"""Mini3 command/teleop refinement with the existing SuperMimic dynamics and BPTT.

Real simulation -> replay -> dynamics correction -> held-out prediction check ->
imagined actor/critic update. The reference-relative termination head is unused.
No training runs on import or --dry-run. Use a new server run directory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True, help="distilled command/teleop model")
    p.add_argument("--world-model-checkpoint", type=Path, required=True, help="original Cuboi checkpoint containing wm")
    p.add_argument("--library", type=Path, required=True, help="reviewed TRAIN compact subset")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--capacity", type=int, default=256)
    p.add_argument("--wm-window", type=int, default=8)
    p.add_argument("--wm-max-nse", type=float, default=.1, help="held-out normalized MSE limit, heuristic")
    p.add_argument("--warmup-steps", type=int, default=40)
    p.add_argument("--update-every", type=int, default=2)
    p.add_argument("--max-seconds", type=float, required=True)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-vx", type=float, default=.3)
    p.add_argument("--max-vy", type=float, default=.15)
    p.add_argument("--max-yaw-rate", type=float, default=.5)
    p.add_argument("--height-range", type=float, nargs=2, default=[.38, .48])
    p.add_argument("--hold-steps", type=int, default=100)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--compile", choices=("none", "default", "reduce-overhead"), default="default")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    positive = (a.num_envs, a.steps, a.horizon, a.batch_size, a.capacity, a.wm_window,
                a.wm_max_nse, a.warmup_steps, a.update_every, a.max_seconds, a.lr,
                a.max_vx, a.max_vy, a.max_yaw_rate, a.hold_steps, a.save_every)
    if any(not math.isfinite(x) or x <= 0 for x in positive) or a.num_envs < 2:
        p.error("positive counts/budgets required, with at least 2 environments for validation")
    if a.capacity <= max(a.horizon, a.wm_window) + 12:
        p.error("replay capacity must exceed history and model-validation horizon")
    if not all(math.isfinite(x) for x in a.height_range) or not .15 < a.height_range[0] <= a.height_range[1]:
        p.error("height range must be finite, ordered and above the fall threshold")
    if a.output.exists():
        p.error("output exists; choose a new run directory")
    if a.dry_run:
        print(json.dumps({**vars(a), "stage": "supermimic", "starts_training": False}, default=str, indent=2))
        return

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import RewardTermCfg, TerminationTermCfg
    from mjlab.utils.torch import configure_torch_backends
    from supermimic.control.dynamics import ControlDynamicsTrainer, imagination_allowed, load_world_model, sample_windows
    from supermimic.control.imagination import ControlImaginationTrainer, ImaginationCfg
    from supermimic.control.model import load_control
    from supermimic.control.reference import ReferenceTargets
    from supermimic.control.runtime import control_reward, fallen
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg

    manifest = json.loads(a.library.with_suffix(".manifest.json").read_text())
    if manifest.get("split") != "train":
        p.error("library manifest must declare split=train")
    torch.manual_seed(a.seed)
    configure_torch_backends()
    control, source = load_control(a.checkpoint, a.device)
    wm = load_world_model(a.world_model_checkpoint, a.device)
    source_env = source["source_config"]["env"]
    args = SuperMimicEnvCfgArgs(
        npz_files=[str(a.library)], compact_library=True, num_envs=a.num_envs,
        history_length=control.mimic.cfg.history, future_frames=source_env["future_frames"],
        record_transitions=True, replay_capacity=a.capacity, sampling_mode="uniform",
        action_noise_std=.05, seed=a.seed,
    )
    cfg = make_env_cfg(args)
    cfg.rewards = {"control": RewardTermCfg(func=control_reward, weight=1.),
                   "action_rate_l2": cfg.rewards["action_rate_l2"],
                   "joint_limit": cfg.rewards["joint_limit"]}
    for name in ("root_height", "root_tilt", "clip_end"):
        cfg.terminations.pop(name)
    cfg.terminations["control_fall"] = TerminationTermCfg(func=fallen)
    if not math.isclose(wm.spec.dt, cfg.sim.mujoco.timestep * cfg.decimation):
        p.error("world-model dt does not match simulation control dt")
    a.output.mkdir(parents=True, exist_ok=False)
    run_config = {**vars(a), "stage": "supermimic", "subset_manifest": manifest,
                  "sites": source["run_config"].get("sites")}
    (a.output / "config.json").write_text(json.dumps(run_config, indent=2, default=str))
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    env = ManagerBasedRlEnv(cfg, device=a.device)
    start_time, step, actor_updates = time.monotonic(), 0, 0
    commands = torch.zeros(a.num_envs, 4, device=a.device)
    age = torch.zeros(a.num_envs, dtype=torch.long, device=a.device)

    def sample_commands(n):
        command = torch.rand(n, 4, device=a.device)
        command[:, :3] = (command[:, :3] * 2 - 1) * command.new_tensor(
            [a.max_vx, a.max_vy, a.max_yaw_rate])
        command[torch.rand(n, device=a.device) < .2, :3] = 0
        command[:, 3] = a.height_range[0] + command[:, 3] * (a.height_range[1] - a.height_range[0])
        return command

    def resample(ids):
        commands[ids] = sample_commands(len(ids))
        age[ids] = 0

    def save():
        package = control.package(source["source_config"])
        package.update(stage="supermimic", step=step, actor_updates=actor_updates, run_config=run_config,
                       wm=wm.state_dict(), wm_optimizer=dynamics.opt.state_dict(),
                       encoder_optimizer=trainer.opt.state_dict(), control_critic=trainer.critic.state_dict(),
                       critic_optimizer=trainer.critic_opt.state_dict(),
                       elapsed_seconds=time.monotonic() - start_time)
        tmp = a.output / "last.pt.tmp"
        torch.save(package, tmp)
        os.replace(tmp, a.output / "last.pt")

    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        if [clip.name for clip in lib.clips] != manifest["clips"]:
            raise ValueError("library differs from reviewed training manifest")
        targets = ReferenceTargets(lib, control.cfg.interface, run_config["sites"])
        env._mini3_control_sites = targets
        env._mini3_control_target = torch.zeros(a.num_envs, control.target_dim, device=a.device)
        recorder = env.recorder_manager.get_term("replay")
        buffer = recorder.buffer
        dynamics = ControlDynamicsTrainer(wm, lr=a.lr)
        robot = env.scene["robot"]
        trainer = ControlImaginationTrainer(
            wm, control, lib.default_joint_pos, sites=targets,
            soft_limits=robot.data.soft_joint_pos_limits[0].clone(),
            cfg=ImaginationCfg(horizon=a.horizon, lr=a.lr, critic_lr=a.lr,
                               compile=None if a.compile == "none" else a.compile))
        obs, _ = env.reset()
        resample(torch.arange(a.num_envs, device=a.device))
        with (a.output / "metrics.jsonl").open("x") as log:
            while step < a.steps and time.monotonic() - start_time < a.max_seconds and not stop:
                resample((age >= a.hold_steps).nonzero().flatten())
                with torch.no_grad():
                    target = targets(cmd.frame)
                    target[:, :4] = commands
                    env._mini3_control_target = target
                    action = control(obs["policy"], target)
                    if not torch.isfinite(action).all():
                        raise FloatingPointError("nonfinite control action")
                    obs, reward, term, timeout, _ = env.step(action)
                    if not torch.isfinite(reward).all() or not torch.isfinite(obs["policy"]).all():
                        raise FloatingPointError("nonfinite measured control observation or reward")
                step += 1
                age += 1
                resample((term | timeout).nonzero().flatten())
                metrics = {"step": step, "real_reward": float(reward.mean()),
                           "resets": int((term | timeout).sum()), "actor_updates": actor_updates}
                if step >= a.warmup_steps and step % a.update_every == 0:
                    # Training/validation dynamics windows are from disjoint physical environments.
                    try:
                        train_window = sample_windows(buffer, a.batch_size, a.wm_window)
                        validation_window = sample_windows(buffer, min(a.batch_size, 128), a.horizon, validation=True)
                    except RuntimeError as error:
                        if "no sampleable rows" not in str(error):
                            raise
                        metrics["wm_waiting_for_windows"] = True
                    else:
                        metrics.update(dynamics.update(train_window))
                        errors = dynamics.validate(validation_window)
                        metrics["wm_validation_nse"] = errors.tolist()
                        allowed = imagination_allowed(errors, a.wm_max_nse)
                        metrics["imagination_allowed"] = allowed
                        if allowed and time.monotonic() - start_time < a.max_seconds and not stop:
                            starts = buffer.sample_starts(a.batch_size, control.mimic.cfg.history)
                            with torch.no_grad():
                                frames = torch.minimum(starts.frame[None] + torch.arange(
                                    a.horizon + 1, device=a.device)[:, None], lib.last_frame_of(starts.frame)[None])
                                goals = targets(frames.flatten()).reshape(a.horizon + 1, a.batch_size, -1)
                                goals[..., :4] = sample_commands(a.batch_size)[None]
                            metrics.update(trainer.update(starts, goals))
                            actor_updates += 1
                if step == 1 or step % 50 == 0:
                    metrics.update(actor_updates=actor_updates, elapsed_seconds=time.monotonic() - start_time)
                    log.write(json.dumps(metrics) + "\n")
                    log.flush()
                    print(json.dumps(metrics), flush=True)
                if step % a.save_every == 0:
                    save()
            save()
    finally:
        env.close()


if __name__ == "__main__":
    main()
