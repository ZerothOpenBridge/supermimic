"""Refine a distilled Mini3 encoder with PPO on independent commands and measured rewards.

Run on the training server only. A reviewed TRAIN library provides initial poses
and, for teleop, reachable site trajectories. Commands are sampled independently
of reference velocity. No future reference is supplied to the actor or critic.
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
    p.add_argument("--checkpoint", type=Path, required=True, help="distilled control checkpoint")
    p.add_argument("--library", type=Path, required=True, help="reviewed TRAIN compact subset")
    p.add_argument("--output", type=Path, required=True, help="new run directory")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--max-seconds", type=float, required=True)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-vx", type=float, default=.3)
    p.add_argument("--max-vy", type=float, default=.15)
    p.add_argument("--max-yaw-rate", type=float, default=.5)
    p.add_argument("--height-range", type=float, nargs=2, default=[.38, .48])
    p.add_argument("--hold-steps", type=int, default=100)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    positive = [a.num_envs, a.iterations, a.horizon, a.max_seconds, a.lr, a.max_vx,
                a.max_vy, a.max_yaw_rate, a.hold_steps, a.save_every]
    if any(not math.isfinite(x) or x <= 0 for x in positive):
        p.error("counts, budgets, rates and velocity limits must be positive")
    if not all(math.isfinite(x) for x in a.height_range) or not .15 < a.height_range[0] <= a.height_range[1]:
        p.error("height range must be finite, ordered and above the fall threshold")
    if a.output.exists():
        p.error("output exists; select a new run directory")
    if a.dry_run:
        print(json.dumps({**vars(a), "stage": "ppo", "starts_training": False}, default=str, indent=2))
        return
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import RewardTermCfg, TerminationTermCfg
    from mjlab.utils.torch import configure_torch_backends
    from supermimic.control.model import load_control
    from supermimic.control.ppo import ActorCritic, advantages, update
    from supermimic.control.reference import ReferenceTargets
    from supermimic.control.runtime import control_reward, fallen
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg

    manifest = json.loads(a.library.with_suffix(".manifest.json").read_text())
    if manifest.get("split") != "train":
        p.error("library manifest must declare split=train")
    torch.manual_seed(a.seed)
    configure_torch_backends()
    control, source = load_control(a.checkpoint, a.device)
    model = ActorCritic(control).to(a.device)
    optimizer = torch.optim.Adam([v for v in model.parameters() if v.requires_grad], lr=a.lr)
    se = source["source_config"]["env"]
    args = SuperMimicEnvCfgArgs(
        npz_files=[str(a.library)], compact_library=True, num_envs=a.num_envs,
        history_length=control.mimic.cfg.history, future_frames=se["future_frames"],
        record_transitions=False, sampling_mode="uniform", action_noise_std=0,
        seed=a.seed,
    )
    cfg = make_env_cfg(args)
    cfg.rewards = {"control": RewardTermCfg(func=control_reward, weight=1.),
                   "action_rate_l2": cfg.rewards["action_rate_l2"],
                   "joint_limit": cfg.rewards["joint_limit"]}
    cfg.terminations.pop("root_height")
    cfg.terminations.pop("root_tilt")
    cfg.terminations.pop("clip_end")
    cfg.terminations["control_fall"] = TerminationTermCfg(func=fallen)
    a.output.mkdir(parents=True, exist_ok=False)
    run_config = {**vars(a), "stage": "ppo", "parent_checkpoint": str(a.checkpoint.resolve()),
                  "subset_manifest": manifest, "sites": source["run_config"].get("sites")}
    (a.output / "config.json").write_text(json.dumps(run_config, default=str, indent=2))
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    env = ManagerBasedRlEnv(cfg, device=a.device)
    start, iteration = time.monotonic(), 0
    command = torch.zeros(a.num_envs, 4, device=a.device)
    age = torch.full((a.num_envs,), a.hold_steps, device=a.device, dtype=torch.long)

    def resample(ids):
        n = len(ids)
        command[ids, :3] = (torch.rand(n, 3, device=a.device) * 2 - 1) * command.new_tensor(
            [a.max_vx, a.max_vy, a.max_yaw_rate])
        # Explicit stop commands are essential for a centered gamepad.
        command[ids[torch.rand(n, device=a.device) < .2], :3] = 0
        command[ids, 3] = a.height_range[0] + torch.rand(n, device=a.device) * (
            a.height_range[1] - a.height_range[0])
        age[ids] = 0

    def current_target():
        target = targets(cmd.frame)
        target[:, :4] = command
        return target

    def save():
        package = control.package(source["source_config"])
        package.update(stage="ppo", iteration=iteration, run_config=run_config,
                       actor_critic=model.state_dict(), optimizer=optimizer.state_dict(),
                       elapsed_seconds=time.monotonic() - start)
        tmp = a.output / "last.pt.tmp"
        torch.save(package, tmp)
        os.replace(tmp, a.output / "last.pt")

    try:
        cmd = env.command_manager.get_term("motion")
        if [clip.name for clip in cmd.library.clips] != manifest["clips"]:
            raise ValueError("library differs from reviewed training manifest")
        targets = ReferenceTargets(cmd.library, control.cfg.interface, run_config["sites"])
        env._mini3_control_sites = targets
        env._mini3_control_target = torch.zeros(a.num_envs, control.target_dim, device=a.device)
        obs, _ = env.reset()
        resample(torch.arange(a.num_envs, device=a.device))
        with (a.output / "metrics.jsonl").open("x") as log:
            while iteration < a.iterations and time.monotonic() - start < a.max_seconds and not stop:
                storage = {key: [] for key in ("obs", "target", "action", "log_prob", "value", "reward", "done")}
                for _ in range(a.horizon):
                    ids = (age >= a.hold_steps).nonzero().flatten()
                    resample(ids)
                    target = current_target()
                    env._mini3_control_target = target
                    history = obs["policy"].detach()
                    if not torch.isfinite(history).all() or not torch.isfinite(target).all():
                        raise FloatingPointError("nonfinite PPO observation or target")
                    with torch.no_grad():
                        action, log_prob, value = model.act(history, target)
                    next_obs, reward, term, timeout, _ = env.step(action)
                    if not torch.isfinite(reward).all():
                        raise FloatingPointError("nonfinite measured control reward")
                    done = term | timeout
                    for key, data in zip(storage, (history, target, action, log_prob, value, reward, done)):
                        storage[key].append(data.detach().clone())
                    obs = next_obs
                    age += 1
                    resample(done.nonzero().flatten())
                    if time.monotonic() - start >= a.max_seconds or stop:
                        break
                batch = {key: torch.stack(data) for key, data in storage.items()}
                # The bootstrap target is exactly the command used at the next policy step.
                resample((age >= a.hold_steps).nonzero().flatten())
                with torch.no_grad():
                    last_value = model.value(obs["policy"], current_target())
                    batch["advantage"], batch["return"] = advantages(
                        batch["reward"], batch["value"], batch["done"], last_value)
                if time.monotonic() - start >= a.max_seconds or stop:
                    break
                metrics = update(model, optimizer, batch)
                iteration += 1
                row = {"iteration": iteration, **metrics,
                       "reward": float(batch["reward"].mean()),
                       "resets": int(batch["done"].sum()),
                       "elapsed_seconds": time.monotonic() - start}
                log.write(json.dumps(row) + "\n")
                log.flush()
                if iteration == 1 or iteration % 10 == 0:
                    print(json.dumps(row), flush=True)
                if iteration % a.save_every == 0:
                    save()
            save()
    finally:
        env.close()


if __name__ == "__main__":
    main()
