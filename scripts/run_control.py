"""Test a distilled Mini3 interface in simulation: reference targets, UDP or Linux gamepad.

There is no hardware transport. Live modes stop on falls, episode timeout, disconnect,
or stale targets; they do not silently reset the robot and continue commanding it.
"""
import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True, help="small held-out compact library; not Full-6000")
    p.add_argument("--mode", choices=("reference", "udp", "gamepad"), default="reference")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--port", type=int, default=5558)
    p.add_argument("--max-age", type=float, default=0.25)
    p.add_argument("--joystick", default="/dev/input/js0")
    p.add_argument("--axes", type=int, nargs=3, default=[1, 0, 3], metavar=("FORWARD", "LEFT", "YAW"))
    p.add_argument("--signs", type=float, nargs=3, default=[-1, -1, -1])
    p.add_argument("--height", type=float, help="Mini3 pelvis height (m); required for gamepad")
    a = p.parse_args()
    if a.steps < 1:
        p.error("steps must be positive")
    import torch
    from supermimic.control.model import load_control
    from supermimic.control.reference import ReferenceTargets
    from supermimic.control.stream import LinuxJoystick, TargetReceiver
    from supermimic.control.targets import gamepad_target, validate_target
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env, make_env_cfg
    from mjlab.envs import ManagerBasedRlEnv

    model, ck = load_control(a.checkpoint, a.device)
    interface = model.cfg.interface
    if a.mode == "gamepad" and (interface != "command" or a.height is None):
        p.error("gamepad requires command checkpoint and --height")
    e = ck["source_config"]["env"]
    args = SuperMimicEnvCfgArgs(
        npz_files=[str(a.library)], compact_library=True, num_envs=1,
        future_frames=e["future_frames"], history_length=model.mimic.cfg.history,
        action_noise_std=0, obs_noise=False, reset_noise=False,
        domain_randomization=False, sampling_mode="start", record_transitions=False,
    )
    receiver = None
    if a.mode == "reference":
        env = make_env(args, device=a.device)
    else:
        # Live targets must not fail due to divergence from an unrelated library pose.
        # Keep absolute height/uprightness checks and bounded episode timeout.
        from mjlab.managers import RewardTermCfg, TerminationTermCfg
        from mjlab.utils.torch import configure_torch_backends
        from supermimic.control.runtime import control_reward, fallen
        configure_torch_backends()
        cfg = make_env_cfg(args)
        cfg.terminations.pop("root_height")
        cfg.terminations.pop("root_tilt")
        cfg.terminations.pop("clip_end")
        cfg.terminations["control_fall"] = TerminationTermCfg(func=fallen)
        cfg.rewards = {"control": RewardTermCfg(func=control_reward, weight=1.)}
        env = ManagerBasedRlEnv(cfg, device=a.device)
    try:
        if a.mode == "udp":
            receiver = TargetReceiver(interface, a.port, a.max_age)
        elif a.mode == "gamepad":
            receiver = LinuxJoystick(a.joystick)
        cmd = env.command_manager.get_term("motion")
        sites = ck["run_config"]["sites"] if interface == "teleop" else None
        targets = ReferenceTargets(cmd.library, interface, sites)
        env._mini3_control_sites = targets
        env._mini3_control_target = torch.zeros(1, model.target_dim, device=a.device)
        obs, _ = env.reset()
        total_reward, failures, timeouts = 0.0, 0, 0
        with torch.inference_mode():
            for step in range(a.steps):
                tick = time.monotonic()
                if a.mode == "reference":
                    target = targets(cmd.frame)
                elif a.mode == "udp":
                    target = torch.tensor([receiver.read()], device=a.device)
                else:
                    axes = torch.tensor([receiver.read(tuple(a.axes), tuple(a.signs))], device=a.device)
                    limits = ck["run_config"]
                    target = gamepad_target(axes, a.height,
                        max_vx=limits.get("max_vx", .3), max_vy=limits.get("max_vy", .15),
                        max_yaw_rate=limits.get("max_yaw_rate", .5))
                validate_target(target, interface)
                env._mini3_control_target = target
                action = model(obs["policy"], target)
                if not torch.isfinite(action).all():
                    raise FloatingPointError("nonfinite control action")
                obs, reward, term, tout, _ = env.step(action)
                total_reward += float(reward.sum())
                failures += int(term.sum())
                timeouts += int(tout.sum())
                if a.mode != "reference" and (term.any() or tout.any()):
                    break
                if a.mode != "reference":
                    time.sleep(max(0, env.step_dt - (time.monotonic() - tick)))
        print(json.dumps({"stage": ck["stage"], "mode": a.mode, "steps": step + 1,
                          "failures": failures, "timeouts": timeouts,
                          "reward": total_reward,
                          "note": "Reference tracking diagnostic; not an arbitrary-command acceptance score."}))
    finally:
        if receiver is not None:
            receiver.close()
        env.close()


if __name__ == "__main__":
    main()
