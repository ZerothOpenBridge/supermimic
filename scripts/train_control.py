"""Bounded on-policy distillation / DAgger for Mini3 command or three-point teleop.

This warm-start is not a trained navigation planner or arbitrary-command RL.
The existing motion-tracking simulator supplies resets and teacher references.
Only the teacher consumes future reference frames. Students consume current
commands/poses and measured proprioception at the same simulation instant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True, help="trusted Cuboi mimic checkpoint")
    p.add_argument("--library", type=Path, required=True, help="reviewed TRAIN subset from prepare_control_library.py")
    p.add_argument("--interface", choices=("command", "teleop"), required=True)
    p.add_argument("--sites", type=Path, help="calibrated Mini3 site JSON; required for teleop")
    p.add_argument("--output", type=Path, required=True, help="new run directory, must not exist")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--max-seconds", type=float, required=True)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--teacher-prob-start", type=float, default=1.0)
    p.add_argument("--teacher-prob-end", type=float, default=0.0)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="validate arguments only; no torch/simulator import or writes")
    return p


def main():
    p = parser()
    a = p.parse_args()
    positive = (a.num_envs, a.steps, a.max_seconds, a.lr, a.save_every)
    if any(not math.isfinite(x) or x <= 0 for x in positive):
        p.error("environment count, steps, time budget, lr and save interval must be positive")
    if not all(0 <= v <= 1 for v in (a.teacher_prob_start, a.teacher_prob_end)):
        p.error("teacher probabilities must lie in [0, 1]")
    if a.interface == "teleop" and a.sites is None:
        p.error("teleop requires calibrated --sites; forearm origins are not hand targets")
    if a.output.exists():
        p.error("output exists; use a new run directory")
    if a.dry_run:
        print(json.dumps({**vars(a), "stage": "distillation", "starts_training": False}, default=str, indent=2))
        return
    import torch
    from supermimic.control.model import ControlCfg, ControlPolicy, load_mimic
    from supermimic.control.reference import ReferenceTargets
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env

    manifest_path = a.library.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("split") != "train":
        p.error("library manifest must declare split=train")
    torch.manual_seed(a.seed)
    teacher, source_config = load_mimic(a.checkpoint, a.device)
    source_env = source_config["env"]
    if source_env.get("anchor_obs", False):
        p.error("anchor_obs teacher is not supported by this control adapter")
    frames = source_env["future_frames"]
    if teacher.cfg.goal_dim != frames * 52:
        p.error("teacher future-frame contract mismatch")
    student = ControlPolicy(teacher, ControlCfg(interface=a.interface)).to(a.device)
    optimizer = torch.optim.Adam(student.encoder.parameters(), lr=a.lr)
    args = SuperMimicEnvCfgArgs(
        npz_files=[str(a.library)], compact_library=True, num_envs=a.num_envs,
        future_frames=frames, history_length=teacher.cfg.history, record_transitions=False,
        sampling_mode="uniform", seed=a.seed,
    )
    # Reserve the destination before simulator allocation; never overwrite an earlier run.
    a.output.mkdir(parents=True, exist_ok=False)
    run_config = {**vars(a), "stage": "distillation", "source_config": source_config,
                  "subset_manifest": manifest,
                  "checkpoint_sha256": sha256(a.checkpoint),
                  "sites": json.loads(a.sites.read_text()) if a.sites else None}
    (a.output / "config.json").write_text(json.dumps(run_config, indent=2, default=str))
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    env = make_env(args, device=a.device)
    start, step = time.monotonic(), 0

    def save():
        package = student.package(source_config)
        package.update(step=step, optimizer=optimizer.state_dict(), run_config=run_config,
                       elapsed_seconds=time.monotonic() - start,
                       rng_state=torch.get_rng_state(), source_checkpoint=str(a.checkpoint))
        tmp = a.output / "last.pt.tmp"
        torch.save(package, tmp)
        os.replace(tmp, a.output / "last.pt")

    try:
        cmd = env.command_manager.get_term("motion")
        if list(cmd.library.clips[i].name for i in range(cmd.library.num_clips)) != manifest["clips"]:
            raise ValueError("library contents differ from reviewed clip manifest")
        targets = ReferenceTargets(cmd.library, a.interface, a.sites)
        obs, _ = env.reset()
        with (a.output / "metrics.jsonl").open("x") as log:
            while step < a.steps and time.monotonic() - start < a.max_seconds and not stop:
                # Capture both labels BEFORE stepping: autoresets must never shift supervision.
                history, goal = obs["policy"].detach(), obs["goal"].detach()
                target = targets(cmd.frame)
                if not all(torch.isfinite(x).all() for x in (history, goal, target)):
                    raise FloatingPointError("nonfinite control sample")
                with torch.no_grad():
                    teacher_action = teacher(history, goal)
                    student_action = student(history, target)
                    fraction = step / max(1, a.steps - 1)
                    beta = a.teacher_prob_start + fraction * (a.teacher_prob_end - a.teacher_prob_start)
                    use_teacher = torch.rand((env.num_envs, 1), device=a.device) < beta
                    action = torch.where(use_teacher, teacher_action, student_action)
                losses = student.losses(history, target, goal)
                if not all(torch.isfinite(v) for v in losses.values()):
                    raise FloatingPointError("nonfinite distillation loss")
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(student.encoder.parameters(), 1.0,
                                                          error_if_nonfinite=True)
                optimizer.step()
                obs, reward, terminated, timeout, _ = env.step(action)
                step += 1
                if step == 1 or step % 50 == 0:
                    row = {"step": step, "teacher_probability": beta,
                           **{k: float(v.detach()) for k, v in losses.items()},
                           "grad_norm": float(grad_norm),
                           "tracking_reward": float(reward.mean()),
                           "terminated_fraction": float(terminated.float().mean()),
                           "timeout_fraction": float(timeout.float().mean()),
                           "elapsed_seconds": time.monotonic() - start}
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(json.dumps(row), flush=True)
                if step % a.save_every == 0:
                    save()
            save()
    finally:
        env.close()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    main()
