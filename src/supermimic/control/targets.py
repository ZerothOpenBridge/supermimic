"""Control targets in the Mini3 heading frame, SI units, WXYZ quaternions.

Command: [vx, vy, yaw_rate, pelvis_height]. Teleop appends head, left-hand,
right-hand poses: pelvis-relative position (3) and rotation columns 0/1 (6).
Positions/orientations must already be calibrated and scaled to Mini3.
Only the current target is exposed to students; future motion belongs to teachers.
"""
from __future__ import annotations

import torch

COMMAND_DIM = 4
TELEOP_DIM = 31
INTERFACE_DIMS = {"command": COMMAND_DIM, "teleop": TELEOP_DIM}


def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    return normalize_quat(q) * q.new_tensor([1, -1, -1, -1])


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.cross(av, bv, dim=-1)), -1)


def rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = normalize_quat(q)
    t = 2 * torch.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q[..., 1:], t, dim=-1)


def heading(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = normalize_quat(q).unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zero = torch.zeros_like(yaw)
    return torch.stack((torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)), -1)


def rotation6d(q: torch.Tensor) -> torch.Tensor:
    """First two matrix columns, flattened in row-major order (same as mimic)."""
    w, x, y, z = normalize_quat(q).unbind(-1)
    return torch.stack((1 - 2*(y*y + z*z), 2*(x*y - w*z),
                        2*(x*y + w*z), 1 - 2*(x*x + z*z),
                        2*(x*z - w*y), 2*(y*z + w*x)), -1)


def command_target(ref: torch.Tensor) -> torch.Tensor:
    """Packed 55-D library state (angular velocity is WORLD) -> current command."""
    inv = quat_inverse(heading(ref[..., 3:7]))
    vel = rotate(inv, ref[..., 28:31])
    omega = rotate(inv, ref[..., 31:34])
    return torch.cat((vel[..., :2], omega[..., 2:3], ref[..., 2:3]), -1)


def teleop_target(ref: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Three reference body poses, in order head/left hand/right hand, -> 31-D target."""
    inv = quat_inverse(heading(ref[..., 3:7]))[..., None, :].expand_as(quat)
    relative_pos = rotate(inv, pos - ref[..., None, :3])
    relative_rot = rotation6d(quat_mul(inv, quat))
    poses = torch.cat((relative_pos, relative_rot), -1).flatten(-2)
    return torch.cat((command_target(ref), poses), -1)


def validate_target(target: torch.Tensor, interface: str) -> None:
    expected = INTERFACE_DIMS[interface]
    if target.ndim < 1 or target.shape[-1] != expected or not torch.isfinite(target).all():
        raise ValueError(f"{interface} requires finite (..., {expected}) targets")


def gamepad_target(axes: torch.Tensor, height: float, *,
                   max_vx: float = 0.5, max_vy: float = 0.3,
                   max_yaw_rate: float = 0.8, deadzone: float = 0.1) -> torch.Tensor:
    """Normalized [forward, left, left-turn] axes. Device axis mapping is caller-owned."""
    if axes.shape[-1] != 3 or not torch.isfinite(axes).all():
        raise ValueError("gamepad axes must be finite (..., 3)")
    if not 0 <= deadzone < 1 or height <= 0 or min(max_vx, max_vy, max_yaw_rate) <= 0:
        raise ValueError("invalid gamepad calibration")
    axes = axes.clamp(-1, 1)
    axes = axes.sign() * ((axes.abs() - deadzone) / (1 - deadzone)).clamp_min(0)
    velocity = axes * axes.new_tensor([max_vx, max_vy, max_yaw_rate])
    return torch.cat((velocity, axes.new_full((*axes.shape[:-1], 1), height)), -1)
