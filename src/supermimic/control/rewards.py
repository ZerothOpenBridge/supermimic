"""Differentiable control reward shared by measured and imagined Mini3 states.

State is MuJoCo layout: root angular velocity is BODY-frame. Library references
instead store WORLD angular velocity; convert explicitly at the site boundary.
"""
import torch

from supermimic.control.targets import command_target, normalize_quat, rotate


def upright(s):
    q = normalize_quat(s[..., 3:7])
    return 1 - 2 * (q[..., 1].square() + q[..., 2].square())


def fallen_state(s, min_height=.15, min_up=.3):
    return ((s[..., 2] < min_height) | (upright(s) < min_up)
            | ~torch.isfinite(s).all(-1))


def fall_probability(s):
    """Smooth physical height/uprightness boundary; never calls the mimic term head."""
    height_alive = torch.sigmoid((s[..., 2] - .15) / .025)
    upright_alive = torch.sigmoid((upright(s) - .3) / .1)
    return 1 - height_alive * upright_alive


def control_reward_values(s, desired, sites=None):
    """Per-step unscaled control reward (same function in simulation and imagination)."""
    ref = torch.cat((s[..., :28], s[..., 28:31],
                     rotate(s[..., 3:7], s[..., 31:34]), s[..., 34:]), -1)
    error = command_target(ref) - desired[..., :4]
    velocity = torch.exp(-error[..., :2].square().sum(-1) / .25)
    turning = torch.exp(-error[..., 2].square() / .25)
    height = torch.exp(-error[..., 3].square() / .01)
    reward = velocity + .5 * turning + .5 * height + .2 * upright(s).clamp(-1, 1)
    if desired.shape[-1] == 31:
        if sites is None:
            raise ValueError("teleop reward requires calibrated sites")
        measured = sites.from_state(ref, s[..., :28], s[..., 28:])
        actual_pose = measured[..., 4:].reshape(*s.shape[:-1], 3, 9)
        target_pose = desired[..., 4:].reshape(*s.shape[:-1], 3, 9)
        distance = (actual_pose[..., :3] - target_pose[..., :3]).square().sum(-1)
        orientation = (actual_pose[..., 3:] - target_pose[..., 3:]).square().sum(-1)
        reward = reward + torch.exp(-distance / .01).mean(-1)
        reward = reward + .3 * torch.exp(-orientation / .5).mean(-1)
    reward = reward - .0001 * s[..., 34:].square().sum(-1)
    return reward - 5 * fallen_state(s).to(reward.dtype)
