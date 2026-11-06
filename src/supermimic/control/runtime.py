"""Absolute fall termination for live simulation control (no reference dependence)."""
import torch

from supermimic.control.rewards import control_reward_values, fallen_state


def fallen(env, min_height: float = 0.15, min_up: float = 0.3):
    qpos = env.sim.data.qpos[:]
    s = qpos.clone()
    s[:, :3] -= env.scene.env_origins
    return fallen_state(s, min_height, min_up)


def control_reward(env):
    """Called by the reward manager BEFORE autoresets, using only measured state."""
    qpos = env.sim.data.qpos[:].clone()
    qpos[:, :3] -= env.scene.env_origins
    qvel = env.sim.data.qvel[:]
    state = torch.cat((qpos, qvel), -1)
    return control_reward_values(state, env._mini3_control_target,
                                 getattr(env, "_mini3_control_sites", None))
