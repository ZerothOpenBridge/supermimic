"""Read-only CPU checks against real Mini3 assets and a Cuboi mimic checkpoint.

Never constructs a simulation environment, calls an optimizer, or writes a model.
Synthetic teleop site offsets below are test fixtures, not hand calibration.
"""
import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--library", required=True, help="small pilot library, not full training library")
    a = p.parse_args()
    import torch
    from supermimic.control.model import ControlCfg, ControlPolicy, load_mimic
    from supermimic.control.reference import ReferenceTargets
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.mini3.spec import CONTROL, DEFAULT_POS, TRACKED_BODIES
    from mjlab.managers import RewardTermCfg, TerminationTermCfg
    from supermimic.control.runtime import control_reward, fallen
    from supermimic.control.dynamics import load_world_model
    from supermimic.control.imagination import observation, flatten_history
    from supermimic.control.rewards import control_reward_values
    from supermimic.world_model.heads import analytic_observation, flatten_history_term_major

    torch.set_num_threads(1)
    teacher, source = load_mimic(a.checkpoint, "cpu")
    lib = MotionLibrary([a.library], TRACKED_BODIES, torch.tensor(DEFAULT_POS),
                        device="cpu", compact=True, store_dtype=torch.float32)
    frame = lib.clip_start[:2]
    goals = lib.goal_window(frame, source["env"]["future_frames"])
    history = torch.zeros(len(frame), teacher.decoder.scale.numel())
    assert len(CONTROL["joint_names"]) == 21
    # Exercise actual FK and site transforms without claiming these offsets are calibrated hands.
    sites = {name: {"body": body, "pos": [0., 0., 0.]} for name, body in zip(
        ("head", "left_hand", "right_hand"),
        ("head_link", "left_elbow_pitch_link", "right_elbow_pitch_link"))}
    results = {}
    with torch.inference_mode():
        original = teacher(history, goals)
        for interface in ("command", "teleop"):
            target = ReferenceTargets(lib, interface, sites if interface == "teleop" else None)(frame)
            student = ControlPolicy(teacher, ControlCfg(interface=interface, hidden=(16,)))
            action = student(history, target)
            assert action.shape == (len(frame), 21) and torch.isfinite(action).all()
            results[interface] = {"target_shape": list(target.shape), "action_shape": list(action.shape)}
        torch.testing.assert_close(teacher(history, goals), original, rtol=0, atol=0)
    # Check manager APIs and config mutation without allocating physics or touching a GPU.
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[a.library], num_envs=1,
                                           compact_library=True, record_transitions=False))
    cfg.rewards = {"control": RewardTermCfg(func=control_reward, weight=1.)}
    cfg.terminations["control_fall"] = TerminationTermCfg(func=fallen)
    wm = load_world_model(a.checkpoint, "cpu")
    qpos, qvel = lib.ref_qpos_qvel(frame)
    state = torch.cat((qpos, qvel), -1)
    xi = wm.spec.xi_offset[None].expand(len(frame), -1)
    previous = torch.zeros(len(frame), 21)
    with torch.inference_mode():
        predicted = wm.step(state, previous, xi)
        assert predicted.shape == state.shape and torch.isfinite(predicted).all()
        ours = observation(state, previous, xi, lib.default_joint_pos)
        native = analytic_observation(state, previous, xi, lib.default_joint_pos)
        torch.testing.assert_close(ours, native)
        torch.testing.assert_close(flatten_history(ours[None].repeat(10, 1, 1)),
                                   flatten_history_term_major(native[None].repeat(10, 1, 1)))
    assert wm.spec.dt == cfg.sim.mujoco.timestep * cfg.decimation
    # Autograd-only check: real dynamics -> calibrated-site transform -> teleop reward.
    # No optimizer runs, and all real checkpoint parameters remain untouched.
    teleop_sites = ReferenceTargets(lib, "teleop", sites)
    desired = teleop_sites(frame).detach().clone()
    desired[:, 4] += .01
    differentiable_action = torch.zeros_like(previous, requires_grad=True)
    future = wm.step(state.detach(), differentiable_action, xi)
    reward = control_reward_values(future, desired, teleop_sites)
    gradient, = torch.autograd.grad(reward.sum(), differentiable_action)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    print(json.dumps({"passed": True, "device": "cpu", "optimizer_steps": 0,
                      "simulation_steps": 0, "clips": lib.num_clips, "interfaces": results,
                      "world_model_state_shape": list(predicted.shape),
                      "native_observation_matches": True,
                      "teleop_reward_gradient_finite": True}, indent=2))


if __name__ == "__main__":
    main()
