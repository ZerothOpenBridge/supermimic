"""CPU-only contracts; synthetic fixtures and one tiny optimizer step, no training job."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from supermimic.control.model import ControlCfg, ControlPolicy, load_control, load_mimic
from supermimic.control.reference import ReferenceTargets, select_clips
from supermimic.control.runtime import fallen
from supermimic.control.ppo import ActorCritic, advantages, update
from supermimic.control.imagination import ControlImaginationTrainer, ImaginationCfg, flatten_history, observation
from supermimic.control.dynamics import ControlDynamicsTrainer, imagination_allowed, sample_windows
from supermimic.control.rewards import control_reward_values, fall_probability
from supermimic.data.replay_buffer import ReplayBuffer, StartBatch, WindowBatch
from supermimic.control.stream import TargetReceiver
from supermimic.control.targets import (
    command_target, gamepad_target, heading, quat_inverse, quat_mul, rotate,
    rotation6d, teleop_target, validate_target,
)
from supermimic.policy.nets import Policy, PolicyCfg


def mimic():
    return Policy(PolicyCfg(encoder_hidden=(16,), decoder_hidden=(16,), history=10))


def ref_state(n=2):
    ref = torch.zeros(n, 55)
    ref[:, 3] = 1
    ref[:, 2] = .45
    return ref


class ToyWorldModel(torch.nn.Module):
    """Small differentiable dynamics fixture, never used for robot training."""
    def __init__(self):
        super().__init__()
        from types import SimpleNamespace
        self.dynamics = torch.nn.Linear(21, 55, bias=False)
        with torch.no_grad():
            self.dynamics.weight.zero_()
            self.dynamics.weight[28, 0] = .5
        self.term_head = torch.nn.Linear(1, 1).requires_grad_(False)
        self.spec = SimpleNamespace(canonical=lambda s: s, normalize_xi=lambda x: x,
                                    error_vector=lambda p, t: p - t, dt=.02)

    def step(self, state, action, xi):
        return state + .02 * self.dynamics(action)

    def termination_logits(self, *args):
        raise AssertionError("reference-relative termination must not be used")


class ControlTests(unittest.TestCase):
    def test_mimic_architecture_and_strict_loading_unchanged(self):
        teacher = mimic()
        obs, goal = torch.randn(2, 690), torch.randn(2, 520)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "mimic.pt"
            from dataclasses import asdict
            torch.save({"config": {"policy": asdict(teacher.cfg)}, "policy": teacher.state_dict()}, path)
            restored, _ = load_mimic(path)
            torch.testing.assert_close(restored(obs, goal), teacher(obs, goal), rtol=0, atol=0)
            self.assertEqual(list(restored.encoders), ["motion"])

    def test_both_encoders_optimize_without_changing_teacher(self):
        for interface, dim in (("command", 4), ("teleop", 31)):
            with self.subTest(interface=interface):
                torch.manual_seed(42)
                model = ControlPolicy(mimic(), ControlCfg(interface=interface, hidden=(16,)))
                before = {k: v.clone() for k, v in model.mimic.state_dict().items()}
                obs, target, goal = torch.randn(4, 690), torch.randn(4, dim), torch.randn(4, 520)
                old = model.encoder.net[0].weight.clone()
                optimizer = torch.optim.Adam(model.encoder.parameters(), lr=.001)
                losses = model.losses(obs, target, goal)
                losses["loss"].backward()
                self.assertTrue(all(p.grad is None for p in model.mimic.parameters()))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.encoder.parameters()))
                optimizer.step()
                self.assertFalse(torch.equal(old, model.encoder.net[0].weight))
                for key, value in model.mimic.state_dict().items():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)
                model.train()
                self.assertFalse(model.mimic.training)
                self.assertEqual(model(obs, target).shape, (4, 21))

    def test_control_package_roundtrip(self):
        model = ControlPolicy(mimic(), ControlCfg(interface="teleop", hidden=(16,)))
        obs, target = torch.randn(2, 690), torch.randn(2, 31)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "control.pt"
            torch.save(model.package({"env": {"future_frames": 10}}), path)
            restored, ck = load_control(path)
            self.assertEqual(ck["stage"], "distillation")
            torch.testing.assert_close(restored(obs, target), model(obs, target), rtol=0, atol=0)

    def test_command_heading_and_world_angular_velocity(self):
        ref = ref_state(1)
        ref[:, 3:7] = torch.tensor([[2**-.5, 0, 0, 2**-.5]])
        ref[:, 28:31] = torch.tensor([[0., 1., 0.]])
        ref[:, 31:34] = torch.tensor([[0., 0., .7]])
        torch.testing.assert_close(command_target(ref), torch.tensor([[1., 0., .7, .45]]), atol=1e-6, rtol=0)

    def test_pose_translation_and_yaw_invariance(self):
        ref = ref_state(2)
        pos = torch.randn(2, 3, 3)
        quat = torch.randn(2, 3, 4)
        quat /= quat.norm(dim=-1, keepdim=True)
        target = teleop_target(ref, pos, quat)
        turn = torch.tensor([[2**-.5, 0, 0, 2**-.5]]).expand(2, -1)
        turn3 = turn[:, None].expand(-1, 3, -1)
        shift = torch.tensor([3., -2., 0.])
        moved = ref.clone()
        moved[:, :3] = rotate(turn, ref[:, :3]) + shift
        moved[:, 3:7] = quat_mul(turn, ref[:, 3:7])
        pos = rotate(turn3, pos) + shift
        quat = quat_mul(turn3, quat)
        torch.testing.assert_close(teleop_target(moved, pos, quat), target, atol=2e-6, rtol=0)

    def test_rotation_columns_and_quaternion_sign(self):
        q = torch.tensor([[1., 0, 0, 0]])
        expected = torch.tensor([[1., 0, 0, 1, 0, 0]])
        torch.testing.assert_close(rotation6d(q), expected)
        torch.testing.assert_close(rotation6d(-q), expected)
        torch.testing.assert_close(quat_mul(q, quat_inverse(q)), q)

    def test_teleop_requires_calibrated_sites(self):
        with self.assertRaises(ValueError):
            ReferenceTargets(None, "teleop")

    def test_clip_selection_is_exact_and_does_not_cross_boundaries(self):
        pack = {"ref": torch.arange(8*55).reshape(8, 55), "names": ["a", "b", "c"],
                "starts": [0, 2, 5], "lengths": [2, 3, 3]}
        selected = select_clips(pack, ["c", "a"])
        self.assertEqual(selected["starts"].tolist(), [0, 3])
        self.assertEqual(selected["lengths"].tolist(), [3, 2])
        torch.testing.assert_close(selected["ref"], torch.cat((pack["ref"][5:8], pack["ref"][:2])))
        self.assertEqual(pack["names"], ["a", "b", "c"])
        with self.assertRaises(ValueError):
            select_clips(pack, ["unknown"])
        with self.assertRaises(ValueError):
            select_clips(pack, ["a", "a"])

    def test_gamepad_deadzone_limits_and_direction(self):
        target = gamepad_target(torch.tensor([[.05, -.05, 0.], [2., -2., 1.]]), .45)
        torch.testing.assert_close(target[0], torch.tensor([0., 0., 0., .45]))
        torch.testing.assert_close(target[1], torch.tensor([.5, -.3, .8, .45]))
        with self.assertRaises(ValueError):
            gamepad_target(torch.zeros(1, 3), .45, deadzone=1.)

    def test_invalid_targets_rejected(self):
        for t in (torch.zeros(1, 3), torch.full((1, 4), float("nan"))):
            with self.assertRaises(ValueError):
                validate_target(t, "command")

    def test_stream_rejects_replay_staleness_and_malformed_packets(self):
        # Test parser without opening sockets, so sandboxed CPU checks are deterministic.
        r = TargetReceiver.__new__(TargetReceiver)
        r.interface, r.max_age, r.sequence, r.target, r.timestamp = "command", .25, -1, None, -1
        packet = {"interface": "command", "sequence": 1, "timestamp": 10., "target": [0, 0, 0, .45]}
        self.assertTrue(r.accept(packet, 10.1))
        for change in ({}, {"sequence": 0}, {"sequence": 2, "timestamp": 9.},
                       {"sequence": 2, "timestamp": 11.}, {"sequence": 2, "target": [float("nan")]*4},
                       {"sequence": 2, "interface": "teleop"}, {"sequence": True}):
            self.assertFalse(r.accept({**packet, **change}, 10.1))
        self.assertFalse(r.accept([], 10.1))
        self.assertEqual(r.sequence, 1)

    def test_absolute_fall_detection_ignores_reference(self):
        from types import SimpleNamespace
        qpos = torch.zeros(3, 28)
        qpos[:, 2] = torch.tensor([.45, .1, .45])
        qpos[:2, 3] = 1
        qpos[2, 4] = 1  # upside-down
        env = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(qpos=qpos)),
                              scene=SimpleNamespace(env_origins=torch.zeros(3, 3)))
        self.assertEqual(fallen(env).tolist(), [False, True, True])

    def test_gae_never_bootstraps_across_reset(self):
        rewards = torch.tensor([[1.], [2.], [3.]])
        values = torch.zeros_like(rewards)
        done = torch.tensor([[False], [True], [False]])
        adv, returns = advantages(rewards, values, done, torch.tensor([4.]), gamma=1., lam=1.)
        torch.testing.assert_close(returns[:, 0], torch.tensor([3., 2., 7.]))
        torch.testing.assert_close(adv, returns)

    def test_ppo_update_is_finite_and_keeps_mimic_frozen(self):
        torch.manual_seed(7)
        model = ActorCritic(ControlPolicy(mimic(), ControlCfg(hidden=(16,))))
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
        obs, target = torch.randn(2, 3, 690), torch.randn(2, 3, 4)
        with torch.no_grad():
            action, log_prob, value = model.act(obs, target)
            adv, returns = advantages(torch.ones(2, 3), value, torch.zeros(2, 3, dtype=torch.bool), torch.zeros(3))
        frozen = {k: v.clone() for k, v in model.control.mimic.state_dict().items()}
        metrics = update(model, optimizer, {"obs": obs, "target": target, "action": action,
                         "log_prob": log_prob, "value": value, "advantage": adv, "return": returns},
                         epochs=1, minibatch=3)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        for key, value in model.control.mimic.state_dict().items():
            torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)

    def test_command_reward_measures_world_velocity_in_robot_heading(self):
        from types import SimpleNamespace
        from supermimic.control.runtime import control_reward
        qpos, qvel = torch.zeros(1, 28), torch.zeros(1, 27)
        qpos[:, 2] = .45
        qpos[:, 3:7] = torch.tensor([[2**-.5, 0, 0, 2**-.5]])
        qvel[:, 1] = .3
        env = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(qpos=qpos, qvel=qvel)),
                              scene=SimpleNamespace(env_origins=torch.zeros(1, 3)),
                              _mini3_control_target=torch.tensor([[.3, 0, 0, .45]]))
        reward = control_reward(env)
        torch.testing.assert_close(reward, torch.tensor([2.2]), atol=1e-6, rtol=0)
        env._mini3_control_target[:, 0] = -.3
        self.assertLess(float(control_reward(env)), float(reward))

    def test_imagined_and_measured_reward_share_identical_function(self):
        from types import SimpleNamespace
        from supermimic.control.runtime import control_reward
        state = ref_state(2)
        target = torch.tensor([[.3, 0, 0, .45], [0, 0, .1, .45]])
        env = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(qpos=state[:, :28], qvel=state[:, 28:])),
                              scene=SimpleNamespace(env_origins=torch.zeros(2, 3)),
                              _mini3_control_target=target)
        torch.testing.assert_close(control_reward(env), control_reward_values(state, target), atol=0, rtol=0)

    def test_fall_probability_uses_physical_boundary(self):
        state = ref_state(2)
        state[1, 2] = .05
        probability = fall_probability(state)
        self.assertLess(float(probability[0]), .01)
        self.assertGreater(float(probability[1]), .95)

    def test_control_observation_history_layout(self):
        state = ref_state(2)
        state[:, 7:28] = 2
        state[:, 34:] = 3
        xi = torch.zeros(2, 25)
        xi[:, 1:22] = .1
        current = observation(state, torch.full((2, 21), 4.), xi, torch.ones(21))
        torch.testing.assert_close(current[:, :21], torch.full((2, 21), 1.1))
        torch.testing.assert_close(current[:, 45:48], torch.tensor([[0., 0., -1.]]).expand(2, -1))
        history = torch.stack((current, current + 10))
        flat = flatten_history(history)
        torch.testing.assert_close(flat[:, :21], current[:, :21])
        torch.testing.assert_close(flat[:, 21:42], current[:, :21] + 10)
        torch.testing.assert_close(flat[:, 42:63], current[:, 21:42])

    def test_supermimic_actor_bptt_reaches_encoder_only(self):
        for use_critic in (False, True):
            with self.subTest(use_critic=use_critic):
                torch.manual_seed(43)
                wm = ToyWorldModel()
                control = ControlPolicy(mimic(), ControlCfg(hidden=(16,)))
                trainer = ControlImaginationTrainer(wm, control, torch.zeros(21),
                    cfg=ImaginationCfg(horizon=2, use_critic=use_critic))
                state = ref_state(2)
                start = StartBatch(state, torch.zeros(2, 25), torch.zeros(2, dtype=torch.long),
                                   torch.zeros(2, 21), state[None].repeat(10, 1, 1), torch.zeros(10, 2, 21))
                targets = torch.tensor([.3, 0., 0., .45]).expand(3, 2, 4).clone()
                original_wm = {k: v.clone() for k, v in wm.state_dict().items()}
                original_teacher = {k: v.clone() for k, v in control.mimic.state_dict().items()}
                original_encoder = control.encoder.net[0].weight.clone()
                metrics = trainer.update(start, targets)
                self.assertTrue(all(torch.isfinite(torch.tensor(x)) for x in metrics.values()))
                self.assertGreater(metrics["grad_norm"], 0.)
                self.assertFalse(torch.equal(original_encoder, control.encoder.net[0].weight))
                self.assertTrue(all(p.grad is None for p in wm.parameters()))
                self.assertFalse(wm.term_head.weight.requires_grad)
                for k, value in wm.state_dict().items():
                    torch.testing.assert_close(value, original_wm[k], atol=0, rtol=0)
                for k, value in control.mimic.state_dict().items():
                    torch.testing.assert_close(value, original_teacher[k], atol=0, rtol=0)

    def test_dynamics_train_validation_environment_split(self):
        buffer = ReplayBuffer(9, 16, 55, 21, 25, "cpu")
        buffer.valid[:, :8] = True
        buffer.ep_id[:, :8] = 0
        buffer.ptr[:] = 8
        buffer.s[:, :, 0] = torch.arange(9)[:, None]
        train = sample_windows(buffer, 128, 2)
        validation = sample_windows(buffer, 128, 2, validation=True)
        self.assertTrue(((train.states[..., 0] != 0) & (train.states[..., 0] != 8)).all())
        self.assertTrue(((validation.states[..., 0] == 0) | (validation.states[..., 0] == 8)).all())

    def test_dynamics_fit_and_multistep_error_gate(self):
        wm = ToyWorldModel()
        trainer = ControlDynamicsTrainer(wm)
        states = ref_state(2)[None].repeat(3, 1, 1)
        actions = torch.ones(2, 2, 21)
        window = WindowBatch(states, actions, actions, torch.zeros(2, 25),
                             torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2, 2, dtype=torch.long))
        metrics = trainer.update(window)
        self.assertTrue(all(torch.isfinite(torch.tensor(x)) for x in metrics.values()))
        errors = trainer.validate(window)
        self.assertEqual(tuple(errors.shape), (2,))
        self.assertTrue(imagination_allowed(torch.tensor([.01, .02]), .1))
        self.assertFalse(imagination_allowed(torch.tensor([.01, .2]), .1))
        self.assertFalse(imagination_allowed(torch.tensor([float("nan")]), .1))


if __name__ == "__main__":
    unittest.main()
