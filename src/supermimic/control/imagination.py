"""SuperMimic-style control optimization: analytic reward + BPTT through frozen dynamics.

The original reference-relative termination head is deliberately unused. Only
the new encoder is optimized; the mimic policy/decoder and dynamics are frozen
while computing actor gradients. Dynamics fitting is a separate real-data step.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from supermimic.control.rewards import control_reward_values, fall_probability
from supermimic.control.targets import quat_inverse, rotate
from supermimic.policy.layout import OBS_TERMS
from supermimic.world_model.dynamics import MLP


def observation(s, action, xi, q0):
    gravity = s.new_tensor([0., 0., -1.]).expand(*s.shape[:-1], 3)
    return torch.cat((s[..., 7:28] + xi[..., 1:22] - q0, s[..., 34:55],
                      s[..., 31:34], rotate(quat_inverse(s[..., 3:7]), gravity), action), -1)


def flatten_history(history):
    # [H,N,69] -> term-major [N,H*69], identical to the native mimic contract.
    chunks, offset = [], 0
    for _, dim in OBS_TERMS:
        term = history[..., offset:offset + dim]
        chunks.append(term.permute(1, 0, 2).flatten(1))
        offset += dim
    return torch.cat(chunks, -1)


@dataclass
class ImaginationCfg:
    horizon: int = 16
    gamma: float = .99
    gae_lambda: float = .95
    lr: float = 2e-4
    critic_lr: float = 2e-4
    grad_clip: float = 1.
    use_critic: bool = True
    compile: str | None = None
    obs_noise: bool = True


class ControlImaginationTrainer:
    def __init__(self, wm, control, q0, *, cfg=None, sites=None, soft_limits=None,
                 action_rate_weight=.1, joint_limit_weight=2.):
        self.wm, self.control, self.q0 = wm, control, q0
        self.cfg = cfg or ImaginationCfg()
        if self.cfg.horizon < 1:
            raise ValueError("imagination horizon must be positive")
        self.sites, self.soft_limits = sites, soft_limits
        self.action_rate_weight, self.joint_limit_weight = action_rate_weight, joint_limit_weight
        # Privileged simulated state/context are critic-only, never actor inputs.
        self.critic = MLP(55 + 25 + control.target_dim, (256, 128), 1).to(q0.device)
        self.opt = torch.optim.Adam(control.encoder.parameters(), lr=self.cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)
        self.step_fn = self._step
        if self.cfg.compile:
            mode = None if self.cfg.compile == "default" else self.cfg.compile
            self.step_fn = torch.compile(self._step, mode=mode, dynamic=False)

    def _step(self, state, action, previous, xi, target):
        state = self.wm.step(state, action, xi)
        reward = control_reward_values(state, target, self.sites)
        reward = reward - self.action_rate_weight * (action - previous).square().sum(-1)
        if self.soft_limits is not None:
            joints = state[..., 7:28]
            violation = ((self.soft_limits[:, 0] - joints).clamp_min(0)
                         + (joints - self.soft_limits[:, 1]).clamp_min(0)).sum(-1)
            reward = reward - self.joint_limit_weight * violation
        return (state, reward * self.wm.spec.dt, fall_probability(state),
                observation(state, action, xi, self.q0))

    def value(self, state, target, xi):
        feat = self.wm.spec.canonical(state)
        context = self.wm.spec.normalize_xi(xi)
        return self.critic(torch.cat((feat, target, context), -1)).squeeze(-1)

    def rollout(self, start, targets):
        cfg = self.cfg
        if targets.shape != (cfg.horizon + 1, len(start.state), self.control.target_dim):
            raise ValueError("targets must contain H+1 current control targets per rollout")
        # External targets are constants, not differentiable paths back into source trajectories.
        targets = targets.detach()
        def noisy(o):
            if not cfg.obs_noise:
                return o
            half = o.new_tensor([.01] * 21 + [.5] * 21 + [.2] * 3 + [.05] * 3 + [0.] * 21)
            return o + (torch.rand_like(o) * 2 - 1) * half

        hist = [noisy(observation(s, a, start.xi, self.q0))
                for s, a in zip(start.hist_states, start.hist_actions)]
        state, previous = start.state.detach(), start.prev_action.detach()
        states, rewards, alive, dones = [state], [], [state.new_ones(len(state))], []
        for t in range(cfg.horizon):
            action = self.control(flatten_history(torch.stack(hist)), targets[t])
            state, reward, death, next_obs = self.step_fn(state, action, previous, start.xi, targets[t])
            rewards.append(reward)
            dones.append(death)
            alive.append(alive[-1] * (1 - death))
            states.append(state)
            hist = hist[1:] + [noisy(next_obs)]
            previous = action
        return torch.stack(states), torch.stack(rewards), torch.stack(alive), torch.stack(dones)

    def update(self, start, targets):
        # Frozen parameters still transmit derivatives with respect to actions/states.
        parameters = list(self.wm.parameters()) + list(self.critic.parameters())
        flags = [p.requires_grad for p in parameters]
        for p in parameters:
            p.requires_grad_(False)
        try:
            states, rewards, alive, done = self.rollout(start, targets)
            discount = self.cfg.gamma ** torch.arange(self.cfg.horizon, device=states.device)
            objective = (discount[:, None] * alive[:-1] * rewards).sum(0)
            if self.cfg.use_critic:
                objective = objective + self.cfg.gamma ** self.cfg.horizon * alive[-1] * self.value(
                    states[-1], targets[-1].detach(), start.xi)
            loss = -objective.mean() / self.cfg.horizon
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite imagined control objective")
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(self.control.encoder.parameters(), self.cfg.grad_clip,
                                           error_if_nonfinite=True)
            self.opt.step()
        finally:
            for p, flag in zip(parameters, flags):
                p.requires_grad_(flag)
        critic_loss = states.new_zeros(())
        if self.cfg.use_critic:
            with torch.no_grad():
                values = torch.stack([self.value(s.detach(), g.detach(), start.xi)
                                      for s, g in zip(states, targets)])
                returns, carry = [], values[-1]
                for t in reversed(range(self.cfg.horizon)):
                    carry = rewards[t].detach() + self.cfg.gamma * (1 - done[t].detach()) * (
                        (1 - self.cfg.gae_lambda) * values[t + 1] + self.cfg.gae_lambda * carry)
                    returns.append(carry)
                returns = torch.stack(list(reversed(returns)))
            predicted = torch.stack([self.value(s.detach(), g.detach(), start.xi)
                                     for s, g in zip(states[:-1], targets[:-1])])
            critic_loss = F.mse_loss(predicted, returns)
            if not torch.isfinite(critic_loss):
                raise FloatingPointError("nonfinite control critic loss")
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip, error_if_nonfinite=True)
            self.critic_opt.step()
        return {"imag_objective": float(objective.detach().mean()), "grad_norm": float(norm),
                "imag_survival": float(alive[-1].detach().mean()),
                "imag_reward": float(rewards.detach().mean()), "critic_loss": float(critic_loss.detach())}
