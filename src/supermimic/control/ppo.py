"""Small PPO refinement of a distilled interface, using measured simulator rewards."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from supermimic.world_model.dynamics import MLP


def advantages(rewards, values, done, last_value, gamma=.99, lam=.95):
    """All environment resets end this bounded episodic objective, including timeouts."""
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(len(rewards))):
        alive = (~done[t]).to(rewards.dtype)
        delta = rewards[t] + gamma * alive * next_value - values[t]
        carry = delta + gamma * lam * alive * carry
        advantage[t] = carry
        next_value = values[t]
    return advantage, advantage + values


class ActorCritic(nn.Module):
    def __init__(self, control):
        super().__init__()
        self.control = control
        self.value_net = MLP(control.obs_dim + control.target_dim, (256, 128), 1)
        self.log_std = nn.Parameter(torch.full((21,), -2.0))

    def distribution(self, obs, target):
        return Normal(self.control(obs, target), self.log_std.clamp(-5, 0).exp())

    def value(self, obs, target):
        x = torch.cat((obs * self.control.mimic.decoder.scale, target), -1)
        return self.value_net(x).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, target):
        dist = self.distribution(obs, target)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), self.value(obs, target)


def update(model, optimizer, batch, *, epochs=4, minibatch=4096, clip=.2):
    """Batch stores [T,N,...] tensors from student rollouts; no teacher goals."""
    obs, target, action, old_log, old_value, advantage, returns = (
        batch[key].flatten(0, 1) for key in
        ("obs", "target", "action", "log_prob", "value", "advantage", "return")
    )
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
    count, stats = 0, {"policy_loss": 0., "value_loss": 0., "entropy": 0., "approx_kl": 0.}
    parameters = [p for p in model.parameters() if p.requires_grad]
    for _ in range(epochs):
        for indices in torch.randperm(len(obs), device=obs.device).split(minibatch):
            dist = model.distribution(obs[indices], target[indices])
            log_prob = dist.log_prob(action[indices]).sum(-1)
            ratio = (log_prob - old_log[indices]).exp()
            pg = -torch.minimum(ratio * advantage[indices],
                                ratio.clamp(1 - clip, 1 + clip) * advantage[indices]).mean()
            value = model.value(obs[indices], target[indices])
            clipped_value = old_value[indices] + (value - old_value[indices]).clamp(-clip, clip)
            vf = torch.maximum((value - returns[indices]).square(),
                               (clipped_value - returns[indices]).square()).mean()
            entropy = dist.entropy().sum(-1).mean()
            loss = pg + .5 * vf - .001 * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                kl = ((ratio - 1) - (log_prob - old_log[indices])).mean()
            for key, value in zip(stats, (pg, vf, entropy, kl)):
                stats[key] += float(value.detach())
            count += 1
    return {key: value / max(count, 1) for key, value in stats.items()}
