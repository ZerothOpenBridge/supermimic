"""Train new encoders against an immutable mimic teacher and shared frozen decoder."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from supermimic.control.targets import INTERFACE_DIMS
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.world_model.dynamics import MLP


@dataclass
class ControlCfg:
    interface: str = "command"
    hidden: tuple[int, ...] = (512, 256, 128)
    action_weight: float = 1.0
    latent_weight: float = 0.1


def load_mimic(path: str | Path, device: str = "cpu") -> tuple[Policy, dict]:
    """Load a trusted Cuboi Mini3 checkpoint. Zita has a different checkpoint contract."""
    ck = torch.load(path, map_location=device, weights_only=False)
    pc = dict(ck["config"]["policy"])
    for key in ("encoder_hidden", "decoder_hidden", "obs_term_scale", "goal_term_scale"):
        if key in pc:
            pc[key] = tuple(pc[key])
    policy = Policy(PolicyCfg(**pc)).to(device)
    policy.load_state_dict(ck["policy"], strict=True)
    policy.eval().requires_grad_(False)
    return policy, ck["config"]


class ControlPolicy(nn.Module):
    """Student sees current external target + term-major proprioceptive history.

    Proprioception is included in the encoder to disambiguate gait phase without
    a reference clock. The original motion encoder and decoder stay unchanged.
    This is a continuous SuperMimic latent, not SONIC's quantized universal token.
    """

    def __init__(self, mimic: Policy, cfg: ControlCfg):
        super().__init__()
        if cfg.interface not in INTERFACE_DIMS:
            raise ValueError(f"unsupported interface: {cfg.interface}")
        if not cfg.hidden or min(cfg.hidden) < 1:
            raise ValueError("hidden layers must be positive")
        self.cfg = cfg
        self.mimic = mimic.eval().requires_grad_(False)
        self.target_dim = INTERFACE_DIMS[cfg.interface]
        self.obs_dim = mimic.decoder.scale.numel()
        self.encoder = MLP(self.target_dim + self.obs_dim, tuple(cfg.hidden), mimic.cfg.latent_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.mimic.eval()
        return self

    def latent(self, obs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.encoder(torch.cat((target, obs * self.mimic.decoder.scale), -1)))

    def forward(self, obs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mimic.decoder(self.latent(obs, target), obs)

    def losses(self, obs: torch.Tensor, target: torch.Tensor, goal: torch.Tensor) -> dict:
        with torch.no_grad():
            teacher_z = self.mimic.encoders["motion"](goal)
            teacher_action = self.mimic.decoder(teacher_z, obs)
        z = self.latent(obs, target)
        action = self.mimic.decoder(z, obs)
        action_loss = F.mse_loss(action, teacher_action)
        latent_loss = F.mse_loss(z, teacher_z)
        return {"loss": self.cfg.action_weight * action_loss + self.cfg.latent_weight * latent_loss,
                "action_mse": action_loss, "latent_mse": latent_loss}

    def package(self, source_config: dict) -> dict:
        return {"format": "mini3-control-v1", "stage": "distillation",
                "control_config": asdict(self.cfg), "source_config": source_config,
                "policy_config": asdict(self.mimic.cfg), "model": self.state_dict()}


def load_control(path: str | Path, device: str = "cpu") -> tuple[ControlPolicy, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("format") != "mini3-control-v1":
        raise ValueError("expected mini3-control-v1 checkpoint")
    model = ControlPolicy(Policy(PolicyCfg(**ck["policy_config"])),
                          ControlCfg(**ck["control_config"])).to(device)
    model.load_state_dict(ck["model"], strict=True)
    return model.eval(), ck
