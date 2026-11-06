"""Reuse Mini3 dynamics, fit on new real transitions, and gate imagination on held-out error."""
import torch

from supermimic.data.replay_buffer import WindowBatch


def load_world_model(path, device="cpu"):
    # Kept lazy so pure control unit tests do not import simulation dependencies.
    from supermimic.world_model.state_spec import StateSpecCfg
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg
    ck = torch.load(path, map_location=device, weights_only=False)
    source_config = ck["source_config"] if ck.get("format") == "mini3-control-v1" else ck["config"]
    config = dict(source_config["world_model"])
    config["state_spec"] = StateSpecCfg(**config["state_spec"])
    wm = WorldModel(WorldModelCfg(**config), heads=None, device=device)
    wm.load_state_dict(ck["wm"], strict=True)
    # This old head predicts deviation from a motion reference, not physical falling.
    wm.term_head.requires_grad_(False)
    return wm


def sample_windows(buffer, n, k, *, validation=False):
    """Reserve env IDs 0,8,... for WM validation; their transitions never fit dynamics."""
    mask = buffer.window_mask(k)
    if buffer.E < 2:
        raise ValueError("world-model fitting requires at least two environments for held-out error")
    heldout = torch.arange(buffer.E, device=buffer.device) % 8 == 0
    mask &= heldout[:, None] if validation else ~heldout[:, None]
    e, t = buffer._sample_from_mask(mask, n)
    tt = (t[None, :] + torch.arange(k + 1, device=buffer.device)[:, None]) % buffer.T
    ee = e[None].expand_as(tt)
    return WindowBatch(states=buffer.s[ee, tt], a_exec=buffer.a_exec[ee[:-1], tt[:-1]],
                       a_clean=buffer.a[ee[:-1], tt[:-1]], xi=buffer.xi[e, t],
                       reward=buffer.r[ee[:-1], tt[:-1]], done=buffer.done[ee[1:], tt[1:]],
                       frame=buffer.frame[ee[:-1], tt[:-1]])


def window_error(wm, window):
    state, errors = window.states[0], []
    for action, true in zip(window.a_exec, window.states[1:]):
        state = wm.step(state, action, window.xi)
        errors.append(wm.spec.error_vector(state, true).square())
    return torch.stack(errors)


class ControlDynamicsTrainer:
    def __init__(self, wm, lr=2e-4):
        self.wm = wm
        self.opt = torch.optim.Adam(wm.dynamics.parameters(), lr=lr)

    def update(self, window):
        sq = window_error(self.wm, window)
        loss = sq.sum(-1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite control dynamics loss")
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.wm.dynamics.parameters(), 1., error_if_nonfinite=True)
        self.opt.step()
        return {"wm_loss": float(loss.detach()), "wm_grad_norm": float(norm)}

    @torch.no_grad()
    def validate(self, window):
        # Return all step errors: a low one-step error is not sufficient for BPTT.
        return window_error(self.wm, window).mean((-1, -2))


def imagination_allowed(errors, limit):
    return bool(torch.isfinite(errors).all() and (errors <= limit).all())
