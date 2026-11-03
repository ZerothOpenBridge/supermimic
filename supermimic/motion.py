"""Reference motion library: LAFAN1 G1 retargets resampled to 50 Hz, with FK of tracked bodies."""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

from .kinematics import G1Kinematics
from .rotations import mat_to_6d, quat_slerp, quat_to_mat, so3_log, yaw_mat

CONTROL_DT = 0.02
BIN_FRAMES = 50  # 1 s bins for return-based reset sampling


class MotionLib:
    def __init__(self, files, kin, device, src_fps=50.):
        from .mini3.compact_motion import initialize
        initialize(self,files,kin,device)

    @staticmethod
    def from_dir(path: str, kin: G1Kinematics, device: str) -> MotionLib:
        files = [path] if path.endswith(".pt") else sorted(glob.glob(os.path.join(path, "*.pt")))
        return MotionLib(files, kin, device)

    def future_frames(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Frames f+1..f+K clamped to the clip's last frame, shape (B, K)."""
        end = self.clip_end[self.clip_id[f]]
        fut = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        return torch.minimum(fut, end.unsqueeze(-1))

    def goal(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """K future reference frames, holding the final pose with zero velocity beyond the clip."""
        fut = self.future_frames(f, K)  # (B,K)
        requested = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        recorded = (requested <= self.clip_end[self.clip_id[f]].unsqueeze(-1)).unsqueeze(-1)
        # Keep the recorded last-frame velocity; only the synthetic held-pose
        # extension is stationary. Do not change velocities in ref_state/rewards.
        v = torch.where(recorded, self.root_v_local[fut], 0.0)
        w = torch.where(recorded, self.root_w_local[fut], 0.0)
        qd = torch.where(recorded, self.qd[fut], 0.0)
        H = yaw_mat(self.root_R[f])  # (B,3,3)
        Ht = H.transpose(-1, -2).unsqueeze(1)
        dp = (Ht @ (self.root_p[fut] - self.root_p[f].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rot = mat_to_6d(Ht @ self.root_R[fut])
        feats = torch.cat(
            (dp, rot, v, w, self.q[fut], qd * 0.1), -1
        )  # (B,K,73)
        return feats.reshape(f.shape[0], -1)

    def ref_state(self, f: torch.Tensor):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]


def _gradient(x: torch.Tensor, dt: float) -> torch.Tensor:
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g
