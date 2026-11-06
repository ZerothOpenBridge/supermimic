"""Reference-to-control supervision, including explicitly configured Mini3 tracking sites."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from supermimic.control.targets import command_target, quat_mul, rotate, teleop_target


class ReferenceTargets:
    def __init__(self, library, interface: str, sites_path: str | Path | dict | None = None):
        self.library = library
        self.interface = interface
        if interface == "command":
            return
        if interface != "teleop" or sites_path is None:
            raise ValueError("teleop requires --sites with calibrated head/left_hand/right_hand sites")
        if not hasattr(library, "_fk"):
            raise ValueError("teleop targets require a compact library with FK")
        self.sites = sites_path if isinstance(sites_path, dict) else json.loads(Path(sites_path).read_text())
        names = ("head", "left_hand", "right_hand")
        entries = [self.sites[name] for name in names]
        self.indices = library._fk.body_indices([entry["body"] for entry in entries])
        self.offset_pos = torch.tensor([entry["pos"] for entry in entries],
                                       device=library.device, dtype=library.dtype)
        self.offset_quat = torch.tensor([entry.get("quat_wxyz", [1, 0, 0, 0]) for entry in entries],
                                        device=library.device, dtype=library.dtype)
        if (self.offset_pos.shape != (3, 3) or self.offset_quat.shape != (3, 4)
                or not torch.isfinite(self.offset_pos).all()
                or not torch.isfinite(self.offset_quat).all()
                or not torch.allclose(self.offset_quat.norm(dim=-1),
                                       self.offset_quat.new_ones(3), atol=1e-4)):
            raise ValueError("sites must contain finite positions and unit WXYZ quaternions")

    @torch.no_grad()
    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        ref = self.library.ref71(frame)
        if self.interface == "command":
            return command_target(ref)
        qpos, qvel = self.library.ref_qpos_qvel(frame)
        return self.from_state(ref, qpos, qvel)

    def from_state(self, ref, qpos, qvel):
        """Same site calibration for reference supervision and measured robot reward."""
        if self.interface == "command":
            return command_target(ref)
        bodies = self.library._fk(qpos, qvel)
        quat = bodies["quat"][:, self.indices]
        pos = bodies["pos"][:, self.indices] + rotate(quat, self.offset_pos.expand_as(bodies["pos"][:, self.indices]))
        quat = quat_mul(quat, self.offset_quat.expand_as(quat))
        return teleop_target(ref, pos, quat)


def select_clips(pack: dict, names: list[str]) -> dict:
    """Build a new compact library; exact clip allowlist, never mutate the source pack."""
    if not names or len(names) != len(set(names)):
        raise ValueError("clip allowlist must be nonempty and unique")
    source_names = list(pack["names"])
    if len(set(source_names)) != len(source_names):
        raise ValueError("source library contains duplicate names")
    lookup = {name: i for i, name in enumerate(source_names)}
    missing = set(names) - lookup.keys()
    if missing:
        raise ValueError(f"clips absent from source library: {sorted(missing)[:5]}")
    chunks, lengths, starts, cursor = [], [], [], 0
    for name in names:
        i = lookup[name]
        start, length = int(pack["starts"][i]), int(pack["lengths"][i])
        if length < 1 or start < 0 or start + length > len(pack["ref"]):
            raise ValueError(f"invalid source clip span: {name}")
        chunks.append(pack["ref"][start:start + length])
        lengths.append(length)
        starts.append(cursor)
        cursor += length
    result = {**pack, "ref": torch.cat(chunks), "names": list(names),
              "starts": torch.tensor(starts), "lengths": torch.tensor(lengths)}
    # Per-clip derived metadata from the input cannot describe a subset.
    return {key: value for key, value in result.items()
            if key in ("ref", "names", "starts", "lengths", "fps", "joint_names")}
