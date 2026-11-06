"""Select a reviewed locomotion/teleop subset from an existing TRAIN compact library."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--clips", type=Path, required=True, help="JSON array of exact training clip IDs")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.with_suffix(".manifest.json").exists():
        p.error("output already exists; select a new destination")
    if "full6000" in str(a.input).lower().replace("-", "").replace("_", ""):
        p.error("Full-6000 is evaluation only")
    import torch
    from supermimic.control.reference import select_clips
    names = json.loads(a.clips.read_text())
    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        p.error("--clips must contain a JSON array of strings")
    pack = torch.load(a.input, map_location="cpu", weights_only=False)
    if pack["ref"].ndim != 2 or pack["ref"].shape[1] != 55:
        p.error("expected a 55-D Mini3 library, not G1 data")
    subset = select_clips(pack, names)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("xb") as f:
        torch.save(subset, f)
    manifest = {"source": str(a.input.resolve()), "clips": names,
                "allowlist_sha256": hashlib.sha256(a.clips.read_bytes()).hexdigest(),
                "frames": len(subset["ref"]), "split": "train",
                "note": "Caller must supply a training source; filename checks are not split proof."}
    a.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"clips": len(names), "frames": len(subset["ref"]), "output": str(a.output)}))


if __name__ == "__main__":
    main()
