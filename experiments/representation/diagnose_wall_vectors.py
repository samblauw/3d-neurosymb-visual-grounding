#!/usr/bin/env python3
"""Count zero baseline wall vectors over the scenes in a frozen parses file.

Thesis representation diagnostic (§5.4). Uses the inherited wall encoder and
the baseline's GT scene loader, including its foreground candidate filtering.
No parser generation, feature-cache replacement, or calibration is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from grounding.paths import PARSES_DEV, RESULTS_STAGE1, resolve_under_root
from grounding.data.benchmark import load_boxes
from grounding.resolution.artifacts import git_commit, relative_path, sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parses", type=Path,
                        default=PARSES_DEV / "nr3d_dev_30b_v4.jsonl")
    parser.add_argument("--out", type=Path,
                        default=RESULTS_STAGE1 / "baseline_wall_vectors.json")
    args = parser.parse_args()
    parses = resolve_under_root(args.parses)
    scan_ids = sorted({json.loads(line)["scan_id"]
                       for line in parses.read_text().splitlines() if line.strip()})

    import torch
    from grounding.resolution import engine

    engine.bind("baseline")
    engine.rel_num()  # Import this tree before accessing its inherited encoder
    from grounding.representation.encoders.against_wall import AgainstTheWall

    scenes = engine.Scenes()
    rows = []
    for scan_id in scan_ids:
        scan = scenes.load(scan_id)
        locations = torch.as_tensor(scan["pred_locs"])
        room = torch.as_tensor(scan["room_points"])
        encoder = AgainstTheWall(locations, room)
        vector = encoder.forward().detach().cpu()
        boxes = load_boxes(scan_id, include_background=True)
        rows.append({
            "scan_id": scan_id,
            "n_candidates": len(scan["obj_ids"]),
            "n_heuristic_walls": len(encoder.walls),
            "n_annotated_walls": sum(box["label"] == "wall" for box in boxes),
            "all_zero": bool((vector == 0).all()),
            "nonzero_entries": int(torch.count_nonzero(vector)),
            "nonfinite_entries": int((~torch.isfinite(vector)).sum()),
            "vector_sha256": hashlib.sha256(vector.numpy().tobytes()).hexdigest(),
        })
        scenes.drop(scan_id)

    report = {
        "producer": "experiments/representation/diagnose_wall_vectors.py",
        "git_commit": git_commit(),
        "parses": {"path": relative_path(parses), "sha256": sha256(parses)},
        "representation": "baseline",
        "label_type": "gt",
        "calibration": "none (raw encoder vectors)",
        "zero_criterion": "all entries equal exactly zero; no tolerance",
        "wall_selection": "foreground boxes: max(dx,dy,dz)>10m and min(dx,dy)<0.5m",
        "n_scenes": len(rows),
        "zero_scenes": sum(row["all_zero"] for row in rows),
        "no_heuristic_wall_scenes": sum(row["n_heuristic_walls"] == 0 for row in rows),
        "no_annotated_wall_scenes": sum(row["n_annotated_walls"] == 0 for row in rows),
        "rows": rows,
    }
    out = resolve_under_root(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Zero baseline wall vector: {report['zero_scenes']}/{len(rows)} scenes")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
