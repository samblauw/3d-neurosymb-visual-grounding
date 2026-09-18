#!/usr/bin/env python3
"""Compare relation-feature tensors concept by concept.

The default inputs check regenerated baseline features against the reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from grounding.resolution.harness import (FEATURE_FILES,
                                          REGENERATED_FEATURE_FILES, tree_dir)

_BASELINE = "baseline"


def load(p: Path) -> dict:
    if not p.exists():
        raise SystemExit(
            f"No such tensor: {p}\n"
            f"  build it with scripts/evaluation/build_features.py")
    return torch.load(p, map_location="cpu", weights_only=False)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", nargs="?", default=None,
                    help=f"reference tensor (default: {FEATURE_FILES[_BASELINE]})")
    ap.add_argument("new", nargs="?", default=None,
                    help=f"tensor to check (default: {REGENERATED_FEATURE_FILES[_BASELINE]})")
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--show", type=int, default=10,
                    help="max differing concepts to list")
    args = ap.parse_args()

    ref_path = Path(args.ref) if args.ref else tree_dir(_BASELINE) / FEATURE_FILES[_BASELINE]
    new_path = Path(args.new) if args.new else tree_dir(_BASELINE) / REGENERATED_FEATURE_FILES[_BASELINE]
    print(f"ref: {ref_path.name}\nnew: {new_path.name}\n")

    ref, new = load(ref_path), load(new_path)

    only_ref, only_new = set(ref) - set(new), set(new) - set(ref)
    shared = sorted(set(ref) & set(new))
    print(f"Scenes: {len(ref)} ref, {len(new)} new, {len(shared)} shared")

    n_concepts = n_same = 0
    missing, extra, mismatched = set(), set(), []

    for s in shared:
        a, b = ref[s], new[s]
        missing |= set(a) - set(b)
        extra |= set(b) - set(a)

        for c in set(a) & set(b):
            n_concepts += 1
            x, y = a[c].float(), b[c].float()

            if x.shape != y.shape:
                mismatched.append((s, c, float("inf"),
                                   f"shape {list(x.shape)} != {list(y.shape)}"))
                continue

            if torch.allclose(x, y, atol=args.atol, rtol=args.rtol):
                n_same += 1
            else:
                d = (x - y).abs()
                max_d, mean_d = float(d.max()), float(d.mean())
                mismatched.append((s, c, max_d, f"maxabs {max_d:.3e} mean {mean_d:.3e}"))

    print(f"Concepts compared: {n_concepts} | Identical: {n_same} | "
          f"Differing: {len(mismatched)}")

    if missing:
        print(f"  In ref only ({len(missing)}): {sorted(missing)[:args.show]}")
    if extra:
        print(f"  In new only ({len(extra)}): {sorted(extra)[:args.show]}")
    if mismatched:
        mismatched.sort(key=lambda t: -t[2])
        print(f"\n  Top {min(args.show, len(mismatched))} differences by max abs error:")
        for s, c, _, why in mismatched[:args.show]:
            print(f"    {s}  {c:<22} {why}")

    is_clean = not (mismatched or missing or extra or only_ref or only_new)
    print("\nIDENTICAL — regeneration reproduces the reference" if is_clean
          else "\nDIFFERS — inspect gaps before trusting the control")
    return 0 if is_clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
