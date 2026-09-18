#!/usr/bin/env python3
"""Create the three symlinks the inherited baseline reads its inputs through.

The baseline resolves data through ``baselines/lasp/{data,benchmark,output}``,
while the extended system reads the same trees directly through the package.
The links are Git-ignored, so every checkout has to make its own.

Nothing is fetched and no data directory is created: a bare checkout is
meant to lack ``data/`` and ``artifacts/``, and tests read that absence as
"not yet populated". A link may be made before its target exists.

Idempotent. A correct link is left alone; a link pointing elsewhere is
repaired; a real directory in a link's place is reported, never removed."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from grounding.paths import BASELINE_TREE, BENCHMARK, FEATURES, RAW, REPO_ROOT

# As documented in the README's tree-layout block.
LINKS = [(BASELINE_TREE / "data", RAW),
         (BASELINE_TREE / "benchmark", BENCHMARK),
         (BASELINE_TREE / "output", FEATURES)]


def ensure_link(link: Path, target: Path, check: bool) -> tuple[str, bool]:
    """Point ``link`` at ``target``. Returns the status and whether it is wrong."""
    relative = Path(os.path.relpath(target, link.parent))
    if link.is_symlink():
        if link.readlink() == relative:
            return "ok", False
        if check:
            return f"WRONG -> {link.readlink()}, expected {relative}", True
        link.unlink()
        link.symlink_to(relative, target_is_directory=True)
        return f"repaired -> {relative}", False
    if link.exists():
        # A real directory here holds somebody's data; not ours to delete.
        return "BLOCKED: exists and is not a symlink", True
    if check:
        return f"MISSING -> {relative}", True
    link.symlink_to(relative, target_is_directory=True)
    return f"created -> {relative}", False


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report what is missing or wrong without changing anything")
    args = ap.parse_args()

    if not BASELINE_TREE.is_dir():
        print(f"{BASELINE_TREE.relative_to(REPO_ROOT)}/ is not checked out")
        return 1

    problems = 0
    pending = []
    for link, target in LINKS:
        status, bad = ensure_link(link, target, args.check)
        problems += bad
        print(f"{link.relative_to(REPO_ROOT)}  {status}")
        if not target.exists():
            pending.append(target.relative_to(REPO_ROOT))

    if problems:
        print(f"\n{problems} link(s) need attention"
              + ("; rerun without --check to create them" if args.check else ""))
        return 1

    print("\nLinks ready.")
    if pending:
        print("Still to be supplied locally: "
              + ", ".join(f"{p}/" for p in pending))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
