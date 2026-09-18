"""Row files, sidecars and the paths every reference-frame stage reads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from grounding.parsing.fields import relation_objects
from grounding.paths import DATA, resolve_under_root

REPO = Path(__file__).resolve().parents[2]


def load_jsonl(path) -> list[dict]:
    """Rows from a JSONL file, failing on the line that is malformed."""
    path = resolve_under_root(path)
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no}: {e}") from e
    return rows


def read_json(path) -> Optional[dict]:
    """A sidecar's contents, or None when it was never written."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def repo_relative(path) -> str:
    """Use repository-relative paths where possible in frozen artifacts."""
    path = Path(path)
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def digest(value) -> str:
    """Stable digest for a JSON value or a CPU/GPU tensor."""
    h = hashlib.sha256()
    if hasattr(value, "detach"):
        a = value.detach().cpu().contiguous().numpy()
        h.update(str(a.dtype).encode())
        h.update(str(tuple(a.shape)).encode())
        h.update(a.tobytes())
    else:
        h.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


def has_lateral_relation(node):
    if not isinstance(node, dict):
        return False
    for relation in node.get("relations") or []:
        if str(relation.get("relation_name") or "").lower() in {"left", "right"}:
            return True
        if any(has_lateral_relation(child) for child in relation_objects(relation)):
            return True
    return False


def selection_key(row):
    row_id = str(row.get("row_idx", row.get("dataset_idx")))
    return hashlib.sha256(row_id.encode()).hexdigest(), int(row_id)


REFFRAME_ROWS = DATA / "probes" / "reference_frame" / "refframe_rows.jsonl"
REFFRAME_V4 = DATA / "parses" / "reference_frame" / "refframe_30b_v4.jsonl"
REFFRAME_FRAMES = DATA / "parses" / "reference_frame" / "refframe_30b_frames.jsonl"
REFFRAME_PROBE = DATA / "probes" / "reference_frame" / "refframe_probe.jsonl"


# The doorway diagnostic uses the reference-frame train population, not test
DOORWAY_ROWS = DATA / "probes" / "reference_frame" / "doorway_rows.jsonl"


DOORWAY_V4 = DATA / "parses" / "reference_frame" / "doorway_30b_v4.jsonl"


#: Frame-pass output for doorway rows; `doorway` gates on it and checks its prompt
DOORWAY_FRAMES = DATA / "parses" / "reference_frame" / "doorway_30b_frames.jsonl"
