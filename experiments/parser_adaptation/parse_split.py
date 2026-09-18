"""Parse CSV or JSONL rows and write lowered executable programs.

Lowering uses the selected representation's vocabulary. Rebuild features from
the output before evaluation; existing rows are skipped unless ``--fresh`` is set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path


from grounding.paths import REPO_ROOT
from grounding.parsing import (REPRESENTATIONS, extract_json_objects, load_rel_num,
                               lower_one, normalise_representation,
                               sft_user_turn)
from grounding.parsing import PROMPTS
from grounding.parsing.generation import encode_chat, load_model, template_prompt
from grounding.resolution.artifacts import git_commit, sha256

# style -> (default prompt version, chat-template?, user message, recover?)
#
# ``sft`` is deliberately opt-in.  The older fused/teacher commands retain
# their prompts and repair policy exactly, while this path mirrors the v4 SFT
# conditioning and its non-repairing clean/lower step
STYLES = {
    "fused":   ("v2", False, "Parse: '{s}'", True),
    "teacher": ("v3", True,  'Input: "{s}"\nOutput:', True),
    "sft":     ("v4", False, None, False),
}


def load_rows(path: Path, limit: int = 0) -> tuple[list[dict], bool]:
    """Rows to parse, and whether they carry a dataset_idx worth writing out."""
    rows = []
    if path.suffix == ".csv":
        with open(path, newline="") as f:
            for idx, r in enumerate(csv.DictReader(f)):
                rows.append({
                    "dataset_idx": idx,
                    "row_idx": idx,
                    "scan_id": r["scan_id"],
                    "target_id": int(r["target_id"]),
                    "target_label": r["instance_type"],
                    "sentence": r["utterance"],
                })
        keyed = True
    else:
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                rows.append({
                    "row_idx": r["row_idx"],
                    "scan_id": r["scan_id"],
                    "target_id": int(r["target_id"]),
                    "target_label": r.get("target_label"),
                    # probe files say "sentence", corpora say "utterance"; both are inputs
                    "sentence": r.get("sentence") or r["utterance"],
                })
        # A probe row_idx indexes the TRAIN csv, so dataset_idx would look up an unrelated split
        keyed = False
    return (rows[:limit] if limit else rows), keyed


def invocation(args, rows_path: Path, version: str, system: str,
               representation: str, n_rows: int) -> dict:
    """Everything that decides what a parse row looks like, recorded once.

    Written next to --out as `<out>.meta.json` so a frozen parse says how it was
    made. A resume is refused unless this record matches the one on disk: two
    invocations that differ in any decoding setting cannot share one file.
    """
    return {
        "producer": "experiments/parser_adaptation/parse_split.py",
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "rows": {"path": str(rows_path.relative_to(REPO_ROOT))
                 if rows_path.is_relative_to(REPO_ROOT) else str(rows_path),
                 "sha256": sha256(rows_path), "n": n_rows, "limit": args.limit},
        "model": args.model, "adapter": args.adapter,
        "representation": representation,
        "style": args.style, "prompt": version,
        "prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "decoding": {"sampler": "greedy", "temp": 0.0,
                     "max_tokens": args.max_tokens, "batch_size": args.batch_size,
                     "no_think": bool(args.no_think)},
    }


def done_rows(path: Path) -> set:
    """row_idx values already written, so a resumed run skips them."""
    done = set()
    if not path.exists():
        return done
    for line in open(path):
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["row_idx"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--representation", dest="representation",
                    required=True, choices=list(REPRESENTATIONS),
                    help="whose rel_num vocabulary the parses lower against: "
                         "baseline | extended. Required, "
                         "not defaulted -- it decides what survives lowering")
    ap.add_argument("--rows", default="data/corpora/nr3d_test.csv",
                    help="utterances to parse: the test csv, or a probe jsonl")
    # Defaults describe the zero-shot 1.7B student; the full system supplies --adapter explicitly
    ap.add_argument("--out", default="data/parses/test/nr3d_test_qwen3_1p7b_v4_ext.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None,
                    help="optional MLX LoRA adapter directory")
    ap.add_argument("--style", default="sft", choices=sorted(STYLES),
                    help="prompt formatting recipe; see the module docstring")
    ap.add_argument("--prompt", default=None, choices=sorted(PROMPTS),
                    help="prompt version; default is the style's")
    # A MoE model reads more expert weights per step, so batch trades sparsity for memory
    ap.add_argument("--batch-size", type=int, default=8,
                    help="prompts per batch_generate call")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--fresh", action="store_true",
                    help="overwrite --out instead of resuming into it")
    ap.add_argument("--no-think", action="store_true",
                    help="append ' /no_think' to each prompt. Required for a "
                         "RAW Qwen3 base; omit for SFT-derived models.")
    args = ap.parse_args()

    rows_path = Path(args.rows)
    rows_path = rows_path if rows_path.is_absolute() else REPO_ROOT / rows_path
    if rows_path.suffix == ".csv":
        assert "train" not in rows_path.name.lower(), (
            f"rows look like a train split: {args.rows!r}")

    version, chat, user_msg, recover = STYLES[args.style]
    if args.style == "sft" and args.prompt not in (None, "v4"):
        ap.error("--style sft always uses the v4 SFT system prompt")
    version = args.prompt or version
    system = PROMPTS[version]
    # sft_user_turn is the shared implementation behind training_set.user_turn
    user_turn = sft_user_turn if args.style == "sft" else None
    representation = normalise_representation(args.representation)
    rel_num = load_rel_num(representation)

    rows, keyed = load_rows(rows_path, args.limit)
    out_path = Path(args.out)
    out_path = out_path if out_path.is_absolute() else REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(str(out_path) + ".meta.json")
    if args.fresh:
        for p in (out_path, meta_path):
            if p.exists():
                p.unlink()

    meta = invocation(args, rows_path, version, system, representation, len(rows))
    # A file with rows but no record of how they were made, or a record that
    # disagrees with this invocation, cannot be resumed into: the rows already
    # there and the rows about to be appended would be two populations under
    # one name. `--fresh` starts over; a different --out keeps both
    if out_path.exists() and done_rows(out_path):
        if not meta_path.exists():
            raise SystemExit(
                f"{out_path} has rows but no {meta_path.name}: its invocation is "
                "unknown, so this run cannot be shown to match it. Pass --fresh "
                "to start over, or a different --out.")
        recorded = json.loads(meta_path.read_text())
        drift = {k: (recorded.get(k), meta[k]) for k in
                 ("rows", "model", "adapter", "representation", "style",
                  "prompt", "prompt_sha256", "decoding")
                 if recorded.get(k) != meta[k]}
        if drift:
            raise SystemExit(
                f"{out_path} was written by a different invocation; refusing "
                f"to mix rows. Differs in: {json.dumps(drift, indent=1)}\n"
                "Pass --fresh to start over, or a different --out.")
    elif not meta_path.exists() or args.fresh:
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    already = done_rows(out_path)
    todo = [r for r in rows if r["row_idx"] not in already]

    print(f"{len(rows)} utterances, {len(already)} already done, "
          f"{len(todo)} to do -> {out_path.name}")
    print(f"prompt {version} ({len(system)} chars), style {args.style} | "
          f"{len(rel_num)} relations in {representation}'s vocabulary")
    if args.adapter:
        print(f"adapter {args.adapter}")
    if not todo:
        print("Nothing to do.")
        return 0

    from mlx_lm import batch_generate

    print(f"Loading {args.model} ...")
    # Greedy decoding, so arms differ by prompt only
    model, tok, sampler = load_model(args.model, args.adapter)
    suffix = " /no_think" if args.no_think else ""

    def build_prompt(sentence: str):
        content = ((user_turn(sentence) if user_turn else user_msg.format(s=sentence))
                   + suffix)
        return (template_prompt(tok, system, content) if chat
                else encode_chat(tok, system, content))

    n, n_fail = len(todo), 0
    dropped: Counter = Counter()   # predicates the engine has no encoder for
    t0 = time.time()

    with open(out_path, "a") as out:
        for start in range(0, n, args.batch_size):
            chunk = todo[start:start + args.batch_size]
            resp = batch_generate(
                model, tok, prompts=[build_prompt(r["sentence"]) for r in chunk],
                max_tokens=args.max_tokens, sampler=sampler, verbose=False)
            for r, raw in zip(chunk, resp.texts):
                text = raw.split("<|im_end|>")[0]
                # The legacy parser styles preserve their repairing behaviour
                # The SFT-v4 path instead matches the supervised program
                # pipeline: omitted constraints remain omitted, not
                # being inferred from the sentence at test time
                json_obj = lower_one(extract_json_objects(text), r["sentence"],
                                     rel_num, recover=recover, dropped=dropped)
                if json_obj is None:
                    n_fail += 1
                # Unparsed rows still written, json_obj null, keeping the full denominator
                row = {"row_idx": r["row_idx"], "scan_id": r["scan_id"],
                       "target_id": r["target_id"],
                       "target_label": r["target_label"],
                       "sentence": r["sentence"], "json_obj": json_obj,
                       "raw": text}
                if keyed:
                    row["dataset_idx"] = r["dataset_idx"]
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

            done = min(start + args.batch_size, n)
            rate = done / max(time.time() - t0, 1e-9)
            print(f"[{done}/{n}] failures {n_fail}  {rate * 60:.1f} rows/min  "
                  f"ETA {(n - done) / rate / 60:.0f}m", flush=True)

    meta["written"] = {"rows": len(done_rows(out_path)), "failures_this_run": n_fail,
                       "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    meta["output_sha256"] = sha256(out_path)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\nWrote {out_path}  (+ {meta_path.name})")
    print(f"  {n - n_fail}/{n} utterances produced a runnable program "
          f"({n_fail} failures)")
    if dropped:
        total = sum(dropped.values())
        print(f"  {total} predicate mentions dropped — no encoder in "
              f"{representation} for these names:")
        for name, count in dropped.most_common(15):
            print(f"      {name:<20} {count}")
        print("  (each dropped mention is one constraint the engine never sees)")
    else:
        print("  every predicate emitted maps onto an encoder")
    try:
        out_display = out_path.relative_to(REPO_ROOT)
    except ValueError:
        out_display = out_path
    print(f"\nRebuild features before scoring:\n"
          f"  python scripts/evaluation/build_features.py --representation "
          f"{representation} --parses {out_display}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
