#!/usr/bin/env python3
"""Generate parse rollouts for Nr3D training utterances."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

from grounding.parsing import PROMPTS, extract_json_objects
from grounding.parsing.generation import load_model, template_prompt


def load_rows(path: Path, limit: int) -> list[dict]:
    """Load and deterministically shuffle Nr3D rows."""
    rows = []

    with path.open(newline="") as f:
        for row_idx, row in enumerate(csv.DictReader(f)):
            utterance = (row.get("utterance") or "").strip()
            if not utterance:
                continue

            rows.append({
                "row_idx": row_idx,
                "stimulus_id": row.get("stimulus_id", ""),
                "scan_id": row.get("scan_id", ""),
                "target_id": row.get("target_id", ""),
                "instance_type": row.get("instance_type", ""),
                "utterance": utterance,
            })

    random.Random(42).shuffle(rows)

    return rows[:limit] if limit > 0 else rows


def completed_rows(path: Path) -> set[int]:
    """Return row indices already present in a rollout file."""
    if not path.exists():
        return set()

    done = set()

    with path.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["row_idx"])
            except (json.JSONDecodeError, KeyError):
                continue

    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/corpora/nr3d_train.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--prompt",
        choices=sorted(PROMPTS),
        default="v3",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-30B-A3B-4bit",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2800,
        help="Number of utterances to process; 0 uses all rows.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Maximum number of generated sequences per batch.",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = Path(
            f"data/rollouts/nr3d_rollouts_{args.prompt}.jsonl"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    from mlx_lm import batch_generate

    rows = load_rows(args.input, args.limit)

    done = completed_rows(args.output)
    rows = [row for row in rows if row["row_idx"] not in done]

    if done:
        print(
            f"Resuming: {len(done)} rows already complete, "
            f"{len(rows)} remaining."
        )

    if not rows:
        print("Nothing to do.")
        return 0

    print(f"Loading {args.model} ...")
    model, tokenizer, sampler = load_model(args.model, temperature=args.temp)

    system = PROMPTS[args.prompt]

    prompts = [
        template_prompt(tokenizer, system,
                        f"Sentence: {row['utterance']}\n\nParsed JSON:")
        for row in rows
    ]

    # Each row contributes n_samples sequences to a generation batch
    rows_per_batch = max(1, args.batch_size // args.n_samples)

    total_samples = 0
    valid_samples = 0
    total_tokens = 0
    start_time = time.time()

    with args.output.open("a") as fout:
        for start in range(0, len(rows), rows_per_batch):
            group = rows[start:start + rows_per_batch]

            batch_prompts = []
            for i in range(len(group)):
                batch_prompts.extend(
                    [prompts[start + i]] * args.n_samples
                )

            response = batch_generate(
                model,
                tokenizer,
                prompts=batch_prompts,
                max_tokens=args.max_tokens,
                sampler=sampler,
                verbose=False,
            )

            for i, row in enumerate(group):
                raw_samples = response.texts[
                    i * args.n_samples:(i + 1) * args.n_samples
                ]

                samples = []

                for raw in raw_samples:
                    parsed = extract_json_objects(raw)

                    samples.append({
                        "raw": raw,
                        "parsed": parsed,
                    })

                    total_samples += 1
                    valid_samples += parsed is not None
                    total_tokens += len(tokenizer.encode(raw))

                fout.write(json.dumps({
                    "row_idx": row["row_idx"],
                    "stimulus_id": row["stimulus_id"],
                    "scan_id": row["scan_id"],
                    "target_id": row["target_id"],
                    "instance_type": row["instance_type"],
                    "utterance": row["utterance"],
                    "samples": samples,
                }) + "\n")

            fout.flush()

            completed = min(start + len(group), len(rows))
            elapsed = time.time() - start_time

            sample_rate = total_tokens / elapsed if elapsed else 0
            valid_pct = (
                100 * valid_samples / total_samples
                if total_samples else 0
            )

            print(
                f"[{completed}/{len(rows)}] "
                f"{sample_rate:.1f} tok/s | "
                f"valid JSON {valid_pct:.0f}%",
                file=sys.stderr,
            )

    elapsed = time.time() - start_time

    print(
        f"Done. {len(rows)} prompts x {args.n_samples} "
        f"= {total_samples} samples "
        f"({valid_samples} parsed). "
        f"{total_tokens / elapsed:.1f} tok/s. "
        f"Wrote {args.output}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
