#!/usr/bin/env python3
"""Recompute parser validation loss over chosen token spans of saved adapters.

Forward passes only: no optimizer, no weight update, nothing written except the
optional --out file, which must not already exist. Token ids, padding and
the loss follow mlx_lm 0.31.3 (`ChatDataset.process`, `iterate_batches`,
`default_loss`). Four spans are reported, each a token-weighted mean
cross-entropy over the tokens it contains:

  mlx        exactly the `default_loss` mask with mask_prompt=false, i.e. what
             `mlx_lm lora` logs as "Val loss": `steps <= lengths` also keeps
             the target one past the last token, a padding id, per row
  full       every real token after the first
  assistant  from the end of the `add_generation_prompt=True` prefix to the end
             (what `--mask-prompt` would train on: includes the empty think
             block the Qwen3 template writes into the final assistant turn)
  program    the JSON program and its closing <|im_end|>; the prefix is rendered
             with `enable_thinking=False`, as parse_split.py prompts at inference

Span offsets are lengths of the templated prefix and are checked to be a token
prefix of the templated conversation; a row where they are not stops the run.

--rows all scores every validation row. --rows logged scores the 25 batches
`mlx_lm lora` drew for the "Val loss" line at --logged-iter, rebuilt from the
run's seed. That line is computed before that iteration's update, so a saved
NNNNNNN_adapters.safetensors is one update later than the logged value.
Omitting --adapter scores the base model, which at iteration 1 is exactly
the model that produced the first logged value (LoRA B starts at zero).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from grounding.resolution.artifacts import sha256

REPO = Path(__file__).resolve().parents[2]
SPANS = ("mlx", "full", "assistant", "program")
PAD_TO = 32  # mlx_lm.tuner.trainer.iterate_batches


def load_run_config(args: argparse.Namespace) -> dict:
    """Take run settings from the adapter's own adapter_config.json."""
    if args.adapter is None:
        if args.data is None:
            sys.exit("--data is required when no --adapter is given")
        return {"model": args.model, "data": str(args.data), "seed": args.seed,
                "batch_size": 4, "max_seq_length": 2048, "val_batches": 25,
                "steps_per_eval": 100, "iters": 1000, "mask_prompt": False}
    config = json.loads((args.adapter / "adapter_config.json").read_text())
    if args.data is not None:
        config["data"] = str(args.data)
    if args.seed is not None:
        config["seed"] = args.seed
    return config


def logged_iterations(steps_per_eval: int, iters: int) -> list[int]:
    """Iterations at which mlx_lm.tuner.trainer.train runs evaluate()."""
    return [it for it in range(1, iters + 1)
            if it == 1 or it % steps_per_eval == 0 or it == iters]


def logged_batches(config: dict, n_valid: int, iteration: int) -> list[int]:
    """Replay the global numpy draws of `mlx_lm lora` up to that evaluation.

    run() seeds numpy with the run seed; the training iterator draws one
    permutation of its batches before the first step, and every evaluate()
    draws one permutation of the validation batches and keeps the first
    val_batches. 1000 updates stay inside the first training epoch, so no
    second training permutation interleaves.
    """
    batch_size = config["batch_size"]
    train_path = REPO / config["data"] / "train.jsonl"
    with train_path.open() as f:
        n_train = sum(1 for _ in f)
    n_train_batches = n_train // batch_size
    if config["iters"] > n_train_batches:
        sys.exit("run spans more than one training epoch; the draw order "
                 "is not replayed for that case")
    iterations = logged_iterations(config["steps_per_eval"], config["iters"])
    if iteration not in iterations:
        sys.exit(f"--logged-iter {iteration} is not an evaluated iteration: "
                 f"{iterations}")
    np.random.seed(config["seed"])
    np.random.permutation(n_train_batches)
    for _ in range(iterations.index(iteration) + 1):
        drawn = np.random.permutation(n_valid // batch_size)
    return drawn[:config["val_batches"]].tolist()


def encode(rows: list[dict], tokenizer) -> list[dict]:
    encoded = []
    for index, row in enumerate(rows):
        messages = row["messages"]
        if messages[-1]["role"] != "assistant":
            sys.exit(f"valid row {index} does not end in an assistant turn")
        tokens = tokenizer.apply_chat_template(messages, return_dict=False)
        assistant = tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, return_dict=False)
        program = tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, enable_thinking=False,
            return_dict=False)
        for name, prefix in (("assistant", assistant), ("program", program)):
            if tokens[:len(prefix)] != prefix:
                sys.exit(f"valid row {index}: the {name} prefix is not a token "
                         "prefix of the templated conversation")
        end = len(tokens) - tokens[::-1].index(tokenizer.eos_token_id)
        if end <= len(program):
            sys.exit(f"valid row {index}: no {tokenizer.eos_token!r} after the program")
        encoded.append({"tokens": tokens, "spans": {
            # len(tokens) is padding; batches pad past the longest row, so that position exists
            "mlx": (1, len(tokens) + 1),
            "full": (1, len(tokens)),
            "assistant": (len(assistant), len(tokens)),
            "program": (len(program), end),
        }})
    return encoded


def batches(n_rows: int, batch_size: int, mode: str, drawn: list[int] | None):
    # iterate_batches sorts by raw JSON key count, not token count; equal counts keep file order
    order = list(range(n_rows))
    groups = [order[i:i + batch_size]
              for i in range(0, n_rows - batch_size + 1, batch_size)]
    if mode == "logged":
        return [groups[i] for i in drawn]
    covered = batch_size * len(groups)
    return groups + ([order[covered:]] if covered < n_rows else [])


def score(model, encoded: list[dict], groups, max_seq_length: int) -> dict:
    import mlx.core as mx
    import mlx.nn as nn

    totals = {span: 0.0 for span in SPANS}
    counts = {span: 0 for span in SPANS}
    for group in groups:
        rows = [encoded[i] for i in group]
        longest = max(len(row["tokens"]) for row in rows)
        if longest > max_seq_length:
            sys.exit(f"a row has {longest} tokens > max_seq_length "
                     f"{max_seq_length}; truncation is not replayed")
        width = min(1 + PAD_TO * ((longest + PAD_TO - 1) // PAD_TO), max_seq_length)
        batch = np.zeros((len(rows), width), np.int32)
        for j, row in enumerate(rows):
            batch[j, :len(row["tokens"])] = row["tokens"]
        batch = mx.array(batch)
        logits = model(batch[:, :-1])
        # Same dtype path as default_loss: loss in model dtype, summed in float32
        ce = nn.losses.cross_entropy(logits, batch[:, 1:]).astype(mx.float32)
        # Position t predicts token t + 1, so tokens [start, end) are scored at [start - 1, end - 1)
        positions = mx.arange(ce.shape[1])
        sums = {}
        for span in SPANS:
            starts = mx.array([row["spans"][span][0] - 1 for row in rows])[:, None]
            ends = mx.array([row["spans"][span][1] - 1 for row in rows])[:, None]
            mask = (positions >= starts) & (positions < ends)
            sums[span] = (ce * mask).sum()
            counts[span] += sum(row["spans"][span][1] - row["spans"][span][0]
                                for row in rows)
        mx.eval(sums)
        for span in SPANS:
            totals[span] += sums[span].item()
    return {span: {"loss": totals[span] / counts[span],
                   "perplexity": math.exp(totals[span] / counts[span]),
                   "tokens": counts[span]} for span in SPANS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path,
                    help="adapter directory (reads its adapter_config.json); "
                         "omit to score the base model")
    ap.add_argument("--weights", type=Path,
                    help="checkpoint file to score with --adapter's config, e.g. "
                         "an archived 0000400_adapters.safetensors (default: "
                         "the adapter's adapters.safetensors)")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B",
                    help="base model when no --adapter is given")
    ap.add_argument("--data", type=Path,
                    help="training-set directory holding valid.jsonl "
                         "(default: the adapter's recorded data path)")
    ap.add_argument("--seed", type=int,
                    help="run seed for --rows logged (default: the adapter's)")
    ap.add_argument("--rows", choices=("all", "logged"), default="all")
    ap.add_argument("--logged-iter", type=int, default=1000)
    ap.add_argument("--out", type=Path, help="write the report here; must not exist")
    args = ap.parse_args()

    if args.out is not None and args.out.exists():
        sys.exit(f"{args.out} exists; not overwriting")
    config = load_run_config(args)
    if config.get("mask_prompt"):
        print("note: this run trained with mask_prompt=true", file=sys.stderr)
    if args.rows == "logged" and config["seed"] is None:
        sys.exit("--rows logged needs the run seed (--seed)")

    valid_path = REPO / config["data"] / "valid.jsonl"
    rows = [json.loads(line) for line in valid_path.open() if line.strip()]
    if len({len(row) for row in rows}) != 1:
        sys.exit("valid rows have differing key counts; mlx_lm's batch order "
                 "is not file order and is not replayed here")

    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import load_adapters
    model, tokenizer = load(config["model"])
    weights = None
    if args.adapter is not None:
        weights = args.weights or args.adapter / "adapters.safetensors"
        # load_adapters loads non-strictly, so the chosen file is checked key- and shape-wise first
        model = load_adapters(model, str(args.adapter))
        lora = {k: v for k, v in tree_flatten(model.parameters()) if "lora_" in k}
        saved = mx.load(str(weights))
        if (saved.keys() != lora.keys()
                or any(saved[k].shape != lora[k].shape for k in lora)):
            sys.exit(f"{weights} does not match the LoRA layers of {args.adapter}")
        model.load_weights(list(saved.items()), strict=False)
    model.eval()

    encoded = encode(rows, tokenizer)
    drawn = (logged_batches(config, len(rows), args.logged_iter)
             if args.rows == "logged" else None)
    groups = batches(len(rows), config["batch_size"], args.rows, drawn)
    metrics = score(model, encoded, groups, config["max_seq_length"])

    report = {
        "model": config["model"],
        "adapter": None if args.adapter is None else str(args.adapter),
        "weights": None if weights is None else str(weights),
        "weights_sha256": None if weights is None else sha256(weights),
        "valid": str(valid_path.relative_to(REPO)),
        "rows": args.rows,
        "logged_iter": args.logged_iter if args.rows == "logged" else None,
        "examples": sum(len(group) for group in groups),
        "metrics": metrics,
    }
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
