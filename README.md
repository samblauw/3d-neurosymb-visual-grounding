# Symbolic 3D grounding on Nr3D

This project turns referring expressions into symbolic programs that identify
objects in ScanNet scenes. It extends LaSP with parser adaptation, colour,
isolation, wall proximity, ordinal features and score calibration. Reference-frame
interpretation is studied in a separate experiment.

## Setup

Use Python 3.10 or later. Run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[evaluation]"
```

For MLX parser generation and LoRA training, use a separate Apple Silicon
environment with `pip install -e ".[training]"`. The evaluation dependencies pin
an older Transformers version and should not be combined with the training extra.
Checkpoint scoring also needs compatible evaluation packages; that combined
setup is not specified here. Dependency versions are not a validated lock.

Data, model weights, adapters and generated results are not included. You need:

- Aligned ReferIt3D scene exports under `data/raw/referit3d/scan_data/`, including
  points, instance locations, labels and colour mixtures.
- Annotations and split lists under `data/raw/`, including
  `data/raw/referit3d/annotations/`, and annotation CSVs under `data/corpora/`.
- CLIP weights for feature extraction and Qwen weights for parser generation.

Prepare the scene exports with external ScanNet/ReferIt3D tooling. Then create
local baseline data links and follow the benchmark builder's input options:

```bash
python scripts/prepare/init_tree.py
python scripts/prepare/build_benchmark.py --help
```

Use `init_tree.py --check` to check the links and `build_benchmark.py --verify`
to check existing caches. Neither supplies the dataset. Keep the frozen dev and
validation splits when comparing runs.

## Running experiments

With development parses available, build features and compare the two systems:

```bash
python scripts/evaluation/build_features.py \
  --parses data/parses/dev/nr3d_dev_30b_v4.jsonl --split dev
python scripts/evaluation/compare_baseline.py \
  --parses data/parses/dev/nr3d_dev_30b_v4.jsonl --split dev --totals parses \
  --norm-mode zscore_sigmoid --out-dir artifacts/results/representation/dev_comparison
```

Feature building replaces the cache for that system and split. Evaluate using
the same parses. `--totals parses` scores the supplied subset; `--totals official`
counts missing test parses as misses.

To generate parses in the MLX environment:

```bash
python -m experiments.parser_adaptation.parse_split \
  --representation extended --style teacher --prompt v4 \
  --model mlx-community/Qwen3-30B-A3B-4bit \
  --rows data/corpora/nr3d_test.csv --out data/parses/test/nr3d_test_30b_v4.jsonl
```

The experiments are grouped by contribution:

| Folder under `experiments/` | Purpose |
| --- | --- |
| `parser_adaptation/` | Rollouts, training sets, LoRA training and adapter selection |
| `representation/` | Feature probes, ablations and sensitivity sweeps |
| `calibration/` | Normalization, anchor binding and combination rules |
| `reference_frame/` | Observer-frame and doorway-entry experiments |

Each command has `--help`. For example:

```bash
python -m experiments.parser_adaptation.train_sft_arms --dry-run
python -m experiments.representation.probe_feature --help
python -m experiments.calibration.score_aggregation --help
python -m experiments.reference_frame --help
```

Training uses the R=8 teacher pool and the IDs in
`experiments/parser_adaptation/validation_split.json`. Recovery rollouts are
excluded. Remove `--dry-run` to train; select adapters on dev data, then pass the
chosen adapter to `parse_split --style sft --adapter <path>`.

Checkpoint selection credits every top tie; headline evaluation selects one
candidate with `torch.topk`. Preserve the scoring settings and row-level results
when comparing experiments.

## Results

The maintained run is `sft_v4_ext_random_size_matched__seed2`. Its recorded test
accuracy is **64.28%** (4,811/7,485), versus **61.88%** (4,632/7,485) for the
baseline using the same parses.

Local results live in `artifacts/results/`, grouped into `final/`, `baselines/`,
`parser_adaptation/`, `representation/`, `calibration/` and `reference_frame/`.
`final/manifest.json` identifies the maintained comparison, predictions and
significance report. Generated results are Git-ignored.

## Contributing

`src/grounding/` contains the parser, scene representation and resolver.
Inherited components live in `src/grounding/_vendor/lasp/`; `baselines/lasp/`
retains independent baseline scoring. Both include local loading adaptations.
Data preparation and evaluation commands live in `scripts/`.

Include regression coverage for behavior changes, and keep datasets, weights,
generated outputs and credentials out of commits. The project was developed with AI coding
assistance.

## License

Original contributions use the [MIT License](LICENSE). Inherited LaSP code retains
its copyright in [baselines/lasp/LICENSE](baselines/lasp/LICENSE) and
[src/grounding/_vendor/lasp/LICENSE](src/grounding/_vendor/lasp/LICENSE).
ScanNet and ReferIt3D data must be obtained under their own terms.
