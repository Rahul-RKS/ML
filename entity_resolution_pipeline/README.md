# Business Entity Resolution - Amazon ML Challenge 2026

For every Source-1 entity, this pipeline finds the matching records in Source-2
and Source-3. It is built for macro F_0.5 (which weights precision twice) and for
keeping the candidate set per Source-1 entity small.

## Approach

Normalize names/addresses, block by country with hashed character n-grams to get
top-k candidates per entity, build pairwise similarity features, train a
gradient-boosted classifier, and pick the decision threshold that maximizes macro
F_0.5 on a held-out validation split.

No external data, APIs, or lookups are used. All libraries are BSD/Apache/MIT
licensed and the model is far under 8B parameters.

## Setup

```bash
pip install -r requirements.txt
```

## Data layout

```
dataset/train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

The dataset files are read only and never modified.

## Run

```bash
python src/run_pipeline.py --data-dir dataset --out-dir output --k 15 --min-sim 0.30
```

This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

## Run the metric unit tests

```bash
python src/evaluate.py
```

## Modules (`src/`)

| file | role |
|------|------|
| `preprocessing.py` | name/address normalization; country kept as an open string set |
| `blocking.py` | hashed char 3-4 gram top-k cosine per country, sparse batched scoring |
| `features.py` | rapidfuzz token-sort/set, Jaro-Winkler, Jaccard, postal + country flags |
| `evaluate.py` | F_0.5 metric + unit tests |
| `train.py` | entity-level split, HistGradientBoosting matcher, threshold sweep |
| `io_utils.py` | tab-separated I/O enforcing the submission format |
| `run_pipeline.py` | runs the whole flow |

## Tuning

- `--k` and `--min-sim` control candidate-set size. A smaller set scores better,
  so lower `k` until the training blocking recall starts to drop.
- The decision threshold is tuned automatically for macro F_0.5 on an
  entity-level validation split, so it is usually above 0.5.
- `--seed` (default 42) makes runs reproducible.
