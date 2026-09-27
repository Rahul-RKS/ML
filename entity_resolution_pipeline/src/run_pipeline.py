"""End-to-end pipeline: data -> blocking -> matching -> output.

Usage:
    python src/run_pipeline.py --data-dir dataset --out-dir output --k 15 --min-sim 0.30

Steps:
  1. Load and preprocess all sources (train and test).
  2. Block on train, build labeled pairs, split by entity, train the matcher,
     tune the F_0.5 threshold on held-out validation entities.
  3. Block on test with the same params -> candidate_pairs.tsv.
  4. Score test candidates, apply the tuned threshold, keep only candidates that
     were generated in blocking -> matching_results.tsv.

Writes both TSVs to --out-dir and prints validation F_0.5 and blocking recall.
"""

import argparse
import os

import pandas as pd

from preprocessing import preprocess_frame
from blocking import generate_candidates, candidates_to_map
from features import build_feature_matrix  # noqa: F401 (re-exported for tests)
from train import (
    split_entities, build_training_pairs, train_matcher,
    predict_scores, tune_threshold,
)
from evaluate import macro_f0_5_detailed
from io_utils import read_source, read_ground_truth, write_id_list_tsv


def _lookup(df, needed=None):
    """Map entity_id -> minimal dict of only the fields features.py needs.

    Only the 5 fields the features use are kept (not the whole row). If ``needed``
    is given, only those entity_ids are stored -- crucial at 10M-row scale, where
    a lookup over every candidate would eat many GB. After blocking we only ever
    need the candidates that actually survived, which is a small fraction.
    """
    ids = df["entity_id"].tolist()
    nc = df["name_core"].tolist()
    an = df["addr_norm"].tolist()
    po = df["postal"].tolist()
    co = df["country_norm"].tolist()
    hl = df["has_landmark"].tolist()
    out = {}
    for i, eid in enumerate(ids):
        if needed is not None and eid not in needed:
            continue
        out[eid] = {
            "name_core": nc[i], "addr_norm": an[i], "postal": po[i],
            "country_norm": co[i], "has_landmark": hl[i],
        }
    return out


def _blocking_recall(candidate_map, ground_truth):
    """Fraction of true matches that survive blocking (recall ceiling)."""
    total = found = 0
    for s1, truth in ground_truth.items():
        if not truth:
            continue
        cands = candidate_map.get(s1, set())
        total += len(truth)
        found += len(truth & cands)
    return (found / total) if total else 1.0


def _avg_candidates(candidate_map, s1_ids):
    if not s1_ids:
        return 0.0
    return sum(len(candidate_map.get(s, set())) for s in s1_ids) / len(s1_ids)


def _candidate_stats(candidate_map, s1_ids):
    """Print min/median/mean/p95/max candidates per entity + zero-candidate count."""
    import numpy as np
    counts = np.array([len(candidate_map.get(s, set())) for s in s1_ids])
    if counts.size == 0:
        print("    (no entities)")
        return
    print(f"    candidates/entity: min={counts.min()} median={int(np.median(counts))} "
          f"mean={counts.mean():.2f} p95={int(np.percentile(counts, 95))} "
          f"max={counts.max()} | zero-candidate entities={int((counts == 0).sum())}")


def run(data_dir, out_dir, k, min_sim, seed=42):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    # ---- 1. Load + preprocess -------------------------------------------- #
    print("[1/4] Loading & preprocessing...", flush=True)

    def _load(path):
        raw = read_source(path)
        print(f"      {os.path.basename(path)}: {len(raw)} rows -> normalizing...",
              flush=True)
        return preprocess_frame(raw)

    tr_s1 = _load(os.path.join(train_dir, "train_source1.tsv"))
    tr_s2 = _load(os.path.join(train_dir, "train_source2.tsv"))
    tr_s3 = _load(os.path.join(train_dir, "train_source3.tsv"))
    tr_cand = pd.concat([tr_s2, tr_s3], ignore_index=True)
    del tr_s2, tr_s3
    gt = read_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))

    # ---- 2. Blocking on train, train matcher, tune threshold ------------- #
    print("[2/4] Blocking (train) + training matcher...")
    tr_rows = generate_candidates(tr_s1, tr_cand, k=k, min_sim=min_sim)
    cand_sim = {(r["source1_entity_id"], r["candidate_entity_id"]): r["sim"] for r in tr_rows}
    cand_map = candidates_to_map(tr_rows)
    recall = _blocking_recall(cand_map, gt)
    print(f"    blocking recall (train): {recall:.4f}"
          f" | avg candidates/entity: {_avg_candidates(cand_map, list(tr_s1['entity_id'])):.2f}")

    train_ids, val_ids = split_entities(list(tr_s1["entity_id"]), seed=seed)
    labeled = build_training_pairs(cand_map, gt, cand_sim)
    train_rows = [r for r in labeled if r[0] in train_ids]
    val_rows = [r for r in labeled if r[0] in val_ids]

    # Build lookups only for the entities that actually appear in labeled pairs.
    # (Building a lookup over all ~10M candidates up front would exhaust memory.)
    need_s1 = {r[0] for r in labeled}
    need_cand = {r[1] for r in labeled}
    s1_lookup = _lookup(tr_s1, need_s1)
    cand_lookup = _lookup(tr_cand, need_cand)
    del tr_cand  # no longer needed once its lookup is built

    model = train_matcher(train_rows, s1_lookup, cand_lookup, seed=seed)
    val_scores = predict_scores(model, val_rows, s1_lookup, cand_lookup)
    best_t, best_f, table = tune_threshold(val_rows, val_scores, val_ids, gt)
    print(f"    tuned threshold: {best_t:.3f} | validation macro F0.5: {best_f:.4f}")
    print("    thr    F0.5    prec    rec    singleton")
    for t, f, p, r, s in table:
        mark = "  <-- best" if abs(t - best_t) < 1e-9 else ""
        print(f"    {t:.2f}  {f:.4f}  {p:.4f}  {r:.4f}  {s:.4f}{mark}")

    # ---- 3. Blocking on test -------------------------------------------- #
    print("[3/4] Blocking (test)...")
    te_s1 = _load(os.path.join(test_dir, "test_source1.tsv"))
    te_s2 = _load(os.path.join(test_dir, "test_source2.tsv"))
    te_s3 = _load(os.path.join(test_dir, "test_source3.tsv"))
    te_cand = pd.concat([te_s2, te_s3], ignore_index=True)
    del te_s2, te_s3

    te_rows = generate_candidates(te_s1, te_cand, k=k, min_sim=min_sim)
    te_cand_sim = {(r["source1_entity_id"], r["candidate_entity_id"]): r["sim"] for r in te_rows}
    te_cand_map = candidates_to_map(te_rows)
    te_s1_ids = list(te_s1["entity_id"])

    # ---- 4. Score, threshold, enforce subset-of-candidates -------------- #
    print("[4/4] Matching + writing output...")
    infer_rows = [(s1, cid, te_cand_sim.get((s1, cid), 0.0), 0)
                  for s1, cids in te_cand_map.items() for cid in cids]
    # Lookups only for entities in the surviving candidate pairs (see step 2).
    te_s1_lookup = _lookup(te_s1, {r[0] for r in infer_rows})
    te_cand_lookup = _lookup(te_cand, {r[1] for r in infer_rows})
    del te_cand
    scores = predict_scores(model, infer_rows, te_s1_lookup, te_cand_lookup)
    matches = {s1: set() for s1 in te_s1_ids}
    for (s1, cid, _sim, _lab), sc in zip(infer_rows, scores):
        if sc >= best_t:
            matches[s1].add(cid)

    write_id_list_tsv(os.path.join(out_dir, "candidate_pairs.tsv"),
                      te_cand_map, te_s1_ids, "candidate_entity_ids")
    write_id_list_tsv(os.path.join(out_dir, "matching_results.tsv"),
                      matches, te_s1_ids, "matched_entity_ids")
    print(f"    wrote {out_dir}/matching_results.tsv and candidate_pairs.tsv")

    # ---- Sanity report (row-count match, candidate stats, singletons) ---- #
    n_required = len(set(te_s1_ids))
    with open(os.path.join(out_dir, "matching_results.tsv"), encoding="utf-8") as f:
        n_written = sum(1 for _ in f) - 1  # minus header
    n_singleton = sum(1 for s1 in te_s1_ids if not matches.get(s1))
    print(f"    required test S1 entities: {n_required} | rows written: {n_written} "
          f"| {'MATCH' if n_required == n_written else 'MISMATCH!!!'}")
    print(f"    predicted: {n_required - n_singleton} with matches, "
          f"{n_singleton} singletons (empty)")
    print("    [blocking] ", end="")
    _candidate_stats(te_cand_map, te_s1_ids)
    if n_required != n_written:
        raise SystemExit("FATAL: output row count != test_source1 entity count. "
                         "Do NOT submit — investigate before validating.")
    return best_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../../dataset")
    ap.add_argument("--out-dir", default="../../output")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--min-sim", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.data_dir, args.out_dir, args.k, args.min_sim, args.seed)


if __name__ == "__main__":
    main()
