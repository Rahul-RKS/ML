"""TSV I/O helpers that enforce the exact submission format.

All reads use an explicit tab separator (a .tsv read without sep='\\t' silently
collapses into one column). All writes are tab-separated, UTF-8, comma-joined ID
lists with no quoting — exactly what the scorer and validate_submission.py expect.
"""

import os
import time
import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for col in SOURCE_COLS:
        if col not in df.columns:
            df[col] = ""
    return df


def read_ground_truth(path):
    """Return {source1_entity_id: set(matched_ids)} (empty set for singletons)."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    out = {}
    source1_ids = df["source1_entity_id"]
    match_lists = df["matched_entity_ids"]
    total = len(df)
    started = time.perf_counter()
    for row_number, (source1_id, raw) in enumerate(zip(source1_ids, match_lists), start=1):
        raw = raw.strip()
        out[source1_id] = set(
            x for x in raw.split(",") if x.strip()
        ) if raw else set()
        if total > 500_000 and row_number % 500_000 == 0:
            elapsed = time.perf_counter() - started
            print(f"      ground truth: {row_number:,}/{total:,} rows ({elapsed:.0f}s)",
                  flush=True)
    return out


def write_id_list_tsv(path, id_map, s1_ids, header_col):
    """Write a matching_results / candidate_pairs style TSV.

    Guarantees the format rules: one row per S1 id in ``s1_ids``, empty string for
    no matches, no duplicate IDs within a list, S2-/S3- only, sorted for
    determinism.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{header_col}\n")
        for s1 in s1_ids:
            ids = id_map.get(s1, set())
            clean = sorted(
                i for i in set(ids)
                if i.startswith(("S2-", "S3-"))  # never self-match S1
            )
            f.write(f"{s1}\t{','.join(clean)}\n")
