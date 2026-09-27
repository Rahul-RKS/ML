"""Pairwise similarity features for the matching model.

Given a Source-1 record and a candidate Source-2/3 record (both already
preprocessed), produce a fixed-length numeric feature vector. All features are
symmetric string-similarity signals plus a couple of structured flags — no
external data, per the fair-play rules.
"""

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

FEATURE_NAMES = [
    "name_token_sort", "name_token_set", "name_partial", "name_jaro_winkler",
    "name_jaccard",
    "addr_token_sort", "addr_token_set", "addr_partial", "addr_jaccard",
    "postal_exact", "postal_both_present",
    "country_match", "has_landmark_cand",
    "blocking_sim",
]


def _jaccard(a, b):
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def pair_features(s1, cand, blocking_sim=0.0):
    """Return a feature vector for one (S1, candidate) pair.

    s1, cand: dict-like rows with name_norm, name_core, addr_norm, postal,
    country_norm, has_landmark.
    """
    n1, n2 = s1["name_core"], cand["name_core"]
    a1, a2 = s1["addr_norm"], cand["addr_norm"]
    p1, p2 = s1["postal"], cand["postal"]

    postal_both = 1.0 if (p1 and p2) else 0.0
    postal_exact = 1.0 if (p1 and p2 and p1 == p2) else 0.0

    return np.array([
        fuzz.token_sort_ratio(n1, n2) / 100.0,
        fuzz.token_set_ratio(n1, n2) / 100.0,
        fuzz.partial_ratio(n1, n2) / 100.0,
        JaroWinkler.similarity(n1, n2),
        _jaccard(n1, n2),
        fuzz.token_sort_ratio(a1, a2) / 100.0,
        fuzz.token_set_ratio(a1, a2) / 100.0,
        fuzz.partial_ratio(a1, a2) / 100.0,
        _jaccard(a1, a2),
        postal_exact,
        postal_both,
        1.0 if s1["country_norm"] == cand["country_norm"] else 0.0,
        float(cand.get("has_landmark", 0)),
        float(blocking_sim),
    ], dtype=np.float32)


def build_feature_matrix(pairs, s1_lookup, cand_lookup):
    """Build an (n_pairs, n_features) matrix.

    pairs: iterable of (s1_id, cand_id, blocking_sim).
    s1_lookup / cand_lookup: {entity_id: row-dict}.
    """
    rows = [
        pair_features(s1_lookup[s1], cand_lookup[cid], sim)
        for s1, cid, sim in pairs
    ]
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.vstack(rows)
