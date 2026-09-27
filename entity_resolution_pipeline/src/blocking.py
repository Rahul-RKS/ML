"""Candidate generation (blocking).

All-pairs comparison is infeasible at this scale, so for each Source-1 entity we
retrieve its top-k most cosine-similar Source-2/3 records using character n-grams
of the name+address blob.

Blocking is done per country (matches never cross country here). Vectors come from
a HashingVectorizer, which maps n-grams into a fixed hash space instead of
building and storing a vocabulary, so there is no vocabulary dict to blow up
memory and no fit pass. The top-k most similar candidates per entity are found
with sparse_dot_topn, which keeps only the top-k during the multiplication so the
near-dense full similarity matrix is never materialized.

k and the min-similarity floor trade recall against candidate-set size.
"""

import numpy as np
from scipy.sparse import vstack as sparse_vstack
from sklearn.feature_extraction.text import HashingVectorizer

# Character n-grams are common enough that most record pairs share at least one,
# so the S1 x candidate cosine matrix is nearly dense and cannot be materialized
# at multi-million scale. sparse_dot_topn keeps only the top-k per row *during*
# the multiplication (C++), so memory stays at n_rows x k. We import it if
# available and fall back to a pure-Python batched path for small inputs.
try:
    from sparse_dot_topn import sp_matmul_topn as _sp_topn

    def _topk_matmul(a, b_t, k, min_sim):
        # a: (n1, V) csr, b_t: (V, nc) csr. Returns (n1, nc) csr, top-k per row.
        return _sp_topn(
            a, b_t, top_n=k, threshold=min_sim, sort=True, n_threads=-1
        ).tocsr()

    _HAVE_SPTOPN = True
except Exception:  # pragma: no cover - fallback for environments without the lib
    _HAVE_SPTOPN = False


def _transform_chunked(vec, texts, chunk=250_000):
    """Hash-transform texts in chunks and stack the sparse result.

    Transforming millions of rows in one call forces a single multi-GB contiguous
    buffer (and an internal doubling resize) that can fail even when total memory
    is sufficient. Chunking caps each transform's transient allocation.
    """
    texts = list(texts)
    if len(texts) <= chunk:
        return vec.transform(texts)
    parts = [vec.transform(texts[i:i + chunk]) for i in range(0, len(texts), chunk)]
    return sparse_vstack(parts, format="csr")


def _topk_for_block(s1_vecs, cand_vecs, k, min_sim, batch=1000):
    """Return (rows, cols, sims) top-k candidate indices per S1 row for one block.

    s1_vecs, cand_vecs: (n, V) sparse CSR, rows L2-normalized so a dot product is
    cosine similarity. We multiply each S1 batch against cand_vecs.T (a cheap CSC
    view, no copy) and keep the result SPARSE, reading top-k only from the nonzero
    entries of each row. Two records with no shared character n-gram have exactly
    zero similarity, so the vast majority of candidates never materialize -- this
    is what makes it safe at multi-million-candidate scale.

    NOTE: this is only the fallback path used when sparse_dot_topn is not
    installed (see _HAVE_SPTOPN above). The primary path in generate_candidates()
    uses sp_matmul_topn directly and never calls this function. Prefer fixing
    your sparse_dot_topn install over relying on this path -- it is much slower
    and, because it still forms one un-topk'd batch product at a time via plain
    `@`, it can OOM on batches where a block is very dense.
    """
    n1 = s1_vecs.shape[0]
    nc = cand_vecs.shape[0]
    if nc == 0 or n1 == 0:
        return np.array([], int), np.array([], int), np.array([], float)
    cand_tv = cand_vecs.T.tocsc()  # transpose as a CSC view (shares data, no copy)
    rows_out, cols_out, sims_out = [], [], []
    for start in range(0, n1, batch):
        end = min(start + batch, n1)
        # Sparse score matrix for this batch: (rows in batch) x nc, but only the
        # nonzero (shared-ngram) pairs are stored.
        scores = (s1_vecs[start:end] @ cand_tv).tocsr()
        indptr, indices, data = scores.indptr, scores.indices, scores.data
        for i in range(end - start):
            lo, hi = indptr[i], indptr[i + 1]
            if lo == hi:
                continue  # no candidate shares any n-gram -> singleton
            cand_idx = indices[lo:hi]
            row_scores = data[lo:hi]
            keep = row_scores >= min_sim
            cand_idx, row_scores = cand_idx[keep], row_scores[keep]
            if cand_idx.size == 0:
                continue
            if cand_idx.size > k:
                part = np.argpartition(-row_scores, k - 1)[:k]
                cand_idx, row_scores = cand_idx[part], row_scores[part]
            order = np.argsort(-row_scores)
            rows_out.append(np.full(cand_idx.size, start + i))
            cols_out.append(cand_idx[order])
            sims_out.append(row_scores[order])
    if not rows_out:
        return np.array([], int), np.array([], int), np.array([], float)
    return (
        np.concatenate(rows_out),
        np.concatenate(cols_out),
        np.concatenate(sims_out),
    )


def generate_candidates(s1_df, cand_df, k=15, min_sim=0.10,
                        analyzer="char_wb", ngram_range=(3, 3)):
    """Generate candidate pairs for every Source-1 entity.

    Args:
        s1_df: preprocessed Source-1 frame (needs entity_id, blob, country_norm).
        cand_df: preprocessed, concatenated Source-2 + Source-3 frame.
        k: max candidates to keep per S1 entity (tune down for a smaller set).
        min_sim: cosine floor below which a candidate is dropped.

    Returns a list of dicts: {source1_entity_id, candidate_entity_id, sim}.
    """
    results = []
    s1_df = s1_df.reset_index(drop=True)
    cand_df = cand_df.reset_index(drop=True)
    countries = sorted(set(s1_df["country_norm"]) | set(cand_df["country_norm"]))
    for country in countries:
        s1_blk = s1_df[s1_df["country_norm"] == country]
        cand_blk = cand_df[cand_df["country_norm"] == country]
        if len(s1_blk) == 0 or len(cand_blk) == 0:
            continue
        print(f"      block '{country}': {len(s1_blk)} S1 x {len(cand_blk)} candidates...",
              flush=True)
        vec = HashingVectorizer(analyzer=analyzer, ngram_range=ngram_range,
                                n_features=2 ** 20, norm="l2", alternate_sign=False,
                                dtype=np.float32)
        # No fit needed - hashing is stateless. Rows come out L2-normalized, so a
        # dot product is cosine similarity. Transform in chunks to bound memory.
        s1_vecs = _transform_chunked(vec, s1_blk["blob"])
        cand_vecs = _transform_chunked(vec, cand_blk["blob"])
        s1_ids = s1_blk["entity_id"].to_numpy()
        cand_ids = cand_blk["entity_id"].to_numpy()

        if _HAVE_SPTOPN:
            # Bound each top-k product so large country blocks remain observable
            # and do not require one oversized temporary multiplication.
            b_t = cand_vecs.T.tocsr()
            del cand_vecs
            block_size = 25_000
            for start in range(0, len(s1_ids), block_size):
                end = min(start + block_size, len(s1_ids))
                C = _topk_matmul(s1_vecs[start:end], b_t, k, min_sim)
                indptr, indices, data = C.indptr, C.indices, C.data
                for i in range(C.shape[0]):
                    for j in range(indptr[i], indptr[i + 1]):
                        results.append({
                            "source1_entity_id": s1_ids[start + i],
                            "candidate_entity_id": cand_ids[indices[j]],
                            "sim": float(data[j]),
                        })
                print(f"        blocked {end:,}/{len(s1_ids):,} Source-1 rows",
                      flush=True)
            del b_t, s1_vecs
        else:
            # Pure-Python fallback (fine for small blocks / testing only).
            rows, cols, sims = _topk_for_block(s1_vecs, cand_vecs, k, min_sim)
            del cand_vecs, s1_vecs
            for r, c, s in zip(rows, cols, sims):
                results.append({
                    "source1_entity_id": s1_ids[r],
                    "candidate_entity_id": cand_ids[c],
                    "sim": float(s),
                })
    return results


def candidates_to_map(candidate_rows):
    """Group flat candidate rows into {s1_id: set(candidate_ids)}."""
    out = {}
    for row in candidate_rows:
        out.setdefault(row["source1_entity_id"], set()).add(row["candidate_entity_id"])
    return out