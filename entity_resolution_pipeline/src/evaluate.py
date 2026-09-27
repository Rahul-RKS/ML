"""Official F_0.5 metric (macro-averaged over Source-1 entities).

This is the single most important file to get right: every threshold decision
depends on it. The rules (from the problem statement):

  * Per Source-1 entity: precision = |pred ∩ true| / |pred|,
    recall = |pred ∩ true| / |true|,
    F_0.5 = (1.25 * P * R) / (0.25 * P + R), and 0.0 if the denominator is 0.
  * A true SINGLETON (no true matches): score 1.0 if predicted empty, else 0.0.
  * An entity missing from predictions is treated as an empty prediction.
  * Macro-average across all Source-1 entities in the ground truth.

Run `python evaluate.py` to execute the built-in unit tests.
"""


def f0_5_for_entity(true_set, pred_set):
    """F_0.5 for a single Source-1 entity following the competition rules."""
    true_set = set(true_set)
    pred_set = set(pred_set)
    if not true_set:  # true singleton
        return 1.0 if not pred_set else 0.0
    if not pred_set:  # missed everything
        return 0.0
    inter = len(true_set & pred_set)
    if inter == 0:
        return 0.0
    precision = inter / len(pred_set)
    recall = inter / len(true_set)
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom


def macro_f0_5(ground_truth, predictions):
    """Macro-average F_0.5 over every entity in ``ground_truth``.

    ground_truth / predictions: {source1_entity_id: set(matched_ids)}.
    Missing keys in ``predictions`` count as an empty prediction.
    """
    if not ground_truth:
        return 0.0
    total = 0.0
    for s1, true_set in ground_truth.items():
        total += f0_5_for_entity(true_set, predictions.get(s1, set()))
    return total / len(ground_truth)


def macro_f0_5_detailed(ground_truth, predictions):
    """Return (macro_f0_5, micro_precision, micro_recall, singleton_accuracy)."""
    if not ground_truth:
        return 0.0, 0.0, 0.0, 0.0
    tp = fp = fn = 0
    singleton_total = singleton_correct = 0
    for s1, true_set in ground_truth.items():
        true_set = set(true_set)
        pred_set = set(predictions.get(s1, set()))
        if not true_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    sing = singleton_correct / singleton_total if singleton_total else 1.0
    return macro_f0_5(ground_truth, predictions), prec, rec, sing


# --------------------------------------------------------------------------- #
# Unit tests — run directly: python evaluate.py
# --------------------------------------------------------------------------- #
def _tests():
    # Singleton predicted empty -> 1.0
    assert f0_5_for_entity(set(), set()) == 1.0
    # Singleton with any false match -> 0.0
    assert f0_5_for_entity(set(), {"S2-1"}) == 0.0
    # Worked example from the problem statement -> ~0.714
    v = f0_5_for_entity({"S2-00047", "S3-00812"},
                        {"S2-00047", "S2-00193", "S3-00812"})
    assert abs(v - 0.7142857142857143) < 1e-9, v
    # Perfect match -> 1.0
    assert abs(f0_5_for_entity({"S2-1", "S2-2"}, {"S2-1", "S2-2"}) - 1.0) < 1e-9
    # No overlap -> 0.0
    assert f0_5_for_entity({"S2-1"}, {"S2-2"}) == 0.0
    # True match but empty prediction -> 0.0
    assert f0_5_for_entity({"S2-1"}, set()) == 0.0
    # Entity entirely missing from predictions == empty prediction
    gt = {"S1-1": {"S2-1"}, "S1-2": set()}
    preds = {"S1-2": set()}  # S1-1 absent
    # S1-1: 0.0 (missed), S1-2: 1.0 (correct singleton) -> macro 0.5
    assert abs(macro_f0_5(gt, preds) - 0.5) < 1e-9, macro_f0_5(gt, preds)
    # Empty ground truth -> 0.0 by convention
    assert macro_f0_5({}, {}) == 0.0
    print("evaluate.py: all unit tests passed.")


if __name__ == "__main__":
    _tests()
