"""Train and evaluate the pairwise entity-resolution matcher."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from evaluate import macro_f0_5_detailed
from features import build_feature_matrix


def split_entities(entity_ids, seed=42, validation_fraction=0.2):
	"""Split Source-1 IDs deterministically so entities never leak across sets."""
	unique_ids = np.asarray(list(dict.fromkeys(entity_ids)), dtype=object)
	if len(unique_ids) < 2:
		raise ValueError("At least two Source-1 entities are required for validation.")
	rng = np.random.default_rng(seed)
	rng.shuffle(unique_ids)
	n_validation = min(
		len(unique_ids) - 1,
		max(1, int(round(len(unique_ids) * validation_fraction))),
	)
	validation_ids = set(unique_ids[:n_validation])
	training_ids = set(unique_ids[n_validation:])
	return training_ids, validation_ids


def build_training_pairs(candidate_map, ground_truth, candidate_sim, max_negatives=5):
	"""Label blocked pairs, retaining the hardest negatives per entity."""
	labeled = []
	for source1_id, candidate_ids in candidate_map.items():
		true_ids = ground_truth.get(source1_id, set())
		positives = []
		negatives = []
		for candidate_id in candidate_ids:
			similarity = candidate_sim.get((source1_id, candidate_id), 0.0)
			row = (source1_id, candidate_id, similarity)
			if candidate_id in true_ids:
				positives.append((*row, 1))
			else:
				negatives.append((*row, 0))
		labeled.extend(positives)
		negatives.sort(key=lambda row: row[2], reverse=True)
		labeled.extend(negatives[:max_negatives])
	return labeled


def train_matcher(training_rows, source1_lookup, candidate_lookup, seed=42):
	"""Fit a compact gradient-boosted classifier on pair similarity features."""
	if not training_rows:
		raise ValueError("No blocked training pairs are available to fit the matcher.")
	labels = np.fromiter((row[3] for row in training_rows), dtype=np.uint8)
	if np.unique(labels).size < 2:
		raise ValueError("Training pairs must include both matches and non-matches.")
	pairs = [(row[0], row[1], row[2]) for row in training_rows]
	features = build_feature_matrix(pairs, source1_lookup, candidate_lookup)
	model = HistGradientBoostingClassifier(
		max_iter=120,
		learning_rate=0.08,
		max_leaf_nodes=31,
		l2_regularization=1.0,
		early_stopping=True,
		validation_fraction=0.1,
		n_iter_no_change=10,
		random_state=seed,
	)
	model.fit(features, labels)
	return model


def predict_scores(model, rows, source1_lookup, candidate_lookup):
	"""Return match probabilities for rows shaped as (S1, candidate, sim, label)."""
	if not rows:
		return np.empty(0, dtype=np.float64)
	pairs = [(row[0], row[1], row[2]) for row in rows]
	features = build_feature_matrix(pairs, source1_lookup, candidate_lookup)
	return model.predict_proba(features)[:, 1]


def tune_threshold(validation_rows, scores, validation_ids, ground_truth):
	"""Choose the threshold with the best macro F0.5 on held-out entities."""
	thresholds = np.arange(0.05, 1.0, 0.025)
	candidates_by_entity = {}
	for row, score in zip(validation_rows, scores):
		candidates_by_entity.setdefault(row[0], []).append((row[1], float(score)))

	best_threshold = 0.5
	best_f = -1.0
	table = []
	for threshold in thresholds:
		predictions = {
			source1_id: {
				candidate_id
				for candidate_id, score in candidates_by_entity.get(source1_id, [])
				if score >= threshold
			}
			for source1_id in validation_ids
		}
		validation_truth = {
			source1_id: ground_truth.get(source1_id, set())
			for source1_id in validation_ids
		}
		score, precision, recall, singleton_accuracy = macro_f0_5_detailed(
			validation_truth, predictions
		)
		threshold = float(threshold)
		table.append((threshold, score, precision, recall, singleton_accuracy))
		if score > best_f:
			best_threshold = threshold
			best_f = score
	return best_threshold, best_f, table