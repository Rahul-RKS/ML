# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Enter team name]
**Team Members:** [Enter all team members]
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
This solution uses country-aware character n-gram blocking to restrict comparisons,
then scores the remaining Source-1 to Source-2/3 pairs with a gradient-boosted
classifier. Its decision threshold is selected on an entity-level validation split
using the challenge's macro F_0.5 metric.

---

## 2. Methodology

### 2.1 Problem Analysis
The supplied data may vary in punctuation, legal suffixes, abbreviations, address
ordering, landmarks, and missing address components. The test set includes France,
which is absent from training, so country handling must remain open-set. No separate
source column is supplied; source identity is encoded in each entity ID.

### 2.2 Solution Strategy
Normalize names, addresses, and country labels; generate a bounded candidate set
within each country; train on labeled blocked pairs; tune a match-probability
threshold on held-out Source-1 entities; then score every test candidate.

**Approach Type:** Blocking + classifier
**Core Innovation:** Open-set country-aware blocking with threshold tuning against
the precision-weighted macro F_0.5 metric.

---

## 3. Candidate Generation (Blocking)
The pipeline uses a stateless `HashingVectorizer` over character 3-grams from the
normalized name-and-address text. It compares records only within matching country
labels and retains up to `k=15` candidates with cosine similarity at least `0.30`.

- **Blocking keys used:** Character 3-gram hashed vectors of normalized names and
	addresses; country equality is a hard block.
- **Candidate pairs generated:** [Fill from the pipeline's candidate statistics.]
- **How you ensured true matches were not lost:** Blocking recall is measured on
	training ground truth. The chosen `k` and similarity floor are recorded above;
	report the measured recall from the run here.

---

## 4. Matching Model

**Features used:**
- Name features: token-sort ratio, token-set ratio, partial ratio, Jaro-Winkler,
	and token Jaccard similarity.
- Address features: token-sort ratio, token-set ratio, partial ratio, and token
	Jaccard similarity.
- Other: exact postal-code match, postal-code presence, country match, candidate
	landmark flag, and blocking cosine similarity.

**Model type:** `HistGradientBoostingClassifier` from scikit-learn, trained on
blocked pairs with up to five highest-similarity negative examples per entity.
**Threshold selection method:** Select the threshold with the highest macro F_0.5
on a deterministic entity-level validation split.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [Fill from the successful pipeline run.]
- **Common false positives (wrong merges):** [Review validation errors and report.]
- **Common false negatives (missed matches):** [Review validation errors and report.]

---

## 6. Conclusion
The pipeline produces a bounded candidate set and ranks candidate pairs using
name, address, postal, country, landmark, and blocking-similarity signals. Add the
measured validation score, blocking recall, and error analysis after a full run.

---

## Appendix

### A. Code Artefacts
The runnable pipeline is packaged under `code/business_entity_resolution/`, with
source modules in `src/`, dependencies in `requirements.txt`, and reproduction
instructions in `README.md`. Run `python src/run_pipeline.py --data-dir dataset
--out-dir output --k 15 --min-sim 0.30` to generate both required TSV files.

### B. Additional Results
Add any plots or detailed validation results here if available.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
