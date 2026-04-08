## Tasks

### Task 1: Sentence embedding pipeline
**Refs:** REQ-EBT-001
**Files:** `python/carnot/inference/ebt_embeddings.py`, `tests/python/test_ebt_embeddings.py`
**Description:** Create a module that embeds QA pairs using sentence-transformers. Functions: `embed_qa_pair(question, answer) → 768-dim vector`, `embed_qa_batch(pairs) → (N, 768) array`. Use all-MiniLM-L6-v2 (384-dim per sentence, concatenated).
**Tests:** SCENARIO-EBT-001-001, SCENARIO-EBT-001-002

### Task 2: Train energy function on TruthfulQA embeddings
**Refs:** REQ-EBT-002
**Files:** `scripts/experiment_37_ebt_reasoning.py`
**Description:** Embed all TruthfulQA QA pairs (correct + incorrect answers), train Gibbs EBM [768 → 256 → 64 → 1] via NCE. Evaluate energy discrimination on held-out test set. Target: >60% accuracy.
**Tests:** SCENARIO-EBT-002-001, SCENARIO-EBT-002-002

### Task 3: Gradient-based answer repair
**Refs:** REQ-EBT-003
**Files:** `python/carnot/inference/ebt_repair.py`, `tests/python/test_ebt_repair.py`
**Description:** Implement `repair_answer_embedding(question_emb, wrong_answer_emb, ebm, n_steps, lr, noise_scale) → repaired_emb`. Fix the question embedding, optimize only the answer embedding via Langevin dynamics on the energy function. Track energy trajectory.
**Tests:** SCENARIO-EBT-003-001, SCENARIO-EBT-003-002

### Task 4: Nearest-neighbor decoding
**Refs:** REQ-EBT-004
**Files:** `python/carnot/inference/ebt_decode.py`, `tests/python/test_ebt_decode.py`
**Description:** Implement `decode_embedding(repaired_emb, candidate_pool, candidate_texts) → (best_text, similarity)`. Pre-embed a candidate pool from TruthfulQA correct answers, find nearest by cosine similarity.
**Tests:** SCENARIO-EBT-004-001, SCENARIO-EBT-004-002

### Task 5: End-to-end experiment
**Refs:** REQ-EBT-002, REQ-EBT-003, REQ-EBT-004
**Files:** `scripts/experiment_37_ebt_reasoning.py`
**Description:** Full pipeline: embed TruthfulQA → train EBM → for each wrong answer, repair via gradient descent → decode → measure improvement rate. Report: how often does repair change a wrong answer to a correct one?
**Tests:** All SCENARIO-EBT-*
