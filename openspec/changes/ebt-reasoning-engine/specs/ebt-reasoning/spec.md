## ADDED Requirements

### REQ-EBT-001: Sentence Embedding Pipeline
The system shall embed question and answer strings into continuous vector representations using a frozen sentence encoder (all-MiniLM-L6-v2, 384-dim). The embeddings shall be concatenated into a single (768-dim) vector for energy scoring.

#### SCENARIO-EBT-001-001: Embed QA pair
- **WHEN** a question string and answer string are provided
- **THEN** each is encoded to a 384-dim vector and concatenated to a 768-dim joint embedding

#### SCENARIO-EBT-001-002: Batch embedding
- **WHEN** a list of (question, answer) pairs is provided
- **THEN** all pairs are embedded efficiently in a single batch, returning an array of shape (N, 768)

### REQ-EBT-002: Energy Function Training on QA Embeddings
The system shall train a Gibbs EBM on embedded (question, correct_answer) vs (question, wrong_answer) pairs using NCE loss. The trained EBM shall assign lower energy to correct QA pairs than wrong QA pairs.

#### SCENARIO-EBT-002-001: Correct answers get lower energy
- **WHEN** a Gibbs EBM is trained on embedded QA pairs from TruthfulQA
- **THEN** mean energy of (question, correct_answer) pairs is lower than (question, wrong_answer) pairs on a held-out test set

#### SCENARIO-EBT-002-002: Energy discrimination accuracy
- **WHEN** test-set QA pairs are scored by the trained EBM
- **THEN** threshold-based classification achieves >60% accuracy (above the 50% we got on raw activations in practice)

### REQ-EBT-003: Gradient-Based Answer Repair in Embedding Space
The system shall optimize a wrong-answer embedding by gradient descent on the energy function while keeping the question embedding fixed. The optimization shall use Langevin dynamics (gradient + noise) to explore the energy landscape.

#### SCENARIO-EBT-003-001: Energy decreases during repair
- **WHEN** gradient descent is applied to a wrong-answer embedding for 100 steps
- **THEN** the energy of the (question, repaired_answer) pair decreases monotonically (modulo noise)

#### SCENARIO-EBT-003-002: Repaired embedding moves toward correct region
- **WHEN** a wrong-answer embedding is repaired via gradient descent
- **THEN** cosine similarity between the repaired embedding and the correct-answer embedding increases

### REQ-EBT-004: Nearest-Neighbor Decoding
The system shall decode continuous embeddings back to text by finding the nearest answer in a pre-embedded candidate pool via cosine similarity.

#### SCENARIO-EBT-004-001: Decode repaired embedding
- **WHEN** a repaired answer embedding is decoded against a candidate pool
- **THEN** the decoded answer is returned along with cosine similarity score

#### SCENARIO-EBT-004-002: Repair improves decoded answer
- **WHEN** wrong answers are repaired and decoded on a test set
- **THEN** the decoded answers are more often correct than the original wrong answers
