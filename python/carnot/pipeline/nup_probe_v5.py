"""NUPProbeV5 — GRPO-style contrastive retrain of NUP Probe v4 on live benchmark pairs.

**Why NUPProbeV5 exists (arXiv 2503.06639, REQ-VERIFY-125):**
    GRPO (Group Relative Policy Optimisation, arXiv 2503.06639) showed that binary
    correct/incorrect labels from live benchmarks ARE a contrastive training signal
    without needing additional human annotation.  For each question, the set of model
    responses that got the answer right (correct) versus wrong (incorrect) forms a
    natural contrastive pair.

    NUP Probe v4 already uses contrastive margin loss.  The v5 upgrade applies that
    same objective to REAL live benchmark data rather than synthetic pairs.  The key
    insight from GRPO is that grouping by question_index extracts the signal: within
    the same question, a correct response and an incorrect response SHOULD produce
    different energies.  The question itself is a controlled variable — the only
    difference is whether the model got it right.

    This is the "free supervision" principle: we already have the answer label, so
    we can produce arbitrarily many contrastive pairs from existing benchmark runs
    without any new annotation budget.

**How GRPOContrastivePairer works:**
    Given a flat list of live_pairs entries (each with question_index and is_correct),
    it groups entries by question_index.  For questions where BOTH correct and incorrect
    responses exist, it yields (correct_entry, incorrect_entry) pairs.  Questions with
    only one label type are skipped — they provide no contrastive signal.

**How NUPProbeV5 works:**
    Wraps NUPProbeV4 directly (no new architecture).  The v5 upgrade is entirely in
    the training data pipeline:
    1. Use GRPOContrastivePairer to extract pairs from live_pairs JSON entries.
    2. Extract step_text from the 'response' field (or first cot_step if present).
       NOTE: live_pairs do NOT store hidden states; we use raw response text as the
       input to the energy function.  This is a text-level proxy, not a hidden-state
       probe.  The limitation is noted in the artifact honest_verdict.
    3. Call NUPProbeV4.train_contrastive with the extracted (correct, incorrect) texts.
    4. Evaluate AUC on the full FOVER corpus.

**Safetensors serialisation:**
    If AUC >= 0.750, weights and bias are serialised to safetensors format via a
    minimal NumPy-compatible approach (no JAX or Torch required).

Spec: REQ-VERIFY-125, SCENARIO-VERIFY-155, SCENARIO-VERIFY-156
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from carnot.pipeline.nup_probe_v4 import NUPProbeV4


# ---------------------------------------------------------------------------
# GRPOContrastivePairer
# ---------------------------------------------------------------------------


class GRPOContrastivePairer:
    """Extract GRPO-style contrastive pairs from live benchmark entries.

    **Why group by question_index:**
        Within the same question, a correct response and an incorrect response
        differ ONLY in correctness.  The question itself is held constant.
        This controlled-variable property is exactly what makes the pair a
        clean contrastive signal: the model's energy should be lower for the
        correct response than for the incorrect one.

    **What 'entry' means here:**
        Each entry is a dict from a live_pairs JSON file with at minimum:
        - 'question_index': int — identifies which question this response answers
        - 'is_correct': bool — whether the response is factually correct
        - 'response': str — the model's full response text (used as step_text)
        - 'cot_steps': list — optional; if present, first step's 'step_text' is used

    Spec: REQ-VERIFY-125-1, SCENARIO-VERIFY-155
    """

    def pairs(self, entries: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """Group entries by question_index and yield (correct, incorrect) pairs.

        **Algorithm:**
            1. Partition entries into correct_by_q[question_index] and
               incorrect_by_q[question_index] lists.
            2. For each question_index that has BOTH correct and incorrect entries,
               emit one (correct_entry, incorrect_entry) pair per combination.
               (In practice, most questions have one correct and one-to-three incorrect
               responses, so the cross-product is small.)
            3. Questions with only correct OR only incorrect entries are skipped —
               they have no contrastive partner and provide no training signal.

        Args:
            entries: List of live_pairs dicts with 'question_index' and 'is_correct'.

        Returns:
            List of (correct_entry, incorrect_entry) tuples.

        Spec: REQ-VERIFY-125-1, SCENARIO-VERIFY-155
        """
        correct_by_q: Dict[int, List[Dict]] = defaultdict(list)
        incorrect_by_q: Dict[int, List[Dict]] = defaultdict(list)

        for entry in entries:
            q_idx = entry.get("question_index", -1)
            if entry.get("is_correct", False):
                correct_by_q[q_idx].append(entry)
            else:
                incorrect_by_q[q_idx].append(entry)

        result: List[Tuple[Dict, Dict]] = []
        for q_idx in correct_by_q:
            if q_idx not in incorrect_by_q:
                continue  # no contrastive partner — skip
            for correct_entry in correct_by_q[q_idx]:
                for incorrect_entry in incorrect_by_q[q_idx]:
                    result.append((correct_entry, incorrect_entry))

        return result


def _extract_step_text(entry: Dict) -> str:
    """Extract the primary text signal from a live_pairs entry.

    **Why prefer cot_steps over response:**
        CoT steps are already segmented; the first step is the most informative
        unit for energy probing.  If no steps are available, fall back to the
        full response text.  Either way, we note this is text-level — no hidden
        states are stored in live_pairs.

    Args:
        entry: A live_pairs dict.

    Returns:
        String text to pass to the energy probe.
    """
    steps = entry.get("cot_steps", [])
    if steps and isinstance(steps, list) and len(steps) > 0:
        step = steps[0]
        if isinstance(step, dict):
            return str(step.get("step_text", ""))
    return str(entry.get("response", ""))


# ---------------------------------------------------------------------------
# NUPProbeV5
# ---------------------------------------------------------------------------


class NUPProbeV5:
    """NUP Probe v5 — GRPO-style contrastive retrain using live benchmark pairs.

    **Architecture note:**
        This class wraps NUPProbeV4 without changing its architecture.  The v5
        upgrade is entirely in the training data pipeline.  We use text-level
        features because hidden states are not stored in live_pairs JSON files.
        This is a known limitation documented in the honest_verdict field.

    **GRPO connection (arXiv 2503.06639):**
        GRPO demonstrated that binary labels from benchmark evaluation ARE a
        sufficient contrastive signal.  For each question, correct responses
        provide negative examples (low energy target) and incorrect responses
        provide positive examples (high energy target).  No additional annotation
        is needed beyond the benchmark correctness label.

    Args:
        energy_dim:    Feature embedding dimension.  Default 32.
        margin:        Contrastive margin.  Default 1.0.
        learning_rate: SGD learning rate.  Default 0.01.
        random_seed:   Weight initialisation seed.  Default 42.

    Spec: REQ-VERIFY-125-2, REQ-VERIFY-125-3, REQ-VERIFY-125-4, SCENARIO-VERIFY-156
    """

    def __init__(
        self,
        energy_dim: int = 32,
        margin: float = 1.0,
        learning_rate: float = 0.01,
        random_seed: int = 42,
    ) -> None:
        self._probe = NUPProbeV4(
            energy_dim=energy_dim,
            margin=margin,
            learning_rate=learning_rate,
            random_seed=random_seed,
        )
        self._pairer = GRPOContrastivePairer()

    def train_from_pairs(
        self,
        entries: List[Dict],
        n_epochs: int = 100,
    ) -> Dict:
        """Train the probe using GRPO-style contrastive pairs extracted from live_pairs.

        **What this does step-by-step:**
            1. Use GRPOContrastivePairer to extract (correct_entry, incorrect_entry) pairs.
            2. Extract step_text from each entry using _extract_step_text().
            3. Build correct_steps and incorrect_steps lists.
            4. Delegate training to NUPProbeV4.train_contrastive().
            5. Return the training result dict augmented with grpo_pairs_built count.

        **NOTE on hidden states:**
            live_pairs JSON files store only text responses, NOT hidden states.
            We use text-level character-bigram features as a proxy.  A future
            v6 upgrade would store the actual last-token hidden state from the
            LLM and use that as the energy input instead.

        Args:
            entries:  Combined live_pairs entries from one or more JSON files.
            n_epochs: Number of training epochs.  Default 100.

        Returns:
            Dict with keys from NUPProbeV4.train_contrastive plus:
                'grpo_pairs_built': int — number of contrastive pairs extracted.
                'n_correct_steps': int — number of unique correct response texts.
                'n_incorrect_steps': int — number of unique incorrect response texts.

        Spec: REQ-VERIFY-125-2, SCENARIO-VERIFY-156
        """
        pairs = self._pairer.pairs(entries)

        correct_steps: List[str] = []
        incorrect_steps: List[str] = []

        for correct_entry, incorrect_entry in pairs:
            correct_steps.append(_extract_step_text(correct_entry))
            incorrect_steps.append(_extract_step_text(incorrect_entry))

        train_result = self._probe.train_contrastive(
            correct_steps=correct_steps,
            incorrect_steps=incorrect_steps,
            n_epochs=n_epochs,
        )

        train_result["grpo_pairs_built"] = len(pairs)
        train_result["n_correct_steps"] = len(correct_steps)
        train_result["n_incorrect_steps"] = len(incorrect_steps)
        return train_result

    def evaluate_auc(self, entries: List[Dict]) -> float:
        """Evaluate AUROC on a FOVER corpus (list of live_pairs entries).

        **How this works:**
            Extract step_text for all entries, split into correct/incorrect by
            is_correct label, and delegate to NUPProbeV4.evaluate_auc().

        Args:
            entries: Live_pairs entries with 'is_correct' and 'response' fields.

        Returns:
            Float AUROC in [0.0, 1.0].

        Spec: REQ-VERIFY-125-3, SCENARIO-VERIFY-156
        """
        correct_steps = [_extract_step_text(e) for e in entries if e.get("is_correct")]
        incorrect_steps = [_extract_step_text(e) for e in entries if not e.get("is_correct")]
        return self._probe.evaluate_auc(correct_steps, incorrect_steps)

    def save_safetensors(self, path: str) -> None:
        """Serialise probe weights and bias to safetensors format.

        **Why safetensors:**
            Carnot's standard serialisation format (cross-language, safe, no
            pickle deserialization vulnerabilities).  We use a minimal NumPy-only
            implementation since NUPProbeV5 runs without JAX or Torch.

        **Format written:**
            A single-tensor safetensors file containing:
            - 'weights': float32 array of shape (energy_dim,)
            - 'bias':    float32 scalar stored as shape (1,)

        Args:
            path: Output file path (e.g. 'results/nup_probe_v5.safetensors').

        Spec: REQ-VERIFY-125-4
        """
        import struct

        weights = self._probe._weights  # list of floats
        bias = self._probe._bias        # float scalar

        # Encode as raw float32 bytes (safetensors header + data layout)
        # We write a minimal safetensors-compatible binary:
        # [uint64 header_len][json header bytes][data bytes]
        import json as _json

        w_data = struct.pack(f"{len(weights)}f", *weights)
        b_data = struct.pack("f", bias)

        header = {
            "weights": {
                "dtype": "F32",
                "shape": [len(weights)],
                "data_offsets": [0, len(w_data)],
            },
            "bias": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [len(w_data), len(w_data) + len(b_data)],
            },
        }
        header_bytes = _json.dumps(header, separators=(",", ":")).encode("utf-8")
        # Pad header to 8-byte alignment (safetensors spec requirement)
        pad = (8 - (len(header_bytes) % 8)) % 8
        header_bytes += b" " * pad

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(w_data)
            f.write(b_data)
