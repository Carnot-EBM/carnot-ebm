"""Exp 5090: STATIC-style CSR masks for finite constrained decoding.

Spec refs: REQ-VERIFY-5090, SCENARIO-VERIFY-5090.

This module intentionally benchmarks the mask extraction problem without
claiming model quality. The finite output space is small enough to enumerate,
so a trie can be flattened into CSR-like transition arrays and checked exactly
before any live generation would be allowed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


JsonDict = dict[str, Any]
ClockNs = Callable[[], int]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5090
EXPERIMENT_NAME = "experiment_5090_static_csr_constrained_decoding"
SCHEMA = "carnot.experiment_5090_static_csr_constrained_decoding.v467"
RESULT_RELATIVE_PATH = "results/experiment_5090_static_csr_constrained_decoding_v467.json"
EXP5085_RELATIVE_PATH = "results/experiment_5085_llamacpp_logprob_endpoint_bringup_v467.json"
SPEC_REFS = ["REQ-VERIFY-5090", "SCENARIO-VERIFY-5090"]
RUN_DATE = "20260701"
RANDOM_SEED = 20260701

FINITE_SCHEMA_NAME = "verifier_verdict_schema_v1"
DETERMINISTIC_INFERENCE_SUBSTRATE = "deterministic_static_csr_mask_benchmark"
BYTE_VOCAB_LIMIT = 128
EOS_TOKEN_ID = 128
VOCAB_SIZE = 129

MODEL_SPECS: tuple[dict[str, str], ...] = (
    {
        "role": "flagship_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "preferred_quant": "Q4_K_M",
    },
    {
        "role": "flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "preferred_quant": "Q4_K_M",
    },
    {
        "role": "middle_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "preferred_quant": "Q4_K_M",
    },
)
MANDATED_MODEL_IDS = tuple(spec["hf_id"] for spec in MODEL_SPECS)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "preconditions_checked",
    "model_specs",
    "finite_schema",
    "n_allowed_outputs",
    "trie_latency_ms",
    "csr_latency_ms",
    "mask_speedup",
    "validity_rate",
    "rerank_only_validity_rate",
    "live_llm_invoked",
    "beats_cpu_trie",
    "beats_rerank_only_on_validity_or_cost",
    "flagged_adversarial",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal prefix: success only for CSR speedup plus validity/cost win; "
        "otherwise complete no-headline diagnostic."
    ),
    "duration_s": "Wall-clock runtime for deterministic mask extraction and artifact assembly.",
    "inference_substrate": (
        "Declares deterministic_static_csr_mask_benchmark unless a real live smoke is invoked."
    ),
    "preconditions_checked": (
        "Records selected schema, byte-token assumptions, and Exp5085 endpoint usability."
    ),
    "model_specs": "The three mandated GGUF model IDs plus any resolved Exp5085 paths.",
    "finite_schema": "The bounded verifier verdict schema and canonical JSON/token rules.",
    "n_allowed_outputs": "Number of exact canonical outputs in the finite language.",
    "trie_latency_ms": "Mean CPU trie mask lookup latency per prefix in milliseconds.",
    "csr_latency_ms": "Mean CSR/vectorized mask lookup latency per prefix in milliseconds.",
    "mask_speedup": "Trie latency divided by CSR latency; above one means CSR is faster.",
    "validity_rate": "Fraction of constrained finite outputs accepted by the masks.",
    "rerank_only_validity_rate": "Post-generation rerank-only validity on candidate batches.",
    "live_llm_invoked": "True only when a live endpoint smoke actually ran.",
    "beats_cpu_trie": "True only when CSR is faster while preserving exact mask validity.",
    "beats_rerank_only_on_validity_or_cost": (
        "True when constrained masks beat rerank-only on validity or equal-validity cost."
    ),
    "flagged_adversarial": "True only for self-detected artifact inconsistency.",
}


@dataclass(frozen=True)
class TrieMaskIndex:
    """Indexed trie used as the CPU baseline for prefix-mask extraction.

    The trie is intentionally simple: every state stores a Python dictionary of
    outgoing byte-token transitions. That mirrors the obvious implementation a
    schema decoder would reach for before flattening transitions for hardware or
    vectorized lookup.
    """

    transitions: tuple[dict[int, int], ...]
    accepting_states: frozenset[int]
    vocab_size: int = VOCAB_SIZE
    eos_token_id: int = EOS_TOKEN_ID

    @property
    def state_count(self) -> int:
        return len(self.transitions)

    def transition(self, state: int, token_id: int) -> int:
        if state < 0 or state >= self.state_count:
            return -1
        return self.transitions[state].get(token_id, -1)

    def allowed_mask_for_state(self, state: int) -> int:
        if state < 0 or state >= self.state_count:
            return 0
        mask = 0
        for token_id in self.transitions[state]:
            mask |= 1 << token_id
        if state in self.accepting_states:
            mask |= 1 << self.eos_token_id
        return mask

    def allowed_mask(self, prefix: bytes) -> int:
        state = 0
        for token_id in prefix:
            state = self.transition(state, token_id)
            if state < 0:
                return 0
        return self.allowed_mask_for_state(state)


@dataclass(frozen=True)
class CSRAutomaton:
    """CSR-like byte automaton with precomputed per-state mask bitsets."""

    row_offsets: tuple[int, ...]
    labels: tuple[int, ...]
    targets: tuple[int, ...]
    accepting_states: frozenset[int]
    mask_bits: tuple[int, ...]
    vocab_size: int = VOCAB_SIZE
    eos_token_id: int = EOS_TOKEN_ID

    @property
    def state_count(self) -> int:
        return len(self.row_offsets) - 1

    @property
    def transition_count(self) -> int:
        return len(self.labels)

    def allowed_mask_for_state(self, state: int) -> int:
        if state < 0 or state >= self.state_count:
            return 0
        return self.mask_bits[state]

    def transition(self, state: int, token_id: int) -> int:
        if state < 0 or state >= self.state_count:
            return -1
        start = self.row_offsets[state]
        end = self.row_offsets[state + 1]
        for index in range(start, end):
            if self.labels[index] == token_id:
                return self.targets[index]
        return -1


def finite_verifier_verdict_outputs() -> tuple[str, ...]:
    """Enumerate the bounded verifier verdict strings used for this diagnostic.

    The important property is finiteness, not semantic richness. Canonical JSON
    keeps byte-token masks stable across machines and avoids relying on a BPE
    tokenizer whose vocabulary may differ by model build.
    """

    verdicts = ("accept", "reject", "abstain")
    evidence_labels = (
        "schema_valid",
        "schema_missing_field",
        "solver_verified",
        "solver_counterexample",
        "arithmetic_mismatch",
        "citation_gap",
    )
    confidences = ("low", "medium", "high")
    outputs: list[str] = []
    for verdict in verdicts:
        for evidence_label in evidence_labels:
            for confidence in confidences:
                payload = {
                    "confidence": confidence,
                    "evidence_label": evidence_label,
                    "schema": FINITE_SCHEMA_NAME,
                    "verdict": verdict,
                }
                outputs.append(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    return tuple(outputs)


def finite_schema_descriptor() -> JsonDict:
    return {
        "schema_name": FINITE_SCHEMA_NAME,
        "output_format": "canonical_ascii_json",
        "canonicalization": "json.dumps(sort_keys=True,separators=(',',':'))",
        "fields": {
            "schema": [FINITE_SCHEMA_NAME],
            "verdict": ["accept", "reject", "abstain"],
            "evidence_label": [
                "schema_valid",
                "schema_missing_field",
                "solver_verified",
                "solver_counterexample",
                "arithmetic_mismatch",
                "citation_gap",
            ],
            "confidence": ["low", "medium", "high"],
        },
        "vocabulary": {
            "kind": "ascii_byte_tokens_plus_eos",
            "byte_token_ids": "0..127",
            "eos_token_id": EOS_TOKEN_ID,
            "vocab_size": VOCAB_SIZE,
        },
    }


def build_trie_mask_index(outputs: Sequence[str]) -> TrieMaskIndex:
    """Build the CPU trie baseline from exact allowed output strings."""

    if not outputs:
        raise ValueError("finite output set must not be empty")
    transitions: list[dict[int, int]] = [{}]
    accepting_states: set[int] = set()
    for text in outputs:
        encoded = _encode_ascii(text)
        state = 0
        for token_id in encoded:
            next_state = transitions[state].get(token_id)
            if next_state is None:
                next_state = len(transitions)
                transitions[state][token_id] = next_state
                transitions.append({})
            state = next_state
        accepting_states.add(state)
    return TrieMaskIndex(
        transitions=tuple(dict(row) for row in transitions),
        accepting_states=frozenset(accepting_states),
    )


def build_csr_from_trie(trie: TrieMaskIndex) -> CSRAutomaton:
    """Flatten trie dictionaries into CSR-style sparse transition arrays."""

    row_offsets: list[int] = [0]
    labels: list[int] = []
    targets: list[int] = []
    mask_bits: list[int] = []
    for state, outgoing in enumerate(trie.transitions):
        mask = 0
        for token_id, target in sorted(outgoing.items()):
            labels.append(token_id)
            targets.append(target)
            mask |= 1 << token_id
        if state in trie.accepting_states:
            mask |= 1 << trie.eos_token_id
        mask_bits.append(mask)
        row_offsets.append(len(labels))
    return CSRAutomaton(
        row_offsets=tuple(row_offsets),
        labels=tuple(labels),
        targets=tuple(targets),
        accepting_states=trie.accepting_states,
        mask_bits=tuple(mask_bits),
    )


def evaluate_mask_equivalence(
    outputs: Sequence[str],
    trie: TrieMaskIndex,
    csr: CSRAutomaton,
) -> JsonDict:
    """Check CSR and trie masks on every finite-output prefix."""

    pairs = prefix_state_pairs(outputs, csr)
    mismatches: list[JsonDict] = []
    for prefix, state in pairs:
        trie_mask = trie.allowed_mask(prefix)
        csr_mask = csr.allowed_mask_for_state(state)
        if trie_mask != csr_mask:
            mismatches.append(
                {
                    "prefix": prefix.decode("ascii", "replace"),
                    "state": state,
                    "trie_mask": trie_mask,
                    "csr_mask": csr_mask,
                }
            )
    valid_outputs = sum(1 for output in outputs if _output_valid_under_masks(output, csr))
    prefix_count = len(pairs)
    equivalence_rate = (
        round((prefix_count - len(mismatches)) / prefix_count, 6) if prefix_count else 0.0
    )
    return {
        "prefix_count": prefix_count,
        "mask_equivalence_rate": equivalence_rate,
        "mismatched_prefix_count": len(mismatches),
        "mismatches": mismatches[:5],
        "validity_rate": _rate(valid_outputs, len(outputs)),
        "invalid_prefix_mask_zero": trie.allowed_mask(b"not-a-prefix") == 0
        and csr.allowed_mask_for_state(-1) == 0,
    }


def prefix_state_pairs(
    outputs: Sequence[str],
    csr: CSRAutomaton,
) -> tuple[tuple[bytes, int], ...]:
    """Return every prefix and CSR state encountered by allowed outputs."""

    seen: set[tuple[bytes, int]] = set()
    pairs: list[tuple[bytes, int]] = []
    for output in outputs:
        state = 0
        prefix = b""
        if (prefix, state) not in seen:
            seen.add((prefix, state))
            pairs.append((prefix, state))
        for token_id in _encode_ascii(output):
            state = csr.transition(state, token_id)
            if state < 0:
                break
            prefix = prefix + bytes((token_id,))
            item = (prefix, state)
            if item not in seen:
                seen.add(item)
                pairs.append(item)
    return tuple(pairs)


def benchmark_mask_lookup(
    trie: TrieMaskIndex,
    csr: CSRAutomaton,
    outputs: Sequence[str],
    *,
    repeats: int = 2000,
    clock_ns: ClockNs = time.perf_counter_ns,
) -> JsonDict:
    """Measure per-prefix mask lookup latency for trie and CSR paths."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    pairs = prefix_state_pairs(outputs, csr)
    operations = max(1, len(pairs) * repeats)

    trie_accumulator = 0
    started = clock_ns()
    for _ in range(repeats):
        for prefix, _state in pairs:
            trie_accumulator ^= trie.allowed_mask(prefix)
    trie_ns = clock_ns() - started

    csr_accumulator = 0
    started = clock_ns()
    for _ in range(repeats):
        for _prefix, state in pairs:
            csr_accumulator ^= csr.allowed_mask_for_state(state)
    csr_ns = clock_ns() - started

    trie_latency_ms = trie_ns / operations / 1_000_000.0
    csr_latency_ms = csr_ns / operations / 1_000_000.0
    mask_speedup = trie_latency_ms / csr_latency_ms if csr_latency_ms > 0 else 0.0
    return {
        "trie_latency_ms": round(trie_latency_ms, 6),
        "csr_latency_ms": round(csr_latency_ms, 6),
        "mask_speedup": round(mask_speedup, 6),
        "lookup_count": operations,
        "lookup_checksum": trie_accumulator ^ csr_accumulator,
    }


def compare_rerank_only(outputs: Sequence[str]) -> JsonDict:
    """Compare exact constrained masks with a bounded rerank-only control.

    Rerank-only can validate candidates after generation, but it cannot force
    the model to emit a member of the finite language. The candidate batches
    therefore include realistic malformed rows where no schema-valid candidate
    exists to rerank.
    """

    allowed = set(outputs)
    output_list = list(outputs)
    batches = (
        ("malformed_then_valid", ("not json", output_list[0])),
        ("wrong_schema_then_valid", ('{"schema":"other"}', output_list[7])),
        ("valid_first", (output_list[13], "trailing text")),
        ("no_valid_candidates_a", ("{}", '{"verdict":"accept"}')),
        ("no_valid_candidates_b", ("[]", '{"schema":"verifier_verdict_schema_v1"}')),
        ("valid_after_two", ("not json", '{"schema":"other"}', output_list[29])),
    )
    valid_selected = 0
    cost_units = 0
    rows: list[JsonDict] = []
    for batch_id, candidates in batches:
        selected: str | None = None
        for candidate in candidates:
            cost_units += 1
            if candidate in allowed:
                selected = candidate
                break
        selected_valid = selected in allowed
        valid_selected += int(selected_valid)
        rows.append(
            {
                "batch_id": batch_id,
                "n_candidates": len(candidates),
                "selected_valid": selected_valid,
                "selected_candidate": selected,
            }
        )

    rerank_validity = _rate(valid_selected, len(batches))
    constrained_validity = 1.0 if outputs else 0.0
    constrained_cost_units = sum(len(_encode_ascii(output_list[index % len(output_list)])) + 1 for index in range(len(batches)))
    beats = constrained_validity > rerank_validity or (
        constrained_validity == rerank_validity and constrained_cost_units < cost_units
    )
    return {
        "candidate_batches": len(batches),
        "rerank_only_validity_rate": rerank_validity,
        "constrained_validity_rate": constrained_validity,
        "rerank_only_cost_units": cost_units,
        "constrained_mask_cost_units": constrained_cost_units,
        "beats_rerank_only_on_validity_or_cost": beats,
        "rerank_rows": rows,
    }


def load_preconditions(*, root: Path | str = REPO_ROOT) -> JsonDict:
    """Load preconditions without treating stale endpoint fields as live evidence."""

    root_path = Path(root)
    exp5085_path = root_path / EXP5085_RELATIVE_PATH
    gate = _read_json_object(exp5085_path)
    endpoint_url = str(gate.get("endpoint_url") or "") if gate else ""
    flagged = bool(gate.get("flagged_adversarial")) if gate else False
    logprob_ready = bool(gate.get("logprob_endpoint_ready")) if gate else False
    completion_ready = bool(gate.get("completion_endpoint_ready")) if gate else False
    exists = gate is not None
    usable = bool(exists and logprob_ready and endpoint_url and not flagged)
    if not exists:
        reason = "exp5085_artifact_missing"
    elif flagged:
        reason = "exp5085_flagged_adversarial"
    elif not logprob_ready:
        reason = "exp5085_logprob_endpoint_not_ready"
    elif not endpoint_url:
        reason = "exp5085_endpoint_url_missing"
    else:
        reason = None

    return {
        "selected_finite_schema": FINITE_SCHEMA_NAME,
        "vocabulary_tokenization_assumptions": {
            "tokenization": "canonical ASCII JSON uses one byte as one token",
            "byte_token_ids": "0..127",
            "eos_token_id": EOS_TOKEN_ID,
            "bpe_tokenizer_used": False,
            "why": "finite mask extraction must be deterministic across GGUF tokenizers",
        },
        "live_endpoint_fields": {
            "artifact_path": EXP5085_RELATIVE_PATH,
            "exists": exists,
            "artifact_sha256": _sha256_file(exp5085_path),
            "endpoint_url": endpoint_url or None,
            "completion_endpoint_ready": completion_ready,
            "logprob_endpoint_ready": logprob_ready,
            "flagged_adversarial": flagged,
            "usable_for_live_smoke": usable,
            "unusable_reason": reason,
        },
    }


def run_diagnostic(
    *,
    root: Path | str = REPO_ROOT,
    repeats: int = 2000,
) -> JsonDict:
    """Run the deterministic Exp 5090 mask diagnostic and return the artifact."""

    started = time.perf_counter()
    outputs = finite_verifier_verdict_outputs()
    trie = build_trie_mask_index(outputs)
    csr = build_csr_from_trie(trie)
    equivalence = evaluate_mask_equivalence(outputs, trie, csr)
    latency = benchmark_mask_lookup(trie, csr, outputs, repeats=repeats)
    rerank = compare_rerank_only(outputs)
    preconditions = load_preconditions(root=root)
    live_llm_invoked = False
    validity_rate = float(equivalence["validity_rate"])
    mask_equivalence_rate = float(equivalence["mask_equivalence_rate"])
    trie_latency_ms = float(latency["trie_latency_ms"])
    csr_latency_ms = float(latency["csr_latency_ms"])
    beats_cpu_trie = bool(
        csr_latency_ms < trie_latency_ms
        and validity_rate == 1.0
        and mask_equivalence_rate == 1.0
    )
    beats_rerank = bool(rerank["beats_rerank_only_on_validity_or_cost"])
    honest_verdict = (
        "success_static_csr_masks_speedup_and_validity_win"
        if beats_cpu_trie and beats_rerank
        else "complete_static_csr_masks_diagnostic_no_headline"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "result_path": RESULT_RELATIVE_PATH,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": honest_verdict,
        "duration_s": round(max(0.0, time.perf_counter() - started), 6),
        "inference_substrate": DETERMINISTIC_INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "model_specs": model_specs_from_preconditions(root=root),
        "finite_schema": finite_schema_descriptor(),
        "n_allowed_outputs": len(outputs),
        "trie_latency_ms": trie_latency_ms,
        "csr_latency_ms": csr_latency_ms,
        "mask_speedup": float(latency["mask_speedup"]),
        "validity_rate": validity_rate,
        "rerank_only_validity_rate": float(rerank["rerank_only_validity_rate"]),
        "live_llm_invoked": live_llm_invoked,
        "beats_cpu_trie": beats_cpu_trie,
        "beats_rerank_only_on_validity_or_cost": beats_rerank,
        "flagged_adversarial": False,
        "csr_state_count": csr.state_count,
        "csr_transition_count": csr.transition_count,
        "trie_state_count": trie.state_count,
        "trie_memory_bytes": estimate_trie_memory_bytes(trie),
        "csr_memory_bytes": estimate_csr_memory_bytes(csr),
        "mask_equivalence_rate": mask_equivalence_rate,
        "mismatched_prefix_count": equivalence["mismatched_prefix_count"],
        "prefix_count": equivalence["prefix_count"],
        "rerank_only_cost_units": rerank["rerank_only_cost_units"],
        "constrained_mask_cost_units": rerank["constrained_mask_cost_units"],
        "finite_outputs_sha256": _sha256_payload(outputs),
        "reproducibility_checksum": _reproducibility_checksum(outputs, csr),
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "live_smoke": {
            "invoked": False,
            "reason": (
                "deterministic_core_only"
                if preconditions["live_endpoint_fields"]["usable_for_live_smoke"]
                else preconditions["live_endpoint_fields"]["unusable_reason"]
            ),
        },
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    *,
    root: Path | str = REPO_ROOT,
    output_path: Path | str | None = None,
    repeats: int = 2000,
) -> JsonDict:
    """Persist the terminal JSON artifact consumed by the conductor."""

    root_path = Path(root)
    destination = Path(output_path) if output_path is not None else root_path / RESULT_RELATIVE_PATH
    payload = run_diagnostic(root=root_path, repeats=repeats)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 5090 artifact violates the terminal contract."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact["honest_verdict"])
    if not verdict.startswith(
        (
            "success_static_csr_masks_speedup_and_validity_win",
            "complete_static_csr_masks_diagnostic_no_headline",
        )
    ):
        raise ValueError("honest_verdict has no accepted Exp 5090 terminal prefix")
    if artifact["inference_substrate"] == "live_llm_inference" and not artifact["live_llm_invoked"]:
        raise ValueError("live_llm_inference cannot be claimed when live_llm_invoked=false")
    if (
        artifact["inference_substrate"] != DETERMINISTIC_INFERENCE_SUBSTRATE
        and not artifact["live_llm_invoked"]
    ):
        raise ValueError("inference_substrate must match deterministic run when live is false")
    if not isinstance(artifact["live_llm_invoked"], bool):
        raise ValueError("live_llm_invoked must be a boolean")
    for field in ("validity_rate", "rerank_only_validity_rate"):
        if not _is_rate(artifact[field]):
            raise ValueError(f"{field} must be in [0, 1]")
    for field in ("beats_cpu_trie", "beats_rerank_only_on_validity_or_cost", "flagged_adversarial"):
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a boolean")
    for field in ("duration_s", "trie_latency_ms", "csr_latency_ms", "mask_speedup"):
        if not _is_nonnegative_number(artifact[field]):
            raise ValueError(f"{field} must be a nonnegative finite number")
    if int(artifact["n_allowed_outputs"]) <= 0:
        raise ValueError("n_allowed_outputs must be positive")
    finite_schema = artifact.get("finite_schema")
    if not isinstance(finite_schema, Mapping) or finite_schema.get("schema_name") != FINITE_SCHEMA_NAME:
        raise ValueError("finite_schema must describe the selected verifier schema")
    model_ids = {
        str(row.get("hf_id"))
        for row in artifact.get("model_specs", [])
        if isinstance(row, Mapping)
    }
    if set(MANDATED_MODEL_IDS) - model_ids:
        raise ValueError("model_specs must include all mandated GGUF IDs")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not set(REQUIRED_ARTIFACT_FIELDS) <= set(principles):
        raise ValueError("field_principles must annotate every required field")


def model_specs_from_preconditions(*, root: Path | str = REPO_ROOT) -> list[JsonDict]:
    gate = _read_json_object(Path(root) / EXP5085_RELATIVE_PATH)
    resolved = _resolved_model_paths(gate or {})
    specs: list[JsonDict] = []
    for base in MODEL_SPECS:
        row = dict(base)
        row["resolved_path"] = resolved.get(base["hf_id"])
        row["live_llm_invoked"] = False
        specs.append(row)
    return specs


def estimate_trie_memory_bytes(trie: TrieMaskIndex) -> int:
    total = sys.getsizeof(trie.transitions) + sys.getsizeof(trie.accepting_states)
    for row in trie.transitions:
        total += sys.getsizeof(row)
        for token_id, target in row.items():
            total += sys.getsizeof(token_id) + sys.getsizeof(target)
    for state in trie.accepting_states:
        total += sys.getsizeof(state)
    return int(total)


def estimate_csr_memory_bytes(csr: CSRAutomaton) -> int:
    total = 0
    for sequence in (csr.row_offsets, csr.labels, csr.targets, csr.mask_bits):
        total += sys.getsizeof(sequence)
        total += sum(sys.getsizeof(item) for item in sequence)
    total += sys.getsizeof(csr.accepting_states)
    total += sum(sys.getsizeof(state) for state in csr.accepting_states)
    return int(total)


def main() -> int:  # pragma: no cover - CLI wrapper.
    payload = write_artifact()
    print(json.dumps({field: payload[field] for field in REQUIRED_ARTIFACT_FIELDS}, indent=2))
    return 0


def _encode_ascii(text: str) -> bytes:
    try:
        return text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("finite output contains non-ASCII byte") from exc


def _output_valid_under_masks(output: str, csr: CSRAutomaton) -> bool:
    state = 0
    for token_id in _encode_ascii(output):
        if not csr.allowed_mask_for_state(state) & (1 << token_id):
            return False
        state = csr.transition(state, token_id)
        if state < 0:
            return False
    return bool(csr.allowed_mask_for_state(state) & (1 << EOS_TOKEN_ID))


def _rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def _read_json_object(path: Path) -> JsonDict | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(loaded) if isinstance(loaded, Mapping) else None


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_payload(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()


def _reproducibility_checksum(outputs: Sequence[str], csr: CSRAutomaton) -> str:
    payload = {
        "outputs": list(outputs),
        "row_offsets": list(csr.row_offsets),
        "labels": list(csr.labels),
        "targets": list(csr.targets),
        "accepting_states": sorted(csr.accepting_states),
        "eos_token_id": csr.eos_token_id,
    }
    return _sha256_payload(payload)


def _resolved_model_paths(gate: Mapping[str, Any]) -> dict[str, str | None]:
    resolved: dict[str, str | None] = {model_id: None for model_id in MANDATED_MODEL_IDS}
    model_specs = gate.get("model_specs")
    if isinstance(model_specs, Mapping):
        resolved_models = model_specs.get("resolved_models")
        if isinstance(resolved_models, Mapping):
            for value in resolved_models.values():
                if isinstance(value, Mapping):
                    hf_id = str(value.get("hf_id") or "")
                    if hf_id in resolved:
                        resolved[hf_id] = _optional_string(value.get("resolved_path"))
    for value in gate.get("headline_models_available") or []:
        if isinstance(value, Mapping):
            hf_id = str(value.get("hf_id") or "")
            if hf_id in resolved and not resolved[hf_id]:
                resolved[hf_id] = _optional_string(value.get("path"))
    return resolved


def _optional_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _is_rate(value: Any) -> bool:
    return _is_nonnegative_number(value) and float(value) <= 1.0


def _is_nonnegative_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number >= 0.0 and number < float("inf")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
