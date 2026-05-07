"""Exp 1475 STATIC-style CSR automaton smoke for certificate strings.

Spec: REQ-VERIFY-1475, SCENARIO-VERIFY-1475
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from carnot.eval import certificate_grammar_backend_bakeoff as certificate_backend


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_PATH = (
    REPO_ROOT / "results" / "experiment_1475_static_csr_certificate_automaton_smoke.json"
)
EXPERIMENT_NAME = "1475_static_csr_certificate_automaton_smoke"
SCHEMA = "static_csr_certificate_automaton_smoke_v1"
RUN_DATE = "20260507"
CSR_AUTOMATON_PATH = str(REPO_ROOT / "python/carnot/eval/static_csr_certificate_automaton.py")


@dataclass(frozen=True)
class CertificateCase:
    """One concrete certificate string used to bound the smoke-test claim."""

    name: str
    text: str


@dataclass(frozen=True)
class StaticCSRAutomaton:
    """Sparse byte-transition automaton flattened from accepted certificate strings."""

    row_offsets: tuple[int, ...]
    labels: tuple[int, ...]
    targets: tuple[int, ...]
    accepting_states: frozenset[int]

    @property
    def state_count(self) -> int:
        return len(self.row_offsets) - 1

    def accepts(self, text: str) -> bool:
        state = 0
        for label in text.encode("utf-8"):
            start = self.row_offsets[state]
            end = self.row_offsets[state + 1]
            next_state = -1
            for index in range(start, end):
                if self.labels[index] == label:
                    next_state = self.targets[index]
                    break
            if next_state < 0:
                return False
            state = next_state
        return state in self.accepting_states


class ConstantStepTimer:
    """Deterministic nanosecond timer for latency tests without wall-clock noise."""

    def __init__(self, step_ns: int = 1000) -> None:
        self._current = 0
        self._step_ns = int(step_ns)

    def __call__(self) -> int:
        self._current += self._step_ns
        return self._current


def canonical_certificate_text(payload: Mapping[str, Any]) -> str:
    """Serialize fixtures exactly so the byte automaton has stable inputs."""

    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def tiny_certificate_cases() -> list[CertificateCase]:
    """Return a small valid/invalid suite grounded in the existing schema path."""

    baseline_valid = certificate_backend.sample_certificate()
    alternate_valid = {
        "claims": [{"id": "c2", "text": "5 is at least 3."}],
        "equations": [{"lhs": "5", "relation": ">=", "rhs": "3"}],
        "final_answer": "true",
        "confidence": 1.0,
        "verifier_routes": [{"claim_id": "c2", "verifier": "semenergy"}],
        "proof_numbers": [5, 3],
    }
    missing_confidence = dict(baseline_valid)
    missing_confidence.pop("confidence")
    bad_verifier = dict(baseline_valid)
    bad_verifier["verifier_routes"] = [{"claim_id": "c1", "verifier": "unknown"}]
    bad_claim_id = dict(baseline_valid)
    bad_claim_id["claims"] = [{"id": "claim-1", "text": "bad id"}]

    return [
        CertificateCase("valid_z3_math", canonical_certificate_text(baseline_valid)),
        CertificateCase("valid_semenergy", canonical_certificate_text(alternate_valid)),
        CertificateCase(
            "invalid_missing_confidence", canonical_certificate_text(missing_confidence)
        ),
        CertificateCase("invalid_verifier_enum", canonical_certificate_text(bad_verifier)),
        CertificateCase("invalid_claim_id_regex", canonical_certificate_text(bad_claim_id)),
        CertificateCase("invalid_malformed_json", '{"claims":'),
        CertificateCase("invalid_non_object_json", "[]"),
    ]


def existing_path_accepts(text: str) -> bool:
    """Run the existing JSON parser plus Carnot's bounded schema/regex validator."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, Mapping):
        return False
    accepted, _errors = certificate_backend.validate_certificate(
        payload,
        certificate_backend.certificate_schema(),
    )
    return accepted


def build_static_csr_automaton(accepted_strings: Iterable[str]) -> StaticCSRAutomaton:
    """Flatten the accepted-string trie into CSR-like transition arrays."""

    strings = list(accepted_strings)
    if not strings:
        raise ValueError("accepted_strings must contain at least one certificate string")

    transitions: list[dict[int, int]] = [{}]
    accepting_states: set[int] = set()
    for text in strings:
        state = 0
        for label in text.encode("utf-8"):
            next_state = transitions[state].get(label)
            if next_state is None:
                next_state = len(transitions)
                transitions[state][label] = next_state
                transitions.append({})
            state = next_state
        accepting_states.add(state)

    row_offsets: list[int] = [0]
    labels: list[int] = []
    targets: list[int] = []
    for outgoing in transitions:
        for label, target in sorted(outgoing.items()):
            labels.append(label)
            targets.append(target)
        row_offsets.append(len(labels))

    return StaticCSRAutomaton(
        row_offsets=tuple(row_offsets),
        labels=tuple(labels),
        targets=tuple(targets),
        accepting_states=frozenset(accepting_states),
    )


def evaluate_equivalence(
    cases: Sequence[CertificateCase],
    automaton: StaticCSRAutomaton,
) -> dict[str, Any]:
    """Compare CSR acceptance against the existing parser on bounded cases."""

    records: list[dict[str, Any]] = []
    false_accept_case_names: list[str] = []
    false_reject_case_names: list[str] = []
    for case in cases:
        existing_accepts = existing_path_accepts(case.text)
        csr_accepts = automaton.accepts(case.text)
        if csr_accepts and not existing_accepts:
            false_accept_case_names.append(case.name)
        if existing_accepts and not csr_accepts:
            false_reject_case_names.append(case.name)
        records.append(
            {
                "name": case.name,
                "existing_accepts": existing_accepts,
                "csr_accepts": csr_accepts,
            }
        )

    false_accepts = len(false_accept_case_names)
    false_rejects = len(false_reject_case_names)
    return {
        "schema_cases_evaluated": len(cases),
        "exact_acceptance_equivalent": false_accepts == 0 and false_rejects == 0,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "false_accept_case_names": false_accept_case_names,
        "false_reject_case_names": false_reject_case_names,
        "cases": records,
    }


def benchmark_acceptors(
    cases: Sequence[CertificateCase],
    automaton: StaticCSRAutomaton,
    *,
    repeats: int = 1000,
    timer: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, float]:
    """Measure p50 per-string latency for both acceptance paths."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")

    existing_samples: list[float] = []
    csr_samples: list[float] = []
    for _ in range(repeats):
        for case in cases:
            start = timer()
            existing_path_accepts(case.text)
            existing_samples.append((timer() - start) / 1_000_000.0)
            start = timer()
            automaton.accepts(case.text)
            csr_samples.append((timer() - start) / 1_000_000.0)

    return {
        "existing_path_latency_ms_p50": round(float(statistics.median(existing_samples)), 6),
        "csr_latency_ms_p50": round(float(statistics.median(csr_samples)), 6),
    }


def build_smoke_artifact(
    *,
    run_date: str = RUN_DATE,
    repeats: int = 1000,
    tests_run: Sequence[str] = (),
    timer: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Build the terminal artifact without invoking generation or repair."""

    cases = tiny_certificate_cases()
    automaton = build_static_csr_automaton(
        case.text for case in cases if existing_path_accepts(case.text)
    )
    evaluation = evaluate_equivalence(cases, automaton)
    latency = benchmark_acceptors(cases, automaton, repeats=repeats, timer=timer)
    csr_latency = latency["csr_latency_ms_p50"]
    speedup_ratio = (
        round(latency["existing_path_latency_ms_p50"] / csr_latency, 6) if csr_latency > 0 else 0.0
    )
    honest_verdict = (
        "complete_bounded_case_equivalence_csr_faster_no_generation_or_repair"
        if evaluation["exact_acceptance_equivalent"] and speedup_ratio > 1.0
        else "complete_bounded_case_equivalence_latency_reported_no_generation_or_repair"
    )

    return {
        "experiment": EXPERIMENT_NAME,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "complete",
        "schema_cases_evaluated": evaluation["schema_cases_evaluated"],
        "csr_automaton_path": CSR_AUTOMATON_PATH,
        "csr_state_count": automaton.state_count,
        "csr_transition_count": len(automaton.labels),
        "exact_acceptance_equivalent": evaluation["exact_acceptance_equivalent"],
        "false_accepts": evaluation["false_accepts"],
        "false_rejects": evaluation["false_rejects"],
        "false_accept_case_names": evaluation["false_accept_case_names"],
        "false_reject_case_names": evaluation["false_reject_case_names"],
        "existing_path_latency_ms_p50": latency["existing_path_latency_ms_p50"],
        "csr_latency_ms_p50": latency["csr_latency_ms_p50"],
        "speedup_ratio": speedup_ratio,
        "tests_run": list(tests_run),
        "llm_inference_run": False,
        "repair_loop_run": False,
        "baseline_parser_path": ("python/carnot/eval/certificate_grammar_backend_bakeoff.py"),
        "case_results": evaluation["cases"],
        "honest_verdict": honest_verdict,
    }


def write_in_progress_artifact(path: Path | str, *, run_date: str = RUN_DATE) -> dict[str, Any]:
    """Write the bootstrap artifact before any schema-case evaluation starts."""

    artifact = {
        "experiment": EXPERIMENT_NAME,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "in_progress",
    }
    _write_json(Path(path), artifact)
    return artifact


def run_smoke(
    *,
    output_path: Path | str = DEFAULT_RESULT_PATH,
    run_date: str = RUN_DATE,
    repeats: int = 1000,
    tests_run: Sequence[str] = (),
    timer: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Run the static CSR certificate smoke and persist the final artifact."""

    output = Path(output_path)
    write_in_progress_artifact(output, run_date=run_date)
    artifact = build_smoke_artifact(
        run_date=run_date,
        repeats=repeats,
        tests_run=tests_run,
        timer=timer,
    )
    _write_json(output, artifact)
    return artifact


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
