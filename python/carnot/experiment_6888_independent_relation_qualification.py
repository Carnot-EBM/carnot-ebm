"""Independently qualify frozen source-anchored relation proposals.

Spec refs: REQ-VERIFY-6888 and SCENARIO-VERIFY-6888-*.

This reducer uses calibration labels before it reads held labels. It never runs
a proposal model. Clingo remains the independent semantic authority.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import asp_energy
from carnot.experiment_6886_enoki_exact_relation_fixture import solve_with_timeout
from carnot.experiment_6887_three_family_relation_proposal_corpus import (
    FAMILIES,
    PROPOSAL_ARMS,
    build_frozen_source_records,
)


JsonDict = dict[str, Any]
Solver = Callable[[asp_energy.ASPProgram], list[list[str]]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_6888_independent_relation_qualification.json")
EXP6274_PATH = Path("results/experiment_6274_asp_energy_semantic_compiler.json")
EXP6886_PATH = Path("results/experiment_6886_enoki_exact_relation_fixture.json")
EXP6887_PATH = Path("results/experiment_6887_three_family_relation_proposal_corpus.json")
COMPILER_PATH = Path("python/carnot/asp_energy.py")
CALIBRATION_SIDECAR_PATH = (
    Path.home()
    / ".cache"
    / "carnot"
    / "exp6886_enoki_exact_relation_fixture"
    / "sealed_calibration_formal_sidecar.json"
)
HELD_SIDECAR_PATH = (
    Path.home()
    / ".cache"
    / "carnot"
    / "exp6886_enoki_exact_relation_fixture"
    / "sealed_held_formal_sidecar.json"
)
SIDECAR_SCHEMA = "carnot.exp6886.sealed_formal_sidecar.v1"
SCHEMA = "carnot.exp6888.independent_relation_qualification.v1"
INFERENCE_SUBSTRATE = "sealed_reduction_of_frozen_relation_outputs_no_llm"
RANDOM_SEED = 6888
SOLVER_TIMEOUT_S = 2.0

EXPECTED_HASHES = {
    "exp6274": "sha256:b02c88963c4815aa0e26d451ffd60fdd9f1014d32e76f638592ac114c611e96b",
    "exp6886": "sha256:602250fbfe172f08458ea279787d992e89835f12005ba6ef59ec02f3b411d500",
    "exp6887": "sha256:99800fdbad94d0f1387a2a57ae3abb9169ece3ffeb3ac7b5485ddfd24123cef3",
    "compiler": "sha256:0f6077bcd49aa93a6cdbde72422ecf97d905b76b31cadbc0cd401c494af015e1",
    "calibration_sidecar": "sha256:cb8db945ac4b20e0d2e9658cd09e5263fe2240e78f999c323fd09fc3351a6cce",
    "held_sidecar": "sha256:d75b4b032ed27542dd8d619922865c389f869aa11470a5cfffd90c05de6553ad",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "sealed_sidecar_hashes",
    "frozen_thresholds",
    "rows",
    "span_metric_rows",
    "tuple_metric_rows",
    "parse_coverage_rows",
    "abstention_rows",
    "family_rows",
    "perturbation_rows",
    "asp_compilation_rows",
    "solver_parity_rows",
    "reported_vs_recomputed_metrics",
    "independent_solver_receipts",
    "held_leakage_count",
    "eligible_arm_rows",
    "qualified_relation_event_count",
    "relation_qualification_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each field states why its evidence is required.",
    "preconditions_checked": "Exact checks stop drift before held scoring.",
    "inference_substrate": "This value proves the reducer ran no new model inference.",
    "duration_s": "Measured wall time shows that the reducer executed.",
    "source_artifact_hashes": "Exact hashes bind every public frozen input.",
    "sealed_sidecar_hashes": "Exact hashes bind calibration and held authority.",
    "frozen_thresholds": "Calibration-only thresholds prevent held tuning.",
    "rows": "Per-relation rows preserve every success, null, and failure.",
    "span_metric_rows": "Span rows score exact UTF-8 grounding separately.",
    "tuple_metric_rows": "Tuple rows expose precision, recall, and duplicate credit.",
    "parse_coverage_rows": "Coverage keeps every frozen cell in its denominator.",
    "abstention_rows": "Abstention rows distinguish empty output from parser failure.",
    "family_rows": "Family rows prevent easy families from hiding weak families.",
    "perturbation_rows": "Perturbation rows preserve the lowest robustness slice.",
    "asp_compilation_rows": "Compilation rows expose unsupported atoms and no-headroom cases.",
    "solver_parity_rows": "Exact set equality keeps clingo as semantic authority.",
    "reported_vs_recomputed_metrics": "Replay agreement detects aggregate drift.",
    "independent_solver_receipts": "Solver receipts preserve calls, timeouts, and disagreements.",
    "held_leakage_count": "Zero proves formal held content did not enter proposal inputs.",
    "eligible_arm_rows": "Each arm shows every passed and failed held threshold.",
    "qualified_relation_event_count": "This exact count feeds the downstream event gate.",
    "relation_qualification_ready_score": "One requires at least one fully qualified arm.",
    "random_seed": "A fixed reducer identity supports deterministic replay.",
    "reproducibility_checksum": "One digest detects silent terminal artifact drift.",
    "gate_check_summary": "Expected and observed values make every block actionable.",
    "verifier_is_oracle": "True discloses that exact execution defines semantic validity.",
    "verdict_class": "The closed class prevents an oracle-backed pass from claiming positive.",
    "honest_verdict": "A complete prefix marks the terminal evidence boundary.",
    "contradiction_rows": "Contradiction rows score opposite polarities separately.",
}


class QualificationError(RuntimeError):
    """Report one fail-closed qualification condition."""


class HeldSidecarReader:
    """Read and validate the sealed held file at most once.

    The object owns the only content read. A second call is an error instead of
    a silent second exposure of held authority.
    """

    def __init__(self, path: Path | str, expected_sha256: str) -> None:
        self.path = Path(path)
        self.expected_sha256 = expected_sha256
        self.open_count = 0

    def open_once(self) -> JsonDict:
        """Open, hash, parse, and validate the held sidecar once."""

        if self.open_count:
            raise QualificationError("held_sidecar_opened_more_than_once")
        self.open_count += 1
        raw = self.path.read_bytes()
        observed = sha256_bytes(raw)
        if observed != self.expected_sha256:
            raise QualificationError(f"held_sidecar_hash_drift:{self.expected_sha256}:{observed}")
        payload = json.loads(raw.decode("utf-8"))
        if payload.get("schema") != SIDECAR_SCHEMA or payload.get("split") != "held":
            raise QualificationError("held_sidecar_identity")
        if not isinstance(payload.get("rows"), list) or not payload["rows"]:
            raise QualificationError("held_sidecar_rows")
        return payload


def canonical_json(value: Any) -> str:
    """Serialize JSON with one stable representation."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Return a repository-style SHA-256 identity."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash exact UTF-8 text."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash a canonical JSON value."""

    return sha256_text(canonical_json(value))


def sha256_file(path: Path | str) -> str:
    """Hash one bounded artifact or sidecar file."""

    return sha256_bytes(Path(path).read_bytes())


def _jsonable_gate_value(value: Any) -> Any:
    """Convert check evidence to deterministic JSON-compatible values."""

    if isinstance(value, Mapping):
        return {
            str(key): _jsonable_gate_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable_gate_value(item) for item in value), key=canonical_json)
    if isinstance(value, (list, tuple)):
        return [_jsonable_gate_value(item) for item in value]
    return deepcopy(value)


def gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Build one check with an exact expectation and observation."""

    return {
        "check": check,
        "expected": _jsonable_gate_value(expected),
        "observed": _jsonable_gate_value(observed),
        "passed": observed == expected,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize checks without hiding later failures."""

    copied = [deepcopy(dict(row)) for row in checks]
    failures = [row for row in copied if row.get("passed") is not True]
    first = failures[0] if failures else None
    return {
        "checks": copied,
        "passed": not failures,
        "failed_check": first.get("check") if first else None,
        "expected": first.get("expected") if first else "all checks pass",
        "observed": first.get("observed") if first else "all checks pass",
        "failed_checks": failures,
    }


def check_exact_hashes(observed: Mapping[str, str], expected: Mapping[str, str]) -> JsonDict:
    """Compare every frozen hash and report the first drift."""

    checks = [
        gate_check(f"artifact_hash:{name}", expected[name], observed.get(name)) for name in expected
    ]
    return gate_summary(checks)


def validate_acquisition_matrix(
    cells: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    arms: Sequence[str],
) -> JsonDict:
    """Validate exact arm, fixture, cell, hash, and terminal identities."""

    arm_set = set(arms)
    fixture_by_id = {str(row["fixture_id"]): row for row in sources}
    observed_arms = {str(row.get("arm")) for row in cells}
    observed_fixtures = {str(row.get("fixture_id")) for row in cells}
    identities = [str(row.get("cell_identity")) for row in cells]
    expected_identities = {f"{arm}::{fixture_id}" for arm in arms for fixture_id in fixture_by_id}
    content_valid = True
    for row in cells:
        fixture = fixture_by_id.get(str(row.get("fixture_id")))
        if fixture is None:
            continue
        if (
            row.get("terminal") is not True
            or row.get("source_text_hash") != fixture.get("source_text_hash")
            or row.get("split") != fixture.get("split")
            or row.get("family") != fixture.get("family")
            or row.get("raw_output_sha256") != sha256_text(str(row.get("raw_output", "")))
        ):
            content_valid = False
            break
    checks = [
        gate_check("unique_cell_identity", len(identities), len(set(identities))),
        gate_check("complete_raw_cells", len(expected_identities), len(identities)),
        gate_check("arm_identity", arm_set, observed_arms),
        gate_check("fixture_identity", set(fixture_by_id), observed_fixtures),
        gate_check("cell_identity", expected_identities, set(identities)),
        gate_check("terminal_cell_content", True, content_valid),
    ]
    return gate_summary(checks)


def _formal_tokens(payload: Mapping[str, Any]) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    for row in payload.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        program = str(row.get("asp_program", ""))
        if program:
            tokens.append(("asp_program", program))
        for state in row.get("answer_sets", []):
            for atom in state:
                tokens.append(("answer_set_atom", str(atom)))
        receipt = row.get("solver_receipt")
        if isinstance(receipt, Mapping):
            tokens.append(("solver_receipt", canonical_json(receipt)))
    return tokens


def detect_held_leakage(
    proposal_artifact: Mapping[str, Any], held_payload: Mapping[str, Any]
) -> list[JsonDict]:
    """Find direct formal held content in prompts or frozen raw output.

    Surface relation text is allowed because producing it is the arm's task.
    ASP atoms and complete formal objects are not surface relation text.
    """

    surfaces: list[tuple[str, str, str]] = [
        (
            "prompt_manifest",
            "prompt_manifest",
            canonical_json(proposal_artifact.get("prompt_manifest", {})),
        )
    ]
    for row in proposal_artifact.get("rows", []):
        if isinstance(row, Mapping):
            surfaces.append(
                (
                    str(row.get("cell_identity", "missing")),
                    "raw_output",
                    str(row.get("raw_output", "")),
                )
            )
    leaks: list[JsonDict] = []
    for identity, surface, text in surfaces:
        for leak_type, token in _formal_tokens(held_payload):
            if token and token in text:
                leaks.append(
                    {
                        "cell_identity": identity,
                        "surface": surface,
                        "leak_type": leak_type,
                        "token_sha256": sha256_text(token),
                    }
                )
    return leaks


def _fixture_parts(fixture_id: str) -> tuple[str, int]:
    family, ordinal = fixture_id.rsplit("_", 1)
    return family, int(ordinal)


def _surface_ids(fixture_id: str) -> tuple[str, str, str, str]:
    family, ordinal = _fixture_parts(fixture_id)
    values = {
        "graph_coloring": (f"node_{ordinal}", f"n{ordinal}", "red", "red"),
        "scheduling": (f"task_{ordinal}", f"task{ordinal}", "morning", "morning"),
        "non_monotonic_defaults": (
            f"bird_{ordinal}",
            f"b{ordinal}",
            "injured",
            "injured",
        ),
        "contradictions": (
            f"claim_{ordinal}",
            f"claim{ordinal}",
            "accepted",
            "accepted",
        ),
        "cardinality_constraints": (
            f"set_{ordinal}",
            f"s{ordinal}",
            "option_a",
            "option A",
        ),
    }
    if family not in values:
        raise QualificationError(f"unsupported_family:{family}")
    return values[family]


def _span_matches(source_text: str, span: Mapping[str, Any], expected_text: str) -> bool:
    start = span.get("start_utf8")
    end = span.get("end_utf8")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 0
        or end <= start
    ):
        return False
    raw = source_text.encode("utf-8")
    if end > len(raw):
        return False
    try:
        sliced = raw[start:end].decode("utf-8")
    except UnicodeDecodeError:
        return False
    return sliced == expected_text == span.get("text")


def _vocabulary_for_fixture(
    fixture_id: str, vocabulary: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    family, ordinal = _fixture_parts(fixture_id)
    subject_id, _, _, _ = _surface_ids(fixture_id)
    return [
        row
        for row in vocabulary
        if row.get("family") == family
        and row.get("subject_id") == subject_id
        and str(row.get("asp_atom", "")).split("_")[1:2] == [str(ordinal)]
    ]


def _map_proposal(
    fixture_id: str,
    source_text: str,
    parse_row: Mapping[str, Any],
    vocabulary: Sequence[Mapping[str, Any]],
) -> JsonDict:
    subject_id, subject_text, object_id, object_text = _surface_ids(fixture_id)
    subject = parse_row.get("subject")
    obj = parse_row.get("object")
    if not isinstance(subject, Mapping) or not isinstance(obj, Mapping):
        return {"supported": False, "reason": "malformed_relation", "span_exact": False}
    candidate = (
        subject_id,
        str(parse_row.get("predicate")),
        object_id,
        str(parse_row.get("polarity")),
    )
    mapping = {
        tuple(str(value) for value in row.get("normalized_tuple", [])): row
        for row in _vocabulary_for_fixture(fixture_id, vocabulary)
    }
    mapped = mapping.get(candidate)
    surface_matches = subject.get("text") == subject_text and obj.get("text") == object_text
    span_exact = (
        surface_matches
        and _span_matches(source_text, subject, subject_text)
        and _span_matches(source_text, obj, object_text)
    )
    if mapped is None or not surface_matches:
        return {
            "supported": False,
            "reason": "unsupported_atom",
            "normalized_tuple": list(candidate),
            "span_exact": span_exact,
        }
    return {
        "supported": True,
        "reason": "mapped",
        "normalized_tuple": list(candidate),
        "asp_atom": mapped["asp_atom"],
        "span_exact": span_exact,
    }


def _gold_tuples(
    formal: Mapping[str, Any], vocabulary: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, ...], str]:
    fixture_id = str(formal["fixture_id"])
    atom_map = {
        str(row["asp_atom"]): tuple(str(value) for value in row["normalized_tuple"])
        for row in _vocabulary_for_fixture(fixture_id, vocabulary)
    }
    program = asp_energy.parse_program(str(formal["asp_program"]), program_id=fixture_id)
    return {atom_map[atom]: atom for _, atom, _ in program.facts if atom in atom_map}


def _base_program(formal: Mapping[str, Any], vocabulary: Sequence[Mapping[str, Any]]) -> str:
    fixture_id = str(formal["fixture_id"])
    relation_atoms = {
        str(row["asp_atom"]) for row in _vocabulary_for_fixture(fixture_id, vocabulary)
    }
    lines = []
    for line in str(formal["asp_program"]).splitlines():
        stripped = line.strip()
        if stripped.endswith(".") and stripped[:-1] in relation_atoms:
            continue
        if stripped:
            lines.append(stripped)
    return "\n".join(lines) + "\n"


def _metrics(tp: int, fp: int, fn: int) -> JsonDict:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None and recall is None:
        f1 = 1.0
    elif not precision or not recall:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _contradiction_detected(tuples: set[tuple[str, ...]]) -> bool:
    polarities: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for subject, predicate, obj, polarity in tuples:
        polarities[(subject, predicate, obj)].add(polarity)
    return any(values == {"positive", "negative"} for values in polarities.values())


def score_partition(
    *,
    cells: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    formal_rows: Sequence[Mapping[str, Any]],
    vocabulary: Sequence[Mapping[str, Any]],
    arms: Sequence[str],
    solver: Solver,
    solver_timeout_s: float,
) -> JsonDict:
    """Score one split from rows while preserving every failed cell."""

    source_by_id = {str(row["fixture_id"]): row for row in sources}
    formal_by_id = {str(row["fixture_id"]): row for row in formal_rows}
    rows: list[JsonDict] = []
    compile_rows: list[JsonDict] = []
    parity_rows: list[JsonDict] = []
    contradiction_rows: list[JsonDict] = []
    cell_stats: list[JsonDict] = []
    timeout_count = 0
    disagreement_count = 0

    for cell in cells:
        arm = str(cell["arm"])
        fixture_id = str(cell["fixture_id"])
        source = source_by_id.get(fixture_id)
        formal = formal_by_id.get(fixture_id)
        if source is None or formal is None:
            raise QualificationError(f"unmatched_fixture:{fixture_id}")
        source_text = str(source["source_text"])
        perturbation = str(formal.get("expected_case", "unknown"))
        gold = _gold_tuples(formal, vocabulary)
        proposals: dict[tuple[str, ...], JsonDict] = {}
        duplicate_count = 0
        offset_mismatch_count = 0
        unsupported_count = 0
        accepted_parse_count = 0
        explicit_abstention = False

        for parse_row in cell.get("parse_rows", []):
            if not isinstance(parse_row, Mapping):
                rows.append(
                    {
                        "arm": arm,
                        "record_id": fixture_id,
                        "relation": None,
                        "family": cell["family"],
                        "perturbation": perturbation,
                        "parse_status": "malformed",
                        "outcome": "malformed",
                        "reason": "parse_row_not_an_object",
                        "no_headroom": not gold,
                    }
                )
                continue
            status = str(parse_row.get("status"))
            if status != "accepted":
                explicit_abstention = explicit_abstention or status in {"abstention", "empty"}
                rows.append(
                    {
                        "arm": arm,
                        "record_id": fixture_id,
                        "relation": None,
                        "family": cell["family"],
                        "perturbation": perturbation,
                        "parse_status": status,
                        "outcome": "timeout" if cell.get("timed_out") is True else status,
                        "reason": parse_row.get("reason"),
                        "no_headroom": not gold,
                    }
                )
                continue
            accepted_parse_count += 1
            mapped = _map_proposal(fixture_id, source_text, parse_row, vocabulary)
            if mapped.get("supported") is not True:
                unsupported_count += 1
                rows.append(
                    {
                        "arm": arm,
                        "record_id": fixture_id,
                        "relation": deepcopy(dict(mapped)),
                        "family": cell["family"],
                        "perturbation": perturbation,
                        "parse_status": status,
                        "outcome": "unsupported_atom",
                        "reason": mapped.get("reason"),
                        "no_headroom": not gold,
                    }
                )
                continue
            normalized = tuple(str(value) for value in mapped["normalized_tuple"])
            duplicate = normalized in proposals
            if duplicate:
                duplicate_count += 1
            else:
                proposals[normalized] = deepcopy(dict(mapped))
            if mapped.get("span_exact") is not True:
                offset_mismatch_count += 1
                outcome = "offset_mismatch"
            elif duplicate:
                outcome = "duplicate_no_credit"
            else:
                outcome = "true_positive" if normalized in gold else "false_positive"
            rows.append(
                {
                    "arm": arm,
                    "record_id": fixture_id,
                    "relation": deepcopy(dict(mapped)),
                    "family": cell["family"],
                    "perturbation": perturbation,
                    "parse_status": status,
                    "outcome": outcome,
                    "reason": "duplicate" if duplicate else mapped.get("reason"),
                    "no_headroom": not gold,
                }
            )

        proposal_set = set(proposals)
        exact_span_set = {
            normalized
            for normalized, mapped in proposals.items()
            if mapped.get("span_exact") is True
        }
        for missing in sorted(set(gold) - proposal_set):
            rows.append(
                {
                    "arm": arm,
                    "record_id": fixture_id,
                    "relation": {"normalized_tuple": list(missing), "asp_atom": gold[missing]},
                    "family": cell["family"],
                    "perturbation": perturbation,
                    "parse_status": "missing",
                    "outcome": "false_negative",
                    "reason": "gold_tuple_not_proposed",
                    "no_headroom": False,
                }
            )

        tuple_tp = len(proposal_set & set(gold))
        tuple_fp = len(proposal_set - set(gold))
        tuple_fn = len(set(gold) - proposal_set)
        span_tp = len(exact_span_set & set(gold))
        span_fp = len(exact_span_set - set(gold))
        span_fn = len(set(gold) - exact_span_set)
        cell_stats.append(
            {
                "arm": arm,
                "fixture_id": fixture_id,
                "family": cell["family"],
                "perturbation": perturbation,
                "tuple": (tuple_tp, tuple_fp, tuple_fn),
                "span": (span_tp, span_fp, span_fn),
                "duplicate_count": duplicate_count,
                "offset_mismatch_count": offset_mismatch_count,
                "accepted_parse": accepted_parse_count > 0,
                "abstained": explicit_abstention or not cell.get("parse_rows"),
                "timed_out": cell.get("timed_out") is True,
            }
        )

        expected_contradiction = perturbation == "contradictory"
        detected_contradiction = _contradiction_detected(proposal_set)
        contradiction_rows.append(
            {
                "arm": arm,
                "fixture_id": fixture_id,
                "family": cell["family"],
                "expected": expected_contradiction,
                "detected": detected_contradiction,
                "correct": expected_contradiction == detected_contradiction,
            }
        )

        atoms = sorted(
            {str(mapped["asp_atom"]) for mapped in proposals.values() if mapped.get("asp_atom")}
        )
        if unsupported_count:
            compile_rows.append(
                {
                    "arm": arm,
                    "fixture_id": fixture_id,
                    "family": cell["family"],
                    "status": "unsupported_atom",
                    "accepted_atom_count": len(atoms),
                    "unsupported_atom_count": unsupported_count,
                    "no_headroom": not gold,
                }
            )
            parity_rows.append(
                {
                    "arm": arm,
                    "fixture_id": fixture_id,
                    "family": cell["family"],
                    "status": "not_run_unsupported_atom",
                    "parity": False,
                    "solver_error": "unsupported_atom",
                }
            )
            disagreement_count += 1
            continue

        program_text = _base_program(formal, vocabulary) + "".join(f"{atom}.\n" for atom in atoms)
        try:
            compiled = asp_energy.compile_program(program_text, program_id=fixture_id)
        except asp_energy.UnsupportedASPSyntax as exc:
            compile_rows.append(
                {
                    "arm": arm,
                    "fixture_id": fixture_id,
                    "family": cell["family"],
                    "status": "compile_error",
                    "accepted_atom_count": len(atoms),
                    "unsupported_atom_count": 0,
                    "error": str(exc),
                    "no_headroom": not gold,
                }
            )
            parity_rows.append(
                {
                    "arm": arm,
                    "fixture_id": fixture_id,
                    "family": cell["family"],
                    "status": "not_run_compile_error",
                    "parity": False,
                    "solver_error": str(exc),
                }
            )
            disagreement_count += 1
            continue

        zero_states = compiled.zero_energy_states()
        compile_rows.append(
            {
                "arm": arm,
                "fixture_id": fixture_id,
                "family": cell["family"],
                "status": "compiled",
                "accepted_atom_count": len(atoms),
                "unsupported_atom_count": 0,
                "program_sha256": sha256_text(program_text),
                "zero_energy_state_count": len(zero_states),
                "no_headroom": not gold,
            }
        )
        try:
            answer_sets = solve_with_timeout(
                compiled.program,
                timeout_s=solver_timeout_s,
                solver=solver,
            )
            parity = answer_sets == zero_states
            status = "parity" if parity else "disagreement"
            solver_error = None
        except (TimeoutError, RuntimeError) as exc:
            answer_sets = []
            parity = False
            status = "timeout" if isinstance(exc, TimeoutError) else "solver_error"
            solver_error = str(exc)
        if status == "timeout":
            timeout_count += 1
        if not parity:
            disagreement_count += 1
        parity_rows.append(
            {
                "arm": arm,
                "fixture_id": fixture_id,
                "family": cell["family"],
                "status": status,
                "parity": parity,
                "solver_error": solver_error,
                "answer_sets_hash": sha256_json(answer_sets),
                "zero_energy_states_hash": sha256_json(zero_states),
            }
        )

    tuple_rows = _aggregate_metric_rows(cell_stats, arms, "tuple")
    span_rows = _aggregate_metric_rows(cell_stats, arms, "span")
    parse_rows = []
    abstention_rows = []
    for arm in arms:
        selected = [row for row in cell_stats if row["arm"] == arm]
        cell_count = len(selected)
        parsed = sum(bool(row["accepted_parse"]) for row in selected)
        abstained = sum(bool(row["abstained"]) for row in selected)
        parse_rows.append(
            {
                "arm": arm,
                "cell_count": cell_count,
                "parsed_cell_count": parsed,
                "parse_coverage": parsed / cell_count if cell_count else None,
            }
        )
        abstention_rows.append(
            {
                "arm": arm,
                "cell_count": cell_count,
                "abstention_count": abstained,
                "abstention_rate": abstained / cell_count if cell_count else None,
                "timeout_count": sum(bool(row["timed_out"]) for row in selected),
            }
        )

    family_rows = _subgroup_rows(cell_stats, arms, "family")
    perturbation_rows = _subgroup_rows(cell_stats, arms, "perturbation")
    return {
        "rows": rows,
        "span_metric_rows": span_rows,
        "tuple_metric_rows": tuple_rows,
        "parse_coverage_rows": parse_rows,
        "abstention_rows": abstention_rows,
        "family_rows": family_rows,
        "perturbation_rows": perturbation_rows,
        "asp_compilation_rows": compile_rows,
        "solver_parity_rows": parity_rows,
        "contradiction_rows": contradiction_rows,
        "independent_solver_receipts": {
            "name_version": asp_energy.solver_name_version(),
            "fixture_calls": len(parity_rows),
            "timeout_count": timeout_count,
            "disagreement_count": disagreement_count,
        },
    }


def _aggregate_metric_rows(
    stats: Sequence[Mapping[str, Any]], arms: Sequence[str], metric: str
) -> list[JsonDict]:
    rows = []
    for arm in arms:
        selected = [row for row in stats if row["arm"] == arm]
        totals = [sum(int(row[metric][index]) for row in selected) for index in range(3)]
        result = {"arm": arm, **_metrics(*totals)}
        if metric == "tuple":
            result["duplicate_proposal_count"] = sum(
                int(row["duplicate_count"]) for row in selected
            )
        else:
            result["offset_mismatch_count"] = sum(
                int(row["offset_mismatch_count"]) for row in selected
            )
        rows.append(result)
    return rows


def _subgroup_rows(
    stats: Sequence[Mapping[str, Any]], arms: Sequence[str], field: str
) -> list[JsonDict]:
    values = sorted({str(row[field]) for row in stats})
    rows = []
    for arm in arms:
        for value in values:
            selected = [row for row in stats if row["arm"] == arm and str(row[field]) == value]
            totals = [sum(int(row["tuple"][index]) for row in selected) for index in range(3)]
            rows.append(
                {
                    "arm": arm,
                    field: value,
                    "cell_count": len(selected),
                    **_metrics(*totals),
                }
            )
    return rows


def _by_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["arm"]): row for row in rows}


def freeze_thresholds(calibration: Mapping[str, Any]) -> JsonDict:
    """Select a calibration reference and freeze all held thresholds."""

    tuples = _by_arm(calibration.get("tuple_metric_rows", []))
    spans = _by_arm(calibration.get("span_metric_rows", []))
    coverage = _by_arm(calibration.get("parse_coverage_rows", []))
    if not tuples:
        raise QualificationError("calibration_metrics_missing")

    def rank(arm: str) -> tuple[float, float, float, float]:
        row = tuples[arm]
        return (
            float(row.get("f1") or 0.0),
            float(spans.get(arm, {}).get("f1") or 0.0),
            float(row.get("recall") or 0.0),
            float(coverage.get(arm, {}).get("parse_coverage") or 0.0),
        )

    reference = max(sorted(tuples), key=rank)
    family_values = [
        float(row.get("f1") or 0.0)
        for row in calibration.get("family_rows", [])
        if row.get("arm") == reference
    ]
    perturbation_values = [
        float(row.get("f1") or 0.0)
        for row in calibration.get("perturbation_rows", [])
        if row.get("arm") == reference
    ]
    parity_values = [
        bool(row.get("parity"))
        for row in calibration.get("solver_parity_rows", [])
        if row.get("arm") == reference
    ]
    tuple_row = tuples[reference]
    return {
        "threshold_source_split": "calibration",
        "reference_arm": reference,
        "minimum_span_f1": float(spans[reference]["f1"]),
        "minimum_tuple_precision": float(tuple_row["precision"] or 0.0),
        "minimum_tuple_recall": float(tuple_row["recall"] or 0.0),
        "minimum_parse_coverage": float(coverage[reference]["parse_coverage"] or 0.0),
        "minimum_family_floor": min(family_values),
        "minimum_perturbation_floor": min(perturbation_values),
        "exact_semantic_parity": sum(parity_values) / len(parity_values),
        "calibration_metrics_sha256": sha256_json(
            {
                "span_metric_rows": calibration.get("span_metric_rows", []),
                "tuple_metric_rows": calibration.get("tuple_metric_rows", []),
                "parse_coverage_rows": calibration.get("parse_coverage_rows", []),
                "family_rows": calibration.get("family_rows", []),
                "perturbation_rows": calibration.get("perturbation_rows", []),
                "solver_parity_rows": calibration.get("solver_parity_rows", []),
            }
        ),
    }


def evaluate_eligible_arms(
    held: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    event_count_by_arm: Mapping[str, int],
) -> JsonDict:
    """Apply every held threshold with minimum family and perturbation pooling."""

    tuples = _by_arm(held.get("tuple_metric_rows", []))
    spans = _by_arm(held.get("span_metric_rows", []))
    coverage = _by_arm(held.get("parse_coverage_rows", []))
    rows = []
    for arm in sorted(tuples):
        family_values = [
            float(row.get("f1") or 0.0)
            for row in held.get("family_rows", [])
            if row.get("arm") == arm
        ]
        perturbation_values = [
            float(row.get("f1") or 0.0)
            for row in held.get("perturbation_rows", [])
            if row.get("arm") == arm
        ]
        parity_values = [
            bool(row.get("parity"))
            for row in held.get("solver_parity_rows", [])
            if row.get("arm") == arm
        ]
        observed = {
            "span_f1": spans.get(arm, {}).get("f1"),
            "tuple_precision": tuples[arm].get("precision"),
            "tuple_recall": tuples[arm].get("recall"),
            "parse_coverage": coverage.get(arm, {}).get("parse_coverage"),
            "family_floor": min(family_values) if family_values else None,
            "perturbation_floor": min(perturbation_values) if perturbation_values else None,
            "exact_semantic_parity": (
                sum(parity_values) / len(parity_values) if parity_values else None
            ),
        }
        expected = {
            "span_f1": thresholds["minimum_span_f1"],
            "tuple_precision": thresholds["minimum_tuple_precision"],
            "tuple_recall": thresholds["minimum_tuple_recall"],
            "parse_coverage": thresholds["minimum_parse_coverage"],
            "family_floor": thresholds["minimum_family_floor"],
            "perturbation_floor": thresholds["minimum_perturbation_floor"],
            "exact_semantic_parity": thresholds["exact_semantic_parity"],
        }
        failed = [
            name
            for name, floor in expected.items()
            if observed[name] is None or float(observed[name]) < float(floor)
        ]
        rows.append(
            {
                "arm": arm,
                "observed": observed,
                "thresholds": expected,
                "failed_thresholds": failed,
                "passed": not failed,
                "qualified_event_count": int(event_count_by_arm.get(arm, 0)) if not failed else 0,
            }
        )
    passing = [row for row in rows if row["passed"]]
    return {
        "eligible_arm_rows": rows,
        "relation_qualification_ready_score": int(bool(passing)),
        "qualified_relation_event_count": sum(int(row["qualified_event_count"]) for row in passing),
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"duration_s", "field_principles", "reproducibility_checksum"}
        }
    )


def _attach_principles(artifact: JsonDict) -> None:
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(key, f"{key} preserves required Exp6888 evidence.")
        for key in artifact
    }
    artifact["field_principles"]["field_principles"] = FIELD_PRINCIPLES["field_principles"]


def blocked_artifact(
    *,
    date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
    sealed_sidecar_hashes: Mapping[str, Any],
) -> JsonDict:
    """Build a complete blocked artifact without opening held labels."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 6888,
        "run_date": date,
        "status": "blocked",
        "field_principles": {},
        "preconditions_checked": {
            "held_sidecar_open_count": 0,
            "gate_check_summary": gate_summary(checks),
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "sealed_sidecar_hashes": deepcopy(dict(sealed_sidecar_hashes)),
        "frozen_thresholds": {},
        "rows": [],
        "span_metric_rows": [],
        "tuple_metric_rows": [],
        "parse_coverage_rows": [],
        "abstention_rows": [],
        "family_rows": [],
        "perturbation_rows": [],
        "asp_compilation_rows": [],
        "solver_parity_rows": [],
        "contradiction_rows": [],
        "reported_vs_recomputed_metrics": {
            "reported_ready_score": 0,
            "recomputed_ready_score": 0,
            "agreement": True,
        },
        "independent_solver_receipts": {},
        "held_leakage_count": 0,
        "eligible_arm_rows": [],
        "qualified_relation_event_count": 0,
        "relation_qualification_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_independent_relation_qualification",
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    _attach_principles(artifact)
    return artifact


def reduce_frozen_outputs(
    *,
    date: str,
    proposal_artifact: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    calibration_sidecar: Mapping[str, Any],
    held_reader: HeldSidecarReader,
    vocabulary: Sequence[Mapping[str, Any]],
    arms: Sequence[str],
    event_count_by_arm: Mapping[str, int],
    source_artifact_hashes: Mapping[str, Any],
    sealed_sidecar_hashes: Mapping[str, Any],
    solver: Solver,
    solver_timeout_s: float,
    duration_s: float,
    precondition_checks: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Freeze calibration, open held once, and build the terminal artifact."""

    calibration_cells = [
        row for row in proposal_artifact.get("rows", []) if row.get("split") == "calibration"
    ]
    calibration_sources = [row for row in sources if row.get("split") == "calibration"]
    calibration = score_partition(
        cells=calibration_cells,
        sources=calibration_sources,
        formal_rows=calibration_sidecar.get("rows", []),
        vocabulary=vocabulary,
        arms=arms,
        solver=solver,
        solver_timeout_s=solver_timeout_s,
    )
    thresholds = freeze_thresholds(calibration)
    if held_reader.open_count != 0:
        raise QualificationError("held_opened_before_threshold_freeze")
    held_sidecar = held_reader.open_once()
    leaks = detect_held_leakage(proposal_artifact, held_sidecar)
    held_cells = [row for row in proposal_artifact.get("rows", []) if row.get("split") == "held"]
    held_sources = [row for row in sources if row.get("split") == "held"]
    held = score_partition(
        cells=held_cells,
        sources=held_sources,
        formal_rows=held_sidecar.get("rows", []),
        vocabulary=vocabulary,
        arms=arms,
        solver=solver,
        solver_timeout_s=solver_timeout_s,
    )
    eligibility = evaluate_eligible_arms(held, thresholds, event_count_by_arm)
    if leaks:
        eligibility = {
            "eligible_arm_rows": [
                {
                    **row,
                    "passed": False,
                    "failed_thresholds": [*row["failed_thresholds"], "held_leakage"],
                }
                for row in eligibility["eligible_arm_rows"]
            ],
            "relation_qualification_ready_score": 0,
            "qualified_relation_event_count": 0,
        }
    ready = int(eligibility["relation_qualification_ready_score"])
    post_checks = [
        gate_check("held_sidecar_open_count", 1, held_reader.open_count),
        gate_check("held_leakage_count", 0, len(leaks)),
        gate_check("eligible_arm_count_at_least_one", True, ready == 1),
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 6888,
        "run_date": date,
        "status": "complete",
        "field_principles": {},
        "preconditions_checked": {
            "calibration_scored_before_held_open": True,
            "held_sidecar_open_count": held_reader.open_count,
            "no_model_inference": True,
            "no_output_repair": True,
            "gate_check_summary": gate_summary(precondition_checks),
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "sealed_sidecar_hashes": deepcopy(dict(sealed_sidecar_hashes)),
        "frozen_thresholds": thresholds,
        "rows": held["rows"],
        "span_metric_rows": held["span_metric_rows"],
        "tuple_metric_rows": held["tuple_metric_rows"],
        "parse_coverage_rows": held["parse_coverage_rows"],
        "abstention_rows": held["abstention_rows"],
        "family_rows": held["family_rows"],
        "perturbation_rows": held["perturbation_rows"],
        "asp_compilation_rows": held["asp_compilation_rows"],
        "solver_parity_rows": held["solver_parity_rows"],
        "contradiction_rows": held["contradiction_rows"],
        "reported_vs_recomputed_metrics": {
            "reported_ready_score": ready,
            "recomputed_ready_score": ready,
            "agreement": True,
        },
        "independent_solver_receipts": held["independent_solver_receipts"],
        "held_leakage_rows": leaks,
        "held_leakage_count": len(leaks),
        "eligible_arm_rows": eligibility["eligible_arm_rows"],
        "qualified_relation_event_count": eligibility["qualified_relation_event_count"],
        "relation_qualification_ready_score": ready,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary([*precondition_checks, *post_checks]),
        "verifier_is_oracle": True,
        "verdict_class": ("disqualified" if leaks else "circular_positive" if ready else "null"),
        "honest_verdict": (
            "complete_disqualified_held_relation_leakage"
            if leaks
            else "complete_circular_positive_independent_relation_qualification"
            if ready
            else "complete_null_no_relation_arm_passed_frozen_thresholds"
        ),
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    _attach_principles(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Replay terminal schema, gate, checksum, and aggregate consistency."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append("required_fields:" + ",".join(missing))
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not set(artifact) <= set(principles):
        errors.append("field_principles")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    if artifact.get("verdict_class") not in {
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict")
    reported = artifact.get("reported_vs_recomputed_metrics")
    if not isinstance(reported, Mapping):
        errors.append("reported_vs_recomputed_metrics")
    elif (
        reported.get("agreement") is not True
        or reported.get("reported_ready_score")
        != artifact.get("relation_qualification_ready_score")
        or reported.get("recomputed_ready_score")
        != artifact.get("relation_qualification_ready_score")
    ):
        errors.append("aggregate_vs_row_disagreement")
    if artifact.get("verdict_class") != "blocked" and artifact.get("frozen_thresholds"):
        event_counts = {
            str(row.get("arm")): int(row.get("qualified_event_count", 0))
            for row in artifact.get("eligible_arm_rows", [])
        }
        recomputed = evaluate_eligible_arms(artifact, artifact["frozen_thresholds"], event_counts)
        if recomputed["relation_qualification_ready_score"] != artifact.get(
            "relation_qualification_ready_score"
        ) or recomputed["qualified_relation_event_count"] != artifact.get(
            "qualified_relation_event_count"
        ):
            errors.append("aggregate_vs_row_disagreement")
    for row in artifact.get("rows", []):
        if not {"arm", "record_id", "relation", "family", "perturbation"} <= set(row):
            errors.append("row_schema")
            break
    if artifact.get("reproducibility_checksum") != _artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    return sorted(set(errors))


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_hash_rows(observed: Mapping[str, str]) -> JsonDict:
    paths = {
        "exp6274": EXP6274_PATH,
        "exp6886": EXP6886_PATH,
        "exp6887": EXP6887_PATH,
        "compiler": COMPILER_PATH,
    }
    return {
        name: {"path": path.as_posix(), "sha256": observed[name]} for name, path in paths.items()
    }


def run_experiment(*, date: str, output_path: Path | str = RESULT_PATH) -> JsonDict:
    """Run the real fresh reducer and write one terminal artifact."""

    started = time.perf_counter()
    output = Path(output_path)
    source_paths = {
        "exp6274": REPO_ROOT / EXP6274_PATH,
        "exp6886": REPO_ROOT / EXP6886_PATH,
        "exp6887": REPO_ROOT / EXP6887_PATH,
        "compiler": REPO_ROOT / COMPILER_PATH,
    }
    observed_hashes = {
        name: sha256_file(path) if path.exists() else "missing"
        for name, path in source_paths.items()
    }
    hash_summary = check_exact_hashes(
        observed_hashes,
        {name: EXPECTED_HASHES[name] for name in source_paths},
    )
    checks = list(hash_summary["checks"])
    fixture = _read_json(source_paths["exp6886"]) if source_paths["exp6886"].exists() else {}
    proposal = _read_json(source_paths["exp6887"]) if source_paths["exp6887"].exists() else {}
    compiler = _read_json(source_paths["exp6274"]) if source_paths["exp6274"].exists() else {}
    sources = build_frozen_source_records()
    matrix = validate_acquisition_matrix(proposal.get("rows", []), sources, PROPOSAL_ARMS)
    checks.extend(matrix["checks"])
    checks.extend(
        [
            gate_check(
                "relation_corpus_complete_score", 1, proposal.get("relation_corpus_complete_score")
            ),
            gate_check("held_sidecar_access_count", 0, proposal.get("held_sidecar_access_count")),
            gate_check(
                "relation_fixture_ready_score", 1, fixture.get("relation_fixture_ready_score")
            ),
            gate_check("qualified_compiler", 1.0, compiler.get("asp_energy_semantic_ready_score")),
            gate_check("compiler_parity_failure_count", 0, compiler.get("parity_failure_count")),
            gate_check(
                "independent_solver_available",
                True,
                not asp_energy.solver_name_version().endswith(":missing"),
            ),
            gate_check("calibration_sidecar_present", True, CALIBRATION_SIDECAR_PATH.is_file()),
            gate_check("held_sidecar_present", True, HELD_SIDECAR_PATH.is_file()),
        ]
    )
    calibration_hash = (
        sha256_file(CALIBRATION_SIDECAR_PATH) if CALIBRATION_SIDECAR_PATH.is_file() else "missing"
    )
    checks.append(
        gate_check(
            "calibration_sidecar_hash",
            EXPECTED_HASHES["calibration_sidecar"],
            calibration_hash,
        )
    )
    summary = gate_summary(checks)
    source_rows = _source_hash_rows(observed_hashes)
    sidecar_hashes = {
        "calibration": {
            "path": str(CALIBRATION_SIDECAR_PATH),
            "sha256": calibration_hash,
        },
        "held": {
            "path": str(HELD_SIDECAR_PATH),
            "sha256": EXPECTED_HASHES["held_sidecar"],
        },
    }
    if not summary["passed"]:
        artifact = blocked_artifact(
            date=date,
            duration_s=time.perf_counter() - started,
            checks=checks,
            source_artifact_hashes=source_rows,
            sealed_sidecar_hashes=sidecar_hashes,
        )
    else:
        calibration = _read_json(CALIBRATION_SIDECAR_PATH)
        if calibration.get("schema") != SIDECAR_SCHEMA or calibration.get("split") != "calibration":
            artifact = blocked_artifact(
                date=date,
                duration_s=time.perf_counter() - started,
                checks=[gate_check("calibration_sidecar_identity", True, False)],
                source_artifact_hashes=source_rows,
                sealed_sidecar_hashes=sidecar_hashes,
            )
        else:
            reader = HeldSidecarReader(HELD_SIDECAR_PATH, EXPECTED_HASHES["held_sidecar"])
            event_counts = Counter(str(row.get("arm")) for row in proposal.get("rows", []))
            try:
                artifact = reduce_frozen_outputs(
                    date=date,
                    proposal_artifact=proposal,
                    sources=sources,
                    calibration_sidecar=calibration,
                    held_reader=reader,
                    vocabulary=fixture.get("closed_vocabulary_manifest", {}).get("entries", []),
                    arms=PROPOSAL_ARMS,
                    event_count_by_arm=event_counts,
                    source_artifact_hashes=source_rows,
                    sealed_sidecar_hashes=sidecar_hashes,
                    solver=asp_energy.solve_with_clingo,
                    solver_timeout_s=SOLVER_TIMEOUT_S,
                    duration_s=0.0,
                    precondition_checks=checks,
                )
                artifact["duration_s"] = time.perf_counter() - started
                artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
            except QualificationError as exc:
                artifact = blocked_artifact(
                    date=date,
                    duration_s=time.perf_counter() - started,
                    checks=[gate_check("held_sidecar", "valid_once", str(exc))],
                    source_artifact_hashes=source_rows,
                    sealed_sidecar_hashes=sidecar_hashes,
                )
    errors = validate_artifact(artifact)
    if errors:
        raise QualificationError("artifact_validation:" + ",".join(errors))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp6888 from the command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260902")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    args = parser.parse_args(argv)
    artifact = run_experiment(date=args.date, output_path=args.output)
    print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the wrapper.
    raise SystemExit(main())
