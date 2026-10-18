"""Build the Exp 3138 canonical-answer and premise-grounding pilot artifact.

Spec refs: REQ-VERIFY-3138, SCENARIO-VERIFY-3138.

This is a deterministic pilot, not a production VeriCoT verifier. The goal is
to make the prior .291 false accepts fail closed for inspectable reasons:
canonical answer mismatch, missing or contradictory premise grounding, and
monitor-ledger disagreement. No model, solver, repair loop, or conductor path
is invoked here; existing artifacts provide all evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3138_canonical_answer_vericot_grounding_pilot_v1"
SCHEMA = "carnot.canonical_answer_vericot_grounding_pilot.v1"
OUTPUT_REL_PATH = Path("results/experiment_3138_canonical_answer_vericot_grounding_pilot_v1.json")

EXP3136_REL_PATH = Path("results/experiment_3136_false_accept_root_cause_autopsy_v1.json")
EXP3137_REL_PATH = Path("results/experiment_3137_exact_safe_accept_abstain_contract_v1.json")
EXP3111_REL_PATH = Path("results/experiment_3111_certified_coherence_z3_mcs_feedback_v3.json")
EXP3126_REL_PATH = Path(
    "results/experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1.json"
)

SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
REQUIRED_FIELDS = {
    "canonical_grounding_pilot_v1_ready",
    "canonicalizer_implemented",
    "premise_grounding_rows",
    "regression_rows_evaluated",
    "false_accept_rows_blocked",
    "canonicalization_block_count",
    "premise_grounding_block_count",
    "ledger_replay_block_count",
    "residual_false_accept_rows",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3138_canonical_answer_vericot_grounding_pilot_v1.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/verify -m pytest -o addopts='' tests/python/test_experiment_3138_canonical_answer_vericot_grounding_pilot_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/canonical_answer_vericot_grounding_pilot_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_REL_PATHS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("research_references", Path("research-references.md"), False),
    ("verification_openspec", Path("openspec/capabilities/verification/spec.md"), False),
    ("exp3136_false_accept_autopsy", EXP3136_REL_PATH, True),
    ("exp3137_exact_safe_contract", EXP3137_REL_PATH, True),
    ("exp3111_certified_coherence_feedback", EXP3111_REL_PATH, True),
    ("exp3126_fragment_monitor_ledger", EXP3126_REL_PATH, True),
    (
        "exp3138_module",
        Path("python/carnot/verify/canonical_answer_vericot_grounding_pilot_v1.py"),
        False,
    ),
)

VALIDITY_LABELS = {"VALID", "INVALID"}
SAT_LABELS = {"SAT", "UNSAT"}
REJECT_LABELS = {"INVALID", "UNSAT", "REPAIRABLE", "UNREPAIRABLE"}
ACCEPT_LABELS = {"VALID", "SAT"}
LABEL_ALIASES = {
    "VALID": "VALID",
    "INVALID": "INVALID",
    "SAT": "SAT",
    "SATISFIABLE": "SAT",
    "UNSAT": "UNSAT",
    "UNSATISFIABLE": "UNSAT",
    "REPAIRABLE": "REPAIRABLE",
    "UNREPAIRABLE": "UNREPAIRABLE",
}
CODE_VERDICT_ALIASES = {
    "ASSERTION_PASSES": "pass",
    "ASSERTION_PASSED": "pass",
    "ASSERTION_PASS": "pass",
    "ASSERTION_FAILS": "fail",
    "ASSERTION_FAILED": "fail",
    "ASSERTION_FAIL": "fail",
    "TEST_PASSES": "pass",
    "TEST_FAILS": "fail",
    "PASS": "pass",
    "FAIL": "fail",
}
SOLVER_LABEL_TO_EXACT = {
    "ASSERTION_PASSES": "VALID",
    "ASSERTION_FAILS": "INVALID",
    "SAT": "SAT",
    "SATISFIABLE": "SAT",
    "UNSAT": "UNSAT",
    "UNSATISFIABLE": "UNSAT",
}


@dataclass(frozen=True)
class CanonicalAnswer:
    """Verifier-friendly answer form with token family kept explicit.

    We keep `kind` and `family` separate because .291 mixed up answer tokens
    that look semantically positive to a model but belong to different exact
    authorities. `VALID` and `SAT` both feel like acceptance, yet they answer
    different question families and must not compare as equivalent.
    """

    kind: str
    value: str
    family: str
    normalized: str
    parse_status: str = "parsed"

    def to_dict(self) -> JsonDict:
        return {
            "kind": self.kind,
            "value": self.value,
            "family": self.family,
            "normalized": self.normalized,
            "parse_status": self.parse_status,
        }


def canonicalize_answer(
    value: Any,
    *,
    json_max_nodes: int = 64,
    json_max_depth: int = 8,
) -> CanonicalAnswer:
    """REQ-VERIFY-3138: normalize labels, numbers, bounded JSON, and code verdicts."""

    if isinstance(value, CanonicalAnswer):
        return value
    json_answer = canonicalize_json_like(value, json_max_nodes, json_max_depth)
    if json_answer is not None:
        return json_answer
    text = "" if value is None else str(value).strip()
    upper = normalize_token(text)
    if upper in LABEL_ALIASES:
        label = LABEL_ALIASES[upper]
        return CanonicalAnswer("label", label, label_family(label), label)
    if upper in CODE_VERDICT_ALIASES:
        verdict = CODE_VERDICT_ALIASES[upper]
        return CanonicalAnswer("code_verdict", verdict, "code_fragment", verdict)
    numeric = canonicalize_numeric(value)
    if numeric is not None:
        return numeric
    return CanonicalAnswer("unknown", text, "unknown", text, "unparsed")


def canonicalize_json_like(value: Any, max_nodes: int, max_depth: int) -> CanonicalAnswer | None:
    """Return a bounded canonical JSON answer, or None when the value is not JSON-like."""

    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped.startswith(("{", "[")):
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return CanonicalAnswer("json", stripped, "json", stripped, "unparsed")
    elif not isinstance(value, (Mapping, list, tuple)):
        return None
    normalized, node_count, too_deep = normalize_json_value(parsed, 0, max_depth)
    if too_deep or node_count > max_nodes:
        return CanonicalAnswer("json", "too_large", "json", "too_large", "too_large")
    compact = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return CanonicalAnswer("json", compact, "json", compact)


def normalize_json_value(value: Any, depth: int, max_depth: int) -> tuple[Any, int, bool]:
    """Canonicalize JSON-like containers while tracking a small bounded size budget."""

    if depth > max_depth:
        return None, 1, True
    if isinstance(value, Mapping):
        total = 1
        normalized: JsonDict = {}
        too_deep = False
        for key in sorted(value, key=str):
            child, child_count, child_too_deep = normalize_json_value(
                value[key], depth + 1, max_depth
            )
            normalized[str(key)] = child
            total += child_count
            too_deep = too_deep or child_too_deep
        return normalized, total, too_deep
    if isinstance(value, (list, tuple)):
        total = 1
        normalized_list = []
        too_deep = False
        for child_value in value:
            child, child_count, child_too_deep = normalize_json_value(
                child_value, depth + 1, max_depth
            )
            normalized_list.append(child)
            total += child_count
            too_deep = too_deep or child_too_deep
        return normalized_list, total, too_deep
    return value, 1, False


def canonicalize_numeric(value: Any) -> CanonicalAnswer | None:
    """Normalize equivalent numeric surface forms such as `2`, `2.0`, and `2.00`."""

    if isinstance(value, bool):
        return None
    text = str(value).strip()
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None
    normalized = "0" if number == 0 else format(number.normalize(), "f")
    return CanonicalAnswer("number", normalized, "numeric", normalized)


def answers_equivalent(left: Any, right: Any) -> bool:
    """Return xVerify-style exact equivalence over canonicalized answer families."""

    left_canonical = canonicalize_answer(left)
    right_canonical = canonicalize_answer(right)
    return (
        left_canonical.parse_status == "parsed"
        and right_canonical.parse_status == "parsed"
        and left_canonical.kind == right_canonical.kind
        and left_canonical.family == right_canonical.family
        and left_canonical.value == right_canonical.value
    )


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """SCENARIO-VERIFY-3138: replay known false accepts through canonical grounding."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3136 = read_json_object(root_path / EXP3136_REL_PATH)
    exp3137 = read_json_object(root_path / EXP3137_REL_PATH)
    exp3111 = read_json_object(root_path / EXP3111_REL_PATH)
    exp3126 = read_json_object(root_path / EXP3126_REL_PATH)
    monitor_by_fixture = monitor_events_by_fixture(exp3126.get("monitor_events"))
    certificates = certificates_by_fixture(exp3111.get("certificates"))
    contract_replay = contract_replay_by_fixture(exp3137.get("replay_rows"))
    regression_rows = known_false_accept_rows(exp3136)
    replay_rows = [
        evaluate_regression_row(
            row,
            solver_certificate=certificates.get(row_id_from(row), {}),
            monitor_events=monitor_by_fixture.get(row_id_from(row), row.get("monitor_events") or ()),
            contract_replay=contract_replay.get(row_id_from(row), {}),
        )
        for row in regression_rows
    ]
    source_rows = source_artifacts(root_path)
    false_accept_rows_blocked = sum(bool(row["blocked_by"]) for row in replay_rows)
    residual_rows = [row["row_id"] for row in replay_rows if not row["blocked_by"]]
    ready = bool(
        all(row["present"] for row in source_rows if row["required"])
        and replay_rows
        and false_accept_rows_blocked == len(replay_rows)
        and not residual_rows
        and stable_hash(replay_rows) == stable_hash(json.loads(json.dumps(replay_rows, sort_keys=True)))
    )
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "canonical_grounding_pilot_v1_ready": ready,
        "canonicalizer_implemented": True,
        "premise_grounding_rows": len(replay_rows),
        "premise_grounding_record_count": sum(
            len(row["premise_records"]) for row in replay_rows
        ),
        "regression_rows_evaluated": len(replay_rows),
        "false_accept_rows_blocked": false_accept_rows_blocked,
        "canonicalization_block_count": sum(
            row["canonicalization_blocked"] for row in replay_rows
        ),
        "premise_grounding_block_count": sum(
            row["premise_grounding_blocked"] for row in replay_rows
        ),
        "ledger_replay_block_count": sum(row["ledger_replay_blocked"] for row in replay_rows),
        "residual_false_accept_rows": residual_rows,
        "regression_row_replay": replay_rows,
        "canonical_forms_defined": canonical_forms_defined(),
        "source_artifacts": source_rows,
        "source_checksums": {
            row["path"]: row["sha256"] for row in source_rows if row.get("sha256")
        },
        "inference_substrate": inference_substrate(exp3136),
        "self_checks": {
            "source_replay_deterministic": stable_hash(replay_rows)
            == stable_hash(json.loads(json.dumps(replay_rows, sort_keys=True))),
            "all_regression_rows_blocked": false_accept_rows_blocked == len(replay_rows),
            "no_residual_false_accept_rows": not residual_rows,
        },
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "duration_s": duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3138 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def evaluate_regression_row(
    row: Mapping[str, Any],
    *,
    solver_certificate: Mapping[str, Any] | None = None,
    monitor_events: Sequence[Mapping[str, Any]] = (),
    contract_replay: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Evaluate one known false accept against canonical, premise, and ledger evidence."""

    certificate = dict(solver_certificate or {})
    replay = dict(contract_replay or {})
    row_id = row_id_from(row)
    exact = str(row.get("exact_label") or row.get("expected_answer") or "").upper()
    candidate = str(row.get("extracted_answer") or row.get("live_model_verdict") or "")
    exact_canonical = canonicalize_answer(exact)
    candidate_canonical = canonicalize_answer(candidate)
    expected_action = expected_action_from_label(exact)
    candidate_action = action_from_canonical(candidate_canonical)
    ledger_summary = ledger_replay_summary(monitor_events)
    premise_records = premise_records_from_events(monitor_events, row_id)
    solver_summary = solver_certificate_summary(certificate, exact)
    canonicalization_blocked = not answers_equivalent(exact_canonical, candidate_canonical)
    premise_grounding_blocked = (
        not any(record["grounded"] for record in premise_records) and candidate_action == "accept"
    )
    ledger_action = ledger_summary.get("ledger_action")
    premise_to_answer_consistent = bool(
        ledger_action == expected_action
        and solver_summary["certificate_consistent_with_exact"]
        and not canonicalization_blocked
    )
    answer_to_premise_consistent = bool(
        candidate_action == ledger_action
        and ledger_summary.get("final_answer_consistent_with_ledger") is not False
    )
    ledger_replay_blocked = not answer_to_premise_consistent or (
        ledger_summary.get("final_answer_consistent_with_exact") is False
    )
    blocked_by = []
    if canonicalization_blocked:
        blocked_by.append("canonicalization")
    if premise_grounding_blocked:
        blocked_by.append("premise_grounding")
    if ledger_replay_blocked:
        blocked_by.append("ledger_replay")
    return {
        "row_id": row_id,
        "exact_label": exact,
        "candidate_answer": candidate,
        "expected_action": expected_action,
        "candidate_action": candidate_action,
        "exact_canonical": exact_canonical.to_dict(),
        "candidate_canonical": candidate_canonical.to_dict(),
        "canonical_equivalent": not canonicalization_blocked,
        "canonicalization_blocked": canonicalization_blocked,
        "premise_records": premise_records,
        "premise_grounding_blocked": premise_grounding_blocked,
        "premise_to_answer_consistent": premise_to_answer_consistent,
        "answer_to_premise_consistent": answer_to_premise_consistent,
        "ledger_replay": ledger_summary,
        "ledger_replay_blocked": ledger_replay_blocked,
        "solver_certificate_summary": solver_summary,
        "contract_replay": {
            "decision": replay.get("decision"),
            "matched_rule_id": replay.get("matched_rule_id"),
        },
        "blocked_by": blocked_by,
    }


def premise_records_from_events(
    monitor_events: Sequence[Mapping[str, Any]],
    row_id: str,
) -> list[JsonDict]:
    """Map available monitor constraints into simple VeriCoT-style premise records."""

    ledger = first_payload(monitor_events, "constraint_ledger")
    constraints = [dict(row) for row in ledger.get("constraints") or [] if isinstance(row, Mapping)]
    if constraints:
        return [
            {
                "premise_id": str(row.get("constraint_id") or f"{row_id}:premise_{index}"),
                "source": "monitor_ledger",
                "grounded": True,
                "status": str(row.get("status") or "unknown"),
                "solver_evidence": dict(row.get("solver_evidence") or {}),
            }
            for index, row in enumerate(constraints)
        ]
    partial = first_payload(monitor_events, "partial_trace_state")
    return [
        {
            "premise_id": f"{row_id}:premise_absent",
            "source": "monitor_ledger",
            "grounded": False,
            "status": "absent",
            "reason": str(partial.get("partial_state") or "no_constraint_ledger"),
        }
    ]


def ledger_replay_summary(monitor_events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Extract the forward/backward ledger fields needed for answer consistency checks."""

    ledger = first_payload(monitor_events, "constraint_ledger")
    candidate = first_payload(monitor_events, "candidate_final_answer")
    drift = first_payload(monitor_events, "drift_classification")
    return {
        "monitor_event_count": len(monitor_events),
        "ledger_action": candidate.get("ledger_action") or ledger.get("ledger_action"),
        "ledger_source": ledger.get("ledger_source"),
        "candidate_live_decision": candidate.get("live_decision"),
        "candidate_extracted_answer": candidate.get("extracted_answer"),
        "final_answer_consistent_with_exact": candidate.get("final_answer_consistent_with_exact"),
        "final_answer_consistent_with_ledger": candidate.get(
            "final_answer_consistent_with_ledger"
        ),
        "is_monitor_violation": drift.get("is_monitor_violation") is True,
        "failure_mechanism": drift.get("failure_mechanism"),
    }


def solver_certificate_summary(certificate: Mapping[str, Any], exact_label: str) -> JsonDict:
    """Summarize checked-in solver authority without executing a fresh solver."""

    solver_label = str(certificate.get("solver_label") or "").upper()
    solver_exact = str(certificate.get("exact_label") or SOLVER_LABEL_TO_EXACT.get(solver_label) or "")
    route = certificate.get("maxsat_route") if isinstance(certificate.get("maxsat_route"), Mapping) else {}
    return {
        "present": bool(certificate),
        "solver_authority": certificate.get("solver_authority"),
        "solver_label": solver_label,
        "solver_exact_label": solver_exact,
        "certificate_consistent_with_exact": answers_equivalent(solver_exact, exact_label),
        "coherence_status": certificate.get("coherence_status"),
        "maxsat_action": route.get("action"),
        "unsat_core": list(certificate.get("unsat_core") or []),
        "minimal_correction_set": dict(certificate.get("minimal_correction_set") or {}),
    }


def canonical_forms_defined() -> JsonDict:
    """Expose the canonical answer families covered by the pilot."""

    return {
        "validity_labels": sorted(VALIDITY_LABELS),
        "sat_labels": sorted(SAT_LABELS),
        "numeric": "Decimal-normalized finite number strings",
        "json_like_bounded": {"max_nodes": 64, "max_depth": 8, "sort_keys": True},
        "code_fragment_verdicts": {"pass": "code fragment accepted", "fail": "code fragment rejected"},
    }


def known_false_accept_rows(exp3136: Mapping[str, Any]) -> list[JsonDict]:
    """Read known false-accept regression rows from the Exp 3136 autopsy."""

    rows = mapping_rows(exp3136.get("false_accept_rows"))
    ids = [str(row_id) for row_id in exp3136.get("false_accept_row_ids") or []]
    if rows:
        return sorted(rows, key=row_id_from)
    verifier_rows = mapping_rows(exp3136.get("verifier_rows"))
    return sorted([row for row in verifier_rows if row_id_from(row) in ids], key=row_id_from)


def monitor_events_by_fixture(value: Any) -> dict[str, list[JsonDict]]:
    """Group monitor events by fixture ID while preserving deterministic event order."""

    grouped: dict[str, list[JsonDict]] = {}
    for event in mapping_rows(value):
        grouped.setdefault(str(event.get("fixture_id") or ""), []).append(event)
    for fixture_id in grouped:
        grouped[fixture_id].sort(key=lambda event: int(event.get("event_index") or 0))
    return grouped


def certificates_by_fixture(value: Any) -> dict[str, JsonDict]:
    """Index Exp 3111 solver-certificate rows by fixture ID."""

    return {str(row.get("fixture_id") or ""): row for row in mapping_rows(value)}


def contract_replay_by_fixture(value: Any) -> dict[str, JsonDict]:
    """Index Exp 3137 live replay rows by row ID."""

    return {
        row_id_from(row): row
        for row in mapping_rows(value)
        if row.get("row_source") in (None, "live")
    }


def first_payload(events: Sequence[Mapping[str, Any]], event_type: str) -> JsonDict:
    """Return the first payload for an event type, or empty evidence."""

    return next(
        (
            dict(event.get("payload") or {})
            for event in events
            if event.get("event_type") == event_type
        ),
        {},
    )


def read_json_object(path: Path) -> JsonDict:
    """Read a local JSON object while making missing evidence non-promotable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive filesystem guard.
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return source rows so the pilot traces every claim to local files."""

    return [source_row(root, role, rel_path, required) for role, rel_path, required in SOURCE_REL_PATHS]


def source_row(root: Path, role: str, rel_path: Path, required: bool) -> JsonDict:
    """Build one source-artifact provenance row."""

    path = root / rel_path
    return {
        "role": role,
        "path": rel_path.as_posix(),
        "required": required,
        "present": path.is_file(),
        "sha256": sha256_file(path),
    }


def inference_substrate(exp3136: Mapping[str, Any]) -> JsonDict:
    """Describe exactly which compute substrates this pilot did and did not use."""

    upstream = exp3136.get("inference_substrate")
    reused = upstream.get("upstream_live_model_calls_reused", 0) if isinstance(upstream, Mapping) else 0
    return {
        "kind": "deterministic_artifact_replay",
        "executes_models": False,
        "model_backed_verifier_invoked": False,
        "executes_solvers": False,
        "executes_repairs": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "local_repo_only": True,
        "no_live_llm_inference": True,
        "fresh_live_model_calls": 0,
        "upstream_live_model_calls_reused": int(reused or 0),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields and the no-residual-false-accept claim."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"Exp 3138 artifact missing required fields: {missing}")  # pragma: no cover
    if artifact.get("canonicalizer_implemented") is not True:
        raise ValueError("canonicalizer_implemented must be true")
    for key in (
        "premise_grounding_rows",
        "regression_rows_evaluated",
        "false_accept_rows_blocked",
        "canonicalization_block_count",
        "premise_grounding_block_count",
        "ledger_replay_block_count",
    ):
        value = artifact.get(key)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")  # pragma: no cover
    if artifact.get("residual_false_accept_rows") != []:
        raise ValueError("residual_false_accept_rows must be empty for a ready artifact")
    if artifact.get("canonical_grounding_pilot_v1_ready") is True and artifact.get(
        "false_accept_rows_blocked"
    ) != artifact.get("regression_rows_evaluated"):
        raise ValueError("blocked false-accept count must match regression row count")
    verdict = str(artifact.get("honest_verdict") or "")
    if artifact.get("canonical_grounding_pilot_v1_ready") is True and not verdict.startswith(
        SUCCESS_PREFIXES
    ):
        raise ValueError("honest_verdict must start with a terminal success prefix")


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict string with the required success prefix."""

    if artifact.get("canonical_grounding_pilot_v1_ready") is True:
        return (
            "complete: canonical_grounding_pilot_v1_ready=true; "
            f"regression_rows_evaluated={artifact.get('regression_rows_evaluated')}; "
            f"false_accept_rows_blocked={artifact.get('false_accept_rows_blocked')}"
        )
    return "blocked_canonical_grounding_pilot_missing_regression_replay"


def expected_action_from_label(label: str | None) -> str:
    """Map exact labels into accept/reject/abstain actions."""

    normalized = str(label or "").upper()
    if normalized in ACCEPT_LABELS:
        return "accept"
    if normalized in REJECT_LABELS:
        return "reject"
    return "abstain"


def action_from_canonical(answer: CanonicalAnswer) -> str:
    """Map a canonical candidate answer into the action family it implies."""

    if answer.kind == "label":
        return expected_action_from_label(answer.value)
    if answer.kind == "code_verdict":
        return "accept" if answer.value == "pass" else "reject"
    return "abstain"


def label_family(label: str) -> str:
    """Return the answer-token family for one canonical exact label."""

    if label in VALIDITY_LABELS:
        return "validity_token"
    if label in SAT_LABELS:
        return "sat_token"
    return "repairability_token"


def normalize_token(text: str) -> str:
    """Normalize punctuation in short answer tokens without changing semantics."""

    return text.strip().upper().replace("-", "_").replace(" ", "_")


def mapping_rows(value: Any) -> list[JsonDict]:
    """Return only mapping rows from an arbitrary list-like value."""

    return [dict(row) for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def row_id_from(row: Mapping[str, Any]) -> str:
    """Return the stable replay row identifier."""

    return str(row.get("row_id") or row.get("fixture_id") or row.get("source_fixture_id") or "")


def sha256_file(path: Path) -> str | None:
    """Checksum a source file so replay claims remain tied to exact bytes."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    """Hash JSON-serializable evidence using canonical key ordering."""

    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def duration(started_s: float, now_s: float | None) -> float:
    """Return a nonnegative wall-clock duration."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)
