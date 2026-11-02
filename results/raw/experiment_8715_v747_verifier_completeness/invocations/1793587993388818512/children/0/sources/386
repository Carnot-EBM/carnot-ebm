"""Exp 3201 FR-11 KAN-CL nonforgetting sidecar audit.

Spec refs: REQ-LEARN-3201, SCENARIO-LEARN-3201,
SCENARIO-LEARN-3201-BLOCKED.

This module audits the Exp 3200 trace-memory controller as a diagnostic
sidecar.  It borrows the KAN-CL idea of checking local retention boundaries,
but it never trains a KAN, updates model weights, or promotes the sidecar to
verifier authority.  The only inputs are checked-in exact replay artifacts.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
SCHEMA_VERSION = "1.0"
EXPERIMENT_ID = "experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1"
SCHEMA = "carnot.fr11.kan_cl_nonforgetting_sidecar_audit.v1"
OUTPUT_REL_PATH = Path("results/experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1.json")
EXP3200_REL_PATH = Path("results/experiment_3200_fr11_verify_trace_memory_controller_v1.json")
EXP3187_REL_PATH = Path("results/experiment_3187_fr11_cross_environment_drift_replay_v1.json")
SPEC_REL_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_REL_PATH = Path("python/carnot/eval/fr11_kan_cl_nonforgetting_sidecar_audit_v1.py")
TEST_REL_PATH = Path("tests/python/test_experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1.py")

REQUIRED_ARTIFACT_FIELDS = {
    "schema_version",
    "experiment_id",
    "source_artifacts",
    "audit_metric_schema",
    "heldout_replay_count",
    "drift_replay_count",
    "negative_control_regression_count",
    "locality_violation_count",
    "rollback_triggered",
    "model_weight_update_performed",
    "sidecar_promotion_allowed",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest -o addopts='' "
    "tests/python/test_experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1.py -q",
    ".venv/bin/coverage run -m pytest -o addopts='' "
    "tests/python/test_experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1.py -q",
    ".venv/bin/coverage report "
    "--include='python/carnot/eval/fr11_kan_cl_nonforgetting_sidecar_audit_v1.py' "
    "--fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_no_hidden_learning_claims", Path("CLAUDE.md"), False),
    ("self_learning_openspec", SPEC_REL_PATH, False),
    ("exp3200_trace_memory_controller", EXP3200_REL_PATH, True),
    ("exp3187_cross_environment_drift_replay", EXP3187_REL_PATH, True),
    ("exp3201_module", MODULE_REL_PATH, False),
    ("exp3201_tests", TEST_REL_PATH, False),
)
MUTATION_FLAGS = (
    "executes_live_model_inference",
    "model_weight_learning",
    "model_weight_training",
    "model_weight_mutation",
    "base_model_weights_updated",
    "kan_model_weight_training",
    "hidden_state_mutation_claimed",
)
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed when evidence is unavailable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def load_sources(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load the checked-in trace controller and drift replay artifacts."""

    root_path = Path(root)
    return {
        "exp3200": read_json_object(root_path / EXP3200_REL_PATH),
        "exp3187": read_json_object(root_path / EXP3187_REL_PATH),
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build a schema-complete Exp 3201 sidecar audit artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = load_sources(root_path)
    blocker = source_blocker(sources)
    if blocker:
        artifact = blocked_artifact(root_path, blocker, start, now_s, tests_run)
        validate_artifact(artifact)
        return artifact

    exp3200 = sources["exp3200"]
    exp3187 = sources["exp3187"]
    audit = audit_traces(rows_from_trace_payload(exp3200), exp3187)
    rollback = bool(audit["rollback_reasons"])
    artifact = {
        "artifact": EXPERIMENT_ID,
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "source_artifacts": source_artifacts(root_path),
        "audit_metric_schema": audit_metric_schema(),
        "heldout_replay_count": audit["heldout_replay_count"],
        "drift_replay_count": audit["drift_replay_count"],
        "negative_control_regression_count": audit["negative_control_regression_count"],
        "locality_violation_count": audit["locality_violation_count"],
        "rollback_triggered": rollback,
        "model_weight_update_performed": False,
        "sidecar_promotion_allowed": False,
        "heldout_regression_count": audit["heldout_regression_count"],
        "drift_regression_count": audit["drift_regression_count"],
        "negative_control_replay_count": audit["negative_control_replay_count"],
        "source_drift_case_count": audit["source_drift_case_count"],
        "trace_regressions": audit["trace_regressions"],
        "locality_violations": audit["locality_violations"],
        "routing_bin_summary": audit["routing_bin_summary"],
        "rollback_reasons": audit["rollback_reasons"],
        "source_preconditions": precondition_checks(sources),
        "inference_substrate": inference_substrate(),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(start, now_s),
        "honest_verdict": honest_verdict(audit, rollback),
    }
    validate_artifact(artifact)
    return artifact


def blocked_artifact(
    root: Path,
    blocker: str,
    started_s: float,
    now_s: float | None,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    """Return a terminal fail-closed artifact for unsafe source evidence."""

    return {
        "artifact": EXPERIMENT_ID,
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "source_artifacts": source_artifacts(root),
        "audit_metric_schema": audit_metric_schema(),
        "heldout_replay_count": 0,
        "drift_replay_count": 0,
        "negative_control_regression_count": 0,
        "locality_violation_count": 0,
        "rollback_triggered": True,
        "model_weight_update_performed": False,
        "sidecar_promotion_allowed": False,
        "heldout_regression_count": 0,
        "drift_regression_count": 0,
        "negative_control_replay_count": 0,
        "source_drift_case_count": 0,
        "trace_regressions": [],
        "locality_violations": [],
        "routing_bin_summary": {},
        "rollback_reasons": ["blocked_precondition"],
        "source_preconditions": {"blocked_reason": blocker},
        "inference_substrate": inference_substrate(mode="blocked_precondition_check"),
        "blocked_reason": blocker,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(started_s, now_s),
        "honest_verdict": f"complete: blocked kan-cl sidecar audit; {blocker}",
    }


def source_blocker(sources: Mapping[str, Any]) -> str:
    """REQ-LEARN-3201-1: fail closed when source artifacts are unsafe."""

    exp3200 = sources.get("exp3200", {})
    exp3187 = sources.get("exp3187", {})
    if not isinstance(exp3200, Mapping) or not is_terminal(exp3200):
        return "exp3200_missing_or_not_terminal"
    if exp3200.get("model_weight_update_performed") is True:
        return "exp3200_model_weight_update_claimed"
    if source_claims_live_or_mutation(exp3200):
        return "exp3200_live_inference_or_weight_update_claimed"
    if (
        not isinstance(exp3187, Mapping)
        or exp3187.get("fr11_cross_environment_drift_replay_v1_ready") is not True
    ):
        return "exp3187_missing_or_not_ready"
    if exp3187.get("no_model_weight_update_claimed") is not True:
        return "exp3187_model_weight_update_claimed"
    if not is_terminal(exp3187) or source_claims_live_or_mutation(exp3187):
        return "exp3187_live_inference_or_weight_update_claimed"
    return ""


def is_terminal(payload: Mapping[str, Any]) -> bool:
    """Return whether a source artifact has a terminal verdict string."""

    verdict = str(payload.get("honest_verdict") or "")
    return verdict.startswith(TERMINAL_PREFIXES)


def source_claims_live_or_mutation(payload: Mapping[str, Any]) -> bool:
    """Return whether a source artifact claims live inference or model mutation."""

    substrate = payload.get("inference_substrate", {})
    if not isinstance(substrate, Mapping):
        return True
    return int(substrate.get("fresh_live_inference_calls") or 0) != 0 or any(
        substrate.get(flag) is True for flag in MUTATION_FLAGS
    )


def rows_from_trace_payload(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Return trace row dictionaries from the Exp 3200 artifact."""

    rows = payload.get("trace_records", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def audit_traces(records: Sequence[Mapping[str, Any]], exp3187: Mapping[str, Any]) -> JsonDict:
    """REQ-LEARN-3201-2/3/4/5: replay exact labels and locality bins."""

    rows = [dict(row) for row in records]
    regressions = [case for row in rows if (case := trace_regression(row)) is not None]
    locality = locality_violations(rows)
    negative_trace_regressions = [
        row for row in regressions if row["replay_role"] == "negative_control"
    ]
    source_negative_count = int(exp3187.get("negative_control_regression_count") or 0)
    source_drift_cases = exp3187.get("drift_cases", [])
    source_drift_count = len(source_drift_cases) if isinstance(source_drift_cases, Sequence) else 0
    heldout_regression_count = sum(1 for row in regressions if row["replay_role"] == "heldout")
    drift_regression_count = sum(1 for row in regressions if row["replay_role"] == "drift")
    negative_count = max(source_negative_count, len(negative_trace_regressions))
    rollback_reasons = rollback_reasons_for(
        heldout_regression_count,
        drift_regression_count,
        negative_count,
        source_drift_count,
        locality,
    )
    return {
        "heldout_replay_count": sum(1 for row in rows if row.get("replay_role") == "heldout"),
        "drift_replay_count": sum(1 for row in rows if row.get("replay_role") == "drift"),
        "negative_control_replay_count": sum(
            1 for row in rows if row.get("replay_role") == "negative_control"
        ),
        "heldout_regression_count": heldout_regression_count,
        "drift_regression_count": drift_regression_count,
        "negative_control_regression_count": negative_count,
        "source_drift_case_count": source_drift_count,
        "locality_violation_count": len(locality),
        "trace_regressions": regressions,
        "locality_violations": locality,
        "routing_bin_summary": routing_bin_summary(rows),
        "rollback_reasons": rollback_reasons,
    }


def trace_regression(record: Mapping[str, Any]) -> JsonDict | None:
    """Return a replay regression when exact labels and routing disagree."""

    exact_action = exact_action_for_label(record.get("exact_label"), record.get("expected_action"))
    decision = normalize_action(record.get("answer_abstain_decision"))
    if normalize_action(record.get("consistency_judgment")) != "consistent":
        return regression_case(record, "inconsistent_exact_replay", exact_action)
    if bool(record.get("observed_action_changed")):
        return regression_case(record, "observed_action_changed", exact_action)
    if exact_action == "accept" and decision != "answer":
        return regression_case(record, "exact_accept_not_answered", exact_action)
    if exact_action == "reject" and decision != "abstain":
        return regression_case(record, "exact_reject_not_abstained", exact_action)
    return None


def regression_case(record: Mapping[str, Any], reason: str, exact_action: str) -> JsonDict:
    """Return compact row-level regression evidence for rollback triage."""

    return {
        "row_id": str(record.get("row_id") or ""),
        "trace_id": str(record.get("trace_id") or ""),
        "replay_role": str(record.get("replay_role") or ""),
        "fixture_family": str(record.get("fixture_family") or "unknown"),
        "reason": reason,
        "exact_action": exact_action,
        "exact_label": str(record.get("exact_label") or ""),
        "answer_abstain_decision": normalize_action(record.get("answer_abstain_decision")),
        "routing_bin": route_bin_for(record),
    }


def locality_violations(records: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """REQ-LEARN-3201-4: detect exact-evidence bins whose boundaries move."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        key = str(record.get("historical_exact_evidence_key") or record.get("row_id") or "")
        if key:
            grouped[key].append(record)

    violations: list[JsonDict] = []
    for key, rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        boundaries = {
            "expected_actions": sorted(
                {normalize_action(row.get("expected_action")) for row in rows}
            ),
            "exact_actions": sorted(
                {
                    exact_action_for_label(row.get("exact_label"), row.get("expected_action"))
                    for row in rows
                }
            ),
            "answer_abstain_decisions": sorted(
                {normalize_action(row.get("answer_abstain_decision")) for row in rows}
            ),
            "routing_bins": sorted({route_bin_for(row) for row in rows}),
        }
        changed = {name: values for name, values in boundaries.items() if len(values) > 1}
        if changed:
            violations.append(
                {
                    "historical_exact_evidence_key": key,
                    "row_ids": sorted({str(row.get("row_id") or "") for row in rows}),
                    "affected_roles": sorted({str(row.get("replay_role") or "") for row in rows}),
                    "boundary_deltas": changed,
                }
            )
    return violations


def routing_bin_summary(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count trace rows by normalized route bin."""

    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[route_bin_for(record)] += 1
    return {key: {"count": counts[key]} for key in sorted(counts)}


def rollback_reasons_for(
    heldout_regressions: int,
    drift_regressions: int,
    negative_regressions: int,
    source_drift_cases: int,
    locality: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return stable rollback reasons from the replay audit metrics."""

    reasons: list[str] = []
    if heldout_regressions:
        reasons.append("heldout_regression")
    locality_touches_drift = any("drift" in row.get("affected_roles", []) for row in locality)
    if drift_regressions or source_drift_cases or locality_touches_drift:
        reasons.append("drift_regression")
    if negative_regressions:
        reasons.append("negative_control_regression")
    if locality:
        reasons.append("locality_violation")
    return reasons


def exact_action_for_label(label: Any, expected_action: Any) -> str:
    """Map existing exact labels to the action the controller must preserve."""

    text = str(label or "").strip().upper()
    if text in {"VALID", "EXACT_ACCEPT", "ACCEPT"}:
        return "accept"
    if text in {"INVALID", "EXACT_REJECT", "REJECT"}:
        return "reject"
    return normalize_action(expected_action)


def route_bin_for(record: Mapping[str, Any]) -> str:
    """Normalize trace routing into KAN-CL-style sidecar bins."""

    decision = normalize_action(record.get("answer_abstain_decision"))
    routing = normalize_action(record.get("routing_outcome"))
    if decision == "answer" and routing in {"verify_then_answer", "skip_redundant_recheck"}:
        return "answer_path"
    if decision == "abstain" or routing in {"abstain_or_escalate", "reject", "escalate"}:
        return "abstain_path"
    return routing or decision or "unknown"


def audit_metric_schema() -> JsonDict:
    """Return the interpretable metric schema for the sidecar audit."""

    return {
        "schema_id": "carnot.fr11.kan_cl_nonforgetting_audit_metrics.v1",
        "schema_version": SCHEMA_VERSION,
        "feature_space": [
            "historical_exact_evidence_key",
            "row_id",
            "fixture_family",
            "exact_label",
            "answer_abstain_decision",
            "routing_outcome",
        ],
        "metrics": {
            "exact_label_consistency": (
                "accepted exact labels must answer; rejected exact labels must abstain"
            ),
            "routing_bin_retention": (
                "trace rows are grouped into answer_path, abstain_path, or explicit route bins"
            ),
            "negative_control_regression": (
                "negative controls must not change exact action, consistency, or routing"
            ),
            "locality_boundary": (
                "the same exact evidence key must retain action, label, decision, and route bin"
            ),
        },
        "rollback_policy": [
            "heldout_regression_count > 0",
            "drift_regression_count > 0",
            "negative_control_regression_count > 0",
            "source_drift_case_count > 0",
            "locality_violation_count > 0",
        ],
        "not_authority_for": [
            "model-weight learning",
            "KAN training",
            "sidecar verifier promotion",
        ],
    }


def precondition_checks(sources: Mapping[str, Any]) -> JsonDict:
    """Expose source readiness and sidecar boundary checks."""

    exp3200 = sources.get("exp3200", {})
    exp3187 = sources.get("exp3187", {})
    return {
        "exp3200_present": isinstance(exp3200, Mapping) and bool(exp3200),
        "exp3200_terminal": isinstance(exp3200, Mapping) and is_terminal(exp3200),
        "exp3200_trace_count": int(exp3200.get("trace_count") or 0)
        if isinstance(exp3200, Mapping)
        else 0,
        "exp3187_present": isinstance(exp3187, Mapping) and bool(exp3187),
        "exp3187_ready": isinstance(exp3187, Mapping)
        and exp3187.get("fr11_cross_environment_drift_replay_v1_ready") is True,
        "exp3187_terminal": isinstance(exp3187, Mapping) and is_terminal(exp3187),
        "source_live_or_mutation_detected": any(
            source_claims_live_or_mutation(payload)
            for payload in sources.values()
            if isinstance(payload, Mapping) and payload
        ),
    }


def inference_substrate(mode: str = "controller_memory_kan_cl_sidecar_audit") -> JsonDict:
    """REQ-LEARN-3201-6: declare diagnostic replay with no model mutation."""

    return {
        "mode": mode,
        "controller_memory_replay_only": True,
        "sidecar_audit_only": True,
        "uses_checked_in_artifacts_only": True,
        "executes_live_model_inference": False,
        "fresh_live_inference_calls": 0,
        "model_weight_learning": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "base_model_weights_updated": False,
        "kan_model_weight_training": False,
        "hidden_state_mutation_claimed": False,
        "sidecar_verifier_authority": False,
    }


def source_artifacts(root: Path) -> list[JsonDict]:
    """List source files and checksums for artifact lineage."""

    rows: list[JsonDict] = []
    for source_id, rel_path, required in SOURCE_ARTIFACTS:
        path = root / rel_path
        exists = path.is_file()
        rows.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "required": required,
                "exists": exists,
                "sha256": sha256_file(path) if exists else None,
            }
        )
    return rows


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the sidecar audit overclaims learning or authority."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("model_weight_update_performed") is not False:
        raise ValueError("model_weight_update_performed must remain false")
    if artifact.get("sidecar_promotion_allowed") is not False:
        raise ValueError("sidecar_promotion_allowed must remain false")
    if not isinstance(artifact.get("audit_metric_schema"), Mapping):
        raise ValueError("audit_metric_schema must be a mapping")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        raise ValueError("inference_substrate must be a mapping")
    if int(substrate.get("fresh_live_inference_calls") or 0) != 0:
        raise ValueError("fresh_live_inference_calls must remain zero")
    if any(substrate.get(flag) is True for flag in MUTATION_FLAGS):
        raise ValueError("live-inference and model mutation flags must remain false")
    if rollback_required(artifact) and artifact.get("rollback_triggered") is not True:
        raise ValueError("rollback_triggered must be true when regressions are present")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith("complete:"):
        raise ValueError("honest_verdict must start with complete:")


def rollback_required(artifact: Mapping[str, Any]) -> bool:
    """Return whether visible audit counts require rollback."""

    return any(
        int(artifact.get(field) or 0) > 0
        for field in (
            "heldout_regression_count",
            "drift_regression_count",
            "negative_control_regression_count",
            "source_drift_case_count",
            "locality_violation_count",
        )
    )


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write deterministic Exp 3201 JSON."""

    root_path = Path(root)
    path = Path(output_path)
    output = path if path.is_absolute() else root_path / path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    write_json(output, artifact)
    return output


def honest_verdict(audit: Mapping[str, Any], rollback_triggered: bool) -> str:
    """Return a terminal truthful verdict that denies overclaim boundaries."""

    return (
        "complete: kan-cl nonforgetting sidecar audit finished; "
        f"heldout_replay_count={audit['heldout_replay_count']}; "
        f"drift_replay_count={audit['drift_replay_count']}; "
        f"negative_control_regression_count={audit['negative_control_regression_count']}; "
        f"locality_violation_count={audit['locality_violation_count']}; "
        f"rollback_triggered={str(rollback_triggered).lower()}; "
        "model_weight_update_performed=false; "
        "sidecar_promotion_allowed=false"
    )


def normalize_action(value: Any) -> str:
    """Normalize small routing/action tokens used by trace replay rows."""

    text = str(value or "").strip().lower().replace(" ", "_")
    return text or "unknown"


def duration(started_s: float, now_s: float | None) -> float:
    """Return stable elapsed seconds for artifact provenance."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def sha256_file(path: Path) -> str | None:
    """Return a file checksum when an audit source exists."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON output for deterministic artifact diffs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
