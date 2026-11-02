"""Build the Exp 3168 conservative repair-gate decision v3 artifact.

Spec refs: REQ-VERIFY-3168, SCENARIO-VERIFY-3168.

This module is a deterministic aggregation gate. It does not run a verifier,
call a model, or attempt repair. The point is to make the repair decision
machine-readable even when the upstream clean rerun is blocked, tainted, or
missing, so downstream conductor work cannot mistake missing evidence for a
safe repair opportunity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
ARTIFACT = "experiment_3168_repair_gate_decision_v3"
SCHEMA = "carnot.repair_gate_decision.v3"
OUTPUT_REL_PATH = Path("results/experiment_3168_repair_gate_decision_v3.json")

EXP3167_REL_PATH = Path("results/experiment_3167_clean_live_sota_verifier_rerun_v9.json")
EXP3166_REL_PATH = Path("results/experiment_3166_verifier_invariance_token_suspicion_audit_v1.json")
EXP3137_REL_PATH = Path("results/experiment_3137_exact_safe_accept_abstain_contract_v1.json")
EXP3138_REL_PATH = Path("results/experiment_3138_canonical_answer_vericot_grounding_pilot_v1.json")
EXP3153_REL_PATH = Path("results/experiment_3153_repair_gate_unlock_decision_v2.json")
EXP3140_REL_PATH = Path("results/experiment_3140_repair_gate_unlock_decision_v1.json")

FALSE_ACCEPT_GATE = 0.10
ALLOWED_STATES = {
    "unblocked",
    "blocked_false_accept",
    "blocked_flagged_verifier",
    "blocked_missing_live_model",
    "blocked_missing_exact_labels",
    "blocked_invariance_failure",
    "blocked_gated_skip",
    "blocked_missing_clean_rerun",
    "blocked_other",
}
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
    "repair_gate_decision_v3_ready",
    "repair_gate_state",
    "clean_rerun_artifact_present",
    "false_accept_rate",
    "false_accept_gate_passed",
    "flagged_adversarial",
    "controlled_invariance_passed",
    "exact_authority_ready",
    "selected_repair_rows",
    "repair_blockers",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3168_repair_gate_decision_v3.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/verify -m pytest -o addopts='' tests/python/test_experiment_3168_repair_gate_decision_v3.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/repair_gate_decision_v3.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)


@dataclass(frozen=True)
class SourceSpec:
    role: str
    path: Path
    required: bool


SOURCE_SPECS = (
    SourceSpec("agents_repo_instructions", Path("AGENTS.md"), False),
    SourceSpec("codex_repo_workflow", Path("CODEX.md"), False),
    SourceSpec("claude_authenticity_rules", Path("CLAUDE.md"), False),
    SourceSpec("verification_openspec", Path("openspec/capabilities/verification/spec.md"), True),
    SourceSpec("exp3167_clean_live_verifier_rerun_v9", EXP3167_REL_PATH, True),
    SourceSpec("exp3166_invariance_token_suspicion_audit", EXP3166_REL_PATH, True),
    SourceSpec("exp3137_exact_safe_contract", EXP3137_REL_PATH, True),
    SourceSpec("exp3138_canonical_grounding", EXP3138_REL_PATH, True),
    SourceSpec("exp3153_thin_gate_decision_v2", EXP3153_REL_PATH, False),
    SourceSpec("exp3140_repair_gate_decision_v1", EXP3140_REL_PATH, False),
    SourceSpec("exp3168_module", Path("python/carnot/verify/repair_gate_decision_v3.py"), False),
    SourceSpec(
        "exp3168_tests",
        Path("tests/python/test_experiment_3168_repair_gate_decision_v3.py"),
        False,
    ),
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3168: build the terminal repair-gate decision without inference."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    clean_path = root_path / EXP3167_REL_PATH
    clean_present = clean_path.is_file()
    exp3167 = read_json_object(clean_path)
    exp3166 = read_json_object(root_path / EXP3166_REL_PATH)
    exp3137 = read_json_object(root_path / EXP3137_REL_PATH)
    exp3138 = read_json_object(root_path / EXP3138_REL_PATH)
    sources = source_artifacts(root_path)
    selected_rows = repair_rows_from_exact_authority(exp3167, exp3137, exp3138, exp3166)

    false_accept_rate = false_accept_rate_from(exp3167, clean_present)
    false_accept_gate_passed = bool(
        clean_present
        and false_accept_rate <= FALSE_ACCEPT_GATE
        and exp3167.get("false_accept_gate_passed") is True
    )
    flagged_adversarial = exp3167.get("flagged_adversarial") is True
    controlled_invariance_passed = exp3167.get("controlled_invariance_passed") is True
    exact_ready = exact_authority_ready(exp3167, exp3166, exp3137, exp3138)
    live_ready = live_model_ready(exp3167)
    clean_ready = clean_present and exp3167.get("clean_live_verifier_rerun_v9_ready") is True
    gated_skip = clean_present and exp3167.get("gated_skip") is True
    regression_rows_included = clean_present and exp3167.get("regression_rows_included") is True
    headline_claim_allowed = clean_present and exp3167.get("headline_claim_allowed") is True
    blockers = repair_blockers(
        clean_present=clean_present,
        clean_ready=clean_ready,
        gated_skip=gated_skip,
        false_accept_rate=false_accept_rate,
        false_accept_gate_passed=false_accept_gate_passed,
        flagged_adversarial=flagged_adversarial,
        live_model_ready=live_ready,
        regression_rows_included=regression_rows_included,
        exact_authority_ready=exact_ready,
        controlled_invariance_passed=controlled_invariance_passed,
        headline_claim_allowed=headline_claim_allowed,
        selected_rows_ready=bool(selected_rows),
        clean_rerun=exp3167,
    )
    state = repair_gate_state(
        clean_present=clean_present,
        flagged_adversarial=flagged_adversarial,
        gated_skip=gated_skip,
        live_model_ready=live_ready,
        exact_authority_ready=exact_ready,
        controlled_invariance_passed=controlled_invariance_passed,
        false_accept_gate_passed=false_accept_gate_passed,
        false_accept_rate=false_accept_rate,
        blockers=blockers,
    )
    terminal_rows = selected_rows if state == "unblocked" else []
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "repair_gate_decision_v3_ready": state in ALLOWED_STATES
        and (state == "unblocked" or bool(blockers)),
        "repair_gate_state": state,
        "clean_rerun_artifact_present": clean_present,
        "clean_live_verifier_rerun_v9_ready": clean_ready,
        "gated_skip": gated_skip,
        "false_accept_rate": false_accept_rate,
        "false_accept_gate_passed": false_accept_gate_passed,
        "flagged_adversarial": flagged_adversarial,
        "controlled_invariance_passed": controlled_invariance_passed,
        "exact_authority_ready": exact_ready,
        "live_model_ready": live_ready,
        "regression_rows_included": regression_rows_included,
        "headline_claim_allowed": headline_claim_allowed,
        "selected_repair_rows": terminal_rows,
        "repair_blockers": blockers,
        "gate_criteria": gate_criteria(),
        "source_artifacts": sources,
        "source_checksums": {row["path"]: row["sha256"] for row in sources if row.get("sha256")},
        "inference_substrate": inference_substrate(exp3167),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
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
    """Build, validate, and persist the Exp 3168 decision JSON."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def false_accept_rate_from(clean_rerun: Mapping[str, Any], clean_present: bool) -> float:
    """Return the carried-forward false-accept rate, failing closed when absent."""

    if not clean_present:
        return 1.0
    value = clean_rerun.get("false_accept_rate")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return 1.0


def exact_authority_ready(
    clean_rerun: Mapping[str, Any],
    exp3166: Mapping[str, Any],
    exp3137: Mapping[str, Any],
    exp3138: Mapping[str, Any],
) -> bool:
    """Return whether exact labels and exact-only acceptance authority are present."""

    if int(clean_rerun.get("exact_ground_truth_count") or 0) <= 0:
        return False
    if exp3137.get("acceptance_contract_v1_ready") is not True:
        return False
    if float(exp3137.get("replay_false_accept_rate") or 0.0) != 0.0:
        return False
    if exp3138.get("canonical_grounding_pilot_v1_ready") is not True:
        return False
    if exp3138.get("residual_false_accept_rows") not in ([], None):
        return False
    if exp3166.get("verifier_invariance_token_suspicion_audit_ready") is not True:
        return False
    policy = exp3166.get("downstream_policy_for_exp3167")
    if not isinstance(policy, Mapping) or policy.get("acceptance_requires_exact_authority") is not True:
        return False
    rows = mapping_rows(exp3166.get("trusted_exact_rows"))
    replay_rows = mapping_rows(exp3137.get("replay_rows"))
    return bool(rows) and bool(replay_rows) and all_exact_rows_have_labels(rows + replay_rows)


def live_model_ready(clean_rerun: Mapping[str, Any]) -> bool:
    """Return whether the v9 artifact actually reused bounded live model evidence."""

    substrate = clean_rerun.get("inference_substrate")
    inference = substrate if isinstance(substrate, Mapping) else {}
    return bool(
        int(clean_rerun.get("live_call_count") or 0) > 0
        and (clean_rerun.get("selected_model_ids") or [])
        and int(inference.get("live_model_calls") or 0) > 0
        and inference.get("executes_models") is True
    )


def all_exact_rows_have_labels(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether exact-authority rows carry non-empty labels and authority marks."""

    for row in rows:
        if not str(row.get("exact_label") or ""):
            return False
        trusted = row.get("trusted_exact_authority")
        authority = row.get("acceptance_authority")
        if trusted is False or authority is False:
            return False
    return True


def repair_rows_from_exact_authority(
    clean_rerun: Mapping[str, Any],
    exp3137: Mapping[str, Any],
    exp3138: Mapping[str, Any],
    exp3166: Mapping[str, Any],
) -> list[JsonDict]:
    """List the repair denominator from exact false-accept regression rows."""

    planned = clean_rerun.get("planned_rerun_set")
    planned_rows = planned if isinstance(planned, Mapping) else {}
    regression_ids = [str(row_id) for row_id in planned_rows.get("regression_row_ids") or [] if row_id]
    exact_by_id = exact_rows_by_id(exp3137, exp3166)
    canonical_by_id = {row_id_from(row): row for row in mapping_rows(exp3138.get("regression_row_replay"))}
    selected: list[JsonDict] = []
    for row_id in regression_ids:
        exact_row = exact_by_id.get(row_id)
        canonical = canonical_by_id.get(row_id, {})
        if not exact_row or not canonical:
            continue
        solver = canonical.get("solver_certificate_summary")
        solver_summary = solver if isinstance(solver, Mapping) else {}
        contract = canonical.get("contract_replay")
        contract_replay = contract if isinstance(contract, Mapping) else {}
        selected.append(
            {
                "row_id": row_id,
                "exact_authority_constraints": {
                    "exact_label": str(exact_row.get("exact_label") or canonical.get("exact_label") or ""),
                    "expected_action": str(
                        exact_row.get("expected_action") or canonical.get("expected_action") or ""
                    ),
                    "exact_safe_decision": str(exact_row.get("decision") or ""),
                    "canonical_decision": str(contract_replay.get("decision") or ""),
                    "solver_or_test_authority": str(solver_summary.get("solver_authority") or ""),
                    "minimal_correction_set": solver_summary.get("minimal_correction_set") or {},
                    "unsat_core": solver_summary.get("unsat_core") or [],
                },
            }
        )
    return selected


def exact_rows_by_id(exp3137: Mapping[str, Any], exp3166: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Merge exact labels from the replay contract and trusted audit rows by ID."""

    rows: dict[str, JsonDict] = {}
    for row in mapping_rows(exp3166.get("trusted_exact_rows")):
        rows[row_id_from(row)] = dict(row)
    for row in mapping_rows(exp3137.get("replay_rows")):
        row_id = row_id_from(row)
        rows[row_id] = {**rows.get(row_id, {}), **dict(row)}
    return rows


def repair_blockers(
    *,
    clean_present: bool,
    clean_ready: bool,
    gated_skip: bool,
    false_accept_rate: float,
    false_accept_gate_passed: bool,
    flagged_adversarial: bool,
    live_model_ready: bool,
    regression_rows_included: bool,
    exact_authority_ready: bool,
    controlled_invariance_passed: bool,
    headline_claim_allowed: bool,
    selected_rows_ready: bool,
    clean_rerun: Mapping[str, Any],
) -> list[str]:
    """Build human-actionable blockers in gate-criteria order."""

    blockers: list[str] = []
    if not clean_present:
        blockers.append("clean v9 rerun artifact is missing")
        return blockers
    if not clean_ready:
        blockers.append("clean_live_verifier_rerun_v9_ready is not true")
    if gated_skip:
        reason = str(clean_rerun.get("gated_skip_reason") or "no reason recorded")
        blockers.append(f"gated_skip=true: {reason}")
    if false_accept_rate > FALSE_ACCEPT_GATE:
        blockers.append(f"false_accept_rate={false_accept_rate} exceeds gate <= {FALSE_ACCEPT_GATE}")
    if not false_accept_gate_passed:
        blockers.append("false_accept_gate_passed is not true")
    if flagged_adversarial:
        blockers.append("flagged_adversarial=true")
    if not live_model_ready:
        blockers.append("bounded live model rerun is missing")
    if not regression_rows_included:
        blockers.append("regression_rows_included is not true")
    if not exact_authority_ready:
        blockers.append("exact authority is not ready")
    if not controlled_invariance_passed:
        blockers.append("controlled_invariance_passed is not true")
    if not headline_claim_allowed:
        blockers.append("headline_claim_allowed is not true")
    if not selected_rows_ready:
        blockers.append("selected repair rows with exact authority constraints are missing")
    return blockers


def repair_gate_state(
    *,
    clean_present: bool,
    flagged_adversarial: bool,
    gated_skip: bool,
    live_model_ready: bool,
    exact_authority_ready: bool,
    controlled_invariance_passed: bool,
    false_accept_gate_passed: bool,
    false_accept_rate: float,
    blockers: Sequence[str],
) -> str:
    """Collapse many checks into the one state downstream tasks can consume."""

    if not clean_present:
        return "blocked_missing_clean_rerun"
    if false_accept_rate > FALSE_ACCEPT_GATE:
        return "blocked_false_accept"
    if flagged_adversarial:
        return "blocked_flagged_verifier"
    if gated_skip:
        return "blocked_gated_skip"
    if not live_model_ready:
        return "blocked_missing_live_model"
    if not exact_authority_ready:
        return "blocked_missing_exact_labels"
    if not controlled_invariance_passed:
        return "blocked_invariance_failure"
    if not false_accept_gate_passed:
        return "blocked_false_accept"
    if blockers:
        return "blocked_other"
    return "unblocked"


def gate_criteria() -> JsonDict:
    """Expose the exact criteria so another runner can audit the decision."""

    return {
        "clean_live_verifier_rerun_v9_ready": True,
        "gated_skip": False,
        "max_false_accept_rate": FALSE_ACCEPT_GATE,
        "false_accept_gate_passed": True,
        "flagged_adversarial": False,
        "regression_rows_included": True,
        "exact_labels_present": True,
        "controlled_invariance_passed": True,
        "headline_claim_allowed": True,
    }


def inference_substrate(clean_rerun: Mapping[str, Any]) -> JsonDict:
    """Declare that this decision reused evidence and performed no live work."""

    substrate = clean_rerun.get("inference_substrate")
    upstream = substrate if isinstance(substrate, Mapping) else {}
    return {
        "kind": "deterministic_repair_gate_decision_v3",
        "aggregation_only": True,
        "executes_models": False,
        "executes_repairs": False,
        "executes_verifiers": False,
        "executes_solvers": False,
        "no_live_inference": True,
        "live_model_calls": 0,
        "repair_calls": 0,
        "source_live_model_calls_reused": int(upstream.get("live_model_calls") or 0),
        "source_executes_models": upstream.get("executes_models") is True,
    }


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return checksummed provenance rows for every local source we read."""

    return [source_row(root, spec) for spec in SOURCE_SPECS]


def source_row(root: Path, spec: SourceSpec) -> JsonDict:
    """Build one source provenance row without treating absence as success."""

    path = root / spec.path
    return {
        "role": spec.role,
        "path": spec.path.as_posix(),
        "required": spec.required,
        "present": path.is_file(),
        "sha256": sha256_file(path),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the complete fail-closed terminal gate contract."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"Exp 3168 artifact missing required fields: {missing}")  # pragma: no cover
    state = str(artifact.get("repair_gate_state") or "")
    if state not in ALLOWED_STATES:
        raise ValueError(f"repair_gate_state must be an allowed state, got {state!r}")
    false_accept_rate = float(artifact.get("false_accept_rate"))
    if not math.isfinite(false_accept_rate) or not 0.0 <= false_accept_rate <= 1.0:
        raise ValueError("false_accept_rate must be a finite rate in [0, 1]")  # pragma: no cover
    if artifact.get("false_accept_gate_passed") is True and false_accept_rate > FALSE_ACCEPT_GATE:
        raise ValueError("false_accept_gate_passed conflicts with false_accept_rate")  # pragma: no cover
    blockers = artifact.get("repair_blockers")
    selected_rows = artifact.get("selected_repair_rows")
    if state == "unblocked" and (blockers or not selected_rows):
        raise ValueError("unblocked gate requires selected rows and no blockers")
    if state != "unblocked" and not blockers:
        raise ValueError("blocked gate requires repair_blockers")  # pragma: no cover
    substrate = artifact.get("inference_substrate")
    inference = substrate if isinstance(substrate, Mapping) else {}
    if inference.get("executes_models") or inference.get("live_model_calls") or inference.get("repair_calls"):
        raise ValueError("Exp 3168 must not execute live inference or repair calls")
    verdict = str(artifact.get("honest_verdict") or "")
    if state == "unblocked" and not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("unblocked verdict must start with a success prefix")  # pragma: no cover
    if state != "unblocked" and not verdict.startswith(f"{state}:"):
        raise ValueError("blocked verdict must start with the blocked state")  # pragma: no cover


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict wording expected by conductor consumers."""

    state = str(artifact.get("repair_gate_state") or "blocked_other")
    if state == "unblocked":
        return (
            "complete: repair_gate_state=unblocked; "
            f"false_accept_rate={artifact.get('false_accept_rate')}; "
            f"selected_repair_rows={len(artifact.get('selected_repair_rows') or [])}"
        )
    blockers = artifact.get("repair_blockers")
    first_blocker = str(blockers[0]) if isinstance(blockers, list) and blockers else "missing blocker"
    return f"{state}: {first_blocker}"


def read_json_object(path: Path) -> JsonDict:
    """Read one checked-in JSON object, returning empty evidence on failure."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def mapping_rows(value: Any) -> list[JsonDict]:
    """Return only object rows from list-like artifact fields."""

    return [dict(row) for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def row_id_from(row: Mapping[str, Any]) -> str:
    """Return the stable ID shared by exact, canonical, and planned rows."""

    return str(row.get("row_id") or row.get("fixture_id") or row.get("source_fixture_id") or "")


def sha256_file(path: Path) -> str | None:
    """Checksum source bytes so the decision traces to exact local evidence."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duration(started_s: float, now_s: float | None) -> float:
    """Return a stable nonnegative elapsed duration."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)
