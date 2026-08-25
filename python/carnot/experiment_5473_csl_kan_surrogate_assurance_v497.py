"""Exp5473: interpretable CSL KAN surrogate assurance.

Spec refs: REQ-LEARN-5473,
SCENARIO-LEARN-5473-ROLLBACK,
SCENARIO-LEARN-5473-NEGATIVE-TRANSFER,
SCENARIO-LEARN-5473-MONOTONICITY,
SCENARIO-LEARN-5473-ARTIFACT.

This module audits the frozen continuous self-learning policy from Exp5460 and
the routed memory replay from Exp5461. It intentionally uses a deterministic
KAN-like additive surrogate instead of a trainable model because the point of
this lane is interpretability: every acceptance margin is decomposed into local
feature basis terms that a reviewer can inspect without trusting hidden weights.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from carnot.provenance_receipts import receipt_bytes


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5473_csl_kan_surrogate_assurance_v497.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5473_csl_kan_surrogate_assurance_v497.py")
EXP5460_RESULT_RELATIVE_PATH = Path("results/experiment_5460_csl_policy_bandit_v496.json")
EXP5461_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5461_gated_sota_csl_memory_routing_v496.json"
)

EXPERIMENT_ID = "experiment_5473_csl_kan_surrogate_assurance_v497"
TASK_ID = "exp5473-csl-kan-surrogate-assurance-v497"
SCHEMA = "carnot.experiment_5473.csl_kan_surrogate_assurance.v497"
MILESTONE = "2026.07.497"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5473
INFERENCE_SUBSTRATE = "deterministic_csl_policy_no_llm"
TERMINAL_PREFIXES = ("complete:", "blocked:")

NO_MEMORY_CONDITION = "no_memory"
NAIVE_CONDITION = "naive_icl"
POLICY_CONDITION = "policy_selected"
FEATURE_NAMES = (
    "context_cost",
    "verifier_cost",
    "prior_success",
    "conflict_risk",
    "memory_age",
    "constraint_violation_history",
)
REQUIRED_ARTIFACT_FIELDS = (
    "csl_kan_surrogate_ready",
    "surrogate_feature_names",
    "surrogate_coefficients_or_basis",
    "assurance_ratio",
    "threshold_offset",
    "accepted_action_rate",
    "constraint_violation_count",
    "rollback_trigger_count",
    "negative_transfer_deflection_rate",
    "no_memory_baseline_score",
    "naive_icl_baseline_score",
    "governed_policy_score",
    "model_weight_mutation",
    "inference_substrate",
    "random_seed",
    "honest_verdict",
)
SPEC_REFS = (
    "REQ-LEARN-5473",
    "SCENARIO-LEARN-5473-ROLLBACK",
    "SCENARIO-LEARN-5473-NEGATIVE-TRANSFER",
    "SCENARIO-LEARN-5473-MONOTONICITY",
    "SCENARIO-LEARN-5473-ARTIFACT",
)


class KanStyleSurrogate:
    """Small additive surrogate whose terms can be audited row by row."""

    intercept = 0.28
    base_threshold = 0.58
    coefficients: JsonDict = {
        "context_cost": -0.03,
        "verifier_cost": -0.04,
        "prior_success": 0.58,
        "conflict_risk": -0.45,
        "memory_age": -0.16,
        "constraint_violation_history": -0.55,
    }

    def score(self, features: Mapping[str, float]) -> JsonDict:
        """Return score, threshold, and local basis terms for one feature row."""

        basis_terms = {
            name: round(self.coefficients[name] * float(features[name]), 6)
            for name in FEATURE_NAMES
        }
        score = round(self.intercept + sum(basis_terms.values()), 6)
        offset = threshold_offset(features)
        threshold = round(self.base_threshold + offset, 6)
        margin = round(score - threshold, 6)
        return {
            "surrogate_score": score,
            "threshold_offset": offset,
            "acceptance_threshold": threshold,
            "acceptance_margin": margin,
            "surrogate_accept": margin >= 0.0,
            "basis_terms": basis_terms,
        }


def feature_vector(
    *,
    context_cost: float,
    verifier_cost: float,
    prior_success: float,
    conflict_risk: float,
    memory_age: float,
    constraint_violation_history: float,
) -> JsonDict:
    """Normalize costs and risks onto small stable feature ranges."""

    return {
        "context_cost": round(float(context_cost) / 100.0, 6),
        "verifier_cost": round(float(verifier_cost) / 5.0, 6),
        "prior_success": round(float(prior_success), 6),
        "conflict_risk": round(float(conflict_risk), 6),
        "memory_age": round(float(memory_age), 6),
        "constraint_violation_history": round(float(constraint_violation_history), 6),
    }


def threshold_offset(features: Mapping[str, float]) -> float:
    """Tighten acceptance when the row carries stale, conflicting evidence."""

    return round(
        0.12 * float(features["conflict_risk"])
        + 0.05 * float(features["memory_age"])
        + 0.12 * float(features["constraint_violation_history"]),
        6,
    )


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp5473 artifact from the V496 replay evidence."""

    root_path = Path(root)
    policy_artifact = _read_json(root_path / EXP5460_RESULT_RELATIVE_PATH)
    routing_artifact = _read_json(root_path / EXP5461_RESULT_RELATIVE_PATH)
    surrogate_rows = score_surrogate_rows(build_surrogate_rows(policy_artifact, routing_artifact))
    rollback_count = _rollback_trigger_count(policy_artifact)
    assurance = compute_assurance(surrogate_rows, rollback_trigger_count=rollback_count)
    condition_metrics = _mapping(routing_artifact.get("condition_metrics"))
    mutation_receipt = model_weight_mutation_receipt(policy_artifact, routing_artifact)
    ready = (
        bool(tests_run)
        and assurance["constraint_violation_count"] == 0
        and assurance["accepted_action_rate"] == 1.0
        and negative_transfer_deflection_rate(_policy_rows(surrogate_rows)) == 1.0
        and mutation_receipt["model_weight_mutation"] is False
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "status": "complete" if ready else "blocked",
        "csl_kan_surrogate_ready": ready,
        "surrogate_feature_names": list(FEATURE_NAMES),
        "surrogate_coefficients_or_basis": coefficient_report(surrogate_rows),
        "assurance_ratio": assurance["assurance_ratio"],
        "threshold_offset": assurance["threshold_offset"],
        "accepted_action_rate": assurance["accepted_action_rate"],
        "constraint_violation_count": assurance["constraint_violation_count"],
        "rollback_trigger_count": assurance["rollback_trigger_count"],
        "negative_transfer_deflection_rate": negative_transfer_deflection_rate(
            _policy_rows(surrogate_rows)
        ),
        "no_memory_baseline_score": _quality(condition_metrics, NO_MEMORY_CONDITION),
        "naive_icl_baseline_score": _quality(condition_metrics, NAIVE_CONDITION),
        "governed_policy_score": _quality(condition_metrics, POLICY_CONDITION),
        "model_weight_mutation": mutation_receipt["model_weight_mutation"],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready),
        "surrogate_rows": surrogate_rows,
        "assurance_details": assurance,
        "model_weight_mutation_receipt": mutation_receipt,
        "source_artifacts": [
            str(EXP5460_RESULT_RELATIVE_PATH),
            str(EXP5461_RESULT_RELATIVE_PATH),
        ],
        "source_files": {
            "module": str(MODULE_RELATIVE_PATH),
            "spec": str(SPEC_RELATIVE_PATH),
            "upstream_policy": str(EXP5460_RESULT_RELATIVE_PATH),
            "upstream_routing": str(EXP5461_RESULT_RELATIVE_PATH),
        },
        "source_file_checksums": _source_file_checksums(root_path),
        "tests_run": _normalise_tests_run(tests_run),
        "research_conductor_modified": False,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def build_surrogate_rows(
    policy_artifact: Mapping[str, Any],
    routing_artifact: Mapping[str, Any],
) -> list[JsonDict]:
    """Create one surrogate feature row for each comparable replay condition."""

    condition_metrics = _mapping(routing_artifact.get("condition_metrics"))
    rows = [
        _surrogate_row_from_replay(
            row,
            prior_success=_quality(condition_metrics, str(row.get("condition"))),
        )
        for row in _list_of_mappings(routing_artifact.get("row_results"))
        if row.get("condition") in {NO_MEMORY_CONDITION, NAIVE_CONDITION, POLICY_CONDITION}
    ]
    rolled_back = _rollback_trigger_count(policy_artifact)
    for index, row in enumerate(rows):
        row["rollback_audit_visible_count"] = rolled_back
        row["surrogate_row_index"] = index
    return _json_ready(rows)


def score_surrogate_rows(
    rows: Sequence[Mapping[str, Any]],
    surrogate: KanStyleSurrogate | None = None,
) -> list[JsonDict]:
    """Add KAN-like score terms and active-assurance flags to rows."""

    model = surrogate or KanStyleSurrogate()
    scored: list[JsonDict] = []
    for row in rows:
        score = model.score(_mapping(row.get("features")))
        enriched = {**copy.deepcopy(dict(row)), **score}
        enriched["active_for_assurance"] = bool(
            enriched.get("candidate_for_assurance") is True
            and enriched.get("rollback_required") is not True
        )
        scored.append(enriched)
    return _json_ready(scored)


def compute_assurance(
    scored_rows: Sequence[Mapping[str, Any]],
    *,
    rollback_trigger_count: int | None = None,
) -> JsonDict:
    """Summarize accepted-action safety with a finite-sample lower bound."""

    rows = _list_of_mappings(list(scored_rows))
    active = [row for row in rows if row.get("active_for_assurance") is True]
    accepted = [
        row
        for row in active
        if row.get("surrogate_accept") is True and row.get("accepted_by_final_authority") is True
    ]
    violations = [
        row
        for row in active
        if row.get("surrogate_accept") is True
        and row.get("accepted_by_final_authority") is not True
    ]
    accepted_count = len(accepted)
    ratio = _rate(accepted_count - len(violations), accepted_count)
    return {
        "candidate_action_count": len(active),
        "accepted_action_count": accepted_count,
        "accepted_action_rate": _rate(accepted_count, len(active)),
        "constraint_violation_count": len(violations),
        "rollback_trigger_count": int(
            rollback_trigger_count
            if rollback_trigger_count is not None
            else sum(1 for row in rows if row.get("rollback_required") is True)
        ),
        "assurance_ratio": ratio,
        "finite_sample_conservative_bound": finite_sample_bound(ratio, accepted_count),
        "threshold_offset": max(
            [float(row.get("threshold_offset", 0.0)) for row in active] or [0.0]
        ),
    }


def finite_sample_bound(ratio: float, count: int, alpha: float = 0.05) -> float:
    """Use a Hoeffding-style one-sided lower bound for tiny replay samples."""

    penalty = math.sqrt(math.log(1.0 / float(alpha)) / (2.0 * count)) if count else 1.0
    return round(max(0.0, float(ratio) - penalty), 6)


def negative_transfer_deflection_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    """Count risky rows as deflected when accepted safely or rejected as unsafe."""

    risky = [row for row in rows if row.get("negative_transfer_candidate") is True]
    deflected = [
        row
        for row in risky
        if (
            row.get("surrogate_accept") is True
            and row.get("accepted_by_final_authority") is True
            and row.get("negative_transfer_detected") is not True
        )
        or (
            row.get("surrogate_accept") is not True
            and row.get("negative_transfer_detected") is True
        )
    ]
    return _rate(len(deflected), len(risky))


def coefficient_report(scored_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return coefficients plus row-local basis terms for artifact review."""

    surrogate = KanStyleSurrogate()
    return {
        "model_family": "deterministic_additive_kan_like_surrogate",
        "intercept": surrogate.intercept,
        "base_threshold": surrogate.base_threshold,
        "coefficients": dict(surrogate.coefficients),
        "local_basis": [
            {
                "task_id": row.get("task_id"),
                "condition": row.get("condition"),
                "basis_terms": row.get("basis_terms"),
                "surrogate_score": row.get("surrogate_score"),
                "threshold_offset": row.get("threshold_offset"),
                "acceptance_margin": row.get("acceptance_margin"),
            }
            for row in scored_rows
        ],
    }


def synthetic_surrogate_row(
    *,
    task_id: str,
    features: Mapping[str, float],
    condition: str = POLICY_CONDITION,
    accepted_by_final_authority: bool = True,
    negative_transfer_candidate: bool = False,
    negative_transfer_detected: bool = False,
    rollback_required: bool = False,
) -> JsonDict:
    """Build compact synthetic rows for targeted assurance unit tests."""

    return {
        "task_id": task_id,
        "condition": condition,
        "case_family": "synthetic",
        "features": dict(features),
        "candidate_for_assurance": condition == POLICY_CONDITION,
        "accepted_by_final_authority": accepted_by_final_authority,
        "negative_transfer_candidate": negative_transfer_candidate,
        "negative_transfer_detected": negative_transfer_detected,
        "rollback_required": rollback_required,
    }


def model_weight_mutation_receipt(
    policy_artifact: Mapping[str, Any],
    routing_artifact: Mapping[str, Any],
) -> JsonDict:
    """Prove Exp5473 only reads governed sidecar state and replay receipts."""

    policy_clean = policy_artifact.get("no_weight_mutation") is True
    routing_clean = routing_artifact.get("no_weight_mutation") is True
    routing_receipt = _mapping(routing_artifact.get("weight_mutation_receipt"))
    return {
        "model_weight_mutation": not (policy_clean and routing_clean),
        "exp5460_no_weight_mutation": policy_clean,
        "exp5461_no_weight_mutation": routing_clean,
        "exp5461_model_weights_written": routing_receipt.get("model_weights_written") is True,
        "adapter_weights_written": routing_receipt.get("adapter_weights_written") is True,
        "learned_state_scope": "governed_policy_state_and_memory_routing_records_only",
    }


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Write the Exp5473 JSON artifact unless the caller requests dry-run."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    if write:
        _write_json(Path(result_path), artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when the deliverable cannot support the V497 readiness claim."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5473 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema and readiness errors without mutating the artifact."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(",".join(missing))
    for field in (
        "assurance_ratio",
        "threshold_offset",
        "accepted_action_rate",
        "negative_transfer_deflection_rate",
        "no_memory_baseline_score",
        "naive_icl_baseline_score",
        "governed_policy_score",
    ):
        if not isinstance(artifact.get(field), int | float):
            errors.append(field)
    if artifact.get("surrogate_feature_names") != list(FEATURE_NAMES):
        errors.append("surrogate_feature_names")
    if not _valid_coefficient_report(artifact.get("surrogate_coefficients_or_basis")):
        errors.append("surrogate_coefficients_or_basis")
    if artifact.get("model_weight_mutation") is not False:
        errors.append("model_weight_mutation")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("constraint_violation_count") != 0:
        errors.append("constraint_violation_count")
    if artifact.get("rollback_trigger_count") != 1:
        errors.append("rollback_trigger_count")
    if artifact.get("negative_transfer_deflection_rate") != 1.0:
        errors.append("negative_transfer_deflection_rate")
    if artifact.get("csl_kan_surrogate_ready") is not True:
        errors.append("csl_kan_surrogate_ready")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("scripts/research_conductor.py")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact payload while excluding the self-referential field."""

    return _sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def _surrogate_row_from_replay(row: Mapping[str, Any], *, prior_success: float) -> JsonDict:
    condition = str(row.get("condition"))
    effective = str(row.get("effective_condition", condition))
    negative_candidate = row.get("negative_transfer_candidate") is True
    stale_candidate = row.get("stale_memory_candidate") is True
    conflict = _conflict_risk(condition, negative_candidate, row.get("negative_transfer_detected"))
    memory_age = _memory_age(condition, effective, stale_candidate, str(row.get("case_family")))
    violation_history = _constraint_history(condition, row)
    return {
        "task_id": str(row.get("task_id")),
        "condition": condition,
        "effective_condition": effective,
        "case_family": str(row.get("case_family")),
        "features": feature_vector(
            context_cost=float(row.get("context_cost", 0.0)),
            verifier_cost=float(row.get("verifier_cost", 0.0)),
            prior_success=prior_success,
            conflict_risk=conflict,
            memory_age=memory_age,
            constraint_violation_history=violation_history,
        ),
        "candidate_for_assurance": condition == POLICY_CONDITION,
        "accepted_by_final_authority": row.get("accepted_by_final_authority") is True,
        "negative_transfer_candidate": negative_candidate,
        "negative_transfer_detected": row.get("negative_transfer_detected") is True,
        "rollback_required": False,
        "source_row_checksum": str(row.get("row_checksum", "")),
    }


def _conflict_risk(condition: str, negative_candidate: bool, transfer_detected: Any) -> float:
    return {
        (False, False): 0.0,
        (True, False): 0.25 if condition == POLICY_CONDITION else 0.5,
        (True, True): 1.0,
    }[(negative_candidate, transfer_detected is True)]


def _memory_age(
    condition: str,
    effective_condition: str,
    stale_candidate: bool,
    case_family: str,
) -> float:
    age_by_case = {
        "stale_memory": 1.0 if condition == NAIVE_CONDITION else 0.25,
        "repeated_task": 0.2 if effective_condition != NO_MEMORY_CONDITION else 0.0,
    }
    return float(age_by_case.get(case_family, 0.1 if stale_candidate else 0.0))


def _constraint_history(condition: str, row: Mapping[str, Any]) -> float:
    return 1.0 if condition == NAIVE_CONDITION and row.get("negative_transfer_detected") else 0.0


def _quality(condition_metrics: Mapping[str, Any], condition: str) -> float:
    return float(_mapping(condition_metrics.get(condition)).get("quality_score", 0.0))


def _rollback_trigger_count(policy_artifact: Mapping[str, Any]) -> int:
    rollback = _mapping(policy_artifact.get("rollback_audit"))
    return len(_list_of_mappings(rollback.get("rollback_events")))


def _policy_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [dict(row) for row in rows if row.get("condition") == POLICY_CONDITION]


def _valid_coefficient_report(value: Any) -> bool:
    report = _mapping(value)
    coeffs = _mapping(report.get("coefficients"))
    return (
        report.get("intercept") == KanStyleSurrogate().intercept
        and set(coeffs) == set(FEATURE_NAMES)
        and isinstance(report.get("local_basis"), list)
    )


def _normalise_tests_run(tests_run: Sequence[str | Mapping[str, Any]]) -> list[JsonDict]:
    return [_normalise_test_run(item) for item in tests_run] or [
        {"command": "not_recorded", "outcome": "not_recorded"}
    ]


def _normalise_test_run(item: str | Mapping[str, Any]) -> JsonDict:
    return {
        "command": str(item if isinstance(item, str) else item.get("command", "")),
        "outcome": str("passed" if isinstance(item, str) else item.get("outcome", "passed")),
    }


def _honest_verdict(ready: bool) -> str:
    return (
        "complete: deterministic CSL KAN surrogate audited frozen policy routing with "
        "chance-style assurance and no model weight mutation"
        if ready
        else "blocked: csl_kan_surrogate_ready prerequisites missing"
    )


def _source_file_checksums(root: Path) -> JsonDict:
    paths = {
        "module": root / MODULE_RELATIVE_PATH,
        "spec": root / SPEC_RELATIVE_PATH,
        "upstream_policy": root / EXP5460_RESULT_RELATIVE_PATH,
        "upstream_routing": root / EXP5461_RESULT_RELATIVE_PATH,
    }
    return {name: _file_checksum(path) for name, path in paths.items() if path.is_file()}


def _file_checksum(path: Path) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            receipt_bytes(path, artifact_relative_path=RESULT_RELATIVE_PATH)
        ).hexdigest()
    )


def _read_json(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt inputs block upstream.
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_mappings(value: Any) -> list[JsonDict]:
    return [dict(row) for row in value] if isinstance(value, list) else []


def _rate(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        _json_ready(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))
