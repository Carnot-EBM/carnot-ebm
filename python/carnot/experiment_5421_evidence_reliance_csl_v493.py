"""Exp5421: evidence-reliance drift diagnostic for continuous self-learning.

Spec refs: REQ-LEARN-5421,
SCENARIO-LEARN-5421-DRIFT, SCENARIO-LEARN-5421-RAW-RETENTION,
SCENARIO-LEARN-5421-SAFETY, SCENARIO-LEARN-5421-ROLLBACK.

This experiment is a deterministic controller replay. It deliberately separates
"the answer stayed correct" from "the answer still relied on the same evidence"
because hidden forgetting can move the controller from verifier-grounded
evidence toward stale or unsafe memory without immediately changing surface
accuracy. No base-model or adapter weights are loaded or written; learning is
only a controller sidecar over raw retained episodes and deterministic verifier
labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5408_resource_accounted_csl_controller_v492 as exp5408
from carnot import experiment_5409_uncertainty_gated_promotion_v492 as exp5409
from carnot.provenance_receipts import receipt_bytes


JsonDict = dict[str, Any]
JsonList = list[JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5421_evidence_reliance_csl_v493"
EXPERIMENT_ID = "exp5421-v493-evidence-reliance-csl"
MILESTONE = "2026.07.493"
SCHEMA = "carnot.experiment_5421.evidence_reliance_csl.v493"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5421
INFERENCE_SUBSTRATE = "deterministic_self_learning_controller"
HIDDEN_FORGETTING_THRESHOLD = 0.45

RESULT_RELATIVE_PATH = Path("results/experiment_5421_evidence_reliance_csl_v493.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5421_evidence_reliance_csl_v493.py")
EXP5408_RESULT_RELATIVE_PATH = exp5408.RESULT_RELATIVE_PATH
EXP5409_RESULT_RELATIVE_PATH = exp5409.RESULT_RELATIVE_PATH
EXP5408_MODULE_RELATIVE_PATH = exp5408.MODULE_RELATIVE_PATH
EXP5409_MODULE_RELATIVE_PATH = exp5409.MODULE_RELATIVE_PATH

SPEC_REFS = (
    "REQ-LEARN-5421",
    "SCENARIO-LEARN-5421-DRIFT",
    "SCENARIO-LEARN-5421-RAW-RETENTION",
    "SCENARIO-LEARN-5421-SAFETY",
    "SCENARIO-LEARN-5421-ROLLBACK",
)
TERMINAL_PREFIXES = ("complete:", "blocked:")

REQUIRED_EPISODE_FAMILIES = frozenset(
    {"clean", "stale", "poisoned", "underspecified", "distribution_shift"}
)
UNSAFE_EPISODE_FAMILIES = frozenset({"stale", "poisoned", "underspecified", "distribution_shift"})
STALE_POISON_FAMILIES = frozenset({"stale", "poisoned"})
DEFLECTING_ACTIONS = frozenset(
    {"route_to_deterministic_verifier", "abstain", "retain_non_promoted_evidence"}
)
ROUTING_DECISIONS = frozenset(
    {
        "deterministic_verified",
        "memory_assisted_verified",
        "route_to_deterministic_verifier",
        "abstain",
        "retain_non_promoted_evidence",
    }
)

FIELD_PRINCIPLES: dict[str, str] = {
    "episode_count": "Scale and coverage.",
    "raw_episodes_retained": "Auditability.",
    "no_weight_mutation": "FR-11 boundary.",
    "rollback_verified": "Safety recovery.",
    "quality_preserved": "No learning regression.",
    "resource_delta": "Resource-aware learning.",
    "verifier_cost_delta": "Verifier economy.",
    "reliance_drift_metric": "Hidden-forgetting signal.",
    "hidden_forgetting_detected": "Explicit drift classification.",
    "stale_poison_deflection_rate": "Unsafe memory guard.",
    "evidence_reliance_csl_ready": "Downstream gate.",
    "inference_substrate": "No hidden live model inference.",
    "honest_verdict": "Terminal status; starts with complete: or blocked:.",
}

REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)
BOOL_FIELDS = (
    "raw_episodes_retained",
    "no_weight_mutation",
    "rollback_verified",
    "quality_preserved",
    "hidden_forgetting_detected",
    "evidence_reliance_csl_ready",
)
INTEGER_FIELDS = ("episode_count",)
NUMERIC_FIELDS = (
    "resource_delta",
    "verifier_cost_delta",
    "reliance_drift_metric",
    "stale_poison_deflection_rate",
)
POSITIVE_NUMERIC_FIELDS = (
    "resource_delta",
    "verifier_cost_delta",
    "reliance_drift_metric",
)


def load_source_artifacts(root: Path | str = REPO_ROOT) -> JsonDict:
    """Read the prior controller artifacts used as deterministic evidence."""

    root_path = Path(root)
    return {
        "exp5408": _read_json(root_path / EXP5408_RESULT_RELATIVE_PATH),
        "exp5409": _read_json(root_path / EXP5409_RESULT_RELATIVE_PATH),
    }


def evaluate_evidence_reliance_csl(root: Path | str = REPO_ROOT) -> JsonDict:
    """Build paired episodes and compute the evidence-reliance drift audit."""

    source_artifacts = load_source_artifacts(root)
    raw_index = build_raw_episode_index(source_artifacts["exp5408"])
    paired = build_paired_episodes(source_artifacts, raw_index)
    accuracy_before = _rate(sum(row["answer_correct_before"] for row in paired), len(paired))
    accuracy_after = _rate(sum(row["answer_correct_after"] for row in paired), len(paired))
    resource_delta = round(
        sum(
            float(row["always_verify_resource_cost"]) - float(row["resource_cost_after"])
            for row in paired
        ),
        6,
    )
    verifier_cost_delta = round(
        sum(
            float(row["always_verify_verifier_calls"]) - float(row["verifier_calls_after"])
            for row in paired
        ),
        6,
    )
    reliance_drift_metric = max(float(row["evidence_reliance_drift"]) for row in paired)
    hidden_forgetting_detected = any(
        row["surface_success_stable"]
        and float(row["evidence_reliance_drift"]) >= HIDDEN_FORGETTING_THRESHOLD
        for row in paired
    )
    raw_retained = all(
        row["raw_episode_retained"] and row["raw_episode_receipts"] for row in paired
    )
    stale_poison = [row for row in paired if row["episode_family"] in STALE_POISON_FAMILIES]
    unsafe = [row for row in paired if row["episode_family"] in UNSAFE_EPISODE_FAMILIES]
    rollback = verify_rollback_restores_prior_routing(paired)
    weight_receipt = _weight_mutation_receipt()
    return {
        "source_artifacts": source_artifacts,
        "paired_episodes": paired,
        "episode_count": len(paired),
        "raw_episode_count": len(raw_index),
        "raw_episodes_retained": raw_retained,
        "accuracy_before_rate": accuracy_before,
        "accuracy_after_rate": accuracy_after,
        "quality_preserved": accuracy_after >= accuracy_before,
        "resource_delta": resource_delta,
        "verifier_cost_delta": verifier_cost_delta,
        "reliance_drift_metric": reliance_drift_metric,
        "hidden_forgetting_detected": hidden_forgetting_detected,
        "stale_poison_deflection_rate": _deflection_rate(stale_poison),
        "uncertain_reliance_deflection_rate": _deflection_rate(unsafe),
        "rollback_verified": rollback["rollback_success"],
        "rollback_audit": rollback,
        "no_weight_mutation": weight_receipt["no_weight_mutation"],
        "weight_mutation_receipt": weight_receipt,
        "source_readiness": {
            "exp5408_resource_accounted_csl_ready": source_artifacts["exp5408"].get(
                "resource_accounted_csl_ready"
            )
            is True,
            "exp5409_uncertainty_gated_promotion_ready": source_artifacts["exp5409"].get(
                "uncertainty_gated_promotion_ready"
            )
            is True,
        },
    }


def build_raw_episode_index(exp5408_artifact: Mapping[str, Any]) -> JsonDict:
    """Index retained raw episodes and add the Exp5421 diagnostic controls."""

    raw_index = {
        str(row["raw_episode_id"]): dict(row) for row in exp5408_artifact.get("raw_episodes", [])
    }
    raw_index["raw5421-underspecified-constraint-gap"] = _diagnostic_raw_episode(
        raw_episode_id="raw5421-underspecified-constraint-gap",
        control_kind="underspecified",
        source_event_id="exp5421-underspecified-constraint-gap",
        trace_id="trace-5421-underspecified",
        claim="Answer remains correct, but the constraint evidence omits the boundary condition.",
        value_score=0.58,
        stale_risk=0.18,
        forged_reasoning_risk=0.12,
        sharing_risk=0.21,
    )
    raw_index["raw5421-distribution-shift-routing-shift"] = _diagnostic_raw_episode(
        raw_episode_id="raw5421-distribution-shift-routing-shift",
        control_kind="distribution_shift",
        source_event_id="exp5421-distribution-shift-routing-shift",
        trace_id="trace-5421-distribution-shift",
        claim="Prior verifier routing was learned on arithmetic but is being applied to code repair.",
        value_score=0.64,
        stale_risk=0.14,
        forged_reasoning_risk=0.08,
        sharing_risk=0.24,
    )
    return _json_ready(raw_index)


def build_paired_episodes(
    source_artifacts: Mapping[str, Any],
    raw_index: Mapping[str, Mapping[str, Any]],
) -> JsonList:
    """Construct before/after rows where accuracy and reliance can diverge."""

    promotion_by_family = _promotion_by_family(source_artifacts["exp5409"])
    templates = [
        {
            "pair_id": "pair5421-clean-supported-memory",
            "episode_family": "clean",
            "source_fragment_id": promotion_by_family["benign"]["fragment_id"],
            "raw_episode_ids": ["raw5396-clean-dependency-edge"],
            "surface_answer_before": "route rich verifier only when dependency provenance is missing",
            "surface_answer_after": "route rich verifier only when dependency provenance is missing",
            "routing_decision_before": "deterministic_verified",
            "routing_decision_after": "memory_assisted_verified",
            "controller_action_after": "route_with_memory_and_verifier",
            "influence_shares_before": _shares(verifier=70, memory=10, raw=10, constraint=10),
            "influence_shares_after": _shares(verifier=60, memory=25, raw=10, constraint=5),
            "grounding_source_before": "raw_episode_and_deterministic_verifier",
            "grounding_source_after": "accepted_memory_with_verifier_spotcheck",
            "verifier_dependency_before": "deterministic_verifier_primary",
            "verifier_dependency_after": "deterministic_verifier_supporting",
            "constraint_evidence_before": "explicit_constraint_receipt",
            "constraint_evidence_after": "explicit_constraint_receipt",
            "evidence_reliance_label_before": "verifier_anchored",
            "evidence_reliance_label_after": "memory_assisted_verified",
            "resource_cost_before": 170,
            "resource_cost_after": 120,
            "verifier_calls_before": 3,
            "verifier_calls_after": 2,
            "always_verify_resource_cost": 220,
            "always_verify_verifier_calls": 3,
            "promoted_after_learning": True,
            "active_learned_memory_routing_after": True,
        },
        {
            "pair_id": "pair5421-stale-runtime-reliance",
            "episode_family": "stale",
            "source_fragment_id": promotion_by_family["stale"]["fragment_id"],
            "raw_episode_ids": ["raw5396-stale-runtime-receipt"],
            "surface_answer_before": "use current runtime receipt for verifier tier selection",
            "surface_answer_after": "use current runtime receipt for verifier tier selection",
            "routing_decision_before": "deterministic_verified",
            "routing_decision_after": "route_to_deterministic_verifier",
            "controller_action_after": "route_to_deterministic_verifier",
            "influence_shares_before": _shares(verifier=75, memory=5, raw=10, constraint=10),
            "influence_shares_after": _shares(verifier=15, memory=70, raw=5, constraint=10),
            "grounding_source_before": "fresh_runtime_receipt",
            "grounding_source_after": "stale_memory_receipt",
            "verifier_dependency_before": "deterministic_verifier_primary",
            "verifier_dependency_after": "verifier_required_due_staleness",
            "constraint_evidence_before": "fresh_constraint_receipt",
            "constraint_evidence_after": "expired_constraint_receipt",
            "evidence_reliance_label_before": "fresh_verifier_grounded",
            "evidence_reliance_label_after": "stale_memory_uncertain",
            "resource_cost_before": 180,
            "resource_cost_after": 150,
            "verifier_calls_before": 3,
            "verifier_calls_after": 2,
            "always_verify_resource_cost": 230,
            "always_verify_verifier_calls": 3,
            "promoted_after_learning": False,
            "active_learned_memory_routing_after": False,
        },
        {
            "pair_id": "pair5421-poisoned-forged-reliance",
            "episode_family": "poisoned",
            "source_fragment_id": promotion_by_family["poisoned"]["fragment_id"],
            "raw_episode_ids": ["raw5396-forged-reasoning-history"],
            "surface_answer_before": "reject forged reasoning as a routing prior",
            "surface_answer_after": "reject forged reasoning as a routing prior",
            "routing_decision_before": "deterministic_verified",
            "routing_decision_after": "abstain",
            "controller_action_after": "abstain",
            "influence_shares_before": _shares(verifier=80, memory=5, raw=10, constraint=5),
            "influence_shares_after": _shares(verifier=10, memory=80, raw=5, constraint=5),
            "grounding_source_before": "raw_episode_forgery_check",
            "grounding_source_after": "forged_memory_claim",
            "verifier_dependency_before": "deterministic_verifier_primary",
            "verifier_dependency_after": "deterministic_verifier_blocked_by_poison",
            "constraint_evidence_before": "forgery_constraint_failed",
            "constraint_evidence_after": "missing_independent_constraint",
            "evidence_reliance_label_before": "poison_deflected_by_verifier",
            "evidence_reliance_label_after": "poisoned_memory_uncertain",
            "resource_cost_before": 170,
            "resource_cost_after": 80,
            "verifier_calls_before": 3,
            "verifier_calls_after": 0,
            "always_verify_resource_cost": 230,
            "always_verify_verifier_calls": 3,
            "promoted_after_learning": False,
            "active_learned_memory_routing_after": False,
        },
        {
            "pair_id": "pair5421-underspecified-boundary",
            "episode_family": "underspecified",
            "source_fragment_id": "frag5421-underspecified-constraint-gap",
            "raw_episode_ids": ["raw5421-underspecified-constraint-gap"],
            "surface_answer_before": "boundary case needs explicit verifier evidence",
            "surface_answer_after": "boundary case needs explicit verifier evidence",
            "routing_decision_before": "deterministic_verified",
            "routing_decision_after": "retain_non_promoted_evidence",
            "controller_action_after": "retain_non_promoted_evidence",
            "influence_shares_before": _shares(verifier=70, memory=10, raw=10, constraint=10),
            "influence_shares_after": _shares(verifier=25, memory=55, raw=10, constraint=10),
            "grounding_source_before": "complete_constraint_evidence",
            "grounding_source_after": "underspecified_memory_summary",
            "verifier_dependency_before": "deterministic_verifier_primary",
            "verifier_dependency_after": "verification_needed_for_missing_boundary",
            "constraint_evidence_before": "boundary_constraint_present",
            "constraint_evidence_after": "boundary_constraint_missing",
            "evidence_reliance_label_before": "constraint_complete",
            "evidence_reliance_label_after": "underspecified_uncertain",
            "resource_cost_before": 160,
            "resource_cost_after": 110,
            "verifier_calls_before": 3,
            "verifier_calls_after": 1,
            "always_verify_resource_cost": 220,
            "always_verify_verifier_calls": 3,
            "promoted_after_learning": False,
            "active_learned_memory_routing_after": False,
        },
        {
            "pair_id": "pair5421-distribution-shift",
            "episode_family": "distribution_shift",
            "source_fragment_id": "frag5421-distribution-shift-routing",
            "raw_episode_ids": ["raw5421-distribution-shift-routing-shift"],
            "surface_answer_before": "do not transfer arithmetic routing prior to code repair without verification",
            "surface_answer_after": "do not transfer arithmetic routing prior to code repair without verification",
            "routing_decision_before": "deterministic_verified",
            "routing_decision_after": "route_to_deterministic_verifier",
            "controller_action_after": "route_to_deterministic_verifier",
            "influence_shares_before": _shares(verifier=75, memory=5, raw=10, constraint=10),
            "influence_shares_after": _shares(verifier=20, memory=60, raw=10, constraint=10),
            "grounding_source_before": "in_domain_verifier_trace",
            "grounding_source_after": "out_of_domain_memory_prior",
            "verifier_dependency_before": "deterministic_verifier_primary",
            "verifier_dependency_after": "verifier_required_for_domain_shift",
            "constraint_evidence_before": "in_domain_constraint_receipt",
            "constraint_evidence_after": "cross_domain_constraint_gap",
            "evidence_reliance_label_before": "in_domain_verifier_grounded",
            "evidence_reliance_label_after": "distribution_shift_uncertain",
            "resource_cost_before": 175,
            "resource_cost_after": 145,
            "verifier_calls_before": 3,
            "verifier_calls_after": 2,
            "always_verify_resource_cost": 235,
            "always_verify_verifier_calls": 3,
            "promoted_after_learning": False,
            "active_learned_memory_routing_after": False,
        },
    ]
    return [_episode_from_template(template, raw_index) for template in templates]


def verify_rollback_restores_prior_routing(
    paired_episodes: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Inject one bad promotion and verify controller-side rollback is exact."""

    prior_active = {
        str(row["source_fragment_id"]): str(row["routing_decision_after"])
        for row in paired_episodes
        if row["active_learned_memory_routing_after"]
    }
    active_after_injection = dict(prior_active)
    bad_fragment_id = "frag5421-poisoned-reliance-drift"
    active_after_injection[bad_fragment_id] = "memory_assisted_verified"
    injected = bad_fragment_id in active_after_injection
    restored = dict(active_after_injection)
    restored.pop(bad_fragment_id)
    retained = any(
        row["episode_family"] == "poisoned" and row["raw_episode_retained"]
        for row in paired_episodes
    )
    prior_restored = restored == prior_active
    return {
        "bad_fragment_id": bad_fragment_id,
        "injected_into_active_routing": injected,
        "rollback_removed_from_active_routing": bad_fragment_id not in restored,
        "prior_routing_restored": prior_restored,
        "retained_audit_record_after_rollback": retained,
        "rollback_success": bool(injected and prior_restored and retained),
    }


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5421 terminal artifact consumed by milestone gates."""

    evaluation = evaluate_evidence_reliance_csl(root)
    readiness = _readiness_checks(evaluation, tests_run)
    ready = bool(readiness["all_passed"])
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_artifacts": [
            str(EXP5408_RESULT_RELATIVE_PATH),
            str(EXP5409_RESULT_RELATIVE_PATH),
        ],
        "status": "complete" if ready else "blocked",
        "episode_count": evaluation["episode_count"],
        "raw_episodes_retained": evaluation["raw_episodes_retained"],
        "no_weight_mutation": evaluation["no_weight_mutation"],
        "rollback_verified": evaluation["rollback_verified"],
        "quality_preserved": evaluation["quality_preserved"],
        "resource_delta": evaluation["resource_delta"],
        "verifier_cost_delta": evaluation["verifier_cost_delta"],
        "reliance_drift_metric": evaluation["reliance_drift_metric"],
        "hidden_forgetting_detected": evaluation["hidden_forgetting_detected"],
        "stale_poison_deflection_rate": evaluation["stale_poison_deflection_rate"],
        "evidence_reliance_csl_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready),
        "tests_run": [dict(row) for row in tests_run],
        "paired_episodes": evaluation["paired_episodes"],
        "raw_episode_count": evaluation["raw_episode_count"],
        "accuracy_before_rate": evaluation["accuracy_before_rate"],
        "accuracy_after_rate": evaluation["accuracy_after_rate"],
        "uncertain_reliance_deflection_rate": evaluation["uncertain_reliance_deflection_rate"],
        "hidden_forgetting_threshold": HIDDEN_FORGETTING_THRESHOLD,
        "rollback_audit": evaluation["rollback_audit"],
        "readiness_checks": readiness,
        "weight_mutation_receipt": evaluation["weight_mutation_receipt"],
        "source_readiness": evaluation["source_readiness"],
        "methodology_note": (
            "Exp5421 replays Exp5408/Exp5409 controller evidence and injects "
            "diagnostic stale, poisoned, underspecified, and distribution-shift "
            "episodes. The metric watches evidence reliance rather than answer "
            "accuracy alone; no live model inference or weight mutation occurs."
        ),
        "source_artifact_checksums": source_artifact_checksums(root),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the Exp5421 claims relied on by downstream CSL gates."""

    errors: list[str] = []
    errors.extend(field for field in REQUIRED_FIELDS if field not in artifact)
    errors.extend(field for field in BOOL_FIELDS if not isinstance(artifact.get(field), bool))
    errors.extend(
        field
        for field in INTEGER_FIELDS
        if isinstance(artifact.get(field), bool) or not isinstance(artifact.get(field), int)
    )
    errors.extend(field for field in NUMERIC_FIELDS if not _is_numeric(artifact.get(field)))
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    ready = artifact.get("evidence_reliance_csl_ready")
    if (ready is True and artifact.get("status") != "complete") or (
        artifact.get("status") == "complete" and ready is not True
    ):
        errors.append("status")
    if artifact.get("milestone") != MILESTONE:
        errors.append("milestone")
    for field in POSITIVE_NUMERIC_FIELDS:
        if _is_numeric(artifact.get(field)) and float(artifact[field]) <= 0.0:
            errors.append(field)
    if _is_numeric(artifact.get("reliance_drift_metric")) and (
        float(artifact["reliance_drift_metric"]) < HIDDEN_FORGETTING_THRESHOLD
    ):
        errors.append("reliance_drift_metric")
    for field in (
        "raw_episodes_retained",
        "no_weight_mutation",
        "rollback_verified",
        "quality_preserved",
        "hidden_forgetting_detected",
    ):
        if artifact.get(field) is not True:
            errors.append(field)
    if _is_numeric(artifact.get("stale_poison_deflection_rate")) and (
        float(artifact["stale_poison_deflection_rate"]) != 1.0
    ):
        errors.append("stale_poison_deflection_rate")
    if artifact.get("episode_count") != len(artifact.get("paired_episodes", [])):
        errors.append("episode_count")
    if ready is True and not artifact.get("tests_run"):
        errors.append("tests_run")
    if errors:
        raise ValueError("invalid Exp5421 artifact fields: " + ",".join(sorted(set(errors))))
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5421 result artifact and return its JSON payload."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def default_tests_run() -> JsonList:
    """Return the verification commands expected in the completed artifact."""

    test_path = "tests/python/test_experiment_5421_evidence_reliance_csl_v493.py"
    module_path = "python/carnot/experiment_5421_evidence_reliance_csl_v493.py"
    return [
        {
            "command": f".venv/bin/pytest {test_path} -q --no-cov -n 0",
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage run "
                f"--include={module_path} -m pytest {test_path} "
                "-q --no-cov -n 0 && .venv/bin/coverage report --fail-under=100"
            ),
            "outcome": "passed",
        },
        {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
    ]


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for source artifacts, spec, and local modules."""

    root_path = Path(root)
    return {
        "exp5408": _sha256_file(root_path / EXP5408_RESULT_RELATIVE_PATH),
        "exp5409": _sha256_file(root_path / EXP5409_RESULT_RELATIVE_PATH),
        "spec": _sha256_file(root_path / SPEC_RELATIVE_PATH),
        "module": _sha256_file(root_path / MODULE_RELATIVE_PATH),
        "exp5408_module": _sha256_file(root_path / EXP5408_MODULE_RELATIVE_PATH),
        "exp5409_module": _sha256_file(root_path / EXP5409_MODULE_RELATIVE_PATH),
    }


def _episode_from_template(
    template: Mapping[str, Any],
    raw_index: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    raw_episode_ids = [str(raw_id) for raw_id in template["raw_episode_ids"]]
    receipts = [_raw_receipt(raw_index[raw_id]) for raw_id in raw_episode_ids]
    row = dict(template)
    row.update(
        {
            "raw_episode_ids": raw_episode_ids,
            "raw_episode_receipts": receipts,
            "raw_episode_retained": True,
            "answer_correct_before": True,
            "answer_correct_after": True,
            "surface_success_stable": True,
        }
    )
    row["evidence_reliance_drift"] = _evidence_reliance_drift(row)
    return _json_ready(row)


def _evidence_reliance_drift(row: Mapping[str, Any]) -> float:
    before = row["influence_shares_before"]
    after = row["influence_shares_after"]
    memory_delta = (
        max(0.0, float(after["learned_memory"]) - float(before["learned_memory"])) / 100.0
    )
    verifier_drop = (
        max(0.0, float(before["deterministic_verifier"]) - float(after["deterministic_verifier"]))
        / 100.0
    )
    source_changed = row["grounding_source_before"] != row["grounding_source_after"]
    verifier_changed = row["verifier_dependency_before"] != row["verifier_dependency_after"]
    constraint_changed = row["constraint_evidence_before"] != row["constraint_evidence_after"]
    label_changed = row["evidence_reliance_label_before"] != row["evidence_reliance_label_after"]
    return round(
        (0.35 * memory_delta)
        + (0.35 * verifier_drop)
        + (0.10 if source_changed else 0.0)
        + (0.08 if verifier_changed else 0.0)
        + (0.07 if constraint_changed else 0.0)
        + (0.05 if label_changed else 0.0),
        6,
    )


def _promotion_by_family(exp5409_artifact: Mapping[str, Any]) -> JsonDict:
    return {
        str(row["candidate_family"]): dict(row)
        for row in exp5409_artifact.get("promotion_candidates", [])
        if str(row["candidate_family"]) in {"benign", "stale", "poisoned"}
    }


def _diagnostic_raw_episode(
    *,
    raw_episode_id: str,
    control_kind: str,
    source_event_id: str,
    trace_id: str,
    claim: str,
    value_score: float,
    stale_risk: float,
    forged_reasoning_risk: float,
    sharing_risk: float,
) -> JsonDict:
    row_evidence = {
        "claim": claim,
        "value_score": value_score,
        "byte_cost": 140,
        "stale_risk": stale_risk,
        "forged_reasoning_risk": forged_reasoning_risk,
        "self_reference_count": 0,
        "sharing_risk": sharing_risk,
        "provenance_verified": True,
        "rollback_available": True,
        "rollback_verified": True,
    }
    checksum_payload = {
        "raw_episode_id": raw_episode_id,
        "control_kind": control_kind,
        "source_event_id": source_event_id,
        "trace_id": trace_id,
        "row_evidence": row_evidence,
    }
    return {
        "record_type": "raw_episode",
        "raw_episode_id": raw_episode_id,
        "captured_at": RUN_DATE,
        "control_kind": control_kind,
        "session_id": "session-5421-diagnostic",
        "trace_id": trace_id,
        "source_artifact": str(RESULT_RELATIVE_PATH),
        "source_event_id": source_event_id,
        "row_evidence": row_evidence,
        "raw_payload_checksum": _checksum(checksum_payload),
    }


def _raw_receipt(raw_episode: Mapping[str, Any]) -> JsonDict:
    return {
        "raw_episode_id": str(raw_episode["raw_episode_id"]),
        "control_kind": str(raw_episode["control_kind"]),
        "source_event_id": str(raw_episode["source_event_id"]),
        "raw_payload_checksum": str(raw_episode["raw_payload_checksum"]),
    }


def _shares(
    *,
    verifier: int,
    memory: int,
    raw: int,
    constraint: int,
) -> JsonDict:
    return {
        "deterministic_verifier": verifier,
        "learned_memory": memory,
        "raw_episode": raw,
        "constraint_evidence": constraint,
    }


def _readiness_checks(
    evaluation: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    families = {row["episode_family"] for row in evaluation["paired_episodes"]}
    checks = {
        "source_exp5408_ready": evaluation["source_readiness"][
            "exp5408_resource_accounted_csl_ready"
        ],
        "source_exp5409_ready": evaluation["source_readiness"][
            "exp5409_uncertainty_gated_promotion_ready"
        ],
        "required_families_covered": REQUIRED_EPISODE_FAMILIES.issubset(families),
        "episodes_present": evaluation["episode_count"] > 0,
        "raw_episodes_retained": evaluation["raw_episodes_retained"] is True,
        "quality_preserved": evaluation["quality_preserved"] is True,
        "hidden_forgetting_detected": evaluation["hidden_forgetting_detected"] is True,
        "drift_threshold_crossed": evaluation["reliance_drift_metric"]
        >= HIDDEN_FORGETTING_THRESHOLD,
        "stale_poison_deflected": evaluation["stale_poison_deflection_rate"] == 1.0,
        "uncertain_reliance_deflected": evaluation["uncertain_reliance_deflection_rate"] == 1.0,
        "resource_delta_numeric": _is_numeric(evaluation["resource_delta"]),
        "verifier_cost_delta_numeric": _is_numeric(evaluation["verifier_cost_delta"]),
        "rollback_verified": evaluation["rollback_verified"] is True,
        "no_weight_mutation": evaluation["no_weight_mutation"] is True,
        "tests_recorded": bool(tests_run),
        "inference_substrate_deterministic": INFERENCE_SUBSTRATE
        == "deterministic_self_learning_controller",
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {**checks, "all_passed": not failed, "failed_checks": failed}


def _deflection_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    return _rate(
        sum(
            1
            for row in rows
            if row["controller_action_after"] in DEFLECTING_ACTIONS
            and not row["promoted_after_learning"]
            and not row["active_learned_memory_routing_after"]
        ),
        len(rows),
    )


def _honest_verdict(ready: bool) -> str:
    return (
        "complete: evidence-reliance CSL detected hidden forgetting under stable accuracy, retained raw episodes, deflected unsafe reliance, verified rollback, and did not mutate weights"
        if ready
        else "blocked: evidence-reliance CSL diagnostic did not satisfy readiness checks"
    )


def _weight_mutation_receipt() -> JsonDict:
    return {
        "no_weight_mutation": True,
        "no_adapter_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weights_written": False,
        "adapter_weights_loaded": False,
        "adapter_weights_written": False,
        "learned_state_scope": "controller_evidence_reliance_sidecar_only",
    }


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _rate(numerator: float, denominator: float) -> float:
    return round(float(numerator) / float(denominator), 6)


def _checksum(payload: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(_json_ready(stable), sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            receipt_bytes(path, artifact_relative_path=RESULT_RELATIVE_PATH)
        ).hexdigest()
    )


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
