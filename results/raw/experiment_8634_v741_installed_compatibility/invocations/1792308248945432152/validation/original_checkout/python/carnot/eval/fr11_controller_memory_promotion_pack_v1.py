"""Exp 3186 FR-11 controller-memory promotion pack v1.

Spec refs: REQ-LEARN-3186, SCENARIO-LEARN-3186,
SCENARIO-LEARN-3186-BLOCKED.

This module packages the controller-memory update that Exp 3172 already proved
against replay rows.  The important boundary is that the learned object is an
auditable controller rule: exact row-id action overrides plus replay gates.  It
does not finetune a model, mutate hidden state, train KAN weights, or make live
model calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
ARTIFACT = "experiment_3186_fr11_controller_memory_promotion_pack_v1"
SCHEMA = "carnot.fr11.controller_memory_promotion_pack.v1"
OUTPUT_REL_PATH = Path("results/experiment_3186_fr11_controller_memory_promotion_pack_v1.json")
EXP3171_REL_PATH = Path("results/experiment_3171_fr11_ledger_counterexample_isolation_v1.json")
EXP3172_REL_PATH = Path("results/experiment_3172_fr11_nonforgetting_self_learning_pilot_v2.json")
EXP3175_REL_PATH = Path("results/experiment_3175_cross_corpus_matrix_v28.json")
EXP3176_REL_PATH = Path("results/experiment_3176_capstone_v294.json")
EXP3187_REL_PATH = Path("results/experiment_3187_fr11_cross_environment_drift_replay_v1.json")
SPEC_REL_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_REL_PATH = Path("python/carnot/eval/fr11_controller_memory_promotion_pack_v1.py")
TEST_REL_PATH = Path(
    "tests/python/test_experiment_3186_fr11_controller_memory_promotion_pack_v1.py"
)

UPDATE_ID = "fr11-controller-memory-exp3172-row-exact-v1"
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
REQUIRED_ARTIFACT_FIELDS = {
    "fr11_controller_memory_promotion_pack_v1_ready",
    "continuous_self_learning_task",
    "learning_tier",
    "no_model_weight_update_claimed",
    "source_update_artifact",
    "promotion_manifest",
    "replay_requirements",
    "rollback_policy",
    "promotion_allowed",
    "inference_substrate",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest -o addopts='' tests/python/test_experiment_3186_fr11_controller_memory_promotion_pack_v1.py -q",
    ".venv/bin/coverage run --source=python/carnot/eval/fr11_controller_memory_promotion_pack_v1.py -m pytest -o addopts='' tests/python/test_experiment_3186_fr11_controller_memory_promotion_pack_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/fr11_controller_memory_promotion_pack_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_3186_fr11_controller_memory_promotion_pack_v1.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md"), True),
    ("codex_repo_workflow", Path("CODEX.md"), True),
    ("research_program", Path("research-program.md"), True),
    ("self_learning_openspec", SPEC_REL_PATH, True),
    ("exp3171_counterexample_isolation", EXP3171_REL_PATH, True),
    ("exp3172_controller_memory_update", EXP3172_REL_PATH, True),
    ("exp3175_matrix_v28", EXP3175_REL_PATH, False),
    ("exp3176_capstone_v294", EXP3176_REL_PATH, False),
    ("exp3186_module", MODULE_REL_PATH, False),
    ("exp3186_tests", TEST_REL_PATH, False),
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object, returning empty evidence when the path is unusable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def load_sources(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load the checked-in .294 FR-11 evidence used by the promotion pack."""

    root_path = Path(root)
    return {
        "exp3171": read_json_object(root_path / EXP3171_REL_PATH),
        "exp3172": read_json_object(root_path / EXP3172_REL_PATH),
        "exp3175": read_json_object(root_path / EXP3175_REL_PATH),
        "exp3176": read_json_object(root_path / EXP3176_REL_PATH),
        "research_program_text": read_text(root_path / "research-program.md"),
    }


def read_text(path: Path) -> str:
    """Return source text used for classification, or empty text if absent."""

    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build a schema-complete Exp 3186 artifact from checked-in evidence."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = load_sources(root_path)
    blocker = source_blocker(sources)
    if blocker:
        artifact = blocked_artifact(root_path, blocker, start, now_s, tests_run, sources)
        validate_artifact(artifact)
        return artifact
    artifact = build_promotion_pack(root_path, sources, start, now_s, tests_run)
    validate_artifact(artifact)
    return artifact


def build_promotion_pack(
    root: Path | str,
    sources: Mapping[str, Any],
    started_s: float = 0.0,
    now_s: float | None = 0.0,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-LEARN-3186-2/3/4/5: package the Exp 3172 controller rule."""

    root_path = Path(root)
    exp3172 = dict(sources["exp3172"])
    manifest = promotion_manifest(exp3172, sources)
    rollback = rollback_policy(active_update=True)
    requirements = replay_requirements()
    return {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_controller_memory_promotion_pack_v1_ready": True,
        "continuous_self_learning_task": True,
        "learning_tier": learning_tier(str(sources.get("research_program_text") or "")),
        "no_model_weight_update_claimed": True,
        "source_update_artifact": EXP3172_REL_PATH.as_posix(),
        "promotion_manifest": manifest,
        "replay_requirements": requirements,
        "rollback_policy": rollback,
        "promotion_allowed": True,
        "inference_substrate": inference_substrate(),
        "source_artifacts": source_artifacts(root_path),
        "precondition_checks": precondition_checks(sources),
        "drift_replay_contract": drift_replay_contract(requirements, rollback),
        "ops_reconciliation": ops_reconciliation(),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(started_s, now_s),
        "honest_verdict": (
            "complete: fr11 controller-memory promotion pack v1 ready; "
            "promotion_allowed=true; learning_tier=controller_memory_tier2; "
            "no model-weight update claimed"
        ),
    }


def source_blocker(sources: Mapping[str, Any]) -> str:
    """REQ-LEARN-3186-1: fail closed unless Exp 3172 is promotable and bounded."""

    exp3172 = sources.get("exp3172", {})
    if not isinstance(exp3172, Mapping) or not exp3172:
        return "exp3172_missing_or_not_ready"
    if exp3172.get("fr11_nonforgetting_self_learning_pilot_v2_ready") is not True:
        return "exp3172_missing_or_not_ready"
    if exp3172.get("model_weight_update_claimed") is not False:
        return "exp3172_model_weight_update_claimed"
    if exp3172.get("promotion_allowed") is not True:
        return "exp3172_promotion_not_allowed"
    if int(exp3172.get("negative_control_regression_count") or 0) != 0:
        return "exp3172_negative_control_regression_present"
    if not updated_rows(exp3172.get("controller_memory_update", {})):
        return "exp3172_controller_memory_update_missing"
    return ""


def blocked_artifact(
    root: Path,
    blocker: str,
    started_s: float,
    now_s: float | None,
    tests_run: Sequence[str] | None,
    sources: Mapping[str, Any],
) -> JsonDict:
    """Return a blocked but schema-complete promotion pack."""

    return {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_controller_memory_promotion_pack_v1_ready": False,
        "continuous_self_learning_task": True,
        "learning_tier": learning_tier(str(sources.get("research_program_text") or "")),
        "no_model_weight_update_claimed": True,
        "source_update_artifact": EXP3172_REL_PATH.as_posix(),
        "promotion_manifest": blocked_manifest(blocker),
        "replay_requirements": replay_requirements(blocked=True),
        "rollback_policy": rollback_policy(active_update=False),
        "promotion_allowed": False,
        "inference_substrate": inference_substrate(mode="blocked_precondition_check"),
        "source_artifacts": source_artifacts(root),
        "precondition_checks": precondition_checks(sources) | {"blocked_reason": blocker},
        "drift_replay_contract": {
            "target_artifact": EXP3187_REL_PATH.as_posix(),
            "ready_for_drift_replay": False,
            "blocked_reason": blocker,
        },
        "ops_reconciliation": ops_reconciliation(status="blocked_until_source_promotable"),
        "blocked_reason": blocker,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(started_s, now_s),
        "honest_verdict": f"blocked_precondition_failed: {blocker}",
    }


def promotion_manifest(exp3172: Mapping[str, Any], sources: Mapping[str, Any]) -> JsonDict:
    """Build the exact activation and rollback metadata for the learned rule."""

    update = dict(exp3172.get("controller_memory_update", {}))
    rows = updated_rows(update)
    overrides = row_overrides(update)
    families = sorted({str(row.get("fixture_family") or "unknown") for row in rows})
    allowed_row_ids = sorted(overrides)
    return {
        "update_id": UPDATE_ID,
        "update_type": "controller_memory_exact_row_action_override",
        "promotion_decision": "promote_controller_memory_only",
        "source_update_artifact": EXP3172_REL_PATH.as_posix(),
        "source_counterexample_artifact": EXP3171_REL_PATH.as_posix(),
        "source_counterexample_families": families,
        "activation_predicate": {
            "mode": "exact_row_id_controller_memory_override",
            "allowed_row_ids": allowed_row_ids,
            "row_action_overrides": {row_id: overrides[row_id] for row_id in allowed_row_ids},
            "requires_exact_authority_consensus": True,
            "requires_exp3187_drift_replay_before_broadening": True,
            "scope": "training_row_ids_only",
        },
        "rollback_predicate": rollback_predicate(),
        "replay_requirements": replay_requirements(),
        "owner_artifact_paths": owner_artifact_paths(),
        "evidence": evidence_summary(exp3172),
        "learning_tier": learning_tier(str(sources.get("research_program_text") or "")),
    }


def blocked_manifest(blocker: str) -> JsonDict:
    """Return manifest metadata that cannot activate a controller update."""

    return {
        "update_id": UPDATE_ID,
        "update_type": "controller_memory_exact_row_action_override",
        "promotion_decision": "blocked",
        "blocked_reason": blocker,
        "activation_predicate": {
            "mode": "blocked_precondition_check",
            "allowed_row_ids": [],
            "row_action_overrides": {},
            "requires_exact_authority_consensus": True,
            "scope": "no_active_update",
        },
        "rollback_predicate": rollback_predicate(),
        "replay_requirements": replay_requirements(blocked=True),
        "owner_artifact_paths": owner_artifact_paths(),
        "evidence": {
            "before_ledger_consistency_rate": 0.0,
            "after_ledger_consistency_rate": 0.0,
            "heldout_consistency_rate": 0.0,
            "negative_control_regression_count": 0,
            "updated_row_count": 0,
        },
    }


def evidence_summary(exp3172: Mapping[str, Any]) -> JsonDict:
    """Extract the promotion evidence that came from Exp 3172."""

    update = exp3172.get("controller_memory_update", {})
    rows = updated_rows(update if isinstance(update, Mapping) else {})
    return {
        "before_ledger_consistency_rate": round_float(
            float(exp3172.get("before_ledger_consistency_rate") or 0.0)
        ),
        "after_ledger_consistency_rate": round_float(
            float(exp3172.get("after_ledger_consistency_rate") or 0.0)
        ),
        "heldout_consistency_rate": round_float(
            float(exp3172.get("heldout_consistency_rate") or 0.0)
        ),
        "negative_control_regression_count": int(
            exp3172.get("negative_control_regression_count") or 0
        ),
        "negative_control_regressions": list(exp3172.get("negative_control_regressions") or []),
        "nonforgetting_passed": exp3172.get("nonforgetting_passed") is True,
        "updated_row_count": len(rows),
        "updated_rows": rows,
    }


def updated_rows(update: Mapping[str, Any]) -> list[JsonDict]:
    """Return copied controller-memory update rows."""

    rows = update.get("updated_rows", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def row_overrides(update: Mapping[str, Any]) -> dict[str, str]:
    """Return exact row-id action overrides from the controller-memory update."""

    overrides = update.get("row_action_overrides", {})
    if not isinstance(overrides, Mapping):
        return {}
    return {str(row_id): normalize_action(action) for row_id, action in overrides.items()}


def replay_requirements(blocked: bool = False) -> list[JsonDict]:
    """REQ-LEARN-3186-3: enumerate replay gates that guard the learned memory."""

    state = "blocked_until_promotion_pack_ready" if blocked else "required"
    return [
        {
            "id": "exp3172_training_and_heldout_replay",
            "source_artifact": EXP3172_REL_PATH.as_posix(),
            "gate": "after_ledger_consistency_rate == 1.0 and heldout_consistency_rate == 1.0",
            "status": state,
        },
        {
            "id": "negative_control_replay",
            "source_artifact": EXP3172_REL_PATH.as_posix(),
            "gate": "negative_control_regression_count == 0",
            "status": state,
        },
        {
            "id": "exp3187_cross_environment_drift_replay",
            "source_artifact": EXP3187_REL_PATH.as_posix(),
            "gate": "cross_environment_drift_failure_count == 0",
            "status": state,
        },
        {
            "id": "ops_documentation_reconciliation",
            "source_artifact": "ops/status.md",
            "gate": "conductor reconciliation records controller-memory-only promotion",
            "status": state,
        },
    ]


def rollback_policy(active_update: bool) -> JsonDict:
    """REQ-LEARN-3186-5: define reversible promotion and monitoring triggers."""

    return {
        "rollback_action": "remove_exact_row_overrides" if active_update else "no_active_update",
        "owner_update_id": UPDATE_ID,
        "triggers": [
            {
                "trigger": "negative_control_regression",
                "predicate": "negative_control_regression_count > 0",
            },
            {
                "trigger": "stale_ledger_evidence",
                "predicate": "source_artifact_sha256 changes without replay refresh",
            },
            {
                "trigger": "drift_replay_failure",
                "predicate": "exp3187 reports any cross-environment consistency regression",
            },
            {
                "trigger": "exact_authority_conflict",
                "predicate": "exact authority rejects an activated override action",
            },
        ],
        "monitoring_plan": {
            "check_frequency": "every_exp3187_or_matrix_replay",
            "metrics": [
                "after_ledger_consistency_rate",
                "heldout_consistency_rate",
                "negative_control_regression_count",
                "cross_environment_drift_failure_count",
                "source_artifact_sha256",
            ],
        },
    }


def rollback_predicate() -> JsonDict:
    """Expose the compact predicate used inside the promotion manifest."""

    return {
        "rollback_if_any": [
            "negative_control_regression_count > 0",
            "heldout_consistency_rate < 1.0",
            "cross_environment_drift_failure_count > 0",
            "exact_authority_conflict_count > 0",
            "source_ledger_stale == true",
        ]
    }


def owner_artifact_paths() -> JsonDict:
    """Return the owned files that operators and Exp 3187 need to inspect."""

    return {
        "source_update_artifact": EXP3172_REL_PATH.as_posix(),
        "source_counterexample_artifact": EXP3171_REL_PATH.as_posix(),
        "promotion_pack_artifact": OUTPUT_REL_PATH.as_posix(),
        "drift_replay_artifact": EXP3187_REL_PATH.as_posix(),
        "module": MODULE_REL_PATH.as_posix(),
        "tests": TEST_REL_PATH.as_posix(),
        "spec": SPEC_REL_PATH.as_posix(),
        "ops_status": "ops/status.md",
        "ops_changelog": "ops/changelog.md",
    }


def learning_tier(research_program_text: str) -> str:
    """REQ-LEARN-3186-4: classify the update using research-program.md."""

    text = research_program_text.lower()
    if "tier 2" in text and "constraint memory" in text:
        return "Tier 2: Constraint Memory / Trace2Skill controller-memory learning"
    return "Tier 2: controller-memory learning (research-program.md tier text unavailable)"


def inference_substrate(mode: str = "controller_memory_promotion_pack_replay") -> JsonDict:
    """REQ-LEARN-3186-6: declare that this is aggregation and replay only."""

    return {
        "mode": mode,
        "aggregation_and_replay_only": True,
        "controller_memory_replay_only": True,
        "uses_checked_in_artifacts_only": True,
        "executes_live_model_inference": False,
        "fresh_live_inference_calls": 0,
        "model_weight_learning": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "base_model_weights_updated": False,
        "kan_model_weight_training": False,
        "hidden_state_mutation_claimed": False,
    }


def drift_replay_contract(
    requirements: Sequence[Mapping[str, Any]],
    rollback: Mapping[str, Any],
) -> JsonDict:
    """Write the handoff contract consumed by Exp 3187 drift replay."""

    return {
        "target_artifact": EXP3187_REL_PATH.as_posix(),
        "ready_for_drift_replay": True,
        "required_replay_ids": [str(req.get("id") or "") for req in requirements],
        "rollback_triggers": [str(row.get("trigger") or "") for row in rollback["triggers"]],
    }


def ops_reconciliation(status: str = "pending_conductor_reconciliation") -> JsonDict:
    """Record the docs that the conductor reconciler should update later."""

    return {
        "status": status,
        "paths": ["ops/status.md", "ops/changelog.md", "_bmad/traceability.md"],
        "claim_boundary": "controller_memory_learning_only_no_model_weight_update",
    }


def precondition_checks(sources: Mapping[str, Any]) -> JsonDict:
    """Expose the source gates that explain either readiness or blocking."""

    exp3172 = sources.get("exp3172", {})
    substrate = exp3172.get("inference_substrate", {}) if isinstance(exp3172, Mapping) else {}
    return {
        "exp3172_present": bool(exp3172),
        "exp3172_ready": isinstance(exp3172, Mapping)
        and exp3172.get("fr11_nonforgetting_self_learning_pilot_v2_ready") is True,
        "exp3172_promotion_allowed": isinstance(exp3172, Mapping)
        and exp3172.get("promotion_allowed") is True,
        "exp3172_model_weight_update_claimed": isinstance(exp3172, Mapping)
        and exp3172.get("model_weight_update_claimed") is True,
        "source_live_inference_calls": int(substrate.get("fresh_live_inference_calls") or 0)
        if isinstance(substrate, Mapping)
        else 0,
        "exp3171_present": bool(sources.get("exp3171")),
        "research_program_present": bool(sources.get("research_program_text")),
    }


def source_artifacts(root: Path) -> list[JsonDict]:
    """List source files and checksums for auditable promotion."""

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
    """Raise when the promotion pack overclaims or loses required gates."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("no_model_weight_update_claimed") is not True:
        raise ValueError("no_model_weight_update_claimed must be true")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        raise ValueError("inference_substrate must be a mapping")
    if int(substrate.get("fresh_live_inference_calls") or 0) != 0:
        raise ValueError("fresh_live_inference_calls must remain zero")
    if any(
        substrate.get(flag) is True
        for flag in (
            "executes_live_model_inference",
            "model_weight_mutation",
            "model_weight_training",
            "base_model_weights_updated",
            "kan_model_weight_training",
            "hidden_state_mutation_claimed",
        )
    ):
        raise ValueError("model and live-inference mutation flags must remain false")
    if artifact.get("fr11_controller_memory_promotion_pack_v1_ready") is not True:
        return
    manifest = artifact.get("promotion_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("promotion_manifest must be a mapping")
    evidence = manifest.get("evidence", {})
    if not isinstance(evidence, Mapping):
        raise ValueError("promotion evidence must be a mapping")
    if artifact.get("promotion_allowed") is not True:
        raise ValueError("promotion_allowed must be true when pack is ready")
    if (
        round_float(float(evidence.get("after_ledger_consistency_rate") or 0.0)) != 1.0
        or round_float(float(evidence.get("heldout_consistency_rate") or 0.0)) != 1.0
        or int(evidence.get("negative_control_regression_count") or 0) != 0
    ):
        raise ValueError("promotion_allowed requires perfect replay evidence")
    if any(
        row.get("required") and not row.get("exists")
        for row in artifact.get("source_artifacts", [])
        if isinstance(row, Mapping)
    ):
        raise ValueError("required source_artifacts must exist")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write deterministic Exp 3186 JSON."""

    root_path = Path(root)
    path = Path(output_path)
    output = path if path.is_absolute() else root_path / path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    write_json(output, artifact)
    return output


def normalize_action(value: Any) -> str:
    """Normalize small action tokens used by controller-memory overrides."""

    text = str(value or "").strip().lower().replace(" ", "_")
    return text or "unknown"


def round_float(value: float) -> float:
    """Round artifact floats to stable six-decimal precision."""

    return round(float(value), 6)


def duration(started_s: float, now_s: float | None) -> float:
    """Return stable elapsed seconds for artifact provenance."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round_float(max(0.0, end - started_s))


def sha256_file(path: Path) -> str | None:
    """Return a file checksum when the path exists."""

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
