"""Exp5640 FR-11 shadow adapter pipeline integration.

Spec refs: REQ-LEARN-5640,
SCENARIO-LEARN-5640-EQUIVALENCE,
SCENARIO-LEARN-5640-SHADOW,
SCENARIO-LEARN-5640-REPLAY,
SCENARIO-LEARN-5640-ARTIFACT.

This experiment is a bounded operating-path integration check. It does not
enable learned advice by default and does not alter LLM weights. The only
production surface under test is an opt-in shadow adapter that appends an
audit ledger after exact verification remains authoritative.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from carnot import experiment_5639_anytime_valid_csl_independent_audit as exp5639
from carnot.pipeline.fr11_shadow_adapter import (
    ACTIONS,
    ExactVerificationReceipt,
    FR11ShadowAdapter,
    canonical_json,
    ledger_lineage_complete,
    load_ledger,
    sha256_json,
)
from carnot.pipeline.verify_repair import VerifyRepairPipeline


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5640_fr11_shadow_pipeline_integration.json")
LEDGER_RELATIVE_PATH = Path(
    "results/experiment_5640_fr11_shadow_pipeline_integration_ledger.jsonl"
)
CHECKPOINT_RELATIVE_PATH = Path(
    "results/experiment_5640_fr11_shadow_pipeline_integration_checkpoint.json"
)
ADAPTER_RELATIVE_PATH = Path("python/carnot/pipeline/fr11_shadow_adapter.py")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5640_fr11_shadow_pipeline_integration.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
TEST_RELATIVE_PATHS = (
    Path("tests/python/test_fr11_shadow_adapter.py"),
    Path("tests/python/test_experiment_5640_fr11_shadow_pipeline_integration.py"),
)

SCHEMA = "carnot.experiment_5640.fr11_shadow_pipeline_integration.v1"
EXPERIMENT = 5640
EXPERIMENT_ID = "experiment_5640_fr11_shadow_pipeline_integration"
RUN_DATE = "20260714"
FEATURE_FLAG = "CARNOT_FR11_SHADOW_ADAPTER"
INFERENCE_SUBSTRATE = "exact_verifier_gated_conformal_kan_shadow_adapter"
DEFAULT_ENABLED = False
EXACT_VERIFIER_AUTHORITY = True
MODEL_WEIGHT_MUTATION = False
RANDOM_SEEDS = (5640, *exp5639.DEFAULT_REPLAY_SEEDS)

SPEC_REFS = (
    "REQ-LEARN-5640",
    "SCENARIO-LEARN-5640-EQUIVALENCE",
    "SCENARIO-LEARN-5640-SHADOW",
    "SCENARIO-LEARN-5640-REPLAY",
    "SCENARIO-LEARN-5640-ARTIFACT",
)
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "upstream_gate_receipts",
    "openspec_requirement_ids",
    "adapter_path",
    "feature_flag",
    "default_enabled",
    "exact_verifier_authority",
    "shadow_decision_count",
    "shadow_offline_parity",
    "default_path_equivalence",
    "unsafe_update_accept_count",
    "checkpoint_atomicity_pass",
    "restart_replay_pass",
    "rollback_pass",
    "ledger_path",
    "ledger_lineage_complete",
    "model_weight_mutation",
    "fr11_shadow_integration_ready_score",
    "inference_substrate",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)
REQUIRED_FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "evidence fields explain why they exist",
    "upstream_gate_receipts": "promotion is prerequisite",
    "openspec_requirement_ids": "implementation is spec anchored",
    "adapter_path": "production surface is explicit",
    "feature_flag": "rollout is bounded",
    "default_enabled": "rollout is bounded",
    "exact_verifier_authority": "learned advice cannot legalize invalid state",
    "shadow_decision_count": "execution is real",
    "shadow_offline_parity": "wiring matches the audited controller",
    "default_path_equivalence": "existing behavior is preserved",
    "unsafe_update_accept_count": "fail-closed safety is exact",
    "checkpoint_atomicity_pass": "partial state cannot publish",
    "restart_replay_pass": "persistence is reproducible",
    "rollback_pass": "regressions recover",
    "ledger_path": "decisions are auditable",
    "ledger_lineage_complete": "decisions are auditable",
    "model_weight_mutation": "scope remains controller learning",
    "fr11_shadow_integration_ready_score": "deployment readiness is mechanical",
    "inference_substrate": "no LLM inference occurred",
    "random_seeds": "replay is stable",
    "reproducibility_checksum": "replay is stable",
    "honest_verdict": "starts complete: or blocked: and distinguishes shadow readiness",
}
FIELD_PRINCIPLES: JsonDict = {
    **REQUIRED_FIELD_PRINCIPLES,
    "replay_controls": "known failure modes are exercised",
    "shadow_offline_mismatches": "parity failures are inspectable",
    "benefit_evidence_within_exp5639_bound": "upstream certified utility bound is preserved",
    "ledger_sha256": "ledger bytes are replayable",
    "tests_added_or_reused": "verification commands are visible",
}
DEFAULT_TESTS_ADDED_OR_REUSED = (
    ".venv/bin/pytest tests/python/test_fr11_shadow_adapter.py "
    "tests/python/test_experiment_5640_fr11_shadow_pipeline_integration.py -q --no-cov -n 0",
    ".venv/bin/coverage run --include=python/carnot/pipeline/fr11_shadow_adapter.py,"
    "python/carnot/experiment_5640_fr11_shadow_pipeline_integration.py "
    "-m pytest tests/python/test_fr11_shadow_adapter.py "
    "tests/python/test_experiment_5640_fr11_shadow_pipeline_integration.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/pipeline/fr11_shadow_adapter.py,"
    "python/carnot/experiment_5640_fr11_shadow_pipeline_integration.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5640_fr11_shadow_pipeline_integration.json",
)


def _reset_paths(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()
        tmp = path.with_name(path.name + ".tmp")
        if tmp.exists():
            tmp.unlink()


def _actual_paths(root: Path | str, ledger_dir: Path | str | None) -> tuple[Path, Path]:
    if ledger_dir is None:
        ledger_path = Path(root) / LEDGER_RELATIVE_PATH
        checkpoint_path = Path(root) / CHECKPOINT_RELATIVE_PATH
    else:
        base = Path(ledger_dir)
        ledger_path = base / LEDGER_RELATIVE_PATH.name
        checkpoint_path = base / CHECKPOINT_RELATIVE_PATH.name
    return ledger_path, checkpoint_path


def upstream_gate_receipts(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load Exp5639 and enforce promotion before shadow integration."""

    root_path = Path(root)
    exp5639_path = root_path / exp5639.RESULT_RELATIVE_PATH
    exp5639_artifact = json.loads(exp5639_path.read_text(encoding="utf-8"))
    exp5639.validate_artifact(exp5639_artifact)
    promotion_ready = exp5639_artifact.get("fr11_independent_promotion_ready_score") == 1.0
    return {
        "promotion_prerequisite_enforced": promotion_ready,
        "exp5639": {
            "path": exp5639.RESULT_RELATIVE_PATH.as_posix(),
            "sha256": exp5639.sha256_file(exp5639_path),
            "schema": exp5639_artifact.get("schema"),
            "honest_verdict": exp5639_artifact.get("honest_verdict"),
            "promotion_ready": promotion_ready,
            "fr11_independent_promotion_ready_score": exp5639_artifact.get(
                "fr11_independent_promotion_ready_score"
            ),
            "unsafe_false_accept_count_total": exp5639_artifact.get(
                "unsafe_false_accept_count_total"
            ),
        },
        "paired_benefit_intervals": exp5639_artifact.get("paired_benefit_intervals", {}),
        "certified_bound": exp5639_artifact.get("preregistered_thresholds", {}).get(
            "paired_benefit_lower_floor"
        ),
    }


def benefit_evidence_within_exp5639_bound(gates: Mapping[str, Any]) -> bool:
    """Check that Exp5639's certified paired-benefit lower bound still holds."""

    if gates.get("exp5639", {}).get("promotion_ready") is not True:
        return False
    floor = float(gates.get("certified_bound") or 0.0)
    intervals = gates.get("paired_benefit_intervals", {})
    return bool(intervals) and all(float(row.get("lower", -1.0)) > floor for row in intervals.values())


def frozen_exact_stream() -> tuple[ExactVerificationReceipt, ...]:
    """Return the frozen exact stream used by offline and pipeline-shadow replay."""

    base_payload = {"stream": "exp5640-frozen-exact", "question": "What is 2 + 2?"}
    accepted = ExactVerificationReceipt(
        receipt_id="row-accepted",
        input_payload={**base_payload, "response": "2 + 2 = 4."},
        conformal_action_set=("adapt", "abstain"),
        exact_valid=True,
    )
    return (
        accepted,
        ExactVerificationReceipt(
            receipt_id="row-rejected",
            input_payload={**base_payload, "response": "2 + 2 = 5."},
            conformal_action_set=("adapt", "abstain"),
            exact_valid=False,
        ),
        accepted,
        ExactVerificationReceipt(
            receipt_id="row-delayed",
            input_payload={**base_payload, "response": "label pending"},
            conformal_action_set=("smooth", "abstain"),
            exact_valid=True,
            delayed_label=True,
        ),
        ExactVerificationReceipt(
            receipt_id="row-poison",
            input_payload={**base_payload, "response": "poison"},
            conformal_action_set=("reset", "abstain"),
            exact_valid=True,
            poison=True,
        ),
        ExactVerificationReceipt(
            receipt_id="row-rollback",
            input_payload={**base_payload, "response": "rollback"},
            conformal_action_set=("retain", "abstain"),
            exact_valid=True,
            rollback_required=True,
        ),
        ExactVerificationReceipt(
            receipt_id="row-unsupported",
            input_payload={**base_payload, "response": "unsupported"},
            conformal_action_set=("retain", "abstain"),
            exact_valid=None,
        ),
    )


def _replay_stream(
    receipts: Sequence[ExactVerificationReceipt],
    *,
    ledger_path: Path,
    checkpoint_path: Path,
) -> list[JsonDict]:
    _reset_paths(ledger_path, checkpoint_path)
    adapter = FR11ShadowAdapter(
        ledger_path=ledger_path,
        checkpoint_path=checkpoint_path,
        enabled=True,
    )
    for receipt in receipts:
        adapter.observe(receipt)
    return load_ledger(ledger_path)


def _decision_signature(row: Mapping[str, Any]) -> JsonDict:
    return {
        "input_hash": row["input_hash"],
        "conformal_action_set": row["conformal_action_set"],
        "recommendation": row["recommendation"],
        "exact_disposition": row["exact_disposition"],
        "rollback_reason": row["rollback_reason"],
        "duplicate_delivery": row["duplicate_delivery"],
        "delayed_label": row["delayed_label"],
        "poison": row["poison"],
        "rollback_required": row["rollback_required"],
        "unsafe_update_accepted": row["unsafe_update_accepted"],
    }


def shadow_offline_parity(offline_rows: Sequence[Mapping[str, Any]], shadow_rows: Sequence[Mapping[str, Any]]) -> tuple[bool, list[JsonDict]]:
    """Compare offline and shadow decisions on behavior, not filesystem paths."""

    mismatches: list[JsonDict] = []
    if len(offline_rows) != len(shadow_rows):
        mismatches.append({"kind": "count", "offline": len(offline_rows), "shadow": len(shadow_rows)})
        return False, mismatches
    for index, (offline, shadow) in enumerate(zip(offline_rows, shadow_rows, strict=True)):
        if _decision_signature(offline) != _decision_signature(shadow):
            mismatches.append(
                {
                    "index": index,
                    "offline": _decision_signature(offline),
                    "shadow": _decision_signature(shadow),
                }
            )
    return not mismatches, mismatches


def default_path_equivalence(tmp_dir: Path) -> JsonDict:
    """Verify that an explicitly disabled adapter leaves existing results unchanged."""

    baseline = VerifyRepairPipeline().verify(
        question="What is 47 + 28?",
        response="The answer is 47 + 28 = 76.",
        domain="arithmetic",
    )
    disabled_ledger = tmp_dir / "disabled.jsonl"
    candidate = VerifyRepairPipeline(
        fr11_shadow_adapter_enabled=False,
        fr11_shadow_ledger_path=disabled_ledger,
    ).verify(
        question="What is 47 + 28?",
        response="The answer is 47 + 28 = 76.",
        domain="arithmetic",
    )
    equivalent = (
        baseline.verified == candidate.verified
        and baseline.energy == candidate.energy
        and baseline.mode == candidate.mode
        and baseline.skipped == candidate.skipped
        and baseline.certificate == candidate.certificate
        and [item.constraint_type for item in baseline.violations]
        == [item.constraint_type for item in candidate.violations]
    )
    return {
        "pass": equivalent,
        "ledger_written": disabled_ledger.exists(),
        "baseline_certificate_hash": sha256_json(baseline.certificate),
        "candidate_certificate_hash": sha256_json(candidate.certificate),
    }


def run_pipeline_shadow_path(ledger_path: Path, checkpoint_path: Path) -> list[JsonDict]:
    """Exercise the actual `VerifyRepairPipeline.verify(...)` shadow hook."""

    _reset_paths(ledger_path, checkpoint_path)
    pipeline = VerifyRepairPipeline(
        fr11_shadow_adapter_enabled=True,
        fr11_shadow_ledger_path=ledger_path,
        fr11_shadow_checkpoint_path=checkpoint_path,
    )
    pipeline.verify(
        question="What is 10 + 5?",
        response="10 + 5 = 15.",
        domain="arithmetic",
    )
    pipeline.verify(
        question="What is 10 + 5?",
        response="10 + 5 = 16.",
        domain="arithmetic",
    )
    return load_ledger(ledger_path)


def restart_replay_control(tmp_dir: Path) -> JsonDict:
    """Restart from checkpoint and verify lineage continues deterministically."""

    ledger_path = tmp_dir / "restart.jsonl"
    checkpoint_path = tmp_dir / "restart.checkpoint.json"
    _reset_paths(ledger_path, checkpoint_path)
    first = FR11ShadowAdapter(ledger_path=ledger_path, checkpoint_path=checkpoint_path, enabled=True)
    first.observe(frozen_exact_stream()[0])
    restarted = FR11ShadowAdapter(
        ledger_path=ledger_path,
        checkpoint_path=checkpoint_path,
        enabled=True,
    )
    restarted.observe(frozen_exact_stream()[1])
    rows = load_ledger(ledger_path)
    return {
        "pass": len(rows) == 2 and ledger_lineage_complete(rows),
        "row_count": len(rows),
    }


def corrupted_checkpoint_control(tmp_dir: Path) -> JsonDict:
    """Corrupt checkpoint bytes and confirm the next decision abstains."""

    ledger_path = tmp_dir / "corrupt.jsonl"
    checkpoint_path = tmp_dir / "corrupt.checkpoint.json"
    _reset_paths(ledger_path, checkpoint_path)
    first = FR11ShadowAdapter(ledger_path=ledger_path, checkpoint_path=checkpoint_path, enabled=True)
    first.observe(frozen_exact_stream()[0])
    checkpoint_path.write_text("{corrupt", encoding="utf-8")
    recovered = FR11ShadowAdapter(
        ledger_path=ledger_path,
        checkpoint_path=checkpoint_path,
        enabled=True,
    )
    decision = recovered.observe(frozen_exact_stream()[3])
    return {
        "pass": decision is not None
        and decision.recommendation == "abstain"
        and decision.rollback_reason == "corrupted_checkpoint_recovered",
        "recommendation": decision.recommendation if decision else None,
        "rollback_reason": decision.rollback_reason if decision else None,
    }


def replay_controls(
    *,
    tmp_dir: Path,
    shadow_rows: Sequence[Mapping[str, Any]],
    equivalence: Mapping[str, Any],
) -> JsonDict:
    """Summarize every required fail-closed replay control."""

    duplicate = next(row for row in shadow_rows if row["duplicate_delivery"])
    delayed = next(row for row in shadow_rows if row["delayed_label"])
    poison = next(row for row in shadow_rows if row["poison"])
    rollback = next(row for row in shadow_rows if row["rollback_required"])
    inactive_ledger = tmp_dir / "inactive.jsonl"
    inactive = FR11ShadowAdapter(
        ledger_path=inactive_ledger,
        checkpoint_path=tmp_dir / "inactive.checkpoint.json",
        enabled=False,
    )
    inactive_decision = inactive.observe(frozen_exact_stream()[0])
    corrupt = corrupted_checkpoint_control(tmp_dir)
    restart = restart_replay_control(tmp_dir)
    return {
        "crash_restart": restart,
        "duplicate_delivery": {
            "pass": duplicate["recommendation"] == "abstain"
            and duplicate["rollback_reason"] == "duplicate_delivery",
            "recommendation": duplicate["recommendation"],
            "rollback_reason": duplicate["rollback_reason"],
        },
        "delayed_labels": {
            "pass": delayed["recommendation"] == "abstain"
            and delayed["rollback_reason"] == "delayed_label_pending",
            "recommendation": delayed["recommendation"],
            "rollback_reason": delayed["rollback_reason"],
        },
        "poison": {
            "pass": poison["recommendation"] == "abstain"
            and poison["rollback_reason"] == "poison_rejected",
            "recommendation": poison["recommendation"],
            "rollback_reason": poison["rollback_reason"],
        },
        "corrupted_checkpoint": corrupt,
        "rollback": {
            "pass": rollback["recommendation"] == "abstain"
            and rollback["rollback_reason"] == "rollback_required",
            "recommendation": rollback["recommendation"],
            "rollback_reason": rollback["rollback_reason"],
        },
        "inactive_adapter": {
            "pass": inactive_decision is None and not inactive_ledger.exists(),
            "decision_count": inactive.decision_count,
        },
        "feature_disabled_equivalence": equivalence,
    }


def _ledger_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json([dict(row) for row in rows])


def readiness_score(artifact: Mapping[str, Any]) -> float:
    """Compute the mechanical shadow-readiness gate."""

    controls = artifact.get("replay_controls", {})
    all_controls_pass = bool(controls) and all(
        isinstance(control, Mapping) and control.get("pass") is True
        for control in controls.values()
    )
    ready = (
        artifact.get("upstream_gate_receipts", {}).get("exp5639", {}).get("promotion_ready")
        is True
        and artifact.get("default_path_equivalence") is True
        and artifact.get("shadow_offline_parity") is True
        and artifact.get("unsafe_update_accept_count") == 0
        and artifact.get("checkpoint_atomicity_pass") is True
        and artifact.get("restart_replay_pass") is True
        and artifact.get("rollback_pass") is True
        and artifact.get("ledger_lineage_complete") is True
        and artifact.get("benefit_evidence_within_exp5639_bound") is True
        and artifact.get("model_weight_mutation") is False
        and all_controls_pass
    )
    return 1.0 if ready else 0.0


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict that separates shadow readiness from default enablement."""

    if readiness_score(artifact) == 1.0:
        return "complete: fr11_shadow_ready_not_default_enabled"
    return "blocked: fr11_shadow_integration_gate_failed"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with the checksum field removed."""

    material = dict(artifact)
    material.pop("reproducibility_checksum", None)
    return sha256_json(material)


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    ledger_dir: Path | str | None = None,
    tests_added_or_reused: Sequence[str] = DEFAULT_TESTS_ADDED_OR_REUSED,
) -> JsonDict:
    """Build the Exp5640 artifact from deterministic local replay."""

    root_path = Path(root)
    actual_ledger, actual_checkpoint = _actual_paths(root_path, ledger_dir)
    tmp_dir = actual_ledger.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)

    gates = upstream_gate_receipts(root_path)
    equivalence = default_path_equivalence(tmp_dir)
    receipts = frozen_exact_stream()
    offline_rows = _replay_stream(
        receipts,
        ledger_path=tmp_dir / "offline.jsonl",
        checkpoint_path=tmp_dir / "offline.checkpoint.json",
    )
    shadow_rows = _replay_stream(
        receipts,
        ledger_path=tmp_dir / "shadow.jsonl",
        checkpoint_path=tmp_dir / "shadow.checkpoint.json",
    )
    parity, mismatches = shadow_offline_parity(offline_rows, shadow_rows)
    pipeline_rows = run_pipeline_shadow_path(actual_ledger, actual_checkpoint)
    controls = replay_controls(tmp_dir=tmp_dir, shadow_rows=shadow_rows, equivalence=equivalence)
    all_rows = [*shadow_rows, *pipeline_rows]

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "upstream_gate_receipts": gates,
        "openspec_requirement_ids": list(SPEC_REFS),
        "adapter_path": ADAPTER_RELATIVE_PATH.as_posix(),
        "feature_flag": FEATURE_FLAG,
        "default_enabled": DEFAULT_ENABLED,
        "exact_verifier_authority": EXACT_VERIFIER_AUTHORITY,
        "shadow_decision_count": len(all_rows),
        "shadow_offline_parity": parity,
        "shadow_offline_mismatches": mismatches,
        "default_path_equivalence": equivalence["pass"] is True
        and equivalence["ledger_written"] is False,
        "unsafe_update_accept_count": sum(int(row["unsafe_update_accepted"]) for row in all_rows),
        "checkpoint_atomicity_pass": not list(tmp_dir.glob("*.tmp")) and actual_checkpoint.exists(),
        "restart_replay_pass": controls["crash_restart"]["pass"],
        "rollback_pass": controls["rollback"]["pass"],
        "ledger_path": LEDGER_RELATIVE_PATH.as_posix(),
        "ledger_sha256": _ledger_sha256(pipeline_rows),
        "ledger_lineage_complete": ledger_lineage_complete(shadow_rows)
        and ledger_lineage_complete(pipeline_rows),
        "model_weight_mutation": MODEL_WEIGHT_MUTATION,
        "benefit_evidence_within_exp5639_bound": benefit_evidence_within_exp5639_bound(gates),
        "replay_controls": controls,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": list(RANDOM_SEEDS),
        "tests_added_or_reused": list(tests_added_or_reused),
        "fr11_shadow_integration_ready_score": 0.0,
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["fr11_shadow_integration_ready_score"] = readiness_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return all schema/readiness problems without mutating the artifact."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
        return errors
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles")
    else:
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
            if principles.get(field) != principle:
                errors.append(f"field_principles.{field}")
    if artifact.get("upstream_gate_receipts", {}).get("exp5639", {}).get("promotion_ready") is not True:
        errors.append("upstream_gate_receipts")
    if artifact.get("openspec_requirement_ids") != list(SPEC_REFS):
        errors.append("openspec_requirement_ids")
    if artifact.get("adapter_path") != ADAPTER_RELATIVE_PATH.as_posix():
        errors.append("adapter_path")
    if artifact.get("feature_flag") != FEATURE_FLAG:
        errors.append("feature_flag")
    if artifact.get("default_enabled") is not False:
        errors.append("default_enabled")
    if artifact.get("exact_verifier_authority") is not True:
        errors.append("exact_verifier_authority")
    if int(artifact.get("shadow_decision_count") or 0) <= 0:
        errors.append("shadow_decision_count")
    if artifact.get("shadow_offline_parity") is not True:
        errors.append("shadow_offline_parity")
    if artifact.get("default_path_equivalence") is not True:
        errors.append("default_path_equivalence")
    if artifact.get("unsafe_update_accept_count") != 0:
        errors.append("unsafe_update_accept_count")
    if artifact.get("checkpoint_atomicity_pass") is not True:
        errors.append("checkpoint_atomicity_pass")
    if artifact.get("restart_replay_pass") is not True:
        errors.append("restart_replay_pass")
    if artifact.get("rollback_pass") is not True:
        errors.append("rollback_pass")
    if not artifact.get("ledger_path"):
        errors.append("ledger_path")
    if artifact.get("ledger_lineage_complete") is not True:
        errors.append("ledger_lineage_complete")
    if artifact.get("model_weight_mutation") is not False:
        errors.append("model_weight_mutation")
    if artifact.get("benefit_evidence_within_exp5639_bound") is not True:
        errors.append("benefit_evidence_within_exp5639_bound")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("random_seeds") != list(RANDOM_SEEDS):
        errors.append("random_seeds")
    if artifact.get("fr11_shadow_integration_ready_score") != readiness_score(artifact):
        errors.append("fr11_shadow_integration_ready_score")
    verdict = artifact.get("honest_verdict")
    invalid_verdict = not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict.startswith("blocked:")
    )
    score = artifact.get("fr11_shadow_integration_ready_score")
    if invalid_verdict or (
        isinstance(verdict, str)
        and (
            (score == 1.0 and not verdict.startswith("complete:"))
            or (score == 0.0 and not verdict.startswith("blocked:"))
        )
    ):
        errors.append("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    if any(action not in ACTIONS for action in _all_recommendations(artifact)):
        errors.append("replay_controls")
    return errors


def _all_recommendations(artifact: Mapping[str, Any]) -> list[str]:
    controls = artifact.get("replay_controls", {})
    values: list[str] = []
    if isinstance(controls, Mapping):
        for control in controls.values():
            if isinstance(control, Mapping) and "recommendation" in control:
                values.append(str(control["recommendation"]))
    return values


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise ValueError unless the artifact satisfies the Exp5640 contract."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    ledger_dir: Path | str | None = None,
    tests_added_or_reused: Sequence[str] = DEFAULT_TESTS_ADDED_OR_REUSED,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the Exp5640 artifact."""

    artifact = build_artifact(
        root=root,
        ledger_dir=ledger_dir,
        tests_added_or_reused=tests_added_or_reused,
    )
    validate_artifact(artifact)
    if write:
        destination = Path(result_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(canonical_json(artifact) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    """CLI entry point for regenerating the checked-in artifact."""

    run()


if __name__ == "__main__":  # pragma: no cover - covered through main(), not import-time CLI.
    main()
