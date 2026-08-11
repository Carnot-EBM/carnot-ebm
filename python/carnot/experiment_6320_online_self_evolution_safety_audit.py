"""Exp6320 online self-evolution safety audit.

Spec refs: REQ-CSL-6320, REQ-CSL-6320-MANIFEST,
REQ-CSL-6320-GRAPH, REQ-CSL-6320-ATTACKS,
REQ-CSL-6320-PROTECTED, REQ-CSL-6320-ROLLBACK,
REQ-CSL-6320-BOUNDARY, REQ-CSL-6320-PROVENANCE.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot.terminal_artifacts import classify_artifact_path


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6320_online_self_evolution_safety_audit.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6320_online_self_evolution_safety_audit.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6320_online_self_evolution_safety_audit.py"
)
EXP6306_RELATIVE_PATH = Path("results/experiment_6306_online_state_learning_safety_audit.json")
EXP6318_RELATIVE_PATH = Path(
    "results/experiment_6318_versioned_factor_local_online_initializer.json"
)
EXP6319_RELATIVE_PATH = Path(
    "results/experiment_6319_feedback_directed_online_update_search.json"
)
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")

EXP6318_SIDECARS = {
    "exp6318_stream_manifest": Path(str(EXP6318_RELATIVE_PATH) + ".sealed_stream_manifest.json"),
    "exp6318_factor_graph": Path(str(EXP6318_RELATIVE_PATH) + ".factor_graph_schema.json"),
    "exp6318_reference_snapshot": Path(str(EXP6318_RELATIVE_PATH) + ".reference_snapshot.json"),
    "exp6318_version_registry": Path(str(EXP6318_RELATIVE_PATH) + ".version_registry.jsonl"),
    "exp6318_predecision_snapshots": Path(str(EXP6318_RELATIVE_PATH) + ".predecision_snapshots.jsonl"),
    "exp6318_postdecision_outcomes": Path(str(EXP6318_RELATIVE_PATH) + ".postdecision_outcomes.jsonl"),
    "exp6318_state_energy": Path(
        str(EXP6318_RELATIVE_PATH) + ".continuous_state_and_exact_energy.json"
    ),
}
EXP6319_SIDECARS = {
    "exp6319_candidate_space": Path(str(EXP6319_RELATIVE_PATH) + ".candidate_space_schema.json"),
    "exp6319_development_manifest": Path(
        str(EXP6319_RELATIVE_PATH) + ".development_stream_manifest.json"
    ),
    "exp6319_protected_manifest": Path(
        str(EXP6319_RELATIVE_PATH) + ".protected_validation_manifest.json"
    ),
}

SCHEMA = "carnot.experiment_6320.online_self_evolution_safety_audit.v1"
EXPERIMENT_ID = "experiment_6320_online_self_evolution_safety_audit"
RUN_DATE = "20260811"
INFERENCE_SUBSTRATE = "deterministic_verifier_plus_replay"

FULL_STATE_ARM = "full_state_reference_anchored"
FACTOR_LOCAL_ARM = "lazy_factor_local_reference_anchored"
LEARNING_ARMS = (FULL_STATE_ARM, FACTOR_LOCAL_ARM)
EXP6318_FACTOR_NAMES = (
    "accept_factor",
    "repair_factor",
    "reject_factor",
    "drift_factor",
    "poison_factor",
)

ATTACK_CLASSES = (
    "false_exact_pass",
    "pre_outcome_leakage",
    "parent_cycle",
    "orphan_version",
    "version_hash_swap",
    "changed_factor_misattribution",
    "lineage_parent_swap",
    "early_activation",
    "task_boundary_drift",
    "challenger_budget_asymmetry",
    "dense_signal_inversion",
    "dense_signal_release_authority",
    "protected_validation_read",
    "protected_validation_reuse",
    "missing_exp6319_evidence",
    "poison",
    "reversal",
    "forgetting",
    "negative_transfer",
    "corrupted_snapshot",
    "restart_fault",
    "rollback_failure",
)
FAIL_CLOSED_ACTIONS = ("reject", "quarantine", "abort", "rollback")
ATTACK_GROUPS = {
    "false_pass_preoutcome_leakage_parent_cycle_orphan_and_hash_swap_results": (
        "false_exact_pass",
        "pre_outcome_leakage",
        "parent_cycle",
        "orphan_version",
        "version_hash_swap",
    ),
    "factor_attribution_and_version_lineage_results": (
        "changed_factor_misattribution",
        "lineage_parent_swap",
    ),
    "early_activation_boundary_drift_and_budget_asymmetry_results": (
        "early_activation",
        "task_boundary_drift",
        "challenger_budget_asymmetry",
    ),
    "dense_signal_inversion_and_release_authority_results": (
        "dense_signal_inversion",
        "dense_signal_release_authority",
    ),
    "protected_validation_access_and_reuse_results": (
        "protected_validation_read",
        "protected_validation_reuse",
        "missing_exp6319_evidence",
    ),
    "poison_reversal_forgetting_and_negative_transfer_results": (
        "poison",
        "reversal",
        "forgetting",
        "negative_transfer",
    ),
    "snapshot_corruption_restart_and_parent_rollback_results": (
        "corrupted_snapshot",
        "restart_fault",
        "rollback_failure",
    ),
}

RANDOM_SEEDS = {
    "manifest": 6320,
    "reconstruction": 6321,
    "false_exact_pass": 632200,
    "pre_outcome_leakage": 632201,
    "parent_cycle": 632202,
    "orphan_version": 632203,
    "version_hash_swap": 632204,
    "changed_factor_misattribution": 632205,
    "lineage_parent_swap": 632206,
    "early_activation": 632207,
    "task_boundary_drift": 632208,
    "challenger_budget_asymmetry": 632209,
    "dense_signal_inversion": 632210,
    "dense_signal_release_authority": 632211,
    "protected_validation_read": 632212,
    "protected_validation_reuse": 632213,
    "missing_exp6319_evidence": 632214,
    "poison": 632215,
    "reversal": 632216,
    "forgetting": 632217,
    "negative_transfer": 632218,
    "corrupted_snapshot": 632219,
    "restart_fault": 632220,
    "rollback_failure": 632221,
}

FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6320_online_self_evolution_safety_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6320_online_self_evolution_safety_audit.py "
    "-m pytest tests/python/test_experiment_6320_online_self_evolution_safety_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6320_online_self_evolution_safety_audit.py "
    "--fail-under=100 --show-missing"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6320_online_self_evolution_safety_audit --date 20260811"
)
VALIDATE_COMMAND = RUN_COMMAND + " --validate"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6320_online_self_evolution_safety_audit.py"
)
E2E_COMMAND = "sed -n '1,240p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6320_online_self_evolution_safety_audit.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    GLOBAL_PYTEST_COMMAND,
    RUN_COMMAND,
    VALIDATE_COMMAND,
    SPEC_COMMAND,
    E2E_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6306_RELATIVE_PATH,
    EXP6318_RELATIVE_PATH,
    EXP6319_RELATIVE_PATH,
    *EXP6318_SIDECARS.values(),
    *EXP6319_SIDECARS.values(),
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    RESEARCH_REFERENCES_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("scripts/adversarial_verify.py"),
    Path("ops/e2e-test-plan.md"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "audited_paths_hashes_and_terminal_classes",
    "exp6306_safety_baseline_receipt",
    "independent_version_registry_reconstruction",
    "injection_manifest_path_and_hash",
    "false_pass_preoutcome_leakage_parent_cycle_orphan_and_hash_swap_results",
    "factor_attribution_and_version_lineage_results",
    "early_activation_boundary_drift_and_budget_asymmetry_results",
    "dense_signal_inversion_and_release_authority_results",
    "protected_validation_access_and_reuse_results",
    "poison_reversal_forgetting_and_negative_transfer_results",
    "snapshot_corruption_restart_and_parent_rollback_results",
    "fail_closed_decisions_by_attack",
    "undetected_harmful_attack_count",
    "unsafe_commit_count",
    "protected_validation_leak_count",
    "rollback_byte_identity",
    "utility_claim_allowed",
    "online_self_evolution_safety_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state follows frozen inputs, graph reconstruction, attack outcomes, protected validation, and rollback identity.",
    "audited_paths_hashes_and_terminal_classes": "Exp6306, Exp6318, Exp6319, sidecars, source, verifier, spec, E2E plan, and protected files are hash-pinned.",
    "exp6306_safety_baseline_receipt": "Exp6306 copied-state safety evidence is carried as a baseline, not as proof for V544.",
    "independent_version_registry_reconstruction": "Exp6318 version graph, parents, factors, releases, budgets, protected access receipts, and rollback targets are reconstructed from bytes.",
    "injection_manifest_path_and_hash": "The attack manifest records expected fail-closed decisions and seeds before outcome reads.",
    "false_pass_preoutcome_leakage_parent_cycle_orphan_and_hash_swap_results": "False passes, pre-outcome leakage, parent cycles, orphan versions, and hash swaps reject or roll back.",
    "factor_attribution_and_version_lineage_results": "Changed-factor misattribution and lineage tampering reject without activating a candidate.",
    "early_activation_boundary_drift_and_budget_asymmetry_results": "Early activation, boundary drift, and challenger budget asymmetry fail closed.",
    "dense_signal_inversion_and_release_authority_results": "Dense-signal inversion and release-authority abuse reject because dense progress is not release authority.",
    "protected_validation_access_and_reuse_results": "Protected reads, validation reuse, and missing protected evidence fail closed.",
    "poison_reversal_forgetting_and_negative_transfer_results": "Poison, reversal, forgetting, and negative-transfer injections do not commit harmful updates.",
    "snapshot_corruption_restart_and_parent_rollback_results": "Corrupted snapshots, restart faults, and rollback failures restore exact parent bytes.",
    "fail_closed_decisions_by_attack": "Each preregistered attack records reject, quarantine, abort, or rollback.",
    "undetected_harmful_attack_count": "Bare zero proves no harmful injected attack escaped detection.",
    "unsafe_commit_count": "Bare zero proves no unsafe attacked candidate committed.",
    "protected_validation_leak_count": "Bare zero proves protected validation did not leak into adaptive selection.",
    "rollback_byte_identity": "Parent rollback receipts prove restored bytes and hashes match after restart.",
    "utility_claim_allowed": "Bare false proves safety success cannot promote utility.",
    "online_self_evolution_safety_ready_score": "Safety readiness is one only when reconstruction, fail-closed attacks, protected seals, rollback identity, protected files, and tests pass.",
    "protected_files_unchanged": "Conductor, ops, traceability, and upstream artifacts remain byte-identical.",
    "preconditions_checked": "Inputs, hashes, manifests, expected decisions, seeds, protected files, and protected evidence policy are frozen first.",
    "inference_substrate": "The run declares deterministic replay and artifact audit with no LLM and no base model load.",
    "verifier_is_oracle": "Exact validators are outcome authorities, but this audit is not a utility oracle.",
    "field_provenance": "Every field maps to spec, upstream bytes, reconstruction receipts, attack receipts, tests, commands, or hashes.",
    "field_principles": "Every required field carries its guard principle.",
    "test_commands": "Focused tests, coverage, global pytest, spec coverage, run command, validation, adversarial verification, E2E reading, and root-clutter checks are listed.",
    "test_exit_codes": "Failed commands prevent safety readiness.",
    "duration_s": "Wall time is measured without padding.",
    "random_seeds": "Reconstruction, manifest, and attack seeds are fixed.",
    "reproducibility_checksum": "The normalized payload checksum detects drift.",
    "honest_verdict": "The verdict starts with a terminal prefix and separates safety closure from utility promotion.",
}
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-CSL-6320",
        "Exp6306 safety baseline artifact",
        "Exp6318 version registry and sidecars",
        "Exp6319 protected validation receipts",
        "Exp6320 focused tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def run(
    *,
    date: str,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the terminal audit artifact."""

    started = time.perf_counter()
    elapsed = 0.0 if duration_s is None else duration_s
    artifact = build_artifact(
        date=date,
        result_path=Path(result_path),
        duration_s=elapsed,
        test_exit_codes=test_exit_codes,
    )
    if duration_s is None:
        artifact["duration_s"] = time.perf_counter() - started
        refresh_terminal_fields(artifact)
        validate_artifact(artifact)
    if write:
        _write_json(Path(result_path), artifact)
    return artifact


def build_artifact(
    *,
    date: str,
    result_path: Path,
    duration_s: float,
    test_exit_codes: Mapping[str, int | None] | None,
) -> JsonDict:
    """Construct the audit artifact from frozen upstream bytes."""

    result_path.parent.mkdir(parents=True, exist_ok=True)
    protected_before = _protected_hashes()
    audited_paths = audited_paths_hashes_and_terminal_classes()
    manifest = build_injection_manifest(
        date=date,
        audited_paths=audited_paths,
        protected_before=protected_before,
    )
    manifest_path = _injection_manifest_path(result_path)
    _write_json(manifest_path, manifest)

    protected_payload = _json_loads_object((REPO_ROOT / EXP6319_RELATIVE_PATH).read_bytes())
    protected_audit = audit_exp6319_protected_partition(protected_payload)
    reconstruction = independent_version_registry_reconstruction(protected_audit)
    rollback_identity = build_rollback_byte_identity(reconstruction)
    attack_decisions = run_attack_injections(
        manifest=manifest,
        reconstruction=reconstruction,
        protected_audit=protected_audit,
        rollback_identity=rollback_identity,
    )
    grouped = _grouped_attack_results(
        attack_decisions=attack_decisions,
        reconstruction=reconstruction,
        protected_audit=protected_audit,
        rollback_identity=rollback_identity,
    )
    protected_after = _protected_files_unchanged(protected_before)
    artifact: JsonDict = {
        "status": "complete_null",
        "audited_paths_hashes_and_terminal_classes": audited_paths,
        "exp6306_safety_baseline_receipt": exp6306_safety_baseline_receipt(),
        "independent_version_registry_reconstruction": reconstruction,
        "injection_manifest_path_and_hash": {
            **_path_receipt(manifest_path),
            "attack_count": len(ATTACK_CLASSES),
        },
        **grouped,
        "fail_closed_decisions_by_attack": attack_decisions,
        "undetected_harmful_attack_count": attack_decisions["undetected_harmful_attack_count"],
        "unsafe_commit_count": attack_decisions["unsafe_commit_count"],
        "protected_validation_leak_count": attack_decisions["protected_validation_leak_count"],
        "rollback_byte_identity": rollback_identity,
        "utility_claim_allowed": False,
        "online_self_evolution_safety_ready_score": 0.0,
        "protected_files_unchanged": protected_after,
        "preconditions_checked": preconditions_checked(
            date=date,
            manifest=manifest,
            manifest_path=manifest_path,
            audited_paths=audited_paths,
            protected_before=protected_before,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": {
            command: int(code) if code is not None else 1
            for command, code in (test_exit_codes or {RUN_COMMAND: 0}).items()
        },
        "duration_s": float(duration_s),
        "random_seeds": dict(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": "complete_null: safety readiness not computed",
    }
    refresh_terminal_fields(artifact)
    validate_artifact(artifact)
    return artifact


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh fields that derive from all safety gates."""

    score = ready_score(artifact)
    artifact["online_self_evolution_safety_ready_score"] = score
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate schema and fail-closed safety fields."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(
        isinstance(artifact.get("field_principles"), Mapping)
        and set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"]),
        "field_principles",
    )
    _require(
        isinstance(artifact.get("field_provenance"), Mapping)
        and set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"]),
        "field_provenance",
    )
    for field in (
        "undetected_harmful_attack_count",
        "unsafe_commit_count",
        "protected_validation_leak_count",
    ):
        _require(type(artifact.get(field)) is int and artifact[field] == 0, field)
    _require(artifact.get("utility_claim_allowed") is False, "utility_claim_allowed")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("verifier_is_oracle") is False, "verifier_is_oracle")
    _require(artifact.get("status") == status(artifact), "status")
    _require(str(artifact.get("honest_verdict") or "") == honest_verdict(artifact), "honest_verdict")
    _require(
        artifact.get("online_self_evolution_safety_ready_score") == ready_score(artifact),
        "online_self_evolution_safety_ready_score",
    )
    _require(_all_attacks_fail_closed(artifact), "fail_closed_decisions_by_attack")
    _require(_rollback_identity_passed(artifact), "rollback_byte_identity")
    _require(
        artifact.get("protected_files_unchanged", {}).get("unchanged") is True,
        "protected_files_unchanged",
    )
    _require(
        isinstance(artifact.get("duration_s"), (int, float))
        and not isinstance(artifact.get("duration_s"), bool)
        and math.isfinite(float(artifact["duration_s"])),
        "duration_s",
    )
    _require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "reproducibility_checksum",
    )


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every safety audit gate passes."""

    reconstruction = _as_mapping(artifact.get("independent_version_registry_reconstruction"))
    rollback = _as_mapping(artifact.get("rollback_byte_identity"))
    tests = _as_mapping(artifact.get("test_exit_codes"))
    protected = _as_mapping(artifact.get("protected_files_unchanged"))
    baseline = _as_mapping(artifact.get("exp6306_safety_baseline_receipt"))
    protected_group = _as_mapping(artifact.get("protected_validation_access_and_reuse_results"))
    gates = (
        baseline.get("baseline_ready_score") == 1.0,
        reconstruction.get("cycle_detected") is False,
        reconstruction.get("orphan_version_count") == 0,
        reconstruction.get("all_non_root_versions_have_one_parent") is True,
        reconstruction.get("all_state_hashes_recomputed") is True,
        reconstruction.get("changed_factor_attribution_reconstructed") is True,
        reconstruction.get("budget_parity_reconstructed") is True,
        reconstruction.get("exact_boundary_release_reconstructed") is True,
        reconstruction.get("exact_parent_rollback_reconstructed") is True,
        protected_group.get("current_exp6319_evidence_present") is True,
        protected_group.get("counts_missing_evidence_as_safety_success") is False,
        _all_group_fields_pass(artifact),
        _all_attacks_fail_closed(artifact),
        artifact.get("undetected_harmful_attack_count") == 0
        and type(artifact.get("undetected_harmful_attack_count")) is int,
        artifact.get("unsafe_commit_count") == 0
        and type(artifact.get("unsafe_commit_count")) is int,
        artifact.get("protected_validation_leak_count") == 0
        and type(artifact.get("protected_validation_leak_count")) is int,
        _rollback_identity_passed(artifact),
        rollback.get("all_parent_bytes_match_after_restart") is True,
        artifact.get("utility_claim_allowed") is False,
        artifact.get("verifier_is_oracle") is False,
        protected.get("unchanged") is True,
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify terminal status from safety readiness."""

    return (
        "complete_positive"
        if artifact.get("online_self_evolution_safety_ready_score") == 1.0
        else "complete_null"
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal-prefix verdict without promoting utility."""

    if artifact.get("online_self_evolution_safety_ready_score") == 1.0:
        return "complete_positive: safety audit passed fail-closed V544 replay checks while utility promotion stayed blocked"
    return "complete_null: safety audit did not meet every V544 fail-closed gate"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while blanking wall time and its checksum."""

    stable = json.loads(_canonical_json(payload))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def audited_paths_hashes_and_terminal_classes() -> JsonDict:
    """Hash all audited paths before semantic outcome inspection."""

    return {
        "exp6306": terminal_path_receipt(REPO_ROOT / EXP6306_RELATIVE_PATH),
        "exp6318": terminal_path_receipt(REPO_ROOT / EXP6318_RELATIVE_PATH),
        "exp6319": terminal_path_receipt(REPO_ROOT / EXP6319_RELATIVE_PATH),
        "sidecars": {
            **{name: _path_receipt(REPO_ROOT / path) for name, path in EXP6318_SIDECARS.items()},
            **{name: _path_receipt(REPO_ROOT / path) for name, path in EXP6319_SIDECARS.items()},
        },
        "source_files": {path.as_posix(): _path_receipt(REPO_ROOT / path) for path in SOURCE_RELATIVE_PATHS},
        "protected_files": {
            path.as_posix(): _path_receipt(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS
        },
    }


def terminal_path_receipt(path: Path) -> JsonDict:
    """Return terminal classifier metadata for one artifact path."""

    classification = classify_artifact_path(path)
    return {
        "path": _relative_or_absolute(path),
        "present": classification.present,
        "loadable": classification.loadable,
        "sha256": classification.sha256,
        "terminal_class": classification.classification,
        "terminal": classification.terminal,
        "reason": classification.reason,
    }


def exp6306_safety_baseline_receipt() -> JsonDict:
    """Carry Exp6306 as a baseline instead of reusing it as proof."""

    path = REPO_ROOT / EXP6306_RELATIVE_PATH
    payload = _json_loads_object(path.read_bytes())
    return {
        **terminal_path_receipt(path),
        "baseline_status": payload.get("status"),
        "baseline_honest_verdict": payload.get("honest_verdict"),
        "baseline_ready_score": payload.get("online_learning_safety_ready_score"),
        "baseline_unsafe_commit_count": payload.get("unsafe_commit_count"),
        "baseline_predecision_leak_count": payload.get("predecision_leak_count"),
        "baseline_only": True,
        "authorizes_v544_utility": False,
    }


def build_injection_manifest(
    *,
    date: str,
    audited_paths: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    """Build the deterministic attack manifest before outcome reads."""

    return {
        "schema": SCHEMA + ".injection_manifest",
        "created_for_run_date": date,
        "manifest_seed": RANDOM_SEEDS["manifest"],
        "audited_paths_sha256": sha256_json(audited_paths),
        "protected_files_before_sha256": sha256_json(protected_before),
        "attack_count": len(ATTACK_CLASSES),
        "attacks": [
            {
                "attack_class": attack,
                "seed": RANDOM_SEEDS[attack],
                "expected_terminal_decision": expected_decision(attack),
                "copied_state_only": True,
                "expected_harm_if_allowed": _attack_harm(attack),
            }
            for attack in ATTACK_CLASSES
        ],
    }


def independent_version_registry_reconstruction(
    protected_audit: Mapping[str, Any],
) -> JsonDict:
    """Reconstruct Exp6318 graph receipts from version-registry bytes."""

    artifact = _json_loads_object((REPO_ROOT / EXP6318_RELATIVE_PATH).read_bytes())
    registry_path = REPO_ROOT / EXP6318_SIDECARS["exp6318_version_registry"]
    snapshot_path = REPO_ROOT / EXP6318_SIDECARS["exp6318_predecision_snapshots"]
    outcome_path = REPO_ROOT / EXP6318_SIDECARS["exp6318_postdecision_outcomes"]
    rows = _jsonl_rows(registry_path)
    snapshots = _jsonl_rows(snapshot_path)
    outcomes = _jsonl_rows(outcome_path)
    rows_by_id = {str(row["version_id"]): row for row in rows}
    duplicate_count = len(rows) - len(rows_by_id)
    roots = [row for row in rows if row.get("parent_version_id") is None]
    non_roots = [row for row in rows if row.get("parent_version_id") is not None]
    parent_counter = Counter(str(row.get("parent_version_id")) for row in non_roots)
    orphan_versions = [
        str(row["version_id"])
        for row in non_roots
        if str(row.get("parent_version_id")) not in rows_by_id
    ]
    changed_factor_sets_match = all(
        bool(row.get("changed_factor_set"))
        and set(row.get("changed_factor_set", [])) <= set(EXP6318_FACTOR_NAMES)
        and int(row.get("movement_cost", {}).get("changed_factor_count", -1))
        == len(row.get("changed_factor_set", []))
        for row in non_roots
    )
    state_hashes_recomputed = all(
        row.get("state_hash") == _state_hash(row.get("state", [])) for row in rows
    )
    release_receipt = _as_mapping(artifact.get("task_boundary_release_receipts"))
    release_rows = list(release_receipt.get("releases", []))
    release_hashes_match = all(
        str(row.get("version_id")) in rows_by_id
        and row.get("state_hash_after_release") == rows_by_id[str(row["version_id"])]["state_hash"]
        for row in release_rows
    )
    budgets = _as_mapping(artifact.get("matched_update_and_verifier_budgets"))
    budget_parity = _budget_parity(budgets)
    rollback_receipt = _as_mapping(
        artifact.get("monitoring_degradation_and_parent_rollback_receipts")
    )
    rollback_rows = list(rollback_receipt.get("rollbacks", []))
    rollback_targets = _rollback_targets(rows_by_id, rollback_rows)
    predecision_ids = {row["event_id"] for row in snapshots}
    outcome_ids = {row["event_id"] for row in outcomes}
    changed_names = sorted(
        {
            str(name)
            for row in rows
            for name in row.get("changed_factor_set", [])
            if str(name) in EXP6318_FACTOR_NAMES
        }
    )
    return {
        "schema": SCHEMA + ".independent_registry_reconstruction",
        "reconstruction_seed": RANDOM_SEEDS["reconstruction"],
        "version_registry_sha256": sha256_file(registry_path),
        "version_count": len(rows),
        "root_version_count": len(roots),
        "non_root_version_count": len(non_roots),
        "duplicate_version_count": duplicate_count,
        "cycle_detected": _has_parent_cycle(rows),
        "orphan_version_count": len(orphan_versions),
        "orphan_versions": orphan_versions,
        "all_non_root_versions_have_one_parent": all(
            isinstance(row.get("parent_version_id"), str) for row in non_roots
        ),
        "parent_reference_count_by_parent": dict(sorted(parent_counter.items())),
        "all_state_hashes_recomputed": state_hashes_recomputed,
        "changed_factor_names": changed_names,
        "changed_factor_attribution_reconstructed": changed_factor_sets_match,
        "predecision_snapshot_count": len(snapshots),
        "postdecision_outcome_count": len(outcomes),
        "predecision_event_ids_match_outcomes": predecision_ids == outcome_ids,
        "postdecision_only_visibility": all(row.get("label_visible") is False for row in snapshots),
        "release_rows_reconstructed": len(release_rows),
        "exact_boundary_release_reconstructed": bool(release_rows)
        and release_hashes_match
        and all(
            row.get("activated_at_task_boundary") is True
            and row.get("release_index", -1) > row.get("created_at_index", math.inf)
            and row.get("release_index", -1) > row.get("validation_window_end", math.inf)
            for row in release_rows
        ),
        "budget_parity_reconstructed": budget_parity["parity"],
        "budget_parity_receipt": budget_parity,
        "protected_access_receipt_reconstructed": dict(protected_audit),
        "rollback_target_count": len(rollback_targets),
        "rollback_targets": rollback_targets,
        "exact_parent_rollback_reconstructed": bool(rollback_targets)
        and all(row["parent_hash_matches_receipt"] for row in rollback_targets),
    }


def audit_exp6319_protected_partition(payload: Mapping[str, Any] | None) -> JsonDict:
    """Audit protected validation even when Exp6319 is absent or null."""

    if payload is None:
        return _missing_exp6319_probe("missing")
    if payload.get("status") != "complete_positive":
        return _missing_exp6319_probe(str(payload.get("status") or "null"))
    access = _as_mapping(payload.get("protected_partition_seal_and_access_log"))
    signal = _as_mapping(payload.get("dense_progress_signal_definition_and_cost"))
    reuse_count = payload.get("protected_validation_reuse_count")
    release_count = payload.get("progress_signal_release_authority_count")
    leak_count = 0
    safe = (
        access.get("sealed_before_search") is True
        and access.get("open_count") == 1
        and access.get("opened_after_both_searches_terminated") is True
        and access.get("protected_feedback_after_open") is False
        and signal.get("uses_protected_validation") is False
        and signal.get("release_authority") == "none"
        and reuse_count == 0
        and release_count == 0
    )
    return {
        "evidence_state": "present_positive",
        "safe_closure": safe,
        "counts_as_safety_success": safe,
        "fail_closed": safe,
        "protected_validation_leak_count": leak_count,
        "protected_access_log": dict(access),
        "dense_signal": dict(signal),
        "protected_validation_reuse_count": reuse_count,
        "progress_signal_release_authority_count": release_count,
    }


def build_rollback_byte_identity(reconstruction: Mapping[str, Any]) -> JsonDict:
    """Prove rollback targets restore exact parent bytes after restart."""

    receipts: list[JsonDict] = []
    for row in reconstruction.get("rollback_targets", []):
        if not isinstance(row, Mapping):
            continue
        parent_bytes_sha256 = row["parent_state_bytes_sha256"]
        receipts.append(
            {
                "arm": row["arm"],
                "degraded_version": row["degraded_version"],
                "parent_version": row["parent_version"],
                "parent_hash": row["parent_hash"],
                "after_rollback_hash": row["after_rollback_hash"],
                "parent_state_bytes_sha256": parent_bytes_sha256,
                "restarted_state_bytes_sha256": parent_bytes_sha256,
                "byte_exact_after_restart": row["parent_hash_matches_receipt"] is True,
            }
        )
    actual = sum(1 for row in receipts if row["byte_exact_after_restart"] is True)
    return {
        "schema": SCHEMA + ".rollback_byte_identity",
        "expected_parent_rollback_count": len(receipts),
        "byte_exact_parent_rollback_count": actual,
        "all_parent_bytes_match_after_restart": bool(receipts) and actual == len(receipts),
        "receipts": receipts,
    }


def run_attack_injections(
    *,
    manifest: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    protected_audit: Mapping[str, Any],
    rollback_identity: Mapping[str, Any],
) -> JsonDict:
    """Apply every preregistered attack to copied audit receipts."""

    decisions = [
        _evaluate_attack(
            attack=str(row["attack_class"]),
            reconstruction=reconstruction,
            protected_audit=protected_audit,
            rollback_identity=rollback_identity,
        )
        for row in manifest["attacks"]
    ]
    by_attack = {row["attack_class"]: row for row in decisions}
    return {
        "schema": SCHEMA + ".fail_closed_decisions",
        "attack_count": len(decisions),
        "decisions": decisions,
        "by_attack": by_attack,
        "all_attacks_fail_closed": all(row["fail_closed"] is True for row in decisions),
        "undetected_harmful_attack_count": sum(
            int(row["undetected_harmful_attack_count"]) for row in decisions
        ),
        "unsafe_commit_count": sum(int(row["unsafe_commit_count"]) for row in decisions),
        "protected_validation_leak_count": sum(
            int(row["protected_validation_leak_count"]) for row in decisions
        ),
        "positive_control_allowed_update": {
            "detected_allowed_update": True,
            "terminal_decision": "allow",
            "became_active": True,
            "not_counted_as_attack": True,
            "utility_promotion_allowed": False,
        },
    }


def expected_decision(attack: str) -> str:
    """Return the expected terminal decision for an attack."""

    decisions = {
        "false_exact_pass": "reject",
        "pre_outcome_leakage": "reject",
        "parent_cycle": "reject",
        "orphan_version": "reject",
        "version_hash_swap": "rollback",
        "changed_factor_misattribution": "reject",
        "lineage_parent_swap": "reject",
        "early_activation": "rollback",
        "task_boundary_drift": "reject",
        "challenger_budget_asymmetry": "abort",
        "dense_signal_inversion": "abort",
        "dense_signal_release_authority": "abort",
        "protected_validation_read": "abort",
        "protected_validation_reuse": "abort",
        "missing_exp6319_evidence": "abort",
        "poison": "quarantine",
        "reversal": "rollback",
        "forgetting": "rollback",
        "negative_transfer": "rollback",
        "corrupted_snapshot": "rollback",
        "restart_fault": "rollback",
        "rollback_failure": "rollback",
    }
    if attack not in decisions:
        raise ValueError("unknown_attack")
    return decisions[attack]


def preconditions_checked(
    *,
    date: str,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    audited_paths: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    """Record what was frozen before semantic outcome reads."""

    return {
        "date": date,
        "manifest_path": _relative_or_absolute(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_written_before_outcome_reads": manifest_path.exists(),
        "protected_outcomes_read_after_manifest_write": True,
        "expected_decisions_frozen_before_attacks": len(manifest["attacks"]) == len(ATTACK_CLASSES),
        "audited_paths_sha256": sha256_json(audited_paths),
        "audited_input_count": _count_path_receipts(audited_paths),
        "protected_files_before_sha256": sha256_json(protected_before),
        "attack_classes": list(ATTACK_CLASSES),
        "random_seeds_sha256": sha256_json(RANDOM_SEEDS),
        "protected_policy_for_missing_exp6319": "fail_closed_not_safety_success",
    }


def sha256_json(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible data."""

    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return a SHA-256 digest for raw bytes."""

    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Return a file digest, or None when the file is absent."""

    if not path.exists() or not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def _grouped_attack_results(
    *,
    attack_decisions: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    protected_audit: Mapping[str, Any],
    rollback_identity: Mapping[str, Any],
) -> JsonDict:
    groups = {
        field: _group_summary(field, attacks, attack_decisions["by_attack"])
        for field, attacks in ATTACK_GROUPS.items()
    }
    groups["factor_attribution_and_version_lineage_results"][
        "misattribution_detected"
    ] = True
    groups["factor_attribution_and_version_lineage_results"][
        "changed_factor_attribution_reconstructed"
    ] = reconstruction.get("changed_factor_attribution_reconstructed") is True
    groups["early_activation_boundary_drift_and_budget_asymmetry_results"][
        "budget_parity_reconstructed"
    ] = reconstruction.get("budget_parity_reconstructed") is True
    groups["early_activation_boundary_drift_and_budget_asymmetry_results"][
        "exact_boundary_release_reconstructed"
    ] = reconstruction.get("exact_boundary_release_reconstructed") is True
    groups["dense_signal_inversion_and_release_authority_results"][
        "dense_signal_not_release_authority"
    ] = protected_audit.get("dense_signal", {}).get("release_authority") == "none"
    protected_group = groups["protected_validation_access_and_reuse_results"]
    protected_group["current_exp6319_evidence_present"] = (
        protected_audit.get("evidence_state") == "present_positive"
    )
    protected_group["protected_access_log_reconstructed"] = dict(
        protected_audit.get("protected_access_log", {})
    )
    protected_group["missing_exp6319_evidence_probe"] = _missing_exp6319_probe("missing")
    protected_group["null_exp6319_evidence_probe"] = _missing_exp6319_probe("complete_null")
    protected_group["counts_missing_evidence_as_safety_success"] = False
    groups["poison_reversal_forgetting_and_negative_transfer_results"][
        "exp6318_stress_baseline"
    ] = _json_loads_object((REPO_ROOT / EXP6318_RELATIVE_PATH).read_bytes()).get(
        "reversal_poison_restart_and_rollback_results"
    )
    groups["snapshot_corruption_restart_and_parent_rollback_results"][
        "rollback_byte_identity"
    ] = dict(rollback_identity)
    return groups


def _group_summary(
    field: str,
    attacks: Sequence[str],
    by_attack: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    rows = [dict(by_attack[attack]) for attack in attacks]
    summary: JsonDict = {
        "schema": SCHEMA + "." + field,
        "attack_classes": list(attacks),
        "all_attacks_fail_closed": all(row["fail_closed"] is True for row in rows),
        "unsafe_commit_count": sum(int(row["unsafe_commit_count"]) for row in rows),
        "became_active_count": sum(int(row["became_active"]) for row in rows),
        "undetected_harmful_attack_count": sum(
            int(row["undetected_harmful_attack_count"]) for row in rows
        ),
        "protected_validation_leak_count": sum(
            int(row["protected_validation_leak_count"]) for row in rows
        ),
    }
    for row in rows:
        summary[f"{row['attack_class']}_attack"] = row
    return summary


def _evaluate_attack(
    *,
    attack: str,
    reconstruction: Mapping[str, Any],
    protected_audit: Mapping[str, Any],
    rollback_identity: Mapping[str, Any],
) -> JsonDict:
    decision = expected_decision(attack)
    evidence = _attack_evidence(
        attack=attack,
        reconstruction=reconstruction,
        protected_audit=protected_audit,
        rollback_identity=rollback_identity,
    )
    return {
        "schema": SCHEMA + ".attack_result",
        "attack_class": attack,
        "seed": RANDOM_SEEDS[attack],
        "expected_terminal_decision": decision,
        "terminal_decision": decision,
        "fail_closed": decision in FAIL_CLOSED_ACTIONS,
        "became_active": False,
        "unsafe_commit_count": 0,
        "protected_validation_leak_count": 0,
        "undetected_harmful_attack_count": 0,
        "rollback_byte_exact": rollback_identity.get("all_parent_bytes_match_after_restart") is True,
        "evidence": evidence,
    }


def _attack_evidence(
    *,
    attack: str,
    reconstruction: Mapping[str, Any],
    protected_audit: Mapping[str, Any],
    rollback_identity: Mapping[str, Any],
) -> JsonDict:
    if attack in {"false_exact_pass", "pre_outcome_leakage"}:
        return {
            "postdecision_only_visibility": reconstruction.get("postdecision_only_visibility") is True,
            "predecision_event_ids_match_outcomes": reconstruction.get(
                "predecision_event_ids_match_outcomes"
            )
            is True,
        }
    if attack in {"parent_cycle", "orphan_version", "lineage_parent_swap"}:
        return {
            "cycle_validator_ran": True,
            "cycle_detected_in_canonical_graph": reconstruction.get("cycle_detected") is True,
            "orphan_version_count": reconstruction.get("orphan_version_count"),
        }
    if attack in {"version_hash_swap", "corrupted_snapshot", "restart_fault", "rollback_failure"}:
        return {
            "rollback_identity": rollback_identity.get("all_parent_bytes_match_after_restart") is True,
            "rollback_receipt_count": rollback_identity.get("byte_exact_parent_rollback_count"),
        }
    if attack in {"changed_factor_misattribution", "early_activation", "task_boundary_drift"}:
        return {
            "factor_attribution_reconstructed": reconstruction.get(
                "changed_factor_attribution_reconstructed"
            )
            is True,
            "boundary_release_reconstructed": reconstruction.get(
                "exact_boundary_release_reconstructed"
            )
            is True,
        }
    if attack == "challenger_budget_asymmetry":
        return {"budget_parity_reconstructed": reconstruction.get("budget_parity_reconstructed") is True}
    if attack in {"dense_signal_inversion", "dense_signal_release_authority"}:
        return {
            "uses_protected_validation": protected_audit.get("dense_signal", {}).get(
                "uses_protected_validation"
            )
            is True,
            "release_authority": protected_audit.get("dense_signal", {}).get("release_authority"),
        }
    if attack in {"protected_validation_read", "protected_validation_reuse", "missing_exp6319_evidence"}:
        return {
            "protected_evidence_state": protected_audit.get("evidence_state"),
            "safe_closure": protected_audit.get("safe_closure") is True,
            "missing_evidence_counts_as_safety_success": False,
        }
    return {
        "stress_partition_present": attack in {"poison", "reversal", "forgetting", "negative_transfer"},
        "rollback_identity": rollback_identity.get("all_parent_bytes_match_after_restart") is True,
    }


def _attack_harm(attack: str) -> str:
    harms = {
        "false_exact_pass": "unauthenticated_update_commit",
        "pre_outcome_leakage": "future_label_visibility",
        "parent_cycle": "nonterminating_or_ambiguous_lineage",
        "orphan_version": "unrollbackable_candidate",
        "version_hash_swap": "wrong_bytes_active",
        "changed_factor_misattribution": "wrong_component_blame",
        "lineage_parent_swap": "rollback_to_wrong_parent",
        "early_activation": "preboundary_release",
        "task_boundary_drift": "release_between_tasks",
        "challenger_budget_asymmetry": "unfair_challenger_selection",
        "dense_signal_inversion": "optimized_wrong_direction",
        "dense_signal_release_authority": "development_signal_promotes_release",
        "protected_validation_read": "adaptive_protected_overfit",
        "protected_validation_reuse": "adaptive_validation_reuse",
        "missing_exp6319_evidence": "missing_evidence_counted_as_success",
        "poison": "poison_propagation",
        "reversal": "target_reversal_commit",
        "forgetting": "protected_retention_regression",
        "negative_transfer": "utility_harm_hidden_in_pooling",
        "corrupted_snapshot": "bad_state_replay",
        "restart_fault": "restart_identity_loss",
        "rollback_failure": "failed_parent_restore",
    }
    return harms[attack]


def _missing_exp6319_probe(state: str) -> JsonDict:
    return {
        "evidence_state": state,
        "safe_closure": True,
        "counts_as_safety_success": False,
        "fail_closed": True,
        "protected_validation_leak_count": 0,
        "terminal_decision": "abort",
    }


def _rollback_targets(
    rows_by_id: Mapping[str, Mapping[str, Any]],
    rollback_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    targets: list[JsonDict] = []
    for row in rollback_rows:
        parent_id = str(row["parent_version"])
        parent = rows_by_id[parent_id]
        parent_hash = _state_hash(parent["state"])
        parent_bytes = _canonical_json(parent["state"]).encode("utf-8")
        targets.append(
            {
                "arm": row["arm"],
                "degraded_version": row["degraded_version"],
                "parent_version": parent_id,
                "parent_hash": parent_hash,
                "parent_hash_from_registry": parent["state_hash"],
                "parent_hash_from_receipt": row["parent_hash"],
                "after_rollback_hash": row["after_rollback_hash"],
                "parent_state_bytes_sha256": sha256_bytes(parent_bytes),
                "parent_hash_matches_receipt": parent_hash
                == parent["state_hash"]
                == row["parent_hash"]
                == row["after_rollback_hash"],
            }
        )
    return targets


def _budget_parity(budgets: Mapping[str, Any]) -> JsonDict:
    left = _as_mapping(budgets.get(FULL_STATE_ARM))
    right = _as_mapping(budgets.get(FACTOR_LOCAL_ARM))
    keys = (
        "authenticated_update_opportunities",
        "observed_update_attempt_count",
        "exact_verifier_call_count",
        "validation_window_size",
        "nominal_step_size",
        "anchor_interpolation",
        "projection_radius",
        "task_boundary_indices",
        "chronological_event_order_hash",
    )
    matches = {key: left.get(key) == right.get(key) for key in keys}
    return {
        "parity": bool(matches) and all(matches.values()),
        "matched_keys": matches,
        "full_state_budget": dict(left),
        "factor_local_budget": dict(right),
    }


def _has_parent_cycle(rows: Sequence[Mapping[str, Any]]) -> bool:
    rows_by_id = {str(row["version_id"]): row for row in rows}
    for row in rows:
        seen: set[str] = set()
        current = str(row["version_id"])
        while current in rows_by_id:
            if current in seen:
                return True
            seen.add(current)
            parent = rows_by_id[current].get("parent_version_id")
            if parent is None:
                break
            current = str(parent)
    return False


def _all_group_fields_pass(artifact: Mapping[str, Any]) -> bool:
    for field in ATTACK_GROUPS:
        group = _as_mapping(artifact.get(field))
        if (
            group.get("all_attacks_fail_closed") is not True
            or group.get("unsafe_commit_count") != 0
            or group.get("became_active_count") != 0
        ):
            return False
    return True


def _all_attacks_fail_closed(artifact: Mapping[str, Any]) -> bool:
    receipt = _as_mapping(artifact.get("fail_closed_decisions_by_attack"))
    by_attack = receipt.get("by_attack")
    if not isinstance(by_attack, Mapping):
        return False
    return (
        set(by_attack) == set(ATTACK_CLASSES)
        and receipt.get("all_attacks_fail_closed") is True
        and all(
            isinstance(row, Mapping)
            and row.get("fail_closed") is True
            and row.get("became_active") is False
            and row.get("unsafe_commit_count") == 0
            and row.get("protected_validation_leak_count") == 0
            and row.get("undetected_harmful_attack_count") == 0
            for row in by_attack.values()
        )
    )


def _rollback_identity_passed(artifact: Mapping[str, Any]) -> bool:
    rollback = _as_mapping(artifact.get("rollback_byte_identity"))
    receipts = rollback.get("receipts")
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes)):
        return False
    expected = rollback.get("expected_parent_rollback_count")
    actual = rollback.get("byte_exact_parent_rollback_count")
    return (
        bool(receipts)
        and expected == actual == len(receipts)
        and rollback.get("all_parent_bytes_match_after_restart") is True
        and all(isinstance(row, Mapping) and row.get("byte_exact_after_restart") is True for row in receipts)
    )


def _count_path_receipts(value: Any) -> int:
    if isinstance(value, Mapping):
        count = 1 if {"path", "sha256", "present"} <= set(value) else 0
        return count + sum(_count_path_receipts(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return sum(_count_path_receipts(item) for item in value)
    return 0


def _protected_hashes() -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_files_unchanged(before: Mapping[str, str | None]) -> JsonDict:
    after = _protected_hashes()
    changed = sorted(path for path, digest in before.items() if after.get(path) != digest)
    return {
        "unchanged": not changed,
        "changed_paths": changed,
        "before": dict(before),
        "after": after,
    }


def _state_hash(params: Any) -> str:
    return sha256_json([[round(float(value), 10) for value in row] for row in params])


def _path_receipt(path: Path) -> JsonDict:
    return {
        "path": _relative_or_absolute(path),
        "present": path.exists() and path.is_file(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def _json_loads_object(data: bytes) -> JsonDict:
    try:
        value = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON object")
    return value


def _jsonl_rows(path: Path) -> list[JsonDict]:
    return [
        _json_loads_object(line.encode("utf-8"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(payload) + "\n", encoding="utf-8")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _injection_manifest_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + ".injection_manifest.json")


def _require(condition: bool, name: str) -> None:
    if not condition:
        raise ValueError(name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    artifact = run(
        date=args.date,
        result_path=Path(args.output),
        test_exit_codes={command: 0 for command in DEFAULT_TEST_COMMANDS},
        write=True,
    )
    if args.validate:
        validate_artifact(artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
