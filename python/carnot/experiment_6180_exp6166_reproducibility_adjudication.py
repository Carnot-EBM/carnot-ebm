"""Exp6180 read-only adjudication of the Exp6166 stochastic result.

Spec refs: REQ-SAMPLE-6180, SCENARIO-SAMPLE-6180-READ-ONLY-ADJUDICATION,
SCENARIO-SAMPLE-6180-HISTORICAL-BLOCK-PRESERVED.

This module treats the old Exp6166 artifact as evidence, not as a training
recipe. It recomputes the declared metrics from stored arm probabilities so a
future reader can separate the scientific software result from unrelated
repository-suite failures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

from carnot import experiment_6166_mode_jumping_factor_thermalization as exp6166
from carnot import experiment_6170_v535_task_artifact_isolation_canary as exp6170


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6180_exp6166_reproducibility_adjudication.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6180_exp6166_reproducibility_adjudication.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6180_exp6166_reproducibility_adjudication.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/samplers/spec.md")
EXP6166_RESULT_RELATIVE_PATH = exp6166.RESULT_RELATIVE_PATH
EXP6170_RESULT_RELATIVE_PATH = exp6170.RESULT_RELATIVE_PATH
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ARTIFACT_SCHEMA_VERSION = "carnot.experiment_6180.exp6166_reproducibility_adjudication.v1"
EXPERIMENT_ID = "experiment_6180_exp6166_reproducibility_adjudication"
RUN_DATE = "20260807"
INFERENCE_SUBSTRATE = "jax_cpu_software_exp6166_artifact_replay"

IMMUTABLE_EXP6166_PATHS: tuple[Path, ...] = (
    EXP6166_RESULT_RELATIVE_PATH,
    exp6166.MODULE_RELATIVE_PATH,
    exp6166.TEST_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
)
PROTECTED_FILES: tuple[Path, ...] = exp6166.PROTECTED_FILES
FOCUSED_TEST_COMMAND = (
    "JAX_PLATFORMS=cpu .venv/bin/pytest "
    "tests/python/test_experiment_6180_exp6166_reproducibility_adjudication.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    "JAX_PLATFORMS=cpu .venv/bin/coverage run --source=python/carnot "
    "-m pytest tests/python/test_experiment_6180_exp6166_reproducibility_adjudication.py "
    "-q --no-cov -n 0 && JAX_PLATFORMS=cpu .venv/bin/coverage report "
    "--include=python/carnot/experiment_6180_exp6166_reproducibility_adjudication.py "
    "--fail-under=100"
)
FULL_SUITE_COMMAND = ".venv/bin/pytest tests/python -q"

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "experiment_id",
    "run_date",
    "preconditions_checked",
    "immutable_exp6166_byte_snapshot",
    "no_refit_receipt",
    "stochastic_replay_receipt",
    "recomputed_metric_determination",
    "old_full_suite_failure_classification",
    "companion_determination",
    "no_hardware_promotion_receipt",
    "protected_files_unchanged",
    "before_after_byte_comparison",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state for the companion adjudication, not a rewrite of Exp6166.",
    "experiment_id": "Names only the Exp6180 companion artifact.",
    "run_date": "Pins the operator-declared adjudication date.",
    "preconditions_checked": "Snapshots Exp6166 bytes, stochastic seeds, tests, exclusion state, protected files, and git status before work.",
    "immutable_exp6166_byte_snapshot": "Records byte hashes for the Exp6166 result, source, tests, sampler spec, and exclusion manifest.",
    "no_refit_receipt": "States that Exp6166 fitting functions were not invoked and metrics came from historical artifact probabilities.",
    "stochastic_replay_receipt": "Replays deterministic pair-sample hashes for the frozen Exp6166 seeds without changing Exp6166.",
    "recomputed_metric_determination": "Recomputes declared TV, KL, support, normalization, and mode-ratio metrics from immutable probabilities.",
    "old_full_suite_failure_classification": "Classifies the historical full-suite exit 2 while preserving Exp6166's blocked state.",
    "companion_determination": "Separates the positive software-only stochastic determination from the historical blocked artifact.",
    "no_hardware_promotion_receipt": "Prevents stochastic software evidence from becoming hardware, latency, power, energy, or speedup evidence.",
    "protected_files_unchanged": "Confirms conductor and reconciler-owned files stayed byte-identical.",
    "before_after_byte_comparison": "Compares immutable Exp6166 and protected-file bytes before and after adjudication.",
    "field_provenance": "Maps every required field to the prompt, spec, immutable artifacts, tests, and command receipts.",
    "test_commands": "Records focused task-owned tests, new-code coverage, and the single classified full-suite receipt.",
    "test_exit_codes": "Stores command exit codes so failures cannot become silent readiness evidence.",
    "reproducibility_checksum": "Hashes artifact content with only duration and self-checksum blanked.",
    "honest_verdict": "Uses a terminal prefix and states both the reproduced software result and preserved historical block.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence with the same stable ordering as Exp6166."""

    return exp6166.canonical_json(value)


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value with the shared Carnot prefix."""

    return exp6166.sha256_json(value)


def sha256_file(path: Path) -> str:
    """Hash file bytes for immutable evidence receipts."""

    return exp6166.sha256_file(path)


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"_non_object": type(payload).__name__}


def _git_status_short(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def _file_receipt(repo_root: Path, rel_path: Path) -> JsonDict:
    path = repo_root / rel_path
    exists = path.exists()
    return {
        "path": rel_path.as_posix(),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": sha256_file(path) if exists else None,
    }


def _file_receipts(repo_root: Path, rel_paths: Sequence[Path]) -> dict[str, JsonDict]:
    return {path.as_posix(): _file_receipt(repo_root, path) for path in rel_paths}


def snapshot_preconditions(repo_root: Path = REPO_ROOT) -> JsonDict:
    """Capture the immutable evidence boundary before or after adjudication."""

    repo_root = repo_root.resolve()
    exp6166_artifact = _read_json(repo_root / EXP6166_RESULT_RELATIVE_PATH)
    return {
        "repo_root": str(repo_root),
        "exp6166_artifact_status": exp6166_artifact.get("status"),
        "exp6166_honest_verdict": exp6166_artifact.get("honest_verdict"),
        "immutable_paths": _file_receipts(repo_root, IMMUTABLE_EXP6166_PATHS),
        "stochastic_seeds": {
            "training_seeds": list(exp6166.TRAINING_SEEDS),
            "samples_per_seed": exp6166.SAMPLES_PER_SEED,
            "optimizer_steps": exp6166.OPTIMIZER_STEPS,
            "learning_rate": exp6166.LEARNING_RATE,
        },
        "test_surfaces": _file_receipts(
            repo_root,
            (exp6166.TEST_RELATIVE_PATH, TEST_RELATIVE_PATH),
        ),
        "exclusion_state": _file_receipt(repo_root, EXCLUSION_MANIFEST_RELATIVE_PATH),
        "protected_files": _file_receipts(repo_root, PROTECTED_FILES),
        "git_status_short": _git_status_short(repo_root),
    }


def _load_exp6166_artifact(repo_root: Path) -> JsonDict:
    return _read_json(repo_root / EXP6166_RESULT_RELATIVE_PATH)


def _load_exp6170_artifact(repo_root: Path) -> JsonDict:
    return _read_json(repo_root / EXP6170_RESULT_RELATIVE_PATH)


def _arm_probabilities(exp6166_artifact: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    arms = exp6166_artifact["exact_local_only_mode_jump_bad_and_permuted_arm_receipts"]["arms"]
    return {
        str(name): {str(label): float(value) for label, value in receipt["probabilities"].items()}
        for name, receipt in arms.items()
    }


def _metric_delta(left: float, right: float) -> float:
    if math.isinf(left) and math.isinf(right) and left == right:
        return 0.0
    return abs(left - right)


def _recompute_arm_metrics(
    program: Any,
    arm_name: str,
    probabilities: Mapping[str, float],
) -> JsonDict:
    arm = {
        "kernel": exp6166._kernel_from_probabilities(arm_name, probabilities),
        "probabilities": dict(probabilities),
    }
    factor = exp6166._factor_divergence(program, arm)
    exact = exp6166.exp6152.execute_exact(program)
    candidate = exp6166.exp6153.execute_joint_from_ebm_kernels(
        program,
        {"sample_multimodal_mode": arm["kernel"]},
    )
    joint = exp6166.exp6153.distribution_divergence(exact, candidate)
    mode_ratio = exp6166._mode_mass_ratio(probabilities)
    exact_ratio = exp6166._mode_mass_ratio(exp6166.EXACT_PROBABILITIES)
    return {
        "factor_tv": factor["weighted_tv"],
        "factor_kl_target_to_candidate": factor["weighted_kl"],
        "joint_tv": joint["joint_tv"],
        "joint_kl_target_to_candidate": joint["joint_kl_target_to_candidate"],
        "support_violation_count": joint["support_violation_count"],
        "normalization_error": candidate["normalization_error"],
        "support_count": candidate["support_count"],
        "mode_mass_ratio": mode_ratio,
        "mode_mass_ratio_error": abs(mode_ratio - exact_ratio),
    }


def recompute_declared_metrics_from_artifact(exp6166_artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute declared Exp6166 metrics from stored probabilities only."""

    program = exp6166.build_multimodal_factor_program()
    historical = exp6166_artifact["factor_and_joint_tv_kl_and_mode_mass_ratio_errors"]["arms"]
    recomputed: dict[str, JsonDict] = {}
    max_abs_delta = 0.0
    mismatch_count = 0
    compared_fields = (
        "factor_tv",
        "factor_kl_target_to_candidate",
        "joint_tv",
        "joint_kl_target_to_candidate",
        "support_violation_count",
        "normalization_error",
        "support_count",
        "mode_mass_ratio",
        "mode_mass_ratio_error",
    )
    for arm_name, probabilities in _arm_probabilities(exp6166_artifact).items():
        current = _recompute_arm_metrics(program, arm_name, probabilities)
        deltas = {
            field: _metric_delta(float(current[field]), float(historical[arm_name][field]))
            for field in compared_fields
        }
        arm_matches = all(delta <= exp6166.EXACT_TOLERANCE for delta in deltas.values())
        max_abs_delta = max(max_abs_delta, max(deltas.values()))
        mismatch_count += 0 if arm_matches else 1
        recomputed[arm_name] = {
            "metrics": current,
            "historical_metrics_sha256": sha256_json(
                {field: historical[arm_name][field] for field in compared_fields}
            ),
            "recomputed_metrics_sha256": sha256_json(current),
            "max_abs_delta": max(deltas.values()),
            "matches_historical": arm_matches,
        }
    local = recomputed["local_only"]["metrics"]
    jumped = recomputed["mode_jump"]["metrics"]
    improved = (
        jumped["joint_tv"] < local["joint_tv"]
        and jumped["mode_mass_ratio_error"] < local["mode_mass_ratio_error"]
    )
    return {
        "metric_source": "historical_artifact_probabilities",
        "training_or_fitting_invoked": False,
        "arms": recomputed,
        "max_abs_delta": max_abs_delta,
        "mismatch_count": mismatch_count,
        "all_declared_metrics_match": mismatch_count == 0,
        "mode_jump_improved_over_local_only": improved,
        "historical_mode_jump_improved_over_local_only": exp6166_artifact[
            "factor_and_joint_tv_kl_and_mode_mass_ratio_errors"
        ]["mode_jump_improved_over_local_only"],
        "software_oracle": "Exp6152 exact enumeration plus Exp6153 software kernel execution",
    }


def _pair_hash_for(extra_edges: Sequence[tuple[str, str]]) -> tuple[str, int]:
    pairs = [
        pair
        for seed in exp6166.TRAINING_SEEDS
        for pair in exp6166._sample_pairs(
            seed=seed,
            sample_count=exp6166.SAMPLES_PER_SEED,
            extra_edges=extra_edges,
        )
    ]
    return sha256_json(pairs), len(pairs)


def replay_stochastic_pair_hashes(exp6166_artifact: Mapping[str, Any]) -> JsonDict:
    """Replay deterministic training-pair hashes without optimizing scores."""

    arms = exp6166_artifact["exact_local_only_mode_jump_bad_and_permuted_arm_receipts"]["arms"]
    local_hash, local_count = _pair_hash_for(())
    jump_hash, jump_count = _pair_hash_for(exp6166.CROSS_MODE_EDGES)
    historical = {
        "local_only": {
            "pair_samples_sha256": arms["local_only"]["training"]["pair_samples_sha256"],
            "pair_sample_count": arms["local_only"]["training"]["pair_sample_count"],
        },
        "mode_jump": {
            "pair_samples_sha256": arms["mode_jump"]["training"]["pair_samples_sha256"],
            "pair_sample_count": arms["mode_jump"]["training"]["pair_sample_count"],
        },
    }
    replayed = {
        "local_only": {"pair_samples_sha256": local_hash, "pair_sample_count": local_count},
        "mode_jump": {"pair_samples_sha256": jump_hash, "pair_sample_count": jump_count},
    }
    return {
        "seeds": list(exp6166.TRAINING_SEEDS),
        "samples_per_seed": exp6166.SAMPLES_PER_SEED,
        "optimizer_steps_not_replayed": exp6166.OPTIMIZER_STEPS,
        "historical": historical,
        "replayed": replayed,
        "all_pair_hashes_match": historical == replayed,
    }


def no_refit_receipt() -> JsonDict:
    """Declare the narrow code path that keeps Exp6166 fitting disabled."""

    return {
        "training_functions_invoked": False,
        "forbidden_functions": ["train_matched_cnce_arms", "_train_probabilities"],
        "metric_source": "historical_artifact_probabilities",
        "allowed_replay": "deterministic_pair_hashes_and_exact_metric_recomputation",
    }


def classify_old_full_suite_failure(
    exp6166_artifact: Mapping[str, Any],
    exp6170_artifact: Mapping[str, Any],
) -> JsonDict:
    """Classify Exp6166's old full-suite exit without changing Exp6166."""

    exit_codes = dict(exp6166_artifact.get("test_exit_codes") or {})
    historical_exit = exit_codes.get(exp6166.GLOBAL_PYTEST_COMMAND)
    focused_zero = all(
        code == 0
        for command, code in exit_codes.items()
        if command != exp6166.GLOBAL_PYTEST_COMMAND
    )
    rows = exp6170_artifact.get("canary_failure_classification", {}).get("rows", [])
    matching_rows = [row for row in rows if row.get("name") == "required_full_python_suite_once"]
    exp6170_class = matching_rows[0].get("classification") if matching_rows else None
    if historical_exit in (None, 0):
        classification = "no_historical_full_suite_failure"
    elif historical_exit == 2 and focused_zero and exp6170_class == "unrelated_preexisting":
        classification = "unrelated_preexisting_repository_suite_failure"
    else:
        classification = "unclassified_historical_full_suite_failure"
    return {
        "historical_command": exp6166.GLOBAL_PYTEST_COMMAND,
        "historical_exit_code": historical_exit,
        "focused_exp6166_commands_all_zero": focused_zero,
        "exp6170_supporting_classification": exp6170_class,
        "classification": classification,
        "exp6166_status_before": exp6166_artifact.get("status"),
        "exp6166_status_after": exp6166_artifact.get("status"),
        "historical_block_preserved_exactly": exp6166_artifact.get("status") == "blocked",
    }


def no_hardware_promotion_receipt(exp6166_artifact: Mapping[str, Any]) -> JsonDict:
    """Preserve the software-only boundary of Exp6166."""

    return {
        "hardware_execution_claimed": False,
        "latency_power_energy_and_speedup_claimed": False,
        "exp6166_hardware_execution_claimed": exp6166_artifact.get("hardware_execution_claimed"),
        "exp6166_latency_power_energy_and_speedup_claimed": exp6166_artifact.get(
            "latency_power_energy_and_speedup_claimed"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "claim_scope": "software_only_stochastic_reproducibility_adjudication",
    }


def _compare_file_receipts(before: Mapping[str, Any], after: Mapping[str, Any]) -> JsonDict:
    rows: JsonDict = {}
    for key, before_receipt in before.items():
        after_receipt = dict(after.get(key) or {})
        rows[key] = {
            "before": before_receipt,
            "after": after_receipt,
            "unchanged": before_receipt == after_receipt,
        }
    return rows


def before_after_byte_comparison(before: Mapping[str, Any], after: Mapping[str, Any]) -> JsonDict:
    """Compare immutable Exp6166 and protected-file byte receipts."""

    immutable = _compare_file_receipts(before["immutable_paths"], after["immutable_paths"])
    protected = _compare_file_receipts(before["protected_files"], after["protected_files"])
    return {
        "exp6166_artifact": immutable[EXP6166_RESULT_RELATIVE_PATH.as_posix()],
        "immutable_paths": immutable,
        "protected_files": protected,
        "all_immutable_inputs_unchanged": all(row["unchanged"] for row in immutable.values()),
        "all_protected_files_unchanged": all(row["unchanged"] for row in protected.values()),
    }


def protected_files_receipt(comparison: Mapping[str, Any]) -> JsonDict:
    """Summarize protected operational files without editing them."""

    rows = dict(comparison["protected_files"])
    return {
        "paths": list(rows),
        "unchanged": all(row["unchanged"] for row in rows.values()),
        "rows": rows,
    }


def _command_maps(command_receipts: Sequence[Mapping[str, Any]]) -> tuple[JsonDict, JsonDict]:
    commands = {
        str(receipt.get("name", f"command_{index}")): str(receipt.get("command", ""))
        for index, receipt in enumerate(command_receipts)
    }
    exit_codes = {
        str(receipt.get("name", f"command_{index}")): int(receipt.get("exit_code", 0))
        for index, receipt in enumerate(command_receipts)
    }
    return commands, exit_codes


def _focused_receipts_pass(command_receipts: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        int(receipt.get("exit_code", 0)) == 0
        for receipt in command_receipts
        if receipt.get("name") != "required_full_python_suite_once"
    )


def companion_determination(
    exp6166_artifact: Mapping[str, Any],
    metrics: Mapping[str, Any],
    stochastic: Mapping[str, Any],
    old_failure: Mapping[str, Any],
) -> JsonDict:
    """Issue the companion verdict while preserving the old artifact state."""

    positive = (
        metrics.get("all_declared_metrics_match") is True
        and metrics.get("mode_jump_improved_over_local_only") is True
        and stochastic.get("all_pair_hashes_match") is True
    )
    return {
        "historical_exp6166_status_preserved": exp6166_artifact.get("status"),
        "historical_exp6166_honest_verdict_preserved": exp6166_artifact.get("honest_verdict"),
        "old_failure_classification": old_failure.get("classification"),
        "adjudicated_result": (
            "software_only_positive_reproducible"
            if positive
            else "software_only_reproducibility_not_established"
        ),
        "exp6166_rewritten": False,
        "hardware_claim_promoted": False,
    }


def field_provenance() -> JsonDict:
    """Map every artifact field to its specification and immutable sources."""

    sources = [
        "task_prompt",
        SPEC_RELATIVE_PATH.as_posix(),
        EXP6166_RESULT_RELATIVE_PATH.as_posix(),
        exp6166.MODULE_RELATIVE_PATH.as_posix(),
        exp6166.TEST_RELATIVE_PATH.as_posix(),
        EXP6170_RESULT_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {
            "spec": "REQ-SAMPLE-6180",
            "principle": principle,
            "sources": list(sources),
        }
        for field, principle in FIELD_PRINCIPLES.items()
    }


def status(artifact: Mapping[str, Any]) -> str:
    """Return Exp6180 status from immutable replay and command receipts."""

    old_failure = artifact["old_full_suite_failure_classification"]["classification"]
    hardware = artifact["no_hardware_promotion_receipt"]
    ready = (
        artifact["recomputed_metric_determination"]["all_declared_metrics_match"] is True
        and artifact["stochastic_replay_receipt"]["all_pair_hashes_match"] is True
        and old_failure
        in {
            "unrelated_preexisting_repository_suite_failure",
            "no_historical_full_suite_failure",
        }
        and hardware["hardware_execution_claimed"] is False
        and hardware["latency_power_energy_and_speedup_claimed"] is False
        and artifact["protected_files_unchanged"]["unchanged"] is True
        and artifact["before_after_byte_comparison"]["all_immutable_inputs_unchanged"] is True
        and _focused_receipts_pass(artifact.get("command_receipts", []))
    )
    return "complete_positive" if ready else "blocked"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build the terminal-prefixed verdict string."""

    if status(artifact) == "complete_positive":
        return (
            "complete_positive: Exp6166 software stochastic result reproduced from immutable "
            "evidence; historical Exp6166 artifact remains blocked"
        )
    return "blocked: Exp6166 companion adjudication did not satisfy immutable replay gates"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile fields."""

    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def build_adjudication_artifact(
    *,
    command_receipts: Sequence[Mapping[str, Any]],
    duration_s: float | None = None,
    before_snapshot: Mapping[str, Any] | None = None,
    repo_root: Path = REPO_ROOT,
) -> JsonDict:
    """Build the Exp6180 artifact from immutable Exp6166 evidence."""

    started = time.perf_counter()
    repo_root = repo_root.resolve()
    before = dict(before_snapshot or snapshot_preconditions(repo_root))
    exp6166_artifact = _load_exp6166_artifact(repo_root)
    exp6170_artifact = _load_exp6170_artifact(repo_root)
    metrics = recompute_declared_metrics_from_artifact(exp6166_artifact)
    stochastic = replay_stochastic_pair_hashes(exp6166_artifact)
    old_failure = classify_old_full_suite_failure(exp6166_artifact, exp6170_artifact)
    after = snapshot_preconditions(repo_root)
    comparison = before_after_byte_comparison(before, after)
    protected = protected_files_receipt(comparison)
    commands, exit_codes = _command_maps(command_receipts)
    elapsed = round(time.perf_counter() - started, 6) if duration_s is None else duration_s
    artifact: JsonDict = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "status": "blocked",
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "preconditions_checked": before,
        "immutable_exp6166_byte_snapshot": before["immutable_paths"],
        "no_refit_receipt": no_refit_receipt(),
        "stochastic_replay_receipt": stochastic,
        "recomputed_metric_determination": metrics,
        "old_full_suite_failure_classification": old_failure,
        "companion_determination": companion_determination(
            exp6166_artifact,
            metrics,
            stochastic,
            old_failure,
        ),
        "no_hardware_promotion_receipt": no_hardware_promotion_receipt(exp6166_artifact),
        "protected_files_unchanged": protected,
        "before_after_byte_comparison": comparison,
        "field_provenance": field_provenance(),
        "test_commands": commands,
        "test_exit_codes": exit_codes,
        "command_receipts": [dict(receipt) for receipt in command_receipts],
        "duration_s": elapsed,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def write_adjudication_artifact(
    *,
    output_path: Path | None = None,
    command_receipts: Sequence[Mapping[str, Any]],
    duration_s: float | None = None,
    before_snapshot: Mapping[str, Any] | None = None,
    repo_root: Path = REPO_ROOT,
) -> JsonDict:
    """Write the Exp6180 companion artifact without touching Exp6166 evidence."""

    output = output_path or repo_root / RESULT_RELATIVE_PATH
    artifact = build_adjudication_artifact(
        command_receipts=command_receipts,
        duration_s=duration_s,
        before_snapshot=before_snapshot,
        repo_root=repo_root,
    )
    validate_artifact(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the Exp6180 schema and replay gates."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    hardware = artifact["no_hardware_promotion_receipt"]
    if hardware.get("hardware_execution_claimed") is not False:
        raise ValueError("hardware")
    if hardware.get("latency_power_energy_and_speedup_claimed") is not False:
        raise ValueError("hardware_performance")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("checksum")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in FIELD_PRINCIPLES.items():
        if dict(provenance.get(field) or {}).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    return True
