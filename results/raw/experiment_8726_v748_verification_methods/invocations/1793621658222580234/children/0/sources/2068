"""Exp5879 hardness-headroom taxonomy corrigendum.

Spec refs: REQ-VERIFY-5879, SCENARIO-VERIFY-5879-TAXONOMY,
SCENARIO-VERIFY-5879-HEADROOM, SCENARIO-VERIFY-5879-BLOCKED-DEBT.

This module does not regenerate Exp5868 rows and does not rewrite Exp5869.
It replays the checked-in exact fixture, preserves the old mixed-control gate
as historical evidence, and then separates non-oracle nuisance controls from
oracle-derived solver telemetry so solver authority cannot masquerade as a
learned-energy result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import argparse
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5869_hardness_surface_headroom_audit as exp5869


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = exp5869.REPO_ROOT
RESULT_RELATIVE_PATH = Path("results/experiment_5879_hardness_headroom_taxonomy_corrigendum.json")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5879_hardness_headroom_taxonomy_corrigendum.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5879_hardness_headroom_taxonomy_corrigendum.py"
)
SCHEMA = "carnot.experiment_5879.hardness_headroom_taxonomy_corrigendum.v1"
EXPERIMENT = 5879
EXPERIMENT_ID = "experiment_5879_hardness_headroom_taxonomy_corrigendum"
MILESTONE = "2026.07.524"
RUN_DATE = "20260724"
INFERENCE_SUBSTRATE = exp5869.INFERENCE_SUBSTRATE
VERIFIER_IS_ORACLE = True
SATURATION_CEILING_AUROC = exp5869.SATURATION_CEILING_AUROC
from carnot.global_suite_baseline import delta as global_suite_delta

FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"

sha256_file = exp5869.sha256_file
sha256_json = exp5869.sha256_json
canonical_json = exp5869.canonical_json

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "immutable_upstream_hashes",
    "original_gate_reproduction",
    "independent_row_integrity_replay",
    "leakage_safe_split_receipts",
    "control_taxonomy",
    "non_oracle_nuisance_control_metrics",
    "oracle_derived_diagnostic_metrics",
    "current_verifier_circularity_matrix",
    "relabel_and_certificate_stability",
    "saturation_and_skip_decision",
    "oracle_distinct_evaluation_design",
    "held_model_and_constraint_plan",
    "test_debt_classification",
    "protected_files_unchanged",
    "hardness_surface_headroom_ready_score",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES = {
    "control_taxonomy": "every feature belongs to exactly one authority-aware class.",
    "non_oracle_nuisance_control_metrics": "only these metrics determine shortcut saturation.",
    "oracle_derived_diagnostic_metrics": "solver telemetry is useful context but circular for learned-energy credit.",
    "saturation_and_skip_decision": "expensive downstream work must fail fast on leakage or failed checks.",
    "hardness_surface_headroom_ready_score": "emit a bare scalar for structured gates.",
    "inference_substrate": "use `deterministic_control_audit_no_llm`.",
    "verifier_is_oracle": "record a per-path authority matrix.",
    "honest_verdict": "use `complete_ready:`, `complete_null:`, `retired:`, or `blocked:`.",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5879_hardness_headroom_taxonomy_corrigendum.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5879_hardness_headroom_taxonomy_corrigendum.py "
    "-m pytest tests/python/test_experiment_5879_hardness_headroom_taxonomy_corrigendum.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5879_hardness_headroom_taxonomy_corrigendum.py "
    "--fail-under=100",
    FULL_TEST_COMMAND,
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5879_hardness_headroom_taxonomy_corrigendum.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5879_hardness_headroom_taxonomy_corrigendum.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\"",
)


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _hash_optional_file(root: Path, relative: Path) -> str:
    return exp5869._hash_optional_file(root, relative)


def _read_json(path: str | Path) -> JsonDict:
    return exp5869._read_json(path)


def read_upstream_rows(root: Path = REPO_ROOT) -> list[JsonDict]:
    """Read the immutable Exp5868 row file through the Exp5869 replay helper."""

    return exp5869.read_upstream_rows(root)


def _output_path_receipt(result_path: Path) -> JsonDict:
    receipt = exp5869._output_path_receipt(result_path)
    receipt["schema"] = SCHEMA + ".output_path_receipt"
    return receipt


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    memory_probe: MemoryProbe = exp5869._memory_probe,
    disk_probe: DiskProbe = exp5869._disk_probe,
) -> JsonDict:
    """Hash prerequisites before the taxonomy corrigendum is built."""

    root = Path(root)
    result_path = Path(result_path)
    base = exp5869.collect_preconditions(
        root=root,
        result_path=result_path,
        memory_probe=memory_probe,
        disk_probe=disk_probe,
    )
    code_hashes = {
        "exp5869_module": _hash_optional_file(root, exp5869.MODULE_RELATIVE_PATH),
        "exp5869_test": _hash_optional_file(root, exp5869.TEST_RELATIVE_PATH),
        "exp5879_module": _hash_optional_file(root, MODULE_RELATIVE_PATH),
        "exp5879_test": _hash_optional_file(root, TEST_RELATIVE_PATH),
        "verification_spec": _hash_optional_file(root, exp5869.VERIFY_SPEC_RELATIVE_PATH),
        "adversarial_verify": _hash_optional_file(root, Path("scripts/adversarial_verify.py")),
    }
    checks = dict(base.get("checks") or {})
    checks["exp5879_code_hashes"] = all(value != "missing" for value in code_hashes.values())
    blocked_reasons = list(base.get("blocked_reasons") or [])
    if not checks["exp5879_code_hashes"]:
        blocked_reasons.append("missing_exp5879_module_test_or_spec")
    base.update(
        {
            "schema": SCHEMA + ".preconditions",
            "run_date": RUN_DATE,
            "code_hashes": code_hashes,
            "output_paths": _output_path_receipt(result_path),
            "checks": checks,
            "preconditions_ready": base.get("preconditions_ready") is True
            and checks["exp5879_code_hashes"],
            "blocked_reasons": sorted(set(blocked_reasons)),
        }
    )
    return base


def immutable_upstream_hashes(root: Path = REPO_ROOT) -> JsonDict:
    """Record immutable inputs and operator-protected code hashes."""

    root = Path(root)
    paths = {
        "exp5868_summary": exp5869.UPSTREAM_ARTIFACT_RELATIVE_PATH,
        "exp5868_rows": exp5869.UPSTREAM_ROWS_RELATIVE_PATH,
        "exp5869_artifact": exp5869.RESULT_RELATIVE_PATH,
        "exp5869_module": exp5869.MODULE_RELATIVE_PATH,
        "exp5869_test": exp5869.TEST_RELATIVE_PATH,
        "exp5879_module": MODULE_RELATIVE_PATH,
        "exp5879_test": TEST_RELATIVE_PATH,
        "verification_spec": exp5869.VERIFY_SPEC_RELATIVE_PATH,
        "adversarial_verify": Path("scripts/adversarial_verify.py"),
        "research_conductor": Path("scripts/research_conductor.py"),
    }
    hashes = {name: _hash_optional_file(root, relative) for name, relative in paths.items()}
    hashes["verify_registry"] = _hash_optional_file(root, exp5869.VERIFIER_REGISTRY_RELATIVE_PATH)
    hashes["verify_dir_hash_root"] = exp5869._verifier_config_receipt(root)[
        "verifier_file_hash_root"
    ]
    return {
        "schema": SCHEMA + ".immutable_upstream_hashes",
        "hashes": hashes,
        "all_present": all(value != "missing" for value in hashes.values()),
        "protected_files": ["scripts/research_conductor.py"],
        "immutable_artifacts_not_rewritten": True,
    }


def original_gate_reproduction(root: Path = REPO_ROOT) -> JsonDict:
    """Reproduce the old mixed-control gate before applying the corrigendum."""

    root = Path(root)
    historical = _read_json(root / exp5869.RESULT_RELATIVE_PATH)
    combined = exp5869.run(
        root=root,
        result_path=root / exp5869.RESULT_RELATIVE_PATH,
        test_exit_codes={command: 0 for command in exp5869.DEFAULT_TEST_COMMANDS},
        duration_s=0.0,
        write=False,
    )
    historical_exit_codes = dict(historical.get("test_exit_codes") or {})
    return {
        "schema": SCHEMA + ".original_gate_reproduction",
        "historical_exp5869_artifact_sha256": _hash_optional_file(
            root,
            exp5869.RESULT_RELATIVE_PATH,
        ),
        "historical_status": historical.get("status"),
        "historical_honest_verdict": historical.get("honest_verdict"),
        "historical_full_suite_exit_code": historical_exit_codes.get(FULL_TEST_COMMAND),
        "recomputed_original_status_with_owned_tests_assumed_zero": combined["status"],
        "recomputed_original_honest_verdict": combined["honest_verdict"],
        "recomputed_combined_no_trivial_control_exceeds_ceiling": combined[
            "saturation_and_skip_decision"
        ]["no_trivial_control_exceeds_ceiling"],
        "recomputed_combined_saturated_controls": combined["saturation_and_skip_decision"][
            "saturated_control_names"
        ],
        "recomputed_max_non_oracle_control_auroc": combined[
            "density_length_width_name_and_order_controls"
        ]["max_non_oracle_control_auroc"],
        "recomputed_max_all_trivial_control_auroc": combined[
            "density_length_width_name_and_order_controls"
        ]["max_all_trivial_control_auroc"],
        "old_gate_conflated_oracle_telemetry_with_nuisance_features": True,
    }


def control_taxonomy(
    control_receipt: Mapping[str, Any],
    no_info_receipt: Mapping[str, Any],
) -> JsonDict:
    """Assign every evaluated feature to one authority-aware class."""

    controls = dict(control_receipt.get("controls") or {})
    non_oracle = sorted(
        name for name, metric in controls.items() if dict(metric).get("oracle_derived") is False
    )
    if no_info_receipt.get("majority_control"):
        non_oracle.append("majority_control")
    if no_info_receipt.get("shuffled_label_control"):
        non_oracle.append("shuffled_label_control")
    oracle = sorted(
        name for name, metric in controls.items() if dict(metric).get("oracle_derived") is True
    )
    oracle.extend(["certificate_checker_results", "direct_solver_outputs"])
    excluded = [
        "certificate_kind_as_label_proxy",
        "expected_label",
        "solver_result_label",
    ]
    overlap = sorted(set(non_oracle) & set(oracle))
    assigned = non_oracle + oracle
    return {
        "schema": SCHEMA + ".control_taxonomy",
        "classes": {
            "non_oracle_nuisance": sorted(non_oracle),
            "oracle_derived_diagnostic": sorted(oracle),
            "excluded_label_proxy": excluded,
        },
        "class_overlap": overlap,
        "assigned_feature_count": len(assigned),
        "unique_assigned_feature_count": len(set(assigned)),
        "all_features_assigned_once": not overlap and len(assigned) == len(set(assigned)),
        "excluded_label_proxy_features": excluded,
        "taxonomy_hash": sha256_json({"assigned": sorted(assigned), "excluded": excluded}),
    }


def non_oracle_nuisance_control_metrics(
    control_receipt: Mapping[str, Any],
    no_info_receipt: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
) -> JsonDict:
    """Freeze nuisance controls as the only shortcut-saturation source."""

    controls = dict(control_receipt.get("controls") or {})
    names = list(dict(taxonomy.get("classes") or {}).get("non_oracle_nuisance") or [])
    metrics: dict[str, JsonDict] = {}
    for name in names:
        if name in controls:
            metric = dict(controls[name])
            auroc = float(metric.get("orientation_free_auroc", 0.5))
            metrics[name] = {
                "authority_class": "non_oracle_nuisance",
                "feature_source": metric.get("feature_source", name),
                "verifier_is_oracle": False,
                "counts_for_shortcut_saturation": True,
                "uses_exact_label_feature": metric.get("uses_exact_label_feature") is True,
                "all_rows_auroc": metric.get("all_rows_auroc", 0.5),
                "orientation_free_auroc": auroc,
                "split_test_balanced_error_rate": metric.get(
                    "split_test_balanced_error_rate",
                    0.5,
                ),
                "saturated": auroc > SATURATION_CEILING_AUROC,
            }
    majority = dict(no_info_receipt.get("majority_control") or {})
    if "majority_control" in names:
        metrics["majority_control"] = {
            "authority_class": "non_oracle_nuisance",
            "feature_source": "majority_label_from_train_split",
            "verifier_is_oracle": False,
            "counts_for_shortcut_saturation": True,
            "uses_exact_label_feature": False,
            "all_rows_auroc": 0.5,
            "orientation_free_auroc": 0.5,
            "split_test_balanced_error_rate": majority.get("balanced_error_rate", 0.5),
            "saturated": False,
        }
    shuffled = dict(no_info_receipt.get("shuffled_label_control") or {})
    if "shuffled_label_control" in names:
        shuffled_auc = float(shuffled.get("orientation_free_auroc", 0.5))
        metrics["shuffled_label_control"] = {
            "authority_class": "non_oracle_nuisance",
            "feature_source": "shuffled_labels_no_information_control",
            "verifier_is_oracle": False,
            "counts_for_shortcut_saturation": True,
            "uses_exact_label_feature": False,
            "all_rows_auroc": shuffled.get("all_rows_auroc", 0.5),
            "orientation_free_auroc": shuffled_auc,
            "split_test_balanced_error_rate": 0.5,
            "saturated": shuffled_auc > SATURATION_CEILING_AUROC,
        }
    max_auroc = max(
        (float(metric["orientation_free_auroc"]) for metric in metrics.values()), default=0.5
    )
    saturated = sorted(name for name, metric in metrics.items() if metric["saturated"])
    return {
        "schema": SCHEMA + ".non_oracle_nuisance_control_metrics",
        "principle": REQUIRED_FIELD_PRINCIPLES["non_oracle_nuisance_control_metrics"],
        "saturation_ceiling_auroc": SATURATION_CEILING_AUROC,
        "control_names": sorted(metrics),
        "control_metrics": dict(sorted(metrics.items())),
        "max_non_oracle_nuisance_auroc": round(max_auroc, 6),
        "saturated_control_names": saturated,
        "no_non_oracle_nuisance_control_exceeds_ceiling": not saturated,
        "label_feature_used": any(
            metric["uses_exact_label_feature"] for metric in metrics.values()
        ),
        "saturation_source": "non_oracle_nuisance_only",
        "receipt_hash": sha256_json(metrics),
    }


def oracle_derived_diagnostic_metrics(
    control_receipt: Mapping[str, Any],
    circularity: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
) -> JsonDict:
    """Report solver/checker telemetry without awarding learned-energy credit."""

    controls = dict(control_receipt.get("controls") or {})
    names = list(dict(taxonomy.get("classes") or {}).get("oracle_derived_diagnostic") or [])
    metrics: dict[str, JsonDict] = {}
    for name in names:
        if name in controls:
            metric = dict(controls[name])
            auroc = float(metric.get("orientation_free_auroc", 0.5))
            metrics[name] = {
                "authority_class": "oracle_derived_diagnostic",
                "feature_source": metric.get("feature_source", name),
                "verifier_is_oracle": True,
                "counts_as_learned_energy_win": False,
                "reduces_oracle_distinct_headroom": False,
                "all_rows_auroc": metric.get("all_rows_auroc", 0.5),
                "orientation_free_auroc": auroc,
                "split_test_balanced_error_rate": metric.get(
                    "split_test_balanced_error_rate",
                    0.5,
                ),
                "diagnostic_saturated": auroc > SATURATION_CEILING_AUROC,
            }
    paths = dict(circularity.get("paths") or {})
    solver_accuracies = [
        float(path.get("accuracy", 0.0))
        for path in paths.values()
        if path.get("executes_same_exact_solver") is True
    ]
    if "direct_solver_outputs" in names:
        accuracy = round(max(solver_accuracies, default=0.0), 6)
        metrics["direct_solver_outputs"] = {
            "authority_class": "oracle_derived_diagnostic",
            "feature_source": "solver_results.label",
            "verifier_is_oracle": True,
            "counts_as_learned_energy_win": False,
            "reduces_oracle_distinct_headroom": False,
            "all_rows_auroc": accuracy,
            "orientation_free_auroc": accuracy,
            "split_test_balanced_error_rate": 0.0 if accuracy == 1.0 else 0.5,
            "diagnostic_saturated": accuracy > SATURATION_CEILING_AUROC,
        }
    certificate = dict(paths.get("certificate_checker") or {})
    if "certificate_checker_results" in names:
        accuracy = float(certificate.get("accuracy", 0.0))
        metrics["certificate_checker_results"] = {
            "authority_class": "oracle_derived_diagnostic",
            "feature_source": "certificate_checker",
            "verifier_is_oracle": True,
            "counts_as_learned_energy_win": False,
            "reduces_oracle_distinct_headroom": False,
            "all_rows_auroc": accuracy,
            "orientation_free_auroc": accuracy,
            "split_test_balanced_error_rate": 0.0 if accuracy == 1.0 else 0.5,
            "diagnostic_saturated": accuracy > SATURATION_CEILING_AUROC,
        }
    max_auroc = max(
        (float(metric["orientation_free_auroc"]) for metric in metrics.values()), default=0.5
    )
    saturated = sorted(name for name, metric in metrics.items() if metric["diagnostic_saturated"])
    return {
        "schema": SCHEMA + ".oracle_derived_diagnostic_metrics",
        "principle": REQUIRED_FIELD_PRINCIPLES["oracle_derived_diagnostic_metrics"],
        "control_names": sorted(metrics),
        "control_metrics": dict(sorted(metrics.items())),
        "max_oracle_derived_auroc": round(max_auroc, 6),
        "saturated_diagnostic_names": saturated,
        "verifier_is_oracle": True,
        "counts_as_learned_energy_win": False,
        "reduces_oracle_distinct_headroom": False,
        "diagnostic_only": True,
        "receipt_hash": sha256_json(metrics),
    }


def current_verifier_circularity_matrix(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Add a compact authority matrix to the exact verifier replay receipt."""

    matrix = exp5869.current_verifier_circularity_matrix(rows)
    paths = dict(matrix.get("paths") or {})
    matrix["schema"] = SCHEMA + ".current_verifier_circularity_matrix"
    matrix["authority_matrix"] = {
        name: {
            "verifier_is_oracle": path.get("verifier_is_oracle") is True,
            "counts_as_oracle_distinct_headroom": path.get("counts_as_oracle_distinct_headroom")
            is True,
        }
        for name, path in paths.items()
    }
    matrix["non_oracle_nuisance_controls_are_oracle"] = False
    return matrix


def relabel_and_certificate_stability(
    rows: Sequence[Mapping[str, Any]],
    split_receipt: Mapping[str, Any],
    integrity_receipt: Mapping[str, Any],
) -> JsonDict:
    """Preserve relabel and certificate stability under the corrected taxonomy."""

    receipt = exp5869.relabel_and_certificate_group_controls(
        rows,
        split_receipt,
        integrity_receipt,
    )
    receipt["schema"] = SCHEMA + ".relabel_and_certificate_stability"
    receipt["stability_hash"] = sha256_json(
        {
            "relabel": receipt.get("relabel_stability_rate"),
            "certificate": receipt.get("certificate_stability_rate"),
            "cross_split": [
                receipt.get("relabel_groups_cross_split"),
                receipt.get("certificate_groups_cross_split"),
            ],
        }
    )
    return receipt


def saturation_and_skip_decision(
    integrity: Mapping[str, Any],
    splits: Mapping[str, Any],
    non_oracle: Mapping[str, Any],
    stability: Mapping[str, Any],
    held_plan: Mapping[str, Any],
) -> JsonDict:
    """Gate downstream science only on non-oracle controls and owned checks."""

    gates = {
        "row_integrity": integrity.get("all_integrity_checks_passed") is True,
        "group_safe_splits": splits.get("all_splits_leakage_safe") is True,
        "non_oracle_nuisance_below_ceiling": non_oracle.get(
            "no_non_oracle_nuisance_control_exceeds_ceiling"
        )
        is True,
        "relabel_and_certificate_stability": stability.get("all_group_controls_passed") is True,
        "held_model_and_constraint_plan": held_plan.get("nonempty_plan") is True,
        "label_feature_absent": non_oracle.get("label_feature_used") is False,
    }
    science_ready = all(gates.values())
    return {
        "schema": SCHEMA + ".saturation_and_skip_decision",
        "principle": REQUIRED_FIELD_PRINCIPLES["saturation_and_skip_decision"],
        "saturation_ceiling_auroc": SATURATION_CEILING_AUROC,
        "gate_receipts": gates,
        "no_non_oracle_nuisance_control_exceeds_ceiling": gates[
            "non_oracle_nuisance_below_ceiling"
        ],
        "saturated_non_oracle_control_names": list(non_oracle.get("saturated_control_names") or []),
        "skip_model_extraction_for_science": not science_ready,
        "skip_reason": ""
        if science_ready
        else "integrity_leakage_stability_or_nuisance_gate_failed",
        "hardness_surface_headroom_ready_score": 1.0 if science_ready else 0.0,
    }


def classify_test_debt(
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
    *,
    science_matrix_ready: bool,
    global_failure_node_ids: Sequence[str] | None = None,
) -> JsonDict:
    """Separate Exp5879-owned checks from unrelated global suite debt.

    `global_failure_node_ids` is what the global suite ACTUALLY failed on. None means no
    evidence was recorded, which is not the same as "nothing failed" -- see the fail-closed
    note below.
    """

    commands = [str(command) for command in test_commands]
    exit_codes = {str(command): int(code) for command, code in test_exit_codes.items()}
    failed = {command: code for command, code in exit_codes.items() if code != 0}
    owned_commands = [command for command in commands if command != FULL_TEST_COMMAND]
    owned_failed = {
        command: exit_codes.get(command, 1)
        for command in owned_commands
        if exit_codes.get(command, 1) != 0
    }
    full_suite_exit = exit_codes.get(FULL_TEST_COMMAND)
    unrelated_global_suite_debt = (
        science_matrix_ready
        and not owned_failed
        and full_suite_exit is not None
        and full_suite_exit != 0
    )
    # REQ-HARNESS-5920 (wired 2026-08-29). This function ALREADY separated owned checks from
    # unrelated global debt and correctly labelled this case `unrelated_global_suite_debt` --
    # and then blocked on it anyway, so the classification had no consequence. The verdict has
    # read `blocked: science_ready_but_unrelated_global_suite_debt` while the science score was
    # 1.0, which is the module saying its own work is done and something else is in the way.
    #
    # The spec's answer is a node-id delta: unrelated debt blocks ONLY when a NEW failing node
    # id appeared, meaning this task caused it. The suite still runs in full and every failure
    # stays visible; nothing is suppressed, deselected or relabelled.
    #
    # Fails CLOSED. If the delta cannot be computed -- no node-id evidence recorded, or an
    # unreadable baseline -- `ready_allowed` is False and the debt blocks exactly as before.
    global_delta = global_suite_delta(global_failure_node_ids or [])
    debt_is_a_regression = not (
        global_delta.get("ready_allowed") is True and global_failure_node_ids is not None
    )
    return {
        "schema": SCHEMA + ".test_debt_classification",
        "science_matrix_ready": bool(science_matrix_ready),
        "owned_commands": owned_commands,
        "owned_checks_passed": not owned_failed
        and all(command in exit_codes for command in owned_commands),
        "owned_failed_commands": owned_failed,
        "full_suite_command": FULL_TEST_COMMAND,
        "full_suite_exit_code": full_suite_exit,
        "failed_commands": failed,
        "unrelated_global_suite_debt": unrelated_global_suite_debt,
        "global_suite_failure_delta": global_delta,
        "unrelated_debt_is_a_regression": bool(debt_is_a_regression),
        "blocks_terminal_ready_status": bool(
            owned_failed or (unrelated_global_suite_debt and debt_is_a_regression)
        ),
        "classification": "unrelated_global_suite_debt"
        if unrelated_global_suite_debt
        else ("owned_check_failure" if owned_failed else "checks_clean"),
    }


def protected_files_unchanged(
    root: Path,
    preconditions_checked: Mapping[str, Any],
) -> JsonDict:
    """Verify protected operator-owned files kept their precondition hashes."""

    receipt = exp5869.protected_files_unchanged(root, preconditions_checked)
    receipt["schema"] = SCHEMA + ".protected_files_unchanged"
    return receipt


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        exp5869.VERIFY_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        exp5869.UPSTREAM_ARTIFACT_RELATIVE_PATH.as_posix(),
        exp5869.UPSTREAM_ROWS_RELATIVE_PATH.as_posix(),
        exp5869.RESULT_RELATIVE_PATH.as_posix(),
        exp5869.MODULE_RELATIVE_PATH.as_posix(),
        exp5869.TEST_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": principle, "sources": sources}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def _owned_tests_pass(artifact: Mapping[str, Any]) -> bool:
    return dict(artifact.get("test_debt_classification") or {}).get("owned_checks_passed") is True


def hardness_surface_headroom_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return the bare science score; unrelated full-suite debt is status-only."""

    decision = dict(artifact.get("saturation_and_skip_decision") or {})
    non_oracle = dict(artifact.get("non_oracle_nuisance_control_metrics") or {})
    ready = bool(
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and dict(artifact.get("immutable_upstream_hashes") or {}).get("all_present") is True
        and dict(artifact.get("independent_row_integrity_replay") or {}).get(
            "all_integrity_checks_passed"
        )
        is True
        and dict(artifact.get("leakage_safe_split_receipts") or {}).get("all_splits_leakage_safe")
        is True
        and non_oracle.get("no_non_oracle_nuisance_control_exceeds_ceiling") is True
        and float(non_oracle.get("max_non_oracle_nuisance_auroc", 1.0)) < SATURATION_CEILING_AUROC
        and dict(artifact.get("relabel_and_certificate_stability") or {}).get(
            "all_group_controls_passed"
        )
        is True
        and dict(artifact.get("oracle_distinct_evaluation_design") or {}).get(
            "nonempty_held_model_and_constraint_design"
        )
        is True
        and dict(artifact.get("held_model_and_constraint_plan") or {}).get("nonempty_plan") is True
        and dict(artifact.get("current_verifier_circularity_matrix") or {}).get(
            "all_exact_paths_marked_oracle"
        )
        is True
        and dict(artifact.get("oracle_derived_diagnostic_metrics") or {}).get(
            "counts_as_learned_energy_win"
        )
        is False
        and decision.get("hardness_surface_headroom_ready_score") == 1.0
        and dict(artifact.get("protected_files_unchanged") or {}).get("all_unchanged") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and _owned_tests_pass(artifact)
    )
    return 1.0 if ready else 0.0


def _hard_blocked(artifact: Mapping[str, Any]) -> bool:
    return not bool(
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and dict(artifact.get("immutable_upstream_hashes") or {}).get("all_present") is True
        and dict(artifact.get("independent_row_integrity_replay") or {}).get(
            "all_integrity_checks_passed"
        )
        is True
        and dict(artifact.get("leakage_safe_split_receipts") or {}).get("all_splits_leakage_safe")
        is True
        and dict(artifact.get("protected_files_unchanged") or {}).get("all_unchanged") is True
        and _owned_tests_pass(artifact)
    )


def status(artifact: Mapping[str, Any]) -> str:
    """Return the terminal status while keeping suite debt separate."""

    score = hardness_surface_headroom_ready_score(artifact)
    debt = dict(artifact.get("test_debt_classification") or {})
    # Key on the DECISION, not the CLASSIFICATION. `unrelated_global_suite_debt` says the debt
    # exists; `blocks_terminal_ready_status` says whether it is this task's fault, which is the
    # question REQ-HARNESS-5920 asks. Keying on the classification is what made the first fix
    # hollow: the decision field was computed correctly and then read by nothing.
    #
    # Falls back to the classification when the decision field is absent, so an artifact
    # written before this change still reads exactly as it did.
    blocks = debt.get("blocks_terminal_ready_status")
    if blocks is None:
        blocks = debt.get("unrelated_global_suite_debt")
    if score == 1.0 and blocks is True:
        return "blocked"
    if score == 1.0:
        return "complete_ready"
    if _hard_blocked(artifact):
        return "blocked"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build the required terminal-prefix verdict."""

    terminal = status(artifact)
    if terminal == "complete_ready":
        return "complete_ready: hardness_headroom_taxonomy_corrigendum_ready"
    if terminal == "complete_null":
        saturated = ",".join(
            dict(artifact.get("non_oracle_nuisance_control_metrics") or {}).get(
                "saturated_control_names"
            )
            or []
        )
        return "complete_null: non_oracle_nuisance_saturation=" + saturated
    if hardness_surface_headroom_ready_score(artifact) == 1.0:
        return "blocked: science_ready_but_unrelated_global_suite_debt"
    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    if not _owned_tests_pass(artifact):
        reasons.append("owned_test_exit_codes")
    return "blocked: " + ",".join(sorted(set(reasons))[:8])


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact after blanking host-variable and self-reference fields."""

    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    if isinstance(stable.get("preconditions_checked"), dict):
        stable["preconditions_checked"]["output_paths"] = {}
    return sha256_json(stable)


def _taxonomy_valid(taxonomy: Mapping[str, Any]) -> bool:
    classes = dict(taxonomy.get("classes") or {})
    non_oracle = set(classes.get("non_oracle_nuisance") or [])
    oracle = set(classes.get("oracle_derived_diagnostic") or [])
    return (
        taxonomy.get("all_features_assigned_once") is True
        and not taxonomy.get("class_overlap")
        and not (non_oracle & oracle)
    )


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
    duration_s: float,
    root: Path = REPO_ROOT,
    global_failure_node_ids: Sequence[str] | None = None,
) -> JsonDict:
    """Build the terminal corrigendum artifact from already-read rows.

    `global_failure_node_ids` is what the global suite actually failed on. None means no
    evidence, which is NOT the same as no failures: `classify_test_debt` then fails closed and
    the debt blocks exactly as it did before REQ-HARNESS-5920 was wired here.

    THIS PARAMETER EXISTS BECAUSE ITS ABSENCE MADE THE FIRST WIRING HOLLOW (2026-08-29).
    `classify_test_debt` gained the delta, and this caller never passed it, so production took
    the fail-closed path on every run and nothing changed. The verification quoted in that
    commit tested `classify_test_debt` directly with evidence handed in -- a path production
    never takes. Test the call site, not the function.
    """

    root = Path(root)
    split_definitions = dict(
        preconditions_checked.get("split_definitions") or exp5869.freeze_splits(rows)
    )
    splits = exp5869.verify_split_leakage(rows, split_definitions)
    upstream = exp5869.read_upstream_artifact(root)
    integrity = exp5869.independent_row_integrity_replay(rows, upstream)
    controls = exp5869.evaluate_trivial_controls(rows, splits)
    no_info = exp5869.shuffled_and_majority_controls(rows, splits)
    circularity = current_verifier_circularity_matrix(rows)
    taxonomy = control_taxonomy(controls, no_info)
    non_oracle = non_oracle_nuisance_control_metrics(controls, no_info, taxonomy)
    oracle = oracle_derived_diagnostic_metrics(controls, circularity, taxonomy)
    stability = relabel_and_certificate_stability(rows, splits, integrity)
    design = exp5869.oracle_distinct_evaluation_design()
    held_plan = exp5869.held_family_and_constraint_cell_plan(rows)
    held_plan["schema"] = SCHEMA + ".held_model_and_constraint_plan"
    decision = saturation_and_skip_decision(integrity, splits, non_oracle, stability, held_plan)
    test_debt = classify_test_debt(
        test_commands,
        test_exit_codes,
        science_matrix_ready=decision["hardness_surface_headroom_ready_score"] == 1.0,
        global_failure_node_ids=global_failure_node_ids,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "blocked",
        "preconditions_checked": dict(preconditions_checked),
        "immutable_upstream_hashes": immutable_upstream_hashes(root),
        "original_gate_reproduction": original_gate_reproduction(root),
        "independent_row_integrity_replay": integrity,
        "leakage_safe_split_receipts": splits,
        "control_taxonomy": taxonomy,
        "non_oracle_nuisance_control_metrics": non_oracle,
        "oracle_derived_diagnostic_metrics": oracle,
        "current_verifier_circularity_matrix": circularity,
        "relabel_and_certificate_stability": stability,
        "saturation_and_skip_decision": decision,
        "oracle_distinct_evaluation_design": design,
        "held_model_and_constraint_plan": held_plan,
        "test_debt_classification": test_debt,
        "protected_files_unchanged": protected_files_unchanged(root, preconditions_checked),
        "hardness_surface_headroom_ready_score": 0.0,
        "duration_s": float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": {str(command): int(code) for command, code in test_exit_codes.items()},
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: artifact_not_validated",
    }
    artifact["hardness_surface_headroom_ready_score"] = hardness_surface_headroom_ready_score(
        artifact
    )
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate schema, taxonomy exclusivity, checksum, score, and status."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing_fields:{missing}")
    if not _taxonomy_valid(dict(artifact.get("control_taxonomy") or {})):
        raise ValueError("control_taxonomy")
    if artifact.get(
        "hardness_surface_headroom_ready_score"
    ) != hardness_surface_headroom_ready_score(artifact):
        raise ValueError("hardness_surface_headroom_ready_score")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    return True


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    """Run Exp5879, optionally writing the terminal JSON artifact."""

    started = time.perf_counter()
    root = Path(root)
    result_path = Path(result_path)
    preconditions = dict(
        preconditions_checked or collect_preconditions(root=root, result_path=result_path)
    )
    rows = read_upstream_rows(root)
    elapsed = round(time.perf_counter() - started, 6) if duration_s is None else float(duration_s)
    artifact = build_artifact(
        rows=rows,
        preconditions_checked=preconditions,
        test_commands=test_commands,
        test_exit_codes=dict(test_exit_codes or {command: 0 for command in test_commands}),
        duration_s=elapsed,
        root=root,
    )
    if write:
        _atomic_write(result_path, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    artifact = run(result_path=args.result, write=True)
    print(json.dumps({"status": artifact["status"], "result": args.result}, sort_keys=True))
    return 0 if artifact["status"] != "blocked" else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
