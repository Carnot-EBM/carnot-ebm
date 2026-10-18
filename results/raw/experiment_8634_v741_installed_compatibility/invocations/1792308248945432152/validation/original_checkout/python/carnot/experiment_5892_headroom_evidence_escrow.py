"""Exp5892 headroom evidence escrow.

Spec refs: REQ-VERIFY-5892, SCENARIO-VERIFY-5892-IMMUTABLE-REPLAY,
SCENARIO-VERIFY-5892-ADMISSION-BOUNDARY,
SCENARIO-VERIFY-5892-NON-INTERFERENCE, SCENARIO-VERIFY-5892-FRESHNESS.

This module changes the admission boundary, not the historical evidence. It
replays the immutable Exp5868 rows, reuses the corrected Exp5879 taxonomy
calculation, and then admits the evidence only when Exp5892-owned checks pass
and any wider suite debt has exact node/path non-interference receipts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any

from carnot import experiment_5868_hardness_controlled_constraint_fixture as exp5868
from carnot import experiment_5869_hardness_surface_headroom_audit as exp5869
from carnot import experiment_5879_hardness_headroom_taxonomy_corrigendum as exp5879


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = exp5869.REPO_ROOT
RESULT_RELATIVE_PATH = Path("results/experiment_5892_headroom_evidence_escrow.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5892_headroom_evidence_escrow.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5892_headroom_evidence_escrow.py")
VERIFY_SPEC_RELATIVE_PATH = exp5869.VERIFY_SPEC_RELATIVE_PATH
RESEARCH_PROGRAM_RELATIVE_PATH = Path("research-program.md")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

SCHEMA = "carnot.experiment_5892.headroom_evidence_escrow.v1"
EXPERIMENT = 5892
EXPERIMENT_ID = "experiment_5892_headroom_evidence_escrow"
MILESTONE = "2026.07.524"
RUN_DATE = "20260724"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
VERIFIER_IS_ORACLE = False
SATURATION_CEILING_AUROC = exp5869.SATURATION_CEILING_AUROC
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
RAM_FLOOR_MB = exp5869.RAM_FLOOR_MB
DISK_FLOOR_MB = exp5869.DISK_FLOOR_MB

sha256_file = exp5869.sha256_file
sha256_json = exp5869.sha256_json
canonical_json = exp5869.canonical_json

UPSTREAM_SUMMARY_PATHS: dict[str, Path] = {
    "exp5868_summary": exp5869.UPSTREAM_ARTIFACT_RELATIVE_PATH,
    "exp5869_summary": exp5869.RESULT_RELATIVE_PATH,
    "exp5879_summary": exp5879.RESULT_RELATIVE_PATH,
}
UPSTREAM_ROW_PATH = exp5869.UPSTREAM_ROWS_RELATIVE_PATH
PROTECTED_RELATIVE_PATHS = (
    exp5869.UPSTREAM_ARTIFACT_RELATIVE_PATH,
    exp5869.UPSTREAM_ROWS_RELATIVE_PATH,
    exp5869.RESULT_RELATIVE_PATH,
    exp5879.RESULT_RELATIVE_PATH,
    RESEARCH_CONDUCTOR_RELATIVE_PATH,
)
HASHED_SOURCE_PATHS: dict[str, Path] = {
    "codex_instructions": Path("CODEX.md"),
    "claude_instructions": Path("CLAUDE.md"),
    "research_program": RESEARCH_PROGRAM_RELATIVE_PATH,
    "exp5869_module": exp5869.MODULE_RELATIVE_PATH,
    "exp5869_test": exp5869.TEST_RELATIVE_PATH,
    "exp5879_module": exp5879.MODULE_RELATIVE_PATH,
    "exp5879_test": exp5879.TEST_RELATIVE_PATH,
    "exp5892_module": MODULE_RELATIVE_PATH,
    "exp5892_test": TEST_RELATIVE_PATH,
    "verification_spec": VERIFY_SPEC_RELATIVE_PATH,
    "adversarial_verify": ADVERSARIAL_VERIFY_RELATIVE_PATH,
}
OWNED_NODE_PATHS = {
    TEST_RELATIVE_PATH.as_posix(),
    MODULE_RELATIVE_PATH.as_posix(),
    VERIFY_SPEC_RELATIVE_PATH.as_posix(),
    ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
    "scripts/root_clutter_sweep.py",
    RESEARCH_CONDUCTOR_RELATIVE_PATH.as_posix(),
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "immutable_upstream_hashes",
    "independent_row_and_certificate_replay",
    "taxonomy_and_oracle_boundary_receipt",
    "leakage_safe_split_receipts",
    "non_oracle_nuisance_metrics",
    "oracle_derived_diagnostic_metrics",
    "owned_check_receipts",
    "unrelated_global_debt_receipts",
    "gate_non_interference_receipts",
    "terminal_artifact_freshness_receipt",
    "headroom_admission_ready_score",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal admission state distinguishes clean escrow admission from blocked or null evidence.",
    "preconditions_checked": "Hash, solver, resource, verifier, output, and protected-file checks prevent stale or fabricated admission.",
    "immutable_upstream_hashes": "Admission may consume prior evidence but never rewrite it.",
    "independent_row_and_certificate_replay": "Row count, labels, witnesses, certificates, and relabels are replayed from rows rather than accepted from prose.",
    "taxonomy_and_oracle_boundary_receipt": "Nuisance controls and exact solver telemetry remain in disjoint authority classes.",
    "leakage_safe_split_receipts": "Semantic, family, relabel, and certificate groups cannot leak across train/test boundaries.",
    "non_oracle_nuisance_metrics": "Only oracle-distinct nuisance metrics determine the headroom ceiling.",
    "oracle_derived_diagnostic_metrics": "Exact solver telemetry is reported as circular diagnostics, not learned-energy credit.",
    "owned_check_receipts": "Every owned check has command, node, owner, exit, and mutability attribution.",
    "unrelated_global_debt_receipts": "Exact node/path evidence is required; a prose claim is insufficient.",
    "gate_non_interference_receipts": "Unrelated debt can be ignored only when rows, computations, schemas, and gates are unaffected.",
    "terminal_artifact_freshness_receipt": "Atomic terminal output must be newer or content-distinct from the pre-task target.",
    "headroom_admission_ready_score": "Emit bare `1.0` only for a clean non-retired admission artifact.",
    "protected_files_unchanged": "Operator-owned and immutable upstream files remain untouched.",
    "duration_s": "Measured runtime distinguishes real replay from bootstrap placeholders.",
    "inference_substrate": "Use `aggregation_from_upstream_artifacts`.",
    "verifier_is_oracle": "Use `false` for nuisance headroom; exact solver telemetry carries its own circular oracle flag.",
    "field_provenance": "Every field traces to task prompt, spec, source, tests, rows, artifacts, or solver receipts.",
    "test_commands": "Verification commands are part of the admission evidence.",
    "test_exit_codes": "Exit codes prevent failed checks from silently promoting.",
    "reproducibility_checksum": "A checksum detects upstream, taxonomy, check, or gate drift.",
    "honest_verdict": "Use `complete_ready:`, `complete_null:`, `retired:`, or `blocked:`.",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5892_headroom_evidence_escrow.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5892_headroom_evidence_escrow.py "
    "-m pytest tests/python/test_experiment_5892_headroom_evidence_escrow.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5892_headroom_evidence_escrow.py "
    "--fail-under=100",
    FULL_TEST_COMMAND,
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5892_headroom_evidence_escrow.py",
    ".venv/bin/python scripts/adversarial_verify.py --json "
    "results/experiment_5892_headroom_evidence_escrow.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\"",
)


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _hash_optional_file(root: Path, relative: Path) -> str:
    path = root / relative
    return sha256_file(path) if path.exists() and path.is_file() else "missing"


def _read_json(path: str | Path) -> JsonDict:
    return exp5869._read_json(path)


def _read_optional_json(root: Path, relative: Path) -> JsonDict:
    path = root / relative
    return _read_json(path) if path.exists() else {}


def read_upstream_rows(root: Path = REPO_ROOT) -> list[JsonDict]:
    """Read immutable Exp5868 rows through the existing row parser."""

    return exp5869.read_upstream_rows(root)


def _memory_probe() -> JsonDict:  # pragma: no cover - host-dependent resource probe.
    return exp5869._memory_probe()


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover - host-dependent resource probe.
    return exp5869._disk_probe(root)


def _path_stats(path: Path) -> JsonDict:
    if not path.exists():
        return {
            "exists": False,
            "sha256": "missing",
            "mtime_ns": None,
            "size_bytes": 0,
            "status": "missing",
        }
    status = "non_json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        status = str(payload.get("status", "json_without_status")) if isinstance(payload, Mapping) else "non_object_json"
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        status = "unreadable_json"
    stat = path.stat()
    return {
        "exists": True,
        "sha256": sha256_file(path),
        "mtime_ns": stat.st_mtime_ns,
        "size_bytes": stat.st_size,
        "status": status,
    }


def _output_path_receipt(result_path: Path) -> JsonDict:
    parent = result_path.parent
    writable = (parent.exists() and os.access(parent, os.W_OK)) or (
        parent.parent.exists() and os.access(parent.parent, os.W_OK)
    )
    prior = _path_stats(result_path)
    return {
        "schema": SCHEMA + ".output_path_receipt",
        "result_path": str(result_path),
        "result_writable": bool(writable),
        "atomic_checkpoint_suffix": ".tmp",
        "prior_exists": prior["exists"],
        "prior_sha256": prior["sha256"],
        "prior_mtime_ns": prior["mtime_ns"],
        "prior_size_bytes": prior["size_bytes"],
        "prior_status": prior["status"],
        "prior_was_bootstrap_running": prior["status"] == "running",
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
) -> JsonDict:
    """Hash every admission prerequisite before replaying the escrow."""

    root = Path(root)
    result_path = Path(result_path)
    upstream_gate = exp5869.upstream_gate_receipt(root)
    rows = read_upstream_rows(root) if upstream_gate.get("upstream_ready") is True else []
    split_definitions = exp5869.freeze_splits(rows) if rows else {"split_definition_hash": "missing"}
    split_receipt = exp5869.verify_split_leakage(rows, split_definitions) if rows else {
        "all_splits_leakage_safe": False
    }
    solver_receipts = exp5868.solver_version_receipts()
    solver_receipts["exp5868_module_sha256"] = _hash_optional_file(
        root,
        exp5868.MODULE_RELATIVE_PATH,
    )
    immutable_hashes = {
        name: _hash_optional_file(root, relative)
        for name, relative in {**UPSTREAM_SUMMARY_PATHS, "exp5868_rows": UPSTREAM_ROW_PATH}.items()
    }
    source_hashes = {
        name: _hash_optional_file(root, relative)
        for name, relative in HASHED_SOURCE_PATHS.items()
    }
    protected_hashes = {
        relative.as_posix(): _hash_optional_file(root, relative)
        for relative in PROTECTED_RELATIVE_PATHS
    }
    memory = memory_probe()
    disk = disk_probe(root)
    output_paths = _output_path_receipt(result_path)
    checks = {
        "all_three_upstream_summaries": all(
            immutable_hashes[name] != "missing" for name in UPSTREAM_SUMMARY_PATHS
        ),
        "exp5868_row_file": immutable_hashes["exp5868_rows"] != "missing",
        "upstream_exp5868_gate": upstream_gate.get("upstream_ready") is True,
        "split_definitions": bool(rows) and split_receipt.get("all_splits_leakage_safe") is True,
        "exact_solver_receipts": solver_receipts.get("ok") is True
        and solver_receipts["exp5868_module_sha256"] != "missing",
        "relevant_source_test_spec_hashes": all(value != "missing" for value in source_hashes.values()),
        "adversarial_verifier_available": source_hashes["adversarial_verify"] != "missing",
        "protected_files": all(value != "missing" for value in protected_hashes.values()),
        "memory": memory.get("ok") is True,
        "disk": disk.get("ok") is True,
        "output_path_atomic_writable": output_paths["result_writable"] is True,
        "no_bootstrap_running_output": output_paths["prior_was_bootstrap_running"] is False,
        "python": sys.version_info >= (3, 11),
    }
    blocked_reasons = [name for name, ok in checks.items() if not ok]
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "upstream_gate_receipt": upstream_gate,
        "split_definitions": split_definitions,
        "split_definition_hash": split_definitions.get("split_definition_hash"),
        "split_receipt_hash": sha256_json(split_receipt),
        "immutable_file_hashes": immutable_hashes,
        "source_hashes": source_hashes,
        "solver_version_receipts": solver_receipts,
        "solver_version_receipts_hash": sha256_json(solver_receipts),
        "resources": {"memory": memory, "disk": disk},
        "output_paths": output_paths,
        "protected_file_hashes": protected_hashes,
        "checks": checks,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "ok": sys.version_info >= (3, 11),
        },
        "preconditions_ready": not blocked_reasons,
        "blocked_reasons": sorted(blocked_reasons),
    }


def immutable_upstream_hashes(
    root: Path = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Record upstream evidence hashes without using upstream status as admission."""

    root = Path(root)
    before = dict((preconditions_checked or {}).get("immutable_file_hashes") or {})
    paths = {**UPSTREAM_SUMMARY_PATHS, "exp5868_rows": UPSTREAM_ROW_PATH}
    after = {name: _hash_optional_file(root, relative) for name, relative in paths.items()}
    summaries: dict[str, JsonDict] = {}
    for name, relative in UPSTREAM_SUMMARY_PATHS.items():
        payload = _read_optional_json(root, relative)
        summaries[name] = {
            "path": relative.as_posix(),
            "sha256": after[name],
            "status": payload.get("status", "missing"),
            "honest_verdict": payload.get("honest_verdict", "missing"),
            "ready_score": payload.get(
                "headroom_admission_ready_score",
                payload.get(
                    "hardness_surface_headroom_ready_score",
                    payload.get("hardness_controlled_fixture_ready_score"),
                ),
            ),
        }
    unchanged = {
        name: before.get(name) == value if before else value != "missing"
        for name, value in after.items()
    }
    return {
        "schema": SCHEMA + ".immutable_upstream_hashes",
        "principle": REQUIRED_FIELD_PRINCIPLES["immutable_upstream_hashes"],
        "summaries": summaries,
        "row_file": {
            "path": UPSTREAM_ROW_PATH.as_posix(),
            "sha256": after["exp5868_rows"],
        },
        "hashes": after,
        "precondition_hashes": before,
        "all_present": all(value != "missing" for value in after.values()),
        "unchanged_since_preconditions": unchanged,
        "all_unchanged_since_preconditions": all(unchanged.values()),
        "admission_consumes_prior_evidence_but_never_rewrites_it": True,
        "exp5879_status_is_not_admission_gate": True,
    }


def _empty_split_receipt() -> JsonDict:
    return {
        "schema": SCHEMA + ".leakage_safe_split_receipts",
        "splits_frozen_before_controls": False,
        "all_splits_leakage_safe": False,
        "duplicate_semantic_instances_across_splits": [],
        "instance_group_split": {"train_groups": [], "test_groups": []},
        "family_holdout_splits": {},
    }


def independent_row_and_certificate_replay(
    rows: Sequence[Mapping[str, Any]],
    upstream_artifact: Mapping[str, Any],
    split_receipt: Mapping[str, Any],
) -> JsonDict:
    """Replay row labels, certificates, witnesses, semantic groups, and stability."""

    integrity = exp5869.independent_row_integrity_replay(rows, upstream_artifact)
    stability = exp5869.relabel_and_certificate_group_controls(rows, split_receipt, integrity)
    semantic_groups = sorted({str(row.get("base_instance_id")) for row in rows})
    certificate_groups = sorted(
        {
            "|".join(
                [
                    str(row.get("base_instance_id")),
                    str(dict(row.get("certificate") or {}).get("kind")),
                    str(row.get("canonical_formula_hash")),
                ]
            )
            for row in rows
        }
    )
    witness_counts = Counter(str(dict(row.get("certificate") or {}).get("kind")) for row in rows)
    all_passed = bool(
        integrity.get("all_integrity_checks_passed") is True
        and stability.get("all_group_controls_passed") is True
        and len(rows) == 84
    )
    return {
        "schema": SCHEMA + ".independent_row_and_certificate_replay",
        "principle": REQUIRED_FIELD_PRINCIPLES["independent_row_and_certificate_replay"],
        "row_count": len(rows),
        "label_counts": dict(integrity.get("label_counts") or {}),
        "integrity_replay": integrity,
        "witness_replay": {
            "certificate_kind_counts": dict(sorted(witness_counts.items())),
            "certificate_failure_count": integrity.get("certificate_failure_count", 0),
            "certificate_failures": integrity.get("certificate_failures", []),
        },
        "semantic_group_count": len(semantic_groups),
        "semantic_groups_hash": sha256_json(semantic_groups),
        "certificate_group_count": len(certificate_groups),
        "certificate_groups_hash": sha256_json(certificate_groups),
        "relabel_stability_rate": stability.get("relabel_stability_rate", 0.0),
        "certificate_stability_rate": stability.get("certificate_stability_rate", 0.0),
        "stability_receipt": stability,
        "all_row_and_certificate_replay_passed": all_passed,
    }


def _empty_controls() -> tuple[JsonDict, JsonDict]:
    controls = {
        "schema": SCHEMA + ".empty_controls",
        "controls": {},
        "label_feature_used": False,
        "max_non_oracle_control_auroc": 0.5,
        "max_all_trivial_control_auroc": 0.5,
        "saturated_control_names": [],
        "non_oracle_controls_saturated": [],
    }
    no_info = {
        "schema": SCHEMA + ".empty_no_information_controls",
        "majority_control": {},
        "shuffled_label_control": {},
    }
    return controls, no_info


def taxonomy_and_metrics(
    rows: Sequence[Mapping[str, Any]],
    split_receipt: Mapping[str, Any],
    root: Path = REPO_ROOT,
) -> tuple[JsonDict, JsonDict, JsonDict]:
    """Compute the corrected taxonomy and its admission boundary receipts."""

    controls, no_info = (
        (exp5869.evaluate_trivial_controls(rows, split_receipt), exp5869.shuffled_and_majority_controls(rows, split_receipt))
        if rows
        else _empty_controls()
    )
    circularity = exp5879.current_verifier_circularity_matrix(rows)
    taxonomy = exp5879.control_taxonomy(controls, no_info)
    non_oracle = exp5879.non_oracle_nuisance_control_metrics(controls, no_info, taxonomy)
    oracle = exp5879.oracle_derived_diagnostic_metrics(controls, circularity, taxonomy)
    non_oracle.update(
        {
            "schema": SCHEMA + ".non_oracle_nuisance_metrics",
            "principle": REQUIRED_FIELD_PRINCIPLES["non_oracle_nuisance_metrics"],
            "verifier_is_oracle": False,
            "admission_gate_field": "headroom_admission_ready_score",
        }
    )
    oracle.update(
        {
            "schema": SCHEMA + ".oracle_derived_diagnostic_metrics",
            "principle": REQUIRED_FIELD_PRINCIPLES["oracle_derived_diagnostic_metrics"],
            "verifier_is_oracle": True,
            "oracle_telemetry_is_circular": True,
        }
    )
    exp5879_summary = _read_optional_json(Path(root), exp5879.RESULT_RELATIVE_PATH)
    boundary_ready = bool(
        taxonomy.get("all_features_assigned_once") is True
        and not taxonomy.get("class_overlap")
        and non_oracle.get("verifier_is_oracle") is False
        and oracle.get("verifier_is_oracle") is True
        and oracle.get("counts_as_learned_energy_win") is False
        and oracle.get("reduces_oracle_distinct_headroom") is False
    )
    boundary = {
        "schema": SCHEMA + ".taxonomy_and_oracle_boundary_receipt",
        "principle": REQUIRED_FIELD_PRINCIPLES["taxonomy_and_oracle_boundary_receipt"],
        "control_taxonomy": taxonomy,
        "current_verifier_circularity_matrix": circularity,
        "nuisance_and_oracle_classes_disjoint": boundary_ready,
        "oracle_telemetry_separately_flagged_circular": oracle.get("oracle_telemetry_is_circular") is True,
        "exp5879_status_observed": exp5879_summary.get("status", "missing"),
        "exp5879_honest_verdict_observed": exp5879_summary.get("honest_verdict", "missing"),
        "exp5879_status_is_admission_gate": False,
        "admission_gate_source": "independent_exp5892_replay",
        "boundary_ready": boundary_ready,
        "receipt_hash": sha256_json(
            {
                "taxonomy": taxonomy,
                "non_oracle": non_oracle.get("receipt_hash"),
                "oracle": oracle.get("receipt_hash"),
            }
        ),
    }
    return boundary, non_oracle, oracle


def extract_pytest_nodes(output: str) -> list[str]:
    """Return exact pytest node ids from failure output."""

    pattern = re.compile(r"\b(?:FAILED|ERROR)\s+([^\s]+?\.py(?:::[^\s]+)*)")
    nodes = list(pattern.findall(output))
    if "Segmentation fault" in output:
        stack_pattern = re.compile(r'File ".+?/carnot/([^"]+?\.py)"')
        nodes.extend(
            f"{path}::segmentation_fault"
            for path in stack_pattern.findall(output)
            if path.startswith("python/carnot/")
        )
    return sorted(set(nodes))


def _command_node_id(command: str) -> str:
    if command == FULL_TEST_COMMAND:
        return "global-python-suite"
    if "coverage run" in command or "coverage report" in command:
        return "exp5892-owned-coverage"
    if "check_spec_coverage.py" in command:
        return "exp5892-spec-coverage"
    if TEST_RELATIVE_PATH.as_posix() in command:
        return "exp5892-owned-unit"
    if "adversarial_verify.py" in command:
        return "exp5892-adversarial-verify"
    if "root_clutter_sweep.py" in command:
        return "exp5892-root-clutter"
    if "research_conductor.py" in command:
        return "exp5892-protected-file"
    return "command:" + sha256_json(command)[7:19]


def _command_owner_path(command: str) -> str:
    if "coverage run" in command or "coverage report" in command:
        return MODULE_RELATIVE_PATH.as_posix()
    for path in (
        TEST_RELATIVE_PATH,
        VERIFY_SPEC_RELATIVE_PATH,
        ADVERSARIAL_VERIFY_RELATIVE_PATH,
        Path("scripts/root_clutter_sweep.py"),
        RESEARCH_CONDUCTOR_RELATIVE_PATH,
    ):
        if path.as_posix() in command:
            return path.as_posix()
    return "tests/python" if command == FULL_TEST_COMMAND else "unknown"


def _owned_path(owner_path: str) -> bool:
    return owner_path in OWNED_NODE_PATHS or owner_path == TEST_RELATIVE_PATH.as_posix()


def _failure_receipt(
    *,
    command: str,
    node_id: str,
    owner_path: str,
    path_owner: str,
    exit_code: int,
    owned: bool,
    exact_node_path_evidence: bool,
) -> JsonDict:
    can_alter_schemas = owned and owner_path == VERIFY_SPEC_RELATIVE_PATH.as_posix()
    can_alter_audit = owned and owner_path == MODULE_RELATIVE_PATH.as_posix()
    can_alter_gate = owned
    return {
        "command": command,
        "node_id": node_id,
        "path_owner": path_owner,
        "owner_path": owner_path,
        "exit_code": int(exit_code),
        "owned": owned,
        "exact_node_path_evidence": exact_node_path_evidence,
        "can_alter_rows": False if not owned else owner_path == UPSTREAM_ROW_PATH.as_posix(),
        "can_alter_audit_computations": bool(can_alter_audit),
        "can_alter_schemas": bool(can_alter_schemas),
        "can_alter_gate_fields": bool(can_alter_gate),
        "receipt_id": sha256_json(
            {
                "command": command,
                "node_id": node_id,
                "owner_path": owner_path,
                "exit_code": int(exit_code),
                "owned": owned,
            }
        ),
    }


def owned_check_receipts(
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    """Attribute focused Exp5892-owned commands and their failures."""

    receipts = []
    failures = []
    for command in [str(value) for value in test_commands if str(value) != FULL_TEST_COMMAND]:
        exit_code = int(test_exit_codes.get(command, 1))
        owner_path = _command_owner_path(command)
        receipt = _failure_receipt(
            command=command,
            node_id=_command_node_id(command),
            owner_path=owner_path,
            path_owner="exp5892_owned",
            exit_code=exit_code,
            owned=True,
            exact_node_path_evidence=True,
        )
        receipt["check_passed"] = exit_code == 0
        receipts.append(receipt)
        if exit_code != 0:
            failures.append(receipt)
    return {
        "schema": SCHEMA + ".owned_check_receipts",
        "principle": REQUIRED_FIELD_PRINCIPLES["owned_check_receipts"],
        "owned_commands": [receipt["command"] for receipt in receipts],
        "check_receipts": receipts,
        "failure_receipts": failures,
        "owned_checks_passed": not failures
        and len(receipts) == len([command for command in test_commands if command != FULL_TEST_COMMAND]),
    }


def unrelated_global_debt_receipts(
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
    check_outputs: Mapping[str, str] | None = None,
) -> JsonDict:
    """Classify full-suite failures only when exact node/path evidence exists."""

    outputs = dict(check_outputs or {})
    if FULL_TEST_COMMAND not in test_commands:
        return {
            "schema": SCHEMA + ".unrelated_global_debt_receipts",
            "principle": REQUIRED_FIELD_PRINCIPLES["unrelated_global_debt_receipts"],
            "full_suite_command": FULL_TEST_COMMAND,
            "full_suite_exit_code": None,
            "classification": "full_suite_not_run",
            "unrelated_failures_present": False,
            "failure_receipts": [],
            "owned_failure_receipts": [],
            "missing_exact_node_path_evidence": True,
        }
    exit_code = int(test_exit_codes.get(FULL_TEST_COMMAND, 1))
    if exit_code == 0:
        return {
            "schema": SCHEMA + ".unrelated_global_debt_receipts",
            "principle": REQUIRED_FIELD_PRINCIPLES["unrelated_global_debt_receipts"],
            "full_suite_command": FULL_TEST_COMMAND,
            "full_suite_exit_code": 0,
            "classification": "global_suite_clean",
            "unrelated_failures_present": False,
            "failure_receipts": [],
            "owned_failure_receipts": [],
            "missing_exact_node_path_evidence": False,
        }
    nodes = extract_pytest_nodes(outputs.get(FULL_TEST_COMMAND, ""))
    if not nodes:
        missing = _failure_receipt(
            command=FULL_TEST_COMMAND,
            node_id="missing_exact_node_path_evidence",
            owner_path="",
            path_owner="unknown_global_suite",
            exit_code=exit_code,
            owned=False,
            exact_node_path_evidence=False,
        )
        return {
            "schema": SCHEMA + ".unrelated_global_debt_receipts",
            "principle": REQUIRED_FIELD_PRINCIPLES["unrelated_global_debt_receipts"],
            "full_suite_command": FULL_TEST_COMMAND,
            "full_suite_exit_code": exit_code,
            "classification": "missing_exact_node_path_evidence",
            "unrelated_failures_present": True,
            "failure_receipts": [missing],
            "owned_failure_receipts": [],
            "missing_exact_node_path_evidence": True,
        }
    unrelated = []
    owned_failures = []
    for node in nodes:
        owner_path = node.split("::", 1)[0]
        owned = _owned_path(owner_path)
        receipt = _failure_receipt(
            command=FULL_TEST_COMMAND,
            node_id=node,
            owner_path=owner_path,
            path_owner="exp5892_owned" if owned else "unrelated_global_suite",
            exit_code=exit_code,
            owned=owned,
            exact_node_path_evidence=True,
        )
        if owned:
            owned_failures.append(receipt)
        else:
            unrelated.append(receipt)
    return {
        "schema": SCHEMA + ".unrelated_global_debt_receipts",
        "principle": REQUIRED_FIELD_PRINCIPLES["unrelated_global_debt_receipts"],
        "full_suite_command": FULL_TEST_COMMAND,
        "full_suite_exit_code": exit_code,
        "classification": "owned_failure_inside_global_suite" if owned_failures else "unrelated_global_suite_debt",
        "unrelated_failures_present": bool(unrelated),
        "failure_receipts": unrelated,
        "owned_failure_receipts": owned_failures,
        "missing_exact_node_path_evidence": False,
    }


def gate_non_interference_receipts(unrelated_debt: Mapping[str, Any]) -> JsonDict:
    """Prove unrelated failures cannot influence rows, computations, schemas, or gates."""

    receipts = []
    for failure in list(unrelated_debt.get("failure_receipts") or []):
        exact = failure.get("exact_node_path_evidence") is True
        safe = bool(
            exact
            and failure.get("can_alter_rows") is False
            and failure.get("can_alter_audit_computations") is False
            and failure.get("can_alter_schemas") is False
            and failure.get("can_alter_gate_fields") is False
        )
        receipts.append(
            {
                "receipt_id": failure.get("receipt_id"),
                "node_id": failure.get("node_id"),
                "owner_path": failure.get("owner_path"),
                "command": failure.get("command"),
                "exit_code": failure.get("exit_code"),
                "non_interference_passed": safe,
                "rows_unchanged_by_failure": failure.get("can_alter_rows") is False,
                "audit_computations_unchanged_by_failure": failure.get(
                    "can_alter_audit_computations"
                )
                is False,
                "schemas_unchanged_by_failure": failure.get("can_alter_schemas") is False,
                "gate_fields_unchanged_by_failure": failure.get("can_alter_gate_fields") is False,
            }
        )
    all_safe = bool(
        unrelated_debt.get("missing_exact_node_path_evidence") is False
        and not unrelated_debt.get("owned_failure_receipts")
        and all(receipt["non_interference_passed"] for receipt in receipts)
    )
    return {
        "schema": SCHEMA + ".gate_non_interference_receipts",
        "principle": REQUIRED_FIELD_PRINCIPLES["gate_non_interference_receipts"],
        "receipts": receipts,
        "all_unrelated_failures_safe": all_safe,
        "owned_failures_inside_global_suite": list(unrelated_debt.get("owned_failure_receipts") or []),
        "missing_exact_node_path_evidence": unrelated_debt.get("missing_exact_node_path_evidence") is True,
    }


def protected_files_unchanged(root: Path, preconditions_checked: Mapping[str, Any]) -> JsonDict:
    """Verify immutable upstream and operator-owned files kept their precondition hashes."""

    before = dict(preconditions_checked.get("protected_file_hashes") or {})
    after = {relative.as_posix(): _hash_optional_file(Path(root), relative) for relative in PROTECTED_RELATIVE_PATHS}
    changed = sorted(path for path, value in after.items() if before.get(path) != value)
    return {
        "schema": SCHEMA + ".protected_files_unchanged",
        "principle": REQUIRED_FIELD_PRINCIPLES["protected_files_unchanged"],
        "before_hashes": before,
        "after_hashes": after,
        "changed_files": changed,
        "all_unchanged": not changed and all(value != "missing" for value in after.values()),
    }


def _initial_terminal_freshness(
    result_path: Path,
    preconditions_checked: Mapping[str, Any],
    *,
    atomic_write_completed: bool,
) -> JsonDict:
    output = dict(preconditions_checked.get("output_paths") or {})
    before_exists = output.get("prior_exists") is True
    return {
        "schema": SCHEMA + ".terminal_artifact_freshness_receipt",
        "principle": REQUIRED_FIELD_PRINCIPLES["terminal_artifact_freshness_receipt"],
        "result_path": str(result_path),
        "pre_task_exists": before_exists,
        "pre_task_sha256": output.get("prior_sha256", "missing"),
        "pre_task_mtime_ns": output.get("prior_mtime_ns"),
        "pre_task_status": output.get("prior_status", "missing"),
        "atomic_checkpoint_suffix": output.get("atomic_checkpoint_suffix", ".tmp"),
        "atomic_write_completed": atomic_write_completed,
        "bootstrap_running_artifact_written": False,
        "first_atomic_write_sha256": "pending" if atomic_write_completed else "not_written",
        "first_atomic_write_mtime_ns": None,
        "first_atomic_write_size_bytes": 0,
        "hash_changed_in_task": not before_exists and atomic_write_completed,
        "mtime_changed_in_task": not before_exists and atomic_write_completed,
        "normalized_payload_sha256": "",
        "freshness_ready": atomic_write_completed
        and output.get("prior_was_bootstrap_running") is False,
        "self_referential_hash_policy": "normalized_payload_sha256 blanks self-referential freshness and checksum fields",
    }


def _stable_for_reproducibility(artifact: Mapping[str, Any]) -> JsonDict:
    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    if isinstance(stable.get("preconditions_checked"), dict):
        stable["preconditions_checked"]["output_paths"] = {}
    if isinstance(stable.get("terminal_artifact_freshness_receipt"), dict):
        stable["terminal_artifact_freshness_receipt"].update(
            {
                "result_path": "",
                "pre_task_exists": False,
                "pre_task_sha256": "",
                "pre_task_mtime_ns": None,
                "pre_task_status": "",
                "first_atomic_write_sha256": "",
                "first_atomic_write_mtime_ns": None,
                "first_atomic_write_size_bytes": 0,
                "hash_changed_in_task": True,
                "mtime_changed_in_task": True,
                "normalized_payload_sha256": "",
            }
        )
    return stable


def normalized_payload_sha256(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while blanking self-referential freshness fields."""

    return sha256_json(_stable_for_reproducibility(artifact))


def _refresh_freshness_hash(artifact: JsonDict) -> None:
    freshness = dict(artifact.get("terminal_artifact_freshness_receipt") or {})
    freshness["normalized_payload_sha256"] = normalized_payload_sha256(artifact)
    artifact["terminal_artifact_freshness_receipt"] = freshness


def _update_terminal_freshness_after_first_write(artifact: JsonDict, result_path: Path) -> None:
    freshness = dict(artifact.get("terminal_artifact_freshness_receipt") or {})
    after = _path_stats(result_path)
    before_sha = freshness.get("pre_task_sha256")
    before_mtime = freshness.get("pre_task_mtime_ns")
    freshness.update(
        {
            "first_atomic_write_sha256": after["sha256"],
            "first_atomic_write_mtime_ns": after["mtime_ns"],
            "first_atomic_write_size_bytes": after["size_bytes"],
            "hash_changed_in_task": before_sha != after["sha256"],
            "mtime_changed_in_task": before_mtime != after["mtime_ns"],
            "freshness_ready": after["exists"] is True
            and freshness.get("atomic_write_completed") is True
            and freshness.get("bootstrap_running_artifact_written") is False
            and (before_sha != after["sha256"] or before_mtime != after["mtime_ns"]),
        }
    )
    artifact["terminal_artifact_freshness_receipt"] = freshness
    _refresh_freshness_hash(artifact)


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        VERIFY_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        exp5869.UPSTREAM_ARTIFACT_RELATIVE_PATH.as_posix(),
        exp5869.UPSTREAM_ROWS_RELATIVE_PATH.as_posix(),
        exp5869.RESULT_RELATIVE_PATH.as_posix(),
        exp5879.RESULT_RELATIVE_PATH.as_posix(),
        exp5869.MODULE_RELATIVE_PATH.as_posix(),
        exp5879.MODULE_RELATIVE_PATH.as_posix(),
        ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": principle, "sources": sources}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    checks = {
        "immutable_upstream_hashes": dict(artifact.get("immutable_upstream_hashes") or {}).get("all_present") is True,
        "independent_row_and_certificate_replay": dict(
            artifact.get("independent_row_and_certificate_replay") or {}
        ).get("all_row_and_certificate_replay_passed")
        is True,
        "leakage_safe_split_receipts": dict(artifact.get("leakage_safe_split_receipts") or {}).get("all_splits_leakage_safe") is True,
        "taxonomy_and_oracle_boundary_receipt": dict(
            artifact.get("taxonomy_and_oracle_boundary_receipt") or {}
        ).get("boundary_ready")
        is True,
        "owned_check_receipts": dict(artifact.get("owned_check_receipts") or {}).get("owned_checks_passed") is True,
        "gate_non_interference_receipts": dict(
            artifact.get("gate_non_interference_receipts") or {}
        ).get("all_unrelated_failures_safe")
        is True,
        "terminal_artifact_freshness_receipt": dict(
            artifact.get("terminal_artifact_freshness_receipt") or {}
        ).get("freshness_ready")
        is True,
        "protected_files_unchanged": dict(artifact.get("protected_files_unchanged") or {}).get("all_unchanged") is True,
        "inference_substrate": artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "verifier_is_oracle": artifact.get("verifier_is_oracle") is False,
    }
    reasons.extend(name for name, ok in checks.items() if not ok)
    return sorted(set(reasons))


def headroom_admission_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return bare readiness only for clean, non-retired admission artifacts."""

    non_oracle = dict(artifact.get("non_oracle_nuisance_metrics") or {})
    oracle = dict(artifact.get("oracle_derived_diagnostic_metrics") or {})
    ready = bool(
        artifact.get("status") != "retired"
        and dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and dict(artifact.get("immutable_upstream_hashes") or {}).get("all_present") is True
        and dict(artifact.get("immutable_upstream_hashes") or {}).get(
            "all_unchanged_since_preconditions"
        )
        is True
        and dict(artifact.get("independent_row_and_certificate_replay") or {}).get(
            "all_row_and_certificate_replay_passed"
        )
        is True
        and dict(artifact.get("leakage_safe_split_receipts") or {}).get(
            "all_splits_leakage_safe"
        )
        is True
        and dict(artifact.get("taxonomy_and_oracle_boundary_receipt") or {}).get(
            "boundary_ready"
        )
        is True
        and non_oracle.get("no_non_oracle_nuisance_control_exceeds_ceiling") is True
        and float(non_oracle.get("max_non_oracle_nuisance_auroc", 1.0))
        < SATURATION_CEILING_AUROC
        and non_oracle.get("verifier_is_oracle") is False
        and oracle.get("verifier_is_oracle") is True
        and oracle.get("oracle_telemetry_is_circular") is True
        and oracle.get("counts_as_learned_energy_win") is False
        and oracle.get("reduces_oracle_distinct_headroom") is False
        and dict(artifact.get("owned_check_receipts") or {}).get("owned_checks_passed") is True
        and dict(artifact.get("gate_non_interference_receipts") or {}).get(
            "all_unrelated_failures_safe"
        )
        is True
        and dict(artifact.get("terminal_artifact_freshness_receipt") or {}).get(
            "freshness_ready"
        )
        is True
        and dict(artifact.get("protected_files_unchanged") or {}).get("all_unchanged") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is False
    )
    return 1.0 if ready else 0.0


def _hard_blocked(artifact: Mapping[str, Any]) -> bool:
    return bool(_blocked_reasons(artifact))


def status(artifact: Mapping[str, Any]) -> str:
    """Compute the terminal admission status."""

    if artifact.get("status") == "retired":
        return "retired"
    if headroom_admission_ready_score(artifact) == 1.0:
        return "complete_ready"
    if _hard_blocked(artifact):
        return "blocked"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build the terminal-prefix verdict required by the admission contract."""

    terminal = status(artifact)
    if terminal == "complete_ready":
        return "complete_ready: headroom_evidence_escrow_admitted"
    if terminal == "retired":
        return "retired: headroom_evidence_escrow_not_admitted"
    if terminal == "complete_null":
        saturated = ",".join(
            dict(artifact.get("non_oracle_nuisance_metrics") or {}).get(
                "saturated_control_names"
            )
            or []
        )
        return "complete_null: non_oracle_nuisance_saturation=" + saturated
    return "blocked: " + ",".join(_blocked_reasons(artifact)[:8])


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable evidence while blanking host-variable and self-reference fields."""

    return sha256_json(_stable_for_reproducibility(artifact))


def _finalize_terminal_fields(artifact: JsonDict) -> None:
    _refresh_freshness_hash(artifact)
    artifact["headroom_admission_ready_score"] = headroom_admission_ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
    check_outputs: Mapping[str, str] | None,
    duration_s: float,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    atomic_write_completed: bool = False,
) -> JsonDict:
    """Build the escrow artifact from already-read immutable rows."""

    root = Path(root)
    result_path = Path(result_path)
    split_definitions = dict(preconditions_checked.get("split_definitions") or {})
    splits = exp5869.verify_split_leakage(rows, split_definitions) if rows else _empty_split_receipt()
    upstream = _read_optional_json(root, exp5869.UPSTREAM_ARTIFACT_RELATIVE_PATH)
    replay = independent_row_and_certificate_replay(rows, upstream, splits)
    boundary, non_oracle, oracle = taxonomy_and_metrics(rows, splits, root)
    owned = owned_check_receipts(test_commands, test_exit_codes)
    unrelated = unrelated_global_debt_receipts(test_commands, test_exit_codes, check_outputs)
    non_interference = gate_non_interference_receipts(unrelated)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "blocked",
        "preconditions_checked": dict(preconditions_checked),
        "immutable_upstream_hashes": immutable_upstream_hashes(root, preconditions_checked),
        "independent_row_and_certificate_replay": replay,
        "taxonomy_and_oracle_boundary_receipt": boundary,
        "leakage_safe_split_receipts": splits,
        "non_oracle_nuisance_metrics": non_oracle,
        "oracle_derived_diagnostic_metrics": oracle,
        "owned_check_receipts": owned,
        "unrelated_global_debt_receipts": unrelated,
        "gate_non_interference_receipts": non_interference,
        "terminal_artifact_freshness_receipt": _initial_terminal_freshness(
            result_path,
            preconditions_checked,
            atomic_write_completed=atomic_write_completed,
        ),
        "headroom_admission_ready_score": 0.0,
        "protected_files_unchanged": protected_files_unchanged(root, preconditions_checked),
        "duration_s": float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "test_commands": [str(command) for command in test_commands],
        "test_exit_codes": {str(command): int(code) for command, code in test_exit_codes.items()},
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: artifact_not_validated",
    }
    _finalize_terminal_fields(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate required fields, checksum, terminal status, and admission score."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing_fields:{missing}")
    if artifact.get("headroom_admission_ready_score") != headroom_admission_ready_score(artifact):
        raise ValueError("headroom_admission_ready_score")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    if str(artifact.get("status")) == "running":
        raise ValueError("status")
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
    check_outputs: Mapping[str, str] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    """Run Exp5892, optionally writing the terminal JSON artifact atomically."""

    started = time.perf_counter()
    root = Path(root)
    result_path = Path(result_path)
    preconditions = dict(
        preconditions_checked
        or collect_preconditions(root=root, result_path=result_path)
    )
    rows = read_upstream_rows(root)
    elapsed = round(time.perf_counter() - started, 6) if duration_s is None else float(duration_s)
    exit_codes = dict(test_exit_codes or {command: 0 for command in test_commands})
    artifact = build_artifact(
        rows=rows,
        preconditions_checked=preconditions,
        test_commands=test_commands,
        test_exit_codes=exit_codes,
        check_outputs=check_outputs,
        duration_s=elapsed,
        root=root,
        result_path=result_path,
        atomic_write_completed=write,
    )
    if write:
        _atomic_write(result_path, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        _update_terminal_freshness_after_first_write(artifact, result_path)
        _finalize_terminal_fields(artifact)
        _atomic_write(result_path, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    artifact = run(result_path=args.result, write=True)
    print(json.dumps({"status": artifact["status"], "result": args.result}, sort_keys=True))
    return 0 if artifact["status"] == "complete_ready" else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
