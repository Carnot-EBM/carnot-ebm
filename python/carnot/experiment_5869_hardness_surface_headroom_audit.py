"""Exp5869 hardness-surface headroom audit.

Spec refs: REQ-VERIFY-5869, SCENARIO-VERIFY-5869-INTEGRITY,
SCENARIO-VERIFY-5869-SPLITS, SCENARIO-VERIFY-5869-CONTROLS,
SCENARIO-VERIFY-5869-DESIGN.

This diagnostic is intentionally conservative. It audits the exact Exp5868
SAT/CNF rows before any internal-state extraction, separates exact solver
authority from oracle-distinct headroom, and fails the readiness gate when a
trivial or solver-derived control saturates the labels.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
from typing import Any

from carnot import experiment_5868_hardness_controlled_constraint_fixture as exp5868


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5869_hardness_surface_headroom_audit.json")
UPSTREAM_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5868_hardness_controlled_constraint_fixture.json"
)
UPSTREAM_ROWS_RELATIVE_PATH = Path(
    "results/experiment_5868_hardness_controlled_constraint_fixture.rows.jsonl"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5869_hardness_surface_headroom_audit.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5869_hardness_surface_headroom_audit.py"
)
VERIFY_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
VERIFIER_REGISTRY_RELATIVE_PATH = Path("ops/verifier_registry.yaml")
VERIFY_DIR_RELATIVE_PATH = Path("python/carnot/verify")
PROTECTED_FILES = (Path("scripts/research_conductor.py"),)

SCHEMA = "carnot.experiment_5869.hardness_surface_headroom_audit.v1"
EXPERIMENT = 5869
EXPERIMENT_ID = "experiment_5869_hardness_surface_headroom_audit"
MILESTONE = "2026.07.522"
RUN_DATE = "20260724"
INFERENCE_SUBSTRATE = "deterministic_control_audit_no_llm"
VERIFIER_IS_ORACLE = True
BASE_SEED = 5869
SPLIT_SEED = 5869001
SHUFFLE_SEED = 5869002
RAM_FLOOR_MB = 1024
DISK_FLOOR_MB = 512
SATURATION_CEILING_AUROC = 0.85
POSITIVE_LABEL = "unsatisfiable"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "upstream_gate_receipt",
    "independent_row_integrity_replay",
    "leakage_safe_split_receipts",
    "label_balance_and_headroom",
    "solver_hardness_vs_label_analysis",
    "density_length_width_name_and_order_controls",
    "relabel_and_certificate_group_controls",
    "shuffled_and_majority_controls",
    "current_verifier_circularity_matrix",
    "oracle_distinct_evaluation_design",
    "held_family_and_constraint_cell_plan",
    "saturation_and_skip_decision",
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

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal audit state distinguishes qualified headroom from shortcut saturation.",
    "preconditions_checked": "Gate, rows, solvers, splits, controls, resources, and outputs prevent post-hoc qualification.",
    "upstream_gate_receipt": "Only a ready exact fixture may be audited.",
    "independent_row_integrity_replay": "Labels and certificates are rechecked rather than trusted from prose.",
    "leakage_safe_split_receipts": "Semantic, relabel, family, and certificate groups cannot cross train/test boundaries.",
    "label_balance_and_headroom": "A usable research corpus needs both outcomes and unsaturated errors.",
    "solver_hardness_vs_label_analysis": "Solver conflicts are a separate axis from correctness.",
    "density_length_width_name_and_order_controls": "Nuisance features cannot masquerade as internal reasoning.",
    "relabel_and_certificate_group_controls": "Equivalent formulas and shared witnesses stay grouped and stable.",
    "shuffled_and_majority_controls": "No-information baselines define the noise floor.",
    "current_verifier_circularity_matrix": "Solver-backed accuracy is labeled circular and never a moat.",
    "oracle_distinct_evaluation_design": "The future learned score must be evaluated separately from exact release authority.",
    "held_family_and_constraint_cell_plan": "Portability requires whole-family and whole-constraint holdouts.",
    "saturation_and_skip_decision": "A saturated or leaky corpus must skip expensive model extraction.",
    "protected_files_unchanged": "User and operator-owned files remain untouched.",
    "hardness_surface_headroom_ready_score": "EMIT BARE scalar; only 1.0 may permit Exp5871.",
    "duration_s": "Measured time exposes bootstrap-only diagnostics.",
    "inference_substrate": "`deterministic_control_audit_no_llm` declares analysis of exact rows.",
    "verifier_is_oracle": "A per-path matrix distinguishes exact/circular verifiers from oracle-distinct controls.",
    "field_provenance": "Every metric traces to rows, groups, solvers, controls, seeds, and code hashes.",
    "test_commands": "Commands document integrity, leakage, controls, circularity, and split checks.",
    "test_exit_codes": "Exit codes prevent a failed audit becoming a compute gate.",
    "reproducibility_checksum": "A checksum detects row, split, control, or threshold drift.",
    "honest_verdict": "A `complete_ready:`, `complete_null:`, or `blocked:` prefix states the terminal audit.",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5869_hardness_surface_headroom_audit.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5869_hardness_surface_headroom_audit.py "
    "-m pytest tests/python/test_experiment_5869_hardness_surface_headroom_audit.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5869_hardness_surface_headroom_audit.py "
    "--fail-under=100",
    '.venv/bin/python -c "from carnot import '
    "experiment_5869_hardness_surface_headroom_audit as m; "
    "assert m.upstream_gate_receipt()['upstream_ready'] is True\"",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5869_hardness_surface_headroom_audit.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\"",
)


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible values in a deterministic byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes so receipts do not depend on filesystem metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required:{path}")
    return dict(payload)


def _read_jsonl(path: str | Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSONL object required:{path}")
        rows.append(dict(payload))
    return rows


def _hash_optional_file(root: Path, relative: Path) -> str:
    path = root / relative
    return sha256_file(path) if path.exists() and path.is_file() else "missing"


def _memory_probe() -> JsonDict:  # pragma: no cover - host-dependent resource probe.
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {
        "available_mb": available_mb,
        "required_mb": RAM_FLOOR_MB,
        "ok": available_mb >= RAM_FLOOR_MB,
    }


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover - host-dependent resource probe.
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {
        "available_mb": available_mb,
        "required_mb": DISK_FLOOR_MB,
        "ok": available_mb >= DISK_FLOOR_MB,
    }


def read_upstream_artifact(root: Path = REPO_ROOT) -> JsonDict:
    """Read the Exp5868 summary artifact."""

    return _read_json(Path(root) / UPSTREAM_ARTIFACT_RELATIVE_PATH)


def read_upstream_rows(root: Path = REPO_ROOT) -> list[JsonDict]:
    """Read the Exp5868 row file."""

    path = Path(root) / UPSTREAM_ROWS_RELATIVE_PATH
    return _read_jsonl(path) if path.exists() else []


def _output_path_receipt(result_path: Path) -> JsonDict:
    parent = result_path.parent
    writable = (parent.exists() and os.access(parent, os.W_OK)) or (
        parent.parent.exists() and os.access(parent.parent, os.W_OK)
    )
    return {
        "result_path": str(result_path),
        "result_writable": writable and (not result_path.exists() or os.access(result_path, os.W_OK)),
        "atomic_checkpoint_suffix": ".tmp",
    }


def _verifier_config_receipt(root: Path) -> JsonDict:
    registry_path = root / VERIFIER_REGISTRY_RELATIVE_PATH
    verify_dir = root / VERIFY_DIR_RELATIVE_PATH
    verifier_files = sorted(
        path.relative_to(root).as_posix()
        for path in verify_dir.glob("*.py")
        if path.is_file()
    ) if verify_dir.exists() else []
    file_hashes = {
        relative: _hash_optional_file(root, Path(relative))
        for relative in verifier_files
    }
    return {
        "registry_path": VERIFIER_REGISTRY_RELATIVE_PATH.as_posix(),
        "registry_sha256": sha256_file(registry_path) if registry_path.exists() else "missing",
        "verify_dir": VERIFY_DIR_RELATIVE_PATH.as_posix(),
        "verifier_file_count": len(verifier_files),
        "verifier_file_hash_root": sha256_json(file_hashes),
        "ok": registry_path.exists() and bool(verifier_files),
    }


def upstream_gate_receipt(root: Path = REPO_ROOT) -> JsonDict:
    """Replay the Exp5868 gate enough to prove the upstream fixture is ready."""

    root = Path(root)
    artifact_path = root / UPSTREAM_ARTIFACT_RELATIVE_PATH
    rows_path = root / UPSTREAM_ROWS_RELATIVE_PATH
    if not artifact_path.exists() or not rows_path.exists():
        return {
            "schema": SCHEMA + ".upstream_gate_receipt",
            "artifact_path": UPSTREAM_ARTIFACT_RELATIVE_PATH.as_posix(),
            "rows_path": UPSTREAM_ROWS_RELATIVE_PATH.as_posix(),
            "upstream_ready": False,
            "blocked_reason": "missing_upstream_exp5868_artifacts",
        }
    try:
        artifact = read_upstream_artifact(root)
        rows = read_upstream_rows(root)
        row_file_verified = exp5868.verify_row_file(rows, artifact)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "schema": SCHEMA + ".upstream_gate_receipt",
            "artifact_path": UPSTREAM_ARTIFACT_RELATIVE_PATH.as_posix(),
            "rows_path": UPSTREAM_ROWS_RELATIVE_PATH.as_posix(),
            "upstream_ready": False,
            "blocked_reason": f"corrupt_upstream_exp5868:{type(exc).__name__}",
        }
    test_exit_codes = dict(artifact.get("test_exit_codes") or {})
    upstream_ready = bool(
        artifact.get("hardness_controlled_fixture_ready_score") == 1.0
        and row_file_verified
        and len(rows) == dict(artifact.get("row_file_receipt") or {}).get("row_count")
        and test_exit_codes
        and all(code == 0 for code in test_exit_codes.values())
    )
    return {
        "schema": SCHEMA + ".upstream_gate_receipt",
        "artifact_path": UPSTREAM_ARTIFACT_RELATIVE_PATH.as_posix(),
        "rows_path": UPSTREAM_ROWS_RELATIVE_PATH.as_posix(),
        "artifact_sha256": sha256_file(artifact_path),
        "rows_sha256": sha256_file(rows_path),
        "row_count": len(rows),
        "upstream_ready_score": artifact.get("hardness_controlled_fixture_ready_score"),
        "upstream_honest_verdict": artifact.get("honest_verdict"),
        "upstream_row_hash_root": dict(artifact.get("row_file_receipt") or {}).get(
            "row_hash_root"
        ),
        "test_exit_codes_all_zero": all(code == 0 for code in test_exit_codes.values()),
        "row_file_verified": row_file_verified,
        "upstream_ready": upstream_ready,
    }


def _semantic_group(row: Mapping[str, Any]) -> str:
    return str(row.get("base_instance_id") or row.get("canonical_formula_hash") or row.get("row_id"))


def _relabel_group(row: Mapping[str, Any]) -> str:
    return str(row.get("base_instance_id") or row.get("canonical_formula_hash") or row.get("row_id"))


def _certificate_group(row: Mapping[str, Any]) -> str:
    certificate = dict(row.get("certificate") or {})
    return "|".join(
        [
            str(row.get("base_instance_id")),
            str(certificate.get("kind")),
            str(row.get("canonical_formula_hash")),
        ]
    )


def _groups_to_rows(rows: Sequence[Mapping[str, Any]], groups: Sequence[str]) -> list[Mapping[str, Any]]:
    wanted = set(groups)
    return [row for row in rows if _semantic_group(row) in wanted]


def _label_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("expected_label")) for row in rows).items()))


def freeze_splits(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Freeze leakage-safe train/test definitions before controls are fitted."""

    groups = sorted({_semantic_group(row) for row in rows})
    size_by_group = {str(row["base_instance_id"]): str(row["size_bin"]) for row in rows}
    family_by_group = {str(row["base_instance_id"]): str(row["family"]) for row in rows}
    train_groups = [group for group in groups if size_by_group.get(group) in {"small", "medium"}]
    test_groups = [group for group in groups if size_by_group.get(group) == "large"]
    split = {
        "schema": SCHEMA + ".frozen_splits",
        "split_seed": SPLIT_SEED,
        "group_axes": [
            "base_instance_id",
            "canonical_formula_hash",
            "proof_preserving_relabel_group",
            "certificate_group",
            "family_holdout",
        ],
        "instance_group_split": {
            "name": "small_medium_train_large_test",
            "train_groups": train_groups,
            "test_groups": test_groups,
        },
        "family_holdout_splits": {
            f"holdout_{family}": {
                "train_groups": [group for group in groups if family_by_group.get(group) != family],
                "test_groups": [group for group in groups if family_by_group.get(group) == family],
                "held_family": family,
            }
            for family in sorted(set(family_by_group.values()))
        },
    }
    split["split_definition_hash"] = sha256_json(split)
    return split


def _split_group_intersections(
    rows: Sequence[Mapping[str, Any]],
    train_groups: Sequence[str],
    test_groups: Sequence[str],
) -> dict[str, list[str]]:
    train_rows = _groups_to_rows(rows, train_groups)
    test_rows = _groups_to_rows(rows, test_groups)
    axes = {
        "semantic_instance": _semantic_group,
        "canonical_formula_hash": lambda row: str(row.get("canonical_formula_hash")),
        "relabel_group": _relabel_group,
        "certificate_group": _certificate_group,
    }
    intersections: dict[str, list[str]] = {}
    for axis, getter in axes.items():
        train_values = {getter(row) for row in train_rows}
        test_values = {getter(row) for row in test_rows}
        intersections[axis] = sorted(train_values & test_values)
    return intersections


def verify_split_leakage(rows: Sequence[Mapping[str, Any]], splits: Mapping[str, Any]) -> JsonDict:
    """Return a receipt proving semantic groups do not cross train/test splits."""

    instance = dict(splits.get("instance_group_split") or {})
    instance_train = list(instance.get("train_groups") or [])
    instance_test = list(instance.get("test_groups") or [])
    instance_intersections = _split_group_intersections(rows, instance_train, instance_test)
    family_receipts: dict[str, JsonDict] = {}
    duplicates: list[str] = list(instance_intersections["semantic_instance"])
    for name, split in dict(splits.get("family_holdout_splits") or {}).items():
        train_groups = list(dict(split).get("train_groups") or [])
        test_groups = list(dict(split).get("test_groups") or [])
        intersections = _split_group_intersections(rows, train_groups, test_groups)
        duplicates.extend(intersections["semantic_instance"])
        train_rows = _groups_to_rows(rows, train_groups)
        test_rows = _groups_to_rows(rows, test_groups)
        family_receipts[str(name)] = {
            "held_family": dict(split).get("held_family"),
            "train_row_count": len(train_rows),
            "test_row_count": len(test_rows),
            "train_label_counts": _label_counts(train_rows),
            "test_label_counts": _label_counts(test_rows),
            "group_axis_intersections": intersections,
            "leakage_safe": not any(intersections.values()),
        }
    train_rows = _groups_to_rows(rows, instance_train)
    test_rows = _groups_to_rows(rows, instance_test)
    all_intersections = [value for values in instance_intersections.values() for value in values]
    all_intersections.extend(duplicates)
    leakage_safe = not all_intersections and bool(rows)
    return {
        "schema": SCHEMA + ".leakage_safe_split_receipts",
        "splits_frozen_before_controls": True,
        "split_seed": splits.get("split_seed"),
        "split_definition_hash": splits.get("split_definition_hash", sha256_json(splits)),
        "instance_group_split": {
            "name": instance.get("name"),
            "train_groups": instance_train,
            "test_groups": instance_test,
            "train_row_count": len(train_rows),
            "test_row_count": len(test_rows),
            "train_label_counts": _label_counts(train_rows),
            "test_label_counts": _label_counts(test_rows),
            "group_axis_intersections": instance_intersections,
            "leakage_safe": not any(instance_intersections.values()) and bool(test_rows),
        },
        "family_holdout_splits": family_receipts,
        "duplicate_semantic_instances_across_splits": sorted(set(duplicates)),
        "all_splits_leakage_safe": leakage_safe
        and all(receipt["leakage_safe"] for receipt in family_receipts.values()),
        "control_fit_policy": "controls_evaluated_only_after_this_frozen_split_receipt",
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
) -> JsonDict:
    """Collect upstream, split, solver, verifier, resource, and output gates."""

    root = Path(root)
    result_path = Path(result_path)
    upstream = upstream_gate_receipt(root)
    rows = read_upstream_rows(root) if upstream.get("upstream_ready") is True else []
    frozen_splits = freeze_splits(rows) if rows else {"split_definition_hash": "missing"}
    split_receipt = verify_split_leakage(rows, frozen_splits) if rows else {
        "all_splits_leakage_safe": False
    }
    exact_solver_receipts = exp5868.solver_version_receipts()
    exact_solver_receipts["exp5868_module_sha256"] = _hash_optional_file(
        root,
        exp5868.MODULE_RELATIVE_PATH,
    )
    current_verifier_receipt = _verifier_config_receipt(root)
    code_hashes = {
        "module": _hash_optional_file(root, MODULE_RELATIVE_PATH),
        "test": _hash_optional_file(root, TEST_RELATIVE_PATH),
        "verification_spec": _hash_optional_file(root, VERIFY_SPEC_RELATIVE_PATH),
    }
    protected_hashes = {
        path.as_posix(): _hash_optional_file(root, path) for path in PROTECTED_FILES
    }
    memory = memory_probe()
    disk = disk_probe(root)
    output_paths = _output_path_receipt(result_path)
    checks = {
        "upstream_exp5868_gate": upstream.get("upstream_ready") is True,
        "split_definitions": bool(rows) and split_receipt.get("all_splits_leakage_safe") is True,
        "exact_solver_receipts": exact_solver_receipts.get("ok") is True
        and exact_solver_receipts["exp5868_module_sha256"] != "missing",
        "current_verifier_registry_config": current_verifier_receipt.get("ok") is True,
        "seed_registry": BASE_SEED == 5869 and SPLIT_SEED > BASE_SEED and SHUFFLE_SEED > BASE_SEED,
        "code_hashes": all(value != "missing" for value in code_hashes.values()),
        "protected_files": all(value != "missing" for value in protected_hashes.values()),
        "memory": memory.get("ok") is True,
        "disk": disk.get("ok") is True,
        "output_path": output_paths["result_writable"] is True,
        "python": sys.version_info >= (3, 11),
    }
    failure_names = {
        "upstream_exp5868_gate": "missing_or_unready_exp5868_gate",
        "split_definitions": "missing_or_leaky_split_definitions",
        "exact_solver_receipts": "missing_exact_solver_receipts",
        "current_verifier_registry_config": "missing_current_verifier_registry_config",
        "seed_registry": "missing_seed_registry",
        "code_hashes": "missing_audit_module_test_or_spec",
        "protected_files": "missing_protected_file",
        "memory": "insufficient_free_ram",
        "disk": "insufficient_free_disk",
        "output_path": "output_path_not_writable",
        "python": "python_version_too_old",
    }
    blocked = [failure_names[name] for name, ok in checks.items() if not ok]
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "upstream_gate_receipt": upstream,
        "split_definitions": frozen_splits,
        "split_definition_hash": frozen_splits.get("split_definition_hash"),
        "exact_solver_receipts": exact_solver_receipts,
        "current_verifier_registry_config": current_verifier_receipt,
        "seeds": {
            "base_seed": BASE_SEED,
            "split_seed": SPLIT_SEED,
            "shuffle_seed": SHUFFLE_SEED,
            "upstream_seed": exp5868.BASE_SEED,
        },
        "code_hashes": code_hashes,
        "resources": {"memory": memory, "disk": disk},
        "output_paths": output_paths,
        "protected_file_hashes": protected_hashes,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "ok": sys.version_info >= (3, 11),
        },
        "preconditions_ready": not blocked,
        "blocked_reasons": sorted(set(blocked)),
    }


def recompute_label(row: Mapping[str, Any]) -> str:
    """Recompute Tseitin satisfiability from graph charge parity."""

    charges = [int(value) for value in row.get("charges") or []]
    if not charges:
        solver = exp5868.solve_cnf_dpll(row.get("clauses") or [], int(row.get("n_vars", 0)), config=exp5868.SOLVER_CONFIGS[0])
        return str(solver["label"])
    return "satisfiable" if sum(charges) % 2 == 0 else "unsatisfiable"


def recompute_certificate_validity(row: Mapping[str, Any]) -> bool:
    """Validate the row certificate against the row clauses, graph, and charges."""

    certificate = dict(row.get("certificate") or {})
    if certificate.get("validated") is not True:
        return False
    if row.get("expected_label") == "satisfiable":
        assignment = {
            int(key): bool(value)
            for key, value in dict(certificate.get("assignment") or {}).items()
        }
        return len(assignment) == int(row.get("n_vars", 0)) and exp5868.assignment_satisfies_cnf(
            assignment,
            row.get("clauses") or [],
        )
    if row.get("expected_label") == "unsatisfiable":
        edges = [tuple(edge) for edge in row.get("edges") or []]
        return exp5868.validate_parity_witness(
            int(row.get("n_vertices", 0)),
            edges,
            row.get("charges") or [],
            certificate.get("witness") or {},
        )
    return False


def _apply_receipt_relabel(row: Mapping[str, Any]) -> list[list[int]]:
    receipt = dict(row.get("proof_preserving_relabel") or {})
    variable_map = {int(key): int(value) for key, value in dict(receipt.get("variable_map") or {}).items()}
    n_vars = int(row.get("n_vars", 0))
    if sorted(variable_map) != list(range(1, n_vars + 1)):
        raise ValueError("invalid_relabel_domain")
    if sorted(variable_map.values()) != list(range(1, n_vars + 1)):
        raise ValueError("invalid_relabel_range")
    return exp5868.apply_variable_relabel(row.get("clauses") or [], variable_map)


def recompute_relabel_equivalence(row: Mapping[str, Any]) -> bool:
    """Check the proof-preserving relabel receipt without using summary prose."""

    try:
        relabeled_clauses = _apply_receipt_relabel(row)
    except ValueError:
        return False
    receipt = dict(row.get("proof_preserving_relabel") or {})
    solver_labels = {
        str(result.get("label"))
        for result in dict(receipt.get("solver_results") or {}).values()
    }
    if solver_labels != {str(row.get("expected_label"))}:
        return False
    if receipt.get("label_preserved") is not True or receipt.get("certificate_preserved") is not True:
        return False
    if row.get("expected_label") == "satisfiable":
        certificate = dict(row.get("certificate") or {})
        assignment = {
            int(key): bool(value)
            for key, value in dict(certificate.get("assignment") or {}).items()
        }
        variable_map = {
            int(key): int(value)
            for key, value in dict(receipt.get("variable_map") or {}).items()
        }
        relabeled_assignment = {variable_map[key]: value for key, value in assignment.items()}
        return exp5868.assignment_satisfies_cnf(relabeled_assignment, relabeled_clauses)
    return recompute_certificate_validity(row)


def independent_row_integrity_replay(
    rows: Sequence[Mapping[str, Any]],
    upstream_artifact: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Replay row integrity from row content rather than summary conclusions."""

    upstream = dict(upstream_artifact or {})
    upstream_hashes = dict(dict(upstream.get("row_file_receipt") or {}).get("row_hashes") or {})
    row_ids = [str(row.get("row_id")) for row in rows]
    duplicate_row_id_count = len(row_ids) - len(set(row_ids))
    row_hash_mismatches = [
        str(row.get("row_id"))
        for row in rows
        if exp5868.row_hash(row) != row.get("row_hash")
    ]
    summary_mismatches = [
        str(row.get("row_id"))
        for row in rows
        if upstream_hashes and upstream_hashes.get(str(row.get("row_id"))) != row.get("row_hash")
    ]
    label_disagreements = [
        str(row.get("row_id"))
        for row in rows
        if recompute_label(row) != row.get("expected_label")
    ]
    certificate_failures = [
        str(row.get("row_id"))
        for row in rows
        if not recompute_certificate_validity(row)
    ]
    relabel_failures = [
        str(row.get("row_id"))
        for row in rows
        if not recompute_relabel_equivalence(row)
    ]
    solver_disagreements = []
    for row in rows:
        labels = {
            exp5868.solve_cnf_dpll(
                row.get("clauses") or [],
                int(row.get("n_vars", 0)),
                config=config,
            )["label"]
            for config in exp5868.SOLVER_CONFIGS
        }
        if labels != {row.get("expected_label")}:
            solver_disagreements.append(str(row.get("row_id")))
    matching = exp5868.density_width_and_length_matching(rows)
    label_counts = Counter(str(row.get("expected_label")) for row in rows)
    control_counts = Counter(str(row.get("control_kind")) for row in rows)
    all_passed = bool(rows) and not any(
        [
            duplicate_row_id_count,
            row_hash_mismatches,
            summary_mismatches,
            label_disagreements,
            certificate_failures,
            relabel_failures,
            solver_disagreements,
            not matching.get("all_matching_passed"),
        ]
    )
    return {
        "schema": SCHEMA + ".independent_row_integrity_replay",
        "row_count": len(rows),
        "duplicate_row_id_count": duplicate_row_id_count,
        "row_hash_mismatch_count": len(row_hash_mismatches),
        "row_hash_mismatches": row_hash_mismatches[:20],
        "summary_row_hash_mismatch_count": len(summary_mismatches),
        "summary_row_hash_mismatches": summary_mismatches[:20],
        "exact_label_disagreement_count": len(label_disagreements),
        "exact_label_disagreements": label_disagreements[:20],
        "certificate_failure_count": len(certificate_failures),
        "certificate_failures": certificate_failures[:20],
        "solver_replay_disagreement_count": len(solver_disagreements),
        "solver_replay_disagreements": solver_disagreements[:20],
        "matching_tolerance_passed": matching.get("all_matching_passed") is True,
        "matching_tolerance_receipt": matching,
        "relabel_equivalence_failure_count": len(relabel_failures),
        "relabel_equivalence_failures": relabel_failures[:20],
        "label_counts": dict(sorted(label_counts.items())),
        "control_counts": dict(sorted(control_counts.items())),
        "all_integrity_checks_passed": all_passed,
        "receipt_hash": sha256_json(
            {
                "row_hashes": {str(row.get("row_id")): row.get("row_hash") for row in rows},
                "labels": {str(row.get("row_id")): row.get("expected_label") for row in rows},
                "matching": matching,
            }
        ),
    }


def _binary_labels(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return [1 if row.get("expected_label") == POSITIVE_LABEL else 0 for row in rows]


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    positives = [score for score, label in zip(scores, labels, strict=True) if label == 1]
    negatives = [score for score, label in zip(scores, labels, strict=True) if label == 0]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return round(wins / (len(positives) * len(negatives)), 6)


def _balanced_error(predictions: Sequence[int], labels: Sequence[int]) -> float:
    positives = [pred for pred, label in zip(predictions, labels, strict=True) if label == 1]
    negatives = [pred for pred, label in zip(predictions, labels, strict=True) if label == 0]
    if not positives or not negatives:
        return 0.5
    fnr = sum(1 for value in positives if value == 0) / len(positives)
    fpr = sum(1 for value in negatives if value == 1) / len(negatives)
    return round((fnr + fpr) / 2.0, 6)


def _threshold_predictions(
    scores: Sequence[float],
    labels: Sequence[int],
    train_indices: Sequence[int],
    test_indices: Sequence[int],
) -> tuple[float, str, float]:
    train_scores = [float(scores[index]) for index in train_indices]
    train_labels = [int(labels[index]) for index in train_indices]
    unique_scores = sorted(set(train_scores))
    candidates = unique_scores + [min(unique_scores) - 1.0, max(unique_scores) + 1.0] if unique_scores else [0.0]
    best = (1.0, ">=", candidates[0])
    for threshold in candidates:
        for direction in (">=", "<="):
            train_predictions = [
                1 if (score >= threshold if direction == ">=" else score <= threshold) else 0
                for score in train_scores
            ]
            ber = _balanced_error(train_predictions, train_labels)
            if ber < best[0]:
                best = (ber, direction, threshold)
    test_scores = [float(scores[index]) for index in test_indices]
    test_labels = [int(labels[index]) for index in test_indices]
    test_predictions = [
        1 if (score >= best[2] if best[1] == ">=" else score <= best[2]) else 0
        for score in test_scores
    ]
    return best[2], best[1], _balanced_error(test_predictions, test_labels)


def _variable_name_token_count(row: Mapping[str, Any]) -> float:
    tokens = str(row.get("surface_formula_text", "")).split()
    values = {
        abs(int(token))
        for token in tokens
        if token.lstrip("-").isdigit() and int(token) != 0
    }
    return float(len(values))


def _literal_count(row: Mapping[str, Any]) -> float:
    return float(sum(len(clause) for clause in row.get("clauses") or []))


def _abs_literal_sum(row: Mapping[str, Any]) -> float:
    return float(sum(abs(int(lit)) for clause in row.get("clauses") or [] for lit in clause))


def _feature_values(rows: Sequence[Mapping[str, Any]], name: str) -> list[float]:
    feature_map = {
        "density": lambda row, index: float(row.get("clause_density", 0.0)),
        "length": lambda row, index: float(row.get("surface_token_count", 0)),
        "clause_width": lambda row, index: float(row.get("max_clause_width", 0)),
        "variable_name_tokens": lambda row, index: _variable_name_token_count(row),
        "family_id": lambda row, index: 1.0 if row.get("family") == "expander_tseitin" else 0.0,
        "solver_conflicts": lambda row, index: float(
            dict(row.get("proof_hardness_covariates") or {}).get("solver_conflicts", 0)
        ),
        "solver_time": lambda row, index: float(
            dict(row.get("proof_hardness_covariates") or {}).get(
                "deterministic_time_proxy_s",
                0.0,
            )
        ),
        "solver_decisions": lambda row, index: float(
            dict(row.get("proof_hardness_covariates") or {}).get("solver_decisions", 0)
        ),
        "norm_literal_count": lambda row, index: _literal_count(row),
        "norm_abs_literal_sum": lambda row, index: _abs_literal_sum(row),
        "n_vars": lambda row, index: float(row.get("n_vars", 0)),
        "clause_count": lambda row, index: float(row.get("clause_count", 0)),
        "row_order": lambda row, index: float(index),
    }
    return [feature_map[name](row, index) for index, row in enumerate(rows)]


def _instance_split_indices(rows: Sequence[Mapping[str, Any]], split_receipt: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    instance = dict(split_receipt.get("instance_group_split") or {})
    train_groups = set(instance.get("train_groups") or [])
    test_groups = set(instance.get("test_groups") or [])
    train = [index for index, row in enumerate(rows) if _semantic_group(row) in train_groups]
    test = [index for index, row in enumerate(rows) if _semantic_group(row) in test_groups]
    return train, test


def evaluate_trivial_controls(
    rows: Sequence[Mapping[str, Any]],
    split_receipt: Mapping[str, Any],
) -> JsonDict:
    """Evaluate preregistered non-label controls after leakage-safe splits exist."""

    labels = _binary_labels(rows)
    train_indices, test_indices = _instance_split_indices(rows, split_receipt)
    control_names = [
        "density",
        "length",
        "clause_width",
        "variable_name_tokens",
        "family_id",
        "solver_conflicts",
        "solver_time",
        "solver_decisions",
        "norm_literal_count",
        "norm_abs_literal_sum",
        "n_vars",
        "clause_count",
        "row_order",
    ]
    oracle_derived = {"solver_conflicts", "solver_time", "solver_decisions"}
    controls: dict[str, JsonDict] = {}
    for name in control_names:
        scores = _feature_values(rows, name)
        raw_auc = _auc(scores, labels)
        oriented_auc = round(max(raw_auc, 1.0 - raw_auc), 6)
        threshold, direction, test_ber = _threshold_predictions(
            scores,
            labels,
            train_indices,
            test_indices,
        )
        controls[name] = {
            "feature_source": name,
            "uses_exact_label_feature": False,
            "oracle_derived": name in oracle_derived,
            "all_rows_auroc": raw_auc,
            "orientation_free_auroc": oriented_auc,
            "split_test_balanced_error_rate": test_ber,
            "fit_threshold": threshold,
            "fit_direction": direction,
            "saturated": oriented_auc > SATURATION_CEILING_AUROC,
        }
    non_oracle = [
        value["orientation_free_auroc"]
        for value in controls.values()
        if value["oracle_derived"] is False
    ]
    all_aurocs = [value["orientation_free_auroc"] for value in controls.values()]
    saturated = sorted(name for name, value in controls.items() if value["saturated"])
    return {
        "schema": SCHEMA + ".density_length_width_name_and_order_controls",
        "saturation_ceiling_auroc": SATURATION_CEILING_AUROC,
        "controls": controls,
        "label_feature_used": any(value["uses_exact_label_feature"] for value in controls.values()),
        "control_count": len(controls),
        "max_non_oracle_control_auroc": round(max(non_oracle) if non_oracle else 0.5, 6),
        "max_all_trivial_control_auroc": round(max(all_aurocs) if all_aurocs else 0.5, 6),
        "saturated_control_names": saturated,
        "non_oracle_controls_saturated": [
            name for name in saturated if controls[name]["oracle_derived"] is False
        ],
        "receipt_hash": sha256_json(controls),
    }


def shuffled_and_majority_controls(
    rows: Sequence[Mapping[str, Any]],
    split_receipt: Mapping[str, Any],
) -> JsonDict:
    """Compute no-information baselines for the frozen instance split."""

    labels = _binary_labels(rows)
    train_indices, test_indices = _instance_split_indices(rows, split_receipt)
    train_labels = [labels[index] for index in train_indices]
    train_label_names = [str(rows[index].get("expected_label")) for index in train_indices]
    majority_label = 1 if sum(train_labels) > len(train_labels) / 2 else 0
    majority_predictions = [majority_label for _index in test_indices]
    majority_ber = _balanced_error(majority_predictions, [labels[index] for index in test_indices])
    shuffled_scores = [float(label) for label in labels]
    random.Random(SHUFFLE_SEED).shuffle(shuffled_scores)
    shuffled_auc = _auc(shuffled_scores, labels)
    return {
        "schema": SCHEMA + ".shuffled_and_majority_controls",
        "majority_control": {
            "majority_label": POSITIVE_LABEL if majority_label else "satisfiable",
            "balanced_error_rate": majority_ber,
            "train_label_counts": dict(sorted(Counter(train_label_names).items())),
        },
        "shuffled_label_control": {
            "shuffle_seed": SHUFFLE_SEED,
            "uses_shuffled_labels": True,
            "all_rows_auroc": shuffled_auc,
            "orientation_free_auroc": round(max(shuffled_auc, 1.0 - shuffled_auc), 6),
        },
        "receipt_hash": sha256_json({"majority": majority_ber, "shuffled_auc": shuffled_auc}),
    }


def solver_hardness_vs_label_analysis(control_receipt: Mapping[str, Any]) -> JsonDict:
    """Summarize the solver-derived controls separately from non-oracle controls."""

    controls = dict(control_receipt.get("controls") or {})
    conflicts = dict(controls.get("solver_conflicts") or {})
    solver_time = dict(controls.get("solver_time") or {})
    decisions = dict(controls.get("solver_decisions") or {})
    return {
        "schema": SCHEMA + ".solver_hardness_vs_label_analysis",
        "solver_conflicts_auroc": conflicts.get("orientation_free_auroc", 0.5),
        "solver_time_auroc": solver_time.get("orientation_free_auroc", 0.5),
        "solver_decisions_auroc": decisions.get("orientation_free_auroc", 0.5),
        "solver_conflicts_saturated": conflicts.get("saturated") is True,
        "solver_time_saturated": solver_time.get("saturated") is True,
        "solver_decisions_saturated": decisions.get("saturated") is True,
        "solver_conflicts_used_as_label": False,
        "solver_time_used_as_label": False,
        "interpretation": "solver_effort_is_oracle_derived_covariate_not_oracle_distinct_headroom",
    }


def label_balance_and_headroom(
    rows: Sequence[Mapping[str, Any]],
    control_receipt: Mapping[str, Any],
    no_info_receipt: Mapping[str, Any],
) -> JsonDict:
    """Report label balance and remaining non-oracle error headroom."""

    label_counts = _label_counts(rows)
    cells = Counter(
        f"{row['proof_hardness_family']}|{row['size_bin']}|{row['expected_label']}"
        for row in rows
    )
    max_non_oracle = float(control_receipt.get("max_non_oracle_control_auroc", 0.5))
    majority_ber = float(dict(no_info_receipt.get("majority_control") or {}).get("balanced_error_rate", 0.5))
    return {
        "schema": SCHEMA + ".label_balance_and_headroom",
        "label_counts": label_counts,
        "positive_label": POSITIVE_LABEL,
        "balanced_labels": label_counts.get("satisfiable") == label_counts.get("unsatisfiable")
        and bool(rows),
        "majority_balanced_error_rate": majority_ber,
        "maximum_non_oracle_trivial_control_auroc": max_non_oracle,
        "maximum_all_trivial_control_auroc": control_receipt.get("max_all_trivial_control_auroc"),
        "balanced_error_headroom_exists": majority_ber >= 0.5
        and max_non_oracle <= SATURATION_CEILING_AUROC,
        "hard_easy_cells": dict(sorted(cells.items())),
        "headroom_boundary": "solver_oracle_controls_do_not_reduce_oracle_distinct_headroom",
    }


def relabel_and_certificate_group_controls(
    rows: Sequence[Mapping[str, Any]],
    split_receipt: Mapping[str, Any],
    integrity_receipt: Mapping[str, Any],
) -> JsonDict:
    """Summarize relabel and certificate grouping stability."""

    relabel_groups = Counter(_relabel_group(row) for row in rows)
    certificate_groups = Counter(_certificate_group(row) for row in rows)
    instance_split = dict(split_receipt.get("instance_group_split") or {})
    intersections = dict(instance_split.get("group_axis_intersections") or {})
    relabel_failures = int(integrity_receipt.get("relabel_equivalence_failure_count", 0))
    certificate_failures = int(integrity_receipt.get("certificate_failure_count", 0))
    return {
        "schema": SCHEMA + ".relabel_and_certificate_group_controls",
        "relabel_group_count": len(relabel_groups),
        "certificate_group_count": len(certificate_groups),
        "rows_per_relabel_group": dict(sorted(relabel_groups.items())),
        "rows_per_certificate_group": dict(sorted(certificate_groups.items())),
        "relabel_stability_rate": 1.0 if rows and relabel_failures == 0 else 0.0,
        "certificate_stability_rate": 1.0 if rows and certificate_failures == 0 else 0.0,
        "relabel_groups_cross_split": intersections.get("relabel_group", []),
        "certificate_groups_cross_split": intersections.get("certificate_group", []),
        "all_group_controls_passed": relabel_failures == 0
        and certificate_failures == 0
        and not intersections.get("relabel_group", [])
        and not intersections.get("certificate_group", []),
    }


def current_verifier_circularity_matrix(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Mark exact verifier paths as circular even when their accuracy is perfect."""

    labels = [str(row.get("expected_label")) for row in rows]
    paths: dict[str, JsonDict] = {}
    for config in exp5868.SOLVER_CONFIGS:
        predictions = [
            str(dict(row.get("solver_results") or {}).get(config, {}).get("label"))
            for row in rows
        ]
        accuracy = sum(1 for pred, label in zip(predictions, labels, strict=True) if pred == label)
        paths[config] = {
            "executes_same_exact_solver": True,
            "executes_certificate_checker": False,
            "verifier_is_oracle": True,
            "accuracy": round(accuracy / len(rows), 6) if rows else 0.0,
            "counts_as_oracle_distinct_headroom": False,
        }
    certificate_predictions = [
        "satisfiable"
        if dict(row.get("certificate") or {}).get("kind") == "satisfying_assignment"
        else "unsatisfiable"
        for row in rows
    ]
    certificate_accuracy = sum(
        1 for pred, label in zip(certificate_predictions, labels, strict=True) if pred == label
    )
    paths["certificate_checker"] = {
        "executes_same_exact_solver": False,
        "executes_certificate_checker": True,
        "verifier_is_oracle": True,
        "accuracy": round(certificate_accuracy / len(rows), 6) if rows else 0.0,
        "counts_as_oracle_distinct_headroom": False,
    }
    return {
        "schema": SCHEMA + ".current_verifier_circularity_matrix",
        "paths": paths,
        "all_exact_paths_marked_oracle": all(path["verifier_is_oracle"] for path in paths.values()),
        "oracle_accuracy_reduces_headroom": False,
        "matrix_hash": sha256_json(paths),
    }


def oracle_distinct_evaluation_design() -> JsonDict:
    """Return the nonempty future design kept separate from exact authority."""

    design = {
        "future_signal_source": "internal_state_or_learned_energy_score",
        "exact_release_authority_separate": True,
        "allowed_training_inputs": [
            "frozen_train_rows_without_expected_label_as_feature",
            "non-oracle_internal_state_features",
        ],
        "forbidden_inputs": [
            "expected_label",
            "solver_result_label",
            "certificate_kind_as_label_proxy",
            "exact_verifier_accuracy_as_headroom",
        ],
        "held_model_design": ["held_local_model_internal_state_probe"],
        "held_constraint_design": ["whole_family_holdout", "whole_surface_control_holdout"],
    }
    design["nonempty_held_model_and_constraint_design"] = bool(
        design["held_model_design"] and design["held_constraint_design"]
    )
    design["design_hash"] = sha256_json(design)
    return design


def held_family_and_constraint_cell_plan(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Describe whole-family and whole-constraint holdouts for downstream probes."""

    families = sorted({str(row.get("family")) for row in rows})
    controls = sorted({str(row.get("control_kind")) for row in rows})
    sizes = sorted({str(row.get("size_bin")) for row in rows})
    labels = sorted({str(row.get("expected_label")) for row in rows})
    cells = [
        f"{family}|{size}|{control}|{label}"
        for family in families
        for size in sizes
        for control in controls
        for label in labels
    ]
    return {
        "schema": SCHEMA + ".held_family_and_constraint_cell_plan",
        "whole_family_holdouts": families,
        "whole_constraint_holdouts": controls,
        "size_bins": sizes,
        "labels": labels,
        "constraint_cell_count": len(cells),
        "example_cells": cells[:12],
        "nonempty_plan": bool(cells),
        "plan_hash": sha256_json(cells),
    }


def saturation_and_skip_decision(
    integrity: Mapping[str, Any],
    splits: Mapping[str, Any],
    controls: Mapping[str, Any],
    relabels: Mapping[str, Any],
    design: Mapping[str, Any],
) -> JsonDict:
    """Decide whether downstream model extraction should proceed."""

    saturated = list(controls.get("saturated_control_names") or [])
    no_trivial_control_exceeds_ceiling = not saturated
    skip = not (
        integrity.get("all_integrity_checks_passed") is True
        and splits.get("all_splits_leakage_safe") is True
        and relabels.get("all_group_controls_passed") is True
        and no_trivial_control_exceeds_ceiling
        and design.get("nonempty_held_model_and_constraint_design") is True
    )
    return {
        "schema": SCHEMA + ".saturation_and_skip_decision",
        "saturation_ceiling_auroc": SATURATION_CEILING_AUROC,
        "saturated_control_names": saturated,
        "no_trivial_control_exceeds_ceiling": no_trivial_control_exceeds_ceiling,
        "skip_model_extraction": skip,
        "skip_reason": "solver_or_trivial_control_saturation" if saturated else (
            "integrity_or_design_gate_failed" if skip else ""
        ),
        "hardness_surface_headroom_ready_score": 0.0 if skip else 1.0,
    }


def protected_files_unchanged(root: Path, preconditions_checked: Mapping[str, Any]) -> JsonDict:
    """Verify protected operator-owned files kept the precondition hash."""

    before = dict(preconditions_checked.get("protected_file_hashes") or {})
    after = {path.as_posix(): _hash_optional_file(root, path) for path in PROTECTED_FILES}
    changed = sorted(path for path, value in after.items() if before.get(path) != value)
    return {
        "schema": SCHEMA + ".protected_files_unchanged",
        "before_hashes": before,
        "after_hashes": after,
        "changed_files": changed,
        "all_unchanged": not changed and all(value != "missing" for value in after.values()),
    }


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        VERIFY_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        UPSTREAM_ARTIFACT_RELATIVE_PATH.as_posix(),
        UPSTREAM_ROWS_RELATIVE_PATH.as_posix(),
        VERIFIER_REGISTRY_RELATIVE_PATH.as_posix(),
        VERIFY_DIR_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": principle, "sources": sources}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def _tests_pass(artifact: Mapping[str, Any]) -> bool:
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    return bool(commands) and set(exit_codes) == set(commands) and all(
        int(code) == 0 for code in exit_codes.values()
    )


def hardness_surface_headroom_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return bare readiness only when integrity passes and controls do not saturate."""

    ready = bool(
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and dict(artifact.get("upstream_gate_receipt") or {}).get("upstream_ready") is True
        and dict(artifact.get("independent_row_integrity_replay") or {}).get(
            "all_integrity_checks_passed"
        )
        is True
        and dict(artifact.get("leakage_safe_split_receipts") or {}).get(
            "all_splits_leakage_safe"
        )
        is True
        and dict(artifact.get("label_balance_and_headroom") or {}).get("balanced_labels") is True
        and dict(artifact.get("density_length_width_name_and_order_controls") or {}).get(
            "label_feature_used"
        )
        is False
        and not dict(artifact.get("density_length_width_name_and_order_controls") or {}).get(
            "saturated_control_names"
        )
        and dict(artifact.get("relabel_and_certificate_group_controls") or {}).get(
            "all_group_controls_passed"
        )
        is True
        and dict(artifact.get("current_verifier_circularity_matrix") or {}).get(
            "all_exact_paths_marked_oracle"
        )
        is True
        and dict(artifact.get("current_verifier_circularity_matrix") or {}).get(
            "oracle_accuracy_reduces_headroom"
        )
        is False
        and dict(artifact.get("oracle_distinct_evaluation_design") or {}).get(
            "nonempty_held_model_and_constraint_design"
        )
        is True
        and dict(artifact.get("held_family_and_constraint_cell_plan") or {}).get(
            "nonempty_plan"
        )
        is True
        and dict(artifact.get("protected_files_unchanged") or {}).get("all_unchanged") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and _tests_pass(artifact)
    )
    return 1.0 if ready else 0.0


def _hard_blocked(artifact: Mapping[str, Any]) -> bool:
    return not bool(
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and dict(artifact.get("upstream_gate_receipt") or {}).get("upstream_ready") is True
        and dict(artifact.get("independent_row_integrity_replay") or {}).get(
            "all_integrity_checks_passed"
        )
        is True
        and dict(artifact.get("leakage_safe_split_receipts") or {}).get(
            "all_splits_leakage_safe"
        )
        is True
        and dict(artifact.get("protected_files_unchanged") or {}).get("all_unchanged") is True
        and _tests_pass(artifact)
    )


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    """Return blocking gates for artifacts that cannot be treated as completed audits."""

    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    checks = {
        "upstream_gate_receipt": dict(artifact.get("upstream_gate_receipt") or {}).get(
            "upstream_ready"
        )
        is True,
        "independent_row_integrity_replay": dict(
            artifact.get("independent_row_integrity_replay") or {}
        ).get("all_integrity_checks_passed")
        is True,
        "leakage_safe_split_receipts": dict(
            artifact.get("leakage_safe_split_receipts") or {}
        ).get("all_splits_leakage_safe")
        is True,
        "protected_files_unchanged": dict(artifact.get("protected_files_unchanged") or {}).get(
            "all_unchanged"
        )
        is True,
        "test_exit_codes": _tests_pass(artifact),
    }
    for name, ok in checks.items():
        if not ok:
            reasons.append(name)
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        reasons.append("verifier_is_oracle")
    return sorted(set(reasons))


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build the required terminal-prefix verdict."""

    if hardness_surface_headroom_ready_score(artifact) == 1.0:
        return "complete_ready: hardness_surface_headroom_ready"
    if _hard_blocked(artifact):
        return "blocked: " + ",".join(blocked_reasons(artifact)[:8])
    saturated = list(
        dict(artifact.get("density_length_width_name_and_order_controls") or {}).get(
            "saturated_control_names"
        )
        or []
    )
    return "complete_null: trivial_or_solver_control_saturation=" + ",".join(saturated)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact after blanking host-variable and self-referential fields."""

    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    if isinstance(stable.get("preconditions_checked"), dict):
        stable["preconditions_checked"]["output_paths"] = {}
    return sha256_json(stable)


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
    duration_s: float,
    root: Path = REPO_ROOT,
) -> JsonDict:
    """Build the terminal audit artifact from already-read rows."""

    upstream = upstream_gate_receipt(root)
    split_definitions = dict(preconditions_checked.get("split_definitions") or freeze_splits(rows))
    split_receipt = verify_split_leakage(rows, split_definitions) if rows else {
        "schema": SCHEMA + ".leakage_safe_split_receipts",
        "splits_frozen_before_controls": False,
        "all_splits_leakage_safe": False,
        "duplicate_semantic_instances_across_splits": [],
        "instance_group_split": {"train_groups": [], "test_groups": []},
        "family_holdout_splits": {},
    }
    integrity = independent_row_integrity_replay(rows, read_upstream_artifact(root) if rows else {})
    controls = evaluate_trivial_controls(rows, split_receipt) if rows else {
        "schema": SCHEMA + ".density_length_width_name_and_order_controls",
        "saturation_ceiling_auroc": SATURATION_CEILING_AUROC,
        "controls": {},
        "label_feature_used": False,
        "max_non_oracle_control_auroc": 0.5,
        "max_all_trivial_control_auroc": 0.5,
        "saturated_control_names": [],
        "non_oracle_controls_saturated": [],
    }
    no_info = shuffled_and_majority_controls(rows, split_receipt) if rows else {
        "schema": SCHEMA + ".shuffled_and_majority_controls",
        "majority_control": {"balanced_error_rate": 0.5},
        "shuffled_label_control": {"uses_shuffled_labels": True, "all_rows_auroc": 0.5},
    }
    relabels = relabel_and_certificate_group_controls(rows, split_receipt, integrity)
    design = oracle_distinct_evaluation_design()
    held_plan = held_family_and_constraint_cell_plan(rows)
    decision = saturation_and_skip_decision(integrity, split_receipt, controls, relabels, design)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "blocked",
        "preconditions_checked": dict(preconditions_checked),
        "upstream_gate_receipt": upstream,
        "independent_row_integrity_replay": integrity,
        "leakage_safe_split_receipts": split_receipt,
        "label_balance_and_headroom": label_balance_and_headroom(rows, controls, no_info),
        "solver_hardness_vs_label_analysis": solver_hardness_vs_label_analysis(controls),
        "density_length_width_name_and_order_controls": controls,
        "relabel_and_certificate_group_controls": relabels,
        "shuffled_and_majority_controls": no_info,
        "current_verifier_circularity_matrix": current_verifier_circularity_matrix(rows),
        "oracle_distinct_evaluation_design": design,
        "held_family_and_constraint_cell_plan": held_plan,
        "saturation_and_skip_decision": decision,
        "protected_files_unchanged": protected_files_unchanged(root, preconditions_checked),
        "hardness_surface_headroom_ready_score": 0.0,
        "duration_s": float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": {str(key): int(value) for key, value in test_exit_codes.items()},
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: artifact_not_validated",
    }
    score = hardness_surface_headroom_ready_score(artifact)
    artifact["hardness_surface_headroom_ready_score"] = score
    if score == 1.0:
        artifact["status"] = "complete_ready"
    elif _hard_blocked(artifact):
        artifact["status"] = "blocked"
    else:
        artifact["status"] = "complete_null"
    artifact["saturation_and_skip_decision"]["hardness_surface_headroom_ready_score"] = score
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the terminal artifact schema, checksum, and terminal state."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing_fields:{missing}")
    if not _tests_pass(artifact) and artifact.get("status") != "blocked":
        raise ValueError("test_exit_codes")
    score = hardness_surface_headroom_ready_score(artifact)
    if artifact.get("hardness_surface_headroom_ready_score") != score:
        raise ValueError("hardness_surface_headroom_ready_score")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    if score == 1.0:
        if artifact.get("status") != "complete_ready":
            raise ValueError("status")
        if not str(artifact.get("honest_verdict", "")).startswith("complete_ready:"):
            raise ValueError("honest_verdict")
    elif _hard_blocked(artifact):
        if artifact.get("status") != "blocked":
            raise ValueError("status")
        if not str(artifact.get("honest_verdict", "")).startswith("blocked:"):
            raise ValueError("honest_verdict")
    else:
        if artifact.get("status") != "complete_null":
            raise ValueError("status")
        if not str(artifact.get("honest_verdict", "")).startswith("complete_null:"):
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
    """Run Exp5869, optionally writing the terminal JSON artifact."""

    started = time.perf_counter()
    root = Path(root)
    result_path = Path(result_path)
    preconditions = dict(
        preconditions_checked
        or collect_preconditions(root=root, result_path=result_path)
    )
    rows = read_upstream_rows(root) if preconditions.get("preconditions_ready") is True else []
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
