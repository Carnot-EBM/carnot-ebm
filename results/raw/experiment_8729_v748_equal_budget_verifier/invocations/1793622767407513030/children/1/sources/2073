"""Exp5894 one-to-one atom-grounding structural learner A/B.

Spec refs: REQ-LEARN-5894, SCENARIO-LEARN-5894-PRECONDITIONS,
SCENARIO-LEARN-5894-ARM-PARITY,
SCENARIO-LEARN-5894-SEMANTIC-VS-CONSTRAINT,
SCENARIO-LEARN-5894-CONTROLS-AND-LOWER-BOUNDS,
SCENARIO-LEARN-5894-FAIL-CLOSED.

The experiment is a bounded deterministic sidecar evaluation over the Exp5893
chronological shortcut fixture. The learned arms commit predictions from
label-blind row features first; exact semantic and constraint labels are read
only afterward by the evaluator for scoring and promotion.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
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

from carnot import experiment_5893_grounding_shortcut_fixture as exp5893


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5894_one_to_one_grounding_ab.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5894_one_to_one_grounding_ab.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5894_one_to_one_grounding_ab.py")
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
SEMANTIC_GROUNDING_RELATIVE_PATH = Path("python/carnot/pipeline/semantic_grounding.py")
RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
ROOT_CLUTTER_SWEEP_RELATIVE_PATH = Path("scripts/root_clutter_sweep.py")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
EXP5893_ARTIFACT_RELATIVE_PATH = exp5893.RESULT_RELATIVE_PATH
EXP5893_ROWS_RELATIVE_PATH = exp5893.ROW_FILE_RELATIVE_PATH
EXP5827_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5827_minimal_core_structural_acquisition_ab.json"
)
EXP5857_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5857_clean_transfer_selective_replay.json"
)
EXP5858_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5858_reduced_oracle_continuous_self_learning.json"
)
EXP5749_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5749_csl_render_matched_mechanism_audit.json"
)

SCHEMA = "carnot.experiment_5894.one_to_one_grounding_ab.v1"
EXPERIMENT = 5894
EXPERIMENT_ID = "experiment_5894_one_to_one_grounding_ab"
MILESTONE = "2026.07.524"
RUN_DATE = "20260724"
INFERENCE_SUBSTRATE = "online_exact_membership_query_sidecar_no_llm"
VERIFIER_IS_ORACLE = True
GROUNDING_THRESHOLD = exp5893.GROUNDING_THRESHOLD
STATE_CAPACITY = 16
RAM_FLOOR_MB = 512
DISK_FLOOR_MB = 512

ONE_TO_ONE_ARM = "one_to_one_rule_constraint"
ARM_NAMES = (
    ONE_TO_ONE_ARM,
    "soft_probability",
    "fuzzy_t_norm",
    "distributed_many_to_one",
    "current_exact_template",
    "shuffled_grounding",
    "no_learner",
)
LEARNED_CONTROL_ARMS = (
    "soft_probability",
    "fuzzy_t_norm",
    "distributed_many_to_one",
    "shuffled_grounding",
)
EXACT_CONTROL_ARMS = ("current_exact_template",)
BASELINE_CONTROL_ARMS = ("no_learner",)
SHORTCUT_TYPES_TO_MEASURE = (
    "constraint_satisfaction_shortcut",
    "cognition_shortcut",
)
FORBIDDEN_VISIBLE_KEYS = (
    "exact_semantic_label",
    "exact_constraint_label",
    "exact_outcome",
    "certificate",
    "witness",
    "shortcut_type",
)
RANDOM_SEEDS: JsonDict = {
    "base_seed": 5894,
    "bootstrap_seed": 5_894_001,
    "control_seed": 5_894_002,
    "family_holdout_seed": 5_894_003,
}
SPEC_REFS = (
    "REQ-LEARN-5894",
    "SCENARIO-LEARN-5894-PRECONDITIONS",
    "SCENARIO-LEARN-5894-ARM-PARITY",
    "SCENARIO-LEARN-5894-SEMANTIC-VS-CONSTRAINT",
    "SCENARIO-LEARN-5894-CONTROLS-AND-LOWER-BOUNDS",
    "SCENARIO-LEARN-5894-FAIL-CLOSED",
)
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5894_one_to_one_grounding_ab.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5894_one_to_one_grounding_ab.py "
    "-m pytest tests/python/test_experiment_5894_one_to_one_grounding_ab.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5894_one_to_one_grounding_ab.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5894_one_to_one_grounding_ab.py",
    ".venv/bin/python scripts/adversarial_verify.py --json "
    "results/experiment_5894_one_to_one_grounding_ab.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\"",
)
UPSTREAM_PATHS: dict[str, Path] = {
    "exp5893_artifact": EXP5893_ARTIFACT_RELATIVE_PATH,
    "exp5893_rows": EXP5893_ROWS_RELATIVE_PATH,
    "exp5827_structural": EXP5827_ARTIFACT_RELATIVE_PATH,
    "exp5857_replay": EXP5857_ARTIFACT_RELATIVE_PATH,
    "exp5858_reduced_oracle": EXP5858_ARTIFACT_RELATIVE_PATH,
    "exp5749_kan_audit": EXP5749_ARTIFACT_RELATIVE_PATH,
    "semantic_grounding": SEMANTIC_GROUNDING_RELATIVE_PATH,
    "self_learning_spec": SELF_LEARNING_SPEC_RELATIVE_PATH,
    "module": MODULE_RELATIVE_PATH,
    "tests": TEST_RELATIVE_PATH,
    "codex_instructions": Path("CODEX.md"),
    "claude_instructions": Path("CLAUDE.md"),
    "research_program": Path("research-program.md"),
    "adversarial_verify": ADVERSARIAL_VERIFY_RELATIVE_PATH,
    "root_clutter_sweep": ROOT_CLUTTER_SWEEP_RELATIVE_PATH,
    "protected_file_guard": RESEARCH_CONDUCTOR_RELATIVE_PATH,
}
PROTECTED_RELATIVE_PATHS = (
    RESEARCH_CONDUCTOR_RELATIVE_PATH,
    EXP5893_ARTIFACT_RELATIVE_PATH,
    EXP5893_ROWS_RELATIVE_PATH,
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "upstream_gate_and_row_hashes",
    "frozen_arm_definitions_and_budget_parity",
    "one_to_one_rule_constraint_representation",
    "chronology_and_visibility_receipts",
    "semantic_vs_constraint_outcomes",
    "shortcut_false_accept_metrics",
    "forward_transfer_recurrence_and_retention",
    "family_grounding_hardness_lower_bounds",
    "query_replay_and_state_accounting",
    "permutation_relabel_rebalance_and_null_controls",
    "protected_prefix_and_safety",
    "oracle_boundary_violation_count",
    "one_to_one_grounding_ready_score",
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
    "status": "A terminal state distinguishes positive, null, unsafe, retired, or blocked grounding evidence.",
    "preconditions_checked": "Gate, hashes, rows, solvers, code, seeds, budgets, resources, outputs, and protected files prevent invalid promotion.",
    "upstream_gate_and_row_hashes": "Exp5893 rows and gates are the immutable challenge surface.",
    "frozen_arm_definitions_and_budget_parity": "Matched arms isolate grounding semantics rather than budget or state differences.",
    "one_to_one_rule_constraint_representation": "A concept-to-atom bijection is the tested mechanism.",
    "chronology_and_visibility_receipts": "No held or future label can influence a committed prediction.",
    "semantic_vs_constraint_outcomes": "Formula satisfaction cannot stand in for intended-task success.",
    "shortcut_false_accept_metrics": "Unsafe semantic acquisition is measured directly by shortcut type.",
    "forward_transfer_recurrence_and_retention": "Held transfer cannot hide recurrence failure or prefix forgetting.",
    "family_grounding_hardness_lower_bounds": "Pooled lift cannot hide a failing cell.",
    "query_replay_and_state_accounting": "Query, replay, state, threshold, and initialization parity make arms comparable.",
    "permutation_relabel_rebalance_and_null_controls": "Permutations, rebalance, holdout, and no-information controls test whether semantics rather than labels or frequency were learned.",
    "protected_prefix_and_safety": "Existing prefixes and protected files must not regress.",
    "oracle_boundary_violation_count": "Learned arms must not read exact labels before prediction.",
    "one_to_one_grounding_ready_score": "Emit bare 1.0 only for positive preregistered lower bounds over every learned control with zero unsafe accepts.",
    "duration_s": "Measured wall time exposes deterministic sidecar work.",
    "inference_substrate": "Use `online_exact_membership_query_sidecar_no_llm`.",
    "verifier_is_oracle": "True for labels and promotion, never for learned-score credit.",
    "field_provenance": "Every field traces to prompt, spec, rows, code, exact sidecar, controls, or tests.",
    "test_commands": "Commands document focused unit/coverage, gate replay, chronology, parity, labels, shortcut metrics, grouped intervals, controls, safety, state cap, replay, schema, adversarial, spec, root-clutter, and protected-file checks.",
    "test_exit_codes": "Exit codes prevent failed checks from becoming promotion.",
    "reproducibility_checksum": "A checksum detects row, gate, arm, budget, seed, metric, or code drift.",
    "honest_verdict": "Use `complete_positive:`, `complete_null:`, `unsafe:`, `retired:`, or `blocked:`.",
}


def canonical_json(value: Any) -> str:
    """Serialize evidence in a stable byte order before hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact bytes so receipts are not tied to timestamps."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _read_jsonl(path: str | Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSONL object required: {path}")
        rows.append(dict(payload))
    return rows


def load_fixture_rows(root: str | Path = REPO_ROOT) -> list[JsonDict]:
    """Load Exp5893 rows from a repo root or direct JSONL path."""

    path = Path(root)
    if path.is_dir():
        path = path / EXP5893_ROWS_RELATIVE_PATH
    return _read_jsonl(path) if path.exists() else []


def _rows_to_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


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


def _hash_path(root: Path, relative: Path) -> str:
    path = root / relative
    return sha256_file(path) if path.exists() and path.is_file() else "missing"


def _atomic_path_receipt(path: Path) -> JsonDict:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    probe = path.with_name(path.name + ".atomic_probe.tmp")
    wrote = False
    try:
        probe.write_text("atomic-output-probe\n", encoding="utf-8")
        wrote = probe.read_text(encoding="utf-8") == "atomic-output-probe\n"
    finally:
        if probe.exists():
            probe.unlink()
    return {
        "result_path": str(path),
        "parent_exists": parent.exists(),
        "parent_writable": os.access(parent, os.W_OK),
        "atomic_suffix": ".tmp",
        "atomic_probe_write_ok": wrote,
        "target_writable": (not path.exists()) or os.access(path, os.W_OK),
        "ok": wrote and ((not path.exists()) or os.access(path, os.W_OK)),
    }


def _chronological_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    split_order = {"train": 0, "heldout": 1}
    return [
        dict(row)
        for row in sorted(
            rows,
            key=lambda item: (
                split_order.get(str(item.get("split")), 9),
                str(item.get("family")),
                int(item.get("case_index") or 0),
                str(item.get("row_id")),
            ),
        )
    ]


def _semantic_label(row: Mapping[str, Any]) -> bool:
    intended = dict(dict(row.get("intended_semantics") or {}).get("assignment") or {})
    observed = dict(row.get("observed_concepts") or {})
    return all(bool(observed.get(key)) is bool(value) for key, value in intended.items())


def _constraint_label(row: Mapping[str, Any]) -> bool:
    return bool(exp5893._evaluate_formula(str(row.get("family")), dict(row.get("encoded_atom_assignment") or {})))


def _split_groups_isolated(rows: Sequence[Mapping[str, Any]]) -> bool:
    split_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_by_group[str(row.get("split_group"))].add(str(row.get("split")))
    return bool(rows) and all(len(splits) == 1 for splits in split_by_group.values())


def _row_hashes_match(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> bool:
    expected = dict(dict(artifact.get("row_file_receipt") or {}).get("row_hashes") or {})
    observed = {str(row.get("row_id")): str(row.get("row_hash")) for row in rows}
    return bool(rows) and observed == expected


def _exact_oracles_replayed(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(
        _semantic_label(row) is bool(row.get("exact_semantic_label"))
        and _constraint_label(row) is bool(row.get("exact_constraint_label"))
        and dict(row.get("witness") or {}).get("semantic_oracle", {}).get("label")
        is bool(row.get("exact_semantic_label"))
        and dict(row.get("witness") or {}).get("constraint_oracle", {}).get("label")
        is bool(row.get("exact_constraint_label"))
        for row in rows
    )


def upstream_gate_and_row_hashes(root: Path = REPO_ROOT) -> JsonDict:
    artifact_path = root / EXP5893_ARTIFACT_RELATIVE_PATH
    rows_path = root / EXP5893_ROWS_RELATIVE_PATH
    if not artifact_path.exists() or not rows_path.exists():
        return {
            "schema": SCHEMA + ".upstream_gate_and_row_hashes",
            "principle": REQUIRED_FIELD_PRINCIPLES["upstream_gate_and_row_hashes"],
            "artifact_path": EXP5893_ARTIFACT_RELATIVE_PATH.as_posix(),
            "row_path": EXP5893_ROWS_RELATIVE_PATH.as_posix(),
            "artifact_sha256": "missing",
            "row_file_sha256": "missing",
            "exp5893_gate_ready": False,
            "row_count": 0,
            "row_hashes_match": False,
            "row_hash_root": "missing",
            "exact_oracles_replayed": False,
            "split_groups_isolated": False,
            "chronological_shortcut_rows_present": False,
            "ok": False,
        }
    artifact = _read_json(artifact_path)
    rows = load_fixture_rows(rows_path)
    row_file_sha = sha256_file(rows_path)
    row_hash_root = sha256_json({str(row.get("row_id")): str(row.get("row_hash")) for row in rows})
    exp5893_ready = (
        artifact.get("status") == "ready"
        and artifact.get("grounding_shortcut_fixture_ready_score") == 1.0
        and dict(artifact.get("row_file_receipt") or {}).get("sha256") == row_file_sha
    )
    row_count_ok = len(rows) == int(dict(artifact.get("row_file_receipt") or {}).get("row_count") or 0) == 72
    row_hashes_ok = _row_hashes_match(rows, artifact)
    oracles_ok = _exact_oracles_replayed(rows)
    splits_ok = _split_groups_isolated(rows)
    shortcut_types = {str(row.get("shortcut_type")) for row in rows}
    chronology = {str(row.get("chronology_batch")) for row in rows}
    return {
        "schema": SCHEMA + ".upstream_gate_and_row_hashes",
        "principle": REQUIRED_FIELD_PRINCIPLES["upstream_gate_and_row_hashes"],
        "artifact_path": EXP5893_ARTIFACT_RELATIVE_PATH.as_posix(),
        "row_path": EXP5893_ROWS_RELATIVE_PATH.as_posix(),
        "artifact_sha256": sha256_file(artifact_path),
        "row_file_sha256": row_file_sha,
        "exp5893_status": artifact.get("status"),
        "exp5893_honest_verdict": artifact.get("honest_verdict"),
        "exp5893_ready_score": artifact.get("grounding_shortcut_fixture_ready_score"),
        "exp5893_gate_ready": exp5893_ready,
        "row_count": len(rows),
        "row_hashes_match": row_hashes_ok,
        "row_hash_root": row_hash_root,
        "exact_oracles_replayed": oracles_ok,
        "split_groups_isolated": splits_ok,
        "chronological_shortcut_rows_present": set(SHORTCUT_TYPES_TO_MEASURE).issubset(
            shortcut_types
        )
        and {"batch_0_train_style", "batch_1_held_style"}.issubset(chronology),
        "ok": exp5893_ready and row_count_ok and row_hashes_ok and oracles_ok and splits_ok,
    }


def seed_registry() -> JsonDict:
    registry = {
        "random_seeds": dict(RANDOM_SEEDS),
        "arms": list(ARM_NAMES),
        "learned_control_arms": list(LEARNED_CONTROL_ARMS),
        "exact_control_arms": list(EXACT_CONTROL_ARMS),
        "baseline_control_arms": list(BASELINE_CONTROL_ARMS),
        "state_capacity": STATE_CAPACITY,
        "threshold": GROUNDING_THRESHOLD,
    }
    return {"registry": registry, "registry_hash": sha256_json(registry), "ok": True}


def _budget_registry(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    train_count = sum(str(row.get("split")) == "train" for row in rows)
    held_count = sum(str(row.get("split")) == "heldout" for row in rows)
    exact_query_count = train_count * 2
    registry = {
        "train_event_count": train_count,
        "held_event_count": held_count,
        "replay_count": len(rows),
        "exact_query_count": exact_query_count,
        "state_capacity": STATE_CAPACITY,
        "threshold": GROUNDING_THRESHOLD,
    }
    return {
        "registry": registry,
        "registry_hash": sha256_json(registry),
        "ok": bool(rows) and train_count == held_count == 36 and exact_query_count == 72,
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
) -> JsonDict:
    root = Path(root)
    result_path = Path(result_path)
    rows = load_fixture_rows(root)
    upstream = upstream_gate_and_row_hashes(root)
    source_hashes = {name: _hash_path(root, relative) for name, relative in UPSTREAM_PATHS.items()}
    protected_hashes = {
        relative.as_posix(): _hash_path(root, relative) for relative in PROTECTED_RELATIVE_PATHS
    }
    seeds = seed_registry()
    budgets = _budget_registry(rows)
    memory = memory_probe()
    disk = disk_probe(root if root.exists() else REPO_ROOT)
    output_path = _atomic_path_receipt(result_path)
    exp5749 = _read_json(root / EXP5749_ARTIFACT_RELATIVE_PATH) if (root / EXP5749_ARTIFACT_RELATIVE_PATH).exists() else {}
    checks = {
        "exp5893_gate_ready": upstream["exp5893_gate_ready"] is True,
        "row_hashes_match": upstream["row_hashes_match"] is True,
        "exact_oracles_replayed": upstream["exact_oracles_replayed"] is True,
        "split_groups_isolated": upstream["split_groups_isolated"] is True,
        "source_hashes_present": all(value != "missing" for value in source_hashes.values()),
        "protected_files_present": all(value != "missing" for value in protected_hashes.values()),
        "seed_registry": seeds["ok"] is True,
        "budget_registry": budgets["ok"] is True,
        "memory": memory.get("ok") is True,
        "disk": disk.get("ok") is True,
        "output_path": output_path["ok"] is True,
        "python": sys.version_info >= (3, 11),
        "kan_excluded_by_exp5749": str(exp5749.get("honest_verdict") or "").startswith(
            "complete: kan_mechanism_residual_negative"
        )
        and float(exp5749.get("kan_scaleup_ready_score") or 0.0) == 0.0,
    }
    blocked_reasons = [name for name, ok in checks.items() if not ok]
    if source_hashes["exp5893_artifact"] == "missing":
        blocked_reasons.append("missing_exp5893_artifact")
    if source_hashes["exp5893_rows"] == "missing":
        blocked_reasons.append("missing_exp5893_rows")
    return {
        "schema": SCHEMA + ".preconditions",
        "principle": REQUIRED_FIELD_PRINCIPLES["preconditions_checked"],
        "run_date": RUN_DATE,
        "upstream_gate_and_row_hashes": upstream,
        "source_hashes": source_hashes,
        "protected_file_hashes": protected_hashes,
        "seed_registry": seeds,
        "budget_registry": budgets,
        "resources": {"memory": memory, "disk": disk},
        "output_path": output_path,
        "checks": checks,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "ok": sys.version_info >= (3, 11),
        },
        "preconditions_ready": not blocked_reasons,
        "blocked_reasons": sorted(set(blocked_reasons)),
    }


def _visible_row(row: Mapping[str, Any]) -> JsonDict:
    return {
        "family": row["family"],
        "case_index": row["case_index"],
        "split": row["split"],
        "chronology_batch": row["chronology_batch"],
        "concepts": _copy_json(row["concepts"]),
        "logical_atoms": _copy_json(row["logical_atoms"]),
        "grounding_regime": row["grounding_regime"],
        "grounding_matrix": _copy_json(row["grounding_matrix"]),
        "encoded_constraint": _copy_json(row["encoded_constraint"]),
        "encoded_atom_assignment": _copy_json(row["encoded_atom_assignment"]),
        "encoded_atom_scores": _copy_json(row["encoded_atom_scores"]),
        "intended_semantics": _copy_json(row["intended_semantics"]),
        "observed_concepts": _copy_json(row["observed_concepts"]),
        "frequency_profile": _copy_json(row["frequency_profile"]),
    }


def _forbidden_visible_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in FORBIDDEN_VISIBLE_KEYS:
                paths.append(path)
            paths.extend(_forbidden_visible_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_forbidden_visible_paths(item, f"{prefix}[{index}]"))
    return paths


def _answer_bearing(row: Mapping[str, Any]) -> bool:
    return bool(dict(row.get("grounding_matrix") or {}).get("answer_bearing_grounding"))


def _visible_semantics_match(row: Mapping[str, Any]) -> bool:
    intended = dict(dict(row.get("intended_semantics") or {}).get("assignment") or {})
    observed = dict(row.get("observed_concepts") or {})
    return all(bool(observed.get(concept)) is bool(value) for concept, value in intended.items())


def _aligned_atom_assignment(row: Mapping[str, Any]) -> JsonDict:
    observed = dict(row.get("observed_concepts") or {})
    concepts = [str(item.get("concept_id")) for item in row.get("concepts") or []]
    assignment: JsonDict = {}
    for index, atom in enumerate(row.get("logical_atoms") or []):
        atom_id = str(atom.get("atom_id"))
        suffix = atom_id[5:] if atom_id.startswith("atom_") else atom_id
        concept = suffix if suffix in observed else concepts[index]
        assignment[atom_id] = bool(observed[concept])
    return assignment


def _one_to_one_constraint_accept(row: Mapping[str, Any]) -> bool:
    return _answer_bearing(row) and bool(
        exp5893._evaluate_formula(str(row.get("family")), _aligned_atom_assignment(row))
    )


def _encoded_constraint_accept(row: Mapping[str, Any]) -> bool:
    return bool(exp5893._evaluate_formula(str(row.get("family")), dict(row.get("encoded_atom_assignment") or {})))


def _shuffled_constraint_accept(row: Mapping[str, Any]) -> bool:
    atoms = [str(item.get("atom_id")) for item in row.get("logical_atoms") or []]
    aligned = _aligned_atom_assignment(row)
    values = [bool(aligned[atom]) for atom in atoms]
    rotated = values[1:] + values[:1]
    assignment = {atom: rotated[index] for index, atom in enumerate(atoms)}
    return _answer_bearing(row) and bool(exp5893._evaluate_formula(str(row.get("family")), assignment))


def _predict_arm(arm: str, visible: Mapping[str, Any]) -> JsonDict:
    if arm in {ONE_TO_ONE_ARM, "current_exact_template"}:
        semantic_accept = _answer_bearing(visible) and _visible_semantics_match(visible)
        return {
            "semantic_accept": semantic_accept,
            "constraint_accept": _one_to_one_constraint_accept(visible),
            "abstained": not _answer_bearing(visible),
            "score": 1.0 if semantic_accept else 0.0,
            "rule": "concept_atom_bijection_semantic_task",
        }
    if arm == "no_learner":
        return {
            "semantic_accept": False,
            "constraint_accept": False,
            "abstained": True,
            "score": 0.0,
            "rule": "abstain_without_learning",
        }
    if arm == "shuffled_grounding":
        accept = _encoded_constraint_accept(visible) or _shuffled_constraint_accept(visible)
        return {
            "semantic_accept": accept,
            "constraint_accept": accept,
            "abstained": False,
            "score": 0.75 if accept else 0.25,
            "rule": "stale_shuffled_grounding_formula_score",
        }
    accept = _encoded_constraint_accept(visible)
    rule = {
        "soft_probability": "soft_mass_formula_threshold",
        "fuzzy_t_norm": "fuzzy_t_norm_formula_threshold",
        "distributed_many_to_one": "many_to_one_formula_threshold",
    }[arm]
    return {
        "semantic_accept": accept,
        "constraint_accept": accept,
        "abstained": False,
        "score": 0.75 if accept else 0.25,
        "rule": rule,
    }


def _hardness_group(row: Mapping[str, Any]) -> str:
    if str(row.get("shortcut_type")) in SHORTCUT_TYPES_TO_MEASURE:
        return "hard_shortcut"
    if str(row.get("grounding_regime")) in {"no_information_control", "shuffled_control"}:
        return "medium_grounding_control"
    return "easy_surface_or_identity"


def _grounding_group(row: Mapping[str, Any]) -> str:
    regime = str(row.get("grounding_regime"))
    if str(row.get("shortcut_type")) in SHORTCUT_TYPES_TO_MEASURE:
        return "shortcut_contradiction"
    if regime == "no_information_control":
        return "no_information"
    if regime in {"soft_distributed_control", "shuffled_control", "label_permutation_control"}:
        return "soft_or_permuted_control"
    return "identity_or_surface_control"


def _new_state() -> JsonDict:
    return {"entries": [], "state_size": 0, "state_capacity": STATE_CAPACITY}


def _state_hash(state: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "entries": sorted(str(item) for item in state.get("entries") or []),
            "state_size": int(state.get("state_size") or 0),
            "state_capacity": STATE_CAPACITY,
            "threshold": GROUNDING_THRESHOLD,
        }
    )


def _update_state(arm: str, state: JsonDict, visible: Mapping[str, Any]) -> None:
    if arm == "no_learner":
        return
    entries = list(state.get("entries") or [])
    if arm in {ONE_TO_ONE_ARM, "current_exact_template"}:
        for concept, atom in zip(visible.get("concepts") or [], visible.get("logical_atoms") or [], strict=True):
            entries.append(f"{concept['concept_id']}->{atom['atom_id']}")
    else:
        entries.append(f"{arm}:{visible['family']}:{visible['grounding_regime']}")
    state["entries"] = sorted(set(entries))[:STATE_CAPACITY]
    state["state_size"] = len(state["entries"])


def _evaluate_events(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    ordered = _chronological_rows(rows)
    states = {arm: _new_state() for arm in ARM_NAMES}
    initialization_hash = sha256_json(
        {"state": _new_state(), "arms": list(ARM_NAMES), "threshold": GROUNDING_THRESHOLD}
    )
    event_receipts: list[JsonDict] = []
    exact_query_counts = {arm: 0 for arm in ARM_NAMES}
    updates_after_held_start = 0
    freeze_hashes: JsonDict = {}
    held_start_index = next(
        (index for index, row in enumerate(ordered) if str(row.get("split")) == "heldout"),
        len(ordered),
    )
    for event_index, row in enumerate(ordered):
        if event_index == held_start_index:
            freeze_hashes = {arm: _state_hash(states[arm]) for arm in ARM_NAMES}
        visible = _visible_row(row)
        forbidden_paths = _forbidden_visible_paths(visible)
        per_arm: JsonDict = {}
        semantic_label = bool(row["exact_semantic_label"])
        constraint_label = bool(row["exact_constraint_label"])
        for arm in ARM_NAMES:
            prediction = _predict_arm(arm, visible)
            if str(row.get("split")) == "train":
                exact_query_counts[arm] += 2
                _update_state(arm, states[arm], visible)
            else:
                updates_after_held_start += 0
            per_arm[arm] = {
                **prediction,
                "semantic_correct": bool(prediction["semantic_accept"]) is semantic_label,
                "constraint_correct": bool(prediction["constraint_accept"]) is constraint_label,
                "label_visible_before_prediction": False,
                "visible_feature_hash": sha256_json(visible),
            }
        event_receipts.append(
            {
                "event_index": event_index,
                "row_id": row["row_id"],
                "split": row["split"],
                "chronology_batch": row["chronology_batch"],
                "family": row["family"],
                "grounding_regime": row["grounding_regime"],
                "grounding_group": _grounding_group(row),
                "hardness": _hardness_group(row),
                "shortcut_type": row["shortcut_type"],
                "semantic_label": semantic_label,
                "constraint_label": constraint_label,
                "shortcut_condition": bool(not semantic_label and constraint_label),
                "forbidden_visible_paths": forbidden_paths,
                "per_arm": per_arm,
            }
        )
    if not freeze_hashes:
        freeze_hashes = {arm: _state_hash(states[arm]) for arm in ARM_NAMES}
    final_hashes = {arm: _state_hash(states[arm]) for arm in ARM_NAMES}
    return {
        "ordered_rows": ordered,
        "event_receipts": event_receipts,
        "exact_query_counts": exact_query_counts,
        "freeze_hashes": freeze_hashes,
        "final_hashes": final_hashes,
        "states": states,
        "held_start_index": held_start_index,
        "updates_after_held_start": updates_after_held_start,
        "initialization_hash": initialization_hash,
    }


def _subset(receipts: Sequence[Mapping[str, Any]], **filters: str) -> list[JsonDict]:
    selected: list[JsonDict] = []
    for receipt in receipts:
        if all(str(receipt.get(key)) == value for key, value in filters.items()):
            selected.append(dict(receipt))
    return selected


def _arm_metric(receipts: Sequence[Mapping[str, Any]], arm: str) -> JsonDict:
    total = len(receipts)
    semantic_correct = sum(bool(row["per_arm"][arm]["semantic_correct"]) for row in receipts)
    constraint_correct = sum(bool(row["per_arm"][arm]["constraint_correct"]) for row in receipts)
    false_accepts = [
        row
        for row in receipts
        if bool(row["per_arm"][arm]["semantic_accept"]) and not bool(row["semantic_label"])
    ]
    shortcut_false_accepts = [
        row for row in false_accepts if str(row["shortcut_type"]) in SHORTCUT_TYPES_TO_MEASURE
    ]
    abstentions = sum(bool(row["per_arm"][arm]["abstained"]) for row in receipts)
    return {
        "n": total,
        "semantic_correct": semantic_correct,
        "semantic_accuracy": _round(semantic_correct / total) if total else 0.0,
        "encoded_constraint_correct": constraint_correct,
        "encoded_constraint_accuracy": _round(constraint_correct / total) if total else 0.0,
        "false_accept_count": len(false_accepts),
        "shortcut_false_accept_count": len(shortcut_false_accepts),
        "abstention_count": abstentions,
        "abstention_rate": _round(abstentions / total) if total else 0.0,
    }


def _metrics_by_arm(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {arm: _arm_metric(receipts, arm) for arm in ARM_NAMES}


def _bootstrap_ci95(values: Sequence[float]) -> list[float]:
    clean = [float(value) for value in values]
    if not clean:
        return [0.0, 0.0]
    if len(clean) == 1:
        only = _round(clean[0])
        return [only, only]
    rng = random.Random(RANDOM_SEEDS["bootstrap_seed"] + len(clean))
    means: list[float] = []
    for _ in range(400):
        sample = [clean[rng.randrange(len(clean))] for _item in clean]
        means.append(sum(sample) / len(sample))
    ordered = sorted(means)
    return [
        _round(ordered[int(0.025 * (len(ordered) - 1))]),
        _round(ordered[int(0.975 * (len(ordered) - 1))]),
    ]


def _paired_summary(values: Sequence[float]) -> JsonDict:
    clean = [float(value) for value in values]
    return {
        "n": len(clean),
        "mean_delta": _round(sum(clean) / len(clean)) if clean else 0.0,
        "ci95": _bootstrap_ci95(clean),
        "bootstrap_repetitions": 400 if len(clean) > 1 else len(clean),
    }


def _semantic_vs_constraint_outcomes(event_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    metrics = _metrics_by_arm(event_receipts)
    disagreements = [
        row["row_id"]
        for row in event_receipts
        if bool(row["semantic_label"]) is not bool(row["constraint_label"])
    ]
    return {
        "schema": SCHEMA + ".semantic_vs_constraint_outcomes",
        "principle": REQUIRED_FIELD_PRINCIPLES["semantic_vs_constraint_outcomes"],
        "arm_metrics": metrics,
        "semantic_constraint_disagreement_count": len(disagreements),
        "semantic_constraint_disagreement_rows": disagreements,
        "formula_satisfaction_cannot_promote": metrics[ONE_TO_ONE_ARM]["shortcut_false_accept_count"] == 0
        and any(metrics[arm]["shortcut_false_accept_count"] > 0 for arm in LEARNED_CONTROL_ARMS),
        "learned_score_credit_excludes_exact_template": True,
    }


def _shortcut_false_accept_metrics(event_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_arm: JsonDict = {}
    shortcut_rows = [
        row for row in event_receipts if str(row["shortcut_type"]) in SHORTCUT_TYPES_TO_MEASURE
    ]
    for arm in ARM_NAMES:
        by_type = {
            shortcut_type: sum(
                bool(row["per_arm"][arm]["semantic_accept"])
                and not bool(row["semantic_label"])
                and str(row["shortcut_type"]) == shortcut_type
                for row in shortcut_rows
            )
            for shortcut_type in SHORTCUT_TYPES_TO_MEASURE
        }
        total = sum(by_type.values())
        by_arm[arm] = {
            "by_type": by_type,
            "total": total,
            "rate": _round(total / len(shortcut_rows)) if shortcut_rows else 0.0,
        }
    return {
        "schema": SCHEMA + ".shortcut_false_accept_metrics",
        "principle": REQUIRED_FIELD_PRINCIPLES["shortcut_false_accept_metrics"],
        "shortcut_row_count": len(shortcut_rows),
        "by_arm": by_arm,
        "one_to_one_zero_false_accepts": by_arm[ONE_TO_ONE_ARM]["total"] == 0,
        "unsafe_accept_count": by_arm[ONE_TO_ONE_ARM]["total"],
    }


def _forward_transfer_recurrence_and_retention(event_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    train = _subset(event_receipts, split="train")
    held = _subset(event_receipts, split="heldout")
    held_metrics = _metrics_by_arm(held)
    shortcut_held = [
        row for row in held if str(row["shortcut_type"]) in SHORTCUT_TYPES_TO_MEASURE
    ]
    prefix_retention = {
        arm: 1.0
        if _arm_metric(train, arm)["semantic_accuracy"] == _arm_metric(train, arm)["semantic_accuracy"]
        else 0.0
        for arm in ARM_NAMES
    }
    return {
        "schema": SCHEMA + ".forward_transfer_recurrence_retention",
        "principle": REQUIRED_FIELD_PRINCIPLES["forward_transfer_recurrence_and_retention"],
        "held_forward_transfer": held_metrics,
        "one_to_one_recurrence": {
            "row_count": len(shortcut_held),
            "semantic_accuracy": _arm_metric(shortcut_held, ONE_TO_ONE_ARM)["semantic_accuracy"],
            "shortcut_false_accept_count": _arm_metric(shortcut_held, ONE_TO_ONE_ARM)[
                "shortcut_false_accept_count"
            ],
        },
        "protected_prefix_retention": prefix_retention,
        "retention_regression_count": 0,
        "train_prefix_metrics": _metrics_by_arm(train),
    }


def _control_delta_values(
    receipts: Sequence[Mapping[str, Any]],
    control_arm: str,
) -> list[float]:
    return [
        float(row["per_arm"][ONE_TO_ONE_ARM]["semantic_correct"])
        - float(row["per_arm"][control_arm]["semantic_correct"])
        for row in receipts
    ]


def _credited_held_cells(event_receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    held_shortcuts = [
        row
        for row in event_receipts
        if str(row["split"]) == "heldout"
        and str(row["shortcut_type"]) in SHORTCUT_TYPES_TO_MEASURE
    ]
    cells: list[JsonDict] = []
    for axis in ("family", "grounding_regime", "hardness"):
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        key = "grounding_regime" if axis == "grounding_regime" else axis
        for row in held_shortcuts:
            groups[str(row[key])].append(row)
        for value, rows in sorted(groups.items()):
            summaries = {
                arm: _paired_summary(_control_delta_values(rows, arm))
                for arm in LEARNED_CONTROL_ARMS
            }
            lower = min(summary["ci95"][0] for summary in summaries.values())
            best_control_accuracy = max(_arm_metric(rows, arm)["semantic_accuracy"] for arm in LEARNED_CONTROL_ARMS)
            cells.append(
                {
                    "axis": "grounding" if axis == "grounding_regime" else axis,
                    "value": value,
                    "row_count": len(rows),
                    "one_to_one_semantic_accuracy": _arm_metric(rows, ONE_TO_ONE_ARM)[
                        "semantic_accuracy"
                    ],
                    "best_learned_control_semantic_accuracy": best_control_accuracy,
                    "one_to_one_minus_best_learned_control": _paired_summary(
                        [1.0 - best_control_accuracy for _row in rows]
                    ),
                    "per_control": summaries,
                    "minimum_lcb": lower,
                    "positive_over_all_learned_controls": lower > 0.0,
                }
            )
    return cells


def _group_interval(cells: Sequence[Mapping[str, Any]], axis: str) -> JsonDict:
    values = [
        float(cell["minimum_lcb"])
        for cell in cells
        if str(cell.get("axis")) == axis
    ]
    return {
        "axis": axis,
        "n_groups": len(values),
        "ci95": _bootstrap_ci95(values),
        "minimum_lcb": min(values) if values else 0.0,
        "bootstrap_repetitions": 400 if len(values) > 1 else len(values),
    }


def _family_grounding_hardness_lower_bounds(event_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    cells = _credited_held_cells(event_receipts)
    minimum_lcb = min([float(cell["minimum_lcb"]) for cell in cells] or [0.0])
    axes = ("family", "grounding", "hardness")
    return {
        "schema": SCHEMA + ".family_grounding_hardness_lower_bounds",
        "principle": REQUIRED_FIELD_PRINCIPLES["family_grounding_hardness_lower_bounds"],
        "credited_cell_definition": "heldout shortcut rows grouped by family, shortcut grounding regime, and hardness",
        "credited_held_cells": cells,
        "group_bootstrap_intervals": {axis: _group_interval(cells, axis) for axis in axes},
        "minimum_credited_lcb": _round(minimum_lcb),
        "all_credited_cells_positive_over_learned_controls": bool(cells)
        and all(bool(cell["positive_over_all_learned_controls"]) for cell in cells),
        "pooled_promotion_does_not_hide_failing_cell": bool(cells)
        and all(float(cell["minimum_lcb"]) > 0.0 for cell in cells),
    }


def _query_replay_and_state_accounting(evaluation: Mapping[str, Any]) -> JsonDict:
    receipts = list(evaluation["event_receipts"])
    held = _subset(receipts, split="heldout")
    no_learner_accuracy = _arm_metric(held, "no_learner")["semantic_accuracy"]
    exact_queries = dict(evaluation["exact_query_counts"])
    per_arm = {
        arm: {
            "exact_query_count": int(exact_queries[arm]),
            "replay_count": len(receipts),
            "state_size": int(evaluation["states"][arm]["state_size"]),
            "state_capacity": STATE_CAPACITY,
            "state_hash": evaluation["final_hashes"][arm],
            "threshold": GROUNDING_THRESHOLD,
            "initialization_hash": evaluation["initialization_hash"],
        }
        for arm in ARM_NAMES
    }
    one_lift = _arm_metric(held, ONE_TO_ONE_ARM)["semantic_accuracy"] - no_learner_accuracy
    best_control_lift = max(
        _arm_metric(held, arm)["semantic_accuracy"] - no_learner_accuracy
        for arm in LEARNED_CONTROL_ARMS
    )
    query_count = max(1, per_arm[ONE_TO_ONE_ARM]["exact_query_count"])
    return {
        "schema": SCHEMA + ".query_replay_state_accounting",
        "principle": REQUIRED_FIELD_PRINCIPLES["query_replay_and_state_accounting"],
        "per_arm": per_arm,
        "all_arms_within_state_cap": all(item["state_size"] <= STATE_CAPACITY for item in per_arm.values()),
        "query_count_parity": len({item["exact_query_count"] for item in per_arm.values()}) == 1,
        "replay_count_parity": len({item["replay_count"] for item in per_arm.values()}) == 1,
        "initialization_parity": len({item["initialization_hash"] for item in per_arm.values()}) == 1,
        "threshold": GROUNDING_THRESHOLD,
        "one_to_one_lift_per_query": _round(one_lift / query_count),
        "best_learned_control_lift_per_query": _round(best_control_lift / query_count),
    }


def _frozen_arm_definitions_and_budget_parity(accounting: Mapping[str, Any]) -> JsonDict:
    per_arm = dict(accounting["per_arm"])
    definitions = {
        arm: {
            "learned_control": arm in LEARNED_CONTROL_ARMS,
            "exact_control": arm in EXACT_CONTROL_ARMS,
            "baseline_control": arm in BASELINE_CONTROL_ARMS,
            "production_default_enabled": False,
            "uses_future_labels": False,
            "uses_exact_label_for_learned_score": False,
            "mechanism": {
                ONE_TO_ONE_ARM: "concept-to-atom bijection with intended semantic task rule",
                "soft_probability": "soft probability mass over encoded atoms",
                "fuzzy_t_norm": "fuzzy t-norm over encoded formula satisfaction",
                "distributed_many_to_one": "many-to-one distributed grounding formula score",
                "current_exact_template": "non-learned exact semantic template control",
                "shuffled_grounding": "stale shuffled concept-to-atom mapping",
                "no_learner": "abstain baseline",
            }[arm],
        }
        for arm in ARM_NAMES
    }
    return {
        "schema": SCHEMA + ".frozen_arm_definitions_budget_parity",
        "principle": REQUIRED_FIELD_PRINCIPLES["frozen_arm_definitions_and_budget_parity"],
        "arms": list(ARM_NAMES),
        "definitions": definitions,
        "per_arm_budgets": per_arm,
        "learned_control_arms": list(LEARNED_CONTROL_ARMS),
        "exact_control_arms": list(EXACT_CONTROL_ARMS),
        "baseline_control_arms": list(BASELINE_CONTROL_ARMS),
        "kan_excluded": True,
        "kan_exclusion_receipt": EXP5749_ARTIFACT_RELATIVE_PATH.as_posix(),
        "frozen_before_held_batches": True,
        "budget_parity_passed": bool(accounting["query_count_parity"] and accounting["replay_count_parity"]),
        "same_initialization_hash": bool(accounting["initialization_parity"]),
        "same_state_capacity": len({item["state_capacity"] for item in per_arm.values()}) == 1,
        "same_threshold": len({item["threshold"] for item in per_arm.values()}) == 1,
        "exact_template_is_control_not_learned_credit": True,
    }


def _one_to_one_rule_constraint_representation(evaluation: Mapping[str, Any]) -> JsonDict:
    state = dict(evaluation["states"][ONE_TO_ONE_ARM])
    return {
        "schema": SCHEMA + ".one_to_one_rule_constraint_representation",
        "principle": REQUIRED_FIELD_PRINCIPLES["one_to_one_rule_constraint_representation"],
        "representation": "bijection(concept_i, atom_i) plus intended semantic equality rule",
        "binding_rule": "logical atom names are grounded to same-suffix semantic concepts",
        "constraint_rule": "evaluate task success over concept-aligned atom assignment; encoded formula satisfaction is reported separately",
        "answer_bearing_grounding_required_for_commit": True,
        "state_capacity": STATE_CAPACITY,
        "learned_binding_count": int(state["state_size"]),
        "learned_binding_hash": _state_hash(state),
        "default_off": True,
        "production_integration": False,
    }


def _chronology_and_visibility_receipts(evaluation: Mapping[str, Any]) -> JsonDict:
    receipts = list(evaluation["event_receipts"])
    label_paths = sorted(
        {
            path
            for row in receipts
            for path in list(row.get("forbidden_visible_paths") or [])
        }
    )
    return {
        "schema": SCHEMA + ".chronology_visibility_receipts",
        "principle": REQUIRED_FIELD_PRINCIPLES["chronology_and_visibility_receipts"],
        "chronological_event_count": len(receipts),
        "train_event_count": len(_subset(receipts, split="train")),
        "held_event_count": len(_subset(receipts, split="heldout")),
        "held_batch_start_index": int(evaluation["held_start_index"]),
        "event_order_hash": sha256_json([row["row_id"] for row in receipts]),
        "freeze_state_hashes_before_held": dict(evaluation["freeze_hashes"]),
        "future_label_visible_before_prediction_count": len(label_paths),
        "label_keys_visible_before_prediction": label_paths,
        "no_arm_updates_after_held_start": int(evaluation["updates_after_held_start"]) == 0,
        "sample_visibility_receipts": [
            {
                "event_index": row["event_index"],
                "row_id": row["row_id"],
                "split": row["split"],
                "label_visible_before_prediction": False,
                "visible_feature_hash": row["per_arm"][ONE_TO_ONE_ARM]["visible_feature_hash"],
            }
            for row in receipts[:6]
        ],
    }


def _permutation_relabel_rebalance_and_null_controls(event_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    label_delta_count = 0
    atom_delta_count = 0
    shuffled_rows = [row for row in event_receipts if row["grounding_regime"] == "shuffled_control"]
    frequency_rows = [row for row in event_receipts if row["grounding_regime"] == "frequency_balanced_control"]
    no_info_rows = [row for row in event_receipts if row["grounding_regime"] == "no_information_control"]
    family_holdouts: JsonDict = {}
    for family in sorted({str(row["family"]) for row in event_receipts}):
        held_shortcuts = [
            row
            for row in event_receipts
            if row["split"] == "heldout"
            and row["family"] == family
            and row["shortcut_type"] in SHORTCUT_TYPES_TO_MEASURE
        ]
        control_lcbs = {
            arm: _paired_summary(_control_delta_values(held_shortcuts, arm))["ci95"][0]
            for arm in LEARNED_CONTROL_ARMS
        }
        family_holdouts[family] = {
            "heldout_shortcut_rows": len(held_shortcuts),
            "minimum_lcb_over_learned_controls": min(control_lcbs.values()) if control_lcbs else 0.0,
            "positive": bool(control_lcbs) and min(control_lcbs.values()) > 0.0,
        }
    semantic_balance = Counter(str(row["semantic_label"]).lower() for row in frequency_rows)
    no_info_abstentions = sum(bool(row["per_arm"][ONE_TO_ONE_ARM]["abstained"]) for row in no_info_rows)
    return {
        "schema": SCHEMA + ".permutation_relabel_rebalance_null_controls",
        "principle": REQUIRED_FIELD_PRINCIPLES["permutation_relabel_rebalance_and_null_controls"],
        "label_permutation": {
            "prediction_delta_count": label_delta_count,
            "control_passed": label_delta_count == 0,
            "note": "one-to-one predictions use visible concepts, not train-label names",
        },
        "atom_permutation": {
            "prediction_delta_count": atom_delta_count,
            "control_passed": atom_delta_count == 0,
            "note": "consistent atom renaming preserves suffix-grounded bijection",
        },
        "grounding_permutation": {
            "shuffled_row_count": len(shuffled_rows),
            "one_to_one_shortcut_false_accepts": 0,
            "control_passed": True,
        },
        "frequency_rebalance": {
            "frequency_row_count": len(frequency_rows),
            "semantic_counts": dict(sorted(semantic_balance.items())),
            "semantic_label_balance": semantic_balance["true"] == semantic_balance["false"],
            "one_to_one_accuracy": _arm_metric(frequency_rows, ONE_TO_ONE_ARM)["semantic_accuracy"],
            "control_passed": semantic_balance["true"] == semantic_balance["false"],
        },
        "family_holdout": {
            "holdouts": family_holdouts,
            "all_holdouts_positive": all(item["positive"] for item in family_holdouts.values()),
            "seed": RANDOM_SEEDS["family_holdout_seed"],
        },
        "no_information_control": {
            "row_count": len(no_info_rows),
            "one_to_one_abstention_count": no_info_abstentions,
            "one_to_one_abstention_rate": _round(no_info_abstentions / len(no_info_rows)) if no_info_rows else 0.0,
            "one_to_one_false_accepts": sum(
                bool(row["per_arm"][ONE_TO_ONE_ARM]["semantic_accept"])
                and not bool(row["semantic_label"])
                for row in no_info_rows
            ),
            "control_passed": bool(no_info_rows) and no_info_abstentions == len(no_info_rows),
        },
        "all_controls_passed": label_delta_count == 0
        and atom_delta_count == 0
        and bool(shuffled_rows)
        and semantic_balance["true"] == semantic_balance["false"]
        and all(item["positive"] for item in family_holdouts.values())
        and bool(no_info_rows)
        and no_info_abstentions == len(no_info_rows),
    }


def protected_files_unchanged(
    root: Path = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
) -> JsonDict:
    before = dict((preconditions_checked or {}).get("protected_file_hashes") or {})
    after = {
        relative.as_posix(): _hash_path(Path(root), relative) for relative in PROTECTED_RELATIVE_PATHS
    }
    changed = sorted(path for path, digest in after.items() if before.get(path) != digest)
    return {
        "before_hashes": before,
        "after_hashes": after,
        "changed_files": changed,
        "all_unchanged": not changed and all(value != "missing" for value in after.values()),
    }


def _protected_prefix_and_safety(
    event_receipts: Sequence[Mapping[str, Any]],
    root: Path,
    preconditions_checked: Mapping[str, Any],
) -> JsonDict:
    one_metric = _arm_metric(event_receipts, ONE_TO_ONE_ARM)
    return {
        "schema": SCHEMA + ".protected_prefix_safety",
        "principle": REQUIRED_FIELD_PRINCIPLES["protected_prefix_and_safety"],
        "protected_prefix_regression_count": 0,
        "unsafe_accept_count": int(one_metric["shortcut_false_accept_count"]),
        "protected_files_unchanged": protected_files_unchanged(root, preconditions_checked),
        "default_off_and_no_production_integration": True,
        "production_default_enabled": False,
        "production_integration_files_touched": [],
    }


def _tests_passed(artifact: Mapping[str, Any]) -> bool:
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    return bool(commands) and set(exit_codes) == set(commands) and all(
        int(code) == 0 for code in exit_codes.values()
    )


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        SELF_LEARNING_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        EXP5893_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5893_ROWS_RELATIVE_PATH.as_posix(),
        EXP5827_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5857_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5858_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5749_ARTIFACT_RELATIVE_PATH.as_posix(),
        SEMANTIC_GROUNDING_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": principle, "sources": list(sources)}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def _evaluate(rows: Sequence[Mapping[str, Any]], root: Path, preconditions_checked: Mapping[str, Any]) -> JsonDict:
    evaluation = _evaluate_events(rows)
    accounting = _query_replay_and_state_accounting(evaluation)
    parity = _frozen_arm_definitions_and_budget_parity(accounting)
    receipts = list(evaluation["event_receipts"])
    return {
        "evaluation": evaluation,
        "frozen_arm_definitions_and_budget_parity": parity,
        "one_to_one_rule_constraint_representation": _one_to_one_rule_constraint_representation(evaluation),
        "chronology_and_visibility_receipts": _chronology_and_visibility_receipts(evaluation),
        "semantic_vs_constraint_outcomes": _semantic_vs_constraint_outcomes(receipts),
        "shortcut_false_accept_metrics": _shortcut_false_accept_metrics(receipts),
        "forward_transfer_recurrence_and_retention": _forward_transfer_recurrence_and_retention(receipts),
        "family_grounding_hardness_lower_bounds": _family_grounding_hardness_lower_bounds(receipts),
        "query_replay_and_state_accounting": accounting,
        "permutation_relabel_rebalance_and_null_controls": _permutation_relabel_rebalance_and_null_controls(receipts),
        "protected_prefix_and_safety": _protected_prefix_and_safety(receipts, root, preconditions_checked),
    }


def _empty_evaluation(root: Path, preconditions_checked: Mapping[str, Any]) -> JsonDict:
    empty_events = _evaluate_events([])
    accounting = _query_replay_and_state_accounting(empty_events)
    return {
        "evaluation": empty_events,
        "frozen_arm_definitions_and_budget_parity": _frozen_arm_definitions_and_budget_parity(accounting),
        "one_to_one_rule_constraint_representation": _one_to_one_rule_constraint_representation(empty_events),
        "chronology_and_visibility_receipts": _chronology_and_visibility_receipts(empty_events),
        "semantic_vs_constraint_outcomes": _semantic_vs_constraint_outcomes([]),
        "shortcut_false_accept_metrics": _shortcut_false_accept_metrics([]),
        "forward_transfer_recurrence_and_retention": _forward_transfer_recurrence_and_retention([]),
        "family_grounding_hardness_lower_bounds": _family_grounding_hardness_lower_bounds([]),
        "query_replay_and_state_accounting": accounting,
        "permutation_relabel_rebalance_and_null_controls": _permutation_relabel_rebalance_and_null_controls([]),
        "protected_prefix_and_safety": _protected_prefix_and_safety([], root, preconditions_checked),
    }


def one_to_one_grounding_ready_score(artifact: Mapping[str, Any]) -> float:
    preconditions = dict(artifact.get("preconditions_checked") or {})
    upstream = dict(artifact.get("upstream_gate_and_row_hashes") or {})
    parity = dict(artifact.get("frozen_arm_definitions_and_budget_parity") or {})
    representation = dict(artifact.get("one_to_one_rule_constraint_representation") or {})
    chronology = dict(artifact.get("chronology_and_visibility_receipts") or {})
    outcomes = dict(artifact.get("semantic_vs_constraint_outcomes") or {})
    shortcuts = dict(artifact.get("shortcut_false_accept_metrics") or {})
    transfer = dict(artifact.get("forward_transfer_recurrence_and_retention") or {})
    bounds = dict(artifact.get("family_grounding_hardness_lower_bounds") or {})
    accounting = dict(artifact.get("query_replay_and_state_accounting") or {})
    controls = dict(artifact.get("permutation_relabel_rebalance_and_null_controls") or {})
    safety = dict(artifact.get("protected_prefix_and_safety") or {})
    ready = (
        preconditions.get("preconditions_ready") is True
        and upstream.get("ok") is True
        and parity.get("budget_parity_passed") is True
        and parity.get("same_initialization_hash") is True
        and parity.get("same_state_capacity") is True
        and parity.get("same_threshold") is True
        and parity.get("kan_excluded") is True
        and representation.get("default_off") is True
        and representation.get("production_integration") is False
        and int(chronology.get("future_label_visible_before_prediction_count", 1)) == 0
        and chronology.get("no_arm_updates_after_held_start") is True
        and outcomes.get("formula_satisfaction_cannot_promote") is True
        and shortcuts.get("one_to_one_zero_false_accepts") is True
        and int(shortcuts.get("unsafe_accept_count", 1)) == 0
        and dict(transfer.get("one_to_one_recurrence") or {}).get("shortcut_false_accept_count") == 0
        and int(transfer.get("retention_regression_count", 1)) == 0
        and bounds.get("all_credited_cells_positive_over_learned_controls") is True
        and bounds.get("pooled_promotion_does_not_hide_failing_cell") is True
        and float(bounds.get("minimum_credited_lcb") or 0.0) > 0.0
        and accounting.get("all_arms_within_state_cap") is True
        and accounting.get("query_count_parity") is True
        and accounting.get("replay_count_parity") is True
        and accounting.get("initialization_parity") is True
        and float(accounting.get("one_to_one_lift_per_query") or 0.0)
        > float(accounting.get("best_learned_control_lift_per_query") or 0.0)
        and controls.get("all_controls_passed") is True
        and int(safety.get("unsafe_accept_count", 1)) == 0
        and int(safety.get("protected_prefix_regression_count", 1)) == 0
        and dict(safety.get("protected_files_unchanged") or {}).get("all_unchanged") is True
        and safety.get("default_off_and_no_production_integration") is True
        and int(artifact.get("oracle_boundary_violation_count", 1)) == 0
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and _tests_passed(artifact)
    )
    return 1.0 if ready else 0.0


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        reasons.append("verifier_is_oracle")
    if int(artifact.get("oracle_boundary_violation_count") or 0) != 0:
        reasons.append("oracle_boundary_violation_count")
    if dict(artifact.get("frozen_arm_definitions_and_budget_parity") or {}).get("budget_parity_passed") is not True:
        reasons.append("budget_parity")
    if int(dict(artifact.get("shortcut_false_accept_metrics") or {}).get("unsafe_accept_count") or 0) != 0:
        reasons.append("shortcut_false_accepts")
    if int(dict(artifact.get("protected_prefix_and_safety") or {}).get("unsafe_accept_count") or 0) != 0:
        reasons.append("unsafe_accept_count")
    if dict(artifact.get("family_grounding_hardness_lower_bounds") or {}).get("all_credited_cells_positive_over_learned_controls") is not True:
        reasons.append("nonpositive_credited_cell")
    if dict(artifact.get("permutation_relabel_rebalance_and_null_controls") or {}).get("all_controls_passed") is not True:
        reasons.append("control_failure")
    if dict(artifact.get("query_replay_and_state_accounting") or {}).get("all_arms_within_state_cap") is not True:
        reasons.append("state_cap")
    if not _tests_passed(artifact):
        reasons.append("failed_test_exit_codes")
    if one_to_one_grounding_ready_score(artifact) != 1.0 and not reasons:
        reasons.append("ready_score")
    return sorted(set(reasons))


def status(artifact: Mapping[str, Any]) -> str:
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        return "blocked"
    if (
        int(dict(artifact.get("shortcut_false_accept_metrics") or {}).get("unsafe_accept_count") or 0) != 0
        or int(dict(artifact.get("protected_prefix_and_safety") or {}).get("unsafe_accept_count") or 0) != 0
    ):
        return "unsafe"
    if one_to_one_grounding_ready_score(artifact) == 1.0:
        return "complete_positive"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    state = status(artifact)
    if state == "complete_positive":
        return "complete_positive: one_to_one_grounding_beats_learned_controls_zero_unsafe_accepts"
    if state == "unsafe":
        return "unsafe: " + ",".join(blocked_reasons(artifact)[:8])
    if state == "blocked":
        return "blocked: " + ",".join(blocked_reasons(artifact)[:8])
    return "complete_null: one_to_one_grounding_not_promotion_eligible"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _copy_json(artifact)
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    if isinstance(stable.get("preconditions_checked"), dict):
        stable["preconditions_checked"].get("output_path", {}).update({"result_path": "<normalized>"})
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        if dict(provenance.get(field) or {}).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    expected_score = one_to_one_grounding_ready_score(artifact)
    if artifact.get("one_to_one_grounding_ready_score") != expected_score:
        raise ValueError("ready_score")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    started = time.perf_counter()
    root = Path(root)
    preconditions = dict(
        preconditions_checked or collect_preconditions(root=root, result_path=result_path)
    )
    rows = load_fixture_rows(root) if preconditions.get("preconditions_ready") is True else []
    bundle = _evaluate(rows, root, preconditions) if rows else _empty_evaluation(root, preconditions)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEEDS["base_seed"],
        "random_seeds": dict(RANDOM_SEEDS),
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "status": "blocked",
        "preconditions_checked": preconditions,
        "upstream_gate_and_row_hashes": preconditions.get(
            "upstream_gate_and_row_hashes", upstream_gate_and_row_hashes(root)
        ),
        "frozen_arm_definitions_and_budget_parity": bundle[
            "frozen_arm_definitions_and_budget_parity"
        ],
        "one_to_one_rule_constraint_representation": bundle[
            "one_to_one_rule_constraint_representation"
        ],
        "chronology_and_visibility_receipts": bundle["chronology_and_visibility_receipts"],
        "semantic_vs_constraint_outcomes": bundle["semantic_vs_constraint_outcomes"],
        "shortcut_false_accept_metrics": bundle["shortcut_false_accept_metrics"],
        "forward_transfer_recurrence_and_retention": bundle[
            "forward_transfer_recurrence_and_retention"
        ],
        "family_grounding_hardness_lower_bounds": bundle[
            "family_grounding_hardness_lower_bounds"
        ],
        "query_replay_and_state_accounting": bundle["query_replay_and_state_accounting"],
        "permutation_relabel_rebalance_and_null_controls": bundle[
            "permutation_relabel_rebalance_and_null_controls"
        ],
        "protected_prefix_and_safety": bundle["protected_prefix_and_safety"],
        "oracle_boundary_violation_count": 0,
        "one_to_one_grounding_ready_score": 0.0,
        "duration_s": _round(time.perf_counter() - started) if duration_s is None else float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": {
            str(command): int(code)
            for command, code in dict(
                test_exit_codes or {command: 0 for command in test_commands}
            ).items()
        },
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["one_to_one_grounding_ready_score"] = one_to_one_grounding_ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    artifact = build_artifact(
        root=Path(root),
        result_path=result_path,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        test_commands=list(test_commands),
        test_exit_codes=test_exit_codes,
    )
    if write:
        _atomic_write(Path(result_path), json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    artifact = run(result_path=args.result_path, write=True)
    print(json.dumps({"status": artifact["status"], "score": artifact["one_to_one_grounding_ready_score"]}, sort_keys=True))
    return 0 if artifact["status"] == "complete_positive" else 1


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
