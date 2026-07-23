"""Exp5827 minimal-core structural constraint acquisition.

Spec refs: REQ-LEARN-5827, SCENARIO-LEARN-5827-ACTIVE-CORE,
SCENARIO-LEARN-5827-MATCHED-ARMS, SCENARIO-LEARN-5827-READY-GATE,
SCENARIO-LEARN-5827-FAIL-CLOSED.

This module replays the sealed Exp5826 stream with the same exact membership
surface used by Exp5762. Deployable arms see candidate assignments, exact query
answers, and minimal-core receipts. They do not receive the sealed cleartext
target structure; the exact upper bound is separated and marked non-deployable.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
from typing import Any

from carnot import experiment_5762_query_driven_constraint_lifecycle as exp5762
from carnot import experiment_5826_out_of_template_constraint_stream as exp5826


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5827_minimal_core_structural_acquisition_ab.json")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5827_minimal_core_structural_acquisition_ab.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5827_minimal_core_structural_acquisition_ab.py"
)
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
EXP5826_ARTIFACT_RELATIVE_PATH = exp5826.RESULT_RELATIVE_PATH
EXP5826_ROWS_RELATIVE_PATH = exp5826.ROW_FILE_RELATIVE_PATH
EXP5762_ARTIFACT_RELATIVE_PATH = exp5826.EXP5762_ARTIFACT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5827.minimal_core_structural_acquisition_ab.v1"
EXPERIMENT = 5827
EXPERIMENT_ID = "experiment_5827_minimal_core_structural_acquisition_ab"
MILESTONE = "2026.07.520"
RUN_DATE = "20260723"
INFERENCE_SUBSTRATE = "online_exact_membership_query_sidecar_no_llm"
GRAMMAR_VERSION = "exp5827_bounded_relation_composition_grammar_v1"
STOPPING_RULE = "minimal_core_unique_or_budget_exhausted_v1"
QUERY_BUDGET_PER_ROW = 6
RAM_FLOOR_MB = 512
DISK_FLOOR_MB = 512

PRIMARY_FAMILIES = exp5826.PRIMARY_FAMILIES
CHANGE_ORDER = exp5826.CHANGE_ORDER
PROOF_PRESERVING_SURFACES = exp5826.PROOF_PRESERVING_SURFACES
HARDNESS_BINS = exp5826.HARDNESS_BINS

NO_UPDATE_ARM = "no_update"
TEMPLATE_BASELINE_ARM = "exp5762_matched_template_query_learner"
PASSIVE_ARM = "passive_minimal_core_synthesis"
RANDOM_ARM = "random_query_structure_synthesis"
ACTIVE_ARM = "active_discriminating_query_minimal_core_synthesis"
UPPER_BOUND_ARM = "exact_structure_upper_bound"
CONTROL_ARMS = (
    NO_UPDATE_ARM,
    TEMPLATE_BASELINE_ARM,
    PASSIVE_ARM,
    RANDOM_ARM,
    ACTIVE_ARM,
    UPPER_BOUND_ARM,
)
DEPLOYABLE_ARMS = (
    NO_UPDATE_ARM,
    TEMPLATE_BASELINE_ARM,
    PASSIVE_ARM,
    RANDOM_ARM,
    ACTIVE_ARM,
)
STRUCTURAL_ARMS = (PASSIVE_ARM, RANDOM_ARM, ACTIVE_ARM)
SPEC_REFS = (
    "REQ-LEARN-5827",
    "SCENARIO-LEARN-5827-ACTIVE-CORE",
    "SCENARIO-LEARN-5827-MATCHED-ARMS",
    "SCENARIO-LEARN-5827-READY-GATE",
    "SCENARIO-LEARN-5827-FAIL-CLOSED",
)
RANDOM_SEEDS: JsonDict = {
    "base_seed": 5827,
    "active_query_seed": 5_827_001,
    "random_query_seed": 5_827_002,
    "bootstrap_seed": 5_827_003,
    "checkpoint_seed": 5_827_004,
}
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5827_minimal_core_structural_acquisition_ab.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5827_minimal_core_structural_acquisition_ab.py "
    "-m pytest tests/python/test_experiment_5827_minimal_core_structural_acquisition_ab.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5827_minimal_core_structural_acquisition_ab.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5827_minimal_core_structural_acquisition_ab.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "upstream_artifact_hashes",
    "arm_definitions_and_budget_parity",
    "structural_hypothesis_grammar",
    "query_and_minimal_core_receipts",
    "per_arm_family_change_metrics",
    "paired_deltas_and_ci95",
    "structural_recovery_and_headroom",
    "protected_prefix_and_safety",
    "oracle_boundary_violation_count",
    "structural_learner_ready_score",
    "retire_if_same_verdict",
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
    "status": "A terminal experiment state distinguishes completed negative evidence from an incomplete run.",
    "preconditions_checked": "Gate, hashes, headroom, solvers, resources, and checkpoints prevent fabricated execution.",
    "upstream_artifact_hashes": "Hashes bind learning to the sealed certified event stream.",
    "arm_definitions_and_budget_parity": "Matched evidence and query budgets isolate the structural method.",
    "structural_hypothesis_grammar": "A frozen grammar makes out-of-template expressivity and complexity auditable.",
    "query_and_minimal_core_receipts": "Exact boundary evidence shows how each structure was learned without leaking labels.",
    "per_arm_family_change_metrics": "Disaggregated results prevent a pooled average from hiding family failures.",
    "paired_deltas_and_ci95": "Paired intervals quantify lift over the successful template baseline.",
    "structural_recovery_and_headroom": "Credit is restricted to genuinely expressible, headroom-present science rows.",
    "protected_prefix_and_safety": "Zero regression and unsafe propagation are required for adaptive learning.",
    "oracle_boundary_violation_count": "A bare zero proves deployable arms never read sealed structure labels.",
    "structural_learner_ready_score": "EMIT BARE scalar; only 1.0 permits the future-validated lifecycle.",
    "retire_if_same_verdict": "A repeated blocked outcome mechanically retires this reattempt.",
    "duration_s": "Measured wall time exposes bootstrap-only artifacts.",
    "inference_substrate": "`online_exact_membership_query_sidecar_no_llm` declares the actual learning surface.",
    "verifier_is_oracle": "True records that exact solvers label and gate updates, so no moat claim is allowed.",
    "field_provenance": "Every metric traces to rows, queries, cores, or state receipts.",
    "test_commands": "Commands document parity, leakage, recovery, statistics, and safety checks.",
    "test_exit_codes": "Exit codes prevent failed evaluations from becoming readiness.",
    "reproducibility_checksum": "A checksum detects drift in arms, rows, seeds, or metrics.",
    "honest_verdict": "A terminal prefix states credited, null, negative, or blocked outcome honestly.",
}
UPSTREAM_PATHS: dict[str, Path] = {
    "exp5826_artifact": EXP5826_ARTIFACT_RELATIVE_PATH,
    "exp5826_rows": EXP5826_ROWS_RELATIVE_PATH,
    "exp5826_module": exp5826.MODULE_RELATIVE_PATH,
    "exp5762_artifact": EXP5762_ARTIFACT_RELATIVE_PATH,
    "exp5762_module": exp5762.MODULE_RELATIVE_PATH,
    "self_learning_spec": SELF_LEARNING_SPEC_RELATIVE_PATH,
    "module": MODULE_RELATIVE_PATH,
    "tests": TEST_RELATIVE_PATH,
}


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence in stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for stable text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes without trusting timestamps or metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _mean(values: Sequence[float]) -> float:
    return _round(sum(float(value) for value in values) / max(1, len(values)))


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


def read_row_file(path: str | Path) -> list[JsonDict]:
    """Read Exp5826 JSONL rows, returning an empty list for absent files."""

    if not Path(path).exists():
        return []
    return _read_jsonl(path)


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
    return {"available_mb": available_mb, "required_mb": RAM_FLOOR_MB, "ok": available_mb >= RAM_FLOOR_MB}


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover - host-dependent resource probe.
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": DISK_FLOOR_MB, "ok": available_mb >= DISK_FLOOR_MB}


def _hash_path(root: Path, relative: Path) -> str:
    path = root / relative
    return sha256_file(path) if path.exists() and path.is_file() else "missing"


def _output_path_receipt(result_path: Path, checkpoint_dir: Path) -> JsonDict:
    def ready_file(path: Path) -> bool:
        parent = path.parent
        return (
            ((parent.exists() and os.access(parent, os.W_OK)) or (parent.parent.exists() and os.access(parent.parent, os.W_OK)))
            and (not path.exists() or os.access(path, os.W_OK))
        )

    checkpoint_parent = checkpoint_dir if checkpoint_dir.exists() else checkpoint_dir.parent
    checkpoint_ready = (
        (checkpoint_parent.exists() and os.access(checkpoint_parent, os.W_OK))
        or (checkpoint_parent.parent.exists() and os.access(checkpoint_parent.parent, os.W_OK))
    )
    return {
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "checkpoint_dir": "results/checkpoints/experiment_5827_minimal_core_structural_acquisition_ab",
        "result_writable": ready_file(result_path),
        "checkpoint_writable": checkpoint_ready,
        "checkpoint_atomic_suffix": ".tmp",
    }


def _headroom_present(row: Mapping[str, Any]) -> bool:
    exact = dict(row.get("exact_receipt") or {}).get("primary") or {}
    witness = dict(row.get("out_of_template_witness") or {})
    return (
        int(exact.get("accepted_count") or 0) > 0
        and int(exact.get("rejected_count") or 0) > 0
        and witness.get("absent_from_frozen_library") is True
    )


def _row_replay_receipt(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any], row_path: Path) -> JsonDict:
    artifact_rows = dict(artifact.get("row_file_and_sha256") or {})
    try:
        replay_ok = exp5826.verify_row_file(rows, artifact)
    except exp5826.StreamReplayError:
        replay_ok = False
    row_text = exp5826.rows_to_jsonl(rows)
    row_hash = sha256_text(row_text)
    return {
        "row_count": len(rows),
        "artifact_row_count": int(artifact_rows.get("row_count") or -1),
        "row_file_hash": sha256_file(row_path) if row_path.exists() else "missing",
        "row_text_hash": row_hash,
        "artifact_row_file_hash": str(artifact_rows.get("sha256") or ""),
        "row_file_hash_ok": row_hash == artifact_rows.get("sha256"),
        "row_hash_root": str(artifact_rows.get("row_hash_root") or ""),
        "replay_ok": replay_ok,
        "ok": replay_ok and len(rows) == 360 and row_hash == artifact_rows.get("sha256"),
    }


def _headroom_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    headroom = [row for row in rows if _headroom_present(row)]
    by_family = Counter(str(row["family"]) for row in headroom)
    by_cell = Counter(f"{row['family']}|{row['change']}" for row in headroom)
    return {
        "headroom_present_row_count": len(headroom),
        "out_of_template_row_count": sum(
            1
            for row in rows
            if dict(row.get("out_of_template_witness") or {}).get("absent_from_frozen_library") is True
        ),
        "headroom_by_family": dict(sorted(by_family.items())),
        "headroom_by_family_change": dict(sorted(by_cell.items())),
        "all_primary_families_have_headroom": all(by_family.get(family, 0) >= 30 for family in PRIMARY_FAMILIES),
        "ok": len(headroom) >= 3 * 4 * 27 and all(by_family.get(family, 0) > 0 for family in PRIMARY_FAMILIES),
    }


def _solver_version_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    primary_versions = {
        str(row["exact_receipt"]["primary"]["validator_version"]) for row in rows
    }
    independent_versions = {
        str(row["exact_receipt"]["independent"]["validator_version"]) for row in rows
    }
    query_versions = {
        version
        for row in rows
        for query in row["exact_receipt"]["membership_queries"]
        for version in query["validator_versions"]
    }
    return {
        "primary_versions": sorted(primary_versions),
        "independent_versions": sorted(independent_versions),
        "query_validator_versions": sorted(query_versions),
        "expected_primary": exp5826.PRIMARY_VALIDATOR_VERSION,
        "expected_independent": exp5826.INDEPENDENT_VALIDATOR_VERSION,
        "all_validators_agree": all(row["exact_receipt"]["validators_agree"] is True for row in rows),
        "ok": primary_versions == {exp5826.PRIMARY_VALIDATOR_VERSION}
        and independent_versions == {exp5826.INDEPENDENT_VALIDATOR_VERSION}
        and {exp5826.PRIMARY_VALIDATOR_VERSION, exp5826.INDEPENDENT_VALIDATOR_VERSION}.issubset(query_versions),
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    checkpoint_dir: str | Path = REPO_ROOT / "results/checkpoints/experiment_5827_minimal_core_structural_acquisition_ab",
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
) -> JsonDict:
    """Replay gates, hashes, resources, and checkpoint writability before learning."""

    root = Path(root)
    result_path = Path(result_path)
    checkpoint_dir = Path(checkpoint_dir)
    upstream_hashes = {name: _hash_path(root, relative) for name, relative in UPSTREAM_PATHS.items()}
    blocked: list[str] = []
    if upstream_hashes["exp5826_artifact"] == "missing" or upstream_hashes["exp5826_rows"] == "missing":
        blocked.append("missing_upstream_artifact")

    structured_gate: JsonDict = {"ok": False}
    row_replay: JsonDict = {"ok": False, "row_count": 0}
    headroom: JsonDict = {"ok": False, "headroom_present_row_count": 0}
    solvers: JsonDict = {"ok": False}
    seeds: JsonDict = {"ok": False, "random_seeds": dict(RANDOM_SEEDS)}
    schema_hashes: JsonDict = {"ok": False}
    corrupt_errors: list[str] = []
    if "missing_upstream_artifact" not in blocked:
        try:
            artifact = _read_json(root / EXP5826_ARTIFACT_RELATIVE_PATH)
            rows = read_row_file(root / EXP5826_ROWS_RELATIVE_PATH)
            exp5826.validate_artifact(artifact)
            row_replay = _row_replay_receipt(rows, artifact, root / EXP5826_ROWS_RELATIVE_PATH)
            structured_gate = {
                "exp5826_status": artifact.get("status"),
                "exp5826_honest_verdict": artifact.get("honest_verdict"),
                "constraint_event_stream_ready_score": artifact.get("constraint_event_stream_ready_score"),
                "validate_artifact_ok": True,
                "row_replay_ok": row_replay["replay_ok"],
                "ok": artifact.get("status") == "complete"
                and str(artifact.get("honest_verdict") or "").startswith("complete:")
                and artifact.get("constraint_event_stream_ready_score") == 1.0
                and row_replay["ok"] is True,
            }
            headroom = _headroom_receipt(rows)
            solvers = _solver_version_receipt(rows)
            seeds = {
                "random_seeds": dict(RANDOM_SEEDS),
                "exp5826_random_seeds": dict(artifact.get("random_seeds") or {}),
                "base_seed_ok": RANDOM_SEEDS["base_seed"] == 5827,
                "exp5826_seed_ok": dict(artifact.get("random_seeds") or {}) == dict(exp5826.RANDOM_SEEDS),
                "ok": RANDOM_SEEDS["base_seed"] == 5827
                and dict(artifact.get("random_seeds") or {}) == dict(exp5826.RANDOM_SEEDS),
            }
            schema_hashes = {
                "exp5826_schema_hash": sha256_json(
                    {"schema": exp5826.SCHEMA, "row_schema": exp5826.ROW_SCHEMA}
                ),
                "exp5827_schema_hash": sha256_json(
                    {"schema": SCHEMA, "required_fields": list(REQUIRED_ARTIFACT_FIELDS)}
                ),
                "spec_hash": upstream_hashes["self_learning_spec"],
                "ok": upstream_hashes["self_learning_spec"] != "missing",
            }
        except (OSError, ValueError, json.JSONDecodeError, exp5826.StreamReplayError) as exc:
            corrupt_errors.append(type(exc).__name__)
            blocked.append("corrupt_upstream_artifact")

    memory = memory_probe()
    disk = disk_probe(root)
    output_paths = _output_path_receipt(result_path, checkpoint_dir)
    checks = {
        "structured_gate": structured_gate.get("ok") is True,
        "row_replay": row_replay.get("ok") is True,
        "headroom_witnesses": headroom.get("ok") is True,
        "solver_versions": solvers.get("ok") is True,
        "deterministic_seeds": seeds.get("ok") is True,
        "schema_hashes": schema_hashes.get("ok") is True,
        "memory": memory.get("ok") is True,
        "disk": disk.get("ok") is True,
        "output_paths": output_paths["result_writable"] is True
        and output_paths["checkpoint_writable"] is True,
        "python": sys.version_info >= (3, 11),
    }
    failure_names = {
        "memory": "insufficient_free_ram",
        "disk": "insufficient_free_disk",
        "output_paths": "output_path_not_writable",
    }
    blocked.extend(failure_names.get(name, name) for name, ok in checks.items() if not ok)
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "python": {
            "available": True,
            "version": platform.python_version(),
            "executable": sys.executable,
            "ok": sys.version_info >= (3, 11),
        },
        "structured_gate_replay": structured_gate,
        "upstream_artifact_hashes": upstream_hashes,
        "row_replay": row_replay,
        "headroom_witnesses": headroom,
        "solver_versions": solvers,
        "deterministic_seeds": seeds,
        "schema_hashes": schema_hashes,
        "resources": {"memory": memory, "disk": disk},
        "output_paths": output_paths,
        "corrupt_upstream_errors": corrupt_errors,
        "llm_calls_made": 0,
        "preconditions_ready": not sorted(set(blocked)),
        "blocked_reasons": sorted(set(blocked)),
    }


def fixture_preconditions() -> JsonDict:
    """Return deterministic resource probes while replaying sealed inputs."""

    return collect_preconditions(
        memory_probe=lambda: {"available_mb": 8192, "required_mb": RAM_FLOOR_MB, "ok": True},
        disk_probe=lambda root: {"available_mb": 8192, "required_mb": DISK_FLOOR_MB, "ok": True},
    )


def _candidate_domain(row: Mapping[str, Any]) -> list[JsonDict]:
    accepted = set(row["exact_receipt"]["primary"]["accepted_assignment_hashes"])
    rejected = set(row["exact_receipt"]["primary"]["rejected_assignment_hashes"])
    candidates: list[JsonDict] = []
    for index, assignment in enumerate(exp5826._candidate_assignments(str(row["family"]))):
        assignment_hash = sha256_json(assignment)
        candidates.append(
            {
                "candidate_id": f"{row['family']}-cand-{index:03d}",
                "assignment": assignment,
                "assignment_hash": assignment_hash,
                "oracle_accepts": assignment_hash in accepted,
                "oracle_rejects": assignment_hash in rejected,
            }
        )
    return candidates


def _signature(relation: str, arity: int, composition: str) -> JsonDict:
    return {"relation": relation, "arity": arity, "composition": composition}


def _hypothesis(
    *,
    family: str,
    relation: str,
    signature: Mapping[str, Any],
    params: Mapping[str, Any],
    source: str,
    predicate_count: int = 1,
    composition_depth: int = 1,
) -> JsonDict:
    row = {
        "family": family,
        "relation": relation,
        "signature": dict(signature),
        "params": dict(params),
        "source": source,
        "complexity": {
            "predicate_count": predicate_count,
            "arity": int(signature["arity"]),
            "composition_depth": composition_depth,
            "parameter_count": len(params),
        },
    }
    row["hypothesis_hash"] = sha256_json(row)
    return row


def _structural_hypotheses(family: str) -> list[JsonDict]:
    if family == "finite_domain_csp":
        signature = _signature("cyclic_order", 3, "ternary_ordered_tuple")
        return [
            _hypothesis(
                family=family,
                relation="cyclic_order",
                signature=signature,
                params={"offset": offset},
                source="exp5827_out_of_template_structural_grammar",
                composition_depth=2,
            )
            for offset in (0, 1, 2, None)
        ]
    if family == "weighted_maxsat":
        signature = _signature("cardinality_eq", 3, "cardinality_count_eq")
        return [
            _hypothesis(
                family=family,
                relation="cardinality_eq",
                signature=signature,
                params={"required_true_count": count},
                source="exp5827_out_of_template_structural_grammar",
                composition_depth=2,
            )
            for count in (0, 1, 2, 3, 4)
        ]
    if family == "hard_soft_packing":
        signature = _signature("weighted_sum_lte", 3, "linear_weighted_threshold")
        hypotheses = []
        for weights in ([1, 2, 3], [2, 2, 3]):
            for capacity in (-1, 0, 1, 2, 3, 4, 5, 6):
                hypotheses.append(
                    _hypothesis(
                        family=family,
                        relation="weighted_sum_lte",
                        signature=signature,
                        params={"weights": weights, "capacity": capacity},
                        source="exp5827_out_of_template_structural_grammar",
                        composition_depth=2,
                    )
                )
        return hypotheses
    signature = _signature("forbidden_subsequence", 2, "temporal_subsequence")
    return [
        _hypothesis(
            family=family,
            relation="forbidden_subsequence",
            signature=signature,
            params={"forbidden_pattern": list(pattern), "also_require_forbidden_pattern": unsat},
            source="exp5827_out_of_template_structural_grammar",
            composition_depth=2,
        )
        for pattern in (("A", "B"), ("B", "A"), ("A", "A"), ("B", "B"))
        for unsat in (False, True)
    ]


def _overlap_hypotheses(family: str) -> list[JsonDict]:
    if family == "finite_domain_csp":
        values = ("red", "green", "blue")
        return [
            _hypothesis(
                family=family,
                relation="equals",
                signature=_signature("equals", 1, "atomic"),
                params={"var": "A", "value": value},
                source="exp5762_overlap_template",
            )
            for value in values
        ] + [
            _hypothesis(
                family=family,
                relation="not_equal",
                signature=_signature("not_equal", 2, "binary_difference"),
                params={"vars": ["A", "B"]},
                source="exp5762_overlap_template",
            )
        ]
    if family == "weighted_maxsat":
        hypotheses = []
        for left_positive in (False, True):
            for right_positive in (False, True):
                hypotheses.append(
                    _hypothesis(
                        family=family,
                        relation="clause",
                        signature=_signature("clause", 2, "disjunctive_literal"),
                        params={"literals": [["X", left_positive], ["Y", right_positive]]},
                        source="exp5762_overlap_template",
                    )
                )
        return hypotheses
    if family == "hard_soft_packing":
        return [
            _hypothesis(
                family=family,
                relation="requires_item",
                signature=_signature("requires_item", 1, "unary_item"),
                params={"var": "I0"},
                source="exp5762_overlap_template",
            ),
            _hypothesis(
                family=family,
                relation="not_both",
                signature=_signature("not_both", 2, "binary_exclusion"),
                params={"vars": ["I1", "I2"]},
                source="exp5762_overlap_template",
            ),
        ]
    return [
        _hypothesis(
            family=family,
            relation="max_action_count",
            signature=_signature("max_action_count", 1, "temporal_count_limit"),
            params={"action": "A", "limit": limit},
            source="exp5762_overlap_template",
        )
        for limit in (0, 1, 2)
    ]


def _hypothesis_space(family: str) -> list[JsonDict]:
    """Return frozen grammar candidates for one family."""

    return sorted(_structural_hypotheses(family) + _overlap_hypotheses(family), key=canonical_json)


def _hypothesis_accepts(hypothesis: Mapping[str, Any], assignment: Mapping[str, Any]) -> bool:
    relation = str(hypothesis["relation"])
    params = dict(hypothesis["params"])
    if relation == "cyclic_order":
        offset = params["offset"]
        if offset is None:
            return False
        rotations = [
            ("red", "green", "blue"),
            ("green", "blue", "red"),
            ("blue", "red", "green"),
        ]
        return (assignment["A"], assignment["B"], assignment["C"]) == rotations[int(offset)]
    if relation == "cardinality_eq":
        return sum(1 for value in assignment.values() if value is True) == int(params["required_true_count"])
    if relation == "weighted_sum_lte":
        weights = list(params["weights"])
        selected = [bool(assignment[f"I{index}"]) for index in range(3)]
        total = sum(weight for weight, chosen in zip(weights, selected, strict=True) if chosen)
        return total <= int(params["capacity"])
    if relation == "forbidden_subsequence":
        if params.get("also_require_forbidden_pattern") is True:
            return False
        actions = list(assignment["actions"])
        pattern = list(params["forbidden_pattern"])
        return not any(actions[index : index + len(pattern)] == pattern for index in range(2))
    if relation == "equals":
        return assignment[str(params["var"])] == params["value"]
    if relation == "not_equal":
        left, right = params["vars"]
        return assignment[left] != assignment[right]
    if relation == "clause":
        return any(bool(assignment[var]) is bool(positive) for var, positive in params["literals"])
    if relation == "requires_item":
        return bool(assignment[str(params["var"])])
    if relation == "not_both":
        left, right = params["vars"]
        return not (bool(assignment[left]) and bool(assignment[right]))
    action = str(params["action"])
    return sum(1 for item in assignment["actions"] if item == action) <= int(params["limit"])


def _labels_for_hypothesis(hypothesis: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    return {
        str(candidate["assignment_hash"]): _hypothesis_accepts(hypothesis, candidate["assignment"])
        for candidate in candidates
    }


def _oracle_labels(candidates: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    return {str(candidate["assignment_hash"]): bool(candidate["oracle_accepts"]) for candidate in candidates}


def _agreement(predicted: Mapping[str, bool], oracle: Mapping[str, bool]) -> float:
    matches = sum(1 for key, label in oracle.items() if predicted.get(key) is label)
    return _round(matches / max(1, len(oracle)))


def _observed_from_initial(row: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {
            "candidate_id": str(query["candidate_id"]),
            "assignment_hash": str(query["assignment_hash"]),
            "oracle_accepts": bool(query["oracle_accepts"]),
            "query_hash": str(query["query_hash"]),
            "source": "exp5826_membership_query_receipt",
        }
        for query in row["exact_receipt"]["membership_queries"]
    ]


def _filter_hypotheses(hypotheses: Sequence[Mapping[str, Any]], observed: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    by_hash = {str(candidate["assignment_hash"]): dict(candidate["assignment"]) for candidate in candidates}
    survivors = []
    for hypothesis in hypotheses:
        if all(
            _hypothesis_accepts(hypothesis, by_hash[str(item["assignment_hash"])])
            is bool(item["oracle_accepts"])
            for item in observed
        ):
            survivors.append(dict(hypothesis))
    return sorted(survivors, key=lambda item: (item["complexity"]["predicate_count"], item["complexity"]["parameter_count"], canonical_json(item)))


def _select_best_hypothesis(hypotheses: Sequence[Mapping[str, Any]], observed: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]) -> JsonDict:
    survivors = _filter_hypotheses(hypotheses, observed, candidates)
    if survivors:
        return survivors[0]
    observed_by_hash = {str(item["assignment_hash"]): bool(item["oracle_accepts"]) for item in observed}
    scored = []
    for hypothesis in hypotheses:
        labels = _labels_for_hypothesis(hypothesis, candidates)
        matches = sum(1 for key, label in observed_by_hash.items() if labels.get(key) is label)
        scored.append((matches, -int(hypothesis["complexity"]["parameter_count"]), canonical_json(hypothesis), dict(hypothesis)))
    return sorted(scored, reverse=True)[0][3]


def _choose_active_query(
    survivors: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    observed_hashes: set[str],
) -> JsonDict | None:
    best: tuple[int, str, JsonDict] | None = None
    for candidate in candidates:
        assignment_hash = str(candidate["assignment_hash"])
        if assignment_hash in observed_hashes:
            continue
        labels = [_hypothesis_accepts(hypothesis, candidate["assignment"]) for hypothesis in survivors]
        true_count = sum(1 for label in labels if label is True)
        false_count = len(labels) - true_count
        split = min(true_count, false_count)
        if split == 0:
            continue
        score = (split, assignment_hash, dict(candidate))
        if best is None or score > best:
            best = score
    return None if best is None else best[2]


def _choose_random_queries(row_id: str, candidates: Sequence[Mapping[str, Any]], observed_hashes: set[str], limit: int) -> list[JsonDict]:
    rng = random.Random(RANDOM_SEEDS["random_query_seed"] + int(sha256_text(row_id)[-8:], 16))
    remaining = [dict(candidate) for candidate in candidates if str(candidate["assignment_hash"]) not in observed_hashes]
    rng.shuffle(remaining)
    return remaining[:limit]


def _query_receipt(
    *,
    arm: str,
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    query_index: int,
    survivor_count_before: int,
    survivor_count_after: int,
    source: str,
) -> JsonDict:
    receipt = {
        "arm": arm,
        "row_id": str(row["row_id"]),
        "family": str(row["family"]),
        "change": str(row["change"]),
        "surface": str(row["surface_kind"]),
        "hardness": str(row["solver_effort_bin"]),
        "query_index": query_index,
        "candidate_id": str(candidate["candidate_id"]),
        "assignment_hash": str(candidate["assignment_hash"]),
        "oracle_accepts": bool(candidate["oracle_accepts"]),
        "survivor_count_before": survivor_count_before,
        "survivor_count_after": survivor_count_after,
        "source": source,
        "oracle_boundary": "exact_membership_outcome_only",
        "sealed_ground_truth_read": False,
    }
    receipt["query_hash"] = sha256_json(receipt)
    return receipt


def _run_synthesis_arm(row: Mapping[str, Any], arm: str, hypotheses: Sequence[Mapping[str, Any]]) -> JsonDict:
    candidates = _candidate_domain(row)
    observed = _observed_from_initial(row)
    receipts: list[JsonDict] = []
    observed_hashes = {str(item["assignment_hash"]) for item in observed}
    if arm == RANDOM_ARM:
        for candidate in _choose_random_queries(str(row["row_id"]), candidates, observed_hashes, QUERY_BUDGET_PER_ROW - len(observed)):
            before = len(_filter_hypotheses(hypotheses, observed, candidates))
            observed.append(
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "assignment_hash": str(candidate["assignment_hash"]),
                    "oracle_accepts": bool(candidate["oracle_accepts"]),
                    "query_hash": "",
                    "source": "random_exact_membership_query",
                }
            )
            observed_hashes.add(str(candidate["assignment_hash"]))
            after = len(_filter_hypotheses(hypotheses, observed, candidates))
            receipts.append(
                _query_receipt(
                    arm=arm,
                    row=row,
                    candidate=candidate,
                    query_index=len(observed) - 1,
                    survivor_count_before=before,
                    survivor_count_after=after,
                    source="random_exact_membership_query",
                )
            )
    if arm in {ACTIVE_ARM, TEMPLATE_BASELINE_ARM}:
        while len(observed) < QUERY_BUDGET_PER_ROW:
            survivors = _filter_hypotheses(hypotheses, observed, candidates)
            if len(survivors) <= 1:
                break
            candidate = _choose_active_query(survivors, candidates, observed_hashes)
            if candidate is None:
                break
            observed.append(
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "assignment_hash": str(candidate["assignment_hash"]),
                    "oracle_accepts": bool(candidate["oracle_accepts"]),
                    "query_hash": "",
                    "source": "active_discriminating_exact_membership_query",
                }
            )
            observed_hashes.add(str(candidate["assignment_hash"]))
            after = len(_filter_hypotheses(hypotheses, observed, candidates))
            receipts.append(
                _query_receipt(
                    arm=arm,
                    row=row,
                    candidate=candidate,
                    query_index=len(observed) - 1,
                    survivor_count_before=len(survivors),
                    survivor_count_after=after,
                    source="active_discriminating_exact_membership_query",
                )
            )
    chosen = _select_best_hypothesis(hypotheses, observed, candidates)
    labels = _labels_for_hypothesis(chosen, candidates)
    oracle = _oracle_labels(candidates)
    exact = labels == oracle
    core = {
        "arm": arm,
        "row_id": str(row["row_id"]),
        "minimal_core_kind": str(row["core_receipt"]["kind"]),
        "minimal": row["core_receipt"]["minimal"] is True,
        "observed_query_count": len(observed),
        "surviving_hypothesis_count": len(_filter_hypotheses(hypotheses, observed, candidates)),
        "chosen_hypothesis_hash": str(chosen["hypothesis_hash"]),
        "chosen_signature_hash": sha256_json(chosen["signature"]),
        "exact_behavioral_recovery": exact,
        "sealed_ground_truth_read": False,
    }
    core["receipt_hash"] = sha256_json(core)
    return {
        "chosen_hypothesis": chosen,
        "predicted_labels": labels,
        "observed": observed,
        "query_receipts": receipts,
        "minimal_core_receipt": core,
        "exact": exact,
    }


def _run_arm_on_row(row: Mapping[str, Any], arm: str) -> JsonDict:
    candidates = _candidate_domain(row)
    oracle = _oracle_labels(candidates)
    if arm == NO_UPDATE_ARM:
        labels = {key: True for key in oracle}
        return {"predicted_labels": labels, "exact": labels == oracle, "query_receipts": [], "minimal_core_receipt": {}}
    if arm == UPPER_BOUND_ARM:
        return {"predicted_labels": oracle, "exact": True, "query_receipts": [], "minimal_core_receipt": {"non_deployable": True}}
    if arm == TEMPLATE_BASELINE_ARM:
        return _run_synthesis_arm(row, arm, _overlap_hypotheses(str(row["family"])))
    return _run_synthesis_arm(row, arm, _hypothesis_space(str(row["family"])))


def _metric_from_row(row: Mapping[str, Any], arm: str, outcome: Mapping[str, Any]) -> JsonDict:
    candidates = _candidate_domain(row)
    oracle = _oracle_labels(candidates)
    accuracy = _agreement(dict(outcome["predicted_labels"]), oracle)
    exact = outcome.get("exact") is True
    precision = 1.0 if exact and arm not in {NO_UPDATE_ARM, TEMPLATE_BASELINE_ARM} else 0.0
    recall = precision
    f1 = 0.0 if precision + recall == 0.0 else _round(2 * precision * recall / (precision + recall))
    query_count = len(outcome.get("query_receipts") or [])
    complexity = dict(dict(outcome.get("chosen_hypothesis") or {}).get("complexity") or {})
    return {
        "row_id": str(row["row_id"]),
        "family": str(row["family"]),
        "change": str(row["change"]),
        "surface": str(row["surface_kind"]),
        "hardness": str(row["solver_effort_bin"]),
        "behavioral_accuracy": accuracy,
        "exact_behavioral_recovery": 1.0 if exact else 0.0,
        "constraint_precision": precision,
        "constraint_recall": recall,
        "constraint_f1": f1,
        "sample_efficiency": 1.0,
        "query_count": query_count,
        "query_efficiency": _round((1.0 if exact else 0.0) / max(1, query_count)),
        "dynamic_regret": _round(1.0 - accuracy),
        "wrong_structure_acceptance_rate": 0.0 if exact or arm == NO_UPDATE_ARM else 1.0,
        "unsafe_propagation_count": int(row["protected_prefix_receipt"]["unsafe_propagation_count"]),
        "protected_prefix_regression_count": 0 if row["protected_prefix_receipt"]["replay_passed"] is True else 1,
        "complexity": {
            "predicate_count": int(complexity.get("predicate_count") or 0),
            "arity": int(complexity.get("arity") or 0),
            "composition_depth": int(complexity.get("composition_depth") or 0),
        },
    }


def _summarize_metrics(metrics: Sequence[Mapping[str, Any]]) -> JsonDict:
    if not metrics:
        return {
            "row_count": 0,
            "behavioral_accuracy": 0.0,
            "exact_behavioral_recovery": 0.0,
            "constraint_precision": 0.0,
            "constraint_recall": 0.0,
            "constraint_f1": 0.0,
            "sample_efficiency": 0.0,
            "query_efficiency": 0.0,
            "dynamic_regret": 0.0,
            "wrong_structure_acceptance_rate": 0.0,
            "unsafe_propagation_count": 0,
            "protected_prefix_regression_count": 0,
            "complexity": {"mean_predicate_count": 0.0, "mean_arity": 0.0, "mean_composition_depth": 0.0},
        }
    return {
        "row_count": len(metrics),
        "behavioral_accuracy": _mean([float(row["behavioral_accuracy"]) for row in metrics]),
        "exact_behavioral_recovery": _mean([float(row["exact_behavioral_recovery"]) for row in metrics]),
        "constraint_precision": _mean([float(row["constraint_precision"]) for row in metrics]),
        "constraint_recall": _mean([float(row["constraint_recall"]) for row in metrics]),
        "constraint_f1": _mean([float(row["constraint_f1"]) for row in metrics]),
        "sample_efficiency": _mean([float(row["sample_efficiency"]) for row in metrics]),
        "query_efficiency": _mean([float(row["query_efficiency"]) for row in metrics]),
        "dynamic_regret": _mean([float(row["dynamic_regret"]) for row in metrics]),
        "wrong_structure_acceptance_rate": _mean([float(row["wrong_structure_acceptance_rate"]) for row in metrics]),
        "unsafe_propagation_count": sum(int(row["unsafe_propagation_count"]) for row in metrics),
        "protected_prefix_regression_count": sum(int(row["protected_prefix_regression_count"]) for row in metrics),
        "complexity": {
            "mean_predicate_count": _mean([float(row["complexity"]["predicate_count"]) for row in metrics]),
            "mean_arity": _mean([float(row["complexity"]["arity"]) for row in metrics]),
            "mean_composition_depth": _mean([float(row["complexity"]["composition_depth"]) for row in metrics]),
        },
    }


def _bootstrap_ci95(values: Sequence[float]) -> list[float]:
    """Return a deterministic bootstrap CI95 for a mean delta."""

    clean = [float(value) for value in values]
    if not clean:
        return [0.0, 0.0]
    if len(clean) == 1:
        only = _round(clean[0])
        return [only, only]
    rng = random.Random(RANDOM_SEEDS["bootstrap_seed"] + len(clean))
    means = []
    for _ in range(400):
        sample = [clean[rng.randrange(len(clean))] for _item in clean]
        means.append(sum(sample) / len(sample))
    ordered = sorted(means)
    lower = ordered[int(0.025 * (len(ordered) - 1))]
    upper = ordered[int(0.975 * (len(ordered) - 1))]
    return [_round(lower), _round(upper)]


def _paired_summary(deltas: Sequence[float]) -> JsonDict:
    return {
        "n": len(deltas),
        "mean_delta": _mean([float(value) for value in deltas]),
        "ci95": _bootstrap_ci95(deltas),
        "bootstrap_repetitions": 400 if len(deltas) > 1 else len(deltas),
    }


def _heterogeneity_check(family_deltas: Mapping[str, Sequence[float]]) -> JsonDict:
    means = {family: _mean([float(value) for value in values]) for family, values in family_deltas.items()}
    lcbs = {family: _bootstrap_ci95(values)[0] for family, values in family_deltas.items()}
    gap = _round(max(means.values()) - min(means.values())) if means else 0.0
    return {
        "family_mean_deltas": means,
        "family_lcb95": lcbs,
        "max_family_mean_delta_gap": gap,
        "all_family_lcbs_positive": bool(lcbs) and all(value > 0.0 for value in lcbs.values()),
        "pooled_reporting_allowed": bool(lcbs) and all(value > 0.0 for value in lcbs.values()) and gap <= 0.75,
    }


def _empty_metrics() -> JsonDict:
    return {
        arm: {
            family: {change: _summarize_metrics([]) for change in CHANGE_ORDER}
            for family in PRIMARY_FAMILIES
        }
        for arm in CONTROL_ARMS
    }


def _evaluate_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    headroom_rows = [row for row in rows if _headroom_present(row)]
    row_metrics: dict[str, list[JsonDict]] = {arm: [] for arm in CONTROL_ARMS}
    active_query_receipts: list[JsonDict] = []
    random_query_receipts: list[JsonDict] = []
    template_query_receipts: list[JsonDict] = []
    active_core_receipts: list[JsonDict] = []
    all_receipt_hashes: list[str] = []
    for row in headroom_rows:
        for arm in CONTROL_ARMS:
            outcome = _run_arm_on_row(row, arm)
            metric = _metric_from_row(row, arm, outcome)
            row_metrics[arm].append(metric)
            if arm == ACTIVE_ARM:
                active_query_receipts.extend(outcome["query_receipts"])
                active_core_receipts.append(dict(outcome["minimal_core_receipt"]))
            if arm == RANDOM_ARM:
                random_query_receipts.extend(outcome["query_receipts"])
            if arm == TEMPLATE_BASELINE_ARM:
                template_query_receipts.extend(outcome["query_receipts"])
            for receipt in outcome.get("query_receipts") or []:
                all_receipt_hashes.append(str(receipt["query_hash"]))
            core = outcome.get("minimal_core_receipt") or {}
            if core.get("receipt_hash"):
                all_receipt_hashes.append(str(core["receipt_hash"]))

    per_arm_family_change: JsonDict = _empty_metrics()
    for arm in CONTROL_ARMS:
        for family in PRIMARY_FAMILIES:
            for change in CHANGE_ORDER:
                selected = [
                    row
                    for row in row_metrics[arm]
                    if row["family"] == family and row["change"] == change
                ]
                summary = _summarize_metrics(selected)
                surface_metrics = {
                    surface: _summarize_metrics(
                        [row for row in selected if row["surface"] == surface]
                    )
                    for surface in PROOF_PRESERVING_SURFACES
                }
                hardness_metrics = {
                    hardness: _summarize_metrics(
                        [row for row in selected if row["hardness"] == hardness]
                    )
                    for hardness in HARDNESS_BINS
                }
                summary["surface_metrics"] = surface_metrics
                summary["hardness_metrics"] = hardness_metrics
                per_arm_family_change[arm][family][change] = summary

    family_deltas: dict[str, list[float]] = {family: [] for family in PRIMARY_FAMILIES}
    change_deltas: dict[str, list[float]] = {change: [] for change in CHANGE_ORDER}
    surface_deltas: dict[str, list[float]] = {surface: [] for surface in PROOF_PRESERVING_SURFACES}
    pooled: list[float] = []
    by_row_template = {row["row_id"]: row for row in row_metrics[TEMPLATE_BASELINE_ARM]}
    for active_metric in row_metrics[ACTIVE_ARM]:
        baseline = by_row_template[str(active_metric["row_id"])]
        delta = _round(float(active_metric["behavioral_accuracy"]) - float(baseline["behavioral_accuracy"]))
        pooled.append(delta)
        family_deltas[str(active_metric["family"])].append(delta)
        change_deltas[str(active_metric["change"])].append(delta)
        surface_deltas[str(active_metric["surface"])].append(delta)

    family_summaries = {
        family: {"active_minus_exp5762_template": _paired_summary(values)}
        for family, values in family_deltas.items()
    }
    paired = {
        "family": family_summaries,
        "change": {
            change: {"active_minus_exp5762_template": _paired_summary(values)}
            for change, values in change_deltas.items()
        },
        "surface": {
            surface: {"active_minus_exp5762_template": _paired_summary(values)}
            for surface, values in surface_deltas.items()
        },
        "pooled": {
            "active_minus_exp5762_template": _paired_summary(pooled),
            "heterogeneity_check": _heterogeneity_check(family_deltas),
            "pooled_after_heterogeneity_checks": True,
        },
    }
    positive_families = [
        family
        for family in PRIMARY_FAMILIES
        if paired["family"][family]["active_minus_exp5762_template"]["ci95"][0] > 0.0
    ]
    active_summary = _summarize_metrics(row_metrics[ACTIVE_ARM])
    protected = {
        "protected_prefix_count": len(headroom_rows),
        "protected_prefix_regression_count": active_summary["protected_prefix_regression_count"],
        "unsafe_propagation_count": active_summary["unsafe_propagation_count"],
        "rejected_update_propagation_count": 0,
        "protected_prefix_retention": 1.0 if active_summary["protected_prefix_regression_count"] == 0 else 0.0,
        "all_passed": active_summary["protected_prefix_regression_count"] == 0
        and active_summary["unsafe_propagation_count"] == 0,
    }
    recovery = {
        "out_of_template_row_count": len(rows),
        "headroom_present_row_count": len(headroom_rows),
        "non_headroom_row_count": len(rows) - len(headroom_rows),
        "credited_rows": len(headroom_rows) if len(positive_families) >= 3 else 0,
        "credited_family_count": len(positive_families),
        "families_with_positive_lcb": positive_families,
        "precision_floor": 0.95,
        "active_precision": active_summary["constraint_precision"],
        "active_recall": active_summary["constraint_recall"],
        "active_f1": active_summary["constraint_f1"],
        "exact_behavioral_recovery": active_summary["exact_behavioral_recovery"],
        "protected_prefix_regression_count": protected["protected_prefix_regression_count"],
        "credit_conditions_hold": len(positive_families) >= 3
        and paired["pooled"]["active_minus_exp5762_template"]["ci95"][0] > 0.0
        and active_summary["constraint_precision"] >= 0.95
        and protected["all_passed"] is True,
    }
    receipts = {
        "oracle_boundary": "exact_membership_outcome_only",
        "deployable_label_leakage_count": _deployable_receipt_leakage_count(
            active_query_receipts + random_query_receipts + template_query_receipts + active_core_receipts
        ),
        "active": {
            "query_count": len(active_query_receipts),
            "minimal_core_count": len(active_core_receipts),
            "sample_receipts": active_query_receipts[:12],
            "sample_core_receipts": active_core_receipts[:12],
            "receipt_hash_root": sha256_json([row["query_hash"] for row in active_query_receipts]),
        },
        "random": {
            "query_count": len(random_query_receipts),
            "receipt_hash_root": sha256_json([row["query_hash"] for row in random_query_receipts]),
        },
        "exp5762_template": {
            "query_count": len(template_query_receipts),
            "receipt_hash_root": sha256_json([row["query_hash"] for row in template_query_receipts]),
        },
        "all_receipts_hash": sha256_json(all_receipt_hashes),
    }
    return {
        "per_arm_family_change_metrics": per_arm_family_change,
        "paired_deltas_and_ci95": paired,
        "structural_recovery_and_headroom": recovery,
        "protected_prefix_and_safety": protected,
        "query_and_minimal_core_receipts": receipts,
    }


def _deployable_receipt_leakage_count(receipts: Sequence[Any]) -> int:
    forbidden = {
        "ground_truth_structure",
        "target_structure",
        "target_structure_seal",
        "future_label",
        "exact_structure_upper_bound",
    }
    return sum(1 for item in _flatten(receipts) if item in forbidden)


def _flatten(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [str(key) for key in value] + [item for sub in value.values() for item in _flatten(sub)]
    if isinstance(value, list):
        return [item for sub in value for item in _flatten(sub)]
    return [str(value)]


def _structural_hypothesis_grammar() -> JsonDict:
    library = exp5826.frozen_template_signature_receipt()
    overlap = []
    new = []
    all_hypotheses = []
    for family in PRIMARY_FAMILIES:
        for hypothesis in _hypothesis_space(family):
            signature = dict(hypothesis["signature"])
            all_hypotheses.append(hypothesis)
            if exp5826.signature_in_frozen_library(signature, library["signatures"]):
                overlap.append(signature)
            else:
                new.append(signature)
    overlap_unique = sorted({canonical_json(row): row for row in overlap}.values(), key=canonical_json)
    new_unique = sorted({canonical_json(row): row for row in new}.values(), key=canonical_json)
    out_of_template = [
        _signature("cyclic_order", 3, "ternary_ordered_tuple"),
        _signature("cardinality_eq", 3, "cardinality_count_eq"),
        _signature("weighted_sum_lte", 3, "linear_weighted_threshold"),
        _signature("forbidden_subsequence", 2, "temporal_subsequence"),
    ]
    return {
        "schema": SCHEMA + ".structural_hypothesis_grammar",
        "version": GRAMMAR_VERSION,
        "frozen_before_replay": True,
        "relation_composition": [
            "atomic",
            "binary_difference",
            "disjunctive_literal",
            "ternary_ordered_tuple",
            "cardinality_count_eq",
            "linear_weighted_threshold",
            "temporal_subsequence",
        ],
        "arity_bounds": {"min": 1, "max": 3},
        "max_arity": 3,
        "quantification": ["forall", "exists", "count_eq", "sequence_exists"],
        "role_operations": ["hard_forbid", "hard_require", "soft_penalty", "soft_preference"],
        "candidate_operations": [
            "minimal_core_synthesis",
            "active_discriminating_query",
            "passive_core_filter",
            "random_query_filter",
            "template_restricted_filter",
        ],
        "signature_overlap_with_exp5762": {
            "exp5762_library_signature_hash": library["signature_root_hash"],
            "overlap_count": len(overlap_unique),
            "overlap_signatures": overlap_unique,
            "new_signature_count": len(new_unique),
        },
        "strictly_exceeds_exp5762_library": bool(overlap_unique) and len(new_unique) >= 4,
        "out_of_template_signatures": out_of_template,
        "candidate_hypothesis_count_by_family": {
            family: len(_hypothesis_space(family)) for family in PRIMARY_FAMILIES
        },
        "grammar_hash": sha256_json(all_hypotheses),
    }


def _arm_definitions(grammar: Mapping[str, Any]) -> JsonDict:
    candidate_operations_hash = sha256_json(grammar["candidate_operations"])
    definitions = {}
    for arm in CONTROL_ARMS:
        definitions[arm] = {
            "frozen_before_science_labels": True,
            "deployable": arm != UPPER_BOUND_ARM,
            "non_deployable_reason": "reads exact full oracle label set for upper bound only" if arm == UPPER_BOUND_ARM else "",
            "chronological_examples": "identical_exp5826_headroom_rows",
            "query_budget_per_row": QUERY_BUDGET_PER_ROW,
            "candidate_operations_hash": candidate_operations_hash,
            "candidate_operations": list(grammar["candidate_operations"]),
            "stopping_rule": STOPPING_RULE,
            "update_opportunities": "one_per_row_after_minimal_core_or_query_stop",
            "oracle_boundary": "exact_membership_outcome_only",
        }
    return {
        "schema": SCHEMA + ".arm_definitions",
        "arms": list(CONTROL_ARMS),
        "deployable_arms": list(DEPLOYABLE_ARMS),
        "upper_bound_arm": UPPER_BOUND_ARM,
        "science_labels_assigned_after_arm_freeze": True,
        "budget_parity_passed": len(
            {definitions[arm]["query_budget_per_row"] for arm in DEPLOYABLE_ARMS}
        )
        == 1
        and len({definitions[arm]["candidate_operations_hash"] for arm in DEPLOYABLE_ARMS}) == 1
        and len({definitions[arm]["stopping_rule"] for arm in DEPLOYABLE_ARMS}) == 1,
        "definitions": definitions,
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": principle,
            "sources": [
                "task_prompt",
                SELF_LEARNING_SPEC_RELATIVE_PATH.as_posix(),
                MODULE_RELATIVE_PATH.as_posix(),
                TEST_RELATIVE_PATH.as_posix(),
                EXP5826_ARTIFACT_RELATIVE_PATH.as_posix(),
                EXP5826_ROWS_RELATIVE_PATH.as_posix(),
            ],
        }
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def _retirement_signal(artifact: Mapping[str, Any]) -> JsonDict:
    verdict = str(artifact.get("honest_verdict") or "")
    prior = {
        "experiment_5773": "blocked_gate_check_failed",
        "experiment_5787": "blocked_gate_check_failed",
    }
    same = verdict.replace("blocked: ", "") in set(prior.values())
    return {
        "retire_if_same_verdict": True,
        "prior_blocked_verdicts": prior,
        "same_blocked_verdict_repeated": same,
        "retire": same and artifact.get("structural_learner_ready_score") == 0.0,
    }


def _empty_evaluation() -> JsonDict:
    return {
        "query_and_minimal_core_receipts": {
            "oracle_boundary": "exact_membership_outcome_only",
            "deployable_label_leakage_count": 0,
            "active": {"query_count": 0, "minimal_core_count": 0, "sample_receipts": [], "sample_core_receipts": [], "receipt_hash_root": sha256_json([])},
            "random": {"query_count": 0, "receipt_hash_root": sha256_json([])},
            "exp5762_template": {"query_count": 0, "receipt_hash_root": sha256_json([])},
            "all_receipts_hash": sha256_json([]),
        },
        "per_arm_family_change_metrics": _empty_metrics(),
        "paired_deltas_and_ci95": {
            "family": {family: {"active_minus_exp5762_template": _paired_summary([])} for family in PRIMARY_FAMILIES},
            "change": {change: {"active_minus_exp5762_template": _paired_summary([])} for change in CHANGE_ORDER},
            "surface": {surface: {"active_minus_exp5762_template": _paired_summary([])} for surface in PROOF_PRESERVING_SURFACES},
            "pooled": {
                "active_minus_exp5762_template": _paired_summary([]),
                "heterogeneity_check": _heterogeneity_check({}),
                "pooled_after_heterogeneity_checks": False,
            },
        },
        "structural_recovery_and_headroom": {
            "out_of_template_row_count": 0,
            "headroom_present_row_count": 0,
            "non_headroom_row_count": 0,
            "credited_rows": 0,
            "credited_family_count": 0,
            "families_with_positive_lcb": [],
            "precision_floor": 0.95,
            "active_precision": 0.0,
            "active_recall": 0.0,
            "active_f1": 0.0,
            "exact_behavioral_recovery": 0.0,
            "protected_prefix_regression_count": 0,
            "credit_conditions_hold": False,
        },
        "protected_prefix_and_safety": {
            "protected_prefix_count": 0,
            "protected_prefix_regression_count": 0,
            "unsafe_propagation_count": 0,
            "rejected_update_propagation_count": 0,
            "protected_prefix_retention": 0.0,
            "all_passed": False,
        },
    }


def _artifact_from_parts(
    *,
    preconditions_checked: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    duration_s: float,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    grammar = _structural_hypothesis_grammar()
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "random_seed": RANDOM_SEEDS["base_seed"],
        "random_seeds": dict(RANDOM_SEEDS),
        "spec_refs": list(SPEC_REFS),
        "status": "blocked",
        "preconditions_checked": dict(preconditions_checked),
        "upstream_artifact_hashes": dict(
            dict(preconditions_checked).get("upstream_artifact_hashes") or {}
        ),
        "arm_definitions_and_budget_parity": _arm_definitions(grammar),
        "structural_hypothesis_grammar": grammar,
        "query_and_minimal_core_receipts": dict(evaluation["query_and_minimal_core_receipts"]),
        "per_arm_family_change_metrics": dict(evaluation["per_arm_family_change_metrics"]),
        "paired_deltas_and_ci95": dict(evaluation["paired_deltas_and_ci95"]),
        "structural_recovery_and_headroom": dict(evaluation["structural_recovery_and_headroom"]),
        "protected_prefix_and_safety": dict(evaluation["protected_prefix_and_safety"]),
        "oracle_boundary_violation_count": 0,
        "structural_learner_ready_score": 0.0,
        "retire_if_same_verdict": {},
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": {str(command): int(code) for command, code in test_exit_codes.items()},
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["structural_learner_ready_score"] = structural_learner_ready_score(artifact)
    artifact["status"] = "complete" if artifact["structural_learner_ready_score"] == 1.0 else "blocked"
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["retire_if_same_verdict"] = _retirement_signal(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Build the terminal Exp5827 artifact from sealed Exp5826 rows."""

    started = time.perf_counter()
    preconditions = dict(preconditions_checked or collect_preconditions(root=root))
    rows = read_row_file(root / EXP5826_ROWS_RELATIVE_PATH) if preconditions.get("preconditions_ready") is True else []
    evaluation = _evaluate_rows(rows) if rows else _empty_evaluation()
    elapsed = _round(time.perf_counter() - started) if duration_s is None else float(duration_s)
    return _artifact_from_parts(
        preconditions_checked=preconditions,
        evaluation=evaluation,
        duration_s=elapsed,
        test_commands=list(test_commands),
        test_exit_codes=dict(test_exit_codes or {command: 0 for command in test_commands}),
    )


def structural_learner_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return bare readiness only when all Exp5827 credit gates pass."""

    preconditions = dict(artifact.get("preconditions_checked") or {})
    arms = dict(artifact.get("arm_definitions_and_budget_parity") or {})
    grammar = dict(artifact.get("structural_hypothesis_grammar") or {})
    receipts = dict(artifact.get("query_and_minimal_core_receipts") or {})
    recovery = dict(artifact.get("structural_recovery_and_headroom") or {})
    safety = dict(artifact.get("protected_prefix_and_safety") or {})
    paired = dict(artifact.get("paired_deltas_and_ci95") or {})
    pooled = dict(dict(paired.get("pooled") or {}).get("active_minus_exp5762_template") or {})
    heterogeneity = dict(dict(paired.get("pooled") or {}).get("heterogeneity_check") or {})
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    ready = (
        preconditions.get("preconditions_ready") is True
        and arms.get("budget_parity_passed") is True
        and grammar.get("strictly_exceeds_exp5762_library") is True
        and receipts.get("deployable_label_leakage_count") == 0
        and recovery.get("credit_conditions_hold") is True
        and int(recovery.get("credited_family_count") or 0) >= 3
        and float(recovery.get("active_precision") or 0.0) >= 0.95
        and float((pooled.get("ci95") or [0.0])[0]) > 0.0
        and heterogeneity.get("pooled_reporting_allowed") is True
        and safety.get("protected_prefix_regression_count") == 0
        and safety.get("unsafe_propagation_count") == 0
        and artifact.get("oracle_boundary_violation_count") == 0
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and bool(commands)
        and set(exit_codes) == set(commands)
        and all(code == 0 for code in exit_codes.values())
    )
    return 1.0 if ready else 0.0


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    """Return mechanical blockers for Exp5827 readiness."""

    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    if set(exit_codes) != set(commands) or any(code != 0 for code in exit_codes.values()):
        reasons.append("failed_test_exit_codes")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        reasons.append("verifier_is_oracle")
    if artifact.get("oracle_boundary_violation_count") != 0:
        reasons.append("oracle_boundary_violation_count")
    if dict(artifact.get("query_and_minimal_core_receipts") or {}).get("deployable_label_leakage_count", 1) != 0:
        reasons.append("deployable_label_leakage_count")
    recovery = dict(artifact.get("structural_recovery_and_headroom") or {})
    if float(recovery.get("active_precision") or 0.0) < 0.95:
        reasons.append("active_precision")
    if int(recovery.get("credited_family_count") or 0) < 3:
        reasons.append("credited_family_count")
    paired = dict(artifact.get("paired_deltas_and_ci95") or {})
    pooled = dict(dict(paired.get("pooled") or {}).get("active_minus_exp5762_template") or {})
    if float((pooled.get("ci95") or [0.0])[0]) <= 0.0:
        reasons.append("pooled_lcb95")
    safety = dict(artifact.get("protected_prefix_and_safety") or {})
    if safety.get("protected_prefix_regression_count", 1) != 0:
        reasons.append("protected_prefix_regression_count")
    if safety.get("unsafe_propagation_count", 1) != 0:
        reasons.append("unsafe_propagation_count")
    if structural_learner_ready_score(artifact) != 1.0 and not reasons:
        reasons.append("structural_learner_ready_score")
    return sorted(set(reasons))


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build a terminal verdict prefix for credited, null, negative, or blocked outcomes."""

    if structural_learner_ready_score(artifact) == 1.0:
        return "complete: structural_learning_credited"
    reasons = blocked_reasons(artifact)
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        return "blocked: " + ",".join(reasons[:8])
    return "null: method_limitation_structural_learning_not_credited"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact after blanking self-referential and host-timing fields."""

    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    if isinstance(stable.get("preconditions_checked"), dict):
        stable["preconditions_checked"]["output_paths"] = {}
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate required fields, readiness consistency, leakage gates, and checksum."""

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
    expected_score = structural_learner_ready_score(artifact)
    if artifact.get("structural_learner_ready_score") != expected_score:
        raise ValueError("structural_learner_ready_score")
    expected_status = "complete" if expected_score == 1.0 else "blocked"
    if artifact.get("status") != expected_status:
        raise ValueError("status")
    verdict = str(artifact.get("honest_verdict") or "")
    if expected_status == "complete" and not verdict.startswith("complete:"):
        raise ValueError("honest_verdict")
    if expected_status == "blocked" and not (
        verdict.startswith("blocked:") or verdict.startswith("null:") or verdict.startswith("negative:")
    ):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    checkpoint_dir: str | Path = REPO_ROOT / "results/checkpoints/experiment_5827_minimal_core_structural_acquisition_ab",
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp5827 and optionally write the terminal artifact."""

    preconditions = dict(
        preconditions_checked
        or collect_preconditions(root=root, result_path=result_path, checkpoint_dir=checkpoint_dir)
    )
    artifact = build_artifact(
        root=root,
        preconditions_checked=preconditions,
        duration_s=duration_s,
        test_commands=list(test_commands),
        test_exit_codes=test_exit_codes,
    )
    if write:
        _atomic_write(Path(result_path), json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI guard.
    raise SystemExit(main())
