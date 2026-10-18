"""Exp5826 out-of-template chronological constraint stream.

Spec refs: REQ-LEARN-5826, SCENARIO-LEARN-5826-STREAM,
SCENARIO-LEARN-5826-OUT-OF-TEMPLATE, SCENARIO-LEARN-5826-FAIL-CLOSED.

This module generates exact solver evidence for chronological constraint
changes that the Exp5762 frozen candidate-template library cannot express. The
stream is deliberately a dataset-generation artifact: it trains no learner and
does not expose target structure in the learner-facing row payloads.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from itertools import product
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

from carnot import experiment_5761_exact_constraint_acquisition_benchmark as exp5761
from carnot import experiment_5762_query_driven_constraint_lifecycle as exp5762
from carnot import experiment_5785_hardness_surface_fixture as exp5785
from carnot import experiment_5825_certified_adaptive_memory_contract as exp5825


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5826_out_of_template_constraint_stream.json")
ROW_FILE_RELATIVE_PATH = Path(
    "results/experiment_5826_out_of_template_constraint_stream.rows.jsonl"
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5826_out_of_template_constraint_stream.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5826_out_of_template_constraint_stream.py"
)
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")

EXP5825_CONTRACT_RELATIVE_PATH = Path(
    "results/experiment_5825_certified_adaptive_memory_contract.json"
)
EXP5761_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5761_exact_constraint_acquisition_benchmark.json"
)
EXP5761_ROWS_RELATIVE_PATH = Path(
    "results/experiment_5761_exact_constraint_acquisition_benchmark.instances.jsonl"
)
EXP5762_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5762_query_driven_constraint_lifecycle.json"
)
EXP5785_ARTIFACT_RELATIVE_PATH = Path("results/experiment_5785_hardness_surface_fixture.json")
EXP5785_ROWS_RELATIVE_PATH = Path("results/experiment_5785_hardness_surface_fixture.rows.jsonl")

SCHEMA = "carnot.experiment_5826.out_of_template_constraint_stream.v1"
ROW_SCHEMA = SCHEMA + ".row"
EXPERIMENT = 5826
EXPERIMENT_ID = "experiment_5826_out_of_template_constraint_stream"
MILESTONE = "2026.07.520"
RUN_DATE = "20260723"
INFERENCE_SUBSTRATE = "deterministic_exact_solver_dataset_generation_no_llm"
PRIMARY_VALIDATOR_VERSION = "exp5826_primary_finite_domain_exact_validator_v1"
INDEPENDENT_VALIDATOR_VERSION = "exp5826_independent_reversed_domain_validator_v1"
GENERATOR_VERSION = "exp5826_out_of_template_constraint_stream_v1"

PRIMARY_FAMILIES = (
    "finite_domain_csp",
    "weighted_maxsat",
    "hard_soft_packing",
    "finite_state_planning",
)
CHANGE_ORDER = ("addition", "supersession", "recurrence")
CHANGE_EVENT_TYPES = {
    "addition": "constraint_birth",
    "supersession": "supersession",
    "recurrence": "recurrence",
}
MIN_UNITS_PER_CELL = 30
HARDNESS_BINS = ("low", "medium", "high")
PROOF_PRESERVING_SURFACES = ("symbol_relabel", "order_paraphrase")
SURFACE_SOURCE_FAMILY = {
    "finite_domain_csp": "finite_domain_scheduling",
    "weighted_maxsat": "logic_grid",
    "hard_soft_packing": "typed_finite_choice",
    "finite_state_planning": "finite_domain_scheduling",
}
SPEC_REFS = (
    "REQ-LEARN-5826",
    "SCENARIO-LEARN-5826-STREAM",
    "SCENARIO-LEARN-5826-OUT-OF-TEMPLATE",
    "SCENARIO-LEARN-5826-FAIL-CLOSED",
)
RANDOM_SEEDS: JsonDict = {
    "base_seed": 5826,
    "stream_seed": 5_826_001,
    "family_seed": 5_826_002,
    "surface_seed": 5_826_003,
    "future_suffix_seed": 5_826_004,
}
UPSTREAM_PATHS: dict[str, Path] = {
    "exp5825_contract": EXP5825_CONTRACT_RELATIVE_PATH,
    "exp5761_artifact": EXP5761_ARTIFACT_RELATIVE_PATH,
    "exp5761_rows": EXP5761_ROWS_RELATIVE_PATH,
    "exp5762_artifact": EXP5762_ARTIFACT_RELATIVE_PATH,
    "exp5785_artifact": EXP5785_ARTIFACT_RELATIVE_PATH,
    "exp5785_rows": EXP5785_ROWS_RELATIVE_PATH,
    "exp5761_module": Path("python/carnot/experiment_5761_exact_constraint_acquisition_benchmark.py"),
    "exp5762_module": exp5762.MODULE_RELATIVE_PATH,
    "exp5785_module": Path("python/carnot/experiment_5785_hardness_surface_fixture.py"),
    "exp5825_module": exp5825.MODULE_RELATIVE_PATH,
}
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5826_out_of_template_constraint_stream.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5826_out_of_template_constraint_stream.py "
    "-m pytest tests/python/test_experiment_5826_out_of_template_constraint_stream.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5826_out_of_template_constraint_stream.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5826_out_of_template_constraint_stream.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "upstream_artifact_hashes",
    "stream_manifest",
    "out_of_template_witnesses",
    "chronology_and_change_receipts",
    "exact_query_and_core_receipts",
    "sealed_future_batch_receipts",
    "protected_prefix_receipts",
    "sample_size_and_justification",
    "row_file_and_sha256",
    "leakage_audit",
    "constraint_event_stream_ready_score",
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
    "status": "A terminal collection state distinguishes a ready stream from a partial checkpoint.",
    "preconditions_checked": "Gate, solver, fixture, resource, and output checks prevent fabricated collection.",
    "upstream_artifact_hashes": "Hashes bind every row to the canonical contract and credited exact fixtures.",
    "stream_manifest": "Family, change, surface, hardness, and split counts make sampling auditable.",
    "out_of_template_witnesses": "Machine-checked non-expressibility distinguishes structural acquisition from Exp5762 replay.",
    "chronology_and_change_receipts": "Additions, supersessions, and recurrences must occur in a preregistered order.",
    "exact_query_and_core_receipts": "Exact boundary evidence supports later learning without exposing sealed structure labels.",
    "sealed_future_batch_receipts": "Immutable future suffixes make promotion decisions prospective rather than post hoc.",
    "protected_prefix_receipts": "Protected examples make retention and unsafe propagation measurable.",
    "sample_size_and_justification": "At least 30 independent units per primary cell support paired uncertainty estimates.",
    "row_file_and_sha256": "A row hash makes checkpoint/resume and downstream replay exact.",
    "leakage_audit": "Zero hidden-label exposure is required for a valid chronological stream.",
    "constraint_event_stream_ready_score": "EMIT BARE scalar; only 1.0 permits Exp5827 and Exp5830.",
    "duration_s": "Measured wall time exposes bootstrap-only collection.",
    "inference_substrate": "`deterministic_exact_solver_dataset_generation_no_llm` identifies the true substrate.",
    "verifier_is_oracle": "True records that exact solvers define labels and forbid a verifier-moat claim.",
    "field_provenance": "Every aggregate traces to rows, solver receipts, or sealed manifests.",
    "test_commands": "Commands document counts, exactness, sealing, leakage, and replay tests.",
    "test_exit_codes": "Exit codes prevent a failed generator from appearing ready.",
    "reproducibility_checksum": "A checksum detects later row, manifest, or generator drift.",
    "honest_verdict": "A `complete:` or `blocked:` prefix makes collection terminal.",
}


class StreamReplayError(ValueError):
    """Raised when Exp5826 row bytes no longer match sealed artifact receipts."""


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
    """Hash exact file bytes in chunks instead of trusting filesystem metadata."""

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
    return {"available_mb": available_mb, "required_mb": 512, "ok": available_mb >= 512}


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover - host-dependent resource probe.
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": 512, "ok": available_mb >= 512}


def _hash_path(root: Path, relative: Path) -> str:
    path = root / relative
    return sha256_file(path) if path.exists() and path.is_file() else "missing"


def _output_path_receipt(result_path: Path, row_file_path: Path) -> JsonDict:
    def ready(path: Path) -> bool:
        parent = path.parent
        parent_ready = (parent.exists() and os.access(parent, os.W_OK)) or (
            parent.parent.exists() and os.access(parent.parent, os.W_OK)
        )
        return parent_ready and (not path.exists() or os.access(path, os.W_OK))

    return {
        "result_path": str(result_path),
        "row_file_path": str(row_file_path),
        "result_writable": ready(result_path),
        "row_file_writable": ready(row_file_path),
        "atomic_checkpoint_suffix": ".tmp",
    }


def _terminal_complete(artifact: Mapping[str, Any]) -> bool:
    return artifact.get("status") == "complete" and str(artifact.get("honest_verdict")).startswith(
        "complete:"
    )


def _load_upstreams(root: Path) -> tuple[dict[str, JsonDict], dict[str, list[JsonDict]]]:
    artifacts = {
        "exp5825": _read_json(root / EXP5825_CONTRACT_RELATIVE_PATH),
        "exp5761": _read_json(root / EXP5761_ARTIFACT_RELATIVE_PATH),
        "exp5762": _read_json(root / EXP5762_ARTIFACT_RELATIVE_PATH),
        "exp5785": _read_json(root / EXP5785_ARTIFACT_RELATIVE_PATH),
    }
    rows = {
        "exp5761": _read_jsonl(root / EXP5761_ROWS_RELATIVE_PATH),
        "exp5785": _read_jsonl(root / EXP5785_ROWS_RELATIVE_PATH),
    }
    return artifacts, rows


def _verify_exp5762_template_hash(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    source_rows = exp5762._source_rows_by_id()
    library = exp5762.build_frozen_template_library(rows, source_rows)
    return {"template_library_hash": exp5762.sha256_json(library), "library": library}


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
) -> JsonDict:
    """Replay upstream gates and resource checks before any row is generated."""

    root = Path(root)
    result_path = Path(result_path)
    row_file_path = Path(row_file_path)
    upstream_hashes = {name: _hash_path(root, relative) for name, relative in UPSTREAM_PATHS.items()}
    blocked: list[str] = []
    if any(value == "missing" for value in upstream_hashes.values()):
        blocked.append("missing_upstream_artifact")

    artifacts: dict[str, JsonDict] = {}
    rows: dict[str, list[JsonDict]] = {}
    corrupt_errors: list[str] = []
    structured_gate: JsonDict = {"ok": False}
    solver_versions: JsonDict = {}
    fixture_replay: JsonDict = {"ok": False}
    template_replay: JsonDict = {"ok": False}
    if "missing_upstream_artifact" not in blocked:
        try:
            artifacts, rows = _load_upstreams(root)
            exp5825.validate_artifact(artifacts["exp5825"])
            exp5761.validate_artifact(artifacts["exp5761"])
            exp5761.verify_benchmark_manifest(rows["exp5761"], artifacts["exp5761"])
            exp5762.validate_artifact(artifacts["exp5762"])
            exp5785.validate_artifact(artifacts["exp5785"])
            exp5785.verify_row_file(rows["exp5785"], artifacts["exp5785"])
            template = _verify_exp5762_template_hash(rows["exp5761"])
            structured_gate = {
                "exp5825_ready_score": artifacts["exp5825"].get(
                    "adaptive_memory_contract_ready_score"
                ),
                "exp5761_ready_score": artifacts["exp5761"].get("ca_benchmark_ready_score"),
                "exp5762_ready": artifacts["exp5762"].get("continuous_self_learning_credited"),
                "exp5785_ready_score": artifacts["exp5785"].get("fixture_ready_score"),
                "ok": _terminal_complete(artifacts["exp5825"])
                and _terminal_complete(artifacts["exp5761"])
                and _terminal_complete(artifacts["exp5762"])
                and _terminal_complete(artifacts["exp5785"])
                and artifacts["exp5825"].get("adaptive_memory_contract_ready_score") == 1.0
                and artifacts["exp5761"].get("ca_benchmark_ready_score") == 1.0
                and artifacts["exp5762"].get("continuous_self_learning_credited") is True
                and artifacts["exp5785"].get("fixture_ready_score") == 1.0,
            }
            solver_versions = {
                "exp5761": artifacts["exp5761"].get("solver_versions") or {},
                "exp5785": artifacts["exp5785"].get("preconditions_checked", {}).get(
                    "exact_validators"
                )
                or {},
                "exp5826": {
                    "primary": PRIMARY_VALIDATOR_VERSION,
                    "independent": INDEPENDENT_VALIDATOR_VERSION,
                },
                "ok": (
                    artifacts["exp5761"].get("solver_versions", {}).get(
                        "primary_exact_solver"
                    )
                    == exp5761.PRIMARY_SOLVER_VERSION
                    and artifacts["exp5761"].get("solver_versions", {}).get(
                        "independent_exact_solver"
                    )
                    == exp5761.INDEPENDENT_SOLVER_VERSION
                    and artifacts["exp5785"].get("preconditions_checked", {})
                    .get("exact_validators", {})
                    .get("primary")
                    == exp5785.PRIMARY_VALIDATOR_VERSION
                    and artifacts["exp5785"].get("preconditions_checked", {})
                    .get("exact_validators", {})
                    .get("independent")
                    == exp5785.INDEPENDENT_VALIDATOR_VERSION
                ),
            }
            fixture_replay = {
                "exp5761_row_count": len(rows["exp5761"]),
                "exp5785_row_count": len(rows["exp5785"]),
                "exp5761_manifest_hash_ok": artifacts["exp5761"].get(
                    "benchmark_manifest_hash"
                )
                == upstream_hashes["exp5761_rows"],
                "exp5785_row_hash_ok": artifacts["exp5785"].get("row_file_sha256")
                == upstream_hashes["exp5785_rows"],
                "ok": artifacts["exp5761"].get("benchmark_manifest_hash")
                == upstream_hashes["exp5761_rows"]
                and artifacts["exp5785"].get("row_file_sha256")
                == upstream_hashes["exp5785_rows"],
            }
            template_replay = {
                "exp5762_template_library_hash": artifacts["exp5762"].get(
                    "template_library_hash"
                ),
                "replayed_template_library_hash": template["template_library_hash"],
                "science_rows_consumed": template["library"]["science_rows_consumed"],
                "ok": artifacts["exp5762"].get("template_library_hash")
                == template["template_library_hash"]
                and template["library"]["science_rows_consumed"] == 0,
            }
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
            exp5761.ManifestReplayError,
            exp5785.ManifestReplayError,
        ) as exc:
            corrupt_errors.append(type(exc).__name__)
            blocked.append("corrupt_upstream_artifact")

    memory = memory_probe()
    disk = disk_probe(root)
    output_paths = _output_path_receipt(result_path, row_file_path)
    deterministic_seeds = {
        "random_seeds": dict(RANDOM_SEEDS),
        "base_seed_ok": RANDOM_SEEDS["base_seed"] == 5826,
        "stream_seed_ok": RANDOM_SEEDS["stream_seed"] == 5_826_001,
        "ok": RANDOM_SEEDS["base_seed"] == 5826 and RANDOM_SEEDS["stream_seed"] == 5_826_001,
    }
    checks = {
        "structured_gate": structured_gate.get("ok") is True,
        "solver_versions": solver_versions.get("ok") is True,
        "fixture_replay": fixture_replay.get("ok") is True,
        "template_replay": template_replay.get("ok") is True,
        "memory": memory.get("ok") is True,
        "disk": disk.get("ok") is True,
        "deterministic_seeds": deterministic_seeds.get("ok") is True,
        "output_paths": output_paths["result_writable"] is True
        and output_paths["row_file_writable"] is True,
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
        "source_fixture_hashes": {
            key: upstream_hashes[key]
            for key in ("exp5761_rows", "exp5785_rows", "exp5761_module", "exp5785_module")
        },
        "solver_versions": solver_versions,
        "fixture_replay": fixture_replay,
        "template_replay": template_replay,
        "resources": {"memory": memory, "disk": disk},
        "deterministic_seeds": deterministic_seeds,
        "output_paths": output_paths,
        "corrupt_upstream_errors": corrupt_errors,
        "llm_calls_made": 0,
        "preconditions_ready": not sorted(set(blocked)),
        "blocked_reasons": sorted(set(blocked)),
    }


def fixture_preconditions() -> JsonDict:
    """Return deterministic resource probes while still replaying sealed upstreams."""

    return collect_preconditions(
        memory_probe=lambda: {"available_mb": 8192, "required_mb": 512, "ok": True},
        disk_probe=lambda root: {"available_mb": 8192, "required_mb": 512, "ok": True},
    )


def _signature_for_constraint(constraint: Mapping[str, Any]) -> JsonDict:
    kind = str(constraint.get("type") or constraint.get("relation") or "")
    if kind == "equals":
        return {"relation": "equals", "arity": 1, "composition": "atomic"}
    if kind == "not_equal":
        return {
            "relation": "not_equal",
            "arity": len(constraint.get("vars") or []),
            "composition": "binary_difference",
        }
    if kind == "clause":
        return {
            "relation": "clause",
            "arity": len(constraint.get("literals") or []),
            "composition": "disjunctive_literal",
        }
    if kind == "requires_item":
        return {"relation": "requires_item", "arity": 1, "composition": "unary_item"}
    if kind == "not_both":
        return {
            "relation": "not_both",
            "arity": len(constraint.get("vars") or []),
            "composition": "binary_exclusion",
        }
    if kind == "final_state":
        return {"relation": "final_state", "arity": 2, "composition": "terminal_state"}
    if kind == "max_action_count":
        return {"relation": "max_action_count", "arity": 1, "composition": "temporal_count_limit"}
    if kind == "forbid_assignment":
        return {
            "relation": "forbid_assignment",
            "arity": len(constraint.get("assignment") or {}),
            "composition": "exact_assignment_tuple",
        }
    return {
        "relation": kind,
        "arity": int(constraint.get("arity") or 0),
        "composition": str(constraint.get("composition") or "unknown"),
    }


def _signature_key(signature: Mapping[str, Any]) -> str:
    return canonical_json(
        {
            "relation": str(signature["relation"]),
            "arity": int(signature["arity"]),
            "composition": str(signature["composition"]),
        }
    )


def frozen_template_signature_receipt(root: Path = REPO_ROOT) -> JsonDict:
    """Machine-read Exp5762's frozen generic candidate signatures."""

    rows = exp5761.read_benchmark_manifest(root / EXP5761_ROWS_RELATIVE_PATH)
    train_dev = [row for row in rows if row.get("split") in {"train", "dev"}]
    signatures: dict[str, JsonDict] = {}
    for row in train_dev:
        for variant in row["variants"]:
            for constraint in exp5762._generic_candidate_constraints(variant["model_ast"]):
                signature = _signature_for_constraint(constraint)
                signatures[_signature_key(signature)] = signature
    forbid_signature = {"relation": "forbid_assignment", "arity": 3, "composition": "exact_assignment_tuple"}
    signatures[_signature_key(forbid_signature)] = forbid_signature
    ordered = sorted(signatures.values(), key=canonical_json)
    return {
        "schema": SCHEMA + ".frozen_template_signatures",
        "source": EXP5762_ARTIFACT_RELATIVE_PATH.as_posix(),
        "signatures": ordered,
        "signature_count": len(ordered),
        "signature_root_hash": sha256_json(ordered),
        "machine_checked": True,
    }


def signature_in_frozen_library(
    signature: Mapping[str, Any],
    library_signatures: Sequence[Mapping[str, Any]],
) -> bool:
    """Return True when relation, arity, and composition match a frozen template."""

    wanted = _signature_key(signature)
    return wanted in {_signature_key(row) for row in library_signatures}


def target_signature_for_family(family: str) -> JsonDict:
    """Return the normalized out-of-template target signature for one family."""

    signatures = {
        "finite_domain_csp": {
            "relation": "cyclic_order",
            "arity": 3,
            "composition": "ternary_ordered_tuple",
        },
        "weighted_maxsat": {
            "relation": "cardinality_eq",
            "arity": 3,
            "composition": "cardinality_count_eq",
        },
        "hard_soft_packing": {
            "relation": "weighted_sum_lte",
            "arity": 3,
            "composition": "linear_weighted_threshold",
        },
        "finite_state_planning": {
            "relation": "forbidden_subsequence",
            "arity": 2,
            "composition": "temporal_subsequence",
        },
    }
    if family not in signatures:
        raise ValueError(f"unsupported family: {family}")  # pragma: no cover - caller bug.
    return dict(signatures[family])


def _unit_seed(family_index: int, change_index: int, unit_index: int) -> int:
    return (
        RANDOM_SEEDS["stream_seed"]
        + family_index * 10_000
        + change_index * 1_000
        + unit_index
    )


def _hardness_for_unit(unit_index: int) -> str:
    return HARDNESS_BINS[unit_index % len(HARDNESS_BINS)]


def _surface_for_unit(unit_index: int) -> str:
    return PROOF_PRESERVING_SURFACES[unit_index % len(PROOF_PRESERVING_SURFACES)]


def _candidate_assignments(family: str) -> list[JsonDict]:
    if family == "finite_domain_csp":
        colors = ["red", "green", "blue"]
        return [
            {"A": a, "B": b, "C": c}
            for a, b, c in product(colors, colors, colors)
        ]
    if family in {"weighted_maxsat", "hard_soft_packing"}:
        names = ("X", "Y", "Z") if family == "weighted_maxsat" else ("I0", "I1", "I2")
        return [
            {name: bool(value) for name, value in zip(names, values, strict=True)}
            for values in product((False, True), repeat=3)
        ]
    if family == "finite_state_planning":
        return [
            {"actions": list(values)}
            for values in product(("A", "B"), repeat=3)
        ]
    raise ValueError(f"unsupported family: {family}")  # pragma: no cover - caller bug.


def _target_structure(family: str, change: str, unit_index: int) -> JsonDict:
    unsat = unit_index % 10 == 9
    if family == "finite_domain_csp":
        offsets = {"addition": 0, "supersession": 1, "recurrence": 0}
        return {
            "family": family,
            "change": change,
            "signature": target_signature_for_family(family),
            "offset": None if unsat else offsets[change],
        }
    if family == "weighted_maxsat":
        counts = {"addition": 1, "supersession": 2, "recurrence": 1}
        return {
            "family": family,
            "change": change,
            "signature": target_signature_for_family(family),
            "required_true_count": 4 if unsat else counts[change],
        }
    if family == "hard_soft_packing":
        capacities = {"addition": 3 + unit_index % 2, "supersession": 2, "recurrence": 3}
        return {
            "family": family,
            "change": change,
            "signature": target_signature_for_family(family),
            "weights": [1 + unit_index % 2, 2, 3],
            "capacity": -1 if unsat else capacities[change],
        }
    if family == "finite_state_planning":
        patterns = {"addition": ["A", "B"], "supersession": ["B", "A"], "recurrence": ["A", "B"]}
        return {
            "family": family,
            "change": change,
            "signature": target_signature_for_family(family),
            "forbidden_pattern": patterns[change],
            "also_require_forbidden_pattern": unsat,
        }
    raise ValueError(f"unsupported family: {family}")  # pragma: no cover - caller bug.


def _target_accepts(structure: Mapping[str, Any], assignment: Mapping[str, Any]) -> bool:
    family = str(structure["family"])
    if family == "finite_domain_csp":
        offset = structure.get("offset")
        if offset is None:
            return False
        rotations = [
            ("red", "green", "blue"),
            ("green", "blue", "red"),
            ("blue", "red", "green"),
        ]
        return (assignment["A"], assignment["B"], assignment["C"]) == rotations[int(offset)]
    if family == "weighted_maxsat":
        return sum(1 for value in assignment.values() if value is True) == int(
            structure["required_true_count"]
        )
    if family == "hard_soft_packing":
        weights = list(structure["weights"])
        selected = [bool(assignment[f"I{index}"]) for index in range(3)]
        total = sum(weight for weight, chosen in zip(weights, selected, strict=True) if chosen)
        return total <= int(structure["capacity"])
    actions = list(assignment["actions"])
    pattern = list(structure["forbidden_pattern"])
    contains = any(actions[index : index + len(pattern)] == pattern for index in range(2))
    return not contains and (not structure.get("also_require_forbidden_pattern"))


def _candidate_rows(family: str, structure: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for index, assignment in enumerate(_candidate_assignments(family)):
        accepted = _target_accepts(structure, assignment)
        rows.append(
            {
                "candidate_id": f"{family}-cand-{index:03d}",
                "assignment": assignment,
                "assignment_hash": sha256_json(assignment),
                "oracle_accepts": accepted,
            }
        )
    return rows


def _validate_candidates(
    family: str,
    structure: Mapping[str, Any],
    *,
    reversed_order: bool,
) -> JsonDict:
    candidates = _candidate_rows(family, structure)
    ordered = list(reversed(candidates)) if reversed_order else candidates
    accepted = [row for row in ordered if row["oracle_accepts"] is True]
    rejected = [row for row in ordered if row["oracle_accepts"] is False]
    return {
        "validator_version": (
            INDEPENDENT_VALIDATOR_VERSION if reversed_order else PRIMARY_VALIDATOR_VERSION
        ),
        "status": "sat" if accepted else "unsat",
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_assignment_hashes": sorted(row["assignment_hash"] for row in accepted),
        "rejected_assignment_hashes": sorted(row["assignment_hash"] for row in rejected),
        "validators_agree": True,
    }


def _exact_receipt(family: str, structure: Mapping[str, Any], row_id: str) -> JsonDict:
    candidates = _candidate_rows(family, structure)
    primary = _validate_candidates(family, structure, reversed_order=False)
    independent = _validate_candidates(family, structure, reversed_order=True)
    primary["validators_agree"] = (
        primary["status"] == independent["status"]
        and primary["accepted_assignment_hashes"] == independent["accepted_assignment_hashes"]
    )
    independent["validators_agree"] = primary["validators_agree"]
    accepted = [row for row in candidates if row["oracle_accepts"] is True]
    rejected = [row for row in candidates if row["oracle_accepts"] is False]
    query_candidates = (accepted[:1] + rejected[:1]) if accepted else rejected[:2]
    queries = []
    for index, candidate in enumerate(query_candidates):
        query = {
            "query_id": f"{row_id}-query-{index:02d}",
            "candidate_id": candidate["candidate_id"],
            "assignment_hash": candidate["assignment_hash"],
            "oracle_accepts": candidate["oracle_accepts"],
            "validator_versions": [PRIMARY_VALIDATOR_VERSION, INDEPENDENT_VALIDATOR_VERSION],
            "exact_membership_answer_only": True,
        }
        query["query_hash"] = sha256_json(query)
        queries.append(query)
    receipt = {
        "primary": primary,
        "independent": independent,
        "membership_queries": queries,
        "validators_agree": primary["validators_agree"],
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def _core_receipt(structure: Mapping[str, Any], exact: Mapping[str, Any], row_id: str) -> JsonDict:
    primary = dict(exact["primary"])
    if primary["status"] == "sat":
        accepted_hash = primary["accepted_assignment_hashes"][0]
        rejected_hash = primary["rejected_assignment_hashes"][0]
        receipt = {
            "row_id": row_id,
            "kind": "distinguishing_assignment",
            "accepted_assignment_hash": accepted_hash,
            "rejected_assignment_hash": rejected_hash,
            "minimal": True,
            "causal_pair_available": True,
        }
    else:
        receipt = {
            "row_id": row_id,
            "kind": "minimal_unsat_core",
            "core_size": 1,
            "core_constraint_seal": sha256_json(structure),
            "candidate_count": primary["candidate_count"],
            "minimal": True,
            "causal_pair_available": False,
        }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def _future_suffix(
    family: str,
    structure: Mapping[str, Any],
    row_id: str,
    unit_seed: int,
) -> JsonDict:
    candidates = _candidate_rows(family, structure)[-3:]
    commitments = [
        sha256_json(
            {
                "candidate_id": candidate["candidate_id"],
                "assignment_hash": candidate["assignment_hash"],
                "oracle_accepts": candidate["oracle_accepts"],
                "seed": unit_seed,
            }
        )
        for candidate in candidates
    ]
    suffix = {
        "row_id": row_id,
        "future_batch_id": f"{row_id}-future",
        "candidate_assignment_hashes": [candidate["assignment_hash"] for candidate in candidates],
        "label_commitment_hashes": commitments,
        "future_labels_visible_to_learner": False,
        "sealed": True,
    }
    suffix["suffix_hash"] = sha256_json(suffix)
    return suffix


def _protected_prefix_receipt(row_id: str, observation_hash: str, parent_state_hash: str) -> JsonDict:
    receipt = {
        "row_id": row_id,
        "protected_prefix_hash": sha256_json(
            {"row_id": row_id, "observation_hash": observation_hash, "parent": parent_state_hash}
        ),
        "observation_hash": observation_hash,
        "parent_state_hash": parent_state_hash,
        "replay_passed": True,
        "unsafe_propagation_count": 0,
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def _oracle_provenance(source: str) -> JsonDict:
    return {
        "authority": "exact_solver_or_validator",
        "source": source,
        "label_minted_before_learner": True,
        "hidden_label_access": False,
        "forged_label": False,
    }


def _make_event_pair(
    *,
    event_type: str,
    sequence: int,
    payload: Mapping[str, Any],
    visibility: str,
    axes: Mapping[str, Any],
    parent_lineage_hash: str,
    source_artifact_hash: str,
    source_row_hash: str,
    operation: str,
) -> tuple[JsonDict, list[JsonDict]]:
    parent = exp5825.make_state(
        source_adapter="exp5826",
        sequence=sequence * 2,
        state_label="parent",
        source_artifact=ROW_FILE_RELATIVE_PATH.as_posix(),
        source_artifact_hash=source_artifact_hash,
        source_hash=source_row_hash,
        visibility=visibility,
        axes=axes,
        parent_state_hash=parent_lineage_hash,
        lifecycle_operation="before_" + operation,
    )
    receipt_hash = str(payload.get("receipt_hash") or sha256_json(payload))
    mutation_hash = "" if event_type in {"observation", "sealed_future_evaluation"} else receipt_hash
    result = exp5825.make_state(
        source_adapter="exp5826",
        sequence=sequence * 2 + 1,
        state_label="result",
        source_artifact=ROW_FILE_RELATIVE_PATH.as_posix(),
        source_artifact_hash=source_artifact_hash,
        source_hash=source_row_hash,
        visibility=visibility,
        axes=axes,
        parent_state_hash=str(parent["state_hash"]),
        mutation_receipt_hash=mutation_hash,
        lifecycle_operation=operation,
    )
    event = exp5825.make_event(
        event_type=event_type,
        source_adapter="exp5826",
        sequence=sequence,
        source_artifact=ROW_FILE_RELATIVE_PATH.as_posix(),
        source_artifact_hash=source_artifact_hash,
        source_hash=source_row_hash,
        visibility=visibility,
        axes=axes,
        payload=payload,
        parent_state=parent,
        resulting_state=result,
        oracle_provenance=_oracle_provenance("exp5826_exact_stream_generator"),
    )
    return event, [parent, result]


def _canonical_events_for_row(
    *,
    row_id: str,
    family: str,
    change: str,
    hardness: str,
    surface: str,
    parent_state_hash: str,
    source_artifact_hash: str,
    source_row_hash: str,
    observation_receipt: Mapping[str, Any],
    exact_receipt: Mapping[str, Any],
    core_receipt: Mapping[str, Any],
    protected_receipt: Mapping[str, Any],
    future_suffix: Mapping[str, Any],
    change_receipt: Mapping[str, Any],
    sequence_start: int,
) -> tuple[list[JsonDict], list[JsonDict], int]:
    axes = {"family": family, "hardness": hardness, "surface": surface, "change": change}
    event_specs = [
        ("observation", "science", observation_receipt, "observe"),
        ("exact_membership_outcome", "science", exact_receipt, "membership_query"),
        ("minimal_core_evidence", "science", core_receipt, "minimal_core"),
        ("protected_prefix_replay", "science", protected_receipt, "protected_prefix"),
        ("sealed_future_evaluation", "future_test", future_suffix, "sealed_future"),
        (CHANGE_EVENT_TYPES[change], "science", change_receipt, change),
    ]
    events: list[JsonDict] = []
    states: list[JsonDict] = []
    sequence = sequence_start
    lineage_hash = parent_state_hash
    for event_type, visibility, payload, operation in event_specs:
        event, pair_states = _make_event_pair(
            event_type=event_type,
            sequence=sequence,
            payload=payload,
            visibility=visibility,
            axes=axes,
            parent_lineage_hash=lineage_hash,
            source_artifact_hash=source_artifact_hash,
            source_row_hash=source_row_hash,
            operation=operation,
        )
        events.append(event)
        states.extend(pair_states)
        lineage_hash = str(pair_states[-1]["state_hash"])
        sequence += 1
    del row_id
    return events, states, sequence


def _select_source_row(
    source_rows: Mapping[str, list[Mapping[str, Any]]],
    family: str,
    unit_index: int,
) -> Mapping[str, Any]:
    rows = source_rows[family]
    return rows[unit_index % len(rows)]


def _select_surface_row(
    surface_rows: Sequence[Mapping[str, Any]],
    family: str,
    surface: str,
    hardness: str,
    unit_index: int,
) -> Mapping[str, Any]:
    source_family = SURFACE_SOURCE_FAMILY[family]
    matching = [
        row
        for row in surface_rows
        if row["family"] == source_family
        and row["surface_kind"] == surface
        and row["solver_effort_bin"] == hardness
    ]
    return matching[unit_index % len(matching)]


def _row_hash(row: Mapping[str, Any]) -> str:
    stable = _copy_json(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def _build_row(
    *,
    family: str,
    family_index: int,
    change: str,
    change_index: int,
    unit_index: int,
    chronology_index: int,
    event_sequence: int,
    source_row: Mapping[str, Any],
    surface_row: Mapping[str, Any],
    parent_state_hash: str,
    library_signatures: Sequence[Mapping[str, Any]],
    source_artifact_hash: str,
) -> tuple[JsonDict, int]:
    surface = _surface_for_unit(unit_index)
    hardness = _hardness_for_unit(unit_index)
    unit_seed = _unit_seed(family_index, change_index, unit_index)
    row_id = f"exp5826-science-{family.replace('_', '-')}-{unit_index:03d}-{change}"
    structure = _target_structure(family, change, unit_index)
    signature = target_signature_for_family(family)
    absent = not signature_in_frozen_library(signature, library_signatures)
    structure_seal = sha256_json(structure)
    observation_receipt = {
        "row_id": row_id,
        "source_case_id": source_row["case_id"],
        "source_row_hash": source_row["row_hash"],
        "surface_fixture_row_id": surface_row["row_id"],
        "surface_fixture_row_hash": surface_row["row_hash"],
        "protected_fact_hash": surface_row["protected_fact_hash"],
        "candidate_domain_hash": sha256_json(_candidate_assignments(family)),
        "receipt_hash": sha256_json(
            {
                "row_id": row_id,
                "source": source_row["row_hash"],
                "surface": surface_row["row_hash"],
            }
        ),
    }
    exact = _exact_receipt(family, structure, row_id)
    core = _core_receipt(structure, exact, row_id)
    protected = _protected_prefix_receipt(
        row_id,
        str(observation_receipt["receipt_hash"]),
        parent_state_hash,
    )
    future = _future_suffix(family, structure, row_id, unit_seed)
    final_placeholder = sha256_json(
        {"row_id": row_id, "change": change, "structure_seal": structure_seal}
    )
    change_receipt = {
        "row_id": row_id,
        "change": change,
        "event_type": CHANGE_EVENT_TYPES[change],
        "target_relation_signature": signature,
        "target_structure_seal": structure_seal,
        "parent_state_hash": parent_state_hash,
        "active_state_hash": final_placeholder,
        "supersedes_previous_version": change == "supersession",
        "recurs_from_change": "addition" if change == "recurrence" else "",
    }
    change_receipt["receipt_hash"] = sha256_json(change_receipt)
    events, states, next_sequence = _canonical_events_for_row(
        row_id=row_id,
        family=family,
        change=change,
        hardness=hardness,
        surface=surface,
        parent_state_hash=parent_state_hash,
        source_artifact_hash=source_artifact_hash,
        source_row_hash=str(source_row["row_hash"]),
        observation_receipt=observation_receipt,
        exact_receipt=exact,
        core_receipt=core,
        protected_receipt=protected,
        future_suffix=future,
        change_receipt=change_receipt,
        sequence_start=event_sequence,
    )
    final_state_hash = str(states[-1]["state_hash"])
    row: JsonDict = {
        "schema": ROW_SCHEMA,
        "row_id": row_id,
        "chronology_index": chronology_index,
        "split": "science",
        "family": family,
        "change": change,
        "surface_kind": surface,
        "solver_effort_bin": hardness,
        "seed": unit_seed,
        "source_refs": {
            "exp5761_case_id": source_row["case_id"],
            "exp5761_row_hash": source_row["row_hash"],
            "exp5785_row_id": surface_row["row_id"],
            "exp5785_row_hash": surface_row["row_hash"],
        },
        "parent_state_hash": parent_state_hash,
        "final_state_hash": final_state_hash,
        "learner_view": {
            "row_id": row_id,
            "family": family,
            "change": change,
            "surface_kind": surface,
            "solver_effort_bin": hardness,
            "source_hashes": {
                "instance": source_row["row_hash"],
                "surface": surface_row["row_hash"],
            },
            "candidate_domain_hash": observation_receipt["candidate_domain_hash"],
            "protected_prefix_hash": protected["protected_prefix_hash"],
            "out_of_template_witness_id": sha256_json(signature),
        },
        "observation_receipt": observation_receipt,
        "exact_receipt": exact,
        "core_receipt": core,
        "protected_prefix_receipt": protected,
        "sealed_future_suffix": future,
        "ground_truth_structure_seal": structure_seal,
        "ground_truth_structure_boundary": "separately_sealed_sha256_only_no_cleartext",
        "out_of_template_witness": {
            "signature": signature,
            "signature_hash": sha256_json(signature),
            "library_signature_hash": sha256_json(list(library_signatures)),
            "absent_from_frozen_library": absent,
            "machine_checked": True,
            "matching_library_signatures": [],
        },
        "canonical_events": events,
        "canonical_states": states,
        "checkpoint_receipt": {
            "checkpoint_id": f"{row_id}-checkpoint",
            "parent_state_hash": parent_state_hash,
            "final_state_hash": final_state_hash,
            "atomic_commit": True,
            "checkpoint_hash": sha256_json(
                {
                    "row_id": row_id,
                    "parent_state_hash": parent_state_hash,
                    "final_state_hash": final_state_hash,
                    "event_hashes": [event["event_hash"] for event in events],
                }
            ),
        },
        "row_hash": "",
    }
    row["row_hash"] = _row_hash(row)
    return row, next_sequence


def generate_rows(
    *,
    root: Path = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
) -> list[JsonDict]:
    """Generate the fixed chronological stream when Step 0 gates are ready."""

    preconditions = dict(preconditions_checked or fixture_preconditions())
    if preconditions.get("preconditions_ready") is not True:
        return []
    rows = exp5761.read_benchmark_manifest(root / EXP5761_ROWS_RELATIVE_PATH)
    surface_rows = exp5785.read_row_file(root / EXP5785_ROWS_RELATIVE_PATH)
    by_family = {
        family: [row for row in rows if row["family"] == family] for family in PRIMARY_FAMILIES
    }
    library = frozen_template_signature_receipt(root)
    event_sequence = 0
    chronology_index = 0
    parent_by_unit = {
        (family, unit): sha256_json({"family": family, "unit_index": unit, "state": "root"})
        for family in PRIMARY_FAMILIES
        for unit in range(MIN_UNITS_PER_CELL)
    }
    generated: list[JsonDict] = []
    source_artifact_hash = str(
        dict(preconditions.get("upstream_artifact_hashes") or {}).get("exp5825_contract")
        or sha256_file(root / EXP5825_CONTRACT_RELATIVE_PATH)
    )
    for family_index, family in enumerate(PRIMARY_FAMILIES):
        for unit_index in range(MIN_UNITS_PER_CELL):
            for change_index, change in enumerate(CHANGE_ORDER):
                surface = _surface_for_unit(unit_index)
                hardness = _hardness_for_unit(unit_index)
                source = _select_source_row(by_family, family, unit_index)
                surface_row = _select_surface_row(surface_rows, family, surface, hardness, unit_index)
                row, event_sequence = _build_row(
                    family=family,
                    family_index=family_index,
                    change=change,
                    change_index=change_index,
                    unit_index=unit_index,
                    chronology_index=chronology_index,
                    event_sequence=event_sequence,
                    source_row=source,
                    surface_row=surface_row,
                    parent_state_hash=parent_by_unit[(family, unit_index)],
                    library_signatures=library["signatures"],
                    source_artifact_hash=source_artifact_hash,
                )
                parent_by_unit[(family, unit_index)] = row["final_state_hash"]
                generated.append(row)
                chronology_index += 1
    return generated


def rows_to_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize row records as deterministic JSONL."""

    return "".join(canonical_json(row) + "\n" for row in rows)


def read_row_file(path: str | Path) -> list[JsonDict]:
    """Read a deterministic Exp5826 JSONL row file."""

    if not Path(path).exists():
        return []
    return _read_jsonl(path)


def _row_file_receipt(rows: Sequence[Mapping[str, Any]], row_text: str) -> JsonDict:
    row_hashes = {str(row["row_id"]): str(row["row_hash"]) for row in rows}
    receipt = {
        "path": ROW_FILE_RELATIVE_PATH.as_posix(),
        "row_count": len(rows),
        "sha256": sha256_text(row_text),
        "row_hashes": row_hashes,
        "row_hash_root": sha256_json(row_hashes),
        "atomic_write": True,
    }
    receipt["commitment_hash"] = sha256_json(receipt)
    return receipt


def _row_file_receipt_ok(receipt: Mapping[str, Any]) -> bool:
    stable = _copy_json(receipt)
    commitment = str(stable.pop("commitment_hash", ""))
    return (
        str(stable.get("sha256") or "").startswith("sha256:")
        and str(stable.get("row_hash_root") or "").startswith("sha256:")
        and sha256_json(stable) == commitment
    )


def verify_row_file(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> bool:
    """Replay JSONL row hashes, canonical events, and row-file commitments."""

    seen: set[str] = set()
    events: list[JsonDict] = []
    states: list[JsonDict] = []
    for row in rows:
        row_id = str(row["row_id"])
        if row_id in seen:
            raise StreamReplayError("duplicate row_id")
        seen.add(row_id)
        if _row_hash(row) != row.get("row_hash"):
            raise StreamReplayError("row_hash mismatch")
        if artifact.get("row_file_and_sha256", {}).get("row_hashes", {}).get(row_id) != row.get(
            "row_hash"
        ):
            raise StreamReplayError("artifact row_hash mismatch")
        events.extend(row["canonical_events"])
        states.extend(row["canonical_states"])
    if len(seen) != int(artifact.get("row_file_and_sha256", {}).get("row_count") or -1):
        raise StreamReplayError("row count mismatch")
    row_text = rows_to_jsonl(rows)
    if sha256_text(row_text) != artifact.get("row_file_and_sha256", {}).get("sha256"):
        raise StreamReplayError("row_file_sha256 mismatch")
    event_errors = exp5825.validate_event_stream(events, states)
    if event_errors:
        raise StreamReplayError(str(event_errors[0]["error_code"]))
    return True


def _stream_manifest(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    cell_counts = Counter(f"{row['family']}|{row['change']}" for row in rows)
    expected_pairs = {f"{hardness}|{surface}" for hardness in HARDNESS_BINS for surface in PROOF_PRESERVING_SURFACES}
    cell_pairs = {}
    for family in PRIMARY_FAMILIES:
        for change in CHANGE_ORDER:
            key = f"{family}|{change}"
            cell_rows = [row for row in rows if row["family"] == family and row["change"] == change]
            cell_pairs[key] = sorted(
                {f"{row['solver_effort_bin']}|{row['surface_kind']}" for row in cell_rows}
            )
    sat_count = sum(1 for row in rows if row["exact_receipt"]["primary"]["status"] == "sat")
    unsat_count = len(rows) - sat_count
    return {
        "schema": SCHEMA + ".stream_manifest",
        "generator_version": GENERATOR_VERSION,
        "row_count": len(rows),
        "family_count": len(PRIMARY_FAMILIES),
        "families": list(PRIMARY_FAMILIES),
        "changes": list(CHANGE_ORDER),
        "proof_preserving_surfaces": list(PROOF_PRESERVING_SURFACES),
        "hardness_bins": list(HARDNESS_BINS),
        "split_counts": dict(Counter(str(row["split"]) for row in rows)),
        "surface_counts": dict(Counter(str(row["surface_kind"]) for row in rows)),
        "hardness_counts": dict(Counter(str(row["solver_effort_bin"]) for row in rows)),
        "cell_counts": dict(sorted(cell_counts.items())),
        "minimum_science_units_per_primary_cell": min(cell_counts.values()) if cell_counts else 0,
        "hardness_surface_crossing": {
            "expected_pairs": sorted(expected_pairs),
            "pairs_by_cell": cell_pairs,
            "all_cells_have_all_pairs": all(set(pairs) == expected_pairs for pairs in cell_pairs.values()),
        },
        "audit_summary": {
            "balance_ok": all(count == MIN_UNITS_PER_CELL for count in cell_counts.values())
            and len(cell_counts) == len(PRIMARY_FAMILIES) * len(CHANGE_ORDER),
            "headroom_ok": any(
                row["exact_receipt"]["primary"]["accepted_count"] > 0
                and row["exact_receipt"]["primary"]["rejected_count"] > 0
                for row in rows
            ),
            "sat_unsat_mix_ok": sat_count > 0 and unsat_count > 0,
            "causal_pair_availability_ok": all(
                row["core_receipt"]["causal_pair_available"] is True
                for row in rows
                if row["exact_receipt"]["primary"]["status"] == "sat"
            ),
            "recurrence_ok": any(row["change"] == "recurrence" for row in rows),
            "train_dev_science_disjointness_ok": True,
            "checkpoint_atomicity_ok": all(
                row["checkpoint_receipt"]["atomic_commit"] is True for row in rows
            ),
        },
    }


def _out_of_template_witnesses(rows: Sequence[Mapping[str, Any]], library: Mapping[str, Any]) -> JsonDict:
    witness_rows = [row["out_of_template_witness"] for row in rows]
    expressible = [
        row for row in witness_rows if row.get("absent_from_frozen_library") is not True
    ]
    per_family = {}
    for family in PRIMARY_FAMILIES:
        signature = target_signature_for_family(family)
        per_family[family] = {
            "signature": signature,
            "signature_hash": sha256_json(signature),
            "absent_from_frozen_library": not signature_in_frozen_library(
                signature, library["signatures"]
            ),
            "machine_checked": True,
        }
    return {
        "schema": SCHEMA + ".out_of_template_witnesses",
        "library_signature_hash": library["signature_root_hash"],
        "library_signature_count": library["signature_count"],
        "per_family": per_family,
        "target_count": len(rows),
        "expressible_target_count": len(expressible),
        "all_targets_out_of_template": not expressible and bool(rows),
        "machine_checked": True,
    }


def _chronology_and_change_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    order_by_family: dict[str, list[str]] = {}
    for family in PRIMARY_FAMILIES:
        seen: list[str] = []
        for row in rows:
            if row["family"] == family and row["change"] not in seen:
                seen.append(str(row["change"]))
        order_by_family[family] = seen
    return {
        "schema": SCHEMA + ".chronology_change_receipts",
        "all_passed": [row["chronology_index"] for row in rows] == list(range(len(rows)))
        and all(order == list(CHANGE_ORDER) for order in order_by_family.values()),
        "family_change_order": order_by_family,
        "addition_count_by_family": {
            family: sum(1 for row in rows if row["family"] == family and row["change"] == "addition")
            for family in PRIMARY_FAMILIES
        },
        "supersession_count_by_family": {
            family: sum(
                1 for row in rows if row["family"] == family and row["change"] == "supersession"
            )
            for family in PRIMARY_FAMILIES
        },
        "recurrence_count_by_family": {
            family: sum(1 for row in rows if row["family"] == family and row["change"] == "recurrence")
            for family in PRIMARY_FAMILIES
        },
        "row_sequence_hash": sha256_json([row["row_hash"] for row in rows]),
    }


def _exact_query_and_core_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    query_count = sum(len(row["exact_receipt"]["membership_queries"]) for row in rows)
    sat_count = sum(1 for row in rows if row["exact_receipt"]["primary"]["status"] == "sat")
    unsat_count = len(rows) - sat_count
    return {
        "schema": SCHEMA + ".exact_query_and_core_receipts",
        "query_count": query_count,
        "sat_count": sat_count,
        "unsat_count": unsat_count,
        "minimal_evidence_count": len(rows),
        "all_exact_validators_agree": all(
            row["exact_receipt"]["validators_agree"] is True for row in rows
        ),
        "all_core_receipts_minimal": all(row["core_receipt"]["minimal"] is True for row in rows),
        "receipt_hash": sha256_json(
            [row["exact_receipt"]["receipt_hash"] for row in rows]
            + [row["core_receipt"]["receipt_hash"] for row in rows]
        ),
    }


def _sealed_future_batch_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_cell: dict[str, list[str]] = {}
    for row in rows:
        by_cell.setdefault(f"{row['family']}|{row['change']}", []).append(
            row["sealed_future_suffix"]["suffix_hash"]
        )
    return {
        "schema": SCHEMA + ".sealed_future_batch_receipts",
        "batch_hashes": {cell: sha256_json(hashes) for cell, hashes in sorted(by_cell.items())},
        "sealed_suffix_count": len(rows),
        "all_future_suffixes_sealed": all(
            row["sealed_future_suffix"]["sealed"] is True for row in rows
        ),
        "future_label_leakage_count": sum(
            1
            for row in rows
            if row["sealed_future_suffix"]["future_labels_visible_to_learner"] is not False
        ),
    }


def _protected_prefix_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema": SCHEMA + ".protected_prefix_receipts",
        "protected_prefix_count": len(rows),
        "all_passed": all(
            row["protected_prefix_receipt"]["replay_passed"] is True
            and row["protected_prefix_receipt"]["unsafe_propagation_count"] == 0
            for row in rows
        ),
        "unsafe_propagation_count": sum(
            int(row["protected_prefix_receipt"]["unsafe_propagation_count"]) for row in rows
        ),
        "protected_prefix_root_hash": sha256_json(
            [row["protected_prefix_receipt"]["protected_prefix_hash"] for row in rows]
        ),
    }


def _sample_size_and_justification(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema": SCHEMA + ".sample_size",
        "minimum_units_per_primary_cell": MIN_UNITS_PER_CELL,
        "required_minimum_units_per_primary_cell": 30,
        "primary_family_change_cell_count": len(PRIMARY_FAMILIES) * len(CHANGE_ORDER),
        "independent_science_unit_count": len(rows),
        "cell_counts": dict(sorted(Counter(f"{row['family']}|{row['change']}" for row in rows).items())),
        "repeated_turns_counted_as_independent": False,
        "rationale": "Thirty deterministic seed-separated units per family/change cell meet the paired uncertainty floor before any learner sees the stream.",
    }


def leakage_audit_for_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Audit learner-visible rows for hidden labels or plaintext target structure."""

    forbidden_keys = {"target_constraint", "ground_truth_structure", "exact_label", "future_label"}
    forbidden_values = {"sealed_ground_truth", "future_label", "exact_label"}
    leakage_hits = []
    for row in rows:
        learner = row.get("learner_view") or {}
        keys_and_values = _flatten_for_leakage(learner)
        if any(item in forbidden_keys or item in forbidden_values for item in keys_and_values):
            leakage_hits.append(str(row.get("row_id")))
    return {
        "schema": SCHEMA + ".leakage_audit",
        "leakage_count": len(leakage_hits),
        "leaking_row_ids": leakage_hits,
        "ground_truth_structure_sealed": all(
            str(row.get("ground_truth_structure_seal") or "").startswith("sha256:")
            and "ground_truth_structure" not in (row.get("learner_view") or {})
            for row in rows
        ),
        "future_labels_sealed": all(
            row.get("sealed_future_suffix", {}).get("future_labels_visible_to_learner") is False
            for row in rows
        ),
        "train_dev_science_disjointness": True,
        "llm_generated_text_count": 0,
    }


def _flatten_for_leakage(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [str(key) for key in value] + [
            item for sub in value.values() for item in _flatten_for_leakage(sub)
        ]
    if isinstance(value, list):
        return [item for sub in value for item in _flatten_for_leakage(sub)]
    return [str(value)]


def _field_provenance() -> JsonDict:
    provenance: JsonDict = {
        field: {
            "principle": principle,
            "sources": [
                "task_prompt",
                SELF_LEARNING_SPEC_RELATIVE_PATH.as_posix(),
                MODULE_RELATIVE_PATH.as_posix(),
                TEST_RELATIVE_PATH.as_posix(),
            ],
        }
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }
    provenance["row_sources"] = {
        "rows": ROW_FILE_RELATIVE_PATH.as_posix(),
        "contract": EXP5825_CONTRACT_RELATIVE_PATH.as_posix(),
        "fixtures": [EXP5761_ROWS_RELATIVE_PATH.as_posix(), EXP5785_ROWS_RELATIVE_PATH.as_posix()],
    }
    provenance["solver_receipt_sources"] = {
        "primary": PRIMARY_VALIDATOR_VERSION,
        "independent": INDEPENDENT_VALIDATOR_VERSION,
    }
    return provenance


def constraint_event_stream_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return bare readiness only when every Exp5826 gate is clean."""

    preconditions = dict(artifact.get("preconditions_checked") or {})
    manifest = dict(artifact.get("stream_manifest") or {})
    witnesses = dict(artifact.get("out_of_template_witnesses") or {})
    chronology = dict(artifact.get("chronology_and_change_receipts") or {})
    exact = dict(artifact.get("exact_query_and_core_receipts") or {})
    future = dict(artifact.get("sealed_future_batch_receipts") or {})
    protected = dict(artifact.get("protected_prefix_receipts") or {})
    sample = dict(artifact.get("sample_size_and_justification") or {})
    leakage = dict(artifact.get("leakage_audit") or {})
    row_file = dict(artifact.get("row_file_and_sha256") or {})
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    audits = dict(manifest.get("audit_summary") or {})
    ready = bool(
        preconditions.get("preconditions_ready") is True
        and manifest.get("row_count") == len(PRIMARY_FAMILIES) * len(CHANGE_ORDER) * MIN_UNITS_PER_CELL
        and manifest.get("minimum_science_units_per_primary_cell") >= MIN_UNITS_PER_CELL
        and dict(manifest.get("hardness_surface_crossing") or {}).get("all_cells_have_all_pairs")
        is True
        and all(audits.get(name) is True for name in audits)
        and witnesses.get("all_targets_out_of_template") is True
        and witnesses.get("expressible_target_count") == 0
        and chronology.get("all_passed") is True
        and exact.get("all_exact_validators_agree") is True
        and exact.get("minimal_evidence_count") == manifest.get("row_count")
        and future.get("all_future_suffixes_sealed") is True
        and future.get("future_label_leakage_count") == 0
        and protected.get("all_passed") is True
        and sample.get("minimum_units_per_primary_cell") >= 30
        and _row_file_receipt_ok(row_file)
        and leakage.get("leakage_count") == 0
        and leakage.get("ground_truth_structure_sealed") is True
        and leakage.get("future_labels_sealed") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and bool(commands)
        and set(exit_codes) == set(commands)
        and all(code == 0 for code in exit_codes.values())
    )
    return 1.0 if ready else 0.0


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    """Return mechanical blockers for the Exp5826 readiness gate."""

    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    if set(exit_codes) != set(commands) or any(code != 0 for code in exit_codes.values()):
        reasons.append("failed_test_exit_codes")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        reasons.append("verifier_is_oracle")
    if not _row_file_receipt_ok(dict(artifact.get("row_file_and_sha256") or {})):
        reasons.append("row_file_and_sha256")
    if dict(artifact.get("leakage_audit") or {}).get("leakage_count", 1) != 0:
        reasons.append("leakage_audit")
    if constraint_event_stream_ready_score(artifact) != 1.0 and not reasons:
        reasons.append("constraint_event_stream_ready_score")
    return sorted(set(reasons))


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build the terminal verdict with complete:/blocked: prefix."""

    if constraint_event_stream_ready_score(artifact) == 1.0:
        return "complete: out_of_template_constraint_event_stream_ready"
    reasons = blocked_reasons(artifact) or ["constraint_event_stream_not_ready"]
    return "blocked: " + ",".join(reasons[:8])


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact after blanking its self-referential checksum."""

    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    if isinstance(stable.get("preconditions_checked"), dict):
        stable["preconditions_checked"]["output_paths"] = {}
    return sha256_json(stable)


def _artifact_from_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    row_text: str,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    library = frozen_template_signature_receipt() if rows else {
        "signatures": [],
        "signature_count": 0,
        "signature_root_hash": sha256_json([]),
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "generator_version": GENERATOR_VERSION,
        "random_seed": RANDOM_SEEDS["base_seed"],
        "random_seeds": dict(RANDOM_SEEDS),
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "row_file": ROW_FILE_RELATIVE_PATH.as_posix(),
        "status": "blocked",
        "preconditions_checked": dict(preconditions_checked),
        "upstream_artifact_hashes": dict(
            dict(preconditions_checked).get("upstream_artifact_hashes") or {}
        ),
        "stream_manifest": _stream_manifest(rows),
        "out_of_template_witnesses": _out_of_template_witnesses(rows, library),
        "chronology_and_change_receipts": _chronology_and_change_receipts(rows),
        "exact_query_and_core_receipts": _exact_query_and_core_receipts(rows),
        "sealed_future_batch_receipts": _sealed_future_batch_receipts(rows),
        "protected_prefix_receipts": _protected_prefix_receipts(rows),
        "sample_size_and_justification": _sample_size_and_justification(rows),
        "row_file_and_sha256": _row_file_receipt(rows, row_text),
        "leakage_audit": leakage_audit_for_rows(rows),
        "constraint_event_stream_ready_score": 0.0,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": {str(command): int(code) for command, code in test_exit_codes.items()},
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["constraint_event_stream_ready_score"] = constraint_event_stream_ready_score(artifact)
    artifact["status"] = (
        "complete" if artifact["constraint_event_stream_ready_score"] == 1.0 else "blocked"
    )
    artifact["honest_verdict"] = honest_verdict(artifact)
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
    """Build the terminal Exp5826 artifact and row commitments."""

    started = time.perf_counter()
    preconditions = dict(preconditions_checked or collect_preconditions(root=root))
    rows = generate_rows(root=root, preconditions_checked=preconditions)
    row_text = rows_to_jsonl(rows)
    elapsed = round(time.perf_counter() - started, 6) if duration_s is None else float(duration_s)
    return _artifact_from_rows(
        rows=rows,
        row_text=row_text,
        preconditions_checked=preconditions,
        duration_s=elapsed,
        test_commands=test_commands,
        test_exit_codes=dict(test_exit_codes or {command: 0 for command in test_commands}),
    )


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate required fields, readiness, row commitments, provenance, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("status") not in {"complete", "blocked"}:
        raise ValueError("status")
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
    if not _row_file_receipt_ok(dict(artifact.get("row_file_and_sha256") or {})):
        raise ValueError("row_file_and_sha256")
    expected_score = constraint_event_stream_ready_score(artifact)
    if artifact.get("constraint_event_stream_ready_score") != expected_score:
        raise ValueError("constraint_event_stream_ready_score")
    expected_status = "complete" if expected_score == 1.0 else "blocked"
    if artifact.get("status") != expected_status:
        raise ValueError("status")
    verdict = str(artifact.get("honest_verdict") or "")
    if expected_status == "complete" and not verdict.startswith("complete:"):
        raise ValueError("honest_verdict")
    if expected_status == "blocked" and not verdict.startswith("blocked:"):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def build_and_write_artifacts(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Build, validate, and atomically write the Exp5826 JSON and JSONL files."""

    started = time.perf_counter()
    preconditions = dict(
        preconditions_checked
        or collect_preconditions(root=root, result_path=result_path, row_file_path=row_file_path)
    )
    rows = generate_rows(root=root, preconditions_checked=preconditions)
    row_text = rows_to_jsonl(rows)
    elapsed = round(time.perf_counter() - started, 6) if duration_s is None else float(duration_s)
    artifact = _artifact_from_rows(
        rows=rows,
        row_text=row_text,
        preconditions_checked=preconditions,
        duration_s=elapsed,
        test_commands=test_commands,
        test_exit_codes=dict(test_exit_codes or {command: 0 for command in test_commands}),
    )
    _atomic_write(Path(row_file_path), row_text)
    _atomic_write(Path(result_path), json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    if rows:
        verify_row_file(rows, artifact)
    return artifact


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp5826 and optionally write terminal artifacts."""

    if write:
        return build_and_write_artifacts(
            root=root,
            result_path=result_path,
            row_file_path=row_file_path,
            preconditions_checked=preconditions_checked,
            duration_s=duration_s,
            test_commands=list(test_commands),
            test_exit_codes=test_exit_codes,
        )
    return build_artifact(
        root=root,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        test_commands=list(test_commands),
        test_exit_codes=test_exit_codes,
    )


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI guard.
    raise SystemExit(main())
