"""Build the Exp 3188 THRML factor-graph API boundary artifact.

Spec refs: REQ-HW-099, SCENARIO-HW-099.

This module checks only a local Python API boundary. It turns two tiny
exact-authority rows into a simple Carnot-side factor graph, then attempts to
construct matching THRML software node/block/factor objects when the installed
package exposes them. The result is intentionally not a sampler benchmark:
construction success says the adapter shape is plausible, not that TSU, Kona,
hardware acceleration, or sampler speedup exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from importlib import metadata
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
MILESTONE = "2026.05.295"
ARTIFACT = "experiment_3188_thrml_factor_graph_api_boundary_v1"
SCHEMA = "carnot.thrml_factor_graph_api_boundary.v1"
OUTPUT_REL_PATH = Path("results/experiment_3188_thrml_factor_graph_api_boundary_v1.json")

EXP3180_REL_PATH = Path("results/experiment_3180_controlled_invariance_executor_v2.json")
SPEC_REL_PATH = Path("openspec/capabilities/fpga/spec.md")
RESEARCH_HARDWARE_WISHLIST_REL_PATH = Path("research-hardware-wishlist.md")
RESEARCH_REFERENCES_REL_PATH = Path("research-references.md")

PREFERRED_EXACT_ROW_IDS = ("resyn-3084-arith-000", "resyn-3084-arith-003")
REQUIRED_THRML_SYMBOLS = (
    "thrml.CategoricalNode",
    "thrml.Block",
    "thrml.models.discrete_ebm.CategoricalEBMFactor",
)
REQUIRED_FIELDS = {
    "thrml_factor_graph_api_boundary_v1_ready",
    "thrml_import_available",
    "thrml_version",
    "selected_exact_rows",
    "factor_graph_translation_records",
    "api_gap_records",
    "local_api_smoke_passed",
    "hardware_speedup_claim_allowed",
    "kona_or_tsu_execution_claimed",
    "inference_substrate",
    "honest_verdict",
}
SOURCE_SPECS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False, "text"),
    ("codex_repo_workflow", Path("CODEX.md"), False, "text"),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False, "text"),
    ("fpga_openspec", SPEC_REL_PATH, True, "text"),
    ("research_hardware_wishlist", RESEARCH_HARDWARE_WISHLIST_REL_PATH, True, "text"),
    ("research_references", RESEARCH_REFERENCES_REL_PATH, True, "text"),
    ("exp3180_controlled_invariance_exact_rows", EXP3180_REL_PATH, True, "json"),
    (
        "exp3188_module",
        Path("python/carnot/reporting/thrml_factor_graph_api_boundary_3188.py"),
        False,
        "python",
    ),
    (
        "exp3188_tests",
        Path("tests/python/test_experiment_3188_thrml_factor_graph_api_boundary_v1.py"),
        False,
        "python",
    ),
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3188_thrml_factor_graph_api_boundary_v1.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/reporting -m pytest -o addopts='' tests/python/test_experiment_3188_thrml_factor_graph_api_boundary_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/reporting/thrml_factor_graph_api_boundary_3188.py' --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
)


@dataclass(frozen=True)
class ThrmlProbe:
    """Local THRML import evidence plus the module object, when import succeeded."""

    import_available: bool
    version: str | None
    import_path: str | None
    import_error: str | None
    module: Any | None
    available_symbols: Mapping[str, bool]
    missing_symbols: list[str]


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    thrml_probe: ThrmlProbe | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-HW-099: build the local API-boundary artifact without hardware work."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    probe = thrml_probe or probe_thrml()
    sources = source_artifacts(root_path)
    source_errors = required_source_errors(sources)
    selected_rows = select_exact_rows(root_path)
    state_labels = state_labels_for_rows(selected_rows)
    translation_records = [
        translate_exact_row(row, state_labels=state_labels, probe=probe)
        for row in selected_rows
    ]
    smoke_passed = bool(translation_records) and all(
        record["thrml_mapping"]["construction_check"] == "passed"
        for record in translation_records
    )
    ready = not source_errors and bool(selected_rows) and smoke_passed
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "spec_refs": ["REQ-HW-099", "SCENARIO-HW-099"],
        "thrml_factor_graph_api_boundary_v1_ready": ready,
        "thrml_import_available": probe.import_available,
        "thrml_version": probe.version,
        "thrml_import_path": probe.import_path,
        "thrml_import_error": probe.import_error,
        "thrml_available_symbols": dict(probe.available_symbols),
        "thrml_missing_symbols": list(probe.missing_symbols),
        "selected_exact_rows": selected_rows,
        "factor_graph_translation_records": translation_records,
        "api_gap_records": api_gap_records(probe, smoke_passed),
        "local_api_smoke_passed": smoke_passed,
        "hardware_speedup_claim_allowed": False,
        "kona_or_tsu_execution_claimed": False,
        "inference_substrate": inference_substrate(probe.import_available),
        "source_artifacts": sources,
        "source_errors": source_errors,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(started, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    thrml_probe: ThrmlProbe | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build and persist the Exp 3188 JSON artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(
        root_path,
        thrml_probe=thrml_probe,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def probe_thrml(
    *,
    importer: Callable[[str], Any] = importlib.import_module,
    version: Callable[[str], str] = metadata.version,
) -> ThrmlProbe:
    """Check local THRML import metadata and required construction symbols."""

    try:
        thrml = importer("thrml")
    except BaseException as exc:
        return ThrmlProbe(
            import_available=False,
            version=None,
            import_path=None,
            import_error=exception_text(exc),
            module=None,
            available_symbols={},
            missing_symbols=list(REQUIRED_THRML_SYMBOLS),
        )

    try:
        thrml_version = str(version("thrml"))
    except metadata.PackageNotFoundError:
        thrml_version = str(getattr(thrml, "__version__", "unknown"))

    symbol_map = {symbol: has_symbol(thrml, symbol) for symbol in REQUIRED_THRML_SYMBOLS}
    return ThrmlProbe(
        import_available=True,
        version=thrml_version,
        import_path=str(getattr(thrml, "__file__", "")) or None,
        import_error=None,
        module=thrml,
        available_symbols=symbol_map,
        missing_symbols=[symbol for symbol, available in symbol_map.items() if not available],
    )


def select_exact_rows(root: Path | str = REPO_ROOT) -> list[JsonDict]:
    """Select one clean and one false-accept exact row from Exp 3180 evidence."""

    payload = read_json_object(Path(root) / EXP3180_REL_PATH)
    rows = [row for row in payload.get("exact_rows_evaluated", []) if isinstance(row, Mapping)]
    by_id = {str(row.get("row_id")): row for row in rows}
    selected = [by_id[row_id] for row_id in PREFERRED_EXACT_ROW_IDS if row_id in by_id]
    if len(selected) < 2:
        selected = rows[:2]
    return [exact_row_summary(row) for row in selected[:2]]


def exact_row_summary(row: Mapping[str, Any]) -> JsonDict:
    """Keep only deterministic authority fields needed by the API boundary."""

    exact_label = str(row.get("exact_label") or "")
    candidate_answers = [str(value) for value in row.get("candidate_answers", [])]
    return {
        "row_id": str(row.get("row_id") or ""),
        "source_artifact": EXP3180_REL_PATH.as_posix(),
        "authority": "exact_authority_decision",
        "deterministic_authority": row.get("acceptance_authority") is True,
        "exact_label": exact_label,
        "candidate_answers": candidate_answers,
        "exact_authority_decision": str(row.get("exact_authority_decision") or ""),
        "expected_action": expected_action(exact_label, row),
        "known_false_accept": row.get("known_false_accept_regression") is True,
        "semantic_false_accept": row.get("semantic_false_accept") is True,
    }


def expected_action(exact_label: str, row: Mapping[str, Any]) -> str:
    """Normalize exact labels into the small accept/reject boundary vocabulary."""

    decision = str(row.get("exact_authority_decision") or "").lower()
    if decision in {"accept", "reject", "abstain"}:
        return decision
    return "accept" if exact_label in {"VALID", "SAT"} else "reject"


def state_labels_for_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return deterministic categorical states used by every tiny graph."""

    labels: set[str] = set()
    for row in rows:
        labels.add(str(row.get("exact_label") or ""))
        labels.update(str(value) for value in row.get("candidate_answers", []))
    labels.discard("")
    return sorted(labels) or ["INVALID", "VALID"]


def translate_exact_row(row: Mapping[str, Any], *, state_labels: Sequence[str], probe: ThrmlProbe) -> JsonDict:
    """Build the internal graph and attempt a THRML construction-only mapping."""

    graph = internal_factor_graph(row, state_labels=state_labels)
    return {
        "row_id": row["row_id"],
        "internal_factor_graph": graph,
        "thrml_mapping": attempt_thrml_mapping(graph, probe),
    }


def internal_factor_graph(row: Mapping[str, Any], *, state_labels: Sequence[str]) -> JsonDict:
    """Represent one exact row as categorical variables plus penalty factors."""

    labels = list(state_labels)
    return {
        "graph_id": f"{row['row_id']}:exact_label_boundary",
        "variables": [
            {
                "name": "candidate_label",
                "kind": "categorical",
                "state_labels": labels,
                "observed": False,
            },
            {
                "name": "exact_label",
                "kind": "categorical",
                "state_labels": labels,
                "observed": True,
                "observed_value": row["exact_label"],
            },
        ],
        "factors": [
            {
                "name": "exact_alignment",
                "kind": "categorical_pair_energy_table",
                "scope": ["candidate_label", "exact_label"],
                "energy_table": alignment_energy_table(labels),
                "authority": "zero energy when candidate label equals exact label",
            },
            {
                "name": "known_false_accept_block",
                "kind": "categorical_unary_energy_table",
                "scope": ["candidate_label"],
                "energy_table": regression_energy_table(row, labels),
                "authority": "known false-accept rows penalize candidate labels that conflict with exact authority",
            },
        ],
    }


def alignment_energy_table(labels: Sequence[str]) -> list[list[float]]:
    """Return a zero-diagonal categorical alignment penalty table."""

    return [
        [0.0 if candidate == exact else 1.0 for exact in labels]
        for candidate in labels
    ]


def regression_energy_table(row: Mapping[str, Any], labels: Sequence[str]) -> list[float]:
    """Return a unary conflict penalty for known false-accept rows."""

    exact_label = str(row.get("exact_label") or "")
    candidate_answers = {str(value) for value in row.get("candidate_answers", [])}
    if row.get("known_false_accept") is not True:
        return [0.0 for _label in labels]
    return [2.0 if label in candidate_answers and label != exact_label else 0.0 for label in labels]


def attempt_thrml_mapping(graph: Mapping[str, Any], probe: ThrmlProbe) -> JsonDict:
    """Attempt construction-only mapping to THRML categorical factor concepts."""

    base: JsonDict = {
        "attempted": probe.import_available,
        "constructed": False,
        "construction_check": "not_attempted",
        "api_symbols": list(REQUIRED_THRML_SYMBOLS),
        "constructed_objects": {
            "node_count": 0,
            "block_count": 0,
            "factor_count": 0,
            "interaction_group_count": 0,
        },
        "error": None,
        "claim_boundary": "local construction only; no sampler benchmark or hardware execution",
    }
    if not probe.import_available:
        base["construction_check"] = "blocked_thrml_import_unavailable"
        base["error"] = probe.import_error
        return base
    if probe.missing_symbols:
        base["construction_check"] = "blocked_missing_thrml_symbols"
        base["error"] = "missing symbols: " + ", ".join(probe.missing_symbols)
        return base
    try:
        summary = construct_thrml_categorical_factors(graph, probe.module)
    except BaseException as exc:
        base["construction_check"] = "failed"
        base["error"] = exception_text(exc)
        return base
    base["constructed"] = True
    base["construction_check"] = "passed"
    base["constructed_objects"] = summary
    return base


def construct_thrml_categorical_factors(graph: Mapping[str, Any], thrml_module: Any) -> JsonDict:
    """Instantiate THRML categorical nodes, blocks, and factors for one graph."""

    import jax.numpy as jnp

    node_cls = get_thrml_symbol(thrml_module, "thrml.CategoricalNode")
    block_cls = get_thrml_symbol(thrml_module, "thrml.Block")
    factor_cls = get_thrml_symbol(thrml_module, "thrml.models.discrete_ebm.CategoricalEBMFactor")

    nodes = {variable["name"]: node_cls() for variable in graph["variables"]}
    blocks = {name: block_cls([node]) for name, node in nodes.items()}
    factor_count = 0
    interaction_group_count = 0
    for factor in graph["factors"]:
        factor_blocks = [blocks[name] for name in factor["scope"]]
        weights = -jnp.asarray([factor["energy_table"]], dtype=jnp.float32)
        thrml_factor = factor_cls(factor_blocks, weights)
        groups = thrml_factor.to_interaction_groups()
        factor_count += 1
        interaction_group_count += len(groups)
    return {
        "node_count": len(nodes),
        "block_count": len(blocks),
        "factor_count": factor_count,
        "interaction_group_count": interaction_group_count,
    }


def api_gap_records(probe: ThrmlProbe, smoke_passed: bool) -> list[JsonDict]:
    """Record missing API/hardware capabilities as adapter work items."""

    if not probe.import_available:
        return [
            {
                "gap_id": "thrml_import_unavailable",
                "severity": "blocked_precondition",
                "missing_symbols": list(REQUIRED_THRML_SYMBOLS),
                "details": probe.import_error,
                "next_adapter_steps": [
                    "install or repair the local THRML package in the project environment",
                    "rerun only the Exp 3188 construction smoke after import succeeds",
                ],
            }
        ]
    if probe.missing_symbols:
        return [
            {
                "gap_id": "thrml_factor_graph_symbols_missing",
                "severity": "blocked_precondition",
                "missing_symbols": list(probe.missing_symbols),
                "details": "Installed THRML does not expose every construction symbol required by the adapter.",
                "next_adapter_steps": [
                    "inspect the installed THRML package for renamed node, block, or categorical factor APIs",
                    "add a compatibility shim before attempting sampler integration",
                ],
            }
        ]
    if smoke_passed:
        return [
            {
                "gap_id": "thrml_semantic_metadata_externalized",
                "severity": "adapter_needed",
                "missing_symbols": [],
                "details": "THRML nodes/factors can be constructed, but exact row ids, authority labels, and state-label names remain Carnot-side metadata.",
                "next_adapter_steps": [
                    "preserve row_id, exact_label, candidate_answers, and state_labels in the Carnot adapter wrapper",
                    "add a round-trip metadata test before any sampler integration",
                ],
            }
        ]
    return [
        {
            "gap_id": "thrml_construction_failed",
            "severity": "blocked_precondition",
            "missing_symbols": [],
            "details": "THRML import and symbols were present, but local factor construction did not pass.",
            "next_adapter_steps": [
                "inspect factor_graph_translation_records error fields",
                "adapt the categorical factor tensor shape to the installed THRML API",
            ],
        }
    ]


def inference_substrate(thrml_import_available: bool) -> JsonDict:
    """Separate local import/API checks from hardware or live inference claims."""

    return {
        "kind": "local_thrml_factor_graph_api_construction",
        "local_repo_only": True,
        "executes_hardware": False,
        "hardware_commands_run": [],
        "board_commands_run": [],
        "retired_kv260_host_storage_checks_used": False,
        "executes_models": False,
        "no_live_model_inference": True,
        "installs_packages": False,
        "sampler_benchmark_run": False,
        "sampler_speedup_reported": False,
        "thrml_import_available": thrml_import_available,
        "local_api_smoke_only": True,
        "kona_or_tsu_execution_claimed": False,
    }


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict that does not overstate blocked preconditions."""

    if artifact.get("source_errors"):
        return "blocked_precondition: required exact-row or workflow sources are missing"
    if artifact.get("selected_exact_rows") == []:
        return "blocked_precondition: no deterministic exact rows available for THRML boundary mapping"
    if artifact.get("thrml_import_available") is not True:
        return "blocked_precondition: local thrml import unavailable; construction smoke not attempted"
    if artifact.get("local_api_smoke_passed") is not True:
        return "blocked_precondition: local THRML factor-graph construction smoke did not pass"
    return (
        "complete: THRML factor-graph API boundary materialized; "
        "local_api_smoke_passed=true; hardware_speedup_claim_allowed=false; "
        "kona_or_tsu_execution_claimed=false"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail fast if a required Exp 3188 field is absent."""

    missing = REQUIRED_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required Exp 3188 artifact fields: {sorted(missing)}")


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return provenance for files read or cited by the boundary builder."""

    rows: list[JsonDict] = []
    for role, rel_path, required, source_type in SOURCE_SPECS:
        path = root / rel_path
        payload = read_json_object(path) if source_type == "json" else {}
        rows.append(
            {
                "role": role,
                "path": rel_path.as_posix(),
                "required": required,
                "source_type": source_type,
                "present": path.is_file(),
                "readable_json_object": bool(payload) if source_type == "json" else None,
                "sha256": sha256_file(path),
            }
        )
    return rows


def required_source_errors(sources: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Expose missing or malformed required sources as preflight blockers."""

    errors: list[JsonDict] = []
    for source in sources:
        if source.get("required") is not True:
            continue
        if source.get("present") is not True:
            errors.append(
                {
                    "path": str(source["path"]),
                    "error": "missing_required_source",
                    "source_type": str(source["source_type"]),
                }
            )
        elif source.get("source_type") == "json" and source.get("readable_json_object") is not True:
            errors.append(
                {
                    "path": str(source["path"]),
                    "error": "malformed_required_json_source",
                    "source_type": str(source["source_type"]),
                }
            )
    return errors


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object and fail closed to an empty mapping."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a SHA-256 digest when the source exists."""

    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def duration(started_s: float, now_s: float | None = None) -> float:
    """Return a stable rounded duration for artifacts and tests."""

    ended = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, ended - float(started_s)), 6)


def has_symbol(thrml_module: Any, symbol: str) -> bool:
    """Check dotted THRML symbols relative to the imported top-level module."""

    try:
        get_thrml_symbol(thrml_module, symbol)
    except AttributeError:
        return False
    return True


def get_thrml_symbol(thrml_module: Any, symbol: str) -> Any:
    """Resolve a dotted symbol such as `thrml.models.discrete_ebm.CategoricalEBMFactor`."""

    parts = symbol.split(".")
    current = thrml_module
    for part in parts[1:]:
        current = getattr(current, part)
    return current


def exception_text(exc: BaseException) -> str:
    """Convert an exception into a compact reproducible preflight string."""

    text = str(exc).strip()
    return f"{exc.__class__.__name__}: {text}" if text else exc.__class__.__name__
