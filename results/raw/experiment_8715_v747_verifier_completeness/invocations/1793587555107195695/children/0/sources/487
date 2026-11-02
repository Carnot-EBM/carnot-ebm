"""Exp 3005 solver-to-validator tree expansion corpus.

Spec refs: REQ-VERIFY-3005, SCENARIO-VERIFY-3005.

This harness keeps every verifier decision deterministic.  It turns small
solver/formalization items into validator trees whose runtime nodes check the
candidate envelope and whose Z3 nodes replay the exact formalization.  Partial
candidate checks are deliberately prefix-based: an incomplete candidate is
accepted only when it is still an exact prefix of the reference formalization,
so downstream repair can keep exploring without letting invalid constraints
survive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - absence is handled by validate_artifact blocking in production.
    import z3 as _z3
except Exception:  # pragma: no cover
    _z3 = None

from carnot.eval import sota_solver_formalization_provenance_reproduction_v1 as exp2992


JsonDict = dict[str, Any]
RUN_DATE = "20260524"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILENAME = "experiment_3005_solver_to_validator_tree_expansion_v1.json"
VALIDATOR_MANIFEST_REL_PATH = Path("results/solver_to_validator_tree_expansion_3005/validator_manifest.jsonl")
Z3_TRANSCRIPT_REL_DIR = Path("results/solver_to_validator_tree_expansion_3005/z3_transcripts")
RUNTIME_TRANSCRIPT_REL_DIR = Path("results/solver_to_validator_tree_expansion_3005/runtime_transcripts")
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILENAME
MIN_SOLVER_ITEMS = 20
INFERENCE_SUBSTRATE = "deterministic_runtime_and_z3_validator_tree_corpus"
EXACT_AUTHORITIES = frozenset({"runtime_json_parser", "z3_solver"})
TERMINAL_PREFIXES = ("ready:", "flagged:", "blocked:")
REQUIRED_ARTIFACT_FIELDS = (
    "validator_tree_expanded",
    "validator_manifest_path",
    "n_solver_items",
    "n_validator_trees",
    "all_trees_exact_checked",
    "partial_viability_checked",
    "z3_transcript_paths",
    "runtime_transcript_paths",
    "rejected_constraints",
    "llm_judge_used",
    "honest_verdict",
)


@dataclass(frozen=True)
class SolverItem:
    """One exact-checkable solver/formalization item.

    The item stores already-normalized SMT-LIB statements because partial
    validation needs stable prefixes.  The prompt is provenance, not authority;
    acceptance comes only from runtime parsing plus Z3 replay.
    """

    item_id: str
    prompt: str
    expected_solver_status: str
    assertions: tuple[str, ...]
    source_family: str
    skill_labels: tuple[str, ...]
    expected_answer_values: Mapping[str, Any]


@dataclass(frozen=True)
class ExperimentConfig:
    """Output locations and clock hooks for deterministic Exp 3005 runs."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    manifest_path: Path | None = None
    z3_transcript_dir: Path | None = None
    runtime_transcript_dir: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME

    def resolved_manifest_path(self) -> Path:
        return self.manifest_path or self.repo_root / VALIDATOR_MANIFEST_REL_PATH

    def resolved_z3_transcript_dir(self) -> Path:
        return self.z3_transcript_dir or self.repo_root / Z3_TRANSCRIPT_REL_DIR

    def resolved_runtime_transcript_dir(self) -> Path:
        return self.runtime_transcript_dir or self.repo_root / RUNTIME_TRANSCRIPT_REL_DIR


def build_solver_items(limit: int = MIN_SOLVER_ITEMS) -> list[SolverItem]:
    """Build the fixed 20+ item corpus from Exp 2992 plus generated cases."""

    items: list[SolverItem] = []
    for source in exp2992.fixed_reproduction_items(12):
        formalization = source.accepted_reference_formalization
        items.append(
            SolverItem(
                item_id=source.item_id,
                prompt=source.prompt,
                expected_solver_status=source.expected_solver_status,
                assertions=tuple(_normalize_assertions(formalization["assertions"])),
                source_family="exp2992_solver_feedback",
                skill_labels=tuple(source.skill_labels),
                expected_answer_values=dict(source.expected_answer_values),
            )
        )
    items.extend(_generated_solver_items())
    return items[:limit]


def build_rejected_constraints() -> list[JsonDict]:
    """Return constraints rejected before corpus inclusion with explicit reasons."""

    return [
        {
            "constraint_id": "rejected-randomized-smoke-test",
            "rejection_reason": "nondeterministic_test",
            "detail": "candidate depended on random sampling instead of exact replay",
        },
        {
            "constraint_id": "rejected-text-only-formalization-label",
            "rejection_reason": "missing_exact_check",
            "detail": "constraint had prose labels but no runtime or Z3 executable node",
        },
        {
            "constraint_id": "rejected-looks-correct-label",
            "rejection_reason": "llm_only_label",
            "detail": "acceptance would have depended on model self-judgment",
        },
    ]


def build_validator_tree(item: SolverItem) -> JsonDict:
    """Build one explicit runtime/Z3 validator tree for an item."""

    nodes = [
        {
            "node_id": f"{item.item_id}:required_fields",
            "kind": "required_fields",
            "authority": "runtime_json_parser",
            "required_fields": ["assertions", "query", "expected_status", "answer_extraction"],
        },
        {
            "node_id": f"{item.item_id}:assertion_list_shape",
            "kind": "assertion_list_shape",
            "authority": "runtime_json_parser",
        },
        {
            "node_id": f"{item.item_id}:query_shape",
            "kind": "query_shape",
            "authority": "runtime_json_parser",
            "expected_query": "(check-sat)",
        },
        {
            "node_id": f"{item.item_id}:expected_status_matches_reference",
            "kind": "expected_status_matches_reference",
            "authority": "runtime_json_parser",
            "reference_status": item.expected_solver_status,
        },
        {
            "node_id": f"{item.item_id}:z3_status",
            "kind": "z3_status",
            "authority": "z3_solver",
            "reference_status": item.expected_solver_status,
        },
    ]
    return {
        "tree_id": item.item_id,
        "source_family": item.source_family,
        "root": {"op": "all", "children": [node["node_id"] for node in nodes]},
        "nodes": nodes,
        "reference": {
            "expected_solver_status": item.expected_solver_status,
            "assertions": list(item.assertions),
            "skill_labels": list(item.skill_labels),
        },
    }


def candidate_from_item(item: SolverItem) -> JsonDict:
    """Return the full exact candidate for one solver item."""

    return {
        "assertions": list(item.assertions),
        "query": "(check-sat)",
        "expected_status": item.expected_solver_status,
        "answer_extraction": {"expected_answer_values": dict(item.expected_answer_values)},
    }


def partial_candidate_fixtures(item: SolverItem) -> tuple[JsonDict, JsonDict]:
    """Return one valid extendable partial and one invalid partial candidate."""

    prefix_len = max(1, min(2, len(item.assertions) - 1))
    valid = {
        "assertions": list(item.assertions[:prefix_len]),
        "query": "(check-sat)",
        "expected_status": item.expected_solver_status,
    }
    invalid = {
        "assertions": list(item.assertions[:prefix_len]) + ["(assert false)"],
        "query": "(check-sat)",
        "expected_status": item.expected_solver_status,
    }
    return valid, invalid


def evaluate_validator_tree(
    validator_tree: Mapping[str, Any],
    candidate_text: str,
    *,
    z3_module: Any = _z3,
) -> JsonDict:
    """Execute every full-candidate validator node with exact authorities."""

    payload, parse_reason = _parse_json_object(candidate_text)
    node_results = [
        _evaluate_node(node, payload, parse_reason, z3_module=z3_module)
        for node in validator_tree["nodes"]
    ]
    failing = [str(row["node_id"]) for row in node_results if not row["accepted"]]
    reasons = _rejection_reasons(node_results)
    return {
        "accepted": not failing,
        "failing_node_ids": failing,
        "rejection_reasons": reasons,
        "node_results": node_results,
        "llm_judge_used": False,
    }


def evaluate_partial_candidate(
    validator_tree: Mapping[str, Any],
    candidate_text: str,
    *,
    z3_module: Any = _z3,
) -> JsonDict:
    """Check whether a partial candidate is parseable and extendable."""

    payload, parse_reason = _parse_json_object(candidate_text)
    if parse_reason:
        return _partial_feedback(False, False, [parse_reason], [], False)
    assertions = payload.get("assertions")
    reference = list(validator_tree["reference"]["assertions"])
    if not isinstance(assertions, list) or not all(isinstance(row, str) for row in assertions):
        return _partial_feedback(False, False, ["partial_assertions_not_list"], [], False)  # pragma: no cover
    normalized = [row.strip() for row in assertions]
    if normalized != reference[: len(normalized)]:
        return _partial_feedback(False, False, ["partial_assertions_not_reference_prefix"], [], True)
    z3_result = execute_z3_status(normalized, z3_module=z3_module)
    if not z3_result["z3_executed"]:
        return _partial_feedback(False, False, ["partial_z3_execution_failed"], [z3_result], True)  # pragma: no cover
    return _partial_feedback(True, True, [], [z3_result], True)


def run_experiment(config: ExperimentConfig | None = None, *, z3_module: Any = _z3) -> JsonDict:
    """Build, execute, and persist the Exp 3005 corpus artifacts."""

    active = config or ExperimentConfig()
    started = active.start_time()
    items = build_solver_items()
    manifest_rows: list[JsonDict] = []
    for item in items:
        tree = build_validator_tree(item)
        full_candidate = candidate_from_item(item)
        full_validation = evaluate_validator_tree(tree, json.dumps(full_candidate, sort_keys=True), z3_module=z3_module)
        valid_partial, invalid_partial = partial_candidate_fixtures(item)
        valid_partial_result = evaluate_partial_candidate(
            tree,
            json.dumps(valid_partial, sort_keys=True),
            z3_module=z3_module,
        )
        invalid_partial_result = evaluate_partial_candidate(
            tree,
            json.dumps(invalid_partial, sort_keys=True),
            z3_module=z3_module,
        )
        z3_transcript = _z3_transcript(item, full_candidate, full_validation, z3_module=z3_module)
        runtime_transcript = _runtime_transcript(
            item,
            tree,
            full_validation,
            valid_partial_result,
            invalid_partial_result,
        )
        z3_info = _write_transcript(active.resolved_z3_transcript_dir(), item.item_id, z3_transcript)
        runtime_info = _write_transcript(active.resolved_runtime_transcript_dir(), item.item_id, runtime_transcript)
        manifest_rows.append(
            _manifest_row(
                active.repo_root,
                item,
                tree,
                full_candidate,
                full_validation,
                valid_partial_result,
                invalid_partial_result,
                z3_info,
                runtime_info,
            )
        )

    _write_jsonl(active.resolved_manifest_path(), manifest_rows)
    artifact = build_artifact(
        active,
        manifest_rows,
        duration_s=round(active.clock() - started, 6),
        rejected_constraints=build_rejected_constraints(),
    )
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def build_artifact(
    config: ExperimentConfig,
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    rejected_constraints: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build the terminal artifact and compute strict corpus gates."""

    z3_paths = [str(row["z3_transcript_path"]) for row in manifest_rows]
    runtime_paths = [str(row["runtime_transcript_path"]) for row in manifest_rows]
    n_items = len(manifest_rows)
    n_trees = len({row["validator_tree"]["tree_id"] for row in manifest_rows})
    all_exact = bool(manifest_rows) and all(_row_exact_checked(row) for row in manifest_rows)
    partial_checked = bool(manifest_rows) and all(_row_partial_checked(row) for row in manifest_rows)
    llm_used = any(bool(row.get("llm_judge_used")) for row in manifest_rows)
    expanded = (
        n_items >= MIN_SOLVER_ITEMS
        and n_trees == n_items
        and all_exact
        and partial_checked
        and bool(rejected_constraints)
        and not llm_used
    )
    return {
        "schema": "carnot.solver_to_validator_tree_expansion.v1",
        "artifact": "experiment_3005_solver_to_validator_tree_expansion_v1",
        "run_date": RUN_DATE,
        "validator_tree_expanded": expanded,
        "validator_manifest_path": str(_relative_to(config.repo_root, config.resolved_manifest_path())),
        "n_solver_items": n_items,
        "n_validator_trees": n_trees,
        "all_trees_exact_checked": all_exact,
        "partial_viability_checked": partial_checked,
        "z3_transcript_paths": z3_paths,
        "runtime_transcript_paths": runtime_paths,
        "rejected_constraints": [dict(row) for row in rejected_constraints],
        "llm_judge_used": llm_used,
        "honest_verdict": (
            "ready: expanded deterministic validator-tree corpus exact-checked"
            if expanded
            else "flagged: validator-tree corpus did not clear exact gates"
        ),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_artifacts": source_artifact_status(config.repo_root),
        "manifest_sha256": sha256_file(config.resolved_manifest_path()),
        "field_provenance": field_provenance(),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 3005 artifact violates its terminal contract."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must state ready, flagged, or blocked")
    if artifact.get("llm_judge_used") is not False:
        raise ValueError("llm_judge_used must remain false")
    if int(artifact.get("n_solver_items") or 0) < MIN_SOLVER_ITEMS:
        raise ValueError("expanded corpus requires at least 20 solver items")
    if artifact.get("n_validator_trees") != artifact.get("n_solver_items"):
        raise ValueError("n_validator_trees must equal n_solver_items")
    if artifact.get("all_trees_exact_checked") is not True:
        raise ValueError("all_trees_exact_checked must be true")
    if artifact.get("partial_viability_checked") is not True:
        raise ValueError("partial_viability_checked must be true")


def load_manifest(path: Path) -> list[JsonDict]:
    """Load an inspectable validator manifest JSONL file."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def execute_z3_status(assertions: Sequence[str], *, z3_module: Any = _z3) -> JsonDict:
    """Execute SMT-LIB assertions and return replayable status evidence."""

    if z3_module is None:
        return {"z3_executed": False, "actual_solver_status": None, "z3_error": "z3_unavailable"}  # pragma: no cover
    solver = z3_module.Solver()
    try:
        solver.add(z3_module.parse_smt2_string("\n".join(assertions) + "\n"))
        status = str(solver.check())
    except Exception as exc:  # pragma: no cover - malformed SMT-LIB is rejected before corpus inclusion.
        return {"z3_executed": False, "actual_solver_status": None, "z3_error": f"{type(exc).__name__}: {exc}"}
    return {"z3_executed": True, "actual_solver_status": status, "z3_error": None}


def source_artifact_status(repo_root: Path) -> JsonDict:
    """Summarize source artifacts used to anchor this corpus."""

    results = repo_root / "results"
    return {
        "exp2992": _summarize_json(
            results / "experiment_2992_sota_solver_formalization_provenance_reproduction_v1.json",
            ("solver_provenance_reproduced", "formalization_clean", "n_items"),
        ),
        "exp2994": _summarize_json(
            results / "experiment_2994_prompt_validator_dialogue_schema_v1.json",
            ("prompt_validator_protocol_ready", "exact_verifier_authority_preserved", "n_validator_tree_fixtures"),
        ),
    }


def field_provenance() -> JsonDict:
    """Explain why each required terminal field exists."""

    return {
        "validator_tree_expanded": {
            "principle": "Downstream diagnostics must gate on a real corpus.",
            "satisfied_by": "20 exact-checked solver items with one validator tree each",
        },
        "validator_manifest_path": {
            "principle": "Expanded corpus must be inspectable.",
            "satisfied_by": str(VALIDATOR_MANIFEST_REL_PATH),
        },
        "partial_viability_checked": {
            "principle": "Prefix/partial validity must be explicit.",
            "satisfied_by": "valid extendable and invalid prefix fixtures per item",
        },
        "llm_judge_used": {
            "principle": "LLM judgments must not become verifiers.",
            "satisfied_by": False,
        },
    }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _generated_solver_items() -> list[SolverItem]:
    rows = [
        (
            "gen-3005-001",
            "Integers x and y can sum to five as x=2 and y=3.",
            "sat",
            ["(declare-const x Int)", "(declare-const y Int)", "(assert (= x 2))", "(assert (= y 3))", "(assert (= (+ x y) 5))"],
        ),
        (
            "gen-3005-002",
            "Integer x cannot be both four and less than three.",
            "unsat",
            ["(declare-const x Int)", "(assert (= x 4))", "(assert (< x 3))"],
        ),
        (
            "gen-3005-003",
            "If a implies b, a and not b are inconsistent.",
            "unsat",
            ["(declare-const a Bool)", "(declare-const b Bool)", "(assert (=> a b))", "(assert a)", "(assert (not b))"],
        ),
        (
            "gen-3005-004",
            "If a implies b, not a and not b remain satisfiable.",
            "sat",
            ["(declare-const a Bool)", "(declare-const b Bool)", "(assert (=> a b))", "(assert (not a))", "(assert (not b))"],
        ),
        (
            "gen-3005-005",
            "At least one of p or q cannot coexist with neither p nor q.",
            "unsat",
            ["(declare-const p Bool)", "(declare-const q Bool)", "(assert (or p q))", "(assert (not p))", "(assert (not q))"],
        ),
        (
            "gen-3005-006",
            "Integer n=7 satisfies the bounded interval zero through ten.",
            "sat",
            ["(declare-const n Int)", "(assert (>= n 0))", "(assert (<= n 10))", "(assert (= n 7))"],
        ),
        (
            "gen-3005-007",
            "Integer k cannot equal one and differ from one.",
            "unsat",
            ["(declare-const k Int)", "(assert (= k 1))", "(assert (not (= k 1)))"],
        ),
        (
            "gen-3005-008",
            "A strict ordered chain can be witnessed by one, two, and three.",
            "sat",
            ["(declare-const a Int)", "(declare-const b Int)", "(declare-const c Int)", "(assert (= a 1))", "(assert (= b 2))", "(assert (= c 3))", "(assert (< a b))", "(assert (< b c))"],
        ),
    ]
    return [
        SolverItem(
            item_id=item_id,
            prompt=prompt,
            expected_solver_status=status,
            assertions=tuple(assertions),
            source_family="deterministic_generated",
            skill_labels=("symbolization", "satisfiability"),
            expected_answer_values={},
        )
        for item_id, prompt, status, assertions in rows
    ]


def _normalize_assertions(value: Any) -> list[str]:
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return [str(line).strip() for line in value if str(line).strip()]  # pragma: no cover


def _parse_json_object(candidate_text: str) -> tuple[JsonDict, str | None]:
    try:
        payload = json.loads(candidate_text)
    except json.JSONDecodeError:
        return {}, "json_parse_error"
    if not isinstance(payload, dict):
        return {}, "json_parse_error"  # pragma: no cover
    return payload, None


def _evaluate_node(node: Mapping[str, Any], payload: Mapping[str, Any], parse_reason: str | None, *, z3_module: Any) -> JsonDict:
    if parse_reason:
        return _node_result(node, False, parse_reason)
    kind = str(node["kind"])
    if kind == "required_fields":
        missing = [field for field in node["required_fields"] if field not in payload]
        return _node_result(node, not missing, None if not missing else "missing_required_field")
    if kind == "assertion_list_shape":
        assertions = payload.get("assertions")
        ok = isinstance(assertions, list) and bool(assertions) and all(isinstance(row, str) and row.strip() for row in assertions)
        return _node_result(node, ok, None if ok else "invalid_assertion_list")
    if kind == "query_shape":
        ok = payload.get("query") == node["expected_query"]
        return _node_result(node, ok, None if ok else "query_shape_mismatch")
    if kind == "expected_status_matches_reference":
        ok = payload.get("expected_status") == node["reference_status"]
        return _node_result(node, ok, None if ok else "candidate_expected_status_mismatch")
    if kind == "z3_status":
        z3_result = execute_z3_status([str(row) for row in payload.get("assertions", [])], z3_module=z3_module)
        ok = bool(
            z3_result["z3_executed"]
            and z3_result["actual_solver_status"] == payload.get("expected_status") == node["reference_status"]
        )
        reason = None if ok else "reference_status_mismatch"
        return _node_result(node, ok, reason, z3_result=z3_result)
    raise ValueError(f"unknown validator node kind: {kind}")  # pragma: no cover


def _node_result(node: Mapping[str, Any], accepted: bool, reason: str | None, *, z3_result: Mapping[str, Any] | None = None) -> JsonDict:
    result = {
        "node_id": node["node_id"],
        "kind": node["kind"],
        "authority": node["authority"],
        "accepted": accepted,
        "rejection_reason": reason,
    }
    if z3_result is not None:
        result["z3_result"] = dict(z3_result)
    return result


def _partial_feedback(
    accepted: bool,
    extendable: bool,
    reasons: Sequence[str],
    node_results: Sequence[Mapping[str, Any]],
    checked: bool,
) -> JsonDict:
    return {
        "accepted": accepted,
        "extendable_to_reference": extendable,
        "rejection_reasons": list(reasons),
        "node_results": [dict(row) for row in node_results],
        "partial_viability_checked": checked,
        "llm_judge_used": False,
    }


def _rejection_reasons(node_results: Sequence[Mapping[str, Any]]) -> list[str]:
    return list(
        dict.fromkeys(
            str(row["rejection_reason"])
            for row in node_results
            if row.get("rejection_reason")
        )
    )


def _z3_transcript(
    item: SolverItem,
    full_candidate: Mapping[str, Any],
    full_validation: Mapping[str, Any],
    *,
    z3_module: Any,
) -> JsonDict:
    return {
        "item_id": item.item_id,
        "candidate_sha256": sha256_text(json.dumps(full_candidate, sort_keys=True)),
        "assertions": list(full_candidate["assertions"]),
        "expected_status": full_candidate["expected_status"],
        "z3_version": z3_module.get_version_string() if z3_module is not None else None,
        "full_validation": full_validation,
        "llm_judge_used": False,
    }


def _runtime_transcript(
    item: SolverItem,
    tree: Mapping[str, Any],
    full_validation: Mapping[str, Any],
    valid_partial_result: Mapping[str, Any],
    invalid_partial_result: Mapping[str, Any],
) -> JsonDict:
    return {
        "item_id": item.item_id,
        "validator_tree_sha256": sha256_text(json.dumps(tree, sort_keys=True)),
        "runtime_node_results": [
            row for row in full_validation["node_results"] if row["authority"] == "runtime_json_parser"
        ],
        "partial_viability": {
            "valid_partial": dict(valid_partial_result),
            "invalid_partial": dict(invalid_partial_result),
        },
        "llm_judge_used": False,
    }


def _write_transcript(directory: Path, item_id: str, payload: Mapping[str, Any]) -> JsonDict:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{item_id}.json"
    _write_json(path, payload)
    return {"path": path, "sha256": sha256_file(path)}


def _manifest_row(
    repo_root: Path,
    item: SolverItem,
    tree: Mapping[str, Any],
    full_candidate: Mapping[str, Any],
    full_validation: Mapping[str, Any],
    valid_partial_result: Mapping[str, Any],
    invalid_partial_result: Mapping[str, Any],
    z3_info: Mapping[str, Any],
    runtime_info: Mapping[str, Any],
) -> JsonDict:
    return {
        "item_id": item.item_id,
        "source_family": item.source_family,
        "prompt_sha256": sha256_text(item.prompt),
        "candidate_sha256": sha256_text(json.dumps(full_candidate, sort_keys=True)),
        "validator_tree": dict(tree),
        "full_validation": dict(full_validation),
        "partial_viability": {
            "valid_partial": dict(valid_partial_result),
            "invalid_partial": dict(invalid_partial_result),
        },
        "z3_transcript_path": str(_relative_to(repo_root, Path(z3_info["path"]))),
        "z3_transcript_sha256": z3_info["sha256"],
        "runtime_transcript_path": str(_relative_to(repo_root, Path(runtime_info["path"]))),
        "runtime_transcript_sha256": runtime_info["sha256"],
        "llm_judge_used": False,
    }


def _row_exact_checked(row: Mapping[str, Any]) -> bool:
    authorities = {node["authority"] for node in row["validator_tree"]["nodes"]}
    return bool(
        row["full_validation"]["accepted"]
        and EXACT_AUTHORITIES <= authorities
        and row.get("z3_transcript_path")
        and row.get("z3_transcript_sha256")
        and row.get("runtime_transcript_path")
        and row.get("runtime_transcript_sha256")
    )


def _row_partial_checked(row: Mapping[str, Any]) -> bool:
    partial = row["partial_viability"]
    return bool(
        partial["valid_partial"]["accepted"]
        and partial["valid_partial"]["extendable_to_reference"]
        and partial["invalid_partial"]["accepted"] is False
        and partial["invalid_partial"]["rejection_reasons"]
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:  # pragma: no cover - explicit external paths are not used by this experiment.
        return path


def _summarize_json(path: Path, keys: Sequence[str]) -> JsonDict:
    if not path.exists():
        return {"present": False, "path": str(path), "fields": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))  # pragma: no cover - covered by default run.
    return {"present": True, "path": str(path), "fields": {key: payload.get(key) for key in keys}}  # pragma: no cover


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point for the deterministic Exp 3005 harness."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args(argv)
    artifact = run_experiment(ExperimentConfig(output_path=Path(args.output)))
    print(
        "[exp3005] "
        f"verdict={artifact['honest_verdict']} "
        f"items={artifact['n_solver_items']} "
        f"trees={artifact['n_validator_trees']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
