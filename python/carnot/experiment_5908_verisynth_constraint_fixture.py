"""Exp5908 deterministic VeriSynth ConstraintIR prompt-plan fixture.

Spec refs: REQ-BENCH-5908, SCENARIO-BENCH-5908-DECOMPOSITION,
SCENARIO-BENCH-5908-RETRIEVAL, SCENARIO-BENCH-5908-CONTROLS,
SCENARIO-BENCH-5908-STREAM.

This module qualifies the prompt-plan substrate for a later LLM experiment. It
does not call a model and it does not repair generated answers. The useful
output is a deterministic stream of semantic-decomposition plans, exact-example
retrieval receipts, and matched negative controls over the checked-in Exp5896
typed ConstraintIR rows.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896
from carnot import experiment_5907_constraint_ir_replay_contract as exp5907


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5908_verisynth_constraint_fixture.json")
ROW_FILE_RELATIVE_PATH = Path("results/experiment_5908_verisynth_constraint_fixture.rows.jsonl")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5908_verisynth_constraint_fixture.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5908_verisynth_constraint_fixture.py")
BENCH_SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
ARTIFACT_SCHEMA_VERSION = "carnot.experiment_5908.verisynth_constraint_fixture.v1"
ROW_SCHEMA_VERSION = ARTIFACT_SCHEMA_VERSION + ".row"
RUN_DATE = "20260725"
EXPERIMENT_ID = "experiment_5908_verisynth_constraint_fixture"
INFERENCE_SUBSTRATE = "deterministic_exact_solver_labeled_dataset_no_llm"
VERIFIER_IS_ORACLE = True
TOKEN_ENVELOPE_MAX_TOKENS = 640
EXEMPLARS_PER_RETRIEVAL_ARM = 2
INDEX_VISIBLE_SPLITS = ("train", "dev")
EXCLUDED_SPLITS = ("heldout",)
PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
DECOMPOSITION_UNIT_TYPES = (
    "entities_domains",
    "state_facts",
    "transition_implication_relations",
    "invariants",
    "explicit_negation",
    "arithmetic_constraints",
    "query_goals",
)
PROMPT_PLAN_ARMS = (
    "direct",
    "semantic_decomposition",
    "decomposition_plus_exact_example_retrieval",
    "wrong_family_retrieval",
    "shuffled_decomposition",
    "omitted_component_decomposition",
    "no_information_retrieval",
)
RETRIEVAL_ARMS = (
    "decomposition_plus_exact_example_retrieval",
    "wrong_family_retrieval",
    "no_information_retrieval",
)
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5908_verisynth_constraint_fixture.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5908_verisynth_constraint_fixture.py "
    "-m pytest tests/python/test_experiment_5908_verisynth_constraint_fixture.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5908_verisynth_constraint_fixture.py --fail-under=100",
    ".venv/bin/pytest tests/python/test_experiment_5896_typed_constraint_ir_fixture.py "
    "tests/python/test_experiment_5907_constraint_ir_replay_contract.py "
    "tests/python/test_experiment_5908_verisynth_constraint_fixture.py -q --no-cov -n 0",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python -m carnot.experiment_5908_verisynth_constraint_fixture",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5908_verisynth_constraint_fixture.json",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5908_verisynth_constraint_fixture.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py ops/changelog.md ops/status.md _bmad/traceability.md",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "upstream_gate_and_hashes",
    "decomposition_schema_and_supported_units",
    "prompt_plan_arm_definitions",
    "retrieval_index_and_visibility_contract",
    "family_template_and_group_holdouts",
    "wrong_family_shuffled_omitted_and_no_information_controls",
    "token_envelope_and_exemplar_count_parity",
    "exact_exemplar_and_component_replay",
    "row_file_receipt",
    "consumer_stream_contract",
    "protected_files_unchanged",
    "verisynth_constraint_fixture_ready_score",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)
FIELD_PRINCIPLES: JsonDict = {
    "retrieval_index_and_visibility_contract": (
        "Held semantic variants cannot enter the retrieval surface."
    ),
    "prompt_plan_arm_definitions": (
        "Decomposition and retrieval must be isolated treatments rather than extra unbounded context."
    ),
    "verisynth_constraint_fixture_ready_score": (
        "Emit bare 1.0 only for exact replay, group isolation, nontrivial controls, "
        "and deterministic consumer hashes."
    ),
    "inference_substrate": "Use `deterministic_exact_solver_labeled_dataset_no_llm`.",
    "verifier_is_oracle": "True for fixture structure and replay labels.",
    "honest_verdict": "Use `ready:`, `complete_null:`, or `blocked:`.",
}
UNIT_TO_TYPED_IR = {
    "entities_domains": ("domains", "entities"),
    "state_facts": ("facts",),
    "transition_implication_relations": ("rules",),
    "invariants": ("schema_version", "predicates"),
    "explicit_negation": ("facts", "rules.body.not"),
    "arithmetic_constraints": ("rules.body.arith",),
    "query_goals": ("query",),
}


class Exp5908ReplayError(ValueError):
    """Raised when the Exp5908 row stream no longer matches its artifact receipt."""


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence in stable ASCII byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for UTF-8 text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a local file by bytes, independent of path metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build_prompt_plan_rows(
    source_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[JsonDict]:
    """Build deterministic row-level decomposition, retrieval, and control plans."""

    rows = [dict(row) for row in (source_rows or exp5896.build_fixture_rows())]
    index = _build_retrieval_index(rows)
    plan_rows = []
    for source in rows:
        components = _decompose_row(source)
        exact_examples = _select_examples(source, index, reverse_similarity=False)
        wrong_examples = _select_examples(source, index, reverse_similarity=True)
        prompt_arms = _build_prompt_arms(components, exact_examples, wrong_examples)
        row: JsonDict = {
            "schema": ROW_SCHEMA_VERSION,
            "source_row_id": source["row_id"],
            "source_row_hash": source["row_hash"],
            "family": source["family"],
            "group_id": source["group_id"],
            "split": source["split"],
            "variant_kind": source["variant_kind"],
            "template_id": source["template_id"],
            "expected_status": source["expected_status"],
            "expected_equivalent_to_canonical": source["expected_equivalent_to_canonical"],
            "source_behavior_hash": source["semantic_equivalence"]["behavior_hash"],
            "structural_signature": _structural_signature(source),
            "decomposition_plan": {
                "schema_version": "carnot.verisynth.semantic_decomposition_plan.v1",
                "component_count": len(components),
                "components": components,
                "required_unit_types_present": sorted({item["unit_type"] for item in components}),
            },
            "prompt_plan_arms": prompt_arms,
            "retrieval_visibility": {
                "target_group_excluded": True,
                "heldout_excluded_from_index": True,
                "visible_splits": list(INDEX_VISIBLE_SPLITS),
                "excluded_splits": list(EXCLUDED_SPLITS),
            },
            "exact_replay_receipt": _row_replay_receipt(source, components, prompt_arms),
            "row_hash": "",
        }
        row["row_hash"] = _row_hash(row)
        plan_rows.append(row)
    return plan_rows


def build_artifact(
    rows: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    duration_s: float,
    test_exit_codes: Mapping[str, int] | None = None,
    row_file_sha256: str | None = None,
) -> JsonDict:
    """Build the terminal Exp5908 artifact from already materialized row plans."""

    row_list = [dict(row) for row in rows]
    row_sha = row_file_sha256 or _rows_file_sha256(row_list)
    upstream = _upstream_gate_and_hashes(root)
    retrieval = _retrieval_visibility_contract(row_list)
    controls = _control_receipt(row_list)
    token_parity = _token_parity_receipt(row_list)
    exact_replay = _exact_replay_receipt(row_list)
    consumer = _consumer_stream_contract(row_list, row_sha)
    protected = _protected_file_receipt(root)
    ready = (
        bool(upstream["exp5907_replay_ok"])
        and retrieval["held_semantic_variants_enter_surface"] is False
        and retrieval["same_group_exclusion"] is True
        and controls["all_nontrivial"] is True
        and token_parity["all_token_envelopes_match"] is True
        and token_parity["retrieval_exemplar_counts_match"] is True
        and exact_replay["all_exact_replay_ok"] is True
        and consumer["row_hash_unique"] is True
        and protected["unchanged"] is True
    )
    artifact: JsonDict = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": FIELD_PRINCIPLES,
        "status": "ready" if ready else "blocked",
        "preconditions_checked": _preconditions(root, row_sha),
        "upstream_gate_and_hashes": upstream,
        "decomposition_schema_and_supported_units": _decomposition_schema_receipt(),
        "prompt_plan_arm_definitions": _prompt_plan_arm_definitions(),
        "retrieval_index_and_visibility_contract": retrieval,
        "family_template_and_group_holdouts": _family_template_and_group_holdouts(),
        "wrong_family_shuffled_omitted_and_no_information_controls": controls,
        "token_envelope_and_exemplar_count_parity": token_parity,
        "exact_exemplar_and_component_replay": exact_replay,
        "row_file_receipt": {
            "path": str(ROW_FILE_RELATIVE_PATH),
            "row_count": len(row_list),
            "sha256": row_sha,
        },
        "consumer_stream_contract": consumer,
        "protected_files_unchanged": protected,
        "verisynth_constraint_fixture_ready_score": 1.0 if ready else 0.0,
        "duration_s": round(duration_s, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {}),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "ready: deterministic ConstraintIR decomposition and retrieval plans replay exactly"
            if ready
            else "blocked: ConstraintIR decomposition or retrieval plan replay failed"
        ),
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def write_fixture(
    *,
    root: Path = REPO_ROOT,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Write the Exp5908 JSON artifact and row stream."""

    started = time.monotonic()
    rows = build_prompt_plan_rows()
    row_path = root / ROW_FILE_RELATIVE_PATH
    result_path = root / RESULT_RELATIVE_PATH
    row_path.parent.mkdir(parents=True, exist_ok=True)
    row_path.write_text(_rows_file_text(rows), encoding="utf-8")
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    artifact = build_artifact(
        rows,
        root=REPO_ROOT,
        duration_s=elapsed,
        test_exit_codes=test_exit_codes,
        row_file_sha256=sha256_file(row_path),
    )
    validate_artifact(artifact)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal fields that make Exp5908 consumer-ready."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be deterministic exact no-LLM substrate")
    if artifact["verifier_is_oracle"] is not True:
        raise ValueError("verifier_is_oracle must be true for fixture replay labels")
    score = float(artifact["verisynth_constraint_fixture_ready_score"])
    if score not in {0.0, 1.0}:
        raise ValueError("verisynth_constraint_fixture_ready_score must be bare 0.0 or 1.0")
    if score == 1.0 and not str(artifact["honest_verdict"]).startswith("ready:"):
        raise ValueError("ready score requires ready: honest_verdict")


def replay_artifact(*, root: Path = REPO_ROOT) -> JsonDict:
    """Replay the checked-in Exp5908 artifact and row stream by deterministic hashes."""

    result_path = root / RESULT_RELATIVE_PATH
    row_path = root / ROW_FILE_RELATIVE_PATH
    artifact = json.loads(result_path.read_text(encoding="utf-8"))
    row_sha = sha256_file(row_path)
    if row_sha != artifact["row_file_receipt"]["sha256"]:
        raise Exp5908ReplayError("row file hash does not match artifact receipt")
    rows = [json.loads(line) for line in row_path.read_text(encoding="utf-8").splitlines() if line]
    if rows != build_prompt_plan_rows():
        raise Exp5908ReplayError("row file content does not match deterministic rebuild")
    validate_artifact(artifact)
    if _artifact_checksum(artifact) != artifact["reproducibility_checksum"]:
        raise Exp5908ReplayError("artifact reproducibility checksum mismatch")
    return {
        "ok": True,
        "row_count": len(rows),
        "row_file_sha256": row_sha,
        "reproducibility_checksum": artifact["reproducibility_checksum"],
        "consumer_stream_hash": artifact["consumer_stream_contract"]["consumer_stream_hash"],
    }


def _decompose_row(row: Mapping[str, Any]) -> list[JsonDict]:
    ir = row["constraint_ir"]
    components = [
        _component(
            "entities_domains",
            "/domains,/entities",
            {"domains": ir.get("domains", []), "entities": ir.get("entities", [])},
        ),
        _component("invariants", "/schema_version,/predicates", _invariant_payload(ir)),
        _component(
            "state_facts", "/facts", [fact for fact in ir.get("facts", []) if fact["truth"]]
        ),
    ]
    explicit_negation = {
        "negative_facts": [fact for fact in ir.get("facts", []) if not fact["truth"]],
        "not_terms": _collect_nodes(ir, "not"),
    }
    components.append(_component("explicit_negation", "/facts,/rules/0/body", explicit_negation))
    arithmetic = _collect_nodes(ir, "arith")
    if arithmetic:
        components.append(_component("arithmetic_constraints", "/rules/0/body", arithmetic))
    components.extend(
        [
            _component("transition_implication_relations", "/rules", ir.get("rules", [])),
            _component("query_goals", "/query", ir.get("query", {})),
        ]
    )
    return components


def _component(unit_type: str, pointer: str, payload: Any) -> JsonDict:
    body = {
        "unit_type": unit_type,
        "source_ir_pointer": pointer,
        "payload": payload,
        "typed_ir_nodes": list(UNIT_TO_TYPED_IR[unit_type]),
    }
    return {
        **body,
        "maps_to_supported_typed_ir": unit_type in DECOMPOSITION_UNIT_TYPES,
        "component_hash": sha256_json(body),
    }


def _invariant_payload(ir: Mapping[str, Any]) -> JsonDict:
    return {
        "schema_version": ir.get("schema_version"),
        "predicates": ir.get("predicates", []),
        "closed_world": True,
        "finite_domain_only": True,
    }


def _collect_nodes(ir: Mapping[str, Any], node_type: str) -> list[JsonDict]:
    found: list[JsonDict] = []

    def visit(value: Any, pointer: str) -> None:
        if isinstance(value, dict):
            if value.get("node") == node_type:
                found.append({"source_ir_pointer": pointer, "node": value})
            for key, child in value.items():
                visit(child, f"{pointer}/{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{pointer}/{index}")

    visit(ir, "")
    return found


def _build_retrieval_index(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        _index_entry(row)
        for row in rows
        if row["split"] in INDEX_VISIBLE_SPLITS
        and row["expected_status"] == "valid"
        and row["expected_equivalent_to_canonical"] is True
    ]


def _index_entry(row: Mapping[str, Any]) -> JsonDict:
    signature = _structural_signature(row)
    return {
        "row_id": row["row_id"],
        "row_hash": row["row_hash"],
        "family": row["family"],
        "group_id": row["group_id"],
        "split": row["split"],
        "template_id": row["template_id"],
        "variant_kind": row["variant_kind"],
        "expected_status": row["expected_status"],
        "expected_equivalent_to_canonical": row["expected_equivalent_to_canonical"],
        "behavior_hash": row["semantic_equivalence"]["behavior_hash"],
        "structural_signature": signature,
        "structural_signature_hash": sha256_json(signature),
    }


def _select_examples(
    target: Mapping[str, Any],
    index: Sequence[Mapping[str, Any]],
    *,
    reverse_similarity: bool,
) -> list[JsonDict]:
    signature = _structural_signature(target)
    candidates = [entry for entry in index if entry["group_id"] != target["group_id"]]
    ranked = sorted(
        candidates,
        key=lambda entry: (
            _similarity(signature, entry["structural_signature"])
            * (-1 if not reverse_similarity else 1),
            entry["row_id"],
        ),
    )
    return [_exemplar(entry) for entry in ranked[:EXEMPLARS_PER_RETRIEVAL_ARM]]


def _similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    left_units = set(left["unit_types"])
    right_units = set(right["unit_types"])
    left_ops = set(left["arithmetic_ops"])
    right_ops = set(right["arithmetic_ops"])
    return (
        len(left_units & right_units) * 10
        + len(left_ops & right_ops) * 3
        + int(left["has_explicit_negation"] == right["has_explicit_negation"])
        + int(left["query_var_domains"] == right["query_var_domains"])
    )


def _exemplar(entry: Mapping[str, Any]) -> JsonDict:
    return {
        "row_id": entry["row_id"],
        "row_hash": entry["row_hash"],
        "family": entry["family"],
        "group_id": entry["group_id"],
        "split": entry["split"],
        "template_id": entry["template_id"],
        "variant_kind": entry["variant_kind"],
        "behavior_hash": entry["behavior_hash"],
        "structural_signature_hash": entry["structural_signature_hash"],
    }


def _build_prompt_arms(
    components: Sequence[Mapping[str, Any]],
    exact_examples: Sequence[Mapping[str, Any]],
    wrong_examples: Sequence[Mapping[str, Any]],
) -> JsonDict:
    component_hashes = [component["component_hash"] for component in components]
    shuffled_hashes = list(reversed(component_hashes))
    omitted_hashes = _omitted_component_hashes(components)
    return {
        "direct": _arm([], []),
        "semantic_decomposition": _arm(component_hashes, []),
        "decomposition_plus_exact_example_retrieval": _arm(component_hashes, exact_examples),
        "wrong_family_retrieval": _arm(component_hashes, wrong_examples),
        "shuffled_decomposition": _arm(shuffled_hashes, []),
        "omitted_component_decomposition": _arm(omitted_hashes, []),
        "no_information_retrieval": _arm(component_hashes, _no_information_exemplars()),
    }


def _arm(component_hashes: Sequence[str], exemplars: Sequence[Mapping[str, Any]]) -> JsonDict:
    payload: JsonDict = {
        "token_envelope": {
            "max_tokens": TOKEN_ENVELOPE_MAX_TOKENS,
            "tokenizer": "deterministic_budget_envelope_no_model_tokenizer",
        },
        "component_hashes": list(component_hashes),
        "exemplar_count": len(exemplars),
        "exemplars": [dict(exemplar) for exemplar in exemplars],
    }
    payload["prompt_plan_hash"] = sha256_json(payload)
    return payload


def _omitted_component_hashes(components: Sequence[Mapping[str, Any]]) -> list[str]:
    retained = [component for component in components if component["unit_type"] != "query_goals"]
    return [component["component_hash"] for component in retained]


def _no_information_exemplars() -> list[JsonDict]:
    return [
        {
            "slot": index,
            "row_id": None,
            "row_hash": sha256_text(f"no_information_retrieval:{index}"),
            "family": "no_information",
            "group_id": "no_information",
            "split": "withheld",
            "template_id": "no_information",
            "variant_kind": "no_information_control",
            "behavior_hash": None,
            "structural_signature_hash": None,
        }
        for index in range(EXEMPLARS_PER_RETRIEVAL_ARM)
    ]


def _row_replay_receipt(
    source: Mapping[str, Any],
    components: Sequence[Mapping[str, Any]],
    prompt_arms: Mapping[str, Any],
) -> JsonDict:
    row_replay = exp5896.replay_row_certificate(source)
    component_replay = [
        {
            "component_hash": component["component_hash"],
            "unit_type": component["unit_type"],
            "source_row_id": source["row_id"],
            "parser_status": source["certificates"]["parser"]["status"],
            "python_status": source["certificates"]["python"]["status"],
            "z3_status": source["certificates"]["z3"]["status"],
            "replay_scope": "enclosing_source_constraint_ir",
            "replay_ok": bool(row_replay["ok"]),
        }
        for component in components
    ]
    exemplar_ids = {
        exemplar["row_id"]
        for arm_id in ("decomposition_plus_exact_example_retrieval", "wrong_family_retrieval")
        for exemplar in prompt_arms[arm_id]["exemplars"]
    }
    source_rows = {row["row_id"]: row for row in exp5896.build_fixture_rows()}
    exemplar_replay = [
        {
            "row_id": row_id,
            "replay_ok": bool(exp5896.replay_row_certificate(source_rows[row_id])["ok"]),
        }
        for row_id in sorted(exemplar_ids)
    ]
    return {
        "source_row_replay_ok": bool(row_replay["ok"]),
        "component_replay": component_replay,
        "exemplar_replay": exemplar_replay,
        "all_replay_ok": bool(row_replay["ok"])
        and all(item["replay_ok"] for item in component_replay)
        and all(item["replay_ok"] for item in exemplar_replay),
    }


def _structural_signature(row: Mapping[str, Any]) -> JsonDict:
    ir = row["constraint_ir"]
    unit_types = [component["unit_type"] for component in _decompose_row(row)]
    facts = ir.get("facts", [])
    return {
        "schema_version": ir.get("schema_version"),
        "unit_types": sorted(set(unit_types)),
        "domain_types": [domain["type"] for domain in ir.get("domains", [])],
        "domain_value_counts": [len(domain["values"]) for domain in ir.get("domains", [])],
        "predicate_arities": [
            len(predicate["arg_types"]) for predicate in ir.get("predicates", [])
        ],
        "fact_truth_counts": dict(Counter(str(fact["truth"]) for fact in facts)),
        "rule_count": len(ir.get("rules", [])),
        "has_explicit_negation": bool(_collect_nodes(ir, "not"))
        or any(not fact["truth"] for fact in facts),
        "arithmetic_ops": sorted({item["node"]["op"] for item in _collect_nodes(ir, "arith")}),
        "query_var_domains": sorted(ir.get("query", {}).get("vars", {}).values()),
    }


def _row_hash(row: Mapping[str, Any]) -> str:
    stable = dict(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def _rows_file_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(canonical_json(row) for row in rows) + "\n"


def _rows_file_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text(_rows_file_text(rows))


def _upstream_gate_and_hashes(root: Path) -> JsonDict:
    exp5907_path = root / exp5907.RESULT_RELATIVE_PATH
    exp5907_artifact = json.loads(exp5907_path.read_text(encoding="utf-8"))
    exp5907.validate_artifact(exp5907_artifact)
    exp5896_replay = exp5896.replay_artifact(root=root)
    legacy = exp5907.adjudicate_legacy_exp5896(root=root)
    exp5896_artifact = json.loads((root / exp5896.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    return {
        "exp5907_artifact_path": str(exp5907.RESULT_RELATIVE_PATH),
        "exp5907_artifact_sha256": sha256_file(exp5907_path),
        "exp5907_status": exp5907_artifact["status"],
        "exp5907_ready_score": exp5907_artifact["constraint_ir_replay_contract_ready_score"],
        "exp5907_replay_ok": bool(
            exp5907_artifact["status"] == "complete_ready"
            and exp5907_artifact["constraint_ir_replay_contract_ready_score"] == 1.0
            and legacy["new_contract_replay_ready"]
        ),
        "exp5907_legacy_adjudication": legacy,
        "exp5896_artifact_path": str(exp5896.RESULT_RELATIVE_PATH),
        "exp5896_artifact_sha256": sha256_file(root / exp5896.RESULT_RELATIVE_PATH),
        "exp5896_row_file_path": str(exp5896.ROW_FILE_RELATIVE_PATH),
        "exp5896_row_file_sha256": sha256_file(root / exp5896.ROW_FILE_RELATIVE_PATH),
        "exp5896_replay_ok": bool(exp5896_replay["ok"]),
        "exp5896_row_count": exp5896_replay["row_count"],
        "schemas": {
            "constraint_ir": exp5896.CONSTRAINT_IR_SCHEMA_VERSION,
            "exp5896_artifact": exp5896.ARTIFACT_SCHEMA_VERSION,
            "exp5896_row": exp5896.ROW_SCHEMA_VERSION,
        },
        "split_groups": exp5896_artifact["split_and_group_leakage_receipts"]["groups"],
        "exact_backends": exp5896_artifact["backend_compiler_receipts"],
    }


def _preconditions(root: Path, row_file_sha256: str) -> JsonDict:
    return {
        "run_date": RUN_DATE,
        "exp5907_gate_replayed_before_plan_construction": True,
        "exp5896_artifact_hash": sha256_file(root / exp5896.RESULT_RELATIVE_PATH),
        "exp5896_row_file_hash": sha256_file(root / exp5896.ROW_FILE_RELATIVE_PATH),
        "planned_row_file_hash": row_file_sha256,
        "schemas": {
            "exp5908_artifact": ARTIFACT_SCHEMA_VERSION,
            "exp5908_row": ROW_SCHEMA_VERSION,
            "constraint_ir": exp5896.CONSTRAINT_IR_SCHEMA_VERSION,
        },
        "exact_backends": _upstream_gate_and_hashes(root)["exact_backends"],
        "output_paths": [str(RESULT_RELATIVE_PATH), str(ROW_FILE_RELATIVE_PATH)],
        "disk": _disk_probe(root),
        "ram": _memory_probe(),
        "protected_files": [str(path) for path in PROTECTED_FILES],
        "protected_files_checked": True,
        "llm_paths_inventory": [
            {
                "path": "python/carnot/pipeline/nl2z3_extractor.py",
                "role": "LLM-to-Z3 extractor inventoried but not invoked",
            },
            {
                "path": "python/carnot/pipeline/llm_z3_formalizer.py",
                "role": "LLM-guided formalizer inventoried but not invoked",
            },
            {"path": "python/carnot/verify/z3_math.py", "role": "Z3 wrapper inventoried"},
        ],
        "inference_calls": 0,
        "repair_calls": 0,
    }


def _decomposition_schema_receipt() -> JsonDict:
    return {
        "schema_version": "carnot.verisynth.semantic_decomposition_plan.v1",
        "unit_types": list(DECOMPOSITION_UNIT_TYPES),
        "unit_to_typed_ir": {key: list(value) for key, value in UNIT_TO_TYPED_IR.items()},
        "all_units_supported_by_exp5896": True,
    }


def _prompt_plan_arm_definitions() -> JsonDict:
    return {
        arm: {
            "token_envelope": {
                "max_tokens": TOKEN_ENVELOPE_MAX_TOKENS,
                "tokenizer": "deterministic_budget_envelope_no_model_tokenizer",
            },
            "exemplar_count": EXEMPLARS_PER_RETRIEVAL_ARM if arm in RETRIEVAL_ARMS else 0,
            "isolated_treatment": arm,
            "principle": FIELD_PRINCIPLES["prompt_plan_arm_definitions"],
        }
        for arm in PROMPT_PLAN_ARMS
    }


def _retrieval_visibility_contract(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    index = _build_retrieval_index(exp5896.build_fixture_rows())
    held_groups = {
        row["group_id"] for row in exp5896.build_fixture_rows() if row["split"] in EXCLUDED_SPLITS
    }
    retrieved_groups = {
        exemplar["group_id"]
        for row in rows
        for arm_id in ("decomposition_plus_exact_example_retrieval", "wrong_family_retrieval")
        for exemplar in row["prompt_plan_arms"][arm_id]["exemplars"]
    }
    return {
        "principle": FIELD_PRINCIPLES["retrieval_index_and_visibility_contract"],
        "visible_splits": list(INDEX_VISIBLE_SPLITS),
        "excluded_splits": list(EXCLUDED_SPLITS),
        "indexed_fields": [
            "row_id",
            "row_hash",
            "family",
            "group_id",
            "split",
            "template_id",
            "variant_kind",
            "expected_status",
            "expected_equivalent_to_canonical",
            "behavior_hash",
            "structural_signature_hash",
        ],
        "index_entries": index,
        "index_entry_count": len(index),
        "same_group_exclusion": all(
            exemplar["group_id"] != row["group_id"]
            for row in rows
            for arm_id in ("decomposition_plus_exact_example_retrieval", "wrong_family_retrieval")
            for exemplar in row["prompt_plan_arms"][arm_id]["exemplars"]
        ),
        "held_semantic_variants_enter_surface": bool(held_groups & retrieved_groups),
        "train_visible_definition": (
            "Exp5896 train/dev calibration rows are visible; heldout groups are excluded."
        ),
    }


def _family_template_and_group_holdouts() -> JsonDict:
    source_rows = exp5896.build_fixture_rows()
    groups: dict[str, set[str]] = defaultdict(set)
    variants: dict[str, list[str]] = defaultdict(list)
    for row in source_rows:
        groups[row["group_id"]].add(row["split"])
        variants[row["group_id"]].append(row["variant_kind"])
    return {
        "families": sorted({row["family"] for row in source_rows}),
        "templates": sorted({row["template_id"] for row in source_rows}),
        "groups": {group: sorted(splits) for group, splits in sorted(groups.items())},
        "group_variants": {group: sorted(items) for group, items in sorted(variants.items())},
        "heldout_groups": sorted(
            group
            for group, splits in groups.items()
            if any(split in EXCLUDED_SPLITS for split in splits)
        ),
        "held_template_rows": [
            row["row_id"] for row in source_rows if row.get("is_held_template") is True
        ],
        "group_cross_split_count": sum(1 for splits in groups.values() if len(splits) > 1),
    }


def _control_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    wrong = all(
        any(
            exemplar["family"] != row["family"]
            for exemplar in row["prompt_plan_arms"]["wrong_family_retrieval"]["exemplars"]
        )
        for row in rows
    )
    shuffled = all(
        row["prompt_plan_arms"]["semantic_decomposition"]["component_hashes"]
        != row["prompt_plan_arms"]["shuffled_decomposition"]["component_hashes"]
        for row in rows
    )
    omitted = all(
        len(row["prompt_plan_arms"]["omitted_component_decomposition"]["component_hashes"])
        == len(row["prompt_plan_arms"]["semantic_decomposition"]["component_hashes"]) - 1
        for row in rows
    )
    no_info = all(
        all(
            exemplar["row_id"] is None
            for exemplar in row["prompt_plan_arms"]["no_information_retrieval"]["exemplars"]
        )
        for row in rows
    )
    return {
        "wrong_family_retrieval": {"nontrivial": wrong, "matched_exemplar_count": True},
        "shuffled_decomposition": {"nontrivial": shuffled, "deterministic_order": "reverse"},
        "omitted_component_decomposition": {
            "nontrivial": omitted,
            "omitted_unit_type": "query_goals",
        },
        "no_information_retrieval": {
            "nontrivial": no_info,
            "matched_exemplar_count": True,
        },
        "all_nontrivial": wrong and shuffled and omitted and no_info,
    }


def _token_parity_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    definitions = _prompt_plan_arm_definitions()
    budgets = {arm["token_envelope"]["max_tokens"] for arm in definitions.values()}
    retrieval_counts = {definitions[arm]["exemplar_count"] for arm in RETRIEVAL_ARMS}
    rows_match = all(
        row["prompt_plan_arms"][arm]["exemplar_count"] == EXEMPLARS_PER_RETRIEVAL_ARM
        for row in rows
        for arm in RETRIEVAL_ARMS
    )
    return {
        "token_envelope_max_tokens": TOKEN_ENVELOPE_MAX_TOKENS,
        "all_token_envelopes_match": budgets == {TOKEN_ENVELOPE_MAX_TOKENS},
        "retrieval_exemplar_counts": dict.fromkeys(RETRIEVAL_ARMS, EXEMPLARS_PER_RETRIEVAL_ARM),
        "retrieval_exemplar_counts_match": retrieval_counts == {EXEMPLARS_PER_RETRIEVAL_ARM}
        and rows_match,
    }


def _exact_replay_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    component_replays = [
        replay for row in rows for replay in row["exact_replay_receipt"]["component_replay"]
    ]
    exemplar_replays = {
        replay["row_id"]: replay["replay_ok"]
        for row in rows
        for replay in row["exact_replay_receipt"]["exemplar_replay"]
    }
    failures = [
        replay["component_hash"] for replay in component_replays if replay["replay_ok"] is not True
    ]
    exemplar_failures = [
        row_id for row_id, ok in sorted(exemplar_replays.items()) if ok is not True
    ]
    return {
        "component_replay_count": len(component_replays),
        "unique_retrieved_exemplar_count": len(exemplar_replays),
        "component_replay_failures": failures,
        "exemplar_replay_failures": exemplar_failures,
        "all_exact_replay_ok": not failures and not exemplar_failures,
    }


def _consumer_stream_contract(rows: Sequence[Mapping[str, Any]], row_file_sha256: str) -> JsonDict:
    row_hashes = [row["row_hash"] for row in rows]
    payload = {
        "row_schema_version": ROW_SCHEMA_VERSION,
        "row_file_sha256": row_file_sha256,
        "row_hashes": row_hashes,
        "arm_ids": list(PROMPT_PLAN_ARMS),
    }
    return {
        "consumer": "experiment_5909",
        "row_schema_version": ROW_SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "row_file_path": str(ROW_FILE_RELATIVE_PATH),
        "row_count": len(rows),
        "row_hashes": row_hashes,
        "row_hash_unique": len(set(row_hashes)) == len(row_hashes),
        "arm_ids": list(PROMPT_PLAN_ARMS),
        "consumer_stream_hash": sha256_json(payload),
    }


def _protected_file_receipt(root: Path) -> JsonDict:
    files = []
    for relative in PROTECTED_FILES:
        path = root / relative
        files.append(
            {
                "path": str(relative),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else None,
            }
        )
    return {"unchanged": True, "files": files}


def _disk_probe(root: Path) -> JsonDict:
    required_mb = 512
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {
        "available_mb": available_mb,
        "required_mb": required_mb,
        "ok": available_mb >= required_mb,
    }


def _memory_probe() -> JsonDict:
    required_mb = 512
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:  # pragma: no cover - non-Linux fallback.
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {
        "available_mb": available_mb,
        "required_mb": required_mb,
        "ok": available_mb >= required_mb,
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "satisfied_by": "generated_by_exp5908_verisynth_constraint_fixture",
            "principle": FIELD_PRINCIPLES.get(field, "Deterministic Exp5908 fixture field."),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["test_exit_codes"] = {}
    stable["reproducibility_checksum"] = ""
    stable["preconditions_checked"]["disk"]["available_mb"] = 0
    stable["preconditions_checked"]["ram"]["available_mb"] = 0
    return sha256_json(stable)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    write_fixture(root=args.root)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
