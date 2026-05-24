"""Exp 3002 deterministic metamorphic oracle audit for hard-set repair.

Spec: REQ-CODE-3002, SCENARIO-CODE-3002.

This module does not generate new repairs. It strengthens the already checked-in
hard-code repair benchmark by creating relation-preserving test variants and by
replaying local validators against reference code, known-bad baselines, cached
Exp 2991 patches, and synthetic overfit probes. The point is to make the oracle
harder to game before any downstream live repair rerun claims promotion.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot.eval import hard_code_stress_manifest as hard


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260524"
ARTIFACT_FILENAME = "experiment_3002_metamorphic_repair_oracle_audit_v1.json"
ARTIFACT = ARTIFACT_FILENAME.removesuffix(".json")
SCHEMA = "carnot.metamorphic_repair_oracle_audit.v1"
EXP2991_ARTIFACT_FILENAME = "experiment_2991_gated_sota_intent_preserving_repair_hard_set_v1.json"
RAW_REL_DIR = Path("results/raw/experiment_3002_metamorphic_repair_oracle_audit_v1")
METAMORPHIC_MANIFEST_REL_PATH = RAW_REL_DIR / "metamorphic_manifest.jsonl"
RECONSTRUCTED_MANIFEST_REL_PATH = RAW_REL_DIR / "reconstructed_hard_manifest.jsonl"
ORIGINAL_TRANSCRIPT_REL_PATH = RAW_REL_DIR / "original_verifier_transcript.jsonl"
METAMORPHIC_TRANSCRIPT_REL_PATH = RAW_REL_DIR / "metamorphic_verifier_transcript.jsonl"
PROBE_TRANSCRIPT_REL_PATH = RAW_REL_DIR / "probe_verifier_transcript.jsonl"
INFERENCE_SUBSTRATE = "deterministic_oracle_audit_no_live_llm"
MIN_SOURCE_ITEMS = 20
RELATION_TYPES: tuple[str, ...] = (
    "alpha_renaming",
    "equivalent_boundary_case",
    "input_permutation",
    "oracle_preserving_refactor",
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "metamorphic_oracle_ready",
    "hard_set_manifest_path",
    "metamorphic_manifest_path",
    "n_source_items",
    "n_metamorphic_variants",
    "relation_types",
    "false_accept_probe_ready",
    "tautology_probe_ready",
    "rejected_variants",
    "verifier_transcript_paths",
    "honest_verdict",
)

BOUNDARY_TESTS: dict[str, tuple[str, ...]] = {
    "repair-hard-0001": ("assert clamp_score(0, 0, 10) == 0",),
    "repair-hard-0004": ("assert median_sorted([2, 8]) == 5.0",),
    "repair-hard-0007": ("assert parse_bool('OFF') is False",),
    "repair-hard-0008": ("assert safe_divide(0, 5, default='x') == 0",),
    "repair-hard-0009": ("assert chunked([1, 2, 3, 4], 4) == [[1, 2, 3, 4]]",),
    "repair-hard-0017": ("assert is_valid_parentheses('()()') is True",),
    "repair-hard-0022": ("assert window_sums([1, 2, 3], 3) == [6]",),
    "repair-hard-0024": ("assert grade_bucket(60) == 'D'",),
}
PERMUTATION_TESTS: dict[str, tuple[str, ...]] = {
    "repair-hard-0003": ("assert count_vowels('noitacudE') == 5",),
    "repair-hard-0020": ("assert longest_common_prefix(['flight', 'flower', 'flow']) == 'fl'",),
    "repair-hard-0021": ("assert anagram_key('room dirty') == 'dimoorrty'",),
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths for the Exp 3002 deterministic audit."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    hard_manifest_path: Path | None = None
    exp2991_artifact_path: Path | None = None
    metamorphic_manifest_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    tests_run: Sequence[str] = field(default_factory=tuple)

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def resolved_hard_manifest_path(self) -> Path:
        return self.hard_manifest_path or self.repo_root / hard.DEFAULT_MANIFEST_REL_PATH

    def resolved_exp2991_artifact_path(self) -> Path:
        return self.exp2991_artifact_path or self.repo_root / "results" / EXP2991_ARTIFACT_FILENAME

    def resolved_metamorphic_manifest_path(self) -> Path:
        return self.metamorphic_manifest_path or self.repo_root / METAMORPHIC_MANIFEST_REL_PATH


def build_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build the terminal Exp 3002 artifact without running live LLM repair."""

    config = config or ExperimentConfig()
    started = config.start_time()
    source = _resolve_source_items(config)
    if not source["items"]:
        return _blocked_artifact(config, started, source)

    accepted, rejected = _generate_variants(source["items"])
    metamorphic_manifest_path = config.resolved_metamorphic_manifest_path()
    _write_jsonl(metamorphic_manifest_path, accepted)
    cached = _load_cached_exp2991_candidates(config, source["items"])
    original_rows = _validate_cases(source["items"], "original_case", cached)
    variant_rows = _validate_cases(accepted, "metamorphic_variant", cached)
    false_accept_summary, false_accept_rows = _run_false_accept_probes(source["items"], accepted)
    tautology_summary, tautology_rows, tautology_rejection = _run_tautology_probe(source["items"])
    rejected = [*rejected, tautology_rejection]
    transcript_paths = _write_transcripts(config, original_rows, variant_rows, false_accept_rows + tautology_rows)
    reference_variant_failures = sum(
        1
        for row in variant_rows
        if row["candidate_key"] == "reference_solution" and row["passed"] is False
    )
    ready = bool(
        len(source["items"]) >= MIN_SOURCE_ITEMS
        and accepted
        and reference_variant_failures == 0
        and false_accept_summary["ready"] is True
        and tautology_summary["ready"] is True
    )
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "metamorphic_oracle_ready": ready,
        "hard_set_manifest_path": source["manifest_path"],
        "metamorphic_manifest_path": str(_relative_or_absolute(config.repo_root, metamorphic_manifest_path)),
        "n_source_items": len(source["items"]),
        "n_metamorphic_variants": len(accepted),
        "relation_types": list(RELATION_TYPES),
        "false_accept_probe_ready": bool(false_accept_summary["ready"]),
        "tautology_probe_ready": bool(tautology_summary["ready"]),
        "rejected_variants": rejected,
        "verifier_transcript_paths": transcript_paths,
        "honest_verdict": (
            "flagged: metamorphic oracle ready; downstream repair promotion must rerun against it"
            if ready
            else "blocked: metamorphic oracle audit did not produce a usable oracle"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "live_llm_repair_run": False,
        "source_manifest_resolution": {
            "mode": source["mode"],
            "source_artifact_path": source["source_artifact_path"],
        },
        "validation_summary": {
            "original_validation_rows": len(original_rows),
            "variant_validation_rows": len(variant_rows),
            "reference_variant_failures": reference_variant_failures,
            "baseline_variant_passes": sum(
                1
                for row in variant_rows
                if row["candidate_key"] == "baseline_candidate" and row["passed"] is True
            ),
            "cached_exp2991_candidates_seen": sum(len(rows) for rows in cached.values()),
        },
        "false_accept_probe_summary": {
            key: value for key, value in false_accept_summary.items() if key != "ready"
        },
        "tautology_probe_summary": {
            key: value for key, value in tautology_summary.items() if key != "ready"
        },
        "source_item_ids": [str(item.get("item_id") or "") for item in source["items"]],
        "metamorphic_manifest_sha256": _sha256_file(metamorphic_manifest_path),
        "duration_s": _elapsed(config, started),
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }


def write_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build and persist the Exp 3002 artifact under ``results/``."""

    config = config or ExperimentConfig()
    artifact = build_artifact(config)
    path = config.artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _resolve_source_items(config: ExperimentConfig) -> JsonDict:
    manifest_path = config.resolved_hard_manifest_path()
    if manifest_path.is_file():
        return {
            "items": hard.load_manifest(manifest_path),
            "manifest_path": str(_relative_or_absolute(config.repo_root, manifest_path)),
            "mode": "manifest",
            "source_artifact_path": str(_relative_or_absolute(config.repo_root, manifest_path)),
        }

    exp2991_path = config.resolved_exp2991_artifact_path()
    payload = _read_json_if_present(exp2991_path)
    selected = [str(item_id) for item_id in payload.get("selected_item_ids") or []]
    defaults = [dict(item) for item in hard.default_items()]
    by_id = {str(item["item_id"]): item for item in defaults}
    items = [by_id[item_id] for item_id in selected if item_id in by_id] or defaults if payload else []
    if not items:
        return {
            "items": [],
            "manifest_path": str(_relative_or_absolute(config.repo_root, manifest_path)),
            "mode": "missing",
            "source_artifact_path": str(_relative_or_absolute(config.repo_root, exp2991_path)),
        }

    reconstructed_path = config.repo_root / RECONSTRUCTED_MANIFEST_REL_PATH
    _write_jsonl(reconstructed_path, items)
    return {
        "items": items,
        "manifest_path": str(RECONSTRUCTED_MANIFEST_REL_PATH),
        "mode": "reconstructed_from_exp2991",
        "source_artifact_path": str(_relative_or_absolute(config.repo_root, exp2991_path)),
    }


def _generate_variants(items: Sequence[JsonDict]) -> tuple[list[JsonDict], list[JsonDict]]:
    accepted: list[JsonDict] = []
    rejected: list[JsonDict] = []
    for item in items:
        for variant in _candidate_variants(item):
            reference = hard.run_candidate_tests(variant, "reference_solution")
            baseline = hard.run_candidate_tests(variant, "baseline_candidate")
            if reference.passed:
                variant["reference_verification"] = reference.as_dict()
                variant["baseline_verification"] = baseline.as_dict()
                accepted.append(variant)
            else:
                rejected.append(
                    _rejected_variant(
                        variant,
                        reason="reference_failed_semantics_changed",
                        reference=reference,
                        baseline=baseline,
                    )
                )
    semantic_reject = _semantic_change_reject_probe(items)
    if semantic_reject:
        rejected.append(semantic_reject)
    return accepted, rejected


def _candidate_variants(item: Mapping[str, Any]) -> list[JsonDict]:
    variants = [_alpha_variant(item)]
    refactor = _refactor_variant(item)
    if refactor is not None:
        variants.append(refactor)
    item_id = str(item.get("item_id") or "")
    for code in BOUNDARY_TESTS.get(item_id, ()):
        variants.append(_extra_test_variant(item, "equivalent_boundary_case", "boundary", [code]))
    for code in PERMUTATION_TESTS.get(item_id, ()):
        variants.append(_extra_test_variant(item, "input_permutation", "permutation", [code]))
    return variants


def _alpha_variant(item: Mapping[str, Any]) -> JsonDict:
    entry = str(item.get("entry_point") or "")
    new_entry = f"{entry}_mr"
    tests = [
        {
            "test_id": f"SCENARIO-CODE-3002-{item['item_id']}-alpha-{index}",
            "code": _rename_identifier(str(test["code"]), entry, new_entry),
        }
        for index, test in enumerate(item.get("tests") or [], start=1)
    ]
    return _variant_item(
        item,
        relation_type="alpha_renaming",
        suffix="alpha",
        entry_point=new_entry,
        tests=tests,
        reference_solution=_rename_identifier(str(item.get("reference_solution") or ""), entry, new_entry),
        baseline_candidate=_rename_identifier(str(item.get("baseline_candidate") or ""), entry, new_entry),
        rationale="Renaming the public function and all verifier calls preserves behavior.",
    )


def _refactor_variant(item: Mapping[str, Any]) -> JsonDict | None:
    tests = []
    for index, test in enumerate(item.get("tests") or [], start=1):
        refactored = _refactor_assert_code(str(test.get("code") or ""))
        if refactored is None:
            return None
        tests.append(
            {
                "test_id": f"SCENARIO-CODE-3002-{item['item_id']}-refactor-{index}",
                "code": refactored,
            }
        )
    return _variant_item(
        item,
        relation_type="oracle_preserving_refactor",
        suffix="refactor",
        entry_point=str(item.get("entry_point") or ""),
        tests=tests,
        reference_solution=str(item.get("reference_solution") or ""),
        baseline_candidate=str(item.get("baseline_candidate") or ""),
        rationale="The assertion is rewritten through actual/expected temporaries only.",
    )


def _extra_test_variant(
    item: Mapping[str, Any],
    relation_type: str,
    suffix: str,
    extra_codes: Sequence[str],
) -> JsonDict:
    original_tests = [dict(test) for test in item.get("tests") or []]
    extra_tests = [
        {
            "test_id": f"SCENARIO-CODE-3002-{item['item_id']}-{suffix}-{index}",
            "code": code,
        }
        for index, code in enumerate(extra_codes, start=1)
    ]
    return _variant_item(
        item,
        relation_type=relation_type,
        suffix=suffix,
        entry_point=str(item.get("entry_point") or ""),
        tests=[*original_tests, *extra_tests],
        reference_solution=str(item.get("reference_solution") or ""),
        baseline_candidate=str(item.get("baseline_candidate") or ""),
        rationale="The added verifier call exercises an equivalent input relation.",
    )


def _variant_item(
    item: Mapping[str, Any],
    *,
    relation_type: str,
    suffix: str,
    entry_point: str,
    tests: Sequence[JsonDict],
    reference_solution: str,
    baseline_candidate: str,
    rationale: str,
) -> JsonDict:
    source_id = str(item.get("item_id") or "")
    variant_id = f"{source_id}__{relation_type}__{suffix}"
    row = dict(item)
    row.update(
        {
            "schema_version": "carnot.metamorphic_repair_oracle.item.v1",
            "item_id": variant_id,
            "variant_id": variant_id,
            "source_item_id": source_id,
            "source_entry_point": str(item.get("entry_point") or ""),
            "entry_point": entry_point,
            "relation_type": relation_type,
            "relation_rationale": rationale,
            "tests": [dict(test) for test in tests],
            "reference_solution": reference_solution,
            "baseline_candidate": baseline_candidate,
        }
    )
    return row


def _semantic_change_reject_probe(items: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    for item in items:
        if item.get("item_id") == "repair-hard-0002":
            variant = _extra_test_variant(
                item,
                "input_permutation",
                "semantic-reject",
                ["assert unique_preserve_order([1, 3, 1, 2, 3]) == [3, 1, 2]"],
            )
            reference = hard.run_candidate_tests(variant, "reference_solution")
            baseline = hard.run_candidate_tests(variant, "baseline_candidate")
            return _rejected_variant(
                variant,
                reason="reference_failed_semantics_changed",
                reference=reference,
                baseline=baseline,
            )
    return None


def _run_false_accept_probes(
    items: Sequence[JsonDict],
    variants: Sequence[JsonDict],
) -> tuple[JsonDict, list[JsonDict]]:
    rows: list[JsonDict] = []
    original_pass_count = 0
    catches = 0
    variants_by_source: dict[str, list[JsonDict]] = {}
    for variant in variants:
        variants_by_source.setdefault(str(variant.get("source_item_id") or ""), []).append(variant)
    for item in items:
        candidate = _make_visible_test_overfit_candidate(item)
        if candidate is None:
            continue
        probe_item = {**item, "visible_test_overfit_candidate": candidate}
        original = hard.run_candidate_tests(probe_item, "visible_test_overfit_candidate")
        rows.append(_transcript_row("false_accept_probe", item, "visible_test_overfit_candidate", original))
        if not original.passed:  # pragma: no cover - generated probes are built from original tests.
            continue
        original_pass_count += 1
        for variant in variants_by_source.get(str(item.get("item_id") or ""), []):
            adapted = _adapt_candidate(
                candidate,
                str(item.get("entry_point") or ""),
                str(variant.get("entry_point") or ""),
            )
            probe_variant = {**variant, "visible_test_overfit_candidate": adapted}
            outcome = hard.run_candidate_tests(probe_variant, "visible_test_overfit_candidate")
            rows.append(_transcript_row("false_accept_probe", variant, "visible_test_overfit_candidate", outcome))
            if not outcome.passed:
                catches += 1
                break
    return (
        {
            "ready": original_pass_count > 0 and catches > 0,
            "original_overfit_pass_count": original_pass_count,
            "metamorphic_catches_count": catches,
        },
        rows,
    )


def _run_tautology_probe(items: Sequence[JsonDict]) -> tuple[JsonDict, list[JsonDict], JsonDict]:
    item = items[0]
    variant = _variant_item(
        item,
        relation_type="tautology_probe",
        suffix="vacuous",
        entry_point=str(item.get("entry_point") or ""),
        tests=[{"test_id": f"SCENARIO-CODE-3002-{item['item_id']}-tautology", "code": "assert True"}],
        reference_solution=str(item.get("reference_solution") or ""),
        baseline_candidate=str(item.get("baseline_candidate") or ""),
        rationale="A vacuous assertion is used only as a rejection probe.",
    )
    reference = hard.run_candidate_tests(variant, "reference_solution")
    baseline = hard.run_candidate_tests(variant, "baseline_candidate")
    rows = [
        _transcript_row("tautology_probe", variant, "reference_solution", reference),
        _transcript_row("tautology_probe", variant, "baseline_candidate", baseline),
    ]
    rejection = _rejected_variant(
        variant,
        reason="tautological_oracle_rejected",
        reference=reference,
        baseline=baseline,
    )
    return (
        {
            "ready": reference.passed and baseline.passed,
            "reference_passes_vacuous_probe": reference.passed,
            "baseline_passes_vacuous_probe": baseline.passed,
        },
        rows,
        rejection,
    )


def _load_cached_exp2991_candidates(
    config: ExperimentConfig,
    items: Sequence[Mapping[str, Any]],
) -> dict[str, list[JsonDict]]:
    payload = _read_json_if_present(config.resolved_exp2991_artifact_path())
    wanted = {str(item.get("item_id") or "") for item in items}
    out: dict[str, list[JsonDict]] = {item_id: [] for item_id in wanted}
    for row in payload.get("candidate_evaluations") or []:
        item_id = str(row.get("item_id") or "")
        rel_path = row.get("candidate_patch_path")
        patch_path = config.repo_root / str(rel_path or "")
        if item_id in wanted and patch_path.is_file():
            out[item_id].append(
                {
                    "candidate_key": f"cached_exp2991_repair_{len(out[item_id])}",
                    "path": str(_relative_or_absolute(config.repo_root, patch_path)),
                    "code": patch_path.read_text(encoding="utf-8"),
                    "model_hf_id": str(row.get("model_hf_id") or ""),
                }
            )
    return out


def _validate_cases(
    cases: Sequence[JsonDict],
    case_kind: str,
    cached: Mapping[str, Sequence[JsonDict]],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for case in cases:
        rows.append(
            _transcript_row(
                case_kind,
                case,
                "reference_solution",
                hard.run_candidate_tests(case, "reference_solution"),
            )
        )
        rows.append(
            _transcript_row(
                case_kind,
                case,
                "baseline_candidate",
                hard.run_candidate_tests(case, "baseline_candidate"),
            )
        )
        source_id = str(case.get("source_item_id") or case.get("item_id") or "")
        for candidate in cached.get(source_id, ()):
            adapted_code = _adapt_candidate(
                str(candidate["code"]),
                str(case.get("source_entry_point") or case.get("entry_point") or ""),
                str(case.get("entry_point") or ""),
            )
            candidate_case = {**case, candidate["candidate_key"]: adapted_code}
            outcome = hard.run_candidate_tests(candidate_case, str(candidate["candidate_key"]))
            rows.append(_transcript_row(case_kind, case, str(candidate["candidate_key"]), outcome, candidate))
    return rows


def _transcript_row(
    case_kind: str,
    item: Mapping[str, Any],
    candidate_key: str,
    outcome: hard.VerificationOutcome,
    candidate: Mapping[str, Any] | None = None,
) -> JsonDict:
    candidate = candidate or {}
    return {
        "case_kind": case_kind,
        "item_id": str(item.get("item_id") or ""),
        "source_item_id": str(item.get("source_item_id") or item.get("item_id") or ""),
        "relation_type": str(item.get("relation_type") or "original"),
        "candidate_key": candidate_key,
        "candidate_path": str(candidate.get("path") or ""),
        "candidate_sha256": outcome.candidate_sha256,
        "passed": outcome.passed,
        "tests_run": outcome.tests_run,
        "failing_test_ids": list(outcome.failing_test_ids),
        "errors": list(outcome.errors),
    }


def _rejected_variant(
    variant: Mapping[str, Any],
    *,
    reason: str,
    reference: hard.VerificationOutcome,
    baseline: hard.VerificationOutcome,
) -> JsonDict:
    return {
        "variant_id": str(variant.get("variant_id") or variant.get("item_id") or ""),
        "source_item_id": str(variant.get("source_item_id") or ""),
        "relation_type": str(variant.get("relation_type") or ""),
        "reason": reason,
        "reference_passed": reference.passed,
        "baseline_passed": baseline.passed,
        "reference_errors": list(reference.errors),
        "baseline_errors": list(baseline.errors),
    }


def _make_visible_test_overfit_candidate(item: Mapping[str, Any]) -> str | None:
    cases = []
    for test in item.get("tests") or []:
        parsed = _literal_call_case(str(item.get("entry_point") or ""), str(test.get("code") or ""))
        if parsed is None:
            return None
        cases.append(parsed)
    mapping = {
        (
            tuple(_freeze_literal(arg) for arg in case["args"]),
            tuple(sorted((key, _freeze_literal(value)) for key, value in case["kwargs"].items())),
        ): case["expected"]
        for case in cases
    }
    return (
        f"def {item['entry_point']}(*args, **kwargs):\n"
        "    def _freeze(value):\n"
        "        if isinstance(value, list):\n"
        "            return tuple(_freeze(item) for item in value)\n"
        "        if isinstance(value, tuple):\n"
        "            return tuple(_freeze(item) for item in value)\n"
        "        if isinstance(value, dict):\n"
        "            return tuple(sorted((key, _freeze(val)) for key, val in value.items()))\n"
        "        return value\n"
        "    raw_kwargs = tuple(sorted(kwargs.items()))\n"
        "    key = (tuple(_freeze(item) for item in args), tuple((kw_key, _freeze(kw_val)) for kw_key, kw_val in raw_kwargs))\n"
        f"    expected = {mapping!r}\n"
        "    return expected.get(key)\n"
    )


def _freeze_literal(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return tuple(_freeze_literal(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_literal(val)) for key, val in value.items()))
    return value


def _literal_call_case(entry_point: str, code: str) -> JsonDict | None:
    tree = ast.parse(code)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return None
    expr = tree.body[0].test
    if not isinstance(expr, ast.Compare) or len(expr.ops) != 1 or len(expr.comparators) != 1:
        return None
    if not isinstance(expr.ops[0], ast.Eq | ast.Is):
        return None
    call = expr.left
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return None
    if call.func.id != entry_point:
        return None
    try:
        args = [ast.literal_eval(arg) for arg in call.args]
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg}
        expected = ast.literal_eval(expr.comparators[0])
    except (ValueError, SyntaxError):
        return None
    return {"args": args, "kwargs": kwargs, "expected": expected}


def _refactor_assert_code(code: str) -> str | None:
    tree = ast.parse(code)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return None
    expr = tree.body[0].test
    if not isinstance(expr, ast.Compare) or len(expr.ops) != 1 or len(expr.comparators) != 1:
        return None
    left = ast.unparse(expr.left)
    right = ast.unparse(expr.comparators[0])
    op = "is" if isinstance(expr.ops[0], ast.Is) else "=="
    return f"actual = {left}\nexpected = {right}\nassert actual {op} expected"


def _rename_identifier(code: str, old: str, new: str) -> str:
    tree = ast.parse(code)

    class Renamer(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            node = self.generic_visit(node)
            if node.name == old:
                node.name = new
            return node

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id == old:
                node.id = new
            return node

    renamed = Renamer().visit(tree)
    ast.fix_missing_locations(renamed)
    return ast.unparse(renamed) + "\n"


def _adapt_candidate(code: str, source_entry: str, target_entry: str) -> str:
    if not source_entry or source_entry == target_entry:
        return code
    try:
        return _rename_identifier(code, source_entry, target_entry)
    except SyntaxError:
        return code


def _write_transcripts(
    config: ExperimentConfig,
    original_rows: Sequence[JsonDict],
    variant_rows: Sequence[JsonDict],
    probe_rows: Sequence[JsonDict],
) -> list[str]:
    paths_and_rows = (
        (config.repo_root / ORIGINAL_TRANSCRIPT_REL_PATH, original_rows),
        (config.repo_root / METAMORPHIC_TRANSCRIPT_REL_PATH, variant_rows),
        (config.repo_root / PROBE_TRANSCRIPT_REL_PATH, probe_rows),
    )
    out = []
    for path, rows in paths_and_rows:
        _write_jsonl(path, rows)
        out.append(str(_relative_or_absolute(config.repo_root, path)))
    return out


def _blocked_artifact(config: ExperimentConfig, started: float, source: Mapping[str, Any]) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "metamorphic_oracle_ready": False,
        "hard_set_manifest_path": str(source.get("manifest_path") or ""),
        "metamorphic_manifest_path": str(METAMORPHIC_MANIFEST_REL_PATH),
        "n_source_items": 0,
        "n_metamorphic_variants": 0,
        "relation_types": list(RELATION_TYPES),
        "false_accept_probe_ready": False,
        "tautology_probe_ready": False,
        "rejected_variants": [],
        "verifier_transcript_paths": [],
        "honest_verdict": "blocked: hard-set source items unavailable",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "live_llm_repair_run": False,
        "source_manifest_resolution": {
            "mode": source.get("mode"),
            "source_artifact_path": source.get("source_artifact_path"),
        },
        "validation_summary": {
            "original_validation_rows": 0,
            "variant_validation_rows": 0,
            "reference_variant_failures": 0,
            "baseline_variant_passes": 0,
            "cached_exp2991_candidates_seen": 0,
        },
        "false_accept_probe_summary": {
            "original_overfit_pass_count": 0,
            "metamorphic_catches_count": 0,
        },
        "tautology_probe_summary": {
            "reference_passes_vacuous_probe": False,
            "baseline_passes_vacuous_probe": False,
        },
        "source_item_ids": [],
        "metamorphic_manifest_sha256": "",
        "duration_s": _elapsed(config, started),
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _read_json_if_present(path: Path) -> JsonDict:
    return dict(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else {}


def _relative_or_absolute(root: Path, path: Path) -> Path:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return path.resolve(strict=False)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _elapsed(config: ExperimentConfig, started: float) -> float:
    return round(max(0.0, config.clock() - started), 6)


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = write_artifact(
        ExperimentConfig(
            tests_run=(
                ".venv/bin/pytest tests/python/test_experiment_3002_metamorphic_repair_oracle_audit.py -q",
                ".venv/bin/pytest tests/python -q",
                "python scripts/check_spec_coverage.py",
            )
        )
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["metamorphic_oracle_ready"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_FILENAME",
    "EXP2991_ARTIFACT_FILENAME",
    "INFERENCE_SUBSTRATE",
    "RECONSTRUCTED_MANIFEST_REL_PATH",
    "RELATION_TYPES",
    "ExperimentConfig",
    "build_artifact",
    "main",
    "write_artifact",
]
