"""Exp1540 scaled product-line staged benchmark.

Spec: REQ-BENCH-1540, SCENARIO-BENCH-1540.

Exp1523 proved that a six-row product-line rescue could reach perfect parser
and oracle agreement after staged feedback.  This module checks whether that
branch is worth carrying forward by scaling the same authority boundary: LLMs
and automata may provide candidate JSON, but only the deterministic
feature-model oracle can accept a final selection.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from carnot.eval import product_line_parser_feasibility_rescue as rescue1523
from carnot.eval import product_line_solver_oracle_benchmark as exp1511
from carnot.verify.xgrammar_abs_contract_decoder_adapter import ABSDFAMask

JsonDict = dict[str, Any]

RUN_DATE = "20260508"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1540_product_line_staged_benchmark_scale_v3.json")
DEFAULT_MANIFEST_PATH = Path("results/product_line_staged_benchmark_scale_1540.jsonl")
DEFAULT_RESCUE_ARTIFACT_PATH = Path("results/experiment_1523_product_line_parser_feasibility_rescue_v2.json")
DEFAULT_RESCUE_MANIFEST_PATH = Path("results/product_line_rescue_1523.jsonl")
TARGET_CASE_COUNT = 40
MANDATED_MODEL_SPECS: tuple[str, ...] = tuple(
    str(spec["hf_id"]) for spec in exp1511.MANDATED_MODEL_SPECS
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "milestone",
    "product_line_scale_ready",
    "branch_retired",
    "model_specs",
    "live_sota_model_inference_used",
    "cases_total",
    "syntax_stage_pass_rate",
    "feature_model_stage_pass_rate",
    "feasibility_stage_pass_rate",
    "oracle_agreement_rate",
    "false_accept_rate",
    "benchmark_manifest_path",
    "retirement_reason",
    "focused_tests_passed",
    "honest_verdict",
)

CachedPairFn = Callable[..., list[JsonDict] | None]
LiveCollectorFn = Callable[[list[exp1511.ProductLineCase], JsonDict, int], JsonDict]


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Write the durable bootstrap artifact before source loading or probing."""

    payload: JsonDict = {
        "status": "in_progress",
        "milestone": run_date,
        "product_line_scale_ready": False,
        "branch_retired": False,
        "model_specs": list(MANDATED_MODEL_SPECS),
        "live_sota_model_inference_used": False,
        "cases_total": 0,
        "syntax_stage_pass_rate": 0.0,
        "feature_model_stage_pass_rate": 0.0,
        "feasibility_stage_pass_rate": 0.0,
        "oracle_agreement_rate": 0.0,
        "false_accept_rate": 0.0,
        "benchmark_manifest_path": _display_path(manifest_path),
        "retirement_reason": "",
        "focused_tests_passed": False,
        "honest_verdict": "complete: in_progress",
    }
    _write_json(Path(output_path), payload)
    return payload


def build_staged_product_line_cases(
    target_count: int = TARGET_CASE_COUNT,
    *,
    base_cases: Sequence[exp1511.ProductLineCase] | None = None,
) -> list[exp1511.ProductLineCase]:
    """Build deterministic product-line variants until the scale gate is met.

    The variants keep the same small feature-model grammar as Exp1511 but vary
    operation kind, include sets, and budgets.  Every emitted case is parsed
    back through the public Exp1511 parser and solved once, so malformed or
    infeasible generated cases never enter the benchmark pack.
    """

    bases = list(base_cases or exp1511.build_feature_model_cases())
    if target_count <= 0 or not bases:
        return []

    per_base = max(1, math.ceil(target_count / len(bases)))
    cases: list[exp1511.ProductLineCase] = []
    seen_case_ids: set[str] = set()
    for base_index, base in enumerate(bases):
        operations = _variant_operations(base, per_base)
        for variant_index, operation in enumerate(operations):
            case_id = f"plc-1540-{_case_slug(base.case_id)}-{variant_index:02d}"
            if case_id in seen_case_ids:  # pragma: no cover - generated IDs are unique.
                continue
            blueprint = render_blueprint(case_id, base.model, operation)
            try:
                case = exp1511.parse_blueprint(blueprint)
            except ValueError:  # pragma: no cover - render_blueprint emits parser-compatible text.
                continue
            if not exp1511.solve_case(case).feasible_exists:  # pragma: no cover - filtered variants.
                continue
            seen_case_ids.add(case_id)
            cases.append(case)
            if len(cases) >= target_count:
                return cases
        if base_index == len(bases) - 1 and len(cases) < target_count:  # pragma: no cover
            cases.extend(_overflow_variants(bases, seen_case_ids, target_count - len(cases)))
    return cases[:target_count]  # pragma: no cover - normal generator returns at scale gate.


def render_blueprint(
    case_id: str,
    model: exp1511.FeatureModel,
    operation: exp1511.AnalysisOperation,
) -> str:
    """Render a feature model back into the semi-formal Exp1511 grammar."""

    lines = [f"CASE: {case_id}", f"MODEL: {model.model_id}", "FEATURES:"]
    for feature in sorted(model.features):
        kind = "mandatory" if feature in model.mandatory else "optional"
        lines.append(
            f"- {feature} {kind} cost={model.costs[feature]} value={model.values[feature]}"
        )
    lines.append("REQUIRES:")
    for source, target in sorted(model.requires):
        lines.append(f"- {source} -> {target}")
    lines.append("EXCLUDES:")
    for left, right in sorted(model.excludes):
        lines.append(f"- {left} x {right}")
    op = operation.kind
    attrs: list[str] = []
    if operation.budget is not None:
        attrs.append(f"budget={operation.budget}")
    attrs.append(f"include={','.join(sorted(operation.include))}")
    lines.append(f"OPERATION: {op} {' '.join(attrs)}")
    return "\n".join(lines) + "\n"


def oracle_label_for_case(case: exp1511.ProductLineCase) -> JsonDict:
    """Return the reproducible oracle label used to compare manifest rows."""

    oracle = exp1511.solve_case(case)
    payload: JsonDict = {
        "case_id": case.case_id,
        "operation": {
            "kind": case.operation.kind,
            "budget": case.operation.budget,
            "include": sorted(case.operation.include),
        },
        "feasible_exists": oracle.feasible_exists,
        "feasible_count": oracle.feasible_count,
        "optimal_features": list(oracle.optimal_features),
        "optimal_cost": oracle.optimal_cost,
        "optimal_value": oracle.optimal_value,
    }
    payload["checksum"] = _checksum(payload)
    return payload


def oracle_label_snapshot(cases: Sequence[exp1511.ProductLineCase]) -> list[JsonDict]:
    """Return oracle labels for a case sequence without depending on row order side effects."""

    return [oracle_label_for_case(case) for case in cases]


def compile_product_line_answer_dfa(case: exp1511.ProductLineCase) -> ABSDFAMask:
    """Compile one oracle answer into the Exp1535 ABS-style exact JSON DFA."""

    return ABSDFAMask(exp1511.compliant_answer_for_case(case))


def build_staged_seed_rows(cases: Sequence[exp1511.ProductLineCase]) -> list[JsonDict]:
    """Create deterministic candidate rows that exercise each staged repair path."""

    rows: list[JsonDict] = []
    model_hf_id = MANDATED_MODEL_SPECS[0]
    for index, case in enumerate(cases):
        mode = ("syntax_failure", "feature_model_repair", "solver_repair", "automata_guided_oracle")[
            index % 4
        ]
        if mode == "syntax_failure":
            output = f"not-json answer for {case.case_id}"
            source = "deterministic_syntax_failure_seed"
        elif mode == "feature_model_repair":
            output = json.dumps(_feature_repair_payload(case), sort_keys=True)
            source = "deterministic_feature_model_seed"
        elif mode == "solver_repair":
            output = json.dumps(_solver_repair_payload(case), sort_keys=True)
            source = "deterministic_solver_repair_seed"
        else:
            output = compile_product_line_answer_dfa(case).generate()
            source = "automata_guided_abs_dfa"
        rows.append(
            {
                "case_id": case.case_id,
                "seed_mode": mode,
                "model_hf_id": model_hf_id,
                "model_name": "Qwen3.6-35B-A3B",
                "generation_source": source,
                "model_output": output,
                "elapsed_seconds": 0.0,
                "blocker": None,
            }
        )
    return rows


def evaluate_staged_case(
    case: exp1511.ProductLineCase,
    source_row: Mapping[str, Any],
) -> JsonDict:
    """Replay one product-line candidate through staged deterministic feedback."""

    replay_row: JsonDict = {
        "case_id": case.case_id,
        "model_hf_id": source_row.get("model_hf_id", MANDATED_MODEL_SPECS[0]),
        "model_name": source_row.get("model_name", "Qwen3.6-35B-A3B"),
        "generation_source": source_row.get("generation_source", "deterministic_seed"),
        "model_output": source_row.get("model_output", ""),
        "elapsed_seconds": source_row.get("elapsed_seconds", 0.0),
        "blocker": source_row.get("blocker"),
    }
    row = rescue1523.apply_staged_feedback(case, replay_row)
    selected = tuple(row["parse_result"]["selected_features"])
    evaluation = exp1511.evaluate_selection(case, selected)
    structural = exp1511.is_selection_feasible(case.model, selected)
    operational = exp1511.selection_satisfies_operation(case, selected)
    oracle_label = oracle_label_for_case(case)
    row.update(
        {
            "seed_mode": source_row.get("seed_mode", "live_sota_raw"),
            "oracle_label": oracle_label,
            "oracle_label_checksum": oracle_label["checksum"],
            "syntax_stage_pass": bool(row["parse_result"]["parse_ok"]),
            "feature_model_stage_pass": bool(structural.ok),
            "feasibility_stage_pass": bool(structural.ok and operational.ok),
            "oracle_agreement_stage_pass": bool(evaluation.oracle_agrees),
            "automata_constraints_used": source_row.get("generation_source")
            == "automata_guided_abs_dfa",
            "raw_model_output_excerpt": str(source_row.get("model_output", ""))[:500],
        }
    )
    return row


def summarize_staged_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute staged pass rates and final false-accept rate."""

    total = len(rows)
    if total == 0:
        return {
            "syntax_stage_pass_rate": 0.0,
            "feature_model_stage_pass_rate": 0.0,
            "feasibility_stage_pass_rate": 0.0,
            "oracle_agreement_rate": 0.0,
            "false_accept_rate": 0.0,
            "false_accept_count": 0,
            "stage_classification_counts": {},
        }
    false_accept_count = sum(
        1 for row in rows if bool(row.get("policy_result", {}).get("false_accept"))
    )
    counts: dict[str, int] = {}
    for row in rows:
        classification = str(row.get("oracle_result", {}).get("classification"))
        counts[classification] = counts.get(classification, 0) + 1
    return {
        "syntax_stage_pass_rate": _rate(rows, "syntax_stage_pass"),
        "feature_model_stage_pass_rate": _rate(rows, "feature_model_stage_pass"),
        "feasibility_stage_pass_rate": _rate(rows, "feasibility_stage_pass"),
        "oracle_agreement_rate": _rate(rows, "oracle_agreement_stage_pass"),
        "false_accept_rate": round(false_accept_count / total, 6),
        "false_accept_count": false_accept_count,
        "stage_classification_counts": counts,
    }


def decide_scale_readiness(
    *,
    cases_total: int,
    metrics: Mapping[str, Any],
    live_sota_model_inference_used: bool,
    focused_tests_passed: bool,
    blockers: Sequence[str],
) -> JsonDict:
    """Apply the scale readiness and retirement gate from REQ-BENCH-1540."""

    false_accept_rate = float(metrics.get("false_accept_rate", 0.0))
    stage_rates = [
        float(metrics.get("syntax_stage_pass_rate", 0.0)),
        float(metrics.get("feature_model_stage_pass_rate", 0.0)),
        float(metrics.get("feasibility_stage_pass_rate", 0.0)),
        float(metrics.get("oracle_agreement_rate", 0.0)),
    ]
    retirement_reason = ""
    if false_accept_rate > 0.0:
        retirement_reason = f"false_accept_rate exceeded zero: {false_accept_rate}"
    elif cases_total < TARGET_CASE_COUNT:
        retirement_reason = (
            f"scaled corpus below {TARGET_CASE_COUNT}-case gate: available={cases_total}"
        )
    elif not live_sota_model_inference_used:
        retirement_reason = "mandated live SOTA GGUF inference did not complete on benchmark prompts"
    elif any(rate < 1.0 for rate in stage_rates):
        retirement_reason = "one or more deterministic staged validators failed on scaled cases"
    elif blockers:
        retirement_reason = "; ".join(str(blocker) for blocker in blockers)

    ready = (
        cases_total >= TARGET_CASE_COUNT
        and live_sota_model_inference_used
        and focused_tests_passed
        and false_accept_rate == 0.0
        and all(rate == 1.0 for rate in stage_rates)
        and not blockers
    )
    retired = bool(retirement_reason)
    return {
        "product_line_scale_ready": bool(ready),
        "branch_retired": retired,
        "retirement_reason": retirement_reason,
    }


def load_rescue_context(
    *,
    artifact_path: Path | str = DEFAULT_RESCUE_ARTIFACT_PATH,
    manifest_path: Path | str = DEFAULT_RESCUE_MANIFEST_PATH,
) -> JsonDict:
    """Summarize the Exp1523 fixes that made the bounded rescue pass."""

    artifact = _read_json(Path(artifact_path))
    rows = _read_jsonl(Path(manifest_path)) if Path(manifest_path).exists() else []
    stage_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        for stage in row.get("stages", []):
            stage_name = str(stage.get("stage"))
            status = str(stage.get("status"))
            stage_counts.setdefault(stage_name, {})
            stage_counts[stage_name][status] = stage_counts[stage_name].get(status, 0) + 1
    return {
        "artifact_path": _display_path(artifact_path),
        "manifest_path": _display_path(manifest_path),
        "product_line_rescue_ready": bool(artifact.get("product_line_rescue_ready")),
        "bounded_cases": len(rows),
        "baseline_parse_rate": artifact.get("baseline_parse_rate"),
        "rescue_parse_rate": artifact.get("rescue_parse_rate"),
        "rescue_oracle_agreement_rate": artifact.get("rescue_oracle_agreement_rate"),
        "rescue_false_accept_rate": artifact.get("false_accept_rate"),
        "fix_summary": {
            "syntax": "parse failures are converted into schema JSON seeded from case contracts",
            "feature_model": "unknown features are removed, required features are added, and requires edges are closed",
            "feasibility": "the exhaustive solver replaces infeasible or suboptimal selections with an optimum",
            "policy": "final accept is true only when deterministic oracle_agrees is true",
        },
        "stage_counts": stage_counts,
    }


def run_benchmark(
    *,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    run_date: str = RUN_DATE,
    target_count: int = TARGET_CASE_COUNT,
    focused_tests_passed: bool = False,
    live_prompt_limit: int = 1,
    cached_pair_fn: CachedPairFn | None = None,
    live_collector_fn: LiveCollectorFn | None = None,
    gpu_probe_fn: Callable[[], JsonDict] | None = None,
) -> JsonDict:
    """Run the scaled staged benchmark and write the final artifact."""

    output = Path(output_path)
    manifest = Path(manifest_path)
    write_in_progress_artifact(output, manifest_path=manifest, run_date=run_date)

    rescue_context = load_rescue_context()
    cases = build_staged_product_line_cases(target_count=target_count)
    blockers: list[str] = []
    if len(cases) < target_count:  # pragma: no cover - normal generator reaches requested gate.
        blockers.append(f"scaled_case_corpus_smaller_than_requested:{len(cases)}<{target_count}")

    model_spec, cached_pair, model_blockers = _resolve_live_model(cached_pair_fn)
    blockers.extend(model_blockers)
    live_result: JsonDict = {"models_used": [], "rows": [], "blockers": []}
    if model_spec:
        collector = live_collector_fn or collect_live_sota_prompt_outputs
        live_result = collector(cases, model_spec, live_prompt_limit)
        blockers.extend(str(blocker) for blocker in live_result.get("blockers", []))

    source_rows = _merge_live_rows(
        cases,
        build_staged_seed_rows(cases),
        list(live_result.get("rows", [])),
    )
    staged_rows = [
        evaluate_staged_case(case, source_row)
        for case, source_row in zip(cases, source_rows, strict=True)
    ]
    _write_jsonl(manifest, staged_rows)

    metrics = summarize_staged_rows(staged_rows)
    live_used = any(
        row.get("generation_source") == "live_sota_llamacpp" and not row.get("blocker")
        for row in live_result.get("rows", [])
    )
    if not focused_tests_passed:
        blockers.append("focused_tests_not_passed")
    decision = decide_scale_readiness(
        cases_total=len(staged_rows),
        metrics=metrics,
        live_sota_model_inference_used=live_used,
        focused_tests_passed=focused_tests_passed,
        blockers=blockers,
    )
    ready = bool(decision["product_line_scale_ready"])
    retired = bool(decision["branch_retired"])
    artifact: JsonDict = {
        "status": "complete" if staged_rows else "blocked",
        "milestone": run_date,
        "schema_version": 1,
        "product_line_scale_ready": ready,
        "branch_retired": retired,
        "model_specs": list(MANDATED_MODEL_SPECS),
        "live_sota_model_inference_used": bool(live_used),
        "cases_total": len(staged_rows),
        "syntax_stage_pass_rate": metrics["syntax_stage_pass_rate"],
        "feature_model_stage_pass_rate": metrics["feature_model_stage_pass_rate"],
        "feasibility_stage_pass_rate": metrics["feasibility_stage_pass_rate"],
        "oracle_agreement_rate": metrics["oracle_agreement_rate"],
        "false_accept_rate": metrics["false_accept_rate"],
        "benchmark_manifest_path": _display_path(manifest),
        "retirement_reason": str(decision["retirement_reason"]),
        "focused_tests_passed": bool(focused_tests_passed),
        "honest_verdict": _honest_verdict(ready, retired, str(decision["retirement_reason"])),
        "false_accept_count": metrics["false_accept_count"],
        "stage_classification_counts": metrics["stage_classification_counts"],
        "target_cases_requested": target_count,
        "available_case_count": len(staged_rows),
        "automata_constraints_used": any(row["automata_constraints_used"] for row in staged_rows),
        "deterministic_validator_final_authority": True,
        "rescue_context": rescue_context,
        "models_used": list(live_result.get("models_used", [])),
        "live_prompt_limit": live_prompt_limit,
        "live_rows_evaluated": len(live_result.get("rows", [])),
        "live_inference_blockers": list(live_result.get("blockers", [])),
        "cached_sota_pair": cached_pair,
        "gpu_probe": (gpu_probe_fn or exp1511.probe_gpu)(),
        "blockers": list(dict.fromkeys(blockers)),
    }
    _write_json(output, artifact)
    return artifact


def collect_live_sota_prompt_outputs(
    cases: list[exp1511.ProductLineCase],
    model_spec: JsonDict,
    prompt_limit: int,
) -> JsonDict:  # pragma: no cover - host-specific GGUF runtime path.
    """Run one mandated local GGUF model on a bounded prefix of benchmark prompts."""

    if prompt_limit <= 0:
        return {
            "models_used": [],
            "rows": [],
            "blockers": ["live_prompt_limit_not_positive"],
        }
    try:
        from llama_cpp import Llama  # noqa: PLC0415
    except Exception as exc:
        return {
            "models_used": [],
            "rows": [],
            "blockers": [f"llama_cpp_import_failed:{type(exc).__name__}: {exc}"],
        }

    llm = None
    rows: list[JsonDict] = []
    try:
        llm = Llama(
            model_path=str(model_spec["model_path"]),
            n_gpu_layers=-1,
            main_gpu=int(model_spec.get("gpu", 0)),
            n_ctx=4096,
            seed=1540,
            verbose=False,
        )
        for case in cases[:prompt_limit]:
            started = time.monotonic()
            try:
                result = llm(
                    case.prompt,
                    max_tokens=160,
                    temperature=0.0,
                    top_p=1.0,
                    stop=["</s>", "<eos>"],
                    echo=False,
                )
                output_text = _completion_text(result)
                blocker = None if output_text.strip() else "empty_generation"
            except Exception as exc:
                output_text = ""
                blocker = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "case_id": case.case_id,
                    "model_hf_id": model_spec["hf_id"],
                    "model_name": model_spec.get("name") or model_spec["hf_id"],
                    "generation_source": "live_sota_llamacpp",
                    "model_output": output_text,
                    "elapsed_seconds": round(time.monotonic() - started, 6),
                    "blocker": blocker,
                }
            )
    except Exception as exc:
        return {
            "models_used": [],
            "rows": rows,
            "blockers": [f"live_sota_model_failed:{type(exc).__name__}: {exc}"],
        }
    finally:
        if hasattr(llm, "close"):
            llm.close()
    return {
        "models_used": [str(model_spec["hf_id"])] if any(not row.get("blocker") for row in rows) else [],
        "rows": rows,
        "blockers": [] if any(not row.get("blocker") for row in rows) else ["no_live_sota_generations"],
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """CLI entry point for manual and conductor runs."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--focused-tests-passed", action="store_true")
    parser.add_argument("--live-prompt-limit", type=int, default=1)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    artifact = run_benchmark(
        focused_tests_passed=args.focused_tests_passed,
        live_prompt_limit=args.live_prompt_limit,
    )
    print(
        "[exp1540] "
        f"ready={artifact['product_line_scale_ready']} "
        f"retired={artifact['branch_retired']} "
        f"cases={artifact['cases_total']} "
        f"oracle={artifact['oracle_agreement_rate']} "
        f"false_accept={artifact['false_accept_rate']}"
    )
    return 0


def _variant_operations(
    base: exp1511.ProductLineCase,
    limit: int,
) -> list[exp1511.AnalysisOperation]:
    selections = _structurally_feasible_selections(base.model)
    operations: list[exp1511.AnalysisOperation] = []
    seen: set[tuple[str, tuple[str, ...], int | None]] = set()
    for index, selection in enumerate(selections):
        optional = sorted(selection - base.model.mandatory)
        if index % 2 == 0:
            include = frozenset(optional[:1] or sorted(base.operation.include)[:1])
            operation = exp1511.AnalysisOperation(
                "max_value",
                include,
                exp1511.selection_cost(base.model, selection),
            )
        else:
            include = frozenset(optional[:2] or sorted(base.operation.include)[:1])
            operation = exp1511.AnalysisOperation("min_cost", include, None)
        key = (operation.kind, tuple(sorted(operation.include)), operation.budget)
        trial = exp1511.ProductLineCase(base.case_id, base.model, operation, "", "")
        if key in seen or not exp1511.solve_case(trial).feasible_exists:
            continue
        seen.add(key)
        operations.append(operation)
        if len(operations) >= limit:
            break
    return operations


def _overflow_variants(
    bases: Sequence[exp1511.ProductLineCase],
    seen_case_ids: set[str],
    needed: int,
) -> list[exp1511.ProductLineCase]:  # pragma: no cover - fallback for unexpectedly tiny corpora.
    rows: list[exp1511.ProductLineCase] = []
    for base, suffix in itertools.product(bases, range(100, 200)):
        operation = exp1511.AnalysisOperation(
            "max_value",
            frozenset(sorted(base.operation.include)[:1]),
            exp1511.selection_cost(base.model, exp1511.solve_case(base).optimal_features) + suffix - 100,
        )
        case_id = f"plc-1540-{_case_slug(base.case_id)}-{suffix:02d}"
        if case_id in seen_case_ids:
            continue
        case = exp1511.parse_blueprint(render_blueprint(case_id, base.model, operation))
        if exp1511.solve_case(case).feasible_exists:
            seen_case_ids.add(case_id)
            rows.append(case)
        if len(rows) >= needed:
            break
    return rows


def _structurally_feasible_selections(model: exp1511.FeatureModel) -> list[frozenset[str]]:
    selections: list[frozenset[str]] = []
    for subset_size in range(len(model.optional) + 1):
        for subset in itertools.combinations(sorted(model.optional), subset_size):
            selection = frozenset(model.mandatory | set(subset))
            if exp1511.is_selection_feasible(model, selection).ok:
                selections.append(selection)
    return sorted(
        selections,
        key=lambda selected: (
            exp1511.selection_cost(model, selected),
            -exp1511.selection_value(model, selected),
            tuple(sorted(selected)),
        ),
    )


def _feature_repair_payload(case: exp1511.ProductLineCase) -> JsonDict:
    selected = set(case.model.mandatory)
    selected.update(sorted(case.operation.include)[:1])
    for source, _target in case.model.requires:
        selected.add(source)
        break
    selected.add("BogusFeature")
    return {
        "selected_features": sorted(selected),
        "objective_cost": 0,
        "objective_value": 0,
        "verifier": {"accept": False},
    }


def _solver_repair_payload(case: exp1511.ProductLineCase) -> JsonDict:
    selected = _suboptimal_feasible_selection(case)
    return {
        "selected_features": sorted(selected),
        "objective_cost": exp1511.selection_cost(case.model, selected),
        "objective_value": exp1511.selection_value(case.model, selected),
        "verifier": {"accept": False},
    }


def _suboptimal_feasible_selection(case: exp1511.ProductLineCase) -> frozenset[str]:
    for selection in _structurally_feasible_selections(case.model):
        if not exp1511.selection_satisfies_operation(case, selection).ok:
            continue
        evaluation = exp1511.evaluate_selection(case, selection)
        if evaluation.feasible and not evaluation.oracle_agrees:
            return selection
    return frozenset(case.model.mandatory | case.operation.include)


def _merge_live_rows(
    cases: Sequence[exp1511.ProductLineCase],
    seed_rows: list[JsonDict],
    live_rows: list[JsonDict],
) -> list[JsonDict]:
    merged = [dict(row) for row in seed_rows]
    case_index = {case.case_id: index for index, case in enumerate(cases)}
    for live_row in live_rows:
        if live_row.get("blocker"):
            continue
        index = case_index.get(str(live_row.get("case_id")))
        if index is None:
            continue
        replacement = dict(live_row)
        replacement["seed_mode"] = "live_sota_raw"
        merged[index] = replacement
    return merged


def _resolve_live_model(cached_pair_fn: CachedPairFn | None) -> tuple[JsonDict | None, list[JsonDict], list[str]]:
    blockers: list[str] = []
    try:
        if cached_pair_fn is None:  # pragma: no cover - host cache path.
            from carnot.inference.sota_models import cached_sota_pair  # noqa: PLC0415

            cached_pair = cached_sota_pair(gpu_indices=(0, 1))
        else:
            cached_pair = cached_pair_fn(gpu_indices=(0, 1))
    except Exception as exc:
        cached_pair = None
        blockers.append(f"cached_sota_pair_error:{type(exc).__name__}: {exc}")
    pair = [dict(item) for item in cached_pair or []]
    for item in pair:
        if item.get("hf_id") in MANDATED_MODEL_SPECS and item.get("model_path"):
            return item, pair, blockers
    blockers.append("no_mandated_sota_gguf_runtime")
    return None, pair, blockers


def _completion_text(result: Any) -> str:  # pragma: no cover - live llama.cpp shape helper.
    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        return ""
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    text = first.get("text")
    return text if isinstance(text, str) else ""


def _honest_verdict(ready: bool, retired: bool, reason: str) -> str:
    if ready:
        return "complete: product-line staged benchmark scaled with zero false accepts"
    if retired:
        return f"complete_retired: product-line branch retired: {reason}"
    return "complete_blocked: product-line staged benchmark incomplete"


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return round(sum(1 for row in rows if bool(row.get(key))) / len(rows), 6)


def _checksum(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def _case_slug(case_id: str) -> str:
    slug = case_id.replace("plc-1511-", "")
    return "".join(ch if ch.isalnum() else "-" for ch in slug).strip("-")


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _display_path(path: Path | str) -> str:
    as_path = Path(path)
    try:
        return str(as_path.resolve().relative_to(exp1511._repo_root()))
    except ValueError:
        return str(as_path)


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover - exercised by conductor.
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_MANIFEST_PATH",
    "MANDATED_MODEL_SPECS",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_staged_product_line_cases",
    "build_staged_seed_rows",
    "compile_product_line_answer_dfa",
    "decide_scale_readiness",
    "evaluate_staged_case",
    "load_rescue_context",
    "main",
    "oracle_label_for_case",
    "oracle_label_snapshot",
    "render_blueprint",
    "run_benchmark",
    "summarize_staged_rows",
    "write_in_progress_artifact",
]
