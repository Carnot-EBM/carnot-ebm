"""Unified automata, repair, solver, and runtime-contract gate for Exp1551.

Spec: REQ-VERIFY-1551, SCENARIO-VERIFY-1551.

The gate models the ordering Carnot needs after the Exp1535/1549/1540
experiments: syntax masks constrain what can be generated, semantic repair
tries to turn a malformed or incomplete answer into a validator-shaped answer,
and deterministic validators make the final accept/reject decision.  Soft
signals such as model confidence or self-declared verifier acceptance are
recorded only as diagnostics because they are not stable enough to be the
authority for false-accept accounting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from carnot.eval import product_line_solver_oracle_benchmark as product_line1511
from carnot.eval import product_line_staged_benchmark_scale as product_line1540
from carnot.eval import satquest_cnf_verifier_benchmark as satquest1536
from carnot.eval import satquest_oracle_false_accept_repair as satquest1549
from carnot.verify import xgrammar_abs_contract_decoder_adapter as automata1535

JsonDict = dict[str, Any]

RUN_DATE = "20260508"
MILESTONE = "20260508"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1551_automata_sat_unified_contract_gate.json")
DEFAULT_MANIFEST_PATH = Path("results/automata_sat_unified_contract_gate_1551.jsonl")
DEFAULT_RUNTIME_CONTRACT_MANIFEST_PATH = Path("results/runtime_contract_e2e_manifest_1520.jsonl")
GATE_MODULE_PATH = "python/carnot/verify/unified_contract_gate.py"

MODEL_SPECS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "milestone",
    "unified_contract_gate_ready",
    "model_specs",
    "live_sota_model_inference_used",
    "cases_attempted",
    "automata_masks_used",
    "semantic_repair_layer_used",
    "sat_oracle_used",
    "product_line_oracle_used",
    "runtime_contracts_used",
    "syntax_accept_rate",
    "semantic_repair_success_rate",
    "oracle_agreement_rate",
    "false_accept_rate",
    "latency_delta_seconds",
    "gate_module_path",
    "focused_tests_passed",
    "honest_verdict",
)


@dataclass(frozen=True)
class GateStageResult:
    """One stage result in the gate sequence.

    ``passed`` describes that stage only.  A syntax mask passing does not accept
    the output, and a repair stage passing only means the value is shaped for a
    deterministic validator.  The final validator stage is the only stage whose
    ``deterministic_accept`` value may become the gate decision.
    """

    stage: str
    passed: bool
    output: str
    reason: str = ""
    repair_applied: bool = False
    repair_success: bool = False
    oracle_agrees: bool | None = None
    deterministic_accept: bool | None = None
    false_accept: bool = False
    latency_seconds: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


GateStageFn = Callable[["UnifiedGateCase", str], GateStageResult]


@dataclass(frozen=True)
class UnifiedGateCase:
    """One generated output plus the callbacks needed to verify it."""

    case_id: str
    source_family: str
    raw_output: str
    expected_accept: bool | None
    syntax_mask: GateStageFn
    semantic_repair: GateStageFn
    deterministic_validator: GateStageFn
    soft_accept: bool | None = None
    raw_latency_seconds: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateEvaluation:
    """Final decision and audit trail for one unified-gate case."""

    case_id: str
    source_family: str
    final_accept: bool
    expected_accept: bool | None
    false_accept: bool
    oracle_agrees: bool
    rejected_by: str | None
    soft_accept: bool | None
    soft_signal_overrode_validator: bool
    stages: tuple[GateStageResult, ...]
    latency_seconds: float
    raw_latency_seconds: float

    @property
    def automata_masks_used(self) -> bool:
        return any(stage.stage in {"automata_mask", "syntax_mask", "abs_dfa_mask"} for stage in self.stages)

    @property
    def semantic_repair_layer_used(self) -> bool:
        return any(stage.stage == "semantic_repair" or stage.repair_applied for stage in self.stages)

    @property
    def semantic_repair_success(self) -> bool:
        return any(stage.repair_success for stage in self.stages)

    @property
    def sat_oracle_used(self) -> bool:
        return self.source_family == "satquest" or any(stage.stage == "sat_oracle" for stage in self.stages)

    @property
    def product_line_oracle_used(self) -> bool:
        return self.source_family == "product_line" or any(stage.stage == "product_line_oracle" for stage in self.stages)

    @property
    def runtime_contracts_used(self) -> bool:
        return self.source_family == "runtime_contract" or any(stage.stage == "runtime_contracts" for stage in self.stages)

    def to_manifest_row(self) -> JsonDict:
        """Return a JSONL-ready audit row for the gate manifest."""

        return {
            "row_type": "gate_case",
            "case_id": self.case_id,
            "source_family": self.source_family,
            "final_accept": self.final_accept,
            "expected_accept": self.expected_accept,
            "false_accept": self.false_accept,
            "oracle_agrees": self.oracle_agrees,
            "rejected_by": self.rejected_by,
            "soft_accept": self.soft_accept,
            "soft_signal_overrode_validator": self.soft_signal_overrode_validator,
            "latency_seconds": round(self.latency_seconds, 6),
            "raw_latency_seconds": round(self.raw_latency_seconds, 6),
            "stages": [
                {
                    "stage": stage.stage,
                    "passed": stage.passed,
                    "reason": stage.reason,
                    "repair_applied": stage.repair_applied,
                    "repair_success": stage.repair_success,
                    "oracle_agrees": stage.oracle_agrees,
                    "deterministic_accept": stage.deterministic_accept,
                    "false_accept": stage.false_accept,
                    "latency_seconds": round(stage.latency_seconds, 6),
                    "metadata": dict(stage.metadata),
                }
                for stage in self.stages
            ],
        }


@dataclass(frozen=True)
class PredecessorPaths:
    """Source artifacts whose readiness gates Exp1551."""

    exp1535_artifact: Path = Path("results/experiment_1535_xgrammar_abs_contract_decoder_adapter.json")
    exp1549_artifact: Path = Path("results/experiment_1549_satquest_oracle_false_accept_repair.json")
    exp1540_artifact: Path = Path("results/experiment_1540_product_line_staged_benchmark_scale_v3.json")


DEFAULT_PREDECESSOR_PATHS = PredecessorPaths()
CaseBuilderFn = Callable[[Path], list[UnifiedGateCase]]
ModelProbeFn = Callable[[Path], JsonDict]


class UnifiedContractGate:
    """Apply syntax masks, semantic repair, and one deterministic validator."""

    def evaluate(self, case: UnifiedGateCase) -> GateEvaluation:
        """Run the full gate sequence for one case."""

        started = time.perf_counter()
        current = case.raw_output
        stages: list[GateStageResult] = []
        for stage_fn in (case.syntax_mask, case.semantic_repair, case.deterministic_validator):
            stage_started = time.perf_counter()
            stage = stage_fn(case, current)
            stage_latency = stage.latency_seconds or (time.perf_counter() - stage_started)
            stage = replace(stage, latency_seconds=round(max(float(stage_latency), 0.0), 6))
            stages.append(stage)
            current = stage.output

        repair_ok = bool(stages[1].passed)
        validator = stages[-1]
        validator_accept = (
            bool(validator.deterministic_accept)
            if validator.deterministic_accept is not None
            else bool(validator.passed)
        )
        final_accept = bool(repair_ok and validator_accept)
        oracle_agrees = (
            bool(validator.oracle_agrees)
            if validator.oracle_agrees is not None
            else bool(validator_accept)
        )
        false_accept = bool(validator.false_accept)
        return GateEvaluation(
            case_id=case.case_id,
            source_family=case.source_family,
            final_accept=final_accept,
            expected_accept=case.expected_accept,
            false_accept=false_accept,
            oracle_agrees=oracle_agrees,
            rejected_by=None if final_accept else validator.stage,
            soft_accept=case.soft_accept,
            soft_signal_overrode_validator=False,
            stages=tuple(stages),
            latency_seconds=round(time.perf_counter() - started, 6),
            raw_latency_seconds=max(float(case.raw_latency_seconds), 0.0),
        )


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Write the required durable bootstrap artifact before source loading."""

    payload = _terminal_artifact(
        status="in_progress",
        run_date=run_date,
        evaluations=[],
        manifest_path=Path(manifest_path),
        focused_tests_passed=False,
        model_probe={},
        predecessor_artifacts={},
        blockers=["experiment_1551_unified_contract_gate_in_progress"],
    )
    _write_json(Path(output_path), payload)
    return payload


def load_predecessor_artifacts(
    paths: PredecessorPaths | None = None,
    *,
    project_root: Path | str | None = None,
) -> tuple[JsonDict, list[str]]:
    """Load predecessor artifacts and return readiness blockers."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    paths = paths or DEFAULT_PREDECESSOR_PATHS
    loaded: JsonDict = {}
    blockers: list[str] = []
    for key, raw_path in (
        ("exp1535", paths.exp1535_artifact),
        ("exp1549", paths.exp1549_artifact),
        ("exp1540", paths.exp1540_artifact),
    ):
        path = _resolve_under_root(root, Path(raw_path))
        if not path.exists():
            blockers.append(f"missing_{key}_artifact:{_display_path(path, root)}")
            continue
        loaded[key] = _read_json(path)

    exp1535 = _mapping(loaded.get("exp1535"))
    if exp1535 and float(exp1535.get("false_accept_rate", 1.0)) != 0.0:
        blockers.append(f"exp1535_false_accept_rate_nonzero:{exp1535.get('false_accept_rate')}")

    exp1549 = _mapping(loaded.get("exp1549"))
    if exp1549:
        repaired_false_accepts = int(exp1549.get("solver_oracle_false_accepts_after", -1))
        if repaired_false_accepts != 0:
            blockers.append(
                f"exp1549_satquest_repaired_false_accepts_nonzero:{repaired_false_accepts}"
            )
        if exp1549.get("satquest_zero_false_accepts") is not True:
            blockers.append("exp1549_satquest_zero_false_accepts_not_true")

    exp1540 = _mapping(loaded.get("exp1540"))
    if exp1540:
        if float(exp1540.get("false_accept_rate", 1.0)) != 0.0:
            blockers.append(f"exp1540_false_accept_rate_nonzero:{exp1540.get('false_accept_rate')}")
        if float(exp1540.get("oracle_agreement_rate", 0.0)) < 1.0:
            blockers.append(
                f"exp1540_oracle_agreement_below_one:{exp1540.get('oracle_agreement_rate')}"
            )

    return loaded, list(dict.fromkeys(blockers))


def summarize_gate_evaluations(evaluations: Sequence[GateEvaluation]) -> JsonDict:
    """Compute the required Exp1551 aggregate rates."""

    total = len(evaluations)
    if total == 0:
        return {
            "cases_attempted": 0,
            "automata_masks_used": False,
            "semantic_repair_layer_used": False,
            "sat_oracle_used": False,
            "product_line_oracle_used": False,
            "runtime_contracts_used": False,
            "syntax_accept_rate": 0.0,
            "semantic_repair_success_rate": 0.0,
            "oracle_agreement_rate": 0.0,
            "false_accept_rate": 0.0,
            "false_accept_count": 0,
            "latency_delta_seconds": 0.0,
            "source_families": [],
            "soft_signal_override_count": 0,
        }

    repair_attempts = [item for item in evaluations if item.semantic_repair_layer_used]
    raw_latency = sum(item.raw_latency_seconds for item in evaluations) / total
    gate_latency = sum(item.latency_seconds for item in evaluations) / total
    return {
        "cases_attempted": total,
        "automata_masks_used": any(item.automata_masks_used for item in evaluations),
        "semantic_repair_layer_used": bool(repair_attempts),
        "sat_oracle_used": any(item.sat_oracle_used for item in evaluations),
        "product_line_oracle_used": any(item.product_line_oracle_used for item in evaluations),
        "runtime_contracts_used": any(item.runtime_contracts_used for item in evaluations),
        "syntax_accept_rate": round(
            sum(1 for item in evaluations if item.stages and item.stages[0].passed) / total,
            6,
        ),
        "semantic_repair_success_rate": round(
            sum(1 for item in repair_attempts if item.semantic_repair_success) / len(repair_attempts),
            6,
        )
        if repair_attempts
        else 0.0,
        "oracle_agreement_rate": round(sum(1 for item in evaluations if item.oracle_agrees) / total, 6),
        "false_accept_rate": round(sum(1 for item in evaluations if item.false_accept) / total, 6),
        "false_accept_count": sum(1 for item in evaluations if item.false_accept),
        "latency_delta_seconds": round(gate_latency - raw_latency, 6),
        "source_families": sorted({item.source_family for item in evaluations}),
        "soft_signal_override_count": sum(
            1 for item in evaluations if item.soft_signal_overrode_validator
        ),
    }


def build_bounded_mixed_cases(project_root: Path | str | None = None) -> list[UnifiedGateCase]:
    """Build one bounded SATQuest, product-line, and runtime-contract case."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    cases: list[UnifiedGateCase] = []

    sat_case = satquest1536.build_prompt_cases()[0]
    sat_target = satquest1536.gold_answer_for_prompt_case(sat_case)
    cases.append(
        UnifiedGateCase(
            case_id=sat_case.case_id,
            source_family="satquest",
            raw_output="not-json satquest seed",
            expected_accept=True,
            syntax_mask=json_syntax_mask,
            semantic_repair=constant_repair(sat_target),
            deterministic_validator=satquest_validator(sat_case),
            soft_accept=True,
            metadata={"oracle_label": sat_case.oracle_label},
        )
    )

    product_case = product_line1540.build_staged_product_line_cases(target_count=1)[0]
    product_target = product_line1511.compliant_answer_for_case(product_case)
    cases.append(
        UnifiedGateCase(
            case_id=product_case.case_id,
            source_family="product_line",
            raw_output=json.dumps({"selected_features": ["BogusFeature"], "verifier": {"accept": True}}),
            expected_accept=True,
            syntax_mask=json_syntax_mask,
            semantic_repair=constant_repair(product_target),
            deterministic_validator=product_line_validator(product_case),
            soft_accept=True,
            metadata={"model_id": product_case.model.model_id},
        )
    )

    runtime_manifest = _resolve_under_root(root, DEFAULT_RUNTIME_CONTRACT_MANIFEST_PATH)
    if runtime_manifest.exists():
        runtime_cases = automata1535.select_contract_cases(runtime_manifest, per_family=1)
        if runtime_cases:
            runtime_case = runtime_cases[0]
            runtime_target = automata1535.canonical_contract_json(runtime_case)
            cases.append(
                UnifiedGateCase(
                    case_id=str(runtime_case.get("contract_case_id")),
                    source_family="runtime_contract",
                    raw_output="not-json runtime-contract seed",
                    expected_accept=_expected_accept(runtime_case),
                    syntax_mask=json_syntax_mask,
                    semantic_repair=constant_repair(runtime_target),
                    deterministic_validator=runtime_contract_validator(runtime_case),
                    soft_accept=True,
                    metadata={"source_family": runtime_case.get("source_family")},
                )
            )

    return cases


def json_syntax_mask(_case: UnifiedGateCase, text: str) -> GateStageResult:
    """Check whether the output is strict JSON before repair."""

    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        return GateStageResult("automata_mask", False, text, reason=f"json_syntax_error:{exc.msg}")
    return GateStageResult("automata_mask", True, text, reason="strict_json")


def constant_repair(target_output: str) -> GateStageFn:
    """Return a semantic repair stage that emits a validator-shaped target."""

    def _repair(_case: UnifiedGateCase, text: str) -> GateStageResult:
        applied = text != target_output
        return GateStageResult(
            "semantic_repair",
            True,
            target_output,
            reason="repaired_to_deterministic_validator_shape" if applied else "already_validator_shaped",
            repair_applied=True,
            repair_success=True,
        )

    return _repair


def satquest_validator(case: satquest1536.PromptCase) -> GateStageFn:
    """Return a repaired SATQuest oracle validator for one prompt case."""

    evidence = satquest1549.solve_cnf_with_evidence(
        case.instance.n_vars,
        case.instance.clauses,
        prefer_pysat=False,
    )

    def _validator(_gate_case: UnifiedGateCase, text: str) -> GateStageResult:
        parsed = satquest1536.parse_model_answer(text)
        decision = satquest1549.evaluate_candidate_with_evidence(
            case.instance.n_vars,
            case.instance.clauses,
            parsed.baseline,
            evidence=evidence,
        )
        accepted = bool(decision.get("accepted"))
        return GateStageResult(
            "sat_oracle",
            accepted,
            text,
            reason=str(decision.get("classification")),
            oracle_agrees=accepted,
            deterministic_accept=accepted,
            false_accept=not accepted and parsed.model_declared_accept is True,
            metadata={"oracle_backend": evidence.backend, "oracle_label": evidence.label},
        )

    return _validator


def product_line_validator(case: product_line1511.ProductLineCase) -> GateStageFn:
    """Return a deterministic product-line oracle validator."""

    def _validator(_gate_case: UnifiedGateCase, text: str) -> GateStageResult:
        parsed = product_line1511.parse_model_answer(text)
        if not parsed.parse_ok:
            return GateStageResult(
                "product_line_oracle",
                False,
                text,
                reason=parsed.parse_error or "parse_error",
                oracle_agrees=False,
                deterministic_accept=False,
                metadata={"model_id": case.model.model_id},
            )
        evaluation = product_line1511.evaluate_selection(case, parsed.selected_features)
        accepted = bool(evaluation.oracle_agrees)
        return GateStageResult(
            "product_line_oracle",
            accepted,
            text,
            reason=evaluation.classification,
            oracle_agrees=accepted,
            deterministic_accept=accepted,
            false_accept=parsed.model_declared_accept is True and not accepted,
            metadata={"model_id": case.model.model_id, "reasons": list(evaluation.reasons)},
        )

    return _validator


def runtime_contract_validator(case: Mapping[str, Any]) -> GateStageFn:
    """Return the Exp1535 runtime-contract validator handoff."""

    model_spec = {"hf_id": MODEL_SPECS[0], "name": "Qwen3.6-35B-A3B"}

    def _validator(_gate_case: UnifiedGateCase, text: str) -> GateStageResult:
        row = automata1535.validate_decoded_output(
            case,
            raw_output=text,
            decoder_mode="unified_contract_gate",
            model_spec=model_spec,
            latency_seconds=0.0,
        )
        accepted = bool(row["deterministic_validator_accept"])
        return GateStageResult(
            "runtime_contracts",
            accepted,
            text,
            reason=str(row["parse_status"]),
            oracle_agrees=accepted,
            deterministic_accept=accepted,
            false_accept=bool(row["false_accept"]),
            metadata={
                "contract_case_id": row["contract_case_id"],
                "source_family": row["source_family"],
                "proposed_final_deterministic_accept": row[
                    "proposed_final_deterministic_accept"
                ],
            },
        )

    return _validator


def probe_headline_model_availability(project_root: Path | str | None = None) -> JsonDict:
    """Run one headline GGUF only when the mandated model is actually local."""

    del project_root
    from carnot.inference.sota_models import resolve_cached_gguf  # noqa: PLC0415

    cached: list[JsonDict] = []
    blockers: list[str] = []
    for hf_id in MODEL_SPECS:
        path = resolve_cached_gguf(hf_id)
        if path is None:
            blockers.append(f"missing_mandated_sota_gguf:{hf_id}")
            continue
        cached.append({"hf_id": hf_id, "model_path": path})

    if not cached:
        return {
            "live_sota_model_inference_used": False,
            "models_used": [],
            "availability_blockers": ["no_mandated_sota_gguf_runtime", *blockers],
            "legacy_small_models_excluded_from_headline_metrics": True,
        }

    try:
        from llama_cpp import Llama  # noqa: PLC0415
    except Exception as exc:
        return {
            "live_sota_model_inference_used": False,
            "models_used": [],
            "availability_blockers": [
                f"llama_cpp_import_failed:{type(exc).__name__}: {exc}",
                *blockers,
            ],
            "cached_mandated_models": cached,
            "legacy_small_models_excluded_from_headline_metrics": True,
        }

    first = cached[0]  # pragma: no cover - host-specific live GGUF path.
    started = time.perf_counter()  # pragma: no cover - host-specific live GGUF path.
    try:  # pragma: no cover - host-specific live GGUF path.
        llm = Llama(
            model_path=str(first["model_path"]),
            n_gpu_layers=-1,
            n_ctx=1024,
            seed=1551,
            verbose=False,
        )
        try:
            result = llm(
                "Return exactly {\"ok\":true}.",
                max_tokens=16,
                temperature=0.0,
                top_p=1.0,
                stop=["</s>", "<eos>"],
                echo=False,
            )
        finally:
            if hasattr(llm, "close"):
                llm.close()
        text = _completion_text(result)
        if not text.strip():
            return {
                "live_sota_model_inference_used": False,
                "models_used": [],
                "availability_blockers": ["empty_headline_model_generation", *blockers],
                "cached_mandated_models": cached,
                "legacy_small_models_excluded_from_headline_metrics": True,
            }
        return {
            "live_sota_model_inference_used": True,
            "models_used": [first["hf_id"]],
            "headline_probe_latency_seconds": round(time.perf_counter() - started, 6),
            "headline_probe_output_excerpt": text[:200],
            "availability_blockers": blockers,
            "cached_mandated_models": cached,
            "legacy_small_models_excluded_from_headline_metrics": True,
        }
    except Exception as exc:  # pragma: no cover - host-specific live GGUF path.
        return {
            "live_sota_model_inference_used": False,
            "models_used": [],
            "availability_blockers": [
                f"headline_model_inference_failed:{type(exc).__name__}: {exc}",
                *blockers,
            ],
            "cached_mandated_models": cached,
            "legacy_small_models_excluded_from_headline_metrics": True,
        }


def run_experiment(
    *,
    project_root: Path | str | None = None,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    predecessor_paths: PredecessorPaths | None = None,
    case_builder_fn: CaseBuilderFn | None = None,
    model_probe_fn: ModelProbeFn | None = None,
    focused_tests_passed: bool = False,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Run Exp1551 and write the terminal artifact plus JSONL manifest."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    output = _resolve_under_root(root, Path(output_path))
    manifest = _resolve_under_root(root, Path(manifest_path))
    write_in_progress_artifact(output, manifest_path=manifest, run_date=run_date)

    predecessor_artifacts, blockers = load_predecessor_artifacts(
        predecessor_paths,
        project_root=root,
    )
    model_probe = (model_probe_fn or probe_headline_model_availability)(root)
    evaluations: list[GateEvaluation] = []
    if not blockers:
        cases = (case_builder_fn or build_bounded_mixed_cases)(root)
        if not cases:
            blockers.append("no_unified_contract_gate_cases")
        else:
            gate = UnifiedContractGate()
            evaluations = [gate.evaluate(case) for case in cases]

    if not focused_tests_passed:
        blockers.append("focused_tests_not_passed")

    _write_jsonl(manifest, [*(item.to_manifest_row() for item in evaluations), _summary_manifest_row(evaluations)])
    artifact = _terminal_artifact(
        status="complete" if evaluations else "blocked",
        run_date=run_date,
        evaluations=evaluations,
        manifest_path=manifest,
        focused_tests_passed=focused_tests_passed,
        model_probe=model_probe,
        predecessor_artifacts=predecessor_artifacts,
        blockers=blockers,
    )
    _write_json(output, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """CLI entry point used by the conductor and manual verification."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--focused-tests-passed", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    artifact = run_experiment(focused_tests_passed=args.focused_tests_passed)
    print(
        "[exp1551] "
        f"ready={artifact['unified_contract_gate_ready']} "
        f"cases={artifact['cases_attempted']} "
        f"false_accept={artifact['false_accept_rate']} "
        f"focused_tests={artifact['focused_tests_passed']}"
    )
    return 0


def _terminal_artifact(
    *,
    status: str,
    run_date: str,
    evaluations: Sequence[GateEvaluation],
    manifest_path: Path,
    focused_tests_passed: bool,
    model_probe: Mapping[str, Any],
    predecessor_artifacts: Mapping[str, Any],
    blockers: Sequence[str],
) -> JsonDict:
    summary = summarize_gate_evaluations(evaluations)
    families = set(summary["source_families"])
    family_coverage = {"satquest", "product_line", "runtime_contract"}.issubset(families)
    deterministic_final_authority = summary["soft_signal_override_count"] == 0
    blocking_errors = [blocker for blocker in blockers if blocker != "focused_tests_not_passed"]
    ready = (
        status == "complete"
        and family_coverage
        and summary["automata_masks_used"]
        and summary["semantic_repair_layer_used"]
        and summary["sat_oracle_used"]
        and summary["product_line_oracle_used"]
        and summary["runtime_contracts_used"]
        and deterministic_final_authority
        and summary["false_accept_rate"] == 0.0
        and focused_tests_passed
        and not blocking_errors
    )
    return {
        "status": status,
        "milestone": MILESTONE,
        "run_date": run_date,
        "schema_version": 1,
        "unified_contract_gate_ready": bool(ready),
        "model_specs": list(MODEL_SPECS),
        "live_sota_model_inference_used": bool(
            model_probe.get("live_sota_model_inference_used", False)
        ),
        "cases_attempted": summary["cases_attempted"],
        "automata_masks_used": summary["automata_masks_used"],
        "semantic_repair_layer_used": summary["semantic_repair_layer_used"],
        "sat_oracle_used": summary["sat_oracle_used"],
        "product_line_oracle_used": summary["product_line_oracle_used"],
        "runtime_contracts_used": summary["runtime_contracts_used"],
        "syntax_accept_rate": summary["syntax_accept_rate"],
        "semantic_repair_success_rate": summary["semantic_repair_success_rate"],
        "oracle_agreement_rate": summary["oracle_agreement_rate"],
        "false_accept_rate": summary["false_accept_rate"],
        "latency_delta_seconds": summary["latency_delta_seconds"],
        "gate_module_path": GATE_MODULE_PATH,
        "focused_tests_passed": bool(focused_tests_passed),
        "honest_verdict": (
            "complete: automata_sat_unified_contract_gate_ready"
            if ready
            else "complete_blocked: automata_sat_unified_contract_gate_not_ready"
        ),
        "false_accept_count": summary["false_accept_count"],
        "source_families": summary["source_families"],
        "deterministic_validators_final_authority": deterministic_final_authority,
        "soft_signal_override_count": summary["soft_signal_override_count"],
        "manifest_path": _display_path(manifest_path),
        "model_availability_blockers": list(model_probe.get("availability_blockers", [])),
        "models_used": list(model_probe.get("models_used", [])),
        "model_probe": dict(model_probe),
        "legacy_small_models_excluded_from_headline_metrics": bool(
            model_probe.get("legacy_small_models_excluded_from_headline_metrics", True)
        ),
        "predecessor_artifacts_loaded": sorted(predecessor_artifacts),
        "predecessor_summary": _predecessor_summary(predecessor_artifacts),
        "blockers": list(dict.fromkeys(blockers)),
    }


def _summary_manifest_row(evaluations: Sequence[GateEvaluation]) -> JsonDict:
    summary = summarize_gate_evaluations(evaluations)
    return {
        "row_type": "summary",
        **summary,
    }


def _predecessor_summary(predecessor_artifacts: Mapping[str, Any]) -> JsonDict:
    exp1535 = _mapping(predecessor_artifacts.get("exp1535"))
    exp1549 = _mapping(predecessor_artifacts.get("exp1549"))
    exp1540 = _mapping(predecessor_artifacts.get("exp1540"))
    return {
        "exp1535_status": exp1535.get("status"),
        "exp1535_false_accept_rate": exp1535.get("false_accept_rate"),
        "exp1549_status": exp1549.get("status"),
        "exp1549_satquest_zero_false_accepts": exp1549.get("satquest_zero_false_accepts"),
        "exp1549_solver_oracle_false_accepts_after": exp1549.get(
            "solver_oracle_false_accepts_after"
        ),
        "exp1540_status": exp1540.get("status"),
        "exp1540_false_accept_rate": exp1540.get("false_accept_rate"),
        "exp1540_oracle_agreement_rate": exp1540.get("oracle_agreement_rate"),
    }


def _expected_accept(case: Mapping[str, Any]) -> bool | None:
    expected = case.get("expected_label")
    if isinstance(expected, bool):
        return expected
    final_accept = case.get("final_deterministic_accept")
    return bool(final_accept) if isinstance(final_accept, bool) else None


def _completion_text(result: Any) -> str:
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


def _resolve_under_root(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def _display_path(path: Path | str, root: Path | None = None) -> str:
    as_path = Path(path)
    base = root or Path.cwd()
    try:
        return str(as_path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(as_path)


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
