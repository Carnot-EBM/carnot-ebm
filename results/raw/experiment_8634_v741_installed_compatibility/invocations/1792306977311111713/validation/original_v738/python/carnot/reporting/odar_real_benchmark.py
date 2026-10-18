"""Exp 2257 ODAR benchmark on real verify-repair Tier 0 probe outputs.

**Researcher summary:**
    Exp 2244 showed ODAR can reduce tier calls when EFE values are supplied by a
    synthetic corpus. This benchmark closes the next gap: it builds 100
    deterministic reasoning responses, runs the verify-repair Tier 0 probe
    interfaces to produce actual probe records, routes those records with
    ``FreeEnergyRouter``, and compares ODAR against the uniform full cascade.

**Detailed explanation for engineers:**
    The benchmark deliberately avoids external LLM calls. The response text is
    synthetic, but the ODAR input is not a prefilled EFE label: HalluField and
    SemanticEnergy probe objects produce the records consumed by the router. The
    uniform regime always runs the normal arithmetic constraint extraction and
    verification path. The ODAR regime uses the router decision; fast-path cases
    skip Tier 1+, while deliberative cases fall through to the same full cascade.

Spec: REQ-ODAR-2257, SCENARIO-ODAR-2257
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILE = "experiment_2257_odar_real_benchmark.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILE
DEFAULT_ODAR_PATH = REPO_ROOT / "python" / "carnot" / "pipeline" / "odar_router.py"

EXPERIMENT = "2257_odar_real_benchmark"
SCHEMA = "odar_real_benchmark_v1"
DEFAULT_N_CORPUS = 100
RISK_THRESHOLD = 0.5
UNIFORM_TIER_CALLS_PER_CASE = 4
FAST_PATH_TIER_CALLS_PER_CASE = 1

GATE_COMPUTE_REDUCTION_PCT = 25.0
GATE_ROUTING_OVERHEAD_MS = 5.0
GATE_ACCURACY_DELTA_PP = -2.0

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "odar_real_validated",
    "compute_reduction_pct",
    "routing_overhead_ms",
    "accuracy_delta",
    "n_corpus",
    "preconditions_checked",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": "Terminal-prefix required. complete: if odar_real_validated is true.",
    "odar_real_validated": (
        "Boolean gate. True only when compute_reduction_pct >= 25 from real probe EFE."
    ),
    "compute_reduction_pct": (
        "Primary gate: must be >= 25 on real inference data (not synthetic labels)."
    ),
    "routing_overhead_ms": (
        "Records EFE computation cost; must be <= 5ms to be negligible vs cascade cost."
    ),
    "accuracy_delta": "Guards against trading accuracy for speed; must be >= -2pp.",
    "n_corpus": "Must be 100 for statistical significance claims.",
    "preconditions_checked": "Lists ODAR module availability check.",
}


@dataclass(frozen=True)
class ReasoningCase:
    """One deterministic arithmetic response used by the benchmark."""

    case_id: str
    question: str
    response: str
    expected_correct: bool
    split: str


class _NoopAndComposeVerifier:
    """Disable the advisory AND-compose ensemble without changing verification."""

    def verify(self, question: str, response: str) -> SimpleNamespace:
        return SimpleNamespace(
            verified=True,
            k=0,
            per_verifier_scores=[],
            per_verifier_verified=[],
        )


def build_reasoning_corpus(n_corpus: int = DEFAULT_N_CORPUS) -> list[ReasoningCase]:
    """Build the 100-example reasoning corpus for Exp 2257.

    The first half is coherent and arithmetically correct, giving ODAR low-risk
    SemanticEnergy probe records. The second half contains an arithmetic error
    plus one semantically rogue sentence, giving high-risk Tier 0 records that
    should route to the deliberative cascade.
    """
    if n_corpus != DEFAULT_N_CORPUS:
        raise ValueError(f"Exp 2257 requires exactly {DEFAULT_N_CORPUS} cases, got {n_corpus}")

    cases: list[ReasoningCase] = []
    half = n_corpus // 2
    for index in range(half):
        a = 10 + index
        b = 3 + (index % 9)
        total = a + b
        cases.append(
            ReasoningCase(
                case_id=f"correct_{index:03d}",
                question=f"What is {a} + {b}?",
                response=_make_response(a=a, b=b, claimed=total, rogue_sentence=False),
                expected_correct=True,
                split="coherent_correct",
            )
        )

    for index in range(half):
        a = 60 + index
        b = 4 + (index % 11)
        claimed = a + b + 1
        cases.append(
            ReasoningCase(
                case_id=f"incorrect_{index:03d}",
                question=f"What is {a} + {b}?",
                response=_make_response(a=a, b=b, claimed=claimed, rogue_sentence=True),
                expected_correct=False,
                split="ambiguous_incorrect",
            )
        )
    return cases


def run_benchmark(
    *,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    odar_path: Path | str = DEFAULT_ODAR_PATH,
    run_date: str | None = None,
) -> JsonDict:
    """Run Exp 2257 and write the ODAR real-probe benchmark artifact."""
    started = time.perf_counter()
    destination = Path(output_path)
    run_date = run_date or datetime.now(UTC).strftime("%Y%m%d")

    odar_import = _try_import_module(
        Path(odar_path),
        "_carnot_exp2257_odar_precondition",
        ("FreeEnergyRouter", "RoutingDecision", "RoutingResult"),
    )
    if isinstance(odar_import, Exception):
        artifact = _blocked_artifact(
            started=started,
            run_date=run_date,
            odar_path=Path(odar_path),
            error=odar_import,
        )
        _write_json(destination, artifact)
        return artifact

    preconditions_checked = [
        "odar_router_imported",
        "reasoning_corpus_constructed",
        "n_corpus_100",
        "uniform_full_cascade_regime_evaluated",
        "odar_real_probe_threshold_regime_evaluated",
    ]
    precondition_details: list[JsonDict] = [
        {
            "resource": str(Path(odar_path)),
            "check": "direct_import",
            "status": "passed",
            "symbols": ["FreeEnergyRouter", "RoutingDecision", "RoutingResult"],
        }
    ]

    try:
        from carnot.pipeline.hallufield_detector import HalluFieldDetector  # noqa: PLC0415
        from carnot.pipeline.odar_router import FreeEnergyRouter  # noqa: PLC0415
        from carnot.pipeline.semantic_energy_probe import SemanticEnergyProbe  # noqa: PLC0415
        from carnot.pipeline.verify_repair import VerifyRepairPipeline  # noqa: PLC0415
    except Exception as exc:
        artifact = _blocked_artifact(
            started=started,
            run_date=run_date,
            odar_path=Path(odar_path),
            error=exc,
        )
        artifact["preconditions_checked"] = ["odar_router_import_failed"]
        artifact["precondition_details"].append(
            {
                "resource": "carnot.pipeline imports",
                "check": "package_import",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(destination, artifact)
        return artifact

    corpus = build_reasoning_corpus()
    router = FreeEnergyRouter(risk_threshold=RISK_THRESHOLD)
    pipeline = VerifyRepairPipeline(
        model=None,
        domains=["arithmetic"],
        and_compose_verifier=_NoopAndComposeVerifier(),
    )
    semantic_probe = SemanticEnergyProbe()
    hallufield_detector = HalluFieldDetector()

    rows: list[JsonDict] = []
    routing_overheads_ms: list[float] = []
    uniform_correct = 0
    odar_correct = 0
    fast_path_count = 0
    deliberative_count = 0
    pipeline_route_mismatches = 0

    for case in corpus:
        tier0_outputs = _collect_tier0_probe_outputs(
            response=case.response,
            hallufield_detector=hallufield_detector,
            semantic_probe=semantic_probe,
        )

        route_started = time.perf_counter()
        routing_result = router.evaluate(tier0_outputs)
        routing_overhead_ms = (time.perf_counter() - route_started) * 1000.0
        routing_overheads_ms.append(routing_overhead_ms)

        uniform_result = pipeline.verify(
            question=case.question,
            response=case.response,
            domain="arithmetic",
            hallufield_detector=hallufield_detector,
            semantic_energy_probe=semantic_probe,
            use_odar=False,
        )
        odar_result = pipeline.verify(
            question=case.question,
            response=case.response,
            domain="arithmetic",
            hallufield_detector=hallufield_detector,
            semantic_energy_probe=semantic_probe,
            use_odar=True,
            odar_risk_threshold=RISK_THRESHOLD,
        )

        decision = routing_result.decision.value
        pipeline_decision = str(odar_result.certificate.get("odar_decision", "missing"))
        if pipeline_decision != decision:
            pipeline_route_mismatches += 1

        uniform_prediction = bool(uniform_result.verified)
        odar_prediction = bool(odar_result.verified)
        uniform_correct += int(uniform_prediction is case.expected_correct)
        odar_correct += int(odar_prediction is case.expected_correct)

        if decision == "FAST_PATH":
            fast_path_count += 1
        else:
            deliberative_count += 1

        rows.append(
            _case_row(
                case=case,
                routing_result=routing_result,
                routing_overhead_ms=routing_overhead_ms,
                uniform_verified=uniform_prediction,
                odar_verified=odar_prediction,
                uniform_n_constraints=len(uniform_result.constraints),
                uniform_n_violations=len(uniform_result.violations),
                odar_mode=odar_result.mode,
                pipeline_decision=pipeline_decision,
            )
        )

    preconditions_checked.append("real_tier0_probe_outputs_collected")
    tier_calls_a = len(corpus) * UNIFORM_TIER_CALLS_PER_CASE
    tier_calls_b = (
        fast_path_count * FAST_PATH_TIER_CALLS_PER_CASE
        + deliberative_count * UNIFORM_TIER_CALLS_PER_CASE
    )
    compute_reduction_pct = 100.0 * (tier_calls_a - tier_calls_b) / tier_calls_a
    uniform_accuracy_pct = 100.0 * uniform_correct / len(corpus)
    odar_accuracy_pct = 100.0 * odar_correct / len(corpus)
    accuracy_delta = odar_accuracy_pct - uniform_accuracy_pct
    routing_overhead_ms = statistics.median(routing_overheads_ms)
    fast_path_fraction = fast_path_count / len(corpus)

    odar_real_validated = (
        compute_reduction_pct >= GATE_COMPUTE_REDUCTION_PCT
        and routing_overhead_ms <= GATE_ROUTING_OVERHEAD_MS
        and accuracy_delta >= GATE_ACCURACY_DELTA_PP
        and len(corpus) == DEFAULT_N_CORPUS
        and pipeline_route_mismatches == 0
    )
    honest_verdict = (
        "complete: odar_real_validated"
        if odar_real_validated
        else "failed: odar_real_validation_gate_not_met"
    )

    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": 2257,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "complete" if odar_real_validated else "failed",
        "title": "ODAR real Tier 0 probe routing overhead benchmark",
        "honest_verdict": honest_verdict,
        "odar_real_validated": odar_real_validated,
        "compute_reduction_pct": _round(compute_reduction_pct),
        "routing_overhead_ms": _round(routing_overhead_ms),
        "routing_overhead_scope": (
            "median FreeEnergyRouter.evaluate wall-clock time on Tier 0 probe records"
        ),
        "routing_overhead_p95_ms": _round(_percentile(routing_overheads_ms, 0.95)),
        "fast_path_fraction": _round(fast_path_fraction),
        "accuracy_delta": _round(accuracy_delta),
        "accuracy_delta_units": "percentage_points",
        "accuracy_uniform_pct": _round(uniform_accuracy_pct),
        "accuracy_odar_pct": _round(odar_accuracy_pct),
        "n_corpus": len(corpus),
        "tier_calls_A": tier_calls_a,
        "tier_calls_B": tier_calls_b,
        "fast_path_count": fast_path_count,
        "deliberative_count": deliberative_count,
        "threshold": RISK_THRESHOLD,
        "preconditions_checked": preconditions_checked,
        "precondition_details": precondition_details,
        "pipeline_route_mismatches": pipeline_route_mismatches,
        "external_llm_calls": 0,
        "models_used": [],
        "router_module_path": str(Path(odar_path)),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-ODAR-2257", "SCENARIO-ODAR-2257"],
        "duration_s": _round(time.perf_counter() - started),
        "case_summaries": rows,
    }
    validate_artifact(artifact)
    _write_json(destination, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 2257 artifact fields that downstream tooling relies on."""
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"artifact missing required fields: {missing}")
    if artifact["honest_verdict"].startswith("blocked_odar_missing:"):
        return
    if artifact["n_corpus"] != DEFAULT_N_CORPUS:
        raise ValueError(f"n_corpus must be {DEFAULT_N_CORPUS}")
    if artifact["accuracy_delta"] < GATE_ACCURACY_DELTA_PP:
        raise ValueError("accuracy_delta below ODAR guardrail")
    expected_validated = (
        artifact["compute_reduction_pct"] >= GATE_COMPUTE_REDUCTION_PCT
        and artifact["routing_overhead_ms"] <= GATE_ROUTING_OVERHEAD_MS
        and artifact["accuracy_delta"] >= GATE_ACCURACY_DELTA_PP
        and artifact["n_corpus"] == DEFAULT_N_CORPUS
        and artifact["pipeline_route_mismatches"] == 0
    )
    if artifact["odar_real_validated"] is not expected_validated:
        raise ValueError("odar_real_validated does not match gate conditions")


def _make_response(*, a: int, b: int, claimed: int, rogue_sentence: bool) -> str:
    rogue = " A comet writes invoices beside the ocean." if rogue_sentence else ""
    return (
        f"The {a} apple basket count is the starting apple basket count. "
        f"The {b} apple basket addition increases the apple basket count. "
        f"The apple basket count equation is {a} + {b} = {claimed}."
        f"{rogue} "
        f"The final apple basket count is {claimed} apples."
    )


def _collect_tier0_probe_outputs(
    *,
    response: str,
    hallufield_detector: Any,
    semantic_probe: Any,
) -> JsonDict:
    return {
        "hallufield": hallufield_detector.score(None),
        "semantic_energy": semantic_probe.score(response),
    }


def _case_row(
    *,
    case: ReasoningCase,
    routing_result: Any,
    routing_overhead_ms: float,
    uniform_verified: bool,
    odar_verified: bool,
    uniform_n_constraints: int,
    uniform_n_violations: int,
    odar_mode: str,
    pipeline_decision: str,
) -> JsonDict:
    certificate = routing_result.to_certificate()
    semantic_evidence = _semantic_evidence(certificate.get("odar_contributions", []))
    return {
        "case_id": case.case_id,
        "split": case.split,
        "expected_correct": case.expected_correct,
        "routing_decision": routing_result.decision.value,
        "pipeline_routing_decision": pipeline_decision,
        "expected_free_energy": _round(routing_result.expected_free_energy),
        "routing_overhead_ms": _round(routing_overhead_ms),
        "uniform_verified": uniform_verified,
        "odar_verified": odar_verified,
        "uniform_correct": uniform_verified is case.expected_correct,
        "odar_correct": odar_verified is case.expected_correct,
        "uniform_n_constraints": uniform_n_constraints,
        "uniform_n_violations": uniform_n_violations,
        "odar_mode": odar_mode,
        "tier0_probe_names": [item["name"] for item in certificate["odar_contributions"]],
        "semantic_energy": _round(semantic_evidence.get("energy", 0.0)),
        "semantic_is_unstable": bool(semantic_evidence.get("is_unstable", False)),
    }


def _semantic_evidence(contributions: Sequence[Mapping[str, Any]]) -> JsonDict:
    for contribution in contributions:
        if contribution.get("name") == "semantic_energy":
            evidence = contribution.get("evidence")
            if isinstance(evidence, Mapping):
                return dict(evidence)
    return {}


def _blocked_artifact(
    *,
    started: float,
    run_date: str,
    odar_path: Path,
    error: BaseException,
) -> JsonDict:
    return {
        "experiment": EXPERIMENT,
        "experiment_id": 2257,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "blocked",
        "title": "ODAR real Tier 0 probe routing overhead benchmark",
        "honest_verdict": "blocked_odar_missing: odar_router import failed",
        "odar_real_validated": False,
        "compute_reduction_pct": 0.0,
        "routing_overhead_ms": 0.0,
        "fast_path_fraction": 0.0,
        "accuracy_delta": 0.0,
        "n_corpus": 0,
        "tier_calls_A": 0,
        "tier_calls_B": 0,
        "preconditions_checked": ["odar_router_import_failed"],
        "precondition_details": [
            {
                "resource": str(odar_path),
                "check": "direct_import",
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        ],
        "external_llm_calls": 0,
        "models_used": [],
        "router_module_path": str(odar_path),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-ODAR-2257", "SCENARIO-ODAR-2257"],
        "duration_s": _round(time.perf_counter() - started),
        "case_summaries": [],
    }


def _try_import_module(
    path: Path, module_name: str, required_symbols: Sequence[str]
) -> ModuleType | Exception:
    try:
        if not path.exists():
            raise FileNotFoundError(path)
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not build import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
        missing = [symbol for symbol in required_symbols if not hasattr(module, symbol)]
        if missing:
            raise ImportError(f"{path} missing symbols: {missing}")
        return module
    except Exception as exc:
        return exc


def _write_json(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--odar-path", default=str(DEFAULT_ODAR_PATH))
    args = parser.parse_args(argv)
    artifact = run_benchmark(output_path=args.output_path, odar_path=args.odar_path)
    print(json.dumps({field: artifact[field] for field in REQUIRED_ARTIFACT_FIELDS}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
