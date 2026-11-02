"""Exp1550 SATQuest SOTA re-evaluation under the repaired oracle.

Spec: REQ-BENCH-1550, SCENARIO-BENCH-1550.

This workflow is deliberately conservative: model outputs are diagnostic
signals, while the Exp1549 repaired solver/proof oracle remains the only
acceptance authority.  Legacy small GGUFs are not substituted for headline
rows when the mandated local SOTA cache is unavailable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from carnot.eval import satquest_cnf_verifier_benchmark as exp1536
from carnot.eval import satquest_oracle_false_accept_repair as exp1549

JsonDict = dict[str, Any]
CollectModelOutputsFn = Callable[[JsonDict, list[exp1536.PromptCase]], JsonDict]
CachedPairFn = Callable[..., list[JsonDict] | None]
ResolverFn = Callable[[str], str | None]

RUN_DATE = "20260508"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1550_satquest_sota_reeval_zero_false_accepts.json")
DEFAULT_MANIFEST_PATH = Path("results/satquest_sota_reeval_zero_false_accepts_1550.jsonl")
DEFAULT_REPAIRED_ARTIFACT_PATH = exp1549.DEFAULT_ARTIFACT_PATH
DEFAULT_REPAIRED_MANIFEST_PATH = exp1549.DEFAULT_REPAIRED_CASE_MANIFEST_PATH

MODEL_SPECS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": MODEL_SPECS[0],
        "role": "headline_flagship_moe",
        "gpu": 0,
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": MODEL_SPECS[1],
        "role": "headline_flagship_dense",
        "gpu": 1,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": MODEL_SPECS[2],
        "role": "headline_middle_moe",
        "gpu": 1,
    },
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "milestone",
    "satquest_sota_reeval_ready",
    "model_specs",
    "live_sota_model_inference_used",
    "models_attempted",
    "cases_attempted",
    "formats_attempted",
    "solver_oracle_false_accepts",
    "false_accept_rate",
    "answer_accuracy",
    "witness_validity_rate",
    "energy_ranking_auc",
    "automata_or_format_constraints_used",
    "model_availability_blockers",
    "focused_tests_passed",
    "honest_verdict",
)

AUTOMATA_OR_FORMAT_CONSTRAINTS_USED: tuple[str, ...] = (
    "json_object_schema_prompt",
    "machine_dimacs_cnf",
    "symbolic_cnf",
    "narrative_natural_language_cnf",
    "repaired_oracle_assignment_or_unsat_certificate",
)

EXTRA_RAW_INSTANCES: tuple[tuple[str, int, tuple[tuple[int, ...], ...], str], ...] = (
    (
        "cnf-1550-sat-forced-choice",
        3,
        ((1, 2), (-1, 3), (-2, 3), (-3, 1)),
        "forced_choice_sat",
    ),
    (
        "cnf-1550-unsat-unit-implication",
        2,
        ((1,), (-1, 2), (-2,)),
        "unit_implication_unsat",
    ),
    (
        "cnf-1550-sat-xor-anchor",
        3,
        ((1, 2), (-1, -2), (3,), (-3, 1)),
        "xor_anchor_sat",
    ),
    (
        "cnf-1550-unsat-three-var-cover",
        3,
        (
            (1, 2, 3),
            (1, 2, -3),
            (1, -2, 3),
            (1, -2, -3),
            (-1, 2, 3),
            (-1, 2, -3),
            (-1, -2, 3),
            (-1, -2, -3),
        ),
        "truth_table_cover_unsat",
    ),
)

_ACCEPTED_REPAIRED_CLASSES = {
    "oracle_agreement_with_sat_witness",
    "oracle_agreement_with_unsat_certificate",
}


class RepairedGateError(ValueError):
    """Raised when Exp1549 did not prove SATQuest zero false accepts."""


@dataclass(frozen=True)
class RepairedSatQuestGate:
    """Compact summary of the Exp1549 repaired oracle gate."""

    ready: bool
    satquest_zero_false_accepts: bool
    rows_checked: int
    repaired_false_accepts: int
    artifact_path: str
    manifest_path: str


def load_repaired_satquest_gate(
    artifact_path: Path | str = DEFAULT_REPAIRED_ARTIFACT_PATH,
    manifest_path: Path | str = DEFAULT_REPAIRED_MANIFEST_PATH,
) -> RepairedSatQuestGate:
    """Load Exp1549 and require a zero-false-accept repaired SATQuest gate."""

    artifact = _read_json(Path(artifact_path))
    rows = _read_jsonl(Path(manifest_path))
    if artifact.get("satquest_zero_false_accepts") is not True:
        raise RepairedGateError("satquest_zero_false_accepts is not true")
    if artifact.get("satquest_oracle_repair_ready") is not True:
        raise RepairedGateError("satquest_oracle_repair_ready is not true")
    if int(artifact.get("solver_oracle_false_accepts_after") or 0) != 0:
        raise RepairedGateError("solver_oracle_false_accepts_after is not zero")
    repaired_false_accepts = sum(bool(row.get("repaired_false_accept")) for row in rows)
    if repaired_false_accepts:
        raise RepairedGateError("repaired manifest contains false accepts")
    return RepairedSatQuestGate(
        ready=True,
        satquest_zero_false_accepts=True,
        rows_checked=len(rows),
        repaired_false_accepts=repaired_false_accepts,
        artifact_path=_display_path(artifact_path),
        manifest_path=_display_path(manifest_path),
    )


def build_reeval_cnf_instances(run_date: str = RUN_DATE) -> list[exp1536.CNFInstance]:
    """Return the fixed Exp1550 CNF suite with at least ten bounded instances."""

    del run_date
    raw_instances = (*exp1536.RAW_INSTANCES, *EXTRA_RAW_INSTANCES)
    return [
        exp1536.CNFInstance(
            instance_id=instance_id,
            n_vars=n_vars,
            clauses=clauses,
            family=family,
            oracle=exp1536.solve_cnf(n_vars, clauses),
        )
        for instance_id, n_vars, clauses, family in raw_instances
    ]


def build_reeval_prompt_cases(min_cases: int = 30) -> list[exp1536.PromptCase]:
    """Return machine, symbolic, and narrative SATQuest prompts for Exp1550."""

    cases = exp1536.build_prompt_cases(build_reeval_cnf_instances())
    if len(cases) < min_cases:
        raise ValueError(f"Exp1550 requires at least {min_cases} prompt cases")
    return cases


def oracle_evidence_for_case(case: exp1536.PromptCase) -> exp1549.OracleEvidence:
    """Return checked SAT assignment or UNSAT contradiction evidence for a case."""

    return exp1549.solve_cnf_with_evidence(case.instance.n_vars, case.instance.clauses)


def aggregate_repaired_rows(rows: list[JsonDict]) -> JsonDict:
    """Compute diagnostic quality metrics without changing oracle authority."""

    if not rows:
        return {
            "solver_oracle_false_accepts": 0,
            "false_accept_rate": 0.0,
            "answer_accuracy": 0.0,
            "witness_validity_rate": 0.0,
            "energy_ranking_auc": None,
            "model_self_false_accepts": 0,
        }
    total = len(rows)
    false_accepts = sum(bool(row["repaired_false_accept"]) for row in rows)
    accepted = sum(bool(row["repaired_decision"]["accepted"]) for row in rows)
    evidence_valid = sum(bool(row["oracle_evidence_valid"]) for row in rows)
    self_false_accepts = sum(bool(row["verifier"]["self_verifier_false_accept"]) for row in rows)
    labels = [bool(row["baseline"]["correct"]) for row in rows]
    scores = [-float(row["baseline"]["energy"]) for row in rows]
    return {
        "solver_oracle_false_accepts": false_accepts,
        "false_accept_rate": round(false_accepts / total, 6),
        "answer_accuracy": round(accepted / total, 6),
        "witness_validity_rate": round(evidence_valid / total, 6),
        "energy_ranking_auc": _binary_auc(labels, scores),
        "model_self_false_accepts": self_false_accepts,
    }


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Write the required durable in-progress artifact before evaluation."""

    payload = _base_artifact(run_date=run_date)
    payload["status"] = "in_progress"
    payload["honest_verdict"] = "complete_blocked: in_progress_satquest_sota_reeval_initialized"
    _write_json(Path(output_path), payload)
    return payload


def run_reeval(
    *,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    repaired_artifact_path: Path | str = DEFAULT_REPAIRED_ARTIFACT_PATH,
    repaired_manifest_path: Path | str = DEFAULT_REPAIRED_MANIFEST_PATH,
    run_date: str = RUN_DATE,
    collect_model_outputs_fn: CollectModelOutputsFn | None = None,
    cached_pair_fn: CachedPairFn | None = None,
    resolver_fn: ResolverFn | None = None,
    gpu_probe_fn: Callable[[], JsonDict] | None = None,
    min_cases: int = 30,
    max_models: int = 1,
    focused_tests_passed: bool = False,
) -> JsonDict:
    """Run Exp1550 and write the terminal artifact plus JSONL row manifest."""

    output = Path(output_path)
    manifest = Path(manifest_path)
    write_in_progress_artifact(output, run_date=run_date)
    try:
        gate = load_repaired_satquest_gate(repaired_artifact_path, repaired_manifest_path)
    except RepairedGateError as exc:
        artifact = _terminal_artifact(
            status="blocked",
            run_date=run_date,
            blockers=[f"repaired_satquest_gate_not_ready:{exc}"],
            focused_tests_passed=focused_tests_passed,
            gpu_probe=(gpu_probe_fn or _probe_gpu)(),
        )
        _write_jsonl(manifest, [])
        _write_json(output, artifact)
        return artifact

    prompt_cases = build_reeval_prompt_cases(min_cases=min_cases)
    resolved_specs, cached_sota_details, blockers = resolve_mandated_model_specs(
        cached_pair_fn=cached_pair_fn,
        resolver_fn=resolver_fn,
    )
    runnable_specs = [spec for spec in resolved_specs if spec.get("model_path")]
    if not runnable_specs:
        artifact = _terminal_artifact(
            status="blocked",
            run_date=run_date,
            gate=gate,
            cached_sota_pair=cached_sota_details,
            blockers=blockers,
            focused_tests_passed=focused_tests_passed,
            gpu_probe=(gpu_probe_fn or _probe_gpu)(),
        )
        _write_jsonl(manifest, [])
        _write_json(output, artifact)
        return artifact

    collector = collect_model_outputs_fn or exp1536.collect_live_model_outputs
    rows: list[JsonDict] = []
    model_attempts: list[JsonDict] = []
    case_by_id = {case.case_id: case for case in prompt_cases}
    for spec in runnable_specs[:max_models]:
        collection = collector(spec, prompt_cases)
        summary = dict(collection.get("summary") or {})
        summary.setdefault("hf_id", spec.get("hf_id"))
        summary.setdefault("model_name", spec.get("name"))
        summary["cases_returned"] = len(collection.get("rows") or [])
        model_attempts.append(summary)
        for generation_row in collection.get("rows") or []:
            case = case_by_id.get(str(generation_row.get("case_id") or ""))
            if case is not None:
                rows.append(build_repaired_manifest_row(case, generation_row))

    _write_jsonl(manifest, rows)
    metrics = aggregate_repaired_rows(rows)
    live_used = any(
        row.get("generation_source") == "live_sota_llamacpp"
        and row.get("blocker") is None
        and row.get("model_hf_id") in MODEL_SPECS
        for row in rows
    )
    formats = sorted({str(row["format_name"]) for row in rows})
    ready = (
        gate.ready
        and live_used
        and len(rows) >= min_cases
        and set(formats) == set(exp1536.FORMAT_ORDER)
        and metrics["solver_oracle_false_accepts"] == 0
    )
    artifact = _terminal_artifact(
        status="complete" if ready else "blocked",
        run_date=run_date,
        gate=gate,
        rows=rows,
        model_attempts=model_attempts,
        cached_sota_pair=cached_sota_details,
        blockers=_dedupe([*blockers, *_collect_attempt_blockers(model_attempts)]),
        focused_tests_passed=focused_tests_passed,
        gpu_probe=(gpu_probe_fn or _probe_gpu)(),
        metrics=metrics,
        manifest_path=manifest,
        live_used=live_used,
        ready=ready,
    )
    _write_json(output, artifact)
    return artifact


def build_repaired_manifest_row(case: exp1536.PromptCase, generation_row: JsonDict) -> JsonDict:
    """Attach repaired-oracle evidence to one model generation row."""

    base = exp1536.build_manifest_row(case, generation_row)
    parsed = exp1536.parse_model_answer(str(generation_row.get("output_text") or ""))
    evidence = oracle_evidence_for_case(case)
    decision = exp1549.evaluate_candidate_with_evidence(
        case.instance.n_vars,
        case.instance.clauses,
        parsed.baseline,
        evidence=evidence,
    )
    evidence_valid = bool(evidence.assignment_witness_checked or evidence.unsat_certificate_checked)
    repaired_false_accept = bool(
        decision["accepted"] and decision["classification"] not in _ACCEPTED_REPAIRED_CLASSES
    )
    base.update(
        {
            "repaired_decision": decision,
            "repaired_false_accept": repaired_false_accept,
            "oracle_evidence": evidence.to_dict(),
            "oracle_evidence_valid": evidence_valid,
        }
    )
    return base


def resolve_mandated_model_specs(
    *,
    cached_pair_fn: CachedPairFn | None = None,
    resolver_fn: ResolverFn | None = None,
) -> tuple[list[JsonDict], list[JsonDict], list[str]]:
    """Resolve mandated SOTA GGUF paths, using cached_sota_pair first."""

    specs = [dict(spec) for spec in MANDATED_MODEL_SPECS]
    blockers: list[str] = []
    cached_details: list[JsonDict] = []
    try:
        pair = (
            cached_pair_fn(gpu_indices=(0, 1))
            if cached_pair_fn is not None
            else _cached_sota_pair(gpu_indices=(0, 1))
        )
    except Exception as exc:
        pair = None
        blockers.append(f"cached_sota_pair_error:{type(exc).__name__}: {exc}")
    cached_pair_missing = False
    if pair:
        cached_details = [dict(item) for item in pair]
        paths = {item.get("hf_id"): item.get("model_path") for item in pair if item.get("model_path")}
        for spec in specs:
            if spec["hf_id"] in paths:
                spec["model_path"] = paths[spec["hf_id"]]
    else:
        cached_pair_missing = True

    resolver = resolver_fn or _resolve_cached_gguf
    missing_specs: list[str] = []
    for spec in specs:
        if spec.get("model_path"):
            continue
        try:
            path = resolver(str(spec["hf_id"]))
        except Exception as exc:
            blockers.append(f"model_resolver_error:{spec['hf_id']}:{type(exc).__name__}: {exc}")
            path = None
        if path:
            spec["model_path"] = path
        else:
            missing_specs.append(str(spec["hf_id"]))
    if not any(spec.get("model_path") for spec in specs):
        if cached_pair_missing:
            blockers.append("cached_sota_pair_not_available")
        blockers.extend(f"mandated_model_not_cached:{hf_id}" for hf_id in missing_specs)
    return specs, cached_details, _dedupe(blockers)


def _terminal_artifact(
    *,
    status: str,
    run_date: str,
    gate: RepairedSatQuestGate | None = None,
    rows: list[JsonDict] | None = None,
    model_attempts: list[JsonDict] | None = None,
    cached_sota_pair: list[JsonDict] | None = None,
    blockers: list[str] | None = None,
    focused_tests_passed: bool,
    gpu_probe: JsonDict,
    metrics: JsonDict | None = None,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    live_used: bool = False,
    ready: bool = False,
) -> JsonDict:
    payload = _base_artifact(run_date=run_date)
    rows = rows or []
    metrics = metrics or aggregate_repaired_rows(rows)
    blockers = _dedupe(blockers or [])
    payload.update(
        {
            "status": status,
            "satquest_sota_reeval_ready": bool(ready),
            "live_sota_model_inference_used": bool(live_used),
            "models_attempted": model_attempts or [],
            "cases_attempted": len(rows),
            "formats_attempted": sorted({str(row["format_name"]) for row in rows}),
            "solver_oracle_false_accepts": metrics["solver_oracle_false_accepts"],
            "false_accept_rate": metrics["false_accept_rate"],
            "answer_accuracy": metrics["answer_accuracy"],
            "witness_validity_rate": metrics["witness_validity_rate"],
            "energy_ranking_auc": metrics["energy_ranking_auc"],
            "model_availability_blockers": blockers,
            "focused_tests_passed": bool(focused_tests_passed),
            "honest_verdict": (
                "complete: satquest_sota_reeval_zero_false_accepts_solver_grounded"
                if ready
                else "complete_blocked: satquest_sota_reeval_not_headline_ready"
            ),
            "manifest_path": _display_path(manifest_path),
            "repaired_gate": gate.__dict__ if gate is not None else None,
            "cached_sota_pair": cached_sota_pair or [],
            "gpu_probe": gpu_probe,
            "model_self_false_accepts": metrics["model_self_false_accepts"],
        }
    )
    return payload


def _base_artifact(*, run_date: str) -> JsonDict:
    return {
        "status": "blocked",
        "milestone": run_date,
        "satquest_sota_reeval_ready": False,
        "model_specs": list(MODEL_SPECS),
        "live_sota_model_inference_used": False,
        "models_attempted": [],
        "cases_attempted": 0,
        "formats_attempted": [],
        "solver_oracle_false_accepts": 0,
        "false_accept_rate": 0.0,
        "answer_accuracy": 0.0,
        "witness_validity_rate": 0.0,
        "energy_ranking_auc": None,
        "automata_or_format_constraints_used": list(AUTOMATA_OR_FORMAT_CONSTRAINTS_USED),
        "model_availability_blockers": [],
        "focused_tests_passed": False,
        "honest_verdict": "complete_blocked: satquest_sota_reeval_not_started",
    }


def _binary_auc(labels: list[bool], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores, strict=True) if label]
    negatives = [score for label, score in zip(labels, scores, strict=True) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for positive in positives:
        for negative in negatives:
            total += 1
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return round(wins / total, 6)


def _collect_attempt_blockers(model_attempts: Iterable[JsonDict]) -> list[str]:
    return [
        str(attempt["blocker"])
        for attempt in model_attempts
        if attempt.get("blocker") and attempt.get("blocker") != "not_attempted_runtime_budget"
    ]


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _cached_sota_pair(**kwargs: Any) -> list[JsonDict] | None:  # pragma: no cover - host cache path.
    from carnot.inference.sota_models import cached_sota_pair

    return cached_sota_pair(**kwargs)


def _resolve_cached_gguf(hf_id: str) -> str | None:  # pragma: no cover - host cache path.
    from carnot.inference.sota_models import resolve_cached_gguf

    return resolve_cached_gguf(hf_id)


def _probe_gpu() -> JsonDict:  # pragma: no cover - host-specific probe.
    return exp1536.probe_gpu()


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _display_path(path: Path | str) -> str:
    as_path = Path(path)
    try:
        return str(as_path.resolve().relative_to(_repo_root()))
    except ValueError:
        return str(as_path)


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_REPAIRED_ARTIFACT_PATH",
    "DEFAULT_REPAIRED_MANIFEST_PATH",
    "MODEL_SPECS",
    "MANDATED_MODEL_SPECS",
    "REQUIRED_ARTIFACT_FIELDS",
    "RepairedGateError",
    "RepairedSatQuestGate",
    "aggregate_repaired_rows",
    "build_reeval_cnf_instances",
    "build_reeval_prompt_cases",
    "build_repaired_manifest_row",
    "load_repaired_satquest_gate",
    "oracle_evidence_for_case",
    "resolve_mandated_model_specs",
    "run_reeval",
    "write_in_progress_artifact",
]
