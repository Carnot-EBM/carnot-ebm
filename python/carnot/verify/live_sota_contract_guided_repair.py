"""Live local-SOTA contract-guided repair adapter for Exp 1521.

Spec: REQ-VERIFY-1521, SCENARIO-VERIFY-1521.

The experiment compares three ways to answer a contract failure: unconstrained
baseline prose, grammar-only structured output, and draft-conditioned structured
output.  The model text is never trusted directly.  Each generated response is
parsed into the same contract-case shape used by Exp 1520, then the Exp 1520
false-accept ledger computes the soundness outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from carnot.verify import runtime_contract_e2e_harness as runtime_contracts

JsonDict = dict[str, Any]
GeneratorFn = Callable[[str, JsonDict, str, JsonDict], str]
CachedPairFn = Callable[..., list[JsonDict] | None]
ResolverFn = Callable[[str], str | None]
GpuProbeFn = Callable[[], JsonDict]

RUN_DATE = "20260508"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1521_live_sota_contract_guided_repair_v1.json")
DEFAULT_SOURCE_MANIFEST_PATH = Path("results/runtime_contract_e2e_manifest_1520.jsonl")
DEFAULT_REPAIR_MANIFEST_PATH = Path("results/live_contract_guided_repair_1521.jsonl")

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe_primary_contract_guided_repair",
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "flagship_dense_secondary_contract_guided_repair",
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "middle_moe_secondary_contract_guided_repair",
    },
)
MANDATED_HF_IDS = frozenset(spec["hf_id"] for spec in MANDATED_MODEL_SPECS)
REPAIR_MODES: tuple[str, ...] = ("baseline", "grammar_only", "draft_conditioned")

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "live_sota_model_inference_used",
    "contract_guided_repair_ready",
    "e2e_cases_loaded",
    "repair_cases_attempted",
    "baseline_accept_rate",
    "grammar_only_accept_rate",
    "draft_conditioned_accept_rate",
    "repair_accept_rate_delta",
    "false_accept_count",
    "false_accept_rate",
    "models_used",
    "gpu_probe",
    "repair_manifest_path",
    "blockers",
    "honest_verdict",
)


def select_repair_cases(manifest_path: Path | str, *, limit: int = 2) -> list[JsonDict]:
    """Load bounded deterministic reject or marginal cases from Exp 1520."""

    selected: list[JsonDict] = []
    for row in _read_jsonl(Path(manifest_path)):
        if row.get("row_type") != "contract_case":
            continue
        expected = row.get("expected_label")
        if not isinstance(expected, bool):
            continue
        final_accept = row.get("final_deterministic_accept")
        is_contract_failing = expected is False
        is_contract_marginal = final_accept is False
        if not is_contract_failing and not is_contract_marginal:
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def validate_repair_output(
    case: Mapping[str, Any],
    *,
    model_spec: Mapping[str, Any],
    mode: str,
    raw_output: str,
) -> JsonDict:
    """Validate one generated response using Exp 1520 contract-case semantics."""

    parsed = _extract_last_json_object(raw_output)
    expected = case.get("expected_label")
    expected_bool = expected if isinstance(expected, bool) else None
    parse_status = "ok"
    proposed_accept = False
    case_id_matches = False

    if parsed is None:
        parse_status = "no_json_object"
    else:
        case_id_matches = parsed.get("contract_case_id") == case.get("contract_case_id")
        decision = parsed.get("final_deterministic_decision")
        accept_value = parsed.get("final_deterministic_accept")
        if not case_id_matches:
            parse_status = "contract_case_id_mismatch"
        elif isinstance(decision, str) and decision.lower() in {"accept", "reject"}:
            proposed_accept = decision.lower() == "accept"
        elif isinstance(accept_value, bool):
            proposed_accept = accept_value
        else:
            parse_status = "missing_final_decision"

    structurally_valid = parse_status == "ok"
    validation_row = _validation_contract_case(case, proposed_accept)
    ledger = runtime_contracts.compute_false_accept_ledger([validation_row])
    false_accept = bool(ledger["false_accept_count"])
    deterministic_accept = structurally_valid and expected_bool is not None and proposed_accept == expected_bool
    repair_outcome = _repair_outcome(
        structurally_valid=structurally_valid,
        parse_status=parse_status,
        false_accept=false_accept,
        expected_label=expected_bool,
        proposed_accept=proposed_accept,
    )

    return {
        "row_type": "repair_result",
        "contract_case_id": case.get("contract_case_id"),
        "prompt_or_case_id": case.get("prompt_or_case_id"),
        "source_family": case.get("source_family"),
        "model_hf_id": model_spec.get("hf_id"),
        "model_name": model_spec.get("name") or model_spec.get("hf_id"),
        "mode": mode,
        "raw_output_sha256": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
        "raw_output_excerpt": raw_output[:500],
        "parsed_contract_output": parsed or {},
        "parse_status": parse_status,
        "case_id_matches": bool(case_id_matches),
        "expected_label": expected_bool,
        "proposed_final_deterministic_accept": bool(proposed_accept),
        "deterministic_validator_accept": bool(deterministic_accept),
        "false_accept": false_accept,
        "repair_outcome": repair_outcome,
        "contract_validation_row": validation_row,
    }


def summarize_repair_rows(rows: Iterable[Mapping[str, Any]]) -> JsonDict:
    """Compute accept-rate and false-accept metrics from repair result rows."""

    row_list = list(rows)
    by_mode: dict[str, list[Mapping[str, Any]]] = {
        mode: [row for row in row_list if row.get("mode") == mode] for mode in REPAIR_MODES
    }
    rates = {
        mode: _rate(
            sum(1 for row in mode_rows if row.get("repair_outcome") == "accepted"),
            len(mode_rows),
        )
        for mode, mode_rows in by_mode.items()
    }
    validation_rows = [
        row["contract_validation_row"]
        for row in row_list
        if isinstance(row.get("contract_validation_row"), dict)
    ]
    ledger = runtime_contracts.compute_false_accept_ledger(validation_rows)
    return {
        "baseline_accept_rate": rates["baseline"],
        "grammar_only_accept_rate": rates["grammar_only"],
        "draft_conditioned_accept_rate": rates["draft_conditioned"],
        "repair_accept_rate_delta": None
        if rates["draft_conditioned"] is None or rates["grammar_only"] is None
        else round(rates["draft_conditioned"] - rates["grammar_only"], 6),
        "false_accept_count": ledger["false_accept_count"],
        "false_accept_rate": ledger["false_accept_rate"],
        "explicit_label_count": ledger["explicit_label_count"],
        "explicit_reject_count": ledger["explicit_reject_count"],
    }


def run_experiment(
    *,
    project_root: Path | str | None = None,
    run_date: str = RUN_DATE,
    source_manifest_path: Path | str = DEFAULT_SOURCE_MANIFEST_PATH,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    repair_manifest_path: Path | str = DEFAULT_REPAIR_MANIFEST_PATH,
    cached_pair_fn: CachedPairFn | None = None,
    resolver_fn: ResolverFn | None = None,
    generator_fn: GeneratorFn | None = None,
    gpu_probe_fn: GpuProbeFn | None = None,
    case_limit: int = 2,
    max_models: int = 1,
) -> JsonDict:
    """Run Exp 1521 and write both the terminal artifact and repair manifest."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    output = _resolve_under_root(root, Path(output_path))
    repair_manifest = _resolve_under_root(root, Path(repair_manifest_path))
    source_manifest = _resolve_under_root(root, Path(source_manifest_path))
    _write_json(output, _in_progress_artifact(run_date=run_date, repair_manifest=repair_manifest))
    pair_resolver = cached_pair_fn or _cached_sota_pair
    gguf_resolver = resolver_fn or _resolve_cached_gguf
    gpu_probe = gpu_probe_fn or _probe_gpu_state

    blockers: list[str] = []
    cases = select_repair_cases(source_manifest, limit=case_limit) if source_manifest.exists() else []
    if not source_manifest.exists():
        blockers.append(f"missing_runtime_contract_manifest:{source_manifest}")
    if not cases and not blockers:
        blockers.append("no_deterministic_contract_failing_or_marginal_cases")

    models = _resolve_runtime_models(pair_resolver, gguf_resolver, max_models=max_models)
    if not models:
        blockers.append("no_mandated_sota_gguf_runtime")
        _write_jsonl(repair_manifest, [])
        artifact = _terminal_artifact(
            status="blocked",
            run_date=run_date,
            e2e_cases_loaded=len(cases),
            repair_cases_attempted=0,
            rows=[],
            models_used=[],
            gpu_probe=gpu_probe(),
            repair_manifest=repair_manifest,
            blockers=blockers,
            honest_verdict="complete: blocked_no_mandated_sota_gguf_runtime",
        )
        _write_json(output, artifact)
        return artifact

    repair_rows: list[JsonDict] = []
    generation_blockers: list[str] = []
    if not blockers:
        if generator_fn is not None:
            repair_rows = _run_injected_generation(cases, models, generator_fn)
        else:  # pragma: no cover - exercised by the live experiment command.
            repair_rows, generation_blockers = _run_live_llama_generation(cases, models)
        blockers.extend(generation_blockers)

    _write_jsonl(repair_manifest, repair_rows)
    models_used = sorted(
        {
            str(row["model_hf_id"])
            for row in repair_rows
            if row.get("model_hf_id") in MANDATED_HF_IDS
        }
    )
    live_used = bool(repair_rows and models_used)
    summary = summarize_repair_rows(repair_rows)
    metrics_reported = all(
        summary[key] is not None
        for key in (
            "baseline_accept_rate",
            "grammar_only_accept_rate",
            "draft_conditioned_accept_rate",
            "repair_accept_rate_delta",
            "false_accept_rate",
        )
    )
    ready = live_used and metrics_reported and summary["false_accept_rate"] == 0.0
    if not live_used and "no_mandated_sota_model_completed_live_inference" not in blockers:
        blockers.append("no_mandated_sota_model_completed_live_inference")
    if summary["false_accept_rate"] not in (None, 0.0):
        blockers.append("false_accept_rate_nonzero")

    artifact = _terminal_artifact(
        status="complete" if repair_rows else "blocked",
        run_date=run_date,
        e2e_cases_loaded=len(cases),
        repair_cases_attempted=len(cases) if repair_rows else 0,
        rows=repair_rows,
        models_used=models_used,
        gpu_probe=gpu_probe(),
        repair_manifest=repair_manifest,
        blockers=blockers,
        honest_verdict=(
            "complete: contract-guided repair ready"
            if ready
            else "complete: contract-guided repair blocked before readiness"
        ),
    )
    _write_json(output, artifact)
    return artifact


def _run_injected_generation(
    cases: Sequence[JsonDict],
    models: Sequence[JsonDict],
    generator_fn: GeneratorFn,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for model in models:
        for case in cases:
            for mode in REPAIR_MODES:
                prompt = build_mode_prompt(case, mode)
                raw = generator_fn(prompt, model, mode, case)
                rows.append(validate_repair_output(case, model_spec=model, mode=mode, raw_output=raw))
    return rows


def build_mode_prompt(case: Mapping[str, Any], mode: str) -> str:
    """Build the exact prompt for one repair mode."""

    contract_context = json.dumps(
        {
            "contract_case_id": case.get("contract_case_id"),
            "source_family": case.get("source_family"),
            "prompt_or_case_id": case.get("prompt_or_case_id"),
            "proposed_output": case.get("proposed_output"),
            "certificate_parse_result": case.get("certificate_parse_result"),
            "safe_dsl_verifier_result": case.get("safe_dsl_verifier_result"),
            "monitor_event_result": case.get("monitor_event_result"),
            "structural_contract_result": case.get("structural_contract_result"),
        },
        sort_keys=True,
    )
    if mode == "baseline":
        instruction = (
            "Decide whether the proposed output should pass Carnot's runtime contract. "
            "Answer naturally."
        )
    elif mode == "grammar_only":
        instruction = (
            "Return strict JSON only with keys contract_case_id and "
            "final_deterministic_decision, where the decision is accept or reject."
        )
    elif mode == "draft_conditioned":
        instruction = (
            "First write a one-sentence draft diagnosis, then write strict JSON with keys "
            "contract_case_id and final_deterministic_decision."
        )
    else:
        raise ValueError(f"unknown repair mode: {mode}")
    return f"{instruction}\n\nRuntime contract context:\n{contract_context}\n"


def _validation_contract_case(case: Mapping[str, Any], final_accept: bool) -> JsonDict:
    validation = {
        key: case.get(key)
        for key in runtime_contracts.REQUIRED_CONTRACT_CASE_FIELDS
        if key in case
    }
    validation["row_type"] = "contract_case"
    validation["contract_schema_version"] = runtime_contracts.CONTRACT_CASE_SCHEMA_VERSION
    validation["final_deterministic_accept"] = bool(final_accept)
    validation["final_deterministic_decision"] = "accept" if final_accept else "reject"
    return validation


def _repair_outcome(
    *,
    structurally_valid: bool,
    parse_status: str,
    false_accept: bool,
    expected_label: bool | None,
    proposed_accept: bool,
) -> str:
    if not structurally_valid:
        return "invalid_structure" if parse_status == "no_json_object" else parse_status
    if false_accept:
        return "false_accept"
    if expected_label is None:
        return "unlabeled"
    return "accepted" if proposed_accept == expected_label else "wrong_decision"


def _resolve_runtime_models(
    cached_pair_fn: CachedPairFn,
    resolver_fn: ResolverFn,
    *,
    max_models: int,
) -> list[JsonDict]:
    models: list[JsonDict] = []
    try:
        pair = cached_pair_fn(gpu_indices=(0, 1), preferred_quant="Q4_K_M")
    except Exception:
        pair = None
    for spec in pair or []:
        hf_id = spec.get("hf_id")
        if hf_id in MANDATED_HF_IDS and spec.get("model_path"):
            models.append(dict(spec))
    if not models:
        for index, mandated in enumerate(MANDATED_MODEL_SPECS):
            model_path = resolver_fn(str(mandated["hf_id"]))
            if model_path:
                models.append(
                    {
                        "name": str(mandated["hf_id"]).rsplit("/", 1)[-1].removesuffix("-GGUF"),
                        "hf_id": mandated["hf_id"],
                        "role": mandated["role"],
                        "gpu": index,
                        "model_path": model_path,
                    }
                )
    return models[:max_models]


def _run_live_llama_generation(
    cases: Sequence[JsonDict],
    models: Sequence[JsonDict],
) -> tuple[list[JsonDict], list[str]]:  # pragma: no cover - live hardware path.
    rows: list[JsonDict] = []
    blockers: list[str] = []
    for model in models:
        try:
            model_rows = _run_one_live_model(cases, model)
        except Exception as exc:
            blockers.append(f"live_generation_failed:{model.get('hf_id')}:{type(exc).__name__}:{exc}")
            continue
        rows.extend(model_rows)
        if model_rows:
            break
    if not rows:
        blockers.append("no_mandated_sota_model_completed_live_inference")
    return rows, blockers


def _run_one_live_model(cases: Sequence[JsonDict], model: JsonDict) -> list[JsonDict]:  # pragma: no cover
    _ensure_cuda_library_path()
    from llama_cpp import Llama  # noqa: PLC0415

    gpu = int(model.get("gpu", 0))
    llm = Llama(
        model_path=str(model["model_path"]),
        n_gpu_layers=-1 if gpu >= 0 else 0,
        main_gpu=max(gpu, 0),
        n_ctx=2048,
        verbose=False,
    )
    rows: list[JsonDict] = []
    try:
        for case in cases:
            for mode in REPAIR_MODES:
                prompt = build_mode_prompt(case, mode)
                completion = llm(
                    prompt,
                    max_tokens=180,
                    temperature=0.0,
                    echo=False,
                    stop=["</s>", "<eos>"],
                )
                raw = _completion_text(completion)
                rows.append(validate_repair_output(case, model_spec=model, mode=mode, raw_output=raw))
    finally:
        if hasattr(llm, "close"):
            llm.close()
    return rows


def _ensure_cuda_library_path() -> None:  # pragma: no cover - host runtime repair.
    site_packages = sorted((Path.cwd() / ".venv" / "lib").glob("python*/site-packages"))
    candidates: list[str] = []
    for site in site_packages:
        candidates.extend(
            [
                str(site / "nvidia" / "cuda_runtime" / "lib"),
                str(site / "nvidia" / "cublas" / "lib"),
            ]
        )
    current_parts = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    repaired: list[str] = []
    seen: set[str] = set()
    for path in [*candidates, *current_parts]:
        if path in seen or not Path(path).is_dir():
            continue
        seen.add(path)
        repaired.append(path)
    if repaired:
        os.environ["LD_LIBRARY_PATH"] = ":".join(repaired)


def _completion_text(result: Any) -> str:  # pragma: no cover - llama.cpp shape adapter.
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
    if isinstance(text, str):
        return text.strip()
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return str(message["content"]).strip()
    return ""


def _terminal_artifact(
    *,
    status: str,
    run_date: str,
    e2e_cases_loaded: int,
    repair_cases_attempted: int,
    rows: Sequence[Mapping[str, Any]],
    models_used: Sequence[str],
    gpu_probe: Mapping[str, Any],
    repair_manifest: Path,
    blockers: Sequence[str],
    honest_verdict: str,
) -> JsonDict:
    summary = summarize_repair_rows(rows)
    ready = (
        bool(models_used)
        and summary["baseline_accept_rate"] is not None
        and summary["grammar_only_accept_rate"] is not None
        and summary["draft_conditioned_accept_rate"] is not None
        and summary["repair_accept_rate_delta"] is not None
        and summary["false_accept_rate"] == 0.0
        and not blockers
    )
    return {
        "status": status,
        "run_date": run_date,
        "schema_version": 1,
        "model_specs": [dict(spec) for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": bool(models_used),
        "contract_guided_repair_ready": bool(ready),
        "e2e_cases_loaded": int(e2e_cases_loaded),
        "repair_cases_attempted": int(repair_cases_attempted),
        "baseline_accept_rate": summary["baseline_accept_rate"],
        "grammar_only_accept_rate": summary["grammar_only_accept_rate"],
        "draft_conditioned_accept_rate": summary["draft_conditioned_accept_rate"],
        "repair_accept_rate_delta": summary["repair_accept_rate_delta"],
        "false_accept_count": summary["false_accept_count"],
        "false_accept_rate": summary["false_accept_rate"],
        "models_used": list(models_used),
        "gpu_probe": dict(gpu_probe),
        "repair_manifest_path": _display_path(repair_manifest),
        "blockers": list(dict.fromkeys(blockers)),
        "honest_verdict": honest_verdict,
        "explicit_label_count": summary["explicit_label_count"],
        "explicit_reject_count": summary["explicit_reject_count"],
    }


def _in_progress_artifact(*, run_date: str, repair_manifest: Path) -> JsonDict:
    return {
        "status": "in_progress",
        "run_date": run_date,
        "schema_version": 1,
        "model_specs": [dict(spec) for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": False,
        "contract_guided_repair_ready": False,
        "e2e_cases_loaded": 0,
        "repair_cases_attempted": 0,
        "baseline_accept_rate": None,
        "grammar_only_accept_rate": None,
        "draft_conditioned_accept_rate": None,
        "repair_accept_rate_delta": None,
        "false_accept_count": 0,
        "false_accept_rate": None,
        "models_used": [],
        "gpu_probe": {},
        "repair_manifest_path": _display_path(repair_manifest),
        "blockers": ["experiment_1521_contract_guided_repair_in_progress"],
        "honest_verdict": "complete: in-progress live SOTA contract-guided repair",
    }


def _extract_last_json_object(text: str) -> JsonDict | None:
    decoder = json.JSONDecoder()
    last: JsonDict | None = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            last = parsed
    return last


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _cached_sota_pair(**kwargs: Any) -> list[JsonDict] | None:  # pragma: no cover
    from carnot.inference.sota_models import cached_sota_pair

    return cached_sota_pair(**kwargs)


def _resolve_cached_gguf(hf_id: str) -> str | None:  # pragma: no cover
    from carnot.inference.sota_models import resolve_cached_gguf

    return resolve_cached_gguf(hf_id)


def _probe_gpu_state() -> JsonDict:  # pragma: no cover
    from carnot.reporting.live_sota_repair_runtime_preflight import probe_gpu_state

    return probe_gpu_state()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", default=RUN_DATE)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--repair-manifest", type=Path, default=DEFAULT_REPAIR_MANIFEST_PATH)
    parser.add_argument("--case-limit", type=int, default=2)
    parser.add_argument("--max-models", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = _parse_args(argv)
    run_experiment(
        run_date=args.run_date,
        source_manifest_path=args.source_manifest,
        output_path=args.output,
        repair_manifest_path=args.repair_manifest,
        case_limit=args.case_limit,
        max_models=args.max_models,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
