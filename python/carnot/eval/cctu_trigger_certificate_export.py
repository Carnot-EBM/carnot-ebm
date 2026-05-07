"""Exp 1493 trigger-token certificate export for CCTU-style cases.

Spec: REQ-VERIFY-1493, SCENARIO-VERIFY-1493.

The lane tested here follows the "think before constraining" pattern: the
model may solve in ordinary prose, must then emit a unique trigger token, and
only after that token may export the structured certificate JSON.  The
certificate is never trusted directly; it is replayed through the deterministic
Exp 1486 CCTU validators.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from carnot.eval import cctu_executable_constraint_microbenchmark as cctu

JsonDict = dict[str, Any]

RUN_DATE = "20260507"
TRIGGER_TOKEN = "<<<CCTU_CERTIFICATE_V1>>>"
TRIGGER_LANE = "trigger_certificate"
ALWAYS_CONSTRAINED_LANE = "always_constrained"
LANES: tuple[str, str] = (TRIGGER_LANE, ALWAYS_CONSTRAINED_LANE)
DEFAULT_ARTIFACT_PATH = Path(
    "results/experiment_1493_trigger_token_certificate_export_v1.json"
)
DEFAULT_MANIFEST_PATH = Path("results/cctu_trigger_certificates_1493.jsonl")

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = cctu.MANDATED_MODEL_SPECS
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "live_sota_model_inference_used",
    "trigger_certificate_ready",
    "cctu_cases_attempted",
    "cctu_cases_completed",
    "certificate_parse_rate",
    "certificate_validation_rate",
    "always_constrained_parse_rate",
    "always_constrained_validation_rate",
    "verifier_false_accept_rate",
    "certificate_manifest_path",
    "models_used",
    "gpu_probe",
    "blockers",
    "honest_verdict",
)

ResolverFn = Callable[[str], str | None]
CachedPairFn = Callable[..., list[JsonDict] | None]
LlamaImporterFn = Callable[[], tuple[bool, type[Any] | None, str | None]]
CollectModelOutputsFn = Callable[[JsonDict, list[cctu.BenchmarkCase]], JsonDict]


def certificate_for_case(case: cctu.BenchmarkCase) -> JsonDict:
    """Return the canonical certificate JSON for one CCTU case.

    Why the certificate repeats the prompt-side tool call: the verifier needs a
    self-contained payload it can replay later from the manifest without
    trusting hidden model state or the original prompt string.
    """

    return {
        "certificate_version": "cctu-trigger-v1",
        "case_id": case.case_id,
        "tool_call": {"name": case.tool_name, "arguments": case.tool_arguments},
        "tool_result": case.expected_tool_result,
        "final_answer": case.expected_final_answer,
        "verifier": {"accept": True},
    }


def certificate_text_for_case(
    case: cctu.BenchmarkCase,
    *,
    lane: str,
    reasoning_text: str = "",
) -> str:
    """Return a deterministic gold output for tests and CPU smoke fixtures."""

    certificate_text = json.dumps(certificate_for_case(case), sort_keys=True)
    if lane == TRIGGER_LANE:
        prefix = reasoning_text.strip()
        return f"{prefix}\n{TRIGGER_TOKEN}\n{certificate_text}".strip()
    return certificate_text


def build_trigger_prompt(case: cctu.BenchmarkCase) -> str:
    """Build the free-form-then-trigger prompt for one CCTU case."""

    schema = certificate_for_case(case)
    return (
        "You are evaluating a CCTU-style executable constraint case.\n"
        f"Case: {case.case_id}\n"
        f"Constraint family: {case.family}\n"
        f"Use exactly one local tool named {case.tool_name} with these arguments:\n"
        f"{json.dumps(case.tool_arguments, sort_keys=True)}\n"
        "First solve freely in prose, bounded to the information above. "
        f"Then emit the exact trigger token {TRIGGER_TOKEN} on its own line. "
        "After the trigger, emit exactly one JSON certificate and no more prose. "
        "The certificate must match this shape:\n"
        f"{json.dumps(schema, sort_keys=True)}"
    )


def build_always_constrained_prompt(case: cctu.BenchmarkCase) -> str:
    """Build the direct structured baseline prompt for one CCTU case."""

    schema = certificate_for_case(case)
    return (
        "You are evaluating a CCTU-style executable constraint case.\n"
        f"Case: {case.case_id}\n"
        f"Constraint family: {case.family}\n"
        f"Use exactly one local tool named {case.tool_name} with these arguments:\n"
        f"{json.dumps(case.tool_arguments, sort_keys=True)}\n"
        "Return exactly one JSON certificate and no prose. "
        "The certificate must match this shape:\n"
        f"{json.dumps(schema, sort_keys=True)}"
    )


def parse_certificate_output(
    output_text: str,
    *,
    lane: str,
    trigger_token: str = TRIGGER_TOKEN,
) -> JsonDict:
    """Parse a model output into reasoning text plus certificate JSON."""

    text = str(output_text or "")
    parsed: JsonDict = {
        "lane": lane,
        "parsed": False,
        "parse_error": None,
        "trigger_token_present": False,
        "free_form_reasoning_text": "",
        "certificate_json": None,
    }
    if lane == TRIGGER_LANE:
        count = text.count(trigger_token)
        parsed["trigger_token_present"] = count > 0
        if count == 0:
            parsed["parse_error"] = "missing_trigger_token"
            return parsed
        if count > 1:
            parsed["parse_error"] = "duplicate_trigger_token"
            return parsed
        reasoning, certificate_tail = text.split(trigger_token, 1)
        parsed["free_form_reasoning_text"] = reasoning.strip()
        obj = cctu.extract_json_object(certificate_tail)
    else:
        obj = cctu.extract_json_object(text)

    if obj is None:
        parsed["parse_error"] = "no_json_object_after_trigger" if lane == TRIGGER_LANE else "no_json_object"
        return parsed

    parsed["parsed"] = True
    parsed["certificate_json"] = obj
    return parsed


def validate_certificate(
    case: cctu.BenchmarkCase,
    certificate_json: JsonDict | None,
) -> JsonDict:
    """Replay a parsed certificate through the executable CCTU validators."""

    if certificate_json is None:
        return _missing_certificate_validation("missing_certificate_json")

    validation = cctu.validate_transcript(
        case,
        json.dumps(certificate_json, sort_keys=True),
    )
    validator = dict(validation["validator_result"])
    case_id_valid = certificate_json.get("case_id") == case.case_id
    validator["case_id_valid"] = bool(case_id_valid)

    verifier = dict(validation["verifier_result"])
    base_valid = bool(verifier["base_valid"]) and bool(case_id_valid)
    accepted = bool(base_valid) and bool(verifier["verifier_outcome_valid"])
    verifier["base_valid"] = base_valid
    verifier["accepted"] = accepted
    verifier["caught_invalid"] = not base_valid and not accepted
    verifier["false_accept"] = not base_valid and accepted
    return {"validator_result": validator, "verifier_result": verifier}


def build_manifest_row(
    case: cctu.BenchmarkCase,
    generation_row: JsonDict,
) -> JsonDict:
    """Join a raw generation row with parser and deterministic validation data."""

    lane = str(generation_row.get("lane") or TRIGGER_LANE)
    output_text = str(generation_row.get("output_text") or "")
    parser_result = parse_certificate_output(output_text, lane=lane)
    validation = validate_certificate(case, parser_result.get("certificate_json"))
    verifier_result = validation["verifier_result"]
    return {
        "case_id": case.case_id,
        "family": case.family,
        "lane": lane,
        "prompt": generation_row.get("prompt"),
        "model_hf_id": generation_row.get("model_hf_id"),
        "model_name": generation_row.get("model_name"),
        "generation_source": generation_row.get("generation_source"),
        "elapsed_seconds": generation_row.get("elapsed_seconds"),
        "blocker": generation_row.get("blocker"),
        "model_output": output_text,
        "free_form_reasoning_text": parser_result["free_form_reasoning_text"],
        "trigger_token_present": parser_result["trigger_token_present"],
        "certificate_json": parser_result["certificate_json"],
        "parser_result": parser_result,
        "validator_result": validation["validator_result"],
        "verifier_result": verifier_result,
        "deterministic_validation_passed": bool(verifier_result["accepted"]),
        "false_accept_status": bool(verifier_result["false_accept"]),
    }


def aggregate_manifest_metrics(rows: list[JsonDict]) -> JsonDict:
    """Compute parse, validation, and false-accept rates from manifest rows."""

    trigger_rows = [row for row in rows if row.get("lane") == TRIGGER_LANE]
    baseline_rows = [row for row in rows if row.get("lane") == ALWAYS_CONSTRAINED_LANE]
    invalid_rows = [row for row in rows if not bool(row["verifier_result"]["base_valid"])]
    false_accepts = sum(bool(row["verifier_result"]["false_accept"]) for row in invalid_rows)
    return {
        "certificate_parse_rate": _rate(
            trigger_rows,
            lambda row: bool(row["parser_result"]["parsed"]),
        ),
        "certificate_validation_rate": _rate(
            trigger_rows,
            lambda row: bool(row["verifier_result"]["accepted"]),
        ),
        "always_constrained_parse_rate": _rate(
            baseline_rows,
            lambda row: bool(row["parser_result"]["parsed"]),
        ),
        "always_constrained_validation_rate": _rate(
            baseline_rows,
            lambda row: bool(row["verifier_result"]["accepted"]),
        ),
        "verifier_false_accept_rate": (
            round(false_accepts / len(invalid_rows), 6) if invalid_rows else 0.0
        ),
    }


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Write the durable bootstrap artifact required by REQ-VERIFY-1493-1."""

    payload: JsonDict = {
        "status": "in_progress",
        "run_date": run_date,
        "model_specs": [spec["hf_id"] for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": False,
        "trigger_certificate_ready": False,
        "cctu_cases_attempted": 0,
        "cctu_cases_completed": 0,
        "certificate_parse_rate": 0.0,
        "certificate_validation_rate": 0.0,
        "always_constrained_parse_rate": 0.0,
        "always_constrained_validation_rate": 0.0,
        "verifier_false_accept_rate": 0.0,
        "certificate_manifest_path": _display_path(DEFAULT_MANIFEST_PATH),
        "models_used": [],
        "gpu_probe": {},
        "blockers": [],
        "honest_verdict": "complete: in-progress Exp 1493 bootstrap artifact",
    }
    _write_json(Path(output_path), payload)
    return payload


def run_experiment(
    *,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    run_date: str = RUN_DATE,
    model_specs: Iterable[JsonDict] | None = None,
    collect_model_outputs_fn: CollectModelOutputsFn | None = None,
    gpu_probe_fn: Callable[[], JsonDict] | None = None,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Run the trigger-token versus always-constrained CCTU certificate study."""

    output = Path(output_path)
    manifest = Path(manifest_path)
    write_in_progress_artifact(output, run_date=run_date)

    cases = cctu.build_benchmark_cases()
    specs = list(resolve_model_specs() if model_specs is None else model_specs)
    gpu_probe = (gpu_probe_fn or probe_gpu)()
    collector = collect_model_outputs_fn or collect_live_model_outputs
    rows: list[JsonDict] = []
    model_attempts: list[JsonDict] = []
    case_by_id = {case.case_id: case for case in cases}

    if not specs:
        _write_jsonl(manifest, rows)
        artifact = _build_terminal_artifact(
            run_date=run_date,
            manifest_path=manifest,
            cases=cases,
            rows=rows,
            model_attempts=[],
            gpu_probe=gpu_probe,
            blockers=["no_mandated_sota_gguf_model_available"],
            tests_run=tests_run,
        )
        _write_json(output, artifact)
        return artifact

    for spec in specs:
        collection = collector(dict(spec), cases)
        summary = dict(collection.get("summary") or {})
        model_attempts.append(summary)
        for generation_row in collection.get("rows") or []:
            case = case_by_id.get(generation_row.get("case_id"))
            if case is not None:
                rows.append(build_manifest_row(case, generation_row))

    _write_jsonl(manifest, rows)
    blockers = [
        str(summary.get("blocker"))
        for summary in model_attempts
        if summary.get("model_used") is not True and summary.get("blocker")
    ]
    if not _live_sota_rows_present(rows):
        blockers.append("live_sota_generation_unavailable")

    artifact = _build_terminal_artifact(
        run_date=run_date,
        manifest_path=manifest,
        cases=cases,
        rows=rows,
        model_attempts=model_attempts,
        gpu_probe=gpu_probe,
        blockers=list(dict.fromkeys(blockers)),
        tests_run=tests_run,
    )
    _write_json(output, artifact)
    return artifact


def resolve_model_specs(
    *,
    cached_pair_fn: CachedPairFn | None = None,
    resolver_fn: ResolverFn | None = None,
) -> list[JsonDict]:
    """Resolve mandated local SOTA GGUF specs without using legacy fallbacks."""

    pair_resolver = cached_pair_fn or _cached_sota_pair
    pair = pair_resolver(gpu_indices=(0, 1))
    if pair:
        return pair

    resolver = resolver_fn or _resolve_cached_gguf
    specs: list[JsonDict] = []
    for spec in MANDATED_MODEL_SPECS:
        model_path = resolver(str(spec["hf_id"]))
        if model_path:
            specs.append({**spec, "model_path": model_path})
    return specs


def collect_live_model_outputs(
    spec: JsonDict,
    cases: list[cctu.BenchmarkCase],
    *,
    resolver: ResolverFn | None = None,
    llama_importer: LlamaImporterFn | None = None,
    env_preparer: Callable[[], JsonDict] | None = None,
) -> JsonDict:
    """Collect trigger and baseline outputs from one local GGUF model."""

    hf_id = str(spec.get("hf_id") or "")
    model_path = spec.get("model_path") or (resolver or _resolve_cached_gguf)(hf_id)
    if not model_path:
        return {
            "summary": {
                "hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_used": False,
                "blocker": "model_not_cached",
            },
            "rows": [],
        }

    env_details = (env_preparer or cctu.prepare_llama_environment)()
    ok, llama_class, import_error = (llama_importer or cctu._default_llama_importer)()
    if not ok or llama_class is None:
        return {
            "summary": {
                "hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_path": str(model_path),
                "model_used": False,
                "blocker": import_error or "llama_cpp_import_failed",
                "env_details": env_details,
            },
            "rows": [],
        }

    load_start = time.monotonic()
    try:
        llm = llama_class(
            model_path=str(model_path),
            n_gpu_layers=-1,
            main_gpu=int(spec.get("gpu") or 0),
            n_ctx=4096,
            seed=1493,
            verbose=False,
        )
    except Exception as exc:
        return {
            "summary": {
                "hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_path": str(model_path),
                "model_used": False,
                "blocker": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.monotonic() - load_start, 6),
                "env_details": env_details,
            },
            "rows": [],
        }

    rows: list[JsonDict] = []
    try:
        for case in cases:
            for lane, prompt in (
                (TRIGGER_LANE, build_trigger_prompt(case)),
                (ALWAYS_CONSTRAINED_LANE, build_always_constrained_prompt(case)),
            ):
                started = time.monotonic()
                try:
                    result = llm(
                        prompt,
                        max_tokens=384 if lane == TRIGGER_LANE else 256,
                        temperature=0.0,
                        top_p=1.0,
                        stop=["</s>", "<eos>"],
                        echo=False,
                    )
                    output_text = cctu._completion_text(result)
                    blocker = None if output_text.strip() else "empty_generation"
                except Exception as exc:
                    output_text = ""
                    blocker = f"{type(exc).__name__}: {exc}"
                rows.append(
                    {
                        "case_id": case.case_id,
                        "lane": lane,
                        "prompt": prompt,
                        "model_hf_id": hf_id,
                        "model_name": spec.get("name"),
                        "model_path": str(model_path),
                        "generation_source": "live_sota_llamacpp",
                        "output_text": output_text,
                        "elapsed_seconds": round(time.monotonic() - started, 6),
                        "blocker": blocker,
                    }
                )
    finally:
        cctu._close_llama(llm)

    model_used = any(row.get("blocker") is None for row in rows)
    return {
        "summary": {
            "hf_id": hf_id,
            "model_name": spec.get("name"),
            "model_path": str(model_path),
            "model_used": model_used,
            "blocker": None if model_used else "no_usable_generations",
            "env_details": env_details,
        },
        "rows": rows,
    }


def probe_gpu() -> JsonDict:
    """Return a small JSON-safe NVIDIA GPU probe for the result artifact."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return {
            "nvidia_smi_available": False,
            "gpu_count": 0,
            "gpus": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

    if result.returncode != 0:  # pragma: no cover - defensive hardware branch.
        return {
            "nvidia_smi_available": False,
            "gpu_count": 0,
            "gpus": [],
            "error": result.stderr.strip() or "nvidia-smi failed",
        }

    gpus = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        name, _, memory = line.partition(",")
        gpus.append({"name": name.strip(), "memory_total": memory.strip()})
    return {"nvidia_smi_available": True, "gpu_count": len(gpus), "gpus": gpus}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the conductor and manual runs."""

    _ = list(sys.argv[1:] if argv is None else argv)
    artifact = run_experiment()
    print(
        "[exp1493] "
        f"ready={artifact['trigger_certificate_ready']} "
        f"parse={artifact['certificate_parse_rate']} "
        f"validation={artifact['certificate_validation_rate']} "
        f"verdict={artifact['honest_verdict']}"
    )
    return 0


def _build_terminal_artifact(
    *,
    run_date: str,
    manifest_path: Path,
    cases: list[cctu.BenchmarkCase],
    rows: list[JsonDict],
    model_attempts: list[JsonDict],
    gpu_probe: JsonDict,
    blockers: list[str],
    tests_run: list[str] | None,
) -> JsonDict:
    metrics = aggregate_manifest_metrics(rows)
    models_used = [
        str(summary["hf_id"])
        for summary in model_attempts
        if summary.get("model_used") is True and summary.get("hf_id")
    ]
    live_used = _live_sota_rows_present(rows)
    ready = live_used and bool([row for row in rows if row.get("lane") == TRIGGER_LANE])
    status = "complete" if ready else "blocked"
    verdict = (
        "complete: trigger-token certificate export measured on live local SOTA GGUF rows"
        if ready
        else "complete: blocked because no mandated live SOTA GGUF certificate rows were produced"
    )
    return {
        "status": status,
        "run_date": run_date,
        "schema_version": 1,
        "model_specs": [spec["hf_id"] for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": bool(live_used),
        "trigger_certificate_ready": bool(ready),
        "cctu_cases_attempted": len(cases),
        "cctu_cases_completed": _completed_case_count(rows),
        "certificate_parse_rate": metrics["certificate_parse_rate"],
        "certificate_validation_rate": metrics["certificate_validation_rate"],
        "always_constrained_parse_rate": metrics["always_constrained_parse_rate"],
        "always_constrained_validation_rate": metrics["always_constrained_validation_rate"],
        "verifier_false_accept_rate": metrics["verifier_false_accept_rate"],
        "certificate_manifest_path": _display_path(manifest_path),
        "models_used": models_used,
        "gpu_probe": gpu_probe,
        "blockers": blockers,
        "honest_verdict": verdict,
        "model_attempts": model_attempts,
        "manifest_rows": len(rows),
        "tests_run": list(tests_run or []),
    }


def _missing_certificate_validation(parse_error: str) -> JsonDict:
    return {
        "validator_result": {
            "parse_error": parse_error,
            "tool_call_structure_valid": False,
            "tool_result_consistent": False,
            "final_answer_valid": False,
            "verifier_outcome_valid": False,
            "tool_result_error": "missing_json",
            "model_declared_accept": None,
            "case_id_valid": False,
        },
        "verifier_result": {
            "base_valid": False,
            "accepted": False,
            "model_declared_accept": None,
            "verifier_outcome_valid": False,
            "caught_invalid": True,
            "false_accept": False,
        },
    }


def _rate(rows: list[JsonDict], predicate: Callable[[JsonDict], bool]) -> float:
    if not rows:
        return 0.0
    return round(sum(bool(predicate(row)) for row in rows) / len(rows), 6)


def _completed_case_count(rows: list[JsonDict]) -> int:
    lanes_by_case: dict[str, set[str]] = {}
    for row in rows:
        if row.get("blocker") is not None:
            continue
        lanes_by_case.setdefault(str(row.get("case_id")), set()).add(str(row.get("lane")))
    return sum(set(LANES) <= lanes for lanes in lanes_by_case.values())


def _live_sota_rows_present(rows: list[JsonDict]) -> bool:
    mandated = {str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS}
    return any(
        row.get("generation_source") == "live_sota_llamacpp"
        and row.get("blocker") is None
        and row.get("model_hf_id") in mandated
        for row in rows
    )


def _cached_sota_pair(**kwargs: Any) -> list[JsonDict] | None:  # pragma: no cover
    from carnot.inference.sota_models import cached_sota_pair  # noqa: PLC0415

    return cached_sota_pair(**kwargs)


def _resolve_cached_gguf(hf_id: str) -> str | None:  # pragma: no cover
    from carnot.inference.sota_models import resolve_cached_gguf  # noqa: PLC0415

    return resolve_cached_gguf(hf_id)


def _display_path(path: Path | str) -> str:
    as_path = Path(path)
    try:
        return str(as_path.resolve().relative_to(cctu._repo_root()))
    except ValueError:
        return str(as_path)


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - exercised by conductor.
    raise SystemExit(main())


__all__ = [
    "ALWAYS_CONSTRAINED_LANE",
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_MANIFEST_PATH",
    "MANDATED_MODEL_SPECS",
    "REQUIRED_ARTIFACT_FIELDS",
    "TRIGGER_LANE",
    "TRIGGER_TOKEN",
    "aggregate_manifest_metrics",
    "build_always_constrained_prompt",
    "build_manifest_row",
    "build_trigger_prompt",
    "certificate_for_case",
    "certificate_text_for_case",
    "collect_live_model_outputs",
    "main",
    "parse_certificate_output",
    "probe_gpu",
    "resolve_model_specs",
    "run_experiment",
    "validate_certificate",
    "write_in_progress_artifact",
]
