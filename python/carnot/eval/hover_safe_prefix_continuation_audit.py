"""Exp 1496 HoVer-style safe-prefix continuation audit.

Spec: REQ-VERIFY-1496, SCENARIO-VERIFY-1496.

This module turns the prior CCTU monitor evidence into a bounded continuation
study.  The important safety rule is that the language model does not decide
what prefix is safe.  Carnot selects the prefix from deterministic monitor
events, asks a mandated local GGUF model to continue only the suffix, and then
checks the final text with the same executable validators used by the CCTU
benchmark.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from carnot.eval import cctu_trigger_certificate_export as certificates
from carnot.eval import constrainprompt_validator_compiler_audit as compiler

JsonDict = dict[str, Any]

RUN_DATE = "20260507"
DEFAULT_ARTIFACT_PATH = Path(
    "results/experiment_1496_hover_safe_prefix_continuation_audit.json"
)
DEFAULT_MANIFEST_PATH = Path("results/safe_prefix_continuations_1496.jsonl")
DEFAULT_MONITOR_ARTIFACT_PATH = Path("results/experiment_1495_interwhen_monitor_prototype.json")
DEFAULT_MONITOR_EVENT_MANIFEST_PATH = Path("results/interwhen_monitor_events_1495.jsonl")
DEFAULT_CERTIFICATE_MANIFEST_PATH = Path("results/cctu_trigger_certificates_1493.jsonl")
DEFAULT_VALIDATOR_MANIFEST_PATH = Path("results/constrainprompt_validator_manifest_1494.jsonl")

NO_CONTINUATION_MODE = "no_continuation"
SAFE_PREFIX_MODE = "safe_prefix_continuation"
FULL_REGENERATION_MODE = "full_regeneration"
EVALUATION_MODES: tuple[str, str, str] = (
    NO_CONTINUATION_MODE,
    SAFE_PREFIX_MODE,
    FULL_REGENERATION_MODE,
)

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = certificates.MANDATED_MODEL_SPECS
MANDATED_MODEL_IDS: frozenset[str] = frozenset(
    str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS
)
LAST_SAFE_PREFIX_SELECTION_RULE = (
    "For each CCTU trigger-certificate row selected by Exp 1495 monitor events, "
    "sort same-case same-lane events by token_offset/poll_index, choose the first "
    "interrupting or error-detected event, and keep only the free-form reasoning "
    "plus the <<<CCTU_CERTIFICATE_V1>>> trigger boundary before the unsafe "
    "certificate suffix. If the trigger boundary is absent, fall back to the text "
    "before the first JSON object."
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "live_sota_model_inference_used",
    "safe_prefix_continuation_ready",
    "cases_attempted",
    "continuations_completed",
    "baseline_validator_pass_rate",
    "safe_prefix_validator_pass_rate",
    "full_regeneration_validator_pass_rate",
    "verifier_false_accept_rate",
    "last_safe_prefix_selection_rule",
    "continuation_manifest_path",
    "models_used",
    "gpu_probe",
    "blockers",
    "honest_verdict",
)

CollectContinuationsFn = Callable[[JsonDict, list[JsonDict]], JsonDict]


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Write the durable bootstrap artifact before any gated inputs are loaded."""

    payload: JsonDict = {
        "status": "in_progress",
        "run_date": run_date,
        "schema_version": 1,
        "model_specs": [spec["hf_id"] for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": False,
        "safe_prefix_continuation_ready": False,
        "cases_attempted": 0,
        "continuations_completed": 0,
        "baseline_validator_pass_rate": None,
        "safe_prefix_validator_pass_rate": None,
        "full_regeneration_validator_pass_rate": None,
        "verifier_false_accept_rate": None,
        "last_safe_prefix_selection_rule": LAST_SAFE_PREFIX_SELECTION_RULE,
        "continuation_manifest_path": _display_path(manifest_path),
        "models_used": [],
        "gpu_probe": {},
        "blockers": [],
        "honest_verdict": "complete: in-progress Exp 1496 bootstrap artifact",
    }
    _write_json(Path(output_path), payload)
    return payload


def run_experiment(
    *,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    monitor_artifact_path: Path | str = DEFAULT_MONITOR_ARTIFACT_PATH,
    monitor_event_manifest_path: Path | str = DEFAULT_MONITOR_EVENT_MANIFEST_PATH,
    certificate_manifest_path: Path | str = DEFAULT_CERTIFICATE_MANIFEST_PATH,
    validator_manifest_path: Path | str = DEFAULT_VALIDATOR_MANIFEST_PATH,
    run_date: str = RUN_DATE,
    model_specs: Iterable[JsonDict] | None = None,
    collect_continuations_fn: CollectContinuationsFn | None = None,
    gpu_probe_fn: Callable[[], JsonDict] | None = None,
    max_cases: int = 3,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Run the bounded safe-prefix audit and write the manifest plus artifact."""

    output = Path(output_path)
    manifest = Path(manifest_path)
    write_in_progress_artifact(output, manifest_path=manifest, run_date=run_date)

    blockers = gated_input_blockers(
        monitor_artifact_path=monitor_artifact_path,
        monitor_event_manifest_path=monitor_event_manifest_path,
        certificate_manifest_path=certificate_manifest_path,
        validator_manifest_path=validator_manifest_path,
    )
    gpu_probe = (gpu_probe_fn or certificates.probe_gpu)()
    if blockers:
        _write_jsonl(manifest, [])
        artifact = _terminal_artifact(
            run_date=run_date,
            manifest_path=manifest,
            rows=[],
            plans=[],
            model_attempts=[],
            gpu_probe=gpu_probe,
            blockers=blockers,
            tests_run=tests_run,
        )
        _write_json(output, artifact)
        return artifact

    monitor_events = _load_jsonl(Path(monitor_event_manifest_path))
    certificate_rows = _load_jsonl(Path(certificate_manifest_path))
    validator_rows = load_validator_rows(Path(validator_manifest_path))
    plans = build_case_plans(
        certificate_rows,
        monitor_events,
        max_cases=max_cases,
    )
    if not plans:
        _write_jsonl(manifest, [])
        artifact = _terminal_artifact(
            run_date=run_date,
            manifest_path=manifest,
            rows=[],
            plans=[],
            model_attempts=[],
            gpu_probe=gpu_probe,
            blockers=["no_interrupting_cctu_trigger_cases_selected"],
            tests_run=tests_run,
        )
        _write_json(output, artifact)
        return artifact

    specs = list(resolve_model_specs() if model_specs is None else model_specs)
    if not specs:
        baseline_rows = _build_evaluation_rows(plans, validator_rows, [])
        _write_jsonl(manifest, baseline_rows)
        artifact = _terminal_artifact(
            run_date=run_date,
            manifest_path=manifest,
            rows=baseline_rows,
            plans=plans,
            model_attempts=[],
            gpu_probe=gpu_probe,
            blockers=[
                "no_mandated_sota_gguf_model_available",
                "legacy_headline_fallback_disallowed",
            ],
            tests_run=tests_run,
        )
        _write_json(output, artifact)
        return artifact

    collector = collect_continuations_fn or collect_live_continuations
    collection = collector(dict(specs[0]), plans)
    model_attempts = [dict(collection.get("summary") or {})]
    generation_rows = [
        dict(row) for row in collection.get("rows") or [] if isinstance(row, dict)
    ]
    rows = _build_evaluation_rows(plans, validator_rows, generation_rows)
    _write_jsonl(manifest, rows)

    model_blockers = [
        str(summary.get("blocker"))
        for summary in model_attempts
        if summary.get("model_used") is not True and summary.get("blocker")
    ]
    if model_blockers:
        model_blockers.append("legacy_headline_fallback_disallowed")
    artifact = _terminal_artifact(
        run_date=run_date,
        manifest_path=manifest,
        rows=rows,
        plans=plans,
        model_attempts=model_attempts,
        gpu_probe=gpu_probe,
        blockers=list(dict.fromkeys(model_blockers)),
        tests_run=tests_run,
    )
    _write_json(output, artifact)
    return artifact


def gated_input_blockers(
    *,
    monitor_artifact_path: Path | str = DEFAULT_MONITOR_ARTIFACT_PATH,
    monitor_event_manifest_path: Path | str = DEFAULT_MONITOR_EVENT_MANIFEST_PATH,
    certificate_manifest_path: Path | str = DEFAULT_CERTIFICATE_MANIFEST_PATH,
    validator_manifest_path: Path | str = DEFAULT_VALIDATOR_MANIFEST_PATH,
) -> list[str]:
    """Return concrete upstream blockers before any model continuation is attempted."""

    blockers: list[str] = []
    monitor_artifact = _load_json_if_exists(Path(monitor_artifact_path))
    if monitor_artifact is None:
        blockers.append("missing_monitor_artifact")
    elif (
        monitor_artifact.get("status") != "complete"
        or monitor_artifact.get("monitor_intervention_ready") is not True
    ):
        blockers.append("monitor_gate_not_ready")

    if not Path(monitor_event_manifest_path).exists():
        blockers.append("missing_monitor_event_manifest")
    if not Path(certificate_manifest_path).exists():
        blockers.append("missing_certificate_manifest")
    if not Path(validator_manifest_path).exists():
        blockers.append("missing_validator_manifest")
    return blockers


def build_case_plans(
    certificate_rows: list[JsonDict],
    monitor_events: list[JsonDict],
    *,
    max_cases: int = 3,
) -> list[JsonDict]:
    """Build matched continuation plans from interrupted trigger-certificate rows."""

    cases_by_id = {case.case_id: case for case in certificates.cctu.build_benchmark_cases()}
    events_by_case_lane: dict[tuple[str, str], list[JsonDict]] = {}
    for event in monitor_events:
        case_id = str(event.get("case_id") or "")
        lane = str(event.get("lane") or "")
        events_by_case_lane.setdefault((case_id, lane), []).append(event)

    candidates: list[JsonDict] = []
    for row in certificate_rows:
        case_id = str(row.get("case_id") or "")
        lane = str(row.get("lane") or "")
        case = cases_by_id.get(case_id)
        same_lane_events = events_by_case_lane.get((case_id, lane), [])
        interrupt_events = [event for event in same_lane_events if _event_interrupts(event)]
        if (
            case is None
            or lane != certificates.TRIGGER_LANE
            or row.get("blocker") is not None
            or row.get("model_hf_id") not in MANDATED_MODEL_IDS
            or bool(row.get("deterministic_validation_passed"))
            or not interrupt_events
        ):
            continue
        selection = select_last_safe_prefix(row, same_lane_events)
        candidates.append(
            {
                "case": case,
                "case_id": case_id,
                "family": case.family,
                "source_row": row,
                "original_prompt": str(row.get("prompt") or certificates.build_trigger_prompt(case)),
                **selection,
            }
        )

    candidates.sort(
        key=lambda plan: (
            int(plan.get("selected_event_token_offset") or 0),
            str(plan.get("case_id") or ""),
        )
    )
    return candidates[: max(0, int(max_cases))]


def select_last_safe_prefix(source_row: JsonDict, monitor_events: list[JsonDict]) -> JsonDict:
    """Select the deterministic last safe prefix for one monitored source row."""

    interrupt_events = sorted(
        [event for event in monitor_events if _event_interrupts(event)],
        key=lambda event: (
            int(event.get("token_offset") or 0),
            int(event.get("poll_index") or 0),
            str(event.get("event_id") or ""),
        ),
    )
    selected = interrupt_events[0] if interrupt_events else {}
    output_text = str(source_row.get("model_output") or "")
    reasoning = str(source_row.get("free_form_reasoning_text") or "").strip()
    trigger_present = bool(source_row.get("trigger_token_present")) or (
        certificates.TRIGGER_TOKEN in output_text
    )

    if trigger_present:
        if reasoning:
            safe_prefix = f"{reasoning}\n{certificates.TRIGGER_TOKEN}\n"
        elif certificates.TRIGGER_TOKEN in output_text:
            before_trigger = output_text.split(certificates.TRIGGER_TOKEN, 1)[0].strip()
            safe_prefix = f"{before_trigger}\n{certificates.TRIGGER_TOKEN}\n"
        else:
            safe_prefix = f"{certificates.TRIGGER_TOKEN}\n"
        reason = "monitor_trigger_boundary_before_unsafe_certificate"
    else:
        safe_prefix = _prefix_before_first_json(output_text)
        reason = "monitor_error_before_json_boundary"

    token_offset = int(selected.get("token_offset") or 0)
    polling_interval = int(selected.get("polling_interval_tokens") or 0)
    return {
        "case_id": str(source_row.get("case_id") or ""),
        "safe_prefix": safe_prefix,
        "selection_reason": reason,
        "selected_event_id": selected.get("event_id"),
        "selected_event_token_offset": token_offset,
        "last_safe_token_offset": max(0, token_offset - polling_interval),
        "safe_prefix_token_count": _token_count(safe_prefix),
    }


def load_validator_rows(path: Path | str) -> dict[str, JsonDict]:
    """Load compiled safe-DSL validator rows keyed by CCTU case ID."""

    rows = _load_jsonl(Path(path))
    return {
        str(row["prompt_id"]): row
        for row in rows
        if row.get("prompt_id") and row.get("validator_compiled") is True
    }


def resolve_model_specs() -> list[JsonDict]:
    """Resolve mandated local SOTA GGUF specs without legacy small-model fallback."""

    return certificates.resolve_model_specs()


def collect_live_continuations(
    spec: JsonDict,
    plans: list[JsonDict],
    *,
    resolver: Callable[[str], str | None] | None = None,
    llama_importer: Callable[[], tuple[bool, type[Any] | None, str | None]] | None = None,
    env_preparer: Callable[[], JsonDict] | None = None,
) -> JsonDict:
    """Ask one mandated local GGUF model for safe-prefix and full-regeneration rows."""

    hf_id = str(spec.get("hf_id") or "")
    model_path = spec.get("model_path") or (resolver or certificates._resolve_cached_gguf)(hf_id)
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

    env_details = (env_preparer or certificates.cctu.prepare_llama_environment)()
    ok, llama_class, import_error = (
        llama_importer or certificates.cctu._default_llama_importer
    )()
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
            n_ctx=8192,
            seed=1496,
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
        for plan in plans:
            for mode, prompt, max_tokens in (
                (SAFE_PREFIX_MODE, build_safe_prefix_prompt(plan), 256),
                (FULL_REGENERATION_MODE, plan["original_prompt"], 384),
            ):
                started = time.monotonic()
                try:
                    result = llm(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        stop=["</s>", "<eos>"],
                        echo=False,
                    )
                    generated_text = certificates.cctu._completion_text(result)
                    blocker = None if generated_text.strip() else "empty_generation"
                except Exception as exc:
                    generated_text = ""
                    blocker = f"{type(exc).__name__}: {exc}"
                output_text = (
                    f"{plan['safe_prefix']}{generated_text}"
                    if mode == SAFE_PREFIX_MODE
                    else generated_text
                )
                rows.append(
                    {
                        "case_id": plan["case_id"],
                        "mode": mode,
                        "prompt": prompt,
                        "model_hf_id": hf_id,
                        "model_name": spec.get("name"),
                        "model_path": str(model_path),
                        "generation_source": "live_sota_llamacpp",
                        "generated_text": generated_text,
                        "output_text": output_text,
                        "elapsed_seconds": round(time.monotonic() - started, 6),
                        "blocker": blocker,
                    }
                )
    finally:
        certificates.cctu._close_llama(llm)

    model_used = any(row.get("blocker") is None for row in rows)
    return {
        "summary": {
            "hf_id": hf_id,
            "model_name": spec.get("name"),
            "model_path": str(model_path),
            "model_used": model_used,
            "blocker": None if model_used else "no_usable_continuations",
            "env_details": env_details,
        },
        "rows": rows,
    }


def build_safe_prefix_prompt(plan: JsonDict) -> str:
    """Build the continuation prompt that preserves the deterministic prefix boundary."""

    return (
        "Continue the CCTU certificate from the verified prefix only.\n"
        "Do not repeat or alter the verified prefix. Emit only the JSON suffix needed "
        "after the trigger token. Use the original task constraints exactly.\n\n"
        "Original task:\n"
        f"{plan['original_prompt']}\n\n"
        "Verified prefix:\n"
        f"{plan['safe_prefix']}"
    )


def _build_evaluation_rows(
    plans: list[JsonDict],
    validator_rows: dict[str, JsonDict],
    generation_rows: list[JsonDict],
) -> list[JsonDict]:
    generation_by_key = {
        (str(row.get("case_id") or ""), str(row.get("mode") or "")): row
        for row in generation_rows
    }
    rows: list[JsonDict] = []
    for plan in plans:
        rows.append(
            _evaluation_row(
                plan,
                mode=NO_CONTINUATION_MODE,
                generation_row=plan["source_row"],
                validator_row=validator_rows.get(plan["case_id"]),
            )
        )
        for mode in (SAFE_PREFIX_MODE, FULL_REGENERATION_MODE):
            generation_row = generation_by_key.get((plan["case_id"], mode))
            if generation_row is None:
                generation_row = {"case_id": plan["case_id"], "mode": mode, "blocker": "missing_generation"}
            rows.append(
                _evaluation_row(
                    plan,
                    mode=mode,
                    generation_row=generation_row,
                    validator_row=validator_rows.get(plan["case_id"]),
                )
            )
    return rows


def _evaluation_row(
    plan: JsonDict,
    *,
    mode: str,
    generation_row: JsonDict,
    validator_row: JsonDict | None,
) -> JsonDict:
    case = plan["case"]
    output_text = _output_text_for_mode(plan, mode, generation_row)
    parser_result = certificates.parse_certificate_output(
        output_text,
        lane=certificates.TRIGGER_LANE,
    )
    validation = certificates.validate_certificate(case, parser_result.get("certificate_json"))
    compiled_result = _run_compiled_validator(validator_row, parser_result, output_text)
    cctu_passed = bool(validation["verifier_result"]["accepted"])
    compiled_available = validator_row is not None
    compiled_accepted = bool(compiled_result.get("accepted"))
    final_passed = bool(cctu_passed and compiled_available and compiled_accepted)
    blocker = generation_row.get("blocker")
    if blocker is not None:
        final_passed = False

    return {
        "schema_version": 1,
        "case_id": plan["case_id"],
        "family": plan["family"],
        "mode": mode,
        "selected_event_id": plan.get("selected_event_id"),
        "selection_reason": plan.get("selection_reason"),
        "last_safe_token_offset": plan.get("last_safe_token_offset"),
        "safe_prefix": plan.get("safe_prefix"),
        "safe_prefix_token_count": plan.get("safe_prefix_token_count"),
        "prompt": generation_row.get("prompt") or plan.get("original_prompt"),
        "model_hf_id": generation_row.get("model_hf_id"),
        "model_name": generation_row.get("model_name"),
        "generation_source": generation_row.get("generation_source"),
        "elapsed_seconds": generation_row.get("elapsed_seconds"),
        "blocker": blocker,
        "output_text": output_text,
        "parser_result": parser_result,
        "cctu_validator_result": validation["validator_result"],
        "cctu_verifier_result": validation["verifier_result"],
        "compiled_validator_available": compiled_available,
        "compiled_validator_result": compiled_result,
        "final_validator_passed": final_passed,
        "verifier_false_accept": bool(validation["verifier_result"]["false_accept"]),
    }


def _run_compiled_validator(
    validator_row: JsonDict | None,
    parser_result: JsonDict,
    output_text: str,
) -> JsonDict:
    if validator_row is None:
        return {"accepted": False, "reason": "compiled_validator_missing"}
    compiled = compiler.CompiledValidator(
        prompt_id=str(validator_row.get("prompt_id") or ""),
        compiled=True,
        dsl=dict(validator_row.get("compiled_validator") or {}),
        manual_review_required=bool(validator_row.get("manual_review_required")),
    )
    certificate_json = parser_result.get("certificate_json")
    candidate_output = (
        json.dumps(certificate_json, sort_keys=True)
        if isinstance(certificate_json, dict)
        else output_text
    )
    return dict(compiler.evaluate_compiled_validator(compiled, candidate_output))


def _output_text_for_mode(plan: JsonDict, mode: str, generation_row: JsonDict) -> str:
    if mode == NO_CONTINUATION_MODE:
        return str(generation_row.get("model_output") or generation_row.get("output_text") or "")
    output_text = generation_row.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    generated_text = str(generation_row.get("generated_text") or "")
    if mode == SAFE_PREFIX_MODE:
        return f"{plan['safe_prefix']}{generated_text}"
    return generated_text


def _terminal_artifact(
    *,
    run_date: str,
    manifest_path: Path,
    rows: list[JsonDict],
    plans: list[JsonDict],
    model_attempts: list[JsonDict],
    gpu_probe: JsonDict,
    blockers: list[str],
    tests_run: list[str] | None,
) -> JsonDict:
    baseline_rate = _pass_rate(rows, NO_CONTINUATION_MODE)
    safe_rate = _pass_rate(rows, SAFE_PREFIX_MODE)
    full_rate = _pass_rate(rows, FULL_REGENERATION_MODE)
    false_accept_rate = _false_accept_rate(rows) if rows else None
    continuations_completed = sum(
        row["mode"] == SAFE_PREFIX_MODE
        and row.get("blocker") is None
        and row.get("model_hf_id") in MANDATED_MODEL_IDS
        for row in rows
    )
    models_used = [
        str(summary["hf_id"])
        for summary in model_attempts
        if summary.get("model_used") is True and summary.get("hf_id") in MANDATED_MODEL_IDS
    ]
    live_used = bool(models_used and continuations_completed > 0)
    metrics_present = all(
        metric is not None for metric in (baseline_rate, safe_rate, full_rate, false_accept_rate)
    )
    ready = bool(live_used and continuations_completed > 0 and metrics_present)
    terminal_blockers = list(blockers)
    if plans and not live_used:
        terminal_blockers.append("live_sota_continuation_unavailable")
    terminal_blockers = list(dict.fromkeys(terminal_blockers))
    status = "complete" if ready else "blocked"
    return {
        "status": status,
        "run_date": run_date,
        "schema_version": 1,
        "model_specs": [spec["hf_id"] for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": live_used,
        "safe_prefix_continuation_ready": ready,
        "cases_attempted": len(plans),
        "continuations_completed": int(continuations_completed),
        "baseline_validator_pass_rate": baseline_rate,
        "safe_prefix_validator_pass_rate": safe_rate,
        "full_regeneration_validator_pass_rate": full_rate,
        "verifier_false_accept_rate": false_accept_rate,
        "last_safe_prefix_selection_rule": LAST_SAFE_PREFIX_SELECTION_RULE,
        "continuation_manifest_path": _display_path(manifest_path),
        "models_used": models_used,
        "gpu_probe": gpu_probe,
        "blockers": terminal_blockers,
        "honest_verdict": _honest_verdict(
            ready=ready,
            baseline_rate=baseline_rate,
            safe_rate=safe_rate,
            false_accept_rate=false_accept_rate,
        ),
        "model_attempts": model_attempts,
        "manifest_rows": len(rows),
        "tests_run": list(tests_run or []),
    }


def _honest_verdict(
    *,
    ready: bool,
    baseline_rate: float | None,
    safe_rate: float | None,
    false_accept_rate: float | None,
) -> str:
    if not ready:
        return "complete: HoVer safe-prefix continuation audit blocked before headline metrics"
    if (
        safe_rate is not None
        and baseline_rate is not None
        and false_accept_rate == 0.0
        and safe_rate > baseline_rate
    ):
        return "complete: safe-prefix continuation improved matched validator pass rate with zero false accepts"
    if safe_rate is not None and baseline_rate is not None and false_accept_rate == 0.0:
        return "complete: safe-prefix continuation measured without increasing verifier false accepts"
    return "complete: safe-prefix continuation measured but false-accept risk remains unresolved"


def _pass_rate(rows: list[JsonDict], mode: str) -> float | None:
    mode_rows = [row for row in rows if row.get("mode") == mode]
    if not mode_rows:
        return None
    return round(sum(bool(row.get("final_validator_passed")) for row in mode_rows) / len(mode_rows), 6)


def _false_accept_rate(rows: list[JsonDict]) -> float:
    invalid_rows = [
        row for row in rows if not bool(row.get("cctu_verifier_result", {}).get("base_valid"))
    ]
    if not invalid_rows:
        return 0.0
    false_accepts = sum(bool(row.get("verifier_false_accept")) for row in invalid_rows)
    return round(false_accepts / len(invalid_rows), 6)


def _event_interrupts(event: JsonDict) -> bool:
    return bool(event.get("interruption_triggered") or event.get("error_detected"))


def _prefix_before_first_json(text: str) -> str:
    index = text.find("{")
    if index < 0:
        return text
    return text[:index]


def _token_count(text: str) -> int:
    return len(text.split())


def _load_json_if_exists(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _display_path(path: Path | str) -> str:
    as_path = Path(path)
    try:
        return str(as_path.resolve().relative_to(certificates.cctu._repo_root()))
    except ValueError:
        return str(as_path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for conductor and manual experiment runs."""

    _ = list([] if argv is None else argv)
    artifact = run_experiment()
    print(
        "[exp1496] "
        f"ready={artifact['safe_prefix_continuation_ready']} "
        f"baseline={artifact['baseline_validator_pass_rate']} "
        f"safe_prefix={artifact['safe_prefix_validator_pass_rate']} "
        f"false_accept={artifact['verifier_false_accept_rate']} "
        f"verdict={artifact['honest_verdict']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by conductor.
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_MANIFEST_PATH",
    "EVALUATION_MODES",
    "FULL_REGENERATION_MODE",
    "LAST_SAFE_PREFIX_SELECTION_RULE",
    "MANDATED_MODEL_SPECS",
    "NO_CONTINUATION_MODE",
    "REQUIRED_ARTIFACT_FIELDS",
    "SAFE_PREFIX_MODE",
    "build_case_plans",
    "build_safe_prefix_prompt",
    "collect_live_continuations",
    "gated_input_blockers",
    "load_validator_rows",
    "main",
    "resolve_model_specs",
    "run_experiment",
    "select_last_safe_prefix",
    "write_in_progress_artifact",
]
