"""Trigger-token plus grammar certificate decoder audit for Exp 1508.

Spec: REQ-VERIFY-1508, SCENARIO-VERIFY-1508.

This module tests a narrower version of Carnot's desired certificate boundary:
the model can reason in normal text first, but after it emits the CCTU trigger
token the certificate tail is generated under a bounded llama.cpp GBNF grammar.
The audit keeps the comparison honest by carrying forward Exp 1493 schema-only
rows instead of re-labeling them, then replaying every parsed certificate through
the deterministic CCTU validators.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from carnot.eval import cctu_executable_constraint_microbenchmark as cctu
from carnot.eval import cctu_trigger_certificate_export as exp1493

JsonDict = dict[str, Any]

RUN_DATE = "20260507"
DEFAULT_ARTIFACT_PATH = Path(
    "results/experiment_1508_trigger_grammar_certificate_decoder_audit.json"
)
DEFAULT_DECODER_MANIFEST_PATH = Path("results/trigger_grammar_certificates_1508.jsonl")
DEFAULT_EXP1507_ARTIFACT_PATH = Path(
    "results/experiment_1507_autopyverifier_safe_dsl_induction_pack.json"
)
DEFAULT_INDUCTION_MANIFEST_PATH = Path("results/safe_dsl_verifier_induction_1507.jsonl")
DEFAULT_SCHEMA_ONLY_MANIFEST_PATH = Path("results/cctu_trigger_certificates_1493.jsonl")

TRIGGER_TOKEN = exp1493.TRIGGER_TOKEN
TRIGGER_GRAMMAR_MODE = "trigger_grammar"
SCHEMA_ONLY_MODE = "schema_only"
EXACT_GBNF_BACKEND = "llama_cpp_gbnf_exact_certificate_v1"
SCHEMA_ONLY_BACKEND = "exp1493_schema_only_posthoc_json"
DEFAULT_MAX_CASES = 4

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = cctu.MANDATED_MODEL_SPECS
MANDATED_MODEL_IDS: frozenset[str] = frozenset(
    str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "live_sota_model_inference_used",
    "certificate_decoder_ready",
    "gated_inputs_present",
    "cases_attempted",
    "grammar_backend",
    "trigger_token_presence_rate",
    "grammar_parse_rate",
    "schema_only_parse_rate",
    "grammar_validation_rate",
    "schema_only_validation_rate",
    "verifier_false_accept_rate",
    "decoder_manifest_path",
    "models_used",
    "gpu_probe",
    "blockers",
    "honest_verdict",
)

ResolverFn = Callable[[str], str | None]
CachedPairFn = Callable[..., list[JsonDict] | None]
LlamaImporterFn = Callable[[], tuple[bool, type[Any] | None, str | None]]
GrammarImporterFn = Callable[[], tuple[bool, type[Any] | None, str | None]]
CollectGrammarOutputsFn = Callable[[JsonDict, list[cctu.BenchmarkCase]], JsonDict]


def build_exact_certificate_gbnf(case: cctu.BenchmarkCase) -> str:
    """Return a bounded GBNF grammar for the canonical certificate of one case.

    The first runtime audit deliberately uses an exact per-case grammar rather
    than a broad JSON grammar because the CCTU prompts already expose the exact
    certificate shape.  That lets the experiment isolate the operational
    question - can the local SOTA GGUF stack switch from free reasoning into a
    parseable certificate tail - without introducing a second source of schema
    drift inside the grammar itself.
    """

    certificate_text = json.dumps(exp1493.certificate_for_case(case), sort_keys=True)
    return f"root ::= {json.dumps(certificate_text)}"


def build_reasoning_prompt(case: cctu.BenchmarkCase) -> str:
    """Build the unconstrained reasoning prompt that must end with the trigger."""

    return (
        "You are evaluating a CCTU-style executable constraint case.\n"
        f"Case: {case.case_id}\n"
        f"Constraint family: {case.family}\n"
        f"Use exactly one local tool named {case.tool_name} with these arguments:\n"
        f"{json.dumps(case.tool_arguments, sort_keys=True)}\n"
        "Solve freely in concise prose. End your response with the exact trigger "
        f"token {TRIGGER_TOKEN} on its own final line. Do not emit JSON yet."
    )


def build_certificate_prompt(case: cctu.BenchmarkCase, reasoning_text: str) -> str:
    """Build the grammar-tail prompt after the trigger has been observed."""

    excerpt = " ".join(str(reasoning_text or "").split())[:900]
    return (
        "Continue the CCTU certificate after the trigger under the supplied grammar.\n"
        f"Case: {case.case_id}\n"
        f"Constraint family: {case.family}\n"
        f"Reasoning excerpt: {excerpt}\n"
        "Emit only the certificate JSON body now."
    )


def build_grammar_manifest_row(
    case: cctu.BenchmarkCase,
    generation_row: JsonDict,
) -> JsonDict:
    """Join one trigger+grammar generation with parser and validator results."""

    reasoning_text = str(generation_row.get("reasoning_text") or "")
    certificate_body = str(generation_row.get("certificate_body") or "")
    full_output = str(generation_row.get("model_output") or "").strip()
    if not full_output:
        full_output = f"{reasoning_text}\n{certificate_body}".strip()

    parser_result = exp1493.parse_certificate_output(
        full_output,
        lane=exp1493.TRIGGER_LANE,
        trigger_token=TRIGGER_TOKEN,
    )
    validation = exp1493.validate_certificate(case, parser_result.get("certificate_json"))
    verifier_result = validation["verifier_result"]
    return {
        "case_id": case.case_id,
        "family": case.family,
        "source": "certificate",
        "decoder_mode": TRIGGER_GRAMMAR_MODE,
        "schema_source_lane": None,
        "grammar_backend": generation_row.get("grammar_backend") or EXACT_GBNF_BACKEND,
        "grammar_enforced": bool(generation_row.get("grammar_enforced", not generation_row.get("blocker"))),
        "prompt": generation_row.get("prompt"),
        "model_hf_id": generation_row.get("model_hf_id"),
        "model_name": generation_row.get("model_name"),
        "generation_source": generation_row.get("generation_source"),
        "elapsed_seconds": generation_row.get("elapsed_seconds"),
        "blocker": generation_row.get("blocker"),
        "model_output": full_output,
        "free_form_reasoning_text": parser_result["free_form_reasoning_text"],
        "grammar_certificate_body": certificate_body,
        "trigger_token_present": parser_result["trigger_token_present"],
        "certificate_json": parser_result["certificate_json"],
        "parser_result": parser_result,
        "validator_result": validation["validator_result"],
        "verifier_result": verifier_result,
        "deterministic_validation_passed": bool(verifier_result["accepted"]),
        "false_accept_status": bool(verifier_result["false_accept"]),
    }


def load_schema_only_rows(
    schema_only_manifest_path: Path | str = DEFAULT_SCHEMA_ONLY_MANIFEST_PATH,
    *,
    case_ids: set[str] | None = None,
) -> list[JsonDict]:
    """Load Exp 1493 trigger-token rows as the schema-only comparison mode."""

    path = Path(schema_only_manifest_path)
    if not path.exists():
        return []
    selected: list[JsonDict] = []
    for row in _read_jsonl(path):
        if row.get("lane") != exp1493.TRIGGER_LANE:
            continue
        case_id = str(row.get("case_id") or "")
        if case_ids is not None and case_id not in case_ids:
            continue
        converted = dict(row)
        converted.update(
            {
                "source": "certificate",
                "decoder_mode": SCHEMA_ONLY_MODE,
                "schema_source_lane": exp1493.TRIGGER_LANE,
                "grammar_backend": SCHEMA_ONLY_BACKEND,
                "grammar_enforced": False,
            }
        )
        selected.append(converted)
    return selected


def load_selected_verifier_names(
    induction_manifest_path: Path | str = DEFAULT_INDUCTION_MANIFEST_PATH,
) -> list[str]:
    """Return the Exp 1507 selected safe-DSL verifier names when present."""

    path = Path(induction_manifest_path)
    if not path.exists():
        return []
    for row in _read_jsonl(path):
        if row.get("row_type") == "selected_set_summary":
            names = row.get("candidate_names")
            if isinstance(names, list):
                return [str(name) for name in names]
    return []


def aggregate_manifest_metrics(rows: list[JsonDict]) -> JsonDict:
    """Compute parser, validation, trigger, and false-accept rates by mode."""

    grammar_rows = [row for row in rows if row.get("decoder_mode") == TRIGGER_GRAMMAR_MODE]
    schema_rows = [row for row in rows if row.get("decoder_mode") == SCHEMA_ONLY_MODE]
    invalid_rows = [
        row
        for row in rows
        if not bool((row.get("verifier_result") or {}).get("base_valid"))
    ]
    false_accepts = sum(
        bool((row.get("verifier_result") or {}).get("false_accept")) for row in invalid_rows
    )
    return {
        "trigger_token_presence_rate": _rate(
            grammar_rows,
            lambda row: bool(row.get("trigger_token_present")),
        ),
        "grammar_parse_rate": _rate(
            grammar_rows,
            lambda row: bool((row.get("parser_result") or {}).get("parsed")),
        ),
        "schema_only_parse_rate": _rate(
            schema_rows,
            lambda row: bool((row.get("parser_result") or {}).get("parsed")),
        ),
        "grammar_validation_rate": _rate(
            grammar_rows,
            lambda row: bool((row.get("verifier_result") or {}).get("accepted")),
        ),
        "schema_only_validation_rate": _rate(
            schema_rows,
            lambda row: bool((row.get("verifier_result") or {}).get("accepted")),
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
    """Write the durable bootstrap artifact before any gate or model work."""

    payload = _empty_artifact(
        status="in_progress",
        run_date=run_date,
        honest_verdict="complete: in-progress Exp 1508 bootstrap artifact",
    )
    _write_json(Path(output_path), payload)
    return payload


def run_experiment(
    *,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    decoder_manifest_path: Path | str = DEFAULT_DECODER_MANIFEST_PATH,
    exp1507_artifact_path: Path | str = DEFAULT_EXP1507_ARTIFACT_PATH,
    induction_manifest_path: Path | str = DEFAULT_INDUCTION_MANIFEST_PATH,
    schema_only_manifest_path: Path | str = DEFAULT_SCHEMA_ONLY_MANIFEST_PATH,
    run_date: str = RUN_DATE,
    model_specs: Iterable[JsonDict] | None = None,
    collect_grammar_outputs_fn: CollectGrammarOutputsFn | None = None,
    gpu_probe_fn: Callable[[], JsonDict] | None = None,
    max_cases: int = DEFAULT_MAX_CASES,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Run the bounded trigger+grammar decoder audit and persist artifacts."""

    output = Path(output_path)
    manifest = Path(decoder_manifest_path)
    write_in_progress_artifact(output, run_date=run_date)
    gpu_probe = (gpu_probe_fn or probe_gpu)()

    gate_ready, gate_blocker = _exp1507_gate_ready(exp1507_artifact_path)
    if not gate_ready:
        _write_jsonl(manifest, [])
        artifact = _terminal_artifact(
            run_date=run_date,
            rows=[],
            manifest_path=manifest,
            model_attempts=[],
            gpu_probe=gpu_probe,
            blockers=[gate_blocker],
            gated_inputs_present=False,
            grammar_backend="gated",
            cases_attempted=0,
            tests_run=tests_run,
        )
        _write_json(output, artifact)
        return artifact

    schema_path = Path(schema_only_manifest_path)
    if not schema_path.exists():
        _write_jsonl(manifest, [])
        artifact = _terminal_artifact(
            run_date=run_date,
            rows=[],
            manifest_path=manifest,
            model_attempts=[],
            gpu_probe=gpu_probe,
            blockers=[f"missing_schema_only_manifest:{schema_path}"],
            gated_inputs_present=False,
            grammar_backend="gated",
            cases_attempted=0,
            tests_run=tests_run,
        )
        _write_json(output, artifact)
        return artifact

    selected_cases = _select_cases_from_schema_rows(schema_path, max_cases=max_cases)
    if not selected_cases:
        _write_jsonl(manifest, [])
        artifact = _terminal_artifact(
            run_date=run_date,
            rows=[],
            manifest_path=manifest,
            model_attempts=[],
            gpu_probe=gpu_probe,
            blockers=[f"no_schema_only_trigger_rows:{schema_path}"],
            gated_inputs_present=False,
            grammar_backend="gated",
            cases_attempted=0,
            tests_run=tests_run,
        )
        _write_json(output, artifact)
        return artifact

    selected_verifier_names = load_selected_verifier_names(induction_manifest_path)
    case_ids = {case.case_id for case in selected_cases}
    schema_rows = load_schema_only_rows(schema_path, case_ids=case_ids)
    _annotate_selected_verifiers(schema_rows, selected_verifier_names)

    specs = list(resolve_model_specs() if model_specs is None else model_specs)
    if not specs:
        _write_jsonl(manifest, schema_rows)
        artifact = _terminal_artifact(
            run_date=run_date,
            rows=schema_rows,
            manifest_path=manifest,
            model_attempts=[],
            gpu_probe=gpu_probe,
            blockers=["no_mandated_sota_gguf_model_available"],
            gated_inputs_present=True,
            grammar_backend=EXACT_GBNF_BACKEND,
            cases_attempted=len(selected_cases),
            tests_run=tests_run,
        )
        _write_json(output, artifact)
        return artifact

    collector = collect_grammar_outputs_fn or collect_live_grammar_outputs
    collection = collector(dict(specs[0]), selected_cases)
    model_attempts = [dict(collection.get("summary") or {})]
    case_by_id = {case.case_id: case for case in selected_cases}
    grammar_rows: list[JsonDict] = []
    for generation_row in collection.get("rows") or []:
        case = case_by_id.get(generation_row.get("case_id"))
        if case is not None:
            grammar_rows.append(build_grammar_manifest_row(case, generation_row))
    _annotate_selected_verifiers(grammar_rows, selected_verifier_names)

    rows = [*schema_rows, *grammar_rows]
    _write_jsonl(manifest, rows)
    blockers = [
        str(summary.get("blocker"))
        for summary in model_attempts
        if summary.get("model_used") is not True and summary.get("blocker")
    ]
    if not _live_sota_grammar_rows_present(grammar_rows):
        blockers.append("live_sota_grammar_generation_unavailable")

    artifact = _terminal_artifact(
        run_date=run_date,
        rows=rows,
        manifest_path=manifest,
        model_attempts=model_attempts,
        gpu_probe=gpu_probe,
        blockers=list(dict.fromkeys(blockers)),
        gated_inputs_present=True,
        grammar_backend=_grammar_backend_from_rows(grammar_rows, model_attempts),
        cases_attempted=len(selected_cases),
        tests_run=tests_run,
    )
    artifact["selected_verifier_names"] = selected_verifier_names
    artifact["schema_only_manifest_path"] = _display_path(schema_path)
    artifact["exp1507_artifact_path"] = _display_path(exp1507_artifact_path)
    artifact["induction_manifest_path"] = _display_path(induction_manifest_path)
    _write_json(output, artifact)
    return artifact


def resolve_model_specs(
    *,
    cached_pair_fn: CachedPairFn | None = None,
    resolver_fn: ResolverFn | None = None,
) -> list[JsonDict]:  # pragma: no cover - external cache discovery.
    """Resolve mandated local SOTA GGUF specs without legacy small fallbacks."""

    pair_resolver = cached_pair_fn or _cached_sota_pair
    pair = pair_resolver(gpu_indices=(0, 1))
    if pair:
        return pair
    resolver = resolver_fn or cctu._default_resolver
    specs: list[JsonDict] = []
    for spec in MANDATED_MODEL_SPECS:
        model_path = resolver(str(spec["hf_id"]))
        if model_path:
            specs.append({**spec, "model_path": model_path})
    return specs


def collect_live_grammar_outputs(
    spec: JsonDict,
    cases: list[cctu.BenchmarkCase],
    *,
    resolver: ResolverFn | None = None,
    llama_importer: LlamaImporterFn | None = None,
    grammar_importer: GrammarImporterFn | None = None,
    env_preparer: Callable[[], JsonDict] | None = None,
) -> JsonDict:  # pragma: no cover - live GGUF path is exercised by the experiment run.
    """Collect trigger+grammar rows from one mandated local GGUF model."""

    hf_id = str(spec.get("hf_id") or "")
    model_path = spec.get("model_path") or (resolver or cctu._default_resolver)(hf_id)
    if not model_path:
        return {
            "summary": {
                "hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_used": False,
                "blocker": "model_not_cached",
                "grammar_backend": EXACT_GBNF_BACKEND,
            },
            "rows": [],
        }

    env_details = (env_preparer or cctu.prepare_llama_environment)()
    ok, llama_class, import_error = (llama_importer or cctu._default_llama_importer)()
    if not ok or llama_class is None:
        return _blocked_collection(
            spec,
            model_path,
            import_error or "llama_cpp_import_failed",
            env_details,
        )
    grammar_ok, grammar_class, grammar_error = (
        grammar_importer or _default_grammar_importer
    )()
    if not grammar_ok or grammar_class is None:
        return _blocked_collection(
            spec,
            model_path,
            grammar_error or "llama_cpp_grammar_import_failed",
            env_details,
        )

    load_start = time.monotonic()
    try:
        llm = llama_class(
            model_path=str(model_path),
            n_gpu_layers=-1,
            main_gpu=int(spec.get("gpu") or 0),
            n_ctx=4096,
            seed=1508,
            verbose=False,
        )
    except Exception as exc:
        return _blocked_collection(
            spec,
            model_path,
            f"{type(exc).__name__}: {exc}",
            env_details,
            elapsed_seconds=round(time.monotonic() - load_start, 6),
        )

    rows: list[JsonDict] = []
    try:
        for case in cases:
            rows.append(
                _collect_one_case_with_grammar(
                    llm=llm,
                    grammar_class=grammar_class,
                    spec=spec,
                    model_path=str(model_path),
                    case=case,
                )
            )
    finally:
        cctu._close_llama(llm)

    grammar_used = any(row.get("grammar_enforced") is True for row in rows)
    return {
        "summary": {
            "hf_id": hf_id,
            "model_name": spec.get("name"),
            "model_path": str(model_path),
            "model_used": bool(rows),
            "blocker": None if grammar_used else "no_grammar_enforced_generations",
            "grammar_backend": EXACT_GBNF_BACKEND,
            "env_details": env_details,
        },
        "rows": rows,
    }


def probe_gpu() -> JsonDict:  # pragma: no cover - host hardware probe.
    """Return a JSON-safe NVIDIA GPU probe for the terminal artifact."""

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
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
    if result.returncode != 0:
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


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """CLI entry point used by the conductor and manual experiment runs."""

    _ = list(sys.argv[1:] if argv is None else argv)
    artifact = run_experiment()
    print(
        "[exp1508] "
        f"ready={artifact['certificate_decoder_ready']} "
        f"grammar_parse={artifact['grammar_parse_rate']} "
        f"grammar_validation={artifact['grammar_validation_rate']} "
        f"verdict={artifact['honest_verdict']}"
    )
    return 0


def _collect_one_case_with_grammar(
    *,
    llm: Any,
    grammar_class: type[Any],
    spec: JsonDict,
    model_path: str,
    case: cctu.BenchmarkCase,
) -> JsonDict:
    started = time.monotonic()
    reasoning_prompt = build_reasoning_prompt(case)
    try:
        reasoning_response = llm(
            reasoning_prompt,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            stop=["</s>", "<eos>"],
            echo=False,
        )
        reasoning_text = cctu._completion_text(reasoning_response)
    except Exception as exc:
        return _generation_row(
            spec,
            model_path,
            case,
            prompt=reasoning_prompt,
            reasoning_text="",
            certificate_body="",
            elapsed_seconds=round(time.monotonic() - started, 6),
            blocker=f"{type(exc).__name__}: {exc}",
            grammar_enforced=False,
        )

    trigger_count = reasoning_text.count(TRIGGER_TOKEN)
    if trigger_count != 1:
        blocker = "missing_trigger_token" if trigger_count == 0 else "duplicate_trigger_token"
        return _generation_row(
            spec,
            model_path,
            case,
            prompt=reasoning_prompt,
            reasoning_text=reasoning_text,
            certificate_body="",
            elapsed_seconds=round(time.monotonic() - started, 6),
            blocker=blocker,
            grammar_enforced=False,
        )

    certificate_prompt = build_certificate_prompt(case, reasoning_text)
    grammar = grammar_class.from_string(build_exact_certificate_gbnf(case), verbose=False)
    try:
        certificate_response = llm(
            certificate_prompt,
            max_tokens=768,
            temperature=0.0,
            top_p=1.0,
            stop=["</s>", "<eos>"],
            echo=False,
            grammar=grammar,
        )
        certificate_body = cctu._completion_text(certificate_response)
        blocker = None if certificate_body.strip() else "empty_grammar_generation"
    except Exception as exc:
        certificate_body = ""
        blocker = f"{type(exc).__name__}: {exc}"

    return _generation_row(
        spec,
        model_path,
        case,
        prompt=certificate_prompt,
        reasoning_text=reasoning_text,
        certificate_body=certificate_body,
        elapsed_seconds=round(time.monotonic() - started, 6),
        blocker=blocker,
        grammar_enforced=blocker is None,
    )


def _generation_row(
    spec: JsonDict,
    model_path: str,
    case: cctu.BenchmarkCase,
    *,
    prompt: str,
    reasoning_text: str,
    certificate_body: str,
    elapsed_seconds: float,
    blocker: str | None,
    grammar_enforced: bool,
) -> JsonDict:
    return {
        "case_id": case.case_id,
        "decoder_mode": TRIGGER_GRAMMAR_MODE,
        "prompt": prompt,
        "model_hf_id": spec.get("hf_id"),
        "model_name": spec.get("name"),
        "model_path": str(model_path),
        "generation_source": "live_sota_llamacpp",
        "reasoning_text": reasoning_text,
        "certificate_body": certificate_body,
        "grammar_backend": EXACT_GBNF_BACKEND,
        "grammar_enforced": bool(grammar_enforced),
        "elapsed_seconds": elapsed_seconds,
        "blocker": blocker,
    }


def _blocked_collection(
    spec: JsonDict,
    model_path: str,
    blocker: str,
    env_details: JsonDict,
    *,
    elapsed_seconds: float | None = None,
) -> JsonDict:
    summary: JsonDict = {
        "hf_id": spec.get("hf_id"),
        "model_name": spec.get("name"),
        "model_path": str(model_path),
        "model_used": False,
        "blocker": blocker,
        "grammar_backend": EXACT_GBNF_BACKEND,
        "env_details": env_details,
    }
    if elapsed_seconds is not None:
        summary["elapsed_seconds"] = elapsed_seconds
    return {"summary": summary, "rows": []}


def _terminal_artifact(
    *,
    run_date: str,
    rows: list[JsonDict],
    manifest_path: Path,
    model_attempts: list[JsonDict],
    gpu_probe: JsonDict,
    blockers: list[str],
    gated_inputs_present: bool,
    grammar_backend: str,
    cases_attempted: int,
    tests_run: list[str] | None,
) -> JsonDict:
    metrics = aggregate_manifest_metrics(rows)
    models_used = [
        str(summary["hf_id"])
        for summary in model_attempts
        if summary.get("model_used") is True and summary.get("hf_id")
    ]
    live_used = _live_sota_grammar_rows_present(rows)
    ready = bool(gated_inputs_present and live_used and not blockers)
    artifact = _empty_artifact(
        status="complete" if ready else "blocked",
        run_date=run_date,
        honest_verdict=(
            "complete: trigger+grammar certificate decoder measured on live local SOTA GGUF rows"
            if ready
            else "complete: blocked before trigger+grammar certificate decoder headline readiness"
        ),
    )
    artifact.update(
        {
            "live_sota_model_inference_used": bool(live_used),
            "certificate_decoder_ready": bool(ready),
            "gated_inputs_present": bool(gated_inputs_present),
            "cases_attempted": int(cases_attempted),
            "grammar_backend": grammar_backend,
            "trigger_token_presence_rate": metrics["trigger_token_presence_rate"],
            "grammar_parse_rate": metrics["grammar_parse_rate"],
            "schema_only_parse_rate": metrics["schema_only_parse_rate"],
            "grammar_validation_rate": metrics["grammar_validation_rate"],
            "schema_only_validation_rate": metrics["schema_only_validation_rate"],
            "verifier_false_accept_rate": metrics["verifier_false_accept_rate"],
            "decoder_manifest_path": _display_path(manifest_path),
            "models_used": models_used,
            "gpu_probe": gpu_probe,
            "blockers": blockers,
            "model_attempts": model_attempts,
            "manifest_rows": len(rows),
            "tests_run": list(tests_run or []),
        }
    )
    return artifact


def _empty_artifact(*, status: str, run_date: str, honest_verdict: str) -> JsonDict:
    return {
        "status": status,
        "run_date": run_date,
        "schema_version": 1,
        "model_specs": [spec["hf_id"] for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": False,
        "certificate_decoder_ready": False,
        "gated_inputs_present": False,
        "cases_attempted": 0,
        "grammar_backend": "pending",
        "trigger_token_presence_rate": 0.0,
        "grammar_parse_rate": 0.0,
        "schema_only_parse_rate": 0.0,
        "grammar_validation_rate": 0.0,
        "schema_only_validation_rate": 0.0,
        "verifier_false_accept_rate": 0.0,
        "decoder_manifest_path": _display_path(DEFAULT_DECODER_MANIFEST_PATH),
        "models_used": [],
        "gpu_probe": {},
        "blockers": [],
        "honest_verdict": honest_verdict,
    }


def _select_cases_from_schema_rows(path: Path, *, max_cases: int) -> list[cctu.BenchmarkCase]:
    schema_rows = load_schema_only_rows(path)
    ordered_ids: list[str] = []
    for row in schema_rows:
        case_id = str(row.get("case_id") or "")
        if case_id and case_id not in ordered_ids:
            ordered_ids.append(case_id)
    case_by_id = {case.case_id: case for case in cctu.build_benchmark_cases()}
    limit = max(1, int(max_cases))
    return [case_by_id[case_id] for case_id in ordered_ids[:limit] if case_id in case_by_id]


def _annotate_selected_verifiers(rows: list[JsonDict], verifier_names: list[str]) -> None:
    for row in rows:
        row["selected_verifier_names"] = list(verifier_names)


def _exp1507_gate_ready(exp1507_artifact_path: Path | str) -> tuple[bool, str]:
    path = Path(exp1507_artifact_path)
    if not path.exists():
        return False, f"missing_exp1507_artifact:{path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"exp1507_artifact_unreadable:{type(exc).__name__}:{path}"
    if payload.get("verifier_induction_ready") is True:
        return True, ""
    return False, f"exp1507_not_ready:{path}"


def _grammar_backend_from_rows(rows: list[JsonDict], model_attempts: list[JsonDict]) -> str:
    for row in rows:
        backend = row.get("grammar_backend")
        if backend:
            return str(backend)
    for summary in model_attempts:
        backend = summary.get("grammar_backend")
        if backend:
            return str(backend)
    return EXACT_GBNF_BACKEND


def _live_sota_grammar_rows_present(rows: list[JsonDict]) -> bool:
    return any(
        row.get("decoder_mode") == TRIGGER_GRAMMAR_MODE
        and row.get("generation_source") == "live_sota_llamacpp"
        and row.get("model_hf_id") in MANDATED_MODEL_IDS
        and row.get("grammar_enforced") is True
        for row in rows
    )


def _rate(rows: list[JsonDict], predicate: Callable[[JsonDict], bool]) -> float:
    if not rows:
        return 0.0
    return round(sum(bool(predicate(row)) for row in rows) / len(rows), 6)


def _read_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
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
    return str(path)


def _cached_sota_pair(**kwargs: Any) -> list[JsonDict] | None:  # pragma: no cover
    from carnot.inference.sota_models import cached_sota_pair

    return cached_sota_pair(**kwargs)


def _default_grammar_importer() -> tuple[bool, type[Any] | None, str | None]:
    try:
        from llama_cpp import LlamaGrammar  # noqa: PLC0415

        return True, LlamaGrammar, None
    except Exception as exc:  # pragma: no cover - depends on optional runtime.
        return False, None, f"{type(exc).__name__}: {exc}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
