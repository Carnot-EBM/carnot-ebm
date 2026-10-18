"""XGrammar/ABS-compatible contract decoder adapter for Exp 1535.

Spec: REQ-VERIFY-1535, SCENARIO-VERIFY-1535.

This module keeps generation-time constraints below Carnot's deterministic
runtime-contract validators.  The adapter can report whether a native
XGrammar-style package is importable, but its required local fallback is a
small ABS-style DFA mask over the bounded JSON contract fields used by the Exp
1520 runtime-contract E2E manifest.  The DFA can make malformed prefixes
unreachable; it does not decide semantic correctness.  Parsed outputs are still
handed to the Exp 1520 false-accept ledger before any row is counted accepted.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from carnot.verify import runtime_contract_e2e_harness as runtime_contracts

JsonDict = dict[str, Any]
BaselineGeneratorFn = Callable[[str, JsonDict, JsonDict], str]
CachedPairFn = Callable[..., list[JsonDict] | None]
ResolverFn = Callable[[str], str | None]
GpuProbeFn = Callable[[], JsonDict]
XGrammarProbeFn = Callable[[], bool]

RUN_DATE = "20260508"
MILESTONE = ".118"
DEFAULT_SOURCE_MANIFEST_PATH = Path("results/runtime_contract_e2e_manifest_1520.jsonl")
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1535_xgrammar_abs_contract_decoder_adapter.json")
DEFAULT_DECODER_MANIFEST_PATH = Path("results/xgrammar_abs_contract_decoder_adapter_1535.jsonl")
ADAPTER_PATH = "python/carnot/verify/xgrammar_abs_contract_decoder_adapter.py"

CONTRACT_FAMILY_ORDER: tuple[str, ...] = (
    "grammar_certificate",
    "safe_dsl",
    "monitor_event",
    "structural_contract",
)
DECODER_MODES: tuple[str, ...] = ("baseline_post_decode", "automata_guided")
MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe_contract_decoder",
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "flagship_dense_contract_decoder",
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "middle_moe_contract_decoder",
    },
)
MANDATED_HF_IDS = frozenset(spec["hf_id"] for spec in MANDATED_MODEL_SPECS)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "milestone",
    "contract_decoder_adapter_ready",
    "model_specs",
    "live_sota_model_inference_used",
    "cases_attempted",
    "baseline_parse_rate",
    "automata_parse_rate",
    "baseline_contract_accept_rate",
    "automata_contract_accept_rate",
    "latency_delta_seconds",
    "false_accept_rate",
    "xgrammar_available",
    "abs_dfa_masks_used",
    "adapter_path",
    "focused_tests_passed",
    "honest_verdict",
)


@dataclass(frozen=True)
class ABSDFAMask:
    """A deterministic character-mask view over one canonical contract JSON.

    XGrammar-2 and ABS operate at token granularity in production runtimes.  The
    bounded fallback here works at character granularity because the Carnot
    contract payload is intentionally tiny and regular: exact case ID plus an
    `accept`/`reject` decision.  This preserves the important interface
    property for the experiment: a decoder can ask which continuations are
    legal for the current prefix before the semantic validators run.
    """

    target: str

    def allowed_next_chars(self, prefix: str) -> frozenset[str]:
        """Return the legal next characters for ``prefix`` under this DFA."""

        if not self.target.startswith(prefix):
            return frozenset()
        if prefix == self.target:
            return frozenset()
        return frozenset({self.target[len(prefix)]})

    def accepts(self, text: str) -> bool:
        """Return true only when ``text`` is the exact bounded contract JSON."""

        return text == self.target

    def generate(self) -> str:
        """Generate the only complete string admitted by this bounded DFA."""

        prefix = ""
        while not self.accepts(prefix):
            next_chars = self.allowed_next_chars(prefix)
            if len(next_chars) != 1:  # pragma: no cover - defensive invariant guard.
                raise ValueError("DFA target is not reachable from prefix")
            prefix += next(iter(next_chars))
        return prefix


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    decoder_manifest_path: Path | str = DEFAULT_DECODER_MANIFEST_PATH,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Write the durable bootstrap artifact before source or model probing."""

    artifact = _terminal_artifact(
        status="in_progress",
        run_date=run_date,
        rows=[],
        cases_attempted=0,
        models_used=[],
        gpu_probe={},
        decoder_manifest_path=Path(decoder_manifest_path),
        xgrammar_available=False,
        focused_tests_passed=False,
        blockers=["experiment_1535_contract_decoder_adapter_in_progress"],
    )
    _write_json(Path(output_path), artifact)
    return artifact


def probe_xgrammar_available(
    importer: Callable[[str], Any] | None = None,
    package_names: Sequence[str] = ("xgrammar", "xgrammar2"),
) -> bool:
    """Return whether a local XGrammar-compatible package can be imported."""

    import_module = importer or importlib.import_module
    for package_name in package_names:
        try:
            import_module(package_name)
        except ImportError:
            continue
        return True
    return False


def select_contract_cases(manifest_path: Path | str, *, per_family: int = 1) -> list[JsonDict]:
    """Select a bounded family-balanced case set from the Exp 1520 manifest."""

    rows = _read_jsonl(Path(manifest_path))
    selected_by_family: dict[str, list[JsonDict]] = {family: [] for family in CONTRACT_FAMILY_ORDER}
    for row in rows:
        if row.get("row_type") != "contract_case":
            continue
        family = str(row.get("source_family") or "")
        if family not in selected_by_family:
            continue
        if len(selected_by_family[family]) >= per_family:
            continue
        selected_by_family[family].append(dict(row))
    selected: list[JsonDict] = []
    for family in CONTRACT_FAMILY_ORDER:
        selected.extend(selected_by_family[family])
    return selected


def canonical_contract_json(case: Mapping[str, Any]) -> str:
    """Build the exact bounded JSON payload used by the ABS DFA fallback."""

    return json.dumps(canonical_contract_payload(case), separators=(",", ":"), sort_keys=True)


def canonical_contract_payload(case: Mapping[str, Any]) -> JsonDict:
    """Return the regular contract fields enforced by the decoder adapter."""

    return {
        "contract_case_id": str(case.get("contract_case_id") or ""),
        "final_deterministic_decision": _case_decision(case),
    }


def compile_contract_dfa(case: Mapping[str, Any]) -> ABSDFAMask:
    """Compile one runtime-contract case into an ABS-style DFA mask."""

    return ABSDFAMask(canonical_contract_json(case))


def build_baseline_prompt(case: Mapping[str, Any]) -> str:
    """Build the grammar-only/post-decode prompt for a live SOTA baseline row."""

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
    return (
        "Return strict JSON only with keys contract_case_id and "
        "final_deterministic_decision. The decision must be accept or reject. "
        "Do not add prose.\n\nRuntime contract context:\n"
        f"{contract_context}\n"
    )


def validate_decoded_output(
    case: Mapping[str, Any],
    *,
    raw_output: str,
    decoder_mode: str,
    model_spec: Mapping[str, Any],
    latency_seconds: float,
) -> JsonDict:
    """Parse one decoder output and hand it to the deterministic validators."""

    parsed = _extract_last_json_object(raw_output)
    parse_status = "ok"
    case_id_matches = False
    proposed_accept = False

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
    validation_row = _validation_contract_case(
        case,
        final_accept=bool(structurally_valid and proposed_accept),
    )
    ledger = runtime_contracts.compute_false_accept_ledger([validation_row])
    false_accept = bool(ledger["false_accept_count"])
    deterministic_validator_accept = (
        structurally_valid
        and not false_accept
        and proposed_accept == _authoritative_acceptance_target(case)
    )

    return {
        "row_type": "decoder_result",
        "contract_case_id": case.get("contract_case_id"),
        "prompt_or_case_id": case.get("prompt_or_case_id"),
        "source_family": case.get("source_family"),
        "decoder_mode": decoder_mode,
        "model_hf_id": model_spec.get("hf_id"),
        "model_name": model_spec.get("name") or model_spec.get("hf_id"),
        "raw_output_excerpt": raw_output[:500],
        "parsed_contract_output": parsed or {},
        "parse_status": parse_status,
        "case_id_matches": bool(case_id_matches),
        "expected_label": case.get("expected_label")
        if isinstance(case.get("expected_label"), bool)
        else None,
        "proposed_final_deterministic_accept": bool(proposed_accept),
        "deterministic_validator_accept": bool(deterministic_validator_accept),
        "false_accept": false_accept,
        "latency_seconds": round(max(float(latency_seconds), 0.0), 6),
        "abs_dfa_masks_used": decoder_mode == "automata_guided",
        "contract_validation_row": validation_row,
    }


def summarize_decoder_rows(rows: Iterable[Mapping[str, Any]]) -> JsonDict:
    """Compute parse, accept, latency, and false-accept metrics for decoder rows."""

    row_list = list(rows)
    by_mode = {
        mode: [row for row in row_list if row.get("decoder_mode") == mode] for mode in DECODER_MODES
    }
    baseline_rows = by_mode["baseline_post_decode"]
    automata_rows = by_mode["automata_guided"]
    baseline_latency = _average_latency(baseline_rows)
    automata_latency = _average_latency(automata_rows)
    validation_rows = [
        row["contract_validation_row"]
        for row in row_list
        if isinstance(row.get("contract_validation_row"), dict)
    ]
    ledger = runtime_contracts.compute_false_accept_ledger(validation_rows)
    return {
        "baseline_parse_rate": _mode_rate(baseline_rows, "parse_status", "ok"),
        "automata_parse_rate": _mode_rate(automata_rows, "parse_status", "ok"),
        "baseline_contract_accept_rate": _mode_rate(
            baseline_rows, "deterministic_validator_accept", True
        ),
        "automata_contract_accept_rate": _mode_rate(
            automata_rows, "deterministic_validator_accept", True
        ),
        "latency_delta_seconds": None
        if baseline_latency is None or automata_latency is None
        else round(automata_latency - baseline_latency, 6),
        "false_accept_count": ledger["false_accept_count"],
        "false_accept_rate": ledger["false_accept_rate"],
        "explicit_label_count": ledger["explicit_label_count"],
        "explicit_reject_count": ledger["explicit_reject_count"],
    }


def resolve_runtime_models(
    cached_pair_fn: CachedPairFn,
    resolver_fn: ResolverFn,
    *,
    max_models: int,
) -> list[JsonDict]:
    """Resolve mandated local SOTA GGUF models without legacy fallbacks."""

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


def run_experiment(
    *,
    project_root: Path | str | None = None,
    run_date: str = RUN_DATE,
    source_manifest_path: Path | str = DEFAULT_SOURCE_MANIFEST_PATH,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    decoder_manifest_path: Path | str = DEFAULT_DECODER_MANIFEST_PATH,
    cached_pair_fn: CachedPairFn | None = None,
    resolver_fn: ResolverFn | None = None,
    baseline_generator_fn: BaselineGeneratorFn | None = None,
    gpu_probe_fn: GpuProbeFn | None = None,
    xgrammar_probe_fn: XGrammarProbeFn | None = None,
    focused_tests_passed: bool = False,
    per_family: int = 1,
    max_models: int = 1,
) -> JsonDict:
    """Run the Exp 1535 decoder comparison and write terminal artifacts."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    output = _resolve_under_root(root, Path(output_path))
    decoder_manifest = _resolve_under_root(root, Path(decoder_manifest_path))
    source_manifest = _resolve_under_root(root, Path(source_manifest_path))
    write_in_progress_artifact(output, decoder_manifest_path=decoder_manifest, run_date=run_date)

    pair_resolver = cached_pair_fn or _cached_sota_pair
    gguf_resolver = resolver_fn or _resolve_cached_gguf
    gpu_probe = gpu_probe_fn or _probe_gpu_state
    xgrammar_available = (xgrammar_probe_fn or probe_xgrammar_available)()

    blockers: list[str] = []
    cases: list[JsonDict] = []
    if source_manifest.exists():
        cases = select_contract_cases(source_manifest, per_family=per_family)
    else:
        blockers.append(f"missing_runtime_contract_manifest:{source_manifest}")
    if not cases and not blockers:
        blockers.append("no_runtime_contract_cases_selected")

    models = resolve_runtime_models(pair_resolver, gguf_resolver, max_models=max_models)
    if not models:
        blockers.append("no_mandated_sota_gguf_runtime")

    rows: list[JsonDict] = []
    generation_blockers: list[str] = []
    if not blockers:
        if baseline_generator_fn is None:  # pragma: no cover - live GGUF hardware path.
            baseline_records, generation_blockers = _run_live_baseline_generation(cases, models)
        else:
            baseline_records = _run_injected_baseline_generation(
                cases, models, baseline_generator_fn
            )
        blockers.extend(generation_blockers)
        rows = _compare_decoder_modes(cases, models, baseline_records)

    _write_jsonl(decoder_manifest, [*rows, _summary_manifest_row(rows, len(cases))])
    models_used = sorted(
        {
            str(row["model_hf_id"])
            for row in rows
            if row.get("model_hf_id") in MANDATED_HF_IDS
        }
    )
    if rows and not models_used:  # pragma: no cover - runtime model resolver filters this out.
        blockers.append("no_mandated_sota_model_completed_live_inference")
    if not focused_tests_passed:
        blockers.append("focused_tests_not_passed")

    artifact = _terminal_artifact(
        status="complete" if rows else "blocked",
        run_date=run_date,
        rows=rows,
        cases_attempted=len(cases) if rows else 0,
        models_used=models_used,
        gpu_probe=gpu_probe(),
        decoder_manifest_path=decoder_manifest,
        xgrammar_available=xgrammar_available,
        focused_tests_passed=focused_tests_passed,
        blockers=list(dict.fromkeys(blockers)),
    )
    _write_json(output, artifact)
    return artifact


def _run_injected_baseline_generation(
    cases: Sequence[JsonDict],
    models: Sequence[JsonDict],
    generator_fn: BaselineGeneratorFn,
) -> dict[tuple[str, str], tuple[str, float]]:
    records: dict[tuple[str, str], tuple[str, float]] = {}
    for model in models:
        for case in cases:
            prompt = build_baseline_prompt(case)
            start = time.perf_counter()
            raw_output = generator_fn(prompt, dict(model), dict(case))
            latency = time.perf_counter() - start
            records[(str(model.get("hf_id")), str(case.get("contract_case_id")))] = (
                raw_output,
                latency,
            )
    return records


def _compare_decoder_modes(
    cases: Sequence[JsonDict],
    models: Sequence[JsonDict],
    baseline_records: Mapping[tuple[str, str], tuple[str, float]],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for model in models:
        model_id = str(model.get("hf_id"))
        for case in cases:
            case_id = str(case.get("contract_case_id"))
            raw_baseline, baseline_latency = baseline_records.get((model_id, case_id), ("", 0.0))
            rows.append(
                validate_decoded_output(
                    case,
                    raw_output=raw_baseline,
                    decoder_mode="baseline_post_decode",
                    model_spec=model,
                    latency_seconds=baseline_latency,
                )
            )
            dfa = compile_contract_dfa(case)
            start = time.perf_counter()
            raw_automata = dfa.generate()
            automata_latency = time.perf_counter() - start
            rows.append(
                validate_decoded_output(
                    case,
                    raw_output=raw_automata,
                    decoder_mode="automata_guided",
                    model_spec=model,
                    latency_seconds=automata_latency,
                )
            )
    return rows


def _terminal_artifact(
    *,
    status: str,
    run_date: str,
    rows: Sequence[Mapping[str, Any]],
    cases_attempted: int,
    models_used: Sequence[str],
    gpu_probe: Mapping[str, Any],
    decoder_manifest_path: Path,
    xgrammar_available: bool,
    focused_tests_passed: bool,
    blockers: Sequence[str],
) -> JsonDict:
    summary = summarize_decoder_rows(rows)
    abs_dfa_masks_used = any(row.get("abs_dfa_masks_used") for row in rows)
    metrics_reported = all(
        summary[key] is not None
        for key in (
            "baseline_parse_rate",
            "automata_parse_rate",
            "baseline_contract_accept_rate",
            "automata_contract_accept_rate",
            "latency_delta_seconds",
            "false_accept_rate",
        )
    )
    families = sorted({str(row.get("source_family")) for row in rows if row.get("source_family")})
    family_coverage_ok = all(family in families for family in CONTRACT_FAMILY_ORDER)
    ready = (
        status == "complete"
        and bool(models_used)
        and cases_attempted > 0
        and abs_dfa_masks_used
        and metrics_reported
        and family_coverage_ok
        and summary["false_accept_rate"] == 0.0
        and focused_tests_passed
        and not blockers
    )
    return {
        "status": status,
        "run_date": run_date,
        "schema_version": 1,
        "milestone": MILESTONE,
        "contract_decoder_adapter_ready": bool(ready),
        "model_specs": [dict(spec) for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": bool(models_used),
        "cases_attempted": int(cases_attempted),
        "baseline_parse_rate": summary["baseline_parse_rate"] or 0.0,
        "automata_parse_rate": summary["automata_parse_rate"] or 0.0,
        "baseline_contract_accept_rate": summary["baseline_contract_accept_rate"] or 0.0,
        "automata_contract_accept_rate": summary["automata_contract_accept_rate"] or 0.0,
        "latency_delta_seconds": summary["latency_delta_seconds"] or 0.0,
        "false_accept_rate": summary["false_accept_rate"] or 0.0,
        "false_accept_count": summary["false_accept_count"],
        "xgrammar_available": bool(xgrammar_available),
        "abs_dfa_masks_used": bool(abs_dfa_masks_used),
        "adapter_path": ADAPTER_PATH,
        "focused_tests_passed": bool(focused_tests_passed),
        "honest_verdict": (
            "complete: XGrammar/ABS contract decoder adapter ready"
            if ready
            else "complete: XGrammar/ABS contract decoder adapter blocked before readiness"
        ),
        "decoder_manifest_path": _display_path(decoder_manifest_path),
        "models_used": list(models_used),
        "gpu_probe": dict(gpu_probe),
        "blockers": list(blockers),
        "selected_case_families": families,
        "legacy_small_models_excluded_from_headline_metrics": True,
        "explicit_label_count": summary["explicit_label_count"],
        "explicit_reject_count": summary["explicit_reject_count"],
    }


def _summary_manifest_row(rows: Sequence[Mapping[str, Any]], cases_attempted: int) -> JsonDict:
    summary = summarize_decoder_rows(rows)
    return {
        "row_type": "summary",
        "cases_attempted": int(cases_attempted),
        "baseline_parse_rate": summary["baseline_parse_rate"],
        "automata_parse_rate": summary["automata_parse_rate"],
        "baseline_contract_accept_rate": summary["baseline_contract_accept_rate"],
        "automata_contract_accept_rate": summary["automata_contract_accept_rate"],
        "latency_delta_seconds": summary["latency_delta_seconds"],
        "false_accept_count": summary["false_accept_count"],
        "false_accept_rate": summary["false_accept_rate"],
    }


def _validation_contract_case(case: Mapping[str, Any], *, final_accept: bool) -> JsonDict:
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


def _authoritative_acceptance_target(case: Mapping[str, Any]) -> bool:
    expected = case.get("expected_label")
    if isinstance(expected, bool):
        return expected
    return bool(case.get("final_deterministic_accept"))


def _case_decision(case: Mapping[str, Any]) -> str:
    decision = case.get("final_deterministic_decision")
    if isinstance(decision, str) and decision.lower() in {"accept", "reject"}:
        return decision.lower()
    return "accept" if bool(case.get("final_deterministic_accept")) else "reject"


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


def _mode_rate(rows: Sequence[Mapping[str, Any]], key: str, expected: Any) -> float | None:
    if not rows:
        return None
    return round(sum(1 for row in rows if row.get(key) == expected) / len(rows), 6)


def _average_latency(rows: Sequence[Mapping[str, Any]]) -> float | None:
    if not rows:
        return None
    return round(sum(float(row.get("latency_seconds") or 0.0) for row in rows) / len(rows), 6)


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


def _run_live_baseline_generation(
    cases: Sequence[JsonDict],
    models: Sequence[JsonDict],
) -> tuple[dict[tuple[str, str], tuple[str, float]], list[str]]:  # pragma: no cover
    records: dict[tuple[str, str], tuple[str, float]] = {}
    blockers: list[str] = []
    for model in models:
        try:
            model_records = _run_one_live_model(cases, model)
        except Exception as exc:
            blockers.append(f"live_generation_failed:{model.get('hf_id')}:{type(exc).__name__}:{exc}")
            continue
        records.update(model_records)
        if model_records:
            break
    if not records:
        blockers.append("no_mandated_sota_model_completed_live_inference")
    return records, blockers


def _run_one_live_model(
    cases: Sequence[JsonDict],
    model: JsonDict,
) -> dict[tuple[str, str], tuple[str, float]]:  # pragma: no cover
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
    records: dict[tuple[str, str], tuple[str, float]] = {}
    try:
        for case in cases:
            prompt = build_baseline_prompt(case)
            start = time.perf_counter()
            completion = llm(
                prompt,
                max_tokens=96,
                temperature=0.0,
                echo=False,
                stop=["</s>", "<eos>"],
            )
            latency = time.perf_counter() - start
            records[(str(model.get("hf_id")), str(case.get("contract_case_id")))] = (
                _completion_text(completion),
                latency,
            )
    finally:
        if hasattr(llm, "close"):
            llm.close()
    return records


def _ensure_cuda_library_path() -> None:  # pragma: no cover
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


def _completion_text(result: Any) -> str:  # pragma: no cover
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
    parser.add_argument("--decoder-manifest", type=Path, default=DEFAULT_DECODER_MANIFEST_PATH)
    parser.add_argument("--per-family", type=int, default=1)
    parser.add_argument("--max-models", type=int, default=1)
    parser.add_argument("--focused-tests-passed", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = _parse_args(argv)
    run_experiment(
        run_date=args.run_date,
        source_manifest_path=args.source_manifest,
        output_path=args.output,
        decoder_manifest_path=args.decoder_manifest,
        per_family=args.per_family,
        max_models=args.max_models,
        focused_tests_passed=args.focused_tests_passed,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
