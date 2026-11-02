"""Exp 5274: gated SOTA retry for solver-scored constraint extraction.

Spec refs: REQ-VERIFY-5274, SCENARIO-VERIFY-5274.

This module runs only after Exp 5273 has made the fixture executable and
solver-clean. The model's job is narrow: translate natural language into the
strict Exp 5273 constraint IR. The solver remains the evaluator, so malformed
JSON, schema-invalid IR, and solver-wrong constraints are kept as separate
failure modes instead of being blended into one quality number.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5273_solver_fixture_rebuild_v482 as fixture_mod
from carnot.inference.sota_models import resolve_cached_gguf

try:  # pragma: no cover - absence is covered through dependency injection.
    import z3 as _z3
except Exception:  # pragma: no cover
    _z3 = None


JsonDict = dict[str, Any]
ProposalFn = Callable[[Mapping[str, Any], fixture_mod.SolverFixture, str], str]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5274_solver_constraint_extraction_retry_gated_v482.json"
)
SCHEMA = "carnot.experiment_5274.solver_constraint_extraction_retry_gated.v482"
SPEC_REFS = ("REQ-VERIFY-5274", "SCENARIO-VERIFY-5274")
INFERENCE_SUBSTRATE = "live_llm_inference_local_gguf_sota"
TERMINAL_PREFIXES = ("complete:", "blocked_")
GGUF_GENERATION_CONFIG = {
    "n_gpu_layers": -1,
    "n_ctx": 2048,
    "max_tokens": 384,
    "temperature": 0.0,
    "seed": 5274,
}

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal Exp 5274 verdict; starts with complete: or blocked_ and states "
        "whether the retry improved, nulled, regressed, or was unmeasured."
    ),
    "inference_substrate": (
        "Declares live local SOTA GGUF inference plus deterministic solver scoring "
        "over the rebuilt Exp 5273 fixture."
    ),
    "preconditions_checked": (
        "Records Exp 5273 fixture readiness, local mandated GGUF availability, "
        "deterministic solver availability, and exclusion of external scorers before "
        "extraction."
    ),
    "MODEL_SPECS": (
        "Records mandated local SOTA GGUF model IDs, roles, quantization/file "
        "receipts, runtime status, and headline inclusion."
    ),
    "solver_extraction_improved": (
        "True only when schema-valid solver-scored extraction beats the rebuilt "
        "baseline and prior V481 validity without unsafe false accepts."
    ),
    "validity_rate": (
        "Fraction of model-fixture rows whose schema-valid executable constraints "
        "matched the deterministic solver label."
    ),
    "baseline_validity": (
        "Baseline validity from the rebuilt Exp 5273 fixture controls used as the "
        "comparison floor."
    ),
    "malformed_rate": (
        "Fraction of model-fixture rows rejected by schema validation before solver "
        "scoring."
    ),
    "unsafe_false_accepts": (
        "Count of expected-UNSAT fixture rows whose schema-valid generated "
        "constraints were satisfiable."
    ),
    "fixture_checksums": (
        "Carries Exp 5273 fixture and per-prompt/output checksums so extraction "
        "scoring cannot silently drift from the gated fixture."
    ),
    "commands_run": "Commands run to create and validate the artifact, with outcomes.",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "MODEL_SPECS",
    "solver_extraction_improved",
    "validity_rate",
    "baseline_validity",
    "malformed_rate",
    "unsafe_false_accepts",
    "fixture_checksums",
    "commands_run",
)
WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "MODEL_SPECS",
    "solver_extraction_improved",
    "validity_rate",
    "baseline_validity",
    "malformed_rate",
    "unsafe_false_accepts",
    "fixture_checksums",
)

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "slot": "flagship_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe",
        "quantization": "Q4_K_M",
        "required": True,
    },
    {
        "slot": "flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "flagship_dense",
        "quantization": "Q4_K_M",
        "required": True,
    },
    {
        "slot": "middle_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "optional_middle_moe_third_family_check",
        "quantization": "Q4_K_M",
        "required": False,
    },
)


def sha256(value: str | bytes) -> str:
    """Return a stable full SHA-256 checksum for prompts and outputs."""

    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def build_model_specs(
    *,
    model_paths: Mapping[str, str | Path | None] | None = None,
    preferred_quant: str = "Q4_K_M",
    cache_root: str | None = None,
) -> JsonDict:
    """Resolve mandated SOTA GGUF model receipts into artifact-ready specs.

    The helper checks files, not model quality. A path receipt means the runner
    can attempt local llama.cpp inference; only generated rows later determine
    whether extraction improved.
    """

    specs: JsonDict = {}
    for mandated in MANDATED_MODEL_SPECS:
        slot = str(mandated["slot"])
        if model_paths is None:  # pragma: no cover - live preflight path.
            resolved = resolve_cached_gguf(
                str(mandated["hf_id"]),
                preferred_quant=preferred_quant,
                cache_root=cache_root,
            )
            path = Path(resolved) if resolved else None
        else:
            candidate = model_paths.get(slot)
            path = Path(candidate) if candidate is not None else None

        ready = bool(path and path.exists() and path.is_file())
        file_receipts = _file_receipts(path) if ready and path is not None else _missing_receipts()
        specs[slot] = {
            "slot": slot,
            "hf_id": str(mandated["hf_id"]),
            "role": str(mandated["role"]),
            "quantization": preferred_quant or str(mandated["quantization"]),
            "required": bool(mandated["required"]),
            "runtime_ready": ready,
            "runtime_status": "local_file_ready" if ready else "missing_local_gguf",
            "headline_included": ready,
            "file_receipts": file_receipts,
        }
    return specs


def empty_encoding() -> JsonDict:
    """Return the strict empty IR control used to expose unsafe false accepts."""

    return {
        "schema_version": fixture_mod.IR_SCHEMA_VERSION,
        "variables": {},
        "constraints": [],
    }


def render_prompt(fixture: fixture_mod.SolverFixture) -> str:
    """Render a deterministic request that asks only for the strict IR."""

    return (
        "Translate the natural-language requirements into executable integer constraints.\n"
        "Return only one JSON object matching this schema:\n"
        "{\n"
        f'  "schema_version": "{fixture_mod.IR_SCHEMA_VERSION}",\n'
        '  "variables": {"x": {"type": "int"}},\n'
        '  "constraints": [{"id": "short_identifier", "expr": "x >= 0"}]\n'
        "}\n"
        "Use only integer variables and expressions with ==, <, <=, >, >=, %, +, and -.\n"
        "Do not state whether the requirements are satisfiable.\n"
        "Do not include markdown fences, explanations, or extra keys.\n\n"
        f"Fixture id: {fixture.fixture_id}\n"
        f"Requirements:\n{fixture.natural_language}\n"
    )


def evaluate_model_output(
    fixture: fixture_mod.SolverFixture,
    *,
    model_slot: str,
    prompt: str,
    raw_output: str,
    z3_module: Any = _z3,
) -> JsonDict:
    """Parse, schema-check, and solver-score one model output row."""

    payload, json_parseable, json_extracted, parse_error = _parse_payload(raw_output)
    if payload is None:
        score = _parse_error_score(fixture, parse_error)
        schema_valid = False
        generated_ir = None
    else:
        score_obj = fixture_mod.score_candidate(fixture, payload, z3_module=z3_module)
        score = score_obj.to_dict()
        schema_valid = bool(score_obj.schema_valid)
        generated_ir = payload if schema_valid else None

    return {
        "fixture_id": fixture.fixture_id,
        "model_slot": model_slot,
        "expected_status": fixture.expected_status,
        "prompt": prompt,
        "prompt_sha256": sha256(prompt),
        "raw_output": raw_output,
        "output_sha256": sha256(raw_output),
        "json_parseable": json_parseable,
        "json_extracted": json_extracted,
        "parse_error": parse_error,
        "schema_valid": schema_valid,
        "malformed": not schema_valid,
        "solver_status": score["solver_status"],
        "matches_expected": bool(score["matches_expected"]),
        "false_accept": bool(score["false_accept"]),
        "generated_ir": generated_ir,
        "score": score,
    }


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_validity: float,
    prior_v481_validity: float,
) -> JsonDict:
    """Compute retry metrics while keeping malformed rows separate."""

    total = len(rows)
    correct = sum(1 for row in rows if row.get("matches_expected") is True)
    malformed = sum(1 for row in rows if row.get("malformed") is True)
    unsafe_false_accepts = sum(1 for row in rows if row.get("false_accept") is True)
    solver_invalid = sum(
        1
        for row in rows
        if row.get("schema_valid") is True and row.get("matches_expected") is not True
    )
    sat_rows = [
        row
        for row in rows
        if row.get("expected_status") == "sat" and row.get("schema_valid") is True
    ]
    sat_correct = sum(1 for row in sat_rows if row.get("matches_expected") is True)
    mismatches = [row for row in rows if row.get("matches_expected") is not True]
    counterexamples = sum(
        1
        for row in mismatches
        if isinstance(row.get("score"), Mapping) and bool(row["score"].get("counterexample"))
    )
    validity = _rate(correct, total)
    malformed_rate = _rate(malformed, total)
    improved = (
        total > 0
        and validity > max(float(baseline_validity), float(prior_v481_validity))
        and unsafe_false_accepts == 0
        and malformed < total
    )
    if total == 0:
        outcome = "unmeasured"
    elif improved:
        outcome = "improved"
    elif validity < float(prior_v481_validity):
        outcome = "regressed"
    else:
        outcome = "nulled"
    return {
        "rows_total": total,
        "validity_rate": validity,
        "baseline_validity": float(baseline_validity),
        "prior_v481_validity": float(prior_v481_validity),
        "malformed_outputs": malformed,
        "malformed_rate": malformed_rate,
        "schema_valid_outputs": total - malformed,
        "solver_invalid_constraints": solver_invalid,
        "unsafe_false_accepts": unsafe_false_accepts,
        "satisfiable_label_accuracy": _rate(sat_correct, len(sat_rows)),
        "counterexample_agreement_rate": _rate(counterexamples, len(mismatches)),
        "improved": improved,
        "retry_outcome": outcome,
    }


def run(
    *,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    fixture_artifact: Mapping[str, Any] | None = None,
    model_specs: Mapping[str, Any] | None = None,
    proposal_fn: ProposalFn | None = None,
    commands_run: Sequence[Mapping[str, Any]] = (),
    z3_module: Any = _z3,
    root: Path = REPO_ROOT,
    write: bool = True,
) -> JsonDict:
    """Run the gated retry or emit a blocked artifact with concrete blockers."""

    started = time.perf_counter()
    active_fixture_artifact = (
        dict(fixture_artifact) if fixture_artifact is not None else load_fixture_artifact(root)
    )
    active_model_specs = (
        dict(model_specs) if model_specs is not None else build_model_specs()
    )
    live_runtime = _live_runtime_precondition(proposal_fn)
    preconditions = _preconditions(
        active_fixture_artifact,
        active_model_specs,
        z3_module,
        live_runtime=live_runtime,
    )
    fixtures = fixture_mod.fixture_set()
    prior_v481_validity = _prior_v481_validity(root)
    baseline_validity, fixture_baselines = _baseline_validity(fixtures, z3_module)

    rows: list[JsonDict] = []
    if not preconditions["blockers"]:
        try:
            proposer = proposal_fn or live_llama_cpp_proposal_fn(active_model_specs)
        except RuntimeError as exc:  # pragma: no cover - live environment guard.
            preconditions["blockers"].append(f"llama_cpp_unavailable: {exc}")
            proposer = None
    if not preconditions["blockers"]:
        assert proposer is not None
        for spec in _headline_model_specs(active_model_specs):
            for fixture in fixtures:
                prompt = render_prompt(fixture)
                try:
                    raw_output = proposer(spec, fixture, prompt)
                    row = evaluate_model_output(
                        fixture,
                        model_slot=str(spec["slot"]),
                        prompt=prompt,
                        raw_output=str(raw_output),
                        z3_module=z3_module,
                    )
                except Exception as exc:  # pragma: no cover - live inference failure path.
                    row = evaluate_model_output(
                        fixture,
                        model_slot=str(spec["slot"]),
                        prompt=prompt,
                        raw_output="",
                        z3_module=z3_module,
                    )
                    row["proposal_error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)

    aggregate = aggregate_rows(
        rows,
        baseline_validity=baseline_validity,
        prior_v481_validity=prior_v481_validity,
    )
    artifact = _build_artifact(
        fixture_artifact=active_fixture_artifact,
        model_specs=active_model_specs,
        preconditions=preconditions,
        rows=rows,
        aggregate=aggregate,
        fixture_baselines=fixture_baselines,
        commands_run=commands_run,
        duration_s=time.perf_counter() - started,
        root=root,
    )
    validate_artifact(artifact)
    if write:
        _write_json(result_path, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 5274 artifact violates the required schema."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"missing required field {field}"
    for field in WRAPPED_FIELDS:
        value = artifact[field]
        assert isinstance(value, Mapping), f"{field} must be principle-wrapped"
        assert "value" in value and "principle" in value, f"{field} must be principle-wrapped"
        assert value["principle"] == FIELD_PRINCIPLES[field], f"{field} principle mismatch"

    verdict = artifact["honest_verdict"]["value"]
    assert isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), (
        "honest_verdict.value must start with complete: or blocked_"
    )
    assert any(word in verdict for word in ("improved", "nulled", "regressed", "unmeasured")), (
        "honest_verdict.value must state improved, nulled, regressed, or unmeasured"
    )
    assert artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE, (
        f"inference_substrate.value must be {INFERENCE_SUBSTRATE}"
    )
    assert isinstance(artifact["solver_extraction_improved"]["value"], bool), (
        "solver_extraction_improved.value must be bool"
    )
    for field in ("validity_rate", "baseline_validity", "malformed_rate"):
        assert _rate_ok(artifact[field]["value"]), f"{field}.value must be numeric in [0, 1]"
    assert isinstance(artifact["unsafe_false_accepts"]["value"], int), (
        "unsafe_false_accepts.value must be int"
    )
    assert isinstance(artifact["fixture_checksums"]["value"], Mapping), (
        "fixture_checksums.value must be object"
    )
    assert isinstance(artifact["commands_run"], list), "commands_run must be a list"
    specs = artifact["MODEL_SPECS"]["value"]
    assert isinstance(specs, Mapping), "MODEL_SPECS.value must be object"
    for mandated in MANDATED_MODEL_SPECS:
        slot = str(mandated["slot"])
        assert slot in specs, f"MODEL_SPECS missing {slot}"
        assert specs[slot]["hf_id"] == mandated["hf_id"], f"{slot} hf_id mismatch"


def load_fixture_artifact(root: Path = REPO_ROOT) -> JsonDict:  # pragma: no cover
    """Load the Exp 5273 gate artifact from the repository results directory."""

    path = root / fixture_mod.RESULT_RELATIVE_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def live_llama_cpp_proposal_fn(model_specs: Mapping[str, Any]) -> ProposalFn:  # pragma: no cover
    """Build a sequential local llama.cpp proposer for the mandated GGUFs."""

    try:
        from llama_cpp import Llama
    except Exception as exc:
        raise RuntimeError(f"llama_cpp unavailable: {exc}") from exc

    loaded: dict[str, Any] = {}

    def propose(model_spec: Mapping[str, Any], fixture: fixture_mod.SolverFixture, prompt: str) -> str:
        del fixture
        slot = str(model_spec["slot"])
        if slot not in loaded:
            path = model_spec["file_receipts"].get("path")
            if not path:
                raise RuntimeError(f"{slot} has no model_path")
            loaded.clear()
            gc.collect()
            loaded[slot] = Llama(
                model_path=str(path),
                n_gpu_layers=int(GGUF_GENERATION_CONFIG["n_gpu_layers"]),
                n_ctx=int(GGUF_GENERATION_CONFIG["n_ctx"]),
                seed=int(GGUF_GENERATION_CONFIG["seed"]),
                verbose=False,
            )
        response = loaded[slot](
            prompt,
            max_tokens=int(GGUF_GENERATION_CONFIG["max_tokens"]),
            temperature=float(GGUF_GENERATION_CONFIG["temperature"]),
        )
        if isinstance(response, Mapping) and response.get("choices"):
            return str(response["choices"][0].get("text", ""))
        return str(response)

    return propose


def _parse_payload(raw_output: str) -> tuple[JsonDict | None, bool, bool, str | None]:
    obj_text, extracted = _extract_json_object(raw_output)
    if obj_text is None:
        return None, False, False, "no_json_object"
    try:
        payload = json.loads(obj_text)
    except json.JSONDecodeError as exc:
        return None, False, extracted, f"json_decode_error: {exc.msg}"
    return payload, True, extracted, None


def _parse_error_score(fixture: fixture_mod.SolverFixture, parse_error: str | None) -> JsonDict:
    return {
        "fixture_id": fixture.fixture_id,
        "schema_valid": False,
        "solver_status": "parse_error",
        "expected_status": fixture.expected_status,
        "matches_expected": False,
        "false_accept": False,
        "assignment": {},
        "counterexample": {"parse_error": parse_error or "unknown_parse_error"},
        "errors": [parse_error or "unknown_parse_error"],
    }


def _extract_json_object(text: str) -> tuple[str | None, bool]:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped, False
    fence_start = stripped.find("```")
    if fence_start >= 0:
        after = stripped.find("\n", fence_start)
        fence_end = stripped.find("```", after + 1 if after >= 0 else fence_start + 3)
        if after >= 0 and fence_end > after:
            fenced = stripped[after + 1 : fence_end].strip()
            if fenced.startswith("{") and fenced.endswith("}"):
                return fenced, True
    start = stripped.find("{")
    if start < 0:
        return None, False
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(stripped[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1], True
    return None, False


def _preconditions(
    fixture_artifact: Mapping[str, Any],
    model_specs: Mapping[str, Any],
    z3_module: Any,
    *,
    live_runtime: Mapping[str, Any],
) -> JsonDict:
    fixture_ready = bool(fixture_artifact.get("solver_fixture_ready"))
    solver_available = _solver_available(z3_module)
    required_ready = all(
        bool(model_specs.get(str(mandated["slot"]), {}).get("runtime_ready"))
        for mandated in MANDATED_MODEL_SPECS
        if mandated["required"]
    )
    headline_slots = [
        str(mandated["slot"])
        for mandated in MANDATED_MODEL_SPECS
        if bool(model_specs.get(str(mandated["slot"]), {}).get("headline_included"))
    ]
    blockers: list[str] = []
    if not fixture_ready:
        blockers.append("exp5273_solver_fixture_ready_not_true")
    if not solver_available:
        blockers.append("deterministic_solver_unavailable")
    if not required_ready:
        blockers.append("required_headline_models_unavailable")
    if live_runtime.get("required") and not live_runtime.get("llama_cpp_gpu_offload_supported"):
        blockers.append("llama_cpp_gpu_offload_unavailable")
    return {
        "exp5273_solver_fixture_ready": fixture_ready,
        "exp5273_artifact_path": str(REPO_ROOT / fixture_mod.RESULT_RELATIVE_PATH),
        "deterministic_solver": "z3",
        "deterministic_solver_available": solver_available,
        "required_headline_models_available": required_ready,
        "live_runtime": dict(live_runtime),
        "headline_model_slots": headline_slots,
        "external_text_scorers_used": False,
        "blockers": blockers,
    }


def _live_runtime_precondition(proposal_fn: ProposalFn | None) -> JsonDict:
    if proposal_fn is not None:
        return {
            "required": False,
            "reason": "injected_proposal_fn_unit_path",
            "llama_cpp_import_ok": True,
            "llama_cpp_gpu_offload_supported": True,
        }
    try:  # pragma: no cover - live environment receipt path.
        import llama_cpp
    except Exception as exc:  # pragma: no cover
        return {
            "required": True,
            "llama_cpp_import_ok": False,
            "llama_cpp_error": f"{type(exc).__name__}: {exc}",
            "llama_cpp_gpu_offload_supported": False,
        }
    supports = getattr(llama_cpp, "llama_supports_gpu_offload", lambda: False)()  # pragma: no cover
    return {  # pragma: no cover
        "required": True,
        "llama_cpp_import_ok": True,
        "llama_cpp_version": getattr(llama_cpp, "__version__", "unknown"),
        "llama_cpp_gpu_offload_supported": bool(supports),
        "principle": (
            "Large mandated SOTA GGUF extraction must not silently fall back to "
            "multi-hour CPU decoding when GPU offload is unavailable."
        ),
    }


def _headline_model_specs(model_specs: Mapping[str, Any]) -> list[JsonDict]:
    ordered: list[JsonDict] = []
    for mandated in MANDATED_MODEL_SPECS:
        spec = model_specs.get(str(mandated["slot"]))
        if isinstance(spec, Mapping) and spec.get("headline_included"):
            ordered.append(dict(spec))
    return ordered


def _baseline_validity(fixtures: Sequence[fixture_mod.SolverFixture], z3_module: Any) -> tuple[float, JsonDict]:
    if not _solver_available(z3_module):
        return 0.0, {}
    baselines = fixture_mod.score_baselines(fixtures, z3_module=z3_module)
    return float(baselines["empty_extraction"]["validity_rate"]), baselines


def _prior_v481_validity(root: Path) -> float:
    path = root / fixture_mod.V481_RESULT_RELATIVE_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError:  # pragma: no cover - checked-in artifact exists in normal runs.
        return 0.0
    return float(payload.get("constraint_validity_rate", {}).get("value", 0.0))


def _build_artifact(
    *,
    fixture_artifact: Mapping[str, Any],
    model_specs: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    fixture_baselines: Mapping[str, Any],
    commands_run: Sequence[Mapping[str, Any]],
    duration_s: float,
    root: Path,
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(duration_s, 6),
        "honest_verdict": _wrap("honest_verdict", _honest_verdict(preconditions, aggregate)),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "preconditions_checked": _wrap("preconditions_checked", dict(preconditions)),
        "MODEL_SPECS": _wrap("MODEL_SPECS", {slot: dict(spec) for slot, spec in model_specs.items()}),
        "solver_extraction_improved": _wrap("solver_extraction_improved", bool(aggregate["improved"])),
        "validity_rate": _wrap("validity_rate", aggregate["validity_rate"]),
        "baseline_validity": _wrap("baseline_validity", aggregate["baseline_validity"]),
        "malformed_rate": _wrap("malformed_rate", aggregate["malformed_rate"]),
        "unsafe_false_accepts": _wrap("unsafe_false_accepts", int(aggregate["unsafe_false_accepts"])),
        "fixture_checksums": _wrap(
            "fixture_checksums",
            _fixture_checksums(fixture_artifact, rows, root),
        ),
        "commands_run": [dict(command) for command in commands_run],
        "extraction_results": [dict(row) for row in rows],
        "aggregate_metrics": dict(aggregate),
        "fixture_baselines": dict(fixture_baselines),
        "prior_v481": {
            "artifact_path": str(root / fixture_mod.V481_RESULT_RELATIVE_PATH),
            "validity_rate": aggregate["prior_v481_validity"],
        },
        "blockers": list(preconditions.get("blockers", [])),
        "external_text_scorer_used": False,
        "retired_solver_feedback_scope_reopened": False,
    }


def _honest_verdict(preconditions: Mapping[str, Any], aggregate: Mapping[str, Any]) -> str:
    blockers = list(preconditions.get("blockers", []))
    if blockers:
        return "blocked_preconditions: " + ",".join(blockers) + "; retry was unmeasured"
    outcome = str(aggregate["retry_outcome"])
    return (
        f"complete: retry {outcome} "
        f"(validity={aggregate['validity_rate']}, "
        f"baseline={aggregate['baseline_validity']}, "
        f"prior_v481={aggregate['prior_v481_validity']}, "
        f"malformed={aggregate['malformed_rate']}, "
        f"unsafe_false_accepts={aggregate['unsafe_false_accepts']})"
    )


def _fixture_checksums(
    fixture_artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    root: Path,
) -> JsonDict:
    source_checksums = _nested_value(fixture_artifact, "fixture_checksums")
    prompt_output: JsonDict = {}
    for row in rows:
        slot = str(row["model_slot"])
        fixture_id = str(row["fixture_id"])
        prompt_output.setdefault(slot, {})[fixture_id] = {
            "prompt_sha256": row["prompt_sha256"],
            "output_sha256": row["output_sha256"],
        }
    return {
        "source_exp5273_path": str(root / fixture_mod.RESULT_RELATIVE_PATH),
        "source_fixture_set_sha256": source_checksums.get("fixture_set_sha256")
        if isinstance(source_checksums, Mapping)
        else None,
        "source_fixture_checksums": source_checksums.get("fixtures", {})
        if isinstance(source_checksums, Mapping)
        else {},
        "prompt_output_checksums": prompt_output,
    }


def _nested_value(payload: Mapping[str, Any], key: str) -> Any:
    value = payload.get(key)
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _file_receipts(path: Path) -> JsonDict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "checksum_sha256": None,
        "checksum_head_1m_sha256": _checksum_head(path),
    }


def _missing_receipts() -> JsonDict:
    return {
        "path": None,
        "size_bytes": None,
        "checksum_sha256": None,
        "checksum_head_1m_sha256": None,
    }


def _checksum_head(path: Path) -> str:
    with path.open("rb") as handle:
        return sha256(handle.read(1024 * 1024))


def _solver_available(z3_module: Any) -> bool:
    return z3_module is not None and hasattr(z3_module, "Solver") and hasattr(z3_module, "Int")


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _rate_ok(value: Any) -> bool:
    return isinstance(value, int | float) and 0.0 <= float(value) <= 1.0


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = run(
        result_path=args.output,
        commands_run=[
            {
                "command": (
                    ".venv/bin/python -m carnot."
                    "experiment_5274_solver_constraint_extraction_retry_gated_v482 "
                    f"--output {args.output}"
                ),
                "outcome": "completed module invocation",
            }
        ],
    )
    print(json.dumps({"honest_verdict": artifact["honest_verdict"]["value"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
