#!/usr/bin/env python3
"""Exp 5326: gated local SOTA paraphrase/rewrite smoke.

Spec refs: REQ-VERIFY-5326, SCENARIO-VERIFY-5326.

This module runs a tiny bounded local GGUF generation smoke only after Exp 5324
has selected a stable native llama.cpp command. The generated text is not judged
by another model and is not treated as a benchmark. It is converted into typed
fixture states and scored with the deterministic Exp 5310 and Exp 5325 checks.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from carnot import experiment_5310_paraphrase_consistency_fixture_v485 as exp5310
from carnot import experiment_5324_runtime_receipt_stabilization_v486 as exp5324
from carnot import experiment_5325_theoria_rewrite_state_fixture_v486 as exp5325
from carnot.inference.sota_models import resolve_cached_gguf


JsonDict = dict[str, Any]
ModelResolver = Callable[[str, str], str | None]
PreconditionsProvider = Callable[[], JsonDict]
GenerationProbe = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5326_gated_sota_paraphrase_rewrite_smoke_v486"
MILESTONE = "2026.07.486"
RESULT_RELATIVE_PATH = Path("results/experiment_5326_gated_sota_paraphrase_rewrite_smoke_v486.json")
SCHEMA = "carnot.experiment_5326.gated_sota_paraphrase_rewrite_smoke.v486"
INFERENCE_SUBSTRATE = "local_sota_gguf_bounded_smoke"
SPEC_REFS = ("REQ-VERIFY-5326", "SCENARIO-VERIFY-5326")
RANDOM_SEED = 5326
DEFAULT_N_PREDICT = 128
TERMINAL_PREFIXES = ("complete:", "blocked_")
MISSING_WRAPPED_VALUE = object()

MANDATED_MODEL_SPECS = exp5324.MANDATED_MODEL_SPECS
EXPECTED_MODEL_IDS = tuple(str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS)

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Traceability for the Exp5326 gated local SOTA paraphrase/rewrite smoke.",
    "milestone": "Milestone accountability for the V486 gated SOTA smoke.",
    "status": "Machine-readable terminal state for downstream quality-smoke gates.",
    "honest_verdict": (
        "Terminal verdict must start with complete: or blocked_ and state whether bounded "
        "generation plus deterministic fixture scoring completed."
    ),
    "inference_substrate": (
        "Declares local_sota_gguf_bounded_smoke so the artifact is read as a tiny local "
        "GGUF generation smoke scored by fixtures, not a broad benchmark."
    ),
    "MODEL_SPECS": (
        "Records the three mandated GGUF model IDs so the smoke cannot silently substitute "
        "a legacy or smaller model."
    ),
    "preconditions_checked": (
        "Records Exp5324 command stability, selected model cache, GPU visibility, and "
        "fixture availability before any generation."
    ),
    "selected_model_spec": (
        "Binds quality-smoke outputs to the stable mandated model selected by Exp5324."
    ),
    "tests_run": (
        "Commands run to validate the Exp5326 module, artifact schema, new-code coverage, "
        "and repository test status."
    ),
    "selected_backend_command": (
        "Preserves the exact Exp5324 backend command whose prompt and token budget are "
        "bounded for the smoke run."
    ),
}

REQUIRED_WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "MODEL_SPECS",
    "preconditions_checked",
    "selected_model_spec",
    "tests_run",
)
WRAPPED_FIELDS = (*REQUIRED_WRAPPED_FIELDS, "selected_backend_command")
REQUIRED_ARTIFACT_FIELDS = (
    *REQUIRED_WRAPPED_FIELDS,
    "prompt_count",
    "paraphrase_label_preservation_rate",
    "rewrite_acceptability_rate",
    "unsafe_false_accepts",
    "sota_quality_measured",
    "headline_quality_claim",
)

DEFAULT_PROMPT_SPECS: tuple[JsonDict, ...] = (
    {
        "prompt_id": "paraphrase_supported",
        "kind": "paraphrase",
        "fixture_group_id": "pcf-001-supported-equivalent",
        "n_predict": DEFAULT_N_PREDICT,
        "prompt": (
            "Return one compact JSON object only. Paraphrase this claim without changing "
            "its facts: The Aster-9 battery ran 47 minutes under the amber-load test. "
            "JSON keys: text, premise_valid, facts. Required facts are "
            "duration_minutes=47, subject=aster-9 battery, test=amber-load."
        ),
    },
    {
        "prompt_id": "rewrite_safe_paraphrase",
        "kind": "rewrite",
        "fixture_case_id": "rsf-001-safe-paraphrase",
        "n_predict": DEFAULT_N_PREDICT,
        "prompt": (
            "Return one compact JSON object only for the target rewrite state. Rewrite: "
            "The Aster-9 battery ran 47 minutes under the amber-load test. Keep the same "
            "facts and citation battery-log-47, but use wording=lasted. JSON keys: "
            "text, premise_valid, facts, attributes, citations."
        ),
    },
    {
        "prompt_id": "rewrite_unsafe_contradiction",
        "kind": "rewrite",
        "fixture_case_id": "rsf-002-contradiction-introduction",
        "n_predict": DEFAULT_N_PREDICT,
        "prompt": (
            "Return one compact JSON object only for the target rewrite state. Use this "
            "proposed target: Under the amber-load test, the Aster-9 battery lasted 74 "
            "minutes. Include citation battery-log-47. JSON keys: text, premise_valid, "
            "facts, attributes, citations."
        ),
    },
)


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> JsonDict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _raw_or_wrapped_value(payload: Mapping[str, Any], field: str) -> Any:
    value = payload.get(field)
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def command_for_prompt(
    command: Sequence[str],
    prompt: str,
    n_predict: int = DEFAULT_N_PREDICT,
    seed: int = RANDOM_SEED,
) -> list[str]:
    """Return the selected llama.cpp command with only bounded prompt knobs changed."""

    rewritten = list(command)

    def set_flag(flag: str, value: str) -> None:
        if flag in rewritten and rewritten.index(flag) + 1 < len(rewritten):
            rewritten[rewritten.index(flag) + 1] = value
        else:
            rewritten.extend([flag, value])

    set_flag("-p", prompt)
    set_flag("-n", str(n_predict))
    set_flag("--seed", str(seed))
    return rewritten


def extract_json_object(text: str) -> JsonDict | None:
    """Parse the first JSON object in model output using Python's JSON decoder."""

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return dict(value)
    return None


def _string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(nested) for key, nested in value.items()}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value)


def _resolve_model_specs(model_resolver: ModelResolver) -> JsonDict:
    return {
        str(spec["role"]): exp5324.exp5323._resolve_model_spec(spec, model_resolver)
        for spec in MANDATED_MODEL_SPECS
    }


def _selected_from_prior(prior_artifact: Mapping[str, Any]) -> tuple[JsonDict | None, JsonDict | None]:
    stable = _raw_or_wrapped_value(prior_artifact, "sota_runtime_unblocked_stable") is True
    status = _raw_or_wrapped_value(prior_artifact, "status")
    selected_command = _raw_or_wrapped_value(prior_artifact, "selected_backend_command")
    selected_model = _raw_or_wrapped_value(prior_artifact, "selected_model_spec")
    if not stable or status != "complete":
        return None, None
    command_ok = isinstance(selected_command, Mapping) and isinstance(
        selected_command.get("command"), list
    )
    model_ok = isinstance(selected_model, Mapping)
    return (
        dict(selected_command) if command_ok else None,
        dict(selected_model) if model_ok else None,
    )


def _fixture_preconditions(
    paraphrase_groups: tuple[exp5310.ParaphraseGroup, ...] | None,
    rewrite_cases: tuple[exp5325.RewriteCase, ...] | None,
) -> tuple[JsonDict, tuple[exp5310.ParaphraseGroup, ...], tuple[exp5325.RewriteCase, ...]]:
    try:
        groups = exp5310.load_fixture() if paraphrase_groups is None else paraphrase_groups
        paraphrase_ready = bool(exp5310.evaluate_fixture(groups)["ready"])
    except Exception as exc:  # pragma: no cover - defensive missing-file path
        groups = ()
        paraphrase_ready = False
        paraphrase_error = f"{type(exc).__name__}: {exc}"
    else:
        paraphrase_error = None

    try:
        cases = exp5325.load_fixture() if rewrite_cases is None else rewrite_cases
        rewrite_ready = bool(exp5325.evaluate_fixture(cases)["ready"])
    except Exception as exc:  # pragma: no cover - defensive missing-file path
        cases = ()
        rewrite_ready = False
        rewrite_error = f"{type(exc).__name__}: {exc}"
    else:
        rewrite_error = None

    return (
        {
            "paraphrase_fixture_path": str(exp5310.FIXTURE_RELATIVE_PATH),
            "paraphrase_fixture_ready": paraphrase_ready,
            "paraphrase_fixture_error": paraphrase_error,
            "rewrite_fixture_path": str(exp5325.FIXTURE_RELATIVE_PATH),
            "rewrite_state_fixture_ready": rewrite_ready,
            "rewrite_state_fixture_error": rewrite_error,
        },
        groups,
        cases,
    )


def _precondition_blockers(
    *,
    selected_command: Mapping[str, Any] | None,
    prior_selected_model: Mapping[str, Any] | None,
    selected_model: Mapping[str, Any] | None,
    current_preconditions: Mapping[str, Any],
    fixture_status: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if selected_command is None:
        blockers.append("exp5324_stable_runtime_missing")
        blockers.append("selected_backend_command_missing")
    if prior_selected_model is None:
        blockers.append("selected_model_spec_missing")
    if not current_preconditions.get("gpu_visible"):
        blockers.append("gpu_not_visible")
    if selected_model is None or selected_model.get("status") != "local_gguf_resolved":
        blockers.append("selected_model_file_missing")
    else:
        model_path = Path(str(selected_model.get("model_path") or ""))
        if not model_path.is_file():
            blockers.append("selected_model_file_missing")
        if selected_model.get("hf_id") not in EXPECTED_MODEL_IDS:
            blockers.append("selected_model_not_mandated")
    if not fixture_status.get("paraphrase_fixture_ready"):
        blockers.append("paraphrase_fixture_unavailable")
    if not fixture_status.get("rewrite_state_fixture_ready"):
        blockers.append("rewrite_state_fixture_unavailable")
    return list(dict.fromkeys(blockers))


def _normalize_generation_receipt(
    raw: Mapping[str, Any],
    *,
    prompt_spec: Mapping[str, Any],
    command: Sequence[str],
    timeout_s: float,
) -> JsonDict:
    output_text = str(raw.get("stdout", ""))
    stderr = str(raw.get("stderr", ""))
    completed = (
        bool(raw.get("completed"))
        and raw.get("returncode") == 0
        and not bool(raw.get("timed_out"))
    )
    return {
        "prompt_id": str(prompt_spec["prompt_id"]),
        "kind": str(prompt_spec["kind"]),
        "command": list(command),
        "completed": completed,
        "timed_out": bool(raw.get("timed_out")),
        "returncode": raw.get("returncode"),
        "timeout_s": timeout_s,
        "wall_clock_s": float(raw.get("wall_clock_s") or 0.0),
        "output_text": output_text,
        "output_checksum": exp5324.exp5323.sha16(output_text),
        "stderr_summary": "\n".join(stderr.strip().splitlines()[-8:])[-800:],
    }


def default_generation_probe(
    *,
    prompt_spec: Mapping[str, Any],
    command: Sequence[str],
    timeout_s: float,
    run_index: int,
) -> JsonDict:  # pragma: no cover - invokes live local llama.cpp subprocess
    _ = prompt_spec, run_index
    started = time.perf_counter()
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "completed": result.returncode == 0,
            "timed_out": False,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "wall_clock_s": time.perf_counter() - started,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "completed": False,
            "timed_out": True,
            "returncode": None,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "").strip() if isinstance(exc.stderr, str) else "timeout",
            "wall_clock_s": time.perf_counter() - started,
        }


def _score_paraphrase_output(
    receipt: Mapping[str, Any],
    prompt_spec: Mapping[str, Any],
    groups: tuple[exp5310.ParaphraseGroup, ...],
) -> JsonDict:
    parsed = extract_json_object(str(receipt.get("output_text", "")))
    group = exp5310.group_by_id(groups, str(prompt_spec["fixture_group_id"]))
    anchor_score = exp5310.score_claim(group.anchor, group)
    if parsed is None:
        return {
            "prompt_id": receipt["prompt_id"],
            "kind": "paraphrase",
            "parse_ok": False,
            "label_preserved": False,
            "passed": False,
            "computed_label": "parse-error",
        }
    claim = exp5310.ParaphraseClaim(
        claim_id=f"{receipt['prompt_id']}-generated",
        text=str(parsed.get("text", "")),
        premise_valid=parsed.get("premise_valid") is True,
        facts=_string_map(parsed.get("facts")),
        expected_label=group.anchor.expected_label,
        expected_label_preservation=True,
        expected_violation_type=None,
    )
    score = exp5310.score_claim(claim, group)
    label_preserved = score.label == anchor_score.label
    passed = label_preserved and score.label == group.anchor.expected_label
    return {
        "prompt_id": receipt["prompt_id"],
        "kind": "paraphrase",
        "parse_ok": True,
        "label_preserved": label_preserved,
        "passed": passed,
        "computed_label": score.label,
        "anchor_label": anchor_score.label,
        "conflict_keys": list(score.conflict_keys),
    }


def _score_rewrite_output(
    receipt: Mapping[str, Any],
    prompt_spec: Mapping[str, Any],
    cases: tuple[exp5325.RewriteCase, ...],
) -> JsonDict:
    parsed = extract_json_object(str(receipt.get("output_text", "")))
    base_case = exp5325.case_by_id(cases, str(prompt_spec["fixture_case_id"]))
    if parsed is None:
        return {
            "prompt_id": receipt["prompt_id"],
            "kind": "rewrite",
            "parse_ok": False,
            "acceptability_matches_expected": False,
            "accepted": False,
            "expected_accept": base_case.expected_accept,
            "unsafe_false_accept": False,
        }
    target = exp5325.RewriteState(
        text=str(parsed.get("text", "")),
        premise_valid=parsed.get("premise_valid") is True,
        facts=_string_map(parsed.get("facts")),
        attributes=_string_map(parsed.get("attributes")),
        citations=_string_tuple(parsed.get("citations")),
        expected_label=base_case.target.expected_label,
    )
    scored_case = replace(base_case, target=target)
    row = exp5325.evaluate_fixture((scored_case,))["case_results"][0]
    unsafe_false_accept = bool(row["expected_accept"] is False and row["accepted"] is True)
    return {
        "prompt_id": receipt["prompt_id"],
        "kind": "rewrite",
        "parse_ok": True,
        "acceptability_matches_expected": bool(row["acceptability_matches_expected"]),
        "accepted": bool(row["accepted"]),
        "expected_accept": bool(row["expected_accept"]),
        "unsafe_false_accept": unsafe_false_accept,
        "target_label": row["target_label"],
        "rejection_reasons": row["rejection_reasons"],
    }


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get(key) is True) / len(rows)


def _score_outputs(
    receipts: Sequence[Mapping[str, Any]],
    prompt_specs: Sequence[Mapping[str, Any]],
    groups: tuple[exp5310.ParaphraseGroup, ...],
    cases: tuple[exp5325.RewriteCase, ...],
) -> JsonDict:
    spec_by_id = {str(spec["prompt_id"]): spec for spec in prompt_specs}
    paraphrase_rows: list[JsonDict] = []
    rewrite_rows: list[JsonDict] = []
    for receipt in receipts:
        if not receipt.get("completed"):
            continue
        spec = spec_by_id[str(receipt["prompt_id"])]
        if spec["kind"] == "paraphrase":
            paraphrase_rows.append(_score_paraphrase_output(receipt, spec, groups))
        else:
            rewrite_rows.append(_score_rewrite_output(receipt, spec, cases))
    return {
        "paraphrase_rows": paraphrase_rows,
        "rewrite_rows": rewrite_rows,
        "paraphrase_label_preservation_rate": _rate(paraphrase_rows, "passed"),
        "rewrite_acceptability_rate": _rate(rewrite_rows, "acceptability_matches_expected"),
        "unsafe_false_accepts": sum(1 for row in rewrite_rows if row["unsafe_false_accept"]),
    }


def _build_artifact(
    *,
    started: float,
    model_specs: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    selected_model: Mapping[str, Any] | None,
    selected_command: Mapping[str, Any] | None,
    generation_receipts: Sequence[Mapping[str, Any]],
    scoring: Mapping[str, Any],
    tests_run: Sequence[Any],
    readiness_blockers: Sequence[str],
    generation_complete: bool,
) -> JsonDict:
    measured = generation_complete and not readiness_blockers
    status = "complete" if measured else "blocked"
    honest = (
        "complete: bounded local SOTA paraphrase/rewrite smoke fixture-scored"
        if measured
        else "blocked_sota_paraphrase_rewrite_smoke_not_measured"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap("honest_verdict", honest),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "MODEL_SPECS": _wrap("MODEL_SPECS", dict(model_specs)),
        "preconditions_checked": _wrap("preconditions_checked", dict(preconditions)),
        "selected_model_spec": _wrap("selected_model_spec", selected_model),
        "selected_backend_command": _wrap("selected_backend_command", selected_command),
        "prompt_count": len(generation_receipts),
        "paraphrase_label_preservation_rate": scoring["paraphrase_label_preservation_rate"],
        "rewrite_acceptability_rate": scoring["rewrite_acceptability_rate"],
        "unsafe_false_accepts": scoring["unsafe_false_accepts"],
        "sota_quality_measured": measured,
        "headline_quality_claim": False,
        "generation_receipts": list(generation_receipts),
        "scoring_results": dict(scoring),
        "readiness_blockers": list(readiness_blockers),
        "tests_run": _wrap("tests_run", list(tests_run)),
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.perf_counter() - started, 6),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = exp5324.exp5323.sha16(
        _stable_json(
            {
                "experiment_id": EXPERIMENT_ID,
                "model_specs": model_specs,
                "selected_model": selected_model,
                "selected_command": selected_command,
                "generation_receipts": generation_receipts,
                "scoring": scoring,
                "seed": RANDOM_SEED,
            }
        )
    )
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    prior_artifact_path: Path | None = None,
    model_resolver: ModelResolver = resolve_cached_gguf,
    current_preconditions_provider: PreconditionsProvider | None = None,
    generation_probe: GenerationProbe = default_generation_probe,
    prompt_specs: Sequence[Mapping[str, Any]] = DEFAULT_PROMPT_SPECS,
    paraphrase_groups: tuple[exp5310.ParaphraseGroup, ...] | None = None,
    rewrite_cases: tuple[exp5325.RewriteCase, ...] | None = None,
    tests_run: Sequence[Any] | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    artifact_path = artifact_path or root / RESULT_RELATIVE_PATH
    prior_artifact_path = prior_artifact_path or root / exp5324.RESULT_RELATIVE_PATH
    prior_artifact = _read_json(prior_artifact_path)
    selected_command, prior_selected_model = _selected_from_prior(prior_artifact)
    current_preconditions_provider = current_preconditions_provider or (
        lambda: exp5324.exp5323.collect_preconditions(root)
    )
    current_preconditions = dict(current_preconditions_provider())
    model_specs = _resolve_model_specs(model_resolver)
    selected_role = str((prior_selected_model or {}).get("role") or "")
    selected_model = model_specs.get(selected_role) if selected_role else None
    fixture_status, groups, cases = _fixture_preconditions(paraphrase_groups, rewrite_cases)
    blockers = _precondition_blockers(
        selected_command=selected_command,
        prior_selected_model=prior_selected_model,
        selected_model=selected_model,
        current_preconditions=current_preconditions,
        fixture_status=fixture_status,
    )
    preconditions: JsonDict = {
        **current_preconditions,
        **fixture_status,
        "exp5324_artifact_path": str(prior_artifact_path),
        "exp5324_selected_backend_command_present": selected_command is not None,
        "selected_model_role": selected_role or None,
        "selected_model_file_present": bool(
            selected_model
            and selected_model.get("model_path")
            and Path(str(selected_model["model_path"])).is_file()
        ),
        "blocked_preconditions": blockers,
    }

    generation_receipts: list[JsonDict] = []
    readiness_blockers = list(blockers)
    if not blockers and selected_command is not None:
        base_command = selected_command["command"]
        timeout_s = float(selected_command.get("timeout_s") or exp5324.exp5323.DEFAULT_TIMEOUT_S)
        for index, prompt_spec in enumerate(prompt_specs, start=1):
            n_predict = int(prompt_spec.get("n_predict") or DEFAULT_N_PREDICT)
            command = command_for_prompt(
                base_command,
                str(prompt_spec["prompt"]),
                n_predict=n_predict,
                seed=RANDOM_SEED + index,
            )
            raw = generation_probe(
                prompt_spec=prompt_spec,
                command=command,
                timeout_s=timeout_s,
                run_index=index,
            )
            receipt = _normalize_generation_receipt(
                raw,
                prompt_spec=prompt_spec,
                command=command,
                timeout_s=timeout_s,
            )
            generation_receipts.append(receipt)
            if not receipt["completed"]:
                readiness_blockers.append(f"generation failed: {prompt_spec['prompt_id']}")
                break

    scoring = _score_outputs(generation_receipts, prompt_specs, groups, cases)
    generation_complete = len(generation_receipts) == len(prompt_specs) and all(
        receipt["completed"] for receipt in generation_receipts
    )
    artifact = _build_artifact(
        started=started,
        model_specs=model_specs,
        preconditions=preconditions,
        selected_model=selected_model,
        selected_command=selected_command,
        generation_receipts=generation_receipts,
        scoring=scoring,
        tests_run=[] if tests_run is None else tests_run,
        readiness_blockers=readiness_blockers,
        generation_complete=generation_complete,
    )
    if write:
        write_json(artifact_path, artifact)
    return artifact


def _wrapped_value(artifact: Mapping[str, Any], field: str) -> Any:
    value = artifact.get(field)
    if not isinstance(value, Mapping):
        return MISSING_WRAPPED_VALUE
    if value.get("principle") != FIELD_PRINCIPLES.get(field):
        return MISSING_WRAPPED_VALUE
    return value.get("value")


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    for field in WRAPPED_FIELDS:
        if field in artifact and _wrapped_value(artifact, field) is MISSING_WRAPPED_VALUE:
            errors.append(f"{field} must be principle-wrapped")
    if _wrapped_value(artifact, "experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id mismatch")
    if _wrapped_value(artifact, "milestone") != MILESTONE:
        errors.append("milestone mismatch")
    if _wrapped_value(artifact, "status") not in {"complete", "blocked"}:
        errors.append("status must be complete or blocked")
    honest = _wrapped_value(artifact, "honest_verdict")
    if not isinstance(honest, str) or not honest.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete: or blocked_")
    if _wrapped_value(artifact, "inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    tests_run = _wrapped_value(artifact, "tests_run")
    if tests_run is not MISSING_WRAPPED_VALUE and not isinstance(tests_run, list):
        errors.append("tests_run must be a list")
    if not isinstance(artifact.get("prompt_count"), int):
        errors.append("prompt_count must be a bare integer")
    for field in ("paraphrase_label_preservation_rate", "rewrite_acceptability_rate"):
        value = artifact.get(field)
        if not isinstance(value, int | float) or not 0.0 <= float(value) <= 1.0:
            errors.append(f"{field} must be in [0, 1]")
    if not isinstance(artifact.get("unsafe_false_accepts"), int):
        errors.append("unsafe_false_accepts must be a bare integer")
    if not isinstance(artifact.get("sota_quality_measured"), bool):
        errors.append("sota_quality_measured must be a bare boolean")
    if artifact.get("headline_quality_claim") is not False:
        errors.append("headline_quality_claim must be bare false")
    model_specs = _wrapped_value(artifact, "MODEL_SPECS")
    if not isinstance(model_specs, Mapping):
        errors.append("MODEL_SPECS must be an object")
    else:
        expected_roles = {str(spec["role"]) for spec in MANDATED_MODEL_SPECS}
        if set(model_specs) != expected_roles:
            errors.append("MODEL_SPECS roles mismatch")
        expected_hf = {str(spec["role"]): str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS}
        for role in expected_roles & set(model_specs):
            spec = model_specs[role]
            if spec.get("hf_id") != expected_hf[role]:
                errors.append("hf_id mismatch for mandated model role")
    selected_model = _wrapped_value(artifact, "selected_model_spec")
    if artifact.get("sota_quality_measured") is True:
        if _wrapped_value(artifact, "status") != "complete":
            errors.append("measured artifact must have complete status")
        if not isinstance(selected_model, Mapping):
            errors.append("selected_model_spec must be an object when measured")
    else:
        if _wrapped_value(artifact, "status") != "blocked":
            errors.append("unmeasured artifact must have blocked status")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise AssertionError("; ".join(errors))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--prior", type=Path, default=REPO_ROOT / exp5324.RESULT_RELATIVE_PATH)
    parser.add_argument(
        "--tests-run-json",
        default="[]",
        help="JSON list of validation commands to embed in the artifact.",
    )
    args = parser.parse_args(argv)
    artifact = run(
        artifact_path=args.out,
        prior_artifact_path=args.prior,
        tests_run=json.loads(args.tests_run_json),
        write=True,
    )
    print(
        f"[exp5326] status={artifact['status']['value']} "
        f"measured={artifact['sota_quality_measured']} out={args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
