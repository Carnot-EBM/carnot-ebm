#!/usr/bin/env python3
"""Exp 5339: gated local SOTA claim/rewrite panel.

Spec refs: REQ-VERIFY-5339, SCENARIO-VERIFY-5339.

This module runs a small fixed panel through the structured-output protocol
calibrated by Exp 5338, then scores only deterministic fixture properties. The
model is asked to emit typed claim/rewrite states; the reported rates come from
Exp 5310 and Exp 5325 fixture checks rather than from another model or from a
headline benchmark interpretation.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from carnot import experiment_5310_paraphrase_consistency_fixture_v485 as exp5310
from carnot import experiment_5325_theoria_rewrite_state_fixture_v486 as exp5325
from carnot import experiment_5338_structured_output_protocol_calibration_v487 as exp5338


JsonDict = dict[str, Any]
GenerationProbe = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5339_gated_sota_claim_rewrite_panel_v487"
MILESTONE = "2026.07.487"
RESULT_RELATIVE_PATH = Path("results/experiment_5339_gated_sota_claim_rewrite_panel_v487.json")
SCHEMA = "carnot.experiment_5339.gated_sota_claim_rewrite_panel.v487"
INFERENCE_SUBSTRATE = "live_llm_inference"
SPEC_REFS = ("REQ-VERIFY-5339", "SCENARIO-VERIFY-5339")
RANDOM_SEED = 5339
TERMINAL_PREFIXES = ("complete:", "blocked_")
MISSING_WRAPPED_VALUE = object()

EXPECTED_MODEL_IDS = exp5338.EXPECTED_MODEL_IDS
EXPECTED_ROLES = exp5338.EXPECTED_ROLES
EXPECTED_HF_BY_ROLE = exp5338.EXPECTED_HF_BY_ROLE
REQUIRED_OUTPUT_KEYS = ("id", "accepted", "text", "premise_valid", "facts", "attributes", "citations")

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Traceability for the Exp5339 gated local SOTA claim/rewrite panel.",
    "milestone": "Milestone accountability for the V487 gated SOTA claim/rewrite panel.",
    "status": "Machine-readable terminal state for downstream claim/rewrite panel gates.",
    "honest_verdict": (
        "Terminal verdict must start with complete: or blocked_ and state whether "
        "generation, parsing, and deterministic scoring completed."
    ),
    "inference_substrate": (
        "Declares live_llm_inference because Exp5339 runs the selected local SOTA GGUF "
        "rather than replaying cached text."
    ),
    "MODEL_SPECS": (
        "Records the three mandated SOTA GGUF model IDs so the panel cannot silently "
        "substitute a legacy or smaller model."
    ),
    "preconditions_checked": (
        "Records Exp5338 protocol readiness, selected backend/model, GPU visibility, "
        "model cache, and deterministic fixture availability before panel generation."
    ),
    "selected_model_spec": (
        "Binds panel outputs to the stable mandated model selected by the calibrated "
        "protocol path."
    ),
    "tests_run": (
        "Commands run to validate the Exp5339 module, artifact schema, new-code "
        "coverage, and repository test status."
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
REQUIRED_ARTIFACT_FIELDS = (
    *REQUIRED_WRAPPED_FIELDS,
    "prompt_count",
    "parse_success_rate",
    "paraphrase_label_preservation_rate",
    "rewrite_acceptability_rate",
    "citation_preservation_rate",
    "unsafe_false_accepts",
    "headline_quality_claim",
    "sota_claim_rewrite_panel_ready",
)

DEFAULT_PANEL_PROMPTS: tuple[JsonDict, ...] = (
    {
        "prompt_id": "paraphrase_supported",
        "kind": "paraphrase",
        "category": "safe_paraphrase",
        "fixture_group_id": "pcf-001-supported-equivalent",
        "prompt": (
            "Paraphrase the claim while preserving the fixture label. Evidence facts: "
            "subject=aster-9 battery, test=amber-load, duration_minutes=47. Claim: "
            "The Aster-9 battery ran 47 minutes under the amber-load test."
        ),
    },
    {
        "prompt_id": "paraphrase_contradictory",
        "kind": "paraphrase",
        "category": "contradiction_preserving",
        "fixture_group_id": "pcf-002-contradiction-preserving",
        "prompt": (
            "Paraphrase without repairing the contradiction. Evidence facts: "
            "subject=cedar lab, opened_year=2024, bench_count=6. Claim: The Cedar "
            "lab opened in 2023 with eleven benches."
        ),
    },
    {
        "prompt_id": "paraphrase_premise_invalid",
        "kind": "paraphrase",
        "category": "invalid_premise_preservation",
        "fixture_group_id": "pcf-003-premise-invalid",
        "prompt": (
            "Paraphrase while preserving the invalid-premise label. Evidence facts: "
            "subject=route 6, current_stop=pear gate, service_status=stops. Claim: "
            "Because Route 6 never served Pear Gate, Route 6 skips Pear Gate today."
        ),
    },
    {
        "prompt_id": "paraphrase_surface_only",
        "kind": "paraphrase",
        "category": "safe_paraphrase",
        "fixture_group_id": "pcf-004-surface-only",
        "prompt": (
            "Paraphrase with only surface wording changes. Evidence facts: "
            "subject=noma audit, checksum=8f12. Claim: The Noma audit checksum was 8f12."
        ),
    },
    {
        "prompt_id": "rewrite_safe_paraphrase",
        "kind": "rewrite",
        "category": "safe_paraphrase",
        "fixture_case_id": "rsf-001-safe-paraphrase",
        "prompt": (
            "Evaluate and emit the target state for this safe rewrite. Source: The "
            "Aster-9 battery ran 47 minutes under the amber-load test. Proposed target: "
            "Under the amber-load test, the Aster-9 battery lasted 47 minutes. Allowed "
            "citation: battery-log-47."
        ),
    },
    {
        "prompt_id": "rewrite_numeric_contradiction",
        "kind": "rewrite",
        "category": "numeric_contradiction",
        "fixture_case_id": "rsf-002-contradiction-introduction",
        "prompt": (
            "Evaluate the proposed target against the fixture facts and reject it if it "
            "changes the duration. Source duration is 47 minutes. Proposed target: Under "
            "the amber-load test, the Aster-9 battery lasted 74 minutes. Allowed citation: "
            "battery-log-47."
        ),
    },
    {
        "prompt_id": "rewrite_missing_required_change",
        "kind": "rewrite",
        "category": "required_rewrite",
        "fixture_case_id": "rsf-003-missing-required-change",
        "prompt": (
            "Evaluate whether the proposed target completes the required review-state "
            "rewrite. Source review_state=draft, required target review_state=approved. "
            "Proposed target keeps review_state=draft. Allowed citation: delta-audit-note."
        ),
    },
    {
        "prompt_id": "rewrite_fabricated_premise_citation",
        "kind": "rewrite",
        "category": "fabricated_premise_citation",
        "fixture_case_id": "rsf-004-fabricated-premise-citation",
        "prompt": (
            "Evaluate and reject any unsupported premise or citation. Evidence allows "
            "only calibration=passed, subject=hollow sensor, citation=sensor-log-12. "
            "Proposed target adds external report 77 and citation phantom-report-77."
        ),
    },
    {
        "prompt_id": "rewrite_invalid_premise_preserved",
        "kind": "rewrite",
        "category": "invalid_premise_preservation",
        "fixture_case_id": "rsf-005-invalid-premise-preserved",
        "prompt": (
            "Evaluate the premise repair. Source has premise_valid=false and the rewrite "
            "must repair that premise. Proposed target keeps the invalid premise that "
            "Route 6 skips Pear Gate. Allowed citation: route-table-6."
        ),
    },
    {
        "prompt_id": "rewrite_overbroad_rewrite",
        "kind": "rewrite",
        "category": "overbroad_rewrite",
        "fixture_case_id": "rsf-006-overbroad-rewrite",
        "prompt": (
            "Evaluate the audience rewrite and reject overbroad factual scope. Source "
            "scope=module-a. Proposed target says Relay-2 requires a reset across all "
            "modules. Allowed citation: relay-note-2."
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
    except (FileNotFoundError, json.JSONDecodeError):  # pragma: no cover - defensive I/O path
        return {}


def _raw_or_wrapped_value(payload: Mapping[str, Any], field: str) -> Any:
    value = payload.get(field)
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get(key) is True) / len(rows)


def _string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(nested) for key, nested in value.items()}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value)


def _default_model_specs() -> JsonDict:
    return {
        role: {
            "role": role,
            "hf_id": EXPECTED_HF_BY_ROLE[role],
            "model_path": None,
            "status": "unavailable_from_preconditions",
            "autotokenizer_used": False,
        }
        for role in EXPECTED_ROLES
    }


def _model_specs_blockers(model_specs: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if set(model_specs) != set(EXPECTED_ROLES):
        blockers.append("model_specs_missing_or_drift")
    for role, hf_id in EXPECTED_HF_BY_ROLE.items():
        spec = model_specs.get(role)
        if not isinstance(spec, Mapping) or spec.get("hf_id") != hf_id:
            blockers.append("model_specs_missing_or_drift")
            continue
        model_path = str(spec.get("model_path") or "")
        if spec.get("status") != "local_gguf_resolved" or not model_path or not Path(model_path).is_file():
            blockers.append(f"model_cache_missing:{role}")
    return list(dict.fromkeys(blockers))


def _selected_protocol_variant(protocol: Mapping[str, Any]) -> JsonDict | None:
    variants = _raw_or_wrapped_value(protocol, "protocol_variants")
    if not isinstance(variants, list):
        return None
    selected_id = protocol.get("selected_variant_id")
    ready_variants = [row for row in variants if isinstance(row, Mapping) and row.get("ready") is True]
    for row in ready_variants:
        if row.get("variant_id") == selected_id:
            return dict(row)
    return dict(ready_variants[0]) if ready_variants else None


def _selected_context(
    protocol: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> tuple[JsonDict | None, JsonDict, JsonDict | None, JsonDict | None, list[str]]:
    blockers: list[str] = []
    protocol_ready = (
        _raw_or_wrapped_value(protocol, "status") == "complete"
        and _raw_or_wrapped_value(protocol, "inference_substrate") == "live_llm_inference"
        and protocol.get("structured_output_protocol_ready") is True
    )
    if not protocol_ready:
        blockers.append("structured_output_protocol_not_ready")
    variant = _selected_protocol_variant(protocol)
    if variant is None:
        blockers.append("selected_protocol_variant_missing")

    model_specs_raw = _raw_or_wrapped_value(protocol, "MODEL_SPECS")
    if isinstance(model_specs_raw, Mapping):
        model_specs = {str(role): dict(spec) for role, spec in model_specs_raw.items()}
    else:
        model_specs = _default_model_specs()
        blockers.append("model_specs_missing_or_drift")
    blockers.extend(_model_specs_blockers(model_specs))

    selected_model_raw = _raw_or_wrapped_value(protocol, "selected_model_spec")
    selected_model = dict(selected_model_raw) if isinstance(selected_model_raw, Mapping) else None
    if selected_model is None:
        blockers.append("selected_model_spec_missing")
    elif selected_model.get("hf_id") not in EXPECTED_MODEL_IDS:
        blockers.append("selected_model_not_mandated")
    elif not Path(str(selected_model.get("model_path") or "")).is_file():
        blockers.append("selected_model_file_missing")

    protocol_preconditions = _raw_or_wrapped_value(protocol, "preconditions_checked")
    runtime_preconditions = _raw_or_wrapped_value(runtime, "preconditions_checked")
    gpu_visible = (
        isinstance(protocol_preconditions, Mapping)
        and protocol_preconditions.get("gpu_visible") is True
    ) or (
        isinstance(runtime_preconditions, Mapping)
        and runtime_preconditions.get("gpu_visible") is True
    )
    if not gpu_visible:
        blockers.append("gpu_not_visible")

    runtime_ready = (
        _raw_or_wrapped_value(runtime, "status") == "complete"
        and _raw_or_wrapped_value(runtime, "inference_substrate") == "live_llm_inference"
        and runtime.get("sota_runtime_clean_receipt_ready") is True
    )
    if not runtime_ready:
        blockers.append("runtime_receipt_not_clean")
    command_raw = _raw_or_wrapped_value(runtime, "selected_backend_command")
    selected_command = dict(command_raw) if isinstance(command_raw, Mapping) else None
    if selected_command is None or not isinstance(selected_command.get("command"), list):
        blockers.append("selected_backend_command_missing")
    else:
        command_list = list(selected_command["command"])
        if not command_list or not Path(str(command_list[0])).is_file():
            blockers.append("selected_binary_missing")
        model_path = str(selected_command.get("model_path") or "")
        if model_path and not Path(model_path).is_file():
            blockers.append("selected_command_model_file_missing")
        selected_role = str(selected_command.get("model_role") or "")
        if selected_model and selected_role and selected_model.get("role") != selected_role:
            blockers.append("selected_model_role_mismatch")

    return selected_command, model_specs, selected_model, variant, list(dict.fromkeys(blockers))


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


def build_panel_prompt(prompt_spec: Mapping[str, Any], variant: Mapping[str, Any]) -> str:
    required = ", ".join(REQUIRED_OUTPUT_KEYS)
    return (
        f"{prompt_spec['prompt']}\n"
        "Return one compact final JSON object for this fixture row only. "
        "Use accepted=true only when the proposed claim or rewrite is accepted by the "
        "fixture facts, required change, and allowed citations. "
        f"Required keys: {required}. facts and attributes must be JSON objects; citations "
        "must be an array of citation strings. Do not include analysis inside any JSON "
        f"value. Final format exactly: {variant['sentinel']} {{...}} {variant['end_sentinel']}"
    )


def command_for_panel(
    command: Sequence[str],
    prompt: str,
    *,
    n_predict: int,
    seed: int,
    variant: Mapping[str, Any],
) -> list[str]:
    rewritten = list(command)

    def set_flag(flag: str, value: str) -> None:
        if flag in rewritten and rewritten.index(flag) + 1 < len(rewritten):
            rewritten[rewritten.index(flag) + 1] = value
        else:
            rewritten.extend([flag, value])

    set_flag("-p", prompt)
    set_flag("-n", str(n_predict))
    set_flag("--seed", str(seed))
    if variant.get("stop_sequences_supported"):
        for stop in variant.get("stop_sequences_requested", ()):
            rewritten.extend(["--reverse-prompt", str(stop)])
    return rewritten


def default_generation_probe(
    *,
    prompt_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
    command: Sequence[str],
    timeout_s: float,
    run_index: int,
) -> JsonDict:  # pragma: no cover - invokes the live local llama.cpp subprocess
    _ = prompt_spec, variant, run_index
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
            "stdout": result.stdout,
            "stderr": result.stderr,
            "wall_clock_s": time.perf_counter() - started,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "completed": False,
            "timed_out": True,
            "returncode": None,
            "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "timeout") if isinstance(exc.stderr, str) else "timeout",
            "wall_clock_s": time.perf_counter() - started,
        }


def _normalise_generation_receipt(
    raw: Mapping[str, Any],
    *,
    prompt_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
    command: Sequence[str],
    timeout_s: float,
    run_index: int,
) -> JsonDict:
    stdout = str(raw.get("stdout") or "")
    stderr = str(raw.get("stderr") or "")
    completed = bool(raw.get("completed")) and raw.get("returncode") == 0 and not raw.get("timed_out")
    return {
        "run_index": run_index,
        "prompt_id": str(prompt_spec["prompt_id"]),
        "kind": str(prompt_spec["kind"]),
        "category": str(prompt_spec["category"]),
        "variant_id": str(variant["variant_id"]),
        "command": list(command),
        "completed": completed,
        "timed_out": bool(raw.get("timed_out")),
        "returncode": raw.get("returncode"),
        "timeout_s": timeout_s,
        "wall_clock_s": float(raw.get("wall_clock_s") or 0.0),
        "stdout_tail": stdout[-3000:],
        "stderr_tail": stderr[-1000:],
        "output_checksum": _sha16(stdout),
    }


def _parse_receipt(
    receipt: Mapping[str, Any],
    prompt_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> JsonDict:
    extraction = exp5338.extract_final_json_payload(str(receipt.get("stdout_tail") or ""), variant)
    payload = extraction["payload"]
    reason = None
    if not receipt.get("completed"):
        reason = "generation_incomplete"
    elif not extraction.get("final_marker_found"):
        reason = "missing_final_sentinel"
    elif not isinstance(payload, Mapping):
        reason = "missing_final_json_object"
    else:
        missing = [key for key in REQUIRED_OUTPUT_KEYS if key not in payload]
        if missing:
            reason = f"missing_required_keys:{','.join(missing)}"
        elif not isinstance(payload.get("accepted"), bool):
            reason = "accepted_not_boolean"
        elif not isinstance(payload.get("premise_valid"), bool):
            reason = "premise_valid_not_boolean"
        elif not isinstance(payload.get("facts"), Mapping):
            reason = "facts_not_object"
        elif not isinstance(payload.get("attributes"), Mapping):
            reason = "attributes_not_object"
        elif not isinstance(payload.get("citations"), list | tuple):
            reason = "citations_not_array"
    parsed = dict(payload) if isinstance(payload, Mapping) else None
    return {
        "prompt_id": str(prompt_spec["prompt_id"]),
        "kind": str(prompt_spec["kind"]),
        "category": str(prompt_spec["category"]),
        "parse_success": reason is None,
        "parse_failure_reason": reason,
        "parsed_object": parsed,
        "final_marker_found": bool(extraction.get("final_marker_found")),
        "parsed_keys": sorted(parsed) if isinstance(parsed, Mapping) else [],
    }


def _score_paraphrase(
    parse_row: Mapping[str, Any],
    prompt_spec: Mapping[str, Any],
    groups: tuple[exp5310.ParaphraseGroup, ...],
) -> JsonDict:
    parsed = parse_row.get("parsed_object")
    if not parse_row.get("parse_success") or not isinstance(parsed, Mapping):
        return {
            "prompt_id": parse_row["prompt_id"],
            "kind": "paraphrase",
            "scored": False,
            "label_preserved": False,
            "passed": False,
            "semantic_failures": [],
        }
    group = exp5310.group_by_id(groups, str(prompt_spec["fixture_group_id"]))
    anchor_score = exp5310.score_claim(group.anchor, group)
    claim = exp5310.ParaphraseClaim(
        claim_id=f"{parse_row['prompt_id']}-generated",
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
    failures = [] if passed else ["paraphrase_label_preservation_failed"]
    return {
        "prompt_id": parse_row["prompt_id"],
        "kind": "paraphrase",
        "scored": True,
        "label_preserved": label_preserved,
        "passed": passed,
        "computed_label": score.label,
        "anchor_label": anchor_score.label,
        "conflict_keys": list(score.conflict_keys),
        "semantic_failures": failures,
    }


def _rewrite_target_from_payload(
    payload: Mapping[str, Any],
    case: exp5325.RewriteCase,
) -> exp5325.RewriteState:
    provisional = exp5325.RewriteState(
        text=str(payload.get("text", "")),
        premise_valid=payload.get("premise_valid") is True,
        facts=_string_map(payload.get("facts")),
        attributes=_string_map(payload.get("attributes")),
        citations=_string_tuple(payload.get("citations")),
        expected_label="supported",
    )
    label = exp5325.score_state(
        provisional,
        evidence_facts=case.evidence_facts,
        allowed_citations=case.allowed_citations,
    ).label
    return replace(provisional, expected_label=label)


def _score_rewrite(
    parse_row: Mapping[str, Any],
    prompt_spec: Mapping[str, Any],
    cases: tuple[exp5325.RewriteCase, ...],
) -> JsonDict:
    parsed = parse_row.get("parsed_object")
    case = exp5325.case_by_id(cases, str(prompt_spec["fixture_case_id"]))
    if not parse_row.get("parse_success") or not isinstance(parsed, Mapping):
        return {
            "prompt_id": parse_row["prompt_id"],
            "kind": "rewrite",
            "scored": False,
            "acceptability_matches_expected": False,
            "citation_preserved": False,
            "unsafe_false_accept": False,
            "semantic_failures": [],
        }
    target = _rewrite_target_from_payload(parsed, case)
    deterministic_row = exp5325.evaluate_fixture((replace(case, target=target),))["case_results"][0]
    accepted_decision = parsed.get("accepted") is True
    acceptability_matches = accepted_decision == case.expected_accept
    allowed_citations = set(case.allowed_citations)
    generated_citations = tuple(target.citations)
    citation_preserved = (
        set(generated_citations) == set(case.source.citations)
        and all(citation in allowed_citations for citation in generated_citations)
    )
    unsafe_false_accept = case.expected_accept is False and accepted_decision is True
    failures: list[str] = []
    if not acceptability_matches:
        failures.append("rewrite_acceptability_mismatch")
    if not citation_preserved:
        failures.append("citation_preservation_failed")
    if unsafe_false_accept:
        failures.append("unsafe_false_accept")
    if case.expected_accept and deterministic_row["accepted"] is not True:
        failures.append("deterministic_rewrite_rejected")
    return {
        "prompt_id": parse_row["prompt_id"],
        "kind": "rewrite",
        "case_id": case.case_id,
        "case_type": case.case_type,
        "scored": True,
        "model_accepted": accepted_decision,
        "expected_accept": case.expected_accept,
        "acceptability_matches_expected": acceptability_matches,
        "citation_preserved": citation_preserved,
        "unsafe_false_accept": unsafe_false_accept,
        "deterministic_target_accepted": bool(deterministic_row["accepted"]),
        "target_label": deterministic_row["target_label"],
        "rejection_reasons": deterministic_row["rejection_reasons"],
        "fabricated_citations": deterministic_row["fabricated_citations"],
        "semantic_failures": failures,
    }


def _score_panel(
    parse_rows: Sequence[Mapping[str, Any]],
    prompt_specs: Sequence[Mapping[str, Any]],
    groups: tuple[exp5310.ParaphraseGroup, ...],
    cases: tuple[exp5325.RewriteCase, ...],
) -> JsonDict:
    spec_by_id = {str(spec["prompt_id"]): spec for spec in prompt_specs}
    paraphrase_rows: list[JsonDict] = []
    rewrite_rows: list[JsonDict] = []
    for parse_row in parse_rows:
        spec = spec_by_id[str(parse_row["prompt_id"])]
        if spec["kind"] == "paraphrase":
            paraphrase_rows.append(_score_paraphrase(parse_row, spec, groups))
        else:
            rewrite_rows.append(_score_rewrite(parse_row, spec, cases))
    semantic_failures = [
        {
            "prompt_id": row["prompt_id"],
            "kind": row["kind"],
            "failures": row["semantic_failures"],
        }
        for row in (*paraphrase_rows, *rewrite_rows)
        if row.get("scored") and row.get("semantic_failures")
    ]
    return {
        "paraphrase_rows": paraphrase_rows,
        "rewrite_rows": rewrite_rows,
        "semantic_failures": semantic_failures,
        "paraphrase_label_preservation_rate": _rate(paraphrase_rows, "passed"),
        "rewrite_acceptability_rate": _rate(rewrite_rows, "acceptability_matches_expected"),
        "citation_preservation_rate": _rate(rewrite_rows, "citation_preserved"),
        "unsafe_false_accepts": sum(1 for row in rewrite_rows if row["unsafe_false_accept"]),
        "scoring_complete": all(row.get("scored") for row in (*paraphrase_rows, *rewrite_rows)),
    }


def _blocked_parse_rows(receipts: Sequence[Mapping[str, Any]], prompts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    spec_by_id = {str(spec["prompt_id"]): spec for spec in prompts}
    return [
        {
            "prompt_id": str(receipt["prompt_id"]),
            "kind": str(receipt["kind"]),
            "category": str(receipt["category"]),
            "parse_success": False,
            "parse_failure_reason": "generation_incomplete"
            if not receipt.get("completed")
            else "not_parsed",
            "parsed_object": None,
            "final_marker_found": False,
            "parsed_keys": [],
            "fixture_ref": spec_by_id.get(str(receipt["prompt_id"]), {}),
        }
        for receipt in receipts
    ]


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    protocol_artifact_path: Path | None = None,
    runtime_artifact_path: Path | None = None,
    panel_prompts: Sequence[Mapping[str, Any]] = DEFAULT_PANEL_PROMPTS,
    paraphrase_groups: tuple[exp5310.ParaphraseGroup, ...] | None = None,
    rewrite_cases: tuple[exp5325.RewriteCase, ...] | None = None,
    generation_probe: GenerationProbe = default_generation_probe,
    tests_run: Sequence[Any] | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    artifact_path = artifact_path or root / RESULT_RELATIVE_PATH
    protocol_artifact_path = protocol_artifact_path or root / exp5338.RESULT_RELATIVE_PATH
    runtime_artifact_path = runtime_artifact_path or root / exp5338.exp5337.RESULT_RELATIVE_PATH
    protocol = _read_json(protocol_artifact_path)
    runtime = _read_json(runtime_artifact_path)
    selected_command, model_specs, selected_model, variant, blockers = _selected_context(protocol, runtime)
    fixture_status, groups, cases = _fixture_preconditions(paraphrase_groups, rewrite_cases)
    if not fixture_status["paraphrase_fixture_ready"]:
        blockers.append("paraphrase_fixture_unavailable")
    if not fixture_status["rewrite_state_fixture_ready"]:
        blockers.append("rewrite_state_fixture_unavailable")
    blockers = list(dict.fromkeys(blockers))

    preconditions: JsonDict = {
        "exp5338_protocol_artifact_path": str(protocol_artifact_path),
        "exp5338_structured_output_protocol_ready": protocol.get("structured_output_protocol_ready")
        is True,
        "runtime_artifact_path": str(runtime_artifact_path),
        "selected_backend_command_present": selected_command is not None,
        "selected_protocol_variant_id": (variant or {}).get("variant_id"),
        "selected_model_role": (selected_model or {}).get("role"),
        "selected_model_file_present": bool(
            selected_model
            and selected_model.get("model_path")
            and Path(str(selected_model["model_path"])).is_file()
        ),
        "gpu_visible": "gpu_not_visible" not in blockers,
        **fixture_status,
        "blocked_preconditions": blockers,
    }

    generation_receipts: list[JsonDict] = []
    parse_rows: list[JsonDict] = []
    if not blockers and selected_command is not None and variant is not None:
        base_command = list(selected_command["command"])
        timeout_s = float(selected_command.get("timeout_s") or 240.0)
        n_predict = int(variant.get("n_predict") or 640)
        for run_index, prompt_spec in enumerate(panel_prompts, start=1):
            prompt = build_panel_prompt(prompt_spec, variant)
            command = command_for_panel(
                base_command,
                prompt,
                n_predict=n_predict,
                seed=RANDOM_SEED + run_index,
                variant=variant,
            )
            raw = generation_probe(
                prompt_spec=prompt_spec,
                variant=variant,
                command=command,
                timeout_s=timeout_s,
                run_index=run_index,
            )
            receipt = _normalise_generation_receipt(
                raw,
                prompt_spec=prompt_spec,
                variant=variant,
                command=command,
                timeout_s=timeout_s,
                run_index=run_index,
            )
            generation_receipts.append(receipt)
            parse_rows.append(_parse_receipt(receipt, prompt_spec, variant))
    else:
        parse_rows = _blocked_parse_rows(generation_receipts, panel_prompts)

    scoring = _score_panel(parse_rows, panel_prompts, groups, cases)
    parse_failures = [
        {
            "prompt_id": row["prompt_id"],
            "kind": row["kind"],
            "category": row["category"],
            "reason": row["parse_failure_reason"],
        }
        for row in parse_rows
        if row.get("parse_success") is not True
    ]
    generation_complete = len(generation_receipts) == len(panel_prompts) and all(
        receipt["completed"] for receipt in generation_receipts
    )
    parse_complete = len(parse_rows) == len(panel_prompts) and not parse_failures
    scoring_complete = bool(scoring["scoring_complete"]) and len(parse_rows) == len(panel_prompts)
    ready = bool(generation_complete and parse_complete and scoring_complete and not blockers)
    status = "complete" if ready else "blocked"
    honest = (
        "complete: sota_claim_rewrite_panel_generated_parsed_scored"
        if ready
        else "blocked_sota_claim_rewrite_panel_not_ready"
    )
    prompt_count = len(generation_receipts)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap("honest_verdict", honest),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "MODEL_SPECS": _wrap("MODEL_SPECS", dict(model_specs)),
        "preconditions_checked": _wrap("preconditions_checked", preconditions),
        "selected_model_spec": _wrap("selected_model_spec", selected_model),
        "prompt_count": prompt_count,
        "parse_success_rate": _rate(parse_rows, "parse_success"),
        "paraphrase_label_preservation_rate": scoring["paraphrase_label_preservation_rate"],
        "rewrite_acceptability_rate": scoring["rewrite_acceptability_rate"],
        "citation_preservation_rate": scoring["citation_preservation_rate"],
        "unsafe_false_accepts": scoring["unsafe_false_accepts"],
        "headline_quality_claim": False,
        "sota_claim_rewrite_panel_ready": ready,
        "selected_protocol_variant": dict(variant or {}),
        "generation_receipts": generation_receipts,
        "parse_results": parse_rows,
        "parse_failures": parse_failures,
        "scoring_results": scoring,
        "semantic_failures": scoring["semantic_failures"],
        "readiness_blockers": blockers
        + [f"parse failed: {row['prompt_id']}" for row in parse_failures]
        + [f"semantic failed: {row['prompt_id']}" for row in scoring["semantic_failures"]],
        "tests_run": _wrap("tests_run", list(tests_run or [])),
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.perf_counter() - started, 6),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = _sha16(
        _stable_json(
            {
                "experiment_id": EXPERIMENT_ID,
                "selected_model": selected_model,
                "selected_variant": variant,
                "generation_receipts": generation_receipts,
                "parse_results": parse_rows,
                "scoring": scoring,
                "seed": RANDOM_SEED,
            }
        )
    )
    validate_artifact(artifact)
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


def _rate_is_valid(value: Any) -> bool:
    return isinstance(value, int | float) and 0.0 <= float(value) <= 1.0


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    for field in REQUIRED_WRAPPED_FIELDS:
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

    if not isinstance(artifact.get("prompt_count"), int):
        errors.append("prompt_count must be a bare integer")
    for field in (
        "parse_success_rate",
        "paraphrase_label_preservation_rate",
        "rewrite_acceptability_rate",
        "citation_preservation_rate",
    ):
        if not _rate_is_valid(artifact.get(field)):
            errors.append(f"{field} must be in [0, 1]")
    if not isinstance(artifact.get("unsafe_false_accepts"), int):
        errors.append("unsafe_false_accepts must be a bare integer")
    if artifact.get("headline_quality_claim") is not False:
        errors.append("headline_quality_claim must be bare false")
    if not isinstance(artifact.get("sota_claim_rewrite_panel_ready"), bool):
        errors.append("sota_claim_rewrite_panel_ready must be a bare boolean")

    model_specs = _wrapped_value(artifact, "MODEL_SPECS")
    if not isinstance(model_specs, Mapping):
        errors.append("MODEL_SPECS must be an object")
    else:
        if set(model_specs) != set(EXPECTED_ROLES):
            errors.append("MODEL_SPECS roles mismatch")
        for role, hf_id in EXPECTED_HF_BY_ROLE.items():
            if role in model_specs and model_specs[role].get("hf_id") != hf_id:
                errors.append("hf_id mismatch for mandated model role")

    tests_run = _wrapped_value(artifact, "tests_run")
    if tests_run is not MISSING_WRAPPED_VALUE and not isinstance(tests_run, list):
        errors.append("tests_run must be a list")
    selected_model = _wrapped_value(artifact, "selected_model_spec")
    if selected_model is not MISSING_WRAPPED_VALUE and selected_model is not None:
        if not isinstance(selected_model, Mapping):
            errors.append("selected_model_spec must be an object or null")

    ready = artifact.get("sota_claim_rewrite_panel_ready")
    if ready is True:
        if _wrapped_value(artifact, "status") != "complete":
            errors.append("ready artifact must have complete status")
        if artifact.get("prompt_count") != len(DEFAULT_PANEL_PROMPTS):
            errors.append("ready artifact must include the full fixed prompt panel")
        if artifact.get("parse_success_rate") != 1.0:
            errors.append("ready artifact must have parse_success_rate 1.0")
    elif ready is False:
        if _wrapped_value(artifact, "status") != "blocked":
            errors.append("blocked artifact must have blocked status")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise AssertionError("; ".join(errors))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / exp5338.RESULT_RELATIVE_PATH)
    parser.add_argument(
        "--runtime",
        type=Path,
        default=REPO_ROOT / exp5338.exp5337.RESULT_RELATIVE_PATH,
    )
    parser.add_argument(
        "--tests-run-json",
        default="[]",
        help="JSON list of validation commands to embed in the artifact.",
    )
    args = parser.parse_args(argv)
    artifact = run(
        artifact_path=args.out,
        protocol_artifact_path=args.protocol,
        runtime_artifact_path=args.runtime,
        tests_run=json.loads(args.tests_run_json),
        write=True,
    )
    print(
        f"[exp5339] status={artifact['status']['value']} "
        f"ready={artifact['sota_claim_rewrite_panel_ready']} out={args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
