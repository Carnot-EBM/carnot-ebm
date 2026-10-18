"""Build the Exp6812 paired operational-handoff proposal corpus.

The experiment freezes source-free action scenarios and presents each one in
two byte-length-matched forms. Three named local GGUF models generate two
candidate actions per prompt. The Exp6811 exact automaton checks those actions
only after their raw bytes are durable.

Spec refs: REQ-CONSTRAINT-6812 and SCENARIO-CONSTRAINT-6812-*.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request

from carnot.agentic.arc_trajectory_supervisor import (
    OperationalObligationSupervisor,
    canonical_event_bytes,
    canonical_obligation_bytes,
    compile_operational_obligations,
)
from carnot.durable_row_checkpoint import (
    DurableRowCheckpoint,
    atomic_write_json,
    complete_row_envelope,
)
from carnot.inference.sota_models import resolve_cached_gguf


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_6812_sota_operational_handoff_corpus_v2.json"
SCHEMA = "carnot.experiment_6812.sota_operational_handoff_corpus_v2.v1"
BLOCKED_STATUS = "complete_blocked_sota_operational_handoff_corpus_v2"
MODEL_SPECS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_ROLES = ("flagship_moe", "flagship_dense", "secondary_moe")
MODEL_HASHES = {
    MODEL_SPECS[0]: "sha256:ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61",
    MODEL_SPECS[1]: "sha256:9fdf3dc8b0384830b4402d151388c140bd8eb2abf8d60588d8224231198254a1",
    MODEL_SPECS[2]: "sha256:34c746b1d50ab813e29cd46c4796e3f43c741901a582f93a67b55b9fc9687b35",
}
SCENARIO_FAMILIES = (
    "stale_prerequisites",
    "competing_authorities",
    "fallback",
    "consequence",
    "already_safe_proposals",
    "soft_conflict",
)
ARMS = ("direct_typed", "compressed_prose")
OPERATIONAL_FIELDS = (
    "prerequisite",
    "authority",
    "fallback",
    "execution_consequence",
    "priority",
)
VERDICT_CLASSES = frozenset(
    {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
)
DECODE_BUDGETS: JsonDict = {
    "candidate_count": 2,
    "context_size": 4096,
    "max_output_tokens": 384,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "repeat_penalty": 1.0,
    "stop_rules": ["</s>", "<|im_end|>", "<|endoftext|>"],
    "retry_budget": 0,
    "repair_budget": 0,
    "request_timeout_s": 240,
    "load_timeout_s": 360,
    "teardown_timeout_s": 90,
}
INFERENCE_SUBSTRATE = "local llama.cpp CUDA GGUF"
FIELD_PRINCIPLES: JsonDict = {
    "schema": "Versioned artifact contract.",
    "experiment_id": "Stable experiment identity.",
    "title": "Human-readable experiment name.",
    "run_date": "Caller-supplied execution date.",
    "status": "Terminal producer state.",
    "field_principles": "Meaning of every top-level field.",
    "inference_substrate": "Only local llama.cpp CUDA GGUF inference is eligible.",
    "duration_s": "Total elapsed wall duration in seconds.",
    "phase_duration_s": "Separate acquisition, load, inference, and teardown durations.",
    "random_seed": "Frozen cell seeds rather than an inferred global seed.",
    "reproducibility_checksum": "Hash binding the manifest and retained rows.",
    "MODEL_SPECS": "Exact immutable ordered hub identifiers.",
    "model_specs": "Resolved files, hashes, revisions, templates, roles, and limits.",
    "models_used": "Required models with complete authentic phases.",
    "live_model_invoked": "True only after a first generated token is received.",
    "frozen_manifest": "Prompt bytes, identities, and matched budgets frozen pre-inference.",
    "gpu_receipts": "Task ownership, CUDA offload, VRAM, process, and teardown evidence.",
    "checkpoint_receipts": "Atomic cell publication and exact-resume evidence.",
    "raw_output_manifest": "Immutable generated and local API response byte receipts.",
    "rows": "One retained model-scenario-seed-arm-candidate unit.",
    "operational_preservation_by_arm": "Row-derived exact operational preservation rates.",
    "hard_violation_rate_by_arm": "Row-derived hard violation rates.",
    "safe_proposal_identity_by_arm": "Row-derived already-safe byte identity rates.",
    "parse_completion_by_arm": "Syntax completion kept separate from semantics.",
    "legal_support_by_arm": "Row-derived legal candidate headroom.",
    "retry_demand_by_arm": "Row-derived demand under a matched zero-retry budget.",
    "operational_handoff_corpus_ready": "Exact downstream completeness gate.",
    "solve_claim": "Always false because no live ARC solve occurs.",
    "solve_provenance": "Development-proxy claim boundary.",
    "gate_check_summary": "Named expected and observed terminal gate evidence.",
    "verifier_is_oracle": "False; the exact checker is post-generation authority only.",
    "verdict_class": "Closed terminal classification derived from rows and gates.",
    "honest_verdict": "Terminal row-supported statement.",
    "preconditions_checked": "Ordered fail-closed fixture, model, host, CUDA, and lease checks.",
    "planned_row_count": "Frozen expected row denominator.",
}


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes without a trailing newline."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    """Return the repository's prefixed SHA-256 form for exact bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def _action(kind: str, variant: int, purpose: str) -> JsonDict:
    """Build one small canonical action object used by frozen scenarios."""

    return {"data": {"purpose": purpose, "variant": variant}, "kind": kind}


def _obligation(
    obligation_id: str,
    action: Mapping[str, Any],
    *,
    issuer: str,
    order: int,
    priority_class: str,
    weight: int,
    all_of: Sequence[str],
    none_of: Sequence[str],
    fallback: Mapping[str, Any],
    add: Sequence[str],
    remove: Sequence[str],
) -> JsonDict:
    """Build one complete Exp6811 five-field obligation record."""

    return {
        "action": deepcopy(dict(action)),
        "contract": {
            "authority": {"issuer": issuer, "order": order},
            "execution_consequence": {"add": sorted(add), "remove": sorted(remove)},
            "fallback": {"action": deepcopy(dict(fallback)), "reason": "bounded safe fallback"},
            "prerequisite": {"all_of": sorted(all_of), "none_of": sorted(none_of)},
            "priority": {"class": priority_class, "weight": weight},
        },
        "obligation_id": obligation_id,
    }


def _candidate(
    candidate_id: str,
    action: Mapping[str, Any],
    authorities: Sequence[str],
    soft_progress: int,
) -> JsonDict:
    """Build one response-schema candidate with canonical authority order."""

    return {
        "action": deepcopy(dict(action)),
        "authority_chain": sorted(authorities),
        "candidate_id": candidate_id,
        "soft_progress": soft_progress,
    }


def _scenario_core(family: str, variant: int) -> JsonDict:
    """Create one semantic scenario before its two prompt encodings are frozen."""

    suffix = f"{variant:02d}"
    required = _action("EXECUTE", variant, family)
    fallback = _action("FALLBACK", variant, family)
    facts = [f"ready_{suffix}"]
    safe_proposal: JsonDict | None = None

    if family in {"stale_prerequisites", "fallback"}:
        required_fact = f"missing_{suffix}"
        obligations = [
            _obligation(
                f"obl_{suffix}_hard",
                required,
                issuer="system_authority",
                order=0,
                priority_class="hard",
                weight=0,
                all_of=[required_fact],
                none_of=[],
                fallback=fallback,
                add=[f"done_{suffix}"],
                remove=[],
            )
        ]
        if family == "fallback":
            facts = [f"fallback_required_{suffix}"]
        first_action = fallback
        second_action = fallback
        authorities = ["system_authority"]
    elif family == "competing_authorities":
        obligations = [
            _obligation(
                f"obl_{suffix}_binding",
                required,
                issuer="operator_authority",
                order=1,
                priority_class="binding",
                weight=0,
                all_of=facts,
                none_of=[],
                fallback=fallback,
                add=[f"bound_{suffix}"],
                remove=[],
            ),
            _obligation(
                f"obl_{suffix}_hard",
                required,
                issuer="system_authority",
                order=0,
                priority_class="hard",
                weight=0,
                all_of=facts,
                none_of=[],
                fallback=fallback,
                add=[f"done_{suffix}"],
                remove=[],
            ),
        ]
        first_action = required
        second_action = required
        authorities = ["operator_authority", "system_authority"]
    elif family == "soft_conflict":
        alternate = _action("EXPLORE", variant, family)
        obligations = [
            _obligation(
                f"obl_{suffix}_soft_a",
                required,
                issuer="planner_authority",
                order=0,
                priority_class="soft",
                weight=2 + variant,
                all_of=facts,
                none_of=[],
                fallback=fallback,
                add=[f"progress_a_{suffix}"],
                remove=[],
            ),
            _obligation(
                f"obl_{suffix}_soft_b",
                alternate,
                issuer="planner_authority",
                order=1,
                priority_class="soft",
                weight=1 + variant,
                all_of=facts,
                none_of=[],
                fallback=fallback,
                add=[f"progress_b_{suffix}"],
                remove=[],
            ),
        ]
        first_action = required
        second_action = alternate
        authorities = ["planner_authority"]
    else:
        consequences = [f"effect_{suffix}"]
        obligations = [
            _obligation(
                f"obl_{suffix}_hard",
                required,
                issuer="system_authority",
                order=0,
                priority_class="hard",
                weight=0,
                all_of=facts,
                none_of=[],
                fallback=fallback,
                add=consequences,
                remove=[f"pending_{suffix}"] if family == "consequence" else [],
            )
        ]
        first_action = required
        second_action = required
        authorities = ["system_authority"]
        if family == "already_safe_proposals":
            safe_proposal = deepcopy(required)

    return {
        "scenario_id": f"{family}-{suffix}",
        "family": family,
        "variant": variant,
        "source_free": True,
        "random_seed": 681200 + SCENARIO_FAMILIES.index(family) * 100 + variant,
        "observed_facts": sorted(facts),
        "obligations": sorted(obligations, key=lambda item: item["obligation_id"]),
        "fallback_action": deepcopy(fallback),
        "safe_proposal": safe_proposal,
        "reference_candidates": [
            _candidate("candidate_0", first_action, authorities, variant),
            _candidate("candidate_1", second_action, authorities, variant + 1),
        ],
    }


def _direct_prompt(scenario: Mapping[str, Any]) -> bytes:
    """Encode the full typed handoff and exact response schema as prompt bytes."""

    payload = {
        "instruction": "Return only one JSON object. Preserve the operational contract.",
        "handoff": {
            "base_proposal": scenario["safe_proposal"],
            "obligations": scenario["obligations"],
            "observed_facts": scenario["observed_facts"],
        },
        "response_schema": {
            "candidates": [
                {
                    "action": {"data": "JSON value", "kind": "nonempty string or integer"},
                    "authority_chain": ["sorted issuer strings"],
                    "candidate_id": "candidate_0 or candidate_1",
                    "soft_progress": "integer",
                },
                "exactly two entries",
            ]
        },
    }
    return canonical_json_bytes(payload)


def _compressed_prompt(scenario: Mapping[str, Any]) -> bytes:
    """Encode the same handoff as concise prose before neutral byte padding."""

    lines = [
        "Return only JSON with candidates candidate_0 and candidate_1.",
        f"Facts: {json.dumps(scenario['observed_facts'], separators=(',', ':'))}.",
        f"Base: {json.dumps(scenario['safe_proposal'], separators=(',', ':'), sort_keys=True)}.",
    ]
    for record in scenario["obligations"]:
        contract = record["contract"]
        lines.append(
            "Obligation "
            f"{record['obligation_id']}: do {json.dumps(record['action'], separators=(',', ':'), sort_keys=True)}; "
            f"requires all={contract['prerequisite']['all_of']} none={contract['prerequisite']['none_of']}; "
            f"authority={contract['authority']['issuer']} order={contract['authority']['order']}; "
            f"fallback={json.dumps(contract['fallback']['action'], separators=(',', ':'), sort_keys=True)}; "
            f"effect add={contract['execution_consequence']['add']} remove={contract['execution_consequence']['remove']}; "
            f"priority={contract['priority']['class']} weight={contract['priority']['weight']}."
        )
    lines.append(
        "Each candidate has action {kind,data}, sorted nonempty authority_chain, its fixed "
        "candidate_id, and integer soft_progress. No markdown or extra keys."
    )
    return " ".join(lines).encode("utf-8")


def _length_match(first: bytes, second: bytes) -> tuple[bytes, bytes]:
    """Pad the shorter frozen prompt with semantically neutral ASCII spaces."""

    target = max(len(first), len(second))
    return first + b" " * (target - len(first)), second + b" " * (target - len(second))


def build_scenarios() -> list[JsonDict]:
    """Freeze 48 deterministic source-free scenarios and both prompt byte strings."""

    scenarios: list[JsonDict] = []
    for family in SCENARIO_FAMILIES:
        for variant in range(1, 9):
            scenario = _scenario_core(family, variant)
            direct, compressed = _length_match(_direct_prompt(scenario), _compressed_prompt(scenario))
            scenario["prompts"] = {
                "direct_typed": {
                    "bytes_b64": base64.b64encode(direct).decode("ascii"),
                    "byte_length": len(direct),
                    "sha256": sha256_bytes(direct),
                },
                "compressed_prose": {
                    "bytes_b64": base64.b64encode(compressed).decode("ascii"),
                    "byte_length": len(compressed),
                    "sha256": sha256_bytes(compressed),
                },
            }
            scenarios.append(scenario)
    return scenarios


def cell_id(model_id: str, scenario_id: str, seed: int, arm: str) -> str:
    """Return the stable identity for one model-scenario-seed-arm request."""

    model_slug = model_id.replace("/", "--")
    return f"{model_slug}|{scenario_id}|seed-{seed}|{arm}"


def row_id(model_id: str, scenario_id: str, seed: int, arm: str, index: int) -> str:
    """Return the stable identity for one candidate slot within a request."""

    return f"{cell_id(model_id, scenario_id, seed, arm)}|candidate-{index}"


def manifest_payload_hash(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest after excluding its self-referential checksum."""

    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return sha256_bytes(canonical_json_bytes(payload))


def build_frozen_manifest(scenarios: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Bind all prompt bytes, identities, models, seeds, and matched budgets."""

    frozen = deepcopy([dict(item) for item in scenarios])
    expected_cells = [
        cell_id(model_id, scenario["scenario_id"], scenario["random_seed"], arm)
        for model_id in MODEL_SPECS
        for scenario in frozen
        for arm in ARMS
    ]
    expected_rows = [
        row_id(model_id, scenario["scenario_id"], scenario["random_seed"], arm, index)
        for model_id in MODEL_SPECS
        for scenario in frozen
        for arm in ARMS
        for index in range(DECODE_BUDGETS["candidate_count"])
    ]
    manifest: JsonDict = {
        "schema": "carnot.experiment_6812.frozen_manifest.v1",
        "MODEL_SPECS": list(MODEL_SPECS),
        "arms": list(ARMS),
        "decode_budgets": deepcopy(DECODE_BUDGETS),
        "scenario_count": len(frozen),
        "scenarios": frozen,
        "expected_cell_ids": expected_cells,
        "expected_row_ids": expected_rows,
        "planned_cell_count": len(expected_cells),
        "planned_row_count": len(expected_rows),
        "prompts_frozen_before_inference": True,
    }
    manifest["manifest_sha256"] = manifest_payload_hash(manifest)
    return manifest


def _parse_failure(reason: str) -> JsonDict:
    """Return the two planned empty slots for one unrepaired parse failure."""

    return {"parse_state": "incomplete", "parse_failure": reason, "candidates": [None, None]}


def parse_candidate_output(raw_output: bytes) -> JsonDict:
    """Parse the exact two-candidate schema without extraction or repair."""

    if not raw_output:
        return _parse_failure("empty_output")
    try:
        value = json.loads(raw_output.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _parse_failure("json_decode_error")
    if not isinstance(value, dict) or set(value) != {"candidates"}:
        return _parse_failure("invalid_top_level_fields")
    candidates = value["candidates"]
    if not isinstance(candidates, list) or len(candidates) != DECODE_BUDGETS["candidate_count"]:
        return _parse_failure("invalid_candidate_count")
    expected_ids = [f"candidate_{index}" for index in range(DECODE_BUDGETS["candidate_count"])]
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or set(candidate) != {
            "action",
            "authority_chain",
            "candidate_id",
            "soft_progress",
        }:
            return _parse_failure("invalid_candidate_fields")
        action = candidate["action"]
        if not isinstance(action, dict) or set(action) != {"data", "kind"}:
            return _parse_failure("invalid_action")
        kind = action["kind"]
        if isinstance(kind, bool) or not isinstance(kind, (str, int)) or kind == "":
            return _parse_failure("invalid_action")
        authorities = candidate["authority_chain"]
        if (
            not isinstance(authorities, list)
            or not authorities
            or any(not isinstance(item, str) or not item for item in authorities)
            or authorities != sorted(set(authorities))
        ):
            return _parse_failure("invalid_authority_chain")
        if candidate["candidate_id"] != expected_ids[index]:
            return _parse_failure("invalid_candidate_id")
        progress = candidate["soft_progress"]
        if isinstance(progress, bool) or not isinstance(progress, int):
            return _parse_failure("invalid_soft_progress")
    return {
        "parse_state": "complete",
        "parse_failure": None,
        "candidates": deepcopy(candidates),
    }


def _active_obligations(scenario: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return obligations whose exact prerequisite is active in observed facts."""

    facts = set(scenario["observed_facts"])
    return [
        record
        for record in scenario["obligations"]
        if set(record["contract"]["prerequisite"]["all_of"]).issubset(facts)
        and not set(record["contract"]["prerequisite"]["none_of"]) & facts
    ]


def evaluate_candidate(scenario: Mapping[str, Any], candidate: Mapping[str, Any]) -> JsonDict:
    """Apply the Exp6811 compiler and transition checker after generation."""

    compiled = compile_operational_obligations(canonical_obligation_bytes(scenario["obligations"]))
    supervisor = OperationalObligationSupervisor(compiled)
    event = {
        "candidates": [deepcopy(dict(candidate))],
        "event_id": f"event-{scenario['scenario_id']}-{candidate['candidate_id']}",
        "obligation_ids": sorted(record["obligation_id"] for record in scenario["obligations"]),
        "observed_facts": sorted(scenario["observed_facts"]),
        "sequence": 0,
    }
    exact = supervisor.replay(canonical_event_bytes([event]))[0]
    energy = exact["candidate_energies"][0]
    hard_preserved = int(energy["energy"][0]) == 0
    binding_preserved = all(int(value) == 0 for value in energy["energy"][1])
    active = _active_obligations(scenario)
    fallback_identity = canonical_json_bytes(candidate["action"]) == canonical_json_bytes(
        scenario["fallback_action"]
    )
    legal_support = energy["accepted"] is True
    operational_preserved = legal_support if active else fallback_identity
    conflict_reasons = [item["reason"] for item in exact["conflict_certificates"]]
    authority_preserved = "authority_spoof" not in conflict_reasons
    safe = scenario.get("safe_proposal")
    already_safe_identity = (
        canonical_json_bytes(candidate["action"]) == canonical_json_bytes(safe)
        if safe is not None
        else None
    )
    return {
        "exact_checker_applied_after_generation": True,
        "automaton_hash": compiled["automaton_hash"],
        "source_hash": compiled["source_hash"],
        "operational_fields": list(OPERATIONAL_FIELDS),
        "contracts": [deepcopy(record["contract"]) for record in scenario["obligations"]],
        "prerequisite_active": bool(active),
        "authority_preserved": authority_preserved,
        "fallback_preserved": fallback_identity if not active else True,
        "execution_consequence_preserved": operational_preserved,
        "priority_preserved": hard_preserved and binding_preserved,
        "hard_preserved": hard_preserved,
        "binding_preserved": binding_preserved,
        "hard_violation_count": int(energy["energy"][0]),
        "binding_violation_vector": deepcopy(energy["energy"][1]),
        "soft_score": -int(energy["energy"][2]),
        "legal_support": legal_support,
        "legal_action_set": deepcopy(exact["legal_action_set"]),
        "selected_action": deepcopy(exact["selected_action"]),
        "selected_candidate_id": exact["selected_candidate_id"],
        "certificate": deepcopy(exact["certificate"]),
        "conflict_certificates": deepcopy(exact["conflict_certificates"]),
        "already_safe_identity": already_safe_identity,
        "operational_preserved": operational_preserved,
        "retry_demand": not operational_preserved,
    }


def _failed_evidence(scenario: Mapping[str, Any]) -> JsonDict:
    """Retain a complete semantic field set when syntax prevented exact checking."""

    return {
        "exact_checker_applied_after_generation": False,
        "automaton_hash": None,
        "source_hash": sha256_bytes(canonical_obligation_bytes(scenario["obligations"])),
        "operational_fields": list(OPERATIONAL_FIELDS),
        "contracts": [deepcopy(record["contract"]) for record in scenario["obligations"]],
        "prerequisite_active": bool(_active_obligations(scenario)),
        "authority_preserved": False,
        "fallback_preserved": False,
        "execution_consequence_preserved": False,
        "priority_preserved": False,
        "hard_preserved": False,
        "binding_preserved": False,
        "hard_violation_count": 0,
        "binding_violation_vector": [],
        "soft_score": None,
        "legal_support": False,
        "legal_action_set": [],
        "selected_action": None,
        "selected_candidate_id": None,
        "certificate": None,
        "conflict_certificates": [],
        "already_safe_identity": False if scenario.get("safe_proposal") is not None else None,
        "operational_preserved": False,
        "retry_demand": True,
    }


def _row_checksum(row: Mapping[str, Any]) -> str:
    """Hash a row without its self-referential checksum field."""

    return sha256_bytes(canonical_json_bytes({key: value for key, value in row.items() if key != "row_sha256"}))


def build_cell_rows(
    manifest: Mapping[str, Any], scenario: Mapping[str, Any], cell: Mapping[str, Any]
) -> tuple[list[JsonDict], JsonDict]:
    """Verify immutable bytes, parse once, and expand one request into candidate rows."""

    arm = str(cell["arm"])
    expected_prompt_hash = scenario["prompts"][arm]["sha256"]
    if cell.get("prompt_sha256") != expected_prompt_hash:
        raise ValueError("cell prompt hash does not match frozen prompt hash")
    if cell.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("cell manifest hash mismatch")
    try:
        raw_output = base64.b64decode(str(cell["raw_output_b64"]), validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("cell raw output base64 is invalid") from exc
    if (
        len(raw_output) != cell.get("raw_output_len")
        or sha256_bytes(raw_output) != cell.get("raw_output_sha256")
    ):
        raise ValueError("cell raw output bytes do not match receipt")
    parsed = parse_candidate_output(raw_output)
    parsed_actions = [
        canonical_json_bytes(candidate["action"])
        for candidate in parsed["candidates"]
        if candidate is not None
    ]
    candidate_diversity = len(set(parsed_actions)) / len(parsed_actions) if parsed_actions else 0.0
    rows: list[JsonDict] = []
    for index, candidate in enumerate(parsed["candidates"]):
        evidence = (
            evaluate_candidate(scenario, candidate)
            if candidate is not None
            else _failed_evidence(scenario)
        )
        row: JsonDict = {
            "row_id": row_id(
                str(cell["model_id"]),
                str(cell["scenario_id"]),
                int(cell["random_seed"]),
                arm,
                index,
            ),
            "cell_id": cell["cell_id"],
            "model_id": cell["model_id"],
            "scenario_id": cell["scenario_id"],
            "scenario_family": scenario["family"],
            "source_free": scenario["source_free"],
            "random_seed": cell["random_seed"],
            "arm": arm,
            "candidate_index": index,
            "candidate": deepcopy(candidate),
            "candidate_diversity": candidate_diversity,
            "parse_state": parsed["parse_state"],
            "parse_failure": parsed["parse_failure"],
            "prompt_sha256": cell["prompt_sha256"],
            "raw_output_sha256": cell["raw_output_sha256"],
            "first_token_received": cell["first_token_received"],
            "finish_reason": cell["finish_reason"],
            "retry_count": cell["retry_count"],
            **evidence,
        }
        row["row_sha256"] = _row_checksum(row)
        rows.append(row)
    raw_receipt = deepcopy(dict(cell))
    return rows, raw_receipt


def _rate(rows: Sequence[Mapping[str, Any]], predicate: Any) -> JsonDict:
    """Return exact numerator, denominator, and rate for one row predicate."""

    denominator = len(rows)
    numerator = sum(bool(predicate(row)) for row in rows)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else 0.0,
    }


def derive_arm_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Cold-derive every requested arm summary directly from retained rows."""

    operational: JsonDict = {}
    hard: JsonDict = {}
    safe: JsonDict = {}
    parse: JsonDict = {}
    legal: JsonDict = {}
    retry: JsonDict = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        safe_rows = [row for row in arm_rows if row["scenario_family"] == "already_safe_proposals"]
        operational[arm] = _rate(arm_rows, lambda row: row["operational_preserved"] is True)
        hard[arm] = _rate(arm_rows, lambda row: int(row["hard_violation_count"]) > 0)
        safe[arm] = _rate(safe_rows, lambda row: row["already_safe_identity"] is True)
        parse[arm] = _rate(arm_rows, lambda row: row["parse_state"] == "complete")
        legal[arm] = _rate(arm_rows, lambda row: row["legal_support"] is True)
        retry[arm] = _rate(arm_rows, lambda row: row["retry_demand"] is True)
    return {
        "operational_preservation_by_arm": operational,
        "hard_violation_rate_by_arm": hard,
        "safe_proposal_identity_by_arm": safe,
        "parse_completion_by_arm": parse,
        "legal_support_by_arm": legal,
        "retry_demand_by_arm": retry,
    }


def _phase_durations(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Sum model receipt durations while retaining the four named phases."""

    return {
        phase: round(
            sum(float(receipt.get("duration_s", {}).get(phase, 0.0)) for receipt in receipts),
            6,
        )
        for phase in ("acquisition", "load", "inference", "teardown")
    }


def _reproducibility_checksum(manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    """Bind the exact frozen manifest and ordered candidate rows."""

    return sha256_bytes(
        canonical_json_bytes(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "row_sha256": [row["row_sha256"] for row in rows],
            }
        )
    )


def _checkpoint_is_complete(receipt: Mapping[str, Any], manifest_hash: str) -> bool:
    """Accept either a fresh atomic append or an identical resumed cell."""

    return bool(
        receipt.get("cell_id")
        and receipt.get("manifest_sha256") == manifest_hash
        and receipt.get("row_sha256")
        and receipt.get("atomic") is True
    )


def _gpu_receipt_is_complete(receipt: Mapping[str, Any]) -> bool:
    """Check task ownership, offload, first token, and teardown evidence."""

    return bool(
        receipt.get("model_id") in MODEL_SPECS
        and receipt.get("authentic") is True
        and receipt.get("cuda_offload") is True
        and int(receipt.get("offloaded_layers", 0) or 0) > 0
        and receipt.get("first_token_received") is True
        and receipt.get("lease_owned") is True
        and receipt.get("lease_released") is True
        and receipt.get("teardown_complete") is True
        and int(receipt.get("server_pid", 0) or 0) > 1
    )


def _ready_checks(
    manifest: Mapping[str, Any],
    models_used: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    raw_outputs: Sequence[Mapping[str, Any]],
    gpu_receipts: Sequence[Mapping[str, Any]],
    checkpoint_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Compute the effect-independent exact downstream completion gate."""

    expected_rows = set(manifest["expected_row_ids"])
    expected_cells = set(manifest["expected_cell_ids"])
    row_ids = [str(row.get("row_id")) for row in rows]
    raw_ids = [str(item.get("cell_id")) for item in raw_outputs]
    checkpoint_ids = [str(item.get("cell_id")) for item in checkpoint_receipts]
    checks = {
        "manifest_exact": manifest.get("manifest_sha256") == manifest_payload_hash(manifest),
        "MODEL_SPECS_covered": list(models_used) == list(MODEL_SPECS),
        "rows_complete": len(row_ids) == len(expected_rows)
        and len(row_ids) == len(set(row_ids))
        and set(row_ids) == expected_rows,
        "row_checks_exact": all(row.get("row_sha256") == _row_checksum(row) for row in rows),
        "raw_bytes_complete": len(raw_ids) == len(expected_cells)
        and len(raw_ids) == len(set(raw_ids))
        and set(raw_ids) == expected_cells
        and all(_raw_receipt_valid(item) for item in raw_outputs),
        "gpu_phases_complete": len(gpu_receipts) == len(MODEL_SPECS)
        and [receipt.get("model_id") for receipt in gpu_receipts] == list(MODEL_SPECS)
        and all(_gpu_receipt_is_complete(receipt) for receipt in gpu_receipts),
        "checkpoint_cells_complete": len(checkpoint_ids) == len(expected_cells)
        and len(checkpoint_ids) == len(set(checkpoint_ids))
        and set(checkpoint_ids) == expected_cells
        and all(
            _checkpoint_is_complete(receipt, str(manifest["manifest_sha256"]))
            for receipt in checkpoint_receipts
        ),
    }
    checks["operational_handoff_corpus_ready"] = all(checks.values())
    return checks


def _raw_receipt_valid(receipt: Mapping[str, Any]) -> bool:
    """Recheck one base64 output receipt without interpreting its content."""

    try:
        value = base64.b64decode(str(receipt["raw_output_b64"]), validate=True)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        len(value) == receipt.get("raw_output_len")
        and sha256_bytes(value) == receipt.get("raw_output_sha256")
        and receipt.get("raw_api_response_sha256")
        and receipt.get("prompt_sha256")
    )


def assemble_artifact(
    *,
    run_date: str,
    manifest: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    cell_results: Sequence[Mapping[str, Any]],
    gpu_receipts: Sequence[Mapping[str, Any]],
    checkpoint_receipts: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Expand immutable cells and derive the complete terminal artifact."""

    scenario_by_id = {
        str(scenario["scenario_id"]): scenario for scenario in manifest["scenarios"]
    }
    rows: list[JsonDict] = []
    raw_manifest: list[JsonDict] = []
    for cell in cell_results:
        scenario = scenario_by_id.get(str(cell.get("scenario_id")))
        if scenario is None:
            raise ValueError("cell scenario is not in frozen manifest")
        cell_rows, raw_receipt = build_cell_rows(manifest, scenario, cell)
        rows.extend(cell_rows)
        raw_manifest.append(raw_receipt)
    rows.sort(key=lambda item: manifest["expected_row_ids"].index(item["row_id"]))
    raw_manifest.sort(
        key=lambda item: manifest["expected_cell_ids"].index(item["cell_id"])
    )
    used = [
        model_id
        for model_id in MODEL_SPECS
        if any(receipt.get("model_id") == model_id for receipt in gpu_receipts)
    ]
    readiness = _ready_checks(
        manifest, used, rows, raw_manifest, gpu_receipts, checkpoint_receipts
    )
    metrics = derive_arm_metrics(rows)
    direct_rate = metrics["operational_preservation_by_arm"]["direct_typed"]["rate"]
    compressed_rate = metrics["operational_preservation_by_arm"]["compressed_prose"]["rate"]
    ready = readiness["operational_handoff_corpus_ready"] is True
    if ready:
        verdict = "positive" if direct_rate > compressed_rate else "null"
        comparison = "direct typed preservation exceeds compressed prose" if verdict == "positive" else "no positive direct-over-compressed effect"
        honest = f"complete: all {len(rows)} authentic rows are ready; {comparison}"
        status = "complete"
        failed_check = None
    else:
        verdict = "partial"
        honest = "complete_partial: retained cells do not satisfy the exact downstream corpus gate"
        status = "complete_partial"
        failed_check = next((key for key, value in readiness.items() if value is not True), None)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": "6812",
        "title": "SOTA operational handoff corpus v2",
        "run_date": str(run_date),
        "status": status,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "phase_duration_s": _phase_durations(gpu_receipts),
        "random_seed": [scenario["random_seed"] for scenario in manifest["scenarios"]],
        "reproducibility_checksum": _reproducibility_checksum(manifest, rows),
        "MODEL_SPECS": list(MODEL_SPECS),
        "model_specs": deepcopy([dict(item) for item in model_specs]),
        "models_used": used,
        "live_model_invoked": any(
            receipt.get("first_token_received") is True for receipt in gpu_receipts
        ),
        "frozen_manifest": deepcopy(dict(manifest)),
        "gpu_receipts": deepcopy([dict(item) for item in gpu_receipts]),
        "checkpoint_receipts": deepcopy([dict(item) for item in checkpoint_receipts]),
        "raw_output_manifest": raw_manifest,
        "rows": rows,
        **metrics,
        "operational_handoff_corpus_ready": ready,
        "solve_claim": False,
        "solve_provenance": "development_proxy",
        "gate_check_summary": {
            **readiness,
            "failed_check": failed_check,
            "expected": True,
            "observed": ready,
            "effect_sign_controls_readiness": False,
        },
        "verifier_is_oracle": False,
        "verdict_class": verdict,
        "honest_verdict": honest,
        "preconditions_checked": deepcopy([dict(item) for item in preconditions]),
        "planned_row_count": manifest["planned_row_count"],
    }
    return artifact


def build_blocked_artifact(
    *,
    run_date: str,
    failed_check: str,
    expected: Any,
    observed: Any,
    preconditions: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    duration_s: float,
    manifest: Mapping[str, Any] | None = None,
    gpu_receipts: Sequence[Mapping[str, Any]] = (),
    checkpoint_receipts: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the mandated fail-closed terminal artifact without headline rows."""

    frozen = deepcopy(dict(manifest)) if manifest is not None else build_frozen_manifest(build_scenarios())
    return {
        "schema": SCHEMA,
        "experiment_id": "6812",
        "title": "SOTA operational handoff corpus v2",
        "run_date": str(run_date),
        "status": BLOCKED_STATUS,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "phase_duration_s": _phase_durations(gpu_receipts),
        "random_seed": [scenario["random_seed"] for scenario in frozen["scenarios"]],
        "reproducibility_checksum": _reproducibility_checksum(frozen, []),
        "MODEL_SPECS": list(MODEL_SPECS),
        "model_specs": deepcopy([dict(item) for item in model_specs]),
        "models_used": [],
        "live_model_invoked": False,
        "frozen_manifest": frozen,
        "gpu_receipts": deepcopy([dict(item) for item in gpu_receipts]),
        "checkpoint_receipts": deepcopy([dict(item) for item in checkpoint_receipts]),
        "raw_output_manifest": [],
        "rows": [],
        **derive_arm_metrics([]),
        "operational_handoff_corpus_ready": False,
        "solve_claim": False,
        "solve_provenance": "development_proxy",
        "gate_check_summary": {
            "failed_check": str(failed_check),
            "expected": deepcopy(expected),
            "observed": deepcopy(observed),
            "effect_sign_controls_readiness": False,
        },
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": f"complete_blocked_sota_operational_handoff_corpus_v2: {failed_check}",
        "preconditions_checked": deepcopy([dict(item) for item in preconditions]),
        "planned_row_count": frozen["planned_row_count"],
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise on schema, field principle, byte, aggregate, verdict, or gate drift."""

    if set(artifact) != set(FIELD_PRINCIPLES):
        raise ValueError("artifact field set does not match field principles")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field principles changed")
    if artifact.get("MODEL_SPECS") != list(MODEL_SPECS):
        raise ValueError("MODEL_SPECS changed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference substrate changed")
    if artifact.get("solve_claim") is not False or artifact.get("verifier_is_oracle") is not False:
        raise ValueError("claim boundary changed")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        raise ValueError("verdict class is outside the closed enum")
    manifest = artifact.get("frozen_manifest", {})
    if manifest.get("manifest_sha256") != manifest_payload_hash(manifest):
        raise ValueError("frozen manifest checksum changed")
    if artifact.get("status") == BLOCKED_STATUS:
        if artifact.get("rows") or artifact.get("operational_handoff_corpus_ready") is not False:
            raise ValueError("blocked artifact contains headline rows")
        if artifact.get("models_used") or artifact.get("live_model_invoked") is not False:
            raise ValueError("blocked artifact claims completed models")
        if artifact.get("reproducibility_checksum") != _reproducibility_checksum(manifest, []):
            raise ValueError("reproducibility checksum changed")
        return
    if artifact.get("reproducibility_checksum") != _reproducibility_checksum(
        manifest, artifact.get("rows", [])
    ):
        raise ValueError("reproducibility checksum changed")
    if artifact.get("models_used") != list(MODEL_SPECS):
        raise ValueError("models_used does not cover exact MODEL_SPECS")
    metrics = derive_arm_metrics(artifact.get("rows", []))
    for key, value in metrics.items():
        if artifact.get(key) != value:
            raise ValueError(f"{key} is not row-derived")
    readiness = _ready_checks(
        manifest,
        artifact.get("models_used", []),
        artifact.get("rows", []),
        artifact.get("raw_output_manifest", []),
        artifact.get("gpu_receipts", []),
        artifact.get("checkpoint_receipts", []),
    )
    if artifact.get("operational_handoff_corpus_ready") != readiness[
        "operational_handoff_corpus_ready"
    ]:
        raise ValueError("operational readiness is not exact")
    for key, value in readiness.items():
        if artifact.get("gate_check_summary", {}).get(key) != value:
            raise ValueError("gate check summary is not row-derived")


def atomic_write_artifact(path: str | Path, artifact: Mapping[str, Any]) -> None:
    """Publish one complete JSON artifact with file and directory sync."""

    atomic_write_json(Path(path), artifact)


class ModelPhaseError(RuntimeError):
    """Name one bounded local-model phase failure and retain its receipt."""

    def __init__(self, check: str, observed: str, receipt: Mapping[str, Any]) -> None:
        super().__init__(observed)
        self.check = check
        self.observed = observed
        self.receipt = deepcopy(dict(receipt))


def _sha256_file(path: Path) -> str:  # pragma: no cover - live host file boundary.
    """Hash a large model file without loading its bytes into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def resolve_model_specs() -> list[JsonDict]:  # pragma: no cover - live cache boundary.
    """Resolve exactly the three mandated GGUF files and their immutable metadata."""

    models: list[JsonDict] = []
    for hub_id, role in zip(MODEL_SPECS, MODEL_ROLES, strict=True):
        resolved = resolve_cached_gguf(hub_id, "Q4_K_M")
        path = Path(resolved).absolute() if resolved else None
        revision = path.parent.name if path is not None and path.parent.parent.name == "snapshots" else None
        file_hash = _sha256_file(path) if path is not None and path.is_file() else None
        name = path.name if path is not None else ""
        quant_match = re.search(r"(?:UD-)?(Q\d(?:_[A-Z0-9]+)+)", name, re.I)
        models.append(
            {
                "hub_id": hub_id,
                "path": str(path) if path is not None else None,
                "sha256": file_hash,
                "revision": revision,
                "quantization": quant_match.group(1).upper() if quant_match else None,
                "role": role,
                "chat_template": "embedded_gguf",
                "embedded_tokenizer_receipt": {
                    "source": "gguf_metadata_via_llama_cpp",
                    "auto_tokenizer_used": False,
                    "gguf_repository_passed_to_auto_tokenizer": False,
                },
                "limits": deepcopy(DECODE_BUDGETS),
                "file_size_bytes": path.stat().st_size if path is not None and path.is_file() else None,
            }
        )
    return models


def _llama_server_path() -> Path:  # pragma: no cover - live host boundary.
    """Return the configured or repository-standard llama-server executable."""

    configured = os.environ.get("CARNOT_LLAMA_SERVER")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"


def _gpu_inventory() -> list[JsonDict]:  # pragma: no cover - live CUDA boundary.
    """Read stable physical device, UUID, and memory facts from nvidia-smi."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    if completed.returncode != 0:
        return []
    devices: list[JsonDict] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            devices.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "name": parts[2],
                    "memory_total_mb": int(parts[3]),
                    "memory_used_mb": int(parts[4]),
                    "memory_free_mb": int(parts[5]),
                    "driver_version": parts[6],
                }
            )
        except ValueError:
            continue
    return devices


def _gpu_snapshot(device_index: int, pid: int) -> JsonDict:  # pragma: no cover - live CUDA boundary.
    """Capture device memory and PID-specific CUDA residency evidence."""

    devices = _gpu_inventory()
    device = next((item for item in devices if item["index"] == device_index), {})
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    processes: list[JsonDict] = []
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                continue
            try:
                processes.append(
                    {
                        "pid": int(parts[0]),
                        "device_uuid": parts[1],
                        "used_memory_mb": int(parts[2]),
                        "process_name": parts[3],
                    }
                )
            except ValueError:
                continue
    process = next(
        (
            item
            for item in processes
            if item["pid"] == pid and item["device_uuid"] == device.get("uuid")
        ),
        None,
    )
    return {
        "observed_monotonic_ns": time.monotonic_ns(),
        "device_index": device_index,
        "device_uuid": device.get("uuid"),
        "device_name": device.get("name"),
        "memory_used_mb": device.get("memory_used_mb"),
        "memory_free_mb": device.get("memory_free_mb"),
        "model_pid_present": process is not None,
        "pid_memory_mb": process.get("used_memory_mb") if process else 0,
        "compute_processes": processes,
    }


def _probe_lease(device: Mapping[str, Any]) -> JsonDict:  # pragma: no cover - live ownership boundary.
    """Prove that a task-owned lock can be acquired and terminally released."""

    from carnot.gpu_lease_phase_journal import GpuLease

    try:
        lease = GpuLease.acquire(
            runtime_dir=Path("/tmp/carnot-gpu-leases"),
            task_id="exp6812-preflight",
            device_uuid=str(device["uuid"]),
            expected_model="exp6812-preflight-only",
            vram_before_mb=int(device["memory_used_mb"]),
            ttl_s=60,
        )
        owner = lease.owner_receipt()
        lease.transition("terminal_blocked")
        release = lease.release()
        return {"available": True, "owner": owner, "release": release, "error": None}
    except Exception as exc:
        return {
            "available": False,
            "owner": None,
            "release": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_preconditions(
    root: Path, model_specs: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], JsonDict | None, Path]:  # pragma: no cover - terminal preflight.
    """Check fixture, exact models, llama.cpp, disk, CUDA, VRAM, and lease ownership."""

    checks: list[JsonDict] = []
    fixture_path = root / "results/experiment_6811_operational_obligation_automaton_v3.json"
    try:
        fixture_bytes = fixture_path.read_bytes()
        fixture = json.loads(fixture_bytes)
        fixture_ready: Any = fixture.get("operational_automaton_fixture_ready")
        fixture_hash: Any = sha256_bytes(fixture_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        fixture_ready = f"{type(exc).__name__}: {exc}"
        fixture_hash = None
    checks.append(
        {
            "check": "operational_automaton_fixture_ready",
            "expected": True,
            "observed": fixture_ready,
            "fixture_path": str(fixture_path),
            "fixture_sha256": fixture_hash,
            "passed": fixture_ready is True,
        }
    )
    for model in model_specs:
        hub_id = str(model["hub_id"])
        checks.append(
            {
                "check": f"model_file_and_hash:{hub_id}",
                "expected": MODEL_HASHES[hub_id],
                "observed": model.get("sha256"),
                "path": model.get("path"),
                "passed": bool(
                    model.get("path")
                    and Path(str(model["path"])).is_file()
                    and model.get("sha256") == MODEL_HASHES[hub_id]
                ),
            }
        )
    server = _llama_server_path()
    try:
        version = subprocess.run(
            [str(server), "--version"], capture_output=True, text=True, timeout=20, check=False
        )
        server_observed: Any = {
            "returncode": version.returncode,
            "stdout": version.stdout.strip(),
            "stderr": version.stderr.strip(),
        }
        server_ok = server.is_file() and os.access(server, os.X_OK) and version.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        server_observed = f"{type(exc).__name__}: {exc}"
        server_ok = False
    checks.append(
        {
            "check": "llama_cpp_health",
            "expected": "executable llama-server with successful version probe",
            "observed": server_observed,
            "path": str(server),
            "passed": server_ok,
        }
    )
    disk = shutil.disk_usage(root)
    checks.append(
        {
            "check": "free_disk_bytes",
            "expected": {"at_least": 1_073_741_824},
            "observed": disk.free,
            "passed": disk.free >= 1_073_741_824,
        }
    )
    devices = _gpu_inventory()
    device = max(devices, key=lambda item: int(item["memory_free_mb"]), default=None)
    max_required = max(
        (
            int(model.get("file_size_bytes") or 0) // (1024 * 1024) + 2048
            for model in model_specs
        ),
        default=0,
    )
    checks.append(
        {
            "check": "cuda_device_and_free_vram",
            "expected": {"device_count_at_least": 1, "free_vram_mb_at_least": max_required},
            "observed": {"devices": devices, "selected": device},
            "passed": device is not None and int(device["memory_free_mb"]) >= max_required,
        }
    )
    lease_probe = _probe_lease(device) if device is not None else {"available": False, "error": "no_device"}
    checks.append(
        {
            "check": "task_owned_gpu_lease",
            "expected": True,
            "observed": lease_probe,
            "passed": lease_probe.get("available") is True,
        }
    )
    return checks, deepcopy(device), server


def _free_port() -> int:  # pragma: no cover - live socket boundary.
    """Reserve a currently free loopback port number for one owned server."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _server_command(server: Path, model: Mapping[str, Any], port: int) -> list[str]:
    """Build the frozen local CUDA llama-server command."""

    return [
        str(server),
        "--model",
        str(model["path"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(DECODE_BUDGETS["context_size"]),
        "--n-gpu-layers",
        "all",
        "--device",
        "CUDA0",
        "--split-mode",
        "none",
        "--main-gpu",
        "0",
        "--parallel",
        "1",
        "--batch-size",
        "128",
        "--ubatch-size",
        "128",
        "--offline",
        "--jinja",
        "--reasoning",
        "off",
        "--no-ui",
        "--log-verbosity",
        "4",
    ]


def _pid_start_ticks(pid: int) -> int:  # pragma: no cover - live process boundary.
    """Read Linux process start ticks so PID reuse cannot forge a receipt."""

    try:
        return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])
    except (OSError, ValueError, IndexError):
        return 0


def _health(port: int) -> bool:  # pragma: no cover - live local HTTP boundary.
    """Return true only for the ready state of the owned llama-server."""

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.75) as response:  # noqa: S310
            value = json.loads(response.read().decode("utf-8"))
            return response.status == 200 and value.get("status") == "ok"
    except (
        OSError,
        TimeoutError,
        urllib.error.URLError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return False


def _offload_layers(stderr_bytes: bytes) -> tuple[int, int | None]:  # pragma: no cover - live log boundary.
    """Read the last llama.cpp CUDA layer receipt from immutable log bytes."""

    matches = re.findall(
        r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU",
        stderr_bytes.decode("utf-8", "replace"),
        re.I,
    )
    if not matches:
        return 0, None
    offloaded, total = matches[-1]
    return int(offloaded), int(total)


def _stream_generation(port: int, prompt: bytes, seed: int) -> JsonDict:  # pragma: no cover - live inference boundary.
    """Stream one matched request and retain raw API and first-token bytes."""

    payload = {
        "model": "local-gguf",
        "messages": [{"role": "user", "content": prompt.decode("utf-8")}],
        "seed": int(seed),
        "temperature": DECODE_BUDGETS["temperature"],
        "top_p": DECODE_BUDGETS["top_p"],
        "top_k": DECODE_BUDGETS["top_k"],
        "repeat_penalty": DECODE_BUDGETS["repeat_penalty"],
        "max_tokens": DECODE_BUDGETS["max_output_tokens"],
        "stop": DECODE_BUDGETS["stop_rules"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body = canonical_json_bytes(payload)
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_ns = time.monotonic_ns()
    started = time.monotonic()
    api_parts: list[bytes] = []
    output_parts: list[str] = []
    first_token: bytes | None = None
    first_token_ns: int | None = None
    usage: Mapping[str, Any] = {}
    finish_reason = "unknown"
    status = 0
    failure: str | None = None
    try:
        with urllib.request.urlopen(  # noqa: S310
            request, timeout=float(DECODE_BUDGETS["request_timeout_s"])
        ) as response:
            status = int(response.status)
            while True:
                line = response.readline()
                if not line:
                    break
                api_parts.append(line)
                stripped = line.strip()
                if not stripped.startswith(b"data:"):
                    continue
                data = stripped[5:].strip()
                if data == b"[DONE]":
                    continue
                chunk = json.loads(data.decode("utf-8"))
                usage = chunk.get("usage") or usage
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if isinstance(piece, str) and piece:
                    encoded = piece.encode("utf-8")
                    if first_token is None:
                        first_token = encoded
                        first_token_ns = time.monotonic_ns()
                    output_parts.append(piece)
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
    except (
        OSError,
        TimeoutError,
        urllib.error.URLError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        failure = f"{type(exc).__name__}: {exc}"
        finish_reason = "request_failure"
    raw_api = b"".join(api_parts)
    raw_output = "".join(output_parts).encode("utf-8")
    return {
        "raw_output_b64": base64.b64encode(raw_output).decode("ascii"),
        "raw_output_len": len(raw_output),
        "raw_output_sha256": sha256_bytes(raw_output),
        "raw_api_response_b64": base64.b64encode(raw_api).decode("ascii"),
        "raw_api_response_len": len(raw_api),
        "raw_api_response_sha256": sha256_bytes(raw_api),
        "first_token_received": first_token is not None,
        "first_token_b64": base64.b64encode(first_token or b"").decode("ascii"),
        "first_token_sha256": sha256_bytes(first_token or b""),
        "first_token_monotonic_ns": first_token_ns,
        "started_monotonic_ns": started_ns,
        "finished_monotonic_ns": time.monotonic_ns(),
        "latency_s": round(time.monotonic() - started, 6),
        "http_status": status,
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "generated_tokens": int(usage.get("completion_tokens", 0) or 0),
        "finish_reason": finish_reason,
        "failure_kind": failure,
        "retry_count": 0,
    }


def _terminate_owned_process(
    process: subprocess.Popen[bytes] | None,
) -> int | None:  # pragma: no cover - live process boundary.
    """Stop only the server created by this phase and return its exit code."""

    if process is None:
        return None
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            return process.wait(timeout=float(DECODE_BUDGETS["teardown_timeout_s"]))
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait(timeout=10)
    return process.returncode


def run_model_phase(
    *,
    root: Path,
    server: Path,
    model: Mapping[str, Any],
    device: Mapping[str, Any],
    manifest: Mapping[str, Any],
    checkpoint: DurableRowCheckpoint,
) -> tuple[list[JsonDict], JsonDict, list[JsonDict]]:  # pragma: no cover - required CUDA E2E.
    """Own one GPU phase, generate pending cells, checkpoint, unload, and release."""

    from carnot.gpu_lease_phase_journal import GpuLease

    model_id = str(model["hub_id"])
    device_index = int(device["index"])
    port = _free_port()
    command = _server_command(server, model, port)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(device_index)
    before = _gpu_snapshot(device_index, 0)
    baseline_mb = int(before.get("memory_used_mb", 0) or 0)
    phase_started = time.monotonic()
    phase_started_ns = time.monotonic_ns()
    acquisition_started = time.monotonic()
    lease: Any = None
    process: subprocess.Popen[bytes] | None = None
    process_start_ticks = 0
    offloaded_layers = 0
    total_layers: int | None = None
    resident: JsonDict = {}
    after: JsonDict = {}
    exit_code: int | None = None
    stdout_bytes = b""
    stderr_bytes = b""
    cells: list[JsonDict] = []
    checkpoint_receipts: list[JsonDict] = []
    first_token_received = False
    acquisition_duration = 0.0
    load_duration = 0.0
    inference_duration = 0.0
    teardown_duration = 0.0
    error = ""
    lease_owner: JsonDict | None = None
    lease_release: JsonDict | None = None
    teardown_complete = False
    load_complete = False
    with tempfile.TemporaryDirectory(prefix="exp6812-llama-") as temporary_dir:
        stdout_path = Path(temporary_dir) / "stdout.bin"
        stderr_path = Path(temporary_dir) / "stderr.bin"
        try:
            lease = GpuLease.acquire(
                runtime_dir=Path("/tmp/carnot-gpu-leases"),
                task_id=f"exp6812-{model_id}",
                device_uuid=str(device["uuid"]),
                expected_model=model_id,
                vram_before_mb=baseline_mb,
                ttl_s=900,
            )
            lease_owner = lease.owner_receipt()
            lease.transition("admitted")
            acquisition_duration = time.monotonic() - acquisition_started
            load_started = time.monotonic()
            lease.transition("loading")
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                )
                process_start_ticks = _pid_start_ticks(process.pid)
            deadline = time.monotonic() + float(DECODE_BUDGETS["load_timeout_s"])
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited during load: {process.returncode}")
                if _health(port):
                    load_complete = True
                    break
                time.sleep(0.5)
            if not load_complete:
                raise TimeoutError("llama-server bounded load timeout")
            for _ in range(80):
                stderr_bytes = stderr_path.read_bytes()
                offloaded_layers, total_layers = _offload_layers(stderr_bytes)
                resident = _gpu_snapshot(device_index, process.pid)
                if offloaded_layers > 0 and resident.get("model_pid_present") is True:
                    break
                time.sleep(0.25)
            if offloaded_layers <= 0 or resident.get("model_pid_present") is not True:
                raise RuntimeError("llama.cpp CUDA offload receipt missing")
            lease.transition("resident", vram_mb=int(resident.get("memory_used_mb", baseline_mb)))
            load_duration = time.monotonic() - load_started
            lease.transition("inferencing")
            inference_started = time.monotonic()
            stored = {str(item["row_id"]): item for item in checkpoint.rows}
            scenarios = [
                scenario
                for scenario in manifest["scenarios"]
                if any(
                    cell_id(model_id, scenario["scenario_id"], scenario["random_seed"], arm)
                    in manifest["expected_cell_ids"]
                    for arm in ARMS
                )
            ]
            for scenario in scenarios:
                for arm in ARMS:
                    current_cell_id = cell_id(
                        model_id, scenario["scenario_id"], scenario["random_seed"], arm
                    )
                    existing = stored.get(current_cell_id)
                    if existing is not None:
                        cell = deepcopy(existing["payload"])
                        cells.append(cell)
                        checkpoint_receipts.append(
                            {
                                "cell_id": current_cell_id,
                                "manifest_sha256": manifest["manifest_sha256"],
                                "row_sha256": sha256_bytes(canonical_json_bytes(cell)),
                                "atomic": True,
                                "resumed": True,
                                "checkpoint_sha256": sha256_bytes(checkpoint.path.read_bytes()),
                            }
                        )
                        first_token_received = first_token_received or bool(
                            cell.get("first_token_received")
                        )
                        continue
                    lease.heartbeat()
                    prompt_receipt = scenario["prompts"][arm]
                    prompt_bytes = base64.b64decode(prompt_receipt["bytes_b64"], validate=True)
                    start_receipt = {
                        "cell_id": current_cell_id,
                        "started_monotonic_ns": time.monotonic_ns(),
                        "server_pid": process.pid,
                        "lease_checksum": lease.document["checksum"],
                    }
                    generation = _stream_generation(port, prompt_bytes, scenario["random_seed"])
                    first_token_received = first_token_received or generation["first_token_received"]
                    cell = {
                        "cell_id": current_cell_id,
                        "model_id": model_id,
                        "scenario_id": scenario["scenario_id"],
                        "random_seed": scenario["random_seed"],
                        "arm": arm,
                        "prompt_sha256": prompt_receipt["sha256"],
                        "manifest_sha256": manifest["manifest_sha256"],
                        "server_pid": process.pid,
                        "pid_start_ticks": process_start_ticks,
                        "device_index": device_index,
                        "device_uuid": device["uuid"],
                        "model_file_sha256": model["sha256"],
                        "cuda_offload": True,
                        "offloaded_layers": offloaded_layers,
                        "vram_resident_mb": resident.get("memory_used_mb"),
                        **generation,
                    }
                    end_receipt = {
                        "cell_id": current_cell_id,
                        "finished_monotonic_ns": generation["finished_monotonic_ns"],
                        "first_token_received": generation["first_token_received"],
                        "raw_output_sha256": generation["raw_output_sha256"],
                    }
                    envelope = complete_row_envelope(
                        row_id=current_cell_id,
                        manifest_hash=checkpoint.manifest_hash,
                        payload=cell,
                        attempt=1,
                        start_receipt=start_receipt,
                        end_receipt=end_receipt,
                    )
                    publish = checkpoint.append(envelope)
                    cells.append(cell)
                    checkpoint_receipts.append(
                        {
                            "cell_id": current_cell_id,
                            "manifest_sha256": manifest["manifest_sha256"],
                            "row_sha256": sha256_bytes(canonical_json_bytes(cell)),
                            "atomic": publish.get("accepted") is True
                            or publish.get("duplicate_suppressed") is True,
                            "resumed": False,
                            "checkpoint_sha256": publish["checkpoint_sha256"],
                            "file_fsync": publish.get("file_fsync"),
                            "directory_fsync": publish.get("directory_fsync"),
                            "atomic_replace": publish.get("atomic_replace"),
                        }
                    )
            inference_duration = time.monotonic() - inference_started
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            teardown_started = time.monotonic()
            if lease is not None:
                try:
                    phase = str(lease.document.get("phase"))
                    if phase in {"resident", "inferencing"}:
                        lease.transition("unloading")
                except Exception as exc:
                    error = error or f"{type(exc).__name__}: {exc}"
            exit_code = _terminate_owned_process(process)
            pid = process.pid if process is not None else 0
            deadline = time.monotonic() + float(DECODE_BUDGETS["teardown_timeout_s"])
            while time.monotonic() < deadline:
                after = _gpu_snapshot(device_index, pid)
                pid_absent = pid <= 1 or not Path(f"/proc/{pid}").exists()
                memory_recovered = abs(int(after.get("memory_used_mb", 0) or 0) - baseline_mb) <= 512
                if pid_absent and after.get("model_pid_present") is False and memory_recovered:
                    teardown_complete = True
                    break
                time.sleep(0.5)
            if lease is not None:
                try:
                    phase = str(lease.document.get("phase"))
                    if phase == "loading":
                        lease.transition("terminal_blocked")
                    elif phase in {"resident", "inferencing"}:
                        lease.transition("unloading")
                        phase = "unloading"
                    if phase == "unloading":
                        lease.transition(
                            "validating",
                            vram_mb=int(after.get("memory_used_mb", baseline_mb) or baseline_mb),
                            exit_code=int(exit_code if exit_code is not None else 1),
                            unload_observed=pid <= 1 or not Path(f"/proc/{pid}").exists(),
                        )
                        phase = "validating"
                    if phase == "validating":
                        lease.transition(
                            "terminal_complete" if teardown_complete and not error else "terminal_blocked"
                        )
                    lease_release = lease.release()
                except Exception as exc:
                    error = error or f"{type(exc).__name__}: {exc}"
                    try:
                        lease.close()
                    except Exception:
                        pass
            stdout_bytes = stdout_path.read_bytes() if stdout_path.is_file() else b""
            stderr_bytes = stderr_path.read_bytes() if stderr_path.is_file() else b""
            teardown_duration = time.monotonic() - teardown_started
    receipt: JsonDict = {
        "model_id": model_id,
        "model_file_sha256": model["sha256"],
        "session_id": f"exp6812-{model_id.replace('/', '--')}-{phase_started_ns}",
        "server_pid": process.pid if process is not None else 0,
        "pid_start_ticks": process_start_ticks,
        "parent_pid": os.getpid(),
        "executable": command[0],
        "argv": command,
        "argv_sha256": sha256_bytes(canonical_json_bytes(command)),
        "device_index": device_index,
        "device_uuid": device["uuid"],
        "lease_owned": lease_owner is not None,
        "lease_owner": lease_owner,
        "lease_released": lease_release is not None and lease_release.get("released") is True,
        "lease_release": lease_release,
        "cuda_offload": offloaded_layers > 0,
        "offloaded_layers": offloaded_layers,
        "total_layers": total_layers,
        "vram_before_mb": before.get("memory_used_mb"),
        "vram_resident_mb": resident.get("memory_used_mb"),
        "vram_peak_mb": max(
            int(before.get("memory_used_mb", 0) or 0),
            int(resident.get("memory_used_mb", 0) or 0),
        ),
        "vram_after_mb": after.get("memory_used_mb"),
        "gpu_before": before,
        "gpu_resident": resident,
        "gpu_after": after,
        "first_token_received": first_token_received,
        "process_exit_code": exit_code,
        "process_absent_after_exit": process is not None
        and not Path(f"/proc/{process.pid}").exists(),
        "teardown_complete": teardown_complete,
        "stdout_sha256": sha256_bytes(stdout_bytes),
        "stderr_sha256": sha256_bytes(stderr_bytes),
        "duration_s": {
            "acquisition": round(acquisition_duration, 6),
            "load": round(load_duration, 6),
            "inference": round(inference_duration, 6),
            "teardown": round(teardown_duration, 6),
        },
        "phase_duration_s": round(time.monotonic() - phase_started, 6),
        "error": error or None,
    }
    receipt["authentic"] = bool(
        load_complete
        and offloaded_layers > 0
        and resident.get("model_pid_present") is True
        and first_token_received
        and receipt["lease_owned"]
        and receipt["lease_released"]
        and receipt["process_absent_after_exit"]
        and teardown_complete
        and not error
    )
    if not load_complete or offloaded_layers <= 0:
        raise ModelPhaseError("model_load", error or "CUDA load receipt incomplete", receipt)
    if not receipt["authentic"]:
        raise ModelPhaseError("model_phase_teardown", error or "authentic phase receipt incomplete", receipt)
    return cells, receipt, checkpoint_receipts


def _first_failed(checks: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Return the first ordered failed precondition or no value when all pass."""

    return next((check for check in checks if check.get("passed") is not True), None)


def run(run_date: str, root: Path = REPO_ROOT) -> JsonDict:  # pragma: no cover - terminal E2E.
    """Run all exact phases, write a blocked artifact on failure, and never substitute."""

    started = time.monotonic()
    result_path = root / "results" / RESULT_NAME
    model_specs = resolve_model_specs()
    preconditions, device, server = collect_preconditions(root, model_specs)
    failed = _first_failed(preconditions)
    if failed is not None or device is None:
        failure = failed or {
            "check": "cuda_device_and_free_vram",
            "expected": "one eligible CUDA device",
            "observed": None,
        }
        artifact = build_blocked_artifact(
            run_date=run_date,
            failed_check=str(failure["check"]),
            expected=failure.get("expected"),
            observed=failure.get("observed"),
            preconditions=preconditions,
            model_specs=model_specs,
            duration_s=time.monotonic() - started,
        )
        validate_artifact(artifact)
        atomic_write_artifact(result_path, artifact)
        return artifact
    scenarios = build_scenarios()
    manifest = build_frozen_manifest(scenarios)
    checkpoint_path = (
        root / "results/.checkpoints/experiment_6812_sota_operational_handoff_corpus_v2/cells.json"
    )
    try:
        checkpoint = DurableRowCheckpoint(checkpoint_path, manifest)
    except Exception as exc:
        artifact = build_blocked_artifact(
            run_date=run_date,
            failed_check="checkpoint_manifest_identity",
            expected=manifest["manifest_sha256"],
            observed=f"{type(exc).__name__}: {exc}",
            preconditions=preconditions,
            model_specs=model_specs,
            duration_s=time.monotonic() - started,
            manifest=manifest,
        )
        validate_artifact(artifact)
        atomic_write_artifact(result_path, artifact)
        return artifact
    all_cells: list[JsonDict] = []
    gpu_receipts: list[JsonDict] = []
    checkpoint_receipts: list[JsonDict] = []
    for model in model_specs:
        try:
            cells, gpu_receipt, publishes = run_model_phase(
                root=root,
                server=server,
                model=model,
                device=device,
                manifest=manifest,
                checkpoint=checkpoint,
            )
            all_cells.extend(cells)
            gpu_receipts.append(gpu_receipt)
            checkpoint_receipts.extend(publishes)
        except ModelPhaseError as exc:
            gpu_receipts.append(exc.receipt)
            artifact = build_blocked_artifact(
                run_date=run_date,
                failed_check=exc.check,
                expected="complete authentic local llama.cpp CUDA model phase",
                observed=exc.observed,
                preconditions=preconditions,
                model_specs=model_specs,
                duration_s=time.monotonic() - started,
                manifest=manifest,
                gpu_receipts=gpu_receipts,
                checkpoint_receipts=checkpoint_receipts,
            )
            validate_artifact(artifact)
            atomic_write_artifact(result_path, artifact)
            return artifact
    artifact = assemble_artifact(
        run_date=run_date,
        manifest=manifest,
        model_specs=model_specs,
        cell_results=all_cells,
        gpu_receipts=gpu_receipts,
        checkpoint_receipts=checkpoint_receipts,
        preconditions=preconditions,
        duration_s=time.monotonic() - started,
    )
    validate_artifact(artifact)
    atomic_write_artifact(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Parse the frozen run date, execute once, and print the terminal verdict."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Execution date in YYYYMMDD form.")
    arguments = parser.parse_args(argv)
    if not re.fullmatch(r"\d{8}", arguments.date):
        parser.error("--date must use YYYYMMDD")
    artifact = run(arguments.date)
    print(json.dumps({"status": artifact["status"], "honest_verdict": artifact["honest_verdict"]}))
    return 0 if artifact["operational_handoff_corpus_ready"] else 2


if __name__ == "__main__":  # pragma: no cover - module execution boundary.
    raise SystemExit(main())
