"""Audit the V621 entrance proposal bank as a support set.

Spec refs: REQ-VERIFY-7087 and SCENARIO-VERIFY-7087-*.

The audit reads saved model output in a new process. It does not load a model.
The exact solver supplies audit labels only after the required support schema is
frozen from source groups, model identities, seeds, and operator families.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_7064_v619_exact_entrance_fixture as exact
from carnot import experiment_7086_v621_three_family_entrance_bank as producer


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260906"
RANDOM_SEED = 7_087_202_609_06
EXPERIMENT_ID = "experiment_7087_v621_entrance_bank_sufficiency_audit"
SCHEMA = "carnot.experiment_7087.v621_entrance_bank_sufficiency_audit.v1"
INFERENCE_SUBSTRATE = "fresh-process deterministic entrance-bank replay"
RESULT_PATH = REPO_ROOT / "results/experiment_7087_v621_entrance_bank_sufficiency_audit.json"
BANK_PATH = REPO_ROOT / "results/experiment_7086_v621_three_family_entrance_bank.json"
FIXTURE_PATH = REPO_ROOT / "results/experiment_7064_v619_exact_entrance_fixture.json"
AUDIT_ROOT = REPO_ROOT / "results/.experiment_7087_v621_entrance_bank_sufficiency_audit"
PINNED_BANK_SHA256 = "sha256:4f9e73adfc2ce0ede707424f6f644705557faa27d4485fb2293f652ab1713495"
PINNED_FIXTURE_SHA256 = "sha256:6b62768e3387d40eebf462c199aab6a440321aa4a1549ff7d54faaba312f2277"
ENTRANCE_FAMILIES = ("+", "-", "*", "/")
MINIMUM_HEADROOM_UNITS = 30

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "inference_substrate_class",
    "duration_s",
    "source_artifact_hashes",
    "cited_upstream_artifacts",
    "upstream_gate_rows",
    "rows",
    "per_game_results",
    "fresh_process_rows",
    "raw_hash_rows",
    "parse_recomputation_rows",
    "label_recomputation_rows",
    "causal_witness_replay_rows",
    "prompt_parity_rows",
    "chat_template_parity_rows",
    "stop_parity_rows",
    "budget_parity_rows",
    "model_identity_rows",
    "seed_coverage_rows",
    "source_group_coverage_rows",
    "runner_receipt_rows",
    "gpu_telemetry_rows",
    "lease_cleanup_rows",
    "required_support_schema",
    "family_support_rows",
    "missing_family_rows",
    "conflict_rows",
    "counterfactual_swap_rows",
    "leakage_attack_rows",
    "headroom_rows",
    "headroom_unit_count",
    "entrance_support_audit_ready_score",
    "entrance_selector_headroom_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "A reason for each field makes evidence omissions visible.",
    "preconditions_checked": "Exact preflight results prevent an invalid upstream run from receiving audit credit.",
    "inference_substrate": "The substrate states that saved bytes, not a new model call, supply proposals.",
    "inference_substrate_class": "The class separates a completed cold replay from a blocked preflight.",
    "duration_s": "Measured wall time exposes nonexecution and gives the replay a cost receipt.",
    "source_artifact_hashes": "Content hashes bind the audit to exact evidence and code revisions.",
    "cited_upstream_artifacts": "Citations identify which upstream claims and fields the audit reused.",
    "upstream_gate_rows": "Bare upstream gates prevent summaries from replacing source checks.",
    "rows": "Independent gate rows make both readiness scores recomputable.",
    "per_game_results": "Unit rows expose local support failures that pooled rates can hide.",
    "fresh_process_rows": "Process receipts prove that replay did not reuse producer memory.",
    "raw_hash_rows": "Byte hashes detect output mutation before parsing or labeling.",
    "parse_recomputation_rows": "Independent parsing detects stale or substituted producer parse rows.",
    "label_recomputation_rows": "Independent exact labels prevent producer summaries from granting correctness.",
    "causal_witness_replay_rows": "Witness replay distinguishes certified reachability from answer-only guesses.",
    "prompt_parity_rows": "Prompt and role-message parity detects changed model inputs.",
    "chat_template_parity_rows": "Template parity detects family-specific transport substitution.",
    "stop_parity_rows": "Stop parity detects hidden termination differences.",
    "budget_parity_rows": "Budget parity keeps proposal opportunity equal across families and seeds.",
    "model_identity_rows": "Identity rows bind each proposal family to one declared checkpoint receipt.",
    "seed_coverage_rows": "Seed rows expose missing stochastic replicates.",
    "source_group_coverage_rows": "Source rows expose a missing independent generator family.",
    "runner_receipt_rows": "Runner rows prove the saved proposals used the declared chat and CUDA path.",
    "gpu_telemetry_rows": "Telemetry rows prove that every producer shard had GPU-resident execution.",
    "lease_cleanup_rows": "Lease, checkpoint, release, and cleanup rows exclude shard contamination.",
    "required_support_schema": "A predeclared cross-product prevents outcome-driven support definitions.",
    "family_support_rows": "Cell states expose missing, duplicate, conflicting, and inapplicable families.",
    "missing_family_rows": "A direct failure projection prevents pooled rates from hiding absent support.",
    "conflict_rows": "Explicit rejected alternatives make label conflicts auditable.",
    "counterfactual_swap_rows": "Attacks prove that preserved pooled counts cannot preserve readiness.",
    "leakage_attack_rows": "Input scans keep exact outcomes and future labels outside model context.",
    "headroom_rows": "Held-unit mixtures measure selector opportunity without fitting a selector.",
    "headroom_unit_count": "A minimum count prevents a small mixed sample from supporting selector work.",
    "entrance_support_audit_ready_score": "One requires authentic, sufficient, conflict-free, recomputed evidence.",
    "entrance_selector_headroom_ready_score": "One requires 30 mixed held units and no perfect non-oracle control.",
    "random_seed": "A fixed controller seed makes attack selection and ordering reproducible.",
    "reproducibility_checksum": "A full-content digest detects later artifact edits.",
    "gate_check_summary": "The first expected and observed mismatch makes failure actionable.",
    "verifier_is_oracle": "True keeps exact replay from becoming a learned-selector claim.",
    "verdict_class": "A closed terminal class supports safe automation.",
    "honest_verdict": "A matching prefix states the audit boundary without a pooled-score claim.",
}

_EXACT_FIXTURE_CACHE: dict[str, dict[str, list[JsonDict]]] = {}


def canonical_json(value: Any) -> str:
    """Serialize evidence with stable keys so the same content has one hash."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash exact UTF-8 text so whitespace and control bytes remain significant."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:  # pragma: no cover - exercised by the dated audit.
    """Hash a file in chunks so large upstream artifacts do not require a second copy."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def gate_row(check: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    """Keep both gate values so a blocked verdict names an exact mismatch."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": observed == expected if passed is None else bool(passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return the first failed check while preserving the full ordered receipt."""

    copied = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in copied if row.get("passed") is not True), None)
    return {
        "passed": failed is None,
        "failed_check": None if failed is None else failed.get("check"),
        "expected_value": "all checks pass" if failed is None else failed.get("expected_value"),
        "observed_value": "all checks pass" if failed is None else failed.get("observed_value"),
        "checks": copied,
    }


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all artifact content except the self-referential checksum field."""

    stable = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_text(canonical_json(stable))


def _models_and_seeds(bank: Mapping[str, Any]) -> tuple[list[str], list[int]]:
    """Read roster metadata without consulting parse or exact outcome fields."""

    models = [str(row.get("hf_id")) for row in bank.get("model_specs", [])]
    seeds = sorted(
        {
            int(row.get("seed"))
            for row in bank.get("raw_proposal_rows", [])
            if row.get("arm", "proposal") == "proposal" and type(row.get("seed")) is int
        }
    )
    return models, seeds


def build_required_support_schema(
    fixture: Mapping[str, Any], models: Sequence[str], seeds: Sequence[int]
) -> JsonDict:
    """Freeze every source, model, seed, and reachable-family cell before replay."""

    groups = list(
        dict.fromkeys(str(row.get("source_group_id")) for row in fixture.get("unit_rows", []))
    )
    reachable = {
        (str(row.get("source_group_id")), str(row.get("operator")))
        for row in fixture.get("entrance_rows", [])
        if row.get("reachable") is True
    }
    cells = [
        {
            "source_group_id": group,
            "model_id": str(model),
            "seed": int(seed),
            "entrance_family": family,
            "applicable": (group, family) in reachable,
        }
        for group in groups
        for model in models
        for seed in seeds
        for family in ENTRANCE_FAMILIES
    ]
    schema = {
        "frozen_before_proposal_outcomes": True,
        "applicability_rule": "reachable fixture entrance exists in source group and operator family",
        "source_group_ids": groups,
        "model_ids": [str(value) for value in models],
        "proposal_seeds": [int(value) for value in seeds],
        "entrance_families": list(ENTRANCE_FAMILIES),
        "cells": cells,
    }
    schema["schema_hash"] = sha256_text(canonical_json(schema))
    return schema


def _independent_entrances(fixture: Mapping[str, Any]) -> dict[str, list[JsonDict]]:
    """Re-enumerate exact labels once per fixture content, never from producer labels."""

    units = [
        {
            "unit_id": row.get("unit_id"),
            "source_group_id": row.get("source_group_id"),
            "split": row.get("split"),
            "numbers": row.get("numbers"),
            "target": row.get("target"),
        }
        for row in fixture.get("unit_rows", [])
    ]
    cache_key = sha256_text(canonical_json(units))
    if cache_key not in _EXACT_FIXTURE_CACHE:
        _EXACT_FIXTURE_CACHE[cache_key] = {
            str(unit["unit_id"]): exact.label_unit_entrances(unit) for unit in units
        }
    return _EXACT_FIXTURE_CACHE[cache_key]


def _index_rows(
    rows: Sequence[Mapping[str, Any]], key: str = "raw_key"
) -> dict[str, list[JsonDict]]:
    """Keep every producer alternative because a dictionary overwrite hides conflicts."""

    indexed: dict[str, list[JsonDict]] = {}
    for row in rows:
        indexed.setdefault(str(row.get(key)), []).append(deepcopy(dict(row)))
    return indexed


def _expected_prompt(unit: Mapping[str, Any]) -> str:
    """Rebuild the prompt from the fixture's public fields only."""

    visible = {
        "unit_id": unit.get("unit_id"),
        "numbers": deepcopy(unit.get("numbers")),
        "target": unit.get("target"),
        "formatting_rules": deepcopy(unit.get("formatting_rules", exact.FORMATTING_RULES)),
    }
    return producer.build_proposal_prompt(visible)


def _recompute_primary_rows(bank: Mapping[str, Any], fixture: Mapping[str, Any]) -> JsonDict:
    """Replay bytes, parses, labels, witnesses, and prompt inputs from primary rows."""

    raw_rows = [deepcopy(dict(row)) for row in bank.get("raw_proposal_rows", [])]
    units = {str(row.get("unit_id")): row for row in fixture.get("unit_rows", [])}
    exact_by_unit = _independent_entrances(fixture)
    exact_index = {
        (unit_id, tuple(row["operand_pair"]), str(row["operator"])): row
        for unit_id, rows in exact_by_unit.items()
        for row in rows
    }
    parse_index = _index_rows(bank.get("parse_rows", []))
    label_index = _index_rows(bank.get("exact_label_rows", []))
    witness_index = _index_rows(bank.get("causal_witness_rows", []))
    raw_key_counts = Counter(str(row.get("raw_key")) for row in raw_rows)
    seen_entrances: Counter[tuple[str, str, tuple[int, ...], str]] = Counter()

    raw_hash_rows: list[JsonDict] = []
    parse_rows: list[JsonDict] = []
    label_rows: list[JsonDict] = []
    witness_rows: list[JsonDict] = []
    prompt_rows: list[JsonDict] = []
    independent_labels: list[JsonDict] = []
    errors: list[str] = []

    for raw in raw_rows:
        raw_key = str(raw.get("raw_key"))
        text = str(raw.get("raw_text", ""))
        try:
            decoded = bytes.fromhex(str(raw.get("raw_bytes_hex", ""))).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            decoded = ""
        raw_passed = decoded == text and sha256_text(decoded) == raw.get("raw_output_hash")
        raw_hash_rows.append(
            {
                "raw_key": raw_key,
                "declared_hash": raw.get("raw_output_hash"),
                "recomputed_hash": sha256_text(decoded),
                "bytes_match_text": decoded == text,
                "passed": raw_passed,
            }
        )
        if not raw_passed:
            errors.append("raw_byte_mismatch")

        parsed = producer.parse_entrance(decoded)
        producer_parses = parse_index.get(raw_key, [])
        parse_passed = len(producer_parses) == 1 and all(
            (
                row.get("parsed_entrance") == parsed
                and row.get("parse_failure") is (parsed is None)
                and row.get("raw_output_hash") == raw.get("raw_output_hash")
            )
            for row in producer_parses
        )
        parse_rows.append(
            {
                "raw_key": raw_key,
                "recomputed_entrance": deepcopy(parsed),
                "producer_row_count": len(producer_parses),
                "passed": parse_passed,
            }
        )
        if not parse_passed:
            errors.append("parse_recomputation_mismatch")

        exact_row = None
        duplicate = False
        if parsed is not None:
            pair = tuple(int(value) for value in parsed["operand_pair"])
            exact_row = exact_index.get((str(raw.get("unit_id")), pair, str(parsed["operator"])))
            duplicate_key = (
                str(raw.get("model_id")),
                str(raw.get("unit_id")),
                pair,
                str(parsed["operator"]),
            )
            duplicate = seen_entrances[duplicate_key] > 0
            seen_entrances[duplicate_key] += 1
        independent = {
            "raw_key": raw_key,
            "raw_output_hash": raw.get("raw_output_hash"),
            "model_id": str(raw.get("model_id")),
            "unit_id": str(raw.get("unit_id")),
            "seed": raw.get("seed"),
            "entrance_id": exact_row.get("entrance_id") if exact_row else None,
            "entrance_family": parsed.get("operator") if parsed else None,
            "legal": exact_row is not None,
            "reachable": exact_row.get("reachable") is True if exact_row else False,
            "duplicate": duplicate,
        }
        independent_labels.append(independent)
        expected_label = {
            key: independent[key] for key in ("entrance_id", "legal", "reachable", "duplicate")
        }
        producer_labels = label_index.get(raw_key, [])
        label_passed = len(producer_labels) == 1 and all(
            all(row.get(key) == value for key, value in expected_label.items())
            and row.get("raw_output_hash") == raw.get("raw_output_hash")
            for row in producer_labels
        )
        label_rows.append(
            {
                "raw_key": raw_key,
                **expected_label,
                "producer_row_count": len(producer_labels),
                "passed": label_passed,
            }
        )
        if not label_passed:
            errors.append("label_recomputation_mismatch")

        causal = bool(
            exact_row
            and exact_row.get("reachable") is True
            and str(raw.get("unit_id")) in units
            and exact.replay_entrance_witness(
                units[str(raw.get("unit_id"))]["numbers"],
                int(units[str(raw.get("unit_id"))]["target"]),
                exact_row,
            )
        )
        producer_witnesses = witness_index.get(raw_key, [])
        witness_passed = len(producer_witnesses) == 1 and all(
            row.get("causal_witness") is causal
            and row.get("entrance_id") == independent["entrance_id"]
            and row.get("raw_output_hash") == raw.get("raw_output_hash")
            for row in producer_witnesses
        )
        witness_rows.append(
            {
                "raw_key": raw_key,
                "entrance_id": independent["entrance_id"],
                "replayed_causal_witness": causal,
                "producer_row_count": len(producer_witnesses),
                "passed": witness_passed,
            }
        )
        if not witness_passed:
            errors.append("causal_witness_replay_mismatch")

        unit = units.get(str(raw.get("unit_id")))
        prompt = str(raw.get("prompt", ""))
        expected_prompt = _expected_prompt(unit) if unit else None
        expected_messages = producer.build_role_messages(prompt)
        expected_raw_key = f"{raw.get('model_id')}|proposal|{raw.get('unit_id')}|{raw.get('seed')}"
        prompt_passed = bool(
            unit
            and prompt == expected_prompt
            and raw.get("prompt_hash") == sha256_text(prompt)
            and raw.get("role_messages") == expected_messages
            and raw.get("role_messages_hash") == sha256_text(canonical_json(expected_messages))
            and raw_key == expected_raw_key
            and raw.get("raw_persisted_before_parse") is True
            and raw.get("terminal_state") == "complete"
        )
        prompt_rows.append(
            {
                "raw_key": raw_key,
                "expected_raw_key": expected_raw_key,
                "unit_known": unit is not None,
                "prompt_hash": raw.get("prompt_hash"),
                "passed": prompt_passed,
            }
        )
        if raw_key != expected_raw_key:
            errors.append("raw_key_identity_mismatch")
        if not prompt_passed:
            errors.append("prompt_or_role_parity_mismatch")

    if any(count != 1 for count in raw_key_counts.values()):
        errors.append("duplicate_raw_key")
    return {
        "raw_hash_rows": raw_hash_rows,
        "parse_recomputation_rows": parse_rows,
        "label_recomputation_rows": label_rows,
        "causal_witness_replay_rows": witness_rows,
        "prompt_parity_rows": prompt_rows,
        "independent_label_rows": independent_labels,
        "raw_key_counts": dict(raw_key_counts),
        "errors": errors,
    }


def _conflict_rows(
    bank: Mapping[str, Any], independent_labels: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Expose every producer label conflict and retain the independently selected label."""

    independent = {str(row.get("raw_key")): row for row in independent_labels}
    rows = []
    for raw_key, alternatives in _index_rows(bank.get("exact_label_rows", [])).items():
        values = {
            canonical_json(
                {key: row.get(key) for key in ("entrance_id", "legal", "reachable", "duplicate")}
            )
            for row in alternatives
        }
        if len(values) <= 1:
            continue
        selected = independent.get(raw_key, {})
        rows.append(
            {
                "raw_key": raw_key,
                "producer_alternatives": [json.loads(value) for value in sorted(values)],
                "independent_label": {
                    key: selected.get(key) for key in ("entrance_id", "legal", "reachable")
                },
                "resolved_by": "independent_exact_replay",
                "rejected_alternative_count": len(values) - 1,
                "passed": False,
            }
        )
    return rows


def _family_support(
    schema: Mapping[str, Any],
    fixture: Mapping[str, Any],
    labels: Sequence[Mapping[str, Any]],
    raw_key_counts: Mapping[str, int],
    conflict_keys: set[str],
) -> list[JsonDict]:
    """Assign one explicit state to every frozen support cell."""

    unit_groups = {
        str(row.get("unit_id")): str(row.get("source_group_id"))
        for row in fixture.get("unit_rows", [])
    }
    rows = []
    for cell in schema.get("cells", []):
        matches = [
            row
            for row in labels
            if unit_groups.get(str(row.get("unit_id"))) == cell.get("source_group_id")
            and row.get("model_id") == cell.get("model_id")
            and row.get("seed") == cell.get("seed")
            and row.get("entrance_family") == cell.get("entrance_family")
            and row.get("legal") is True
        ]
        keys = [str(row.get("raw_key")) for row in matches]
        if cell.get("applicable") is not True:
            status = "inapplicable"
        elif not matches:
            status = "missing"
        elif any(key in conflict_keys for key in keys):
            status = "conflicting"
        elif any(int(raw_key_counts.get(key, 0)) > 1 for key in keys):
            status = "duplicate"
        else:
            status = "present"
        rows.append(
            {
                **deepcopy(dict(cell)),
                "status": status,
                "proposal_count": len(matches),
                "distinct_raw_key_count": len(set(keys)),
                "raw_keys": sorted(set(keys)),
            }
        )
    return rows


def _parity_rows(
    bank: Mapping[str, Any], fixture: Mapping[str, Any], schema: Mapping[str, Any]
) -> JsonDict:
    """Reduce primary producer receipts without using producer summary rates."""

    raw = [dict(row) for row in bank.get("raw_proposal_rows", [])]
    models = list(schema.get("model_ids", []))
    seeds = list(schema.get("proposal_seeds", []))
    units = [str(row.get("unit_id")) for row in fixture.get("unit_rows", [])]
    unit_groups = {
        str(row.get("unit_id")): str(row.get("source_group_id"))
        for row in fixture.get("unit_rows", [])
    }
    identity_index = {str(row.get("model_id")): row for row in bank.get("model_identity_rows", [])}
    file_index = {str(row.get("model_id")): row for row in bank.get("model_file_hash_rows", [])}
    spec_index = {str(row.get("hf_id")): row for row in bank.get("model_specs", [])}

    template_rows = []
    stop_rows = []
    budget_rows = []
    identity_rows = []
    for model in models:
        model_raw = [row for row in raw if str(row.get("model_id")) == model]
        identity = identity_index.get(model, {})
        template_passed = bool(
            model_raw
            and identity.get("passed") is True
            and identity.get("identity_matches") is True
            and identity.get("chat_template_present") is True
            and all(
                row.get("chat_template_present") is True
                and row.get("chat_template_hash") == identity.get("chat_template_hash")
                and row.get("transport_method") == "create_chat_completion"
                for row in model_raw
            )
        )
        template_rows.append(
            {
                "model_id": model,
                "row_count": len(model_raw),
                "chat_template_hash": identity.get("chat_template_hash"),
                "passed": template_passed,
            }
        )
        stop_rows.append(
            {
                "model_id": model,
                "expected_stop": deepcopy(dict(bank.get("sampling_config") or {}).get("stop")),
                "passed": bool(model_raw)
                and all(
                    row.get("stop_config") == dict(bank.get("sampling_config") or {}).get("stop")
                    and dict(row.get("generation_config") or {}).get("stop")
                    == dict(bank.get("sampling_config") or {}).get("stop")
                    for row in model_raw
                ),
            }
        )
        budget = int(
            dict(bank.get("sampling_config") or {}).get("completion_budget_tokens", 0) or 0
        )
        budget_rows.append(
            {
                "model_id": model,
                "expected_completion_budget_tokens": budget,
                "passed": bool(model_raw)
                and all(
                    int(row.get("requested_completion_budget_tokens", 0) or 0) == budget
                    and int(row.get("effective_completion_budget_tokens", 0) or 0)
                    + int(row.get("prefix_token_count", 0) or 0)
                    == budget
                    and row.get("generation_config") == bank.get("sampling_config")
                    for row in model_raw
                ),
            }
        )
        file_row = file_index.get(model, {})
        spec = spec_index.get(model, {})
        identity_rows.append(
            {
                "model_id": model,
                "model_path": spec.get("model_path"),
                "model_file_hash": file_row.get("sha256"),
                "metadata_hash": identity.get("metadata_hash"),
                "passed": bool(identity)
                and identity.get("passed") is True
                and identity.get("identity_matches") is True
                and identity.get("tokenizer_source") == "embedded_gguf"
                and file_row.get("path") == spec.get("model_path")
                and str(file_row.get("sha256", "")).startswith("sha256:"),
            }
        )

    expected_units = set(units)
    seed_rows = []
    source_rows = []
    for model in models:
        for seed in seeds:
            selected = [
                row for row in raw if str(row.get("model_id")) == model and row.get("seed") == seed
            ]
            observed_units = [str(row.get("unit_id")) for row in selected]
            seed_rows.append(
                {
                    "model_id": model,
                    "seed": seed,
                    "expected_unit_count": len(expected_units),
                    "observed_unit_count": len(set(observed_units)),
                    "passed": set(observed_units) == expected_units
                    and len(observed_units) == len(expected_units),
                }
            )
            for group in schema.get("source_group_ids", []):
                expected_group_units = {
                    unit for unit, source_group in unit_groups.items() if source_group == group
                }
                observed_group_units = {
                    str(row.get("unit_id"))
                    for row in selected
                    if unit_groups.get(str(row.get("unit_id"))) == group
                }
                source_rows.append(
                    {
                        "source_group_id": group,
                        "model_id": model,
                        "seed": seed,
                        "expected_unit_count": len(expected_group_units),
                        "observed_unit_count": len(observed_group_units),
                        "passed": observed_group_units == expected_group_units,
                    }
                )

    runner = dict(bank.get("runner_receipt") or {})
    runner_rows = [
        {
            "backend": runner.get("backend"),
            "version": runner.get("version"),
            "transport_method": runner.get("transport_method"),
            "passed": runner.get("cuda_offload") is True
            and runner.get("transport_method") == "create_chat_completion"
            and all(row.get("transport_method") == "create_chat_completion" for row in raw),
        }
    ]
    executions = [dict(row) for row in bank.get("model_execution_rows", [])]
    shards = {(str(row.get("model_id")), str(row.get("phase"))) for row in executions}

    def telemetry_present(rows: Sequence[Mapping[str, Any]], shard: tuple[str, str]) -> bool:
        selected = [
            row for row in rows if (str(row.get("model_id")), str(row.get("phase"))) == shard
        ]
        return bool(selected) and any(
            int(row.get("used_memory_mb", 0) or 0) > 0
            or any(
                int(device.get("memory_used_mb", 0) or 0) > 0 for device in row.get("devices", [])
            )
            for row in selected
        )

    gpu_rows = []
    lease_rows = []
    for shard in sorted(shards):
        execution = next(
            row for row in executions if (str(row.get("model_id")), str(row.get("phase"))) == shard
        )
        gpu_rows.append(
            {
                "model_id": shard[0],
                "phase": shard[1],
                "offloaded_layers": execution.get("offloaded_layers"),
                "used_both_gpus": execution.get("used_both_gpus"),
                "passed": int(execution.get("offloaded_layers", 0) or 0) > 0
                and execution.get("used_both_gpus") is True
                and int(execution.get("model_load_count", 0) or 0) == 1
                and telemetry_present(bank.get("stage_gpu_telemetry_rows", []), shard)
                and telemetry_present(bank.get("task_gpu_telemetry_rows", []), shard),
            }
        )
        leases = [
            row
            for row in bank.get("gpu_lease_rows", [])
            if (str(row.get("model_id")), str(row.get("phase"))) == shard
        ]
        cleanups = [
            row
            for row in bank.get("cleanup_rows", [])
            if (str(row.get("model_id")), str(row.get("phase"))) == shard
        ]
        releases = [
            row
            for row in bank.get("vram_release_rows", [])
            if (str(row.get("model_id")), str(row.get("phase"))) == shard
        ]
        checkpoints = [
            row
            for row in bank.get("checkpoint_rows", [])
            if (str(row.get("model_id")), str(row.get("phase"))) == shard
        ]
        lease_rows.append(
            {
                "model_id": shard[0],
                "phase": shard[1],
                "lease_count": len(leases),
                "checkpoint_paths": [row.get("path") for row in checkpoints],
                "passed": bool(leases)
                and all(
                    row.get("owner_preserved") is True
                    and row.get("released") is True
                    and row.get("lease_lost") is False
                    for row in leases
                )
                and bool(cleanups)
                and all(row.get("passed") is True for row in cleanups)
                and bool(releases)
                and all(row.get("passed") is True for row in releases)
                and len(checkpoints) == 1
                and str(checkpoints[0].get("sha256", "")).startswith("sha256:"),
            }
        )
    return {
        "chat_template_parity_rows": template_rows,
        "stop_parity_rows": stop_rows,
        "budget_parity_rows": budget_rows,
        "model_identity_rows": identity_rows,
        "seed_coverage_rows": seed_rows,
        "source_group_coverage_rows": source_rows,
        "runner_receipt_rows": runner_rows,
        "gpu_telemetry_rows": gpu_rows,
        "lease_cleanup_rows": lease_rows,
    }


def _leakage_rows(bank: Mapping[str, Any]) -> list[JsonDict]:
    """Scan only model-input structures for exact, witness, solver, or future-label keys."""

    forbidden = (
        "reachable",
        "unreachable",
        "entrance_id",
        "causal_witness",
        "continuation_witness",
        "residual_numbers",
        "exact_label",
        "solver_derived",
        "future_label",
    )

    def paths(value: Any, prefix: str = "") -> list[str]:
        found = []
        if isinstance(value, Mapping):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if any(token in str(key).lower() for token in forbidden):
                    found.append(path)
                found.extend(paths(child, path))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                found.extend(paths(child, f"{prefix}[{index}]"))
        return found

    rows = []
    for raw in bank.get("raw_proposal_rows", []):
        model_input = {
            "prompt": raw.get("prompt"),
            "role_messages": raw.get("role_messages"),
            "generation_config": raw.get("generation_config"),
            "stop_config": raw.get("stop_config"),
        }
        found = paths(model_input)
        rows.append(
            {
                "raw_key": raw.get("raw_key"),
                "searched_fields": list(model_input),
                "forbidden_paths": found,
                "attack_detected": bool(found),
                "passed": not found,
            }
        )
    return rows


def measure_headroom(
    labels: Sequence[Mapping[str, Any]],
    fixture: Mapping[str, Any],
    *,
    minimum_units: int = MINIMUM_HEADROOM_UNITS,
) -> JsonDict:
    """Count held units with both outcomes without fitting or scoring a selector."""

    held = {
        str(row.get("unit_id"))
        for row in fixture.get("unit_rows", [])
        if row.get("split") == "held"
    }
    if not held:
        held_groups = set(
            dict(fixture.get("split_manifest") or {}).get("held_source_group_ids", [])
        )
        held = {
            str(row.get("unit_id"))
            for row in fixture.get("unit_rows", [])
            if row.get("source_group_id") in held_groups
        }
    unit_rows = []
    mixed_count = 0
    for unit_id in sorted(held):
        selected = [row for row in labels if str(row.get("unit_id")) == unit_id]
        reachable = sum(row.get("reachable") is True for row in selected)
        unreachable = sum(row.get("reachable") is False for row in selected)
        mixed = reachable > 0 and unreachable > 0
        mixed_count += int(mixed)
        unit_rows.append(
            {
                "kind": "held_unit",
                "unit_id": unit_id,
                "reachable_proposal_count": reachable,
                "unreachable_proposal_count": unreachable,
                "mixed_reachability": mixed,
            }
        )
    controls = []
    for seed in sorted({row.get("seed") for row in labels}, key=str):
        selected = [row for row in labels if row.get("seed") == seed and row.get("unit_id") in held]
        rate = (
            sum(row.get("reachable") is True for row in selected) / len(selected)
            if selected
            else 1.0
        )
        controls.append(
            {
                "kind": "non_oracle_base_control",
                "control": f"pooled_proposal_seed:{seed}",
                "proposal_count": len(selected),
                "reachability_rate": rate,
                "perfect": rate == 1.0,
            }
        )
    no_perfect = bool(controls) and all(row["perfect"] is False for row in controls)
    return {
        "headroom_rows": [*unit_rows, *controls],
        "headroom_unit_count": mixed_count,
        "no_perfect_base_control": no_perfect,
        "ready": mixed_count >= int(minimum_units) and no_perfect,
    }


def recompute_audit(
    bank: Mapping[str, Any], fixture: Mapping[str, Any], schema: Mapping[str, Any]
) -> JsonDict:
    """Recompute all support evidence without reading producer aggregate scores."""

    primary = _recompute_primary_rows(bank, fixture)
    labels = primary.pop("independent_label_rows")
    raw_key_counts = primary.pop("raw_key_counts")
    conflicts = _conflict_rows(bank, labels)
    conflict_keys = {str(row.get("raw_key")) for row in conflicts}
    support = _family_support(schema, fixture, labels, raw_key_counts, conflict_keys)
    parity = _parity_rows(bank, fixture, schema)
    leakage = _leakage_rows(bank)
    headroom = measure_headroom(labels, fixture)
    expected_raw_keys = {
        f"{model}|proposal|{unit.get('unit_id')}|{seed}"
        for model in schema.get("model_ids", [])
        for seed in schema.get("proposal_seeds", [])
        for unit in fixture.get("unit_rows", [])
    }
    observed_raw_keys = {str(row.get("raw_key")) for row in bank.get("raw_proposal_rows", [])}
    errors = list(primary.pop("errors"))
    if observed_raw_keys != expected_raw_keys or len(bank.get("raw_proposal_rows", [])) != len(
        expected_raw_keys
    ):
        errors.append("proposal_cross_product_mismatch")
    parity_rows = [row for rows in parity.values() for row in rows]
    if any(row.get("passed") is not True for row in parity_rows):
        errors.append("primary_receipt_parity_mismatch")
    if conflicts:
        errors.append("conflicting_producer_labels")
    if any(row.get("status") not in {"present", "inapplicable"} for row in support):
        errors.append("family_support_incomplete")
    if any(row.get("attack_detected") is True for row in leakage):
        errors.append("future_label_leakage")
    errors = list(dict.fromkeys(errors))
    recomputation_passed = all(
        row.get("passed") is True
        for name in (
            "raw_hash_rows",
            "parse_recomputation_rows",
            "label_recomputation_rows",
            "causal_witness_replay_rows",
            "prompt_parity_rows",
        )
        for row in primary[name]
    )
    telemetry_passed = all(
        row.get("passed") is True
        for name in ("runner_receipt_rows", "gpu_telemetry_rows", "lease_cleanup_rows")
        for row in parity[name]
    )
    authenticity_passed = (
        observed_raw_keys == expected_raw_keys
        and len(bank.get("raw_proposal_rows", [])) == len(expected_raw_keys)
        and recomputation_passed
        and all(row.get("passed") is True for row in parity_rows)
    )
    family_passed = all(row.get("status") in {"present", "inapplicable"} for row in support)
    return {
        "required_support_schema": deepcopy(dict(schema)),
        **primary,
        **parity,
        "family_support_rows": support,
        "missing_family_rows": [
            deepcopy(row)
            for row in support
            if row.get("applicable") is True and row.get("status") != "present"
        ],
        "conflict_rows": conflicts,
        "leakage_attack_rows": leakage,
        "headroom_rows": headroom["headroom_rows"],
        "headroom_unit_count": headroom["headroom_unit_count"],
        "independent_label_rows": labels,
        "per_game_results": _per_game_results(labels, fixture),
        "authenticity_passed": authenticity_passed,
        "family_sufficiency_passed": family_passed,
        "conflict_resolution_passed": not conflicts,
        "recomputation_passed": recomputation_passed,
        "telemetry_passed": telemetry_passed,
        "leakage_passed": all(row.get("passed") is True for row in leakage),
        "errors": errors,
    }


def _per_game_results(
    labels: Sequence[Mapping[str, Any]], fixture: Mapping[str, Any]
) -> list[JsonDict]:
    """Keep proposal outcomes visible per unit so pooled support cannot dominate."""

    results = []
    for unit in fixture.get("unit_rows", []):
        unit_id = str(unit.get("unit_id"))
        selected = [row for row in labels if str(row.get("unit_id")) == unit_id]
        results.append(
            {
                "unit_id": unit_id,
                "source_group_id": unit.get("source_group_id"),
                "split": unit.get("split"),
                "proposal_count": len(selected),
                "reachable_proposal_count": sum(row.get("reachable") is True for row in selected),
                "unreachable_proposal_count": sum(
                    row.get("reachable") is False for row in selected
                ),
                "entrance_families": sorted(
                    {str(row.get("entrance_family")) for row in selected if row.get("legal")}
                ),
            }
        )
    return results


def _mutate_bank(
    bank: Mapping[str, Any], field: str, rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Copy only the changed row list so attacks do not multiply the 43 MB bank."""

    changed = dict(bank)
    changed[field] = [deepcopy(dict(row)) for row in rows]
    return changed


def run_counterfactual_attacks(
    bank: Mapping[str, Any], fixture: Mapping[str, Any], schema: Mapping[str, Any]
) -> list[JsonDict]:
    """Delete, swap, mutate, and conflict evidence while preserving pooled counts when possible."""

    baseline = recompute_audit(bank, fixture, schema)
    raw = [dict(row) for row in bank.get("raw_proposal_rows", [])]
    labels = baseline["independent_label_rows"]
    target = next(
        (
            row
            for row in baseline["family_support_rows"]
            if row.get("status") == "present" and row.get("raw_keys")
        ),
        None,
    )
    delete_keys = set(target.get("raw_keys", [])) if target else set()
    deletion_bank = _mutate_bank(
        bank, "raw_proposal_rows", [row for row in raw if row.get("raw_key") not in delete_keys]
    )
    deletion = recompute_audit(deletion_bank, fixture, schema)

    swapped_rows = [deepcopy(row) for row in raw]
    if len(swapped_rows) >= 2:
        swapped_rows[0]["unit_id"], swapped_rows[-1]["unit_id"] = (
            swapped_rows[-1].get("unit_id"),
            swapped_rows[0].get("unit_id"),
        )
    source_swap = recompute_audit(
        _mutate_bank(bank, "raw_proposal_rows", swapped_rows), fixture, schema
    )

    mutated_rows = [deepcopy(row) for row in raw]
    if mutated_rows:
        mutated_rows[0]["raw_bytes_hex"] = b"counterfactual mutation".hex()
    raw_mutation = recompute_audit(
        _mutate_bank(bank, "raw_proposal_rows", mutated_rows), fixture, schema
    )

    label_rows = [dict(row) for row in bank.get("exact_label_rows", [])]
    if label_rows:
        conflict = deepcopy(label_rows[0])
        conflict["reachable"] = not bool(conflict.get("reachable"))
        label_rows.append(conflict)
    label_conflict = recompute_audit(
        _mutate_bank(bank, "exact_label_rows", label_rows), fixture, schema
    )
    baseline_count = len(raw)
    cases = (
        ("family_deletion", deletion, len(deletion_bank["raw_proposal_rows"])),
        ("source_swap", source_swap, len(swapped_rows)),
        ("raw_byte_mutation", raw_mutation, len(mutated_rows)),
        ("label_conflict", label_conflict, baseline_count),
    )
    return [
        {
            "attack": name,
            "target_support_cell": deepcopy(target),
            "baseline_raw_row_count": baseline_count,
            "attacked_raw_row_count": count,
            "pooled_row_count_preserved": count == baseline_count,
            "attack_detected": bool(result.get("errors"))
            and (
                result.get("authenticity_passed") is False
                or result.get("family_sufficiency_passed") is False
                or result.get("conflict_resolution_passed") is False
            ),
            "failed_checks": deepcopy(result.get("errors", [])),
        }
        for name, result, count in cases
    ]


def _artifact_gate_rows(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Recompute terminal gates from detailed rows instead of saved summary booleans."""

    def all_passed(*fields: str) -> bool:
        return all(row.get("passed") is True for field in fields for row in artifact.get(field, []))

    preconditions = dict(artifact.get("preconditions_checked") or {})
    support_status = all(
        row.get("status") in {"present", "inapplicable"}
        for row in artifact.get("family_support_rows", [])
    ) and not artifact.get("missing_family_rows")
    no_perfect = all(
        row.get("perfect") is False
        for row in artifact.get("headroom_rows", [])
        if row.get("kind") == "non_oracle_base_control"
    ) and any(
        row.get("kind") == "non_oracle_base_control" for row in artifact.get("headroom_rows", [])
    )
    checks = [
        gate_row("preconditions", True, preconditions.get("all_passed") is True),
        gate_row(
            "fresh_process_replay",
            True,
            bool(artifact.get("fresh_process_rows")) and all_passed("fresh_process_rows"),
        ),
        gate_row(
            "authenticity",
            True,
            all_passed(
                "raw_hash_rows",
                "parse_recomputation_rows",
                "label_recomputation_rows",
                "causal_witness_replay_rows",
                "prompt_parity_rows",
                "chat_template_parity_rows",
                "stop_parity_rows",
                "budget_parity_rows",
                "model_identity_rows",
                "seed_coverage_rows",
                "source_group_coverage_rows",
            ),
        ),
        gate_row("family_sufficiency", True, support_status),
        gate_row(
            "conflict_resolution",
            True,
            all_passed("conflict_rows") and not artifact.get("conflict_rows"),
        ),
        gate_row(
            "independent_recomputation",
            True,
            all_passed(
                "raw_hash_rows",
                "parse_recomputation_rows",
                "label_recomputation_rows",
                "causal_witness_replay_rows",
            ),
        ),
        gate_row(
            "telemetry_and_cleanup",
            True,
            all_passed("runner_receipt_rows", "gpu_telemetry_rows", "lease_cleanup_rows"),
        ),
        gate_row(
            "leakage",
            True,
            all_passed("leakage_attack_rows")
            and not any(
                row.get("attack_detected") is True
                for row in artifact.get("leakage_attack_rows", [])
            ),
        ),
        gate_row(
            "counterfactual_attacks",
            True,
            not artifact.get("counterfactual_swap_rows")
            or all(
                row.get("attack_detected") is True
                for row in artifact.get("counterfactual_swap_rows", [])
            ),
        ),
        gate_row(
            "selector_headroom",
            True,
            int(artifact.get("headroom_unit_count", 0) or 0) >= MINIMUM_HEADROOM_UNITS
            and no_perfect,
        ),
    ]
    return checks


def _empty_rows() -> JsonDict:
    """Create every evidence list before a precondition can stop the audit."""

    return {
        field: []
        for field in REQUIRED_ARTIFACT_FIELDS
        if field.endswith("_rows") or field in {"rows", "per_game_results"}
    }


def build_blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    preconditions: Mapping[str, Any],
    source_artifact_hashes: Mapping[str, Any],
) -> JsonDict:
    """Write a complete terminal block without pretending replay took place."""

    checks = list(preconditions.get("checks", []))
    artifact = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": str(run_date),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(dict(preconditions)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "duration_s": float(duration_s),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "cited_upstream_artifacts": [],
        "upstream_gate_rows": [],
        **_empty_rows(),
        "required_support_schema": {},
        "headroom_unit_count": 0,
        "entrance_support_audit_ready_score": 0,
        "entrance_selector_headroom_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked: entrance support audit precondition failed",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    bank: Mapping[str, Any],
    fixture: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    source_artifact_hashes: Mapping[str, Any],
    replay: Mapping[str, Any],
    fresh_process_rows: Sequence[Mapping[str, Any]],
    minimum_headroom_units: int = MINIMUM_HEADROOM_UNITS,
    counterfactual_swap_rows: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build one terminal audit from cold replay rows and independently reduced gates."""

    headroom = measure_headroom(
        replay.get("independent_label_rows", []), fixture, minimum_units=minimum_headroom_units
    )
    artifact = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": str(run_date),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(dict(preconditions)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "no_model_load",
        "duration_s": float(duration_s),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "cited_upstream_artifacts": [
            {
                "path": "results/experiment_7086_v621_three_family_entrance_bank.json",
                "sha256": source_artifact_hashes.get(
                    "experiment_7086_v621_three_family_entrance_bank.json"
                ),
                "fields_imported": ["raw_proposal_rows", "primary execution receipts"],
            },
            {
                "path": "results/experiment_7064_v619_exact_entrance_fixture.json",
                "sha256": source_artifact_hashes.get(
                    "experiment_7064_v619_exact_entrance_fixture.json"
                ),
                "fields_imported": ["unit_rows", "entrance_rows", "split_manifest"],
            },
        ],
        "upstream_gate_rows": deepcopy(list(bank.get("upstream_gate_rows", []))),
        "rows": [],
        "per_game_results": deepcopy(list(replay.get("per_game_results", []))),
        "fresh_process_rows": [deepcopy(dict(row)) for row in fresh_process_rows],
        "raw_hash_rows": deepcopy(list(replay.get("raw_hash_rows", []))),
        "parse_recomputation_rows": deepcopy(list(replay.get("parse_recomputation_rows", []))),
        "label_recomputation_rows": deepcopy(list(replay.get("label_recomputation_rows", []))),
        "causal_witness_replay_rows": deepcopy(list(replay.get("causal_witness_replay_rows", []))),
        "prompt_parity_rows": deepcopy(list(replay.get("prompt_parity_rows", []))),
        "chat_template_parity_rows": deepcopy(list(replay.get("chat_template_parity_rows", []))),
        "stop_parity_rows": deepcopy(list(replay.get("stop_parity_rows", []))),
        "budget_parity_rows": deepcopy(list(replay.get("budget_parity_rows", []))),
        "model_identity_rows": deepcopy(list(replay.get("model_identity_rows", []))),
        "seed_coverage_rows": deepcopy(list(replay.get("seed_coverage_rows", []))),
        "source_group_coverage_rows": deepcopy(list(replay.get("source_group_coverage_rows", []))),
        "runner_receipt_rows": deepcopy(list(replay.get("runner_receipt_rows", []))),
        "gpu_telemetry_rows": deepcopy(list(replay.get("gpu_telemetry_rows", []))),
        "lease_cleanup_rows": deepcopy(list(replay.get("lease_cleanup_rows", []))),
        "required_support_schema": deepcopy(dict(replay.get("required_support_schema", {}))),
        "family_support_rows": deepcopy(list(replay.get("family_support_rows", []))),
        "missing_family_rows": deepcopy(list(replay.get("missing_family_rows", []))),
        "conflict_rows": deepcopy(list(replay.get("conflict_rows", []))),
        "counterfactual_swap_rows": [deepcopy(dict(row)) for row in counterfactual_swap_rows],
        "leakage_attack_rows": deepcopy(list(replay.get("leakage_attack_rows", []))),
        "headroom_rows": deepcopy(headroom["headroom_rows"]),
        "headroom_unit_count": headroom["headroom_unit_count"],
        "entrance_support_audit_ready_score": 0,
        "entrance_selector_headroom_ready_score": int(headroom["ready"]),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "verdict_class": "null",
        "honest_verdict": "null: entrance support evidence is insufficient",
    }
    artifact["rows"] = _artifact_gate_rows(artifact)
    support_checks = artifact["rows"][:9]
    support_ready = all(row.get("passed") is True for row in support_checks)
    artifact["entrance_support_audit_ready_score"] = int(support_ready)
    artifact["gate_check_summary"] = gate_summary(artifact["rows"])
    if support_ready and headroom["ready"]:
        artifact["verdict_class"] = "circular_positive"
        artifact["honest_verdict"] = "circular_positive: support and selector headroom audit ready"
    elif support_ready:
        artifact["verdict_class"] = "partial"
        artifact["honest_verdict"] = "partial: support ready but selector headroom is insufficient"
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Any) -> list[str]:
    """Cold-check schema, reductions, terminal class, and full-content checksum."""

    if not isinstance(artifact, Mapping):
        return ["artifact_object_required"]
    errors = [
        f"missing_field:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact
    ]
    if errors:
        return errors
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS) or any(
        not str(value).strip() for value in artifact.get("field_principles", {}).values()
    ):
        errors.append("field_principles_mismatch")
    blocked = artifact.get("verdict_class") == "blocked"
    expected_class = "blocked_no_run" if blocked else "no_model_load"
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("inference_substrate_class") != expected_class:
        errors.append("inference_substrate_class_mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_mismatch")
    for field in (
        "entrance_support_audit_ready_score",
        "entrance_selector_headroom_ready_score",
    ):
        if type(artifact.get(field)) is not int or artifact.get(field) not in (0, 1):
            errors.append(f"{field}_not_bare_int")
    verdict_class = str(artifact.get("verdict_class", ""))
    if verdict_class not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if not str(artifact.get("honest_verdict", "")).startswith(f"{verdict_class}:"):
        errors.append("honest_verdict_prefix_mismatch")
    preconditions = dict(artifact.get("preconditions_checked") or {})
    if blocked:
        expected_summary = gate_summary(preconditions.get("checks", []))
        if preconditions.get("all_passed") is not False:
            errors.append("blocked_precondition_mismatch")
        if artifact.get("entrance_support_audit_ready_score") != 0:
            errors.append("support_score_mismatch")
        if artifact.get("entrance_selector_headroom_ready_score") != 0:
            errors.append("headroom_score_mismatch")
        if artifact.get("gate_check_summary") != expected_summary:
            errors.append("gate_summary_mismatch")
    else:
        expected_rows = _artifact_gate_rows(artifact)
        if artifact.get("rows") != expected_rows:
            errors.append("gate_rows_mismatch")
        expected_support = int(all(row.get("passed") is True for row in expected_rows[:9]))
        expected_headroom = int(expected_rows[-1].get("passed") is True)
        if artifact.get("entrance_support_audit_ready_score") != expected_support:
            errors.append("support_score_mismatch")
        if artifact.get("entrance_selector_headroom_ready_score") != expected_headroom:
            errors.append("headroom_score_mismatch")
        if artifact.get("headroom_unit_count") != sum(
            row.get("kind") == "held_unit" and row.get("mixed_reachability") is True
            for row in artifact.get("headroom_rows", [])
        ):
            errors.append("headroom_count_mismatch")
        if artifact.get("gate_check_summary") != gate_summary(expected_rows):
            errors.append("gate_summary_mismatch")
        expected_verdict = (
            "circular_positive"
            if expected_support and expected_headroom
            else "partial"
            if expected_support
            else "null"
        )
        if verdict_class != expected_verdict:
            errors.append("verdict_class_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _fresh_worker(
    bank_path: Path, fixture_path: Path, output_path: Path
) -> int:  # pragma: no cover
    """Run the exact replay in a child and leave one private JSON result."""

    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    models, seeds = _models_and_seeds(bank)
    schema = build_required_support_schema(fixture, models, seeds)
    replay = recompute_audit(bank, fixture, schema)
    replay["_fresh_process"] = {
        "fresh_process": True,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "model_loaded": False,
        "inference_substrate_class": "no_model_load",
        "passed": not any(
            error
            for error in replay.get("errors", [])
            if error not in {"family_support_incomplete"}
        ),
    }
    output_path.write_text(json.dumps(replay, sort_keys=True), encoding="utf-8")
    return 0


def run_fresh_replay(
    bank_path: Path, fixture_path: Path, audit_dir: Path
) -> tuple[JsonDict, list[JsonDict]]:
    """Start a distinct interpreter and return its deterministic replay receipt."""

    audit_dir.mkdir(parents=True, exist_ok=False)
    output_path = audit_dir / "fresh-replay.json"
    command = [
        sys.executable,
        "-m",
        "carnot.experiment_7087_v621_entrance_bank_sufficiency_audit",
        "--fresh-worker",
        "--bank-path",
        str(bank_path),
        "--fixture-path",
        str(fixture_path),
        "--worker-output",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=audit_dir,
        env={**os.environ, "PYTHONHASHSEED": "0"},
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            f"fresh_replay_failed:returncode={completed.returncode}:stderr={completed.stderr[-500:]}"
        )
    replay = json.loads(output_path.read_text(encoding="utf-8"))
    process = dict(replay.pop("_fresh_process"))
    process.update(
        {
            "command": command,
            "returncode": completed.returncode,
            "stdout_hash": sha256_text(completed.stdout),
            "stderr_hash": sha256_text(completed.stderr),
            "child_distinct_from_parent": process.get("pid") != os.getpid(),
        }
    )
    process["passed"] = bool(process.get("passed")) and process["child_distinct_from_parent"]
    return replay, [process]


def _public_preconditions(preconditions: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    """Remove loaded 53 MB objects while preserving every measured gate row."""

    return {
        key: deepcopy(value)
        for key, value in preconditions.items()
        if key not in {"bank", "fixture"}
    }


def collect_preconditions(
    *,
    bank_path: Path,
    fixture_path: Path,
    result_path: Path,
    audit_root: Path,
) -> JsonDict:  # pragma: no cover - exercised by the dated audit.
    """Check pinned upstream bytes, cold validators, checkpoints, and private storage."""

    checks: list[JsonDict] = []
    bank: JsonDict = {}
    fixture: JsonDict = {}
    bank_hash = None
    fixture_hash = None
    bank_errors: list[str] = []
    fixture_errors: list[str] = []
    try:
        bank_hash = sha256_file(bank_path)
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
        bank_errors = producer.validate_artifact(bank)
    except (OSError, json.JSONDecodeError) as exc:
        bank_errors = [f"{type(exc).__name__}:{exc}"]
    try:
        fixture_hash = sha256_file(fixture_path)
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        exact.validate_artifact(fixture)
        fixture_errors = []
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        fixture_errors = [f"{type(exc).__name__}:{exc}"]
    checks.extend(
        [
            gate_row("entrance_proposal_bank_source_hash", PINNED_BANK_SHA256, bank_hash),
            gate_row(
                "entrance_proposal_bank_complete_score",
                1,
                bank.get("entrance_proposal_bank_complete_score"),
                bank.get("entrance_proposal_bank_complete_score") == 1 and not bank_errors,
            ),
            gate_row("entrance_fixture_source_hash", PINNED_FIXTURE_SHA256, fixture_hash),
            gate_row(
                "entrance_fixture_ready_score",
                1,
                fixture.get("entrance_fixture_ready_score"),
                fixture.get("entrance_fixture_ready_score") == 1 and not fixture_errors,
            ),
        ]
    )
    checkpoint_receipts = []
    for row in bank.get("checkpoint_rows", []):
        path = Path(str(row.get("path", "")))
        observed = None
        try:
            observed = sha256_file(path)
        except OSError:
            pass
        passed = observed == row.get("sha256")
        receipt = {
            "model_id": row.get("model_id"),
            "phase": row.get("phase"),
            "path": str(path),
            "expected_sha256": row.get("sha256"),
            "observed_sha256": observed,
            "passed": passed,
        }
        checkpoint_receipts.append(receipt)
        checks.append(
            gate_row(f"checkpoint:{row.get('model_id')}:{row.get('phase')}", True, passed)
        )
    raw_receipts = []
    raw_root = REPO_ROOT / "results/raw/experiment_7086_v621_three_family_entrance_bank"
    for phase in ("proposal", "forced_prefix"):
        phase_root = raw_root / phase
        for path in sorted(phase_root.glob("*/raw.jsonl")) if phase_root.is_dir() else []:
            readable = False
            try:
                with path.open("rb") as handle:
                    readable = bool(handle.read(1))
            except OSError:
                pass
            raw_receipts.append({"path": str(path), "readable": readable, "passed": readable})
    expected_raw_count = len(bank.get("checkpoint_rows", []))
    raw_ready = len(raw_receipts) == expected_raw_count and all(
        row["passed"] for row in raw_receipts
    )
    checks.append(gate_row("raw_checkpoint_readability", True, raw_ready))
    storage_ready = False
    storage_error = None
    try:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        audit_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="write-probe-", dir=audit_root) as probe:
            probe_path = Path(probe) / "probe"
            probe_path.write_text("ok", encoding="utf-8")
            storage_ready = probe_path.read_text(encoding="utf-8") == "ok"
    except OSError as exc:
        storage_error = f"{type(exc).__name__}:{exc}"
    checks.append(gate_row("isolated_audit_paths_writable", True, storage_ready))
    return {
        "all_passed": all(row.get("passed") is True for row in checks),
        "checks": checks,
        "bank": bank,
        "fixture": fixture,
        "bank_validation_errors": bank_errors,
        "fixture_validation_errors": fixture_errors,
        "checkpoint_receipts": checkpoint_receipts,
        "raw_checkpoint_receipts": raw_receipts,
        "storage_error": storage_error,
    }


def _source_hashes() -> JsonDict:  # pragma: no cover - hashes the live checkout.
    """Bind the artifact to reviewed instructions, specs, implementation, and tests."""

    paths = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-references.md"),
        Path("openspec/capabilities/verification/spec.md"),
        Path("python/carnot/experiment_7087_v621_entrance_bank_sufficiency_audit.py"),
        Path("tests/python/test_experiment_7087_v621_entrance_bank_sufficiency_audit.py"),
        Path("scripts/experiments/experiment_7087_v621_entrance_bank_sufficiency_audit.py"),
        BANK_PATH.relative_to(REPO_ROOT),
        FIXTURE_PATH.relative_to(REPO_ROOT),
    )
    return {path.name: sha256_file(REPO_ROOT / path) for path in paths}


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:  # pragma: no cover
    """Create the terminal artifact once so a later run cannot replace evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run(
    *,
    run_date: str = RUN_DATE,
    result_path: Path = RESULT_PATH,
    bank_path: Path = BANK_PATH,
    fixture_path: Path = FIXTURE_PATH,
    audit_root: Path = AUDIT_ROOT,
) -> JsonDict:  # pragma: no cover - required dated fresh-process execution.
    """Check preconditions, run the cold child, attack support, and write once."""

    started = time.perf_counter()
    preconditions = collect_preconditions(
        bank_path=bank_path,
        fixture_path=fixture_path,
        result_path=result_path,
        audit_root=audit_root,
    )
    source_hashes = _source_hashes()
    public_preconditions = _public_preconditions(preconditions)
    if preconditions.get("all_passed") is not True:
        artifact = build_blocked_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions=public_preconditions,
            source_artifact_hashes=source_hashes,
        )
        _write_new_json(result_path, artifact)
        return artifact
    bank = dict(preconditions["bank"])
    fixture = dict(preconditions["fixture"])
    with tempfile.TemporaryDirectory(prefix="fresh-replay-", dir=audit_root) as temporary:
        replay, process_rows = run_fresh_replay(bank_path, fixture_path, Path(temporary) / "worker")
    attacks = run_counterfactual_attacks(bank, fixture, replay["required_support_schema"])
    artifact = build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        bank=bank,
        fixture=fixture,
        preconditions=public_preconditions,
        source_artifact_hashes=source_hashes,
        replay=replay,
        fresh_process_rows=process_rows,
        counterfactual_swap_rows=attacks,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"artifact_validation_failed:{errors}")
    _write_new_json(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - command surface.
    """Run, validate, or serve the private fresh-process worker command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--bank-path", type=Path, default=BANK_PATH)
    parser.add_argument("--fixture-path", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--audit-root", type=Path, default=AUDIT_ROOT)
    parser.add_argument("--fresh-worker", action="store_true")
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.fresh_worker:
        if args.worker_output is None:
            parser.error("--fresh-worker requires --worker-output")
        return _fresh_worker(args.bank_path, args.fixture_path, args.worker_output)
    if args.validate:
        artifact = json.loads(args.result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact)
        print(canonical_json({"ok": not errors, "errors": errors}))
        return int(bool(errors))
    artifact = run(
        run_date=args.date,
        result_path=args.result_path,
        bank_path=args.bank_path,
        fixture_path=args.fixture_path,
        audit_root=args.audit_root,
    )
    errors = validate_artifact(artifact)
    print(
        canonical_json(
            {
                "result_path": str(args.result_path),
                "entrance_support_audit_ready_score": artifact[
                    "entrance_support_audit_ready_score"
                ],
                "entrance_selector_headroom_ready_score": artifact[
                    "entrance_selector_headroom_ready_score"
                ],
                "honest_verdict": artifact["honest_verdict"],
                "validation_errors": errors,
            }
        )
    )
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
