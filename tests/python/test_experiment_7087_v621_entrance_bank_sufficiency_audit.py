"""Tests for the cold V621 entrance-bank support audit.

Spec refs: REQ-VERIFY-7087 and SCENARIO-VERIFY-7087-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7064_v619_exact_entrance_fixture as exact
from carnot import experiment_7086_v621_three_family_entrance_bank as producer
from carnot import experiment_7087_v621_entrance_bank_sufficiency_audit as audit


MODELS = ("model-a", "model-b")
SEEDS = (11, 12)
OPERATORS = ("+", "-", "*", "/")


def _fixture() -> dict:
    units = []
    entrances = []
    plans = (("+", 6), ("-", 4), ("*", 6), ("/", 5))
    for group_index, split in enumerate(("calibration", "held")):
        group_id = f"group-{group_index}"
        for unit_index, (_operator, target) in enumerate(plans):
            unit = {
                "unit_id": f"unit-{group_index}-{unit_index}",
                "source_group_id": group_id,
                "split": split,
                "numbers": [1, 2, 3],
                "target": target,
                "formatting_rules": ["Return one compact JSON object."],
            }
            units.append(unit)
            for row in exact.label_unit_entrances(unit):
                entrances.append({**row, "source_group_id": group_id, "split": split})
    return {
        "entrance_fixture_ready_score": 1,
        "unit_rows": units,
        "entrance_rows": entrances,
        "split_manifest": {
            "held_source_group_ids": ["group-1"],
            "calibration_source_group_ids": ["group-0"],
        },
    }


def _entrance_id(fixture: dict, unit_id: str, operator: str) -> str:
    return next(
        row["entrance_id"]
        for row in fixture["entrance_rows"]
        if row["unit_id"] == unit_id
        and row["operator"] == operator
        and row["operand_pair"] == [1, 2]
    )


def _bank(fixture: dict | None = None) -> dict:
    fixture = fixture or _fixture()
    raw_rows = []
    parse_rows = []
    label_rows = []
    witness_rows = []
    seen: set[tuple[str, str, str]] = set()
    template_hashes = {model: audit.sha256_text(f"template:{model}") for model in MODELS}
    for model in MODELS:
        for seed in SEEDS:
            for unit in fixture["unit_rows"]:
                operator = OPERATORS[int(unit["unit_id"].rsplit("-", 1)[1])]
                raw_text = json.dumps(
                    {"operand_pair": [1, 2], "operator": operator},
                    separators=(",", ":"),
                )
                raw_hash = audit.sha256_text(raw_text)
                raw_key = f"{model}|proposal|{unit['unit_id']}|{seed}"
                prompt = producer.build_proposal_prompt(unit)
                messages = producer.build_role_messages(prompt)
                raw_rows.append(
                    {
                        "raw_key": raw_key,
                        "arm": "proposal",
                        "model_id": model,
                        "unit_id": unit["unit_id"],
                        "seed": seed,
                        "raw_text": raw_text,
                        "raw_bytes_hex": raw_text.encode().hex(),
                        "raw_output_hash": raw_hash,
                        "prompt": prompt,
                        "prompt_hash": audit.sha256_text(prompt),
                        "role_messages": messages,
                        "role_messages_hash": audit.sha256_text(audit.canonical_json(messages)),
                        "generation_config": deepcopy(producer.GENERATION_CONFIG),
                        "stop_config": deepcopy(producer.GENERATION_CONFIG["stop"]),
                        "requested_completion_budget_tokens": 192,
                        "effective_completion_budget_tokens": 192,
                        "prefix_token_count": 0,
                        "chat_template_present": True,
                        "chat_template_hash": template_hashes[model],
                        "chat_format": "chat_template.default",
                        "transport_method": "create_chat_completion",
                        "rendered_prompt_hash": audit.sha256_text(f"rendered:{raw_key}"),
                        "terminal_state": "complete",
                        "raw_persisted_before_parse": True,
                    }
                )
                parsed = {"operand_pair": [1, 2], "operator": operator}
                parse_rows.append(
                    {
                        "raw_key": raw_key,
                        "raw_output_hash": raw_hash,
                        "model_id": model,
                        "unit_id": unit["unit_id"],
                        "seed": seed,
                        "parsed_entrance": parsed,
                        "parse_failure": False,
                    }
                )
                entrance_id = _entrance_id(fixture, unit["unit_id"], operator)
                source = next(
                    row for row in fixture["entrance_rows"] if row["entrance_id"] == entrance_id
                )
                duplicate_key = (model, unit["unit_id"], entrance_id)
                label_rows.append(
                    {
                        "raw_key": raw_key,
                        "raw_output_hash": raw_hash,
                        "model_id": model,
                        "unit_id": unit["unit_id"],
                        "seed": seed,
                        "entrance_id": entrance_id,
                        "legal": True,
                        "reachable": source["reachable"],
                        "duplicate": duplicate_key in seen,
                        "label_source": "experiment_7064_exhaustive_enumerator",
                    }
                )
                witness_rows.append(
                    {
                        "raw_key": raw_key,
                        "raw_output_hash": raw_hash,
                        "model_id": model,
                        "unit_id": unit["unit_id"],
                        "seed": seed,
                        "entrance_id": entrance_id,
                        "causal_witness": source["reachable"],
                        "witness_source": (
                            "experiment_7064_exhaustive_enumerator" if source["reachable"] else None
                        ),
                    }
                )
                seen.add(duplicate_key)
    identity_rows = [
        {
            "model_id": model,
            "passed": True,
            "identity_matches": True,
            "chat_template_present": True,
            "chat_template_hash": template_hashes[model],
            "tokenizer_source": "embedded_gguf",
        }
        for model in MODELS
    ]
    return {
        "entrance_proposal_bank_complete_score": 1,
        "model_specs": [
            {"hf_id": model, "model_path": f"/models/{model}.gguf"} for model in MODELS
        ],
        "sampling_config": deepcopy(producer.GENERATION_CONFIG),
        "raw_proposal_rows": raw_rows,
        "parse_rows": parse_rows,
        "exact_label_rows": label_rows,
        "causal_witness_rows": witness_rows,
        "model_identity_rows": identity_rows,
        "model_file_hash_rows": [
            {"model_id": model, "path": f"/models/{model}.gguf", "sha256": "sha256:" + "1" * 64}
            for model in MODELS
        ],
        "runner_receipt": {
            "backend": "llama_cpp.Llama",
            "cuda_offload": True,
            "transport_method": "create_chat_completion",
        },
        "model_execution_rows": [
            {
                "model_id": model,
                "phase": "proposal",
                "terminal_state": "complete",
                "offloaded_layers": 1,
                "used_both_gpus": True,
                "model_load_count": 1,
            }
            for model in MODELS
        ],
        "stage_gpu_telemetry_rows": [
            {"model_id": model, "phase": "proposal", "used_memory_mb": 1} for model in MODELS
        ],
        "task_gpu_telemetry_rows": [
            {"model_id": model, "phase": "proposal", "used_memory_mb": 1} for model in MODELS
        ],
        "gpu_lease_rows": [
            {
                "model_id": model,
                "phase": "proposal",
                "lease_id": f"lease:{model}",
                "owner_preserved": True,
                "released": True,
                "lease_lost": False,
            }
            for model in MODELS
        ],
        "vram_release_rows": [
            {"model_id": model, "phase": "proposal", "passed": True} for model in MODELS
        ],
        "cleanup_rows": [
            {"model_id": model, "phase": "proposal", "passed": True} for model in MODELS
        ],
        "checkpoint_rows": [
            {
                "model_id": model,
                "phase": "proposal",
                "path": f"/checkpoints/{model}.json",
                "sha256": "sha256:" + "3" * 64,
                "manifest_hash": "sha256:" + "4" * 64,
                "row_count": len(SEEDS) * len(fixture["unit_rows"]),
            }
            for model in MODELS
        ],
        "upstream_gate_rows": [],
        "cited_upstream_artifacts": [],
    }


def _replay() -> tuple[dict, dict, dict]:
    fixture = _fixture()
    bank = _bank(fixture)
    schema = audit.build_required_support_schema(fixture, MODELS, SEEDS)
    return fixture, bank, audit.recompute_audit(bank, fixture, schema)


def test_required_schema_is_frozen_before_outcomes() -> None:
    """REQ-VERIFY-7087 fixes all source/model/seed/operator cells first."""

    fixture = _fixture()
    schema = audit.build_required_support_schema(fixture, MODELS, SEEDS)
    assert schema["frozen_before_proposal_outcomes"] is True
    assert schema["entrance_families"] == list(OPERATORS)
    assert len(schema["cells"]) == 2 * 2 * 2 * 4
    assert all(row["applicable"] for row in schema["cells"])


@pytest.mark.parametrize("kind", ["model", "unit", "seed", "source_group", "template"])
def test_missing_primary_dimensions_fail(kind: str) -> None:
    """SCENARIO-VERIFY-7087-COVERAGE rejects every required missing dimension."""

    fixture = _fixture()
    bank = _bank(fixture)
    if kind == "model":
        bank["raw_proposal_rows"] = [
            row for row in bank["raw_proposal_rows"] if row["model_id"] != MODELS[0]
        ]
    elif kind == "unit":
        bank["raw_proposal_rows"] = [
            row for row in bank["raw_proposal_rows"] if row["unit_id"] != "unit-0-0"
        ]
    elif kind == "seed":
        bank["raw_proposal_rows"] = [
            row for row in bank["raw_proposal_rows"] if row["seed"] != SEEDS[0]
        ]
    elif kind == "source_group":
        group_units = {
            row["unit_id"] for row in fixture["unit_rows"] if row["source_group_id"] == "group-0"
        }
        bank["raw_proposal_rows"] = [
            row for row in bank["raw_proposal_rows"] if row["unit_id"] not in group_units
        ]
    else:
        bank["raw_proposal_rows"][0]["chat_template_present"] = False
    schema = audit.build_required_support_schema(fixture, MODELS, SEEDS)
    replay = audit.recompute_audit(bank, fixture, schema)
    assert replay["authenticity_passed"] is False
    assert replay["errors"]


def test_raw_byte_mutation_fails_even_with_unchanged_labels() -> None:
    """SCENARIO-VERIFY-7087-AUTHENTICITY hashes bytes instead of summaries."""

    fixture = _fixture()
    bank = _bank(fixture)
    bank["raw_proposal_rows"][0]["raw_bytes_hex"] = b"mutated".hex()
    replay = audit.recompute_audit(
        bank, fixture, audit.build_required_support_schema(fixture, MODELS, SEEDS)
    )
    assert replay["raw_hash_rows"][0]["passed"] is False
    assert "raw_byte_mismatch" in replay["errors"]


def test_family_deletion_is_not_hidden_by_pooled_rate() -> None:
    """SCENARIO-VERIFY-7087-COVERAGE treats support as a set property."""

    fixture = _fixture()
    bank = _bank(fixture)
    deleted_units = {
        row["unit_id"]
        for row in fixture["unit_rows"]
        if row["source_group_id"] == "group-0" and row["unit_id"].endswith("-3")
    }
    bank["raw_proposal_rows"] = [
        row
        for row in bank["raw_proposal_rows"]
        if not (
            row["model_id"] == MODELS[0]
            and row["seed"] == SEEDS[0]
            and row["unit_id"] in deleted_units
        )
    ]
    replay = audit.recompute_audit(
        bank, fixture, audit.build_required_support_schema(fixture, MODELS, SEEDS)
    )
    target = next(
        row
        for row in replay["family_support_rows"]
        if row["source_group_id"] == "group-0"
        and row["model_id"] == MODELS[0]
        and row["seed"] == SEEDS[0]
        and row["entrance_family"] == "/"
    )
    assert target["status"] == "missing"
    assert replay["family_sufficiency_passed"] is False


def test_source_swap_and_duplicate_support_are_explicit() -> None:
    """SCENARIO-VERIFY-7087-CONFLICTS does not pool swapped or copied rows."""

    fixture = _fixture()
    bank = _bank(fixture)
    first, second = bank["raw_proposal_rows"][0], bank["raw_proposal_rows"][-1]
    first["unit_id"], second["unit_id"] = second["unit_id"], first["unit_id"]
    replay = audit.recompute_audit(
        bank, fixture, audit.build_required_support_schema(fixture, MODELS, SEEDS)
    )
    assert "raw_key_identity_mismatch" in replay["errors"]

    bank = _bank(fixture)
    bank["raw_proposal_rows"].append(deepcopy(bank["raw_proposal_rows"][0]))
    replay = audit.recompute_audit(
        bank, fixture, audit.build_required_support_schema(fixture, MODELS, SEEDS)
    )
    assert any(row["status"] == "duplicate" for row in replay["family_support_rows"])
    assert replay["family_sufficiency_passed"] is False


def test_conflicting_producer_labels_are_resolved_and_fail_closed() -> None:
    """SCENARIO-VERIFY-7087-CONFLICTS records rejected conflicting labels."""

    fixture = _fixture()
    bank = _bank(fixture)
    conflict = deepcopy(bank["exact_label_rows"][0])
    conflict["reachable"] = not conflict["reachable"]
    bank["exact_label_rows"].append(conflict)
    replay = audit.recompute_audit(
        bank, fixture, audit.build_required_support_schema(fixture, MODELS, SEEDS)
    )
    assert replay["conflict_rows"][0]["resolved_by"] == "independent_exact_replay"
    assert replay["conflict_rows"][0]["passed"] is False
    assert replay["conflict_resolution_passed"] is False


def test_parse_labels_and_causal_witnesses_recompute() -> None:
    """REQ-VERIFY-7087 independently replays parse, label, and witness rows."""

    _fixture_value, _bank_value, replay = _replay()
    assert replay["errors"] == []
    assert all(row["passed"] for row in replay["parse_recomputation_rows"])
    assert all(row["passed"] for row in replay["label_recomputation_rows"])
    assert all(row["passed"] for row in replay["causal_witness_replay_rows"])


def test_future_label_leakage_fails() -> None:
    """SCENARIO-VERIFY-7087-LEAKAGE rejects solver fields in model input."""

    fixture = _fixture()
    bank = _bank(fixture)
    bank["raw_proposal_rows"][0]["generation_config"]["future_label"] = {"reachable": True}
    replay = audit.recompute_audit(
        bank, fixture, audit.build_required_support_schema(fixture, MODELS, SEEDS)
    )
    assert replay["leakage_attack_rows"][0]["attack_detected"] is True
    assert replay["leakage_passed"] is False


def test_headroom_requires_thirty_mixed_units_and_imperfect_controls() -> None:
    """SCENARIO-VERIFY-7087-HEADROOM measures diversity without a selector."""

    labels = []
    units = []
    for index in range(30):
        unit_id = f"held-{index}"
        units.append({"unit_id": unit_id, "source_group_id": "held", "split": "held"})
        labels.extend(
            [
                {"unit_id": unit_id, "model_id": "a", "seed": 1, "reachable": True},
                {"unit_id": unit_id, "model_id": "b", "seed": 1, "reachable": False},
            ]
        )
    result = audit.measure_headroom(labels, {"unit_rows": units}, minimum_units=30)
    assert result["headroom_unit_count"] == 30
    assert result["no_perfect_base_control"] is True
    assert result["ready"] is True
    labels = [dict(row, reachable=True) if row["model_id"] == "b" else row for row in labels]
    assert audit.measure_headroom(labels, {"unit_rows": units}, minimum_units=30)["ready"] is False


def test_counterfactual_attacks_detect_swaps_mutation_and_conflict() -> None:
    """SCENARIO-VERIFY-7087-ATTACKS preserves no pooled-score escape hatch."""

    fixture, bank, replay = _replay()
    rows = audit.run_counterfactual_attacks(bank, fixture, replay["required_support_schema"])
    assert {row["attack"] for row in rows} == {
        "family_deletion",
        "source_swap",
        "raw_byte_mutation",
        "label_conflict",
    }
    assert all(row["attack_detected"] for row in rows)
    assert all(
        row["pooled_row_count_preserved"] for row in rows if row["attack"] != "family_deletion"
    )


def test_aggregate_recomputation_rejects_forged_score_and_checksum() -> None:
    """SCENARIO-VERIFY-7087-ATTACKS cold validation recomputes aggregates."""

    fixture, bank, replay = _replay()
    artifact = audit.build_artifact(
        run_date="20260906",
        duration_s=1.25,
        bank=bank,
        fixture=fixture,
        preconditions={"all_passed": True, "checks": []},
        source_artifact_hashes={"bank": "sha256:" + "2" * 64},
        replay=replay,
        fresh_process_rows=[{"passed": True, "fresh_process": True}],
        minimum_headroom_units=1,
        counterfactual_swap_rows=[],
    )
    assert audit.validate_artifact(artifact) == []
    forged = deepcopy(artifact)
    forged["entrance_support_audit_ready_score"] ^= 1
    assert "support_score_mismatch" in audit.validate_artifact(forged)
    forged = deepcopy(artifact)
    forged["rows"][0]["passed"] = not forged["rows"][0]["passed"]
    assert "gate_rows_mismatch" in audit.validate_artifact(forged)
    forged = deepcopy(artifact)
    forged["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum_mismatch" in audit.validate_artifact(forged)


def test_blocked_artifact_has_exact_diagnostics() -> None:
    """REQ-VERIFY-7087 emits a terminal and schema-complete preflight block."""

    checks = [audit.gate_row("upstream_score", 1, 0, False)]
    artifact = audit.build_blocked_artifact(
        run_date="20260906",
        duration_s=0.1,
        preconditions={"all_passed": False, "checks": checks},
        source_artifact_hashes={},
    )
    assert set(audit.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"] == {
        "passed": False,
        "failed_check": "upstream_score",
        "expected_value": 1,
        "observed_value": 0,
        "checks": checks,
    }
    assert audit.validate_artifact(artifact) == []


def test_fresh_worker_uses_a_distinct_process(tmp_path: Path) -> None:
    """REQ-VERIFY-7087 performs deterministic replay in a child process."""

    fixture = _fixture()
    bank = _bank(fixture)
    bank_path = tmp_path / "bank.json"
    fixture_path = tmp_path / "fixture.json"
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    replay, rows = audit.run_fresh_replay(bank_path, fixture_path, tmp_path / "audit")
    assert replay["errors"] == []
    assert rows[0]["fresh_process"] is True
    assert rows[0]["passed"] is True
