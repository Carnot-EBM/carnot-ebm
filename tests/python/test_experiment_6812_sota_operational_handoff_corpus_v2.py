"""Focused tests for the Exp6812 operational-handoff corpus.

Spec refs: REQ-CONSTRAINT-6812 and SCENARIO-CONSTRAINT-6812-*.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6812_sota_operational_handoff_corpus_v2 as exp


def _raw_candidates(scenario: dict[str, Any]) -> bytes:
    return exp.canonical_json_bytes({"candidates": scenario["reference_candidates"]})


def _models() -> list[dict[str, Any]]:
    return [
        {
            "hub_id": hub_id,
            "path": f"/models/{index}.gguf",
            "sha256": f"sha256:{index:064x}",
            "revision": f"revision-{index}",
            "quantization": "Q4_K_M",
            "role": role,
            "chat_template": "embedded_gguf",
            "embedded_tokenizer_receipt": {
                "source": "gguf_metadata",
                "auto_tokenizer_used": False,
            },
            "limits": deepcopy(exp.DECODE_BUDGETS),
        }
        for index, (hub_id, role) in enumerate(
            zip(exp.MODEL_SPECS, exp.MODEL_ROLES, strict=True), start=1
        )
    ]


def _cell(
    manifest: dict[str, Any],
    scenario: dict[str, Any],
    model_id: str,
    arm: str,
    raw: bytes | None = None,
) -> dict[str, Any]:
    prompt = scenario["prompts"][arm]
    output = raw if raw is not None else _raw_candidates(scenario)
    cell_id = exp.cell_id(model_id, scenario["scenario_id"], scenario["random_seed"], arm)
    return {
        "cell_id": cell_id,
        "model_id": model_id,
        "scenario_id": scenario["scenario_id"],
        "random_seed": scenario["random_seed"],
        "arm": arm,
        "prompt_sha256": prompt["sha256"],
        "raw_output_b64": base64.b64encode(output).decode("ascii"),
        "raw_output_len": len(output),
        "raw_output_sha256": exp.sha256_bytes(output),
        "raw_api_response_sha256": "sha256:" + "a" * 64,
        "first_token_received": bool(output),
        "generated_tokens": 12,
        "prompt_tokens": 128,
        "finish_reason": "stop",
        "retry_count": 0,
        "latency_s": 0.2,
        "manifest_sha256": manifest["manifest_sha256"],
    }


def _phase_receipts() -> list[dict[str, Any]]:
    return [
        {
            "model_id": model_id,
            "authentic": True,
            "cuda_offload": True,
            "first_token_received": True,
            "lease_owned": True,
            "lease_released": True,
            "teardown_complete": True,
            "server_pid": 9000 + index,
            "offloaded_layers": 99,
            "vram_peak_mb": 22000,
            "duration_s": {
                "acquisition": 0.1,
                "load": 1.0,
                "inference": 2.0,
                "teardown": 0.2,
            },
        }
        for index, model_id in enumerate(exp.MODEL_SPECS)
    ]


def _complete_artifact() -> dict[str, Any]:
    scenarios = exp.build_scenarios()
    manifest = exp.build_frozen_manifest(scenarios)
    cells = [
        _cell(manifest, scenario, model_id, arm)
        for model_id in exp.MODEL_SPECS
        for scenario in scenarios
        for arm in exp.ARMS
    ]
    checkpoints = [
        {
            "cell_id": cell["cell_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "row_sha256": exp.sha256_bytes(exp.canonical_json_bytes(cell)),
            "atomic": True,
            "resumed": False,
        }
        for cell in cells
    ]
    return exp.assemble_artifact(
        run_date="20260831",
        manifest=manifest,
        model_specs=_models(),
        cell_results=cells,
        gpu_receipts=_phase_receipts(),
        checkpoint_receipts=checkpoints,
        preconditions=[{"check": "fixture", "passed": True}],
        duration_s=3.3,
    )


def test_scenario_constraint_6812_frozen_pairs_are_complete_and_equal_length() -> None:
    """SCENARIO-CONSTRAINT-6812-FROZEN-PAIRS freezes every required pair."""

    scenarios = exp.build_scenarios()
    manifest = exp.build_frozen_manifest(scenarios)

    assert len(scenarios) == 48
    assert {item["family"] for item in scenarios} == set(exp.SCENARIO_FAMILIES)
    assert all(item["source_free"] is True for item in scenarios)
    assert len({item["scenario_id"] for item in scenarios}) == 48
    assert manifest["MODEL_SPECS"] == list(exp.MODEL_SPECS)
    assert manifest["planned_cell_count"] == 288
    assert manifest["planned_row_count"] == 576
    for scenario in scenarios:
        direct = scenario["prompts"]["direct_typed"]
        compressed = scenario["prompts"]["compressed_prose"]
        assert direct["byte_length"] == compressed["byte_length"]
        assert base64.b64decode(direct["bytes_b64"]) != base64.b64decode(
            compressed["bytes_b64"]
        )
        assert exp.sha256_bytes(base64.b64decode(direct["bytes_b64"])) == direct["sha256"]


def test_scenario_constraint_6812_manifest_is_deterministic_and_binds_budgets() -> None:
    """REQ-CONSTRAINT-6812 makes resume identity cover prompts and budgets."""

    first = exp.build_frozen_manifest(exp.build_scenarios())
    second = exp.build_frozen_manifest(exp.build_scenarios())

    assert first == second
    assert first["decode_budgets"] == exp.DECODE_BUDGETS
    assert len(first["expected_cell_ids"]) == len(set(first["expected_cell_ids"]))
    assert len(first["expected_row_ids"]) == len(set(first["expected_row_ids"]))
    changed = deepcopy(first)
    changed["decode_budgets"]["max_output_tokens"] += 1
    assert exp.manifest_payload_hash(changed) != first["manifest_sha256"]


def test_scenario_constraint_6812_parser_accepts_only_exact_unrepaired_json() -> None:
    """SCENARIO-CONSTRAINT-6812-EXACT-POSTCHECK parses without repair."""

    scenario = exp.build_scenarios()[0]
    parsed = exp.parse_candidate_output(_raw_candidates(scenario))

    assert parsed["parse_state"] == "complete"
    assert parsed["parse_failure"] is None
    assert parsed["candidates"] == scenario["reference_candidates"]


@pytest.mark.parametrize(
    ("raw", "failure"),
    [
        (b"", "empty_output"),
        (b"```json\n{}\n```", "json_decode_error"),
        (b'{"candidate":[]}', "invalid_top_level_fields"),
        (b'{"candidates":[]}', "invalid_candidate_count"),
        (
            b'{"candidates":[{},{}]}',
            "invalid_candidate_fields",
        ),
        (
            b'{"candidates":[{"action":{"data":null,"kind":"NOOP"},'
            b'"authority_chain":[],"candidate_id":"candidate_0","soft_progress":0},'
            b'{"action":{"data":null,"kind":"NOOP"},"authority_chain":["x"],'
            b'"candidate_id":"candidate_1","soft_progress":0}]}',
            "invalid_authority_chain",
        ),
        (
            b'{"candidates":[{"action":{},"authority_chain":["x"],'
            b'"candidate_id":"candidate_0","soft_progress":0},{}]}',
            "invalid_action",
        ),
        (
            b'{"candidates":[{"action":{"data":null,"kind":true},'
            b'"authority_chain":["x"],"candidate_id":"candidate_0",'
            b'"soft_progress":0},{}]}',
            "invalid_action",
        ),
        (
            b'{"candidates":[{"action":{"data":null,"kind":"NOOP"},'
            b'"authority_chain":["x"],"candidate_id":"wrong","soft_progress":0},{}]}',
            "invalid_candidate_id",
        ),
        (
            b'{"candidates":[{"action":{"data":null,"kind":"NOOP"},'
            b'"authority_chain":["x"],"candidate_id":"candidate_0",'
            b'"soft_progress":true},{}]}',
            "invalid_soft_progress",
        ),
    ],
)
def test_scenario_constraint_6812_parser_fails_closed(raw: bytes, failure: str) -> None:
    """REQ-CONSTRAINT-6812 retains parse failure instead of extracting text."""

    parsed = exp.parse_candidate_output(raw)

    assert parsed == {
        "parse_state": "incomplete",
        "parse_failure": failure,
        "candidates": [None, None],
    }


@pytest.mark.parametrize("family", exp.SCENARIO_FAMILIES)
def test_scenario_constraint_6812_exact_checker_covers_every_family(family: str) -> None:
    """SCENARIO-CONSTRAINT-6812-EXACT-POSTCHECK uses the Exp6811 authority."""

    scenario = next(item for item in exp.build_scenarios() if item["family"] == family)
    evidence = exp.evaluate_candidate(scenario, scenario["reference_candidates"][0])

    assert evidence["exact_checker_applied_after_generation"] is True
    assert evidence["operational_fields"] == list(exp.OPERATIONAL_FIELDS)
    assert evidence["hard_preserved"] is True
    assert evidence["binding_preserved"] is True
    assert evidence["operational_preserved"] is True
    assert evidence["retry_demand"] is False
    if family == "already_safe_proposals":
        assert evidence["already_safe_identity"] is True


def test_scenario_constraint_6812_bad_candidate_retains_exact_conflict() -> None:
    """REQ-CONSTRAINT-6812 records hard, binding, authority, and retry evidence."""

    scenario = next(
        item for item in exp.build_scenarios() if item["family"] == "competing_authorities"
    )
    candidate = deepcopy(scenario["reference_candidates"][0])
    candidate["action"] = {"data": {"variant": 999}, "kind": "IGNORE"}
    evidence = exp.evaluate_candidate(scenario, candidate)

    assert evidence["operational_preserved"] is False
    assert evidence["hard_violation_count"] > 0
    assert evidence["binding_preserved"] is False
    assert evidence["legal_support"] is False
    assert evidence["retry_demand"] is True
    assert evidence["conflict_certificates"]


def test_scenario_constraint_6812_cell_rows_keep_parse_failure_slots() -> None:
    """REQ-CONSTRAINT-6812 emits both planned candidate rows on parse failure."""

    scenario = exp.build_scenarios()[0]
    manifest = exp.build_frozen_manifest(exp.build_scenarios())
    cell = _cell(manifest, scenario, exp.MODEL_SPECS[0], exp.ARMS[0], b"not json")
    rows, raw_receipt = exp.build_cell_rows(manifest, scenario, cell)

    assert len(rows) == exp.DECODE_BUDGETS["candidate_count"] == 2
    assert [row["candidate_index"] for row in rows] == [0, 1]
    assert all(row["parse_state"] == "incomplete" for row in rows)
    assert all(row["operational_preserved"] is False for row in rows)
    assert all(row["retry_count"] == 0 for row in rows)
    assert raw_receipt["raw_output_b64"] == cell["raw_output_b64"]


def test_scenario_constraint_6812_cell_rejects_mutated_prompt_or_output_bytes() -> None:
    """REQ-CONSTRAINT-6812 binds each cell to frozen and immutable bytes."""

    scenarios = exp.build_scenarios()
    scenario = scenarios[0]
    manifest = exp.build_frozen_manifest(scenarios)
    cell = _cell(manifest, scenario, exp.MODEL_SPECS[0], exp.ARMS[0])

    broken_prompt = deepcopy(cell)
    broken_prompt["prompt_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="prompt hash"):
        exp.build_cell_rows(manifest, scenario, broken_prompt)

    broken_output = deepcopy(cell)
    broken_output["raw_output_len"] += 1
    with pytest.raises(ValueError, match="raw output"):
        exp.build_cell_rows(manifest, scenario, broken_output)

    broken_manifest = deepcopy(cell)
    broken_manifest["manifest_sha256"] = "sha256:" + "1" * 64
    with pytest.raises(ValueError, match="manifest hash"):
        exp.build_cell_rows(manifest, scenario, broken_manifest)

    broken_base64 = deepcopy(cell)
    broken_base64["raw_output_b64"] = "%%%"
    with pytest.raises(ValueError, match="base64"):
        exp.build_cell_rows(manifest, scenario, broken_base64)


def test_scenario_constraint_6812_complete_artifact_is_row_derived() -> None:
    """SCENARIO-CONSTRAINT-6812-RESUME-AND-READINESS derives all metrics."""

    artifact = _complete_artifact()

    exp.validate_artifact(artifact)
    assert artifact["operational_handoff_corpus_ready"] is True
    assert artifact["live_model_invoked"] is True
    assert artifact["models_used"] == list(exp.MODEL_SPECS)
    assert len(artifact["rows"]) == artifact["frozen_manifest"]["planned_row_count"]
    assert set(artifact["operational_preservation_by_arm"]) == set(exp.ARMS)
    assert artifact["solve_claim"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] in exp.VERDICT_CLASSES


def test_scenario_constraint_6812_readiness_does_not_depend_on_effect_sign() -> None:
    """REQ-CONSTRAINT-6812 keeps completeness independent of the arm effect."""

    artifact = _complete_artifact()
    direct = artifact["operational_preservation_by_arm"]["direct_typed"]["rate"]
    compressed = artifact["operational_preservation_by_arm"]["compressed_prose"]["rate"]

    assert direct == compressed
    assert artifact["operational_handoff_corpus_ready"] is True
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_constraint_6812_validator_rejects_missing_model_and_bad_metric() -> None:
    """REQ-CONSTRAINT-6812 never pools an incomplete model phase."""

    missing_model = _complete_artifact()
    missing_model["models_used"].pop()
    with pytest.raises(ValueError, match="models_used"):
        exp.validate_artifact(missing_model)

    bad_metric = _complete_artifact()
    bad_metric["hard_violation_rate_by_arm"][exp.ARMS[0]]["rate"] = 0.5
    with pytest.raises(ValueError, match="row-derived"):
        exp.validate_artifact(bad_metric)


def test_scenario_constraint_6812_partial_and_defensive_validation_paths() -> None:
    """REQ-CONSTRAINT-6812 keeps every structural gate fail closed."""

    complete = _complete_artifact()
    partial = exp.assemble_artifact(
        run_date="20260831",
        manifest=complete["frozen_manifest"],
        model_specs=complete["model_specs"],
        cell_results=complete["raw_output_manifest"][:-1],
        gpu_receipts=complete["gpu_receipts"],
        checkpoint_receipts=complete["checkpoint_receipts"],
        preconditions=complete["preconditions_checked"],
        duration_s=1.0,
    )
    exp.validate_artifact(partial)
    assert partial["status"] == "complete_partial"
    assert partial["verdict_class"] == "partial"

    unknown = deepcopy(complete["raw_output_manifest"][0])
    unknown["scenario_id"] = "unknown"
    with pytest.raises(ValueError, match="scenario"):
        exp.assemble_artifact(
            run_date="20260831",
            manifest=complete["frozen_manifest"],
            model_specs=complete["model_specs"],
            cell_results=[unknown],
            gpu_receipts=[],
            checkpoint_receipts=[],
            preconditions=[],
            duration_s=0.0,
        )

    mutations = [
        (lambda value: value.pop("title"), "field set"),
        (lambda value: value["field_principles"].update({"title": "changed"}), "principles"),
        (lambda value: value.update({"MODEL_SPECS": []}), "MODEL_SPECS"),
        (lambda value: value.update({"inference_substrate": "provider"}), "substrate"),
        (lambda value: value.update({"solve_claim": True}), "claim boundary"),
        (lambda value: value.update({"verdict_class": "unknown"}), "closed enum"),
        (
            lambda value: value["frozen_manifest"].update(
                {"manifest_sha256": "sha256:" + "0" * 64}
            ),
            "manifest checksum",
        ),
        (lambda value: value.update({"reproducibility_checksum": "bad"}), "reproducibility"),
        (lambda value: value.update({"operational_handoff_corpus_ready": False}), "readiness"),
        (
            lambda value: value["gate_check_summary"].update({"rows_complete": False}),
            "gate check",
        ),
    ]
    for mutation, message in mutations:
        changed = deepcopy(complete)
        mutation(changed)
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(changed)


def test_scenario_constraint_6812_raw_receipt_and_phase_helpers_fail_closed() -> None:
    """REQ-CONSTRAINT-6812 validates raw bytes and bounded phase diagnostics."""

    assert exp._raw_receipt_valid({}) is False
    error = exp.ModelPhaseError("model_load", "failed", {"model_id": "m"})
    assert error.check == "model_load"
    assert error.observed == "failed"
    assert error.receipt == {"model_id": "m"}
    command = exp._server_command(Path("/llama-server"), {"path": "/model.gguf"}, 8123)
    assert command[:3] == ["/llama-server", "--model", "/model.gguf"]
    assert command[command.index("--port") + 1] == "8123"
    assert exp._first_failed([{"passed": True}, {"check": "x", "passed": False}]) == {
        "check": "x",
        "passed": False,
    }
    assert exp._first_failed([{"passed": True}]) is None


def test_scenario_constraint_6812_blocked_artifact_names_gate_and_stops() -> None:
    """REQ-CONSTRAINT-6812 writes the mandated bounded-load terminal state."""

    artifact = exp.build_blocked_artifact(
        run_date="20260831",
        failed_check="model_load",
        expected="healthy llama.cpp CUDA phase",
        observed="server exited before health",
        preconditions=[{"check": "model_load", "passed": False}],
        model_specs=_models(),
        duration_s=1.25,
    )

    exp.validate_artifact(artifact)
    assert artifact["status"] == "complete_blocked_sota_operational_handoff_corpus_v2"
    assert artifact["operational_handoff_corpus_ready"] is False
    assert artifact["rows"] == []
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == "model_load"

    with_rows = deepcopy(artifact)
    with_rows["rows"] = [{}]
    with pytest.raises(ValueError, match="headline rows"):
        exp.validate_artifact(with_rows)
    with_model = deepcopy(artifact)
    with_model["models_used"] = [exp.MODEL_SPECS[0]]
    with pytest.raises(ValueError, match="completed models"):
        exp.validate_artifact(with_model)
    bad_repro = deepcopy(artifact)
    bad_repro["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="reproducibility"):
        exp.validate_artifact(bad_repro)


def test_scenario_constraint_6812_atomic_writer_replaces_complete_json(tmp_path: Path) -> None:
    """SCENARIO-CONSTRAINT-6812-RESUME-AND-READINESS leaves no partial artifact."""

    path = tmp_path / "artifact.json"
    payload = {"ready": True, "rows": [1, 2]}

    exp.atomic_write_artifact(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert list(tmp_path.glob("*.tmp")) == []
