"""Tests for Exp6294 matched ARC mechanic-router causal canary.

Spec refs: REQ-ARC-WMTE-6294,
SCENARIO-ARC-WMTE-6294-MATCHED-ARMS,
SCENARIO-ARC-WMTE-6294-PROPOSAL-PATH-METRICS,
SCENARIO-ARC-WMTE-6294-PROVENANCE-GUARDS.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_6294_arc_mechanic_router_causal_canary as exp6294


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def test_req_arc_wmte_6294_spec_declares_matched_causal_contract() -> None:
    """REQ-ARC-WMTE-6294: OpenSpec names the matched canary and artifact fields."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-ARC-WMTE-6294") :]
    for marker in (
        "SCENARIO-ARC-WMTE-6294-MATCHED-ARMS",
        "SCENARIO-ARC-WMTE-6294-PROPOSAL-PATH-METRICS",
        "SCENARIO-ARC-WMTE-6294-PROVENANCE-GUARDS",
        exp6294.RESULT_RELATIVE_PATH.as_posix(),
        "proposal routing only",
        "at least 60 seconds of actual measured work",
    ):
        assert marker in section
    for model_id in exp6294.MANDATED_MODEL_IDS:
        assert model_id in section
    for field in exp6294.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_arc_wmte_6294_fresh_fixture_and_window_seals(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6294-MATCHED-ARMS: fixtures and windows are sealed before arms."""

    fixtures = exp6294.build_fresh_fixtures(seed=6294, per_mechanic=2)
    fixture_payload = exp6294.fixture_manifest_payload(fixtures, seed=6294)
    windows = exp6294.build_live_transition_windows(fixtures, seeds=(101, 202))
    window_payload = exp6294.live_window_manifest_payload(windows)

    fixture_receipt = exp6294.write_manifest(
        tmp_path / "fixtures.json", fixture_payload, write=True
    )
    window_receipt = exp6294.write_manifest(tmp_path / "windows.json", window_payload, write=True)

    assert fixture_payload["family_counts"] == {"push_block": 2, "toggle_move": 2}
    assert fixture_payload["freshness"] == "exp6294_seeded_synthetic_not_exp6282_reuse"
    assert all(row["game_id"] is None for row in fixture_payload["fixtures"])
    assert all(row["hidden_source_used"] is False for row in window_payload["windows"])
    assert fixture_receipt["sha256"] == exp6294.sha256_json(fixture_payload)
    assert window_receipt["sha256"] == exp6294.sha256_json(window_payload)


def test_scenario_arc_wmte_6294_matched_arm_contract() -> None:
    """SCENARIO-ARC-WMTE-6294-MATCHED-ARMS: only the route block differs."""

    fixtures = exp6294.build_fresh_fixtures(seed=6294, per_mechanic=1)
    windows = exp6294.build_live_transition_windows(fixtures, seeds=(303,))
    models = [
        {
            "hf_id": exp6294.MANDATED_MODEL_IDS[0],
            "name": "Qwen3.6-35B-A3B",
            "model_path": "/tmp/qwen.gguf",
            "quantization": "Q4_K_M",
        },
        {
            "hf_id": exp6294.MANDATED_MODEL_IDS[1],
            "name": "Gemma4-31B-it",
            "model_path": "/tmp/gemma.gguf",
            "quantization": "Q4_K_M",
        },
    ]

    requests = exp6294.build_matched_arm_requests(windows, models)
    exp6294.validate_matched_requests(requests)

    assert len(requests) == 8
    for pair in exp6294.group_requests_by_pair(requests).values():
        off = next(row for row in pair if row["arm"] == "router_off")
        on = next(row for row in pair if row["arm"] == "router_on")
        for key in ("fixture_id", "seed", "action_budget", "model_budget_tokens", "model_id"):
            assert off[key] == on[key]
        assert off["starting_history_hash"] == on["starting_history_hash"]
        assert off["route_block_present"] is False
        assert on["route_block_present"] is True


def test_scenario_arc_wmte_6294_artifact_schema_and_metrics(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6294-PROPOSAL-PATH-METRICS: artifact records positive causal delta."""

    artifact = exp6294.run(
        date="20260811",
        result_path=tmp_path / exp6294.RESULT_RELATIVE_PATH.name,
        fixture_manifest_path=tmp_path / exp6294.FIXTURE_MANIFEST_RELATIVE_PATH.name,
        live_window_manifest_path=tmp_path / exp6294.LIVE_WINDOW_MANIFEST_RELATIVE_PATH.name,
        raw_output_dir=tmp_path / "raw",
        duration_s=65.25,
        test_exit_codes={exp6294.RUN_COMMAND: 0},
        model_resolver=exp6294.deterministic_test_model_resolver,
        llm_runner=exp6294.deterministic_test_llm_runner,
        write=True,
    )

    loaded = json.loads((tmp_path / exp6294.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))
    assert loaded == artifact
    assert set(exp6294.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp6294.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_principles"])
    assert set(exp6294.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_provenance"])
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["MODEL_SPECS"][0]["hf_id"] == exp6294.MANDATED_MODEL_IDS[0]
    assert artifact["MODEL_SPECS"][1]["hf_id"] == exp6294.MANDATED_MODEL_IDS[1]
    assert artifact["models_used"] == list(exp6294.MANDATED_MODEL_IDS)
    assert artifact["duration_padding_count"] == 0
    assert artifact["actual_work_duration_receipt"]["measured_actual_work_s"] == 65.25
    assert (
        artifact["paired_causal_deltas_intervals_and_sample_sizes"]["ready_delta_positive"] is True
    )
    assert artifact["baseline_harm_controls"]["baseline_harm_detected"] is False
    assert artifact["arc_mechanic_causal_ready_score"] > 0.0
    assert artifact["reproducibility_checksum"] == exp6294.payload_checksum(artifact)
    assert artifact["raw_output_paths_and_hashes"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hidden_game_source_access_count", 1, "hidden_game_source_access_count"),
        ("outer_loop_ground_truth_search_count", 1, "outer_loop_ground_truth_search_count"),
        ("arc_level_solve_claim_count", 1, "arc_level_solve_claim_count"),
        ("registry_update_count", 1, "registry_update_count"),
        ("source_model_weight_mutation_count", 1, "source_model_weight_mutation_count"),
        ("duration_padding_count", 1, "duration_padding_count"),
        ("solve_provenance", "development_proxy", "solve_provenance"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
    ],
)
def test_scenario_arc_wmte_6294_provenance_guard_rejects_forbidden_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    """SCENARIO-ARC-WMTE-6294-PROVENANCE-GUARDS: forbidden counters stay bare zero."""

    artifact = exp6294.run(
        date="20260811",
        result_path=tmp_path / "artifact.json",
        fixture_manifest_path=tmp_path / "fixtures.json",
        live_window_manifest_path=tmp_path / "windows.json",
        raw_output_dir=tmp_path / "raw",
        duration_s=65.25,
        model_resolver=exp6294.deterministic_test_model_resolver,
        llm_runner=exp6294.deterministic_test_llm_runner,
        write=False,
    )
    bad = dict(artifact)
    bad[field] = value
    bad["reproducibility_checksum"] = exp6294.payload_checksum(bad)

    with pytest.raises(ValueError, match=message):
        exp6294.validate_artifact(bad)


def test_scenario_arc_wmte_6294_validation_rejects_model_substitution(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6294-PROVENANCE-GUARDS: both mandated model ids are required."""

    artifact = exp6294.run(
        date="20260811",
        result_path=tmp_path / "artifact.json",
        fixture_manifest_path=tmp_path / "fixtures.json",
        live_window_manifest_path=tmp_path / "windows.json",
        raw_output_dir=tmp_path / "raw",
        duration_s=65.25,
        model_resolver=exp6294.deterministic_test_model_resolver,
        llm_runner=exp6294.deterministic_test_llm_runner,
        write=False,
    )
    bad = dict(artifact)
    bad["MODEL_SPECS"] = [dict(bad["MODEL_SPECS"][0])]
    bad["models_used"] = [exp6294.MANDATED_MODEL_IDS[0]]
    bad["reproducibility_checksum"] = exp6294.payload_checksum(bad)

    with pytest.raises(ValueError, match="MODEL_SPECS"):
        exp6294.validate_artifact(bad)


def test_scenario_arc_wmte_6294_validation_rejects_arm_history_and_harm_drift(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6294-PROVENANCE-GUARDS: matching and harm gates fail closed."""

    artifact = exp6294.run(
        date="20260811",
        result_path=tmp_path / "artifact.json",
        fixture_manifest_path=tmp_path / "fixtures.json",
        live_window_manifest_path=tmp_path / "windows.json",
        raw_output_dir=tmp_path / "raw",
        duration_s=65.25,
        model_resolver=exp6294.deterministic_test_model_resolver,
        llm_runner=exp6294.deterministic_test_llm_runner,
        write=False,
    )

    bad_arm = dict(artifact)
    bad_arm["matched_router_off_and_on_arm_contract"] = {
        **artifact["matched_router_off_and_on_arm_contract"],
        "all_pairs_matched": False,
    }
    bad_arm["reproducibility_checksum"] = exp6294.payload_checksum(bad_arm)
    with pytest.raises(ValueError, match="matched_router_off_and_on_arm_contract"):
        exp6294.validate_artifact(bad_arm)

    bad_history = dict(artifact)
    bad_history["matched_fixture_seed_action_model_and_history_receipts"] = {
        **artifact["matched_fixture_seed_action_model_and_history_receipts"],
        "history_mismatch_count": 1,
    }
    bad_history["reproducibility_checksum"] = exp6294.payload_checksum(bad_history)
    with pytest.raises(ValueError, match="matched_fixture_seed_action_model_and_history_receipts"):
        exp6294.validate_artifact(bad_history)

    bad_harm = dict(artifact)
    bad_harm["baseline_harm_controls"] = {
        **artifact["baseline_harm_controls"],
        "baseline_harm_detected": True,
    }
    bad_harm["reproducibility_checksum"] = exp6294.payload_checksum(bad_harm)
    with pytest.raises(ValueError, match="baseline_harm_controls"):
        exp6294.validate_artifact(bad_harm)


def test_scenario_arc_wmte_6294_duplicate_registry_target_guard(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6294-PROVENANCE-GUARDS: public registry targets are rejected."""

    artifact = exp6294.run(
        date="20260811",
        result_path=tmp_path / "artifact.json",
        fixture_manifest_path=tmp_path / "fixtures.json",
        live_window_manifest_path=tmp_path / "windows.json",
        raw_output_dir=tmp_path / "raw",
        duration_s=65.25,
        model_resolver=exp6294.deterministic_test_model_resolver,
        llm_runner=exp6294.deterministic_test_llm_runner,
        write=False,
    )
    bad = dict(artifact)
    bad["registry_precheck_path_hash_and_target_receipt"] = {
        **artifact["registry_precheck_path_hash_and_target_receipt"],
        "target_present_in_registry": True,
    }
    bad["reproducibility_checksum"] = exp6294.payload_checksum(bad)

    with pytest.raises(ValueError, match="registry_precheck_path_hash_and_target_receipt"):
        exp6294.validate_artifact(bad)


def test_req_arc_wmte_6294_blocked_precondition_artifact_is_terminal(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6294: blocked model resolution writes zero-credit terminal receipt."""

    artifact = exp6294.run(
        date="20260811",
        result_path=tmp_path / "artifact.json",
        fixture_manifest_path=tmp_path / "fixtures.json",
        live_window_manifest_path=tmp_path / "windows.json",
        raw_output_dir=tmp_path / "raw",
        duration_s=0.5,
        model_resolver=exp6294.missing_gemma_test_model_resolver,
        llm_runner=exp6294.deterministic_test_llm_runner,
        write=True,
    )

    assert artifact["status"].startswith("blocked_")
    assert artifact["honest_verdict"].startswith("complete: blocked_")
    assert artifact["arc_mechanic_causal_ready_score"] == 0.0
    assert artifact["duration_padding_count"] == 0
    assert artifact["raw_output_paths_and_hashes"] == {}
    assert json.loads((tmp_path / "artifact.json").read_text(encoding="utf-8")) == artifact
    exp6294.validate_artifact(artifact)


def test_req_arc_wmte_6294_helper_edges_cover_fail_closed_paths(monkeypatch) -> None:
    """REQ-ARC-WMTE-6294: helper edges preserve the same fail-closed contract."""

    fixtures = exp6294.build_fresh_fixtures(seed=6294, per_mechanic=1)
    windows = exp6294.build_live_transition_windows(fixtures, seeds=(404,))
    models = exp6294.deterministic_test_model_resolver(False)
    requests = exp6294.build_matched_arm_requests(windows[:1], models[:1])

    missing_arm = [dict(requests[0])]
    with pytest.raises(ValueError, match="matched pair arms"):
        exp6294.validate_matched_requests(missing_arm)

    mismatched = [dict(row) for row in requests[:2]]
    mismatched[1]["seed"] = 999
    with pytest.raises(ValueError, match="matched pair seed"):
        exp6294.validate_matched_requests(mismatched)

    route_bad = [dict(row) for row in requests[:2]]
    route_bad[1]["seed"] = route_bad[0]["seed"]
    route_bad[1]["route_block_present"] = False
    with pytest.raises(ValueError, match="matched pair route block"):
        exp6294.validate_matched_requests(route_bad)

    monkeypatch.setenv("CARNOT_ARC_MECHANIC_CLASS_ROUTER", "old")
    exp6294._prompt_for_window(windows[0], route_on=False)
    assert exp6294.os.environ["CARNOT_ARC_MECHANIC_CLASS_ROUTER"] == "old"

    assert exp6294._executable_acceptance("no code here") is False
    assert exp6294._executable_acceptance("```python\ndef engine(:\n```") is False
    assert exp6294._invalid_action_rate("no action ids") == 0.0
    assert exp6294._mean_record([]) == {"mean": 0.0, "sample_size": 0}
    assert (
        exp6294._paired_delta_receipt([{"request": {"pair_key": "x", "arm": "router_off"}}])[
            "sample_size_pairs"
        ]
        == 0
    )

    off_request = {"pair_key": "p", "arm": "router_off", "mechanic": "push_block", "model_id": "m"}
    on_request = {"pair_key": "p", "arm": "router_on", "mechanic": "push_block", "model_id": "m"}
    one_pair = [
        {"request": off_request, "text": "CANARY_ROUTE_CLASS=none", "latency_s": 0.1},
        {"request": on_request, "text": "CANARY_ROUTE_CLASS=push_block", "latency_s": 0.2},
    ]
    assert (
        exp6294._paired_delta_receipt(one_pair)["ci95"][0]
        == exp6294._paired_delta_receipt(one_pair)["ci95"][1]
    )

    harm_rows = [
        {
            "request": off_request,
            "text": "CANARY_ROUTE_CLASS=push_block ACTION4\n```python\ndef engine(grid, action, data):\n    return grid\n```",
            "latency_s": 0.1,
        },
        {
            "request": on_request,
            "text": "CANARY_ROUTE_CLASS=none ACTION9",
            "latency_s": 11.0,
        },
        {"request": {"pair_key": "missing", "arm": "router_off"}, "text": "", "latency_s": 0.0},
    ]
    harm = exp6294._baseline_harm(harm_rows)
    assert harm["baseline_harm_detected"] is True
    assert {row["reason"] for row in harm["harm_rows"]} >= {
        "missing_arm",
        "invalid_rate_increase",
        "proposal_score_drop",
        "latency_excess",
    }

    assert exp6294._model_revision("/cache/snapshots/rev123/model.gguf") == "rev123"
    assert exp6294._model_revision("") is None
    assert exp6294._model_revision("/cache/model.gguf") is None
    assert exp6294._quant_from_path("/cache/model-UD-Q4_K_M.gguf") == "UD-Q4_K_M"
    assert exp6294._quant_from_path("/cache/model.gguf") == exp6294.PREFERRED_QUANT


def test_req_arc_wmte_6294_validation_rejects_checksum_drift(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6294: reproducibility checksum catches artifact edits."""

    artifact = exp6294.run(
        date="20260811",
        result_path=tmp_path / "artifact.json",
        fixture_manifest_path=tmp_path / "fixtures.json",
        live_window_manifest_path=tmp_path / "windows.json",
        raw_output_dir=tmp_path / "raw",
        duration_s=65.25,
        model_resolver=exp6294.deterministic_test_model_resolver,
        llm_runner=exp6294.deterministic_test_llm_runner,
        write=False,
    )
    bad = dict(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp6294.validate_artifact(bad)
