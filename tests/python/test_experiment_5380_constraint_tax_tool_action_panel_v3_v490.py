"""Tests for Exp 5380 constraint-tax tool/action reachability panel.

Spec refs: REQ-VERIFY-5380, SCENARIO-VERIFY-5380.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5379_live_structured_clean_gate_rerun_v490 as exp5379
from carnot import experiment_5380_constraint_tax_tool_action_panel_v3_v490 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5380_constraint_tax_tool_action_panel_v3_v490.py -q"
)


def _ready_exp5379() -> dict[str, Any]:
    return json.loads((REPO / exp5379.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))


def test_req_verify_5380_spec_declares_constraint_tax_panel_contract() -> None:
    """REQ-VERIFY-5380: OpenSpec anchors the Exp5380 state-evidence panel."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5380") : spec.index("### REQ-VERIFY-5379")]

    for marker in (
        "REQ-VERIFY-5380",
        "SCENARIO-VERIFY-5380",
        str(mod.RESULT_RELATIVE_PATH),
        "structured_protocol_clean=true",
        "constraint_tax_panel_ready",
        "initial state",
        "expected final state",
        "tool-call trace",
        "verifier predicate",
        "response-text fallback",
        "wrong-valid outputs",
        "unsafe_false_accepts=0",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "llama.cpp/GGUF",
        "AutoTokenizer",
        "AutoModel",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_verify_5380_clean_gate_writes_state_evidence_panel(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5380: clean Exp5379 enables paired constraint-tax metrics."""

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        exp5379_artifact=_ready_exp5379(),
        tests_run=[TEST_COMMAND],
        write=True,
    )

    mod.validate_artifact(artifact)
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["upstream_structured_protocol_clean"] is True
    assert artifact["constraint_tax_panel_ready"] is True
    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(mod.MANDATED_HF_IDS)
    assert artifact["selected_model_spec"] is None
    assert artifact["inference_substrate"]["live_llm_calls_ran"] is False
    assert artifact["inference_substrate"]["kind"] == "deterministic_fixture_replay"
    assert artifact["fixture_count"] == len(mod.DEFAULT_PANEL_FIXTURES)
    assert artifact["constrained_schema_validity_rate"] == pytest.approx(1.0)
    assert artifact["unconstrained_schema_validity_rate"] == pytest.approx(1 / 3)
    assert artifact["constrained_semantic_success_rate"] == pytest.approx(1.0)
    assert artifact["unconstrained_semantic_success_rate"] == pytest.approx(0.0)
    assert artifact["wrong_valid_count"] == 1
    assert artifact["deterministic_state_evidence_count"] == artifact["fixture_count"]
    assert artifact["tool_action_reachability_rate"] == pytest.approx(1.0)
    assert artifact["latency_or_token_overhead"]["latency_s_delta"] > 0
    assert artifact["latency_or_token_overhead"]["token_delta"] > 0
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["honest_verdict"].startswith("complete:")

    for row in artifact["paired_fixture_results"]:
        assert row["initial_state"]
        assert row["expected_final_state"]
        assert row["verifier_predicate"]
        assert row["constrained"]["final_state"] == row["expected_final_state"]
        for arm in ("unconstrained", "constrained"):
            assert "tool_call_trace" in row[arm]
            assert "response_text_fallback" in row[arm]
            assert row[arm]["response_text_fallback"]["accepted_as_semantic_evidence"] is False

    bad_artifact = dict(artifact)
    bad_artifact.pop("status")
    assert any(
        "missing required fields" in error for error in mod.artifact_schema_errors(bad_artifact)
    )
    with pytest.raises(ValueError, match="Exp 5380 artifact validation failed"):
        mod.validate_artifact(bad_artifact)


def test_scenario_verify_5380_blocks_when_exp5379_gate_is_not_clean(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5380: false upstream clean gate blocks the panel run."""

    upstream = _ready_exp5379()
    upstream["structured_protocol_clean"] = False

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked.json",
        exp5379_artifact=upstream,
        tests_run=[TEST_COMMAND],
        write=True,
    )

    mod.validate_artifact(artifact)
    assert json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "blocked"
    assert artifact["upstream_structured_protocol_clean"] is False
    assert artifact["constraint_tax_panel_ready"] is False
    assert artifact["fixture_count"] == 0
    assert artifact["deterministic_state_evidence_count"] == 0
    assert artifact["tool_action_reachability_rate"] == pytest.approx(0.0)
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["paired_fixture_results"] == []
    assert artifact["honest_verdict"].startswith("blocked_")

    fallback_specs = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked-missing-specs.json",
        exp5379_artifact={"structured_protocol_clean": False},
        tests_run=[TEST_COMMAND],
        write=False,
    )
    assert {row["status"] for row in fallback_specs["MODEL_SPECS"]} == {
        "missing_from_exp5379_artifact"
    }

    no_fixture_panel = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked-empty-panel.json",
        exp5379_artifact=_ready_exp5379(),
        panel_fixtures=[],
        tests_run=[TEST_COMMAND],
        write=False,
    )
    assert no_fixture_panel["status"] == "blocked"
    assert no_fixture_panel["honest_verdict"] == "blocked_no_paired_fixtures"


def test_scenario_verify_5380_unsafe_false_accept_blocks_readiness(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5380: unsafe semantic accepts are counted and block readiness."""

    fixtures = copy.deepcopy(mod.DEFAULT_PANEL_FIXTURES[:1])
    fixtures[0]["constrained"]["unsafe_response"] = True

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "unsafe.json",
        exp5379_artifact=_ready_exp5379(),
        panel_fixtures=fixtures,
        tests_run=[TEST_COMMAND],
        write=False,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["fixture_count"] == 1
    assert artifact["unsafe_false_accepts"] == 1
    assert artifact["constraint_tax_panel_ready"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert "unsafe_false_accepts=1" in artifact["honest_verdict"]


def test_scenario_verify_5380_main_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5380: CLI entrypoint writes the terminal artifact."""

    upstream_path = tmp_path / exp5379.RESULT_RELATIVE_PATH
    upstream_path.parent.mkdir(parents=True, exist_ok=True)
    upstream_path.write_text(json.dumps(_ready_exp5379()), encoding="utf-8")
    out_path = tmp_path / mod.RESULT_RELATIVE_PATH

    exit_code = mod.main(
        [
            "--root",
            str(tmp_path),
            "--artifact-path",
            str(out_path),
            "--exp5379-path",
            str(upstream_path),
        ]
    )

    assert exit_code == 0
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    assert artifact["status"] == "complete"
    mod.validate_artifact(artifact)
