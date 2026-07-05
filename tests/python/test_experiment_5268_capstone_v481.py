"""Tests for Exp 5268 V481 capstone synthesis.

Spec refs: REQ-REPORT-5268, SCENARIO-REPORT-5268,
SCENARIO-REPORT-5268-BLOCKED-MISSING-INPUT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5268_capstone_v481 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _wrap(value: Any, principle: str = "fixture principle") -> dict[str, Any]:
    return {"principle": principle, "value": value}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _base(verdict: str, substrate: str = mod.INFERENCE_SUBSTRATE) -> dict[str, Any]:
    return {
        "duration_s": 1.0,
        "honest_verdict": _wrap(verdict),
        "inference_substrate": _wrap(substrate),
        "reproducibility_checksum": "sha256:fixture",
    }


def _payloads() -> dict[int, dict[str, Any]]:
    return {
        5257: _base("complete: .480 archived and .481 activation-ready"),
        5258: {
            **_base("complete: 7 new actionable findings appended"),
            "new_references_added": _wrap(7),
        },
        5259: {
            **_base(
                "complete: sota_runtime_ready=true ready through flagship_moe",
                "llama_cpp_runtime_preflight_no_quality_claim",
            ),
            "sota_runtime_ready": True,
            "no_quality_claim": _wrap(True),
        },
        5260: {
            **_base("complete: cross-model typed memory null; delta_over_no_memory=0.000000"),
            "cross_model_memory_useful": False,
            "delta_over_no_memory": _wrap(0.0),
            "delta_over_shuffled_memory": _wrap(0.0),
            "unsafe_false_accepts": _wrap(0),
            "rollback_exercised": _wrap(False),
        },
        5261: {
            **_base(
                "complete: memory policy ready for cached fixture replay; harmful_rollback_passed=true"
            ),
            "memory_policy_ready": True,
            "retention_rate": _wrap(1.0),
            "interference_rate": _wrap(0.0),
            "harmful_memory_rollback_passed": _wrap(True),
        },
        5262: {
            **_base(
                "complete: solver-grounded extraction produced no useful oracle-distinct signal",
                "live_llm_inference_local_gguf_sota",
            ),
            "flagged_adversarial": True,
            "solver_grounded_extractor_ready": False,
            "constraint_validity_rate": _wrap(0.25),
            "false_accepts": _wrap(0),
        },
        5263: {
            **_base(
                "complete: null logit-energy unsupported-minus-supported delta=0.004811",
                "live_llm_inference_local_gguf_sota",
            ),
            "flagged_adversarial": True,
            "internal_signal_available": True,
            "hidden_energy_probe_signal_delta": 0.004811,
            "external_text_scorer_used": _wrap(False),
        },
        5264: {
            **_base("complete: useful scheduler replay preserved always-full decision quality"),
            "scheduler_ready": True,
            "full_verifier_calls_avoided_rate": _wrap(0.857143),
            "decision_quality_delta": _wrap(0.0),
            "false_accept_delta": _wrap(0.0),
        },
        5265: {
            **_base(
                "complete: refinement added certificate value",
                "offline_deterministic_certificate_no_llm",
            ),
            "certificate_refinement_ready": True,
            "true_property_certified": _wrap(True),
            "false_property_rejected": _wrap(True),
            "slack_before_after": _wrap({"envelope_gap_bound_reduction": 0.05859375}),
        },
        5266: {
            **_base(
                "blocked_board_reachability: kv260=blocked_kv260_ssh_unreachable polarfire=blocked_polarfire_ssh_unreachable gatemate=blocked_physical_jtag no_speedup_claim",
                "hardware_probe_no_speedup_claim",
            ),
            "kv260_status": _wrap("blocked_kv260_ssh_unreachable"),
            "polarfire_status": _wrap("blocked_polarfire_ssh_unreachable"),
            "gatemate_status": _wrap("blocked_physical_jtag"),
            "speedup_claimed": False,
        },
        5267: {
            **_base("complete: producer-side normalizer adoption is ready"),
            "producer_normalizer_ready": True,
            "gate_fields_preserved": _wrap(True),
            "safe_repairs_supported": _wrap(["top_level_principle_wrapper_unwrap"]),
            "unsafe_repairs_rejected": _wrap(["missing_methodology_receipt"]),
        },
    }


def _make_repo(root: Path, *, omit: set[int] | None = None) -> None:
    omit = omit or set()
    by_number = _payloads()
    for source in mod.PRIMARY_SOURCES:
        if source.experiment_number not in omit:
            _write_json(root / source.relative_path, by_number[source.experiment_number])
    _write_json(
        root / "results/experiment_5265_kan_certificate_explanation_refinement_v481_explanation.json",
        {
            "schema": "carnot.experiment_5265.kan_certificate_explanation_refinement.v481.explanation",
            "experiment_id": "exp5265-kan-certificate-explanation-refinement-v481",
            "methodology_note": "auxiliary explanation artifact",
        },
    )
    (root / "research-roadmap.yaml").write_text('milestone: "2026.07.481"\n', encoding="utf-8")
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/research_conductor.py").write_text("# fixture\n", encoding="utf-8")
    for context in mod.SOURCE_CONTEXT_PATHS:
        path = root / context
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"context for {context}\n", encoding="utf-8")


def _commands() -> list[dict[str, str]]:
    return [{"command": ".venv/bin/pytest tests/python/test_experiment_5268_capstone_v481.py -q", "outcome": "PASS"}]


def _ids(rows: list[dict[str, Any]]) -> set[int]:
    return {int(row["experiment_number"]) for row in rows if "experiment_number" in row}


def test_req_report_5268_spec_declares_capstone_contract() -> None:
    """REQ-REPORT-5268: OpenSpec anchors the V481 capstone fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5268") : spec.index("REQ-REPORT-5257")]

    for marker in (
        "REQ-REPORT-5268",
        "SCENARIO-REPORT-5268",
        "SCENARIO-REPORT-5268-BLOCKED-MISSING-INPUT",
        str(mod.RESULT_RELATIVE_PATH),
        "cached_fixture_replay_no_llm",
        "flagged_adversarial=true",
        "conductor_modified.value=false",
        "roadmap_modified.value=false",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5268_classifies_v481_fixture_outcomes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5268: clean, null, flagged, and blocked outcomes stay separate."""

    _make_repo(tmp_path)
    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260705",
        duration_s=1.0,
        commands_run=_commands(),
        conductor_modified=False,
        roadmap_modified=False,
        research_complete_updated=False,
    )

    mod.validate_artifact(artifact)
    assert mod.value_of(artifact["honest_verdict"]).startswith("complete:")
    assert mod.value_of(artifact["inference_substrate"]) == mod.INFERENCE_SUBSTRATE
    assert _ids(mod.value_of(artifact["clean_positives"])) == {5257, 5258, 5259, 5261, 5264, 5265, 5267}
    assert _ids(mod.value_of(artifact["clean_nulls"])) == {5260}
    assert mod.value_of(artifact["harmful_results"]) == []

    blocked = mod.value_of(artifact["blocked_or_skipped"])
    assert _ids(blocked) == {5262, 5263, 5266}
    assert all(row.get("classification") != "clean_null" for row in blocked)
    assert {row["experiment_number"] for row in blocked if row["classification"] == "flagged_adversarial"} == {5262, 5263}
    assert any("speedup_claimed=false" in row["summary"] for row in blocked)

    retirements = mod.value_of(artifact["retirements_or_exclusions"])
    assert any("Phase D external text scorer" in row["scope"] for row in retirements)
    assert any("Exp5262/Exp5263" in row["scope"] for row in retirements)

    gaps = mod.value_of(artifact["next_top_gaps"])
    assert [gap["priority_rank"] for gap in gaps] == [1, 2, 3]
    assert {gap["category"] for gap in gaps} == {
        "internal_verification",
        "continuous_self_learning_and_sota_runtime",
        "hardware_reachability",
    }
    assert mod.value_of(artifact["conductor_modified"]) is False
    assert mod.value_of(artifact["roadmap_modified"]) is False
    assert artifact["source_artifacts_read"]["auxiliary_count"] == 1


def test_scenario_report_5268_missing_required_artifact_blocks_without_mutation(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5268-BLOCKED-MISSING-INPUT: missing upstreams fail closed."""

    _make_repo(tmp_path, omit={5264})
    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260705",
        duration_s=1.0,
        commands_run=_commands(),
        conductor_modified=False,
        roadmap_modified=False,
        research_complete_updated=False,
    )

    mod.validate_artifact(artifact)
    assert mod.value_of(artifact["honest_verdict"]).startswith("blocked_missing_required")
    blocked = mod.value_of(artifact["blocked_or_skipped"])
    assert any(row["experiment_number"] == 5264 and row["classification"] == "missing" for row in blocked)
    assert mod.value_of(artifact["conductor_modified"]) is False
    assert mod.value_of(artifact["roadmap_modified"]) is False


def test_req_report_5268_research_complete_entry_builder_is_minimal() -> None:
    """REQ-REPORT-5268: research-complete updates are minimal and milestone-scoped."""

    entry = mod.build_research_complete_entry()

    assert entry["id"] == mod.MILESTONE
    assert entry["completed"] == "2026-07-05"
    assert len(entry["tasks"]) == len(mod.MILESTONE_TASKS)
    assert entry["tasks"][-1]["id"] == mod.EXPERIMENT_ID
    assert "flagged internal-verification pilots quarantined" in entry["finding"]


def test_req_report_5268_defensive_helpers_and_malformed_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-5268: helper paths fail closed instead of inventing evidence."""

    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    payload, info = mod.read_json_mapping(malformed)
    assert payload == {}
    assert info["loadable"] is False
    assert str(info["error"]).startswith("malformed_json")

    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    payload, info = mod.read_json_mapping(scalar)
    assert payload == {}
    assert info["error"] == "not_json_object"

    out = tmp_path / "nested" / "out.json"
    mod.write_json(out, {"ok": True})
    assert json.loads(out.read_text(encoding="utf-8")) == {"ok": True}

    assert mod._float(None) == 0.0
    assert mod._float(True) == 0.0
    assert mod._artifact_summary(9999, {}) == "no verdict"
    assert mod._classify_loaded(1, {"honest_verdict": "complete: harmful regression"}) == "harmful"
    assert (
        mod._classify_loaded(
            1,
            {"honest_verdict": "complete: memory ready; harmful_rollback_passed=true"},
        )
        == "clean_positive"
    )
    assert mod._classify_loaded(1, {"honest_verdict": "ambiguous"}) == "blocked"

    _make_repo(tmp_path)
    bad_source = mod.PRIMARY_SOURCES[0]
    (tmp_path / bad_source.relative_path).write_text("{", encoding="utf-8")
    row, loaded = mod._row_for_source(bad_source, tmp_path)
    assert loaded is None
    assert row["classification"] == "malformed"

    assert mod.load_commands(None) == []
    assert mod.load_commands(tmp_path / "missing.json") == []
    commands_json = tmp_path / "commands.json"
    commands_json.write_text(json.dumps([{"command": "x"}, "skip"]), encoding="utf-8")
    assert mod.load_commands(commands_json) == [{"command": "x"}]
    commands_json.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="commands JSON"):
        mod.load_commands(commands_json)

    assert mod.append_research_complete_if_missing(tmp_path) is True
    assert mod.append_research_complete_if_missing(tmp_path) is False
    assert f"id: {mod.MILESTONE}" in (tmp_path / "research-complete.yaml").read_text(
        encoding="utf-8"
    )

    class _Completed:
        stdout = " M scripts/research_conductor.py\n"

    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: _Completed())
    assert mod.git_file_modified(tmp_path, "scripts/research_conductor.py") is True

    def _raise_oserror(*args: Any, **kwargs: Any) -> None:
        raise OSError("git unavailable")

    monkeypatch.setattr(mod.subprocess, "run", _raise_oserror)
    assert mod.git_file_modified(tmp_path, "scripts/research_conductor.py") is False


def test_req_report_5268_main_writes_artifact_with_command_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-5268: the CLI writes the reproducible capstone artifact."""

    _make_repo(tmp_path)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "git_file_modified", lambda root, relative: False)
    commands_path = tmp_path / "commands.json"
    commands_path.write_text(json.dumps(_commands()), encoding="utf-8")
    output = tmp_path / mod.RESULT_RELATIVE_PATH

    assert (
        mod.main(
            [
                "--output",
                str(output),
                "--commands-json",
                str(commands_path),
                "--update-research-complete",
            ]
        )
        == 0
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["commands_run"] == _commands()
    assert artifact["research_complete_updated"]["value"] is True
