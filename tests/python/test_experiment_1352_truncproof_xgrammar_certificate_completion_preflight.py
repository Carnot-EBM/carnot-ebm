"""Tests for Exp 1352 certificate completion-budget preflight.

Spec: REQ-VERIFY-1352, SCENARIO-VERIFY-1352
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import truncproof_xgrammar_certificate_completion_preflight as mod


def test_req1352_allows_sota_when_budget_and_dispatch_pass() -> None:
    """REQ-VERIFY-1352: sufficient token budget plus preserved dispatch opens the gate."""

    artifact = mod.build_preflight_artifact(
        run_date="20260505",
        project_root="/repo",
        runtime_settings={"max_tokens": 96},
        import_checker=lambda _name: False,
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["grammar_states"] == ["SAT", "UNSAT", "UNKNOWN", "REPAIR_HINT"]
    assert set(artifact["min_completion_tokens_by_state"]) == set(artifact["grammar_states"])
    assert all(tokens > 0 for tokens in artifact["min_completion_tokens_by_state"].values())
    assert artifact["max_token_budget_sufficient"] is True
    assert artifact["structural_tag_supported"] is True
    assert artifact["xgrammar_backend_available"] is False
    assert artifact["dynamic_dispatch_preserved"] is True
    assert artifact["tested_dispatch_backend"] == "pure_python_tagdispatch_shim"
    assert artifact["sota_run_allowed"] is True
    assert artifact["blocker_if_not_allowed"] is None
    assert artifact["sota_model_called"] is False
    assert artifact["honest_verdict"] == "preflight_allows_exp1353_pure_python_fallback_xgrammar_absent"


def test_req1352_blocks_sota_when_completion_budget_is_too_small() -> None:
    """REQ-VERIFY-1352: budget failures name the active max-token blocker."""

    native_allowed = mod.build_preflight_artifact(
        run_date="20260505",
        project_root="/repo",
        runtime_settings={"max_tokens": 96},
        import_checker=lambda name: name == "xgrammar",
    )
    artifact = mod.build_preflight_artifact(
        run_date="20260505",
        project_root="/repo",
        runtime_settings={"max_tokens": 1},
        import_checker=lambda name: name == "xgrammar",
    )

    assert native_allowed["honest_verdict"] == "preflight_allows_exp1353_native_xgrammar_available"
    assert artifact["xgrammar_backend_available"] is True
    assert artifact["max_token_budget_sufficient"] is False
    assert artifact["dynamic_dispatch_preserved"] is True
    assert artifact["sota_run_allowed"] is False
    assert artifact["blocker_if_not_allowed"] == (
        f"max_token_budget_insufficient: max_tokens=1 "
        f"required={artifact['max_min_completion_tokens']}"
    )
    assert artifact["honest_verdict"] == "blocked_max_token_budget_insufficient"


def test_req1352_structural_tags_preserve_dynamic_dispatch() -> None:
    """REQ-VERIFY-1352: structural tags select branches before body parsing."""

    grammar = mod.compile_branch_grammars()
    cases = mod.synthetic_completion_cases()
    rows = mod.evaluate_tagged_dispatch(cases, grammar=grammar)
    tag_state, body = mod.parse_structural_tag(cases[0].tagged_text)
    repair_state, _repair_body = mod.parse_structural_tag("<CARNOT_CERT_STATE:repair>\nREPAIR_HINT: fix x.")
    no_tag_state, no_tag_body = mod.parse_structural_tag("SAT")

    assert tag_state == cases[0].state
    assert body == cases[0].body
    assert repair_state == "REPAIR_HINT"
    assert no_tag_state is None
    assert no_tag_body == "SAT"
    assert mod.estimate_completion_tokens("") == 0
    assert {row["expected_state"] for row in rows} == set(mod.GRAMMAR_STATES)
    assert all(row["tag_state"] == row["dispatched_state"] for row in rows)
    assert all(row["dynamic_parseable"] for row in rows)
    assert mod.dynamic_dispatch_preserved(rows) is True
    assert mod.structural_tag_supported(rows) is True


def test_req1352_blocks_sota_when_dynamic_dispatch_breaks() -> None:
    """REQ-VERIFY-1352: dispatch failures close the SOTA gate even with enough tokens."""

    broken_cases = [
        mod.CompletionCase(
            state="SAT",
            body="No bounded label is present here.",
            tagged_text="<CARNOT_CERT_STATE:SAT>\nNo bounded label is present here.",
        )
    ]

    artifact = mod.build_preflight_artifact(
        run_date="20260505",
        project_root="/repo",
        runtime_settings={"max_tokens": 96},
        import_checker=lambda _name: False,
        cases=broken_cases,
    )

    assert artifact["max_token_budget_sufficient"] is True
    assert artifact["dynamic_dispatch_preserved"] is False
    assert artifact["sota_run_allowed"] is False
    assert artifact["blocker_if_not_allowed"] == "dynamic_dispatch_not_preserved"
    assert artifact["honest_verdict"] == "blocked_dynamic_dispatch_not_preserved"


def test_scenario1352_run_experiment_writes_in_progress_then_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-1352: the runner persists bootstrap and terminal artifacts."""

    output_path = tmp_path / "experiment_1352.json"
    exp1323_path = tmp_path / "experiment_1323.json"
    exp1323_path.write_text(
        json.dumps({"recommended_certificate_runtime_settings": {"max_tokens": 96}}),
        encoding="utf-8",
    )
    writes: list[dict[str, Any]] = []
    real_write = mod._write_json

    def recording_write(path: Path, payload: dict[str, Any]) -> None:
        writes.append(payload)
        real_write(path, payload)

    monkeypatch.setattr(mod, "_write_json", recording_write)

    artifact = mod.run_experiment(
        output_path=output_path,
        exp1323_path=exp1323_path,
        run_date="20260505",
        project_root=tmp_path,
        import_checker=lambda _name: False,
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert [write["status"] for write in writes] == ["in_progress", "complete"]
    assert persisted == artifact
    assert artifact["artifact_metadata"]["run_date"] == "20260505"
    assert artifact["artifact_metadata"]["project_root"] == str(tmp_path)
    assert artifact["runtime_settings_used"]["max_tokens"] == 96
