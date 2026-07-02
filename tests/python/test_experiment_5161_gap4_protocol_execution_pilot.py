"""Tests for Exp 5161 GAP-4 protocol execution pilot.

Spec refs: REQ-REPORT-5161, SCENARIO-REPORT-5161,
SCENARIO-REPORT-5161-BLOCKED-SANDBOX.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5161_gap4_protocol_execution_pilot as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _row(domain: str, index: int, *, vote: bool, gated: bool, demo: bool = True) -> JsonDict:
    return {
        "pilot_key": f"{domain}:{index}",
        "domain": domain,
        "task": f"{domain}_task_{index:02d}",
        "entry_i": index,
        "cluster_id": f"{domain}:task_{index:02d}",
        "vote_top2": vote,
        "gated_top2": gated,
        "demo_perfect": demo,
        "pred_is_gold": gated and not vote,
        "oracle_hit": vote or gated,
    }


def _pilot_rows() -> list[JsonDict]:
    rows: list[JsonDict] = []
    # ARC-1 reproduces the positive direction: four gated-only wins, no losses.
    rows.extend(_row("arc1", i, vote=True, gated=True) for i in range(10))
    rows.extend(_row("arc1", 10 + i, vote=False, gated=True) for i in range(4))
    rows.extend(_row("arc1", 14 + i, vote=False, gated=False, demo=i < 14) for i in range(16))
    # ARC-2 is a held-out venue with no exact-match gate movement in this pilot.
    rows.extend(_row("arc2", i, vote=i < 2, gated=i < 2, demo=i < 16) for i in range(30))
    return rows


def _local_block() -> JsonDict:
    return {
        "status": "blocked_local_model_not_cached",
        "checked_model_ids": list(mod.MANDATED_LOCAL_MODEL_IDS),
        "cached_model_paths": [],
    }


def test_req_report_5161_spec_declares_pilot_contract() -> None:
    """REQ-REPORT-5161: OpenSpec declares the pilot artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-5161",
        "SCENARIO-REPORT-5161",
        "SCENARIO-REPORT-5161-BLOCKED-SANDBOX",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_report_5161_builds_n60_artifact_with_exact_tests() -> None:
    """SCENARIO-REPORT-5161: the artifact reports actual N and statistics."""

    artifact = mod.build_artifact(
        pilot_rows=_pilot_rows(),
        sandbox_smoke=mod.SandboxSmokeResult(
            passed=True,
            honest_verdict="complete: smoke",
            artifact_path="/tmp/smoke.json",
            transcript_paths=["/tmp/a.txt", "/tmp/b.txt"],
            duration_s=25.0,
        ),
        local_generator_arm_result=_local_block(),
        duration_s=3.0,
        partial=False,
        checkpoint_path=mod.CHECKPOINT_RELATIVE_PATH,
        source_artifacts=[
            {"path": "results/arc3_gap4_rule_exec_verifier.json", "exists": True},
            {"path": "results/arc3_gap4_arc2_rule_exec_verifier.json", "exists": True},
        ],
        transcript_archive={
            "fresh_sandbox_smoke_transcripts": ["/tmp/a.txt", "/tmp/b.txt"],
            "pilot_rows_replayed_from_saved_programs": 60,
        },
    )

    assert artifact["pilot_n_target"]["value"] == 60
    assert artifact["pilot_n_achieved"]["value"] == 60
    assert artifact["checkpoint_resume_used"]["value"] is True
    assert artifact["arc1_slice_result"]["n_entries"] == 30
    assert artifact["arc1_slice_result"]["pass2_delta_vs_vote"] == pytest.approx(4 / 30)
    assert artifact["arc2_heldout_slice_result"]["n_entries"] == 30
    assert artifact["arc2_heldout_slice_result"]["pass2_delta_vs_vote"] == 0.0
    assert artifact["exact_test_discordant_wins"]["value"] == 4
    assert artifact["exact_test_discordant_losses"] == 0
    assert artifact["exact_test_p_value_two_sided"] == 0.125
    assert artifact["exact_test_passes_min6_rule"]["value"] is False
    assert artifact["gap4_status_recommendation"]["value"] == "scale_up_recommended"
    assert artifact["local_generator_arm_result"]["value"]["status"] == (
        "blocked_local_model_not_cached"
    )
    assert artifact["replicated_prior_direction"] is True
    assert "not_significant" in artifact["honest_verdict"]
    assert artifact["reproducibility_checksum"]["value"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_report_5161_blocked_sandbox_smoke_failed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5161-BLOCKED-SANDBOX: sandbox failure blocks early."""

    artifact = mod.run(
        root=tmp_path,
        sandbox_smoke_checker=lambda _root: mod.SandboxSmokeResult(
            passed=False,
            honest_verdict="blocked_codex_cli",
            artifact_path="/tmp/missing.json",
            transcript_paths=[],
            duration_s=0.5,
        ),
        local_generator_checker=lambda _root: _local_block(),
        pilot_row_loader=lambda _root: _pilot_rows(),
        now=lambda: 10.0,
        sleep_fn=lambda _seconds: None,
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["honest_verdict"] == "blocked_sandbox_smoke_failed"
    assert artifact["pilot_n_achieved"]["value"] == 0
    assert artifact["checkpoint_resume_used"]["value"] is True
    assert artifact["arc1_slice_result"]["blocked"] == "sandbox_smoke_failed"
    assert artifact["arc2_heldout_slice_result"]["blocked"] == "sandbox_smoke_failed"
    assert mod.artifact_schema_errors(artifact) == []


def test_checkpoint_resume_flushes_rows_and_marks_partial(tmp_path: Path) -> None:
    """REQ-REPORT-5161: checkpoint/resume flushes each attempted task."""

    rows = _pilot_rows()[:3]
    calls = iter([0.0, 1000.0, 2000.0])
    attempted, partial, remaining = mod.run_rows_checkpointed(
        root=tmp_path,
        candidate_rows=rows,
        now=lambda: next(calls),
        soft_budget_s=1500.0,
    )

    assert partial is True
    assert [row["pilot_key"] for row in attempted] == [rows[0]["pilot_key"]]
    assert [row["pilot_key"] for row in remaining] == [rows[1]["pilot_key"], rows[2]["pilot_key"]]
    assert mod.load_checkpoint(tmp_path)["rows"] == attempted

    resumed, partial, remaining = mod.run_rows_checkpointed(
        root=tmp_path,
        candidate_rows=rows,
        now=lambda: 0.0,
        soft_budget_s=10_000.0,
    )

    assert partial is False
    assert remaining == []
    assert [row["pilot_key"] for row in resumed] == [row["pilot_key"] for row in rows]
    assert not (tmp_path / mod.CHECKPOINT_RELATIVE_PATH).exists()


def test_deterministic_edge_branches_for_checkpoint_stats_and_schema(tmp_path: Path) -> None:
    """REQ-REPORT-5161: deterministic helpers handle empty/corrupt/edge inputs."""

    assert mod.payload_checksum({"experiment": "x"}).startswith("sha256:")
    assert mod.resolve_soft_budget_s({}) == mod.DEFAULT_SOFT_BUDGET_S
    assert mod.resolve_soft_budget_s({mod.SOFT_BUDGET_ENV: "12.5"}) == 12.5
    assert mod.resolve_soft_budget_s({mod.SOFT_BUDGET_ENV: "0"}) == mod.DEFAULT_SOFT_BUDGET_S
    assert mod.resolve_soft_budget_s({mod.SOFT_BUDGET_ENV: "bad"}) == mod.DEFAULT_SOFT_BUDGET_S

    path = tmp_path / mod.CHECKPOINT_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert mod.load_checkpoint(tmp_path) == {"rows": []}
    path.write_text(json.dumps({"rows": "bad"}), encoding="utf-8")
    assert mod.load_checkpoint(tmp_path) == {"rows": []}

    assert mod._slice_result([])["precision_kind"] == "no_rows"
    override = {"precision": 0.25, "numerator": 1, "denominator": 4, "kind": "true_gold"}
    assert mod._slice_result([_row("arc1", 0, vote=False, gated=True)], override)["precision"] == 0.25
    assert mod.exact_test([_row("arc1", 0, vote=True, gated=True)])["p_value_two_sided"] == 1.0
    assert mod.cluster_bootstrap_delta_ci([]) is None
    assert mod._recommendation([], {"wins": 0, "losses": 0}) == "still_open"
    assert mod._recommendation(
        [_row("arc1", 0, vote=True, gated=False)], {"wins": 0, "losses": 1}
    ) == "retired"
    assert mod._recommendation(
        [_row("arc1", 0, vote=False, gated=True)], {"wins": 6, "losses": 0, "passes_min6_rule": True}
    ) == "scale_up_recommended"


def test_local_generator_cache_probe_reports_blocked_when_no_sota_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-5161: missing local GGUF cache blocks only the local arm."""

    monkeypatch.setattr(mod, "resolve_local_model_path", lambda _hf_id, _root: None)

    result = mod.check_local_generator_arm(tmp_path)

    assert result["value"]["status"] == "blocked_local_model_not_cached"
    assert result["value"]["cached_model_paths"] == []
    assert result["value"]["checked_model_ids"] == list(mod.MANDATED_LOCAL_MODEL_IDS)

    monkeypatch.setattr(mod, "resolve_local_model_path", lambda _hf_id, _root: "/models/q.gguf")
    monkeypatch.setattr(
        mod,
        "run_local_generator_subset",
        lambda model_path: {
            "status": "attempted_local_generator_subset",
            "model_path": model_path,
            "response_preview": "def transform(grid): return grid",
        },
    )
    cached = mod.check_local_generator_arm(tmp_path)
    assert cached["value"]["status"] == "attempted_local_generator_subset"
    assert cached["value"]["cached_model_paths"][0]["model_path"] == "/models/q.gguf"


def test_scenario_report_5161_runner_success_path_writes_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-5161: run() writes the successful pilot artifact."""

    monkeypatch.setattr(
        mod,
        "describe_source_artifacts",
        lambda _root: [{"path": "source.json", "exists": True}],
    )
    monkeypatch.setattr(
        mod,
        "transcript_archive_report",
        lambda _root, _sandbox, n_replayed: {"pilot_rows_replayed_from_saved_programs": n_replayed},
    )
    monkeypatch.setattr(mod, "load_precision_overrides", lambda _root: {})
    monkeypatch.setattr(mod, "resolve_soft_budget_s", lambda: 10_000.0)

    artifact = mod.run(
        root=tmp_path,
        sandbox_smoke_checker=lambda _root: mod.SandboxSmokeResult(
            passed=True,
            honest_verdict="complete: smoke",
            artifact_path="results/smoke.json",
            transcript_paths=["results/t.txt"],
            duration_s=2.0,
        ),
        local_generator_checker=lambda _root: _local_block(),
        pilot_row_loader=lambda _root: _pilot_rows(),
        now=lambda: 0.0,
        sleep_fn=lambda _seconds: None,
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["pilot_n_achieved"]["value"] == 60
    assert artifact["source_artifacts"] == [{"path": "source.json", "exists": True}]
    assert mod.artifact_schema_errors(artifact) == []


def test_artifact_schema_rejects_checksum_and_shape_errors() -> None:
    """REQ-REPORT-5161: schema validation protects required pilot fields."""

    artifact = mod.build_artifact(
        pilot_rows=_pilot_rows(),
        sandbox_smoke=mod.SandboxSmokeResult(
            passed=True,
            honest_verdict="complete: smoke",
            artifact_path="/tmp/smoke.json",
            transcript_paths=[],
            duration_s=1.0,
        ),
        local_generator_arm_result=_local_block(),
        duration_s=1.0,
        partial=False,
        checkpoint_path=mod.CHECKPOINT_RELATIVE_PATH,
        source_artifacts=[],
        transcript_archive={},
    )
    bad = dict(artifact)
    bad["honest_verdict"] = "not_terminal"
    bad["checkpoint_resume_used"] = {"value": False}
    bad["solve_provenance"] = {"value": "fabricated"}
    bad["inference_substrate"] = {"value": "aggregation_from_upstream_artifacts"}
    bad["pilot_n_target"] = {"value": 61}
    bad["pilot_n_achieved"] = {"value": 61}
    bad["exact_test_discordant_wins"] = {"value": 6}
    bad["exact_test_discordant_losses"] = 0
    bad["exact_test_passes_min6_rule"] = {"value": False}
    bad["random_seed"] = {"value": 0}
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = {"value": "sha256:bad"}

    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict_terminal_prefix" in errors
    assert "field_principles" in errors
    assert "pilot_n_target_60" in errors
    assert "pilot_n_achieved_bounds" in errors
    assert "checkpoint_resume_used_true" in errors
    assert "exact_test_passes_min6_rule" in errors
    assert "solve_provenance_development_proxy" in errors
    assert "inference_substrate_live_llm_inference" in errors
    assert "random_seed" in errors
    assert "reproducibility_checksum" in errors

    missing = dict(artifact)
    missing.pop("duration_s")
    missing["reproducibility_checksum"] = {"value": mod.payload_checksum(missing)}
    assert "missing required field duration_s" in mod.artifact_schema_errors(missing)

    with pytest.raises(ValueError):
        mod.write_artifact(Path("/tmp"), bad)
