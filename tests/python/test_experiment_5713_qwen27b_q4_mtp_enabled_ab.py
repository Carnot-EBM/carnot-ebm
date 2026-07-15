"""Tests for Exp 5713: does enabling MTP change Qwen3.6-27B-MTP-GGUF's (Q4_K_M, Q8 KV-cache)
reinduction reliability -- the operator's "let's try Qwen3.6-27B 4bit quant one last time with a
Q8 kv-cache" request, resolved by finding that exact config already measured in exp5599 (MTP off)
and isolating the one untested variable (MTP on).

Spec refs: REQ-ARC-WMTE-5599-4.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5713_qwen27b_q4_mtp_enabled_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


class _DummyProposer:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def _build_with_rows(monkeypatch: pytest.MonkeyPatch, rows: list[object]) -> dict:
    proposer = _DummyProposer()
    row_iter = iter(rows)

    def fake_run_one_draw(*, proposer: object, repeat: int) -> dict:
        row = next(row_iter)
        if isinstance(row, BaseException):
            raise row
        return dict(row) | {"repeat": repeat}

    monkeypatch.setattr(mod, "preconditions", lambda root=mod.REPO_ROOT: {"ok": True})
    monkeypatch.setattr(mod, "_make_mtp_proposer", lambda: proposer)
    monkeypatch.setattr(mod, "_wait_for_port_down", lambda port: None)
    monkeypatch.setattr(mod, "_run_one_draw", fake_run_one_draw)

    artifact = mod.build_artifact(n_repeats=len(rows))
    assert proposer.stopped is True
    return artifact


def test_req_arc_wmte_5599_4_spec_declares_the_precheck_and_oom_finding() -> None:
    """REQ-ARC-WMTE-5599-4: OpenSpec declares the pre-check that found the requested config
    already measured, the one isolated variable (MTP), and the real OOM root cause found via
    direct manual diagnosis after the automated run stalled."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5599-4") :]

    for marker in (
        "already ran EXACTLY Q4_K_M",
        "doomed rerun",
        "cudaMalloc failed: out of memory",
        "self-speculative MTP",
        "loads the SAME GGUF file a SECOND time",
        "exceeds the single RTX 3090's 24GB",
        "blocked_gpu1_free_vram_sufficient_for_mtp_dual_load",
    ):
        assert marker in section


def test_first_precondition_miss_reports_failing_key() -> None:
    assert mod._first_precondition_miss({"ok": False, "a": True, "b": False}) == "b"
    assert mod._first_precondition_miss({"ok": True}) is None


def test_first_precondition_miss_skips_diagnostic_only_fields() -> None:
    """mtp_dual_load_estimated_mb and gpu1_free_mb are informational floats, not booleans --
    they must never be reported as the 'failing key' even when numerically truthy/falsy."""
    preconds = {
        "ok": False,
        "a": True,
        "mtp_dual_load_estimated_mb": 32628.6,
        "gpu1_free_mb": 24120.0,
        "b": False,
    }
    assert mod._first_precondition_miss(preconds) == "b"


def test_build_artifact_blocks_on_first_precondition_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {"ok": False, "dependency_ready": False},
    )

    artifact = mod.build_artifact(n_repeats=2)

    assert artifact["honest_verdict"] == "complete: blocked_dependency_ready"
    assert artifact["per_draw_results"] == []
    assert artifact["arm_summary"] == {}
    assert artifact["preconditions_checked"]["dependency_ready"] is False
    assert len(artifact["reproducibility_checksum"]) == 64


def test_build_artifact_live_branch_summarizes_rows_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _build_with_rows(
        monkeypatch,
        [
            {
                "levelup_reached": True,
                "planned": True,
                "reinduce_duration_s": 2.0,
                "heldout_accuracy": 0.5,
            },
            RuntimeError("server died"),
        ],
    )

    assert artifact["honest_verdict"] == (
        "complete: qwen27b_mtp_plans_more_reliably_than_current_9b"
    )
    assert artifact["arm_summary"]["n_attempted"] == 2
    assert artifact["arm_summary"]["n_levelup_reached"] == 1
    assert artifact["arm_summary"]["n_planned"] == 1
    assert artifact["arm_summary"]["mean_reinduce_duration_s"] == 2.0
    assert artifact["arm_summary"]["mean_heldout_accuracy"] == 0.5
    assert artifact["per_draw_results"][1]["error"].startswith("RuntimeError")


def test_build_artifact_no_levelup_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = _build_with_rows(monkeypatch, [{"levelup_reached": False}])

    assert artifact["arm_summary"]["plan_rate_given_levelup"] is None
    assert artifact["honest_verdict"] == "complete: qwen27b_mtp_lp85_never_leveled_up_inconclusive"


def test_build_artifact_tie_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        mod.EXP5599_HISTORICAL["current_9b_q4"],
        "plan_rate_given_levelup",
        0.5,
    )

    artifact = _build_with_rows(
        monkeypatch,
        [
            {"levelup_reached": True, "planned": True},
            {"levelup_reached": True, "planned": False},
        ],
    )

    assert artifact["arm_summary"]["plan_rate_given_levelup"] == 0.5
    assert artifact["honest_verdict"] == "complete: qwen27b_mtp_ties_current_9b_plan_rate"


def test_build_artifact_improves_over_no_mtp_but_below_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _build_with_rows(
        monkeypatch,
        [
            {"levelup_reached": True, "planned": True},
            {"levelup_reached": True, "planned": False},
            {"levelup_reached": True, "planned": False},
            {"levelup_reached": True, "planned": False},
        ],
    )

    assert artifact["arm_summary"]["plan_rate_given_levelup"] == 0.25
    assert artifact["honest_verdict"] == (
        "complete: qwen27b_mtp_improves_on_no_mtp_but_below_current_9b"
    )


def test_build_artifact_no_plan_ties_no_mtp_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = _build_with_rows(monkeypatch, [{"levelup_reached": True, "planned": False}])

    assert artifact["arm_summary"]["mean_reinduce_duration_s"] is None
    assert artifact["arm_summary"]["mean_heldout_accuracy"] is None
    assert artifact["honest_verdict"] == (
        "complete: qwen27b_mtp_plans_less_reliably_than_current_9b_ties_no_mtp"
    )


@pytest.mark.memory_watchdog_skip  # loads a real lp85 game env via preconditions(); legit
# one-time footprint (arc_agi game module + scorecard), not a leak
def test_mtp_dual_load_precondition_computes_real_arithmetic(monkeypatch) -> None:
    """The precondition is a real computed check (2x on-disk file size vs free VRAM), not a
    magic number -- pin free VRAM low and confirm the dual-load estimate (computed from the
    REAL cached GGUF's on-disk size) correctly exceeds it."""
    monkeypatch.setattr(mod, "_gpu1_free_mb", lambda: 1)  # force insufficient regardless of box

    preconds = mod.preconditions()
    assert preconds["mtp_dual_load_estimated_mb"] > 0  # real file size was found, not zero
    assert preconds["mtp_dual_load_estimated_mb"] > preconds["gpu1_free_mb"]
    assert preconds["gpu1_free_vram_sufficient_for_mtp_dual_load"] is False


def test_req_arc_wmte_5599_4_repository_artifact_is_a_real_blocked_measurement() -> None:
    """The checked-in artifact is an honest, FAST (duration_s=0.0) precondition block -- not a
    fabricated performance result. The model was never invoked; the block is a real arithmetic
    finding backed by a direct manual crash reproduction (embedded verbatim in the artifact).
    Adversarially clean after the adversarial_verify.py terminal-prefix fix this incident
    motivated (see test_adversarial_verify_blocked_verdict_duration_exemption.py)."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["honest_verdict"] == (
        "complete: blocked_gpu1_free_vram_sufficient_for_mtp_dual_load"
    )
    assert result["weight_precision"] == "q4_k_m"
    assert result["kv_cache_precision"] == "q8_0"
    assert result["mtp_enabled"] is True
    assert result["per_draw_results"] == []
    assert result["arm_summary"] == {}
    assert result["preconditions_checked"]["ok"] is False
    assert (
        result["preconditions_checked"]["mtp_dual_load_estimated_mb"]
        > result["preconditions_checked"]["gpu1_free_mb"]
    )
    assert "cudaMalloc failed: out of memory" in result["manual_diagnostic_crash_confirmation"]
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
    assert len(result["reproducibility_checksum"]) == 64
