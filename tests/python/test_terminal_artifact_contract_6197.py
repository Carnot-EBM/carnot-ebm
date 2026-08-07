"""Tests for the shared Exp6197 terminal-artifact classifier.

Spec refs: REQ-INFRA-6197, SCENARIO-INFRA-6197-1,
SCENARIO-INFRA-6197-2, SCENARIO-INFRA-6197-3,
SCENARIO-INFRA-6197-4.
"""

from __future__ import annotations

from pathlib import Path

from carnot import terminal_artifacts as mod


REPO = Path(__file__).resolve().parents[2]


def test_scenario_infra_6197_replays_committed_fixture_classes() -> None:
    """SCENARIO-INFRA-6197-3: committed fixtures classify by artifact state."""

    cases = {
        "complete": ("results/experiment_482_think_probe_live_v3.json", "complete", True),
        "ready": ("results/experiment_6194_mode_jump_rust_pyo3_parity.json", "ready", True),
        "positive": (
            "results/experiment_6195_arc_task_aware_prospective_fresh_transition.json",
            "positive",
            True,
        ),
        "skipped": (
            "results/experiment_6193_prospective_continuous_strategy_learning_ab.json",
            "skipped",
            True,
        ),
        "retired": ("results/experiment_6175_cctu_headroom_audit.json", "retired", True),
        "flagged": (
            "results/experiment_6187_livecodebench_authentic_k8_pool.json",
            "flagged",
            True,
        ),
        "blocked": ("results/experiment_411_humaneval_live.json", "blocked", True),
        "running": (
            "results/experiment_1239_nrgpt_frozen_prefix_evaluation.json",
            "running",
            False,
        ),
        "running_bootstrap_6183": (
            "results/experiment_6183_transition_v536.json",
            "running_bootstrap",
            False,
        ),
        "running_bootstrap_6196": (
            "results/experiment_6196_v536_capstone_reconciliation.json",
            "running_bootstrap",
            False,
        ),
        "malformed": ("results/experiment_2436_pcib_tier0l.json", "malformed", False),
        "non_object_json": ("results/experiment_2352_nsvif_corpus.json", "malformed", False),
        "missing": (
            "results/experiment_6189_matching_base_code_hidden_state_surface.json",
            "missing",
            False,
        ),
    }

    for name, (rel_path, expected_class, expected_terminal) in cases.items():
        got = mod.classify_artifact_path(REPO / rel_path)
        assert got.classification == expected_class, name
        assert got.terminal is expected_terminal, name
        assert got.receipt_overrode is False, name


def test_scenario_infra_6197_cross_product_fails_closed() -> None:
    """SCENARIO-INFRA-6197-2: status/verdict cross-products fail closed."""

    terminal_pairs = {
        ("complete", "complete: finished"): "complete",
        ("complete_ready", "complete_ready: ready"): "ready",
        ("complete_positive", "complete_positive: positive"): "positive",
        ("complete_null", "complete_null: null"): "null",
        ("blocked", "blocked_precondition"): "blocked",
        ("retired", "retired: closed"): "retired",
        ("skipped", "skipped_gate_closed"): "skipped",
        ("complete", "complete_null: null"): "null",
        ("complete_ready", "complete: finished"): "ready",
    }
    for (status, verdict), expected in terminal_pairs.items():
        got = mod.classify_artifact_payload({"status": status, "honest_verdict": verdict})
        assert got.terminal is True
        assert got.classification == expected

    nonterminal_pairs = {
        ("running", "running"),
        ("running_bootstrap", "blocked: bootstrap only"),
        ("bootstrap_only", "bootstrap_only"),
        ("complete_partial", "complete_partial: partial"),
        ("complete_positive", "blocked_precondition"),
        ("complete_ready", "running"),
        ("", "complete: verdict without status"),
        ("complete_ready", ""),
        ("unknown_new_status", "complete: unknown status"),
    }
    for status, verdict in nonterminal_pairs:
        got = mod.classify_artifact_payload({"status": status, "honest_verdict": verdict})
        assert got.terminal is False, (status, verdict, got)
        assert got.classification in {
            "running",
            "running_bootstrap",
            "bootstrap_only",
            "partial",
            "contradictory",
            "unknown",
        }


def test_scenario_infra_6197_aliases_and_flagged_artifacts() -> None:
    """REQ-INFRA-6197: aliases are explicit and flags are artifact-owned."""

    assert "complete:" in mod.ACCEPTED_TERMINAL_PREFIXES
    assert "complete_ready" in mod.ACCEPTED_TERMINAL_PREFIXES
    assert "running_bootstrap" in mod.REJECTED_NONTERMINAL_PREFIXES
    assert "complete_partial" in mod.REJECTED_NONTERMINAL_PREFIXES

    assert mod.normalize_marker("completed").classification == "complete"
    assert mod.normalize_marker("success: shipped").classification == "complete"
    assert mod.normalize_marker("passed_ready").classification == "complete"
    assert mod.normalize_marker("complete-ready: ok").classification == "ready"
    assert mod.normalize_marker("in_progress").classification == "running"
    assert mod.normalize_marker(None).classification == "unknown"
    assert mod.normalize_marker("blocked: bootstrap only; still running").classification == (
        "bootstrap_only"
    )

    flagged = mod.classify_artifact_payload(
        {
            "status": "complete_partial",
            "honest_verdict": "complete_partial: quarantined",
            "flagged_adversarial": True,
        }
    )
    assert flagged.terminal is True
    assert flagged.classification == "flagged"

    corrigendum_flagged = mod.classify_artifact_payload(
        {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: ok",
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT"}],
        }
    )
    assert corrigendum_flagged.classification == "flagged"

    gate_list_skip = mod.classify_artifact_payload(
        {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gates_evaluated": [{"passed": False}],
        }
    )
    assert gate_list_skip.classification == "skipped"

    no_receipt_status = mod.classify_artifact_payload([], conductor_receipt={"status": None})
    assert no_receipt_status.classification == "malformed"
    assert no_receipt_status.conductor_receipt_status is None

    unowned_flag = mod.classify_artifact_payload(
        {"status": "complete_ready", "honest_verdict": "complete_ready: ok"},
        conductor_receipt={"status": "FLAGGED"},
    )
    assert unowned_flag.terminal is True
    assert unowned_flag.classification == "ready"
    assert unowned_flag.receipt_overrode is False


def test_scenario_infra_6197_receipts_cannot_override_bootstrap_or_missing() -> None:
    """SCENARIO-INFRA-6197-1: completion receipts never terminalize bad paths."""

    receipt = {"status": "OK", "detail": "conductor says complete"}
    for rel_path in (
        "results/experiment_6183_transition_v536.json",
        "results/experiment_6196_v536_capstone_reconciliation.json",
        "results/experiment_6189_matching_base_code_hidden_state_surface.json",
        "results/experiment_2436_pcib_tier0l.json",
    ):
        got = mod.classify_artifact_path(REPO / rel_path, conductor_receipt=receipt)
        assert got.terminal is False
        assert got.receipt_override_attempted is True
        assert got.receipt_overrode is False


def test_scenario_infra_6197_protected_files_are_read_only() -> None:
    """SCENARIO-INFRA-6197-3: classifier reads historical fixtures immutably."""

    paths = [
        REPO / "results/experiment_6183_transition_v536.json",
        REPO / "results/experiment_6194_mode_jump_rust_pyo3_parity.json",
        REPO / "results/experiment_6195_arc_task_aware_prospective_fresh_transition.json",
        REPO / "results/experiment_6196_v536_capstone_reconciliation.json",
    ]
    before = {path: mod.path_sha256(path) for path in paths}

    for path in paths:
        mod.classify_artifact_path(path)

    after = {path: mod.path_sha256(path) for path in paths}
    assert after == before


def test_req_infra_6197_classifier_has_no_conductor_import() -> None:
    """REQ-INFRA-6197: shared classifier is independent of the conductor."""

    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "research_conductor" not in source
    assert "scripts." not in source
