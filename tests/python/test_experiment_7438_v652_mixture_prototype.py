"""Tests for REQ-AUTO-7438 causal four-expert probability aggregation."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7438_v652_mixture_prototype as exp


ROOT = Path(__file__).resolve().parents[2]


def _features(index: int) -> dict[str, float]:
    return {
        name: ((index + offset) % 9) / 8.0 for offset, name in enumerate(exp.SOURCE_FEATURE_NAMES)
    }


def _passing_receipts() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "required": True,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
        }
        for name in (*exp.AFFECTED_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES)
    ]


def test_req_auto_7438_spec_and_authenticated_context() -> None:
    """REQ-AUTO-7438: the requirement and branch-local context precede code."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "### REQ-AUTO-7438:" in text
    for number in range(1, 9):
        assert f"SCENARIO-AUTO-7438-{number:02d}" in text
    checks, hashes, upstream = exp.collect_preconditions(ROOT)
    assert checks and all(row["passed"] for row in checks)
    assert upstream["online_capture_complete_score"] == 1
    assert upstream["online_value_score"] == 0
    assert upstream["flagged_adversarial"] is False
    source = hashes[exp.UPSTREAM_PATH.as_posix()]
    assert source["original_verdict_class"] == "null"
    assert source["original_flagged_adversarial"] is False


def test_scenario_auto_7438_01_probability_mixture_and_energy() -> None:
    """SCENARIO-AUTO-7438-01: no feedback equals the frozen prior mixture."""

    controller = exp.FourExpertMixture(exp.build_numeric_experts(seed=65201))
    prediction = controller.predict("event-0", _features(0), index=0, delay=8)
    probabilities = prediction["expert_probabilities"]
    expected = (probabilities[exp.FROZEN_SPLINE] + probabilities[exp.FROZEN_GIBBS]) / 2.0
    assert prediction["mixture_probability"] == pytest.approx(expected, abs=1e-12)
    assert prediction["mixture_probability"] == pytest.approx(
        sum(probabilities.values()) / 4.0, abs=1e-12
    )
    assert prediction["energy"] == pytest.approx(-math.log(expected / (1.0 - expected)), abs=1e-12)
    assert controller.weights == {name: 0.25 for name in exp.EXPERT_NAMES}
    assert prediction["prediction_hash"] == exp.prediction_hash(prediction)


def test_scenario_auto_7438_02_stored_loss_updates_only_adaptive_experts() -> None:
    """SCENARIO-AUTO-7438-02: one reveal uses stored probabilities exactly once."""

    controller = exp.FourExpertMixture(exp.build_numeric_experts(seed=65202))
    prediction = controller.predict("event-1", _features(1), index=1, delay=8)
    before = controller.expert_state_hashes
    with pytest.raises(exp.FutureFeedbackError):
        controller.commit_feedback("event-1", 1, visible_at=8)
    committed = controller.commit_feedback("event-1", 1, visible_at=9)
    after = controller.expert_state_hashes
    assert committed["status"] == "committed"
    assert committed["probabilities_used"] == prediction["expert_probabilities"]
    assert committed["probability_source"] == "stored_at_prediction"
    assert committed["hindsight_prediction_count"] == 0
    assert committed["expert_update_count"] == 2
    assert before[exp.FROZEN_SPLINE] == after[exp.FROZEN_SPLINE]
    assert before[exp.FROZEN_GIBBS] == after[exp.FROZEN_GIBBS]
    assert before[exp.ADAPTIVE_SPLINE] != after[exp.ADAPTIVE_SPLINE]
    assert before[exp.ADAPTIVE_GIBBS] != after[exp.ADAPTIVE_GIBBS]
    assert sum(controller.weights.values()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("delay", [0, 8])
def test_scenario_auto_7438_03_delayed_replay_is_causal_and_bounded(
    delay: int, tmp_path: Path
) -> None:
    """SCENARIO-AUTO-7438-03: both delays preserve chronology and byte bounds."""

    result = exp.replay_numeric_stream(
        delay=delay,
        count=20,
        restart_after=9,
        checkpoint_path=tmp_path / f"delay-{delay}.json",
    )
    assert result["future_label_reads"] == 0
    assert result["duplicate_commit_count"] == 0
    assert result["pending_at_end"] == 0
    assert result["peak_pending_count"] <= exp.MAX_PENDING_EVENTS
    assert result["peak_pending_bytes"] > 0
    assert result["restart_equal"] is True
    assert result["restart_prediction_equal"] is True
    assert result["completed_events"] == 20
    assert all(row["prediction_before_reveal"] for row in result["rows"])
    assert all(row["update_duration_ns"] >= 0 for row in result["rows"])
    assert all(row["state_bytes_after"] > 0 for row in result["rows"])


def test_scenario_auto_7438_04_duplicate_restart_and_interrupted_write(
    tmp_path: Path,
) -> None:
    """SCENARIO-AUTO-7438-04: repeats and interrupted replacement do not mutate state."""

    controller = exp.FourExpertMixture(exp.build_numeric_experts(seed=65203))
    controller.predict("event-a", _features(2), index=0, delay=0)
    controller.commit_feedback("event-a", 0, visible_at=0)
    checkpoint = tmp_path / "mixture.json"
    first = controller.save_checkpoint(checkpoint)
    stable_bytes = checkpoint.read_bytes()
    restored = exp.FourExpertMixture.load_checkpoint(checkpoint)
    assert restored.to_state() == controller.to_state()
    state_before_duplicate = restored.state_hash
    duplicate = restored.commit_feedback("event-a", 0, visible_at=10)
    assert duplicate["status"] == "late_duplicate"
    assert duplicate["update_admitted"] is False
    assert restored.state_hash == state_before_duplicate
    assert restored.duplicate_commit_count == 0

    restored.predict("event-b", _features(3), index=1, delay=0)
    restored.commit_feedback("event-b", 1, visible_at=1)
    with pytest.raises(RuntimeError, match="interruption"):
        restored.save_checkpoint(checkpoint, interrupt_before_replace=True)
    assert checkpoint.read_bytes() == stable_bytes
    assert exp.FourExpertMixture.load_checkpoint(checkpoint).state_hash == controller.state_hash
    assert first["byte_size"] == len(stable_bytes)


def test_scenario_auto_7438_05_revocation_replays_from_safe_checkpoint() -> None:
    """SCENARIO-AUTO-7438-05: revocation preserves prediction identities."""

    controller = exp.FourExpertMixture(
        exp.build_numeric_experts(seed=65204), safe_checkpoint_interval=2
    )
    for index in range(7):
        controller.predict(f"event-{index}", _features(index), index=index, delay=0)
        controller.commit_feedback(f"event-{index}", index % 2, visible_at=index)
    original_hashes = controller.prediction_hashes
    receipt = controller.revoke_feedback("event-4")
    assert receipt["status"] == "revoked"
    assert receipt["safe_checkpoint_commit_count"] == 4
    assert receipt["replayed_event_count"] == 2
    assert receipt["prediction_hashes_preserved"] is True
    assert controller.prediction_hashes == original_hashes
    assert "event-4" in controller.revoked_event_ids
    assert controller.commit_feedback("event-4", 0, visible_at=20)["status"] == "revoked"


def test_scenario_auto_7438_06_analytic_expert_controls() -> None:
    """SCENARIO-AUTO-7438-06: constant, harmful, shift, and extreme controls pass."""

    constant = exp.analytic_probability_replay([[0.7, 0.7, 0.7, 0.7]] * 6, [1, 0, 1, 0, 1, 0])
    assert all(row["mixture_probability"] == pytest.approx(0.7) for row in constant["rows"])
    assert constant["final_weights"] == pytest.approx([0.25] * 4)

    harmful = exp.analytic_probability_replay([[0.99, 0.1, 0.1, 0.1]] * 12, [0] * 12)
    assert harmful["final_weights"][0] < min(harmful["final_weights"][1:])

    shift_probabilities = [[0.9, 0.1, 0.6, 0.4]] * 10 + [[0.9, 0.1, 0.6, 0.4]] * 24
    shift_labels = [1] * 10 + [0] * 24
    shifted = exp.analytic_probability_replay(shift_probabilities, shift_labels)
    assert shifted["rows"][9]["weights_after"][0] > shifted["rows"][9]["weights_after"][1]
    assert shifted["final_weights"][1] > shifted["final_weights"][0]

    extreme = exp.analytic_probability_replay(
        [[0.0, 1.0, 1e-300, 1.0 - 1e-300], [1.0, 0.0, 0.5, 0.5]],
        [0, 1],
    )
    assert all(math.isfinite(row["mixture_probability"]) for row in extreme["rows"])
    assert all(math.isfinite(row["energy"]) for row in extreme["rows"])
    assert all(math.isfinite(weight) and weight > 0 for weight in extreme["final_weights"])


def test_scenario_auto_7438_07_full_information_no_share_identity() -> None:
    """SCENARIO-AUTO-7438-07: eta-one no-share loss equals marginal likelihood."""

    probabilities = [
        [0.8, 0.4, 0.6, 0.2],
        [0.7, 0.3, 0.5, 0.9],
        [0.2, 0.8, 0.4, 0.6],
        [0.9, 0.1, 0.7, 0.3],
    ]
    identity = exp.full_information_no_share_identity(probabilities, [1, 0, 1, 1])
    assert identity["absolute_error"] <= 1e-12
    assert identity["mixture_cumulative_log_loss"] == pytest.approx(
        identity["negative_log_marginal_likelihood"], abs=1e-12
    )
    assert identity["deployment_guarantee_inherited"] is False


def test_scenario_auto_7438_08_artifact_reduction_and_mutations(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7438-08: cold reduction binds protocol and readiness evidence."""

    artifact = exp.build_fixture_artifact(
        root=ROOT,
        protocol_path=tmp_path / "protocol.json",
        validation_receipts=_passing_receipts(),
    )
    assert exp.validate_artifact(artifact, root=ROOT) == []
    reduced = exp.independent_reduce(artifact, root=ROOT)
    assert reduced["mixture_prototype_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["promotion_score"] == 0
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["mixture_definition"]["fixed_share"] == 0.01
    assert artifact["hardware_path"]["weight_update_count"] == 4

    changed = deepcopy(artifact)
    changed["learning_control_rows"][0]["future_label_reads"] = 1
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "independent_reduction_mismatch:mixture_prototype_ready_score" in exp.validate_artifact(
        changed, root=ROOT
    )
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:changed"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed, root=ROOT)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.cold_replay(candidate, root=ROOT) == []
    candidate.write_text("[]", encoding="utf-8")
    assert exp.cold_replay(candidate, root=ROOT) == ["artifact_unreadable_or_not_object"]


def test_req_auto_7438_blocked_and_cli_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-AUTO-7438: unavailable inputs block and public modes stay bounded."""

    failed = {
        "check": "source_bytes:missing",
        "upstream": "missing",
        "path": "missing",
        "field": "bytes",
        "operator": "==",
        "expected": "readable_nonempty_bytes",
        "observed": None,
        "passed": False,
    }
    blocked = exp.build_blocked_artifact(failed)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["gate_check_summary"]["observed"] is None
    with pytest.raises(SystemExit, match="--date"):
        exp.parse_args(["--date", "20260919"])

    monkeypatch.setattr(exp, "cold_replay", lambda *_args, **_kwargs: [])
    assert exp.main(["--date", exp.RUN_DATE, "--cold-replay", "candidate.json"]) == 0
    monkeypatch.setattr(exp, "_load_object", lambda _path: {})
    assert exp.main(["--date", exp.RUN_DATE, "--independent-reduce", "candidate.json"]) == 1
    called: list[tuple[Path, str, Path]] = []
    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda root, date, *, output_path: called.append((root, date, output_path)),
    )
    assert exp.main(["--date", exp.RUN_DATE]) == 0
    assert called == [(exp.REPO_ROOT, exp.RUN_DATE, exp.RESULT_PATH)]


def test_req_auto_7438_input_and_probability_guards(tmp_path: Path) -> None:
    """REQ-AUTO-7438: malformed probability and checkpoint inputs fail closed."""

    with pytest.raises(ValueError, match="four probabilities"):
        exp.mixture_probability([0.5], [0.25] * 4)
    with pytest.raises(ValueError, match="weights"):
        exp.mixture_probability([0.5] * 4, [0.25, 0.25, -0.1, 0.6])
    with pytest.raises(ValueError, match="finite"):
        exp.clip_probability(math.nan)
    with pytest.raises(ValueError, match="binary"):
        exp.bernoulli_log_loss(2, 0.5)
    with pytest.raises(ValueError, match="event identity"):
        exp.FourExpertMixture(exp.build_numeric_experts()).predict(
            "", _features(0), index=0, delay=0
        )
    with pytest.raises(ValueError, match="delay"):
        exp.FourExpertMixture(exp.build_numeric_experts()).predict(
            "event", _features(0), index=0, delay=-1
        )
    controller = exp.FourExpertMixture(exp.build_numeric_experts(), max_pending=1)
    controller.predict("one", _features(0), index=0, delay=8)
    with pytest.raises(OverflowError, match="pending"):
        controller.predict("two", _features(1), index=1, delay=8)
    assert controller.commit_feedback("unknown", 0, visible_at=0)["status"] == "unknown_event"
    with pytest.raises(ValueError, match="binary"):
        controller.commit_feedback("one", 3, visible_at=8)

    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        exp.FourExpertMixture.load_checkpoint(bad)
    checkpoint = tmp_path / "checkpoint.json"
    controller.save_checkpoint(checkpoint)
    changed = json.loads(checkpoint.read_text(encoding="utf-8"))
    changed["log_weights"][0] += 1.0
    checkpoint.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        exp.FourExpertMixture.load_checkpoint(checkpoint)


def test_req_auto_7438_defensive_branches_and_cold_reader_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-AUTO-7438: each malformed state and artifact branch fails closed."""

    assert exp._load_object(tmp_path / "missing.json") == {}  # noqa: SLF001
    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]", encoding="utf-8")
    assert exp._load_object(sequence) == {}  # noqa: SLF001
    with pytest.raises(ValueError, match="log weights"):
        exp._normalize_log_weights([0.0])  # noqa: SLF001
    with pytest.raises(ValueError, match="four probabilities"):
        exp.update_log_weights([0.0] * 4, [0.5], 0)
    with pytest.raises(ValueError, match="eta"):
        exp.update_log_weights([0.0] * 4, [0.5] * 4, 0, eta=-1.0)
    with pytest.raises(ValueError, match="fixed share"):
        exp.update_log_weights([0.0] * 4, [0.5] * 4, 0, fixed_share=1.0)

    experts = exp.build_numeric_experts()
    with pytest.raises(ValueError, match="four-name"):
        exp.FourExpertMixture({name: experts[name] for name in exp.EXPERT_NAMES[:-1]})
    with pytest.raises(ValueError, match="capacity"):
        exp.FourExpertMixture(exp.build_numeric_experts(), max_pending=0)
    with pytest.raises(ValueError, match="interval"):
        exp.FourExpertMixture(exp.build_numeric_experts(), safe_checkpoint_interval=0)

    controller = exp.FourExpertMixture(exp.build_numeric_experts())
    state = controller.to_state()
    with pytest.raises(ValueError, match="schema"):
        exp.FourExpertMixture.from_state({**state, "schema": "bad"})
    with pytest.raises(ValueError, match="experts"):
        exp.FourExpertMixture.from_state({**state, "experts": {}})
    controller.predict("hash-event", _features(0), index=0, delay=0)
    changed_state = controller.to_state()
    changed_state["predictions"]["hash-event"]["energy"] += 1.0
    with pytest.raises(ValueError, match="prediction hash"):
        exp.FourExpertMixture.from_state(changed_state)
    assert controller.revoke_feedback("unknown")["status"] == "unknown_event"

    controller = exp.FourExpertMixture(exp.build_numeric_experts())
    controller.predict("rejected", _features(0), index=0, delay=0)
    monkeypatch.setattr(
        controller.experts[exp.ADAPTIVE_GIBBS],
        "commit_feedback",
        lambda *_args, **_kwargs: {"update_admitted": False},
    )
    with pytest.raises(ValueError, match="rejected trusted feedback"):
        controller.commit_feedback("rejected", 0, visible_at=0)
    with pytest.raises(ValueError, match="equal length"):
        exp.analytic_probability_replay([[0.5] * 4], [])
    with pytest.raises(ValueError, match="registered delay"):
        exp.replay_numeric_stream(
            delay=7, count=1, restart_after=0, checkpoint_path=tmp_path / "unused.json"
        )

    artifact = exp.build_fixture_artifact(
        root=ROOT,
        protocol_path=tmp_path / "protocol.json",
        validation_receipts=_passing_receipts(),
    )
    changed = deepcopy(artifact)
    changed["schema"] = "bad"
    assert "declaration_mismatch:schema" in exp.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["verdict_class"] = "invented"
    assert "verdict_class_invalid" in exp.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["field_principles"] = {}
    assert "field_principles_mismatch" in exp.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["verdict_class"] = "blocked"
    changed["honest_verdict"] = "complete_bad"
    assert "blocked_verdict_prefix_invalid" in exp.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["protocol_manifest"] = "not-a-mapping"
    assert any(
        item.startswith("independent_reduction_failed:")
        for item in exp.validate_artifact(changed, root=ROOT)
    )
    changed = deepcopy(artifact)
    changed["fixture_artifact"] = False
    changed["source_artifact_hashes"] = {
        "bad": "not-a-row",
        "missing": {"path": "missing", "sha256": "sha256:none"},
    }
    errors = exp.validate_artifact(changed, root=ROOT)
    assert "source_artifact_hash_row_invalid" in errors
    assert "source_artifact_hash_mismatch:missing" in errors

    span = exp._span("test", 1.0, 0.0, 2)  # noqa: SLF001
    assert span["phase"] == "test" and span["completed_units"] == 2
