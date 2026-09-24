"""Tests for REQ-CL-7598, REQ-VERIFY-7598, and their scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from carnot import experiment_7585_v662_portable_service as exp7585
from carnot import experiment_7598_v663_rust_consumer as exp
from carnot.pipeline.calibrated_decision_service import (
    DURABILITY_POLICY,
    CalibratedDecisionService,
    frozen_decision_costs,
)
from carnot.pipeline.probability_calibration_verifier import (
    ProbabilityCalibrationVerifier,
    ProbabilityEvidence,
)


ROOT = Path(__file__).resolve().parents[2]
RUST_BINARY = ROOT / "target/release/portable-recalibration-service"


@pytest.fixture(scope="module", autouse=True)
def built_rust_worker() -> None:
    """Build the exact worker that every process-boundary test exercises."""

    completed = subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "carnot-core",
            "--bin",
            "portable-recalibration-service",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    assert RUST_BINARY.is_file()


def _client(state: Path, **kwargs: Any) -> CalibratedDecisionService:
    return CalibratedDecisionService(
        state_path=state,
        binary_path=RUST_BINARY,
        response_timeout_s=2.0,
        **kwargs,
    )


def test_public_client_predicts_late_feedback_and_restarts(tmp_path: Path) -> None:
    """REQ-CL-7598; SCENARIO-CL-7598-LIFECYCLE."""

    state = tmp_path / "session" / "state.json"
    with _client(state) as client:
        accept = client.predict("accept", 0.01)
        reject = client.predict("reject", 0.99)
        escalated = client.predict("escalate", 0.50)

        assert accept.available is True and accept.action == "accept"
        assert reject.available is True and reject.action == "reject"
        assert escalated.available is True and escalated.action == "escalate"
        assert all(math.isfinite(row.error_probability) for row in (accept, reject, escalated))
        assert all(row.verified is False for row in (accept, reject, escalated))

        first = client.release_feedback("reject", 1)
        late = client.release_feedback("accept", 0)
        assert first.available and first.acknowledged and first.durable
        assert late.available and late.acknowledged and late.durable
        assert first.durability_policy == DURABILITY_POLICY

        duplicate = client.release_feedback("accept", 0)
        assert duplicate.available is False
        assert duplicate.acknowledged is False
        assert duplicate.error == "duplicate_feedback:accept"

    with _client(state) as restarted:
        resumed = restarted.predict("resumed", 0.25)
        assert resumed.available is True
        assert resumed.action in {"accept", "reject", "escalate"}
        assert restarted.owned_pid != 0

    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["sample_count"] == 2
    assert saved["processed_event_ids"] == ["accept", "reject"]


def test_client_rejects_nonfinite_unknown_and_nonbinary_inputs(tmp_path: Path) -> None:
    """SCENARIO-CL-7598-FAILURE rejects inputs before unsafe state changes."""

    state = tmp_path / "state.json"
    with _client(state) as client:
        for value in (float("nan"), float("inf"), -0.01, 1.01):
            result = client.predict(f"bad-{value}", value)
            assert result.available is False
            assert result.action == "escalate"
            assert result.verified is False
            assert math.isfinite(result.error_probability)

        unknown = client.release_feedback("missing", 1)
        invalid = client.release_feedback("missing", 2)
        assert unknown.error == "unknown_prediction:missing"
        assert invalid.error == "binary_label_required"
        assert json.loads(state.read_text(encoding="utf-8"))["sample_count"] == 0


def test_invalid_state_schema_and_crashed_child_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-7598; SCENARIO-VERIFY-7598-UNAVAILABLE."""

    state = tmp_path / "bad-state.json"
    payload = exp7585.SufficientStatisticMap.create().to_payload()
    payload["schema"] = "wrong"
    state.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with _client(state) as invalid:
        result = invalid.predict("bad-schema", 0.2)
        assert result.available is False
        assert result.action == "escalate"
        assert "schema" in (result.error or "")

    healthy = tmp_path / "healthy.json"
    client = _client(healthy)
    pid = client.owned_pid
    client._owned_process.kill()
    client._owned_process.wait(timeout=2)
    result = client.predict("after-crash", 0.2)
    assert result.available is False and result.action == "escalate"
    client.close()
    assert client.owned_pid == pid


@pytest.mark.parametrize(
    ("program", "expected"),
    [
        ("import sys,time;sys.stdin.readline();time.sleep(1)", "response_timeout"),
        ("import sys;sys.stdin.readline();print('[]',flush=True)", "response_not_object"),
    ],
)
def test_bounded_transport_rejects_timeout_and_malformed_reply(
    tmp_path: Path, program: str, expected: str
) -> None:
    """SCENARIO-CL-7598-FAILURE bounds all JSON-lines responses."""

    client = CalibratedDecisionService(
        state_path=tmp_path / f"{expected}.json",
        binary_path=RUST_BINARY,
        response_timeout_s=0.05,
        process_command=(sys.executable, "-u", "-c", program),
    )
    result = client.predict("event", 0.1)
    assert result.available is False
    assert result.action == "escalate"
    assert result.verified is False
    assert result.error == expected
    client.close()


def test_transport_pipe_guards_and_owned_kill(tmp_path: Path) -> None:
    """SCENARIO-CL-7598-FAILURE covers missing pipes and bounded owned cleanup."""

    missing = _client(tmp_path / "missing-pipe.json")
    original_stdin = missing._owned_process.stdin
    missing._owned_process.stdin = None
    assert missing.predict("event", 0.1).error == "worker_pipe_missing"
    missing._owned_process.stdin = original_stdin
    missing.close()

    broken = _client(tmp_path / "broken-pipe.json")
    assert broken._owned_process.stdin is not None
    broken._owned_process.stdin.close()
    assert broken.predict("event", 0.1).error == "process_exited"
    broken.close()

    program = (
        "import signal,sys,time;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "print('ready',flush=True);sys.stdin.readline();time.sleep(10)"
    )
    stubborn = CalibratedDecisionService(
        state_path=tmp_path / "stubborn.json",
        binary_path=RUST_BINARY,
        response_timeout_s=0.05,
        process_command=(sys.executable, "-u", "-c", program),
    )
    assert stubborn.predict("event", 0.1).error == "response_json_invalid"
    stubborn.close()
    assert stubborn._owned_process.returncode is not None


def test_close_ignores_owned_stdin_cleanup_error(tmp_path: Path) -> None:
    """SCENARIO-CL-7598-FAILURE makes pipe cleanup best effort."""

    client = _client(tmp_path / "close-error.json")
    original_stdin = client._owned_process.stdin

    class BrokenClose:
        def close(self) -> None:
            raise OSError("synthetic close failure")

    client._owned_process.stdin = BrokenClose()  # type: ignore[assignment]
    client.close()
    assert client._owned_process.returncode is not None
    assert original_stdin is not None
    original_stdin.close()


@pytest.mark.parametrize("stage", ["before_rename", "after_rename"])
def test_rust_crash_boundaries_emit_no_ack_and_preserve_visible_state(
    tmp_path: Path, stage: str
) -> None:
    """SCENARIO-CL-7598-CRASH exercises the real Rust rename boundary."""

    state = tmp_path / f"rust-{stage}.json"
    client = _client(
        state,
        extra_env={
            "CARNOT_RECALIBRATION_TEST_MODE": "1",
            "CARNOT_RECALIBRATION_TEST_CRASH_STAGE": stage,
        },
    )
    assert client.predict("event", 0.2).available is True
    feedback = client.release_feedback("event", 1)
    assert feedback.available is False
    assert feedback.acknowledged is False
    client.close()

    saved = exp7585._load_state(state)
    expected_count = 0 if stage == "before_rename" else 1
    assert saved.sample_count == expected_count
    assert ("event" in saved.processed_event_ids) is (stage == "after_rename")


@pytest.mark.parametrize("stage", ["before_rename", "after_rename"])
def test_python_reference_has_matching_crash_visibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """SCENARIO-CL-7598-CRASH uses the same Python fsync and rename order."""

    state = tmp_path / f"python-{stage}.json"
    exp7585.initialize_state(state)
    if stage == "before_rename":
        monkeypatch.setattr(
            Path,
            "replace",
            lambda _self, _target: (_ for _ in ()).throw(OSError("before_rename")),
        )
    else:
        monkeypatch.setattr(
            exp7585,
            "_directory_fsync",
            lambda _path: (_ for _ in ()).throw(OSError("after_rename")),
        )
    result = exp7585.run_python_service_request(
        {
            "operation": "trace",
            "state_path": str(state),
            "events": [{"event_id": "event", "probability": 0.2, "label": 1}],
        }
    )
    assert result["ok"] is False
    assert stage in result["error"]
    expected_count = 0 if stage == "before_rename" else 1
    assert exp7585._load_state(state).sample_count == expected_count


def test_probability_surface_requires_an_explicit_service(tmp_path: Path) -> None:
    """REQ-VERIFY-7598; SCENARIO-VERIFY-7598-OPT-IN and -CALL."""

    verifier = ProbabilityCalibrationVerifier(tolerance=0.05)
    baseline = verifier.score("1 out of 100 comparable cases", "P(error)=1%")
    assert baseline.verdict == "pass"

    with _client(tmp_path / "surface.json") as service:
        decision = verifier.decide_with_service(
            service=service,
            event_id="surface-event",
            error_probability=0.01,
        )
    assert decision.available is True
    assert decision.action == "accept"
    assert decision.verified is False


def test_frozen_costs_are_finite_and_tie_order_is_stable() -> None:
    """REQ-CL-7598 keeps the registered action costs unchanged."""

    assert frozen_decision_costs(0.01) == {
        "escalate": 0.2,
        "accept": 0.05,
        "reject": 0.99,
    }
    assert frozen_decision_costs(0.5) == {
        "escalate": 0.2,
        "accept": 2.5,
        "reject": 0.5,
    }
    with pytest.raises(ValueError, match="finite_probability_required"):
        frozen_decision_costs(float("nan"))


def test_client_constructor_and_duplicate_guards(tmp_path: Path) -> None:
    """SCENARIO-CL-7598-FAILURE covers local lifecycle guards."""

    with pytest.raises(ValueError, match="positive_response_timeout_required"):
        CalibratedDecisionService(
            state_path=tmp_path / "never.json",
            binary_path=RUST_BINARY,
            response_timeout_s=0.0,
        )
    with pytest.raises(FileNotFoundError):
        CalibratedDecisionService(
            state_path=tmp_path / "never.json",
            binary_path=tmp_path / "missing-worker",
        )
    with _client(tmp_path / "duplicates.json") as client:
        assert client.predict("", 0.2).error == "event_id_required"
        assert client.predict("pending", 0.2).available is True
        assert client.predict("pending", 0.2).error == "duplicate_prediction:pending"
        assert client.release_feedback("pending", 1).durable is True
        assert client.predict("pending", 0.2).error == "duplicate_feedback:pending"
        client.close()


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("not-json", "response_json_invalid"),
        ('{"ok":false,"error":"typed_failure"}', "typed_failure"),
        ('{"ok":true,"predictions":[]}', "prediction_count_invalid"),
        ('{"ok":true,"predictions":[1]}', "prediction_not_object"),
        (
            '{"ok":true,"predictions":[{"event_id":"event","probability":"bad","action":"accept"}]}',
            "prediction_probability_invalid",
        ),
        (
            '{"ok":true,"predictions":[{"event_id":"wrong","probability":0.1,"action":"accept"}]}',
            "prediction_contract_invalid",
        ),
    ],
)
def test_client_rejects_malformed_prediction_contracts(
    tmp_path: Path, reply: str, expected: str
) -> None:
    """REQ-VERIFY-7598 keeps malformed service replies unavailable."""

    program = f"import sys;sys.stdin.readline();print({reply!r},flush=True)"
    client = CalibratedDecisionService(
        state_path=tmp_path / f"{expected}.json",
        binary_path=RUST_BINARY,
        response_timeout_s=1.0,
        process_command=(sys.executable, "-u", "-c", program),
    )
    result = client.predict("event", 0.1)
    assert result.error == expected
    assert result.action == "escalate"
    client.close()


def test_client_rejects_failed_and_nondurable_feedback(tmp_path: Path) -> None:
    """SCENARIO-CL-7598-FAILURE requires a complete durable acknowledgment."""

    state = tmp_path / "failed-feedback.json"
    with _client(state) as client:
        assert client.predict("event", 0.2).available is True
        state.write_text("[]\n", encoding="utf-8")
        failed = client.release_feedback("event", 1)
        assert failed.available is False
        assert "state_json_invalid" in (failed.error or "")

    predict_reply = {
        "ok": True,
        "predictions": [{"event_id": "event", "probability": 0.1, "action": "escalate"}],
        "stage_ns": {},
    }
    ack_reply = {"ok": True, "acknowledgments": [], "state": {"processed_event_ids": []}}
    program = (
        "import json,sys;"
        f"replies={json.dumps([predict_reply, ack_reply])!r};replies=json.loads(replies);"
        "\nfor reply in replies:\n sys.stdin.readline();print(json.dumps(reply),flush=True)"
    )
    client = CalibratedDecisionService(
        state_path=tmp_path / "nondurable.json",
        binary_path=RUST_BINARY,
        response_timeout_s=1.0,
        process_command=(sys.executable, "-u", "-c", program),
    )
    assert client.predict("event", 0.1).available is True
    assert client.release_feedback("event", 1).error == "durable_acknowledgment_invalid"
    client.close()


def test_probability_parser_defensive_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-1414 remains fully covered after the additive caller change."""

    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        ProbabilityCalibrationVerifier(tolerance=-0.1)
    verifier = ProbabilityCalibrationVerifier()
    assert verifier.parse_claim("nothing probabilistic") is None
    assert verifier.parse_claim("There is a 25% chance of rain") is not None
    assert verifier.parse_claim("probability of rain is 0.25") is not None
    assert verifier.score("no evidence", "not a claim").verdict == "abstain"
    assert verifier.score("P(error)=20%", "P(error)=20%").verdict == "abstain"
    evidence = verifier.extract_evidence(
        "2 out of 1 similar cases; base rate is 200%; 200% among comparable cases"
    )
    assert evidence == []
    assert verifier._parse_probability("bad") is None
    assert verifier._parse_probability("-1") is None
    assert verifier._weighted_mean([ProbabilityEvidence(0.5, 0.0, "zero", "private")]) == 0.0
    monkeypatch.setattr(verifier, "_parse_probability", lambda _raw: None)
    assert verifier.parse_claim("P(error)=20%") is None
    assert verifier.extract_claims("P(error)=20%") == []

    overlap = ProbabilityCalibrationVerifier()
    monkeypatch.setattr(overlap, "_PERCENT_CHANCE", overlap._P_CLAIM)
    assert len(overlap.extract_claims("P(error)=20%")) == 1
    assert overlap.extract_evidence("20% among comparable cases")[0].probability == 0.2


def _comparison_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in ("cold", "warm"):
        for batch_size in (1, 8):
            for repeat in range(30):
                pair_id = f"{mode}:{batch_size}:{repeat}"
                for arm, elapsed in (
                    ("rust", 1_000 + repeat),
                    ("python_service", 3_000 + repeat),
                    ("python_inprocess", 2_000 + repeat),
                ):
                    rows.append(
                        {
                            "row_type": "consumer_comparison",
                            "unit_id": pair_id,
                            "pair_id": pair_id,
                            "mode": mode,
                            "batch_size": batch_size,
                            "repeat": repeat,
                            "arm": arm,
                            "seed": exp.RANDOM_SEED + repeat,
                            "numerator": elapsed,
                            "denominator": batch_size,
                            "metric": elapsed,
                            "metric_name": "whole_consumer_ns",
                            "metric_direction": "lower_is_better",
                            "setup_ns": 200 if mode == "cold" else 0,
                            "per_request_ns": elapsed / batch_size,
                            "kernel_ns": 100,
                            "censored": False,
                            "missing": False,
                            "provenance": "private_equal_durability_fixture",
                            "durability_policy": DURABILITY_POLICY,
                            "decision_parity": True,
                            "reload_parity": True,
                            "durable_acknowledgments": batch_size,
                        }
                    )
    return rows


def _receipts() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "command": f"private {name}",
            "command_argv": ["private", name],
            "scope": "private_fixture",
            "exit_code": 0,
            "duration_s": 0.01,
            "log_path": f"/tmp/{name}.log",
            "log_sha256": "sha256:" + "a" * 64,
            "passed": True,
            "timed_out": False,
            "output_tail": "private fixture",
        }
        for name in (*exp.REQUIRED_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES)
    ]


def test_preconditions_authenticate_exp7585_and_require_current_requalification() -> None:
    """SCENARIO-CL-7598-PARITY rejects unauthenticated historical reuse."""

    context = exp.collect_preconditions(ROOT)
    assert context["blocker"] is None
    assert context["exp7585"]["portable_parity_score"] == 1
    assert context["exp7585"]["flagged_adversarial"] is False
    assert context["historical_equal_durability"] is True
    assert context["requalification_required"] is True
    assert all(row["passed"] for row in context["rows"])


def test_comparison_reducer_selects_fastest_eligible_comparator() -> None:
    """SCENARIO-CL-7598-BENCHMARK uses complete paired public-call rows."""

    rows = _comparison_rows()
    summary = exp.summarize_consumer_rows(rows)

    assert set(summary) == {"cold:1", "cold:8", "warm:1", "warm:8"}
    assert all(row["pair_count"] == 30 for row in summary.values())
    assert all(row["primary_comparator"] == "python_inprocess" for row in summary.values())
    assert all(row["paired_ratio"]["lower95"] > 1.0 for row in summary.values())
    assert all(
        row["rust"]["p50_ns"] < row["python_inprocess"]["p50_ns"] for row in summary.values()
    )

    missing = deepcopy(rows[:-1])
    with pytest.raises(ValueError, match="consumer_pair_incomplete"):
        exp.summarize_consumer_rows(missing)
    changed = deepcopy(rows)
    changed[0]["durability_policy"] = "memory_only"
    with pytest.raises(ValueError, match="durability_policy_mismatch"):
        exp.summarize_consumer_rows(changed)
    ineligible = deepcopy(rows)
    for row in ineligible:
        if row["arm"] != "rust":
            row["decision_parity"] = False
    with pytest.raises(ValueError, match="eligible_comparator_missing"):
        exp.summarize_consumer_rows(ineligible)
    with pytest.raises(ValueError, match="percentile_requires_values"):
        exp._percentile([], 0.5)
    assert exp._percentile([3.0], 0.5) == 3.0
    with pytest.raises(ValueError, match="paired_ratio_requires_30_blocks"):
        exp._paired_ratio_interval([2.0], exp.RANDOM_SEED)
    malformed = {"consumer_comparison_rows": [{"bad": True}], "validation_receipts": []}
    assert exp.independent_reduce(malformed)["measurement_complete"] is False


def test_benchmark_outcome_comparison_handles_structured_state() -> None:
    """SCENARIO-CL-7598-BENCHMARK compares numerical and exact state fields."""

    state = exp7585.SufficientStatisticMap.create().to_payload()
    result = {
        "decisions": [("event", 0.25, "escalate")],
        "resumed": (0.37, "escalate"),
        "state": state,
    }
    results = {arm: deepcopy(result) for arm in exp.ARMS}
    assert exp._outcomes_match(results) == (True, True)

    results["rust"]["state"]["sample_count"] = 1
    assert exp._outcomes_match(results) == (True, False)


def test_fixture_artifact_separates_readiness_benefit_and_quality(tmp_path: Path) -> None:
    """REQ-CL-7598; SCENARIO-CL-7598-TERMINAL."""

    artifact = exp.build_test_artifact(
        tmp_path,
        rows=_comparison_rows(),
        validation_receipts=_receipts(),
    )
    reduction = exp.independent_reduce(artifact)

    assert reduction["consumer_ready_score"] == 1
    assert reduction["consumer_speed_benefit_score"] == 1
    assert artifact["honest_verdict"].startswith("complete_positive_")
    assert artifact["verdict_class"] == "positive"
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["target_model"] == "none:no_model_load"
    assert artifact["no_model_load"] is True
    assert artifact["production_default_changed"] is False
    assert artifact["calibration_quality_claimed"] is False
    assert artifact["verifier_is_oracle"] is False
    assert exp.validate_artifact(artifact, root=tmp_path, verify_sources=False) == []

    null_rows = deepcopy(_comparison_rows())
    for row in null_rows:
        if row["arm"] == "rust":
            row["metric"] = row["numerator"] = 4_000 + row["repeat"]
    null_artifact = exp.build_test_artifact(
        tmp_path,
        rows=null_rows,
        validation_receipts=_receipts(),
    )
    assert null_artifact["consumer_ready_score"] == 1
    assert null_artifact["consumer_speed_benefit_score"] == 0
    assert null_artifact["verdict_class"] == "null"
    assert exp.validate_artifact(null_artifact, root=tmp_path, verify_sources=False) == []


def test_artifact_mutations_and_blocked_gate_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CL-7598-TERMINAL preserves every gate operand."""

    artifact = exp.build_test_artifact(
        tmp_path,
        rows=_comparison_rows(),
        validation_receipts=_receipts(),
    )
    mutations = [
        ("artifact_identity_mismatch", lambda value: value.update(schema="bad")),
        ("task_binding_mismatch", lambda value: value.update(run_date="bad")),
        ("model_specs_not_empty", lambda value: value.update(MODEL_SPECS=["bad"])),
        ("no_model_load_invalid", lambda value: value.update(no_model_load=False)),
        (
            "inference_substrate_class_invalid",
            lambda value: value.update(inference_substrate_class="live_llm"),
        ),
        (
            "planned_inference_substrate_class_invalid",
            lambda value: value.update(planned_inference_substrate_class="live_llm"),
        ),
        ("consumer_ready_score_mismatch", lambda value: value.update(consumer_ready_score=0)),
        (
            "consumer_speed_benefit_score_mismatch",
            lambda value: value.update(consumer_speed_benefit_score=0),
        ),
        (
            "calibration_quality_claim_invalid",
            lambda value: value.update(calibration_quality_claimed=True),
        ),
        (
            "production_default_changed_invalid",
            lambda value: value.update(production_default_changed=True),
        ),
        (
            "empirical_head_install_invalid",
            lambda value: value.update(empirical_head_installed=True),
        ),
        ("arc_default_changed_invalid", lambda value: value.update(arc_agent_changed=True)),
        (
            "verifier_oracle_declaration_invalid",
            lambda value: value.update(verifier_is_oracle=True),
        ),
        ("comparison_rows_mismatch", lambda value: value.update(rows=[])),
        (
            "current_invocation_claim_invalid",
            lambda value: value["invocation_counts"].update(generation_calls=1),
        ),
        (
            "acceptance_gate_results_mismatch",
            lambda value: value.update(acceptance_gate_results=[]),
        ),
        (
            "score_not_bare_numeric:consumer_ready_score",
            lambda value: value.update(consumer_ready_score=True),
        ),
        (
            "unblocked_gate_summary_must_be_null",
            lambda value: value.update(gate_check_summary={}),
        ),
        ("verdict_class_invalid", lambda value: value.update(verdict_class="other")),
        (
            "flagged_adversarial_invalid",
            lambda value: value.update(flagged_adversarial=True),
        ),
        (
            "current_binary_requalification_invalid",
            lambda value: value.update(current_binary_requalification={}),
        ),
        (
            "field_principles_incomplete",
            lambda value: value["field_principles"].pop("consumer_ready_score"),
        ),
    ]
    for expected, mutate in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        assert expected in exp.validate_artifact(changed, root=tmp_path, verify_sources=False)

    blocker = {
        "check": "exp7585_portable_parity_score",
        "upstream": "Exp7585",
        "path": exp.UPSTREAM_PATH.as_posix(),
        "field": "portable_parity_score",
        "op": "eq",
        "expected": 1,
        "observed": 0,
    }
    blocked = exp.build_blocked_artifact(blocker, tmp_path, _receipts())
    assert blocked["honest_verdict"] == "complete_blocked_exp7585_portable_parity_score"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"] == blocker
    assert blocked["consumer_ready_score"] == 0
    assert blocked["consumer_speed_benefit_score"] == 0
    assert exp.validate_artifact(blocked, root=tmp_path, verify_sources=False) == []
    blocked_bad = deepcopy(blocked)
    blocked_bad["gate_check_summary"] = {"check": "short"}
    blocked_bad["reproducibility_checksum"] = exp.reproducibility_checksum(blocked_bad)
    assert "external_blocker_incomplete" in exp.validate_artifact(
        blocked_bad, root=tmp_path, verify_sources=False
    )
    blocked_ready = deepcopy(blocked)
    blocked_ready["consumer_ready_score"] = 1
    blocked_ready["reproducibility_checksum"] = exp.reproducibility_checksum(blocked_ready)
    assert "blocked_measurement_must_be_unready" in exp.validate_artifact(
        blocked_ready, root=tmp_path, verify_sources=False
    )


def test_artifact_validation_contains_reducer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7598-TERMINAL converts reducer exceptions into validation errors."""

    artifact = exp.build_test_artifact(
        tmp_path,
        rows=_comparison_rows(),
        validation_receipts=_receipts(),
    )
    monkeypatch.setattr(exp, "independent_reduce", lambda _value: 1 / 0)
    errors = exp.validate_artifact(artifact, root=tmp_path, verify_sources=False)
    assert "independent_reduction_failed" in errors
    assert "comparison_summary_mismatch" in errors


def test_source_hash_and_checksum_detect_changed_evidence(tmp_path: Path) -> None:
    """REQ-CL-7598 binds current binary and source bytes."""

    source = tmp_path / "source.txt"
    source.write_text("first\n", encoding="utf-8")
    artifact = exp.build_test_artifact(
        tmp_path,
        rows=_comparison_rows(),
        validation_receipts=_receipts(),
    )
    artifact["source_artifact_hashes"] = {"source.txt": exp.source_row(source, tmp_path)}
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    assert exp.validate_artifact(artifact, root=tmp_path) == []

    source.write_text("second\n", encoding="utf-8")
    assert "source_hash_invalid:source.txt" in exp.validate_artifact(artifact, root=tmp_path)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(
        changed, root=tmp_path, verify_sources=False
    )
    malformed = deepcopy(artifact)
    malformed["source_artifact_hashes"] = {"bad": 3}
    malformed["reproducibility_checksum"] = exp.reproducibility_checksum(malformed)
    assert "source_row_invalid:bad" in exp.validate_artifact(malformed, root=tmp_path)

    binary = tmp_path / "binary"
    binary.write_bytes(b"\xff\xfe\x00")
    inside = exp.source_row(binary, tmp_path)
    outside = exp.source_row(binary, tmp_path / "other-root")
    assert inside["path"] == "binary"
    assert outside["path"] == str(binary.resolve())
    assert inside["sha256"] == outside["sha256"]


def test_validation_plan_is_scoped_and_private(tmp_path: Path) -> None:
    """REQ-CL-7598 freezes affected tests and changed-module coverage."""

    commands = exp.build_validation_commands(ROOT, tmp_path / "private")
    names = [row.name for row in commands]
    assert names == list(exp.REQUIRED_CHECK_NAMES)
    joined = [argument for row in commands for argument in row.argv]
    assert exp.TEST_PATH.as_posix() in joined
    assert "tests/python" not in joined
    assert "--fail-under=100" in joined
    assert any(argument.startswith("--basetemp=/tmp/") for argument in joined)
    assert all(Path(path).parent.exists() for path in exp.private_basetemps(commands))

    terminal = exp.terminal_commands(tmp_path / "candidate.json", ROOT)
    assert [row.name for row in terminal] == list(exp.TERMINAL_CHECK_NAMES)
    assert terminal[-1].argv[-2] == "--strict"


def test_cold_replay_parser_and_python_worker_modes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-7598-TERMINAL keeps the CLI thin and replayable."""

    artifact = exp.build_test_artifact(
        tmp_path,
        rows=_comparison_rows(),
        validation_receipts=_receipts(),
    )
    candidate = tmp_path / "candidate.json"
    exp.atomic_json(candidate, artifact)
    common = ["--date", exp.RUN_DATE, "--root", str(tmp_path), "--no-source-check"]
    assert exp.main([*common, "--cold-replay", str(candidate)]) == 0
    assert exp.main([*common, "--independent-reduce", str(candidate)]) == 0
    assert '"valid": true' in capsys.readouterr().out.lower()

    args = exp.parse_args(["--date", exp.RUN_DATE, "--root", str(ROOT)])
    assert args.root == ROOT
    assert args.output == exp.RESULT_PATH
    with pytest.raises(ValueError, match="run_date_must_equal_20260924"):
        exp.parse_args(["--date", "20260923", "--root", str(ROOT)])

    state = tmp_path / "python-worker.json"
    exp7585.initialize_state(state)
    response = exp.run_python_service_request(
        {
            "operation": "predict",
            "state_path": str(state),
            "queries": [{"event_id": "e", "probability": 0.1}],
        }
    )
    assert response["ok"] is True
    assert response["predictions"][0]["event_id"] == "e"
    assert exp.run_python_service_request({"operation": "bad"})["error"] == "unknown_operation"
    fresh = tmp_path / "auto-init.json"
    auto = exp.run_python_service_request(
        {
            "operation": "predict",
            "state_path": str(fresh),
            "queries": [{"event_id": "same", "probability": 0.1}],
        }
    )
    assert auto["ok"] is True and fresh.is_file()
    duplicate = exp.run_python_service_request(
        {
            "operation": "predict",
            "state_path": str(fresh),
            "queries": [
                {"event_id": "same", "probability": 0.1},
                {"event_id": "same", "probability": 0.1},
            ],
        }
    )
    assert duplicate["error"] == "duplicate_feedback:same"
    assert (
        exp.run_python_service_request(
            {"operation": "predict", "state_path": str(fresh), "queries": [{"bad": True}]}
        )["ok"]
        is False
    )

    invalid = tmp_path / "invalid-candidate.json"
    invalid.write_text("[]\n", encoding="utf-8")
    assert exp.cold_replay(invalid, root=tmp_path) == ["artifact_not_object"]
    assert exp.independent_replay(invalid, root=tmp_path) == ["artifact_not_object"]
    changed = deepcopy(artifact)
    changed["independent_reduction"] = {}
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    exp.atomic_json(invalid, changed)
    assert "independent_reduction_mismatch" in exp.independent_replay(
        invalid, root=tmp_path, verify_sources=False
    )

    monkeypatch.setattr(exp, "python_service_worker_loop", lambda _stdin, _stdout: 7)
    assert exp.main([*common, "--python-service-worker"]) == 7
    called: dict[str, Any] = {}

    def fake_run(root: Path, date: str, *, output_path: Path) -> dict[str, Any]:
        called.update(root=root, date=date, output=output_path)
        return {}

    monkeypatch.setattr(exp, "run_experiment", fake_run)
    assert exp.main(["--date", exp.RUN_DATE, "--root", str(tmp_path)]) == 0
    assert called["output"] == tmp_path / exp.RESULT_PATH
