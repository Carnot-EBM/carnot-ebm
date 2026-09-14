"""Verify bounded fixed-share selection under delayed feedback.

Spec refs: REQ-CL-7295 and SCENARIO-CL-7295-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_7295_v641_mixture_prototype as exp


def _masks(parameter: int) -> dict[str, int]:
    """Build one complete finite hypothesis for each constraint family."""

    return {family: 1 << parameter for family in exp.FAMILIES}


def _release(index: int, *, label: str = "accept") -> dict[str, object]:
    """Build one label that becomes visible four steps after its request."""

    source_index = 128 + 7 * index
    return {
        "event_id": f"label-{index:03d}",
        "family_id": exp.FAMILIES[index % len(exp.FAMILIES)],
        "numeric_value": index % len(exp.PARAMETER_DOMAIN),
        "observed_label": label,
        "source_index": source_index,
        "release_index": source_index + exp.FEEDBACK_DELAY,
    }


@pytest.fixture(scope="module")
def built_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[exp.ExperimentPaths, dict[str, object]]:
    """Build the complete development fixture once for cold artifact tests."""

    paths = exp.ExperimentPaths.under(tmp_path_factory.mktemp("exp7295"))
    artifact = exp.build_and_seal(exp.REPO_ROOT, paths, progress=True)
    return paths, artifact


def test_req_cl_7295_freezes_contract_and_no_model_work() -> None:
    """REQ-CL-7295 freezes mixture constants, streams, arms, and CPU work."""

    spec = (exp.REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-CL-7295" in spec
    assert len(set(exp.SCENARIO_PATTERN.findall(spec))) == 8
    assert exp.MODEL_SPECS == []
    assert exp.MODEL_INVOKED is False
    assert set(exp.INVOCATION_COUNTS.values()) == {0}
    assert (exp.ETA, exp.FIXED_SHARE) == (0.5, 0.02)
    assert (exp.ARCHIVE_CAP, exp.MEMORY_CAP_BYTES) == (4, 69_632)
    assert (exp.DEVELOPMENT_STREAM_COUNT, exp.EVALUATION_STREAM_COUNT) == (8, 24)
    assert (exp.EVENTS_PER_STREAM, exp.WARMUP_COUNT) == (1_024, 128)
    assert exp.FUTURE_LABEL_POSITIONS == tuple(128 + 7 * index for index in range(128))
    assert exp.ARMS == (
        "fixed_share_mixture",
        "frozen_uniform_voting",
        "reset",
        "unconditional_recognition",
        "label_shuffled_fixed_share",
        "unbounded_memory_reference",
        "frozen_warmup",
    )
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(exp.FIELD_PRINCIPLES)


def test_scenario_cl_7295_update_uses_exact_loss_share_and_ties_abstain() -> None:
    """SCENARIO-CL-7295-UPDATE applies the exact update after prediction."""

    controller = exp.FixedShareController.from_masks(_masks(0))
    controller.install_nominee(_masks(8), birth_index=100)
    before = {row["expert_id"]: row["weight"] for row in controller.experts()}
    event = {
        "event_id": "public",
        "family_id": "lower_bound",
        "numeric_value": 4,
    }
    prediction = controller.predict(event)
    assert prediction in {"accept", "reject", "abstain"}

    receipt = controller.apply_release(_release(0), current_index=132, collect_nominee=False)
    raw = {}
    for row in controller.experts():
        expert_prediction = exp.prototype.predict_masks(row["masks"], _release(0))
        error = int(expert_prediction != "accept")
        raw[row["expert_id"]] = before[row["expert_id"]] * math.exp(-0.5 * error)
    total = sum(raw.values())
    expected = {
        key: (1.0 - exp.FIXED_SHARE) * value / total + exp.FIXED_SHARE / len(raw)
        for key, value in raw.items()
    }
    assert receipt["prediction_preceded_release"] is True
    assert {row["expert_id"]: row["weight"] for row in controller.experts()} == pytest.approx(
        expected
    )

    tied = exp.FixedShareController.from_masks(_masks(0))
    tied.install_nominee(_masks(8), birth_index=100)
    state = tied.state_dict()
    for row in [state["reset_expert"], *state["archives"]]:
        row["weight"] = 0.5
    tied = exp.FixedShareController.from_state(state)
    disagreement = next(
        {
            "event_id": "tie",
            "family_id": "lower_bound",
            "numeric_value": value,
        }
        for value in exp.PARAMETER_DOMAIN
        if exp.prototype.predict_masks(
            _masks(0), {"family_id": "lower_bound", "numeric_value": value}
        )
        != exp.prototype.predict_masks(
            _masks(8), {"family_id": "lower_bound", "numeric_value": value}
        )
    )
    assert tied.predict(disagreement) == "abstain"
    with pytest.raises(exp.MixtureRejected, match="private_authority_in_prediction"):
        tied.predict({**disagreement, "exact_label": "accept"})


def test_scenario_cl_7295_nominee_birth_follows_sixteenth_release() -> None:
    """SCENARIO-CL-7295-NOMINEE uses only one released 16-label block."""

    controller = exp.FixedShareController.from_masks(_masks(0))
    for index in range(exp.NOMINATION_INTERVAL - 1):
        receipt = controller.apply_release(
            _release(index), current_index=132 + 7 * index, collect_nominee=True
        )
        assert receipt["nominee"] is None
    receipt = controller.apply_release(_release(15), current_index=237, collect_nominee=True)

    nominee = receipt["nominee"]
    assert nominee["birth_index"] == 237
    assert nominee["source_release_count"] == 16
    assert nominee["source_release_max_index"] == 237
    assert nominee["future_label_used"] is False
    assert len(controller.archives()) == 1
    assert controller.state_dict()["nomination_buffer"] == []
    assert controller.state_dict()["retained_full_label_history"] is False

    parent = controller.state_bytes()
    with pytest.raises(exp.MixtureRejected, match="release_not_due"):
        controller.apply_release(_release(16), current_index=243)
    assert controller.state_bytes() == parent
    with pytest.raises(exp.MixtureRejected, match="duplicate_release"):
        controller.apply_release(_release(15), current_index=237)
    assert controller.state_bytes() == parent


def test_scenario_cl_7295_nominee_evicts_lowest_then_oldest() -> None:
    """SCENARIO-CL-7295-NOMINEE preserves reset and resolves ties by age."""

    controller = exp.FixedShareController.from_masks(_masks(0))
    for order in range(exp.ARCHIVE_CAP):
        controller.install_nominee(_masks(order + 1), birth_index=200 + order)
    state = controller.state_dict()
    state["reset_expert"]["weight"] = 0.6
    for row in state["archives"]:
        row["weight"] = 0.1
    controller = exp.FixedShareController.from_state(state)
    oldest = controller.archives()[0]["expert_id"]
    reset_id = controller.experts()[0]["expert_id"]

    receipt = controller.install_nominee(_masks(12), birth_index=300)

    assert receipt["evicted_expert_id"] == oldest
    assert reset_id in {row["expert_id"] for row in controller.experts()}
    assert oldest not in {row["expert_id"] for row in controller.experts()}
    assert len(controller.archives()) == exp.ARCHIVE_CAP


def test_scenario_cl_7295_memory_rejection_and_restart_preserve_bytes(tmp_path: Path) -> None:
    """SCENARIO-CL-7295-MEMORY enforces bytes and restart parity."""

    baseline = exp.FixedShareController.from_masks(_masks(0))
    tight_cap = len(baseline.state_bytes()) + 48
    controller = exp.FixedShareController.from_masks(_masks(0), memory_cap_bytes=tight_cap)
    parent = controller.state_bytes()
    with pytest.raises(exp.MixtureRejected, match="mixture_memory_cap"):
        controller.install_nominee(_masks(1), birth_index=200)
    assert controller.state_bytes() == parent

    controller = exp.FixedShareController.from_masks(_masks(0))
    controller.install_nominee(_masks(1), birth_index=200)
    path = tmp_path / "controller.json"
    save_receipt = controller.save(path)
    restored = exp.FixedShareController.load(path)
    event = {"event_id": "restart", "family_id": "upper_bound", "numeric_value": 3}
    assert save_receipt["sha256"] == controller.state_hash()
    assert restored.state_bytes() == controller.state_bytes()
    assert restored.predict(event) == controller.predict(event)

    changed = controller.state_dict()
    changed["archives"] = changed["archives"] * 5
    with pytest.raises(ValueError, match="archive_capacity"):
        exp.FixedShareController.from_state(changed)


def test_scenario_cl_7295_streams_are_fresh_balanced_and_scorer_only(tmp_path: Path) -> None:
    """SCENARIO-CL-7295-STREAMS seals independent authority-separated splits."""

    development = exp.build_stream_views("development")
    evaluation = exp.build_stream_views("evaluation")
    assert exp.stream_conformance_errors(development, "development") == []
    assert exp.stream_conformance_errors(evaluation, "evaluation") == []
    assert development.manifest["strata"] == {
        "separated_recurrence": 4,
        "overlapping_recurrence": 4,
    }
    assert evaluation.manifest["strata"] == {
        "separated_recurrence": 12,
        "overlapping_recurrence": 12,
    }
    assert set(exp.DEVELOPMENT_STREAM_SEEDS).isdisjoint(exp.EVALUATION_STREAM_SEEDS)
    assert len(evaluation.public) == 24 * 1_024
    assert len(evaluation.releases) == 24 * 256
    assert not (set(evaluation.public[0]) & exp.FORBIDDEN_PUBLIC_FIELDS)

    paths = exp.ExperimentPaths.under(tmp_path)
    manifest = exp.seal_streams(paths, development, evaluation)
    loaded = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert loaded == manifest
    assert loaded["scorer_only_evaluation_labels"] is True
    assert loaded["evaluation"]["label_hash"]
    assert "labels" not in loaded["evaluation"]
    assert exp.seal_streams(paths, development, evaluation) == manifest

    changed = deepcopy(evaluation)
    changed.public[0]["observed_label"] = "accept"
    assert "public_authority_leakage" in exp.stream_conformance_errors(changed, "evaluation")
    with pytest.raises(ValueError, match="invalid_stream_kind"):
        exp.build_stream_views("other")


def test_scenario_cl_7295_feedback_replay_is_causal_and_common() -> None:
    """SCENARIO-CL-7295-FEEDBACK preserves order and common nominees."""

    views = exp.build_stream_views("development")
    panel = exp.run_development_panel(views, stream_ids=("development-01",), progress=True)

    assert len(panel.rows) == len(exp.ARMS)
    assert panel.completed_stream_count == 1
    assert (
        exp.development_row_errors(
            panel.event_rows,
            panel.nominee_rows,
            panel.rows,
            ("development-01",),
        )
        == []
    )
    assert all(row["warmup_label_count"] == 128 for row in panel.rows)
    assert all(row["future_label_count"] == 128 for row in panel.rows)
    assert all(row["future_prediction_count"] == 896 for row in panel.rows)
    assert all(row["future_label_read_count"] == 0 for row in panel.rows)
    mixture_nominees = [row for row in panel.nominee_rows if row["arm"] == "fixed_share_mixture"]
    uniform_nominees = [row for row in panel.nominee_rows if row["arm"] == "frozen_uniform_voting"]
    assert [row["candidate_state_hash"] for row in mixture_nominees] == [
        row["candidate_state_hash"] for row in uniform_nominees
    ]
    assert [row["evicted_expert_id"] for row in mixture_nominees] == [
        row["evicted_expert_id"] for row in uniform_nominees
    ]
    assert panel.changed_weight_prediction_difference_count >= 1
    reference = next(row for row in panel.rows if row["arm"] == "unbounded_memory_reference")
    bounded = next(row for row in panel.rows if row["arm"] == "fixed_share_mixture")
    assert reference["maximum_memory_bytes"] > bounded["maximum_memory_bytes"]
    assert reference["bounded_deployment_eligible"] is False


def test_scenario_cl_7295_controls_cover_chronology_bytes_and_restart(tmp_path: Path) -> None:
    """SCENARIO-CL-7295-CONTROLS injects each required causal failure."""

    rows = exp.run_chronology_controls(tmp_path)

    assert {row["control"] for row in rows} == {
        "delayed_release",
        "candidate_birth",
        "deterministic_eviction",
        "zero_future_label_reads",
        "byte_enforcement",
        "restart_parity",
    }
    assert all(row["passed"] is True for row in rows)
    assert all(row["parent_bytes_preserved"] is True for row in rows if row["rejected"])


def test_scenario_cl_7295_terminal_builds_and_cold_reduces(
    built_artifact: tuple[exp.ExperimentPaths, dict[str, object]],
) -> None:
    """SCENARIO-CL-7295-TERMINAL cold-reduces complete per-stream rows."""

    paths, artifact = built_artifact
    assert artifact["status"] == "complete"
    assert artifact["mixture_fixture_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["rows"] == exp.independent_reduce(paths.development_event_rows)
    assert len(artifact["rows"]) == 8 * len(exp.ARMS)
    assert exp.validate_artifact(artifact) == []
    assert artifact["stream_manifest_path"] == str(paths.manifest)
    assert artifact["memory_budget_bytes"]["inherited_limit"] == 69_632
    assert artifact["feedback_schedule"]["future_reveal_count"] == 128
    assert artifact["learning_contract"]["efficacy_gates"]["coverage_lower_delta"] == -0.02
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate_class"] == "cpu_exact_solver_or_simulator"
    assert artifact["reducer_receipt"]["inference_substrate_class"] == "aggregation"


def test_req_cl_7295_validation_rejects_mutated_rows_and_writes_atomically(
    built_artifact: tuple[exp.ExperimentPaths, dict[str, object]], tmp_path: Path
) -> None:
    """REQ-CL-7295 rejects changed evidence before terminal publication."""

    _, artifact = built_artifact
    invalid = deepcopy(artifact)
    invalid["rows"][0]["future_error"] += 1
    invalid["reproducibility_checksum"] = exp.reproducibility_checksum(invalid)
    assert "raw_reduction" in exp.validate_artifact(invalid)
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        exp.write_artifact(tmp_path / "bad.json", invalid)

    output = tmp_path / "good.json"
    exp.write_artifact(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_scenario_cl_7295_preconditions_block_external_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7295-PRECONDITIONS makes external absence terminal."""

    paths = exp.ExperimentPaths.under(tmp_path)
    failed = exp.gate_check("missing", "exp7281", "status", "complete", None)
    monkeypatch.setattr(exp, "collect_preconditions", lambda *args: ([failed], {}))
    artifact = exp.build_and_seal(exp.REPO_ROOT, paths, progress=True)
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["gate_check_summary"]["first_failure"]["field"] == "status"


def test_req_cl_7295_cli_helpers_and_thin_entrypoint(
    built_artifact: tuple[exp.ExperimentPaths, dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7295 keeps terminal orchestration behind a thin entrypoint."""

    paths, artifact = built_artifact
    args = exp._parse_args(["--date", "20260914"])
    assert args.date == "20260914"
    with pytest.raises(SystemExit):
        exp._parse_args(["--date", "20260913"])
    commands = exp._validation_commands(paths.terminal_candidate)
    flattened = [part for command in commands for part in command]
    assert "scripts/check_spec_coverage.py" in flattened
    assert "scripts/adversarial_verify.py" in flattened
    assert "scripts/verdict_row_consistency_lint.py" in flattened

    monkeypatch.setattr(exp, "build_and_seal", lambda *args, **kwargs: deepcopy(artifact))
    monkeypatch.setattr(exp, "_validation_commands", lambda *args: [])
    monkeypatch.setattr(exp.ExperimentPaths, "defaults", classmethod(lambda cls: paths))
    assert exp.main(["--date", "20260914"]) == 0
    assert paths.artifact.exists()

    wrapper = exp.REPO_ROOT / exp.WRAPPER_PATH
    monkeypatch.setattr(exp, "main", lambda: 0)
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(wrapper), run_name="__main__")
    assert result.value.code == 0


def test_req_cl_7295_controller_and_raw_defenses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7295 rejects malformed state, feedback, streams, and raw rows."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    exp.ExperimentPaths.defaults()
    with pytest.raises(ValueError, match="archive_capacity"):
        exp.FixedShareController.from_masks(_masks(0), archive_cap=-1)
    with pytest.raises(ValueError, match="mixture_memory_cap"):
        exp.FixedShareController.from_masks(_masks(0), memory_cap_bytes=1)
    with pytest.raises(ValueError, match="invalid_mixture_state"):
        exp.FixedShareController.from_state({})

    controller = exp.FixedShareController.from_masks(_masks(0))
    base = controller.state_dict()
    mutations = (
        ("invalid_expert", lambda state: state.update(reset_expert=None)),
        ("expert_identity", lambda state: state["reset_expert"].update(state_hash="bad")),
        ("expert_weight", lambda state: state["reset_expert"].update(weight=-1.0)),
        ("expert_weight_sum", lambda state: state["reset_expert"].update(weight=0.5)),
        (
            "nomination_buffer",
            lambda state: state.update(nomination_buffer=[{}] * exp.NOMINATION_INTERVAL),
        ),
        ("full_label_history", lambda state: state.update(label_history=[{}])),
        ("mixture_memory_cap", lambda state: state.update(memory_cap_bytes=1)),
    )
    for expected, mutate in mutations:
        state = deepcopy(base)
        mutate(state)
        with pytest.raises(ValueError, match=expected):
            exp.FixedShareController.from_state(state)

    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="invalid_mixture_state"):
        exp.FixedShareController.load(missing)
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_mixture_state"):
        exp.FixedShareController.load(invalid)

    state = controller.state_dict()
    state["reset_expert"]["weight"] = 0.0
    zero = exp.FixedShareController.__new__(exp.FixedShareController)
    zero._state = state
    with pytest.raises(exp.MixtureRejected, match="zero_expert_weight"):
        zero._renormalize()
    with pytest.raises(exp.MixtureRejected, match="invalid_public_event"):
        controller.predict({"event_id": "bad", "family_id": "bad", "numeric_value": True})

    for parameter in range(exp.ARCHIVE_CAP):
        controller.install_nominee(_masks(parameter + 1), birth_index=200 + parameter)
    with pytest.raises(exp.MixtureRejected, match="forced_eviction_missing"):
        controller.install_nominee(_masks(10), birth_index=300, forced_eviction_id="missing")
    with pytest.raises(exp.MixtureRejected, match="invalid_release"):
        controller.apply_release({}, current_index=132)
    bad_release = _release(0)
    bad_release["observed_label"] = "unknown"
    with pytest.raises(exp.MixtureRejected, match="invalid_release"):
        controller.apply_release(bad_release, current_index=132)

    cap_controller = exp.FixedShareController.from_masks(_masks(0))
    parent = cap_controller.state_bytes()
    monkeypatch.setattr(cap_controller, "_within_cap", lambda: False)
    with pytest.raises(exp.MixtureRejected, match="mixture_memory_cap"):
        cap_controller.apply_release(_release(0), current_index=132, collect_nominee=False)
    assert cap_controller.state_bytes() == parent

    assert exp.stream_conformance_errors(exp.build_stream_views("development"), "bad") == [
        "invalid_stream_kind"
    ]
    with pytest.raises(ValueError, match="incomplete_warmup"):
        exp._warmup_masks([], "missing")
    with pytest.raises(ValueError, match="incomplete_stream"):
        exp.run_development_panel(exp.build_stream_views("development"), stream_ids=("missing",))
    assert exp.development_row_errors([], [], [], ("development-01",))
    with pytest.raises(ValueError, match="raw_rows_unavailable"):
        exp.independent_reduce(tmp_path / "absent.jsonl")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="raw_rows_unavailable"):
        exp.independent_reduce(empty)


def test_req_cl_7295_precondition_and_build_defenses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7295 preserves exact failures in authentication and fixture build."""

    assert exp._task_identity("[") == {}
    assert exp._task_identity("{}") == {}
    assert exp._task_identity("tasks: []") == {}
    assert exp._excluded_experiment({"experiment_id": "exp7295"}, 7295) is True
    assert exp._excluded_experiment({"experiment_ids": [7295]}, 7295) is True
    assert exp._excluded_experiment({"nested": {"experiment_id": 7295}}, 7295) is True
    assert exp._excluded_experiment([{"experiment_id": 1}], 7295) is False

    repo = tmp_path / "repo"
    (repo / "ops").mkdir(parents=True)
    (repo / "ops/exclusion_manifest.yaml").write_text("[", encoding="utf-8")
    checks, hashes = exp.collect_preconditions(repo, exp.ExperimentPaths.under(tmp_path / "out"))
    assert checks
    assert hashes

    passed = exp.gate_check("pass", "test", "field", True, True)
    monkeypatch.setattr(exp, "collect_preconditions", lambda *args: ([passed], {}))
    monkeypatch.setattr(exp, "build_stream_views", lambda kind: object())
    monkeypatch.setattr(exp, "stream_conformance_errors", lambda *args: ["forced"])
    with pytest.raises(ValueError, match="stream_conformance_failed"):
        exp.build_and_seal(exp.REPO_ROOT, exp.ExperimentPaths.under(tmp_path / "stream"))
    monkeypatch.undo()

    monkeypatch.setattr(exp, "collect_preconditions", lambda *args: ([passed], {}))
    monkeypatch.setattr(exp, "build_stream_views", lambda kind: object())
    monkeypatch.setattr(exp, "stream_conformance_errors", lambda *args: [])
    monkeypatch.setattr(exp, "seal_streams", lambda *args: {})
    panel = exp.MixturePanel([], [], [{"forced": True}], 8, 1, 2, 24)
    monkeypatch.setattr(exp, "run_development_panel", lambda *args, **kwargs: panel)
    monkeypatch.setattr(exp, "independent_reduce", lambda *args: [])
    with pytest.raises(ValueError, match="independent_reducer_mismatch"):
        exp.build_and_seal(exp.REPO_ROOT, exp.ExperimentPaths.under(tmp_path / "reduce"))


def test_req_cl_7295_validator_subprocess_and_main_defenses(
    built_artifact: tuple[exp.ExperimentPaths, dict[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7295 records child outcomes and refuses failed validation."""

    paths, artifact = built_artifact
    blocked = exp.gate_check("missing", "upstream", "status", "complete", None)
    blocked_artifact = exp.build_blocked_artifact(
        [blocked],
        {},
        started_at="2026-09-14T00:00:00+00:00",
        completed_at="2026-09-14T00:00:01+00:00",
        duration_s=1.0,
    )
    assert exp.validate_artifact(blocked_artifact) == []

    malformed = deepcopy(artifact)
    malformed["raw_evidence_receipts"]["bad"] = []
    malformed["raw_evidence_receipts"]["development_event_rows"]["path"] = str(
        tmp_path / "missing.jsonl"
    )
    malformed["reproducibility_checksum"] = exp.reproducibility_checksum(malformed)
    errors = exp.validate_artifact(malformed)
    assert "raw_receipts" in errors
    assert "raw_reduction" in errors

    receipt, output = exp._command_receipt([sys.executable, "-c", "print('receipt-ok')"])
    assert receipt["exit_code"] == 0
    assert "receipt-ok" in output
    receipt, _ = exp._command_receipt([sys.executable, "-c", "while True: pass"], timeout_s=0.01)
    assert receipt["exit_code"] == 124

    class FakeStdout:
        def read(self) -> str:
            return "tail-output\n"

    class FakeProcess:
        def __init__(self, stdout: object) -> None:
            self.stdout = stdout
            self.returncode = 0
            self.poll_count = 0

        def poll(self) -> int | None:
            self.poll_count += 1
            return None if self.poll_count == 1 else 0

        def wait(self) -> int:
            return 0

    class FakeSelector:
        def register(self, *args: object) -> None:
            return None

        def select(self, timeout: float) -> list[object]:
            return []

        def close(self) -> None:
            return None

    clock = iter((0.0, 61.0, 62.0))
    monkeypatch.setattr(exp.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(exp.selectors, "DefaultSelector", FakeSelector)
    monkeypatch.setattr(exp.subprocess, "Popen", lambda *args, **kwargs: FakeProcess(FakeStdout()))
    receipt, output = exp._command_receipt(["fake"])
    assert receipt["exit_code"] == 0
    assert output == "tail-output\n"
    monkeypatch.undo()

    monkeypatch.setattr(exp.subprocess, "Popen", lambda *args, **kwargs: FakeProcess(None))
    with pytest.raises(RuntimeError, match="validation_stdout_unavailable"):
        exp._command_receipt(["fake"])
    monkeypatch.undo()

    assert exp.main(["--date", "20260914", "--validate-raw", str(paths.terminal_candidate)]) == 0

    monkeypatch.setattr(exp, "build_and_seal", lambda *args, **kwargs: deepcopy(blocked_artifact))
    monkeypatch.setattr(exp.ExperimentPaths, "defaults", classmethod(lambda cls: paths))
    assert exp.main(["--date", "20260914"]) == 0
    monkeypatch.undo()

    monkeypatch.setattr(exp, "build_and_seal", lambda *args, **kwargs: deepcopy(artifact))
    monkeypatch.setattr(exp.ExperimentPaths, "defaults", classmethod(lambda cls: paths))
    monkeypatch.setattr(
        exp,
        "_validation_commands",
        lambda candidate: [[sys.executable, "-c", "raise SystemExit(1)"]],
    )
    assert exp.main(["--date", "20260914"]) == 1

    monkeypatch.undo()
    monkeypatch.setattr(exp, "build_and_seal", lambda *args, **kwargs: deepcopy(artifact))
    monkeypatch.setattr(exp.ExperimentPaths, "defaults", classmethod(lambda cls: paths))
    monkeypatch.setattr(
        exp,
        "_validation_commands",
        lambda candidate: [[sys.executable, "tests/python"]],
    )
    monkeypatch.setattr(
        exp,
        "_command_receipt",
        lambda command: (
            {
                "command": " ".join(command),
                "exit_code": 1,
                "duration_s": 0.1,
                "log_sha256": exp.transactional.sha256_bytes(b"global failure\n"),
                "classification": "failed",
            },
            "global failure\n",
        ),
    )
    assert exp.main(["--date", "20260914"]) == 0
    observed = exp._load_object(paths.artifact)
    assert observed["global_suite_observation"]["exit_code"] == 1
    assert "repository-wide suite retained unrelated failures" in observed["honest_verdict"]

    monkeypatch.undo()
    old_argv = sys.argv
    sys.argv = [
        str(exp.REPO_ROOT / "python/carnot/experiment_7295_v641_mixture_prototype.py"),
        "--date",
        "20260914",
        "--validate-raw",
        str(paths.terminal_candidate),
    ]
    try:
        with pytest.raises(SystemExit) as result:
            runpy.run_path(
                str(exp.REPO_ROOT / "python/carnot/experiment_7295_v641_mixture_prototype.py"),
                run_name="__main__",
            )
        assert result.value.code == 0
    finally:
        sys.argv = old_argv

    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: ["forced"])
    with pytest.raises(ValueError, match="artifact_validation_failed:forced"):
        exp.build_and_seal(
            exp.REPO_ROOT,
            exp.ExperimentPaths.under(tmp_path / "forced-validation"),
        )
