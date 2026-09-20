"""Tests for REQ-ARC-WMTE-7463 and its four named scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
import zipfile

import pytest

from carnot import experiment_7463_v654_semif_e0_logprob_parity as subject


REQ = "REQ-ARC-WMTE-7463"


def _pair(index: int, *, changed: bool = False) -> dict[str, object]:
    native = {"A": 0.8, "B": 0.2}
    server = {"A": 0.8, "B": 0.2}
    if changed:
        server = {"A": 0.2, "B": 0.8}
    return {
        "row_kind": "parity_pair",
        "unit_id": f"development-{index:02d}",
        "disposition": "complete",
        "native": {"probabilities": native, "option_logits": {"A": 4.0, "B": 2.0}},
        "local_server": {
            "probabilities": server,
            "option_logits": {"A": 4.0, "B": 2.0},
        },
        "input_ids_equal": True,
        "weights_equal": True,
        "quantization_equal": True,
        "tokenizer_equal": True,
        "logit_bias": None,
        "emitted_tokens": 1,
        "local_parity_score": 0.4 if changed else 1.0,
    }


def test_frozen_prompts_are_disjoint_balanced_and_hash_stable() -> None:
    """REQ-ARC-WMTE-7463: the 64 development units are frozen before inference."""

    prompts = subject.freeze_development_prompts()

    assert len(prompts) == 64
    assert len({row["unit_id"] for row in prompts}) == 64
    assert len({row["prompt"] for row in prompts}) == 64
    assert [row["expected_label"] for row in prompts].count("A") == 32
    assert [row["expected_label"] for row in prompts].count("B") == 32
    assert subject.canonical_hash(prompts) == subject.DEVELOPMENT_PROMPTS_SHA256


def test_native_controls_cover_both_orders() -> None:
    """SCENARIO-ARC-WMTE-7463-NATIVE-CONTROLS has exactly eight ordered cases."""

    controls = subject.freeze_positive_controls()

    assert len(controls) == 8
    assert {row["order"] for row in controls} == {"original", "reversed"}
    assert all(row["expected_label"] in {"A", "B"} for row in controls)


def test_distribution_requires_both_finite_option_logits() -> None:
    """SCENARIO-ARC-WMTE-7463-LOCAL-PARITY never invents a missing probability."""

    assert subject.option_distribution({"A": 2.0, "B": 1.0}) == pytest.approx(
        {"A": 0.7310585786, "B": 0.2689414214}
    )
    with pytest.raises(ValueError, match="missing_option_logits"):
        subject.option_distribution({"A": 2.0})
    with pytest.raises(ValueError, match="nonfinite_option_logit"):
        subject.option_distribution({"A": float("nan"), "B": 1.0})

    assert subject._lower_confidence_bound(0, 64) == 0.0
    assert 0.0 < subject._lower_confidence_bound(31, 64) < 0.5


def test_parity_reducer_computes_bounds_tv_and_gate() -> None:
    """REQ-ARC-WMTE-7463 reduces complete raw pairs without trusting summaries."""

    rows = [_pair(index) for index in range(64)]
    reduced = subject.reduce_parity_rows(rows, expected_units=64)

    assert reduced == {
        "planned": 64,
        "complete": 64,
        "unavailable": 0,
        "argmax_agreements": 64,
        "argmax_agreement": 1.0,
        "argmax_agreement_95_lower": pytest.approx(0.954270),
        "median_tv_distance": 0.0,
        "all_options_present": True,
        "runtime_identity_equal": True,
        "passed": True,
    }
    assert subject.reduce_parity_rows(rows[:40], expected_units=40)["passed"] is False


def test_parity_reducer_marks_missing_and_mismatched_rows_unavailable() -> None:
    """SCENARIO-ARC-WMTE-7463-LOCAL-PARITY fails closed on incomplete evidence."""

    rows = [_pair(index) for index in range(64)]
    del rows[0]["local_server"]["option_logits"]["B"]  # type: ignore[index]
    rows[1]["input_ids_equal"] = False
    rows[2] = _pair(2, changed=True)

    reduced = subject.reduce_parity_rows(rows, expected_units=64)

    assert reduced["complete"] == 62
    assert reduced["unavailable"] == 2
    assert reduced["argmax_agreements"] == 61
    assert reduced["passed"] is False

    assert subject._valid_pair({}) is False
    missing_probs = _pair(0)
    del missing_probs["native"]["probabilities"]  # type: ignore[index]
    assert subject._valid_pair(missing_probs) is False


def test_scored_manifest_requires_mounted_wheel_bytes(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7463-SCORED-RUNTIME-BLOCK rejects host-wheel substitution."""

    manifest = tmp_path / "kernel-metadata.json"
    manifest.write_text(
        json.dumps({"dataset_sources": ["iancblenke/carnot-vllm-wheels-py312"]}),
        encoding="utf-8",
    )
    host_metadata = tmp_path / "host-vllm-METADATA"
    host_metadata.write_text("Name: vllm\nVersion: 0.29.0\n", encoding="utf-8")

    blocked = subject.inspect_scored_runtime(
        manifest, tmp_path / "absent-kaggle-input", host_metadata=host_metadata
    )

    assert blocked["available"] is False
    assert blocked["failed_check"] == "exact_mounted_vllm_wheel_bytes"
    assert blocked["host_wheel_is_evidence"] is False
    assert blocked["observed"] == []
    assert blocked["proposed_request_change"]["max_tokens"] == 1
    assert blocked["proposed_request_change"]["logprobs"] >= 2
    assert blocked["proposed_request_change"]["preserve_response_field"] == (
        "choices[0].logprobs.top_logprobs[0]"
    )

    wheel_dir = tmp_path / "mounted" / "wheels"
    wheel_dir.mkdir(parents=True)
    wheel = wheel_dir / "vllm-0.29.0-cp312-cp312-manylinux.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("vllm-0.29.0.dist-info/METADATA", "Name: vllm\nVersion: 0.29.0\n")
    available = subject.inspect_scored_runtime(manifest, tmp_path / "mounted")
    assert available["available"] is True
    assert available["wheel"]["sha256"] == subject.sha256_file(wheel)

    manifest.write_text(json.dumps({"dataset_sources": []}), encoding="utf-8")
    undeclared = subject.inspect_scored_runtime(manifest, tmp_path / "absent")
    assert undeclared["failed_check"] == "scored_manifest_vllm_dataset_source"


def test_scores_separate_native_local_and_scored_runtime() -> None:
    """REQ-ARC-WMTE-7463 keeps the three bare readiness scores independent."""

    controls = [{"correct": True, "nonuniform": True, "offload_observed": True} for _ in range(8)]
    local = {"passed": True}

    assert subject.derive_scores(controls, local, {"passed": False}) == {
        "native_readout_ready_score": 1,
        "local_runtime_parity_score": 1,
        "scored_runtime_parity_score": 0,
    }
    controls[0]["nonuniform"] = False
    assert (
        subject.derive_scores(controls, local, {"passed": True})["native_readout_ready_score"] == 0
    )


def test_artifact_fixture_validates_and_mutations_fail(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7463-TERMINAL independently checks identity and reductions."""

    rows = [_pair(index) for index in range(64)]
    controls = [{"correct": True, "nonuniform": True, "offload_observed": True} for _ in range(8)]
    artifact = subject.build_artifact_for_test(rows, controls)

    assert subject.validate_artifact(artifact, require_validation=False) == []
    assert artifact["MODEL_SPECS"] == [subject.MODEL_HF_ID]
    assert artifact["model_specs"] == [subject.MODEL_HF_ID]
    assert artifact["inference_substrate_class"] == "model_bounded_generation"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["scored_runtime_parity_score"] == 0
    assert artifact["gate_check_summary"]["first_failure"]["check"] == (
        "exact_scored_runtime_available"
    )

    changed = deepcopy(artifact)
    changed["local_runtime_parity_score"] = 0
    assert "score_mismatch:local_runtime_parity_score" in subject.validate_artifact(
        changed, require_validation=False
    )
    changed = deepcopy(artifact)
    changed["rows"][0]["local_server"]["probabilities"] = {"A": 0.1, "B": 0.9}
    assert "local_reduction_mismatch" in subject.validate_artifact(
        changed, require_validation=False
    )
    assert subject.validate_artifact(None, require_validation=False) == ["artifact_not_object"]

    for field, expected_error in (
        ("schema", "identity_mismatch:schema"),
        ("acceptance_gate_results", "gate_summary_mismatch"),
        ("field_principles", "field_principles_mismatch"),
        ("verdict_class", "verdict_class_invalid"),
    ):
        changed = deepcopy(artifact)
        changed[field] = "broken"
        assert expected_error in subject.validate_artifact(changed, require_validation=False)


def test_validation_manifest_and_command_plan_are_narrow(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7463-TERMINAL freezes only affected test and module paths."""

    commands = subject.build_validation_commands(subject.REPO_ROOT, tmp_path)

    assert subject.VALIDATION_MANIFEST.test_paths == (subject.TEST_PATH.as_posix(),)
    assert subject.VALIDATION_MANIFEST.changed_modules == (subject.MODULE_PATH.as_posix(),)
    assert {command.name for command in commands} == set(subject.AFFECTED_CHECK_NAMES)
    focused = next(command for command in commands if command.name == "focused_pytest")
    assert "tests/python" not in focused.argv
    assert "--no-cov" in focused.argv
    assert "-n" in focused.argv and "0" in focused.argv

    candidate = (
        subject.REPO_ROOT / "results/raw/experiment_7463_v654_semif_e0_logprob_parity/x.json"
    )
    assert {row.name for row in subject._terminal_commands(subject.REPO_ROOT, candidate)} == set(
        subject.TERMINAL_CHECK_NAMES
    )


def test_validation_receipts_gate_terminal_artifact() -> None:
    """SCENARIO-ARC-WMTE-7463-TERMINAL disqualifies any missing required receipt."""

    artifact = subject.build_artifact_for_test(
        [_pair(index) for index in range(64)],
        [{"correct": True, "nonuniform": True, "offload_observed": True} for _ in range(8)],
    )
    receipts = [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in (*subject.AFFECTED_CHECK_NAMES, *subject.TERMINAL_CHECK_NAMES)
    ]
    final = subject.finalize_validation(artifact, receipts)
    assert final["verdict_class"] == "blocked"
    assert final["flagged_adversarial"] is False
    assert subject.validate_artifact(final, require_validation=True) == []

    receipts[-1]["passed"] = False
    failed = subject.finalize_validation(artifact, receipts)
    assert failed["verdict_class"] == "disqualified"
    assert failed["honest_verdict"] == "complete_disqualified_required_validation_failed"
    assert subject.validate_artifact(failed, require_validation=True)
    assert any(
        error.startswith("receipt_count:") for error in subject._receipt_errors([], terminal=True)
    )


def test_small_audit_helpers_cover_errors_and_server_logprobs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7463 keeps malformed inputs and absent options visible."""

    assert subject._load_json(tmp_path / "missing.json") == {}
    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    assert subject._load_json(malformed) == {}
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    assert subject._load_json(scalar) == {}
    obj = tmp_path / "obj.json"
    obj.write_text('{"a": 1}', encoding="utf-8")
    assert subject._load_json(obj) == {"a": 1}

    checked = subject._check_row("x", obj, 1, 1, artifact_field="a")
    assert checked["passed"] is True
    assert subject.utc_now().endswith("Z")
    subject.progress(time.monotonic(), "test", "boundary", completed_units=1)
    assert "completed_units=1" in capsys.readouterr().out

    response = {
        "completion_probabilities": [
            {
                "top_logprobs": [
                    {"id": 10, "token": " A", "logprob": -0.2},
                    {"id": 11, "token": " B", "logprob": -1.7},
                    "ignored",
                ]
            }
        ]
    }
    assert subject._server_option_logits(response, {"A": 10, "B": 11}) == {
        "A": -0.2,
        "B": -1.7,
    }
    with pytest.raises(ValueError, match="missing_option_logits"):
        subject._server_option_logits({}, {"A": 10, "B": 11})


def test_invalid_generated_validation_plan_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7463-TERMINAL refuses command-plan drift."""

    monkeypatch.setattr(subject, "validate_command_plan", lambda *_args: ["drift"])
    with pytest.raises(ValueError, match="invalid_validation_plan:drift"):
        subject.build_validation_commands(subject.REPO_ROOT, tmp_path)


def test_date_argument_is_fixed() -> None:
    """REQ-ARC-WMTE-7463 pins the execution date."""

    assert subject.date_argument("20260920") == "20260920"
    with pytest.raises(ValueError, match="run_date_mismatch"):
        subject.date_argument("20260921")
