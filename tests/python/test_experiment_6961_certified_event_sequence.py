"""Tests for the certified prospective event sequence.

Spec refs: REQ-LEARN-6961 and SCENARIO-LEARN-6961-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6961_certified_event_sequence as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def source_objects() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Load the exact upstream rows that own seed admission."""

    certification = json.loads((REPO_ROOT / exp.CERTIFICATION_PATH).read_text())
    fixture = json.loads((REPO_ROOT / exp.FIXTURE_PATH).read_text())
    replay = json.loads((REPO_ROOT / exp.CERTIFICATION_REPLAY_PATH).read_text())
    return certification, fixture, replay


@pytest.fixture(scope="module")
def sequence_data(
    source_objects: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> dict[str, object]:
    """Generate the complete deterministic sequence without writing the repository."""

    certification, fixture, replay = source_objects
    seeds = exp.select_seed_certificates(certification, fixture, replay)
    return exp.generate_sequence(seeds)


@pytest.fixture(scope="module")
def complete_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Build once with a temporary sealed checkpoint and a fresh replay process."""

    directory = tmp_path_factory.mktemp("exp6961")
    return exp.build_from_paths(
        run_date="20260904",
        repo_root=REPO_ROOT,
        sealed_checkpoint_path=directory / "sealed.json",
    )


def test_req_learn_6961_spec_precedes_code_and_declares_contract() -> None:
    """REQ-LEARN-6961 lists every field and every required safety scenario."""

    text = (REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-6961") :]

    assert all(f"`{field}`" in section for field in exp.REQUIRED_ARTIFACT_FIELDS)
    assert all(
        f"SCENARIO-LEARN-6961-{name}" in section
        for name in (
            "PRECONDITIONS",
            "CHRONOLOGY",
            "LEAKAGE",
            "IDENTITY",
            "RETRIEVAL",
            "HEADROOM",
            "COPIES",
            "REPLAY",
        )
    )


def test_req_learn_6961_selects_only_exact_authority_rows(
    source_objects: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> None:
    """REQ-LEARN-6961 excludes confidence, rationale, and learned scores from seeds."""

    certification, fixture, replay = source_objects
    seeds = exp.select_seed_certificates(certification, fixture, replay)

    assert len(seeds) == exp.SEED_EVENT_COUNT
    assert len({row["source_certificate_id"] for row in seeds}) == exp.SEED_EVENT_COUNT
    assert {row["problem_family"] for row in seeds} == set(exp.PROBLEM_FAMILIES)
    assert all(row["exact_mapping_correct"] is True for row in seeds)
    assert all(row["authorities_agree"] is True for row in seeds)
    assert all(row["terminal"] is True and row["quarantined"] is False for row in seeds)
    assert all(row["admission_authority"] == "exact_dual_engine_certificate" for row in seeds)
    assert all(row["confidence_used"] is False for row in seeds)
    assert all(row["rationale_used"] is False for row in seeds)
    assert all(row["learned_score_used"] is False for row in seeds)
    assert all("confidence" not in row and "rationale" not in row for row in seeds)


def test_scenario_learn_6961_preconditions_block_incomplete_and_short_sources(
    tmp_path: Path,
    source_objects: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> None:
    """SCENARIO-LEARN-6961-PRECONDITIONS preserves failed expected and observed values."""

    certification, fixture, replay = source_objects
    bad = deepcopy(certification)
    bad["smt_certification_run_complete_score"] = 0
    bad["rows"] = [row for row in bad["rows"] if row["exact_mapping_correct"]][:5]
    certification_path = tmp_path / "certification.json"
    fixture_path = tmp_path / "fixture.json"
    replay_path = tmp_path / "replay.json"
    certification_path.write_text(json.dumps(bad), encoding="utf-8")
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    replay_path.write_text(json.dumps(replay), encoding="utf-8")

    artifact = exp.build_from_paths(
        run_date="20260904",
        repo_root=REPO_ROOT,
        certification_path=certification_path,
        fixture_path=fixture_path,
        certification_replay_path=replay_path,
        sealed_checkpoint_path=tmp_path / "sealed.json",
    )

    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_certified_event_sequence"
    assert artifact["certified_event_sequence_ready_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] == "smt_certification_run_complete_score"
    assert artifact["gate_check_summary"]["expected_value"] == 1
    assert artifact["gate_check_summary"]["observed_value"] == 0
    assert not exp.validate_artifact(artifact)


def test_scenario_learn_6961_preconditions_detect_hash_drift(
    tmp_path: Path,
    source_objects: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> None:
    """SCENARIO-LEARN-6961-PRECONDITIONS blocks changed fixture bytes."""

    certification, fixture, replay = source_objects
    certification_path = tmp_path / "certification.json"
    fixture_path = tmp_path / "fixture.json"
    replay_path = tmp_path / "replay.json"
    certification_path.write_text(json.dumps(certification), encoding="utf-8")
    fixture_path.write_text(json.dumps(fixture, indent=1), encoding="utf-8")
    replay_path.write_text(json.dumps(replay), encoding="utf-8")

    artifact = exp.build_from_paths(
        run_date="20260904",
        repo_root=REPO_ROOT,
        certification_path=certification_path,
        fixture_path=fixture_path,
        certification_replay_path=replay_path,
        sealed_checkpoint_path=tmp_path / "sealed.json",
    )

    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == "fixture_artifact_hash"
    assert artifact["gate_check_summary"]["expected_value"].startswith("sha256:")
    assert artifact["gate_check_summary"]["observed_value"].startswith("sha256:")


def test_scenario_learn_6961_preconditions_check_generator_and_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    source_objects: tuple[dict[str, object], dict[str, object], dict[str, object]],
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6961-PRECONDITIONS tests deterministic generation and writable seals."""

    certification, fixture, replay = source_objects
    monkeypatch.setattr(exp, "generator_is_deterministic", lambda seeds: False)
    monkeypatch.setattr(exp, "checkpoint_is_writable", lambda path: False)
    checks = exp.collect_preconditions(
        certification=certification,
        fixture=fixture,
        certification_replay=replay,
        fixture_path=REPO_ROOT / exp.FIXTURE_PATH,
        sealed_checkpoint_path=tmp_path / "sealed.json",
    )

    observed = {row["check"]: row for row in checks}
    assert observed["deterministic_generators"]["passed"] is False
    assert observed["writable_sealed_checkpoint"]["passed"] is False


def test_scenario_learn_6961_chronology_rejects_time_reversal(
    sequence_data: dict[str, object],
) -> None:
    """SCENARIO-LEARN-6961-CHRONOLOGY rejects a future certificate as prior memory."""

    bad = deepcopy(sequence_data)
    bad["event_rows"][8]["eligible_prior_certificate_ids"].append(
        bad["event_rows"][20]["certificate_id"]
    )

    assert "time_reversal" in exp.sequence_conformance_errors(bad)


def test_scenario_learn_6961_leakage_rejects_future_label_and_isomorphic_answer(
    sequence_data: dict[str, object],
) -> None:
    """SCENARIO-LEARN-6961-LEAKAGE catches labels and renamed copies of the answer."""

    label_leak = deepcopy(sequence_data)
    event = label_leak["event_rows"][10]
    event["prompt_payload"]["certified_relation"] = event["exact_outcome"]
    assert "future_label_leakage" in exp.sequence_conformance_errors(label_leak)

    answer_leak = deepcopy(sequence_data)
    event = answer_leak["event_rows"][11]
    copied = deepcopy(event["sealed_answer_mapping"])
    for index, row in enumerate(copied["variables"]):
        row["source"] = f"renamed_source_{index}"
        row["target"] = f"renamed_target_{index}"
    event["prompt_payload"]["retrieved_memory"].append({"mapping": copied})
    assert "isomorphic_answer_leakage" in exp.sequence_conformance_errors(answer_leak)


def test_scenario_learn_6961_identity_rejects_duplicates_and_family_collision(
    sequence_data: dict[str, object],
) -> None:
    """SCENARIO-LEARN-6961-IDENTITY rejects copied inputs and cross-family relevance."""

    duplicate = deepcopy(sequence_data)
    duplicate["event_rows"][9]["scientific_input_hash"] = duplicate["event_rows"][8][
        "scientific_input_hash"
    ]
    assert "duplicate_event" in exp.sequence_conformance_errors(duplicate)

    collision = deepcopy(sequence_data)
    event = collision["event_rows"][12]
    event["prompt_payload"]["retrieved_memory"].append(
        {
            "certificate_id": "prior-different-family",
            "problem_family": next(
                family for family in exp.PROBLEM_FAMILIES if family != event["problem_family"]
            ),
            "relevance": "relevant",
            "factor_ids": ["bidirectional_domain_coverage"],
        }
    )
    assert "family_collision" in exp.sequence_conformance_errors(collision)


def test_scenario_learn_6961_retrieval_has_noop_and_explicit_distractors(
    sequence_data: dict[str, object],
) -> None:
    """SCENARIO-LEARN-6961-RETRIEVAL keeps no-op empty and FIFO noise visible."""

    no_op = next(row for row in sequence_data["event_rows"] if row["control_class"] == "no_op")
    arm_rows = [
        row for row in sequence_data["opportunity_rows"] if row["event_id"] == no_op["event_id"]
    ]

    assert {row["arm"] for row in arm_rows} == set(exp.ARMS)
    assert all(row["selected_certificate_ids"] == [] for row in arm_rows)
    assert all(row["inference_ran"] is False for row in arm_rows)
    assert sequence_data["distractor_rows"]
    assert all(row["relevance"] == "irrelevant" for row in sequence_data["distractor_rows"])


def test_scenario_learn_6961_headroom_is_positive_or_complete_null(
    sequence_data: dict[str, object],
) -> None:
    """SCENARIO-LEARN-6961-HEADROOM requires 12 later opportunities for every family."""

    ready = exp.reduce_terminal_gate(sequence_data)
    assert ready["ready_score"] == 1
    assert ready["verdict_class"] == "circular_positive"

    bad = deepcopy(sequence_data)
    family = exp.HEADLINE_MODEL_FAMILIES[0]
    for row in bad["headroom_rows"]:
        if row["model_family"] == family:
            row["positive_headroom"] = False
            row["structural_headroom"] = 0
    for row in bad["opportunity_rows"]:
        if row["model_family"] == family and row["arm"] == "queue":
            row["structural_opportunity"] = False

    reduced = exp.reduce_terminal_gate(bad)
    assert reduced["ready_score"] == 0
    assert reduced["verdict_class"] == "null"
    assert reduced["honest_verdict"].startswith("complete_null_")


def test_scenario_learn_6961_copies_reject_reused_seed_certificate(
    sequence_data: dict[str, object],
) -> None:
    """SCENARIO-LEARN-6961-COPIES rejects copied source certificates."""

    bad = deepcopy(sequence_data)
    bad["seed_certificate_rows"][1]["source_certificate_id"] = bad["seed_certificate_rows"][0][
        "source_certificate_id"
    ]

    assert "copied_seed_certificate" in exp.sequence_conformance_errors(bad)


def test_scenario_learn_6961_replay_and_complete_artifact(
    complete_artifact: dict[str, object],
) -> None:
    """SCENARIO-LEARN-6961-REPLAY owns a circular-positive conformance result."""

    artifact = complete_artifact
    assert len(artifact["event_rows"]) == exp.EXPECTED_EVENT_COUNT
    assert len(artifact["rows"]) == exp.EXPECTED_EVENT_COUNT
    assert len(artifact["chronology_rows"]) == exp.EXPECTED_EVENT_COUNT
    assert len(artifact["fresh_process_replay_rows"]) == exp.EXPECTED_EVENT_COUNT
    assert all(row["replay_matches"] for row in artifact["fresh_process_replay_rows"])
    assert all(row["prompt_matches"] for row in artifact["fresh_process_replay_rows"])
    assert all(row["retrieval_matches"] for row in artifact["fresh_process_replay_rows"])
    assert len(artifact["seed_certificate_rows"]) == exp.SEED_EVENT_COUNT
    assert all(row["event_count"] == exp.EVENTS_PER_MODEL_FAMILY for row in artifact["family_rows"])
    assert all(row["genuine_memory_opportunity_count"] >= 12 for row in artifact["family_rows"])
    assert artifact["certified_event_sequence_ready_score"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete_circular_positive_")
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert not exp.validate_artifact(artifact)


def test_scenario_learn_6961_replay_hash_drift_and_validation_fail_closed(
    complete_artifact: dict[str, object],
) -> None:
    """SCENARIO-LEARN-6961-REPLAY detects replay, checksum, and verdict drift."""

    bad = deepcopy(complete_artifact)
    bad["fresh_process_replay_rows"][0]["child_event_hash"] = "sha256:drift"
    assert "fresh_process_replay_mismatch" in exp.validate_artifact(bad)

    bad = deepcopy(complete_artifact)
    bad["reproducibility_checksum"] = "sha256:drift"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(bad)

    bad = deepcopy(complete_artifact)
    bad["verdict_class"] = "positive"
    bad["honest_verdict"] = "complete_positive_wrong_class"
    assert "conforming_sequence_requires_circular_positive" in exp.validate_artifact(bad)

    bad = deepcopy(complete_artifact)
    bad.pop("rows")
    assert exp.validate_artifact(bad)[0].startswith("missing_required_fields")


def test_req_learn_6961_replay_cli_writer_and_normal_cli(
    tmp_path: Path,
    complete_artifact: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-LEARN-6961 covers private replay, validated writes, and command summaries."""

    checkpoint = Path(complete_artifact["sealed_checkpoint_path"])
    replay_output = tmp_path / "replay.json"
    assert (
        exp.main(
            [
                "--replay-checkpoint",
                str(checkpoint),
                "--replay-output",
                str(replay_output),
            ]
        )
        == 0
    )
    replay = json.loads(replay_output.read_text(encoding="utf-8"))
    assert replay["sequence_hash"].startswith("sha256:")
    with pytest.raises(SystemExit):
        exp.main(["--replay-checkpoint", str(checkpoint)])

    output = tmp_path / "result.json"
    monkeypatch.setattr(exp, "build_from_paths", lambda **kwargs: deepcopy(complete_artifact))
    written = exp.run(date="20260904", repo_root=tmp_path, output_path=output)
    assert written["experiment_id"] == 6961
    assert json.loads(output.read_text(encoding="utf-8"))["experiment_id"] == 6961

    invalid = deepcopy(complete_artifact)
    invalid["reproducibility_checksum"] = "bad"
    monkeypatch.setattr(exp, "build_from_paths", lambda **kwargs: invalid)
    with pytest.raises(ValueError, match="artifact_validation"):
        exp.run(date="20260904", repo_root=tmp_path, output_path=output)

    monkeypatch.setattr(exp, "run", lambda **kwargs: deepcopy(complete_artifact))
    assert exp.main(["--date", "20260904"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["event_count"] == exp.EXPECTED_EVENT_COUNT
    assert summary["certified_event_sequence_ready_score"] == 1


def test_req_learn_6961_seed_filter_skips_incomplete_replay_rows(
    source_objects: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> None:
    """REQ-LEARN-6961 admits no exact row whose replay mapping is absent."""

    certification, fixture, replay = deepcopy(source_objects)
    exact_attempt = next(
        row["attempt_key"] for row in certification["rows"] if row["exact_mapping_correct"]
    )
    replay_row = next(row for row in replay["inputs"] if row["attempt_key"] == exact_attempt)
    replay_row["parse"]["parsed_candidate"] = None

    seeds = exp.select_seed_certificates(certification, fixture, replay)
    assert all(row["source_attempt_key"] != exact_attempt for row in seeds)


def test_req_learn_6961_conformance_fail_closed_branches(
    sequence_data: dict[str, object],
) -> None:
    """REQ-LEARN-6961 names every invalid sequence condition without inference."""

    bad = deepcopy(sequence_data)
    bad["seed_certificate_rows"][0]["confidence"] = 0.99
    bad["seed_certificate_rows"][0]["terminal"] = False
    bad["event_rows"].pop()
    bad["event_rows"][0]["ordinal"] = 1
    event = bad["event_rows"][10]
    event["prompt_payload"]["surface_context"] = event["prohibited_future_certificate_ids"][0]
    copied = deepcopy(event["sealed_answer_mapping"])
    copied["variables"][0]["scale"] = "77"
    event["prompt_payload"]["retrieved_memory"].append({"mapping": copied})
    no_op = next(row for row in bad["event_rows"] if row["control_class"] == "no_op")
    no_op["retrieval_plan"]["fifo"] = [no_op["eligible_prior_certificate_ids"][0]]
    event["exact_success"] = False
    event["correct_proposal_possible_without_copying"] = False
    bad["event_rows"][0]["event_role"] = "later"
    bad["event_rows"][0]["split"] = "evaluation"
    for row in bad["event_rows"]:
        if row["control_class"] in {
            "contradiction",
            "delayed_correction",
            "retention_probe",
            "interference_probe",
        }:
            row["control_class"] = "standard"
    bad["distractor_rows"] = []
    bad["correction_rows"] = []
    bad["retention_probe_rows"] = []

    errors = exp.sequence_conformance_errors(bad)
    expected = {
        "self_report_seed_authority",
        "non_exact_seed_certificate",
        "event_count",
        "chronology_order",
        "future_label_leakage",
        "copied_answer_mapping",
        "no_op_retrieval_not_empty",
        "uncertified_event_outcome",
        "answer_copy_required",
        f"family_event_count:{exp.HEADLINE_MODEL_FAMILIES[2]}",
        "seed_event_count",
        "split_boundary",
        "safety_case_coverage",
        "distractor_coverage",
        "correction_coverage",
        "retention_coverage",
    }
    assert expected <= set(errors)
    assert exp.reduce_terminal_gate(bad)["verdict_class"] == "disqualified"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema_version", "wrong", "checkpoint_schema_version"),
        ("random_seed", -1, "checkpoint_random_seed"),
        ("seed_certificate_rows", None, "checkpoint_seed_rows"),
    ),
)
def test_req_learn_6961_replay_checkpoint_rejects_invalid_seal(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """REQ-LEARN-6961 rejects a seal before replay when its contract is invalid."""

    checkpoint = {
        "schema_version": exp.CHECKPOINT_SCHEMA_VERSION,
        "random_seed": exp.RANDOM_SEED,
        "seed_certificate_rows": [],
    }
    checkpoint[field] = value
    path = tmp_path / f"{field}.json"
    path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        exp.replay_checkpoint(path)


def test_req_learn_6961_json_reader_requires_an_object(tmp_path: Path) -> None:
    """REQ-LEARN-6961 rejects non-object source artifacts."""

    path = tmp_path / "array.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object_required"):
        exp._read_json(path)


def test_req_learn_6961_validator_rejects_schema_and_blocked_drift(
    complete_artifact: dict[str, object],
) -> None:
    """REQ-LEARN-6961 validates principles, substrate, oracle, and blocked metadata."""

    bad = deepcopy(complete_artifact)
    bad["field_principles"] = {}
    bad["inference_substrate"] = "llm"
    bad["verifier_is_oracle"] = False
    bad["verdict_class"] = "mystery"
    errors = exp.validate_artifact(bad)
    assert {
        "field_principles_incomplete",
        "inference_substrate_mismatch",
        "verifier_is_oracle_must_be_true",
        "verdict_class_invalid",
    } <= set(errors)

    blocked = deepcopy(complete_artifact)
    blocked["verdict_class"] = "blocked"
    blocked["certified_event_sequence_ready_score"] = 1
    blocked["honest_verdict"] = "wrong"
    blocked["gate_check_summary"] = {"failed_check": None, "passed": True}
    errors = exp.validate_artifact(blocked)
    assert {
        "blocked_ready_score_nonzero",
        "blocked_verdict_mismatch",
        "blocked_gate_summary_incomplete",
    } <= set(errors)


def test_req_learn_6961_validator_rejects_terminal_and_source_drift(
    complete_artifact: dict[str, object],
) -> None:
    """REQ-LEARN-6961 recomputes readiness, verdict prefixes, and source hashes."""

    bad = deepcopy(complete_artifact)
    bad["certified_event_sequence_ready_score"] = 0
    bad["honest_verdict"] = "wrong"
    errors = exp.validate_artifact(bad)
    assert "ready_score_mismatch" in errors
    assert "honest_verdict_prefix_mismatch" in errors

    bad = deepcopy(complete_artifact)
    bad["event_rows"][0]["exact_success"] = False
    bad["certified_event_sequence_ready_score"] = 0
    bad["verdict_class"] = "circular_positive"
    assert "terminal_verdict_class_mismatch" in exp.validate_artifact(bad)

    bad["verdict_class"] = "disqualified"
    bad["honest_verdict"] = "wrong"
    assert "honest_verdict_prefix_mismatch" in exp.validate_artifact(bad)

    bad = deepcopy(complete_artifact)
    source = next(iter(bad["source_artifact_hashes"].values()))
    source["sha256"] = "sha256:drift"
    assert "source_artifact_hash_drift" in exp.validate_artifact(bad)
