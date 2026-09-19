"""Tests for the source-aware feature and decision protocol.

Spec refs: REQ-AUTO-7412 and SCENARIO-AUTO-7412-01 through
SCENARIO-AUTO-7412-07.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pytest

from carnot import experiment_7412_v650_source_features as source_features


ROOT = Path(__file__).resolve().parents[2]


def _predictor_row(**changes: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "row_key": "row-1",
        "group_id": "group-1",
        "partition": "train",
        "question": "How many packages arrived?",
        "context": "In 2020, Alice received 3 packages. Each weighed 5 kg.",
        "answer": "In 2021, Alice received 3 packages. Therefore the total was 7 kg.",
        "sentence": "Therefore the total was 7 kg.",
    }
    row.update(changes)
    return row


def _validation_receipts() -> list[dict[str, Any]]:
    return [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in source_features.validation_scope.REQUIRED_CHECK_NAMES
    ]


def test_req_auto_7412_spec_owns_protocol() -> None:
    """REQ-AUTO-7412 declares the protocol fields and scenarios first."""

    text = (ROOT / source_features.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-AUTO-7412") :]
    for suffix in ("01", "02", "03", "04", "05", "06", "07"):
        assert f"SCENARIO-AUTO-7412-{suffix}" in section
    for field in source_features.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section or field in {
            "schema",
            "experiment_id",
            "milestone",
            "status",
        }


def test_scenario_7412_01_features_are_bounded_numeric_and_label_blind() -> None:
    """SCENARIO-AUTO-7412-01 fixes source features without teacher fields."""

    row = _predictor_row()
    feature_row = source_features.extract_feature_row(row)
    values = feature_row["source_features"]
    assert list(values) == list(source_features.SOURCE_FEATURE_NAMES)
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values.values())
    assert values["numeric_novelty_with_context"] == 1.0
    assert values["normalized_number_token_overlap"] == 0.0
    assert values["missing_or_empty_source"] == 0.0
    assert set(feature_row["response_only_ablation"]) == {
        "entity_uptake",
        "falsifiability_score",
    }

    injected = {**row, "label": 1, "expected_verdict": "incorrect"}
    assert source_features.extract_feature_row(injected) == feature_row
    empty = source_features.extract_feature_row(_predictor_row(context=" \n "))
    assert empty["source_features"]["missing_or_empty_source"] == 1.0
    assert empty["source_features"]["numeric_novelty_with_context"] == 1.0
    assert empty["source_features"]["normalized_number_token_overlap"] == 0.0
    assert empty["source_features"]["normalized_content_token_overlap"] == 0.0
    assert empty["source_features"]["max_answer_source_sentence_overlap"] == 0.0


def test_scenario_7412_01_numeric_conversion_and_token_caps() -> None:
    """SCENARIO-AUTO-7412-01 canonicalizes decimals and bounds all text work."""

    assert source_features.canonical_number_tokens("1,000 1000.0 1e3 -0 0.000") == {
        "0",
        "1000",
    }
    tokens = source_features.bounded_word_tokens(
        " ".join(f"Token{index}" for index in range(30)), token_limit=7
    )
    assert len(tokens) == 7
    with pytest.raises(ValueError, match="token_limit"):
        source_features.bounded_word_tokens("text", token_limit=0)
    row = _predictor_row(
        context="source 1.0",
        answer="first claim 1. second claim 2.",
        sentence=None,
    )
    values = source_features.extract_feature_row(row)["source_features"]
    assert values["numeric_novelty_with_context"] == 0.5
    assert values["normalized_number_token_overlap"] == 0.5


def test_scenario_7412_02_checkpoint_reload_and_probability_are_safe(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7412-02 reloads finite 6-4-1 state and rejects mutations."""

    checkpoint = source_features.initial_gibbs_checkpoint(seed=65_001, input_dim=6)
    path = tmp_path / "checkpoint.json"
    source_features.write_checkpoint(path, checkpoint)
    reloaded = source_features.load_checkpoint(path, input_dim=6)
    vector = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert source_features.gibbs_energy(reloaded, vector) == pytest.approx(
        source_features.gibbs_energy(checkpoint, vector)
    )
    assert source_features.probability_from_energy(1_000.0) == 1.0
    assert source_features.probability_from_energy(-1_000.0) == 0.0
    with pytest.raises(ValueError, match="finite"):
        source_features.probability_from_energy(float("nan"))

    mutations = []
    changed = deepcopy(checkpoint)
    changed["w1"][0][0] = float("nan")
    mutations.append(changed)
    changed = deepcopy(checkpoint)
    changed["w1"] = changed["w1"][:-1]
    mutations.append(changed)
    changed = deepcopy(checkpoint)
    changed["extra"] = 1
    mutations.append(changed)
    changed = deepcopy(checkpoint)
    changed["b_out"] = True
    mutations.append(changed)
    changed = deepcopy(checkpoint)
    changed["b_out"] = "__import__('os').system('false')"
    mutations.append(changed)
    for mutation in mutations:
        with pytest.raises(ValueError):
            source_features.validate_gibbs_checkpoint(mutation, input_dim=6)


def test_scenario_7412_03_fixture_training_handles_constants_and_imbalance() -> None:
    """SCENARIO-AUTO-7412-03 keeps both frozen trainers finite on hard fixtures."""

    features = np.full((41, 6), 0.25, dtype=np.float64)
    labels = np.asarray([0] * 40 + [1], dtype=np.float64)
    gibbs = source_features.fit_gibbs_head(features, labels, seed=65_001, steps=12)
    logistic = source_features.fit_logistic_control(features, labels, seed=65_001, steps=12)
    assert gibbs["update_count"] == logistic["update_count"] == 12
    assert all(math.isfinite(row["loss"]) for row in gibbs["loss_curve"])
    assert all(math.isfinite(row["loss"]) for row in logistic["loss_curve"])
    source_features.validate_gibbs_checkpoint(gibbs["checkpoint"], input_dim=6)
    with pytest.raises(ValueError, match="both labels"):
        source_features.fit_gibbs_head(features[:4], np.zeros(4), seed=65_001, steps=1)
    with pytest.raises(ValueError, match="shape"):
        source_features.fit_logistic_control(features[:, :5], labels, seed=65_001, steps=1)
    with pytest.raises(ValueError, match="steps"):
        source_features.fit_gibbs_head(features, labels, seed=65_001, steps=501)


def test_scenario_7412_02_typed_decisions_have_stable_confidence() -> None:
    """SCENARIO-AUTO-7412-02 defines accept, reject, and safe escalation."""

    accept = source_features.typed_decision(0.01, 0.01, 0.90, "fixture")
    reject = source_features.typed_decision(0.90, 0.01, 0.90, "fixture")
    escalate = source_features.typed_decision(0.5, 0.01, 0.90, "fixture")
    invalid = source_features.typed_decision(float("nan"), 0.01, 0.90, "fixture")
    assert [accept["decision"], reject["decision"], escalate["decision"]] == [
        "accept",
        "reject",
        "escalate",
    ]
    assert accept["confidence_correct"] == 0.99
    assert invalid["decision"] == "escalate"
    assert invalid["confidence_correct"] is None


def test_scenario_7412_04_group_representatives_and_certificates() -> None:
    """SCENARIO-AUTO-7412-04 counts each group once and disables empty actions."""

    rows = [
        {"group_id": "g1", "row_key": "b", "label": 1},
        {"group_id": "g1", "row_key": "a", "label": 0},
        {"group_id": "g2", "row_key": "c", "label": 1},
    ]
    representatives = source_features.label_blind_group_representatives(rows)
    assert [(row["group_id"], row["row_key"]) for row in representatives] == [
        ("g1", "a"),
        ("g2", "c"),
    ]
    relabeled = [{**row, "label": 1 - row["label"]} for row in rows]
    assert [
        row["row_key"] for row in source_features.label_blind_group_representatives(relabeled)
    ] == [
        "a",
        "c",
    ]
    empty = source_features.exact_risk_certificate([], risk_budget=0.05)
    safe = source_features.exact_risk_certificate([0] * 300, risk_budget=0.05)
    assert empty["upper_risk_bound"] is None and empty["action_enabled"] is False
    assert safe["alpha_per_test"] == pytest.approx(0.05 / 360)
    assert safe["action_enabled"] is True
    with pytest.raises(ValueError, match="binary"):
        source_features.exact_risk_certificate([2], risk_budget=0.05)


def test_scenario_7412_05_challenges_are_complete_masked_and_reloadable(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7412-05 seals 16 pair cases and eight format controls."""

    manifest = source_features.build_challenge_manifest()
    assert len(manifest["predictor_records"]) == 24
    assert len(manifest["evaluator_records"]) == 24
    assert len({row["pair_id"] for row in manifest["predictor_records"]}) == 8
    assert (
        sum(row["case_kind"] == "equivalent_control" for row in manifest["predictor_records"]) == 8
    )
    denied = {"expected_verdict", "expected_scope", "source_relation", "answer_span"}
    assert all(denied.isdisjoint(row) for row in manifest["predictor_records"])
    for row in manifest["evaluator_records"]:
        predictor = next(
            item for item in manifest["predictor_records"] if item["case_id"] == row["case_id"]
        )
        start, end = row["answer_span"]
        assert predictor["answer"][start:end] == row["answer_span_text"]
    path = tmp_path / "challenge.json"
    source_features.write_challenge_manifest(path, manifest)
    assert source_features.load_challenge_manifest(path) == manifest
    changed = deepcopy(manifest)
    changed["predictor_records"][0]["answer"] += " drift"
    with pytest.raises(ValueError, match="challenge_manifest_hash"):
        source_features.load_challenge_manifest_value(changed)


def test_scenario_7412_06_actual_corpus_reader_never_requests_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-AUTO-7412-06 reads every role only through predictor methods."""

    def deny(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("label reader must stay unopened")

    monkeypatch.setattr(source_features.CorpusReaders, "read_labels", deny)
    rows, receipt = source_features.load_predictor_feature_rows(ROOT)
    assert rows and receipt["official_labels_opened"] is False
    assert {row["partition"] for row in rows} >= {
        "train",
        "probability_calibration",
        "policy_calibration",
        "final_test",
    }
    assert all("label" not in row for row in rows)


def test_protocol_manifests_freeze_real_work_as_unstarted(tmp_path: Path) -> None:
    """REQ-AUTO-7412 freezes every later arm, seed, and threshold condition."""

    feature_rows = [source_features.extract_feature_row(_predictor_row())]
    outputs = source_features.write_protocol_manifests(
        tmp_path,
        feature_rows,
        source_manifest_hash="sha256:source",
    )
    protocol = source_features.load_protocol_manifest(outputs["protocol_path"])
    assert protocol["official_test_labels_opened"] is False
    assert protocol["optimizer"]["steps"] == 500
    assert protocol["architecture"]["source_aware"] == [6, 4, 1]
    assert len(protocol["planned_conditions"]) == 4 * 5 * 9
    assert {row["status"] for row in protocol["planned_conditions"]} == {"unstarted"}
    assert outputs["feature_rows_sha256"] == source_features.sha256_file(
        outputs["feature_rows_path"]
    )


def test_upstream_authentication_and_blocked_contract() -> None:
    """REQ-AUTO-7412 names exact Exp7410 mutations before feature work."""

    upstream = json.loads((ROOT / source_features.UPSTREAM_PATH).read_text(encoding="utf-8"))
    manifest = json.loads(
        (ROOT / source_features.UPSTREAM_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    rows = source_features.upstream_gate_rows(upstream, manifest, ROOT)
    assert rows and all(row["passed"] for row in rows)
    changed = deepcopy(upstream)
    changed["flagged_adversarial"] = True
    failures = source_features.upstream_gate_rows(changed, manifest, ROOT)
    assert any(row["field"] == "flagged_adversarial" and not row["passed"] for row in failures)
    failed = next(row for row in failures if not row["passed"])
    blocked = source_features.build_blocked_artifact(failed)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["source_feature_protocol_ready_score"] == 0
    assert blocked["gate_check_summary"]["blocked_field"] == failed["field"]
    assert source_features.validate_artifact(blocked) == []


def test_scenario_7412_07_artifact_contract_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7412-07 separates ready protocol from unmeasured value."""

    features = [source_features.extract_feature_row(_predictor_row())]
    outputs = source_features.write_protocol_manifests(
        tmp_path / "raw",
        features,
        source_manifest_hash=source_features.EXPECTED_UPSTREAM_MANIFEST_HASH,
    )
    fixture = source_features.run_fixture_training(steps=8)
    artifact = source_features.build_artifact_for_test(
        feature_rows=features,
        protocol_outputs=outputs,
        fixture_training=fixture,
        validation_receipts=_validation_receipts(),
    )
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "null"
    assert artifact["source_feature_protocol_ready_score"] == 1
    assert artifact["promotion_score"] == 0
    assert artifact["annotation_risk_scope"] == "machine_annotation_only_not_truth"
    assert len(artifact["rows"]) == 4 * 5 * 9
    assert source_features.validate_artifact(artifact) == []

    path = tmp_path / "candidate.json"
    source_features.atomic_json(path, artifact)
    assert source_features.cold_replay(path) == []
    changed = deepcopy(artifact)
    changed["promotion_score"] = 1
    changed["reproducibility_checksum"] = source_features.artifact_checksum(changed)
    assert "declaration_mismatch:promotion_score" in source_features.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["rows"].pop()
    changed["reproducibility_checksum"] = source_features.artifact_checksum(changed)
    assert "planned_row_count_mismatch" in source_features.validate_artifact(changed)


def test_scenario_7412_01_and_03_defensive_numeric_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-AUTO-7412-01/03 fail closed on malformed text and arrays."""

    source_features.progress(0.0, "fixture", "boundary", completed=1)
    assert "phase=fixture" in capsys.readouterr().out
    assert source_features._canonical_decimal("bad") is None
    assert source_features._canonical_decimal("NaN") is None
    with pytest.raises(ValueError, match="token_limit"):
        source_features.canonical_number_tokens("1", token_limit=0)

    monkeypatch.setattr(
        source_features.PCIBProbe,
        "compute_falsifiability_score",
        lambda *_args: float("nan"),
    )
    with pytest.raises(ValueError, match="source features"):
        source_features.extract_feature_row(_predictor_row())
    monkeypatch.undo()

    with pytest.raises(ValueError, match="non-empty and finite"):
        source_features.fit_gibbs_head(
            np.full((2, 6), float("nan")), np.asarray([0, 1]), seed=65_001, steps=1
        )
    with pytest.raises(ValueError, match="strings"):
        source_features._numeric_tree({1: 1})
    with pytest.raises(ValueError, match="2 or 6"):
        source_features.initial_gibbs_checkpoint(seed=65_001, input_dim=5)
    with pytest.raises(ValueError, match="unreadable"):
        source_features.load_checkpoint(tmp_path / "absent.json")
    with pytest.raises(ValueError, match="finite 2-vector or 6-vector"):
        source_features.gibbs_energy(
            source_features.initial_gibbs_checkpoint(seed=65_001), [0.0] * 5
        )
    with pytest.raises(ValueError, match="steps"):
        source_features.fit_logistic_control(
            np.zeros((2, 6)), np.asarray([0, 1]), seed=65_001, steps=501
        )
    with pytest.raises(ValueError, match="group_id"):
        source_features.label_blind_group_representatives([{"group_id": "", "row_key": "a"}])


def test_scenario_7412_05_challenge_corruptions_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7412-05 rejects leaks, identity drift, and forged spans."""

    original = source_features.build_challenge_manifest()

    def rehash(value: dict[str, Any]) -> dict[str, Any]:
        value["manifest_hash"] = source_features._challenge_hash(value)
        return value

    changed = deepcopy(original)
    changed["predictor_records"] = None
    with pytest.raises(ValueError, match="records_missing"):
        source_features.load_challenge_manifest_value(rehash(changed))
    changed = deepcopy(original)
    changed["predictor_records"].pop()
    with pytest.raises(ValueError, match="case_count"):
        source_features.load_challenge_manifest_value(rehash(changed))
    changed = deepcopy(original)
    changed["predictor_records"][0]["expected_verdict"] = "leak"
    with pytest.raises(ValueError, match="predictor_leak"):
        source_features.load_challenge_manifest_value(rehash(changed))
    changed = deepcopy(original)
    changed["predictor_records"][0]["case_id"] = changed["predictor_records"][1]["case_id"]
    with pytest.raises(ValueError, match="case_identity"):
        source_features.load_challenge_manifest_value(rehash(changed))
    changed = deepcopy(original)
    changed["evaluator_records"][0]["case_id"] = "absent"
    with pytest.raises(ValueError, match="evaluator_identity"):
        source_features.load_challenge_manifest_value(rehash(changed))
    changed = deepcopy(original)
    changed["evaluator_records"][0]["answer_span"] = [True, 2]
    with pytest.raises(ValueError, match="answer_span"):
        source_features.load_challenge_manifest_value(rehash(changed))
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        source_features.load_challenge_manifest(malformed)


def test_protocol_and_precondition_corruptions_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-AUTO-7412 authenticates inputs and rejects protocol mutations."""

    events: list[int] = []
    monkeypatch.setattr(
        source_features, "progress", lambda *_args, **kwargs: events.append(kwargs["completed"])
    )
    _, receipt = source_features.load_predictor_feature_rows(ROOT, emit_progress=True)
    assert events[-1] == receipt["feature_rows_completed"]
    monkeypatch.undo()
    checks, hashes, upstream, manifest = source_features.collect_preconditions(ROOT)
    assert checks and hashes and upstream and manifest
    assert all(row["passed"] for row in checks)

    empty_root = tmp_path / "empty-root"
    empty_root.mkdir()
    monkeypatch.setattr(source_features, "INPUT_PATHS", ())
    _, _, missing_upstream, missing_manifest = source_features.collect_preconditions(empty_root)
    assert missing_upstream == missing_manifest == {}
    upstream_path = empty_root / source_features.UPSTREAM_PATH
    manifest_path = empty_root / source_features.UPSTREAM_MANIFEST_PATH
    upstream_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    upstream_path.write_text("[]", encoding="utf-8")
    manifest_path.write_text("[]", encoding="utf-8")
    _, _, typed_upstream, typed_manifest = source_features.collect_preconditions(empty_root)
    assert typed_upstream == typed_manifest == {}
    monkeypatch.undo()

    outputs = source_features.write_protocol_manifests(
        tmp_path / "raw",
        [source_features.extract_feature_row(_predictor_row())],
        source_manifest_hash="sha256:fixture",
    )
    path = outputs["protocol_path"]
    protocol = json.loads(path.read_text(encoding="utf-8"))
    malformed = tmp_path / "bad-protocol.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        source_features.load_protocol_manifest(malformed)
    changed = deepcopy(protocol)
    changed["manifest_hash"] = "sha256:wrong"
    source_features.atomic_json(path, changed)
    with pytest.raises(ValueError, match="hash_mismatch"):
        source_features.load_protocol_manifest(path)
    changed = deepcopy(protocol)
    changed["planned_conditions"].pop()
    changed["manifest_hash"] = source_features._protocol_hash(changed)
    source_features.atomic_json(path, changed)
    with pytest.raises(ValueError, match="planned_conditions"):
        source_features.load_protocol_manifest(path)
    changed = deepcopy(protocol)
    changed["official_test_labels_opened"] = True
    changed["manifest_hash"] = source_features._protocol_hash(changed)
    source_features.atomic_json(path, changed)
    with pytest.raises(ValueError, match="official_labels"):
        source_features.load_protocol_manifest(path)


def test_artifact_and_replay_defensive_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-AUTO-7412-07 independently rejects artifact and replay drift."""

    repo = tmp_path / "repo"
    outputs = source_features.write_protocol_manifests(
        repo / "raw",
        [source_features.extract_feature_row(_predictor_row())],
        source_manifest_hash="sha256:fixture",
    )
    artifact = source_features.build_artifact_for_test(
        feature_rows=[source_features.extract_feature_row(_predictor_row())],
        protocol_outputs=outputs,
        fixture_training=source_features.run_fixture_training(steps=1),
        validation_receipts=_validation_receipts(),
    )

    def errors_after(**changes: Any) -> list[str]:
        changed = deepcopy(artifact)
        changed.update(changes)
        changed["reproducibility_checksum"] = source_features.artifact_checksum(changed)
        return source_features.validate_artifact(changed)

    assert "verdict_class_invalid" in errors_after(verdict_class="unknown")
    principles = deepcopy(artifact["field_principles"])
    principles.pop("schema")
    assert "field_principles_mismatch" in errors_after(field_principles=principles)
    assert "feature_definitions_mismatch" in errors_after(feature_definitions={})
    checksum_drift = deepcopy(artifact)
    checksum_drift["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum_mismatch" in source_features.validate_artifact(checksum_drift)
    blocked = source_features.build_blocked_artifact(
        {"check": "x", "upstream": "u", "path": "p", "field": "f", "expected": 1, "observed": 0}
    )
    blocked["source_feature_protocol_ready_score"] = 1
    blocked["reproducibility_checksum"] = source_features.artifact_checksum(blocked)
    assert "blocked_disposition_invalid" in source_features.validate_artifact(blocked)
    assert "source_feature_protocol_ready_score_mismatch" in errors_after(
        source_feature_protocol_ready_score=0
    )
    assert "protocol_manifest_identity_mismatch" in errors_after(
        protocol_manifest_hash="sha256:wrong"
    )
    assert "challenge_manifest_identity_mismatch" in errors_after(
        challenge_manifest_hash="sha256:wrong"
    )
    assert any(
        "unreadable" in error for error in errors_after(protocol_manifest_path="missing.json")
    )
    assert any(
        "unreadable" in error for error in errors_after(challenge_manifest_path="missing.json")
    )

    monkeypatch.setattr(source_features, "REPO_ROOT", repo)
    relative = deepcopy(artifact)
    relative["protocol_manifest_path"] = "raw/protocol_manifest.json"
    relative["challenge_manifest_path"] = "raw/challenge_manifest.json"
    relative["feature_rows_path"] = "raw/source_feature_rows.json"
    relative["reproducibility_checksum"] = source_features.artifact_checksum(relative)
    assert source_features.validate_artifact(relative) == []
    candidate = repo / "candidate.json"
    source_features.atomic_json(candidate, relative)
    assert source_features.cold_replay(candidate) == []

    missing = tmp_path / "missing-candidate.json"
    assert "artifact_unreadable" in source_features.cold_replay(missing)[0]
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    assert source_features.cold_replay(array) == ["artifact_not_object"]
    unreadable_feature = deepcopy(relative)
    unreadable_feature["feature_rows_path"] = "raw/missing-features.json"
    unreadable_feature["reproducibility_checksum"] = source_features.artifact_checksum(
        unreadable_feature
    )
    source_features.atomic_json(candidate, unreadable_feature)
    assert any(
        "feature_rows_unreadable" in error for error in source_features.cold_replay(candidate)
    )
    wrong_hash = deepcopy(relative)
    wrong_hash["feature_rows_sha256"] = "sha256:wrong"
    wrong_hash["reproducibility_checksum"] = source_features.artifact_checksum(wrong_hash)
    source_features.atomic_json(candidate, wrong_hash)
    assert "feature_rows_sha256_mismatch" in source_features.cold_replay(candidate)

    real_replay = deepcopy(relative)
    real_replay["fixture_artifact"] = False
    real_replay["reproducibility_checksum"] = source_features.artifact_checksum(real_replay)
    source_features.atomic_json(candidate, real_replay)
    monkeypatch.setattr(source_features, "validate_artifact", lambda _value: [])
    monkeypatch.setattr(
        source_features,
        "load_predictor_feature_rows",
        lambda _root: ([{"different": True}], {"official_labels_opened": True}),
    )
    replay_errors = source_features.cold_replay(candidate)
    assert "independent_feature_row_recompute_mismatch" in replay_errors
    assert "independent_reader_opened_labels" in replay_errors
    span = source_features._span("fixture", time.monotonic(), time.monotonic(), 1)
    assert span["completed_units"] == 1


def test_cli_parse_and_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-AUTO-7412 keeps the public script a thin fixed-date dispatcher."""

    args = source_features.parse_args(["--date", source_features.RUN_DATE])
    assert args.date == source_features.RUN_DATE
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(source_features, "cold_replay", lambda _path: [])
    assert (
        source_features.main(["--date", source_features.RUN_DATE, "--cold-replay", str(candidate)])
        == 0
    )
    monkeypatch.setattr(source_features, "cold_replay", lambda _path: ["failure"])
    assert (
        source_features.main(["--date", source_features.RUN_DATE, "--cold-replay", str(candidate)])
        == 1
    )
    calls: list[tuple[Path, str, Path]] = []
    monkeypatch.setattr(
        source_features,
        "run_experiment",
        lambda root, date, output_path: calls.append((root, date, output_path)),
    )
    output = tmp_path / "result.json"
    assert source_features.main(["--date", source_features.RUN_DATE, "--output", str(output)]) == 0
    assert calls == [(source_features.REPO_ROOT, source_features.RUN_DATE, output)]


def test_req_auto_7412_monotonic_receipt_uses_relative_bounds() -> None:
    """REQ-AUTO-7412 preserves duration without timestamp-like metric collisions."""

    assert source_features.relative_monotonic_bounds(2_100_574_761, 2_100_591_782) == (
        0,
        17_021,
    )
    with pytest.raises(ValueError, match="boundary order"):
        source_features.relative_monotonic_bounds(2, 1)
