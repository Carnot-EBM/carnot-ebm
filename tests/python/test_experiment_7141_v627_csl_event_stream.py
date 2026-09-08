"""Tests for the authentic V626 chronological event stream.

Spec refs: REQ-SELF-7141, SCENARIO-SELF-7141-INITIALIZE,
SCENARIO-SELF-7141-SELECTION, SCENARIO-SELF-7141-CHRONOLOGY,
SCENARIO-SELF-7141-REVEAL, SCENARIO-SELF-7141-REPLAY,
SCENARIO-SELF-7141-MUTATION, and SCENARIO-SELF-7141-VERDICT.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

import pytest

from carnot import experiment_7141_v627_csl_event_stream as mod


REPO = Path(__file__).resolve().parents[2]
RESULT = REPO / "results/experiment_7141_v627_csl_event_stream.json"


@pytest.fixture(scope="module")
def source_bundle() -> dict[str, Any]:
    """REQ-SELF-7141 loads only durable Exp7130 and Exp7129 source bytes."""

    return mod.load_source_bundle()


@pytest.fixture(scope="module")
def selected(source_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """SCENARIO-SELF-7141-SELECTION seals selection before exact scoring."""

    return mod.select_event_sources(source_bundle)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-SELF-7141 builds one isolated complete artifact for focused checks."""

    path = tmp_path_factory.mktemp("exp7141") / "artifact.json"
    return mod.run(result_path=path)


def test_spec_precedes_implementation_and_principles_are_complete() -> None:
    """REQ-SELF-7141 exists before code and explains every required field."""

    spec = (REPO / "openspec/capabilities/self-learning/spec.md").read_text(encoding="utf-8")
    for anchor in (
        "REQ-SELF-7141",
        "SCENARIO-SELF-7141-INITIALIZE",
        "SCENARIO-SELF-7141-SELECTION",
        "SCENARIO-SELF-7141-CHRONOLOGY",
        "SCENARIO-SELF-7141-REVEAL",
        "SCENARIO-SELF-7141-REPLAY",
        "SCENARIO-SELF-7141-MUTATION",
        "SCENARIO-SELF-7141-VERDICT",
    ):
        assert anchor in spec
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(mod.FIELD_PRINCIPLES)
    assert all(str(value).strip() for value in mod.FIELD_PRINCIPLES.values())


def test_raw_manifest_and_every_selected_receipt_are_content_addressed(
    source_bundle: dict[str, Any], selected: list[dict[str, Any]]
) -> None:
    """REQ-SELF-7141 hashes raw bytes and each selected evidence surface."""

    assert len(source_bundle["raw_manifest"]) == 3
    assert source_bundle["raw_manifest_hash"].startswith("sha256:")
    assert len(source_bundle["raw_rows"]) == 541
    assert len(selected) == 108
    for row in selected:
        assert row["arm"] == "single_shot"
        assert row["stage"] == "proposal"
        assert row["prompt_hash"] == mod.sha256_text(row["prompt"])
        assert row["raw_output_hash"] == mod.sha256_text(row["raw_text"])
        assert row["raw_line_hash"].startswith("sha256:")
        assert row["model_receipt_hash"].startswith("sha256:")
        assert row["family_receipt_hash"].startswith("sha256:")
        assert row["hardness_receipt_hash"].startswith("sha256:")


def test_selection_is_balanced_and_sealed_before_outcome_inspection(
    source_bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-SELF-7141-SELECTION uses strata before parse or exact labels."""

    def forbidden(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("selection inspected an exact outcome")

    monkeypatch.setattr(mod, "_score_raw", forbidden)
    rows = mod.select_event_sources(source_bundle)
    cells = Counter((row["model_id"], row["family"], row["hardness"]) for row in rows)

    assert len(cells) == 9
    assert set(cells.values()) == {12}
    assert [row["source_position"] for row in rows] == sorted(
        row["source_position"] for row in rows
    )


def test_complete_artifact_has_four_disjoint_immutable_splits(built: dict[str, Any]) -> None:
    """SCENARIO-SELF-7141-CHRONOLOGY freezes all events into one split each."""

    assert built["event_count"] == 108
    assert [row["chronology_index"] for row in built["event_rows"]] == list(range(108))
    assert [row["split"] for row in built["split_rows"]] == list(mod.SPLIT_NAMES)
    assert [row["event_count"] for row in built["split_rows"]] == [27, 27, 27, 27]
    split_ids = [event_id for row in built["split_rows"] for event_id in row["event_ids"]]
    assert len(split_ids) == len(set(split_ids)) == 108
    assert [row["event_id"] for row in built["event_rows"]] == split_ids
    assert all(row["split_hash"].startswith("sha256:") for row in built["split_rows"])


def test_decision_view_contains_no_outcome_and_reveal_requires_action(
    built: dict[str, Any],
) -> None:
    """SCENARIO-SELF-7141-REVEAL withholds the exact result until action seal."""

    stream = mod.SealedEventStream(built["event_rows"], built["outcome_reveal_rows"])
    first = stream.current_event()
    forbidden = {"parsed", "parse_success", "exact_success", "exact_outcome", "label", "witness"}

    assert not (mod.nested_keys(first) & forbidden)
    with pytest.raises(mod.OutcomeAccessError, match="sealed_action_receipt_required"):
        stream.reveal_outcome(first["event_id"], {})
    receipt = stream.seal_action(first["event_id"], {"decision": "accept"})
    revealed = stream.reveal_outcome(first["event_id"], receipt)
    assert revealed["event_id"] == first["event_id"]
    assert "exact_outcome_receipt" in revealed
    assert stream.current_event()["chronology_index"] == 1


def test_future_label_and_tampered_action_receipts_fail_closed(built: dict[str, Any]) -> None:
    """SCENARIO-SELF-7141-REVEAL rejects future and forged receipt access."""

    stream = mod.SealedEventStream(built["event_rows"], built["outcome_reveal_rows"])
    first, second = built["event_rows"][:2]
    with pytest.raises(mod.OutcomeAccessError, match="event_not_current"):
        stream.reveal_outcome(second["event_id"], {})
    with pytest.raises(mod.OutcomeAccessError, match="event_not_current"):
        stream.seal_action(second["event_id"], {"decision": "accept"})
    with pytest.raises(mod.OutcomeAccessError, match="action_contains_hidden_outcome"):
        stream.seal_action(first["event_id"], {"exact_success": True})
    receipt = stream.seal_action(first["event_id"], {"decision": "accept"})
    receipt["action_hash"] = "sha256:" + "0" * 64
    with pytest.raises(mod.OutcomeAccessError, match="action_receipt_mismatch"):
        stream.reveal_outcome(first["event_id"], receipt)


def test_independent_loader_reproduces_order_splits_labels_and_hashes(
    built: dict[str, Any],
) -> None:
    """SCENARIO-SELF-7141-REPLAY reconstructs all sealed evidence from raw rows."""

    replay = mod.independent_replay()
    expected = mod.artifact_replay_projection(built)

    assert replay == expected
    assert {row["check"] for row in built["independent_loader_rows"]} == {
        "ordering",
        "partitions",
        "labels",
        "receipt_hashes",
    }
    assert all(row["passed"] for row in built["independent_loader_rows"])


def test_required_mutations_are_detected_and_invalidate_readiness(built: dict[str, Any]) -> None:
    """SCENARIO-SELF-7141-MUTATION catches all four required attacks."""

    rows = built["mutation_rows"]
    assert {row["mutation"] for row in rows} == {
        "future_label_access",
        "reordered_events",
        "missing_raw_bytes",
        "cross_split_duplication",
    }
    assert all(row["detected"] and row["readiness_after_mutation"] == 0 for row in rows)


def test_validator_rejects_reorder_missing_bytes_duplicate_split_and_learning_claim(
    built: dict[str, Any],
) -> None:
    """SCENARIO-SELF-7141-MUTATION makes stored evidence own readiness."""

    attacks: list[tuple[str, Any]] = []
    reordered = deepcopy(built)
    reordered["event_rows"][0], reordered["event_rows"][1] = (
        reordered["event_rows"][1],
        reordered["event_rows"][0],
    )
    attacks.append(("chronological_order_mismatch", reordered))
    missing = deepcopy(built)
    missing["event_rows"][0]["raw_text"] = missing["event_rows"][0]["raw_text"][:-1]
    attacks.append(("raw_output_hash_mismatch", missing))
    duplicate = deepcopy(built)
    duplicate["split_rows"][1]["event_ids"].append(duplicate["split_rows"][0]["event_ids"][0])
    attacks.append(("cross_split_duplication", duplicate))
    learning = deepcopy(built)
    learning["learning_claim_made"] = True
    attacks.append(("learning_claim_forbidden", learning))

    for expected, attacked in attacks:
        attacked["reproducibility_checksum"] = mod.artifact_checksum(attacked)
        assert expected in mod.validate_artifact(attacked, replay_sources=False)


def test_blocked_source_is_schema_complete_and_never_partial(tmp_path: Path) -> None:
    """SCENARIO-SELF-7141-INITIALIZE writes exact diagnostics for a missing source."""

    result = tmp_path / "blocked.json"
    artifact = mod.run(
        result_path=result,
        source_artifact_path=tmp_path / "missing-7130.json",
        bank_path=mod.BANK_PATH,
        raw_dir=mod.RAW_DIR,
    )

    assert result.is_file()
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["csl_event_stream_ready_score"] == 0
    assert artifact["gate_check_summary"] == {
        "checks": [
            {
                "check": "source_artifact_exists",
                "expected_value": True,
                "observed_value": False,
                "passed": False,
            }
        ],
        "failed_check": "source_artifact_exists",
        "expected_value": True,
        "observed_value": False,
        "passed": False,
    }
    assert mod.validate_artifact(artifact, replay_sources=False) == []


def test_raw_byte_loss_is_disqualified_with_exact_manifest_diagnostic(tmp_path: Path) -> None:
    """SCENARIO-SELF-7141-MUTATION never fills a missing raw row from counts."""

    raw_dir = tmp_path / "raw"
    shutil.copytree(mod.RAW_DIR, raw_dir)
    damaged = raw_dir / Path(mod.load_source_bundle()["raw_manifest"][0]["path"]).name
    damaged.write_bytes(damaged.read_bytes()[:-1])
    result = tmp_path / "disqualified.json"
    artifact = mod.run(result_path=result, raw_dir=raw_dir)

    assert artifact["verdict_class"] == "disqualified"
    assert artifact["honest_verdict"].startswith("disqualified_")
    assert artifact["csl_event_stream_ready_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] == "raw_shard_sha256"
    assert artifact["gate_check_summary"]["expected_value"].startswith("sha256:")
    assert artifact["gate_check_summary"]["observed_value"].startswith("sha256:")
    assert mod.validate_artifact(artifact, replay_sources=False) == []


def test_initial_artifact_exists_before_manifest_loader_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-SELF-7141-INITIALIZE writes the complete schema unconditionally."""

    result = tmp_path / "initialized.json"

    def stop_after_check(**_kwargs: Any) -> dict[str, Any]:
        initialized = json.loads(result.read_text(encoding="utf-8"))
        assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(initialized)
        raise mod.SourceEvidenceError("blocked", "probe_stop", True, False)

    monkeypatch.setattr(mod, "load_source_bundle", stop_after_check)
    artifact = mod.run(result_path=result)

    assert artifact["gate_check_summary"]["failed_check"] == "probe_stop"
    assert artifact["verdict_class"] == "blocked"


def test_complete_readiness_is_not_a_learning_claim_and_artifact_validates(
    built: dict[str, Any],
) -> None:
    """SCENARIO-SELF-7141-VERDICT keeps readiness separate from learning value."""

    assert built["csl_event_stream_ready_score"] == 1
    assert built["learning_claim_made"] is False
    assert built["verifier_is_oracle"] is False
    assert built["verdict_class"] == "positive"
    assert built["honest_verdict"] == "positive_csl_event_stream_ready_no_learning_claim"
    assert mod.validate_artifact(built) == []


def test_command_writes_and_validates_only_requested_artifact(tmp_path: Path) -> None:
    """REQ-SELF-7141 provides an isolated command and independent validation path."""

    result = tmp_path / "command.json"
    assert mod.main(["--date", "20260908", "--result-path", str(result)]) == 0
    assert mod.main(["--result-path", str(result), "--validate"]) == 0
    artifact = json.loads(result.read_text(encoding="utf-8"))
    assert artifact["event_count"] == 108
    assert artifact["csl_event_stream_ready_score"] == 1


def test_repository_deliverable_is_present_and_valid() -> None:
    """REQ-SELF-7141 leaves the requested stable result in the research record."""

    artifact = json.loads(RESULT.read_text(encoding="utf-8"))
    assert artifact["run_date"] == "20260908"
    assert artifact["event_count"] >= 96
    assert mod.validate_artifact(artifact) == []
