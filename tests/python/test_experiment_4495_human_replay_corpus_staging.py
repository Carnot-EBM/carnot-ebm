"""Tests for Exp 4495 ARC human replay corpus staging.

Spec refs: REQ-ARC-FCP-4495, SCENARIO-ARC-FCP-4495.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_4495_human_replay_corpus_staging as exp4495
from carnot.agentic import arc_human_replay_corpus as corpus


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _session_row() -> dict[str, object]:
    trajectory = [
        {
            "timestamp": "2026-03-02T00:00:00+00:00",
            "data": {"frame": [[0, 0], [0, 0]], "action": {"id": 6, "x": 0, "y": 0}},
        },
        {
            "timestamp": "2026-03-02T00:00:01+00:00",
            "data": {"frame": [[0, 1], [0, 0]], "action": {"id": 1}},
        },
        {
            "timestamp": "2026-03-02T00:00:02+00:00",
            "data": {"frame": [[0, 1], [2, 0]], "action": {"id": 2}},
        },
    ]
    return {
        "env": "sk48",
        "guid": "fixture.recording",
        "trajectory": json.dumps(json.dumps(trajectory)),
        "total_actions": 2,
        "actions_by_level": [[[1, 1], [2, 2]]],
    }


def test_req_arc_fcp_4495_spec_declares_staging_contract() -> None:
    """REQ-ARC-FCP-4495: OpenSpec names the replay shard and artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4495" in spec
    assert "SCENARIO-ARC-FCP-4495" in spec
    assert exp4495.RESULT_RELATIVE_PATH in spec
    assert "frame_delta" in spec
    assert "level_progress" in spec
    for field, principle in exp4495.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_fcp_4495_shards_load_without_upstream_download(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-4495: loader reads staged rows from a gitignored data dir."""

    manifest = corpus.write_training_shards(
        [_session_row()],
        tmp_path / exp4495.DATA_RELATIVE_DIR,
        source_metadata={"source": "fixture"},
        max_examples_per_shard=1,
    )
    rows = list(corpus.load_training_shards(tmp_path / exp4495.DATA_RELATIVE_DIR))

    assert manifest["example_count"] == 2
    assert manifest["shard_count"] == 2
    assert len(rows) == 2
    assert set(rows[0]) >= {"frame", "action", "frame_delta", "level_progress"}
    assert rows[0]["frame"] == [[0, 0], [0, 0]]
    assert rows[0]["frame_delta"] == pytest.approx(0.25)
    assert rows[0]["level_progress"] == pytest.approx(0.5)
    assert rows[1]["level_progress"] == pytest.approx(1.0)


def test_req_arc_fcp_4495_loader_edge_cases_stay_deterministic(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4495: defensive loader branches stay local and deterministic."""

    assert corpus.decode_trajectory("") == []
    assert corpus.decode_trajectory("{bad json") == []
    assert corpus.extract_frame("not-a-frame") is None
    assert corpus.extract_frame({"data": {"other": 1}}) is None
    assert corpus.extract_frame([[[0]], [[1]]]) == [[1]]
    assert corpus.extract_action("not-an-event") == {"missing": True}
    assert corpus.extract_action({"action": 1}) == 1
    assert corpus.extract_action({"data": {"action_input": {"id": "4"}}}) == {"id": "4"}
    assert corpus.extract_action({"data": {"other": 1}}) == {"missing": True}
    assert corpus.frame_delta([], [[1]]) == 0.0
    assert corpus.frame_delta([[1]], [[1, 2]]) == 1.0
    assert corpus.frame_delta([[]], [[]]) == 0.0
    assert corpus.level_progress({"total_actions": 4}, 2) == pytest.approx(0.5)
    assert corpus.level_progress({}, 2) == 0.0

    rows = [
        {
            "env": "edge",
            "guid": "fallback",
            "trajectory": json.dumps(
                [
                    {"frame": [[0]], "action": {"id": 6}},
                    {"frame": [[1]]},
                    "ignored",
                    {"frame": [[2]]},
                ]
            ),
            "actions_by_level": [[[1, 1], [2, 2]]],
        }
    ]
    manifest = corpus.write_training_shards(
        rows,
        tmp_path / "edge",
        max_examples_per_shard=4,
        max_examples=1,
    )
    limited_rows = list(corpus.load_training_shards(tmp_path / "edge", limit=1))

    assert corpus.extract_frame([[1, 2], [3]]) is None
    assert corpus.extract_frame([[1, "bad"]]) is None
    assert manifest["example_count"] == 1
    assert limited_rows[0]["action"] == {"id": 6}

    corpus.write_training_shards(rows, tmp_path / "edge", max_examples_per_shard=4)
    assert (tmp_path / "edge" / corpus.MANIFEST_NAME).exists()

    two_row_manifest = corpus.write_training_shards(
        rows,
        tmp_path / "limit",
        max_examples_per_shard=4,
    )
    assert two_row_manifest["example_count"] == 2
    assert len(list(corpus.load_training_shards(tmp_path / "limit", limit=1))) == 1

    assert corpus.extract_frame([[]]) is None


def test_req_arc_fcp_4495_artifact_records_provenance_and_no_weights(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4495: attributed mirror artifacts are terminal and non-fabricated."""

    corpus.write_training_shards(
        [_session_row()],
        tmp_path / exp4495.DATA_RELATIVE_DIR,
        source_metadata={"source": "fixture"},
    )
    preconditions = {
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade_import_smoke": True,
        "torch_import": True,
        "official_arc_shortlink_reachable": False,
        "hf_mirror_reachable": True,
        "source_shards_cached": True,
        "training_shards_present": True,
    }

    artifact = exp4495.build_artifact(
        root=tmp_path,
        preconditions_checked=preconditions,
        download_manifest={
            "source_kind": "hf_mirror",
            "mirror_url": exp4495.HF_DATASET_URL,
            "license_status": "mirror_attribution_required",
            "license_name": "CC BY 4.0-compatible attribution path only",
        },
    )
    errors = exp4495.artifact_schema_errors(artifact)

    assert errors == []
    assert artifact["honest_verdict"] == "complete: staged_attributed_mirror_no_weights"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["official_license_verified"] is False
    assert artifact["weights_committed"] is False
    assert artifact["training_shard_count"] == 1
    assert "ARC Prize" in artifact["attribution"]


def test_req_arc_fcp_4495_preconditions_are_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-ARC-FCP-4495: preconditions record exactly which resources were checked."""

    (tmp_path / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# test\n", encoding="utf-8")
    import carnot.agentic as agentic_pkg

    fake_kit = SimpleNamespace(offline_arcade=lambda: object())
    monkeypatch.setattr(agentic_pkg, "arc_solver_kit", fake_kit, raising=False)
    monkeypatch.setitem(sys.modules, "carnot.agentic.arc_solver_kit", fake_kit)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="fixture"))
    monkeypatch.setattr(
        exp4495,
        "_url_status",
        lambda url: {"url": url, "reachable": url == exp4495.HF_API_URL, "status_code": 200},
    )

    preconditions = exp4495.check_preconditions(tmp_path)

    assert preconditions["agents_md_read"] is True
    assert preconditions["codex_md_read"] is True
    assert preconditions["offline_arcade_import_smoke"] is True
    assert preconditions["torch_import"] is True
    assert preconditions["official_arc_shortlink_reachable"] is False
    assert preconditions["hf_mirror_reachable"] is True
    assert preconditions["training_shards_present"] is False


def test_req_arc_fcp_4495_missing_and_official_verdict_branches(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4495: verdicts distinguish missing shards from official licensing."""

    missing = exp4495.build_artifact(
        root=tmp_path,
        preconditions_checked={"training_shards_present": False},
        download_manifest={"source_kind": "blocked"},
    )
    assert missing["honest_verdict"] == "complete: blocked_human_replay_shards_missing"

    corpus.write_training_shards([_session_row()], tmp_path / exp4495.DATA_RELATIVE_DIR)
    official = exp4495.build_artifact(
        root=tmp_path,
        preconditions_checked={"training_shards_present": True},
        download_manifest={
            "source_kind": "official",
            "official_license_verified": True,
            "license_status": "official_cc0_mit0_verified",
        },
    )
    assert official["honest_verdict"] == "complete: official_human_replay_shards_staged_no_weights"
    assert official["official_license_verified"] is True


def test_req_arc_fcp_4495_run_writes_stable_json_from_cached_shards(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4495: run reuses cached shards and writes the required result JSON."""

    corpus.write_training_shards(
        [_session_row()],
        tmp_path / exp4495.DATA_RELATIVE_DIR,
        source_metadata={"source": "fixture"},
    )
    artifact = exp4495.run(
        root=tmp_path,
        preconditions_checked={
            "agents_md_read": True,
            "codex_md_read": True,
            "offline_arcade_import_smoke": True,
            "torch_import": True,
            "official_arc_shortlink_reachable": False,
            "hf_mirror_reachable": True,
            "source_shards_cached": True,
            "training_shards_present": True,
        },
        write=True,
        fetch_if_missing=False,
    )

    written = json.loads((tmp_path / exp4495.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["training_example_count"] == 2
    assert artifact["preconditions_checked"]["training_shards_present"] is True


def test_req_arc_fcp_4495_run_can_stage_when_cache_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-FCP-4495: run fetch path stages shards once, then reports them."""

    def fake_stage(root: Path, *, max_examples: int | None = None) -> dict[str, object]:
        assert max_examples == 3
        corpus.write_training_shards([_session_row()], root / exp4495.DATA_RELATIVE_DIR)
        return {"source_kind": "hf_mirror", "license_status": "mirror_attribution_required"}

    monkeypatch.setattr(exp4495, "stage_from_hf_mirror", fake_stage)
    artifact = exp4495.run(
        root=tmp_path,
        preconditions_checked={"hf_mirror_reachable": True},
        write=False,
        fetch_if_missing=True,
        max_examples=3,
    )

    assert artifact["training_example_count"] == 2
    assert artifact["preconditions_checked"]["training_shards_present"] is True


def test_req_arc_fcp_4495_schema_rejects_bad_principled_fields(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4495: schema rejects non-terminal verdicts and weight fabrication."""

    corpus.write_training_shards([_session_row()], tmp_path / exp4495.DATA_RELATIVE_DIR)
    valid = exp4495.build_artifact(
        root=tmp_path,
        preconditions_checked={"training_shards_present": True},
        download_manifest={"source_kind": "fixture"},
    )

    for mutate, expected in (
        (lambda item: item.pop("data_relative_dir"), "missing required"),
        (lambda item: item.__setitem__("honest_verdict", "done"), "terminal prefix"),
        (
            lambda item: item.__setitem__("inference_substrate", "live_llm_inference"),
            "inference_substrate",
        ),
        (lambda item: item.__setitem__("preconditions_checked", []), "preconditions_checked"),
        (lambda item: item.__setitem__("field_principles", {}), "field_principles"),
        (
            lambda item: item.update(weights_committed=True, official_license_verified=False),
            "weights require official",
        ),
        (lambda item: item.__setitem__("training_shard_count", -1), "training_shard_count"),
        (lambda item: item.__setitem__("training_example_count", -1), "training_example_count"),
        (lambda item: item.__setitem__("source_provenance", []), "source_provenance"),
    ):
        artifact = dict(valid)
        mutate(artifact)
        assert any(expected in error for error in exp4495.artifact_schema_errors(artifact))


def test_req_arc_fcp_4495_run_refuses_invalid_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-ARC-FCP-4495: run validates before writing terminal JSON."""

    corpus.write_training_shards([_session_row()], tmp_path / exp4495.DATA_RELATIVE_DIR)
    monkeypatch.setattr(exp4495, "artifact_schema_errors", lambda _artifact: ["bad artifact"])

    with pytest.raises(ValueError, match="bad artifact"):
        exp4495.run(
            root=tmp_path,
            preconditions_checked={"training_shards_present": True},
            write=True,
            fetch_if_missing=False,
        )
