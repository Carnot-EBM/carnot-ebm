"""Tests for Exp 3018 BEAVER-style validator frontier certificate.

Spec refs: REQ-VERIFY-3018, SCENARIO-VERIFY-3018.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import beaver_style_validator_frontier_certificate_v1 as exp
from carnot.eval import nsvif_instruction_validator_tree_expansion_v1 as exp3017


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_3018_beaver_style_validator_frontier_certificate_v1.py"


def _exp3017_config(tmp_path: Path) -> exp3017.ExperimentConfig:
    return exp3017.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp3017.ARTIFACT_FILENAME,
        manifest_path=tmp_path / exp3017.VALIDATOR_MANIFEST_REL_PATH,
        z3_transcript_dir=tmp_path / exp3017.Z3_TRANSCRIPT_REL_DIR,
        runtime_transcript_dir=tmp_path / exp3017.RUNTIME_TRANSCRIPT_REL_DIR,
        started_at=10.0,
        clock=lambda: 12.0,
    )


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        certificate_manifest_path=tmp_path / exp.CERTIFICATE_MANIFEST_REL_PATH,
        transcript_dir=tmp_path / exp.TRANSCRIPT_REL_DIR,
        source_artifact_path=tmp_path / "results" / exp3017.ARTIFACT_FILENAME,
        source_manifest_path=tmp_path / exp3017.VALIDATOR_MANIFEST_REL_PATH,
        started_at=20.0,
        clock=lambda: 23.5,
    )


def _write_exp3017(tmp_path: Path) -> dict[str, object]:
    return exp3017.run_experiment(_exp3017_config(tmp_path))


def test_req_verify_3018_spec_and_template_script_anchor_exists() -> None:
    """REQ-VERIFY-3018: Exp 3018 is OpenSpec anchored and template-runnable."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3018" in spec
    assert "SCENARIO-VERIFY-3018" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "frontier_certificate_ready" in spec
    assert "certificate_manifest_path" in spec
    assert "unresolved_count" in spec
    assert SCRIPT_PATH.exists()
    assert "ExperimentTemplate" in script


def test_scenario_verify_3018_classifies_prefix_and_nonprefix_nodes() -> None:
    """SCENARIO-VERIFY-3018: validator trees expose prefix and frontier status."""

    items = {item.item_id: item for item in exp3017.build_instruction_items()}

    forbidden_tree = exp3017.build_validator_tree(items["if-3017-003"])
    forbidden_class = exp.classify_validator_tree(forbidden_tree, cached_candidates_available=True)
    assert "json_forbidden_tokens" in forbidden_class["prefix_closed_node_kinds"]
    assert forbidden_class["frontier_explorable"] is True
    assert forbidden_class["non_prefix_closed_node_ids"] == []

    semantic_tree = exp3017.build_validator_tree(items["if-3017-001"])
    semantic_class = exp.classify_validator_tree(semantic_tree, cached_candidates_available=True)
    assert semantic_class["frontier_explorable"] is True
    assert semantic_class["non_prefix_closed_node_ids"] == ["if-3017-001:semantic_boundary:3"]

    z3_tree = exp3017.build_validator_tree(items["if-3017-012"])
    z3_class = exp.classify_validator_tree(z3_tree, cached_candidates_available=True)
    assert z3_class["prefix_closed_node_ids"] == []
    assert z3_class["bounded_frontier_node_kinds"] == ["z3_linear_relation"]


def test_scenario_verify_3018_writes_manifest_transcripts_and_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3018: cached candidates become replayable certificate rows."""

    _write_exp3017(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text(encoding="utf-8"))
    manifest_path = tmp_path / artifact["certificate_manifest_path"]
    manifest_rows = exp.load_certificate_manifest(manifest_path)

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["frontier_certificate_ready"] is True
    assert artifact["certificate_manifest_path"] == str(exp.CERTIFICATE_MANIFEST_REL_PATH)
    assert artifact["n_frontier_items"] == len(manifest_rows)
    assert artifact["n_prefix_closed_items"] == sum(
        1 for row in manifest_rows if row["prefix_closed_assumption_applies"]
    )
    assert artifact["certified_safe_count"] == exp3017.MIN_INSTRUCTION_ITEMS
    assert artifact["certified_violating_count"] == exp3017.MIN_INSTRUCTION_ITEMS
    assert artifact["unresolved_count"] == 2
    assert artifact["non_prefix_closed_count"] >= 2
    assert artifact["enumerator_fallback_separated"] is True
    assert artifact["live_llm_evidence_used"] is False
    assert len(artifact["transcript_paths"]) == exp3017.MIN_INSTRUCTION_ITEMS
    assert artifact["duration_s"] == pytest.approx(3.5)
    assert artifact["honest_verdict"].startswith("complete:")

    candidate_rows = [row for row in manifest_rows if row["row_type"] == "candidate_frontier"]
    assert len(candidate_rows) == exp3017.MIN_INSTRUCTION_ITEMS * 2
    assert {row["certificate_status"] for row in candidate_rows} == {
        "certified_safe",
        "certified_violating",
    }
    assert {row["candidate_role"] for row in candidate_rows} == {"known_good", "known_bad"}

    for row in candidate_rows:
        assert row["frontier_exploration"]["mode"] == "cached_candidate_set"
        assert row["frontier_exploration"]["candidate_set_size"] == 2
        assert row["deterministic_validator_outcome"]["llm_judge_used"] is False
        assert row["probability_bound_placeholder"] == exp.PROBABILITY_NOT_COMPUTED
        if row["certificate_status"] == "certified_safe":
            assert row["deterministic_validator_outcome"]["accepted"] is True
        else:
            assert row["deterministic_validator_outcome"]["accepted"] is False
            assert row["deterministic_validator_outcome"]["failing_node_ids"]

    non_prefix_rows = [row for row in manifest_rows if row["certificate_status"] == "non_prefix_closed"]
    unresolved_rows = [row for row in manifest_rows if row["certificate_status"] == "unresolved"]
    assert any(row["row_type"] == "non_prefix_closed_node" for row in non_prefix_rows)
    assert {row["source_rejection_reason"] for row in unresolved_rows} == {
        "nondeterministic_validator",
        "llm_only_label",
    }

    for rel_path in artifact["transcript_paths"]:
        path = tmp_path / rel_path
        assert path.is_file()
        assert exp.sha256_file(path) == next(
            row["transcript_sha256"]
            for row in manifest_rows
            if row.get("transcript_path") == rel_path
        )

    exp.validate_artifact(artifact)


def test_req_verify_3018_validation_rejects_contaminated_or_inconsistent_artifacts(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3018: certificate readiness rejects mixed or inconsistent evidence."""

    _write_exp3017(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))
    exp.validate_artifact(artifact)

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="live_llm_evidence_used"):
        exp.validate_artifact(artifact | {"live_llm_evidence_used": True})
    with pytest.raises(ValueError, match="enumerator_fallback_separated"):
        exp.validate_artifact(artifact | {"enumerator_fallback_separated": False})
    with pytest.raises(ValueError, match="certified_safe_count"):
        exp.validate_artifact(artifact | {"certified_safe_count": 0})
    with pytest.raises(ValueError, match="certified_violating_count"):
        exp.validate_artifact(artifact | {"certified_violating_count": 0})
    with pytest.raises(ValueError, match="n_frontier_items"):
        exp.validate_artifact(artifact | {"n_frontier_items": 0})
    with pytest.raises(ValueError, match="n_prefix_closed_items"):
        exp.validate_artifact(artifact | {"n_prefix_closed_items": 0})
    with pytest.raises(ValueError, match="transcript_paths"):
        exp.validate_artifact(artifact | {"transcript_paths": []})
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(artifact | {"honest_verdict": "ready: wrong prefix"})


def test_req_verify_3018_missing_exp3017_blocks_honestly(tmp_path: Path) -> None:
    """REQ-VERIFY-3018: missing source manifest blocks without fabricating rows."""

    artifact = exp.run_experiment(_config(tmp_path))

    assert artifact["frontier_certificate_ready"] is False
    assert artifact["n_frontier_items"] == 0
    assert artifact["n_prefix_closed_items"] == 0
    assert artifact["certified_safe_count"] == 0
    assert artifact["certified_violating_count"] == 0
    assert artifact["unresolved_count"] == 1
    assert artifact["live_llm_evidence_used"] is False
    assert artifact["enumerator_fallback_separated"] is True
    assert artifact["honest_verdict"].startswith("blocked:")
    assert (tmp_path / "results" / exp.ARTIFACT_FILENAME).is_file()
    assert not (tmp_path / exp.CERTIFICATE_MANIFEST_REL_PATH).exists()

    exp.validate_artifact(artifact)


def test_req_verify_3018_source_blockers_are_distinct(tmp_path: Path) -> None:
    """REQ-VERIFY-3018: malformed, unready, and missing-manifest sources block."""

    cfg = _config(tmp_path)
    source_path = cfg.resolved_source_artifact_path()
    source_path.parent.mkdir(parents=True, exist_ok=True)

    source_path.write_text("{", encoding="utf-8")
    malformed = exp.run_experiment(cfg)
    assert malformed["blocked_reason"] == "exp3017_artifact_malformed"

    source_path.write_text(
        json.dumps({"instruction_validator_tree_ready": False}),
        encoding="utf-8",
    )
    not_ready = exp.run_experiment(cfg)
    assert not_ready["blocked_reason"] == "exp3017_not_ready"

    source_path.write_text(
        json.dumps({"instruction_validator_tree_ready": True}),
        encoding="utf-8",
    )
    missing_manifest = exp.run_experiment(cfg)
    assert missing_manifest["blocked_reason"] == "exp3017_manifest_missing"

    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(missing_manifest | {"honest_verdict": "waiting"})


def test_req_verify_3018_unresolved_manifest_rows_and_hash_mismatches(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3018: source rows without matching cached candidates are unresolved."""

    source_artifact = _write_exp3017(tmp_path)
    manifest = exp3017.load_manifest(tmp_path / exp3017.VALIDATOR_MANIFEST_REL_PATH)
    missing_item_row = dict(manifest[0], item_id="if-3017-unknown")
    good_hash_mismatch = dict(manifest[0], known_good_candidate_sha256="bad-good")
    bad_hash_mismatch = dict(manifest[0], known_bad_candidate_sha256="bad-bad")

    rows = exp.build_certificate_rows(
        _config(tmp_path),
        [missing_item_row, good_hash_mismatch, bad_hash_mismatch],
        source_artifact,
    )

    assert [row["source_rejection_reason"] for row in rows[:3]] == [
        "cached_candidate_missing",
        "known_good_candidate_hash_mismatch",
        "known_bad_candidate_hash_mismatch",
    ]
    assert all(row["certificate_status"] == "unresolved" for row in rows[:3])


def test_req_verify_3018_exp3004_provenance_stays_separate(tmp_path: Path) -> None:
    """REQ-VERIFY-3018: cited live and fallback transcript paths remain separated."""

    _write_exp3017(tmp_path)
    exp3004_path = (
        tmp_path
        / "results"
        / "experiment_3004_aquaforte_beaver_live_retry_provenance_v2.json"
    )
    exp3004_path.write_text(
        json.dumps(
            {
                "enumerator_fallback_separated": True,
                "live_transcript_paths": ["results/raw/live.json"],
                "enumerator_fallback_paths": ["results/raw/fallback.json"],
            }
        ),
        encoding="utf-8",
    )

    artifact = exp.run_experiment(_config(tmp_path))

    assert artifact["frontier_certificate_ready"] is True
    assert artifact["source_artifacts"]["exp3004_boundary_only"]["present"] is True
    assert artifact["enumerator_fallback_provenance"] == {
        "present": True,
        "paths": ["results/raw/fallback.json"],
        "separated_from_live": True,
    }

    exp3004_path.write_text(
        json.dumps(
            {
                "enumerator_fallback_separated": True,
                "live_transcript_paths": ["results/raw/shared.json"],
                "enumerator_fallback_paths": ["results/raw/shared.json"],
            }
        ),
        encoding="utf-8",
    )
    contaminated = exp.run_experiment(_config(tmp_path))
    assert contaminated["frontier_certificate_ready"] is False
    assert contaminated["enumerator_fallback_separated"] is False
    assert contaminated["honest_verdict"].startswith("blocked:")
    exp.validate_artifact(contaminated)


def test_req_verify_3018_cli_and_json_reader_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3018: the module CLI can run against explicit cached sources."""

    _write_exp3017(tmp_path)
    output = tmp_path / "results" / "cli_artifact.json"
    manifest = tmp_path / "results" / "cli_manifest.jsonl"
    transcripts = tmp_path / "results" / "cli_transcripts"

    exit_code = exp.main(
        [
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--transcript-dir",
            str(transcripts),
            "--source-artifact",
            str(tmp_path / "results" / exp3017.ARTIFACT_FILENAME),
            "--source-manifest",
            str(tmp_path / exp3017.VALIDATOR_MANIFEST_REL_PATH),
        ]
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert loaded["frontier_certificate_ready"] is True
    assert manifest.is_file()

    list_json = tmp_path / "results" / "not_a_dict.json"
    list_json.write_text("[]", encoding="utf-8")
    assert exp._read_json(list_json) == {}
