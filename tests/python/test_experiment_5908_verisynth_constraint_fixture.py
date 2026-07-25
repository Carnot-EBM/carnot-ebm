"""Tests for Exp5908 VeriSynth ConstraintIR fixture planning.

Spec refs: REQ-BENCH-5908, SCENARIO-BENCH-5908-DECOMPOSITION,
SCENARIO-BENCH-5908-RETRIEVAL, SCENARIO-BENCH-5908-CONTROLS,
SCENARIO-BENCH-5908-STREAM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896
from carnot import experiment_5908_verisynth_constraint_fixture as exp5908


def _load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# REQ-BENCH-5908, SCENARIO-BENCH-5908-DECOMPOSITION
def test_decomposition_units_map_to_supported_ir_and_replay() -> None:
    rows = exp5908.build_prompt_plan_rows()

    seen_unit_types: set[str] = set()
    for row in rows:
        plan = row["decomposition_plan"]
        components = plan["components"]
        seen_unit_types.update(str(component["unit_type"]) for component in components)

        assert components
        assert all(component["component_hash"].startswith("sha256:") for component in components)
        assert all(component["maps_to_supported_typed_ir"] is True for component in components)
        assert all(component["source_ir_pointer"].startswith("/") for component in components)
        assert len(row["exact_replay_receipt"]["component_replay"]) == len(components)
        assert all(
            replay["replay_ok"] is True
            for replay in row["exact_replay_receipt"]["component_replay"]
        )

    assert seen_unit_types == set(exp5908.DECOMPOSITION_UNIT_TYPES)


# REQ-BENCH-5908, SCENARIO-BENCH-5908-RETRIEVAL
def test_retrieval_visibility_excludes_held_and_same_groups() -> None:
    rows = exp5908.build_prompt_plan_rows()
    artifact = exp5908.build_artifact(rows, root=exp5908.REPO_ROOT, duration_s=0.0)
    visibility = artifact["retrieval_index_and_visibility_contract"]

    assert visibility["held_semantic_variants_enter_surface"] is False
    assert "heldout" in visibility["excluded_splits"]
    assert visibility["same_group_exclusion"] is True
    assert all(entry["split"] != "heldout" for entry in visibility["index_entries"])

    rows_by_id = {row["source_row_id"]: row for row in rows}
    for row in rows_by_id.values():
        for arm_id in ("decomposition_plus_exact_example_retrieval", "wrong_family_retrieval"):
            exemplars = row["prompt_plan_arms"][arm_id]["exemplars"]
            assert len(exemplars) == exp5908.EXEMPLARS_PER_RETRIEVAL_ARM
            for exemplar in exemplars:
                assert exemplar["split"] != "heldout"
                assert exemplar["group_id"] != row["group_id"]
                if row["split"] == "heldout":
                    assert exemplar["family"] != row["family"]


# REQ-BENCH-5908, SCENARIO-BENCH-5908-CONTROLS
def test_prompt_plan_controls_are_matched_and_nontrivial() -> None:
    rows = exp5908.build_prompt_plan_rows()
    artifact = exp5908.build_artifact(rows, root=exp5908.REPO_ROOT, duration_s=0.0)

    assert set(artifact["prompt_plan_arm_definitions"]) == set(exp5908.PROMPT_PLAN_ARMS)
    token_budgets = {
        arm["token_envelope"]["max_tokens"]
        for arm in artifact["prompt_plan_arm_definitions"].values()
    }
    assert token_budgets == {exp5908.TOKEN_ENVELOPE_MAX_TOKENS}
    assert artifact["token_envelope_and_exemplar_count_parity"]["all_token_envelopes_match"] is True
    assert (
        artifact["token_envelope_and_exemplar_count_parity"]["retrieval_exemplar_counts_match"]
        is True
    )

    controls = artifact["wrong_family_shuffled_omitted_and_no_information_controls"]
    assert controls["wrong_family_retrieval"]["nontrivial"] is True
    assert controls["shuffled_decomposition"]["nontrivial"] is True
    assert controls["omitted_component_decomposition"]["nontrivial"] is True
    assert controls["no_information_retrieval"]["nontrivial"] is True

    for row in rows:
        arms = row["prompt_plan_arms"]
        assert arms["direct"]["component_hashes"] == []
        assert arms["semantic_decomposition"]["component_hashes"] != []
        assert (
            arms["semantic_decomposition"]["component_hashes"]
            != arms["shuffled_decomposition"]["component_hashes"]
        )
        assert len(arms["omitted_component_decomposition"]["component_hashes"]) == (
            len(arms["semantic_decomposition"]["component_hashes"]) - 1
        )
        assert (
            arms["decomposition_plus_exact_example_retrieval"]["exemplar_count"]
            == (arms["wrong_family_retrieval"]["exemplar_count"])
            == arms["no_information_retrieval"]["exemplar_count"]
        )


# REQ-BENCH-5908, SCENARIO-BENCH-5908-STREAM
def test_write_artifact_rows_and_consumer_stream_contract_are_stable(tmp_path: Path) -> None:
    artifact = exp5908.write_fixture(root=tmp_path, duration_s=0.0)
    result_path = tmp_path / exp5908.RESULT_RELATIVE_PATH
    row_path = tmp_path / exp5908.ROW_FILE_RELATIVE_PATH
    rows = _load_rows(row_path)

    assert result_path.exists()
    assert row_path.exists()
    assert set(exp5908.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["status"] == "ready"
    assert artifact["honest_verdict"].startswith("ready:")
    assert artifact["inference_substrate"] == exp5908.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["verisynth_constraint_fixture_ready_score"] == 1.0
    assert artifact["row_file_receipt"]["sha256"] == exp5908.sha256_file(row_path)
    assert (
        artifact["row_file_receipt"]["row_count"] == len(rows) == len(exp5896.build_fixture_rows())
    )
    assert artifact["upstream_gate_and_hashes"]["exp5907_replay_ok"] is True
    assert artifact["exact_exemplar_and_component_replay"]["all_exact_replay_ok"] is True
    assert artifact["consumer_stream_contract"]["consumer"] == "experiment_5909"
    assert artifact["consumer_stream_contract"]["row_hashes"] == [row["row_hash"] for row in rows]
    assert artifact["consumer_stream_contract"]["consumer_stream_hash"].startswith("sha256:")
    assert artifact["protected_files_unchanged"]["unchanged"] is True

    exp5908.validate_artifact(artifact)
    replay = exp5908.replay_artifact(root=tmp_path)
    assert replay["ok"] is True
    assert replay["row_file_sha256"] == artifact["row_file_receipt"]["sha256"]

    second = exp5908.write_fixture(root=tmp_path, duration_s=0.0)
    assert second["reproducibility_checksum"] == artifact["reproducibility_checksum"]


# REQ-BENCH-5908, SCENARIO-BENCH-5908-STREAM
def test_artifact_validation_fails_closed(tmp_path: Path) -> None:
    artifact = exp5908.write_fixture(root=tmp_path, duration_s=0.0)

    for key, value, message in [
        ("honest_verdict", "complete: bad prefix", "ready score"),
        ("verisynth_constraint_fixture_ready_score", 0.5, "bare"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("verifier_is_oracle", False, "verifier_is_oracle"),
    ]:
        broken = json.loads(json.dumps(artifact))
        broken[key] = value
        with pytest.raises(ValueError, match=message):
            exp5908.validate_artifact(broken)

    missing = dict(artifact)
    del missing["consumer_stream_contract"]
    with pytest.raises(ValueError, match="missing required"):
        exp5908.validate_artifact(missing)

    row_path = tmp_path / exp5908.ROW_FILE_RELATIVE_PATH
    row_path.write_text(row_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(exp5908.Exp5908ReplayError, match="row file hash"):
        exp5908.replay_artifact(root=tmp_path)

    artifact = exp5908.write_fixture(root=tmp_path, duration_s=0.0)
    result_path = tmp_path / exp5908.RESULT_RELATIVE_PATH
    rows = _load_rows(row_path)
    rows[0]["source_row_id"] = "tampered-row"
    row_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )
    updated = json.loads(result_path.read_text(encoding="utf-8"))
    updated["row_file_receipt"]["sha256"] = exp5908.sha256_file(row_path)
    result_path.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(exp5908.Exp5908ReplayError, match="row file content"):
        exp5908.replay_artifact(root=tmp_path)

    exp5908.write_fixture(root=tmp_path, duration_s=0.0)
    broken_checksum = json.loads(result_path.read_text(encoding="utf-8"))
    broken_checksum["reproducibility_checksum"] = "sha256:" + "0" * 64
    result_path.write_text(
        json.dumps(broken_checksum, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(exp5908.Exp5908ReplayError, match="reproducibility checksum"):
        exp5908.replay_artifact(root=tmp_path)

    assert artifact["schema"] == exp5908.ARTIFACT_SCHEMA_VERSION
