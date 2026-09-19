"""Tests for the human-annotated RAGTruth protocol.

Spec refs: REQ-AUTO-7423 and SCENARIO-AUTO-7423-01 through
SCENARIO-AUTO-7423-05.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

import pytest

from carnot import experiment_7412_v650_source_features as source_features
from carnot import experiment_7423_v651_annotated_protocol as protocol


ROOT = Path(__file__).resolve().parents[2]


def _source(source_id: str, task_type: str = "Summary", value: Any = None) -> dict[str, Any]:
    source_value = value if value is not None else f"Source evidence for {source_id}."
    return {
        "source_id": source_id,
        "task_type": task_type,
        "source": "fixture-origin",
        "source_info": source_value,
        "prompt": "private generation prompt",
    }


def _label(text: str = "A supported", *, implicit_true: bool = False) -> dict[str, Any]:
    return {
        "start": 0,
        "end": len(text),
        "text": text,
        "label_type": "Evident Baseless Info",
        "due_to_null": False,
        "implicit_true": implicit_true,
        "meta": "private annotator note",
    }


def _response(
    row_id: str,
    source_id: str,
    *,
    split: str = "train",
    quality: str = "good",
    labels: list[dict[str, Any]] | None = None,
    text: str = "A supported response.",
) -> dict[str, Any]:
    return {
        "id": row_id,
        "source_id": source_id,
        "model": "private-generator",
        "temperature": 0.7,
        "labels": labels or [],
        "split": split,
        "quality": quality,
        "response": text,
    }


def _fixture_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = [
        _source("summary", "Summary", "Three packages arrived in 2020."),
        _source(
            "qa",
            "QA",
            {"question": "How many?", "passages": "The source says three packages."},
        ),
        _source("data", "Data2txt", {"name": "Example", "count": 3}),
    ]
    responses = [
        _response("r0", "summary"),
        _response("r1", "summary", labels=[_label()]),
        _response("r2", "qa", labels=[_label(implicit_true=True)]),
        _response("r3", "qa", quality="incorrect_refusal"),
        _response("r4", "data", split="test"),
        _response("r5", "data", split="test", labels=[_label()]),
    ]
    return sources, responses


def _validation_receipts() -> list[dict[str, Any]]:
    names = [
        *protocol.validation_scope.REQUIRED_CHECK_NAMES,
        "declared_entrypoint_cold_replay",
        "independent_cold_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    return [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "duration_s": 0.01,
        }
        for name in names
    ]


def test_req_auto_7423_spec_owns_protocol() -> None:
    """REQ-AUTO-7423 declares all human-label scenarios before implementation."""

    text = (ROOT / protocol.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-AUTO-7423") :]
    for suffix in ("01", "02", "03", "04", "05"):
        assert f"SCENARIO-AUTO-7423-{suffix}" in section
    for field in protocol.REQUIRED_ARTIFACT_FIELDS:
        assert field in section


def test_scenario_7423_01_join_preserves_human_source_support_semantics() -> None:
    """SCENARIO-AUTO-7423-01 keeps spans private and defines both frozen labels."""

    sources, responses = _fixture_rows()
    joined, dispositions = protocol.join_release(sources, responses)
    by_id = {row["response_id"]: row for row in joined}
    assert by_id["r0"]["primary_label"] == 1
    assert by_id["r1"]["primary_label"] == 0
    assert by_id["r1"]["implicit_true_excluded_label"] == 0
    assert by_id["r2"]["primary_label"] == 0
    assert by_id["r2"]["implicit_true_excluded_label"] == 1
    assert by_id["r2"]["annotations"][0]["implicit_true"] is True
    assert by_id["r2"]["annotations"][0]["label_type"] == "Evident Baseless Info"
    assert {row["reason"] for row in dispositions} == {"excluded_incorrect_refusal"}
    assert "r3" not in by_id

    broken = deepcopy(responses)
    broken[1]["labels"][0]["end"] = 500
    with pytest.raises(protocol.CorpusInvalid, match="annotation_span_invalid"):
        protocol.join_release(sources, broken)


def test_scenario_7423_01_rejects_missing_sources_and_unknown_release_values() -> None:
    """SCENARIO-AUTO-7423-01 fails closed on malformed joins and release enums."""

    source = _source("s")
    with pytest.raises(protocol.CorpusInvalid, match="source_id_missing"):
        protocol.join_release([source], [_response("r", "missing")])
    with pytest.raises(protocol.CorpusInvalid, match="task_type_invalid"):
        protocol.join_release([_source("s", "Other")], [_response("r", "s")])
    with pytest.raises(protocol.CorpusInvalid, match="split_invalid"):
        protocol.join_release([source], [_response("r", "s", split="dev")])
    with pytest.raises(protocol.CorpusInvalid, match="quality_invalid"):
        protocol.join_release([source], [_response("r", "s", quality="unknown")])


def test_scenario_7423_02_grouping_and_partitioning_are_label_blind() -> None:
    """SCENARIO-AUTO-7423-02 groups source bytes and keeps test overlaps final-only."""

    sources = [
        _source("a", value=" Same  source\nbytes "),
        _source("b", value="same source bytes"),
        _source("c", value="independent"),
        _source("d", value="final only"),
    ]
    responses = [
        _response("a0", "a"),
        _response("a1", "a", labels=[_label()]),
        _response("b0", "b", split="test"),
        _response("c0", "c"),
        _response("d0", "d", split="test"),
    ]
    joined, _ = protocol.join_release(sources, responses)
    assigned = protocol.assign_groups_and_partitions(joined)
    by_id = {row["response_id"]: row for row in assigned}
    assert by_id["a0"]["group_id"] == by_id["a1"]["group_id"] == by_id["b0"]["group_id"]
    assert by_id["a0"]["partition"] == "excluded_test_overlap"
    assert by_id["b0"]["partition"] == "final_test"
    assert by_id["d0"]["partition"] == "final_test"
    assert by_id["c0"]["partition"] in protocol.DEVELOPMENT_PARTITIONS

    relabeled = deepcopy(joined)
    for row in relabeled:
        row["primary_label"] = 1 - row["primary_label"]
        row["implicit_true_excluded_label"] = 1 - row["implicit_true_excluded_label"]
    reassigned = protocol.assign_groups_and_partitions(relabeled)
    assert [(row["response_id"], row["group_id"], row["partition"]) for row in assigned] == [
        (row["response_id"], row["group_id"], row["partition"]) for row in reassigned
    ]


def test_scenario_7423_02_caps_and_representatives_ignore_labels() -> None:
    """SCENARIO-AUTO-7423-02 applies hash ranks to groups and sibling responses."""

    rows = []
    for index in range(8):
        for sibling in range(2):
            rows.append(
                {
                    "response_id": f"r-{index}-{sibling}",
                    "group_id": f"g-{index}",
                    "partition": "fit",
                    "primary_label": sibling,
                }
            )
    selected = protocol.apply_group_caps(rows, caps={"fit": 3})
    assert len({row["group_id"] for row in selected}) == 3
    assert all(
        sum(row["group_id"] == other["group_id"] for other in selected) == 2 for row in selected
    )
    representatives = protocol.label_blind_representatives(selected)
    assert len(representatives) == 3
    relabeled = [{**row, "primary_label": 1 - row["primary_label"]} for row in selected]
    assert [row["response_id"] for row in representatives] == [
        row["response_id"] for row in protocol.label_blind_representatives(relabeled)
    ]


def test_scenario_7423_03_serialization_caps_and_feature_values_match_exp7412() -> None:
    """SCENARIO-AUTO-7423-03 reuses all six bounded feature definitions exactly."""

    qa = {"question": "How many?", "passages": "Three packages arrived."}
    assert protocol.serialize_source("QA", qa).startswith('{"passages"')
    assert protocol.serialize_source("Data2txt", {"z": 1, "a": 2}) == '{"a":2,"z":1}'
    assert protocol.serialize_source("Summary", "Summary text") == "Summary text"
    with pytest.raises(protocol.CorpusInvalid, match="source_info_invalid"):
        protocol.serialize_source("QA", "not an object")

    source_text = " ".join(["source"] * 4_100 + ["7"])
    response_text = " ".join(["answer"] * 1_024 + ["9"])
    row = {
        "response_id": "r",
        "group_id": "g",
        "partition": "fit",
        "task_type": "Summary",
        "source_text": source_text,
        "response_text": response_text,
    }
    predictor = protocol.project_predictor(row)
    assert predictor["source_token_count"] == 4_096
    assert predictor["response_token_count"] == 1_024
    assert predictor["source_truncated"] is True
    assert predictor["response_truncated"] is True
    assert list(predictor["features"]) == list(source_features.SOURCE_FEATURE_NAMES)
    expected = source_features.extract_feature_row(
        {
            "row_key": predictor["row_key"],
            "group_id": "g",
            "partition": "fit",
            "question": "",
            "context": predictor["source_text"],
            "answer": predictor["response_text"],
            "sentence": predictor["response_text"],
        }
    )["source_features"]
    assert predictor["features"] == expected
    assert set(predictor) == set(protocol.PREDICTOR_FIELDS)


def test_scenario_7423_01_and_05_shards_reload_with_masked_readers(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7423-01/05 separate views and bind every shard byte."""

    sources, responses = _fixture_rows()
    manifest = protocol.seal_corpus(tmp_path, sources, responses)
    reloaded = protocol.reload_corpus(tmp_path)
    readers = protocol.ProtocolReaders(reloaded)
    predictors = readers.read_predictors("final_test")
    assert predictors
    assert all(set(row) == set(protocol.PREDICTOR_FIELDS) for row in predictors)
    denied = {"labels", "annotations", "quality", "model", "source_id", "split"}
    assert all(not denied.intersection(row) for row in predictors)
    with pytest.raises(PermissionError, match="evaluator access denied"):
        readers.read_evaluators("final_test", "wrong-token")
    evaluators = readers.read_evaluators("final_test", protocol.EVALUATOR_TOKEN)
    assert {row["primary_label"] for row in evaluators} == {0, 1}
    assert manifest["manifest_hash"] == reloaded["manifest_hash"]

    predictor_path = tmp_path / manifest["shards"][0]["path"]
    predictor_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(protocol.CorpusInvalid, match="shard_hash_mismatch"):
        protocol.reload_corpus(tmp_path)


def test_scenario_7423_04_support_is_domain_specific_and_readiness_independent() -> None:
    """SCENARIO-AUTO-7423-04 records exact shortfalls without invalidating a protocol."""

    rows = []
    for partition, n_rows in {
        "probability_calibration": 99,
        "policy_calibration": 100,
        "prospective_stream": 399,
    }.items():
        for index in range(n_rows):
            rows.append(
                {
                    "group_id": f"{partition}-{index}",
                    "partition": partition,
                    "task_type": "QA" if index % 2 else "Summary",
                    "primary_label": index % 2,
                    "certificate_selected": True,
                }
            )
    support = protocol.reduce_support(rows)
    assert support["probability_calibration"]["independent_groups"] == 99
    assert support["policy_calibration"]["support_ready"] is True
    assert support["prospective_stream"]["support_ready"] is False
    verdict = protocol.classify_support(support, contract_valid=True)
    assert verdict == ("complete_null_insufficient_human_label_support", "null", 1)


def test_req_auto_7423_protocol_plan_and_validation_scope_are_frozen(tmp_path: Path) -> None:
    """REQ-AUTO-7423 seals seeds, roles, schedules, and the exact affected command plan."""

    plan = protocol.protocol_plan()
    assert plan["training_seeds"] == [65_101, 65_102, 65_103, 65_104, 65_105]
    assert plan["training_budget_steps"] == 500
    assert plan["probability_calibration_role"] != plan["policy_calibration_role"]
    assert plan["accept_thresholds"] == [0.01, 0.025, 0.05]
    assert plan["reject_thresholds"] == [0.9, 0.95, 0.99]
    assert len(protocol.planned_rows()) == 160
    assert all(row["status"] == "unstarted" for row in protocol.planned_rows())

    commands = protocol.build_validation_plan(ROOT, tmp_path)
    assert [command.name for command in commands] == list(
        protocol.validation_scope.REQUIRED_CHECK_NAMES
    )
    assert protocol.validate_validation_plan(ROOT, commands) == []
    assert all("tests/python" not in command.argv for command in commands)


def test_scenario_7423_05_artifact_reduces_and_detects_mutations(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7423-05 recomputes readiness, checksums, and shard identity."""

    sources, responses = _fixture_rows()
    artifact = protocol.build_artifact_for_test(
        tmp_path,
        sources,
        responses,
        validation_receipts=_validation_receipts(),
    )
    assert protocol.validate_artifact(artifact) == []
    assert protocol.independent_reduce_artifact(artifact) == []
    assert artifact["annotated_protocol_ready_score"] == 1
    assert artifact["promotion_score"] == 0
    assert artifact["label_authority"] == "human_annotation_source_support"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert not any(artifact["invocation_counts"].values())

    changed = deepcopy(artifact)
    changed["promotion_score"] = 1
    assert "declaration_mismatch:promotion_score" in protocol.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:changed"
    assert "reproducibility_checksum_mismatch" in protocol.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["support_counts"]["prospective_stream"]["independent_groups"] += 1
    changed["reproducibility_checksum"] = protocol.artifact_checksum(changed)
    assert "support_counts_mismatch" in protocol.independent_reduce_artifact(changed)


def test_req_auto_7423_blocked_artifact_and_cli_fail_closed(tmp_path: Path) -> None:
    """REQ-AUTO-7423 preserves unavailable science as a complete blocked record."""

    blocked = protocol.build_blocked_artifact(
        check="ragtruth_assets_available",
        upstream=protocol.RAGTRUTH_REPO,
        path="dataset/response.jsonl",
        field="sha256",
        expected="pinned bytes",
        observed=None,
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["annotated_protocol_ready_score"] == 0
    assert blocked["gate_check_summary"]["blocked_observed"] is None
    assert protocol.validate_artifact(blocked) == []
    assert protocol.independent_reduce_artifact(blocked) == []

    assert protocol.parse_args(["--date", protocol.RUN_DATE]).date == protocol.RUN_DATE
    with pytest.raises(SystemExit, match="--date must be"):
        protocol.run_experiment(ROOT, "20200101", output_path=tmp_path / "no.json")


def test_scenario_7423_05_cold_replay_reads_json_from_fresh_bytes(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7423-05 rejects malformed and valid cold artifact files."""

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    assert protocol.cold_replay(malformed) == ["artifact_not_object"]

    sources, responses = _fixture_rows()
    artifact = protocol.build_artifact_for_test(
        tmp_path / "raw",
        sources,
        responses,
        validation_receipts=_validation_receipts(),
    )
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert protocol.cold_replay(candidate) == []

    missing = tmp_path / "missing.json"
    assert protocol.cold_replay(missing)[0].startswith("artifact_unreadable:")


def test_req_auto_7423_source_receipts_authenticate_exact_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-AUTO-7423 binds both immutable URLs, sizes, hashes, and attribution."""

    payloads = {
        "dataset/source_info.jsonl": b'{"source_id":"1"}\n',
        "dataset/response.jsonl": b'{"id":"1"}\n',
    }
    paths = {}
    for relative, payload in payloads.items():
        path = protocol._asset_path(tmp_path, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        paths[relative] = path
    monkeypatch.setattr(
        protocol,
        "EXPECTED_ASSETS",
        {
            relative: {
                "size_bytes": len(payload),
                "sha256": protocol._sha256_bytes(payload),
            }
            for relative, payload in payloads.items()
        },
    )
    receipt = protocol._receipt_for_paths(tmp_path, paths)
    assert receipt["commit"] == protocol.RAGTRUTH_COMMIT
    assert receipt["license"] == "MIT"
    assert all(protocol.RAGTRUTH_COMMIT in row["url"] for row in receipt["files"])
    with pytest.raises(ValueError, match="asset_not_allowed"):
        protocol.asset_url("dataset/other.jsonl")

    missing = dict(paths)
    missing[protocol.ASSET_PATHS[0]] = tmp_path / "missing"
    with pytest.raises(protocol.SourceBlocked, match="cache_asset_missing"):
        protocol._receipt_for_paths(tmp_path, missing)
    changed = dict(paths)
    changed_path = tmp_path / "changed.jsonl"
    changed_path.write_bytes(b"x")
    changed[protocol.ASSET_PATHS[0]] = changed_path
    with pytest.raises(protocol.SourceBlocked, match="cache_size_mismatch"):
        protocol._receipt_for_paths(tmp_path, changed)
    monkeypatch.setitem(
        protocol.EXPECTED_ASSETS,
        protocol.ASSET_PATHS[0],
        {"size_bytes": 1, "sha256": "sha256:not-the-byte-hash"},
    )
    with pytest.raises(protocol.SourceBlocked, match="cache_hash_mismatch"):
        protocol._receipt_for_paths(tmp_path, changed)


def test_scenario_7423_01_malformed_identity_and_annotation_rows_fail_closed() -> None:
    """SCENARIO-AUTO-7423-01 rejects duplicate IDs and malformed span containers."""

    source = _source("s")
    with pytest.raises(protocol.CorpusInvalid, match="source_identity_invalid"):
        protocol.join_release([source, deepcopy(source)], [])
    response = _response("r", "s")
    with pytest.raises(protocol.CorpusInvalid, match="response_identity_invalid"):
        protocol.join_release([source], [response, deepcopy(response)])
    broken = _response("shape", "s")
    broken["response"] = None
    with pytest.raises(protocol.CorpusInvalid, match="response_shape_invalid"):
        protocol.join_release([source], [broken])
    broken = _response("annotation", "s", labels=[_label()])
    broken["labels"] = [{}]
    with pytest.raises(protocol.CorpusInvalid, match="annotation_shape_invalid"):
        protocol.join_release([source], [broken])
    with pytest.raises(protocol.CorpusInvalid, match="source_info_invalid:Summary"):
        protocol.serialize_source("Summary", {})


def test_scenario_7423_02_all_hash_ranges_and_exclusion_reasons_are_reachable() -> None:
    """SCENARIO-AUTO-7423-02 fixes all ranges and records overlap plus cap reasons."""

    found: dict[str, str] = {}
    index = 0
    while set(found) != set(protocol.DEVELOPMENT_PARTITIONS):
        candidate = f"candidate-{index}"
        found.setdefault(protocol._development_partition(candidate), candidate)
        index += 1
    assert set(found) == set(protocol.DEVELOPMENT_PARTITIONS)

    assigned = [
        {
            "response_id": "overlap",
            "source_id": "s1",
            "official_split": "train",
            "group_id": "g1",
            "partition": "excluded_test_overlap",
        },
        {
            "response_id": "capped",
            "source_id": "s2",
            "official_split": "train",
            "group_id": "g2",
            "partition": "fit",
        },
        {
            "response_id": "kept",
            "source_id": "s3",
            "official_split": "train",
            "group_id": "g3",
            "partition": "fit",
        },
    ]
    rows = protocol._disposition_rows(assigned, [assigned[2]], [])
    assert {row["reason"] for row in rows} == {
        "excluded_cross_split_duplicate_component",
        "excluded_partition_group_cap",
    }


def _update_manifest_shard(raw_dir: Path, shard_index: int, payload: bytes) -> None:
    manifest_path = raw_dir / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard = manifest["shards"][shard_index]
    (raw_dir / shard["path"]).write_bytes(payload)
    shard["sha256"] = protocol._sha256_bytes(payload)
    manifest["manifest_hash"] = protocol._manifest_hash(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _mutated_raw_copy(base: Path, target: Path) -> tuple[Path, dict[str, Any]]:
    shutil.copytree(base, target)
    manifest_path = target / "corpus_manifest.json"
    return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))


def test_scenario_7423_05_shard_split_and_reader_failures_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-AUTO-7423-05 rejects oversized rows and malformed local JSONL."""

    monkeypatch.setattr(protocol, "MAX_SHARD_BYTES", 30)
    records = [{"value": "a" * 10}, {"value": "b" * 10}]
    shards = protocol._write_jsonl_shards(tmp_path, "split", "fixture", records)
    assert len(shards) == 2
    with pytest.raises(protocol.CorpusInvalid, match="single_record_exceeds_shard_limit"):
        protocol._write_jsonl_shards(tmp_path, "large", "fixture", [{"value": "x" * 40}])

    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(protocol.CorpusInvalid, match="shard_unreadable"):
        protocol._load_jsonl(bad_json)
    not_object = tmp_path / "list.jsonl"
    not_object.write_text("[]\n", encoding="utf-8")
    with pytest.raises(protocol.CorpusInvalid, match="shard_row_not_object"):
        protocol._load_jsonl(not_object)


def test_scenario_7423_05_reload_rejects_each_structural_mutation(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7423-05 checks manifest, counts, fields, keys, and group isolation."""

    sources, responses = _fixture_rows()
    base = tmp_path / "base"
    protocol.seal_corpus(base, sources, responses)
    with pytest.raises(protocol.CorpusInvalid, match="manifest_unreadable"):
        protocol.reload_corpus(tmp_path / "missing")

    manifest_path, manifest = _mutated_raw_copy(base, tmp_path / "manifest-hash")
    manifest["license"] = "changed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(protocol.CorpusInvalid, match="manifest_hash_mismatch"):
        protocol.reload_corpus(manifest_path.parent)

    cases = (
        ("predictor-count", "predictor_row_count", "predictor_row_count_mismatch"),
        ("evaluator-count", "evaluator_row_count", "evaluator_row_count_mismatch"),
    )
    for name, field, error in cases:
        case_path, case_manifest = _mutated_raw_copy(base, tmp_path / name)
        case_manifest[field] += 1
        case_manifest["manifest_hash"] = protocol._manifest_hash(case_manifest)
        case_path.write_text(json.dumps(case_manifest), encoding="utf-8")
        with pytest.raises(protocol.CorpusInvalid, match=error):
            protocol.reload_corpus(case_path.parent)

    row_count_dir = tmp_path / "row-count"
    _, row_count_manifest = _mutated_raw_copy(base, row_count_dir)
    shard = row_count_manifest["shards"][0]
    payload = (row_count_dir / shard["path"]).read_bytes()
    _update_manifest_shard(row_count_dir, 0, payload + payload.splitlines(keepends=True)[0])
    with pytest.raises(protocol.CorpusInvalid, match="shard_row_count_mismatch"):
        protocol.reload_corpus(row_count_dir)

    for kind, field_error in (
        ("predictor", "predictor_fields_mismatch"),
        ("evaluator", "evaluator_fields_mismatch"),
    ):
        target = tmp_path / f"fields-{kind}"
        _, target_manifest = _mutated_raw_copy(base, target)
        shard_index = next(
            index for index, shard in enumerate(target_manifest["shards"]) if shard["kind"] == kind
        )
        shard_path = target / target_manifest["shards"][shard_index]["path"]
        lines = shard_path.read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[0])
        row.pop(next(iter(row)))
        lines[0] = json.dumps(row)
        _update_manifest_shard(target, shard_index, ("\n".join(lines) + "\n").encode())
        with pytest.raises(protocol.CorpusInvalid, match=field_error):
            protocol.reload_corpus(target)

    key_dir = tmp_path / "keys"
    _, key_manifest = _mutated_raw_copy(base, key_dir)
    evaluator_index = next(
        index for index, shard in enumerate(key_manifest["shards"]) if shard["kind"] == "evaluator"
    )
    evaluator_path = key_dir / key_manifest["shards"][evaluator_index]["path"]
    lines = evaluator_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["row_key"] = "changed-row-key"
    lines[0] = json.dumps(row)
    _update_manifest_shard(key_dir, evaluator_index, ("\n".join(lines) + "\n").encode())
    with pytest.raises(protocol.CorpusInvalid, match="view_row_keys_mismatch"):
        protocol.reload_corpus(key_dir)

    leak_dir = tmp_path / "group-leak"
    _, leak_manifest = _mutated_raw_copy(base, leak_dir)
    predictor_index = next(
        index
        for index, shard in enumerate(leak_manifest["shards"])
        if shard["kind"] == "predictor" and shard["rows"] >= 2
    )
    predictor_path = leak_dir / leak_manifest["shards"][predictor_index]["path"]
    lines = predictor_path.read_text(encoding="utf-8").splitlines()
    first, second = json.loads(lines[0]), json.loads(lines[1])
    second["group_id"] = first["group_id"]
    second["partition"] = "final_test" if first["partition"] != "final_test" else "fit"
    lines[1] = json.dumps(second)
    _update_manifest_shard(leak_dir, predictor_index, ("\n".join(lines) + "\n").encode())
    with pytest.raises(protocol.CorpusInvalid, match="group_partition_leak"):
        protocol.reload_corpus(leak_dir)

    readers = protocol.ProtocolReaders(protocol.reload_corpus(base))
    with pytest.raises(ValueError, match="partition_invalid"):
        readers.read_predictors("unknown")
    with pytest.raises(ValueError, match="partition_invalid"):
        readers.read_evaluators("unknown", protocol.EVALUATOR_TOKEN)


def test_scenario_7423_04_and_05_closed_classification_and_artifact_guards(
    tmp_path: Path,
) -> None:
    """SCENARIO-AUTO-7423-04/05 cover ready, disqualified, and mutated terminal states."""

    full_support = {
        name: {"support_ready": True}
        for name in ("probability_calibration", "policy_calibration", "prospective_stream")
    }
    assert protocol.classify_support(full_support, contract_valid=True) == (
        "complete_null_protocol_only_no_benefit_test",
        "null",
        1,
    )
    assert protocol.classify_support(full_support, contract_valid=False) == (
        "complete_disqualified_protocol_contract",
        "disqualified",
        0,
    )

    sources, responses = _fixture_rows()
    artifact = protocol.build_artifact_for_test(
        tmp_path / "raw",
        sources,
        responses,
        validation_receipts=_validation_receipts(),
    )
    mutations = [
        ("verdict_class", "unknown", "verdict_class_invalid"),
        ("field_principles", {}, "field_principles_mismatch"),
        ("honest_verdict", "bad", "terminal_verdict_prefix_invalid"),
        ("rows", [], "planned_rows_mismatch"),
        ("annotated_protocol_ready_score", 0, "annotated_protocol_ready_score_mismatch"),
    ]
    for field, value, expected_error in mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = protocol.artifact_checksum(changed)
        assert expected_error in protocol.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["corpus_manifest"] = {}
    changed["reproducibility_checksum"] = protocol.artifact_checksum(changed)
    assert "corpus_manifest_invalid" in protocol.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["corpus_manifest"]["attribution"] = "changed embedded manifest"
    changed["corpus_manifest"]["manifest_hash"] = protocol._manifest_hash(
        changed["corpus_manifest"]
    )
    changed["reproducibility_checksum"] = protocol.artifact_checksum(changed)
    assert "corpus_manifest_reload_mismatch" in protocol.validate_artifact(changed)

    blocked = protocol.build_blocked_artifact(
        check="missing",
        upstream="source",
        path="path",
        field="field",
        expected=1,
        observed=0,
    )
    blocked["annotated_protocol_ready_score"] = 1
    blocked["honest_verdict"] = "wrong"
    blocked["reproducibility_checksum"] = protocol.artifact_checksum(blocked)
    errors = protocol.validate_artifact(blocked)
    assert "blocked_readiness_invalid" in errors
    assert "blocked_verdict_invalid" in errors

    for field, value, expected_error in (
        ("status", "complete_changed", "terminal_verdict_mismatch"),
        ("verdict_class", "positive", "terminal_class_mismatch"),
        ("annotated_protocol_ready_score", 0, "terminal_readiness_mismatch"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = protocol.artifact_checksum(changed)
        assert expected_error in protocol.independent_reduce_artifact(changed)

    broken_artifact = deepcopy(artifact)
    shard_path = (
        Path(broken_artifact["corpus_manifest_path"]).parent
        / broken_artifact["corpus_manifest"]["shards"][0]["path"]
    )
    shard_path.write_text("{}\n", encoding="utf-8")
    assert any(
        error.startswith("shard_hash_mismatch")
        for error in protocol.independent_reduce_artifact(broken_artifact)
    )
