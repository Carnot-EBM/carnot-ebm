"""Tests for the pinned EnokiQA source-context corpus.

Spec refs: REQ-AUTO-7410 and SCENARIO-AUTO-7410-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7410_v650_source_corpus as corpus


ROOT = Path(__file__).resolve().parents[2]


def _sentence(index: int, text: str, probability: float = 0.8) -> dict[str, Any]:
    start = 0
    return {
        "sentence_index": index,
        "sentence": text,
        "hall_prob": probability,
        "triples": [
            {
                "triplet": ["subject", "is", text[:4]],
                "span": [start, min(len(text), 4)],
                "hypothesis": f"private-{text}",
                "entailment": 0.2,
                "neutral": probability,
                "contradiction": 0.0,
                "hall_prob": probability,
            }
        ],
    }


def _row(
    row_id: str,
    *,
    split: str = "dev",
    title: str | None = None,
    question: str | None = None,
    answer: str | None = None,
    context: str | None = None,
) -> dict[str, Any]:
    text = answer or f"answer {row_id}"
    return {
        "id": row_id,
        "title": title or f"title {row_id}",
        "question": question or f"question {row_id}",
        "answer": text,
        "context": context or f"context {row_id}",
        "n_sentences": 1,
        "n_triplets": 1,
        "sentences": [_sentence(0, text)],
        "_official_split": split,
        "_source_index": 0,
    }


def _assets(tmp_path: Path) -> dict[str, Any]:
    files = []
    for relative, content in {
        "README.md": b"---\nlicense: cc-by-sa-4.0\n---\n",
        corpus.DEV_PARQUET: b"dev",
        corpus.TEST_PARQUET: b"test",
    }.items():
        path = tmp_path / corpus.ENOKIQA_REVISION / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files.append(
            {
                "path": relative,
                "cache_path": str(path),
                "url": corpus.asset_url(relative),
                "size_bytes": len(content),
                "sha256": corpus.sha256_file(path),
            }
        )
    return {
        "revision": corpus.ENOKIQA_REVISION,
        "repository": corpus.ENOKIQA_REPO,
        "files": files,
        "total_bytes": sum(row["size_bytes"] for row in files),
        "license": corpus.LICENSE,
        "attribution": corpus.ATTRIBUTION,
    }


def test_req_auto_7410_spec_owns_contract() -> None:
    """REQ-AUTO-7410 declares each corpus scenario before implementation."""

    text = (ROOT / corpus.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-AUTO-7410") :]
    for suffix in ("01", "02", "03", "04", "05"):
        assert f"SCENARIO-AUTO-7410-{suffix}" in section
    for field in corpus.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section or field in {
            "schema",
            "experiment_id",
            "milestone",
            "status",
        }


def test_scenario_7410_01_fetch_is_pinned_bounded_and_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-AUTO-7410-01 fetches only pinned assets and checks total bytes."""

    payloads = {
        "README.md": b"---\nlicense: cc-by-sa-4.0\n---\n",
        corpus.DEV_PARQUET: b"dev-bytes",
        corpus.TEST_PARQUET: b"test-bytes",
    }
    monkeypatch.setattr(
        corpus,
        "EXPECTED_ASSETS",
        {
            path: {"size_bytes": len(value), "sha256": corpus.sha256_bytes(value)}
            for path, value in payloads.items()
        },
    )
    calls: list[tuple[str, float, int]] = []

    def fetch(url: str, timeout_s: float, byte_limit: int) -> bytes:
        relative = next(path for path in payloads if url.endswith(path))
        calls.append((url, timeout_s, byte_limit))
        return payloads[relative]

    receipt = corpus.fetch_assets(tmp_path, fetcher=fetch, timeout_s=12.0, byte_limit=100)
    assert receipt["revision"] == corpus.ENOKIQA_REVISION
    assert [row["path"] for row in receipt["files"]] == list(corpus.ASSET_PATHS)
    assert len(calls) == 3
    assert all(corpus.ENOKIQA_REVISION in url for url, _, _ in calls)
    assert all(0.0 < timeout <= 12.0 for _, timeout, _ in calls)
    assert corpus.fetch_assets(tmp_path, fetcher=lambda *_: pytest.fail("cache missed")) == receipt

    with pytest.raises(corpus.SourceBlocked, match="transfer_limit_exceeded"):
        corpus.fetch_assets(tmp_path / "too-large", fetcher=fetch, byte_limit=4)


def test_scenario_7410_01_fetch_and_decode_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7410-01 reports transport, revision, and decoder failures."""

    with pytest.raises(corpus.SourceBlocked, match="network_failure"):
        corpus.fetch_assets(
            tmp_path / "network",
            fetcher=lambda *_: (_ for _ in ()).throw(OSError("offline")),
        )
    drift = _assets(tmp_path / "drift")
    drift["revision"] = "moving-head"
    with pytest.raises(corpus.SourceBlocked, match="cache_revision_mismatch"):
        corpus.authenticate_assets(tmp_path / "drift", drift)
    with pytest.raises(corpus.SourceBlocked, match="decoder_unavailable"):
        corpus.decode_rows(
            _assets(tmp_path / "decode"),
            decoder=lambda _: (_ for _ in ()).throw(ImportError("no parquet")),
        )


def test_scenario_7410_02_connected_groups_and_test_overlap_are_label_blind() -> None:
    """SCENARIO-AUTO-7410-02 joins all keys and excludes dev-to-test overlap."""

    rows = [
        _row("d-title", title=" Shared  title "),
        _row("d-context", context="common context"),
        _row("d-question", question="SAME question?"),
        _row("d-answer", answer="duplicate answer"),
        _row("d-chain", title="shared title", context="common context"),
        _row(
            "d-chain-2",
            title="shared title",
            question="same question?",
            answer="duplicate answer",
        ),
        _row("t-overlap", split="test", answer="duplicate answer"),
        _row("d-safe"),
        _row("t-safe", split="test"),
    ]
    changed = deepcopy(rows)
    for row in changed:
        row["sentences"][0]["hall_prob"] = 1.0 - row["sentences"][0]["hall_prob"]
    grouped = corpus.build_connected_groups(rows)
    regrouped = corpus.build_connected_groups(changed)
    assert grouped == regrouped
    dispositions = corpus.assign_source_dispositions(rows, grouped)
    by_id = {row["source_id"]: row for row in dispositions}
    assert by_id["d-title"]["disposition"] == "excluded_test_overlap"
    assert by_id["d-safe"]["disposition"] == "eligible_dev"
    assert by_id["t-overlap"]["disposition"] == "official_test"
    assert by_id["t-safe"]["disposition"] == "official_test"
    assert "excluded_test_overlap" in {
        row["partition"] for row in corpus.assign_partitions(dispositions)
    }


def test_scenario_7410_03_selection_and_predictor_view_deny_teacher_fields() -> None:
    """SCENARIO-AUTO-7410-03 selects without labels and exposes four text fields."""

    row = _row("answer-id", answer="first sentence. second sentence.")
    row["sentences"] = [
        _sentence(0, "first sentence.", 0.1),
        _sentence(1, "second sentence.", 0.9),
    ]
    selected = corpus.select_sentence(row)
    mutated = deepcopy(row)
    for sentence in mutated["sentences"]:
        sentence["hall_prob"] = 1.0 - sentence["hall_prob"]
        sentence["triples"][0]["hypothesis"] = "changed teacher value"
    assert corpus.select_sentence(mutated)["sentence_index"] == selected["sentence_index"]

    projected = corpus.project_row(row, "train", "group-1")
    assert set(projected["predictor_view"]) == {"question", "context", "answer", "sentence"}
    denied = {
        "triples",
        "hypothesis",
        "entailment",
        "neutral",
        "contradiction",
        "hall_prob",
        "span",
        "model",
        "source_id",
        "label",
    }
    assert denied.isdisjoint(projected["predictor_view"])
    assert projected["evaluator_view"]["label"] in {0, 1}


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda sentence: sentence.update({"triples": []}), "empty_annotation"),
        (lambda sentence: sentence.update({"sentence": "not in answer"}), "sentence_unmatched"),
        (lambda sentence: sentence["triples"][0].update({"span": [4, 2]}), "malformed_span"),
    ],
)
def test_scenario_7410_04_unsupported_annotations_remain_unscored(
    mutation: Any, reason: str
) -> None:
    """SCENARIO-AUTO-7410-04 retains unsupported answers without a false label."""

    row = _row(f"bad-{reason}")
    mutation(row["sentences"][0])
    projected = corpus.project_row(row, "train", "group-bad")
    assert projected["evaluator_view"]["label"] is None
    assert projected["disposition_reason"] == reason
    assert projected["predictor_view"]["sentence"] is None


def test_scenario_7410_05_partition_and_reload_are_deterministic(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7410-05 seals memberships, shards, masks, and hashes."""

    rows = [_row(f"dev-{index}") for index in range(400)]
    rows += [_row(f"test-{index}", split="test") for index in range(25)]
    groups = corpus.build_connected_groups(rows)
    dispositions = corpus.assign_source_dispositions(rows, groups)
    memberships = corpus.assign_partitions(dispositions)
    assert memberships == corpus.assign_partitions(dispositions)
    assert {row["partition"] for row in memberships} == {
        "train",
        "probability_calibration",
        "policy_calibration",
        "online_stream",
        "final_test",
    }

    manifest = corpus.write_corpus(tmp_path / "raw", rows, groups, memberships, shard_size=37)
    reloaded = corpus.reload_corpus(tmp_path / "raw", manifest)
    assert reloaded["manifest_hash"] == manifest["manifest_hash"]
    assert reloaded["predictor_row_count"] == len(rows)
    assert reloaded["evaluator_row_count"] == len(rows)
    assert all(shard["label_authority"] == "machine_annotation" for shard in manifest["shards"])
    predictor = reloaded["predictor_rows"][0]
    assert "label" not in predictor
    readers = corpus.CorpusReaders(reloaded)
    assert readers.read_predictors("final_test")
    with pytest.raises(PermissionError, match="sealed_final_test_labels"):
        readers.read_labels("final_test")
    assert readers.read_labels("final_test", token=corpus.FINAL_TEST_TOKEN)

    shard_path = tmp_path / "raw" / manifest["shards"][0]["path"]
    shard_path.write_text(shard_path.read_text() + "{}\n", encoding="utf-8")
    with pytest.raises(corpus.CorpusInvalid, match="shard_hash_mismatch"):
        corpus.reload_corpus(tmp_path / "raw", manifest)


def test_artifact_contract_rejects_mutations(tmp_path: Path) -> None:
    """REQ-AUTO-7410 rejects declaration, overlap, checksum, and readiness drift."""

    rows = [_row(f"dev-{index}") for index in range(400)]
    rows += [_row(f"test-{index}", split="test") for index in range(25)]
    groups = corpus.build_connected_groups(rows)
    dispositions = corpus.assign_source_dispositions(rows, groups)
    memberships = corpus.assign_partitions(dispositions)
    manifest = corpus.write_corpus(tmp_path / "raw", rows, groups, memberships)
    artifact = corpus.build_artifact_for_test(
        asset_receipt=_assets(tmp_path / "assets"),
        rows=rows,
        groups=groups,
        memberships=memberships,
        corpus_manifest=manifest,
    )
    assert corpus.validate_artifact(artifact) == []
    for field, value in (
        ("MODEL_SPECS", ["forbidden"]),
        ("execution_venue", "host_cpu"),
        ("label_authority", "exact_truth"),
        ("promotion_score", 1),
        ("source_corpus_ready_score", 0),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        assert corpus.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:changed"
    assert "reproducibility_checksum_mismatch" in corpus.validate_artifact(changed)


def test_blocked_artifact_names_exact_external_failure() -> None:
    """SCENARIO-AUTO-7410-01 keeps unavailable source failures terminal and exact."""

    artifact = corpus.build_blocked_artifact(
        check="enoki_revision_available",
        upstream="s-nlp/EnokiQA",
        field="revision",
        expected=corpus.ENOKIQA_REVISION,
        observed=None,
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["source_corpus_ready_score"] == 0
    assert artifact["gate_check_summary"]["blocked_revision"] == {
        "upstream": "s-nlp/EnokiQA",
        "path": None,
        "check": "enoki_revision_available",
        "field": "revision",
        "expected": corpus.ENOKIQA_REVISION,
        "observed": None,
    }
    assert corpus.validate_artifact(artifact) == []


def test_manifest_json_is_compact_and_has_no_teacher_fields(tmp_path: Path) -> None:
    """REQ-AUTO-7410 keeps teacher values in evaluator shards, not the manifest."""

    rows = [_row("one"), _row("two", split="test")]
    groups = corpus.build_connected_groups(rows)
    dispositions = corpus.assign_source_dispositions(rows, groups)
    memberships = corpus.assign_partitions(dispositions)
    manifest = corpus.write_corpus(tmp_path, rows, groups, memberships, shard_size=1)
    text = json.dumps(manifest, sort_keys=True)
    for teacher in ("hall_prob", "triplet", "hypothesis", "entailment", "contradiction"):
        assert teacher not in text
    assert len(text) < 20_000


def test_source_asset_failure_boundaries_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-AUTO-7410-01 names cache and transport identity failures."""

    with pytest.raises(ValueError, match="asset_not_allowed"):
        corpus.asset_url("weights.bin")
    with pytest.raises(corpus.SourceBlocked, match="cache_manifest_unavailable"):
        corpus.authenticate_assets(tmp_path / "absent")

    manifest = tmp_path / "bad-json" / corpus.ENOKIQA_REVISION / "asset_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(corpus.SourceBlocked, match="cache_manifest_invalid"):
        corpus.authenticate_assets(tmp_path / "bad-json")

    valid = _assets(tmp_path / "valid")
    changed = deepcopy(valid)
    changed["files"] = []
    with pytest.raises(corpus.SourceBlocked, match="cache_asset_paths_mismatch"):
        corpus.authenticate_assets(tmp_path / "valid", changed)
    missing = deepcopy(valid)
    Path(missing["files"][0]["cache_path"]).unlink()
    with pytest.raises(corpus.SourceBlocked, match="cache_asset_missing"):
        corpus.authenticate_assets(tmp_path / "valid", missing)

    corrupt = _assets(tmp_path / "corrupt")
    Path(corrupt["files"][0]["cache_path"]).write_bytes(b"changed")
    with pytest.raises(corpus.SourceBlocked, match="cache_asset_identity_mismatch"):
        corpus.authenticate_assets(tmp_path / "corrupt", corrupt)

    bounded = _assets(tmp_path / "bounded")
    monkeypatch.setattr(corpus, "MAX_TRANSFER_BYTES", 1)
    monkeypatch.setattr(
        corpus,
        "EXPECTED_ASSETS",
        {
            row["path"]: {"size_bytes": row["size_bytes"], "sha256": row["sha256"]}
            for row in bounded["files"]
        },
    )
    with pytest.raises(corpus.SourceBlocked, match="transfer_limit_exceeded"):
        corpus.authenticate_assets(tmp_path / "bounded", bounded)


def test_fetch_timeout_rethrow_and_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-AUTO-7410-01 keeps typed fetch failures distinct."""

    ticks = iter((0.0, 2.0))
    monkeypatch.setattr(corpus.time, "monotonic", lambda: next(ticks))
    with pytest.raises(corpus.SourceBlocked, match="fetch_timeout:total"):
        corpus.fetch_assets(tmp_path / "timeout", fetcher=lambda *_: b"", timeout_s=1.0)

    monkeypatch.undo()

    def typed(*_args: Any) -> bytes:
        raise corpus.SourceBlocked("typed_failure")

    with pytest.raises(corpus.SourceBlocked, match="typed_failure"):
        corpus.fetch_assets(tmp_path / "typed", fetcher=typed)
    with pytest.raises(corpus.SourceBlocked, match="source_identity_mismatch"):
        corpus.fetch_assets(tmp_path / "identity", fetcher=lambda *_: b"wrong")


def test_decoder_success_progress_and_published_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-AUTO-7410 decodes both official splits and enforces the row ceiling."""

    assets = _assets(tmp_path)
    decoded = corpus.decode_rows(assets, decoder=lambda _: [{"id": "row"} for _ in range(250)])
    assert len(decoded) == 500
    assert decoded[0]["_official_split"] == "dev"
    assert decoded[-1]["_official_split"] == "test"
    monkeypatch.setattr(corpus, "MAX_PUBLISHED_ROWS", 1)
    with pytest.raises(corpus.SourceBlocked, match="published_row_limit"):
        corpus.decode_rows(assets, decoder=lambda _: [{"id": "a"}, {"id": "b"}])


def test_group_partition_and_annotation_adverse_branches() -> None:
    """SCENARIO-AUTO-7410-02/04 fail closed on malformed local structures."""

    empty = {
        **_row("empty"),
        "title": "",
        "question": "",
        "answer": "",
        "context": "",
    }
    groups = corpus.build_connected_groups([empty])
    assert len(groups) == 1
    with pytest.raises(corpus.CorpusInvalid, match="group_partition_conflict"):
        corpus.assign_partitions(
            [
                {"row_index": 0, "group_id": "same", "disposition": "eligible_dev"},
                {"row_index": 1, "group_id": "same", "disposition": "official_test"},
            ]
        )
    assert corpus.select_sentence({"sentences": []}) is None
    assert corpus.select_sentence({"sentences": ["not-a-map"]}) is None
    no_sentences = _row("none")
    no_sentences["sentences"] = []
    assert (
        corpus.project_row(no_sentences, "train", "g")["disposition_reason"] == "empty_annotation"
    )

    no_text = _row("no-text")
    no_text["sentences"][0]["sentence"] = ""
    assert corpus.project_row(no_text, "train", "g")["disposition_reason"] == "empty_annotation"
    invalid_label = _row("label")
    invalid_label["sentences"][0]["hall_prob"] = float("nan")
    assert (
        corpus.project_row(invalid_label, "train", "g")["disposition_reason"]
        == "invalid_machine_label"
    )
    non_mapping = _row("non-map")
    non_mapping["sentences"][0]["triples"] = ["bad"]
    assert corpus.project_row(non_mapping, "train", "g")["disposition_reason"] == "malformed_span"


def _small_manifest(tmp_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [_row("dev"), _row("test", split="test")]
    groups = corpus.build_connected_groups(rows)
    dispositions = corpus.assign_source_dispositions(rows, groups)
    memberships = corpus.assign_partitions(dispositions)
    return corpus.write_corpus(tmp_path, rows, groups, memberships, shard_size=1), rows


def test_reload_rejects_manifest_and_shard_mutations(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7410-05 rejects each stored boundary mutation."""

    with pytest.raises(ValueError, match="shard_size_must_be_positive"):
        corpus.write_corpus(tmp_path / "zero", [], [], [], shard_size=0)
    with pytest.raises(corpus.CorpusInvalid, match="manifest_unreadable"):
        corpus.reload_corpus(tmp_path / "absent")
    bad_manifest = tmp_path / "list"
    bad_manifest.mkdir()
    (bad_manifest / "corpus_manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(corpus.CorpusInvalid, match="manifest_not_object"):
        corpus.reload_corpus(bad_manifest)

    manifest, _ = _small_manifest(tmp_path / "base")
    assert corpus.reload_corpus(tmp_path / "base")["predictor_row_count"] == 2
    changed = deepcopy(manifest)
    changed["manifest_hash"] = "sha256:changed"
    with pytest.raises(corpus.CorpusInvalid, match="manifest_hash_mismatch"):
        corpus.reload_corpus(tmp_path / "base", changed)

    cases = (
        ("unreadable", lambda payload: "{bad", "shard_unreadable"),
        (
            "metadata",
            lambda payload: json.dumps({**payload, "label_authority": "truth"}),
            "shard_metadata_invalid",
        ),
        (
            "count",
            lambda payload: json.dumps({**payload, "records": []}),
            "shard_row_count_mismatch",
        ),
    )
    for name, mutate, expected in cases:
        case_dir = tmp_path / name
        case_manifest, _ = _small_manifest(case_dir)
        shard = case_manifest["shards"][0]
        path = case_dir / shard["path"]
        payload = json.loads(path.read_text())
        path.write_text(mutate(payload), encoding="utf-8")
        shard["sha256"] = corpus.sha256_file(path)
        case_manifest["manifest_hash"] = corpus._manifest_hash(case_manifest)
        with pytest.raises(corpus.CorpusInvalid, match=expected):
            corpus.reload_corpus(case_dir, case_manifest)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda predictor, evaluator: evaluator[0].update({"row_key": "different"}),
            "view_identity_mismatch",
        ),
        (lambda predictor, evaluator: predictor[0].update({"label": 1}), "predictor_teacher_field"),
        (
            lambda predictor, evaluator: predictor[1].update(
                {"group_id": predictor[0]["group_id"], "partition": "final_test"}
            ),
            "group_partition_overlap",
        ),
    ],
)
def test_reload_rejects_cross_view_and_group_leaks(
    tmp_path: Path, mutation: Any, expected: str
) -> None:
    """SCENARIO-AUTO-7410-05 rejects cross-view identity and role leaks."""

    manifest, _ = _small_manifest(tmp_path)
    predictor_shards = [row for row in manifest["shards"] if row["kind"] == "predictor"]
    evaluator_shards = [row for row in manifest["shards"] if row["kind"] == "evaluator"]
    predictor_payloads = [
        json.loads((tmp_path / row["path"]).read_text()) for row in predictor_shards
    ]
    evaluator_payloads = [
        json.loads((tmp_path / row["path"]).read_text()) for row in evaluator_shards
    ]
    predictors = [payload["records"][0] for payload in predictor_payloads]
    evaluators = [payload["records"][0] for payload in evaluator_payloads]
    mutation(predictors, evaluators)
    for shard, payload in zip(predictor_shards, predictor_payloads, strict=True):
        path = tmp_path / shard["path"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        shard["sha256"] = corpus.sha256_file(path)
    for shard, payload in zip(evaluator_shards, evaluator_payloads, strict=True):
        path = tmp_path / shard["path"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        shard["sha256"] = corpus.sha256_file(path)
    manifest["manifest_hash"] = corpus._manifest_hash(manifest)
    with pytest.raises(corpus.CorpusInvalid, match=expected):
        corpus.reload_corpus(tmp_path, manifest)


def test_artifact_validator_and_local_preconditions_cover_fail_closed_paths(
    tmp_path: Path,
) -> None:
    """REQ-AUTO-7410 validates terminal fields and hashes local authorities."""

    manifest, rows = _small_manifest(tmp_path / "raw")
    groups = corpus.build_connected_groups(rows)
    dispositions = corpus.assign_source_dispositions(rows, groups)
    memberships = corpus.assign_partitions(dispositions)
    with pytest.raises(corpus.CorpusInvalid, match="ready_artifact_requires_rows"):
        corpus.build_artifact(
            asset_receipt=_assets(tmp_path / "assets"),
            rows=[],
            groups=[],
            memberships=[],
            corpus_manifest=manifest,
            validation_receipts=[],
            preconditions=[],
            source_hashes={},
            phase_spans=[],
            started_at_utc="start",
            completed_at_utc="end",
            started_ns=0,
            ended_ns=0,
        )
    artifact = corpus.build_artifact_for_test(
        asset_receipt=_assets(tmp_path / "assets-2"),
        rows=rows,
        groups=groups,
        memberships=memberships,
        corpus_manifest=manifest,
    )
    mutations = [
        ("verdict_class", "unknown", "verdict_class_invalid"),
        ("source_disposition_rows", [], "source_disposition_count_mismatch"),
        ("field_principles", {}, "field_principles_mismatch"),
    ]
    for field, value, expected in mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = corpus.artifact_checksum(changed)
        assert expected in corpus.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["split_manifest"]["group_memberships"].append(
        {
            "group_id": changed["split_manifest"]["group_memberships"][0]["group_id"],
            "partition": "other",
        }
    )
    changed["reproducibility_checksum"] = corpus.artifact_checksum(changed)
    assert "split_group_overlap" in corpus.validate_artifact(changed)
    blocked = corpus.build_blocked_artifact(
        check="x", upstream="u", field="f", expected=1, observed=0
    )
    blocked["source_corpus_ready_score"] = 1
    blocked["reproducibility_checksum"] = corpus.artifact_checksum(blocked)
    assert "blocked_disposition_invalid" in corpus.validate_artifact(blocked)

    checks, hashes = corpus.collect_preconditions(ROOT)
    assert all(row["passed"] for row in checks)
    assert corpus.SPEC_PATH.as_posix() in hashes
    assert "enokiqa:README.md" in corpus._source_hashes(ROOT, _assets(tmp_path / "hashes"))
    assert corpus._span("test", 2.0, 1.0, 3)["completed_units"] == 3


def test_parse_and_main_cold_replay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-AUTO-7410-05 exposes a fresh-process replay CLI."""

    args = corpus.parse_args(["--date", corpus.RUN_DATE, "--cold-replay", str(tmp_path)])
    assert args.cold_replay == tmp_path
    monkeypatch.setattr(corpus, "cold_replay", lambda _: [])
    assert corpus.main(["--date", corpus.RUN_DATE, "--cold-replay", str(tmp_path)]) == 0
    monkeypatch.setattr(corpus, "cold_replay", lambda _: ["failure"])
    assert corpus.main(["--date", corpus.RUN_DATE, "--cold-replay", str(tmp_path)]) == 1
    called: list[tuple[Path, str, Path, Path]] = []
    monkeypatch.setattr(
        corpus,
        "run_experiment",
        lambda root, date, output_path, cache_root: called.append(
            (root, date, output_path, cache_root)
        ),
    )
    assert corpus.main(["--date", corpus.RUN_DATE, "--output", str(tmp_path / "out")]) == 0
    assert called[0][1] == corpus.RUN_DATE
