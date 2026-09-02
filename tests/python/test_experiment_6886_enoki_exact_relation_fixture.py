"""Tests for pinned Enoki assets and exact anchored relations.

Spec refs: REQ-CONSTRAINT-6886 and SCENARIO-CONSTRAINT-6886-*.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import pytest

from carnot import experiment_6886_enoki_exact_relation_fixture as mod


ROOT = Path(__file__).resolve().parents[2]


def _graph_record(
    *, polarity: str = "positive"
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    vocabulary = mod.build_closed_vocabulary("graph_coloring", 0)
    entry = next(row for row in vocabulary if row["polarity"] == polarity)
    if polarity == "positive":
        source = "Café evidence: Node n0 has color red."
        evidence = "Node n0 has color red."
    else:
        source = "Café evidence: Node n0 does not have color red."
        evidence = "Node n0 does not have color red."
    record = mod.build_anchored_relation(
        record_id=f"record-{polarity}",
        source_text=source,
        evidence_text=evidence,
        subject_id=entry["subject_id"],
        subject_text="n0",
        predicate=entry["predicate"],
        object_id=entry["object_id"],
        object_text="red",
        polarity=polarity,
        asp_atom=entry["asp_atom"],
    )
    return source, vocabulary, record


def _asset_bundle(cache_root: Path) -> dict[str, Any]:
    shard = cache_root / "enokiqa_source_shard.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text('{"id":"row-0","question":"Q","paragraph_context":"E"}\n')
    return {
        "hashes_valid": True,
        "asset_receipts": [
            {
                "asset_id": "enoki_encoder",
                "revision": mod.ENCODER_REVISION,
                "cache_path": str(cache_root / "encoder"),
                "files": [{"path": "model.safetensors", "sha256": "a" * 64}],
            },
            {
                "asset_id": "enokiqa_source_shard",
                "revision": mod.ENOKIQA_REVISION,
                "cache_path": str(shard),
                "files": [{"path": str(shard), "sha256": mod.sha256_path(shard)}],
            },
        ],
        "revision_rows": [
            {"asset_id": "enoki_encoder", "revision": mod.ENCODER_REVISION, "passed": True},
            {"asset_id": "enokiqa_source_shard", "revision": mod.ENOKIQA_REVISION, "passed": True},
        ],
        "license_rows": [
            {
                "asset_id": "enoki_encoder",
                "declared_license": None,
                "license_status": "model_card_has_no_derivative_license_declaration",
            },
            {
                "asset_id": "enokiqa_source_shard",
                "declared_license": "cc-by-sa-4.0",
                "license_status": "declared",
            },
        ],
        "row_hashes": [
            mod.sha256_json({"id": f"row-{index}"}) for index in range(mod.ENOKIQA_SHARD_SIZE)
        ],
        "shard_size": mod.ENOKIQA_SHARD_SIZE,
        "cache_manifest_path": str(cache_root / "asset_manifest.json"),
    }


@pytest.fixture(scope="module")
def ready_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Build the expensive 150-fixture result once for artifact checks."""

    root = tmp_path_factory.mktemp("exp6886-ready")
    return mod.build_artifact(
        date="20260902",
        cache_root=root / "cache",
        asset_bundle=_asset_bundle(root / "assets"),
        duration_s=0.25,
    )


def test_req_constraint_6886_spec_owns_contract() -> None:
    """REQ-CONSTRAINT-6886 declares fields and all required scenarios first."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-CONSTRAINT-6886") :]
    for scenario in (
        "SCENARIO-CONSTRAINT-6886-ASSETS",
        "SCENARIO-CONSTRAINT-6886-ANCHORS",
        "SCENARIO-CONSTRAINT-6886-CLOSED-MAP",
        "SCENARIO-CONSTRAINT-6886-FIXTURES",
        "SCENARIO-CONSTRAINT-6886-NONEXPOSURE",
        "SCENARIO-CONSTRAINT-6886-PARITY",
        "SCENARIO-CONSTRAINT-6886-FAIL-CLOSED",
        "SCENARIO-CONSTRAINT-6886-ARTIFACT",
    ):
        assert scenario in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section or field == "field_principles"


def test_scenario_6886_assets_are_exact_and_source_only() -> None:
    """SCENARIO-CONSTRAINT-6886-ASSETS pins the published immutable revisions."""

    assert mod.ENCODER_REVISION == "3be7767049d8db73ede6eab5c27c0d98faddeaf8"
    assert mod.ENOKIQA_REVISION == "d764e01aa55ab90ca623b3a5fda24e122155b61e"
    assert "model.safetensors" in mod.ENCODER_FILE_SHA256
    assert "enoki-openie-banner.png" not in mod.ENCODER_FILE_SHA256
    assert "answer" not in mod.ENOKIQA_SHARD_COLUMNS
    assert "full_page_context" not in mod.ENOKIQA_SHARD_COLUMNS


def test_scenario_6886_anchor_builder_and_family_fail_closed() -> None:
    """SCENARIO-CONSTRAINT-6886-ANCHORS rejects absent anchors and unknown families."""

    with pytest.raises(ValueError, match="anchor_not_found"):
        mod.build_anchored_relation(
            record_id="missing",
            source_text="No relation is present.",
            evidence_text="missing evidence",
            subject_id="s",
            subject_text="s",
            predicate="p",
            object_id="o",
            object_text="o",
            polarity="positive",
            asp_atom="a",
        )
    with pytest.raises(ValueError, match="unsupported_family"):
        mod.build_closed_vocabulary("unknown_family", 0)


def test_scenario_6886_cache_revision_drift_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-CONSTRAINT-6886-ASSETS rejects an incompatible target cache."""

    manifest = tmp_path / mod.CACHE_MANIFEST_NAME
    manifest.write_text(
        json.dumps(
            {
                "schema": mod.CACHE_SCHEMA,
                "encoder_revision": "changed",
                "enokiqa_revision": mod.ENOKIQA_REVISION,
                "shard_size": mod.ENOKIQA_SHARD_SIZE,
                "shard_columns": list(mod.ENOKIQA_SHARD_COLUMNS),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(mod.AssetUnavailable, match="cache_hash_drift:encoder_revision"):
        mod.check_cache_identity(tmp_path)


def test_scenario_6886_unreadable_cache_manifest_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-CONSTRAINT-6886-ASSETS rejects an unreadable cache identity."""

    (tmp_path / mod.CACHE_MANIFEST_NAME).write_text("{not-json", encoding="utf-8")
    with pytest.raises(mod.AssetUnavailable, match="cache_manifest_unreadable"):
        mod.check_cache_identity(tmp_path)


def test_scenario_6886_asset_preparation_hashes_files_and_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CONSTRAINT-6886-ASSETS records bounded files and projected rows."""

    contents = {name: f"fixture:{name}".encode() for name in mod.ENCODER_FILE_SHA256}
    expected = {name: mod.sha256_bytes(value) for name, value in contents.items()}
    monkeypatch.setattr(mod, "ENCODER_FILE_SHA256", expected)

    def fake_download(
        repo_id: str, filename: str, *, revision: str, repo_type: str | None = None
    ) -> str:
        assert repo_id == mod.ENCODER_REPO
        assert revision == mod.ENCODER_REVISION
        assert repo_type is None
        path = tmp_path / "hub" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents[filename])
        return str(path)

    rows = [
        {
            "id": f"row-{index}",
            "title": f"Title {index}",
            "question": f"Question {index}?",
            "paragraph_context": f"Evidence {index}.",
            "context_id": f"context-{index}",
            "wiki_url": f"https://example.test/{index}",
        }
        for index in range(mod.ENOKIQA_SHARD_SIZE)
    ]
    bundle = mod.prepare_enoki_assets(
        tmp_path / "cache",
        hub_download=fake_download,
        dataset_reader=lambda: rows,
    )

    assert bundle["hashes_valid"] is True
    assert bundle["shard_size"] == mod.ENOKIQA_SHARD_SIZE
    assert len(bundle["row_hashes"]) == mod.ENOKIQA_SHARD_SIZE
    assert Path(bundle["asset_receipts"][1]["cache_path"]).is_file()
    assert mod.check_cache_identity(tmp_path / "cache")["compatible"] is True


def test_scenario_6886_default_asset_adapters_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CONSTRAINT-6886-ASSETS uses bounded default hub adapters."""

    import huggingface_hub
    import pyarrow.parquet

    rows = [
        {column: f"{column}-{index}" for column in mod.ENOKIQA_SHARD_COLUMNS}
        for index in range(mod.ENOKIQA_SHARD_SIZE)
    ]

    class FakeSource:
        def __enter__(self) -> object:
            return object()

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeFileSystem:
        def open(self, uri: str, mode: str) -> FakeSource:
            assert mod.ENOKIQA_REVISION in uri
            assert mode == "rb"
            return FakeSource()

    class FakeTable:
        def slice(self, start: int, size: int) -> "FakeTable":
            assert (start, size) == (0, mod.ENOKIQA_SHARD_SIZE)
            return self

        def to_pylist(self) -> list[dict[str, Any]]:
            return rows

    class FakeParquet:
        def read_row_group(self, group: int, *, columns: list[str]) -> FakeTable:
            assert group == 0
            assert columns == list(mod.ENOKIQA_SHARD_COLUMNS)
            return FakeTable()

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", lambda: FakeFileSystem())
    monkeypatch.setattr(pyarrow.parquet, "ParquetFile", lambda _source: FakeParquet())
    assert mod.read_enokiqa_source_rows() == rows

    contents = {name: name.encode() for name in mod.ENCODER_FILE_SHA256}
    monkeypatch.setattr(
        mod,
        "ENCODER_FILE_SHA256",
        {name: mod.sha256_bytes(value) for name, value in contents.items()},
    )

    def default_download(repo_id: str, filename: str, **kwargs: Any) -> str:
        assert repo_id == mod.ENCODER_REPO
        assert kwargs["revision"] == mod.ENCODER_REVISION
        path = tmp_path / "default-hub" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents[filename])
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", default_download)
    bundle = mod.prepare_enoki_assets(tmp_path / "default-cache", dataset_reader=lambda: rows)
    assert bundle["hashes_valid"] is True


@pytest.mark.parametrize("failure", ["hash", "generic", "size"])
def test_scenario_6886_asset_preparation_failures_are_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """SCENARIO-CONSTRAINT-6886-ASSETS types hash, transport, and shard failures."""

    filename = next(iter(mod.ENCODER_FILE_SHA256))
    monkeypatch.setattr(mod, "ENCODER_FILE_SHA256", {filename: mod.sha256_bytes(b"expected")})
    path = tmp_path / f"{failure}.bin"
    path.write_bytes(b"wrong" if failure == "hash" else b"expected")

    def downloader(*_args: Any, **_kwargs: Any) -> str:
        if failure == "generic":
            raise OSError("offline")
        return str(path)

    rows = (
        []
        if failure == "size"
        else [
            {column: f"{column}-{index}" for column in mod.ENOKIQA_SHARD_COLUMNS}
            for index in range(mod.ENOKIQA_SHARD_SIZE)
        ]
    )
    expected = {
        "hash": "cache_hash_drift",
        "generic": "asset_unavailable",
        "size": "bounded_shard_size",
    }[failure]
    with pytest.raises(mod.AssetUnavailable, match=expected):
        mod.prepare_enoki_assets(
            tmp_path / f"cache-{failure}",
            hub_download=downloader,
            dataset_reader=lambda: rows,
        )


def test_scenario_6886_unavailable_assets_write_complete_blocked_artifact(tmp_path: Path) -> None:
    """SCENARIO-CONSTRAINT-6886-FAIL-CLOSED preserves unavailable asset evidence."""

    def unavailable(_: Path) -> dict[str, Any]:
        raise mod.AssetUnavailable("asset_unavailable:encoder")

    result_path = tmp_path / "blocked.json"
    artifact = mod.run(
        date="20260902",
        result_path=result_path,
        cache_root=tmp_path / "cache",
        asset_loader=unavailable,
        write=True,
    )

    assert result_path.is_file()
    assert artifact["relation_fixture_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "complete_blocked_enoki_exact_relation_fixture"
    assert artifact["gate_check_summary"]["failed_check"] == "enoki_assets_available"
    assert artifact["gate_check_summary"]["expected"] is True
    assert "asset_unavailable" in artifact["gate_check_summary"]["observed"]


def test_scenario_6886_utf8_byte_offsets_are_exact() -> None:
    """SCENARIO-CONSTRAINT-6886-ANCHORS accepts bytes and rejects character offsets."""

    source, vocabulary, record = _graph_record()
    mapped = mod.validate_relation(record, source, vocabulary)
    assert mapped["accepted"] is True
    assert mapped["asp_atom"] == record["asp_atom"]

    wrong = deepcopy(record)
    char_start = source.index("n0")
    wrong["subject"]["span"]["start_utf8"] = char_start
    wrong["subject"]["span"]["end_utf8"] = char_start + 2
    with pytest.raises(mod.RelationValidationError, match="invalid_span"):
        mod.validate_relation(wrong, source, vocabulary)


def test_scenario_6886_utf8_boundaries_and_span_text_fail_closed() -> None:
    """SCENARIO-CONSTRAINT-6886-ANCHORS separates byte boundaries from text drift."""

    with pytest.raises(mod.RelationValidationError, match="utf8_boundary"):
        mod._slice_span("éx", {"start_utf8": 1, "end_utf8": 2, "text": "x"})
    with pytest.raises(mod.RelationValidationError, match="span_text_mismatch"):
        mod._slice_span("abc", {"start_utf8": 0, "end_utf8": 1, "text": "z"})


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("reversed", "invalid_span"),
        ("out_of_range", "invalid_span"),
        ("wrong_text", "span_text_mismatch"),
        ("outside_evidence", "span_outside_evidence"),
        ("source_hash", "source_text_hash"),
        ("provenance", "provenance_hash"),
    ],
)
def test_scenario_6886_invalid_spans_and_hashes_fail(mutation: str, reason: str) -> None:
    """SCENARIO-CONSTRAINT-6886-ANCHORS rejects malformed source evidence."""

    source, vocabulary, record = _graph_record()
    changed = deepcopy(record)
    if mutation == "reversed":
        changed["evidence_span"]["end_utf8"] = changed["evidence_span"]["start_utf8"]
    elif mutation == "out_of_range":
        changed["object"]["span"]["end_utf8"] = len(source.encode()) + 1
    elif mutation == "wrong_text":
        changed["object"]["text"] = "blue"
    elif mutation == "outside_evidence":
        changed["evidence_span"]["start_utf8"] = changed["subject"]["span"]["end_utf8"] + 1
    elif mutation == "source_hash":
        changed["source_text_hash"] = "sha256:" + "0" * 64
    else:
        changed["provenance_hash"] = "sha256:" + "0" * 64

    with pytest.raises(mod.RelationValidationError, match=reason):
        mod.validate_relation(changed, source, vocabulary)


@pytest.mark.parametrize("mutation", ["missing", "outside_source", "short_source"])
def test_scenario_6886_structural_span_failures_are_specific(mutation: str) -> None:
    """SCENARIO-CONSTRAINT-6886-ANCHORS rejects missing and structurally nested spans."""

    source, vocabulary, record = _graph_record()
    changed = deepcopy(record)
    expected = mutation
    if mutation == "missing":
        changed.pop("evidence_span")
        expected = "missing_field"
    elif mutation == "outside_source":
        changed["source_span"] = deepcopy(changed["evidence_span"])
        changed["evidence_span"] = deepcopy(record["source_span"])
        expected = "span_outside_source"
    else:
        changed["source_span"] = deepcopy(changed["evidence_span"])
        expected = "source_span_not_complete"
    with pytest.raises(mod.RelationValidationError, match=expected):
        mod.validate_relation(changed, source, vocabulary)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("unknown_subject", "unknown_entity"),
        ("unknown_object", "unknown_entity"),
        ("predicate", "unsupported_predicate"),
        ("polarity", "invalid_polarity"),
        ("atom", "asp_atom_mismatch"),
        ("tuple", "normalized_tuple"),
    ],
)
def test_scenario_6886_closed_map_rejects_unsupported_rows(mutation: str, reason: str) -> None:
    """SCENARIO-CONSTRAINT-6886-CLOSED-MAP rejects records outside the map."""

    source, vocabulary, record = _graph_record()
    changed = deepcopy(record)
    if mutation == "unknown_subject":
        changed["subject"]["entity_id"] = "unknown"
        changed["normalized_tuple"][0] = "unknown"
    elif mutation == "unknown_object":
        changed["object"]["entity_id"] = "unknown"
        changed["normalized_tuple"][2] = "unknown"
    elif mutation == "predicate":
        changed["predicate"] = "invented_predicate"
        changed["normalized_tuple"][1] = "invented_predicate"
    elif mutation == "polarity":
        changed["polarity"] = "uncertain"
        changed["normalized_tuple"][3] = "uncertain"
    elif mutation == "atom":
        changed["asp_atom"] = "wrong_atom"
    else:
        changed["normalized_tuple"] = list(reversed(changed["normalized_tuple"]))
    changed["provenance_hash"] = mod.relation_provenance_hash(changed)

    with pytest.raises(mod.RelationValidationError, match=reason):
        mod.validate_relation(changed, source, vocabulary)


@pytest.mark.parametrize("duplicate_kind", ["record_id", "normalized_tuple"])
def test_scenario_6886_duplicates_fail_closed(duplicate_kind: str) -> None:
    """SCENARIO-CONSTRAINT-6886-CLOSED-MAP rejects both duplicate identities."""

    source, vocabulary, record = _graph_record()
    duplicate = deepcopy(record)
    if duplicate_kind == "normalized_tuple":
        duplicate["record_id"] = "different-record"
        duplicate["provenance_hash"] = mod.relation_provenance_hash(duplicate)

    rows = mod.validate_relation_set([record, duplicate], source, vocabulary)

    assert rows[0]["accepted"] is True
    assert rows[1]["accepted"] is False
    assert rows[1]["reason"] == f"duplicate_{duplicate_kind}"


def test_scenario_6886_polarity_and_negation_map_to_distinct_atoms() -> None:
    """SCENARIO-CONSTRAINT-6886-CLOSED-MAP keeps explicit negation injective."""

    positive_source, vocabulary, positive = _graph_record(polarity="positive")
    negative_source, _, negative = _graph_record(polarity="negative")

    positive_row = mod.validate_relation(positive, positive_source, vocabulary)
    negative_row = mod.validate_relation(negative, negative_source, vocabulary)

    assert positive_row["accepted"] is True
    assert negative_row["accepted"] is True
    assert positive_row["asp_atom"] != negative_row["asp_atom"]
    assert negative["polarity"] == "negative"
    assert "not" in negative["evidence_span"]["text"]


def test_scenario_6886_atom_collision_is_reported() -> None:
    """SCENARIO-CONSTRAINT-6886-CLOSED-MAP rejects two tuples for one atom."""

    vocabulary = mod.build_closed_vocabulary("graph_coloring", 0)
    collision = deepcopy(vocabulary)
    collision[1]["asp_atom"] = collision[0]["asp_atom"]

    rows = mod.validate_vocabulary(collision)

    assert rows == [
        {
            "asp_atom": collision[0]["asp_atom"],
            "normalized_tuples": [
                collision[0]["normalized_tuple"],
                collision[1]["normalized_tuple"],
            ],
            "passed": False,
        }
    ]


def test_scenario_6886_unmapped_tuple_and_record_atom_collision_fail() -> None:
    """SCENARIO-CONSTRAINT-6886-CLOSED-MAP rejects cross-pairs and set collisions."""

    source, vocabulary, positive = _graph_record()
    crossed = deepcopy(positive)
    crossed["object"]["entity_id"] = "blue"
    crossed["normalized_tuple"][2] = "blue"
    crossed["provenance_hash"] = mod.relation_provenance_hash(crossed)
    cross_vocabulary = deepcopy(vocabulary)
    cross_vocabulary.append(
        {
            **deepcopy(vocabulary[0]),
            "subject_id": "other",
            "object_id": "blue",
            "normalized_tuple": ["other", "has_color", "blue", "positive"],
            "asp_atom": "other_blue",
        }
    )
    with pytest.raises(mod.RelationValidationError, match="unmapped_tuple"):
        mod.validate_relation(crossed, source, cross_vocabulary)

    negative_source, _, negative = _graph_record(polarity="negative")
    combined_source = f"{source} {negative_source}"
    first = mod.build_anchored_relation(
        record_id="first",
        source_text=combined_source,
        evidence_text=positive["evidence_span"]["text"],
        subject_id=positive["subject"]["entity_id"],
        subject_text="n0",
        predicate=positive["predicate"],
        object_id=positive["object"]["entity_id"],
        object_text="red",
        polarity="positive",
        asp_atom=positive["asp_atom"],
    )
    colliding_vocabulary = deepcopy(vocabulary)
    colliding_vocabulary[1]["asp_atom"] = positive["asp_atom"]
    second = mod.build_anchored_relation(
        record_id="second",
        source_text=combined_source,
        evidence_text=negative["evidence_span"]["text"],
        subject_id=negative["subject"]["entity_id"],
        subject_text="n0",
        predicate=negative["predicate"],
        object_id=negative["object"]["entity_id"],
        object_text="red",
        polarity="negative",
        asp_atom=positive["asp_atom"],
    )
    rows = mod.validate_relation_set([first, second], combined_source, colliding_vocabulary)
    assert rows[1]["reason"] == "atom_collision"


def test_scenario_6886_malformed_relation_set_row_is_preserved() -> None:
    """SCENARIO-CONSTRAINT-6886-CLOSED-MAP preserves malformed nested records."""

    source, vocabulary, record = _graph_record()
    malformed = deepcopy(record)
    malformed["subject"] = None
    rows = mod.validate_relation_set([malformed], source, vocabulary)
    assert rows == [
        {"record_id": record["record_id"], "accepted": False, "reason": "malformed_record"}
    ]


def test_scenario_6886_fixture_balance_and_split_isolation() -> None:
    """SCENARIO-CONSTRAINT-6886-FIXTURES freezes 150 balanced fixtures."""

    fixtures = mod.build_fixtures()
    families = Counter(row["family"] for row in fixtures)
    cases = Counter((row["family"], row["expected_case"]) for row in fixtures)
    calibration = {row["group_id"] for row in fixtures if row["split"] == "calibration"}
    held = {row["group_id"] for row in fixtures if row["split"] == "held"}

    assert len(fixtures) == 150
    assert set(families.values()) == {30}
    assert set(cases.values()) == {6}
    assert calibration.isdisjoint(held)
    assert len(calibration) == len(held) == 15


def test_scenario_6886_prompt_nonexposure_covers_fields_values_and_sidecars() -> None:
    """SCENARIO-CONSTRAINT-6886-NONEXPOSURE mechanically checks every hidden item."""

    fixture = mod.build_fixtures()[0]
    prompt = mod.public_prompt_view(fixture)
    hidden = {
        "expected_case": fixture["expected_case"],
        "asp_program": fixture["base_program"],
        "answer_sets": [["hidden_atom"]],
        "solver_receipt": {"solver": "hidden-receipt-token"},
    }
    rows = mod.audit_prompt_nonexposure(prompt, hidden, ["/cache/sealed-held.json"])
    assert len(rows) == 5
    assert all(row["passed"] for row in rows)

    leaking = dict(prompt)
    leaking["asp_program"] = fixture["base_program"]
    leaked = mod.audit_prompt_nonexposure(leaking, hidden, ["/cache/sealed-held.json"])
    assert (
        next(row for row in leaked if row["check"] == "hidden_field:asp_program")["passed"] is False
    )


def test_scenario_6886_solver_timeout_fails_closed() -> None:
    """SCENARIO-CONSTRAINT-6886-FAIL-CLOSED rejects a timed-out solver call."""

    compiled = mod.asp_energy.compile_program("a.", program_id="timeout")

    def slow_solver(_: Any) -> list[list[str]]:
        time.sleep(0.02)
        return [["a"]]

    with pytest.raises(mod.SolverTimeoutError, match="solver_timeout"):
        mod.solve_with_timeout(compiled.program, timeout_s=0.001, solver=slow_solver)
    with pytest.raises(mod.SolverTimeoutError, match="solver_timeout"):
        mod.solve_with_timeout(compiled.program, timeout_s=0.0)

    fixture = mod.build_fixtures()[0]
    report = mod.evaluate_fixture(fixture, timeout_s=0.0)
    assert report["solver_parity_row"]["passed"] is False
    assert "solver_timeout" in report["solver_parity_row"]["solver_error"]


def test_scenario_6886_solver_disagreement_is_not_parity() -> None:
    """SCENARIO-CONSTRAINT-6886-FAIL-CLOSED keeps independent disagreement visible."""

    fixture = mod.build_fixtures()[0]
    report = mod.evaluate_fixture(
        fixture,
        solver=lambda _: [["deliberately_wrong_atom"]],
        timeout_s=1.0,
    )

    assert report["solver_parity_row"]["semantic_parity"] is False
    assert report["solver_parity_row"]["passed"] is False


def test_scenario_6886_cardinality_and_local_receipts_are_preserved() -> None:
    """SCENARIO-CONSTRAINT-6886-PARITY preserves cardinality energy causes."""

    fixture = next(
        row for row in mod.build_fixtures() if row["family"] == "cardinality_constraints"
    )
    report = mod.evaluate_fixture(fixture, timeout_s=1.0)

    assert report["solver_parity_row"]["passed"] is True
    assert any(row["kind"] == "cardinality" for row in report["rule_violation_receipts"])
    assert all(row["energy"] > 0 for row in report["rule_violation_receipts"])


def test_scenario_6886_ready_artifact_replays_all_rows(ready_artifact: dict[str, Any]) -> None:
    """SCENARIO-CONSTRAINT-6886-ARTIFACT opens readiness only on exact row gates."""

    artifact = ready_artifact

    assert artifact["relation_fixture_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["split_overlap_count"] == 0
    assert len(artifact["solver_parity_rows"]) == 150
    assert all(row["passed"] for row in artifact["solver_parity_rows"])
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("score", "ready_score"),
        ("verdict", "verdict_class"),
        ("oracle", "verifier_is_oracle"),
        ("substrate", "inference_substrate"),
        ("principle", "field_principles"),
        ("checksum", "reproducibility_checksum"),
    ],
)
def test_scenario_6886_artifact_tampering_fails_validation(
    ready_artifact: dict[str, Any], mutation: str, message: str
) -> None:
    """SCENARIO-CONSTRAINT-6886-ARTIFACT rejects summary or checksum drift."""

    artifact = deepcopy(ready_artifact)
    if mutation == "score":
        artifact["relation_fixture_ready_score"] = 0
    elif mutation == "verdict":
        artifact["verdict_class"] = "positive"
    elif mutation == "oracle":
        artifact["verifier_is_oracle"] = False
    elif mutation == "substrate":
        artifact["inference_substrate"] = "live_llm_inference"
    elif mutation == "principle":
        artifact["field_principles"].pop("rows")
    else:
        artifact["reproducibility_checksum"] = "sha256:" + "0" * 64

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(artifact)


@pytest.mark.parametrize("mutation", ["required", "honest", "ready_class"])
def test_scenario_6886_artifact_shape_failures_are_typed(
    ready_artifact: dict[str, Any], mutation: str
) -> None:
    """SCENARIO-CONSTRAINT-6886-ARTIFACT rejects missing and overclaimed terminal fields."""

    artifact = deepcopy(ready_artifact)
    if mutation == "required":
        artifact.pop("rows")
        expected = "required_fields"
    elif mutation == "honest":
        artifact["honest_verdict"] = "not_terminal"
        expected = "honest_verdict"
    else:
        artifact["verdict_class"] = "null"
        expected = "verdict_class"
    with pytest.raises(ValueError, match=expected):
        mod.validate_artifact(artifact)


def test_req_constraint_6886_run_writes_only_requested_result_and_external_cache(
    tmp_path: Path,
) -> None:
    """REQ-CONSTRAINT-6886 keeps test writes inside explicit temporary paths."""

    result_path = tmp_path / "results" / "experiment_6886.json"
    artifact = mod.run(
        date="20260902",
        result_path=result_path,
        cache_root=tmp_path / "cache",
        asset_loader=lambda _: _asset_bundle(tmp_path / "assets"),
        write=True,
    )

    assert result_path.is_file()
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert not (ROOT / "experiment_6886_enoki_exact_relation_fixture.py").exists()


def test_req_constraint_6886_missing_compiler_and_cli_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-CONSTRAINT-6886 reports a missing compiler and keeps the CLI terminal."""

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    receipt = mod._compiler_qualified()
    assert receipt["available"] is False

    monkeypatch.setattr(
        mod,
        "run",
        lambda **_kwargs: {"honest_verdict": "complete_fixture_cli_test"},
    )
    assert mod.main(["--date", "20260902"]) == 0
    assert "complete_fixture_cli_test" in capsys.readouterr().out
