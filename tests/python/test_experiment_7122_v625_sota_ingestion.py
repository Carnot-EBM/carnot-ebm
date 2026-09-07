"""Tests for REQ-REPORT-7122 and SCENARIO-REPORT-7122-* contracts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from carnot import experiment_7122_v625_sota_ingestion as exp


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _root(tmp_path: Path, *, planner_marker: bool = True) -> Path:
    """Create private inputs so a test cannot change the research record."""

    for relative in exp.REQUIRED_INPUT_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "test input\n"
        if relative == exp.REFERENCE_PATH and planner_marker:
            text += f"\n{exp.PLANNER_START_MARKER}\n{exp.PLANNER_END_MARKER}\n"
        path.write_text(text, encoding="utf-8")
    (tmp_path / exp.RESULT_PATH).parent.mkdir(parents=True, exist_ok=True)
    return tmp_path


def _cache(tmp_path: Path, *, include_models: bool = True) -> Path:
    """Build a small Hugging Face cache with real content hashes."""

    cache = tmp_path / "hub"
    cache.mkdir(parents=True)
    if not include_models:
        return cache
    names = {
        exp.MODEL_REPOSITORY_IDS[0]: "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        exp.MODEL_REPOSITORY_IDS[1]: "gemma-4-31B-it-Q4_K_M.gguf",
        exp.MODEL_REPOSITORY_IDS[2]: "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
    }
    for index, (hf_id, file_name) in enumerate(names.items(), start=1):
        repo = cache / f"models--{hf_id.replace('/', '--')}"
        blob = repo / "blobs" / hashlib.sha256(f"model-{index}".encode()).hexdigest()
        blob.parent.mkdir(parents=True)
        blob.write_bytes(f"model-{index}".encode())
        snapshot = repo / "snapshots" / f"revision-{index}"
        snapshot.mkdir(parents=True)
        (snapshot / file_name).symlink_to(Path("../..") / "blobs" / blob.name)
        (snapshot / "mmproj-F16.gguf").write_bytes(b"projector")
    return cache


def _routes(**changes: bool) -> dict[str, bool]:
    """Return a complete reachable-route fixture."""

    routes = {name: True for name in exp.SOURCE_COLLECTION_FIELDS}
    routes.update(changes)
    return routes


def _valid_artifact(tmp_path: Path) -> dict[str, object]:
    """Build one complete artifact under private paths."""

    root = _root(tmp_path)
    cache = _cache(tmp_path)
    return exp.build_artifact(
        root,
        exp.RUN_DATE,
        cache_root=cache,
        output_path=root / exp.RESULT_PATH,
        route_reachability=_routes(),
        update_references=True,
        duration_s=1.25,
    )


def test_req_report_7122_spec_precedes_implementation() -> None:
    """REQ-REPORT-7122 owns every required field and named scenario."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7122") :]
    for scenario in (
        "PREFLIGHT",
        "CLASSIFY",
        "DEDUP",
        "DATES",
        "MODELS",
        "MAP",
        "APPEND",
        "ARTIFACT",
    ):
        assert f"SCENARIO-REPORT-7122-{scenario}" in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_7122_classify_has_direct_bounded_sources(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7122-CLASSIFY separates all evidence classes."""

    artifact = _valid_artifact(tmp_path)
    classes = {row["source_type"] for row in artifact["source_class_rows"]}
    assert set(exp.SOURCE_TYPES).issubset(classes)
    assert all(exp.valid_direct_url(row["url"]) for row in artifact["source_class_rows"])
    assert all(row["terminal"] is True for row in artifact["source_class_rows"])
    assert all(row["claim_boundary"] for row in artifact["source_class_rows"])
    assert artifact["extropic_rows"][0]["claim_boundary_label"] == exp.VENDOR_BOUNDARY
    assert (
        artifact["logical_intelligence_rows"][0]["claim_boundary_label"]
        == exp.PRODUCT_BOUNDARY
    )


def test_scenario_report_7122_dedup_prefers_primary_source(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7122-DEDUP removes secondary copies."""

    rows = [
        {
            "source_id": "arxiv:2609.04343",
            "url": "https://huggingface.co/papers/2609.04343",
            "source_type": "secondary_discovery",
        },
        {
            "source_id": "arxiv:2609.04343",
            "url": "https://arxiv.org/abs/2609.04343",
            "source_type": "preprint",
        },
    ]
    assert exp.deduplicate_source_rows(rows) == [rows[1]]

    artifact = _valid_artifact(tmp_path)
    assert len({row["source_id"] for row in artifact["source_class_rows"]}) == len(
        artifact["source_class_rows"]
    )
    forged = deepcopy(artifact)
    forged["source_class_rows"].append(deepcopy(forged["source_class_rows"][0]))
    assert exp.validate_artifact(forged)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2025-01-01", True),
        ("2026-09-07", True),
        ("2024-12-31", False),
        ("2026-09-08", False),
        ("bad", False),
        (None, False),
    ],
)
def test_scenario_report_7122_dates_are_bounded(value: str | None, expected: bool) -> None:
    """SCENARIO-REPORT-7122-DATES applies inclusive 2025-2026 bounds."""

    assert exp.publication_date_in_window(value, exp.RUN_DATE) is expected


def test_scenario_report_7122_url_fields_reject_indirect_or_bad_urls(tmp_path: Path) -> None:
    """REQ-REPORT-7122 requires direct HTTPS source and repository URLs."""

    artifact = _valid_artifact(tmp_path)
    for collection in exp.SOURCE_COLLECTION_FIELDS:
        for row in artifact[collection]:
            for key, value in row.items():
                if key == "url" or key.endswith("_url"):
                    assert exp.valid_direct_url(value)
    for value in ("http://arxiv.org/abs/2609.04343", "not-a-url", None):
        assert not exp.valid_direct_url(value)

    forged = deepcopy(artifact)
    forged["arxiv_rows"][0]["query_url"] = "http://example.com"
    assert exp.validate_artifact(forged)


def test_scenario_report_7122_models_record_exact_cache_facts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7122-MODELS records files without a download."""

    artifact = _valid_artifact(tmp_path)
    repositories = artifact["model_repository_rows"]
    assert [row["hf_id"] for row in repositories] == list(exp.MODEL_REPOSITORY_IDS)
    assert all(row["download_performed"] is False for row in repositories)
    assert all(row["repository_url"].startswith("https://huggingface.co/") for row in repositories)

    cached = artifact["cached_model_rows"]
    assert {row["hf_id"] for row in cached} == set(exp.MODEL_REPOSITORY_IDS)
    assert all(row["cache_status"] == "resolved" for row in cached)
    assert all(row["file_name"].endswith(".gguf") for row in cached)
    assert all("mmproj" not in row["file_name"].lower() for row in cached)
    assert all(row["quantization"] == "Q4_K_M" for row in cached)
    assert all(row["size_bytes"] > 0 for row in cached)
    assert all(str(row["sha256"]).startswith("sha256:") for row in cached)
    assert all(row["download_performed"] is False for row in cached)

    empty = exp.inspect_cached_models(_cache(tmp_path / "empty", include_models=False))
    assert len(empty) == 3
    assert all(row["cache_status"] == "missing" for row in empty)
    assert all(row["path"] is None and row["sha256"] is None for row in empty)


def test_scenario_report_7122_map_is_exact_and_other_work_is_deferred(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7122-MAP limits promotion to four exact tasks."""

    artifact = _valid_artifact(tmp_path)
    assert [row["target_task_id"] for row in artifact["task_method_map_rows"]] == [
        "exp7125-arc-loo-causal-audit",
        "exp7127-verifier-committed-revision",
        "exp7129-cross-model-memory-portability",
        "exp7131-wcrg-multiscale-sampler-prototype",
    ]
    assert all(row["decision_changing"] is True for row in artifact["task_method_map_rows"])
    assert all(row["claim_boundary"] for row in artifact["task_method_map_rows"])
    assert artifact["deferred_rows"]
    assert all(row["reason"] and row["target_task_id"] is None for row in artifact["deferred_rows"])

    forged = deepcopy(artifact)
    forged["task_method_map_rows"].append(deepcopy(forged["task_method_map_rows"][0]))
    assert exp.validate_artifact(forged)


def test_scenario_report_7122_append_markers_are_idempotent(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7122-APPEND writes one complete marker pair."""

    path = _root(tmp_path) / exp.REFERENCE_PATH
    cached = exp.inspect_cached_models(_cache(tmp_path))
    first = exp.append_references_delta(path, exp.task_method_map_rows(), cached, exp.RUN_DATE)
    once = path.read_bytes()
    second = exp.append_references_delta(path, exp.task_method_map_rows(), cached, exp.RUN_DATE)
    text = path.read_text(encoding="utf-8")
    assert first is True
    assert second is False
    assert path.read_bytes() == once
    assert text.count(exp.REFERENCE_START_MARKER) == 1
    assert text.count(exp.REFERENCE_END_MARKER) == 1
    for row in exp.task_method_map_rows():
        assert row["target_task_id"] in text


@pytest.mark.parametrize(
    ("routes", "planner_marker", "cache_exists", "failed_check"),
    [
        ({name: False for name in exp.SOURCE_COLLECTION_FIELDS}, True, True, "network_access"),
        (_routes(), False, True, "v625_planner_marker"),
        (_routes(), True, False, "model_cache_metadata"),
    ],
)
def test_scenario_report_7122_preflight_blocks_exactly(
    tmp_path: Path,
    routes: dict[str, bool],
    planner_marker: bool,
    cache_exists: bool,
    failed_check: str,
) -> None:
    """SCENARIO-REPORT-7122-PREFLIGHT preserves the first failed gate."""

    root = _root(tmp_path, planner_marker=planner_marker)
    cache = _cache(tmp_path) if cache_exists else tmp_path / "missing-cache"
    artifact = exp.build_artifact(
        root,
        exp.RUN_DATE,
        cache_root=cache,
        output_path=root / exp.RESULT_PATH,
        route_reachability=routes,
        update_references=True,
        duration_s=0.1,
    )
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["verdict_class"] == "blocked"
    assert str(artifact["honest_verdict"]).startswith("blocked_")
    assert artifact["gate_check_summary"]["failed_check"] == failed_check
    assert artifact["gate_check_summary"]["expected_value"] is not None
    assert artifact["gate_check_summary"]["observed_value"] is not None
    assert artifact["v625_sota_ingestion_complete_score"] == 0
    assert exp.validate_artifact(artifact) == []


def test_scenario_report_7122_artifact_recomputes_rows_score_and_checksum(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7122-ARTIFACT rejects forged derived fields."""

    artifact = _valid_artifact(tmp_path)
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["rows"] == exp.combined_rows(artifact)
    assert artifact["inference_substrate"].startswith(
        "aggregation_from_external_primary_sources"
    )
    assert artifact["v625_sota_ingestion_complete_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert str(artifact["honest_verdict"]).startswith("positive_")
    assert artifact["reproducibility_checksum"] == exp.payload_checksum(artifact)
    assert exp.validate_artifact(artifact) == []

    for field, value in (
        ("rows", []),
        ("field_principles", {}),
        ("run_date", "20260908"),
        ("inference_substrate", "model inference"),
        ("inference_substrate_class", "no_model_load"),
        ("execution_venue", "gpu"),
        ("duration_s", -1),
        ("v625_sota_ingestion_complete_score", 0),
        ("random_seed", 0),
        ("gate_check_summary", {}),
        ("verifier_is_oracle", True),
        ("verdict_class", "partial"),
        ("verdict_class", "unknown"),
        ("honest_verdict", "blocked_forged"),
        ("reproducibility_checksum", "sha256:bad"),
    ):
        forged = deepcopy(artifact)
        forged[field] = value
        assert exp.validate_artifact(forged), field

    missing = deepcopy(artifact)
    missing.pop("arxiv_rows")
    assert "artifact fields mismatch" in exp.validate_artifact(missing)[0]
    assert exp.validate_artifact([]) == ["artifact_not_object"]
    assert exp.validate_artifact(tmp_path / "missing.json") == ["artifact_missing"]


def test_req_report_7122_io_and_command_line_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-7122 covers storage, route probes, and command output."""

    root = _root(tmp_path)
    cache = _cache(tmp_path)
    artifact = _valid_artifact(tmp_path / "valid")
    output = root / "results" / "custom.json"
    assert exp.write_artifact(artifact, output) == output
    assert exp.validate_artifact(output) == []
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    class _Response:
        status = 200

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(exp, "urlopen", lambda *_args, **_kwargs: _Response())
    assert exp.http_reachable("https://example.com")
    monkeypatch.setattr(exp, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()))
    assert not exp.http_reachable("https://example.com")

    monkeypatch.setattr(exp, "find_repo_root", lambda: root)
    monkeypatch.setattr(exp, "probe_routes", _routes)
    monkeypatch.setattr(exp.time, "monotonic", iter((10.0, 11.5)).__next__)
    assert exp.main(
        ["--date", exp.RUN_DATE, "--output", str(output), "--cache-root", str(cache)]
    ) == 0
    stdout = capsys.readouterr().out
    assert '"v625_sota_ingestion_complete_score": 1' in stdout
    assert '"verdict_class": "positive"' in stdout

    assert exp.main(["--date", "20260908", "--output", str(output)]) == 2
    assert exp.main(["--date", "bad", "--output", str(output)]) == 2


def test_req_report_7122_defensive_source_and_filesystem_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7122 makes failed probes and missing cache candidates explicit."""

    empty_repo = tmp_path / "empty-repo"
    (empty_repo / "snapshots" / "revision").mkdir(parents=True)
    (empty_repo / "snapshots" / "revision" / "mmproj-F16.gguf").write_bytes(b"x")
    assert exp._preferred_cached_file(empty_repo) is None

    error = HTTPError("https://example.com", 403, "denied", {}, None)
    monkeypatch.setattr(exp, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))
    assert exp.http_reachable("https://example.com")

    seen: list[str] = []
    monkeypatch.setattr(exp, "http_reachable", lambda url: not seen.append(url))
    assert set(exp.probe_routes()) == set(exp.SOURCE_COLLECTION_FIELDS)
    assert len(seen) == len(exp.SOURCE_COLLECTION_FIELDS)

    class _BadRead:
        def is_file(self) -> bool:
            return True

        def read_bytes(self) -> bytes:
            raise OSError("unreadable")

    assert not exp._readable_nonempty(_BadRead())  # type: ignore[arg-type]
    assert not exp._writable_target(tmp_path / "missing-parent" / "artifact.json")
    writable = tmp_path / "writable"
    writable.mkdir()
    monkeypatch.setattr(
        exp.tempfile,
        "NamedTemporaryFile",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("not writable")),
    )
    assert not exp._writable_target(writable / "artifact.json")

    reference = _root(tmp_path / "missing-models") / exp.REFERENCE_PATH
    missing = exp.inspect_cached_models(_cache(tmp_path / "no-models", include_models=False))
    assert exp.append_references_delta(reference, exp.task_method_map_rows(), missing, exp.RUN_DATE)
    assert "no local language-model GGUF candidate resolved" in reference.read_text()


def test_req_report_7122_validation_rejects_every_derived_contract_branch(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7122-ARTIFACT exercises every rejection boundary."""

    artifact = _valid_artifact(tmp_path)
    mutations = []
    for field in exp.SOURCE_COLLECTION_FIELDS:
        forged = deepcopy(artifact)
        forged[field] = []
        mutations.append(forged)
    for field, value in (
        ("source_class_rows", "bad"),
        ("publication_date_rows", []),
        ("model_repository_rows", []),
        ("cached_model_rows", []),
        ("deferred_rows", []),
        ("reference_append_marker", {}),
    ):
        forged = deepcopy(artifact)
        forged[field] = value
        mutations.append(forged)

    forged = deepcopy(artifact)
    forged["source_class_rows"][0]["source_type"] = "unknown"
    mutations.append(forged)
    forged = deepcopy(artifact)
    forged["source_class_rows"][0]["claim_boundary"] = ""
    mutations.append(forged)
    forged = deepcopy(artifact)
    forged["publication_date_rows"][0]["in_window"] = False
    mutations.append(forged)
    forged = deepcopy(artifact)
    forged["model_repository_rows"][0]["revision"] = ""
    mutations.append(forged)
    forged = deepcopy(artifact)
    forged["cached_model_rows"][0]["download_performed"] = True
    mutations.append(forged)
    forged = deepcopy(artifact)
    forged["cached_model_rows"][0]["size_bytes"] = 0
    mutations.append(forged)
    forged = deepcopy(artifact)
    forged["cached_model_rows"][0].update(cache_status="missing", path="/forged")
    mutations.append(forged)
    forged = deepcopy(artifact)
    forged["cached_model_rows"][0]["cache_status"] = "unknown"
    mutations.append(forged)
    forged = deepcopy(artifact)
    forged["arxiv_rows"][0] = "bad"
    mutations.append(forged)

    assert all(exp._coverage_errors(item) for item in mutations)

    disqualified = deepcopy(artifact)
    disqualified["deferred_rows"] = []
    assert exp.recompute_artifact(disqualified)["verdict_class"] == "disqualified"


def test_req_report_7122_blocked_and_cli_error_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7122-PREFLIGHT and ARTIFACT reject forged terminal state."""

    root = _root(tmp_path)
    blocked = exp.build_artifact(
        root,
        exp.RUN_DATE,
        cache_root=_cache(tmp_path),
        output_path=Path("results/relative.json"),
        route_reachability={name: False for name in exp.SOURCE_COLLECTION_FIELDS},
    )
    blocked.update(
        gate_check_summary={},
        inference_substrate_class="aggregation",
        v625_sota_ingestion_complete_score=1,
        verdict_class="positive",
        honest_verdict="positive_forged",
    )
    assert len(exp._terminal_errors(blocked)) == 5
    missing_preconditions = deepcopy(blocked)
    missing_preconditions["preconditions_checked"] = []
    assert exp._terminal_errors(missing_preconditions) == ["preconditions_checked missing"]

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert exp.validate_artifact(bad_json) == ["artifact_unreadable"]
    assert exp.validate_artifact(list_json) == ["artifact_not_object"]
    with pytest.raises(ValueError):
        exp.write_artifact(blocked, tmp_path / "must-not-write.json")

    valid_path = tmp_path / "valid.json"
    exp.write_artifact(_valid_artifact(tmp_path / "valid"), valid_path)
    assert exp.main(["--validate", str(valid_path)]) == 0
    assert '"valid": true' in capsys.readouterr().out
    assert exp.main(["--validate", str(bad_json)]) == 1

    monkeypatch.setattr(exp, "find_repo_root", lambda: root)
    monkeypatch.setattr(exp, "probe_routes", _routes)
    monkeypatch.setattr(exp, "build_artifact", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(exp, "validate_artifact", lambda _value: ["forced-invalid"])
    assert exp.main(["--date", exp.RUN_DATE]) == 1
    assert "forced-invalid" in capsys.readouterr().out
