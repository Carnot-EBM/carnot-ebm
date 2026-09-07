"""Tests for REQ-REPORT-7125 and SCENARIO-REPORT-7125-* contracts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from carnot import experiment_7125_v626_source_delta as exp


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _root(tmp_path: Path, *, planner_marker: bool = True) -> Path:
    """Create private inputs so tests cannot change the research record."""

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
    """Build a small Hugging Face cache with content-addressed files."""

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
        revision = repo / "snapshots" / f"revision-{index}"
        revision.mkdir(parents=True)
        (revision / file_name).symlink_to(Path("../..") / "blobs" / blob.name)
        (revision / "mmproj-F16.gguf").write_bytes(b"projector")
    return cache


def _routes(**changes: bool) -> dict[str, bool]:
    """Return a complete external-route fixture."""

    routes = {name: True for name in exp.SOURCE_COLLECTION_FIELDS}
    routes.update(changes)
    return routes


def _valid_artifact(tmp_path: Path) -> dict[str, object]:
    """Build one complete artifact under private paths."""

    root = _root(tmp_path)
    return exp.build_artifact(
        root,
        exp.RUN_DATE,
        cache_root=_cache(tmp_path),
        output_path=root / exp.RESULT_PATH,
        route_reachability=_routes(),
        update_references=True,
        duration_s=1.25,
    )


def test_req_report_7125_spec_precedes_implementation() -> None:
    """REQ-REPORT-7125 owns each required field and named scenario."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7125") :]
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
        assert f"SCENARIO-REPORT-7125-{scenario}" in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_7125_classify_keeps_execution_boundaries(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7125-CLASSIFY keeps every evidence class bounded."""

    artifact = _valid_artifact(tmp_path)
    classes = {row["source_type"] for row in artifact["source_class_rows"]}
    assert set(exp.SOURCE_TYPES).issubset(classes)
    assert all(exp.valid_direct_url(row["url"]) for row in artifact["source_class_rows"])
    assert all(row["terminal"] is True for row in artifact["source_class_rows"])
    assert all(row["claim_boundary"] for row in artifact["source_class_rows"])
    assert artifact["source_pages_are_execution_oracles"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["extropic_rows"][0]["claim_boundary_label"] == exp.VENDOR_BOUNDARY
    assert (
        artifact["logical_intelligence_rows"][0]["claim_boundary_label"]
        == exp.PRODUCT_BOUNDARY
    )


def test_scenario_report_7125_dedup_prefers_primary_source(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7125-DEDUP removes secondary copies."""

    rows = [
        {
            "source_id": "arxiv:2605.18871",
            "url": "https://huggingface.co/papers/2605.18871",
            "source_type": "secondary_discovery",
        },
        {
            "source_id": "arxiv:2605.18871",
            "url": "https://arxiv.org/abs/2605.18871",
            "source_type": "preprint",
        },
    ]
    assert exp.deduplicate_source_rows(rows) == [rows[1]]

    artifact = _valid_artifact(tmp_path)
    ids = [row["source_id"] for row in artifact["source_class_rows"]]
    assert len(ids) == len(set(ids))
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
def test_scenario_report_7125_dates_are_bounded(value: str | None, expected: bool) -> None:
    """SCENARIO-REPORT-7125-DATES applies the inclusive date bounds."""

    assert exp.publication_date_in_window(value, exp.RUN_DATE) is expected


def test_req_report_7125_url_fields_are_direct(tmp_path: Path) -> None:
    """REQ-REPORT-7125 rejects indirect or malformed URL fields."""

    artifact = _valid_artifact(tmp_path)
    for collection in exp.URL_COLLECTION_FIELDS:
        for row in artifact[collection]:
            for key, value in row.items():
                if key == "url" or key.endswith("_url"):
                    assert exp.valid_direct_url(value)
    for value in ("http://arxiv.org/abs/2605.18871", "not-a-url", None):
        assert not exp.valid_direct_url(value)

    forged = deepcopy(artifact)
    forged["arxiv_rows"][0]["query_url"] = "http://example.com"
    assert exp.validate_artifact(forged)


def test_scenario_report_7125_models_record_exact_cache_facts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7125-MODELS records candidates without downloads."""

    artifact = _valid_artifact(tmp_path)
    repositories = artifact["model_repository_rows"]
    assert [row["hf_id"] for row in repositories] == list(exp.MODEL_REPOSITORY_IDS)
    assert all(row["download_performed"] is False for row in repositories)
    assert all(row["repository_url"].startswith("https://huggingface.co/") for row in repositories)

    cached = artifact["cached_model_rows"]
    assert [row["hf_id"] for row in cached] == list(exp.MODEL_REPOSITORY_IDS)
    assert all(row["cache_status"] == "resolved" for row in cached)
    assert all(row["file_name"].endswith(".gguf") for row in cached)
    assert all("mmproj" not in row["file_name"].lower() for row in cached)
    assert all(row["quantization"] == "Q4_K_M" for row in cached)
    assert all(row["size_bytes"] > 0 for row in cached)
    assert all(str(row["sha256"]).startswith("sha256:") for row in cached)
    assert all(row["revision"].startswith("revision-") for row in cached)
    assert all(row["download_performed"] is False for row in cached)

    empty = exp.inspect_cached_models(_cache(tmp_path / "empty", include_models=False))
    assert len(empty) == 3
    assert all(row["cache_status"] == "missing" for row in empty)
    assert all(row["path"] is None and row["sha256"] is None for row in empty)


def test_scenario_report_7125_models_call_cached_sota_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7125-MODELS uses cached_sota_pair in production."""

    cache = _cache(tmp_path)
    paths: dict[str, str] = {}
    for hf_id in exp.MODEL_REPOSITORY_IDS:
        repo = cache / f"models--{hf_id.replace('/', '--')}" / "snapshots"
        paths[hf_id] = str(next(path for path in repo.rglob("*.gguf") if "mmproj" not in path.name))
    calls: list[tuple[int, int]] = []

    def fake_pair(*, model_indices: tuple[int, int], **_kwargs: object) -> list[dict[str, object]]:
        calls.append(model_indices)
        return [
            {"hf_id": exp.SOTA_INDEX_TO_ID[index], "model_path": paths[exp.SOTA_INDEX_TO_ID[index]]}
            for index in model_indices
        ]

    monkeypatch.setattr(exp, "cached_sota_pair", fake_pair)
    rows = exp.inspect_cached_models(exp.DEFAULT_CACHE_ROOT)
    assert calls == [(0, 2), (1, 0)]
    assert [row["hf_id"] for row in rows] == list(exp.MODEL_REPOSITORY_IDS)


def test_scenario_report_7125_map_is_exact_and_other_work_is_deferred(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7125-MAP limits promotion to five exact maps."""

    artifact = _valid_artifact(tmp_path)
    assert [row["target_task_id"] for row in artifact["task_method_map_rows"]] == [
        "exp7129-hardness-controlled-sota-constraint-bank",
        "exp7130-verifier-committed-uncertainty-routing",
        "exp7130-verifier-committed-uncertainty-routing",
        "exp7132-directional-memory-portability-audit",
        "exp7133-wcrg-multiscale-sampler-prototype",
    ]
    assert all(row["decision_changing"] is True for row in artifact["task_method_map_rows"])
    assert all(row["execution_delta_new"] is False for row in artifact["task_method_map_rows"])
    assert all(row["claim_boundary"] for row in artifact["task_method_map_rows"])
    assert all(row["carnot_result_claimed"] is False for row in artifact["task_method_map_rows"])
    assert artifact["deferred_rows"]
    assert all(row["reason"] and row["target_task_id"] is None for row in artifact["deferred_rows"])

    forged = deepcopy(artifact)
    forged["task_method_map_rows"].append(deepcopy(forged["task_method_map_rows"][0]))
    assert exp.validate_artifact(forged)


def test_scenario_report_7125_append_markers_are_idempotent(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7125-APPEND writes one complete marker pair."""

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
    assert "no post-planner method changed the V626 task contract" in text
    for row in exp.task_method_map_rows():
        assert row["target_task_id"] in text


@pytest.mark.parametrize(
    ("planner_marker", "cache_exists", "failed_check"),
    [
        (False, True, "v626_planner_marker"),
        (True, False, "model_cache_metadata"),
    ],
)
def test_scenario_report_7125_local_preflight_blocks_exactly(
    tmp_path: Path,
    planner_marker: bool,
    cache_exists: bool,
    failed_check: str,
) -> None:
    """SCENARIO-REPORT-7125-PREFLIGHT preserves the first local failure."""

    root = _root(tmp_path, planner_marker=planner_marker)
    cache = _cache(tmp_path) if cache_exists else tmp_path / "missing-cache"
    artifact = exp.build_artifact(
        root,
        exp.RUN_DATE,
        cache_root=cache,
        output_path=root / exp.RESULT_PATH,
        route_reachability=_routes(),
        update_references=True,
        duration_s=0.1,
    )
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["verdict_class"] == "blocked"
    assert str(artifact["honest_verdict"]).startswith("blocked_")
    assert artifact["gate_check_summary"]["failed_check"] == failed_check
    assert artifact["gate_check_summary"]["expected_value"] is not None
    assert artifact["gate_check_summary"]["observed_value"] is not None
    assert artifact["v626_source_delta_complete_score"] == 0
    assert exp.validate_artifact(artifact) == []


def test_scenario_report_7125_external_failure_is_terminal_row(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7125-PREFLIGHT does not block an external outage."""

    root = _root(tmp_path)
    artifact = exp.build_artifact(
        root,
        exp.RUN_DATE,
        cache_root=_cache(tmp_path),
        output_path=root / exp.RESULT_PATH,
        route_reachability=_routes(openreview_rows=False),
        update_references=True,
        duration_s=0.2,
    )
    assert artifact["v626_source_delta_complete_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert all(row["available"] is False for row in artifact["openreview_rows"])
    network = next(row for row in artifact["preconditions_checked"] if row["check"] == "network_access")
    assert network["passed"] is True
    assert "openreview_rows" in network["observed_value"]["unavailable"]
    assert exp.validate_artifact(artifact) == []


def test_scenario_report_7125_artifact_recomputes_state_and_checksum(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7125-ARTIFACT rejects forged derived fields."""

    artifact = _valid_artifact(tmp_path)
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["rows"] == exp.combined_rows(artifact)
    assert artifact["inference_substrate"].startswith(
        "aggregation_from_external_primary_sources: source and cache delta"
    )
    assert artifact["v626_source_delta_complete_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert str(artifact["honest_verdict"]).startswith("null_")
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
        ("source_pages_are_execution_oracles", True),
        ("v626_source_delta_complete_score", 0),
        ("random_seed", 0),
        ("gate_check_summary", {}),
        ("verifier_is_oracle", True),
        ("verdict_class", "partial"),
        ("honest_verdict", "positive_forged"),
        ("reproducibility_checksum", "sha256:bad"),
    ):
        forged = deepcopy(artifact)
        forged[field] = value
        assert exp.validate_artifact(forged), field

    missing = deepcopy(artifact)
    missing.pop("kan_rows")
    assert "artifact fields mismatch" in exp.validate_artifact(missing)[0]
    assert exp.validate_artifact([]) == ["artifact_not_object"]
    assert exp.validate_artifact(tmp_path / "missing.json") == ["artifact_missing"]


def test_req_report_7125_io_and_command_line_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-7125 covers storage, probes, and command output."""

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
    error = HTTPError("https://example.com", 403, "denied", {}, None)
    monkeypatch.setattr(exp, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))
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
    assert '"v626_source_delta_complete_score": 1' in stdout
    assert '"verdict_class": "null"' in stdout

    assert exp.main(["--date", "20260908", "--output", str(output)]) == 2
    assert exp.main(["--date", "bad", "--output", str(output)]) == 2

    assert exp.main(["--validate", str(output)]) == 0
    assert '"valid": true' in capsys.readouterr().out
    assert exp.main(["--validate", str(tmp_path / "missing.json")]) == 1

    monkeypatch.setattr(exp, "build_artifact", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(exp, "validate_artifact", lambda _value: ["forced-invalid"])
    assert exp.main(["--date", exp.RUN_DATE, "--output", str(output)]) == 1
    assert "forced-invalid" in capsys.readouterr().out


def test_req_report_7125_defensive_filesystem_and_validation_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7125 makes failed local checks and bad artifacts explicit."""

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
    monkeypatch.undo()

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert exp.validate_artifact(bad_json) == ["artifact_unreadable"]
    assert exp.validate_artifact(list_json) == ["artifact_not_object"]

    artifact = _valid_artifact(tmp_path / "valid")
    for field in exp.SOURCE_COLLECTION_FIELDS:
        forged = deepcopy(artifact)
        forged[field] = []
        assert exp.validate_artifact(forged), field

    disqualified = deepcopy(artifact)
    disqualified["deferred_rows"] = []
    assert exp.recompute_artifact(disqualified)["verdict_class"] == "disqualified"

    blocked = exp.build_artifact(
        _root(tmp_path / "blocked", planner_marker=False),
        exp.RUN_DATE,
        cache_root=_cache(tmp_path / "blocked"),
        output_path=Path("results/relative.json"),
        route_reachability=_routes(),
    )
    with pytest.raises(ValueError):
        exp.write_artifact({**blocked, "verdict_class": "positive"}, tmp_path / "no.json")


def test_req_report_7125_cache_and_append_defensive_branches(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7125-MODELS keeps missing files as explicit facts."""

    empty_repo = tmp_path / "empty-repo"
    (empty_repo / "snapshots" / "revision").mkdir(parents=True)
    (empty_repo / "snapshots" / "revision" / "mmproj-F16.gguf").write_bytes(b"x")
    assert exp._preferred_cached_file(empty_repo) is None
    assert exp._snapshot_revision(tmp_path / "model.gguf") is None

    root = _root(tmp_path / "append-missing")
    missing = exp.inspect_cached_models(_cache(tmp_path / "no-models", include_models=False))
    assert exp.append_references_delta(
        root / exp.REFERENCE_PATH,
        exp.task_method_map_rows(),
        missing,
        exp.RUN_DATE,
    )
    assert "no local language-model GGUF candidate resolved" in (
        root / exp.REFERENCE_PATH
    ).read_text(encoding="utf-8")


def test_req_report_7125_validation_rejects_each_contract_boundary(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7125-ARTIFACT exercises each rejection boundary."""

    artifact = _valid_artifact(tmp_path)
    mutations: list[tuple[dict[str, object], str]] = []

    forged = deepcopy(artifact)
    forged["arxiv_rows"][0]["available"] = "yes"
    mutations.append((forged, "incomplete source receipt"))
    forged = deepcopy(artifact)
    forged["arxiv_rows"][0] = "bad"
    mutations.append((forged, "direct URL field invalid"))
    forged = deepcopy(artifact)
    forged["source_class_rows"] = "bad"
    mutations.append((forged, "source class rows missing"))
    forged = deepcopy(artifact)
    forged["source_class_rows"] = [
        row for row in forged["source_class_rows"] if row["source_type"] != "vendor_claim"
    ]
    mutations.append((forged, "source type coverage mismatch"))
    forged = deepcopy(artifact)
    forged["source_class_rows"][0]["claim_boundary"] = ""
    mutations.append((forged, "source class boundary"))
    forged = deepcopy(artifact)
    forged["publication_date_rows"] = []
    mutations.append((forged, "publication date rows missing"))
    forged = deepcopy(artifact)
    forged["publication_date_rows"][0]["in_window"] = False
    mutations.append((forged, "scientific publication date outside window"))
    forged = deepcopy(artifact)
    forged["model_repository_rows"] = []
    mutations.append((forged, "model repository order mismatch"))
    forged = deepcopy(artifact)
    forged["model_repository_rows"][0]["revision"] = ""
    mutations.append((forged, "model repository metadata incomplete"))
    forged = deepcopy(artifact)
    forged["cached_model_rows"] = []
    mutations.append((forged, "cached model coverage mismatch"))
    forged = deepcopy(artifact)
    forged["cached_model_rows"][0]["download_performed"] = True
    mutations.append((forged, "cached model download boundary mismatch"))
    forged = deepcopy(artifact)
    forged["cached_model_rows"][0]["resolution_api"] = "unknown"
    mutations.append((forged, "cached model resolver mismatch"))
    forged = deepcopy(artifact)
    forged["cached_model_rows"][0]["size_bytes"] = 0
    mutations.append((forged, "resolved cached model facts incomplete"))
    forged = deepcopy(artifact)
    forged["cached_model_rows"][0].update(cache_status="missing", path="/forged")
    mutations.append((forged, "missing cached model row has file facts"))
    forged = deepcopy(artifact)
    forged["cached_model_rows"][0]["cache_status"] = "unknown"
    mutations.append((forged, "cached model status invalid"))
    forged = deepcopy(artifact)
    forged["reference_append_marker"] = {}
    mutations.append((forged, "reference append marker mismatch"))

    for forged, expected in mutations:
        assert any(expected in error for error in exp._coverage_errors(forged)), expected

    assert exp._terminal_errors({**artifact, "preconditions_checked": []}) == [
        "preconditions_checked missing"
    ]
    blocked = exp.build_artifact(
        _root(tmp_path / "blocked", planner_marker=False),
        exp.RUN_DATE,
        cache_root=_cache(tmp_path / "blocked"),
        output_path=Path("results/relative.json"),
        route_reachability=_routes(),
    )
    blocked.update(
        gate_check_summary={},
        inference_substrate_class="aggregation",
        v626_source_delta_complete_score=1,
        verdict_class="positive",
        honest_verdict="positive_forged",
    )
    assert len(exp._terminal_errors(blocked)) == 5

    forged = deepcopy(artifact)
    forged["verdict_class"] = "unknown"
    assert "verdict_class invalid" in exp.validate_artifact(forged)


def test_req_report_7125_probe_routes_calls_each_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7125 probes each named external surface once."""

    seen: list[str] = []
    monkeypatch.setattr(exp, "http_reachable", lambda url: not seen.append(url))
    assert set(exp.probe_routes()) == set(exp.SOURCE_COLLECTION_FIELDS)
    assert len(seen) == len(exp.SOURCE_COLLECTION_FIELDS)
