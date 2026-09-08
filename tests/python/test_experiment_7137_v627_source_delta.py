"""Tests for REQ-REPORT-7137 and SCENARIO-REPORT-7137-* contracts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from carnot import experiment_7137_v627_source_delta as exp


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _root(tmp_path: Path, *, planner_marker: bool = True) -> Path:
    """Create private inputs so tests never change the real research ledger."""

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
    """Create a small cache that has the same snapshot layout as Hugging Face."""

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
    """Return an explicit result for every external source route."""

    routes = {name: True for name in exp.SOURCE_COLLECTION_FIELDS}
    routes.update(changes)
    return routes


def _valid_artifact(tmp_path: Path) -> dict[str, object]:
    """Build one complete null artifact under private paths."""

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


def test_req_report_7137_spec_precedes_implementation() -> None:
    """REQ-REPORT-7137 owns all required fields and focused scenarios."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7137") :]
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
        assert f"SCENARIO-REPORT-7137-{scenario}" in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_7137_classify_preserves_source_boundaries(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7137-CLASSIFY keeps receipts below execution evidence."""

    artifact = _valid_artifact(tmp_path)
    classes = {row["source_type"] for row in artifact["source_class_rows"]}
    assert set(exp.SOURCE_TYPES).issubset(classes)
    assert all(row["claim_boundary"] for row in artifact["source_class_rows"])
    assert all(row["terminal"] is True for row in artifact["source_class_rows"])
    assert artifact["source_pages_are_execution_oracles"] is False
    assert artifact["verifier_is_oracle"] is False

    unavailable = exp.source_collections(_routes(huggingface_rows=False))
    assert all(row["available"] is False for row in unavailable["huggingface_rows"])
    assert all(
        row["access_outcome"] == "unavailable_at_execution"
        for row in unavailable["huggingface_rows"]
    )


def test_scenario_report_7137_dedup_prefers_primary_source(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7137-DEDUP removes weaker duplicate discovery rows."""

    rows = [
        {
            "source_id": "arxiv:2609.03241",
            "url": "https://huggingface.co/papers/2609.03241",
            "source_type": "secondary_discovery",
        },
        {
            "source_id": "arxiv:2609.03241",
            "url": "https://arxiv.org/abs/2609.03241",
            "source_type": "preprint",
        },
    ]
    assert exp.deduplicate_source_rows(rows) == [rows[1]]
    artifact = _valid_artifact(tmp_path)
    identities = [row["source_id"] for row in artifact["source_class_rows"]]
    assert len(identities) == len(set(identities))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2025-01-01", True),
        ("2026-09-08", True),
        ("2024-12-31", False),
        ("2026-09-09", False),
        ("bad", False),
        (None, False),
    ],
)
def test_scenario_report_7137_dates_are_bounded(value: str | None, expected: bool) -> None:
    """SCENARIO-REPORT-7137-DATES enforces the fixed inclusive source window."""

    assert exp.publication_date_in_window(value, exp.RUN_DATE) is expected


def test_scenario_report_7137_models_record_exact_cache_facts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7137-MODELS records candidates without downloads."""

    cached = exp.inspect_cached_models(_cache(tmp_path))
    assert [row["hf_id"] for row in cached] == list(exp.MODEL_REPOSITORY_IDS)
    assert all(row["cache_status"] == "resolved" for row in cached)
    assert all(row["candidate_path"] == row["path"] for row in cached)
    assert all(row["quantization"] == "Q4_K_M" for row in cached)
    assert all(row["size_bytes"] > 0 for row in cached)
    assert all(str(row["sha256"]).startswith("sha256:") for row in cached)
    assert all(row["revision"].startswith("revision-") for row in cached)
    assert all(row["download_performed"] is False for row in cached)

    missing = exp.inspect_cached_models(_cache(tmp_path / "empty", include_models=False))
    assert len(missing) == 3
    assert all(row["cache_status"] == "missing" for row in missing)
    assert all(row["candidate_path"] is None and row["sha256"] is None for row in missing)


def test_scenario_report_7137_models_call_cached_sota_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-7137-MODELS calls the mandated production resolver."""

    calls: list[tuple[int, int]] = []

    def fake_pair(*, model_indices: tuple[int, int], **_kwargs: object) -> None:
        calls.append(model_indices)
        return None

    monkeypatch.setattr(exp.base, "cached_sota_pair", fake_pair)
    exp.inspect_cached_models(exp.DEFAULT_CACHE_ROOT)
    assert calls == [(0, 2), (1, 0)]


def test_scenario_report_7137_map_is_exact_and_other_work_is_deferred() -> None:
    """SCENARIO-REPORT-7137-MAP assigns five methods to five exact tasks."""

    rows = exp.task_method_map_rows()
    assert [row["target_task_id"] for row in rows] == list(exp.TARGET_TASK_IDS)
    assert len({row["method_id"] for row in rows}) == 5
    assert all(row["decision_changing"] is True for row in rows)
    assert all(row["execution_delta_new"] is False for row in rows)
    assert all(row["carnot_result_claimed"] is False for row in rows)
    assert exp.deferred_rows()
    assert all(row["target_task_id"] is None and row["reason"] for row in exp.deferred_rows())


def test_scenario_report_7137_append_is_idempotent(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7137-APPEND writes one stable marker pair."""

    path = _root(tmp_path) / exp.REFERENCE_PATH
    cached = exp.inspect_cached_models(_cache(tmp_path))
    first = exp.append_references_delta(path, exp.task_method_map_rows(), cached, exp.RUN_DATE, [])
    once = path.read_bytes()
    second = exp.append_references_delta(path, exp.task_method_map_rows(), cached, exp.RUN_DATE, [])
    text = path.read_text(encoding="utf-8")
    assert first is True
    assert second is False
    assert path.read_bytes() == once
    assert text.count(exp.REFERENCE_START_MARKER) == 1
    assert text.count(exp.REFERENCE_END_MARKER) == 1
    assert "no post-planner method changed the V627 task contract" in text
    for row in exp.task_method_map_rows():
        assert row["target_task_id"] in text


def test_scenario_report_7137_preflight_blocks_only_local_failures(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7137-PREFLIGHT blocks locally and records site failure."""

    blocked_root = _root(tmp_path / "blocked", planner_marker=False)
    blocked = exp.build_artifact(
        blocked_root,
        exp.RUN_DATE,
        cache_root=_cache(tmp_path / "blocked"),
        output_path=blocked_root / exp.RESULT_PATH,
        route_reachability=_routes(),
        duration_s=0.1,
    )
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["failed_check"] == "v627_planner_marker"
    assert exp.validate_artifact(blocked) == []

    root = _root(tmp_path / "external")
    artifact = exp.build_artifact(
        root,
        exp.RUN_DATE,
        cache_root=_cache(tmp_path / "external"),
        output_path=root / exp.RESULT_PATH,
        route_reachability=_routes(openreview_rows=False),
        update_references=True,
        duration_s=0.2,
    )
    assert artifact["verdict_class"] == "null"
    assert all(row["available"] is False for row in artifact["openreview_rows"])
    assert exp.validate_artifact(artifact) == []


def test_scenario_report_7137_artifact_recomputes_and_cli_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-7137-ARTIFACT rejects derived-state changes."""

    artifact = _valid_artifact(tmp_path / "artifact")
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["rows"] == exp.combined_rows(artifact)
    assert artifact["v627_source_delta_complete_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["reproducibility_checksum"] == exp.payload_checksum(artifact)
    assert exp.validate_artifact(artifact) == []

    for field, value in (
        ("rows", []),
        ("field_principles", {}),
        ("inference_substrate", "model inference"),
        ("inference_substrate_class", "model_bounded_generation"),
        ("execution_venue", "gpu"),
        ("duration_s", -1),
        ("source_pages_are_execution_oracles", True),
        ("v627_source_delta_complete_score", 0),
        ("random_seed", 0),
        ("verifier_is_oracle", True),
        ("verdict_class", "partial"),
        ("honest_verdict", "positive_forged"),
        ("reproducibility_checksum", "sha256:bad"),
    ):
        forged = deepcopy(artifact)
        forged[field] = value
        assert exp.validate_artifact(forged), field

    output = tmp_path / "written.json"
    assert exp.write_artifact(artifact, output) == output
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert exp.validate_artifact(output) == []
    assert exp.validate_artifact(tmp_path / "missing.json") == ["artifact_missing"]

    root = _root(tmp_path / "cli")
    monkeypatch.setattr(exp, "find_repo_root", lambda: root)
    monkeypatch.setattr(exp, "probe_routes", _routes)
    monkeypatch.setattr(exp, "DEFAULT_CACHE_ROOT", _cache(tmp_path / "cli"))
    assert exp.main(["--date", exp.RUN_DATE]) == 0
    assert '"v627_source_delta_complete_score": 1' in capsys.readouterr().out
    assert exp.main(["--date", "bad"]) == 2
    assert exp.main(["--date", "20260909"]) == 2
    assert exp.main(["--validate", str(root / exp.RESULT_PATH)]) == 0


def test_req_report_7137_schema_checkpoint_precedes_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7137 writes the complete blocked shape before route checks."""

    root = _root(tmp_path)
    output = root / exp.RESULT_PATH
    observed: list[dict[str, object]] = []
    real_write = exp._checkpoint_artifact

    def capture(artifact: dict[str, object], path: Path) -> Path:
        observed.append(deepcopy(artifact))
        return real_write(artifact, path)

    def probe() -> dict[str, bool]:
        assert observed
        assert set(observed[0]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
        assert observed[0]["inference_substrate_class"] == "blocked_no_run"
        return _routes()

    monkeypatch.setattr(exp, "_checkpoint_artifact", capture)
    monkeypatch.setattr(exp, "probe_routes", probe)
    exp.build_artifact(
        root,
        exp.RUN_DATE,
        cache_root=_cache(tmp_path),
        output_path=output,
        duration_s=0.1,
    )
    assert len(observed) >= 2
