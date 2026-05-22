"""Tests for Exp 2849 local evaluation dataset materialization.

Spec: REQ-BENCH-2849, SCENARIO-BENCH-2849.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import local_dataset_materialization as mod


def _rows(count: int, **fields: Any) -> list[dict[str, Any]]:
    return [{"idx": idx, **fields} for idx in range(count)]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_req_bench_2849_materializes_manifests_and_artifact_schema(tmp_path: Path) -> None:
    """REQ-BENCH-2849: ready local corpora write JSONL manifests, counts, and digests."""

    source = tmp_path / "cache" / "dataset.arrow"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"local arrow")
    fixture_rows = {
        "mbpp": mod.LoadedDataset(
            rows=_rows(
                101,
                task_id=11,
                prompt="Write a function.",
                code="def f(): return 1",
                test_list=["assert f() == 1"],
                test_imports=[],
            ),
            split_name="test",
            source_path=source,
            source_name="fixture/mbpp",
        ),
        "humaneval": mod.LoadedDataset(
            rows=_rows(
                164,
                task_id="HumanEval/0",
                prompt="def f():",
                canonical_solution="    return 1",
                test="def check(candidate): assert candidate() == 1",
                entry_point="f",
            ),
            split_name="test",
            source_path=source,
            source_name="fixture/humaneval",
        ),
        "truthfulqa": mod.LoadedDataset(
            rows=_rows(
                201,
                question="What is true?",
                best_answer="This answer.",
                correct_answers=["This answer."],
                incorrect_answers=["That answer."],
                category="fixture",
                type="Adversarial",
                source="fixture source",
            ),
            split_name="validation",
            source_path=source,
            source_name="fixture/truthfulqa",
        ),
        "halueval": mod.LoadedDataset(
            rows=_rows(
                260,
                knowledge="Known fact.",
                question="Question?",
                right_answer="Grounded.",
                hallucinated_answer="Unsupported.",
            ),
            split_name="data",
            source_path=source,
            source_name="fixture/halueval",
        ),
        "fever": mod.LoadedDataset(
            rows=_rows(
                501,
                id=1,
                claim="Claim.",
                evidence="Evidence.",
                label="SUPPORTS",
                verifiable="VERIFIABLE",
            ),
            split_name="train",
            source_path=source,
            source_name="fixture/fever",
        ),
    }

    artifact = mod.build_artifact(
        mod.ManifestConfig(
            repo_root=tmp_path,
            manifest_dir=tmp_path / "data" / "eval_manifests",
            results_dir=tmp_path / "results",
            started_at=2.0,
            clock=lambda: 6.5,
        ),
        dataset_loader=lambda _config, dataset_key: fixture_rows[dataset_key],
        write=True,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["run_date"] == "20260522"
    assert artifact["synthetic_rows_created"] is False
    assert artifact["duration_s"] == pytest.approx(4.5)
    assert artifact["manifest_counts"] == {
        "mbpp": 101,
        "humaneval": 164,
        "truthfulqa": 201,
        "halueval": 520,
        "fever": 501,
    }
    assert all(artifact[f"{name}_ready"] for name in mod.DATASET_ORDER)

    mbpp_manifest = Path(artifact["manifest_paths"]["mbpp"])
    mbpp_rows = _load_jsonl(mbpp_manifest)
    assert mbpp_rows[0]["stable_id"] == "mbpp-11"
    assert mbpp_rows[0]["prompt"] == "Write a function."
    assert mbpp_rows[0]["tests"] == ["assert f() == 1"]
    assert mbpp_rows[0]["split_name"] == "test"
    assert mbpp_rows[0]["source_path"] == str(source)
    digest = hashlib.sha256(mbpp_manifest.read_bytes()).hexdigest()
    assert artifact["manifest_sha256"]["mbpp"] == digest

    halueval_rows = _load_jsonl(Path(artifact["manifest_paths"]["halueval"]))
    assert {halueval_rows[0]["label"], halueval_rows[1]["label"]} == {0, 1}
    assert halueval_rows[1]["candidate"] == "Unsupported."

    saved = json.loads(
        (tmp_path / "results" / mod.OUTPUT_FILENAME).read_text(encoding="utf-8")
    )
    assert saved == artifact


def test_scenario_bench_2849_marks_short_local_dataset_not_ready(tmp_path: Path) -> None:
    """SCENARIO-BENCH-2849: short local corpora block honestly without synthetic rows."""

    source = tmp_path / "cache" / "short.arrow"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"short")

    def loader(_config: mod.ManifestConfig, dataset_key: str) -> mod.LoadedDataset:
        if dataset_key == "humaneval":
            return mod.LoadedDataset(
                rows=_rows(
                    2,
                    task_id="HumanEval/0",
                    prompt="def f():",
                    canonical_solution="    return 1",
                    test="def check(candidate): assert candidate() == 1",
                    entry_point="f",
                ),
                split_name="test",
                source_path=source,
                source_name="fixture/humaneval",
            )
        raise mod.DatasetUnavailable("missing fixture resource")

    artifact = mod.build_artifact(
        mod.ManifestConfig(
            repo_root=tmp_path,
            manifest_dir=tmp_path / "data" / "eval_manifests",
            results_dir=tmp_path / "results",
        ),
        dataset_loader=loader,
        write=False,
    )

    assert artifact["honest_verdict"] == "blocked_mbpp_dataset"
    assert artifact["synthetic_rows_created"] is False
    assert artifact["humaneval_ready"] is False
    assert artifact["manifest_counts"]["humaneval"] == 2
    assert "below target 164" in artifact["dataset_status"]["humaneval"]["detail"]
    assert artifact["dataset_status"]["mbpp"]["detail"] == "missing fixture resource"
    assert artifact["manifest_paths"]["mbpp"] == ""
    assert artifact["manifest_sha256"]["mbpp"] == ""


def test_req_bench_2849_precondition_failure_writes_blocked_artifact(tmp_path: Path) -> None:
    """REQ-BENCH-2849: failed startup checks exit with blocked_<resource> verdict."""

    artifact = mod.build_artifact(
        mod.ManifestConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        precondition_checks=[
            mod.PreconditionCheck("python_json_pathlib", True, "python ok"),
            mod.PreconditionCheck("datasets_package", False, "ModuleNotFoundError"),
        ],
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_datasets_package"
    assert artifact["preconditions_checked"][1] == {
        "resource": "datasets_package",
        "available": False,
        "detail": "ModuleNotFoundError",
    }
    assert artifact["manifest_counts"] == {name: 0 for name in mod.DATASET_ORDER}
    saved = json.loads(
        (tmp_path / "results" / mod.OUTPUT_FILENAME).read_text(encoding="utf-8")
    )
    assert saved == artifact


def test_req_bench_2849_default_loader_finds_local_cache_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-BENCH-2849: default discovery prefers local files for every corpus."""

    files = {
        "mbpp": tmp_path / "data" / "local_mbpp" / "v" / "mbpp-test.arrow",
        "humaneval": tmp_path / "data" / "local_humaneval" / "v" / "openai_humaneval-test.arrow",
        "truthfulqa": (
            tmp_path / "data" / "truthful_qa" / "generation" / "v" / "truthful_qa-validation.arrow"
        ),
        "halueval": tmp_path / "data" / "pminervini___halu_eval" / "qa" / "v" / "halu_eval-data.arrow",
        "fever": tmp_path / "data" / "maxzoech___fever" / "default" / "v" / "fever-train.arrow",
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"arrow")

    def fake_arrow_rows(path: Path, limit: int) -> list[dict[str, Any]]:
        return [{"path": str(path), "limit": limit}]

    monkeypatch.setattr(mod, "_load_arrow_rows", fake_arrow_rows)
    monkeypatch.setattr(mod.Path, "home", staticmethod(lambda: tmp_path / "no-home"))

    config = mod.ManifestConfig(repo_root=tmp_path)
    assert config.manifest_output_dir() == tmp_path / "data" / "eval_manifests"
    for dataset_key, path in files.items():
        loaded = mod.load_local_dataset(config, dataset_key)
        assert loaded.source_path == path
        assert loaded.rows == [{"path": str(path), "limit": mod.COUNT_TARGETS[dataset_key]}] or (
            dataset_key == "halueval"
            and loaded.rows == [{"path": str(path), "limit": mod.COUNT_TARGETS[dataset_key] // 2}]
        )

    monkeypatch.setattr(mod, "_load_arrow_rows", lambda _path, _limit: [])
    with pytest.raises(mod.DatasetUnavailable, match="zero rows"):
        mod.load_local_dataset(config, "mbpp")
    with pytest.raises(mod.DatasetUnavailable, match="unknown dataset key"):
        mod.load_local_dataset(config, "unknown")
    with pytest.raises(mod.DatasetUnavailable, match="missing local asset"):
        mod._find_first([tmp_path / "missing"], ["*.arrow"])


def test_req_bench_2849_real_arrow_reader_and_edge_normalization(tmp_path: Path) -> None:
    """REQ-BENCH-2849: Arrow rows and edge labels normalize deterministically."""

    from datasets import Dataset

    dataset_dir = tmp_path / "arrow_ds"
    Dataset.from_list([{"value": "a"}, {"value": "b"}]).save_to_disk(str(dataset_dir))
    arrow_path = next(dataset_dir.glob("*.arrow"))

    assert mod._load_arrow_rows(arrow_path, 1) == [{"value": "a"}]
    assert mod._list_value(None) == []
    assert mod._list_value(("x", "y")) == ["x", "y"]
    assert mod._list_value("z") == ["z"]
    assert mod._fever_label_to_int("REFUTES") == 1

    loaded = mod.LoadedDataset(
        rows=[{"idx": 0}],
        split_name="test",
        source_path=arrow_path,
        source_name="fixture",
    )
    with pytest.raises(mod.DatasetUnavailable, match="unknown dataset key"):
        mod.materialize_rows("unknown", loaded)


def test_req_bench_2849_generic_loader_error_and_cli_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-BENCH-2849: generic loader failures are recorded and CLI reports status."""

    def broken_loader(_config: mod.ManifestConfig, dataset_key: str) -> mod.LoadedDataset:
        if dataset_key == "mbpp":
            raise ValueError("bad local file")
        raise mod.DatasetUnavailable("missing")

    artifact = mod.build_artifact(
        mod.ManifestConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        dataset_loader=broken_loader,
        write=False,
    )

    assert artifact["honest_verdict"] == "blocked_mbpp_dataset"
    assert artifact["dataset_status"]["mbpp"]["detail"] == "ValueError: bad local file"

    monkeypatch.setattr(
        mod,
        "build_artifact",
        lambda _config: {"honest_verdict": "complete: cli fixture"},
    )
    assert mod.main(["--repo-root", str(tmp_path), "--results-dir", str(tmp_path / "out")]) == 0
    assert "complete: cli fixture" in capsys.readouterr().out

    monkeypatch.setattr(mod, "build_artifact", lambda _config: {"honest_verdict": "blocked_cli"})
    assert mod.main(["--repo-root", str(tmp_path)]) == 1
