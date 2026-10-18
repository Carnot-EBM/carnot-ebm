"""Exp 2849 local benchmark manifest materialization.

The downstream corpus experiments need to know whether their benchmark rows are
actually available before they spend a model call.  This module turns local
HuggingFace Arrow cache files into small, explicit JSONL manifests and records
the exact count, SHA256, and readiness status for each corpus.

Spec: REQ-BENCH-2849, SCENARIO-BENCH-2849.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OUTPUT_FILENAME = "experiment_2849_local_dataset_materialization_v1.json"
RUN_DATE = "20260522"
REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ORDER = ("mbpp", "humaneval", "truthfulqa", "halueval", "fever")
COUNT_TARGETS = {
    "mbpp": 100,
    "humaneval": 164,
    "truthfulqa": 200,
    "halueval": 500,
    "fever": 500,
}


class DatasetUnavailable(RuntimeError):
    """Raised when a benchmark dataset cannot be loaded from local assets."""


@dataclass(frozen=True)
class PreconditionCheck:
    """One startup precondition checked before manifest materialization."""

    resource: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "resource": self.resource,
            "available": self.available,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class LoadedDataset:
    """Rows and provenance for one locally loaded benchmark split."""

    rows: Sequence[Mapping[str, Any]]
    split_name: str
    source_path: Path
    source_name: str


@dataclass(frozen=True)
class ManifestConfig:
    """Runtime configuration for the Exp 2849 manifest builder."""

    repo_root: Path = REPO_ROOT
    manifest_dir: Path | None = None
    results_dir: Path | None = None
    run_date: str = RUN_DATE
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def output_dir(self) -> Path:
        return self.results_dir if self.results_dir is not None else self.repo_root / "results"

    def manifest_output_dir(self) -> Path:
        if self.manifest_dir is not None:
            return self.manifest_dir
        return self.repo_root / "data" / "eval_manifests"

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


DatasetLoader = Callable[[ManifestConfig, str], LoadedDataset]


def default_precondition_checks(config: ManifestConfig) -> list[PreconditionCheck]:
    checks = [
        PreconditionCheck("python_json_pathlib", True, "python ok"),
        _datasets_package_check(),
    ]
    try:
        (config.repo_root / "data").mkdir(parents=True, exist_ok=True)
        config.manifest_output_dir().mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - filesystem permission dependent.
        checks.append(PreconditionCheck("data_directory", False, f"{type(exc).__name__}: {exc}"))
    else:
        checks.append(PreconditionCheck("data_directory", True, str(config.manifest_output_dir())))
    checks.append(
        PreconditionCheck(
            "network_download",
            True,
            "not required before local cache search; no synthetic fallback is allowed",
        )
    )
    return checks


def _datasets_package_check() -> PreconditionCheck:
    if importlib.util.find_spec("datasets") is None:
        return PreconditionCheck(  # pragma: no cover - environment dependent.
            "datasets_package", False, "datasets package is not installed"
        )
    try:
        import datasets
    except Exception as exc:  # pragma: no cover - broken install dependent.
        return PreconditionCheck("datasets_package", False, f"{type(exc).__name__}: {exc}")
    return PreconditionCheck("datasets_package", True, str(getattr(datasets, "__version__", "ok")))


def build_artifact(
    config: ManifestConfig | None = None,
    *,
    dataset_loader: DatasetLoader | None = None,
    precondition_checks: Sequence[PreconditionCheck] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Build the Exp 2849 artifact and, when requested, write it to results/.

    The builder stops immediately on failed startup checks because those are
    environmental blockers.  Dataset-level misses are handled more narrowly:
    they get per-dataset `ready=false` entries so downstream tasks can gate
    only the corpus they need.
    """

    config = config or ManifestConfig()
    started_at = config.start_time()
    checks = list(precondition_checks or default_precondition_checks(config))
    failed_check = next((check for check in checks if not check.available), None)
    if failed_check is not None:
        artifact = _base_artifact(
            config=config,
            started_at=started_at,
            preconditions=checks,
            honest_verdict=f"blocked_{failed_check.resource}",
        )
        _write_artifact(config, artifact, write=write)
        return artifact

    loader = dataset_loader or load_local_dataset
    artifact = _base_artifact(
        config=config,
        started_at=started_at,
        preconditions=checks,
        honest_verdict="complete: local benchmark manifests materialized",
    )

    for dataset_key in DATASET_ORDER:
        try:
            loaded = loader(config, dataset_key)
            manifest_rows = materialize_rows(dataset_key, loaded)
        except DatasetUnavailable as exc:
            _record_dataset_failure(artifact, dataset_key, str(exc))
            continue
        except Exception as exc:
            _record_dataset_failure(artifact, dataset_key, f"{type(exc).__name__}: {exc}")
            continue

        manifest_path = config.manifest_output_dir() / f"{dataset_key}_{config.run_date}.jsonl"
        digest, count = _write_jsonl_manifest(manifest_path, manifest_rows, write=write)
        target = COUNT_TARGETS[dataset_key]
        ready = count >= target
        detail = (
            f"ready: {count} rows from {loaded.source_name}"
            if ready
            else f"loaded {count} rows from {loaded.source_name}; below target {target}"
        )
        artifact[f"{dataset_key}_ready"] = ready
        artifact["manifest_paths"][dataset_key] = str(manifest_path) if count else ""
        artifact["manifest_counts"][dataset_key] = count
        artifact["manifest_sha256"][dataset_key] = digest if count else ""
        artifact["dataset_status"][dataset_key] = {
            "ready": ready,
            "count": count,
            "target": target,
            "detail": detail,
            "split_name": loaded.split_name,
            "source_name": loaded.source_name,
            "source_path": str(loaded.source_path),
            "manifest_path": str(manifest_path) if count else "",
        }

    blocked_dataset = next((name for name in DATASET_ORDER if not artifact[f"{name}_ready"]), None)
    if blocked_dataset is not None:
        artifact["honest_verdict"] = f"blocked_{blocked_dataset}_dataset"
    artifact["duration_s"] = max(0.0, config.clock() - started_at)
    _write_artifact(config, artifact, write=write)
    return artifact


def _base_artifact(
    *,
    config: ManifestConfig,
    started_at: float,
    preconditions: Sequence[PreconditionCheck],
    honest_verdict: str,
) -> dict[str, Any]:
    return {
        "artifact": "experiment_2849_local_dataset_materialization_v1",
        "schema": "carnot.local_dataset_materialization.v1",
        "honest_verdict": honest_verdict,
        "mbpp_ready": False,
        "humaneval_ready": False,
        "truthfulqa_ready": False,
        "halueval_ready": False,
        "fever_ready": False,
        "dataset_status": {
            name: {
                "ready": False,
                "count": 0,
                "target": COUNT_TARGETS[name],
                "detail": "not checked",
                "split_name": "",
                "source_name": "",
                "source_path": "",
                "manifest_path": "",
            }
            for name in DATASET_ORDER
        },
        "manifest_paths": {name: "" for name in DATASET_ORDER},
        "manifest_counts": {name: 0 for name in DATASET_ORDER},
        "manifest_sha256": {name: "" for name in DATASET_ORDER},
        "preconditions_checked": [check.as_dict() for check in preconditions],
        "synthetic_rows_created": False,
        "duration_s": max(0.0, config.clock() - started_at),
        "run_date": config.run_date,
    }


def _record_dataset_failure(artifact: dict[str, Any], dataset_key: str, detail: str) -> None:
    artifact[f"{dataset_key}_ready"] = False
    artifact["dataset_status"][dataset_key] = {
        "ready": False,
        "count": 0,
        "target": COUNT_TARGETS[dataset_key],
        "detail": detail,
        "split_name": "",
        "source_name": "",
        "source_path": "",
        "manifest_path": "",
    }


def materialize_rows(dataset_key: str, loaded: LoadedDataset) -> list[dict[str, Any]]:
    if dataset_key == "mbpp":
        return [_mbpp_row(row, idx, loaded) for idx, row in enumerate(loaded.rows)]
    if dataset_key == "humaneval":
        return [_humaneval_row(row, idx, loaded) for idx, row in enumerate(loaded.rows)]
    if dataset_key == "truthfulqa":
        return [_truthfulqa_row(row, idx, loaded) for idx, row in enumerate(loaded.rows)]
    if dataset_key == "halueval":
        return _halueval_rows(loaded)
    if dataset_key == "fever":
        return [_fever_row(row, idx, loaded) for idx, row in enumerate(loaded.rows)]
    raise DatasetUnavailable(f"unknown dataset key: {dataset_key}")


def _common_fields(dataset: str, stable_id: str, loaded: LoadedDataset) -> dict[str, object]:
    return {
        "dataset": dataset,
        "stable_id": stable_id,
        "split_name": loaded.split_name,
        "source_name": loaded.source_name,
        "source_path": str(loaded.source_path),
    }


def _list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _mbpp_row(row: Mapping[str, Any], idx: int, loaded: LoadedDataset) -> dict[str, Any]:
    task_id = row.get("task_id", idx)
    return {
        **_common_fields("MBPP", f"mbpp-{task_id}", loaded),
        "prompt": str(row.get("prompt", "")),
        "tests": [str(item) for item in _list_value(row.get("test_list"))],
        "test_imports": [str(item) for item in _list_value(row.get("test_imports"))],
        "canonical_code": str(row.get("code", "")),
        "source_file": str(row.get("source_file", "")),
    }


def _humaneval_row(row: Mapping[str, Any], idx: int, loaded: LoadedDataset) -> dict[str, Any]:
    task_id = str(row.get("task_id") or f"HumanEval/{idx}")
    return {
        **_common_fields("HumanEval", task_id, loaded),
        "prompt": str(row.get("prompt", "")),
        "canonical_solution": str(row.get("canonical_solution", "")),
        "tests": str(row.get("test", "")),
        "entry_point": str(row.get("entry_point", "")),
    }


def _truthfulqa_row(row: Mapping[str, Any], idx: int, loaded: LoadedDataset) -> dict[str, Any]:
    return {
        **_common_fields("TruthfulQA", f"truthfulqa-{loaded.split_name}-{idx}", loaded),
        "question": str(row.get("question", "")),
        "best_answer": str(row.get("best_answer", "")),
        "correct_answers": [str(item) for item in _list_value(row.get("correct_answers"))],
        "incorrect_answers": [str(item) for item in _list_value(row.get("incorrect_answers"))],
        "category": str(row.get("category", "")),
        "type": str(row.get("type", "")),
        "reference_source": str(row.get("source", "")),
    }


def _halueval_rows(loaded: LoadedDataset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(loaded.rows):
        knowledge = str(row.get("knowledge") or row.get("document") or "").strip()
        question = str(row.get("question") or row.get("user_query") or "").strip()
        prompt = f"Context: {knowledge}\nQuestion: {question}".strip()
        right = row.get("right_answer") or row.get("right_response") or row.get("right_summary")
        hallucinated = (
            row.get("hallucinated_answer")
            or row.get("hallucinated_response")
            or row.get("hallucinated_summary")
        )
        if right:
            rows.append(
                {
                    **_common_fields("HaluEval", f"halueval-{idx}-right", loaded),
                    "prompt": prompt,
                    "candidate": str(right),
                    "label": 0,
                    "reference": str(right),
                }
            )
        if hallucinated:
            rows.append(
                {
                    **_common_fields("HaluEval", f"halueval-{idx}-hallucinated", loaded),
                    "prompt": prompt,
                    "candidate": str(hallucinated),
                    "label": 1,
                    "reference": str(right) if right else "",
                }
            )
    return rows


def _fever_row(row: Mapping[str, Any], idx: int, loaded: LoadedDataset) -> dict[str, Any]:
    label_text = str(row.get("label", "")).strip()
    return {
        **_common_fields("FEVER", f"fever-{row.get('id', idx)}", loaded),
        "prompt": str(row.get("evidence", "")),
        "claim": str(row.get("claim", "")),
        "label": _fever_label_to_int(label_text),
        "label_text": label_text,
        "verifiable": str(row.get("verifiable", "")),
    }


def _fever_label_to_int(label: str) -> int:
    normalized = label.upper().replace("_", " ")
    if normalized == "SUPPORTS":
        return 0
    return 1


def _write_jsonl_manifest(path: Path, rows: Sequence[Mapping[str, Any]], *, write: bool) -> tuple[str, int]:
    payload = "".join(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n" for row in rows
    )
    data = payload.encode("utf-8")
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return hashlib.sha256(data).hexdigest() if data else "", len(rows)


def _write_artifact(config: ManifestConfig, artifact: Mapping[str, Any], *, write: bool) -> None:
    if not write:
        return
    config.output_dir().mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir() / OUTPUT_FILENAME
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_local_dataset(config: ManifestConfig, dataset_key: str) -> LoadedDataset:
    source_path, split_name, source_name, row_limit = _dataset_spec(config, dataset_key)
    rows = _load_arrow_rows(source_path, row_limit)
    if not rows:
        raise DatasetUnavailable(f"{source_name} local asset had zero rows: {source_path}")
    return LoadedDataset(rows=rows, split_name=split_name, source_path=source_path, source_name=source_name)


def _dataset_spec(config: ManifestConfig, dataset_key: str) -> tuple[Path, str, str, int]:
    roots = [config.repo_root / "data", Path.home() / ".cache" / "huggingface" / "datasets"]
    if dataset_key == "mbpp":
        return (
            _find_first(
                roots,
                [
                    "**/*mbpp*/**/mbpp-test.arrow",
                    "**/*google-research-datasets___mbpp*/**/mbpp-test.arrow",
                ],
            ),
            "test",
            "google-research-datasets/mbpp:sanitized:test",
            COUNT_TARGETS["mbpp"],
        )
    if dataset_key == "humaneval":
        return (
            _find_first(roots, ["**/*humaneval*/**/openai_humaneval-test.arrow"]),
            "test",
            "openai_humaneval:test",
            COUNT_TARGETS["humaneval"],
        )
    if dataset_key == "truthfulqa":
        return (
            _find_first(roots, ["**/truthful_qa/generation/**/truthful_qa-validation.arrow"]),
            "validation",
            "truthful_qa:generation:validation",
            COUNT_TARGETS["truthfulqa"],
        )
    if dataset_key == "halueval":
        return (
            _find_first(
                roots,
                ["**/*halu_eval/qa/**/halu_eval-data.arrow", "**/*HaluEval*/qa/**/halu_eval-data.arrow"],
            ),
            "data",
            "pminervini/HaluEval:qa:data",
            (COUNT_TARGETS["halueval"] + 1) // 2,
        )
    if dataset_key == "fever":
        return (
            _find_first(roots, ["**/*fever*/**/fever-train.arrow"]),
            "train",
            "maxzoech/fever:train",
            COUNT_TARGETS["fever"],
        )
    raise DatasetUnavailable(f"unknown dataset key: {dataset_key}")


def _find_first(roots: Sequence[Path], patterns: Sequence[str]) -> Path:
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches.extend(path for path in root.glob(pattern) if path.is_file())
    if not matches:
        searched = ", ".join(str(root / pattern) for root in roots for pattern in patterns)
        raise DatasetUnavailable(f"missing local asset; searched {searched}")
    return sorted(matches, key=lambda path: str(path))[0]


def _load_arrow_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    try:
        from datasets import Dataset
    except Exception as exc:  # pragma: no cover - broken install dependent.
        raise DatasetUnavailable(f"datasets unavailable while reading {path}: {exc}") from exc
    dataset = Dataset.from_file(str(path))
    row_count = min(len(dataset), limit)
    return [dict(dataset[idx]) for idx in range(row_count)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    artifact = build_artifact(
        ManifestConfig(
            repo_root=args.repo_root,
            manifest_dir=args.manifest_dir,
            results_dir=args.results_dir,
        )
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if str(artifact["honest_verdict"]).startswith(("complete:", "success:")) else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
