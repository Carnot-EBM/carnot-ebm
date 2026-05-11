"""Normalize malformed ROCE/HILED evidence into gate-ready Carnot artifacts.

Spec: REQ-REPORT-1877, SCENARIO-REPORT-1877.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
EXPERIMENT = "1877_artifact_contract_normalization"
TITLE = "ROCE/HILED Artifact Contract Normalization"
SCHEMA = "carnot.artifact_contract_normalization.v1"

DEFAULT_OUT_PATH = REPO_ROOT / "results" / "experiment_1877_artifact_contract_normalization.json"

SOURCE_FILES = {
    "roce": "experiment_1864_roce.json",
    "hiled": "experiment_1869_hiled.json",
}

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "honest_verdict",
    "gate_contract_normalization_ready",
    "roce_success_rate",
    "hiled_simulator_ready",
    "normalized_artifacts",
    "tests_run",
}


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_date() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%d")


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_roce_success_rate(payload: Mapping[str, Any]) -> float | None:
    explicit_rate = _coerce_float(payload.get("success_rate"))
    if explicit_rate is not None:
        return explicit_rate

    successes = _coerce_float(payload.get("successes"))
    dataset_size = _coerce_float(payload.get("dataset_size"))
    if successes is None or dataset_size is None or dataset_size <= 0.0:
        return None
    return successes / dataset_size


def _hiled_simulator_ready(payload: Mapping[str, Any]) -> bool:
    simulated_steps = _coerce_float(payload.get("simulated_steps"))
    enforcement_rate = _coerce_float(payload.get("constraint_enforcement_rate"))
    return bool(
        payload.get("hiled_enabled") is True
        and simulated_steps is not None
        and simulated_steps > 0.0
        and enforcement_rate is not None
        and enforcement_rate > 0.0
    )


def _standard_wrapper_base(*, experiment: int, title: str, ready: bool) -> dict[str, Any]:
    timestamp = _utc_now()
    status = "complete" if ready else "blocked"
    return {
        "experiment": experiment,
        "title": title,
        "run_date": _run_date(),
        "started_at": timestamp,
        "finished_at": timestamp,
        "duration_s": 0.0,
        "status": status,
        "honest_verdict": f"{status}: artifact_contract_normalized",
    }


def _with_schema(wrapper: dict[str, Any]) -> dict[str, Any]:
    wrapper["schema"] = sorted(wrapper.keys())
    return wrapper


def _normalize_roce(payload: Mapping[str, Any]) -> dict[str, Any]:
    success_rate = _extract_roce_success_rate(payload)
    ready = success_rate is not None
    wrapper = _standard_wrapper_base(
        experiment=1864,
        title="ROCE Open Constraint Elicitation Prototype (normalized)",
        ready=ready,
    )
    wrapper.update(
        {
            "source_experiment_id": "exp1864",
            "source_artifact_path": "results/experiment_1864_roce.json",
            "normalization_kind": "roce",
            "roce_success_rate": success_rate,
            "raw_metrics": copy.deepcopy(dict(payload)),
        }
    )
    return _with_schema(wrapper)


def _normalize_hiled(payload: Mapping[str, Any]) -> dict[str, Any]:
    ready = _hiled_simulator_ready(payload)
    wrapper = _standard_wrapper_base(
        experiment=1869,
        title="HILED Simulator Evidence (normalized)",
        ready=ready,
    )
    wrapper.update(
        {
            "source_experiment_id": "exp1869",
            "source_artifact_path": "results/experiment_1869_hiled.json",
            "normalization_kind": "hiled",
            "hiled_simulator_ready": ready,
            "raw_metrics": copy.deepcopy(dict(payload)),
        }
    )
    return _with_schema(wrapper)


def _load_sources(results_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    sources: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for source_id, filename in SOURCE_FILES.items():
        payload = _read_json(results_dir / filename)
        if payload is None:
            missing.append(f"results/{filename}")
        else:
            sources[source_id] = payload
    return sources, missing


def _source_inputs_read(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, bool]]:
    return {
        f"results/{filename}": {"exists": source_id in sources}
        for source_id, filename in SOURCE_FILES.items()
    }


def _normalize_sources(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if "roce" in sources:
        normalized.append(_normalize_roce(sources["roce"]))
    if "hiled" in sources:
        normalized.append(_normalize_hiled(sources["hiled"]))
    return normalized


def build_artifact(
    *,
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_paths: Sequence[str],
    tests_run: Sequence[str],
) -> dict[str, Any]:
    """REQ-REPORT-1877: build the terminal ROCE/HILED normalization artifact."""

    normalized_artifacts = _normalize_sources(sources)
    normalized_by_kind = {str(row.get("normalization_kind")): row for row in normalized_artifacts}
    roce_success_rate = normalized_by_kind.get("roce", {}).get("roce_success_rate")
    hiled_simulator_ready = bool(
        normalized_by_kind.get("hiled", {}).get("hiled_simulator_ready") is True
    )
    wrappers_have_gate_fields = all(
        "status" in row and "honest_verdict" in row for row in normalized_artifacts
    )

    blocked_reasons: list[str] = []
    if missing_source_paths:
        blocked_reasons.append("listed source artifacts are missing")
    if not isinstance(roce_success_rate, int | float):
        blocked_reasons.append("roce success-rate gate is not numeric")
    if not hiled_simulator_ready:
        blocked_reasons.append("hiled simulator gate is not ready")
    if len(normalized_artifacts) != len(SOURCE_FILES) or not wrappers_have_gate_fields:
        blocked_reasons.append("normalized wrappers do not expose standard gate fields")

    gate_contract_normalization_ready = not blocked_reasons
    status = "complete" if gate_contract_normalization_ready else "blocked"
    timestamp = _utc_now()

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "title": TITLE,
        "schema": SCHEMA,
        "run_date": _run_date(),
        "started_at": timestamp,
        "finished_at": timestamp,
        "duration_s": 0.0,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": status,
        "gate_contract_normalization_ready": gate_contract_normalization_ready,
        "roce_success_rate": roce_success_rate,
        "hiled_simulator_ready": hiled_simulator_ready,
        "normalized_artifacts": normalized_artifacts,
        "source_inputs_read": _source_inputs_read(sources),
        "blocked_reasons": blocked_reasons,
        "tests_run": list(tests_run),
    }
    if status == "complete":
        artifact["honest_verdict"] = "complete: roce_hiled_gate_contract_normalization_ready"
    else:
        artifact["honest_verdict"] = "blocked: " + "; ".join(blocked_reasons)
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-1877: persist a started marker before source evidence reads."""

    artifact: dict[str, Any] = {field: False for field in REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "experiment": EXPERIMENT,
            "title": TITLE,
            "schema": SCHEMA,
            "run_date": _run_date(),
            "started_at": _utc_now(),
            "finished_at": None,
            "duration_s": 0.0,
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "status": "in_progress",
            "honest_verdict": "in_progress",
            "normalized_artifacts": [],
            "tests_run": [],
        }
    )
    return _write_json(Path(out_path), artifact)


def run(
    *,
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    tests_run: Sequence[str] = (),
) -> dict[str, Any]:
    """SCENARIO-REPORT-1877: write in-progress and terminal normalized artifacts."""

    root_path = Path(root)
    out = Path(out_path)
    write_in_progress_artifact(out)
    sources, missing = _load_sources(root_path / "results")
    artifact = build_artifact(
        sources=sources,
        missing_source_paths=missing,
        tests_run=tests_run,
    )
    return _write_json(out, artifact)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--tests-run", action="append", default=[])
    args = parser.parse_args(argv)
    run(root=args.root, out_path=args.out, tests_run=args.tests_run)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
