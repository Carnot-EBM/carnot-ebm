"""REQ-VERIFY-8165: qualify frozen methods while preserving historical failure.

The V705 implementation remains the numerical authority. This adapter binds
its exact protocol bytes and prior disposition to a new qualification receipt;
private target success gives no credit for natural or independent learning.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import admission_horizon_methods_8152 as legacy

Json = dict[str, Any]
ROOT = legacy.ROOT
NAME = "experiment_8165_v706_learning_qualification"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = "python/carnot/verify/learning_qualification_8165.py"
RUNNER = "python/carnot/reporting/learning_qualification_execution_8165.py"
TEST = "tests/python/test_learning_qualification_8165.py"
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real phase counts so observers can distinguish waiting from completion."""
    print(f"[exp8165] phase={phase} completed={completed} pending={pending}", flush=True)


def measure(root: Path, raw: Path, *, fixture_mode: bool = False) -> Json:
    """Authenticate the prior failure before reusing the unchanged private benchmark.

    Historical readiness is deliberately not a prerequisite: qualification is
    meant to repair coverage, while preserving the old disqualification bytes.
    Missing inputs and retirement remain terminal external gate failures.
    """
    started = time.monotonic()
    progress("before_preconditions")
    binder = legacy.engine.methods.Custody(raw / "citation_inputs")
    cited, predecessor = [], {}
    try:
        exclusion = root / "ops/exclusion_manifest.yaml"
        binder.bind(exclusion)
        retired = yaml.safe_load(exclusion.read_text()).get("retired_experiments", [])
        binder.require(
            exclusion,
            "experiment8165_not_retired",
            True,
            not any(r.get("experiment_id") == 8165 for r in retired),
        )
        for n, name, fields in [
            (8152, legacy.NAME, ["honest_verdict", "verdict_class", "validation_receipts"]),
            (
                8111,
                "experiment_8111_v702_methods_and_stream_custody",
                ["original_slot_mask", "methods_ready_score", "stream_input_ready_score"],
            ),
        ]:
            path = root / "results" / (name + ".json")
            value = binder.read(path)
            cited.append(dict(experiment_id=n, fields_imported=fields, sha256=sha256_file(path)))
            if n == 8152:
                binder.require(
                    path, "predecessor_verdict_class", "disqualified", value.get("verdict_class")
                )
                predecessor = dict(
                    path=str(path),
                    sha256=sha256_file(path),
                    verdict_class=value["verdict_class"],
                    failed_checks=[
                        r["name"] for r in value["validation_receipts"] if not r["passed"]
                    ],
                )
    except (OSError, ValueError, KeyError, TypeError):
        pass
    progress("after_preconditions", len(cited), 2 - len(cited))
    work = legacy.measure(root, raw, fixture_mode=fixture_mode)
    work["gate_check_summary"].extend(
        binder.checks + [r for r in binder.failures if r not in binder.checks]
    )
    work["input_ready"] = int(
        work["input_ready"] and all(r["passed"] for r in work["gate_check_summary"])
    )
    work["source_artifact_hashes"].extend(binder.refs)
    work["cited_upstream_artifacts"] = cited
    work["predecessor_disposition"] = predecessor
    protocol_path = Path(work["protocol_path"])
    protocol_path.write_bytes((ROOT / legacy.PROTOCOL).read_bytes())
    work["protocol_sha256"] = sha256_file(protocol_path)
    for ref in work["raw_shard_hashes"]:
        if ref["path"] == str(protocol_path):
            ref["sha256"] = work["protocol_sha256"]
    work["code_config_hashes"].update(
        {p: sha256_file(ROOT / p) for p in [MODULE, RUNNER, CLI, TEST]}
    )
    work["duration_s"] = time.monotonic() - started
    work["phase_spans"].append(
        dict(
            phase="qualification_citation_custody",
            duration_s=work["duration_s"] - sum(r["duration_s"] for r in work["phase_spans"]),
        )
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_normal_exit", len(work["event_reference_rows"]))
    return work


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Use the existing readiness equations, then name this independent receipt.

    The receipt shares fixture primitives and numerical rules with V705; the
    new identity records current validation without rewriting prior outcomes.
    """
    value = legacy.build(work, raw, receipts)
    value.pop("reproducibility_checksum")
    value.update(
        experiment_id=8165,
        task_id="exp8165-learning-qualification",
        milestone="2026.10.706",
        fixture_rows=work["event_reference_rows"],
    )
    value["field_principles"].update(
        {
            k: "New qualification preserves the frozen protocol and historical disqualification; fixture credit is circular."
            for k in ["fixture_rows", "predecessor_disposition", "cited_upstream_artifacts"]
        }
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Require byte custody and receipt identity as well as independent scalar replay.

    The existing replay checks reject forged rows even when someone computes a
    fresh outer checksum. These extra bindings prevent an old identity or a
    rewritten protocol file from being presented as the new qualification.
    """
    try:
        value = json.loads(path.read_text())
        return bool(
            value["experiment_id"] == 8165
            and value["task_id"] == "exp8165-learning-qualification"
            and value["fixture_rows"] == value["event_reference_rows"]
            and value["protocol_sha256"] == legacy.PROTOCOL_HASH
            and Path(value["protocol_path"]).read_bytes() == (ROOT / legacy.PROTOCOL).read_bytes()
            and legacy.replay(path)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
