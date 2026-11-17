"""REQ-REPORT-9166-V781: administrative readiness with no current model activity."""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v779_contract as legacy
from carnot.reporting import v781_contract_evidence as e
from carnot.reporting.current_work_receipt import atomic_json as atomic_json
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
_measure, _build, _replay = legacy.measure, legacy.build, legacy.replay


@contextmanager
def binding() -> Iterator[None]:
    """Reuse the existing checked lifecycle under this producer's exact owned paths."""
    with patch.multiple(legacy, e=e, measure=measure, build=build, replay=replay):
        yield


def measure(root: Path, raw: Path, date: str) -> Json:
    """Seal immutable protocols and resource receipts before any validation outcomes."""
    with binding():
        work = dict(_measure(root, raw, date))
    refs = {}
    for roster in work["inspection"]["execution_rosters"]:
        path = raw / ("shard_" + roster["shard"] + "_roster.json")
        atomic_json(path, roster)
        refs["shard_" + roster["shard"] + "_roster_reference"] = e.ref(path)
    path = raw / "diagnostic_protocol.json"
    atomic_json(path, work["inspection"]["diagnostic_protocol"])
    refs["diagnostic_protocol_reference"] = e.ref(path)
    work["protocol_sidecars"] = refs
    work["preconditions_checked"] = [
        dict(resource=r["check"], available=r["passed"], check_receipt=r)
        for r in work["inspection"]["checks"][-3:]
    ]
    work["preconditions_checked"].append(
        dict(
            resource="CPU and import limits",
            available=bool(os.cpu_count()),
            check_receipt=dict(
                cpu_count=os.cpu_count(),
                cpu_time_limit=resource.getrlimit(resource.RLIMIT_CPU),
                memory_limit=resource.getrlimit(resource.RLIMIT_AS),
                numpy_import=e.np.__version__,
                private_directory_mode=oct(raw.stat().st_mode & 0o777),
            ),
        )
    )
    command = next(c for c in work["scope"]["commands"] if c["name"] == "direct_consumers")
    command["argv"].extend(
        [
            "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            "tests/python/test_restricted_decision_audit_8210.py",
        ]
    )
    work["scope"]["code_test_hashes"].extend(e.ref(e.ROOT / path) for path in command["argv"][-2:])
    atomic_json(raw / "frozen_scope.json", work["scope"])
    atomic_json(raw / "work.json", work)
    return work


def qualify(work: Json) -> Json:
    """Run real owned/child coverage and required checks, with broad health only once."""
    with binding():
        return dict(legacy.qualify(work))


def build(work: Json, validation: Json) -> Json:
    """Keep all four readiness gates distinct from historical or scientific qualification."""
    with binding():
        value = dict(_build(work, validation))
    i = work["inspection"]
    tasks = [
        dict(
            t,
            canonical_task_hash=canonical_hash(t),
            prompt_hash=canonical_hash(t["prompt"]),
            gates_hash=canonical_hash(t.get("gated_on", [])),
            model_metadata_hash=canonical_hash(
                {k: t[k] for k in ["MODEL_SPECS", "inference_substrate_class"]}
            ),
            previous_outcome_rows=[
                deepcopy(r)
                for r in i["previous_outcome_rows"]
                if r["task_id"] in {p["experiment_id"] for p in t.get("prior_failures", [])}
            ],
        )
        for t in i["contract"]["tasks"]
    ]
    value.update(
        schema="carnot.v781_contract.v1",
        experiment_id=9166,
        task_id=e.TASK,
        milestone="2026.11.781",
        honest_verdict="complete_" + value["verdict_class"] + "_v781_contract",
        full_task_objects=tasks,
        canonical_task_hash=canonical_hash(tasks),
        retired_schedule=i["retired_schedule"],
        validation_obligations=i["validation_obligations"],
        current_science_paper_ready=False,
        preconditions_checked=work["preconditions_checked"],
        model_invocation_status="not_requested",
        diagnostic_protocol=i["diagnostic_protocol"],
    )
    if not i["contract"]["rows"]:
        for n, row in enumerate(value["rows"]):
            row.update(unit_id=f"exp{9166 + n}", group_id=f"exp{9166 + n}")
    value["protocol_references"] = dict(value["protocol_references"], **work["protocol_sidecars"])
    value.update(work["protocol_sidecars"])
    value["acceptance_gates"].update(
        diagnostic="Identical supports, equal-qid population moments, signed residual within1e-12; descriptive only, never rescues a primary bound",
        validation_obligations="621+53 original assertions and affected V780/current obligations; historical32/642 grants zero current credit",
    )
    value["acceptance_gates"] = {
        name: dict(
            condition=condition,
            principle="Frozen original operands and thresholds cannot be changed by outcomes; readiness needs actual affected checks.",
        )
        for name, condition in value["acceptance_gates"].items()
    }
    value["field_principles"].update(
        {
            k: "Exact original operands and executed checks bind this field; no current scientific benefit is inferred."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = legacy.checksum(value)
    return value


def replay(path: Path) -> Json:
    """Reject rehashed invented sidecars, then rebuild aggregates from authenticated inputs."""
    value = json.loads(path.read_bytes())
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    expected = {
        "shard_" + r["shard"] + "_roster_reference": r
        for r in work["inspection"]["execution_rosters"]
    }
    expected["diagnostic_protocol_reference"] = work["inspection"]["diagnostic_protocol"]
    for key, payload in expected.items():
        reference = work["protocol_sidecars"][key]
        sidecar = Path(reference["path"])
        if e.ref(sidecar) != reference or json.loads(sidecar.read_bytes()) != payload:
            raise ValueError("protocol_sidecar_drift")
    with binding():
        return dict(_replay(path))


def main(argv: list[str] | None = None) -> int:
    """Publish only checker-approved bytes; private fixtures cannot replace the live primary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261117")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--from-work", type=Path)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args(argv)
    e.progress("start_no_model_load")
    try:
        if args.date != "20261117":
            raise ValueError("run_date")
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        if args.from_work and args.output.absolute().is_relative_to(e.ROOT):
            raise ValueError("private_work_requires_external_output")
        raw = args.output.absolute().parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
        work = (
            json.loads(args.from_work.read_bytes())
            if args.from_work
            else measure(args.root, raw, args.date)
        )
        validation: Json = {}
        if not args.from_work and not args.checks_only:
            if (time.monotonic_ns() - work["start_ns"]) / 1e9 >= 3300:
                raise TimeoutError("measurement_deadline")
            e.progress("validation_before")
            validation = qualify(work)
            e.progress("validation_after")
            atomic_json(raw / "validation.json", validation)
            if (time.monotonic_ns() - work["start_ns"]) / 1e9 >= 4800:
                raise TimeoutError("hard_deadline")
        value = build(work, validation)
        atomic_json(raw / "candidate.json", value)
        if args.checks_only:
            print(json.dumps(dict(candidate=str(raw / "candidate.json"))), flush=True)
            return 0
        if not args.from_work and not value["required_checks_passed"]:
            e.progress("owned_validation_failed")
            return 1
        e.progress("publication_before")
        with binding():
            receipt = publish_primary(
                args.output, value, lambda p: legacy.terminal(p, raw / "terminal")
            )
        atomic_json(raw / "publication.json", receipt)
        print(json.dumps(dict(receipt, honest_verdict=value["honest_verdict"])), flush=True)
        e.progress("publication_after")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        e.progress("refused:" + str(error))
        return 1
