"""REQ-REPORT-9194-V783: checked administrative binding without model work or activation."""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v782_contract as prior
from carnot.reporting import v783_contract_evidence as e
from carnot.reporting.current_work_receipt import atomic_json as atomic_json
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
legacy = prior.legacy
_measure, _build, _replay = prior.measure, prior.build, prior.replay


@contextmanager
def binding() -> Iterator[None]:
    """Keep the established publisher and receipt checks under this producer's identity."""
    with (
        patch.multiple(prior, e=e, measure=measure, build=build, replay=replay, qualify=qualify),
        prior.binding(),
        prior.prior.binding(),
    ):
        yield


def measure(root: Path, raw: Path, date: str) -> Json:
    """Freeze original protocols, affected assertions and ambiguity before any outcomes."""
    with binding():
        work = dict(_measure(root, raw, date))
    tools = {
        name: importlib.util.find_spec(name) is not None
        for name in ["pytest", "coverage", "ruff", "mypy", "yaml", "numpy"]
    }
    work["preconditions_checked"].extend(
        [
            dict(resource="required_tools", available=all(tools.values()), check_receipt=tools),
            dict(
                resource="host_resources",
                available=bool(os.cpu_count()),
                check_receipt=dict(
                    cpu_count=os.cpu_count(),
                    private_free_disk_bytes=shutil.disk_usage(raw).free,
                    memory=Path("/proc/meminfo").read_text().splitlines()[:3],
                ),
            ),
        ]
    )
    path = raw / "label_ambiguity_protocol.json"
    atomic_json(path, work["inspection"]["label_ambiguity_protocol"])
    path.chmod(0o400)
    work["protocol_sidecars"]["label_ambiguity_protocol_reference"] = e.ref(path)
    work["scope"]["validation_obligations"] = work["inspection"]["validation_obligations"]
    command = next(c for c in work["scope"]["commands"] if c["name"] == "direct_consumers")
    affected = [
        "tests/python/test_v782_contract_9180.py",
        "tests/python/test_causal_validation_9181.py",
        "tests/python/test_v782_capstone_9193.py",
    ]
    command["argv"].extend(affected)
    work["scope"]["code_test_hashes"].extend(e.ref(e.ROOT / p) for p in affected)
    atomic_json(raw / "frozen_scope.json", work["scope"])
    atomic_json(raw / "work.json", work)
    return work


def qualify(work: Json) -> Json:
    """Run owned coverage, affected consumers and the requested repository suite once."""
    with binding():
        return dict(legacy.qualify(work))


def build(work: Json, validation: Json) -> Json:
    """Administrative readiness stays separate from every original scientific effect."""
    with binding():
        value = dict(_build(work, validation))
    value.update(
        schema="carnot.v783_contract.v1",
        experiment_id=9194,
        task_id=e.TASK,
        milestone="2026.11.783",
        honest_verdict="complete_" + value["verdict_class"] + "_v783_contract",
        preserved_v782=work["inspection"].get("preserved_v782", {}),
        label_ambiguity_protocol=work["inspection"]["label_ambiguity_protocol"],
    )
    if not work["inspection"]["contract"]["rows"]:
        for index, row in enumerate(value["rows"]):
            row.update(unit_id=f"exp{9194 + index}", group_id=f"exp{9194 + index}")
    value["acceptance_gates"]["label_ambiguity"] = dict(
        condition="observed disagreement and missing-A2 envelope; exact affine endpoints; fixed original weights",
        principle="Sensitivity never selects an update or replaces original primary gates.",
    )
    value["acceptance_gates"]["corpus_scan"] = dict(
        condition="Exp9195 corpus_scan_ready_score; Exp9181/9182 remain retired",
        principle="A new lossless corpus correction cannot rewrite a historical disqualification.",
    )
    value["claim_scope"]["allowed_claims"] = [
        "administrative task binding",
        "authenticated original human and method authority",
        "frozen secondary label sensitivity protocol",
    ]
    value["reproducibility_checksum"] = legacy.checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rehashed sidecars cannot change the predeclared label sensitivity protocol."""
    value = json.loads(path.read_bytes())
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    reference = work["protocol_sidecars"]["label_ambiguity_protocol_reference"]
    sidecar = Path(reference["path"])
    if (
        e.ref(sidecar) != reference
        or json.loads(sidecar.read_bytes()) != e.FROZEN["label_ambiguity_protocol"]
    ):
        raise ValueError("label_ambiguity_protocol_drift")
    with binding():
        return dict(_replay(path))


def main(argv: list[str] | None = None) -> int:
    """Publish checker-approved terminal accounting even when owned validation disqualifies it."""
    started = time.monotonic()
    deadline = min(float(os.environ.get("CARNOT_V783_DEADLINE", started + 4800)), started + 4800)
    supervisor = legacy.lifecycle.child

    def bounded(name: str, *args: Any, **kwargs: Any) -> Json:
        """A descendant cannot outlive the parent's measurement and publication reserves."""
        remaining = (
            deadline
            - (0 if name in {"cold_replay", "adversarial", "strict_rows"} else 300)
            - time.monotonic()
        )
        if remaining <= 0:
            raise ValueError("shared_deadline")
        kwargs["deadline"] = min(kwargs.get("deadline", 600), remaining)
        kwargs["heartbeat"] = min(kwargs.get("heartbeat", 60), 60)
        return dict(supervisor(name, *args, **kwargs))

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
        with (
            binding(),
            patch.object(legacy.lifecycle, "child", bounded),
            patch.object(legacy.lifecycle.prior, "child", bounded),
            patch.dict(os.environ, CARNOT_V783_DEADLINE=str(deadline)),
        ):
            if args.date != "20261117":
                raise ValueError("run_date")
            if args.cold_replay:
                print(json.dumps(replay(args.cold_replay)), flush=True)
                return 0
            if args.from_work and args.output.absolute().is_relative_to(e.ROOT):
                raise ValueError("private_work_requires_external_output")
            raw = (
                args.output.absolute().parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
            )
            work = (
                json.loads(args.from_work.read_bytes())
                if args.from_work
                else measure(args.root, raw, args.date)
            )
            validation: Json = {}
            if not args.from_work and not args.checks_only:
                if time.monotonic() > deadline - 1500:
                    raise ValueError("measurement_deadline")
                e.progress("validation_before")
                validation = qualify(work)
                atomic_json(raw / "validation.json", validation)
                e.progress("validation_after")
            value = build(work, validation)
            atomic_json(raw / "candidate.json", value)
            if args.checks_only:
                print(json.dumps(dict(candidate=str(raw / "candidate.json"))), flush=True)
                return 0
            e.progress("publication_before")
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
