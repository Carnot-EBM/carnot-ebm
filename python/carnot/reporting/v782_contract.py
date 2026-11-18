"""REQ-REPORT-9180-V782: bind the next contract without activating or measuring science."""

from collections.abc import Iterator
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v781_contract as prior
from carnot.reporting import v782_contract_evidence as e
from carnot.reporting.protocol_owners import validate as validate_protocol
from carnot.reporting.current_work_receipt import atomic_json as atomic_json

Json = dict[str, Any]
legacy = prior.legacy
_measure, _build, _replay, _main = prior.measure, prior.build, prior.replay, prior.main


@contextmanager
def binding() -> Iterator[None]:
    """Reuse the checked lifecycle while leaving every historical producer unchanged."""
    with patch.multiple(prior, e=e, measure=measure, build=build, replay=replay, qualify=qualify):
        yield


def measure(root: Path, raw: Path, date: str) -> Json:
    """Seal the diagnostic reference before any validation or future TEST outcomes."""
    with binding():
        work = dict(_measure(root, raw, date))
    path = raw / "entropy_reference_protocol.json"
    atomic_json(path, work["inspection"]["entropy_reference_protocol"])
    work["protocol_sidecars"]["entropy_reference_protocol_reference"] = e.ref(path)
    atomic_json(raw / "work.json", work)
    return work


def qualify(work: Json) -> Json:
    """Execute the existing owned, consumer and private child checks once."""
    with binding(), prior.binding():
        return dict(legacy.qualify(work))


def build(work: Json, validation: Json) -> Json:
    """Administrative readiness cannot turn the historical failed diagnostic into science."""
    with binding():
        value = dict(_build(work, validation))
    value.update(
        schema="carnot.v782_contract.v1",
        experiment_id=9180,
        task_id=e.TASK,
        milestone="2026.11.782",
        honest_verdict="complete_" + value["verdict_class"] + "_v782_contract",
        entropy_reference_protocol=work["inspection"]["entropy_reference_protocol"],
        preserved_v781=work["inspection"]["preserved_v781"],
    )
    if not work["inspection"]["contract"]["rows"]:
        for index, row in enumerate(value["rows"]):
            row.update(unit_id=f"exp{9180 + index}", group_id=f"exp{9180 + index}")
    value["acceptance_gates"]["entropy_reference"] = dict(
        condition="TRAIN-correct token empirical CDF; fixed three AUROC contrasts, qid bootstrap and no TEST tuning; auxiliary only",
        principle="Frozen secondary comparisons cannot rescue original static or online acceptance bounds.",
    )
    value["claim_scope"]["allowed_claims"] = [
        "administrative task binding",
        "authenticated source and method authority",
        "frozen diagnostic supports",
    ]
    value["reproducibility_checksum"] = legacy.checksum(value)
    return value


def replay(path: Path) -> Json:
    """Fresh hashes cannot bless invented entropy protocol bytes or changed aggregates."""
    value = json.loads(path.read_bytes())
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    try:
        validate_protocol(work, "entropy_reference_protocol")
    except ValueError as error:
        raise ValueError("entropy_protocol_drift") from error
    with binding():
        return dict(_replay(path))


def main(argv: list[str] | None = None) -> int:
    """Preserve the existing private-output guard, deadlines and checked atomic publication."""
    started = time.monotonic()
    deadline = min(float(os.environ.get("CARNOT_V782_DEADLINE", started + 4800)), started + 4800)
    supervisor = legacy.lifecycle.child

    def bounded(name: str, *args: Any, **kwargs: Any) -> Json:
        reserve = 0 if name in {"cold_replay", "adversarial", "strict_rows"} else 300
        remaining = deadline - reserve - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("shared_deadline")
        kwargs["deadline"] = min(kwargs.get("deadline", 600), remaining)
        kwargs["heartbeat"] = min(kwargs.get("heartbeat", 60), 60)
        return dict(supervisor(name, *args, **kwargs))

    with (
        binding(),
        patch.object(legacy.lifecycle, "child", bounded),
        patch.object(legacy.lifecycle.prior, "child", bounded),
        patch.dict(os.environ, CARNOT_V782_DEADLINE=str(deadline)),
    ):
        return int(_main(argv))
