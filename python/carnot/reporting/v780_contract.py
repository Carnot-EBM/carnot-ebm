"""REQ-REPORT-9152-V780: reuse checked publication without repeating scientific work."""

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v779_contract as legacy
from carnot.reporting import v780_contract_evidence as e
from carnot.reporting.current_work_receipt import atomic_json as atomic_json
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
_measure, _build, _replay, _main = legacy.measure, legacy.build, legacy.replay, legacy.main


@contextmanager
def binding() -> Iterator[None]:
    """The existing lifecycle validates this producer's files and foreign CLI."""
    with patch.multiple(legacy, e=e, measure=measure, build=build, replay=replay):
        yield


def measure(root: Path, raw: Path, date: str) -> Json:
    """Seal shared protocol files without placing evaluator labels in execution rosters."""
    with binding():
        work = dict(_measure(root, raw, date))
    references = {}
    for roster in work["inspection"]["execution_rosters"]:
        path = raw / ("shard_" + roster["shard"] + "_roster.json")
        atomic_json(path, roster)
        references["shard_" + roster["shard"] + "_roster_reference"] = e.ref(path)
    path = raw / "diagnostic_protocol.json"
    atomic_json(path, work["inspection"]["diagnostic_protocol"])
    references["diagnostic_protocol_reference"] = e.ref(path)
    work["v780_protocol_references"] = references
    atomic_json(raw / "work.json", work)
    return work


def qualify(work: Json) -> Json:
    """Run owned tests and child coverage, lint, typing, spec tracing and broad health once."""
    with binding():
        return dict(legacy.qualify(work))


def build(work: Json, validation: Json) -> Json:
    """Administrative task equality never grants scientific or historical runtime credit."""
    with binding():
        value = dict(_build(work, validation))
    inspection = work["inspection"]
    tasks = [
        dict(
            t,
            canonical_task_hash=canonical_hash(t),
            prompt_hash=canonical_hash(t["prompt"]),
            gates_hash=canonical_hash(t.get("gated_on", [])),
            deliverable_hash=canonical_hash(t["deliverable"]),
            model_metadata_hash=canonical_hash(
                {k: t[k] for k in ["MODEL_SPECS", "inference_substrate_class"]}
            ),
            previous_outcome_rows=[
                deepcopy(r)
                for r in inspection["previous_outcome_rows"]
                if r["task_id"] in {p["experiment_id"] for p in t.get("prior_failures", [])}
            ],
        )
        for t in inspection["contract"]["tasks"]
    ]
    value.update(
        schema="carnot.v780_contract.v1",
        experiment_id=9152,
        task_id=e.TASK,
        milestone="2026.11.780",
        honest_verdict="complete_" + value["verdict_class"] + "_v780_contract",
        full_task_objects=tasks,
        canonical_task_hash=canonical_hash(tasks),
        retired_schedule=inspection["retired_schedule"],
        partition_union_equals_original=inspection["partition_union_equals_original"],
        current_science_paper_ready=False,
    )
    if not inspection["contract"]["rows"]:
        for i, row in enumerate(value["rows"]):
            row.update(unit_id=f"exp{9152 + i}", group_id=f"exp{9152 + i}")
    references = work["v780_protocol_references"]
    value["protocol_references"] = dict(value["protocol_references"], **references)
    value.update(references)
    value["acceptance_gates"].update(
        execution_partition="Three disjoint lexical100 rosters retain all original qids, siblings and roles",
        diagnostic="Original evaluator-only masks and predeclared TEST100 scale/ranking definitions; no new promotion threshold",
        retired_schedule="Only forecast accounting retires; 621+53 original obligations and all quarantines remain",
    )
    value["field_principles"].update(
        {
            k: "Exact retained operands and executed checks bind accounting; no scientific benefit is inferred."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = legacy.checksum(value)
    return value


def replay(path: Path) -> Json:
    """Recompute sidecars from authenticated inspection so rehashing cannot invent a roster."""
    value = json.loads(path.read_bytes())
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    expected = {
        "shard_" + r["shard"] + "_roster_reference": r
        for r in work["inspection"]["execution_rosters"]
    }
    expected["diagnostic_protocol_reference"] = work["inspection"]["diagnostic_protocol"]
    for key, payload in expected.items():
        reference = work["v780_protocol_references"][key]
        sidecar = Path(reference["path"])
        if e.ref(sidecar) != reference or json.loads(sidecar.read_bytes()) != payload:
            raise ValueError("protocol_sidecar_drift")
    with binding():
        return dict(_replay(path))


def main(argv: list[str] | None = None) -> int:
    """Keep the existing private-output guard and checked primary publication intact."""
    with binding():
        return int(_main(argv))
