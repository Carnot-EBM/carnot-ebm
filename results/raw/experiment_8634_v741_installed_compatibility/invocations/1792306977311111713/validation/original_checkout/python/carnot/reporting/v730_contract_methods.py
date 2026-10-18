"""REQ-REPORT-8485: complete new authority cannot repair a truncated old design."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import time
from typing import Any, Iterator
from unittest.mock import patch

import yaml

from carnot.reporting import v729_contract_methods as shipped
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.roadmap_contract import parse_design

Json = dict[str, Any]
base, history = shipped.base, shipped.history
ROOT, DESIGN, ACTIVE, STAGED = shipped.ROOT, shipped.DESIGN, shipped.ACTIVE, shipped.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8485_v730_contract_methods",
    "exp8485-contract-methods",
    "2026.10.730",
)
PROTOCOL = "openspec/change-proposals/v730-methods-protocol.json"
METHODS = "docs/research-notes/v730-contract-methods.md"
PROTOCOL_PIN = "sha256:63e49630cc1dacfa733d9bef057d48c268f83c9f8054145419eaf73f741a871c"
METHODS_PIN = "sha256:2293dc552233e2bd6e5a5d160adc4e806ccf2965df1498ab0a5a5e1edc68a85c"
TASKS_PIN = "37ffae6a91ff0b935c365948c7d3f9cc51f56b25ece016f05677d0dd10cf53c0"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v730_contract_methods_8485.py"
OWNED = [
    f"python/carnot/reporting/v730_{n}.py" for n in ["contract_methods", "contract_runner"]
] + [CLI]
MODEL_SPECS: list[Json] = []
FIELDS = shipped.FIELDS
INHERITED = (*shipped.INHERITED, "append_protocol", "append_cost")
read, reference, checksum, sha256_file, tasks_digest = (
    shipped.read,
    shipped.reference,
    shipped.checksum,
    shipped.sha256_file,
    shipped.tasks_digest,
)
_AUTHORITY, _MEASURE, _BUILD, _REPLAY = (
    base.authority,
    shipped.measure,
    shipped.build,
    shipped.replay,
)
_ASSESS = shipped._ASSESS
measurement_clock = time.monotonic_ns
SECTION_PINS = {
    "validation_repairs": "sha256:0dda99a5acae8d6d92a6051ca99330a34f40435d7d2fc7a7f5ad396e2649f856",
    "evidence_canary": "sha256:7cfac4e68e44135d8275997f714e7f4229611ff22d34ffb51483701f0b4bde57",
    "protocol_delta": "sha256:726edc38834f0a98dabe6de0363e81f6f508a1e7d5746445e9afd04483ff5829",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so the supervisor can see bounded work advancing."""
    print(f"[exp8485] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Independent seals preserve scientific operands even after a candidate is rehashed."""
    if sha256_file(ROOT / METHODS) != METHODS_PIN:
        raise ValueError("methods_hash")
    p = dict(read(dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_PIN)))
    old = read(p["inherited_protocol_reference"])
    if any(p[k] != old[k] for k in INHERITED):
        raise ValueError("inherited_scientific_operand")
    ref = p["scalar_basis_reference"]
    if sha256_file(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("shipped_scalar_basis_hash")
    if any(canonical_hash(p[k]) != pin for k, pin in SECTION_PINS.items()):
        raise ValueError("separate_validation_or_view_operand")
    return p


@contextmanager
def binding() -> Iterator[None]:
    """Narrow dated adapters retain shipped assertions and the original historical readers."""
    constants: Json = dict(
        NAME=NAME,
        TASK=TASK,
        MILESTONE=MILESTONE,
        PROTOCOL=PROTOCOL,
        METHODS=METHODS,
        PROTOCOL_PIN=PROTOCOL_PIN,
        METHODS_PIN=METHODS_PIN,
        TASKS_PIN=TASKS_PIN,
        OWNED=OWNED,
        CLI=CLI,
        TEST=TEST,
    )
    with (
        patch.multiple(
            base,
            **constants,
            protocol=protocol,
            authority=authority,
            build=build,
            progress=progress,
            assess_authorities=lambda *a, **kw: _ASSESS(*a, **dict(kw, first_id=8485)),
        ),
        patch.multiple(base.legacy, TASK=TASK, MILESTONE=MILESTONE),
        patch.multiple(
            shipped,
            **constants,
            binding=binding,
            protocol=protocol,
            historical=historical,
            measurement_clock=measurement_clock,
            build=build,
        ),
    ):
        yield


def authority(root: Path, raw: Path) -> Json:
    """Full object checks supplement the table reader without inventing consumed staging."""
    with binding():
        result = dict(_AUTHORITY(root, raw))
    text = (root / DESIGN).read_text() if (root / DESIGN).is_file() else ""
    if any(f"<!-- V730_TASK_CONTRACT_{m} -->" not in text for m in ["START", "END"]):
        result["gate_check_summary"].append(
            dict(base.failure(root / DESIGN, "complete_json_markers", True, None), upstream_id=TASK)
        )
        result["activated"] = False
    if (root / STAGED).is_file():
        staged = yaml.safe_load((root / STAGED).read_bytes())
        machine = parse_design(text, milestone=MILESTONE)[1]
        if staged.get("tasks") != machine:
            result["gate_check_summary"].append(
                dict(
                    base.failure(root / STAGED, "staged_full_task_objects", True, False),
                    upstream_id=TASK,
                )
            )
            result["activated"] = False
    for i, row in enumerate(result["contract_rows"]):
        present = i < len(result["tasks"])
        row.update(
            unit_id=result["tasks"][i]["id"] if present else f"exp{8485 + i}",
            family=f"exp{8485 + i}",
            status="completed" if present else "unstarted",
            censored=not present,
            effective_independent_groups=0,
            missing_reason=None if row["matched"] else "authority_absent_or_mismatched",
        )
    return result


def historical(root: Path, raw: Path) -> Json:
    """Separate immutable roots replay qualified inputs without current model execution."""
    return dict(history.historical(root, raw, CLI, protocol()))


def measure(root: Path, raw: Path) -> Json:
    """Seal current authority before independently bounded historical qualification."""
    with binding():
        work = dict(_MEASURE(root, raw))
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Reuse the complete schema while keeping methods and independent input gates distinct."""
    with binding():
        value = dict(_BUILD(work, receipts, raw, output))
    value.update(
        experiment_id=8485,
        task_id=TASK,
        milestone=MILESTONE,
        honest_verdict=f"complete_{value['verdict_class']}_v730_contract_methods",
        random_seed=7308485,
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8486, 8499)
        ],
        protocol_path=str(ROOT / PROTOCOL),
        protocol_sha256=PROTOCOL_PIN,
        contract_rows=value["rows"],
        authority_snapshots=work["contract"]["authority_snapshots"],
        historical_authority_limits=dict(
            value["historical_authority_limits"],
            V729="truncated design preserved byte-identically; ten cascade-blocked tasks; source/objective/append science unmeasured; Exp8482 original ARC line350 untested",
        ),
        methodology_note="Administrative complete V730 table/JSON/digest/activated authority with full dictionary equality, and separately qualified original V726/V728 inputs. No model loads, generations, CUDA or board calls. Methods readiness is not scientific benefit; exposed evidence keeps both generalization scores zero.",
    )
    value["field_principles"].update(
        {
            k: "Bind to sealed invocation operands; administrative qualification does not establish scientific benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Fresh shipped reducers authenticate primitive rows, receipts and original family operands."""
    with binding():
        return bool(_REPLAY(path))
