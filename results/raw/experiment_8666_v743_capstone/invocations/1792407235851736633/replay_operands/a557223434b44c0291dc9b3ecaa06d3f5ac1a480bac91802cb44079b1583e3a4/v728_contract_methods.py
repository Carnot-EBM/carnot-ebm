"""REQ-REPORT-8457: complete authority grants methods admission, never scientific benefit."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import time
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import v727_contract_methods as legacy
from carnot.reporting import v727_historical_inputs as h
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v685_authority_lifecycle import assess_authorities
from carnot.verify import sparse_energy_7996 as sparse

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = legacy.ROOT, legacy.DESIGN, legacy.ACTIVE, legacy.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8457_v728_contract_methods",
    "exp8457-contract-methods",
    "2026.10.728",
)
PROTOCOL = "openspec/change-proposals/v728-methods-protocol.json"
METHODS = "docs/research-notes/v728-contract-methods.md"
PROTOCOL_PIN = "sha256:8787b4a8cf72c7e1301785110328ed16dd8cfd018d810a1d605c3d668fed1b09"
METHODS_PIN = "sha256:174a26a6ac06c60eb094d3ddc601272cd70b75e723689627688cc064829a67ab"
TASKS_PIN = "1a6bb695f9ef6e97d0891d20ae2aad7a9f04fd3bdf3707cdf4a5e79bacaed7ae"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_v728_contract_methods_8457.py"
OWNED = [
    "python/carnot/reporting/v728_contract_methods.py",
    "python/carnot/reporting/v728_contract_runner.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
INHERITED = (
    *legacy.INHERITED,
    "source_controls",
    "source_audit",
    "objective_updates",
    "objective_audit",
    "persistence_profile",
    "journal",
    "journal_cost",
)
read, reference, checksum, sha256_file, tasks_digest = (
    legacy.read,
    legacy.reference,
    legacy.checksum,
    legacy.sha256_file,
    legacy.tasks_digest,
)
_AUTHORITY, _MEASURE, _BUILD, _REPLAY = (
    legacy.authority,
    legacy.measure,
    legacy.build,
    legacy.replay,
)
measurement_clock = time.monotonic_ns


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed actual counts prevent bounded child work from appearing stalled."""
    print(f"[exp8457] phase={phase} completed={completed} pending={pending}", flush=True)


def scalar_dimension() -> int:
    """One cubic input uses the original knots plus bias, without loading or fitting a model."""
    return int(sparse.BSpline.design_matrix([0.0, 0.5, 1.0], sparse.KNOTS, 3).shape[1]) + 1


def protocol() -> Json:
    """Independent byte seals stop repaired candidate hashes from changing scientific roles."""
    if sha256_file(ROOT / METHODS) != METHODS_PIN:
        raise ValueError("methods_hash")
    p = dict(read(dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_PIN)))
    basis_ref = p["scalar_basis_reference"]
    if sha256_file(Path(basis_ref["path"])) != basis_ref["sha256"]:
        raise ValueError("shipped_scalar_basis_hash")
    old = read(p["inherited_protocol_reference"])
    if any(p[k] != old[k] for k in INHERITED):
        raise ValueError("inherited_scientific_operand")
    if p["scalar_fit"] != dict(old["scalar_fit"], coefficients=scalar_dimension()):
        raise ValueError("one_input_scalar_dimension")
    return p


@contextmanager
def binding() -> Iterator[None]:
    """Only dated constants and narrow adapters change; all shipped assertions remain intact."""
    with patch.multiple(
        legacy,
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
        EXTRA_TEST=TEST,
        protocol=protocol,
        authority=authority,
        build=build,
        progress=progress,
        _ASSESS=lambda *args, **kwargs: assess_authorities(*args, **dict(kwargs, first_id=8457)),
    ):
        yield


def authority(root: Path, raw: Path) -> Json:
    """Full dictionary equality supplements the existing table, digest and lifecycle reader."""
    with binding():
        result = dict(_AUTHORITY(root, raw))
    text = (root / DESIGN).read_text() if (root / DESIGN).is_file() else ""
    if any(f"<!-- V728_TASK_CONTRACT_{marker} -->" not in text for marker in ["START", "END"]):
        result["gate_check_summary"].append(
            dict(
                legacy.legacy.failure(root / DESIGN, "complete_json_markers", True, False),
                upstream_id=TASK,
            )
        )
        result["activated"] = False
    for i, row in enumerate(result["contract_rows"]):
        row.update(
            unit_id=f"exp{8457 + i}" if i >= len(result["tasks"]) else result["tasks"][i]["id"],
            family=f"exp{8457 + i}",
        )
    return result


def measure(root: Path, raw: Path) -> Json:
    """Reuse original family workers while sealing this invocation before validation."""
    start = measurement_clock()
    reader_refs = [
        reference(ROOT / name)
        for name in [
            "python/carnot/reporting/v727_contract_methods.py",
            "python/carnot/reporting/v727_historical_inputs.py",
            "python/carnot/reporting/v726_contract_methods.py",
            "python/carnot/reporting/v726_capstone_evidence.py",
            "python/carnot/verify/sparse_energy_7996.py",
        ]
    ]
    with binding():
        work = dict(_MEASURE(root, raw))
    work["code_refs"] += reader_refs
    elapsed = (measurement_clock() - start) / 1e9
    if elapsed > 3600:
        work["failures"].append(
            dict(legacy.legacy.failure(raw, "measurement_cap_s", 3600, elapsed), upstream_id=TASK)
        )
    work["phase_spans"].append(dict(name="historical_inputs", duration_s=elapsed))
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Independent readiness describes inputs only; the oracle cannot establish benefit."""
    with binding():
        value = dict(_BUILD(deepcopy(work), receipts, raw, output))
    qualified = value["required_checks_passed"]
    families = {f["experiment_id"]: f["qualified"] for f in work["history"]["families"]}
    readiness = {
        "source": int(qualified and all(families.get(n, False) for n in [8430, 8431])),
        "learning": int(
            qualified and all(families.get(n, False) for n in [8430, 8431, 8433, 8434])
        ),
        "service": int(qualified and families.get(8407, False)),
    }
    value.update(
        experiment_id=8457,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261012",
        honest_verdict=f"complete_{value['verdict_class']}_v728_contract_methods",
        random_seed=7288457,
        canonical_tasks_sha256=work["contract"]["canonical_tasks_sha256"],
        protocol_delta=work["protocol"]["protocol_delta"],
        literature_mapping=work["protocol"]["literature_mapping"],
        scalar_coefficient_dimension=work["protocol"]["scalar_fit"]["coefficients"],
        methods_protocol_reference=reference(ROOT / PROTOCOL),
        methods_protocol_path=str(ROOT / PROTOCOL),
        methods_protocol_sha256=PROTOCOL_PIN,
        methods_note_path=str(ROOT / METHODS),
        methods_note_sha256=METHODS_PIN,
        protocol_path=str(ROOT / PROTOCOL),
        protocol_sha256=PROTOCOL_PIN,
        historical_authority_limits=dict(
            V727="six blocked primaries, eight skipped; methods score0; no source/objective/journal science",
            V726="qualified retrospective heads/checkpoints; null admission audit; kernel0.093%, persistence96.44%",
            V717_H1=-0.00390625,
            V717_H2=0,
            public_labels="retrospective exposed labels; older overlap unknown",
        ),
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8458, 8471)
        ],
        methodology_note="Oracle administrative V728 full-object authority and independent original V726 input qualification. Zero current LLM calls. Exposed labels grant no independent factuality, generalized learning, hidden-game transfer or service speedup.",
    )
    for family, ready in readiness.items():
        value[f"{family}_inputs_ready_score"] = ready
        value["acceptance_gates"][f"{family}_inputs"] = bool(ready)
    value["field_principles"].update(
        {
            key: "Bind this field to original sealed bytes and this invocation; methods and input qualification do not measure scientific benefit."
            for key in value
            if key not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        source_inputs_ready_score="Original Exp8430 feature_panel_ready_score and Exp8431 head_ready_score both equal1 after separate frozen replay.",
        learning_inputs_ready_score="Original source panel/head plus Exp8433 block_store_ready_score and Exp8434 learning_stream_ready_score equal1 after replay.",
        service_inputs_ready_score="Original Exp8407 python_cost_ready_score equals1 with independently authenticated five-input workload.",
        scalar_coefficient_dimension="Shipped cubic one-input basis has12 columns plus bias; the109-coefficient full head is a separate operand.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """The shipped fresh reducer verifies original family primitives and independent authority."""
    with binding():
        return bool(_REPLAY(path))
