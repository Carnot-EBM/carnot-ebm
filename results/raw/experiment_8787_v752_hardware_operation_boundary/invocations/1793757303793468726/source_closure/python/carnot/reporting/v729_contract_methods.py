"""REQ-REPORT-8471: missing independent design cannot be repaired from activation."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any, Iterator
from unittest.mock import patch

import yaml

from carnot.reporting import v726_contract_methods as base
from carnot.reporting import v729_historical_inputs as history
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = base.ROOT, base.DESIGN, base.ACTIVE, base.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8471_v729_contract_methods",
    "exp8471-contract-methods",
    "2026.10.729",
)
PROTOCOL = "openspec/change-proposals/v729-methods-protocol.json"
METHODS = "docs/research-notes/v729-contract-methods.md"
PROTOCOL_PIN = "sha256:9890e64878938b6c6ec482ebab79158f1f7ce6cc137f940737c55c1411fa8ec1"
METHODS_PIN = "sha256:406278c95daaa6241f70afc775f9640ed2cb4580ff1ee192788fa59480da7b99"
TASKS_PIN = "f8e15329b32c9f068fbb3f5e55f7c13fa895803acc1140a32ecbb98d115b4990"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v729_contract_methods_8471.py"
EXTRA_TEST = TEST
OWNED = [
    f"python/carnot/reporting/v729_{suffix}.py"
    for suffix in ["contract_methods", "historical_inputs", "contract_runner"]
] + [CLI]
MODEL_SPECS: list[Json] = []
INHERITED = (
    "partition",
    "features",
    "decision",
    "feedback",
    "learning_audit",
    "cost",
    "source_controls",
    "scalar_fit",
    "source_audit",
    "objective_updates",
    "objective_audit",
)
FIELDS = history.FIELDS
read, reference, checksum, sha256_file, tasks_digest = (
    base.read,
    base.reference,
    base.checksum,
    base.sha256_file,
    base.tasks_digest,
)
_AUTHORITY, _MEASURE, _BUILD, _REPLAY, _ASSESS = (
    base.authority,
    base.measure,
    base.legacy.build,
    base.replay,
    base.assess_authorities,
)
measurement_clock = time.monotonic_ns
SECTION_PINS = {
    "append_protocol": "sha256:645b814c73136daf99456f816dd8b96748f0618bad1b6ad42bfb147b5db1b4e7",
    "append_cost": "sha256:9ff80cc383b766e5465d103973fb25e5fd555d234f7bedbfebdd51cfc86ef994",
    "validation_repairs": "sha256:648c435f0eea2d64e990edfa0476237c69a7e99699e0fedfd57e847d86f683ee",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual work counts so bounded historical children remain visible."""
    print(f"[exp8471] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Independent byte and section seals prevent repairs from changing scientific roles."""
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
        raise ValueError("separate_method_operand")
    return p


@contextmanager
def binding() -> Iterator[None]:
    """Reuse shipped readers and assertions while changing only current identity and adapters."""
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
            assess_authorities=lambda *args, **kwargs: _ASSESS(
                *args, **dict(kwargs, first_id=8471)
            ),
        ),
        patch.multiple(base.legacy, TASK=TASK, MILESTONE=MILESTONE),
    ):
        yield


def authority(root: Path, raw: Path) -> Json:
    """An absent independent table remains missing even when fourteen activated slots exist."""
    with binding():
        result = dict(_AUTHORITY(root, raw))
    plan = yaml.safe_load((root / ACTIVE).read_bytes()) if (root / ACTIVE).is_file() else {}
    tasks = plan.get("tasks", [])
    result.update(
        observed_count=len(tasks), tasks=tasks, observed_active_tasks_sha256=tasks_digest(tasks)
    )
    text = (root / DESIGN).read_text() if (root / DESIGN).is_file() else ""
    for marker in ["START", "END"]:
        if f"<!-- V729_TASK_CONTRACT_{marker} -->" not in text:
            result["gate_check_summary"].append(
                dict(
                    base.failure(root / DESIGN, "complete_json_" + marker.lower(), True, None),
                    upstream_id=TASK,
                )
            )
            result["activated"] = False
    if any(g["artifact_field"] == "required_authority" for g in result["gate_check_summary"]):
        result["canonical_tasks_sha256"] = None
        for field, expected in [
            ("independent_visible_task_table", 14),
            ("embedded_complete_task_objects", 14),
            ("printed_canonical_task_digest", TASKS_PIN),
        ]:
            result["gate_check_summary"].append(
                dict(base.failure(root / DESIGN, field, expected, None), upstream_id=TASK)
            )
    for i, row in enumerate(result["contract_rows"]):
        present = i < len(tasks)
        row.update(
            unit_id=tasks[i]["id"] if present else f"exp{8471 + i}",
            family=f"exp{8471 + i}",
            status="completed" if present else "unstarted",
            censored=not present,
            effective_independent_groups=0,
            missing_reason=None if row["matched"] else "independent_authority_absent_or_mismatched",
        )
    return result


def historical(root: Path, raw: Path) -> Json:
    """Separate immutable roots qualify each source or service family without a model call."""
    return dict(history.historical(root, raw, CLI, protocol()))


def measure(root: Path, raw: Path) -> Json:
    """Seal current methods before bounded original-authority historical qualification."""
    start = measurement_clock()
    with binding():
        work = dict(_MEASURE(root, raw))
    work["preconditions"].update(measurement_cap_s=3600, task_cap_s=4800, validation_reserve_s=900)
    work["history"] = historical(root, raw / "historical")
    elapsed = (measurement_clock() - start) / 1e9
    if elapsed > 3600:
        work["failures"].append(
            dict(base.failure(raw, "measurement_cap_s", 3600, elapsed), upstream_id=TASK)
        )
    work["phase_spans"] += [
        dict(name=r["name"], duration_s=r["duration_s"]) for r in work["history"]["receipts"]
    ]
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["invocation_argv"] = list(sys.argv)
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Readiness describes authenticated inputs only; exposed history cannot establish benefit."""
    with binding():
        value = dict(_BUILD(deepcopy(work), receipts, raw, output))
    qualified = value["required_checks_passed"] and not work.get("adversarial_findings")
    families = {f["experiment_id"]: f["qualified"] for f in work["history"]["families"]}
    ready = qualified and work["contract"]["activated"] and not work["failures"]
    verdict = "disqualified" if not qualified else "circular_positive" if ready else "blocked"
    groups = dict(
        source=[8430, 8431], learning=[8430, 8431, 8433, 8434], service=[8407, 8464, 8465, 8466]
    )
    provenance = [
        dict(
            producer=f["experiment_id"],
            producer_hash=f.get("primary_sha256"),
            imported_fields=[f.get("field"), "historical_model_provenance"],
            historical_model_provenance=f.get("historical_model_provenance", []),
        )
        for f in work["history"]["families"]
    ]
    refs = [
        dict(path=r["snapshot_path"], sha256=r["sha256"], source_path=r["path"])
        for r in work["history"]["references"]
        if r.get("exists")
    ]
    value.update(
        experiment_id=8471,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261013",
        honest_verdict=f"complete_{verdict}_v729_contract_methods",
        verdict_class=verdict,
        required_checks_passed=bool(qualified),
        flagged_adversarial=not qualified,
        intended_count=14,
        random_seed=7298471,
        methods_ready_score=int(ready),
        contract_ready_score=int(ready),
        current_contract_ready_score=int(ready),
        canonical_tasks_sha256=work["contract"]["canonical_tasks_sha256"],
        contract_digest=work["contract"]["canonical_tasks_sha256"],
        observed_active_tasks_sha256=work["contract"]["observed_active_tasks_sha256"],
        historical_family_receipts=work["history"]["families"],
        historical_validation_receipts=work["history"]["receipts"],
        historical_model_provenance=provenance,
        methods_protocol_reference=reference(ROOT / PROTOCOL),
        methods_protocol_path=str(ROOT / PROTOCOL),
        methods_protocol_sha256=PROTOCOL_PIN,
        methods_note_path=str(ROOT / METHODS),
        methods_note_sha256=METHODS_PIN,
        protocol_delta=work["protocol"]["protocol_delta"],
        literature_mapping=work["protocol"]["literature_mapping"],
        invocation_argv=work["invocation_argv"],
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8472, 8485)
        ],
        current_llm_invocations=0,
        numeric_learning_substrate="verifier_ensemble_against_cached_candidates",
        science_head_inputs=9,
        science_head_coefficients=109,
        scalar_coefficient_dimension=13,
        service_head_inputs=5,
        historical_authority_limits=dict(
            V726="retrospective head/checkpoint qualification; admission contrast null; arithmetic complete-service ceiling closed",
            V728="Exp8458/8461/8468 disqualified; source/objective comparisons unmeasured; Exp8465 journal equivalence qualified; Exp8466 speedup1.0447833367 CI95[0.9942346768,1.1095895448] null",
            V717_H1=-0.00390625,
            V717_H2=0,
            public_labels="exposed question roles; older overlap unknown",
        ),
        methodology_note="Administrative full-object contract and separate original V726/V728 input qualification. Independent design is missing after Phase1. Zero current LLM calls. No scientific benefit or timing is measured.",
    )
    for group, numbers in groups.items():
        score = int(qualified and all(families.get(n, False) for n in numbers))
        value[group + "_inputs_ready_score"] = score
        value["acceptance_gates"][group + "_inputs"] = bool(score)
    value["runtime_repair_ready_score"] = int(qualified and work["runtime"]["ready"])
    value["source_artifact_hashes"] += refs
    value["raw_shard_hashes"] += [dict(path=r["path"], sha256=r["sha256"]) for r in refs]
    value["gate_check_summary"] += [
        g for f in work["history"]["families"] for g in f["gate_check_summary"]
    ]
    value["cited_upstream_artifacts"] = [
        {k: row[k] for k in ("producer", "producer_hash", "imported_fields")}
        for row in value["cited_upstream_artifacts"]
    ] + provenance
    value["sample_size_budget"]["intended_task_slots"] = 14
    value["acceptance_gates"].update(
        current_contract=bool(ready),
        complete_fourteen_objects=bool(ready),
        sealed_methods=True,
        owned_classification=bool(qualified),
    )
    value["field_principles"].update(
        {
            k: "Bind to sealed invocation operands; administrative or input readiness never establishes scientific benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        canonical_tasks_sha256="Independent complete design digest only; null when missing. Observed activation is separately named.",
        source_inputs_ready_score="Original Exp8430 and8431 replay independently under V726 authority.",
        learning_inputs_ready_score="Original V726 panel/head/store/stream replay independently.",
        service_inputs_ready_score="Original Exp8407 five-input workload plus Exp8464/8465/8466 qualify under their own authority; old speed benefit remains null.",
        methods_ready_score="Complete current independent authority and sealed protocol plus owned qualification; historical absence and optional hardware are separate gates.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Fresh original-family reduction rejects repaired hashes and reconstructed summary drift."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        if value != build(
            work,
            value["validation_receipts"],
            Path(value["primitive_reference"]["path"]).parent,
            Path(value["publication_output"]),
        ):
            return False
        with TemporaryDirectory(prefix="exp8471-replay-", dir="/var/tmp") as directory:
            actual = historical(Path(work["root"]), Path(directory) / "history")
            families = json.loads(
                json.dumps(actual["families"]).replace(
                    str(Path(directory) / "history"),
                    str(Path(value["primitive_reference"]["path"]).parent / "historical"),
                )
            )
            if families != work["history"]["families"]:
                return False
        with binding():
            return bool(_REPLAY(path))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
