"""REQ-REPORT-8902: account for signed CPU work without inventing fabric support."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

from carnot.reporting import hardware_operation_budget_8838 as shared
from carnot.reporting import hardware_operation_map_8889 as previous
from carnot.reporting import hardware_operation_runner_8825 as validation_runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]
ROOT = shared.ROOT
NAME = "experiment_8902_v761_hardware_operation_map"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/hardware_operation_map_8902.py", CLI]
TEST = "tests/python/test_hardware_operation_map_8902.py"
DEFAULT_OUTPUT = ROOT / "results" / (NAME + ".json")
PRODUCERS = {
    "labels": (
        "results/experiment_8896_v761_shared_terminal_evaluation.json",
        "sha256:4874e84d0b36fc1b7891e1ccaf93b53b60e75821bab156a4ace0c4f7307bc5a8",
        "certificate_rows_ready_score",
        8896,
    ),
    "attribution": (
        "results/experiment_8899_v761_buffered_cost_attribution.json",
        "sha256:7fc5c7e61d73abc501040b5569b1bdaae7b2f1da91d211abca15e4e94c1c290b",
        "cost_attribution_ready_score",
        8899,
    ),
    "adaptation": (
        "results/experiment_8901_v761_signed_adaptation_cost.json",
        "sha256:c5f35def899c0a2298121dace6f2ba04350f6ea2eefc3d68fb3399119f99d33d",
        "adaptation_cost_ready_score",
        8901,
    ),
}
HISTORY = {
    "results/experiment_8889_v760_hardware_operation_map.json": "sha256:e224e617ccac4bd8640cdf9f32e5717d7248a774aa11c1730310a47bf327d06a",
    "openspec/change-proposals/research-methods-v761-20261107.json": "sha256:e02928f8cce705aa5e62d57ad3a591f9b3cb9cda1c011fddd1867245fb0efe88",
    **{
        p: h
        for p, h in previous.engine.HISTORY.items()
        if any(n in p for n in ("8230", "8259", "8302"))
    },
    **{
        f"results/raw/{previous.NAME}/authorities/{h}": "sha256:" + h
        for _, h in previous.EXTERNAL.values()
    },
}
engine = namespace(shared, "hardware8902")
engine.NAME, engine.CLI, engine.OWNED, engine.TEST = NAME, CLI, OWNED, TEST
engine.PRODUCERS, engine.HISTORY = PRODUCERS, HISTORY
engine.SOURCES = list(
    dict.fromkeys(
        [
            *OWNED,
            TEST,
            *previous.engine.SOURCES[3:],
            "python/carnot/reporting/hardware_operation_map_8889.py",
            "python/carnot/reporting/v748_collection_contract.py",
            "openspec/capabilities/research-reporting/spec.md",
        ]
    )
)
base_build = engine.build
checker = namespace(validation_runner, "hardware8902_validation")
checker.m = sys.modules[__name__]
amdahl = previous.amdahl


def progress(phase: str) -> None:
    """Flush real boundaries so quiet evidence reading cannot conceal a stopped run."""
    print(f"[exp8902] phase={phase} completed=0 pending=0", flush=True)


def admission(value: Json, branch: str) -> str | None:
    """A blocked zero-attribution producer supplies diagnostics, never compatible timing."""
    field = PRODUCERS[branch][2]
    if value.get(field) is None:
        return "missing_field"
    if value[field] == 0 and branch != "attribution":
        return "observed_zero"
    if (
        value[field] not in (0, 1)
        or value.get("experiment_id") != PRODUCERS[branch][3]
        or value.get("required_checks_passed") is not True
        or value.get("flagged_adversarial", False)
        or (value.get("verifier_is_oracle") is True and value.get("verdict_class") == "positive")
        or value.get("verdict_class")
        not in (
            {"blocked", "null", "positive", "circular_positive"}
            if branch == "attribution" and value[field] == 0
            else {"null", "positive", "circular_positive"}
        )
        or not str(value.get("honest_verdict", "")).startswith("complete_")
        or not value.get("rows")
        or any(
            r.get("status") not in {"complete", "completed", "failed", "censored"}
            or r.get("denominator", 0) <= 0
            for r in value["rows"]
        )
    ):
        return "rejected_producer"
    return None


def collect(raw: Path) -> Json:
    """Reuse independent admission and deduplicate exact retained bytes, including old sources."""
    candidates: dict[str, Path] = {}
    for name, pin in [*HISTORY.items(), *((p, h) for p, h, _, _ in PRODUCERS.values())]:
        path = ROOT / name
        if path.is_file() and sha256_file(path) == pin:
            value = json.loads(path.read_bytes()) if path.suffix == ".json" else {}
            if not isinstance(value, dict):
                continue
            for key in ("source_artifact_hashes", "code_config_hashes", "raw_shard_hashes"):
                refs = value.get(key, [])
                for ref in refs.values() if isinstance(refs, dict) else refs:
                    candidates[ref["sha256"]] = Path(ref["path"])
    cache: dict[tuple[Path, str | None], Json] = {}

    def retain(path: Path, destination: Path, expected: str | None = None) -> Json:
        """Only byte-matching snapshots can stand in for an appended historical document."""
        key = (path, expected)
        if key in cache:
            return cache[key]
        source = path
        if expected and (not path.is_file() or sha256_file(path) != expected):
            source = candidates.get(expected, path)
        data = source.read_bytes()
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        if expected and digest != expected:
            raise ValueError("operand_hash_drift:" + str(path))
        target = destination / "operands" / digest.split(":")[1]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise ValueError("immutable_operand_drift")
        if not target.exists():
            target.write_bytes(data)
            target.chmod(0o444)
        ref = dict(path=str(target.absolute()), sha256=digest, original_path=str(path))
        cache[key] = ref
        if len(cache) % 1000 == 0:
            print(
                f"[exp8902] phase=operand_retention completed={len(cache)} pending=unknown",
                flush=True,
            )
        return ref

    engine.retain = retain
    work: Json = engine.collect(raw)
    work["diagnostic_evidence"] = {}
    for branch, (name, pin, _, _) in PRODUCERS.items():
        path = ROOT / name
        if branch not in work["evidence"] and path.is_file() and sha256_file(path) == pin:
            ref = retain(path, raw, pin)
            work["diagnostic_evidence"][branch] = ref
            work["refs"].append(ref)
    work["checks"].update(
        installed_dependencies={
            n: importlib.util.find_spec(n) is not None
            for n in ("pytest", "coverage", "ruff", "mypy")
        },
        running_repository_virtualenv=Path(sys.prefix) == ROOT / ".venv",
        time_budget_s=4200,
        hard_cap_s=4800,
        physical_probes_performed=0,
        execution_authority="read-only accounting authorized by current task; no device access",
    )
    return work


def reduce(evidence: Json) -> Json:
    """Keep measured counts and whole clocks separate from unsupported fabric work."""
    operands = deepcopy(evidence)
    for branch, value in operands.items():
        value[previous.PRODUCERS[branch][2]] = value[PRODUCERS[branch][2]]
    reduced = previous.reduce(operands)
    operations = {
        "signed_atom_checks": "CPU signed fact reads and integer/string atom comparisons",
        "local_binding_intersections": "CPU candidate-pair equality on shared variable bindings; content-dependent exact intersections",
        "small_linear_head": "CPU float32 27-coordinate signed head (21 old); three logits, softmax and gradient updates",
        "exact_certificate_bounds": "CPU exact binomial confidence bound evaluations and scheduled prefix checks; no Ising sampling",
        "host_parsing": "CPU syntax/schema admission and decoded object construction",
        "template_lookup": "CPU normalized topology strings and exact joins",
        "serialization": "host JSON encoding of parameters, predictions and acknowledgements",
        "durable_storage": "host atomic writes, file fsync, rename and directory fsync",
        "sampling": "zero Ising sampler calls in authenticated deterministic learner and certificate code",
    }
    qualified = evidence.get("attribution", {}).get("cost_attribution_ready_score") == 1
    boards = [
        dict(
            unit_id=f"{board}/{op}",
            cluster_id=board,
            arm=op,
            seed=0,
            metric=None,
            denominator=1,
            status="completed",
            missing_reason=None,
            board=board,
            operation=op,
            required_operation=description,
            boundary=boundary["boundary"],
            fabric_supported=False,
            operation_status="blocked_physical"
            if board == "GateMate"
            else "host_only"
            if op in {"serialization", "durable_storage", "host_parsing"}
            else "unsupported_workload",
            measured_board_time_s=None,
            transfer_bytes=None,
        )
        for board, boundary in engine.BOARDS.items()
        for op, description in operations.items()
    ]
    reduced.update(
        operation_map=operations,
        board_operation_rows=boards,
        unsupported_operations=[r["unit_id"] for r in boards],
        compatible_fraction_rows=[
            dict(
                board=board,
                operation=op,
                qualified_fraction=0.0 if qualified else None,
                missing_reason="no_qualified_fabric_kernel"
                if qualified
                else "unqualified_attribution",
                attribution_ready_score=evidence.get("attribution", {}).get(
                    "cost_attribution_ready_score"
                ),
            )
            for board in engine.BOARDS
            for op in operations
        ],
        amdahl_bounds=[
            dict(
                board=board,
                **amdahl(0.0 if qualified else None),
                claim="analytical best-case limit; never measured acceleration",
                transfer_fraction_best_case=0.0 if qualified else None,
                host_durability_remainder=1.0 if qualified else None,
                scope="current qualified fabrics support none of this deterministic workload",
            )
            for board in engine.BOARDS
        ],
        sampling_call_count=0 if evidence.get("labels") or evidence.get("adaptation") else None,
        signed_operation_counts=evidence.get("adaptation", {}).get("operation_counts"),
        cpu_memory_counters=evidence.get("adaptation", {}).get("cpu_memory_counters"),
        certificate_bound_rows=evidence.get("labels", {}).get("bound_rows"),
    )
    return reduced


def build(work: Json, validation: Json) -> Json:
    """A ready accounting report can preserve a blocked cost branch without claiming gain."""
    value: Json = base_build(work, validation)
    ready = value.pop("hardware_budget_ready_score")
    refs = {**work.get("diagnostic_evidence", {}), **work["evidence"]}
    evidence = {b: json.loads(Path(r["path"]).read_bytes()) for b, r in refs.items()}
    for branch in work.get("diagnostic_evidence", {}):
        value["rows"].extend(
            dict(
                r,
                source_branch=branch,
                source_admitted=False,
                status="completed" if r["status"] == "complete" else r["status"],
            )
            for r in evidence[branch]["rows"]
        )
    diagnostics = list(work["failures"])
    for branch, producer in evidence.items():
        field = PRODUCERS[branch][2]
        if producer[field] == 0:
            ref = refs[branch]
            diagnostics.append(
                dict(
                    check="compatible_fraction_admission",
                    upstream=branch,
                    artifact_field=field,
                    expected=1,
                    observed=0,
                    path=ref["path"],
                    sha256=ref["sha256"],
                    missing_reason="observed_zero",
                )
            )
        diagnostics.extend(
            dict(r, upstream_branch=branch) for r in producer.get("gate_check_summary", [])
        )
    blocked = [
        dict(
            unit_id=r["upstream"],
            cluster_id=r["upstream"],
            arm=r["upstream"],
            seed=7618902,
            metric=None,
            denominator=1,
            status="blocked",
            missing_reason=r["missing_reason"],
            source_gate=r,
        )
        for r in work["failures"]
    ]
    value["rows"].extend(blocked)
    verdict = (
        "disqualified"
        if value["verdict_class"] == "disqualified"
        else "blocked"
        if diagnostics or value["verdict_class"] == "blocked"
        else "null"
    )
    physical = deepcopy(engine.BOARDS)
    for ref in work["refs"]:
        if ref["original_path"].endswith("experiment_8889_v760_hardware_operation_map.json"):
            physical = json.loads(Path(ref["path"]).read_bytes())["physical_evidence_state"]
    value.update(
        experiment_id=8902,
        task_id="exp8902-hardware-operation-map",
        milestone="2026.11.761",
        run_date="20261107",
        random_seed=7618902,
        execution_venue="host",
        honest_verdict="complete_" + verdict + "_hardware_operation_map",
        verdict_class=verdict,
        hardware_map_ready_score=ready,
        gate_check_summary=diagnostics,
        intended_count=len(value["rows"]),
        blocked_count=len(blocked),
        independent_count=0,
        invocation=work.get("invocation", {}),
        cited_upstream_artifacts=[
            dict(
                ref,
                producer_task_id=evidence[b].get("task_id"),
                producer_verdict=evidence[b].get("verdict_class"),
                source_admitted=b in work["evidence"],
                fields=[PRODUCERS[b][2], "rows", "code_config_hashes", "raw_shard_hashes"],
            )
            for b, ref in sorted(refs.items())
        ],
        physical_evidence_state=physical,
        hardware_learning_path=dict(
            cpu_counter="Retained fact/sign/binding/candidate-pair counts and CPU/RSS counters; no isolated instruction or joule timings",
            small_head="SIMD/native CPU or GPU batching hypothesis for 27 coordinates; measure parity, launch/transfer and full durable cost first",
            programmable_features="Future programmable signed comparisons and local intersections need a new qualified kernel, precision and state-traffic evidence",
            certificate="CPU exact binomial confidence bounds; standalone labels saved do not reduce all-policy union work",
            sampling="Zero sampling calls; a faster Ising sampler cannot accelerate this learner",
            physical="KV260 quadratic k_max<=5; PolarFire Linux CPU only; GateMate last0xffffffff physical/JTAG block",
            durability="Host parsing, serialization, atomic writes and storage barriers remain paid",
        ),
        acquisition_priority=[
            dict(
                candidate=c,
                status="deferred",
                purchase_authorized=False,
                condition="Establish a measured compatible bottleneck and full host/transfer/durability costs first",
            )
            for c in (
                "existing CPU SIMD/GPU small-head path",
                "larger FPGA / Alveo V80",
                "Extropic TSU / Z1",
            )
        ],
        external_compatibility={
            k: dict(
                url=url,
                sha256="sha256:" + h,
                scope="retained paper/vendor evidence; no local device timing",
            )
            for k, (url, h) in previous.EXTERNAL.items()
        },
        methodology_note="Read-only byte-authenticated signed-feature and certificate operation accounting. Exp8899 blocked attribution supplies diagnostic censored units only; no numerical compatible fractions or speed ceiling qualify. Standalone certificate label savings and unchanged all-policy shared work are different scopes. Extropic and FPGA co-design papers motivate measuring host work; vendor announcements and TSU simulation give no Carnot physical timing. No current model loading, numerical training, hardware probe, purchase or independent generalization.",
    )
    for status in ("completed", "censored", "failed"):
        value[status + "_count"] = sum(r["status"] == status for r in value["rows"])
    value["field_principles"].update(
        {
            k: "Preserve authenticated producer scope; accounting readiness does not establish device speed or independent generalization."
            for k in value
            if k not in value["field_principles"]
        }
    )
    return value


engine.progress, engine.admission, engine.reduce, engine.build = progress, admission, reduce, build
replay = engine.replay


def main(argv: list[str] | None = None) -> int:
    """Publish checked terminal bytes after private validation and cold replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", choices=["20261107"], default="20261107")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--replay", "--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        try:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError) as exc:
            print(json.dumps(dict(passed=False, reason=str(exc))), flush=True)
            return 1
    output = (args.private_output or DEFAULT_OUTPUT).absolute()
    raw = output.parent / "raw" / NAME / "invocations" / str(time.monotonic_ns())
    template = ExperimentTemplate(
        8902, "Hardware operation map", output.name, repo_root=output.parent, seed=7618902
    )
    progress("operand_admission_before")
    with template.phase("operand_admission"):
        work = collect(raw)
    work["invocation"] = dict(
        argv=sys.orig_argv if argv is None else [sys.executable, str(ROOT / CLI), *argv],
        cwd=str(Path.cwd()),
        owner_pid=work["owner_pid"],
        run_id=str(work["started_ns"]),
    )
    atomic_json(raw / "work.json", work)
    progress("owned_validation_before")
    validation = (
        dict(passed=False, receipts=[], coverage=dict(complete=False))
        if args.private_output
        else checker.validate(raw)
    )
    progress("owned_validation_after")
    value = build(work, validation)
    progress("publication_before")
    report = publish_primary(
        output, value, lambda p: replay(p) if args.private_output else checker.terminal(p, raw)
    )
    atomic_json(raw / "publication.json", report)
    progress("complete")
    return 0
