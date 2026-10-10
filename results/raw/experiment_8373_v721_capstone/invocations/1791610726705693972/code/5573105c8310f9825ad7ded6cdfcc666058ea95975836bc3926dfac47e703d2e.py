"""REQ-REPORT-8373: frozen terminal accounting keeps absence separate from null science."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import v721_contract_methods as contract
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.primary_publication import validate_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v719_capstone_evidence import memory
from carnot.reporting.v720_terminal_replay import gate as gate

Json = dict[str, Any]
ROOT, DESIGN, STAGED, ACTIVE = contract.ROOT, contract.DESIGN, contract.STAGED, contract.ACTIVE
PROTOCOL = "openspec/change-proposals/v717-local-learning-protocol.json"
PROTOCOL_PIN = "sha256:853709123024de763e96dd688e819f0430205ae6d97d6561a2b95cca23b81c6f"
NAME, TASK, MILESTONE = "experiment_8373_v721_capstone", "exp8373-capstone", "2026.10.721"
CLI, TEST = "scripts/experiments/" + NAME + ".py", "tests/python/test_v721_capstone_8373.py"
OWNED = [
    "python/carnot/reporting/v721_capstone_evidence.py",
    "python/carnot/reporting/v721_capstone.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
READERS = {
    8360: "v721_contract_methods",
    8361: "utility_audit_qualification_8361",
    8362: "threshold_guard_8362",
    8370: "arc_outcome_delta_8370",
}
NEXT = [
    "Continue exact frozen input custody while full task and source seals match.",
    "Retire only frozen procedure/budget null scopes; new preregistered evidence must satisfy unchanged V717 utility and retention thresholds.",
    "Require certified interval/probability fidelity and a useful fast path before atomic table work.",
    "Defer atomic crash recovery until guard_ready_score=1 and real crash receipts exist.",
    "Defer actual delayed table training until atomic_table_ready_score=1; preserve earlier continuous-learning null.",
    "Defer complete local request costs until an authenticated atomic serving transaction exists.",
    "Defer native parity until actual Rust/PyO3 guarded serving and atomic state qualify.",
    "Defer native service costs until local costs and native parity both qualify.",
    "Require an authenticated CUDA environment delta and successful context/copy parity.",
    "Keep canary pre-gated until typed runtime, actual change and CUDA context all qualify.",
    "Require new authenticated cross-game supervisor outcomes beyond the recorded frontier.",
    "Require complete local service/transfer evidence and supported actual board operations; preserve original owned coverage failure.",
    "Recover both exact source hashes, then dated cable/port/power change, IDCODE0x20000001, n16 flash and device sample/hash smoke.",
    "Continue changed evidence only; do not retry unchanged terminal blocks or claim publication readiness.",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed boundaries let the conductor distinguish slow work from a stall."""
    print(f"[exp8373] phase={phase} completed={completed} pending={pending}", flush=True)


def freeze(path: Path, raw: Path, expected: str | None = None) -> Json:
    """Stream custody copies so a large upstream artifact cannot inflate parent memory."""
    path = path.absolute()
    digest = sha256_file(path) if path.is_file() else None
    ref = dict(path=str(path), sha256=digest, exists=digest is not None, expected_sha256=expected)
    if digest is not None:
        target = raw / (digest[7:] + path.suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with path.open("rb") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            target.chmod(0o400)
        ref["snapshot_path"] = str(target)
    return ref


def load(ref: Json) -> Json:
    """Every read checks sealed bytes before a scientific or accounting field is used."""
    return dict(
        json.loads(
            checked(
                dict(path=ref.get("snapshot_path", ref["path"]), sha256=ref["sha256"])
            ).read_bytes()
        )
    )


def references(value: Any) -> list[Json]:
    """Nested producer manifests carry explicit input roots as well as top-level references."""
    result = []
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            result.append(value)
        elif isinstance(value.get("snapshot_path"), str) and isinstance(value.get("sha256"), str):
            result.append(
                dict(
                    path=value["snapshot_path"],
                    sha256=value["sha256"],
                )
            )
        for item in value.values():
            result.extend(references(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(references(item))
    return result


@contextmanager
def frozen_inputs(refs: list[Json], scratch: Path) -> Iterator[None]:
    """Redirect declared input reads and reconstruction writes away from mutable authority."""
    mapping = {r["path"]: r["snapshot_path"] for r in refs if r["exists"]}
    for r in refs:
        if r.get("authority_alias") and r["exists"]:
            mapping[r["authority_alias"]] = r["snapshot_path"]
    original = Path.open
    original_replace, original_mkdir = Path.replace, Path.mkdir
    write_root = ROOT / "results"

    def location(path: Path) -> Path:
        return (
            scratch / "writes" / path.absolute().relative_to(write_root)
            if path.absolute().is_relative_to(write_root)
            else path
        )

    def moved(path: Path, target: Any) -> Path:
        destination = location(Path(target))
        destination.parent.mkdir(parents=True, exist_ok=True)
        return original_replace(location(path), destination)

    def directory(path: Path, *args: Any, **kwargs: Any) -> None:
        original_mkdir(location(path), *args, **kwargs)

    def opened(path: Path, *args: Any, **kwargs: Any) -> Any:
        mode = str(args[0] if args else kwargs.get("mode", "r"))
        label = str(path.absolute())
        relocated = location(path)
        if any(flag in mode for flag in ["w", "a", "+"]) and relocated != path:
            relocated.parent.mkdir(parents=True, exist_ok=True)
            return original(relocated, *args, **kwargs)
        if any(flag in mode for flag in ["w", "a", "+"]) and label in mapping:
            target = scratch / canonical_hash(label)[7:]
            return original(target, *args, **kwargs)
        if not any(flag in mode for flag in ["w", "a", "+"]):
            if relocated != path and relocated.is_file():
                return original(relocated, *args, **kwargs)
            if label in mapping:
                return original(Path(mapping[label]), *args, **kwargs)
            if path.name in {
                "research-roadmap.yaml",
                "research-roadmap-next.yaml",
                "research-roadmap-vNEXT.md",
            } and path.is_relative_to(ROOT):
                raise ValueError("mutable_authority_fallback")
        return original(path, *args, **kwargs)

    with (
        patch.object(Path, "open", opened),
        patch.object(Path, "replace", moved),
        patch.object(Path, "mkdir", directory),
    ):
        yield


def utility(value: Json, refs: list[Json]) -> Json:
    """Independent reductions adopt qualified audits while preserving their original failures."""
    from carnot.verify import static_benefit_audit_8350 as h1
    from carnot.verify import learning_retention_audit_8351 as h2

    work = load(value["measurement_reference"])
    result = {}
    for key, number in [("H1", "8350"), ("H2", "8351")]:
        state = load(work["audits"][number]["measurement_reference"])
        actual = (
            h1.reduce(
                state["predictions"], state["targets"], state["comparator"], state["optimizer"]
            )
            if key == "H1"
            else h2.reduce(state["state"], state["targets"], state["checks"])
        )
        if actual != value[key]:
            raise ValueError("independent_utility_reduction")
        result[key] = dict(
            actual,
            qualified=True,
            intended_count=128 if key == "H1" else 88,
            independent_reduction_sha256=canonical_hash(actual),
        )
    result["original_dispositions"] = value["original_dispositions"]
    result["qualified_scope_decision"] = value["qualified_scope_decision"]
    return result


def outcome(task: Json, primary: Json, raw: Path, sealed: list[Json] | None = None) -> Json:
    """Authenticate one terminal before importing its measured fields or replaying its reader."""
    value = load(primary)
    number = int(task["id"][3:7])
    refs = list(sealed or [primary])
    checks = []
    pre = value.get("blocked_at_layer") == "conductor_pre_gate"
    if pre:
        if value["experiment"] != number:
            raise ValueError("pre_gate_identity")
        for g in value["gates_evaluated"]:
            checked(dict(path=g["artifact_path"], sha256=g["artifact_sha256"]))
        return dict(
            disposition="conductor_pre_gate",
            honest_verdict=value["honest_verdict"],
            verdict_class="blocked",
            producer_executed=False,
            eligible=False,
            selected={},
            checks=[
                dict(
                    gate(
                        g["artifact_path"],
                        g["artifact_field"],
                        g["expected"],
                        g["actual"],
                        g["artifact_sha256"],
                    ),
                    upstream=g["upstream"],
                )
                for g in value["gates_evaluated"]
            ],
            closure=refs,
            branch_replay=dict(status="not_executed_pre_gate", passed=False),
        )
    validate_primary(value, Path(primary["path"]))
    if value["task_id"] != task["id"]:
        raise ValueError("wrong_authority")
    if sealed is None:
        terminal_path = Path(value["terminal_validation_sidecar_path"])
        terminal = json.loads(terminal_path.read_bytes())
        side = json.loads(Path(terminal["publication"]["sidecar_path"]).read_bytes())
        if (
            terminal["publication"]["primary_sha256"] != primary["sha256"]
            or terminal["publication"]["primary_path"] != primary["path"]
            or side["primary_sha256"] != primary["sha256"]
            or side["primary_path"] != primary["path"]
            or side["report"]["passed"] is not True
        ):
            raise ValueError("terminal_binding")
        operands = [
            reference(terminal_path),
            reference(Path(terminal["publication"]["sidecar_path"])),
        ]
        from carnot.reporting.v720_replay_closure import requirements

        operands.extend(requirements(value))
        operands.extend(references(value))
        roots = [value]
        for key in ["work_reference", "measurement_reference", "primitive_reference"]:
            if value.get(key):
                state = load(value[key])
                operands.extend(references(state))
                roots.append(state)
        if number == 8361:
            audit = load(value["measurement_reference"])
            for a in audit["audits"].values():
                operands.extend(references(load(a["measurement_reference"])))
        for state in roots:
            for a in [state, state.get("authority", {}), state.get("contract", {})]:
                snapshots = a.get("authority_snapshots", {})
                for snap in snapshots.values() if isinstance(snapshots, dict) else snapshots:
                    if snap.get("exists") and snap.get("source_path"):
                        operands.append(
                            dict(
                                path=snap["snapshot_path"],
                                sha256=snap["sha256"],
                                authority_alias=snap["source_path"],
                            )
                        )
        unique = {(r.get("snapshot_path", r["path"]), r["sha256"]): r for r in operands}
        for index, r in enumerate(unique.values()):
            if index % 50 == 0:
                progress("custody", index, len(unique) - index)
            saved = freeze(Path(r.get("snapshot_path", r["path"])), raw / "custody", r["sha256"])
            saved["path"] = r.get("snapshot_path", r["path"])
            if r.get("authority_alias"):
                saved["authority_alias"] = r["authority_alias"]
            refs.append(saved)
    if sealed is not None:
        terminal, side = load(refs[1]), load(refs[2])
        if (
            terminal["publication"]["primary_sha256"] != primary["sha256"]
            or side["primary_sha256"] != primary["sha256"]
            or terminal["publication"]["primary_path"] != primary["path"]
            or side["primary_path"] != primary["path"]
            or side["report"]["passed"] is not True
        ):
            raise ValueError("terminal_binding")
    for r in refs:
        checks.append(
            gate(
                r["path"],
                "source.sha256",
                r.get("expected_sha256") or r["sha256"],
                r["sha256"],
                r["sha256"],
            )
        )
        if r["exists"]:
            checked(dict(path=r["snapshot_path"], sha256=r["sha256"]))
    selected = {
        k: v
        for k, v in value.items()
        if k.endswith("_score")
        or k
        in [
            "original_table_flip_count",
            "fast_path_fraction",
            "fallback_counts",
            "guarded_action_mismatch_count",
            "history_authentication",
            "missing_source_hashes",
            "original_transcript_sha256",
            "board_obligations",
            "operation_rows",
            "current_frontier",
            "prior_frontier",
            "support_frontier",
            "arm_support_rows",
            "model_invocation_counts",
            "historical_model_provenance",
            "original_dispositions",
            "qualified_scope_decision",
        ]
    }
    eligible = (
        value.get("required_checks_passed") is True
        and value.get("flagged_adversarial") is False
        and value["verdict_class"] not in ["blocked", "disqualified"]
    )
    branch = dict(status="terminal_disposition_preserved", passed=False)
    reader_source = (
        ROOT / "python/carnot/reporting" / (READERS[number] + ".py") if eligible else None
    )
    reader_observed = sha256_file(reader_source) if reader_source is not None else None
    with TemporaryDirectory(prefix="exp8373-branch-", dir="/var/tmp") as directory:
        with frozen_inputs(refs, Path(directory)):
            if number == 8361 and eligible:
                selected.update(utility(value, refs))
            if eligible and all(g["passed"] for g in checks):
                source = ROOT / "python/carnot/reporting" / (READERS[number] + ".py")
                pin = next(
                    r["sha256"]
                    for r in refs
                    if r["path"].endswith("/python/carnot/reporting/" + READERS[number] + ".py")
                )
                if reader_observed != pin:
                    checks.append(gate(str(source), "reader.source_sha256", pin, reader_observed))
                    branch = dict(status="blocked_changed_source", passed=False)
                else:
                    module = importlib.import_module("carnot.reporting." + READERS[number])
                    passed = module.replay(
                        value if number == 8370 else Path(primary["snapshot_path"])
                    )
                    branch = dict(
                        status="cold_replay_passed" if passed else "disqualified_cold_replay",
                        passed=bool(passed),
                    )
            elif eligible:
                branch = dict(status="blocked_missing_source_closure", passed=False)
    return dict(
        disposition="qualified" if eligible else value["verdict_class"],
        honest_verdict=value["honest_verdict"],
        verdict_class=value["verdict_class"],
        producer_executed=True,
        eligible=eligible,
        selected=selected,
        checks=checks + value.get("gate_check_summary", []),
        closure=refs,
        branch_replay=branch,
    )


def worker(request: Path, output: Path) -> int:
    """A real bounded child records peak memory even when authentication rejects its input."""
    progress("worker_before")
    before = memory()
    args = json.loads(request.read_bytes())
    try:
        if canonical_hash(args["task"]) != args["task_sha256"]:
            raise ValueError("wrong_authority")
        result = outcome(args["task"], args["primary"], output.parent, args.get("closure"))
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        result = dict(
            disposition="blocked",
            honest_verdict=None,
            verdict_class="blocked",
            producer_executed=True,
            eligible=False,
            selected={},
            checks=[gate(str(request), "authenticated_input", True, str(error))],
            closure=args.get("closure") or [args["primary"]],
            branch_replay=dict(status="blocked_input_authentication", passed=False),
        )
    after = memory()
    result["memory"] = dict(
        before=before,
        after=after,
        growth_mb=after["peak_rss_mb"] - before["peak_rss_mb"],
        peak_bound_mb=1500,
        growth_bound_mb=500,
        passed=after["peak_rss_mb"] <= 1500 and after["peak_rss_mb"] - before["peak_rss_mb"] <= 500,
    )
    atomic_json(output, result)
    progress("worker_after", 1, 0)
    return int(
        not result["memory"]["passed"]
        or result["branch_replay"]["status"]
        in ["blocked_input_authentication", "disqualified_cold_replay"]
    )


def invoke(
    task: Json, primary: Json, raw: Path, closure: list[Json] | None = None
) -> tuple[Json, Json]:
    """Keep real child argv, deadlines, clocks, exits and hashed streams for every branch."""
    request, output = raw / "request.json", raw / "summary.json"
    atomic_json(
        request, dict(task=task, task_sha256=canonical_hash(task), primary=primary, closure=closure)
    )
    receipt = child(
        "branch",
        [
            str(ROOT / ".venv/bin/python"),
            "-u",
            str(ROOT / CLI),
            "--worker-request",
            str(request),
            "--worker-output",
            str(output),
        ],
        raw / "logs",
        deadline=180,
        heartbeat=20,
        scope="branch",
    )
    if not output.is_file():
        value = load(primary)
        return dict(
            disposition=value["verdict_class"],
            honest_verdict=value["honest_verdict"],
            verdict_class=value["verdict_class"],
            producer_executed=True,
            eligible=False,
            selected={},
            checks=[gate(str(output), "owned_child_result.exists", True, None)],
            closure=[primary],
            branch_replay=dict(status="disqualified_owned_child", passed=False),
            memory=dict(passed=False),
        ), receipt
    return load(reference(output)), receipt


def authority(work: Json) -> Json:
    """Rebuild exact task authority from frozen copies instead of today's roadmap."""
    with TemporaryDirectory(prefix="exp8373-authority-", dir="/var/tmp") as directory:
        root = Path(directory)
        for name, ref in zip([DESIGN, STAGED, ACTIVE], work["authority_refs"], strict=True):
            if ref["exists"]:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ref["snapshot_path"], target)
        return dict(contract.authority(root, root / "assessment"))


def historical(path: Path, raw: Path) -> Json:
    """Authenticate original terminal seals without executing historical code or importing readiness."""
    primary = freeze(path, raw)
    value = load(primary)
    terminal_ref = freeze(Path(value["terminal_validation_sidecar_path"]), raw)
    terminal = load(terminal_ref)
    side_ref = freeze(Path(terminal["publication"]["sidecar_path"]), raw)
    side = load(side_ref)
    authenticated = (
        terminal["publication"]["primary_sha256"] == primary["sha256"]
        and terminal["publication"]["primary_path"] == str(path.absolute())
        and side["primary_sha256"] == primary["sha256"]
        and side["primary_path"] == str(path.absolute())
        and side["report"]["passed"] is True
    )
    return dict(
        reference=primary,
        refs=[primary, terminal_ref, side_ref],
        experiment_id=value["experiment_id"],
        task_id=value["task_id"],
        honest_verdict=value["honest_verdict"],
        verdict_class=value["verdict_class"],
        authenticated=authenticated,
        preserved=True,
        readiness_imported=False,
        repository_health=value.get("repository_health_attempt"),
    )


def retire(task: Json, current: Json, history: list[Json]) -> list[Json]:
    """Exact terminal repetition retires only its unchanged inspection, never an unmeasured hypothesis."""
    return [
        dict(
            task_id=task["id"],
            prior_task_id=prior["experiment_id"],
            exact_verdict=prior["verdict"],
            prior_reference=old["reference"],
            scope="unchanged no-new-supervisor-outcomes inspection only",
            hypothesis_retired=False,
            reopening_condition="new authenticated cross-game supervisor outcomes beyond the frozen frontier",
        )
        for prior in task.get("prior_failures", [])
        for old in history
        if old["task_id"] == prior["experiment_id"]
        and old["authenticated"]
        and current["eligible"]
        and prior.get("retire_if_same_verdict")
        and old["honest_verdict"] == prior["verdict"] == current["honest_verdict"]
    ]


def measure(root: Path, raw: Path) -> Json:
    """Freeze all fourteen intended slots once; unavailable future outputs remain dependencies."""
    progress("measurement_before", 0, 14)
    start, before = time.monotonic_ns(), memory()
    actual = contract.authority(root, raw / "authority")
    tasks = actual["tasks"]
    if (
        [t["id"].split("-")[0] for t in tasks] != [f"exp{i}" for i in range(8360, 8374)]
        or tasks[-1]["id"] != TASK
        or tasks[-1].get("gated_on")
    ):
        raise ValueError("exact_fourteen_authority")
    refs = [freeze(root / name, raw / "custody") for name in [DESIGN, STAGED, ACTIVE]]
    protocol = freeze(root / PROTOCOL, raw / "custody", PROTOCOL_PIN)
    checks = [
        gate(str(root / DESIGN), "authority.activated", True, actual["activated"]),
        gate(str(root / PROTOCOL), "protocol.sha256", PROTOCOL_PIN, protocol["sha256"]),
    ]
    inputs, receipts = [], []
    for index, task in enumerate(tasks[:-1]):
        progress("producer_before", index, 13 - index)
        declared = root / task["deliverable"]
        candidates = sorted((root / "results").glob("experiment_" + task["id"][3:7] + "_*.json"))
        path = (
            declared if declared.exists() else candidates[0] if len(candidates) == 1 else declared
        )
        primary = freeze(path, raw / "custody")
        if primary["exists"]:
            summary, receipt = invoke(task, primary, raw / "branches" / task["id"])
            receipts.append(receipt)
        else:
            summary = dict(
                disposition="absent",
                honest_verdict=None,
                verdict_class="blocked",
                producer_executed=False,
                eligible=False,
                selected={},
                checks=[gate(str(declared), "producer.exists", True, None)],
                closure=[primary],
                branch_replay=dict(status="blocked_absent_input", passed=False),
                memory=dict(passed=True),
            )
        inputs.append(dict(task=task, primary=primary, summary=summary))
        progress("producer_after", index + 1, 12 - index)
    history = []
    for number in [8359, 8350, 8351, 8355, 8356, 8358]:
        paths = sorted((root / "results").glob(f"experiment_{number}_*.json"))
        if paths:
            history.append(historical(paths[0], raw / "history"))
    after = memory()
    work = dict(
        root=str(root),
        tasks=tasks,
        authority_refs=refs,
        canonical_tasks_sha256=actual["canonical_tasks_sha256"],
        protocol_reference=protocol,
        inputs=inputs,
        history=history,
        checks=checks,
        branch_receipts=receipts,
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
        memory_measurements=dict(
            parent_before=before,
            parent_after=after,
            parent_growth_mb=after["peak_rss_mb"] - before["peak_rss_mb"],
            peak_bound_mb=1500,
            growth_bound_mb=500,
            workers=[i["summary"]["memory"] for i in inputs],
        ),
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_after", 13, 1)
    return work


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Qualification, scientific utility and deployment completion are independent conclusions."""
    auth = authority(work)
    if (
        work["tasks"] != auth["tasks"]
        or work["canonical_tasks_sha256"] != auth["canonical_tasks_sha256"]
    ):
        raise ValueError("full_task_authority")
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    owned &= (
        work["memory_measurements"]["parent_growth_mb"] <= 500
        and work["memory_measurements"]["parent_after"]["peak_rss_mb"] <= 1500
    )
    owned &= all(i["summary"]["memory"]["passed"] for i in work["inputs"])
    owned &= all(r["passed"] for r in work["branch_receipts"])
    rows = [
        dict(
            s["summary"],
            experiment_id=8360 + index,
            task_id=work["tasks"][index]["id"],
            unit_id=work["tasks"][index]["id"],
            primary_reference=s["primary"],
        )
        for index, s in enumerate(work["inputs"])
    ]
    for row in rows:
        for key in ["closure", "selected", "checks", "memory"]:
            row.pop(key)
    rows.append(
        dict(
            experiment_id=8373,
            task_id=TASK,
            unit_id=TASK,
            disposition="capstone",
            producer_executed=True,
            eligible=owned,
            honest_verdict=None,
            verdict_class="blocked",
            primary_reference=None,
        )
    )
    gates = work["checks"] + [
        g for i in work["inputs"] for g in i["summary"]["checks"] if not g["passed"]
    ]
    gates += [
        gate(
            r.get("stdout_path", "owned_validation"),
            r.get("name", "passed"),
            True,
            False,
            r.get("stdout_sha256"),
        )
        for r in receipts
        if not r["passed"]
    ]
    selected = [i["summary"]["selected"] for i in work["inputs"]]
    h1 = selected[1].get(
        "H1",
        dict(
            qualified=False, intended_count=128, qualified_count=None, status="blocked_unmeasured"
        ),
    )
    h2 = selected[1].get(
        "H2",
        dict(qualified=False, intended_count=88, qualified_count=None, status="blocked_unmeasured"),
    )
    h2 = dict(
        h2, stream_intended_count=96, retention_intended_count=32, retention_windows=[0, 32, 64, 96]
    )
    kind = (
        "disqualified"
        if not owned
        else "blocked"
        if any(
            r["disposition"] in ["absent", "conductor_pre_gate", "blocked", "disqualified"]
            for r in rows[:-1]
        )
        or not all(g["passed"] for g in work["checks"])
        else "null"
    )
    rows[-1].update(honest_verdict="complete_" + kind + "_v721_capstone", verdict_class=kind)
    return dict(
        rows=rows,
        task_dispositions=rows,
        honest_verdict=rows[-1]["honest_verdict"],
        verdict_class=kind,
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        capstone_execution_ready_score=int(owned),
        science_ready_score=int(h1["qualified"] and h2["qualified"]),
        gate_check_summary=gates,
        actual_executed_task_count=sum(r["producer_executed"] for r in rows),
        pre_gate_count=sum(r["disposition"] == "conductor_pre_gate" for r in rows),
        missing_output_count=sum(r["disposition"] == "absent" for r in rows),
        H1=h1,
        H2=h2,
        deployment_results=dict(
            guard_fallback=selected[2],
            atomic_recovery=selected[3] or None,
            actual_continuous_training=selected[4] or None,
            complete_local_service_costs=selected[5] or None,
            native_parity=selected[6] or None,
            native_service_costs=selected[7] or None,
            whole_service_benefit_proven=False,
            semantic_correctness_proven=False,
        ),
        arc_support=selected[10],
        board_obligations=dict(
            KV260="quadratic Ising k<=5 only; spline/database/transfer service acceleration unproved",
            PolarFire="authenticated Exp8259 board-local Linux CPU only; fabric unmeasured",
            GateMate=selected[12],
        ),
        historical_dispositions=work["history"],
        original_utility_dispositions=selected[1].get("original_dispositions", []),
        retirements=retire(work["tasks"][10], work["inputs"][10]["summary"], work["history"]),
        qualified_scope_decision=selected[1].get(
            "qualified_scope_decision", dict(retire_all_joint_reasoning=False)
        ),
        next_evidence_conditions=NEXT,
        three_prd_gaps=[
            dict(gap=name, closed=False, next_evidence_condition=NEXT[index])
            for name, index in [
                ("useful_verified_decisions", 1),
                ("later_learning_and_retention", 4),
                ("request_scale_deployment", 5),
            ]
        ],
    )


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Bind every terminal field to primitive evidence, including exact owned validation logs."""
    value = reduce(work, receipts)
    atomic_json(raw / "measurement.json", work)
    source = (
        work["authority_refs"]
        + [work["protocol_reference"]]
        + [r for i in work["inputs"] for r in i["summary"]["closure"]]
        + [ref for h in work["history"] for ref in h["refs"]]
    )
    publication = work.get(
        "publication",
        dict(
            gates={f"G{i}": dict(pass_=False) for i in range(1, 5)},
            paper_ready=False,
            unmet_gates=["G1", "G2", "G3", "G4"],
        ),
    )
    value.update(
        experiment_id=8373,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261010",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        no_model_load=True,
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=[
            dict(
                task_id=i["task"]["id"],
                provenance=i["summary"]["selected"].get("historical_model_provenance", []),
            )
            for i in work["inputs"]
        ],
        intended_count=14,
        completed_count=sum(r["eligible"] for r in value["rows"]),
        failed_count=sum(r["verdict_class"] == "disqualified" for r in value["rows"]),
        censored_count=sum(r["disposition"] in ["absent", "blocked"] for r in value["rows"]),
        excluded_count=value["pre_gate_count"],
        independent_count=0,
        sample_size_budget=dict(
            intended_tasks=14,
            independent_scientific_examples=0,
            timing_repetitions_are_independent=False,
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed_development_cached_observations_and_constructed_controls",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        acceptance_gates=dict(
            owned_execution=value["required_checks_passed"],
            utility_audit_qualified=value["science_ready_score"] == 1,
            semantic_benefit=False,
            whole_service_benefit=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["checks"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=[
            dict(
                phase="authenticate_reduce_frozen_producers",
                duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
            )
        ],
        random_seed=7218373,
        canonical_tasks_sha256=work["canonical_tasks_sha256"],
        source_artifact_hashes=source,
        code_config_hashes=work.get("code_refs", []),
        raw_shard_hashes=[reference(raw / "measurement.json")],
        work_reference=reference(raw / "measurement.json"),
        publication_output=str(output),
        branch_replay_receipts=work["branch_receipts"],
        memory_measurements=work["memory_measurements"],
        memory_bounds=dict(parent_peak_mb=1500, child_peak_mb=1500, growth_mb=500),
        repository_health=next(
            (h["repository_health"] for h in work["history"] if h.get("repository_health")),
            dict(
                status="V720_repository_suite_exceeded_quota; no_current_repository_wide_pass_claimed"
            ),
        ),
        next_evidence_conditions=NEXT,
        paper_ready=publication["paper_ready"],
        unmet_gates=publication["unmet_gates"],
        publication_gate=publication,
        methodology_note="Read-only aggregation authenticates fourteen full-task slots and terminal receipts. Frozen primitive utility reductions independently reproduce qualified exposed-development H1/H2 nulls under unchanged V717 thresholds. All missing deployment operations remain unmeasured; cached observations are historical, current model calls are zero, and no generator weights or device state changes. Administrative reference-oracle qualification is separate from scientific benefit.",
    )
    for i in range(1, 5):
        value["g" + str(i)] = publication["gates"]["G" + str(i)].get("pass", False)
    value["cited_upstream_artifacts"] = [
        dict(
            r,
            fields_imported=[
                "original terminal identity, immutable primitives or authority; no historical readiness inherited"
            ],
        )
        for r in source
    ]
    value["field_principles"] = {
        k: "Bind "
        + k
        + " to sealed operands; preserve missing inputs, original outcomes and separate qualification from benefit."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["field_principles"].update(
        completed_count="Qualified producer or capstone execution slots; no scientific independence implied.",
        failed_count="Original disqualified slots including this capstone when owned checks fail.",
        censored_count="Absent outputs and terminal external blocks; missing metrics stay null.",
        excluded_count="Authenticated conductor pre-gates, never executed experiments.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Fresh reductions and child reads reject even self-consistently rehashed report tampering."""
    try:
        value = json.loads(path.read_bytes())
        work = load(value["work_reference"])
        for ref in value["source_artifact_hashes"] + value["code_config_hashes"]:
            if ref["exists"]:
                checked(dict(path=ref["snapshot_path"], sha256=ref["sha256"]))
        for receipt in value["validation_receipts"] + work["branch_receipts"]:
            for stream in ["stdout", "stderr"]:
                if receipt.get(stream + "_path"):
                    checked(
                        dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                    )
            if receipt.get("stdout_path"):
                original = json.loads(
                    Path(receipt["stdout_path"]).with_suffix(".receipt.json").read_bytes()
                )
                if any(
                    original.get(k) != receipt.get(k)
                    for k in ["argv", "exit_code", "passed", "stdout_sha256", "stderr_sha256"]
                ):
                    return False
        with TemporaryDirectory(prefix="exp8373-cold-", dir="/var/tmp") as directory:
            raw = Path(directory)
            for index, item in enumerate(work["inputs"]):
                if item["primary"]["exists"]:
                    summary, receipt = invoke(
                        item["task"], item["primary"], raw / str(index), item["summary"]["closure"]
                    )
                    if not receipt["passed"] or any(
                        summary[k] != item["summary"][k]
                        for k in [
                            "disposition",
                            "honest_verdict",
                            "verdict_class",
                            "producer_executed",
                            "eligible",
                            "selected",
                            "checks",
                            "branch_replay",
                        ]
                    ):
                        return False
            actual = build(
                work, value["validation_receipts"], raw, Path(value["publication_output"])
            )
            for key in ["work_reference", "raw_shard_hashes"]:
                actual[key] = value[key]
            actual["reproducibility_checksum"] = canonical_hash(
                {k: v for k, v in actual.items() if k != "reproducibility_checksum"}
            )
            return bool(actual == value)
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
