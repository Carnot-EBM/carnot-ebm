"""REQ-VERIFY-8245: qualify transport bytes without claiming a learning benefit.

Current learner qualification supplies one explicitly selected state. A static
fallback remains a mechanics fixture and retains the unavailable learner gate.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from carnot.reporting import polarfire_packet_evaluator_8245 as e
from carnot.reporting import polarfire_state_boundary_8231 as old
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.recorder_execution_8213 import execute
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.request_trace_inventory_8200 import (
    copy_bytes as copy_bytes,
    operand as operand,
)

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8245_v712_polarfire_state_dispatch"
CLI = "scripts/experiments/" + NAME + ".py"
EVALUATOR = "python/carnot/reporting/polarfire_packet_evaluator_8245.py"
PINS = {
    8240: "sha256:5dcfe543abe6e6c28dd108d188e7771269adab2478f0a10a9d11f1770e40ac67",
    8221: "sha256:5ff6f09312a81ce95b752d38dec94d8a1c57bbf894cde785ebe7494ca7c639e5",
}
SSH = ("ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", "polarfire")


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts keep bounded work visible to the parent supervisor."""
    print(f"[exp8245] phase={phase} completed={completed} pending={pending}", flush=True)


def load(root: Path, raw: Path) -> Json:
    """Authenticate producer identity, qualification and terminal bytes before copying state."""
    data: Json = dict(checks=[], references=[], cited=[], learner_block=[], state=None, queries=[])
    for eid, suffix, score in [
        (8240, "v712_qualified_delayed_learning", "utility_trajectory_ready_score"),
        (8221, "v711_utility_kernel", "static_kernel_ready_score"),
    ]:
        progress("authenticate_before_" + str(eid))
        path = root / "results" / f"experiment_{eid}_{suffix}.json"
        begin = len(data["checks"])
        data["checks"].append(operand("exists", path, True, path.is_file()))
        try:
            value = json.loads(path.read_bytes())
            data["checks"].append(operand("object_schema", path, True, isinstance(value, dict)))
            for field, expected, actual in [
                ("sha256", PINS[eid], e.digest(path.read_bytes())),
                ("experiment_id", eid, value.get("experiment_id")),
                (
                    "task_id",
                    "exp8240-qualified-delayed-learning"
                    if eid == 8240
                    else "exp8221-utility-kernel",
                    value.get("task_id"),
                ),
                (score, 1, value.get(score)),
                ("required_checks_passed", True, value.get("required_checks_passed")),
                ("flagged_adversarial", False, value.get("flagged_adversarial")),
                (
                    "qualified_verdict",
                    True,
                    value.get("verdict_class") in {"null", "circular_positive"},
                ),
                ("fixture_mode", False, bool(value.get("fixture_mode"))),
            ]:
                data["checks"].append(operand(field, path, expected, actual))
            if not all(c["passed"] for c in data["checks"][begin:]):
                raise ValueError("producer_qualification")
            terminal = Path(value["terminal_validation_sidecar_path"])
            publication = json.loads(terminal.read_bytes())["publication"]
            bound = Path(publication["sidecar_path"])
            report = read_bound_sidecar(path, bound)
            data["checks"].append(
                operand("terminal.report.passed", bound, True, report["report"]["passed"])
            )
            if report["report"]["passed"] is not True:
                raise ValueError("terminal_qualification")
            refs = [
                reference(path),
                reference(terminal),
                reference(bound),
                *value["code_config_hashes"],
            ]
            for index, receipt in enumerate(value["validation_receipts"]):
                for field in ["passed", "normal_exit"]:
                    data["checks"].append(
                        operand(f"validation_receipts.{index}.{field}", path, True, receipt[field])
                    )
                if not receipt["passed"] or not receipt["normal_exit"]:
                    raise ValueError("validation_receipt")
                refs += [
                    dict(path=receipt[s + "_path"], sha256=receipt[s + "_sha256"])
                    for s in ["stdout", "stderr"]
                ]
            query_ref = next(
                r for r in value["raw_shard_hashes"] if Path(r["path"]).name == "restart-input.json"
            )
            state_ref = (
                next(
                    r for r in value["raw_shard_hashes"] if r["path"] == value["final_states_path"]
                )
                if eid == 8240
                else value["measurement_reference"]
            )
            refs += [query_ref, state_ref]
            for ref in refs:
                source = Path(ref["path"])
                data["checks"].append(
                    operand(
                        "sha256",
                        source,
                        ref["sha256"],
                        e.digest(source.read_bytes()) if source.is_file() else None,
                    )
                )
                data["references"].append(copy_bytes(checked(ref), raw))
            queries = json.loads(checked(query_ref).read_bytes())["rows"]
            saved = json.loads(checked(state_ref).read_bytes())
            state = (
                saved[0]["local_only"]
                if eid == 8240
                else dict(schema_version=1, model=saved["static"]["model"])
            )
            old.encode(state)
            data["checks"].append(
                operand("queries.count", checked(query_ref), 1, len(queries), ">=")
            )
            if not queries:
                raise ValueError("empty_queries")
            data.update(
                state=state,
                queries=queries,
                state_origin=dict(
                    kind="qualified_current_learner" if eid == 8240 else "qualified_static_fixture",
                    arm="local_only" if eid == 8240 else "static",
                    seed=state.get("seed"),
                    producer=reference(path),
                    state_reference=state_ref,
                    query_reference=query_ref,
                    selection="first seed local_only; every query"
                    if eid == 8240
                    else "static model; every query",
                    mechanics_only=eid == 8221,
                ),
            )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            StopIteration,
            IndexError,
        ) as error:
            if all(c["passed"] for c in data["checks"][begin:]):
                data["checks"].append(
                    operand(
                        "required_input_schema_and_hash",
                        path,
                        "qualified authenticated primitives",
                        str(error),
                    )
                )
            if eid == 8240:
                data["learner_block"] = data["checks"][begin:]
        data["cited"].append(
            dict(
                path=str(path),
                sha256=e.digest(path.read_bytes()) if path.is_file() else None,
                imported_fields=[
                    score,
                    "verdict_class",
                    "validation_receipts",
                    "final_states_path",
                    "measurement_reference",
                    "raw_shard_hashes",
                    "terminal_validation_sidecar_path",
                ],
            )
        )
        progress("authenticate_after_" + str(eid), 1, 0)
        if data["state"] is not None:
            break
    return data


def expected(data: Json) -> Json:
    """Existing host arithmetic supplies the reference independently of the receiver."""
    predictions = old.predictions(data["state"], data["queries"])
    return dict(
        state_sha256=e.digest(old.encode(data["state"])),
        query_sha256=e.digest(e.canonical(data["queries"])),
        output_sha256=e.digest(e.canonical(predictions)),
        predictions=predictions,
    )


def packet(data: Json, raw: Path) -> Json:
    """Freeze exact evaluator and full state, preserving missing query rows."""
    result = dict(
        schema=e.SCHEMA,
        state_bytes=old.encode(data["state"]).decode(),
        query_bytes=e.canonical(data["queries"]).decode(),
        evaluator_sha256=e.digest((ROOT / EVALUATOR).read_bytes()),
        **{k: v for k, v in expected(data).items() if k != "predictions"},
    )
    result["expected_output_sha256"] = result.pop("output_sha256")
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "evaluator.py").write_bytes((ROOT / EVALUATOR).read_bytes())
    atomic_json(raw / "packet.json", result)
    return result


def board(raw: Path) -> Json:
    """One bounded contact retains absent evidence and only removes task-owned scratch."""
    result: Json = dict(ready=False, executed=False, output=None, block=None, receipts=[])
    transcript = raw / "board_transcript.json"

    def call(name: str, argv: tuple[str, ...]) -> Json:
        receipts = execute([CommandSpec(name, argv, "board_preflight_or_dispatch", 60)], raw / name)
        result["receipts"] += receipts
        receipt: Json = receipts[0]
        if not receipt["passed"]:
            result["block"] = operand(
                name + ".actual_exit", Path(receipt["stdout_path"]), 0, receipt["actual_exit"]
            )
        atomic_json(transcript, result)
        return receipt

    if not call("board_ssh", (*SSH, "true"))["passed"]:
        return result
    probe = (
        "python3 -c 'import pathlib,tempfile,sys; assert sys.version_info >= (3,10); "
        'p=pathlib.Path(tempfile.mkdtemp(prefix="carnot8245-",dir="/tmp")); '
        'p.chmod(0o700); (p/"probe").write_bytes(b"writable"); '
        'assert (p/"probe").read_bytes()==b"writable"; print(p)\''
    )
    receipt = call("board_python_scratch", (*SSH, probe))
    if not receipt["passed"]:
        return result
    remote = Path(receipt["stdout_path"]).read_text().strip()
    if re.fullmatch(r"/tmp/carnot8245-[A-Za-z0-9_-]+", remote) is None:
        result["block"] = operand(
            "private_remote_directory",
            Path(receipt["stdout_path"]),
            "/tmp/carnot8245-<private token>",
            remote,
        )
        atomic_json(transcript, result)
        return result
    try:
        receipt = call(
            "board_transfer",
            (
                "scp",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "BatchMode=yes",
                str(raw / "packet.json"),
                str(raw / "evaluator.py"),
                "polarfire:" + remote + "/",
            ),
        )
        if receipt["passed"]:
            result["executed"] = True
            receipt = call(
                "board_evaluate", (*SSH, f"python3 -u {remote}/evaluator.py {remote}/packet.json")
            )
            if receipt["passed"]:
                try:
                    result["output"] = json.loads(Path(receipt["stdout_path"]).read_bytes())
                    result["ready"] = True
                except ValueError as error:
                    result["block"] = operand(
                        "board_output_schema",
                        Path(receipt["stdout_path"]),
                        "JSON output",
                        str(error),
                    )
    finally:
        if not call("board_cleanup", (*SSH, "rm -rf -- " + remote))["passed"]:
            result["ready"] = False
        atomic_json(transcript, result)
    return result


def reduce(data: Json, host: Json | None, device: Json, owned: bool) -> Json:
    """Readiness certifies host validation, while device parity and benefit stay separate."""
    schema_ready = data["state"] is not None
    wanted = expected(data) if schema_ready else None
    parity = schema_ready and host == wanted
    device_parity = device["output"] is not None and device["output"] == wanted
    ready = bool(owned and parity)
    blockers = ([c for c in data["checks"] if not c["passed"]] if not schema_ready else []) + (
        [device["block"]] if device["block"] else []
    )
    kind = (
        "disqualified"
        if not owned or schema_ready and not parity or device["ready"] and not device_parity
        else "blocked"
        if blockers
        else "circular_positive"
        if device_parity or data.get("fixture")
        else "null"
    )
    ready = ready and kind != "disqualified"
    rows = []
    for row in data["queries"]:
        for condition, done, success in [
            ("host_parity", host is not None, bool(parity)),
            ("board_parity", bool(device["executed"]), device_parity),
        ]:
            rows.append(
                dict(
                    unit_id=row["unit_id"],
                    source_cluster_id=row["source_cluster_id"],
                    arm=data["state_origin"]["arm"],
                    condition=condition,
                    metric="exact_output_and_state_hash_parity",
                    numerator=int(success) if done else None,
                    denominator=1,
                    status="completed" if done and success else "failed" if done else "excluded",
                    probability_missing=row["p"] is None,
                    exclusion_reason=blockers if not done else None,
                )
            )
    if not rows:
        rows = [
            dict(
                unit_id="required_state",
                source_cluster_id="unavailable",
                arm="unavailable",
                condition="state_operand",
                metric="state_available",
                numerator=None,
                denominator=1,
                status="excluded",
                exclusion_reason=blockers,
            )
        ]
    return dict(
        honest_verdict="complete_"
        + kind
        + "_"
        + (
            str(blockers[0]["artifact_field"]).replace(".", "_")
            if kind == "blocked"
            else "polarfire_state_dispatch"
        ),
        verdict_class=kind,
        polarfire_validation_ready_score=int(ready),
        polarfire_workload_validated=bool(owned and device_parity),
        current_device_execution_count=int(device["executed"]),
        schema_validation_ready=schema_ready,
        host_parity=bool(parity),
        board_output_hashes={
            k: v for k, v in (device["output"] or {}).items() if k != "predictions"
        },
        host_expected_hashes={k: v for k, v in (wanted or {}).items() if k != "predictions"},
        rows=rows,
        intended_count=len(rows),
        **{
            s + "_count": sum(r["status"] == s for r in rows)
            for s in ["completed", "failed", "excluded"]
        },
        censored_count=0,
        independent_count=len({r["source_cluster_id"] for r in rows if r["status"] == "completed"}),
        verifier_is_oracle=True,
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        acceptance_gates=dict(
            schema=schema_ready,
            host_parity=bool(parity),
            owned_checks=owned,
            board_parity=device_parity,
            scientific_benefit=False,
        ),
        polarfire_obligation=dict(
            actual_substrate="PolarFire Linux CPU",
            fabric_acceleration=False,
            speedup=None,
            learning_benefit=None,
            external_blocks=blockers,
            learner_blocks=data["learner_block"],
            retained=[
                "durable activation and restart",
                "transfer-inclusive performance",
                "independent benefit",
            ],
        ),
    )
