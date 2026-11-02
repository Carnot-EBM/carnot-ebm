"""SCENARIO-REPORT-8743-INSTALLED: retain actual installed baseline and reuse frames."""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import subprocess
import time
from typing import Any
from collections import Counter
from unittest.mock import patch

from carnot.reporting import single_extraction_8743 as e
from carnot.reporting import installed_batch_measure_8702 as installed
from carnot.reporting.v709_execution import child, progress
from carnot.verify import native_nine_input_8533 as build_native

Json = dict[str, Any]


def reduce(cells: list[Json], inputs: Json) -> Json:
    """Reconstruct exact parity and per-item extraction counts from bound observations."""
    rows, counts, parity, outputs = [], [], [], {}
    expected, _ = installed.wire.numeric.batch(inputs["head"], inputs["requests"])
    expected = {r["unit_id"]: json.loads(r["response"]) for r in expected}
    for cell in cells:
        slot = cell["slot"]
        session, observed = e.load(cell["session_reference"]), e.load(cell["counts_reference"])
        config = e.load(cell["config_reference"])
        if (
            session["actual_exit"] != 0
            or observed["actual_exit"] != 0
            or observed["forbidden_imports"]
            or session["ready"]["loaded_extension"]
            != (config["extension"] if slot["arm"] == "native" else None)
            or config["head"] != inputs["head"]
            or config.get("reuse_request_features", False) != (slot["mode"] == "reuse")
            or session["observations"][-1]["shutdown"]["status"] != "shutdown"
        ):
            raise ValueError("installed_identity")
        successes = [r for r in observed["extractions"] if r["successful"]]
        multiplier = 2 if slot["mode"] == "baseline" else 1
        target_counts = Counter(
            installed.wire.numeric.canonical_hash(r["public"]) for r in inputs["requests"]
        )
        if (
            len(successes) != 64 * multiplier
            or len(successes) != len(observed["extractions"])
            or Counter(r["public_hash"] for r in successes)
            != Counter({k: v * multiplier for k, v in target_counts.items()})
        ):
            raise ValueError("extraction_count")
        wire, item_count = [], 0
        for index, obs in enumerate(session["observations"][:-1]):
            request, response = obs["request"], obs["response"]
            operands = inputs["requests"][index * slot["batch"] : (index + 1) * slot["batch"]]
            if (
                request["rows"] != operands
                or response["status"] != "ok"
                or len(response["responses"]) != len(operands)
            ):
                raise ValueError("wire_operands")
            for operand, actual, digest in zip(
                operands, response["responses"], response["feature_hashes"], strict=True
            ):
                target = expected[operand["unit_id"]]
                if (
                    actual["unit_id"] != target["unit_id"]
                    or actual["decision"] != target["decision"]
                    or abs(actual["probability"] - target["probability"]) > 1e-12
                    or digest != installed.wire.numeric.extract(operand["public"])["feature_hash"]
                ):
                    raise ValueError("original_numerical_parity")
                item_count += 1
            wire.append(
                dict(responses=response["responses"], feature_hashes=response["feature_hashes"])
            )
        if item_count != 64:
            raise ValueError("panel_count")
        outputs[slot["arm"], slot["batch"], slot["mode"]] = wire
        counts.append(
            dict(
                unit_id=slot["unit_id"],
                mode=slot["mode"],
                arm=slot["arm"],
                batch=slot["batch"],
                intended_items=64,
                successful_extractions=len(successes),
                expected_extractions=64 * multiplier,
                passed=True,
            )
        )
        rows.append(
            dict(
                slot,
                status="completed",
                passed=True,
                duration_s=session["duration_s"],
                absolute_metric=1,
                effective_independent_groups=0,
                censored=False,
                excluded=False,
            )
        )
    for arm in ["python", "native"]:
        for size in [1, 4, 16]:
            baseline, reused = (outputs[arm, size, mode] for mode in ["baseline", "reuse"])
            if json.dumps(baseline, sort_keys=True) != json.dumps(reused, sort_keys=True):
                raise ValueError("same_arm_wire_bytes")
            for obs in reused:
                for response, digest in zip(obs["responses"], obs["feature_hashes"], strict=True):
                    parity.append(
                        dict(
                            unit_id=f"{arm}-B{size}-{response['unit_id']}",
                            arm=arm,
                            batch=size,
                            response=response,
                            feature_hash=digest,
                            passed=True,
                        )
                    )
    return dict(rows=rows, feature_extraction_count_rows=counts, wire_parity_rows=parity)


def frame_control(
    command: list[str], config: Path, arm: str, payload: bytes, raw: Path, scratch: Path
) -> Json:
    """Keep full malformed-frame streams instead of losing bytes during child cleanup."""
    raw.mkdir(parents=True)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    counts = raw / "counts.json"
    argv = command + [
        "--counts-output",
        str(counts),
        str(config),
        e.reference(config)["sha256"],
        arm,
    ]
    progress("8743_before_frame_" + raw.name)
    result = subprocess.run(
        argv, input=payload, cwd=scratch, env=env, capture_output=True, timeout=30
    )
    (raw / "stdout").write_bytes(result.stdout)
    (raw / "stderr").write_bytes(result.stderr)
    stream, frames = result.stdout, []
    while stream:
        size = struct.unpack("!I", stream[:4])[0]
        frames.append(json.loads(stream[4 : 4 + size]))
        stream = stream[4 + size :]
    malformed_json = payload == struct.pack("!I", 1) + b"{"
    passed = (
        result.returncode == (0 if malformed_json else 1)
        and frames[0]["status"] == "ready"
        and (
            len(frames) == 2
            and frames[1]["status"] == "error"
            and not {"response", "responses", "feature_hashes"} & frames[1].keys()
            if malformed_json
            else len(frames) == 1
        )
    )
    progress("8743_after_frame_" + raw.name)
    return dict(
        unit_id=raw.name,
        passed=passed,
        actual_exit=result.returncode,
        frames=frames,
        stdout_reference=e.reference(raw / "stdout"),
        stderr_reference=e.reference(raw / "stderr"),
        counts_reference=e.reference(counts),
    )


def measure(raw: Path, scratch: Path, inputs: Json, *, deadline_ns: int = 2**63) -> Json:
    """Build once and stop prototype execution before the reserved validation window."""
    raw.mkdir(parents=True)
    deadline_ns = min(deadline_ns, time.monotonic_ns() + 2400 * 10**9)
    with patch.dict(os.environ):
        os.environ.pop("CARNOT_8533_EXTENSION", None)
        _, native = build_native.extension(scratch, raw / "native-build")
    e.atomic_json(raw / "native_build.json", native)
    e.atomic_json(raw / "frozen-head.json", inputs["head"])
    operands = dict(inputs, extension=native)
    (scratch / "installed").mkdir()
    command, baseline, receipts, identities = installed.prepare(
        raw / "installed", scratch / "installed", operands
    )
    command[-1] = "carnot.inference.resident_reuse_probe_8743"
    configs = {}
    for mode in ["baseline", "reuse"]:
        path = raw / (mode + ".json")
        e.atomic_json(
            path, dict(e.load(e.reference(baseline)), reuse_request_features=mode == "reuse")
        )
        configs[mode] = path
    e.atomic_json(
        raw / "method.json",
        dict(roster=e.roster(), batches=[1, 4, 16], panel=inputs["requests"], timing_claim=False),
    )
    cells, traces, rejects = [], [], []
    for slot in e.roster():
        if time.monotonic_ns() + 90 * 10**9 >= deadline_ns:
            raise TimeoutError("prototype_stop_deadline")
        folder = raw / slot["unit_id"]
        count_path = raw / (slot["unit_id"] + "-counts.json")
        config = configs[slot["mode"]]
        frames = [
            dict(operation="batch", rows=inputs["requests"][i : i + slot["batch"]])
            for i in range(0, 64, slot["batch"])
        ]
        installed.session(
            command + ["--counts-output", str(count_path)],
            config,
            slot["arm"],
            frames,
            scratch,
            folder,
            deadline_ns=deadline_ns,
        )
        cell = dict(
            slot=slot,
            session_reference=e.reference(folder / "session.json"),
            counts_reference=e.reference(count_path),
            config_reference=e.reference(config),
        )
        cells.append(cell)
        traces += [
            cell["session_reference"],
            cell["counts_reference"],
            cell["config_reference"],
            e.load(cell["session_reference"])["stderr_reference"],
        ]
        progress("8743_cells", len(cells), 12 - len(cells))
    for arm in ["python", "native"]:
        for mode in ["baseline", "reuse"]:
            row = inputs["requests"][0]
            good = dict(operation="batch", rows=[row], request_id="duplicate")
            frames = [good, good] + [
                dict(operation="batch", rows=items)
                for items in [
                    [],
                    [row] * 17,
                    [row, row],
                    [row, dict(row, unit_id="invalid", q=-1)],
                    [row, dict(row, unit_id="invalid", public={})],
                    None,
                ]
            ]
            frames += [
                dict(operation="batch", rows=[row], config_hash="changed"),
                dict(operation="batch", rows=[row], deadline_ns=1),
            ]
            config = configs[mode]
            folder = raw / f"{arm}-{mode}-invalid"
            count_path = raw / f"{arm}-{mode}-invalid-counts.json"
            observation = installed.session(
                command + ["--counts-output", str(count_path)], config, arm, frames, scratch, folder
            )
            traces += [
                e.reference(folder / "session.json"),
                e.reference(count_path),
                observation["stderr_reference"],
            ]
            for i, obs in enumerate(observation["observations"][:-1]):
                response = obs["response"]
                passed = response["status"] == ("ok" if i == 0 else "error") and (
                    i == 0 or not {"response", "responses", "feature_hashes"} & response.keys()
                )
                rejects.append(
                    dict(unit_id=f"{arm}-{mode}-invalid-{i}", passed=passed, response=response)
                )
            for i, payload in enumerate(
                [
                    struct.pack("!I", 1) + b"{",
                    b"\0",
                    struct.pack("!I", 65537),
                    struct.pack("!I", 5) + b"x",
                    struct.pack("!I", 0),
                ]
            ):
                rejects.append(
                    frame_control(
                        command, config, arm, payload, raw / f"{arm}-{mode}-frame-{i}", scratch
                    )
                )
    reduced = reduce(cells, inputs)
    mutation = installed_controls(command[0], scratch, raw)
    traces += [
        e.reference(raw / p) for p in ["method.json", "frozen-head.json", "native_build.json"]
    ] + [
        r[k]
        for r in rejects
        for k in ["stdout_reference", "stderr_reference", "counts_reference"]
        if k in r
    ]
    frozen = dict(
        wheel=identities["installed_wheel_reference"],
        extension=identities["loaded_extension_reference"],
        source_references=[
            e.reference(e.ROOT / p) for p in e.OWNED + e.MODIFIED + build_native.SOURCES
        ],
        head=e.reference(raw / "frozen-head.json"),
        native_build_reference=e.reference(raw / "native_build.json"),
        configuration_templates={m: e.load(e.reference(p)) for m, p in configs.items()},
        requests=inputs["requests"],
        entry=command[-1],
    )
    return dict(
        reduced,
        **identities,
        passed=all(r["passed"] for r in reduced["rows"] + rejects + mutation),
        receipts=receipts + native["receipts"],
        traces=traces,
        cell_primitives=cells,
        invalid_input_rows=rejects,
        ownership_mutation_rows=mutation,
        original_import_contract_rows=[
            r for r in receipts if r["name"] == "installed_public_contract"
        ],
        frozen_8744_workload=frozen,
    )


def installed_controls(python: str, scratch: Path, raw: Path) -> list[Json]:
    """Run the original mutation assertions against installed imports in a reaped child."""
    source = e.ROOT / e.TESTS[0]
    code = (
        "import os,runpy,tempfile,json;from pathlib import Path;import pytest;"
        f"os.chdir({str(scratch)!r});s=runpy.run_path({str(source)!r});"
        "h=s['head'].__wrapped__();r=s['row'].__wrapped__();"
        "s['test_immutable_ownership'](h,r);"
        "s['test_abstention_preserved'](h,s['row'].__wrapped__());"
        "cases=['ids','mixed','public','q_nan','q_inf','too_many','empty','fields','config','deadline','operation'];"
        "[s['test_fresh_identical_errors'](h,s['row'].__wrapped__(),c) for c in cases];"
        "context=pytest.MonkeyPatch();"
        "[s['test_nonfinite_features'](h,s['row'].__wrapped__(),context,v) for v in [float('nan'),float('inf'),-float('inf')]];"
        "context.undo();"
        "s['test_changed_configuration'](h,r,Path(tempfile.mkdtemp(dir="
        + repr(str(scratch))
        + ")),context);"
        "context.undo();"
        "print(json.dumps([dict(unit_id=c,passed=True) for c in ['ownership','abstention','nonfinite_features','head_extension_config']+cases]))"
    )
    receipt = child(
        "installed_mutations",
        ["env", "-u", "PYTHONPATH", python, "-u", "-c", code],
        raw,
        deadline=120,
    )
    if not receipt["passed"]:
        raise ValueError("installed_mutation_controls")
    rows: list[Json] = json.loads(Path(receipt["stdout_path"]).read_text().splitlines()[-1])
    return [dict(r, receipt=receipt) for r in rows]
