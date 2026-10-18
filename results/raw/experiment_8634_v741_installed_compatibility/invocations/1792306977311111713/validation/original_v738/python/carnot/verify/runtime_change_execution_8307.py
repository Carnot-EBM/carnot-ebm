"""REQ-REPORT-8307: retain checks and publish an authenticated runtime boundary.

The existing supervisor provides deadlines and owned process cleanup. Private
fixtures exercise publication without claiming a repaired GPU or model call.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

import yaml  # type: ignore[import-untyped]

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.primary_publication import publish_primary, validate_primary
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v709_execution import execute
from carnot.reporting.v710_contract_replay import snapshot, require_reference
from carnot.verify import runtime_change_boundary_8307 as q
from carnot.verify import cuda_primitive_8290 as cuda_probe

Json = dict[str, Any]
TEST = "tests/python/test_runtime_change_boundary_8307.py"
OWNED = [
    "python/carnot/verify/runtime_change_boundary_8307.py",
    "python/carnot/verify/runtime_change_execution_8307.py",
    q.CLI,
]


def checksum(value: Json) -> str:
    """Exclude the checksum itself so byte identity has one stable definition."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def authority(root: Path, raw: Path) -> Json:
    """Bind the complete activated task to its declared deliverable and design bytes."""
    result: Json = dict(refs=[], checks=[], task={}, activated=False)
    try:
        paths = [
            root / "openspec/change-proposals/research-roadmap-vNEXT.md",
            root / "research-roadmap.yaml",
        ]
        result["refs"] = [snapshot(p, raw / "authority", p.stem) for p in paths]
        _, tasks = parse_design(paths[0].read_text(), milestone="2026.10.717")
        active = yaml.safe_load(paths[1].read_bytes())
        task = next(t for t in tasks if t["id"] == q.TASK)
        actual = next(t for t in active["tasks"] if t["id"] == q.TASK)
        result["checks"] = [
            q.gate(paths[1], "full_activated_task", task, actual),
            q.gate(paths[1], "milestone", "2026.10.717", active.get("milestone")),
        ]
        result.update(task=task, activated=all(c["passed"] for c in result["checks"]))
    except (OSError, ValueError, KeyError, TypeError, StopIteration, yaml.YAMLError) as error:
        result["checks"].append(
            q.gate(root / "research-roadmap.yaml", "execution_authority", True, str(error))
        )
    return result


def plan(private: Path) -> list[Json]:
    """Freeze only owned coverage and affected private consumers; omit broad repository work."""
    private.mkdir(parents=True, exist_ok=True)
    config = private / "coverage.ini"
    config.write_text(
        f"[run]\nparallel = True\ndata_file = {private / '.coverage'}\n[report]\nexclude_lines =\n"
    )
    py = str(q.ROOT / ".venv/bin/python")
    pytest = [str(q.ROOT / ".venv/bin/pytest"), "-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    specs = [
        (
            "tools_and_private_scratch",
            [
                py,
                "-c",
                "import pytest,coverage,ruff,mypy,shutil,pathlib,sys;p=pathlib.Path(sys.argv[1]);p.write_bytes(b'private');assert p.read_bytes()==b'private';assert shutil.disk_usage(p.parent).free>10000000;print(sys.version)",
                str(private / "probe"),
            ],
            15,
        ),
        (
            "owned_coverage",
            [
                "/usr/bin/env",
                "CARNOT8307_COVERAGE_CONFIG=" + str(config),
                py,
                "-m",
                "coverage",
                "run",
                "-p",
                "--rcfile",
                str(config),
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                TEST,
                "--basetemp=" + str(private / "owned"),
            ],
            180,
        ),
        (
            "coverage_combine",
            [py, "-m", "coverage", "combine", "--rcfile", str(config), "--keep"],
            20,
        ),
        (
            "coverage_json",
            [
                py,
                "-m",
                "coverage",
                "json",
                "--rcfile",
                str(config),
                "--include=" + ",".join(str(q.ROOT / p) for p in OWNED),
                "--fail-under=100",
                "-o",
                str(private / "coverage.json"),
            ],
            20,
        ),
        (
            "e2e_018_private",
            [
                *pytest,
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                "--basetemp=" + str(private / "authority_cases"),
            ],
            180,
        ),
        (
            "runtime_and_publication_consumers",
            [
                *pytest,
                "tests/python/test_runtime_localization_8290.py",
                "tests/python/test_primary_publication_7928.py",
                "--basetemp=" + str(private / "consumers"),
            ],
            180,
        ),
        ("ruff_check", [str(q.ROOT / ".venv/bin/ruff"), "check", *OWNED, TEST], 20),
        ("ruff_format", [str(q.ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, TEST], 20),
        (
            "strict_mypy",
            [
                str(q.ROOT / ".venv/bin/mypy"),
                "--strict",
                "--follow-imports=silent",
                "--config-file=/dev/null",
                *OWNED,
            ],
            60,
        ),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", "--files", TEST], 20),
    ]
    return [
        dict(
            name=name,
            argv=argv,
            deadline=deadline,
            expected=0,
            scope="external_preconditions" if name == "tools_and_private_scratch" else "owned",
        )
        for name, argv, deadline in specs
    ]


def validate(path: Path, raw: Path) -> Json:
    """Run unchanged terminal auditors on a private candidate's actual bytes."""
    py = str(q.ROOT / ".venv/bin/python")
    commands = [
        ("cold_replay", [py, "-u", str(q.ROOT / q.CLI), "--cold-replay", str(path)]),
        ("adversarial", [py, "scripts/adversarial_verify.py", "--json", str(path)]),
        ("strict_rows", [py, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)]),
    ]
    receipts = execute(
        [dict(name=n, argv=a, deadline=60, expected=0, scope="terminal") for n, a in commands], raw
    )
    return dict(passed=all(r["passed"] for r in receipts), checks=receipts)


def replay(path: Path) -> Json:
    """A rehashed headline must still equal a fresh reduction of authenticated primitives."""
    value = json.loads(path.read_bytes())
    if value["reproducibility_checksum"] != checksum(value):
        raise ValueError("candidate_checksum")
    if (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"]) != (
        8307,
        q.TASK,
        "2026.10.717",
        "20261008",
    ):
        raise ValueError("invocation_identity")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        require_reference(ref)
    data = json.loads(checked(value["primitive_reference"]).read_bytes())
    q.verify_primitives(data)
    for receipt in value["validation_receipts"]:
        for prefix in ("stdout", "stderr"):
            checked(dict(path=receipt[prefix + "_path"], sha256=receipt[prefix + "_sha256"]))
    passed = all(
        r["passed"] for r in value["validation_receipts"] if r["scope"] != "external_preconditions"
    )
    if passed != value["required_checks_passed"]:
        raise ValueError("owned_check_drift")
    for key, expected in q.reduce(data, passed).items():
        if value[key] != expected:
            raise ValueError("primitive_reduction_drift:" + key)
    if value["MODEL_SPECS"] or value["model_invocation_counts"] != ZERO_INVOCATION_COUNTS:
        raise ValueError("current_model_provenance")
    return dict(
        passed=True, replay_passed=True, cuda_context_ready_score=value["cuda_context_ready_score"]
    )


def main(argv: list[str] | None = None) -> int:
    """Complete one audit; private fixtures can never become the live primary."""
    q.progress("start_no_model_load", 0, 1)
    args_list = list(sys.argv[1:] if argv is None else argv)
    if "--cuda-probe" in args_list:
        cuda_probe.main(args_list)
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261008"], default="20261008")
    parser.add_argument("--root", type=Path, default=q.ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(args_list)
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or args.root / "results" / (q.NAME + ".json")).absolute()
        if args.input and output.resolve().is_relative_to((q.ROOT / "results").resolve()):
            raise ValueError("private_fixture_requires_private_output")
        began, wall = time.monotonic_ns(), time.time_ns()
        raw = output.parent / "raw" / q.NAME / "invocations" / str(wall)
        raw.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="carnot8307-") as directory:
            private = Path(directory)
            private.chmod(0o700)
            candidate = private / (q.NAME + ".json")
            commands = [] if args.input else plan(private / "checks")
            manifest = raw / "validation_commands.json"
            atomic_json(
                manifest,
                dict(
                    commands=commands,
                    private_candidate=str(candidate),
                    frozen_before_measurement=True,
                    terminal_argv=[
                        [
                            str(q.ROOT / ".venv/bin/python"),
                            "-u",
                            str(q.ROOT / q.CLI),
                            "--cold-replay",
                            str(candidate),
                        ],
                        [
                            str(q.ROOT / ".venv/bin/python"),
                            "scripts/adversarial_verify.py",
                            "--json",
                            str(candidate),
                        ],
                        [
                            str(q.ROOT / ".venv/bin/python"),
                            "scripts/verdict_row_consistency_lint.py",
                            "--strict",
                            str(candidate),
                        ],
                    ],
                    negative_argv=[
                        [
                            str(q.ROOT / ".venv/bin/python"),
                            "-u",
                            str(q.ROOT / q.CLI),
                            "--cold-replay",
                            str(private / (label + ".json")),
                        ]
                        for label in ("negative_checksum", "rehashed_tamper")
                    ],
                ),
            )
            receipts = execute(commands[:1], raw / "preconditions")
            q.progress("preconditions_and_authority_before")
            auth = (
                dict(refs=[], checks=[], task={}, activated=False)
                if args.input
                else authority(args.root, raw)
            )
            data = (
                json.loads(args.input.read_bytes())
                if args.input
                else q.load(
                    args.root,
                    raw,
                    observe=bool(auth["activated"]) and all(r["passed"] for r in receipts),
                )
            )
            data["checks"] += [
                q.gate(Path(r["stdout_path"]), "required_tools.normal_exit", True, r["passed"])
                for r in receipts
                if not r["passed"]
            ]
            if args.input:
                data["fixture"] = True
                data["refs"].append(snapshot(args.input, raw / "inputs", "private_fixture"))
            data["checks"].extend(auth["checks"])
            data["refs"].extend(auth["refs"])
            data["boundary_path"] = str(raw / "primitive_rows.json")
            q.progress("preconditions_and_authority_after")
            measured_start = time.monotonic_ns()
            if (
                not args.input
                and all(c["passed"] for c in data["checks"])
                and q.reduce(data, True)["reopen_contract"]["eligible"]
            ):
                data["diagnostic"] = q.probe_changed(data, raw / "changed_cuda")
            binding = raw / "runtime_binding.json"
            atomic_json(binding, data["current"])
            data["current_reference"] = reference(binding)
            primitive = raw / "primitive_rows.json"
            atomic_json(primitive, data)
            q.progress("validation_before")
            receipts += execute(commands[1:], raw / "validation")
            passed = all(
                r["passed"] and r["normal_exit"]
                for r in receipts
                if r["scope"] != "external_preconditions"
            )
            value = q.reduce(data, passed)
            ended = time.monotonic_ns()
            shards = [reference(primitive), reference(binding), reference(manifest)]
            coverage_path = private / "checks/coverage.json"
            coverage: Json = {}
            if coverage_path.is_file():
                coverage = json.loads(coverage_path.read_bytes())
                for path in (private / "checks").glob(".coverage*"):
                    shards.append(snapshot(path, raw / "coverage", path.name))
                shards.append(snapshot(coverage_path, raw / "coverage", "report"))
            terminal = raw / "terminal_validation.json"
            value.update(
                experiment_id=8307,
                task_id=q.TASK,
                milestone="2026.10.717",
                run_date=args.date,
                invocation=dict(
                    pid=os.getpid(),
                    argv=args_list,
                    started_wall_ns=wall,
                    started_monotonic_ns=began,
                    ended_monotonic_ns=ended,
                ),
                execution_authority=auth,
                inference_substrate="aggregation_from_upstream_artifacts",
                inference_substrate_class="no_model_load",
                no_model_load=True,
                MODEL_SPECS=[],
                model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
                model_invoked=False,
                historical_model_provenance=dict(
                    exp8290_primary_sha256=q.PIN,
                    current_calls_imported=False,
                    failure_layers=data["historical"].get("failure_layer", []),
                ),
                verifier_is_oracle=False,
                exposure_scope="exposed development: historical runtime receipts; private controls are constructed mechanics",
                independent_generalization_score=0,
                generalized_learning_benefit_score=0,
                required_checks_passed=passed,
                flagged_adversarial=False,
                validation_receipts=receipts,
                preconditions_checked=True,
                precondition_receipts=receipts[:1],
                observation_receipts=data["observation_receipts"],
                terminal_validation_sidecar_path=str(terminal),
                duration_s=(ended - began) / 1e9,
                phase_spans=[
                    dict(
                        phase=n,
                        started_monotonic_ns=s,
                        ended_monotonic_ns=t,
                        duration_s=(t - s) / 1e9,
                    )
                    for n, s, t in [
                        ("preconditions", began, measured_start),
                        ("audit_and_validation", measured_start, ended),
                    ]
                ],
                random_seed=7178307,
                source_artifact_hashes=data["refs"],
                code_config_hashes=[reference(q.ROOT / p) for p in OWNED + [TEST]],
                raw_shard_hashes=shards,
                primitive_reference=reference(primitive),
                runtime_binding_path=str(binding),
                runtime_binding_sha256=sha256_file(binding),
                fixture_mode=bool(args.input),
                owned_statement_coverage=coverage.get("totals"),
                owned_statement_counts=coverage.get("files", {}),
                cited_upstream_artifacts=[
                    dict(
                        path=str(args.root / q.UPSTREAM),
                        sha256=q.PIN,
                        fields_imported=[
                            "failure_layer",
                            "device_inventory",
                            "runtime_binding_path",
                            "lease_receipt",
                            "work_reference",
                        ],
                    )
                ],
                methodology="Byte-authenticated read-only runtime identity comparison; unchanged conditions prohibit probes. No model or generalization measurement.",
            )
            value["field_principles"] = {
                k: "Bind the current invocation and its primitive evidence; no independent science claim follows."
                for k in value
            }
            value["field_principles"].update(
                runtime_changed_score="Only an authenticated causal operand delta permits a new attempt.",
                cuda_context_ready_score="Actual driver initialization, context and32-byte parity plus native compatibility and owned checks are required.",
                current_probe_count="Zero denotes no attempted probe, not a measured CUDA result.",
                environment_delta_rows="Preserve every changed, unchanged and unavailable operand.",
                runtime_binding_sha256="Exp8313 must revalidate these exact operands immediately before load.",
            )
            value["field_principles"]["reproducibility_checksum"] = (
                "Bind all recorded fields while excluding the checksum itself."
            )
            value["reproducibility_checksum"] = checksum(value)
            validate_primary(value, output)
            atomic_json(candidate, value)
            controls = []
            for label in ("negative_checksum", "rehashed_tamper"):
                bad = deepcopy(value)
                bad["runtime_changed_score"] = 1 - bad["runtime_changed_score"]
                bad["reproducibility_checksum"] = (
                    checksum(bad) if label == "rehashed_tamper" else "sha256:invalid"
                )
                path = private / (label + ".json")
                atomic_json(path, bad)
                controls.append(
                    dict(
                        name=label,
                        argv=[
                            str(q.ROOT / ".venv/bin/python"),
                            "-u",
                            str(q.ROOT / q.CLI),
                            "--cold-replay",
                            str(path),
                        ],
                        deadline=60,
                        expected=1,
                        scope="terminal_negative",
                    )
                )
            negative = execute(controls, raw / "cold_controls")
            report = validate(candidate, raw / "terminal")
            report["negative_controls"] = negative
            report["passed"] &= all(r["passed"] for r in negative)
            if not report["passed"]:
                value.update(
                    required_checks_passed=False,
                    cuda_context_ready_score=0,
                    honest_verdict="complete_disqualified_owned_checks",
                    verdict_class="disqualified",
                )
                atomic_json(raw / "failed_terminal_candidate.json", value)
                atomic_json(
                    terminal,
                    dict(normal_process_exit=True, report=report, owned_checks_passed=False),
                )
                return 1
            atomic_json(raw / "checked_private_candidate.json", value)
            publication = publish_primary(
                output,
                value,
                lambda p: (
                    report if sha256_file(p) == sha256_file(candidate) else dict(passed=False)
                ),
            )
            atomic_json(
                terminal,
                dict(
                    normal_process_exit=True,
                    owned_checks_passed=passed,
                    publication=publication,
                    report=report,
                ),
            )
            q.progress("published", 1, 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
