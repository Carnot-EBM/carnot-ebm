"""REQ-REPORT-8263: current protocol mechanics grant no independent benefit.

Qualified subprocess supervision and publication stay unchanged. The current
adapters own focal requests and typed causal controls, with separate receipts
so a causal fixture cannot qualify transport by association.
"""

from __future__ import annotations

from copy import deepcopy
import atexit
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting import coverage_custody_8262 as custody
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import evidence_view_execution_8249 as upstream
from carnot.verify import focal_protocol_8263 as f
from carnot.verify import typed_admission_8263 as a
from carnot.verify import focal_capture_8263 as c
from carnot.verify.sentence_transport_methods_8179 import TOKENIZER_PATH, TOKENIZER_PIN
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8263_v714_protocol_conformance"
TASK = "exp8263-protocol-conformance"
RUN_DATE = "20261008"
MODULE = "python/carnot/verify/protocol_conformance_8263.py"
RUNNER = MODULE
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_protocol_conformance_8263.py"
OWNED = [
    MODULE,
    "python/carnot/verify/focal_protocol_8263.py",
    "python/carnot/verify/typed_admission_8263.py",
    "python/carnot/verify/focal_capture_8263.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
PROTOCOL = upstream.PROTOCOL
SCIENCE_PIN = "sha256:f4c06e17a9f8eb72f1be80b265ab7e3507cbc5f6f3b918df6421fe49dae02018"
reference, gate = upstream.reference, upstream.gate
BASE_CHECK = upstream.run_check
BASE_MANIFEST = execution.manifest


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts at phase boundaries and around bounded child work."""
    print(f"[exp8263] phase={phase} completed={completed} pending={pending}", flush=True)


def tokenizer(work: Json) -> tuple[Any, Json]:
    """Read only authenticated embedded vocabulary and bind its actual metadata."""
    progress("before_tokenizer_vocabulary_read")
    observed = sha256_file(TOKENIZER_PATH) if TOKENIZER_PATH.is_file() else None
    # Avoid reading a multi-gigabyte operand twice through the generic gate helper.
    work["checks"].append(
        dict(
            upstream="Qwen3.8_GGUF",
            path=str(TOKENIZER_PATH),
            hash=observed,
            artifact_field="tokenizer_gguf_sha256",
            op="==",
            expected=TOKENIZER_PIN,
            observed=observed,
            passed=observed == TOKENIZER_PIN,
        )
    )
    if observed != TOKENIZER_PIN:
        progress("after_tokenizer_vocabulary_read_blocked")
        return None, {}
    from llama_cpp import Llama

    try:
        vocab = Llama(
            model_path=str(TOKENIZER_PATH), vocab_only=True, n_gpu_layers=0, verbose=False
        )
    except (OSError, ValueError, RuntimeError) as error:
        work["checks"].append(
            dict(
                upstream="Qwen3.8_GGUF",
                path=str(TOKENIZER_PATH),
                hash=observed,
                artifact_field="embedded_vocabulary_available",
                op="==",
                expected=True,
                observed=str(error),
                passed=False,
            )
        )
        progress("after_tokenizer_vocabulary_read_blocked")
        return None, {}
    metadata = dict(vocab.metadata)
    atexit.register(vocab.close)
    receipt = dict(
        path=str(TOKENIZER_PATH),
        sha256=observed,
        metadata=metadata,
        metadata_sha256=canonical_hash(metadata),
        vocab_only=True,
        neural_weights_loaded=False,
        add_bos=False,
        special=False,
    )
    progress("after_tokenizer_vocabulary_read", 1)
    return lambda text: len(vocab.tokenize(text.encode(), add_bos=False, special=False)), receipt


def authenticate(root: Path, work: Json) -> None:
    """Keep failed historical8249 bytes while requiring qualified current operands."""
    upstream.authenticate(root, work)
    path = root / "results/experiment_8262_v714_coverage_custody.json"
    gate(work, path, "exists", True, True if path.is_file() else None)
    if path.is_file():
        value = json.loads(path.read_bytes())
        for field, expected in [
            ("coverage_custody_ready_score", 1),
            ("current_contract_ready_score", 1),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            gate(work, path, field, expected, value.get(field))
        terminal = Path(value["terminal_validation_sidecar_path"])
        report = json.loads(terminal.read_bytes())
        sidecar = Path(report["publication"]["sidecar_path"])
        gate(
            work,
            terminal,
            "publication.primary_sha256",
            sha256_file(path),
            report["publication"]["primary_sha256"],
        )
        gate(
            work,
            sidecar,
            "report.passed",
            True,
            read_bound_sidecar(path, sidecar)["report"]["passed"],
        )
        work["refs"].extend(
            dict(reference(p), fields_imported=fields)
            for p, fields in [
                (path, ["coverage_custody_ready_score", "current_contract_ready_score"]),
                (terminal, ["publication"]),
                (sidecar, ["report.passed"]),
            ]
        )
    for name in [
        "results/experiment_8249_v713_evidence_view_kernel.json",
        "openspec/change-proposals/v714-evidence-execution-contract.json",
    ]:
        path = root / name
        gate(work, path, "exists", True, True if path.is_file() else None)
        if path.is_file():
            work["refs"].append(
                dict(
                    reference(path),
                    fields_imported=["honest_verdict"]
                    if "8249" in name
                    else ["science_protocol_sha256"],
                    purpose="Historical failure remains unchanged; current custody is qualified separately.",
                )
            )
    path = root / PROTOCOL
    gate(
        work,
        path,
        "frozen_science_sha256",
        SCIENCE_PIN,
        sha256_file(path) if path.is_file() else None,
    )
    if path.is_file():
        protocol = json.loads(path.read_bytes())
        for ref in protocol["source_artifact_hashes"]:
            original = root / Path(ref["path"]).relative_to(ROOT)
            gate(
                work,
                original,
                "source_artifact_sha256",
                ref["sha256"],
                sha256_file(original) if original.is_file() else None,
            )
            if original.is_file():
                work["refs"].append(
                    dict(
                        reference(original),
                        fields_imported=["public source/cached prediction custody"],
                    )
                )


def measure(root: Path, raw: Path, *, fixture: bool = False, **kwargs: Any) -> Json:
    """Authenticate private scratch, tools and terminals before owned measurements."""
    progress("before_preconditions")
    start, wall = time.monotonic_ns(), time.time_ns()
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(
        checks=[],
        refs=[],
        evidence={},
        tokenizer={},
        owned_failure="",
        fixture=fixture,
        invocation_argv=list(sys.argv),
    )
    with TemporaryDirectory(prefix="carnot8263-private-") as directory:
        private = Path(directory)
        probe = private / "probe"
        probe.write_bytes(b"private writable scratch")
        gate(
            work,
            probe,
            "private_scratch_writable",
            True,
            probe.read_bytes() == b"private writable scratch",
        )
        for name in ["python", "pytest", "coverage", "ruff", "mypy"]:
            gate(
                work,
                ROOT / ".venv/bin" / name,
                "required_tool",
                True,
                (ROOT / ".venv/bin" / name).is_file(),
            )
        count = None
        if not fixture:
            try:
                authenticate(root, work)
            except (OSError, ValueError, KeyError, TypeError) as error:
                gate(work, root, "authenticated_input_schema", "qualified", str(error))
            if all(r["passed"] for r in work["checks"]):
                count, work["tokenizer"] = tokenizer(work)
        else:
            count = lambda text: len(
                text
            )  # Explicit private adapter double; earns no tokenizer readiness.
        progress("after_preconditions", len(work["checks"]))
        if count is not None:
            progress("before_private_benchmark")
            source = "Café is open. 東京 station. English words go here."
            row = dict(
                source_bytes=source.encode().hex(),
                answer_bytes=("Other sentence. " * 9 + "Café is open, but only today.")
                .encode()
                .hex(),
            )
            plan = f.views(row, [dict(sentence_index=9, p_unsupported=0.8, relation="E")], count)
            slots = [
                dict(unit_id=f"private-{i}", role="fit", mode=mode, view=plan["views"]["original"])
                for i, mode in enumerate(["valid"] * 8 + ["partial", "duplicate", "role", "drop"])
            ]
            progress("before_capture_subprocess")
            captured = c.capture(
                slots,
                private / "capture.jsonl",
                c.PipePeer(
                    [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / CLI), "--scripted-peer"]
                ),
            )
            progress("after_capture_subprocess", len(slots))
            controls = a.controls(private / "typed")
            crash_receipts = []
            roster_path = private / "natural_roster.json"
            atomic_json(roster_path, controls["rosters"]["natural"])
            for crash in [40, 72]:
                ledger = private / f"restart-{crash}.jsonl"
                command = [
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(ROOT / CLI),
                    "--typed-roster",
                    str(roster_path),
                    "--typed-journal",
                    str(ledger),
                ]
                for name, argv, expected in [
                    ("hard_exit", command + ["--crash-slot", str(crash)], 73),
                    ("resume", command, 0),
                ]:
                    progress("before_" + name + "_subprocess")
                    receipt = run_check(
                        ROOT,
                        dict(
                            name=f"{name}_{crash}", argv=argv, expected_exit=expected, deadline_s=60
                        ),
                        private,
                        raw / "crash_logs",
                    )
                    crash_receipts.append(receipt)
                    progress("after_" + name + "_subprocess")
                saved = json.loads(ledger.with_suffix(".pending.json").read_bytes())
                (raw / f"pending_{crash}.json").write_bytes(
                    ledger.with_suffix(".pending.json").read_bytes()
                )
                if (
                    a.load(ledger) != controls["states"]["natural"]
                    or crash - 8 not in saved["pending"]
                ):
                    work["owned_failure"] = "hard_exit_resume_drift"
            work["evidence"] = dict(
                controls,
                request_control_rows=[dict(original=row, plan=plan)],
                capture=captured,
                capture_events=c.journal(private / "capture.jsonl"),
                hard_exit_receipts=crash_receipts,
                pending_state_hashes=[
                    sha256_file(private / f"restart-{crash}.pending.json") for crash in [40, 72]
                ],
            )
            atomic_json(raw / "primitive_evidence.json", work["evidence"])
            progress("after_private_benchmark", 96 + 384)
    work.update(
        duration_s=(time.monotonic_ns() - start) / 1e9,
        clock=dict(
            started_monotonic_ns=start, ended_monotonic_ns=time.monotonic_ns(), started_wall_ns=wall
        ),
        code_config_hashes=[reference(ROOT / p) for p in OWNED + [TEST]],
        frozen_science_sha256=SCIENCE_PIN,
    )
    atomic_json(raw / "measurement.json", work)
    return work


def run_check(root: Path, spec: Json, private: Path, raw: Path, *, heartbeat_s: float = 20) -> Json:
    """Preserve the exact measured coverage operand before its scratch owner exits."""
    from carnot.reporting.v709_execution import child

    receipt = dict(
        child(
            spec["name"],
            spec["argv"],
            raw,
            expected=spec.get("expected_exit", 0),
            deadline=spec["deadline_s"],
            heartbeat=heartbeat_s,
            scope=spec.get("classification", "required"),
        )
    )
    if spec["name"] == "coverage_json":
        try:
            binding = custody.preserve(ROOT, spec, receipt, OWNED, raw.parent / "coverage")
            atomic_json(raw.parent / "coverage_binding.json", binding)
        except (OSError, ValueError, KeyError) as error:
            receipt.update(passed=False, custody_error=str(error))
    return receipt


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze component checks and subprocess/crash coverage before measurement."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = BASE_MANIFEST(private, candidate)
    config = private / "coverage.ini"
    config.write_text(
        config.read_text().replace("[run]", "[run]\npatch = subprocess, _exit")
        + "[report]\nexclude_lines=\n"
    )
    common = [str(ROOT / ".venv/bin/pytest"), "-n", "0", "-o", "addopts=", "--no-cov", "-q", TEST]
    specs["commands"].insert(
        1,
        dict(
            name="view_component",
            argv=common + ["-k", "focal or capture"],
            deadline_s=180,
            expected_exit=0,
            classification="required",
        ),
    )
    specs["commands"].insert(
        2,
        dict(
            name="admission_component",
            argv=common + ["-k", "typed or hard_exit"],
            deadline_s=180,
            expected_exit=0,
            classification="required",
        ),
    )
    specs["execution_contract"] = dict(
        random_group_encoding=f.ENCODING,
        science_protocol_sha256=SCIENCE_PIN,
        generation=dict(temperature=0, seed=7138250, max_tokens=64, prompt_format="raw_completion"),
        capture=dict(server_lifetimes=1, checkpoint_sources=8, durable_issue_before_dispatch=True),
        typed_feedback=dict(
            delay=8, minimum_distinct_sources=8, prior="Beta(1,1)", mixture="n/(n+16)"
        ),
    )
    for spec in specs["commands"]:
        if spec["name"] == "owned_unit_and_private_CLI":
            spec["deadline_s"] = 480
        if spec["name"] == "consumer_and_E2E015_019":
            spec.update(
                name="consumer_E2E019_020",
                argv=common[:-1]
                + [
                    "tests/python/test_primary_publication_7928.py",
                    "tests/python/test_experiment_7942_v689_sentence_labels.py",
                    "tests/python/test_hard_exit_learning_qualification_8206.py",
                ],
                deadline_s=900,
            )
        if spec["name"] == "strict_mypy":
            spec["argv"] = [
                str(ROOT / ".venv/bin/mypy"),
                "--config-file=/dev/null",
                "--strict",
                "--follow-imports=skip",
                "--ignore-missing-imports",
                *OWNED,
            ]
        if spec["name"] == "spec_coverage":
            spec["argv"].insert(-1, "--files")
    return dict(specs)


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Gate transport independently of synthetic gain and any human-label direction."""
    failed = [r for r in work["checks"] if not r["passed"]]
    evidence = work["evidence"]
    path = raw / "coverage_binding.json"
    binding = json.loads(path.read_bytes()) if path.is_file() else {}
    measured = custody.replay(binding) if binding else {}
    owned = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and (fixture or bool(measured))
        and not work["owned_failure"]
        and all(r["passed"] for r in evidence.get("hard_exit_receipts", []))
    )
    component = {
        name: any(r["name"] == name + "_component" and r["passed"] for r in receipts)
        for name in ["view", "admission"]
    }
    view_ready = int(
        owned
        and not failed
        and component["view"]
        and bool(work["tokenizer"])
        and evidence.get("capture", {}).get("completed_count") == 8
    )
    admission_ready = int(
        owned
        and not failed
        and component["admission"]
        and evidence.get("positive_control_passed", False)
    )
    verdict = (
        "disqualified"
        if not owned
        else "blocked"
        if failed
        else "circular_positive"
        if admission_ready
        else "null"
    )
    rows = (
        evidence.get("natural_shape_control_rows", [])
        + evidence.get("learnable_control_rows", [])
        + evidence.get("decision_control_rows", [])
    )
    if failed:
        rows = [
            dict(
                unit_id=r["upstream"],
                condition="missing_external_operand",
                status="excluded",
                exclusion_reason=r["artifact_field"],
                numerator=None,
                denominator=0,
            )
            for r in failed
        ]
    requests = evidence.get("request_control_rows", [])
    schemas = [
        v["request"]["response_schema_sha256"]
        for r in requests
        for _, v in sorted(r["plan"]["views"].items())
    ]
    value = dict(
        experiment_id=8263,
        task_id=TASK,
        milestone="2026.10.714",
        run_date=RUN_DATE,
        honest_verdict="complete_"
        + verdict
        + "_"
        + (failed[0]["upstream"] if verdict == "blocked" else "protocol_conformance"),
        verdict_class=verdict,
        gate_check_summary=work["checks"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        rows=rows,
        intended_count=len(rows),
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=0,
        censored_count=0,
        excluded_count=sum(r["status"] == "excluded" for r in rows),
        independent_count=0,
        verifier_is_oracle=True,
        exposure_scope="exposed_development_and_private_synthetic_controls",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=False,
        acceptance_gates=dict(
            owned_validation=owned,
            changed_statement_coverage=bool(measured),
            view_kernel=bool(view_ready),
            admission_kernel=bool(admission_ready),
            natural_benefit=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["checks"],
        duration_s=work["duration_s"]
        + sum(r.get("duration_s", 0) for r in receipts)
        + work.get("global_health", {}).get("duration_s", 0),
        random_seed=7138250,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[
            reference(p)
            for p in sorted(raw.glob("*.json"))
            if p.name.startswith("pending_")
            or p.name
            in {
                "measurement.json",
                "primitive_evidence.json",
                "validation_commands.json",
                "validation_receipts.json",
                "coverage_binding.json",
            }
        ],
        phase_spans=[dict(phase="measurement", duration_s=work["duration_s"], **work["clock"])]
        + [
            dict(
                phase=r["name"],
                **{
                    k: r.get(k)
                    for k in [
                        "duration_s",
                        "started_monotonic_ns",
                        "ended_monotonic_ns",
                        "started_wall_ns",
                    ]
                },
            )
            for r in receipts
        ],
        cited_upstream_artifacts=work["refs"],
        view_kernel_ready_score=view_ready,
        admission_kernel_ready_score=admission_ready,
        tokenizer_metadata_sha256=work["tokenizer"].get("metadata_sha256"),
        tokenizer_receipt=work["tokenizer"],
        response_schema_sha256=canonical_hash(schemas),
        frozen_science_sha256=SCIENCE_PIN,
        component_validation_receipts={
            name: [r for r in receipts if r["name"] == name + "_component"] for name in component
        },
        request_control_rows=requests,
        view_byte_maps=[
            v["sentence_map"] for r in requests for _, v in sorted(r["plan"]["views"].items())
        ],
        random_group_encoding=f.ENCODING,
        admission_events=evidence.get("admission_events", {}),
        decision_control_rows=evidence.get("decision_control_rows", []),
        natural_shape_control_rows=evidence.get("natural_shape_control_rows", []),
        learnable_control_rows=evidence.get("learnable_control_rows", []),
        positive_control_passed=evidence.get("positive_control_passed", False),
        control_costs={
            k: evidence.get(k)
            for k in [
                "later_costs",
                "false_accepts",
                "retention",
                "improvements",
                "zero_admission_count",
            ]
        },
        state_hashes=evidence.get("state_hashes", {}),
        capture_receipt=evidence.get("capture", {}),
        coverage_command_receipt=binding,
        owned_statement_counts=measured,
        fixture_mode=fixture,
        invocation_argv=work["invocation_argv"],
        repository_health=work.get("global_health", {}),
        scientific_benefit_measured=False,
        claim_scope="Focal tokenizer/capture conformance and causal typed-action controls only. Private oracle gains are circular_positive mechanics; no natural benefit or independent generalization established.",
        methodology_note="Reuse qualified UTF-8 mapping and fsynced issue/release journal. Frozen focal64 raw-completion requests retain full inputs and exact GGUF token admission. Beta(1,1) soft counters, source-hash random groups, delayed feedback and frozen allowed-action costs run on private predeclared rosters; separate component checks and durable measured statement custody qualify execution only.",
        reconciliation_note="Conductor owns ops and traceability updates.",
    )
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Bind actual current execution custody; readiness never supplies scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        random_group_encoding="Label-blind negative-control assignment only; hashes are never learned source memory.",
        component_validation_receipts="Transport and causal controls have independent check authority.",
        control_costs="Actual typed allowed-action costs on private oracle rosters; circular mechanics only.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return dict(value)


def replay(path: Path) -> bool:
    """Cold rebuild primitive requests and causal state rather than trust headlines."""
    try:
        value = json.loads(path.read_bytes())
        checksum = value.pop("reproducibility_checksum")
        if checksum != canonical_hash(value):
            return False
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for receipt in value["validation_receipts"]:
            for prefix in ["stdout", "stderr"]:
                if (
                    prefix + "_path" in receipt
                    and sha256_file(Path(receipt[prefix + "_path"])) != receipt[prefix + "_sha256"]
                ):
                    return False
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        work = json.loads((raw / "measurement.json").read_bytes())
        if work["evidence"]:
            primitive = json.loads((raw / "primitive_evidence.json").read_bytes())
            if primitive != work["evidence"]:
                return False
            with TemporaryDirectory(prefix="carnot8263-replay-") as directory:
                expected = a.controls(Path(directory))
            if any(primitive[k] != v for k, v in expected.items()):
                return False
            for index, crash in enumerate([40, 72]):
                pending = raw / f"pending_{crash}.json"
                if sha256_file(pending) != primitive["pending_state_hashes"][index]:
                    return False
                state = a.initial()
                for event in primitive["states"]["natural"]["events"]:
                    a.transition(state, event)
                    if event["kind"] == "issue" and event["record"]["slot"] == crash:
                        break
                if state != json.loads(pending.read_bytes()):
                    return False
            if work["fixture"]:
                count = lambda text: len(text)
            else:
                checks: Json = dict(checks=[])
                count, metadata = tokenizer(checks)
                if count is None or metadata != work["tokenizer"]:
                    return False
            for row in primitive["request_control_rows"]:
                if (
                    f.views(
                        row["original"],
                        [dict(sentence_index=9, p_unsupported=0.8, relation="E")],
                        count,
                    )
                    != row["plan"]
                ):
                    return False
            issues = {r["slot"]: r for r in primitive["capture_events"] if r["kind"] == "issue"}
            replies = [r for r in primitive["capture_events"] if r["kind"] == "reply"]
            if len(issues) != 12 or len(replies) != 12 or len({r["slot"] for r in replies}) != 12:
                return False
            slots = primitive["capture"]["intended_slots"]
            for reply in replies:
                row = reply["row"]
                slot = slots[row["slot"]]
                key = canonical_hash(
                    dict(role=slot["role"], unit_id=slot["unit_id"], view=slot["view"])
                )
                if (
                    row["resume_key"] != key
                    or issues[row["slot"]]["resume_key"] != key
                    or issues[row["slot"]]["plan_hash"] != canonical_hash(slots)
                ):
                    return False
                if row["error"] is None and (
                    f.parse(
                        slot["view"], row["transcript"], canonical_hash(slot["view"]["request"])
                    )
                    != row["parsed"]
                    or row["status"] != row["parsed"]["status"]
                ):
                    return False
            if [r["row"] for r in replies] != primitive["capture"]["rows"] or primitive["capture"][
                "completed_count"
            ] != sum(r["row"]["status"] == "completed" for r in replies):
                return False
        return build(
            work, raw, value["validation_receipts"], fixture=value["fixture_mode"]
        ) == dict(value, reproducibility_checksum=checksum)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main(argv: list[str] | None = None) -> int:
    """Expose real CLI, scripted peers and genuine crash controls under supervision."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--scripted-peer" in args:
        return c.peer()
    if "--typed-roster" in args:
        roster = Path(args[args.index("--typed-roster") + 1])
        ledger = Path(args[args.index("--typed-journal") + 1])
        crash = int(args[args.index("--crash-slot") + 1]) if "--crash-slot" in args else -1
        state = a.run(json.loads(roster.read_bytes()), ledger, crash_slot=crash)
        atomic_json(ledger.with_suffix(".final.json"), state)
        return 0
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
        patch.object(execution, "manifest", manifest),
        patch.object(execution, "run_check", run_check),
    ):
        return int(execution.main(args))
