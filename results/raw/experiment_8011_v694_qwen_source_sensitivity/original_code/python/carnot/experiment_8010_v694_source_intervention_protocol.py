"""REQ-REPORT-8010: prepare source conditioning inputs without model evidence.

The same answer is paired with another complete source and an exact duplicate.
This measures whether a later model uses source inputs. It cannot establish
hallucination accuracy because the intervention has no human truth labels.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import random
import shutil
from tempfile import TemporaryDirectory
import threading
import time
from typing import Any

from carnot import experiment_7995_v693_qwen_development_capture as upstream
from carnot.inference.llama_cpp_process import OwnedLlamaCppProcess
from carnot.inference.qwen_sufficiency_7920 import QwenRuntime
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt

Json = dict[str, Any]
c = upstream.c
ROOT = upstream.ROOT
NAME = "experiment_8010_v694_source_intervention_protocol"
TASK = "exp8010-source-intervention-protocol"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = f"python/carnot/{NAME}.py"
TEST = "tests/python/test_source_intervention_protocol_8010.py"
OWNED = [MODULE, CLI]
MODEL_SPECS: list[str] = []
PIN = "sha256:df6eb8b559181455348b1d806f23c36d13c7f49202c2d49b2233aa80edebe02d"
ARMS = ("original", "swap", "duplicate")
METHODS = dict(
    seed=69410,
    intended_groups=64,
    intended_calls=192,
    temperature=0,
    max_tokens=32,
    output_token_budget=6144,
    call_timeout_s=120,
    measured_work_cap_s=2400,
    input_tokens=6000,
    selection="eligible_stream_source_hash_ascending",
    context_admission="conservative_UTF8_byte_upper_bound_then_embedded_tokenizer_at_capture",
    primary_metric="mean_source_paired_swap_minus_original_risk",
    placebo_metric="mean_source_paired_duplicate_minus_original_risk",
    minimum_complete_pairs=48,
    sensitivity_lower_CI_gate=0,
    placebo_absolute_mean_gate=0.01,
    bootstrap_draws=10000,
    natural_truth_labels=False,
    independent_unit="original_source_group",
)


def progress(phase: str, started: float, units: int = 0, pending: int = 0) -> None:
    """Expose actual phase boundaries so a silent child does not hide a stall."""
    print(
        f"[exp8010] phase={phase} elapsed_s={time.monotonic() - started:.3f} units={units} pending={pending}",
        flush=True,
    )


def freeze(view: Json) -> Json:
    """Select public identities first; context exclusions never cause replacements."""
    features = c.risk.custody.index_unique(view["features"], "family_id")
    if set(features) != {r["family_id"] for r in view["request_rows"]}:
        raise ValueError("public_feature_roster")
    candidates = []
    for public in view["request_rows"]:
        frozen = c.risk.freeze([public], lambda _: 0)[0]
        feature = features[public["family_id"]]
        if frozen["eligible"] and feature["abstention"] is None:
            candidates.append(dict(public, source_hash=feature["source_normalized_hash"]))
    roster = sorted(candidates, key=lambda r: (r["source_hash"], r["family_id"]))[:64]
    if len(roster) != 64 or len({r["source_hash"] for r in roster}) != 64:
        raise ValueError("independent_roster_floor")
    rng = random.Random(69410)
    cycle = list(range(64))
    rng.shuffle(cycle)
    donors = {cycle[i]: cycle[(i + 1) % 64] for i in range(64)}
    slots, pairs, hashes = [], [], {}
    for i, group in enumerate(roster):
        order = list(ARMS)
        rng.shuffle(order)
        group_slots = []
        for arm in order:
            donor = roster[donors[i]] if arm == "swap" else group
            public = {k: group[k] for k in c.risk.custody.PUBLIC_KEYS}
            public["source_bytes"] = donor["source_bytes"]
            frozen = c.risk.freeze([public], lambda _: 0)[0]
            request = frozen["requests"]["full_source"]
            request.update(seed=69410, max_tokens=32)
            size = len(json.dumps(request["messages"], ensure_ascii=False).encode())
            group_slots.append(
                dict(
                    family_id=group["family_id"] + ":" + arm,
                    group_id=group["family_id"],
                    role="stream",
                    arm=arm,
                    seed=69410,
                    source_cluster_id=group["source_hash"],
                    source_hash=c.risk.custody.digest(bytes.fromhex(donor["source_bytes"])),
                    donor_group_id=donor["family_id"],
                    request=request,
                    visible_ids=frozen["visible_ids"],
                    public_eligible=size <= 6000,
                    exclusion_reason=None if size <= 6000 else "context_upper_bound_overrun",
                    input_byte_upper_bound=size,
                    public_hash=canonical_hash(public),
                )
            )
        # Exclude the complete triple when any view fails admission; do not trim qualifiers.
        if any(not r["public_eligible"] for r in group_slots):
            for row in group_slots:
                row.update(
                    public_eligible=False, exclusion_reason="paired_context_upper_bound_overrun"
                )
        slots.extend(group_slots)
        hashes[group["family_id"]] = {r["arm"]: r["source_hash"] for r in group_slots}
        pairs.append(
            dict(
                group_id=group["family_id"],
                original=group["family_id"] + ":original",
                duplicate=group["family_id"] + ":duplicate",
            )
        )
    return dict(
        methods=METHODS,
        roster=roster,
        slots=slots,
        source_view_hashes=hashes,
        derangement={roster[i]["family_id"]: roster[j]["family_id"] for i, j in donors.items()},
        placebo_pairs=pairs,
        role_hash=canonical_hash(view),
        response_schema=dict(
            unsupported_probability="finite number in [0,1]",
            source_sentence_id="visible integer or null",
        ),
    )


def authenticate(root: Path) -> Json:
    """Reuse upstream qualification while requiring exact producer contract fields."""
    plan = upstream.authenticate(root)
    path = root / "results/experiment_7995_v693_qwen_development_capture.json"
    checks = plan["checks"]
    checks.append(
        upstream.operand(
            7995, path, "sha256", PIN, sha256_file(path) if path.is_file() else "missing_file"
        )
    )
    if path.is_file():
        value = json.loads(path.read_text())
        for key, expected in dict(
            experiment_id=7995,
            run_date="20261001",
            capture_ready_score=1,
            stream_capture_ready_score=1,
            flagged_adversarial=False,
        ).items():
            checks.append(
                upstream.operand(
                    7995, path, key, expected, value.get(key, "missing_field_contract_error")
                )
            )
        plan["references"].append(
            dict(
                upstream.reference(path),
                scope="historical",
                imported_fields=[
                    "capture_ready_score",
                    "stream_capture_ready_score",
                    "protocol_fingerprint",
                ],
            )
        )
    return plan


def capture_peer(panel: Json, output: Path, mode: str) -> None:
    """Use the qualified runtime HTTP methods without launching model weights.

    The loopback peer supplies explicit canned replies. Separate request IDs
    and a call ledger prove duplicate inputs use two independent transports.
    These calls qualify plumbing only and never enter current model counts.
    """

    if len({s["family_id"] for s in panel["slots"]}) != len(panel["slots"]):
        raise ValueError("duplicate_request_ids")
    for slot in panel["slots"]:
        body = json.loads(slot["request"]["messages"][1]["content"])
        if set(body) != {
            "complete_source",
            "original_answer",
            "target",
            "visible_source_sentence_ids",
            "source_evidence_erased",
        }:
            raise ValueError("target_bearing_predictor_payload")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_POST(self) -> None:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/apply-template":
                response = dict(prompt=json.dumps(payload["messages"], ensure_ascii=False))
            elif self.path == "/tokenize":
                response = dict(tokens=list(range(6001 if mode == "context_overrun" else 30)))
            elif mode == "timeout":
                time.sleep(0.05)
                self.close_connection = True
                return
            else:
                response = upstream.FixtureRuntime().generate(payload)
                if mode == "wrong_model":
                    response["model"] = "wrong-model"
            blob = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

    class Worker(OwnedLlamaCppProcess):
        def post_json(self, path: str, payload: Any, timeout_s: float) -> Json:
            return super().post_json(
                path,
                payload,
                0.02 if mode == "timeout" and path == "/v1/chat/completions" else timeout_s,
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    raw = output.parent / (output.stem + "_checkpoints")
    runtime = QwenRuntime(raw / "never-loaded.gguf", raw, 0)
    runtime.worker = Worker(
        command=[],
        port=server.server_port,
        env={},
        log_path=raw / "unused",
        state_path=raw / "unused-owner",
    )
    ledger = c.Ledger(raw / "scripted_ledger.json")
    try:
        rows = c.capture(
            panel["slots"],
            runtime,
            raw / "slots",
            canonical_hash(panel),
            ledger=ledger,
            deadline_s=2400,
            token_budget=192 * 96,
        )
        atomic_json(
            output,
            dict(
                scope="circular_scripted_HTTP_transport_only",
                rows=rows,
                ledger=ledger.rows,
                runtime_receipts=runtime.receipts,
                checkpoint_references=[
                    upstream.reference(p) for p in sorted((raw / "slots").glob("*.json"))
                ],
            ),
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def natural_rows(panel: Json) -> list[Json]:
    """Retain every intended call without inventing a model probability."""
    return [
        dict(
            s,
            id=s["family_id"],
            metric="unsupported_probability",
            numerator=0,
            denominator=1,
            started=False,
            probability=None,
            status="censored" if s["public_eligible"] else "excluded",
            exclusion=s["exclusion_reason"],
            censor_reason="protocol_only_no_model_requested" if s["public_eligible"] else None,
        )
        for s in panel.get("slots", [])
    ]


def budget(panel: Json) -> Json:
    """Count preparation separately from independently completed natural groups."""
    rows = natural_rows(panel)
    eligible = sum(r["public_eligible"] for r in rows)
    return dict(
        intended=192,
        eligible=eligible,
        started=0,
        completed=0,
        excluded=len(rows) - eligible,
        failed=0,
        censored=eligible,
        independent=0,
        prepared_independent_groups=len(panel.get("roster", [])),
        unit="source_view_call",
        independent_unit="original_source_group",
    )


def replay(value: Json) -> None:
    """Rebuild public views and rows from bound durable bytes in a cold process."""
    for ref in (
        value["raw_shard_hashes"] + value["checkpoint_references"] + value["code_config_hashes"]
    ):
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("checkpoint_or_code_hash_drift")
    panel = json.loads(Path(value["panel_reference"]["path"]).read_text())
    public = json.loads(Path(value["public_reference"]["path"]).read_text())
    if panel and freeze(public) != panel:
        raise ValueError("protocol_drift")
    if (
        natural_rows(panel) != value["rows"]
        or canonical_hash(panel) != value["reproducibility_checksum"]
        or budget(panel) != value["sample_size_budget"]
        or any(
            value[k] != panel.get(k, [] if k in {"roster", "placebo_pairs"} else {})
            for k in (
                "roster",
                "source_view_hashes",
                "derangement",
                "placebo_pairs",
                "response_schema",
            )
        )
    ):
        raise ValueError("raw_reduction_drift")
    fixture = json.loads(Path(value["fixture_reference"]["path"]).read_text())
    if fixture.get("rows", []) != value["fixture_rows"]:
        raise ValueError("fixture_drift")
    slots = {s["family_id"]: s for s in panel.get("slots", [])}
    ledger = {r["call_id"]: r for r in fixture.get("ledger", [])}
    if len(ledger) != len(fixture.get("ledger", [])):
        raise ValueError("duplicate_request_ids")
    for row in value["fixture_rows"]:
        if any(row[k] != v for k, v in slots[row["family_id"]].items()) or row[
            "parsed"
        ] != c.risk.transport.parse_response(row["raw_response"], row["visible_ids"]):
            raise ValueError("fixture_parse_or_request_drift")
        if row["started"] and ledger[row["family_id"]]["request_sha256"] != canonical_hash(
            row["request"]
        ):
            raise ValueError("transport_binding")
    if value["model_invocation_counts"] != ZERO_INVOCATION_COUNTS:
        raise ValueError("unexpected_current_model_calls")
    if value["protocol_ready_score"] and (
        value["verdict_class"] in {"blocked", "disqualified"}
        or not all(r["passed"] for r in value["validation_receipts"] if r["scope"] == "owned")
    ):
        raise ValueError("unsafe_readiness")


def commands(scratch: Path) -> list[CommandSpec]:
    """Freeze required owned checks separately from one bounded health command."""
    py = str(ROOT / ".venv/bin/python")
    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel = True\ninclude =\n    "
        + "\n    ".join(str(ROOT / p) for p in OWNED)
        + "\n"
    )
    specs = [
        (
            "focused",
            [
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                TEST,
            ],
        ),
        ("combine", [py, "-m", "coverage", "combine", "--rcfile=" + str(config), str(scratch)]),
        (
            "coverage",
            [
                py,
                "-m",
                "coverage",
                "json",
                "--rcfile=" + str(config),
                "--fail-under=100",
                "-o",
                str(scratch / "coverage.json"),
            ],
        ),
        (
            "consumers_E2E015_7995",
            [
                py,
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_source_boundary_7852.py",
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_current_work_receipt.py",
                "tests/python/test_qwen_development_capture_7995.py",
                "tests/python/test_experiment_7995_v693_qwen_development_capture.py",
                "tests/python/test_llama_cpp_process.py",
            ],
        ),
        (
            "E2E016_fixture",
            [
                py,
                "-u",
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "--date",
                "20260929",
                "--fixture-e2e",
                str(scratch / "e2e016.json"),
            ],
        ),
        (
            "E2E016_replay",
            [
                py,
                "-u",
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "--date",
                "20260929",
                "--cold-replay",
                str(scratch / "e2e016.json"),
            ],
        ),
        ("ruff", [py, "-m", "ruff", "check", *OWNED, TEST]),
        ("format", [py, "-m", "ruff", "format", "--check", *OWNED, TEST]),
        ("mypy", [py, "-m", "mypy", "--strict", *OWNED]),
        ("spec", [py, "scripts/check_spec_coverage.py", *OWNED, TEST]),
    ]
    return [CommandSpec(n, tuple(a), "owned", 180) for n, a in specs] + [
        CommandSpec(
            "repository_health",
            (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
            "repository_health",
            120,
        )
    ]


def build(
    panel: Json, plan: Json, raw: Path, receipts: list[Json], elapsed: float, fixture: bool
) -> Json:
    """Readiness credits completed preparation, while scientific outcomes stay absent."""
    rows = natural_rows(panel)
    failed = [r for r in plan["checks"] if not r["passed"]]
    transcript = json.loads((raw / "fixture.json").read_text())
    fixture_ok = len(transcript.get("rows", [])) == len(panel.get("slots", [])) and all(
        r["parsed"]["completed"]
        and r["raw_response"].get("usage", {}).get("completion_tokens", 33) <= 32
        if r["public_eligible"]
        else r["status"] == "excluded"
        for r in transcript.get("rows", [])
    )
    owned_ok = all(r["passed"] for r in receipts if r["scope"] == "owned") and fixture_ok
    verdict = (
        "blocked"
        if failed
        else "disqualified"
        if not owned_ok
        else "circular_positive"
        if fixture
        else "null"
    )
    ready = int(bool(panel) and not failed and owned_ok and not fixture)
    value = dict(
        experiment_id=8010,
        task_id=TASK,
        milestone="2026.10.694",
        run_date="20261002",
        execution_date="20261002",
        schema="carnot.exp8010.source_intervention_protocol.v1",
        honest_verdict=f"complete_{verdict}_source_intervention_protocol",
        verdict_class=verdict,
        claim_scope="This invocation freezes a natural source conditioning panel and qualifies scripted transport only. No natural sensitivity, hallucination accuracy or causal mitigation is measured.",
        gate_check_summary=[dict(r, artifact_field=r["field"]) for r in failed],
        preconditions_checked=plan["checks"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        planned_model_specs=[c.risk.MODEL],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        duration_s=elapsed,
        phase_spans=[dict(phase="protocol_preparation", start_s=0, end_s=elapsed)],
        rows=rows,
        sample_size_budget=budget(panel),
        random_seed=69410,
        reproducibility_checksum=canonical_hash(panel),
        verifier_is_oracle=False,
        acceptance_gate_results=dict(
            protocol_preparation=bool(ready), natural_sensitivity=False, scientific_benefit=False
        ),
        genuine_headroom=None,
        positive_control_results=dict(
            passed=bool(panel) and fixture_ok, scope="circular_scripted_HTTP_transport_only"
        ),
        protocol_ready_score=ready,
        roster=panel.get("roster", []),
        source_view_hashes=panel.get("source_view_hashes", {}),
        derangement=panel.get("derangement", {}),
        placebo_pairs=panel.get("placebo_pairs", []),
        request_budget=METHODS,
        response_schema=panel.get("response_schema", {}),
        fixture_rows=transcript.get("rows", []),
        fixture_scope="circular_scripted_HTTP_transport_only",
        evaluation_labels_opened=False,
        natural_estimates=dict(
            swap_minus_original=None, duplicate_minus_original=None, complete_pairs=0
        ),
        cited_upstream_artifacts=plan["references"],
        validation_receipts=receipts,
        repository_health=[r for r in receipts if r["scope"] == "repository_health"],
        coverage_statement_counts={},
        flagged_adversarial=False,
        panel_reference=upstream.reference(raw / "panel.json"),
        public_reference=upstream.reference(raw / "public.json"),
        fixture_reference=upstream.reference(raw / "fixture.json"),
        checkpoint_references=transcript.get("checkpoint_references", []),
        raw_shard_hashes=[
            upstream.reference(raw / p)
            for p in ["panel.json", "public.json", "fixture.json", "validation_commands.json"]
        ],
        code_config_hashes=json.loads((raw / "validation_commands.json").read_text())[
            "code_config_hashes"
        ],
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
    )
    value["field_principles"] = {
        k: "Bind this invocation to durable public bytes and actual checks; no model or intervention truth labels are supplied."
        for k in value
    }
    return value


def terminal(path: Path) -> Json:
    """Validate exact bytes and retain actual terminal command results."""
    replay(json.loads(path.read_text()))
    specs = [
        CommandSpec(n, (str(ROOT / ".venv/bin/python"), "-u", s, f, str(path)), "terminal", 60)
        for n, s, f in [
            ("adversarial", "scripts/adversarial_verify.py", "--json"),
            ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        ]
    ]
    logs = (
        path.parent / "terminal_logs"
        if (path.parent / "panel.json").is_file()
        else path.parent / "raw" / path.stem / "published_terminal_logs"
    )
    receipts = run_commands(ROOT, specs, log_dir=logs, heartbeat_s=10)
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Expose private capture and replay routes before checked primary publication."""
    started = time.monotonic()
    progress("start", started)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261002"], default="20261002")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    routes = parser.add_mutually_exclusive_group()
    routes.add_argument("--fixture", type=Path)
    routes.add_argument("--capture-peer", type=Path)
    routes.add_argument("--cold-replay", type=Path)
    parser.add_argument(
        "--peer-mode",
        choices=["normal", "wrong_model", "timeout", "context_overrun"],
        default="normal",
    )
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            replay(json.loads(args.cold_replay.read_text()))
            progress("replay_passed", started)
            return 0
        output = args.output.absolute()
        if args.capture_peer:
            capture_peer(json.loads(args.capture_peer.read_text()), output, args.peer_mode)
            progress("scripted_capture_complete", started)
            return 0
        if args.fixture and output.is_relative_to(ROOT):
            raise ValueError("fixture_publication_must_be_private")
        raw = output.parent / "raw" / output.stem
        raw.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="carnot-8010-") as directory:
            scratch = Path(directory)
            specs = commands(scratch)
            atomic_json(
                raw / "validation_commands.json",
                dict(
                    commands=[asdict(s) for s in specs],
                    methods=METHODS,
                    code_config_hashes=[upstream.reference(ROOT / p) for p in OWNED + [TEST]],
                    transport_qualification_argv=[
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        CLI,
                        "--capture-peer",
                        str(raw / "panel.json"),
                        "--output",
                        str(raw / "fixture.json"),
                    ],
                    cold_reduction_argv=[
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        CLI,
                        "--cold-replay",
                        str(raw / (NAME + ".json")),
                    ],
                    terminal_argv=[
                        [str(ROOT / ".venv/bin/python"), "-u", s, f, str(p)]
                        for p in [raw / (NAME + ".json"), output]
                        for s, f in [
                            ("scripts/adversarial_verify.py", "--json"),
                            ("scripts/verdict_row_consistency_lint.py", "--strict"),
                        ]
                    ],
                ),
            )
            plan = dict(checks=[], references=[]) if args.fixture else authenticate(args.root)
            progress("public_roster_freeze", started)
            public = (
                json.loads(args.fixture.read_text())
                if args.fixture
                else json.loads(Path(plan["public_role_manifests"]["stream"]["path"]).read_text())
                if all(r["passed"] for r in plan["checks"])
                else {}
            )
            panel = freeze(public) if public else {}
            atomic_json(raw / "public.json", public)
            atomic_json(raw / "panel.json", panel)
            progress("scripted_HTTP_capture", started, len(panel.get("roster", [])), 192)
            if panel:
                peer = CommandSpec(
                    "private_script_capture",
                    (
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        CLI,
                        "--capture-peer",
                        str(raw / "panel.json"),
                        "--output",
                        str(raw / "fixture.json"),
                    ),
                    "owned",
                    120,
                )
                peer_receipts = run_commands(
                    ROOT, [peer], log_dir=raw / "peer_logs", heartbeat_s=10
                )
            else:
                atomic_json(raw / "fixture.json", dict(rows=[], ledger=[]))
                peer_receipts = []
            receipts = peer_receipts
            if not args.fixture and args.root.resolve() == ROOT:
                progress("owned_validation", started)
                receipts += run_commands(
                    ROOT,
                    specs,
                    log_dir=raw / "validation_logs",
                    heartbeat_s=10,
                    extra_env=dict(
                        CARNOT_8010_COVERAGE_CONFIG=str(scratch / "coverage.ini"),
                        COVERAGE_FILE=str(scratch / ".coverage"),
                        JAX_PLATFORMS="cpu",
                    ),
                )
            value = build(
                panel, plan, raw, receipts, time.monotonic() - started, bool(args.fixture)
            )
            coverage = scratch / "coverage.json"
            if coverage.is_file():
                summary = json.loads(coverage.read_text())
                value["coverage_statement_counts"] = dict(summary["totals"], files=summary["files"])
                shutil.copy2(coverage, raw / "coverage.json")
            progress("cold_reduction", started, len(value["rows"]))
            candidate = raw / (NAME + ".json")
            atomic_json(candidate, value)
            cold = CommandSpec(
                "fresh_process_cold_reduction",
                (str(ROOT / ".venv/bin/python"), "-u", CLI, "--cold-replay", str(candidate)),
                "terminal",
                60,
            )
            cold_receipts = run_commands(ROOT, [cold], log_dir=raw / "cold_logs", heartbeat_s=10)
            if not all(r["passed"] for r in cold_receipts):
                raise ValueError("cold_reduction_failed")
            progress("terminal_validation", started)
            report = terminal(candidate)
            if not report["passed"]:
                atomic_json(raw / "rejected_terminal.json", report)
                raise ValueError("terminal_rejected")
            digest = sha256_file(candidate)
            publication = publish_primary(
                output, value, lambda p: report if sha256_file(p) == digest else dict(passed=False)
            )
            atomic_json(
                raw / "terminal_validation.json", dict(publication, cold_receipts=cold_receipts)
            )
            replay(json.loads(output.read_text()))
            published_report = terminal(output)
            atomic_json(raw / "published_validation.json", published_report)
            readers = reader_receipt(
                TASK,
                output.parent,
                field="protocol_ready_score",
                expected=value["protocol_ready_score"],
            )
            atomic_json(raw / "primary_readers.json", readers)
            if not readers["passed"] or not published_report["passed"]:
                raise ValueError("published_readers_rejected")
        progress("complete", started, len(value["rows"]))
        return 0
    except (OSError, RuntimeError, TimeoutError, ValueError, KeyError) as error:
        print(f"[exp8010] rejected={type(error).__name__}:{error}", flush=True)
        return 1
