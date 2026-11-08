"""REQ-REPORT-8918: measure final-answer errors without a detector claim."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import random
import re
import tempfile
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8918_v763_bfs_error_pilot"
TASK = "exp8918-bfs-error-pilot"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/bfs_error_pilot_8918.py",
    "python/carnot/inference/bfs_pilot_runtime_8918.py",
    CLI,
]
TEST = "tests/python/test_bfs_error_pilot_8918.py"
SEED = 763018
SETTINGS = dict(
    model="unsloth/Qwen3.8-27B-GGUF",
    max_tokens=16,
    temperature=0,
    top_p=1,
    seed=SEED,
    chat_template_kwargs=dict(enable_thinking=False),
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase boundaries so a quiet native call remains visible."""
    print(f"[exp8918] phase={phase} completed={completed} pending={pending}", flush=True)


def distance_two(nodes: list[str], edges: list[list[str]], start: str) -> list[str]:
    """Shortest distance excludes neighbors that also have a two-edge walk."""
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    distances = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return sorted(node for node, distance in distances.items() if distance == 2)


def graph_manifest() -> list[Json]:
    """Seal independent graph seeds and select a valid start before inference."""
    graphs = []
    seeds = random.Random(SEED).sample(range(2**31), 50)
    labels = [a + b for a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for b in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    for index, seed in enumerate(seeds):
        rng = random.Random(seed)
        resamples = 0
        while True:
            n = rng.randint(50, 60)
            nodes = sorted(rng.sample(labels, n))
            edges = [
                [a, b]
                for i, a in enumerate(nodes)
                for b in nodes[i + 1 :]
                if rng.random() < 3 / (n - 1)
            ]
            start = next(
                (node for node in rng.sample(nodes, n) if distance_two(nodes, edges, node)),
                None,
            )
            if start is not None:
                break
            resamples += 1
        expected = distance_two(nodes, edges, start)
        prompt = (
            "Undirected simple graph. Nodes: "
            + ", ".join(nodes)
            + ".\nEdges: "
            + ", ".join(a + "-" + b for a, b in edges)
            + f".\nName a node exactly 2 hops from {start} (shortest distance). Reply with only its two-letter label."
        )
        graphs.append(
            dict(
                unit_id=f"graph-{index:02d}",
                seed=seed,
                nodes=nodes,
                edges=edges,
                start=start,
                expected=expected,
                prompt=prompt,
                k=2,
                edge_probability=3 / (n - 1),
                resampling_count=resamples,
                edges_sha256=canonical_hash(edges),
                start_sha256=canonical_hash(start),
                expected_sha256=canonical_hash(expected),
                prompt_sha256=canonical_hash(prompt),
            )
        )
    return graphs


def parse_answer(answer: str, nodes: list[str]) -> tuple[str | None, str | None]:
    """A strict single label prevents post-hoc extraction from multiple guesses."""
    match = re.fullmatch(r"\s*([A-Z]{2})[.!]?\s*", answer)
    if match is None:
        return None, "invalid_single_node"
    node = match.group(1)
    return (node, None) if node in nodes else (None, "out_of_graph")


def reduce_answers(graphs: list[Json], responses: list[Json]) -> Json:
    """Retain every scheduled graph; uncertain transport never becomes an error."""
    if (
        len(graphs) != 50
        or len({g["seed"] for g in graphs}) != 50
        or len({g["unit_id"] for g in graphs}) != 50
    ):
        raise ValueError("graph_roster")
    by_id = {r["unit_id"]: r for r in responses}
    if len(by_id) != len(responses) or not set(by_id) <= {g["unit_id"] for g in graphs}:
        raise ValueError("response_roster")
    rows = []
    for graph in graphs:
        response = by_id.get(
            graph["unit_id"], dict(status="censored", missing_reason="not_started")
        )
        status = response["status"]
        if status not in {"completed", "runtime_failed", "censored"}:
            raise ValueError("response_status")
        node, reason = parse_answer(response.get("answer", ""), graph["nodes"])
        error = int(node not in graph["expected"]) if status == "completed" else None
        rows.append(
            dict(
                unit_id=graph["unit_id"],
                cluster_id=str(graph["seed"]),
                arm="qwen_reasoning_off_k2",
                seed=graph["seed"],
                metric="final_answer_error",
                numerator=error,
                denominator=1,
                status=status,
                missing_reason=None
                if status == "completed"
                else response.get("missing_reason", "runtime_failure"),
                answer=response.get("answer"),
                parsed_node=node,
                parser_reason=reason,
                error=error,
                duration_s=response.get("duration_s"),
                usage=response.get("usage", {}),
            )
        )
    completed = sum(r["status"] == "completed" for r in rows)
    errors = sum(r["error"] or 0 for r in rows)
    rate = errors / 50
    z = 1.959963984540054
    center = (rate + z * z / 100) / (1 + z * z / 50)
    radius = z * (rate * (1 - rate) / 50 + z * z / 10000) ** 0.5 / (1 + z * z / 50)
    return dict(
        rows=rows,
        answer_rows=rows,
        intended_count=50,
        n_questions=50,
        completed_count=completed,
        failed_count=sum(r["status"] == "runtime_failed" for r in rows),
        censored_count=50 - completed,
        excluded_count=0,
        independent_count=completed,
        n_errors=errors,
        error_rate=rate if completed == 50 else None,
        error_interval=dict(
            method="Wilson",
            confidence=0.95,
            low=max(0, center - radius),
            high=min(1, center + radius),
            denominator=50,
        )
        if completed == 50
        else None,
    )


def gates(result: Json, *, live: bool, owned: bool) -> tuple[int, int, int, str, str]:
    """Runtime, pilot completion and oracle error supply answer separate questions."""
    runtime = int(live and owned)
    ready = int(runtime and result["completed_count"] == 50)
    error_gate = int(ready and result["n_errors"] >= 10)
    if not owned:
        return 0, 0, 0, "disqualified", "complete_disqualified_owned_validation"
    if not live:
        return 0, 0, 0, "blocked", "complete_blocked_runtime"
    if not ready:
        return 0, 0, runtime, "blocked", "complete_blocked_incomplete_generations"
    return (
        ready,
        error_gate,
        runtime,
        "circular_positive" if error_gate else "null",
        "complete_pilot_error_source_gate_passed"
        if error_gate
        else "complete_pilot_error_rate_below_gate",
    )


def checksum(value: Json) -> str:
    """Bind the whole artifact while allowing its digest to be recomputed."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def reference(path: Path) -> Json:
    """Retain bytes rather than trusting a mutable path name."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def validation_plan(private: Path) -> list[CommandSpec]:
    """Measure only new statements, including direct CLI child processes."""
    private.mkdir(parents=True, exist_ok=True)
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\npatch = subprocess\ninclude =\n"
        + "".join("    */" + path + "\n" for path in OWNED)
        + "[report]\nexclude_lines =\n"
    )
    mypy = private / "mypy.ini"
    mypy.write_text("[mypy]\nstrict = true\nfollow_imports = skip\nignore_missing_imports = true\n")
    py = str(ROOT / ".venv/bin/python")
    cov = str(ROOT / ".venv/bin/coverage")
    common = ("-n0", "-o", "addopts=", "--no-cov", "-q")
    commands = [
        CommandSpec(
            "owned_unit_child",
            (
                cov,
                "run",
                "--rcfile=" + str(config),
                "--data-file=" + str(private / ".coverage"),
                "-m",
                "pytest",
                *common,
                "--basetemp=" + str(private / "pytest"),
                TEST,
            ),
            "owned",
            240,
        ),
        CommandSpec(
            "coverage_combine",
            (cov, "combine", "--data-file=" + str(private / ".coverage"), str(private)),
            "owned",
            30,
        ),
        CommandSpec(
            "coverage_json",
            (
                cov,
                "json",
                "--rcfile=" + str(config),
                "--data-file=" + str(private / ".coverage"),
                "-o",
                str(private / "coverage.json"),
                "--fail-under=100",
            ),
            "owned",
            30,
        ),
        CommandSpec(
            "ruff_check", (str(ROOT / ".venv/bin/ruff"), "check", *OWNED, TEST), "owned", 30
        ),
        CommandSpec(
            "ruff_format",
            (str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, TEST),
            "owned",
            30,
        ),
        CommandSpec(
            "strict_mypy",
            (str(ROOT / ".venv/bin/mypy"), "--config-file=" + str(mypy), *OWNED),
            "owned",
            120,
        ),
        CommandSpec("spec_coverage", (py, "scripts/check_spec_coverage.py", TEST), "owned", 30),
    ]
    for name, test in [
        ("e2e018", "tests/python/test_experiment_7891_v685_authority_lifecycle.py"),
        ("e2e021", "tests/python/test_restricted_decision_audit_8210.py"),
    ]:
        commands.append(
            CommandSpec(
                name,
                (
                    str(ROOT / ".venv/bin/pytest"),
                    *common,
                    "--basetemp=" + str(private / name),
                    test,
                ),
                "private_e2e",
                240,
            )
        )
    return commands


def qualification(
    runtime: Json, receipts: list[Json], coverage: Json, fixture: bool
) -> tuple[bool, bool]:
    """Recompute readiness from primitive controls, including during cold replay."""
    owned = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and coverage.get("percent_covered") == 100
        and coverage.get("excluded_lines") == 0
        and not runtime.get("owned_failure")
    )
    model = runtime.get("model_receipt", {})
    offload = model.get("offload_layers", [0, 0])
    live = bool(
        not fixture
        and runtime["live"]
        and model.get("authenticated")
        and model.get("gguf_sha256") == model.get("preflight_gguf_sha256")
        and offload[0] > 0
        and offload[0] == offload[1]
        and runtime["counts"]["generation_calls_completed"] > 0
    )
    return owned, live


def build(
    graphs: list[Json],
    runtime: Json,
    receipts: list[Json],
    coverage: Json,
    raw: Path,
    duration: float,
    *,
    fixture: bool = False,
) -> Json:
    """Qualification cannot turn a BFS oracle into independent detector benefit."""
    reduced = reduce_answers(graphs, runtime["responses"])
    owned, live = qualification(runtime, receipts, coverage, fixture)
    model = runtime.get("model_receipt", {})
    counts = runtime["counts"]
    ready, error_gate, runtime_ready, cls, verdict = gates(
        reduced, live=live, owned=owned or fixture
    )
    if cls == "blocked" and runtime.get("blocker"):
        verdict = "complete_blocked_" + re.sub(r"[^a-zA-Z0-9_]+", "_", runtime["blocker"])
    sources = [
        reference(path) for path in sorted((raw / "retained_sources").rglob("*")) if path.is_file()
    ]
    shards = [
        reference(path)
        for path in sorted(raw.rglob("*"))
        if path.is_file()
        and "retained_sources" not in path.parts
        and "terminal_candidate" not in path.name
    ]
    template = ExperimentTemplate(
        8918,
        "Fifty-question BFS final-answer error pilot",
        f"results/{NAME}.json",
    )
    value: Json = template.build_result(
        dict(
            experiment_id=8918,
            task_id=TASK,
            milestone="2026.11.763",
            run_date="20261108",
            honest_verdict=verdict,
            verdict_class=cls,
            verifier_is_oracle=True,
            exposure_scope="constructed_random_graph_final_answers_only",
            independent_generalization_score=0,
            inference_substrate="live_llm_inference"
            if counts["generation_calls_attempted"]
            else "model_loading"
            if counts["model_loads_attempted"]
            else "blocked_no_run",
            inference_substrate_class="model_bounded_generation"
            if counts["generation_calls_attempted"]
            else "model_load_no_generation"
            if counts["model_loads_attempted"]
            else "blocked_no_run",
            execution_venue="host",
            MODEL_SPECS=[]
            if fixture
            else [dict(hf_id=SETTINGS["model"], format="GGUF", quantization="Q4_K_M")],
            model_specs=[]
            if fixture
            else [dict(hf_id=SETTINGS["model"], format="GGUF", quantization="Q4_K_M")],
            model_invocation_counts=counts,
            model_invoked=bool(counts["model_loads_attempted"]),
            numerical_training_calls=0,
            duration_s=duration,
            phase_spans=runtime.get("phase_spans", []),
            preconditions_checked=runtime.get("preconditions_checked", {}),
            random_seed=SEED,
            source_artifact_hashes=sources,
            code_config_hashes=sources,
            raw_shard_hashes=shards,
            cited_upstream_artifacts=[],
            graph_manifest=graphs,
            generator_settings=runtime.get("settings", SETTINGS),
            model_receipt=model,
            resource_lease=runtime.get("resource_lease", {}),
            gate_check_summary=runtime["checks"],
            acceptance_gates=dict(
                all_fifty_completed=reduced["completed_count"] == 50,
                at_least_ten_errors=reduced["n_errors"] >= 10,
                owned_checks=owned,
                actual_offloaded_generation=live,
            ),
            required_checks_passed=owned,
            validation_receipts=receipts,
            owned_statement_coverage=coverage,
            pilot_ready_score=ready,
            pilot_error_gate_score=error_gate,
            qwen_runtime_ready_score=runtime_ready,
            fixture_only=fixture,
            flagged_adversarial=False,
            recommendation="Future task may raise k or graph size; no additional run is authorized here."
            if ready and not error_gate
            else "Pilot only; no detector, localization or natural-language generalization claim.",
            **reduced,
        ),
        status="blocked" if cls == "blocked" else "error" if cls == "disqualified" else "success",
    )
    value["field_principles"] = {
        key: "This field describes only this invocation and constructed BFS final-answer evidence; it establishes no detector benefit."
        for key in value
    }
    value["field_principles"].update(
        rows="One row per scheduled graph; only completed generations receive BFS labels.",
        model_invocation_counts="Counts derive from the retained owned call ledger, never from fixtures or upstream work.",
        pilot_ready_score="All fifty final answers and owned checks must complete.",
        pilot_error_gate_score="At least ten BFS errors among fifty completed answers; oracle-only evidence.",
        qwen_runtime_ready_score="Actual authenticated GPU offload and generation plus owned checks, independent of error rate.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Recompute labels and gates from authenticated retained runtime operands."""
    value = json.loads(path.read_text())
    if value["reproducibility_checksum"] != checksum(value):
        raise ValueError("checksum_drift")
    for ref in value["raw_shard_hashes"] + value["source_artifact_hashes"]:
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("operand_hash_drift")
    graphs = graph_manifest()
    if value["graph_manifest"] != graphs:
        raise ValueError("graph_truth_drift")
    runtime_ref = next(
        r for r in value["raw_shard_hashes"] if Path(r["path"]).name == "runtime.json"
    )
    runtime = json.loads(Path(runtime_ref["path"]).read_text())
    reduced = reduce_answers(graphs, runtime["responses"])
    if any(value[key] != actual for key, actual in reduced.items()):
        raise ValueError("reduction_drift")
    if (
        value["model_invocation_counts"] != runtime["counts"]
        or value["model_receipt"] != runtime["model_receipt"]
        or value["generator_settings"] != runtime["settings"]
    ):
        raise ValueError("runtime_binding_drift")
    owned, live = qualification(
        runtime,
        value["validation_receipts"],
        value["owned_statement_coverage"],
        value["fixture_only"],
    )
    if (
        owned != value["required_checks_passed"]
        or live != value["acceptance_gates"]["actual_offloaded_generation"]
    ):
        raise ValueError("qualification_drift")
    expected = gates(reduced, live=live, owned=owned or value["fixture_only"])
    if (
        value["pilot_ready_score"],
        value["pilot_error_gate_score"],
        value["qwen_runtime_ready_score"],
        value["verdict_class"],
    ) != expected[:4] or value["verifier_is_oracle"] is not True:
        raise ValueError("gate_drift")
    return dict(passed=True, replay_passed=True, n_questions=50)


def terminal_passed(receipts: list[Json]) -> bool:
    """Keep auditor exit codes; its informational zero-gate flags are diagnostic."""
    for receipt in receipts:
        if receipt["passed"]:
            continue
        if receipt["name"] != "adversarial" or receipt["exit_code"] != 1:
            return False
        report = json.loads((ROOT / receipt["log_path"]).read_text())
        if not report["reports"] or any(
            not row["loaded"] or row["max_severity"] > 0 for row in report["reports"]
        ):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    """Publish only checked evidence; private CLI fixtures cannot load a model."""
    from carnot.inference.bfs_pilot_runtime_8918 import run
    from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--fixture-e2e", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay is not None:
        try:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        except (ValueError, KeyError, OSError, StopIteration) as error:
            print(json.dumps(dict(passed=False, error=str(error))), flush=True)
            return 1
    if args.date != "20261108":
        parser.error("this invocation requires --date 20261108")
    os.environ.update(CARNOT_FORCE_LIVE="1", CUDA_VISIBLE_DEVICES="0")
    started = time.monotonic()
    progress("invocation_start")
    fixture = args.fixture_e2e is not None
    output = args.fixture_e2e or ROOT / "results" / (NAME + ".json")
    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    graphs = graph_manifest()
    atomic_json(raw / "graph_manifest.json", dict(graphs=graphs))
    for name in OWNED + [
        TEST,
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "research-program.md",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/change-proposals/research-roadmap-vNEXT.md",
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/inference/qwen_sufficiency_7920.py",
        "python/carnot/inference/sota_models.py",
        "python/carnot/verify/gguf_inference.py",
        "ops/e2e-test-plan.md",
        "ops/known-issues.md",
        "docs/research-notes/bfs-labeled-error-source-plan-2026-11-07.md",
        "docs/research-notes/label-provenance-audit-step1-2026-11-07.md",
        "docs/research-notes/implicithop-knowledge-integration-2026-11-06.md",
    ]:
        retained = raw / "retained_sources" / name
        retained.parent.mkdir(parents=True, exist_ok=True)
        retained.write_bytes((ROOT / name).read_bytes())
    runtime: Json = dict(
        responses=[],
        counts=dict(ZERO_INVOCATION_COUNTS),
        live=False,
        checks=[
            dict(
                check="fixture" if fixture else "owned_validation",
                upstream=TASK,
                artifact_field="runtime",
                expected="authenticated live GPU-0 generation",
                observed="fixture_no_run" if fixture else "not_started",
                path=str(raw),
                hash=None,
            )
        ],
        model_receipt={},
        settings=dict(SETTINGS),
        blocker="fixture_no_run" if fixture else "owned_validation",
    )
    receipts: list[Json] = []
    coverage: Json = {}
    health: list[Json] = []
    if not fixture:
        private = Path(tempfile.mkdtemp(prefix="carnot-8918-validation-"))
        progress("before_owned_checks")
        receipts = run_commands(
            ROOT, validation_plan(private), log_dir=raw / "validation", heartbeat_s=15
        )
        if (private / "coverage.json").is_file():
            coverage = json.loads((private / "coverage.json").read_text())["totals"]
            (raw / "coverage.json").write_bytes((private / "coverage.json").read_bytes())
        for name in ["coverage.ini", "mypy.ini"]:
            (raw / name).write_bytes((private / name).read_bytes())
        progress("after_owned_checks")
        health = run_commands(
            ROOT,
            [
                CommandSpec(
                    "full_python_suite",
                    (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
                    "repository_health",
                    300,
                )
            ],
            log_dir=raw / "repository_health",
            heartbeat_s=15,
        )
        if (
            receipts
            and all(r["passed"] for r in receipts)
            and coverage.get("percent_covered") == 100
            and coverage.get("excluded_lines") == 0
        ):
            runtime = run(graphs, raw)
    atomic_json(raw / "runtime.json", runtime)
    value = build(
        graphs, runtime, receipts, coverage, raw, time.monotonic() - started, fixture=fixture
    )
    value["repository_health"] = dict(receipts=health, affects_owned_readiness=False)
    value["field_principles"]["repository_health"] = (
        "Unrelated full-suite health is recorded separately from the new pilot's owned checks."
    )
    value["reproducibility_checksum"] = checksum(value)
    progress("before_terminal_validation")
    terminal_receipts: list[Json] = []

    def validate(candidate: Path) -> Json:
        report = replay(candidate)
        if not fixture:
            py = str(ROOT / ".venv/bin/python")
            commands = [
                CommandSpec(
                    "cold_replay_cli",
                    (
                        py,
                        "-u",
                        "-c",
                        "import os,subprocess,sys,tempfile; env=dict(os.environ); env.pop('PYTHONPATH',None); raise SystemExit(subprocess.run(sys.argv[1:],cwd=tempfile.gettempdir(),env=env,timeout=45).returncode)",
                        py,
                        "-u",
                        str(ROOT / CLI),
                        "--cold-replay",
                        str(candidate),
                    ),
                    "terminal",
                    60,
                ),
                CommandSpec(
                    "adversarial",
                    (py, "scripts/adversarial_verify.py", "--json", str(candidate)),
                    "terminal",
                    60,
                ),
                CommandSpec(
                    "strict_rows",
                    (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
                    "terminal",
                    60,
                ),
            ]
            terminal = run_commands(
                ROOT, commands, log_dir=raw / "terminal_validation", heartbeat_s=15
            )
            terminal_receipts.extend(terminal)
            report.update(
                passed=terminal_passed(terminal),
                receipts=terminal,
                policy="Information-only flags are retained; warnings and critical flags reject publication.",
            )
        return report

    try:
        published = publish_primary(output, value, validate)
    except ValueError as error:
        if str(error) != "candidate_rejected":
            raise
        value["validation_receipts"].extend(terminal_receipts)
        value.update(
            required_checks_passed=False,
            pilot_ready_score=0,
            pilot_error_gate_score=0,
            qwen_runtime_ready_score=0,
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_terminal_validation",
            flagged_adversarial=True,
            status="error",
        )
        value["acceptance_gates"]["owned_checks"] = False
        value["raw_shard_hashes"].extend(
            reference(ROOT / row["log_path"]) for row in terminal_receipts
        )
        value["reproducibility_checksum"] = checksum(value)
        published = publish_primary(output, value, replay)
    progress("terminal_published", value["completed_count"], value["censored_count"])
    print(
        json.dumps(
            dict(**published, honest_verdict=value["honest_verdict"], n_errors=value["n_errors"])
        ),
        flush=True,
    )
    return 0
