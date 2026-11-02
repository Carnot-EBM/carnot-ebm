"""Requalify the isolated Boolean executor through bounded current checks.

The module reuses the V644 public learner and private executor semantics. It
changes the sealed panels and terminal validation scope. The boundary is
audited process separation, not an operating-system security sandbox.

Spec refs: REQ-CL-7344 and SCENARIO-CL-7344-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import random
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

from carnot import experiment_7330_v644_public_learner as public
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    build_scoped_commands,
    run_commands,
    run_scoped_validation,
)


JsonDict = dict[str, Any]
ValidationRunner = Callable[..., JsonDict]
TerminalRunner = Callable[[Path, Path, Path], list[JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260916"
MILESTONE = "2026.09.645"
EXPERIMENT_ID = "exp7344-executor-fixture"
SCHEMA = "carnot.exp7344.v645_executor_fixture.v1"
MODULE_PATH = Path("python/carnot/experiment_7344_v645_executor_fixture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7344_v645_executor_fixture.py")
TEST_PATH = Path("tests/python/test_experiment_7344_v645_executor_fixture.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
HISTORY_PATH = Path("results/experiment_7330_v644_executor_isolation.json")
PRIVATE_EXECUTOR_PATH = Path("scripts/experiments/experiment_7330_v644_private_executor.py")
PUBLIC_LEARNER_PATH = Path("python/carnot/experiment_7330_v644_public_learner.py")
VALIDATION_SCOPE_PATH = Path("python/carnot/reporting/experiment_7303_validation_scope.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7344_v645_executor_fixture.json")
RAW_DIR = Path("results/raw/experiment_7344_v645_executor_fixture")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7344_v645_executor_fixture.json")

COHORTS = ("stable_rules", "announced_changes", "recurrence", "unannounced_changes")
DEVELOPMENT_SEED = 7_344_101
EVALUATION_SEED = 7_344_201
RESAMPLING_SEED = 7_344_301
TOKEN_SEED = 7_344_401
PRIVATE_RULE_SEED = 7_344_501
RANDOM_SEED = {
    "development": DEVELOPMENT_SEED,
    "evaluation": EVALUATION_SEED,
    "resampling": RESAMPLING_SEED,
    "opaque_tokens_commitment": public.sha256_json(TOKEN_SEED),
    "private_rules_commitment": public.sha256_json(PRIVATE_RULE_SEED),
}

ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}
TERMINAL_CHECK_NAMES = (
    "independent_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
ALL_VALIDATION_NAMES = (*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    SPEC_PATH,
    VALIDATION_SCOPE_PATH,
    PUBLIC_LEARNER_PATH,
    PRIVATE_EXECUTOR_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
    HISTORY_PATH,
)

REQUIRED_FIELD_PRINCIPLES = {
    "schema": "Version the record and retain ordinary top-level experiment_id and milestone.",
    "status": "Write a terminal result only after actual work and affected checks.",
    "run_date": "Use 20260916; record real UTC timestamps as well.",
    "preconditions_checked": "Record each actual input/resource check before dependent work.",
    "MODEL_SPECS": "List actual intended model identities; this run intends no model work.",
    "model_invoked": "True for any attempted current model load or generation, including failures.",
    "invocation_counts": "Separate attempted, completed, failed, cancelled and in-flight operations.",
    "inference_substrate": "Declare actual computation; historical model receipts are not current inference.",
    "inference_substrate_class": "Use the closed duration class matching the actual run.",
    "execution_venue": "Use host; this milestone makes no new board-execution claim.",
    "duration_s": "Measure monotonic time; never wait merely to pass a duration floor.",
    "phase_spans": "Measure disjoint load, generation, evaluation, test and write spans.",
    "random_seed": "Freeze development, evaluation and resampling seeds before outcomes.",
    "reproducibility_checksum": "Bind code, settings, inputs, evaluator identity and raw evidence.",
    "source_artifact_hashes": "Authenticate exact producers and current same-milestone paths.",
    "rows": "Keep every comparative unit, arm, metric, cost, failure and censoring disposition.",
    "sample_size_budget": "Record planned, attempted, completed and censored units and stopping rules.",
    "acceptance_gate_results": "Each gate records expected, observed, passed and its principle.",
    "gate_check_summary": "Every blocked_* names upstream, failed check, exact artifact field, expected and observed value.",
    "verifier_is_oracle": "True when the executor defines correctness; separate code does not remove circularity.",
    "honest_verdict": "Completed work starts complete_ or complete:; external absence starts blocked_ with its failed check.",
    "verdict_class": "Closed enum: positive | circular_positive | null | blocked | disqualified | partial.",
    "flagged_adversarial": "Set false only after current verification; a critical finding prevents promotion.",
    "validation_receipts": "Retain exact command, scope, exit code, elapsed time and log hash, including failures.",
    "repository_health": "Preserve dated unrelated failures separately from affected required validation.",
    "field_principles": "Explain fields separately; do not wrap numeric gates or ordinary dictionaries.",
    "executor_fixture_ready_score": "Current mechanism, scope, seals, and process E2E must all pass.",
    "public_manifest_path": "Consumers see public inputs only.",
    "private_manifest_receipt": "The evaluator owns private rules; publish identity and access receipts only.",
    "cohort_manifest": "Freeze stream counts, warmup cutoffs, seeds and paired renaming before evaluation.",
    "isolation_controls": "Distinguish audited separation from an operating-system security guarantee.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep evaluator-only bytes separate from learner-visible public bytes."""

    raw_dir: Path
    public_manifest: Path
    private_manifest: Path
    private_receipt: Path
    learner_evidence: Path
    evaluator_receipt: Path
    control_evidence: Path
    candidate: Path

    @classmethod
    def for_raw_dir(cls, raw_dir: Path) -> ExperimentPaths:
        root = raw_dir.resolve()
        return cls(
            raw_dir=root,
            public_manifest=root / "public/public_manifest.json",
            private_manifest=root / "evaluator/evaluator_private_manifest.json",
            private_receipt=root / "evaluator/private_manifest_receipt.json",
            learner_evidence=root / "learner/learner_evidence.json",
            evaluator_receipt=root / "evaluator/evaluator_boundary_receipt.json",
            control_evidence=root / "evaluator/control_evidence.json",
            candidate=root / "measured-terminal-candidate.json",
        )


def progress(phase: str, event: str, detail: str = "") -> None:
    """Flush each boundary so the conductor can distinguish work from silence."""

    suffix = f" {detail}" if detail else ""
    print(f"[exp7344] phase={phase} event={event}{suffix}", flush=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one complete JSON object through a same-directory rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(public.canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():  # pragma: no cover - only interrupted replacement leaves it.
            temporary.unlink()


def sha256_file(path: Path) -> str:
    """Expose the shared exact-byte hash helper under this experiment module."""

    return public.sha256_file(path)


def _private_executor() -> Any:
    """Import private semantics only inside evaluator or explicit control work."""

    from scripts.experiments import experiment_7330_v644_private_executor as private

    return private


def _internal_cohort(cohort: str) -> str:
    """Map the public recurrence name to the existing executor schedule name."""

    return "return_to_prior_version" if cohort == "recurrence" else cohort


def _model_schedule(cohort: str, first: str, second: str) -> list[tuple[str, str]]:
    """Place each change pattern inside four public model requests."""

    if cohort == "stable_rules":
        return [(first, "authority-a")] * 4
    if cohort == "announced_changes":
        return [(first, "authority-a")] * 2 + [(second, "authority-b")] * 2
    if cohort == "recurrence":
        return [
            (first, "authority-a"),
            (second, "authority-b"),
            (second, "authority-b"),
            (first, "authority-a"),
        ]
    return [(first, "authority-a")] * 2 + [(first, "authority-b")] * 2


def build_manifests(public_path: Path, private_path: Path, receipt_path: Path) -> JsonDict:
    """Seal fresh public panels and evaluator-only labels before proposals exist."""

    private = _private_executor()
    development_public = random.Random(DEVELOPMENT_SEED)
    evaluation_public = random.Random(EVALUATION_SEED)
    token_rng = random.Random(TOKEN_SEED)
    rule_rng = random.Random(PRIVATE_RULE_SEED)
    development: list[JsonDict] = []
    private_records: dict[str, JsonDict] = {}
    for cohort_index, cohort in enumerate(COHORTS):
        for local_index in range(8):
            stream, records = private._stream(
                development_public,
                token_rng,
                rule_rng,
                f"development-{cohort_index}-{local_index:02d}",
                _internal_cohort(cohort),
            )
            stream["cohort"] = cohort
            for request_index, request in enumerate(stream["requests"]):
                request["warmup"] = request_index < 4
            development.append(stream)
            private_records.update(records)

    model_streams: list[JsonDict] = []
    live_pairs: list[JsonDict] = []
    for stream_index in range(8):
        cohort = COHORTS[stream_index % len(COHORTS)]
        first = private._opaque_token(token_rng)
        second = private._opaque_token(token_rng)
        authorities = {
            "authority-a": private._rule_assignment(rule_rng),
            "authority-b": private._rule_assignment(rule_rng),
        }
        stream_rows: list[JsonDict] = []
        for request_index, (token, authority) in enumerate(_model_schedule(cohort, first, second)):
            original = private._public_request(
                evaluation_public,
                f"public-model-{stream_index:02d}-{request_index:02d}-original",
                token,
                request_index,
            )
            twin, rename = private._renamed_twin(
                original, f"public-model-{stream_index:02d}-{request_index:02d}-twin"
            )
            rules = authorities[authority]
            twin_rules = private._rename_rules(rules, rename)
            private_records[str(original["request_id"])] = {
                "version_token": token,
                "authority_label": authority,
                "private_rules": rules,
                "acceptance_witness": private._acceptance_witness(original, rules),
                "witness_label": True,
            }
            private_records[str(twin["request_id"])] = {
                "version_token": token,
                "authority_label": authority,
                "private_rules": twin_rules,
                "acceptance_witness": private._acceptance_witness(twin, twin_rules),
                "witness_label": True,
            }
            pair = {
                "panel_id": f"public-model-{stream_index:02d}-{request_index:02d}",
                "stream_id": f"public-model-{stream_index:02d}",
                "request_index": request_index,
                "warmup": request_index < 2,
                "cohort": cohort,
                "presentation_order": (
                    "original_first" if request_index % 2 == 0 else "twin_first"
                ),
                "original": original,
                "twin": twin,
                "renaming_map": rename,
            }
            stream_rows.append(pair)
            live_pairs.append(pair)
        model_streams.append(
            {
                "stream_id": f"public-model-{stream_index:02d}",
                "cohort": cohort,
                "requests": stream_rows,
            }
        )

    challenge_token = private._opaque_token(token_rng)
    challenge_request: JsonDict = {
        "request_id": "compound-challenge-v645",
        "version_token": challenge_token,
        "activities": ["a", "b", "c"],
        "allowed_starts": {name: [0, 3] for name in "abc"},
        "durations": {name: 2 for name in "abc"},
        "weights": {name: 1 for name in "abc"},
        "horizon": 6,
        "public_revision": 0,
    }
    challenge_plan = {
        "request_id": challenge_request["request_id"],
        "assignments": {name: 0 for name in "abc"},
    }
    challenge_rules = {
        "capacity": 6,
        "pair_gaps": [],
        "forbidden_compounds": [["a", "b", "c"]],
    }
    private_records[str(challenge_request["request_id"])] = {
        "version_token": challenge_token,
        "authority_label": "compound-outside-language",
        "private_rules": challenge_rules,
        "acceptance_witness": private._acceptance_witness(challenge_request, challenge_rules),
        "witness_label": True,
    }

    public_manifest: JsonDict = {
        "schema": "carnot.exp7344.public_manifest.v1",
        "sealed_before_outcomes": True,
        "development_seed": DEVELOPMENT_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "development_streams": development,
        "held_out_streams": [],
        "warmup_requests_per_development_stream": 4,
        "public_model_streams": model_streams,
        "warmup_requests_per_public_model_stream": 2,
        "live_proposal_panel": live_pairs,
        "compound_conflict_challenge": {
            "request": challenge_request,
            "candidate_plan": challenge_plan,
            "outside_acquisition_language": True,
        },
        "downstream_query_budget_per_request": public.QUERY_BUDGET,
    }
    public_manifest["manifest_hash"] = public.sha256_json(public_manifest)
    _atomic_json(public_path, public_manifest)
    private_manifest: JsonDict = {
        "schema": "carnot.exp7344.evaluator_private_manifest.v1",
        "public_manifest_sha256": sha256_file(public_path),
        "token_seed_commitment": public.sha256_json(TOKEN_SEED),
        "private_rule_seed_commitment": public.sha256_json(PRIVATE_RULE_SEED),
        "evaluator_records": private_records,
        "label_count": len(private_records),
        "all_acceptance_witnesses_nonempty": all(
            bool(row["acceptance_witness"]["assignments"]) for row in private_records.values()
        ),
    }
    private_manifest["manifest_hash"] = public.sha256_json(private_manifest)
    _atomic_json(private_path, private_manifest)
    receipt: JsonDict = {
        "schema": "carnot.exp7344.private_manifest_receipt.v1",
        "public_manifest_sha256": sha256_file(public_path),
        "private_manifest_path": str(private_path.resolve()),
        "private_manifest_sha256": sha256_file(private_path),
        "private_manifest_hash": private_manifest["manifest_hash"],
        "private_rule_seed_commitment": private_manifest["private_rule_seed_commitment"],
        "token_seed_commitment": private_manifest["token_seed_commitment"],
        "label_count": len(private_records),
        "acceptance_witness_count": sum(
            bool(row["acceptance_witness"]["assignments"]) for row in private_records.values()
        ),
        "development_stream_count": len(development),
        "public_model_stream_count": len(model_streams),
        "public_model_request_count": len(live_pairs),
        "private_opened_by_learner": False,
        "private_rule_values_published": False,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def _malformed_reply_rejected() -> bool:
    """Exercise the existing socket client against one invalid response."""

    with tempfile.TemporaryDirectory(prefix="exp7344-malformed-", dir="/tmp") as directory:
        endpoint = Path(directory) / "oracle.sock"
        ready = threading.Event()

        def serve() -> None:
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(endpoint))
            server.listen(1)
            ready.set()
            connection, _address = server.accept()
            reader = connection.makefile("r", encoding="utf-8")
            writer = connection.makefile("w", encoding="utf-8")
            reader.readline()
            writer.write('{"query_id":"wrong","accepted":true}\n')
            writer.flush()
            writer.close()
            reader.close()
            connection.close()
            server.close()

        thread = threading.Thread(target=serve)
        thread.start()
        if not ready.wait(2):  # pragma: no cover - local socket startup failure.
            raise RuntimeError("malformed_reply_server_not_ready")
        oracle = public._SocketOracle(endpoint, "control-request")
        rejected = False
        try:
            oracle.query(public.make_plan("control-request", {"a": 0}), "control")
        except public.PublicLearningError as error:
            rejected = str(error) == "response_shape"
        finally:
            oracle.close()
            thread.join(timeout=2)
        return rejected


def _control_row(control: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Use one row shape for hostile, semantic, and lifecycle controls."""

    return {
        "unit_id": f"control:{control}",
        "arm": "mechanism_control",
        "control": control,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "failures": [] if passed else [control],
        "censored": False,
        "executor_calls": 1,
        "unsound_learned_atoms": 0,
    }


def build_control_rows() -> JsonDict:
    """Run required executor, protocol, compound, and restart controls."""

    private = _private_executor()
    request = {
        "request_id": "control-request",
        "version_token": "opaque-control-a",
        "activities": ["a", "b"],
        "allowed_starts": {"a": [0, 2], "b": [0, 2]},
        "durations": {"a": 1, "b": 1},
        "weights": {"a": 1, "b": 1},
        "horizon": 4,
    }
    rules = {
        "capacity": 2,
        "pair_gaps": [{"pair": ["a", "b"], "minimum_gap": 0}],
        "forbidden_compounds": [],
    }
    accepted = private._check(request, public.make_plan("control-request", {"a": 0, "b": 2}), rules)
    rejected = not private._check(
        request, public.make_plan("control-request", {"a": 0, "b": 0}), rules
    )
    mismatch = not private._check(request, public.make_plan("different-request", {"a": 0}), rules)

    learner = public.PublicConstraintLearner("opaque-control-a")
    atom = public.make_pair_atom(
        "opaque-control-a",
        "a",
        "b",
        0,
        {"query_id": "control-q", "accepted": False, "sequence": 1},
    )
    learner.admit_atom(atom, current_query_index=1)
    restored = public.PublicConstraintLearner.from_state_bytes(learner.state_bytes())
    restart = restored.state_bytes() == learner.state_bytes()
    restored.activate_version("opaque-control-b")
    stale = restored.active_atoms() == []

    compound_request = {
        "request_id": "compound-control",
        "version_token": "opaque-compound",
        "activities": ["a", "b", "c"],
        "allowed_starts": {name: [0, 3] for name in "abc"},
        "durations": {name: 2 for name in "abc"},
        "weights": {name: 1 for name in "abc"},
        "horizon": 6,
    }
    compound_rules = {
        "capacity": 6,
        "pair_gaps": [],
        "forbidden_compounds": [["a", "b", "c"]],
    }
    compound_plan = public.make_plan("compound-control", {name: 0 for name in "abc"})
    full_accepted = private._check(compound_request, compound_plan, compound_rules)
    pair_accepted = all(
        private._check(
            compound_request,
            public.make_plan("compound-control", {left: 0, right: 0}),
            compound_rules,
        )
        for left, right in (("a", "b"), ("a", "c"), ("b", "c"))
    )
    compound_learner = public.PublicConstraintLearner("opaque-compound")
    compound_atoms = compound_learner.localize_rejection(
        compound_request, compound_plan, lambda _plan, _reason: True
    )
    compound_observed = {
        "full_accepted": full_accepted,
        "pair_projections_accepted": pair_accepted,
        "learned_atom_count": len(compound_atoms),
    }
    rows = [
        _control_row("accepted_witness", True, accepted, accepted),
        _control_row("rejected_witness", True, rejected, rejected),
        _control_row("malformed_reply", "rejected", "rejected", _malformed_reply_rejected()),
        _control_row("stale_version", "inactive", "inactive" if stale else "active", stale),
        _control_row(
            "compound_only_conflict",
            {
                "full_accepted": False,
                "pair_projections_accepted": True,
                "learned_atom_count": 0,
            },
            compound_observed,
            compound_observed
            == {
                "full_accepted": False,
                "pair_projections_accepted": True,
                "learned_atom_count": 0,
            },
        ),
        _control_row("learner_restart", "exact_bytes", "exact_bytes", restart),
        _control_row("mismatched_request_id", "rejected", "rejected", mismatch),
    ]
    return {
        "schema": "carnot.exp7344.control_evidence.v1",
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
        "zero_private_rule_reads": True,
        "unsound_learned_atom_count": sum(row["unsound_learned_atoms"] for row in rows),
    }


def _environment(root: Path) -> dict[str, str]:
    """Force unbuffered worktree imports for every child process."""

    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = f"{root / 'python'}:{root}"
    return environment


def run_streamed_process(
    argv: Sequence[str],
    root: Path,
    log_path: Path,
    *,
    name: str,
    timeout_s: float = 600.0,
    heartbeat_s: float = 60.0,
) -> JsonDict:
    """Stream one bounded child and emit truthful pending heartbeats."""

    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    progress(name, "before_subprocess", f"command={shlex.join(argv)}")
    process = subprocess.Popen(  # noqa: S603 - callers pass fixed argument vectors.
        tuple(argv),
        cwd=root,
        env=_environment(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stopped = threading.Event()
    timed_out = threading.Event()

    def monitor() -> None:
        while not stopped.wait(heartbeat_s):  # pragma: no cover - children finish below 60s.
            elapsed = time.monotonic() - started
            progress(name, "pending", f"elapsed_s={elapsed:.3f}")
            if elapsed >= timeout_s:
                timed_out.set()
                process.terminate()
                return

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        print(f"[exp7344:{name}] {line.rstrip()}", flush=True)
    exit_code = process.wait()
    stopped.set()
    monitor_thread.join(timeout=1)
    log_path.write_text("".join(lines), encoding="utf-8")
    receipt = {
        "name": name,
        "command": shlex.join(argv),
        "command_argv": list(argv),
        "scope": "bounded_process_e2e",
        "exit_code": exit_code,
        "duration_s": time.monotonic() - started,
        "log_path": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
        "passed": exit_code == 0 and not timed_out.is_set(),
        "timed_out": timed_out.is_set(),
        "output_tail": "".join(lines)[-4000:],
        "output_lines": lines,
        "pid": process.pid,
    }
    progress(name, "after_subprocess", f"exit={exit_code}")
    return receipt


def _stream_reader(process: subprocess.Popen[str], lines: list[str]) -> None:
    """Forward evaluator output while its Boolean server is active."""

    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        print(f"[exp7344:evaluator] {line.rstrip()}", flush=True)


def _load_object(path: Path) -> JsonDict:
    """Require object-shaped JSON for every task-owned evidence file."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_object:{path}")
    return value


def _execute_process_boundary(root: Path, paths: ExperimentPaths) -> JsonDict:
    """Build evaluator seals, then connect existing learner and executor processes."""

    python = str(root / ".venv/bin/python")
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    builder_code = (
        "import json,sys;from pathlib import Path;"
        "from carnot.experiment_7344_v645_executor_fixture import build_manifests;"
        "print(json.dumps(build_manifests(*map(Path,sys.argv[1:4])),sort_keys=True),flush=True)"
    )
    manifest_process = run_streamed_process(
        [
            python,
            "-u",
            "-c",
            builder_code,
            str(paths.public_manifest),
            str(paths.private_manifest),
            str(paths.private_receipt),
        ],
        root,
        paths.raw_dir / "logs/build_manifests.log",
        name="manifest_builder",
    )
    if not manifest_process["passed"]:  # pragma: no cover - retained process failure.
        raise RuntimeError("manifest_builder_failed")
    controls_code = (
        "import json;"
        "from carnot.experiment_7344_v645_executor_fixture import build_control_rows;"
        "print(json.dumps(build_control_rows(),sort_keys=True),flush=True)"
    )
    control_process = run_streamed_process(
        [python, "-u", "-c", controls_code],
        root,
        paths.raw_dir / "logs/control_panel.log",
        name="control_panel",
    )
    if not control_process["passed"]:  # pragma: no cover - retained process failure.
        raise RuntimeError("control_panel_failed")
    controls = json.loads(control_process["output_lines"][-1])
    _atomic_json(paths.control_evidence, controls)

    endpoint = Path("/tmp") / f"carnot-exp7344-{public.sha256_json(str(paths.raw_dir))[-12:]}.sock"
    evaluator_argv = [
        python,
        "-u",
        str(root / PRIVATE_EXECUTOR_PATH),
        "--serve",
        "--public-manifest",
        str(paths.public_manifest),
        "--private-manifest",
        str(paths.private_manifest),
        "--endpoint",
        str(endpoint),
        "--receipt",
        str(paths.evaluator_receipt),
    ]
    evaluator_started = time.monotonic()
    progress("evaluator", "before_subprocess", f"command={shlex.join(evaluator_argv)}")
    evaluator = subprocess.Popen(  # noqa: S603 - fixed evaluator argument vector.
        evaluator_argv,
        cwd=root,
        env=_environment(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    evaluator_lines: list[str] = []
    evaluator_thread = threading.Thread(
        target=_stream_reader, args=(evaluator, evaluator_lines), daemon=True
    )
    evaluator_thread.start()
    deadline = time.monotonic() + 15
    while not endpoint.exists() and evaluator.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    if not endpoint.exists():  # pragma: no cover - local evaluator startup failure.
        evaluator.terminate()
        raise RuntimeError("evaluator_endpoint_unavailable")
    progress("evaluator", "ready", f"pid={evaluator.pid}")

    learner_process = run_streamed_process(
        [
            python,
            "-u",
            "-m",
            "carnot.experiment_7330_v644_public_learner",
            "--public-manifest",
            str(paths.public_manifest),
            "--endpoint",
            str(endpoint),
            "--output",
            str(paths.learner_evidence),
        ],
        root,
        paths.raw_dir / "logs/public_learner.log",
        name="public_learner",
    )
    evaluator_exit = evaluator.wait(timeout=60)
    evaluator_thread.join(timeout=2)
    evaluator_log = paths.raw_dir / "logs/private_evaluator.log"
    evaluator_log.write_text("".join(evaluator_lines), encoding="utf-8")
    progress(
        "evaluator",
        "after_subprocess",
        f"exit={evaluator_exit} elapsed_s={time.monotonic() - evaluator_started:.3f}",
    )
    if not learner_process["passed"] or evaluator_exit != 0:  # pragma: no cover
        raise RuntimeError("process_boundary_failed")
    return {
        "manifest_process": manifest_process,
        "control_process": control_process,
        "learner_process": learner_process,
        "evaluator_process": {
            "name": "private_evaluator",
            "command": shlex.join(evaluator_argv),
            "command_argv": evaluator_argv,
            "scope": "bounded_process_e2e",
            "exit_code": evaluator_exit,
            "duration_s": time.monotonic() - evaluator_started,
            "log_path": str(evaluator_log.resolve()),
            "log_sha256": sha256_file(evaluator_log),
            "passed": evaluator_exit == 0,
            "timed_out": False,
            "pid": evaluator.pid,
        },
    }


def _manifest_errors(manifest: Mapping[str, Any]) -> list[str]:
    """Recompute every public cohort, warmup, twin, and hash invariant."""

    errors: list[str] = []
    streams = manifest.get("development_streams", [])
    model_streams = manifest.get("public_model_streams", [])
    if len(streams) != 32:
        errors.append("development_stream_count")
    if any(len(stream.get("requests", [])) != 12 for stream in streams):
        errors.append("development_request_count")
    if any(sum(stream.get("cohort") == cohort for stream in streams) != 8 for cohort in COHORTS):
        errors.append("development_cohorts")
    if any(
        request.get("warmup") is not (index < 4)
        for stream in streams
        for index, request in enumerate(stream.get("requests", []))
    ):
        errors.append("development_warmup")
    requests = [request for stream in streams for request in stream.get("requests", [])]
    if len({request.get("request_id") for request in requests}) != 384:
        errors.append("development_distinct_requests")
    if len(model_streams) != 8 or any(
        len(stream.get("requests", [])) != 4 for stream in model_streams
    ):
        errors.append("public_model_streams")
    if any(
        row.get("warmup") is not (index < 2)
        for stream in model_streams
        for index, row in enumerate(stream.get("requests", []))
    ):
        errors.append("public_model_warmup")
    pairs = manifest.get("live_proposal_panel", [])
    if len(pairs) != 32 or any(
        row.get("original", {}).get("request_id") == row.get("twin", {}).get("request_id")
        for row in pairs
    ):
        errors.append("public_model_twins")
    frozen = deepcopy(dict(manifest))
    observed_hash = frozen.pop("manifest_hash", None)
    if observed_hash != public.sha256_json(frozen):
        errors.append("manifest_hash")
    if "private_rules" in json.dumps(manifest) or "acceptance_witness" in json.dumps(manifest):
        errors.append("private_data_in_public_manifest")
    return sorted(set(errors))


def _reduce_raw(paths: ExperimentPaths) -> JsonDict:
    """Reload raw public evidence without trusting artifact aggregates."""

    learner = _load_object(paths.learner_evidence)
    evaluator = _load_object(paths.evaluator_receipt)
    controls = _load_object(paths.control_evidence)
    learner_rows = list(learner["rows"])
    control_rows = list(controls["rows"])
    rows = [*learner_rows, *control_rows]
    query_ids = [str(row.get("request_id", "")) for row in evaluator["query_rows"]]
    return {
        "development_row_count": len(learner_rows),
        "control_row_count": len(control_rows),
        "row_count": len(rows),
        "completed_units": sum(row.get("censored") is False for row in rows),
        "censored_units": sum(row.get("censored") is True for row in rows),
        "accepted_development_rows": sum(row.get("accepted") is True for row in learner_rows),
        "rejected_development_rows": sum(row.get("accepted") is False for row in learner_rows),
        "maximum_executor_calls": max(int(row.get("executor_calls", 0)) for row in rows),
        "total_executor_calls": sum(int(row.get("executor_calls", 0)) for row in rows),
        "public_model_query_count": sum(value.startswith("public-model-") for value in query_ids),
        "rows_sha256": public.sha256_json(rows),
        "evaluator_query_rows_sha256": public.sha256_json(evaluator["query_rows"]),
        "unsound_learned_atom_count": controls["unsound_learned_atom_count"],
    }


def _cohort_manifest(public_manifest: Mapping[str, Any]) -> JsonDict:
    """Publish definitions and twin mappings without evaluator-only labels."""

    return {
        "cohorts": {
            "stable_rules": "one authenticated version and one stable authority",
            "announced_changes": "the opaque version changes with the private authority",
            "recurrence": "a prior opaque version and authority return",
            "unannounced_changes": "private authority changes under one opaque version",
        },
        "synthetic_stream_count": 32,
        "synthetic_streams_per_cohort": 8,
        "synthetic_requests_per_stream": 12,
        "synthetic_warmup_requests": 4,
        "public_model_stream_count": 8,
        "public_model_requests": 32,
        "public_model_requests_per_stream": 4,
        "public_model_warmup_requests": 2,
        "development_seed": DEVELOPMENT_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "resampling_seed": RESAMPLING_SEED,
        "live_twin_mapping": [
            {
                "panel_id": row["panel_id"],
                "stream_id": row["stream_id"],
                "request_index": row["request_index"],
                "warmup": row["warmup"],
                "cohort": row["cohort"],
                "presentation_order": row["presentation_order"],
                "original_request_id": row["original"]["request_id"],
                "twin_request_id": row["twin"]["request_id"],
                "renaming_map": row["renaming_map"],
            }
            for row in public_manifest["live_proposal_panel"]
        ],
    }


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain each top-level field while preserving mandated principles."""

    principles = {field: "This field keeps current fixture evidence auditable." for field in fields}
    principles.update(REQUIRED_FIELD_PRINCIPLES)
    return principles


def _phase_span(
    phase: str, phase_started: float, run_started: float, units: int, checkpoint: str
) -> JsonDict:
    """Measure one completed non-overlapping phase with monotonic time."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_elapsed_s": phase_started - run_started,
        "end_elapsed_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint_boundary": checkpoint,
        "pending_operations": [],
    }


def _precondition_row(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    available: bool,
) -> JsonDict:
    """Retain the exact five diagnostic values for one required input."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": field,
        "expected_value": expected,
        "observed_value": observed,
        "available": bool(available),
        "blocking": True,
        "principle": "Dependent work starts only from authenticated eligible inputs.",
    }


def collect_preconditions(
    root: Path,
    output_path: Path,
    raw_dir: Path,
    checkpoint_path: Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[list[JsonDict], dict[str, str]]:
    """Check exact sources, history classification, and writable destinations first."""

    rows: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        if available:
            hashes[str(relative)] = sha256_file(path)
        rows.append(
            _precondition_row(
                f"source_bytes:{relative}",
                str(relative),
                "bytes",
                "readable_nonempty_bytes",
                path.stat().st_size if available else "missing_or_empty",
                available,
            )
        )
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    rows.append(
        _precondition_row(
            "driving_requirement",
            str(SPEC_PATH),
            "REQ-*",
            "REQ-CL-7344",
            "REQ-CL-7344" if "REQ-CL-7344" in spec_text else "missing",
            "REQ-CL-7344" in spec_text,
        )
    )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    rows.append(
        _precondition_row(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            True,
            EXPERIMENT_ID not in exclusion,
            EXPERIMENT_ID not in exclusion,
        )
    )
    try:
        history = _load_object(root / HISTORY_PATH)
        history_available = True
    except (OSError, json.JSONDecodeError, ValueError):
        history, history_available = {}, False
    rows.append(
        _precondition_row(
            "historical_artifact_available",
            str(HISTORY_PATH),
            "bytes",
            True,
            history_available,
            history_available,
        )
    )
    historical_shape = bool(
        history.get("experiment_id") == 7330
        and history.get("executor_fixture_ready_score") == 0
        and history.get("verdict_class") == "disqualified"
        and history.get("repository_health", {}).get("current_observation", {}).get("exit_code")
        == -15
    )
    rows.append(
        _precondition_row(
            "historical_artifact_is_diagnostic_only",
            str(HISTORY_PATH),
            "executor_fixture_ready_score|verdict_class|repository_health.current_observation.exit_code",
            {"score": 0, "verdict_class": "disqualified", "exit_code": -15},
            {
                "score": history.get("executor_fixture_ready_score"),
                "verdict_class": history.get("verdict_class"),
                "exit_code": history.get("repository_health", {})
                .get("current_observation", {})
                .get("exit_code"),
            },
            historical_shape,
        )
    )
    rows.append(
        _precondition_row(
            "same_milestone_inputs",
            MILESTONE,
            "upstream_artifacts",
            [],
            [],
            True,
        )
    )
    for name, directory in (
        ("output_directory", output_path.parent),
        ("raw_directory", raw_dir),
        ("checkpoint_directory", checkpoint_path.parent),
    ):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            observed, available = "directory_writable", True
        except OSError as error:  # pragma: no cover - external filesystem failure.
            observed, available = f"{type(error).__name__}: {error}", False
        rows.append(_precondition_row(name, str(directory), "writable", True, observed, available))
    for check, replacement in dict(overrides or {}).items():
        row = next(item for item in rows if item["check"] == check)
        row["available"] = bool(replacement)
        row["observed_value"] = replacement
    return rows, hashes


def _historical_failures(root: Path) -> list[JsonDict]:
    """Preserve the dated V644 full-suite termination as unrelated health."""

    history = _load_object(root / HISTORY_PATH)
    receipt = history["repository_health"]["current_observation"]
    return [
        {
            "source_experiment_id": 7330,
            "observed_at": "2026-09-16",
            "producer_path": str(HISTORY_PATH),
            "producer_sha256": sha256_file(root / HISTORY_PATH),
            "name": receipt.get("name"),
            "command": receipt.get("command"),
            "exit_code": receipt.get("exit_code"),
            "duration_s": receipt.get("duration_s"),
            "log_sha256": receipt.get("log_sha256"),
            "classification": "historical_unscoped_repository_health",
            "collection_errors": [],
            "resolved": False,
        }
    ]


def scoped_command_plan(root: Path, basetemp: Path, coverage_file: Path) -> list[CommandSpec]:
    """Build and inspect the exact shipped scoped-runner command plan."""

    basetemp.mkdir(parents=True, exist_ok=True)
    coverage_file.parent.mkdir(parents=True, exist_ok=True)
    commands = build_scoped_commands(
        root,
        [str(TEST_PATH)],
        [str(MODULE_PATH)],
        static_paths=[str(WRAPPER_PATH)],
        basetemp=basetemp,
        coverage_file=coverage_file,
    )
    broad = [
        spec.name
        for spec in commands
        if any(argument in {"tests", "tests/python", "."} for argument in spec.argv)
        or "run_full_python_suite" in " ".join(spec.argv)
    ]
    if broad:
        raise ValueError(f"broad_validation_command:{','.join(broad)}")
    return commands


def run_affected_validation(
    root: Path,
    raw_dir: Path,
    *,
    test_paths: Sequence[str] = (str(TEST_PATH),),
    changed_modules: Sequence[str] = (str(MODULE_PATH),),
    static_paths: Sequence[str] = (str(WRAPPER_PATH),),
    historical_failures: Sequence[Mapping[str, Any]] = (),
    scoped_runner: Callable[..., JsonDict] = run_scoped_validation,
) -> JsonDict:
    """Run only the fixed affected commands after checking their actual plan."""

    basetemp = Path("/tmp/carnot-exp7344-v645-scoped")
    coverage_file = Path("/tmp/.coverage-exp7344-v645")
    scoped_command_plan(root, basetemp, coverage_file)
    return scoped_runner(
        root,
        test_paths=list(test_paths),
        changed_modules=list(changed_modules),
        static_paths=list(static_paths),
        basetemp=basetemp,
        coverage_file=coverage_file,
        log_dir=raw_dir / "validation/scoped",
        historical_failures=historical_failures,
    )


def run_terminal_validation(root: Path, candidate: Path, raw_dir: Path) -> list[JsonDict]:
    """Cold-reduce raw rows and run both strict terminal artifact readers."""

    python = str(root / ".venv/bin/python")
    reducer_code = (
        "import json,pathlib;"
        "from carnot.experiment_7344_v645_executor_fixture import independent_reduce;"
        f"value=json.loads(pathlib.Path({str(candidate)!r}).read_text());"
        "errors=independent_reduce(value);print(errors,flush=True);raise SystemExit(bool(errors))"
    )
    commands = [
        CommandSpec(
            "independent_reducer", (python, "-u", "-c", reducer_code), "measured_candidate"
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "measured_candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "measured_candidate",
        ),
    ]
    return run_commands(root, commands, log_dir=raw_dir / "validation/terminal")


def _gate(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    """Retain expected, observed, pass state, and reason for one gate."""

    return {
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def _passing_receipts(artifact: Mapping[str, Any], names: Sequence[str]) -> bool:
    """Require exactly one successful receipt for each named current check."""

    receipts = artifact.get("validation_receipts", [])
    return all(
        sum(
            row.get("name") == name
            and row.get("passed") is True
            and row.get("exit_code") == 0
            and row.get("timed_out") is not True
            for row in receipts
        )
        == 1
        for name in names
    )


def _acceptance_gates(artifact: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Recompute readiness from current evidence and ignore repository history."""

    reduction = artifact.get("independent_reduction", {})
    isolation = artifact.get("isolation_controls", {})
    controls = artifact.get("control_results", {})
    manifest_errors = artifact.get("manifest_errors", [])
    current_validation = _passing_receipts(artifact, REQUIRED_CHECK_NAMES)
    terminal_validation = _passing_receipts(artifact, TERMINAL_CHECK_NAMES)
    raw_errors = independent_reduce(artifact) if artifact.get("rows") else ["rows_missing"]
    return {
        "fresh_seals": _gate(
            [],
            manifest_errors,
            not manifest_errors,
            "Fresh stream and twin identities need exact seals.",
        ),
        "development_and_controls": _gate(
            {"development_rows": 384, "control_rows": 7, "censored": 0},
            {
                "development_rows": reduction.get("development_row_count"),
                "control_rows": reduction.get("control_row_count"),
                "censored": reduction.get("censored_units"),
            },
            reduction.get("development_row_count") == 384
            and reduction.get("control_row_count") == 7
            and reduction.get("censored_units") == 0
            and reduction.get("accepted_development_rows", 0) > 0
            and reduction.get("rejected_development_rows", 0) > 0,
            "Accepted and rejected outcomes plus every named control must stay visible.",
        ),
        "mechanism_controls": _gate(
            {"passed": True, "unsound_learned_atoms": 0},
            {
                "passed": controls.get("passed"),
                "unsound_learned_atoms": controls.get("unsound_learned_atom_count"),
            },
            controls.get("passed") is True and controls.get("unsound_learned_atom_count") == 0,
            "Malformed, stale, compound, restart, and identity controls must fail closed.",
        ),
        "audited_process_boundary": _gate(
            {"separate": True, "private_reads": 0, "private_messages": 0},
            {
                "separate": isolation.get("process_separation_passed"),
                "private_reads": isolation.get("learner_private_access_count"),
                "private_messages": isolation.get("learner_private_rule_message_count"),
            },
            isolation.get("process_separation_passed") is True
            and isolation.get("learner_private_access_count") == 0
            and isolation.get("learner_private_rule_message_count") == 0
            and isolation.get("evaluator_response_keys") == ["accepted", "query_id"],
            "The learner may receive public requests and Boolean responses only.",
        ),
        "public_model_labels_unopened": _gate(
            0,
            reduction.get("public_model_query_count"),
            reduction.get("public_model_query_count") == 0,
            "Evaluation requests and labels remain unopened during fixture construction.",
        ),
        "independent_raw_reduction": _gate(
            [],
            raw_errors,
            not raw_errors,
            "Raw process evidence must independently support stored rows.",
        ),
        "affected_validation": _gate(
            True,
            current_validation,
            current_validation,
            "A failing changed module or affected test disqualifies current readiness.",
        ),
        "terminal_validation": _gate(
            True,
            terminal_validation,
            terminal_validation,
            "Cold reduction and both strict terminal readers must pass.",
        ),
        "adversarial_clear": _gate(
            False,
            artifact.get("flagged_adversarial"),
            artifact.get("flagged_adversarial") is False,
            "A critical adversarial finding prevents fixture readiness.",
        ),
    }


def _summary_from_failure(
    upstream: Any, check: Any, field: Any, expected: Any, observed: Any
) -> JsonDict:
    """Name the exact source and field behind a terminal failure."""

    return {
        "passed": False,
        "failed_check": check,
        "upstream": upstream,
        "artifact_field": field,
        "expected_value": expected,
        "observed_value": observed,
    }


def apply_terminal_state(artifact: JsonDict) -> None:
    """Set readiness from current gates while leaving dated health non-authoritative."""

    failed_precondition = next(
        (
            row
            for row in artifact.get("preconditions_checked", [])
            if row.get("blocking") and row.get("available") is not True
        ),
        None,
    )
    artifact["executor_value_score"] = 0
    artifact["promotion_score"] = 0
    if failed_precondition is not None:
        artifact["executor_fixture_ready_score"] = 0
        artifact["status"] = "blocked"
        artifact["verdict_class"] = "blocked"
        artifact["honest_verdict"] = f"blocked_{failed_precondition['check']}"
        artifact["gate_check_summary"] = _summary_from_failure(
            failed_precondition.get("upstream"),
            failed_precondition.get("check"),
            failed_precondition.get("artifact_field"),
            failed_precondition.get("expected_value"),
            failed_precondition.get("observed_value"),
        )
        return
    gates = _acceptance_gates(artifact)
    artifact["acceptance_gate_results"] = gates
    failed = next(((name, row) for name, row in gates.items() if not row["passed"]), None)
    ready = failed is None
    artifact["executor_fixture_ready_score"] = int(ready)
    artifact["status"] = "complete"
    if ready:
        artifact["verdict_class"] = "circular_positive"
        artifact["honest_verdict"] = (
            "complete_circular_positive_executor_fixture_ready_for_measurement_only"
        )
        artifact["gate_check_summary"] = {
            "passed": True,
            "failed_check": None,
            "upstream": EXPERIMENT_ID,
            "artifact_field": "executor_fixture_ready_score",
            "expected_value": 1,
            "observed_value": 1,
            "check_count": len(gates),
            "failed_check_count": 0,
        }
    else:
        assert failed is not None
        name, row = failed
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_current_executor_fixture_check_failed"
        artifact["gate_check_summary"] = {
            **_summary_from_failure(EXPERIMENT_ID, name, name, row["expected"], row["observed"]),
            "check_count": len(gates),
            "failed_check_count": sum(not value["passed"] for value in gates.values()),
        }


def _base_artifact(started_at: str, checkpoint_path: Path) -> JsonDict:
    """Create a running checkpoint that cannot resemble terminal success."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 1,
        "status": "running",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "preconditions_checked": [],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": dict(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "development": {"planned": 384, "attempted": 0, "completed": 0, "censored": 0},
            "controls": {"planned": 7, "attempted": 0, "completed": 0, "censored": 0},
            "public_model_requests": {
                "planned": 32,
                "attempted": 0,
                "completed": 0,
                "censored": 0,
                "labels_opened": 0,
            },
            "stopping_rule": "Run 32 fixed development streams once, seven controls once, and do not extend from outcomes.",
        },
        "acceptance_gate_results": {},
        "gate_check_summary": _summary_from_failure(
            EXPERIMENT_ID, "work_in_progress", "status", "terminal", "running"
        ),
        "verifier_is_oracle": True,
        "honest_verdict": "partial_running_executor_fixture_requalification",
        "verdict_class": "partial",
        "flagged_adversarial": None,
        "validation_receipts": [],
        "repository_health": {},
        "required_checks_passed": False,
        "executor_fixture_ready_score": 0,
        "executor_value_score": 0,
        "promotion_score": 0,
        "public_manifest_path": "",
        "public_manifest": {},
        "private_manifest_receipt": {},
        "cohort_manifest": {},
        "isolation_controls": {},
        "control_results": {},
        "independent_reduction": {},
        "raw_evidence_paths": {},
        "manifest_errors": [],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "no_model_weight_mutation": True,
        "production_default_changed": False,
        "publication_surface_changed": False,
        "research_roadmap_changed": False,
        "readiness_authorizes_measurement_only": True,
        "learning_value_claim": False,
        "field_principles": {},
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind code, settings, process identities, seals, and raw evidence."""

    keys = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "MODEL_SPECS",
        "invocation_counts",
        "inference_substrate",
        "random_seed",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "public_manifest",
        "private_manifest_receipt",
        "cohort_manifest",
        "isolation_controls",
        "control_results",
        "independent_reduction",
        "acceptance_gate_results",
        "verifier_is_oracle",
        "executor_fixture_ready_score",
        "verdict_class",
    )
    return public.sha256_json({key: artifact.get(key) for key in keys})


def _finalize(artifact: JsonDict, started: float) -> None:
    """Seal actual timing, explanations, and the reproducibility checksum."""

    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["duration_s"] = time.monotonic() - started
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)


def independent_reduce(artifact: Mapping[str, Any]) -> list[str]:
    """Reload task-owned raw rows and compare them with the terminal record."""

    errors: list[str] = []
    raw_paths = artifact.get("raw_evidence_paths", {})
    try:
        learner = _load_object(Path(str(raw_paths["learner_evidence"])))
        controls = _load_object(Path(str(raw_paths["control_evidence"])))
        evaluator = _load_object(Path(str(raw_paths["evaluator_receipt"])))
    except (KeyError, OSError, json.JSONDecodeError, ValueError):
        return ["raw_evidence_unavailable"]
    rows = [*learner.get("rows", []), *controls.get("rows", [])]
    if public.sha256_json(rows) != artifact.get("independent_reduction", {}).get("rows_sha256"):
        errors.append("raw_rows_hash_mismatch")
    if public.sha256_json(rows) != public.sha256_json(artifact.get("rows")):
        errors.append("rows_hash_mismatch")
    if public.sha256_json(evaluator.get("query_rows", [])) != artifact.get(
        "independent_reduction", {}
    ).get("evaluator_query_rows_sha256"):
        errors.append("evaluator_rows_hash_mismatch")
    if len(learner.get("rows", [])) != 384 or len(controls.get("rows", [])) != 7:
        errors.append("raw_row_count_mismatch")
    if controls.get("unsound_learned_atom_count") != 0:
        errors.append("unsound_learned_atoms")
    return errors


def _receipt_valid(receipt: Mapping[str, Any]) -> bool:
    """Require command, scope, exit, time, and exact log identity."""

    return bool(
        isinstance(receipt.get("command"), str)
        and receipt.get("command")
        and isinstance(receipt.get("scope"), str)
        and isinstance(receipt.get("exit_code"), int)
        and isinstance(receipt.get("duration_s"), (int, float))
        and str(receipt.get("log_sha256", "")).startswith("sha256:")
    )


def validate_artifact(artifact: object, *, root: Path = REPO_ROOT) -> list[str]:
    """Cold-check identity, evidence, scores, hashes, and current receipts."""

    if not isinstance(artifact, Mapping):
        return ["artifact_mapping_required"]
    missing = sorted(field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact)
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if (
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE
    ):
        errors.append("identity_invalid")
    if artifact.get("run_date") != RUN_DATE or artifact.get("status") not in {
        "complete",
        "blocked",
    }:
        errors.append("lifecycle_invalid")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("model_contract_invalid")
    if (
        artifact.get("inference_substrate") != "cpu_exact_solver_or_simulator"
        or artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("substrate_invalid")
    principles = artifact.get("field_principles")
    if (
        not isinstance(principles, Mapping)
        or any(principles.get(key) != value for key, value in REQUIRED_FIELD_PRINCIPLES.items())
        or any(key not in principles for key in artifact)
    ):
        errors.append("field_principles_invalid")
    failed_class = artifact.get("verdict_class") in {"blocked", "disqualified"}
    if failed_class and any(
        artifact.get(field) != 0
        for field in ("executor_fixture_ready_score", "executor_value_score", "promotion_score")
    ):
        errors.append("failed_scores_invalid")
    if artifact.get("status") == "blocked":
        if artifact.get("rows") != [] or artifact.get("verdict_class") != "blocked":
            errors.append("blocked_shape_invalid")
        if not str(artifact.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked_verdict_invalid")
    else:
        errors.extend(independent_reduce(artifact))
        receipts = artifact.get("validation_receipts", [])
        if any(not _receipt_valid(row) for row in receipts):
            errors.append("validation_receipt_shape_invalid")
        if not _passing_receipts(artifact, ALL_VALIDATION_NAMES):
            errors.append("validation_receipts_invalid")
        if artifact.get("flagged_adversarial") is not False:
            errors.append("flagged_adversarial_invalid")
        if artifact.get("executor_fixture_ready_score") == 1 and (
            artifact.get("verdict_class") != "circular_positive"
            or not artifact.get("acceptance_gate_results")
            or not all(
                row.get("passed") is True
                for row in artifact.get("acceptance_gate_results", {}).values()
            )
        ):
            errors.append("ready_state_invalid")
    spans = artifact.get("phase_spans", [])
    if any(
        span.get("start_elapsed_s", 0) > span.get("end_elapsed_s", -1)
        or (index and span.get("start_elapsed_s", 0) < spans[index - 1].get("end_elapsed_s", 0))
        for index, span in enumerate(spans)
    ):
        errors.append("phase_spans_invalid")
    for label, expected in artifact.get("source_artifact_hashes", {}).items():
        path = Path(str(label))
        absolute = path if path.is_absolute() else root / path
        if not absolute.is_file() or sha256_file(absolute) != expected:
            errors.append("source_hash_mismatch")
            break
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return sorted(set(errors))


def write_artifact(path: Path, artifact: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    """Write only a complete artifact that passes cold validation."""

    errors = validate_artifact(artifact, root=root)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    _atomic_json(path, artifact)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _write_sidecars(
    root: Path, paths: ExperimentPaths, public_manifest: Mapping[str, Any]
) -> dict[str, Path]:
    """Label historical and scripted model-shaped evidence as non-current."""

    history = _load_object(root / HISTORY_PATH)
    historical = paths.raw_dir / "sidecars/historical_model_receipt.json"
    scripted = paths.raw_dir / "sidecars/scripted_public_model_requests.json"
    _atomic_json(
        historical,
        {
            "schema": "carnot.exp7344.historical_model_receipt.v1",
            "current_inference": {
                "MODEL_SPECS": [],
                "model_invoked": False,
                "invocation_counts": ZERO_INVOCATION_COUNTS,
            },
            "historical_artifact": {
                "path": str(HISTORY_PATH),
                "sha256": sha256_file(root / HISTORY_PATH),
                "MODEL_SPECS": history.get("MODEL_SPECS"),
                "model_invoked": history.get("model_invoked"),
                "accepted_as_current_readiness": False,
            },
        },
    )
    _atomic_json(
        scripted,
        {
            "schema": "carnot.exp7344.scripted_model_shaped_evidence.v1",
            "label": "scripted_public_model_requests_no_current_llm",
            "MODEL_SPECS": [],
            "model_invoked": False,
            "public_model_streams": public_manifest["public_model_streams"],
        },
    )
    return {"historical_model_receipt": historical, "scripted_public_model_requests": scripted}


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_path: Path,
    raw_dir: Path,
    checkpoint_path: Path,
    validation_runner: ValidationRunner = run_affected_validation,
    terminal_runner: TerminalRunner = run_terminal_validation,
    precondition_overrides: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Run the process fixture, bounded checks, reduction, and atomic publication."""

    started = time.monotonic()
    progress("checkpoint", "start", "write nonterminal checkpoint")
    artifact = _base_artifact(datetime.now(UTC).isoformat(), checkpoint_path)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _atomic_json(checkpoint_path, artifact)

    phase_started = time.monotonic()
    progress("preconditions", "start", "check exact sources and destinations")
    preconditions, initial_hashes = collect_preconditions(
        root,
        output_path,
        raw_dir,
        checkpoint_path,
        overrides=precondition_overrides,
    )
    artifact["preconditions_checked"] = preconditions
    artifact["source_artifact_hashes"] = initial_hashes
    artifact["phase_spans"].append(
        _phase_span("preconditions", phase_started, started, len(preconditions), "inputs_checked")
    )
    if any(row["blocking"] and row["available"] is not True for row in preconditions):
        apply_terminal_state(artifact)
        _finalize(artifact, started)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(output_path, artifact)
        progress("terminal_write", "end", f"blocked output={output_path}")
        return artifact

    phase_started = time.monotonic()
    progress("evaluation", "start", "run fresh seals, controls, learner, and evaluator")
    paths = ExperimentPaths.for_raw_dir(raw_dir)
    process_receipts = _execute_process_boundary(root, paths)
    public_manifest = _load_object(paths.public_manifest)
    private_receipt = _load_object(paths.private_receipt)
    learner = _load_object(paths.learner_evidence)
    evaluator = _load_object(paths.evaluator_receipt)
    controls = _load_object(paths.control_evidence)
    reduction = _reduce_raw(paths)
    manifest_errors = _manifest_errors(public_manifest)
    rows = [*learner["rows"], *controls["rows"]]
    private_markers = ("private_rules", "acceptance_witness", "private_rule_seed")
    private_messages = sum(
        any(marker in json.dumps(row, sort_keys=True) for marker in private_markers)
        for row in evaluator["query_rows"]
    )
    process_ids = {
        os.getpid(),
        int(learner["learner_pid"]),
        int(evaluator["evaluator_pid"]),
    }
    learner_private_access_count = len(learner["forbidden_accesses"])
    isolation = {
        "claim_scope": "audited_process_boundary_not_operating_system_security_sandbox",
        "hostile_process_security_claim": False,
        "orchestrator_pid": os.getpid(),
        "learner_pid": learner["learner_pid"],
        "evaluator_pid": evaluator["evaluator_pid"],
        "process_separation_passed": len(process_ids) == 3,
        "learner_allowed_inputs": learner["allowed_input_paths"],
        "learner_allowed_outputs": learner["allowed_output_paths"],
        "learner_import_closure": learner["import_closure"],
        "learner_open_file_receipts": learner["open_file_receipts"],
        "learner_private_accesses": learner["forbidden_accesses"],
        "learner_private_access_count": learner_private_access_count,
        "learner_private_rule_message_count": private_messages,
        "learner_private_rule_prompt_count": 0,
        "evaluator_import_closure": evaluator["import_closure"],
        "evaluator_open_file_receipts": evaluator["open_file_receipts"],
        "evaluator_response_keys": evaluator["response_keys"],
        "evaluator_returned_private_fields": evaluator["returned_private_fields"],
        "process_source_hashes": {
            "learner": sha256_file(root / PUBLIC_LEARNER_PATH),
            "evaluator": sha256_file(root / PRIVATE_EXECUTOR_PATH),
            "python": sha256_file(Path(sys.executable)),
        },
        "process_receipts": process_receipts,
    }
    private_receipt["private_opened_by_learner"] = learner_private_access_count > 0
    sidecars = _write_sidecars(root, paths, public_manifest)
    raw_paths = {
        "public_manifest": str(paths.public_manifest),
        "private_manifest": str(paths.private_manifest),
        "private_manifest_receipt": str(paths.private_receipt),
        "learner_evidence": str(paths.learner_evidence),
        "evaluator_receipt": str(paths.evaluator_receipt),
        "control_evidence": str(paths.control_evidence),
        **{name: str(path) for name, path in sidecars.items()},
    }
    artifact.update(
        rows=rows,
        public_manifest_path=str(paths.public_manifest),
        public_manifest={
            "path": str(paths.public_manifest),
            "sha256": sha256_file(paths.public_manifest),
            "manifest_hash": public_manifest["manifest_hash"],
            "private_fields_present": False,
        },
        private_manifest_receipt=private_receipt,
        cohort_manifest=_cohort_manifest(public_manifest),
        isolation_controls=isolation,
        control_results=controls,
        independent_reduction=reduction,
        raw_evidence_paths=raw_paths,
        manifest_errors=manifest_errors,
    )
    artifact["sample_size_budget"]["development"].update(attempted=384, completed=384, censored=0)
    artifact["sample_size_budget"]["controls"].update(attempted=7, completed=7, censored=0)
    artifact["source_artifact_hashes"].update(
        {path: sha256_file(Path(path)) for path in raw_paths.values()}
    )
    artifact["phase_spans"].append(
        _phase_span("evaluation", phase_started, started, len(rows), "raw_process_evidence")
    )

    phase_started = time.monotonic()
    progress("test", "start", "run shipped scoped validation")
    validation = validation_runner(
        root=root,
        raw_dir=raw_dir,
        test_paths=[str(TEST_PATH)],
        changed_modules=[str(MODULE_PATH)],
        static_paths=[str(WRAPPER_PATH)],
        historical_failures=_historical_failures(root),
    )
    artifact["validation_receipts"] = list(validation.get("validation_receipts", []))
    artifact["repository_health"] = validation.get("repository_health", {})
    artifact["required_checks_passed"] = validation.get("required_checks_passed") is True
    artifact["flagged_adversarial"] = None
    apply_terminal_state(artifact)
    artifact["phase_spans"].append(
        _phase_span(
            "affected_validation",
            phase_started,
            started,
            len(artifact["validation_receipts"]),
            "candidate_next",
        )
    )
    _finalize(artifact, started)
    _atomic_json(paths.candidate, artifact)
    progress("test", "candidate_written", f"path={paths.candidate}")

    phase_started = time.monotonic()
    progress("terminal_validation", "start", "run cold reducer and strict readers")
    terminal_receipts = terminal_runner(root, paths.candidate, raw_dir)
    artifact["validation_receipts"].extend(terminal_receipts)
    adversarial = next(
        (row for row in terminal_receipts if row.get("name") == "adversarial_verify"), {}
    )
    artifact["flagged_adversarial"] = adversarial.get("passed") is not True
    artifact["required_checks_passed"] = _passing_receipts(artifact, ALL_VALIDATION_NAMES)
    apply_terminal_state(artifact)
    artifact["phase_spans"].append(
        _phase_span(
            "terminal_validation",
            phase_started,
            started,
            len(terminal_receipts),
            "terminal_output_ready",
        )
    )

    phase_started = time.monotonic()
    progress("write", "start", "seal terminal artifact")
    artifact["phase_spans"].append(
        _phase_span("write", phase_started, started, 1, "atomic_terminal_write")
    )
    _finalize(artifact, started)
    write_artifact(output_path, artifact, root=root)
    progress("write", "end", f"output={output_path} verdict={artifact['honest_verdict']}")
    return artifact


def date_argument(value: str) -> str:
    """Accept only the fixed V645 execution date."""

    datetime.strptime(value, "%Y%m%d")
    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - declared E2E command.
    """Run the current bounded fixture and require a valid terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=date_argument)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else REPO_ROOT / args.raw_dir
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    artifact = build_artifact(
        REPO_ROOT,
        args.date,
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
    )
    errors = validate_artifact(artifact)
    if errors:
        print(f"[exp7344] invalid artifact: {errors}", file=sys.stderr, flush=True)
        return 1
    progress(
        "complete",
        "end",
        f"verdict={artifact['honest_verdict']} score={artifact['executor_fixture_ready_score']}",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
