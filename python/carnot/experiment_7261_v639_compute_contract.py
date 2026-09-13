"""Measure the V639 compute-floor and hermetic-precondition repair.

This task invokes no model. It exercises CPU verification code, stores injected
model claims only as raw sidecars, and publishes a terminal null receipt only
after the boundary, contradiction, fixture, and command-line checks agree.

Spec refs: REQ-SUBSTRATE-CLASS-1 and SCENARIO-SUBSTRATE-CLASS-9 through
SCENARIO-SUBSTRATE-CLASS-11.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import sys
import time
from typing import Any

from carnot import experiment_7240_v637_recurrence_fixture as exp7240
from carnot import experiment_7246_v638_source_map as shipped


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover - normal package setup.
    sys.path.insert(0, str(SCRIPTS_ROOT))

import adversarial_verify as av  # noqa: E402


JsonDict = dict[str, Any]
EXPERIMENT_ID = "exp7261-compute-contract"
MILESTONE = "2026.09.639"
RUN_DATE = "20260913"
RANDOM_SEED = 7_261_202_609_13
SCHEMA = "carnot.exp7261.v639_compute_contract.v1"
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "usable_answers": 0,
}
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"

MODULE_PATH = Path("python/carnot/experiment_7261_v639_compute_contract.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7261_v639_compute_contract.py")
TEST_PATH = Path("tests/python/test_experiment_7261_v639_compute_contract.py")
EXP7240_ACTIVE_TEST_PATH = Path("tests/python/test_experiment_7240_v637_recurrence_fixture.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7261_v639_compute_contract.json")
DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/experiment_7261_v639_compute_contract.json")
DEFAULT_RAW_DIR = Path("results/raw/experiment_7261")
RAW_CANDIDATE_NAME = "measured-terminal-candidate.json"

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/summarize_artifact.py"),
    Path("python/carnot/experiment_7240_v637_recurrence_fixture.py"),
    EXP7240_ACTIVE_TEST_PATH,
    Path("tests/python/test_adversarial_verify_substrate_class_20260905.py"),
    Path("tests/python/test_adversarial_verify_claim_provenance_6974.py"),
    Path("tests/python/test_adversarial_verify_local_sota_gguf_small_n_substrate.py"),
    Path("tests/python/test_substrate_class_cutover_20260907.py"),
    Path("results/experiment_7237_v637_mention_canary.json"),
    Path("openspec/capabilities/research-harnesses/spec.md"),
)

FIELD_PRINCIPLES = {
    "schema": "Version the result; retain ordinary top-level experiment_id and milestone.",
    "status": "Use complete or blocked only for terminal work; unfinished work stays in a separate checkpoint.",
    "run_date": "Use 20260913, with actual UTC start/end timestamps, so dated evidence is auditable.",
    "field_principles": "Store explanations here; consumers read ordinary top-level values, not nested wrappers.",
    "preconditions_checked": "Retain observed input hashes, resource ownership and failures before expensive work.",
    "MODEL_SPECS": "Declare models executable in this invocation; keep historical model metadata in hashed sidecars.",
    "model_invoked": "Derive from actual calls; a parse failure does not erase a model invocation.",
    "invocation_counts": "Separate attempted/completed loads and generation calls from usable answers.",
    "inference_substrate": "Use an existing recognized literal that describes actual computation.",
    "inference_substrate_class": "Declare actual compute: full generation 60s, bounded generation 10s, load-only 2s; never pad time.",
    "execution_venue": "Use host for host orchestration; identify real boards separately in board rows.",
    "duration_s": "Measure monotonic invocation time and disjoint phase spans; do not invent elapsed time.",
    "random_seed": "Freeze independent-unit seeds before inspecting outcomes.",
    "reproducibility_checksum": "Bind code, input manifests, configuration and raw evidence to the result.",
    "source_artifact_hashes": "Authenticate exact inputs and preserve quarantine and retirement state.",
    "rows": "Retain each independent unit, arm, seed, metric, error, abstention and censoring state for recomputation.",
    "sample_size_budget": "Record planned, attempted, completed and censored units and the fixed stopping rule.",
    "acceptance_gate_results": "Each criterion retains expected, observed, passed and principle; completion is separate from value.",
    "gate_check_summary": "For blocked_* name the upstream, exact field/check, observed value and expected value.",
    "verifier_is_oracle": "Expose shared evaluator/verifier authority; exact conformance is not learned correctness.",
    "honest_verdict": "Use complete_* for terminal measurements, blocked_* for external absence, and explain the finding.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Oracle=true forbids positive; failed scientific gates forbid positive. Only incomplete own work is partial; unchanged external blocks are blocked.",
    "validation_receipts": "Record actual command, exit code and log hash; preserve failures and never suppress checks.",
    "compute_contract_ready_score": "One requires the corrected shared floor and the original pre-test behavior to be exercised cleanly.",
    "boundary_rows": "Below/at-floor pairs and contradiction controls detect an overbroad relaxation.",
    "fixture_repair_receipt": "Record failing old dependency, replacement fixture hashes and preserved assertions.",
}
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)


def _fixture(
    fixture_id: str,
    control_kind: str,
    substrate_class: object,
    duration_s: float,
    expected_floor_s: float | None,
    expected_duration_flag: bool,
    expected_class_flag: bool,
    *,
    substrate: str = "live_llm_inference",
    typed: Mapping[str, object] | None = None,
    verdict: str = "complete_null_compute_floor_fixture",
) -> JsonDict:
    """Create one frozen independent input without executing it."""

    payload: JsonDict = {
        "run_date": RUN_DATE,
        "honest_verdict": verdict,
        "inference_substrate": substrate,
        "duration_s": duration_s,
        **dict(typed or {}),
    }
    if substrate_class is not None:
        payload["inference_substrate_class"] = substrate_class
    return {
        "fixture_id": fixture_id,
        "control_kind": control_kind,
        "payload": payload,
        "expected_floor_s": expected_floor_s,
        "expected_duration_flag": expected_duration_flag,
        "expected_class_flag": expected_class_flag,
    }


BOUNDARY_FIXTURES = (
    _fixture(
        "full_below",
        "duration_boundary",
        "model_full_generation",
        59.999,
        60.0,
        True,
        True,
        typed={"model_invoked": True},
    ),
    _fixture(
        "full_at",
        "duration_boundary",
        "model_full_generation",
        60.0,
        60.0,
        False,
        False,
        typed={"model_invoked": True},
    ),
    _fixture(
        "full_above",
        "duration_boundary",
        "model_full_generation",
        60.001,
        60.0,
        False,
        False,
        typed={"model_invoked": True},
    ),
    _fixture(
        "bounded_below",
        "duration_boundary",
        "model_bounded_generation",
        9.999,
        10.0,
        True,
        True,
        typed={"model_invoked": True},
    ),
    _fixture(
        "bounded_at",
        "duration_boundary",
        "model_bounded_generation",
        10.0,
        10.0,
        False,
        False,
        typed={"model_invoked": True},
    ),
    _fixture(
        "bounded_exp7237_59_466359",
        "duration_boundary",
        "model_bounded_generation",
        59.46635937620886,
        10.0,
        False,
        False,
        typed={"model_invoked": True, "generation_call_count": 48},
    ),
    _fixture(
        "bounded_above",
        "duration_boundary",
        "model_bounded_generation",
        10.001,
        10.0,
        False,
        False,
        typed={"model_invoked": True},
    ),
    _fixture(
        "load_only_below",
        "duration_boundary",
        "model_load_no_generation",
        1.999,
        2.0,
        True,
        True,
        typed={"model_invoked": True},
    ),
    _fixture(
        "load_only_at",
        "duration_boundary",
        "model_load_no_generation",
        2.0,
        2.0,
        False,
        False,
        typed={"model_invoked": True},
    ),
    _fixture(
        "load_only_above",
        "duration_boundary",
        "model_load_no_generation",
        2.001,
        2.0,
        False,
        False,
        typed={"model_invoked": True},
    ),
    _fixture(
        "aggregation_below",
        "duration_boundary",
        "aggregation",
        0.00009,
        0.0001,
        True,
        True,
        substrate="aggregation_from_upstream_artifacts",
        typed={"model_invoked": False},
    ),
    _fixture(
        "aggregation_at",
        "duration_boundary",
        "aggregation",
        0.0001,
        0.0001,
        False,
        False,
        substrate="aggregation_from_upstream_artifacts",
        typed={"model_invoked": False},
    ),
    _fixture(
        "no_model_below",
        "duration_boundary",
        "no_model_load",
        0.00009,
        0.0001,
        False,
        True,
        substrate="artifact_qa_lint_tests",
        typed={"model_invoked": False},
    ),
    _fixture(
        "no_model_at",
        "duration_boundary",
        "no_model_load",
        0.0001,
        0.0001,
        False,
        False,
        substrate="artifact_qa_lint_tests",
        typed={"model_invoked": False},
    ),
    _fixture(
        "cpu_exact_below",
        "duration_boundary",
        "cpu_exact_solver_or_simulator",
        0.00009,
        0.0001,
        False,
        True,
        substrate="cpu_exact_solver_or_simulator",
        typed={"model_invoked": False},
    ),
    _fixture(
        "cpu_exact_at",
        "duration_boundary",
        "cpu_exact_solver_or_simulator",
        0.0001,
        0.0001,
        False,
        False,
        substrate="cpu_exact_solver_or_simulator",
        typed={"model_invoked": False},
    ),
    _fixture(
        "bounded_negative_boolean",
        "contradiction",
        "model_bounded_generation",
        30.0,
        60.0,
        True,
        True,
        typed={"model_invoked": False},
    ),
    _fixture(
        "bounded_mixed_booleans",
        "contradiction",
        "model_bounded_generation",
        30.0,
        60.0,
        True,
        True,
        typed={"model_invoked": True, "llm_invoked": False},
    ),
    _fixture(
        "bounded_zero_count",
        "contradiction",
        "model_bounded_generation",
        30.0,
        60.0,
        True,
        True,
        typed={"model_invoked": True, "generation_calls_completed": 0},
    ),
    _fixture(
        "no_model_positive_count",
        "contradiction",
        "no_model_load",
        30.0,
        60.0,
        True,
        True,
        typed={"generation_calls_attempted": 1},
    ),
    _fixture(
        "aggregation_positive_count",
        "contradiction",
        "aggregation",
        30.0,
        60.0,
        True,
        True,
        typed={"model_loads_completed": 1},
    ),
    _fixture(
        "cpu_positive_count",
        "contradiction",
        "cpu_exact_solver_or_simulator",
        30.0,
        60.0,
        True,
        True,
        typed={"generation_calls_completed": 1},
    ),
    _fixture(
        "malformed_class",
        "contradiction",
        {"kind": "model_bounded_generation"},
        30.0,
        60.0,
        True,
        True,
        typed={"model_invoked": True},
    ),
    _fixture(
        "unknown_class",
        "contradiction",
        "new_model_class",
        30.0,
        60.0,
        True,
        True,
        typed={"model_invoked": True},
    ),
    _fixture(
        "retired_board_class",
        "contradiction",
        "hardware_board",
        30.0,
        60.0,
        True,
        True,
        typed={"model_invoked": True},
    ),
    _fixture(
        "absent_class", "contradiction", None, 30.0, 60.0, True, True, typed={"model_invoked": True}
    ),
    _fixture(
        "blocked_class_nonblocked_verdict",
        "contradiction",
        "blocked_no_run",
        30.0,
        60.0,
        True,
        True,
        typed={"model_invoked": True},
    ),
    _fixture(
        "model_class_blocked_verdict",
        "contradiction",
        "model_bounded_generation",
        0.1,
        None,
        False,
        True,
        typed={"model_invoked": False},
        verdict="blocked_external_model_absent",
    ),
    _fixture(
        "full_class_over_bounded_prose",
        "duration_boundary",
        "model_full_generation",
        30.0,
        60.0,
        True,
        True,
        substrate="live_llm_inference_local_gguf_sota",
        typed={"model_invoked": True},
    ),
    _fixture(
        "bounded_class_over_embedding_prose",
        "duration_boundary",
        "model_bounded_generation",
        5.0,
        10.0,
        True,
        True,
        substrate="live_llm_embedding_extraction",
        typed={"model_invoked": True},
    ),
)


def progress(started: float, phase: int, boundary: str, detail: str) -> None:
    """Emit one flushed phase line with truthful monotonic elapsed time."""

    print(
        f"phase={phase} boundary={boundary} elapsed_s={time.monotonic() - started:.3f} {detail}",
        flush=True,
    )


def sha256_bytes(content: bytes) -> str:
    """Return the repository's prefixed SHA-256 receipt form."""

    return "sha256:" + hashlib.sha256(content).hexdigest()


def sha256_path(path: Path) -> str | None:
    """Hash exact file bytes, retaining absence as an observation."""

    try:
        return sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _canonical_bytes(value: object) -> bytes:
    """Encode stable JSON bytes for sidecars and result publication."""

    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _atomic_write(path: Path, value: object) -> None:
    """Publish complete JSON bytes with flush, fsync, and one rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(_canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _copy(source: Path, target: Path) -> None:
    """Copy one authenticated fixture input and create only its parents."""

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _receipt_path(root: Path, path: Path) -> str:
    """Prefer repository-relative receipts while retaining private absolute paths."""

    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def build_hermetic_exp7240_fixture(root: Path) -> JsonDict:
    """Build a complete private repository view for the unchanged Exp7240 reader."""

    for relative in exp7240.SOURCE_PATHS:
        target = root / relative
        if relative == Path("research-roadmap.yaml"):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "- id: exp7240-recurrence-fixture\n"
                f"  milestone: {exp7240.MILESTONE}\n"
                f"  deliverable: {exp7240.DEFAULT_ARTIFACT}\n",
                encoding="utf-8",
            )
        else:
            source = REPO_ROOT / relative
            _copy(source, target)
    upstream_source = REPO_ROOT / exp7240.DEFAULT_UPSTREAM_ARTIFACT
    _copy(upstream_source, root / exp7240.DEFAULT_UPSTREAM_ARTIFACT)
    upstream = json.loads(upstream_source.read_text(encoding="utf-8"))
    receipt_paths = [exp7240.DEFAULT_UPSTREAM_ARTIFACT]
    decision = exp7240.unwrap_principled(upstream["decision_rows_path"])
    receipt_paths.append(Path(str(decision["path"])))
    receipt_paths.append(
        Path("results/checkpoints/experiment_7227_v636_belief_learning_state.json")
    )
    for relative in receipt_paths[1:]:
        _copy(REPO_ROOT / relative, root / relative)
    output_root = root / "results/hermetic_exp7240_outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    fixture_hashes = {
        str(relative): sha256_path(root / relative)
        for relative in (*exp7240.SOURCE_PATHS, *receipt_paths)
    }
    return {
        "root": str(root),
        "output_root": str(output_root),
        "fixture_hashes": fixture_hashes,
        "failing_observed_input": {
            "check": "required_source_bytes|v637_task_identity",
            "field": "tests/python/test_experiment_7240_v637_recurrence_fixture.py|id,milestone,deliverable",
            "observed_value": "missing|{id: null, milestone: null, deliverable: null}",
            "expected_value": "nonempty|exp7240-recurrence-fixture,2026.09.637,results/experiment_7240_v637_recurrence_fixture.json",
        },
        "preserved_assertions": [
            "positive_gate_summary_passes",
            "mutated_receipt_gate_summary_fails",
        ],
    }


def read_hermetic_exp7240_fixture(
    receipt: Mapping[str, Any],
) -> tuple[list[JsonDict], dict[str, str | None], JsonDict]:
    """Send the private fixture through the shipped Exp7240 precondition reader."""

    root = Path(str(receipt["root"]))
    paths = exp7240.ExperimentPaths.from_results_root(Path(str(receipt["output_root"])))
    return exp7240.collect_preconditions(root, paths)


def evaluate_boundary_fixtures() -> list[JsonDict]:
    """Run every frozen boundary through both shared checker paths."""

    rows: list[JsonDict] = []
    for index, fixture in enumerate(BOUNDARY_FIXTURES):
        payload = deepcopy(fixture["payload"])
        floor = av.duration_floor_for_artifact(payload)
        duration_flags: list[av.Flag] = []
        class_flags: list[av.Flag] = []
        av.check_duration_vs_claim(payload, duration_flags)
        av.check_substrate_class(payload, class_flags)
        observed_floor = None if floor is None else float(floor["min_duration_s"])
        duration_flag = any(flag.kind == "DURATION_TOO_SHORT" for flag in duration_flags)
        class_flag = any(
            flag.kind in {av.SUBSTRATE_CLASS_MISMATCH_KIND, av.SUBSTRATE_CLASS_MISSING_KIND}
            for flag in class_flags
        )
        passed = (
            observed_floor == fixture["expected_floor_s"]
            and duration_flag is fixture["expected_duration_flag"]
            and class_flag is fixture["expected_class_flag"]
        )
        rows.append(
            {
                "fixture_id": fixture["fixture_id"],
                "control_kind": fixture["control_kind"],
                "seed": RANDOM_SEED + index,
                "arm": payload.get("inference_substrate_class", "absent"),
                "metric": "duration_floor_s",
                "expected_floor_s": fixture["expected_floor_s"],
                "observed_floor_s": observed_floor,
                "expected_duration_flag": fixture["expected_duration_flag"],
                "observed_duration_flag": duration_flag,
                "expected_class_flag": fixture["expected_class_flag"],
                "observed_class_flag": class_flag,
                "error": None if passed else "boundary_or_contradiction_mismatch",
                "abstention": False,
                "censored": False,
                "passed": passed,
            }
        )
    return rows


def _complete_model_fixture(substrate_class: str, duration_s: float) -> JsonDict:
    """Build a complete injected artifact used only by the two public readers."""

    return {
        "schema": "carnot.exp7261.injected_floor_fixture.v1",
        "experiment_id": f"fixture-{substrate_class}-{duration_s}",
        "status": "complete",
        "run_date": RUN_DATE,
        "honest_verdict": "complete_null_injected_compute_floor_fixture",
        "verdict_class": "null",
        "inference_substrate": "live_llm_inference",
        "inference_substrate_class": substrate_class,
        "execution_venue": "host",
        "duration_s": duration_s,
        "model_invoked": True,
        "MODEL_SPECS": [{"fixture_model": "historical_sidecar_only.gguf"}],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "sha256:" + "1" * 64,
        "preconditions_checked": [{"check": "injected_fixture", "passed": True}],
    }


def _run_capture(command: Sequence[str], cwd: Path) -> JsonDict:
    """Run one bounded public CLI and retain its exact combined output."""

    started = time.monotonic()
    print(f"before subprocess: {shlex.join(command)}", flush=True)
    completed = shipped._run_streaming_command(
        list(command),
        cwd=cwd,
        timeout_s=1200,
        heartbeat_s=60,
        operation="exp7261_validation",
    )
    duration = time.monotonic() - started
    exit_code = int(completed["exit_code"])
    output = str(completed.get("stdout", ""))
    print(f"after subprocess: exit={exit_code} elapsed_s={duration:.3f}", flush=True)
    return {
        "command": shlex.join(command),
        "exit_code": exit_code,
        "duration_s": duration,
        "timed_out": bool(completed.get("timed_out", False)),
        "stdout": output,
        "stdout_sha256": sha256_bytes(output.encode()),
    }


def run_floor_e2e(root: Path, directory: Path) -> JsonDict:
    """Use the real checker and summarizer CLIs on complete boundary pairs."""

    directory.mkdir(parents=True, exist_ok=True)
    python = str(root / ".venv/bin/python")
    rows: list[JsonDict] = []
    for substrate_class, floor in (
        ("model_full_generation", 60.0),
        ("model_bounded_generation", 10.0),
        ("model_load_no_generation", 2.0),
    ):
        pair: dict[str, Path] = {}
        for label, duration in (("below", floor - 0.001), ("at", floor)):
            path = directory / f"{substrate_class}-{label}.json"
            _atomic_write(path, _complete_model_fixture(substrate_class, duration))
            pair[label] = path
        checker = _run_capture(
            [
                python,
                "-u",
                "scripts/adversarial_verify.py",
                "--json",
                str(pair["below"]),
                str(pair["at"]),
            ],
            root,
        )
        checker_payload = json.loads(checker["stdout"])
        reports = checker_payload["reports"]
        below_flags = {flag["kind"] for flag in reports[0]["flags"]}
        at_flags = {flag["kind"] for flag in reports[1]["flags"]}
        summary = _run_capture(
            [python, "-u", "scripts/summarize_artifact.py", str(pair["at"])], root
        )
        match = re.search(r"duration floor\s*:\s*\S+\s+>=(\d+(?:\.\d+)?)s", summary["stdout"])
        summary_floor = None if match is None else float(match.group(1))
        passed = (
            av.SUBSTRATE_CLASS_MISMATCH_KIND in below_flags
            and "DURATION_TOO_SHORT" in below_flags
            and av.SUBSTRATE_CLASS_MISMATCH_KIND not in at_flags
            and "DURATION_TOO_SHORT" not in at_flags
            and summary_floor == floor
        )
        rows.append(
            {
                "substrate_class": substrate_class,
                "checker_floor_s": floor,
                "summary_floor_s": summary_floor,
                "below_floor_flags": sorted(below_flags),
                "at_floor_flags": sorted(at_flags),
                "checker_receipt": {
                    key: value for key, value in checker.items() if key != "stdout"
                },
                "summary_receipt": {
                    key: value for key, value in summary.items() if key != "stdout"
                },
                "passed": passed,
            }
        )
    mismatches = sum(row["passed"] is not True for row in rows)
    return {"rows": rows, "mismatch_count": mismatches, "passed": mismatches == 0}


def collect_preconditions(
    root: Path, output_path: Path, raw_dir: Path, checkpoint: Path
) -> list[JsonDict]:
    """Authenticate exact listed inputs, the driving REQ, and owned output paths."""

    rows = []
    for relative in INPUT_PATHS:
        observed = sha256_path(root / relative)
        rows.append(
            {
                "check": "required_input",
                "upstream": str(relative),
                "field": "sha256",
                "expected_value": "sha256:<64 hex>",
                "observed_value": observed or "missing",
                "passed": observed is not None,
                "resource_owner": "repository",
            }
        )
    spec_path = root / "openspec/capabilities/research-harnesses/spec.md"
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.is_file() else ""
    rows.append(
        {
            "check": "driving_capability",
            "upstream": str(spec_path.relative_to(root)),
            "field": "REQ-SUBSTRATE-CLASS-1|SCENARIO-SUBSTRATE-CLASS-9",
            "expected_value": True,
            "observed_value": all(
                marker in spec_text
                for marker in ("REQ-SUBSTRATE-CLASS-1", "SCENARIO-SUBSTRATE-CLASS-9")
            ),
            "passed": all(
                marker in spec_text
                for marker in ("REQ-SUBSTRATE-CLASS-1", "SCENARIO-SUBSTRATE-CLASS-9")
            ),
            "resource_owner": "repository",
        }
    )
    for name, path in (("output", output_path), ("raw", raw_dir), ("checkpoint", checkpoint)):
        parent = path if path.suffix == "" else path.parent
        ancestor = parent
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        writable = ancestor.is_dir() and os.access(ancestor, os.W_OK)
        rows.append(
            {
                "check": "writable_output",
                "upstream": "host_filesystem",
                "field": name,
                "expected_value": True,
                "observed_value": writable,
                "passed": writable,
                "resource_owner": "current_process",
            }
        )
    return rows


def _gate_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first exact failed check or report complete preconditions."""

    failed = next((row for row in rows if row.get("passed") is not True), None)
    if failed is None:
        return {
            "failed_check": None,
            "upstream": None,
            "field": None,
            "expected_value": "all_required_checks_pass",
            "observed_value": "all_required_checks_pass",
            "passed": True,
        }
    return {
        "failed_check": failed.get("check"),
        "upstream": failed.get("upstream"),
        "field": failed.get("field"),
        "expected_value": failed.get("expected_value"),
        "observed_value": failed.get("observed_value"),
        "passed": False,
    }


VALIDATION_NAMES = (
    "focused_pytest",
    "named_substrate_suites",
    "scoped_coverage",
    "scoped_coverage_report",
    "ruff_check",
    "ruff_format",
    "mypy",
    "scoped_spec_coverage",
    "independent_raw_reducer",
    "adversarial_verify",
    "verdict_row_consistency",
)
INITIAL_VALIDATION_FAILURES = [
    {
        "name": "named_substrate_suites_initial_timeout",
        "command": "pytest -n 0 Exp7240 and four named substrate/provenance suites",
        "exit_code": 124,
        "timed_out": True,
        "timeout_s": 120,
        "observed": "the bounded Exp7240 CPU fixture had not completed",
        "classification": "task_own_orchestration_timeout_repaired_to_1200s_streaming",
    }
]


def fixture_validation_receipts() -> list[JsonDict]:
    """Return explicit passing receipts for artifact-focused unit tests only."""

    return [
        {
            "name": name,
            "command": f"unit-fixture:{name}",
            "exit_code": 0,
            "duration_s": 0.001,
            "log_sha256": sha256_bytes(name.encode()),
            "passed": True,
        }
        for name in VALIDATION_NAMES
    ]


def _source_hashes(root: Path) -> dict[str, str]:
    """Hash every present source without pretending absent bytes exist."""

    paths = (*INPUT_PATHS, MODULE_PATH, WRAPPER_PATH, TEST_PATH)
    return {
        str(relative): digest
        for relative in paths
        if (digest := sha256_path(root / relative)) is not None
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind configuration, sources, raw receipts, rows, and terminal values."""

    stable = deepcopy(dict(artifact))
    stable.pop("reproducibility_checksum", None)
    return sha256_bytes(_canonical_bytes(stable))


def independent_reduce(rows_path: Path) -> list[str]:
    """Re-read raw rows and recompute the boundary acceptance independently."""

    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    errors = []
    if not isinstance(rows, list) or len(rows) < 24:
        return ["boundary_rows_missing"]
    if len({row.get("fixture_id") for row in rows if isinstance(row, Mapping)}) != len(rows):
        errors.append("fixture_ids_not_independent")
    if any(not isinstance(row, Mapping) or row.get("passed") is not True for row in rows):
        errors.append("boundary_mismatch")
    return errors


def _acceptance_results(
    rows: Sequence[Mapping[str, Any]],
    fixture_receipt: Mapping[str, Any],
    e2e: Mapping[str, Any],
    validations: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Retain expected, observed, pass state, and principle for every gate."""

    contradiction_rows = [row for row in rows if row.get("control_kind") == "contradiction"]
    criteria = (
        (
            "boundary_mismatches",
            0,
            sum(row.get("passed") is not True for row in rows),
            all(row.get("passed") is True for row in rows),
            "Every frozen floor boundary must match.",
        ),
        (
            "contradiction_controls",
            "all_fire",
            sum(row.get("passed") is True for row in contradiction_rows),
            bool(contradiction_rows)
            and all(row.get("passed") is True for row in contradiction_rows),
            "Malformed and contradictory declarations must remain strict.",
        ),
        (
            "hermetic_exp7240_precondition",
            True,
            fixture_receipt.get("passed"),
            fixture_receipt.get("passed") is True,
            "The unchanged positive and negative assertions must pass on authenticated private bytes.",
        ),
        (
            "checker_summarizer_e2e",
            0,
            e2e.get("mismatch_count"),
            e2e.get("passed") is True,
            "Both public readers must expose the same class floor.",
        ),
        (
            "scoped_validation",
            list(VALIDATION_NAMES),
            [row.get("name") for row in validations],
            len(validations) == len(VALIDATION_NAMES)
            and all(row.get("passed") is True for row in validations),
            "Only the requested focused validation may determine readiness.",
        ),
    )
    return [
        {
            "criterion": name,
            "expected": expected,
            "observed": observed,
            "passed": passed,
            "principle": principle,
        }
        for name, expected, observed, passed, principle in criteria
    ]


def _base_artifact(started_at: str, run_date: str) -> JsonDict:
    """Create a non-success checkpoint shape before measurement."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "running",
        "run_date": run_date,
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": dict(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned": len(BOUNDARY_FIXTURES),
            "attempted": 0,
            "completed": 0,
            "censored": 0,
            "independent_units": len(BOUNDARY_FIXTURES),
            "stopping_rule": "Evaluate every frozen fixture exactly once; do not stop on outcome.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "honest_verdict": "partial_compute_contract_measurement",
        "verdict_class": "partial",
        "validation_receipts": [],
        "validation_attempt_failures": deepcopy(INITIAL_VALIDATION_FAILURES),
        "compute_contract_ready_score": 0,
        "boundary_rows": [],
        "fixture_repair_receipt": {},
        "e2e_floor_receipt": {},
        "raw_rows_receipt": {},
        "historical_model_sidecar_receipt": {},
        "publication_performed": False,
        "submission_performed": False,
        "upload_performed": False,
        "production_default_changed": False,
        "external_message_performed": False,
        "research_conductor_modified": False,
        "research_roadmap_modified": False,
    }


def _apply_terminal_state(artifact: JsonDict) -> None:
    """Derive blocked or complete-null state from stored measurements."""

    gate = artifact["gate_check_summary"]
    if gate.get("passed") is not True:
        artifact.update(
            status="blocked",
            inference_substrate="blocked_no_run",
            inference_substrate_class="blocked_no_run",
            verdict_class="blocked",
            honest_verdict="blocked_required_compute_contract_input_absent",
            compute_contract_ready_score=0,
            rows=[],
            boundary_rows=[],
        )
        artifact["acceptance_gate_results"] = []
        return
    acceptance = _acceptance_results(
        artifact["boundary_rows"],
        artifact["fixture_repair_receipt"],
        artifact["e2e_floor_receipt"],
        artifact["validation_receipts"],
    )
    ready = all(row["passed"] is True for row in acceptance)
    artifact.update(
        status="complete",
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        verdict_class="null",
        honest_verdict=(
            "complete_null_compute_contract_repaired_no_scientific_value_claim"
            if ready
            else "complete_null_compute_contract_measured_validation_not_clean"
        ),
        compute_contract_ready_score=1 if ready else 0,
        acceptance_gate_results=acceptance,
        rows=artifact["boundary_rows"],
    )


def validate_artifact(value: object, *, root: Path) -> list[str]:
    """Cold-check identity, evidence, terminal class, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping_required"]
    artifact = dict(value)
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - artifact.keys())
    if missing:
        errors.append("required_fields:" + ",".join(missing))
    if (
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE
    ):
        errors.append("identity_invalid")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_invalid")
    if artifact.get("status") not in {"complete", "blocked"}:
        errors.append("status_invalid")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("current_invocation_invalid")
    if artifact.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue_invalid")
    if (
        not isinstance(artifact.get("duration_s"), (int, float))
        or artifact.get("duration_s", 0) <= 0
    ):
        errors.append("duration_invalid")
    if not isinstance(artifact.get("source_artifact_hashes"), Mapping) or any(
        not isinstance(digest, str) or not digest.startswith("sha256:")
        for digest in artifact.get("source_artifact_hashes", {}).values()
    ):
        errors.append("source_hashes_invalid")
    if artifact.get("status") == "blocked":
        if (
            artifact.get("verdict_class") != "blocked"
            or not str(artifact.get("honest_verdict", "")).startswith("blocked_")
            or artifact.get("compute_contract_ready_score") != 0
            or artifact.get("rows") != []
            or artifact.get("boundary_rows") != []
            or artifact.get("inference_substrate_class") != "blocked_no_run"
        ):
            errors.append("blocked_contract_invalid")
    else:
        rows = artifact.get("boundary_rows")
        validations = artifact.get("validation_receipts")
        if artifact.get("verdict_class") != "null" or not str(
            artifact.get("honest_verdict", "")
        ).startswith("complete_null_"):
            errors.append("complete_verdict_invalid")
        if (
            artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
            or artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS
        ):
            errors.append("substrate_invalid")
        if (
            not isinstance(rows, list)
            or len(rows) < 24
            or artifact.get("rows") != rows
            or any(row.get("passed") is not True for row in rows if isinstance(row, Mapping))
        ):
            errors.append("boundary_rows_invalid")
        if (
            not isinstance(validations, list)
            or [row.get("name") for row in validations if isinstance(row, Mapping)]
            != list(VALIDATION_NAMES)
            or any(row.get("passed") is not True for row in validations if isinstance(row, Mapping))
        ):
            errors.append("validation_receipts_invalid")
        if artifact.get("compute_contract_ready_score") != 1:
            errors.append("readiness_invalid")
        if (
            not isinstance(artifact.get("fixture_repair_receipt"), Mapping)
            or artifact["fixture_repair_receipt"].get("passed") is not True
        ):
            errors.append("fixture_receipt_invalid")
        if (
            not isinstance(artifact.get("e2e_floor_receipt"), Mapping)
            or artifact["e2e_floor_receipt"].get("passed") is not True
        ):
            errors.append("e2e_receipt_invalid")
        raw_receipt = artifact.get("raw_rows_receipt")
        if not isinstance(raw_receipt, Mapping) or sha256_path(
            root / str(raw_receipt.get("path", ""))
        ) != raw_receipt.get("sha256"):
            errors.append("raw_rows_receipt_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return errors


def _validation_commands(
    root: Path, candidate: Path, raw_rows: Path
) -> tuple[tuple[str, list[str]], ...]:
    """Return the exact focused commands; never include the full repository suite."""

    python = str(root / ".venv/bin/python")
    coverage_file = "/tmp/.coverage-exp7261-v639"
    base = "/tmp/exp7261-v639-focused"
    named = [
        TEST_PATH,
        EXP7240_ACTIVE_TEST_PATH,
        Path("tests/python/test_adversarial_verify_substrate_class_20260905.py"),
        Path("tests/python/test_adversarial_verify_claim_provenance_6974.py"),
        Path("tests/python/test_adversarial_verify_local_sota_gguf_small_n_substrate.py"),
        Path("tests/python/test_substrate_class_cutover_20260907.py"),
    ]
    reducer = (
        "import pathlib,sys;from carnot.experiment_7261_v639_compute_contract import independent_reduce;"
        f"errors=independent_reduce(pathlib.Path({str(raw_rows)!r}));print(errors);sys.exit(bool(errors))"
    )
    return (
        (
            "focused_pytest",
            [
                python,
                "-u",
                "-m",
                "pytest",
                "-o",
                "addopts=",
                "-n",
                "0",
                f"--basetemp={base}-new",
                str(TEST_PATH),
                "-q",
            ],
        ),
        (
            "named_substrate_suites",
            [
                python,
                "-u",
                "-m",
                "pytest",
                "-o",
                "addopts=",
                "-n",
                "0",
                f"--basetemp={base}-named",
                *map(str, named[1:]),
                "-q",
            ],
        ),
        (
            "scoped_coverage",
            [
                str(root / ".venv/bin/coverage"),
                "run",
                f"--data-file={coverage_file}",
                f"--include=*/{MODULE_PATH.name}",
                "-m",
                "pytest",
                "-o",
                "addopts=",
                "-n",
                "0",
                f"--basetemp={base}-coverage",
                str(TEST_PATH),
                "-q",
            ],
        ),
        (
            "scoped_coverage_report",
            [
                str(root / ".venv/bin/coverage"),
                "report",
                f"--data-file={coverage_file}",
                f"--include=*/{MODULE_PATH.name}",
                "--show-missing",
                "--fail-under=100",
            ],
        ),
        (
            "ruff_check",
            [
                str(root / ".venv/bin/ruff"),
                "check",
                str(MODULE_PATH),
                str(WRAPPER_PATH),
                str(TEST_PATH),
                str(EXP7240_ACTIVE_TEST_PATH),
                "scripts/adversarial_verify.py",
            ],
        ),
        (
            "ruff_format",
            [
                str(root / ".venv/bin/ruff"),
                "format",
                "--check",
                str(MODULE_PATH),
                str(WRAPPER_PATH),
                str(TEST_PATH),
                str(EXP7240_ACTIVE_TEST_PATH),
                "scripts/adversarial_verify.py",
            ],
        ),
        (
            "mypy",
            [
                str(root / ".venv/bin/mypy"),
                str(MODULE_PATH),
                "scripts/adversarial_verify.py",
                "scripts/summarize_artifact.py",
            ],
        ),
        (
            "scoped_spec_coverage",
            [python, "-u", "scripts/check_spec_coverage.py", *map(str, named)],
        ),
        ("independent_raw_reducer", [python, "-u", "-c", reducer]),
        ("adversarial_verify", [python, "-u", "scripts/adversarial_verify.py", str(candidate)]),
        (
            "verdict_row_consistency",
            [python, "-u", "scripts/verdict_row_consistency_lint.py", str(candidate)],
        ),
    )


def _run_validations(
    root: Path, raw_dir: Path, artifact: JsonDict, started: float
) -> list[JsonDict]:  # pragma: no cover - exercised by the required CLI run.
    """Run scoped checks against the complete/null raw candidate."""

    candidate = raw_dir / RAW_CANDIDATE_NAME
    raw_rows = root / str(artifact["raw_rows_receipt"]["path"])
    _apply_terminal_state(artifact)
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["duration_s"] = max(time.monotonic() - started, 0.0001)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _atomic_write(candidate, artifact)
    log_dir = raw_dir / "validation"
    receipts: list[JsonDict] = []
    commands = _validation_commands(root, candidate, raw_rows)
    for index, (name, command) in enumerate(commands, 1):
        progress(
            started, 6, "before_subprocess", f"completed={index - 1}/{len(commands)} name={name}"
        )
        result = _run_capture(command, root)
        log_path = log_dir / f"{name}.log"
        _atomic_write(log_path, {"stdout": result.pop("stdout")})
        receipts.append(
            {
                "name": name,
                **result,
                "log_path": str(log_path.relative_to(root)),
                "log_sha256": sha256_path(log_path),
                "passed": result["exit_code"] == 0,
            }
        )
        progress(
            started,
            6,
            "after_subprocess",
            f"completed={index}/{len(commands)} name={name} exit={result['exit_code']}",
        )
    return receipts


def build_artifact(
    root: Path,
    run_date: str = RUN_DATE,
    *,
    output_path: Path,
    raw_dir: Path,
    checkpoint_path: Path,
    validation_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Measure, validate, and atomically publish one terminal receipt."""

    started = time.monotonic()
    artifact = _base_artifact(datetime.now(UTC).isoformat(), run_date)
    progress(started, 0, "start", "authenticate inputs and writable output paths")
    preconditions = collect_preconditions(root, output_path, raw_dir, checkpoint_path)
    artifact["preconditions_checked"] = preconditions
    artifact["gate_check_summary"] = _gate_summary(preconditions)
    artifact["source_artifact_hashes"] = _source_hashes(root)
    artifact["duration_s"] = max(time.monotonic() - started, 0.0001)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _atomic_write(checkpoint_path, artifact)
    progress(
        started,
        0,
        "end",
        f"completed={len(preconditions)} passed={artifact['gate_check_summary']['passed']}",
    )

    if artifact["gate_check_summary"]["passed"] is True:
        progress(started, 1, "start", "evaluate frozen duration boundaries and contradictions")
        rows = evaluate_boundary_fixtures()
        raw_rows_path = raw_dir / "boundary-rows.json"
        _atomic_write(raw_rows_path, rows)
        artifact["boundary_rows"] = rows
        artifact["rows"] = rows
        artifact["sample_size_budget"].update(
            attempted=len(rows), completed=len(rows), censored=sum(row["censored"] for row in rows)
        )
        artifact["raw_rows_receipt"] = {
            "path": _receipt_path(root, raw_rows_path),
            "sha256": sha256_path(raw_rows_path),
            "independent_reducer_errors": independent_reduce(raw_rows_path),
        }
        artifact["source_artifact_hashes"][_receipt_path(root, raw_rows_path)] = sha256_path(
            raw_rows_path
        )
        progress(
            started,
            1,
            "end",
            f"completed={len(rows)} mismatches={sum(not row['passed'] for row in rows)}",
        )

        progress(started, 2, "start", "build authenticated hermetic Exp7240 repository fixture")
        fixture = build_hermetic_exp7240_fixture(raw_dir / "hermetic-exp7240-repo")
        fixture_checks, _, fixture_upstream = read_hermetic_exp7240_fixture(fixture)
        fixture["passed"] = exp7240.gate_summary(fixture_checks)["passed"] is True
        fixture["observed_upstream_ready_score"] = fixture_upstream.get("belief_run_complete_score")
        artifact["fixture_repair_receipt"] = fixture
        progress(started, 2, "end", f"completed=1 passed={fixture['passed']}")

        progress(started, 3, "start", "write hashed historical model receipt sidecar")
        historical_path = raw_dir / "historical-model-receipts.json"
        historical = json.loads(
            (root / "results/experiment_7237_v637_mention_canary.json").read_text(encoding="utf-8")
        )
        _atomic_write(
            historical_path,
            {
                "scope": "historical",
                "source_path": "results/experiment_7237_v637_mention_canary.json",
                "source_sha256": sha256_path(
                    root / "results/experiment_7237_v637_mention_canary.json"
                ),
                "inference_substrate_class": historical.get("inference_substrate_class"),
                "duration_s": historical.get("duration_s"),
                "model_invoked": historical.get("model_invoked"),
                "current_invocation_counts": dict(ZERO_INVOCATION_COUNTS),
            },
        )
        artifact["historical_model_sidecar_receipt"] = {
            "path": _receipt_path(root, historical_path),
            "sha256": sha256_path(historical_path),
        }
        artifact["source_artifact_hashes"][_receipt_path(root, historical_path)] = sha256_path(
            historical_path
        )
        progress(started, 3, "end", "completed=1 current_model_calls=0")

        progress(started, 4, "start", "run actual checker and summarize-artifact floor E2E")
        artifact["e2e_floor_receipt"] = run_floor_e2e(root, raw_dir / "e2e")
        progress(
            started,
            4,
            "end",
            f"completed=3 mismatches={artifact['e2e_floor_receipt']['mismatch_count']}",
        )

        artifact["validation_receipts"] = [dict(row) for row in validation_receipts or []]
        if validation_receipts is None:
            progress(started, 5, "start", "prepare complete/null candidate for scoped validation")
            _apply_terminal_state(artifact)
            artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
            artifact["duration_s"] = max(time.monotonic() - started, 0.0001)
            artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
            _atomic_write(raw_dir / RAW_CANDIDATE_NAME, artifact)
            progress(started, 5, "end", "completed=1 candidate_is_raw_not_terminal_output")
            artifact["validation_receipts"] = _run_validations(root, raw_dir, artifact, started)

    progress(started, 7, "start", "derive and cold-check terminal artifact")
    _apply_terminal_state(artifact)
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["duration_s"] = max(time.monotonic() - started, 0.0001)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact, root=root)
    if errors:  # pragma: no cover - fail-closed CLI guard.
        raise ValueError(f"invalid Exp7261 artifact: {errors}")
    _atomic_write(checkpoint_path, artifact)
    _atomic_write(output_path, artifact)
    progress(
        started,
        7,
        "end",
        f"published={output_path} readiness={artifact['compute_contract_ready_score']}",
    )
    return artifact


def _date(value: str) -> str:
    """Accept only the fixed audited execution date."""

    datetime.strptime(value, "%Y%m%d")
    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(
    argv: Sequence[str] | None = None,
) -> int:  # pragma: no cover - exercised by the required CLI run.
    """Parse the thin command surface and publish the terminal artifact."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, type=_date)
    args = parser.parse_args(argv)
    build_artifact(
        REPO_ROOT,
        args.date,
        output_path=REPO_ROOT / DEFAULT_OUTPUT_PATH,
        raw_dir=REPO_ROOT / DEFAULT_RAW_DIR,
        checkpoint_path=REPO_ROOT / DEFAULT_CHECKPOINT_PATH,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
