"""REQ-REPORT-8071: compare process lifetime using unchanged fixed-answer checks.

This diagnostic owns full target-vocabulary scores before native memory changes.
Repeatability qualifies a measurement route; it cannot establish factual truth.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.inference import fixed_answer_likelihood_8022 as base
from carnot.inference import scoring_isolation_8033 as isolation
from carnot.inference.likelihood_runtime_8022 import EmbeddedRuntime
from carnot.inference.qwen_sufficiency_7920 import bounded
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v686_contract_validation import run_check
from carnot.reporting.v698_fixture_consumer_contract import clean_terminal, failure

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8071_v699_process_isolation_diagnostic"
TASK = "exp8071-process-isolation-diagnostic"
CLI = ROOT / "scripts/experiments" / (NAME + ".py")
OWNED = [Path(__file__), CLI]
TEST = ROOT / "tests/python/test_process_isolation_8071.py"
MODEL = "unsloth/Qwen3.8-27B-GGUF"
SEED = 6988059
CONFIG = dict(
    n_ctx=6384,
    n_batch=256,
    n_ubatch=256,
    flash_attn=False,
    n_gpu_layers=-1,
    main_gpu=0,
    tensor_split=[0.5, 0.5],
    logits_all=True,
    seed=SEED,
)
LIMITS = dict(
    seconds=2400,
    child_seconds=180,
    forward_seconds=120,
    forwards=64,
    duplicate_tolerance=1e-6,
    normalization_tolerance=1e-10,
)
START = time.monotonic()


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report real completed units so waiting for a child never implies more work."""
    print(
        f"[exp8071] phase={phase} elapsed_s={time.monotonic() - START:.3f} "
        f"completed={completed} pending={pending}",
        flush=True,
    )


def schedule(panel: list[Json]) -> list[Json]:
    """Two sweeps separate duplicates by other sources without consulting labels."""
    if len(panel) != 8 or len({r["source_cluster_id"] for r in panel}) != 8:
        raise ValueError("eight_complete_groups")
    return [
        dict(
            id=f"pass-{i:03d}",
            family_id=r["family_id"],
            source=r["source_cluster_id"],
            unit=f"pass-{i:03d}",
            arm=view,
            condition=view,
            lifetime_arm=arm,
            repeat=repeat,
            seed=SEED,
            denominator=len(r["target_tokens"]),
            numerator=None,
            status="planned",
            exclusion_reason=None,
        )
        for i, (repeat, r, view, arm) in enumerate(
            (repeat, r, view, arm)
            for repeat in ["A", "B"]
            for r in panel
            for view in ["full", "no_source"]
            for arm in ["current", "fresh_process"]
        )
    ]


def own_logits(matrix: Any, view: Json, raw: Path) -> Json:
    """Copy preceding-token vocabulary rows before a context frees mutable buffers."""
    raw.mkdir(parents=True, exist_ok=True)
    owned = np.array(matrix, copy=True)
    targets = view["tokens"][view["response_start"] :]
    if owned.ndim != 2 or owned.shape[0] != len(targets):
        raise ValueError("target_alignment")
    path = raw / "logits.npy"
    np.save(path, owned, allow_pickle=False)
    value = base.target_likelihood(owned, [0, *targets], 1)
    value["token_rows"] = [
        dict(token_id=t, logit_position=z, log_probability=p, probability=math.exp(p))
        for t, z, p in zip(
            targets,
            range(view["response_start"] - 1, len(view["tokens"]) - 1),
            value["target_logprobs"],
            strict=True,
        )
    ]
    return dict(value, logits_reference=dict(path=str(path), sha256=sha256_file(path)))


def reconstruct(row: Json, view: Json) -> float:
    """An independent reduction uses sealed logits rather than reported means."""
    ref = row["logits_reference"]
    path = Path(ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("logits_hash")
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    targets = view["tokens"][view["response_start"] :]
    positions = list(range(view["response_start"] - 1, len(view["tokens"]) - 1))
    ts = row["token_rows"]
    if (
        [t["token_id"] for t in ts] != targets
        or [t["logit_position"] for t in ts] != positions
        or len(matrix) != len(ts)
        or not row["context_closed"]
    ):
        raise ValueError("token_alignment")
    probabilities = []
    for values, target, token in zip(matrix, targets, ts, strict=True):
        values = np.asarray(values, dtype=np.float64)
        logz = float(np.logaddexp.reduce(values))
        lp = float(values[target] - logz)
        norm = abs(float(np.exp(values - logz).sum()) - 1)
        if (
            not np.isfinite(values).all()
            or norm > 1e-10
            or abs(lp - token["log_probability"]) > 1e-10
            or abs(math.exp(lp) - token["probability"]) > 1e-10
        ):
            raise ValueError("probability_or_normalization")
        probabilities.append(lp)
    numerator = math.fsum(-p for p in probabilities)
    if (
        row["denominator"] != len(ts)
        or abs(row["numerator"] - numerator) > 1e-8
        or abs(row["mean_nll"] - numerator / len(ts)) > 1e-10
    ):
        raise ValueError("aggregate")
    return numerator / len(ts)


def capture(
    panel: list[Json], current: Any, fresh: Any, raw: Path, *, deadline: float
) -> list[Json]:
    """Durable starts and nonstarts keep failures from shrinking the denominator."""
    slots = schedule(panel)
    items = {r["family_id"]: r for r in panel}
    stopped: set[str] = set()
    for row in slots:
        atomic_json(raw / row["id"] / "slot.json", row)
    for i, row in enumerate(slots):
        arm = row["lifetime_arm"]
        if arm in stopped or time.monotonic() >= deadline:
            row.update(status="censored", censor_reason="arm_failure_or_model_budget")
        else:
            row.update(status="started", started_monotonic_ns=time.monotonic_ns())
            atomic_json(raw / row["id"] / "slot.json", row)
            progress(
                "before_forward", sum(r["status"] == "completed" for r in slots), len(slots) - i
            )
            try:
                result = (current if arm == "current" else fresh)(
                    items[row["family_id"]]["views"][row["arm"]], raw / row["id"]
                )
                row.update(result)
                row["numerator"] = math.fsum(-t["log_probability"] for t in row["token_rows"])
                row.update(status="completed", mean_nll=row["numerator"] / row["denominator"])
                reconstruct(row, items[row["family_id"]]["views"][row["arm"]])
            except (OSError, RuntimeError, TimeoutError, ValueError, KeyError) as error:
                row.update(status="failed", failure_reason=f"{type(error).__name__}:{error}")
                stopped.add(arm)
            row["ended_monotonic_ns"] = time.monotonic_ns()
            progress(
                "after_forward", sum(r["status"] == "completed" for r in slots), len(slots) - i - 1
            )
        atomic_json(raw / row["id"] / "slot.json", row)
    return slots


def reduce(panel: list[Json], rows: list[Json]) -> Json:
    """Recompute duplicate drift and distinguish numerical failure from unfinished work."""
    roster = schedule(panel)
    if len(rows) != len(roster) or any(
        any(
            r[k] != s[k]
            for k in ["id", "family_id", "arm", "repeat", "lifetime_arm", "source", "denominator"]
        )
        for r, s in zip(rows, roster, strict=True)
    ):
        raise ValueError("slot_roster")
    items = {r["family_id"]: r for r in panel}
    means, completed = {}, {}
    for r in rows:
        if r["status"] == "completed":
            key = (r["lifetime_arm"], r["family_id"], r["arm"], r["repeat"])
            means[key] = reconstruct(r, items[r["family_id"]]["views"][r["arm"]])
            completed[key] = r
    drifts, divergences = [], []
    for arm in ["current", "fresh_process"]:
        for item in panel:
            for view in ["full", "no_source"]:
                keys = [(arm, item["family_id"], view, repeat) for repeat in ["A", "B"]]
                pair = [means.get(k) for k in keys]
                drift = abs(pair[0] - pair[1]) if all(x is not None for x in pair) else None
                differences = (
                    [
                        b["log_probability"] - a["log_probability"]
                        for a, b in zip(
                            completed[keys[0]]["token_rows"],
                            completed[keys[1]]["token_rows"],
                            strict=True,
                        )
                    ]
                    if drift is not None
                    else []
                )
                row = dict(
                    lifetime_arm=arm,
                    family_id=item["family_id"],
                    arm=view,
                    drift=drift,
                    tolerance=1e-6,
                    passed=drift is not None and drift <= 1e-6,
                    raw_score_differences=differences,
                    process_ids=[completed[k]["process_id"] for k in keys if k in completed],
                    context_ids=[completed[k]["context_identity"] for k in keys if k in completed],
                )
                drifts.append(row)
                first = next((i for i, x in enumerate(differences) if x != 0), None)
                if first is not None:
                    divergences.append(
                        dict(
                            row,
                            first_token_index=first,
                            token_id=item["target_tokens"][first],
                            score_difference=differences[first],
                        )
                    )
    passed = {
        arm: all(d["passed"] for d in drifts if d["lifetime_arm"] == arm)
        for arm in ["current", "fresh_process"]
    }
    return dict(
        arm_passed=passed,
        duplicate_drift_rows=drifts,
        first_divergent_token_rows=divergences,
        scored_tokens=sum(r["denominator"] for r in rows if r["status"] == "completed"),
        current_failure_reproduced=any(
            d["drift"] is not None and d["drift"] > 1e-6
            for d in drifts
            if d["lifetime_arm"] == "current"
        ),
        token_alignment_checks=dict(
            passed=True, completed_count=len(completed), positions="exact_preceding_token"
        ),
        normalization_checks=dict(passed=True, tolerance=1e-10, dtype="float64"),
    )


def controls(raw: Path) -> Json:
    """Negative fixtures show that scalar agreement cannot hide shifted or stale data."""
    view = dict(tokens=[0, 2, 0], response_start=1)
    row = dict(
        own_logits([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]], view, raw),
        context_closed=True,
        denominator=2,
    )
    row.update(numerator=math.fsum(-t["log_probability"] for t in row["token_rows"]))
    row["mean_nll"] = row["numerator"] / 2
    reconstruct(row, view)
    row["token_rows"][0]["logit_position"] += 1
    shifted = False
    try:
        reconstruct(row, view)
    except ValueError:
        shifted = True
    row["token_rows"][0]["logit_position"] -= 1
    np.save(Path(row["logits_reference"]["path"]), np.ones((2, 3)))
    stale = False
    try:
        reconstruct(row, view)
    except ValueError:
        stale = True
    return dict(
        passed=shifted and stale,
        shifted_token_rejected=shifted,
        stale_buffer_rejected=stale,
        verifier_is_oracle=True,
        independent_count=0,
    )


def preconditions(root: Path, raw: Path) -> Json:
    """Authenticate historical bytes without treating failed likelihoods as qualified features."""
    names = [
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "ops/e2e-test-plan.md",
        "openspec/capabilities/research-reporting/spec.md",
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "ops/exclusion_manifest.yaml",
        "openspec/change-proposals/research-roadmap-vNEXT.md",
        "python/carnot/inference/scoring_isolation_8033.py",
        "python/carnot/inference/likelihood_isolation_runtime_8033.py",
        "python/carnot/inference/likelihood_runtime_8022.py",
        "python/carnot/inference/fixed_answer_likelihood_8022.py",
        "python/carnot/experiment_8059_v698_fit_source_scoring.py",
        "research-references.md",
        "results/experiment_8059_v698_fit_source_scoring.json",
    ]
    plan: Json = dict(panel=[], references=[], failures=[])
    for name in names + [
        f".venv/bin/{n}" for n in ["python", "pytest", "coverage", "ruff", "mypy"]
    ]:
        path = root / name
        if not path.is_file():
            plan["failures"].append(failure(path, "resource_exists", True, False, "exp8059"))
        else:
            digest = sha256_file(path)
            snapshot = raw / "inputs" / (digest[7:] + path.suffix)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(path.read_bytes())
            plan["references"].append(
                dict(path=str(path), sha256=digest, snapshot_path=str(snapshot))
            )
    primary = root / names[-1]
    if not primary.is_file():
        return plan
    try:
        terminal = clean_terminal(primary)
        upstream = json.loads(primary.read_text())
        for path in [Path(terminal["terminal_path"]), Path(terminal["validator_path"])]:
            plan["references"].append(dict(path=str(path), sha256=sha256_file(path)))
        panel_path = primary.parent / "raw" / primary.stem / "frozen_panel.json"
        ref = next(r for r in upstream["raw_shard_hashes"] if Path(r["path"]) == panel_path)
        if sha256_file(panel_path) != ref["sha256"]:
            raise ValueError("frozen_panel_hash")
        runtime_ref = next(
            r
            for r in upstream["raw_shard_hashes"]
            if Path(r["path"]) == panel_path.with_name("runtime.json")
        )
        if sha256_file(Path(runtime_ref["path"])) != runtime_ref["sha256"]:
            raise ValueError("historical_runtime_hash")
        plan["references"].append(runtime_ref)
        all_rows = json.loads(panel_path.read_text())["rows"]
        panel = [r for r in all_rows if r["role"] == "fit" and r["slot"] < 8]
        if any(not r["eligible"] or not r["source_bytes"] or not r["answer_bytes"] for r in panel):
            raise ValueError("complete_pilot")
        schedule(panel)
        plan.update(panel=panel, historical_model=upstream["model_identity_receipt"])
        plan["references"].append(ref)
        atomic_json(raw / "panel.json", dict(rows=panel))
    except (OSError, ValueError, KeyError, StopIteration) as error:
        plan["failures"].append(failure(primary, "historical_custody", True, str(error), "exp8059"))
    return plan


def execute(
    argv: list[str], raw: Path, timeout: float, *, cwd: Path = ROOT, expected: int = 0
) -> Json:
    """Keep parent heartbeats active and hash logs only after the real child exits."""
    raw.mkdir(parents=True, exist_ok=True)
    log = raw / "child.log"
    began = time.monotonic()
    progress("before_subprocess")
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    with log.open("w") as stream:
        child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        deadline, heartbeat, timed_out = began + timeout, began, False
        while child.poll() is None:
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
                break
            if now - heartbeat >= 30:
                progress("subprocess_pending", 0, 1)
                heartbeat = now
            time.sleep(min(0.2, max(0.001, deadline - now)))
        code = child.wait(timeout=5)
    progress("after_subprocess", 1, 0)
    return dict(
        argv=argv,
        exit_code=code,
        duration_s=time.monotonic() - began,
        timed_out=timed_out,
        normal_exit=code >= 0 and not timed_out,
        passed=code == expected and not timed_out,
        log_path=str(log),
        log_sha256=sha256_file(log),
    )


def build_identity() -> Json:
    """Bind the Python and native decoder so different executables cannot hide a build change."""
    import inspect
    import llama_cpp

    return dict(
        native=sha256_file(Path(llama_cpp.llama_cpp._lib._name)),
        binding=sha256_file(Path(inspect.getfile(llama_cpp.Llama))),
        version=llama_cpp.__version__,
    )


def gpu_observation() -> Json:
    """Read allocation and utilization without changing any unrelated GPU process."""
    p = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if p.returncode:
        raise RuntimeError("nvidia_smi:" + p.stderr)
    devices = [
        dict(
            index=int(z[0]),
            uuid=z[1].strip(),
            used_mb=int(z[2]),
            free_mb=int(z[3]),
            utilization_pct=int(z[4]),
        )
        for z in [line.split(",") for line in p.stdout.strip().splitlines()]
    ]
    return dict(
        devices=devices,
        observed_monotonic_ns=time.monotonic_ns(),
        command_argv=p.args,
        stdout=p.stdout,
    )


def load_model(plan: Json, raw: Path, deadline: float) -> tuple[Any, Json]:
    """Load the same cached build and verify embedded tokenizer bytes before scoring."""
    import llama_cpp
    from carnot.experiment_8059_v698_fit_source_scoring import freeze

    started = time.monotonic()
    receipt: Json = dict(
        process_id=os.getpid(),
        status="started",
        started_monotonic_ns=time.monotonic_ns(),
        model_build_hashes=build_identity(),
    )
    atomic_json(raw / "load.json", receipt)
    path = Path(plan["model_path"])
    stat = path.stat()
    if [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns] != plan["model_stat"]:
        raise ValueError("cached_model_stat_changed")
    if receipt["model_build_hashes"] != plan["build"]:
        raise ValueError("model_build_changed")
    progress("before_model_load")
    model = bounded(
        lambda: llama_cpp.Llama(model_path=str(path), **CONFIG, verbose=False),
        min(120, deadline - time.monotonic()),
    )
    progress("after_model_load")
    try:
        actual = freeze(plan["panel"], EmbeddedRuntime(model))
        if any(
            any(a[k] != b[k] for k in ["target_tokens", "views", "response_token_offsets"])
            for a, b in zip(actual, plan["panel"], strict=True)
        ):
            raise ValueError("frozen_tokenizer_alignment")
        if (
            canonical_hash(model.metadata["tokenizer.chat_template"])
            != plan["chat_template_sha256"]
        ):
            raise ValueError("tokenizer_template_hash")
        gpu = gpu_observation()
        before = {d["index"]: d["used_mb"] for d in plan["gpu_before"]["devices"]}
        if sum(d["used_mb"] - before[d["index"]] for d in gpu["devices"]) < 10000:
            raise ValueError("cuda_allocation_missing")
        receipt.update(
            status="completed",
            load_seconds=time.monotonic() - started,
            offload_evidence=gpu,
            ended_monotonic_ns=time.monotonic_ns(),
        )
        atomic_json(raw / "load.json", receipt)
        return model, receipt
    except (ValueError, RuntimeError, OSError):
        progress("before_model_cleanup")
        model.close()
        progress("after_model_cleanup")
        raise


def manifest(private: Path) -> list[Json]:
    """Freeze exact commands and limit coverage to the new producer and its thin CLI."""
    py, cov, pytest, ruff, mypy = [
        str(ROOT / ".venv/bin" / n) for n in ["python", "coverage", "pytest", "ruff", "mypy"]
    ]
    include = "--include=" + ",".join(map(str, OWNED))
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    e2e = ROOT / "scripts/experiments/experiment_7868_v683_intervention_protocol.py"
    commands = [
        (
            "unit_coverage",
            [
                cov,
                "run",
                f"--data-file={private / '.coverage'}",
                include,
                "-m",
                "pytest",
                *common,
                f"--basetemp={private / 'unit'}",
                str(TEST),
            ],
            300,
        ),
        (
            "coverage_report",
            [
                cov,
                "report",
                f"--data-file={private / '.coverage'}",
                include,
                "--show-missing",
                "--fail-under=100",
            ],
            60,
        ),
        (
            "coverage_json",
            [
                cov,
                "json",
                f"--data-file={private / '.coverage'}",
                include,
                "-o",
                str(private / "coverage.json"),
            ],
            60,
        ),
        (
            "consumer_tests",
            [
                pytest,
                *common,
                f"--basetemp={private / 'consumer'}",
                str(ROOT / "tests/python/test_fit_source_scoring_8059.py"),
                str(ROOT / "tests/python/test_scoring_isolation_8033.py"),
                str(ROOT / "tests/python/test_scorer_workspace_8045.py"),
            ],
            300,
        ),
        ("ruff_check", [ruff, "check", *map(str, OWNED), str(TEST)], 60),
        ("ruff_format", [ruff, "format", "--check", *map(str, OWNED), str(TEST)], 60),
        ("mypy_strict", [mypy, "--strict", *map(str, OWNED)], 180),
        ("scoped_spec", [py, str(ROOT / "scripts/check_spec_coverage.py"), str(TEST)], 180),
        (
            "e2e_015",
            [
                pytest,
                *common,
                f"--basetemp={private / 'e2e015'}",
                str(ROOT / "tests/python/test_source_boundary_7852.py"),
            ],
            180,
        ),
        (
            "e2e_016_fixture",
            [py, str(e2e), "--date", "20260929", "--fixture-e2e", str(private / "e2e016.json")],
            120,
        ),
        (
            "e2e_016_replay",
            [py, str(e2e), "--date", "20260929", "--cold-replay", str(private / "e2e016.json")],
            60,
        ),
        ("repository_full_suite", [pytest, "tests/python", "-q"], 180),
    ]
    return [
        dict(
            name=name,
            argv=argv,
            deadline_s=deadline,
            expected_exit=0,
            classification="diagnostic" if name == "repository_full_suite" else "required",
        )
        for name, argv, deadline in commands
    ]


def fixture_measure(panel: list[Json], raw: Path, mode: str) -> list[Json]:
    """Oracle fixtures validate execution routes and earn no usable live scoring credit."""

    def score(view: Json, directory: Path) -> Json:
        if mode == "failure":
            raise RuntimeError("fixture_owned_child_failure")
        targets = view["tokens"][view["response_start"] :]
        return dict(
            own_logits(np.ones((len(targets), max(targets) + 1)), view, directory),
            process_id=os.getpid(),
            context_identity=str(directory),
            context_closed=True,
            model_build_hashes={"fixture": "exact_oracle"},
            load_seconds=0,
            forward_seconds=0,
            cleanup_seconds=0,
            offload_evidence={},
        )

    return capture(panel, score, score, raw, deadline=0 if mode == "blocked" else math.inf)


def artifact(
    plan: Json, rows: list[Json], raw: Path, validation: list[Json], *, fixture: bool
) -> Json:
    """Keep completed diagnoses separate from usable routes and independent scientific benefit."""
    panel = plan["panel"]
    reduction = (
        reduce(panel, rows)
        if panel
        else dict(
            arm_passed={},
            duplicate_drift_rows=[],
            first_divergent_token_rows=[],
            scored_tokens=0,
            current_failure_reproduced=False,
            token_alignment_checks={},
            normalization_checks={},
        )
    )
    required = [v for v in validation if v.get("classification") == "required"]
    checks = bool(required) and all(v["passed"] for v in required)
    complete = len(rows) == 64 and all(r["status"] == "completed" for r in rows)
    blocked = bool(plan["failures"])
    loads = [json.loads(p.read_text()) for p in raw.rglob("load.json")]
    fresh_pids = [
        r.get("process_id")
        for r in rows
        if r["lifetime_arm"] == "fresh_process" and r["status"] == "completed"
    ]
    custody = (
        len(fresh_pids) == 32
        and len(set(fresh_pids)) == 32
        and len(loads) == 33
        and all(r.get("status") == "completed" for r in loads)
    )
    ready = int(
        not fixture
        and not blocked
        and complete
        and checks
        and custody
        and reduction["arm_passed"].get("fresh_process", False)
    )
    diagnosis = int(not blocked and complete and (checks or fixture))
    cls = (
        "blocked"
        if blocked
        else "disqualified"
        if not complete or (not fixture and not checks)
        else ("circular_positive" if fixture else "positive" if ready else "null")
    )
    load_times = [r["load_seconds"] for r in loads if r.get("status") == "completed"]
    forward_times = [r.get("forward_seconds", 0) for r in rows if r["status"] == "completed"]
    cleanup_times = [r.get("cleanup_seconds", 0) for r in rows] + [
        plan.get("current_cleanup_seconds", 0)
    ]
    unit_upper = (
        max(load_times, default=0) + max(forward_times, default=0) + max(cleanup_times, default=0)
    )
    projection = dict(
        groups=192,
        views=4,
        forwards=768,
        method="2x_max_observed_load_forward_cleanup",
        measured_upper_seconds=2 * 768 * unit_upper if load_times and forward_times else None,
        hard_budget_upper_seconds=768 * 180,
        scheduled=False,
        includes_reloads=True,
    )
    refs = [
        dict(path=str(p), sha256=sha256_file(p))
        for p in sorted(raw.rglob("*"))
        if p.is_file()
        and p.name
        not in ["terminal_candidate.json", "terminal_validation.json", "publication.lock"]
    ]
    counts = {
        s: sum(r["status"] == s for r in rows)
        for s in ["completed", "censored", "failed", "excluded"]
    }
    events = dict(
        model_loads_attempted=len(loads),
        model_loads_completed=len(load_times),
        model_loads_failed=len(loads) - len(load_times),
        model_loads_cancelled=0,
        model_loads_in_flight=0,
        generation=dict(attempted=0, completed=0, failed=0, cancelled=0, in_flight=0),
    )
    value: Json = dict(
        experiment_id=8071,
        task_id=TASK,
        schema="carnot.v699.process_isolation.v1",
        run_date=plan["date"],
        milestone="2026.10.699",
        honest_verdict="complete_" + cls + "_process_isolation",
        verdict_class=cls,
        verifier_is_oracle=fixture,
        claim_scope="fixed_answer_numerical_lifetime_diagnostic",
        flagged_adversarial=False,
        required_checks_passed=checks,
        validation_receipts=validation,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        diagnostic_ready_score=diagnosis,
        process_isolation_ready_score=ready,
        generalized_learning_benefit_score=0,
        inference_substrate="no_model_load" if not loads else "live_llm_embedding_extraction",
        inference_substrate_class="no_model_load" if not loads else "model_load_no_generation",
        MODEL_SPECS=[MODEL],
        model_invocation_counts=events,
        model_invocation_counts_schema="carnot.operation_counters.separate_generation.v1",
        substrate_declaration=dict(
            substrate="live_llm_embedding_extraction",
            mode="model_load_no_generation",
            operation="teacher_forced_scoring",
            generated_tokens=0,
            floor_s=2,
            duration_floor_s=2,
        ),
        rows=rows,
        lifetime_arm_rows=rows,
        intended_count=64,
        eligible_count=len(rows),
        independent_count=len(panel),
        completed_count=counts["completed"],
        censored_count=counts["censored"],
        excluded_count=counts["excluded"],
        failed_count=counts["failed"],
        sample_size_budget=dict(
            groups=8,
            forwards=64,
            unit="forward_slots",
            independent_unit="frozen_original_source_group",
            limits=LIMITS,
        ),
        gate_check_summary=plan["failures"],
        random_seed=SEED,
        source_artifact_hashes=plan["references"],
        raw_shard_hashes=refs,
        code_config_hashes=dict(
            files={str(p): sha256_file(p) for p in OWNED}, config=canonical_hash(CONFIG)
        ),
        execution_configuration=CONFIG,
        reproducibility_checksum=canonical_hash(dict(panel=panel, config=CONFIG, seed=SEED)),
        phase_spans=plan.get("phase_spans", []),
        process_ids=sorted({r["process_id"] for r in rows if "process_id" in r}),
        context_ids=[r["context_identity"] for r in rows if "context_identity" in r],
        model_build_hashes=plan.get("build", {}),
        offload_evidence=plan.get("offload_evidence", {}),
        load_forward_cleanup_seconds=dict(
            loads=load_times, forwards=forward_times, cleanup=cleanup_times
        ),
        full_capture_budget_projection=projection,
        forward_pass_counts=sum(r["status"] in ["completed", "failed"] for r in rows),
        generated_tokens=0,
        model_load_counts=events["model_loads_attempted"],
        duration_s=time.monotonic() - START,
        causal_interpretation="Hypothesis remains unresolved when current-arm drift does not recur; no cause is inferred.",
        methodology_note="Same eight complete frozen groups; separated duplicates and full-vocabulary preceding-token teacher forcing. No labels, generation, retries or full capture.",
        repository_health=[v for v in validation if v.get("classification") == "diagnostic"],
        **reduction,
    )
    value["field_principles"] = {
        k: "Own evidence for " + k + " prevents claims beyond the frozen numerical diagnostic."
        for k in value
    }
    value["field_principles"].update(
        causal_interpretation="Nonrecurrence cannot identify the cause of a historical failure.",
        arm_passed="Every unchanged group/view duplicate gate must pass; averages cannot hide a failure.",
        current_failure_reproduced="A fresh-process success alone does not explain the earlier failure.",
        execution_configuration="Both arms must share kernels and placement to compare lifetime.",
        methodology_note="Fixed supplied-answer scoring cannot establish generation quality or detection benefit.",
        repository_health="Existing global failures cannot be presented as passes or silently folded into owned checks.",
        model_invocation_counts_schema="Load attempts and generation calls are different operations.",
        run_date="The requested date identifies the experiment rather than implying an earlier observation.",
        schema="Consumers must distinguish this lifetime diagnostic from a full source capture.",
        task_id="Exact task identity prevents filename fallback from selecting unrelated evidence.",
        experiment_id="One producer identity prevents ambiguous primary publication.",
        milestone="Development exposure carries forward across milestone names.",
        duration_s="Elapsed invocation time must not be mistaken for forward-only throughput.",
    )
    if blocked:
        value["honest_verdict"] = "complete_blocked_" + str(
            plan["failures"][0].get("field", "resource")
        )
    return value


def native_score(model: Any, view: Json, raw: Path, deadline: float) -> Json:
    """Reuse the qualified fresh-context controller but take custody of raw vocabulary rows."""
    owned: Json = {}
    started = time.monotonic()

    def take(matrix: Any, tokens: list[int], boundary: int) -> Json:
        owned["forward_gpu_activity"] = gpu_observation()
        owned.update(own_logits(np.asarray(matrix)[boundary - 1 : len(tokens) - 1], view, raw))
        return base.target_likelihood(matrix, tokens, boundary)

    progress("before_native_forward")
    with patch.object(isolation, "likelihood", take):
        score = isolation.NativeController(EmbeddedRuntime(model), deadline).score(
            view, "fresh_full"
        )
    progress("after_native_forward")
    return dict(
        score,
        **{k: v for k, v in owned.items() if k not in score},
        process_id=os.getpid(),
        forward_seconds=time.monotonic() - started,
        model_build_hashes=build_identity(),
        context_identity=f"pid-{os.getpid()}:addr-{score['native_context_address']}:slot-{raw.name}",
    )


def worker(plan_path: Path, output: Path) -> None:
    """A fresh executable owns exactly one load and view; failure remains terminal evidence."""
    plan = json.loads(plan_path.read_text())
    model, result = None, dict(passed=False)
    began = time.monotonic()
    try:
        model, load = load_model(plan, output.parent, min(plan["deadline"], began + 180))
        result.update(load)
        result.update(
            native_score(model, plan["view"], output.parent, min(plan["deadline"], began + 180))
        )
        result["passed"] = True
    except (OSError, RuntimeError, TimeoutError, ValueError, KeyError) as error:
        result["error"] = f"{type(error).__name__}:{error}"
    finally:
        cleanup = time.monotonic()
        progress("before_model_cleanup")
        if model is not None:
            model.close()
        result.update(
            cleanup_seconds=time.monotonic() - cleanup,
            child_seconds=time.monotonic() - began,
            gpu_after_cleanup=gpu_observation(),
        )
        atomic_json(output, result)
        progress("after_model_cleanup")


def live(plan: Json, raw: Path) -> list[Json]:
    """Retain current weights while owned children load identical weights on leased GPUs."""
    from llama_cpp import llama_cpp as native
    from carnot.gpu_lease_phase_journal import GpuLease, LeaseError
    from carnot.inference.sota_models import cached_current_model

    began = time.monotonic()
    deadline = began + LIMITS["seconds"]
    model, leases = None, []
    rows = schedule(plan["panel"])
    for row in rows:
        row.update(status="censored", censor_reason="external_precondition")
        atomic_json(raw / "forwards" / row["id"] / "slot.json", row)
    try:
        cached = cached_current_model()
        if not cached or cached["hf_id"] != MODEL or "Q4_K_M" not in cached["model_path"]:
            plan["failures"].append(
                failure(
                    raw / "model_plan.json", "cached_Q4_K_M_model", MODEL, cached, "cached_qwen_GPU"
                )
            )
            raise ValueError("mandated_Q4_K_M_cache")
        path = Path(cached["model_path"])
        progress("before_model_hash")
        actual = bounded(lambda: sha256_file(path), 120)
        progress("after_model_hash")
        if actual != plan["historical_model"]["gguf_sha256"]:
            plan["failures"].append(
                failure(
                    path, "gguf_sha256", plan["historical_model"]["gguf_sha256"], actual, "exp8059"
                )
            )
            raise ValueError("gguf_sha256:" + str(actual))
        plan["references"].append(dict(path=str(path), sha256=actual))
        build = build_identity()
        historical = json.loads(
            (ROOT / "results/raw/experiment_8059_v698_fit_source_scoring/runtime.json").read_text()
        )
        if build["native"] != historical["llama_cpp_build"]["native_library_sha256"]:
            plan["failures"].append(
                failure(
                    ROOT / "results/raw/experiment_8059_v698_fit_source_scoring/runtime.json",
                    "native_library_sha256",
                    historical["llama_cpp_build"]["native_library_sha256"],
                    build["native"],
                    "exp8059",
                )
            )
            raise ValueError("native_build_hash:" + build["native"])
        gpu = gpu_observation()
        if (
            not native.llama_supports_gpu_offload()
            or len(gpu["devices"]) != 2
            or any(d["used_mb"] >= 1000 or d["free_mb"] < 20000 for d in gpu["devices"])
        ):
            plan["failures"].append(
                failure(
                    raw / "model_plan.json",
                    "cuda_offload_and_idle_dual_gpu",
                    dict(supported=True, count=2, used_mb_lt=1000, free_mb_gte=20000),
                    dict(supported=bool(native.llama_supports_gpu_offload()), observation=gpu),
                    "cached_qwen_GPU",
                )
            )
            raise ValueError("cuda_offload_or_free_gpu:" + json.dumps(gpu))
        stat = path.stat()
        plan.update(
            model_path=str(path),
            model_stat=[stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns],
            build=build,
            gpu_before=gpu,
            chat_template_sha256=plan["historical_model"]["chat_template_sha256"],
        )
        for d in gpu["devices"]:
            lease = GpuLease.acquire(
                runtime_dir="/tmp/carnot-gpu-leases",
                task_id=TASK,
                device_uuid=d["uuid"],
                expected_model=str(path),
                vram_before_mb=d["used_mb"],
                ttl_s=2520,
            )
            leases.append(lease)
            lease.transition("admitted")
            lease.transition("loading")
        atomic_json(raw / "model_plan.json", plan)
        model, load = load_model(plan, raw / "current", deadline)
        plan["offload_evidence"] = dict(
            current_load=load, lease_owners=[l.owner_receipt() for l in leases]
        )
        resident = gpu_observation()
        for lease, d in zip(leases, resident["devices"], strict=True):
            lease.transition("resident", vram_mb=d["used_mb"])
            lease.transition("inferencing")

        def current(view: Json, directory: Path) -> Json:
            return native_score(model, view, directory, deadline)

        def fresh(view: Json, directory: Path) -> Json:
            child_plan = dict(plan, view=view, deadline=deadline, gpu_before=gpu_observation())
            atomic_json(directory / "plan.json", child_plan)
            receipt = execute(
                [
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(CLI),
                    "--worker",
                    str(directory / "plan.json"),
                    "--worker-output",
                    str(directory / "worker.json"),
                ],
                directory / "subprocess",
                min(180, max(0.001, deadline - time.monotonic())),
            )
            atomic_json(directory / "child_receipt.json", receipt)
            if not receipt["passed"]:
                raise RuntimeError("fresh_child_exit:" + str(receipt["exit_code"]))
            result = json.loads((directory / "worker.json").read_text())
            if not result["passed"]:
                raise RuntimeError(result["error"])
            return result

        progress("before_benchmark", 0, 64)
        rows = capture(plan["panel"], current, fresh, raw / "forwards", deadline=deadline - 30)
        progress(
            "after_benchmark",
            sum(r["status"] == "completed" for r in rows),
            sum(r["status"] == "censored" for r in rows),
        )
    except (OSError, RuntimeError, TimeoutError, ValueError, KeyError, LeaseError) as error:
        plan["failures"].append(
            failure(
                raw / "model_plan.json",
                "model_prerequisite",
                True,
                f"{type(error).__name__}:{error}",
                "cached_qwen_GPU",
            )
        )
    finally:
        cleanup = time.monotonic()
        progress("before_model_cleanup")
        if model is not None:
            bounded(model.close, min(30, max(0.001, deadline - time.monotonic())))
        after = gpu_observation() if leases else {}
        for lease in leases:
            if lease.document["phase"] == "inferencing":
                lease.transition("unloading")
                d = next(d for d in after["devices"] if d["uuid"] == lease.device_uuid)
                lease.transition(
                    "validating", vram_mb=d["used_mb"], exit_code=0, unload_observed=True
                )
                lease.transition("terminal_complete")
            else:
                lease.transition("terminal_blocked")
            receipt = lease.release()
            atomic_json(
                raw / ("lease-" + lease.device_uuid + ".json"),
                dict(receipt=receipt, journal=lease.document),
            )
        plan["phase_spans"] = [
            dict(
                phase="model_work_including_hash_load_cleanup", duration_s=time.monotonic() - began
            )
        ]
        plan.setdefault("offload_evidence", {})["after_cleanup"] = after
        plan["current_cleanup_seconds"] = time.monotonic() - cleanup
        progress("after_model_cleanup")
    return rows


def check_candidate(path: Path) -> Json:
    """Reconstruct owned evidence before atomic publication exposes candidate bytes."""
    v = json.loads(path.read_text())
    for ref in v["raw_shard_hashes"] + v["source_artifact_hashes"]:
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("evidence_hash:" + ref["path"])
    if v["rows"]:
        panel_path = path.parent / "panel.json"
        reduced = reduce(json.loads(panel_path.read_text())["rows"], v["rows"])
        if any(v[k] != reduced[k] for k in reduced):
            raise ValueError("reduction_changed")
    if v["completed_count"] != sum(r["status"] == "completed" for r in v["rows"]):
        raise ValueError("completed_count")
    return dict(passed=True, independent_primitive_reconstruction=True)


def replay(path: Path) -> bool:
    """Cold replay authenticates terminal bytes and every original primitive without model calls."""
    from carnot.reporting.primary_publication import read_bound_sidecar

    try:
        v = json.loads(path.read_text())
        terminal = json.loads(Path(v["terminal_validation_sidecar_path"]).read_text())
        if terminal["primary_sha256"] != sha256_file(path):
            return False
        report = read_bound_sidecar(path, Path(terminal["sidecar_path"]))
        if not report["report"]["passed"]:
            return False
        candidate = path.parent / "raw" / path.stem / "terminal_candidate.json"
        if candidate.read_bytes() != path.read_bytes():
            return False
        check_candidate(candidate)
        return True
    except (OSError, ValueError, KeyError):
        return False


def main(argv: list[str] | None = None) -> int:
    """Own measurement, normal child exits and current validation before terminal publication."""
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261003")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument(
        "--fixture-mode", choices=["success", "blocked", "failure"], default="success"
    )
    parser.add_argument("--fixture-worker", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        return 0 if replay(args.cold_replay) else 1
    if args.worker:
        worker(args.worker, args.worker_output)
        return 0
    if args.fixture_worker:
        p = json.loads(args.fixture_worker.read_text())
        rows = fixture_measure(p["panel"], args.worker_output.parent / "forwards", p["mode"])
        atomic_json(args.worker_output, dict(rows=rows))
        return 0
    output = args.output.absolute()
    if args.fixture_input and output.is_relative_to(ROOT / "results"):
        raise ValueError("private_fixture_output_required")
    if output.is_file():
        progress("existing_terminal_no_retry")
        return 0 if replay(output) else 1
    raw = output.parent / "raw" / output.stem
    raw.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="carnot-8071-") as work:
        private = Path(work)
        specs = manifest(private)
        atomic_json(raw / "validation_commands.json", specs)
        progress("preconditions_before")
        if args.fixture_input:
            panel = json.loads(args.fixture_input.read_text())["rows"]
            plan = dict(
                panel=panel,
                references=[
                    dict(path=str(args.fixture_input), sha256=sha256_file(args.fixture_input))
                ],
                failures=[],
            )
            if args.fixture_mode == "blocked":
                plan["failures"].append(
                    failure(args.fixture_input, "fixture_resource", True, False)
                )
            atomic_json(raw / "panel.json", dict(rows=panel))
        else:
            plan = preconditions(ROOT, raw)
        plan["date"] = args.date
        atomic_json(
            raw / "measurement_code_checkpoint.json",
            dict(files={str(p): sha256_file(p) for p in OWNED}, config=CONFIG, limits=LIMITS),
        )
        atomic_json(raw / "fixture_controls.json", controls(private / "controls"))
        progress("preconditions_after")
        validation = []
        if args.fixture_input:
            atomic_json(
                raw / "fixture_plan.json", dict(panel=plan["panel"], mode=args.fixture_mode)
            )
            receipt = execute(
                [
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(CLI),
                    "--fixture-worker",
                    str(raw / "fixture_plan.json"),
                    "--worker-output",
                    str(raw / "fixture_runtime.json"),
                ],
                raw / "fixture_child",
                60,
                cwd=private,
            )
            validation.append(
                dict(receipt, name="private_fixture_child", classification="required")
            )
            rows = json.loads((raw / "fixture_runtime.json").read_text())["rows"]
        else:
            rows = live(plan, raw) if not plan["failures"] else []
            progress("validation_before")
            for spec in specs:
                receipt = execute(
                    spec["argv"], raw / "validation" / spec["name"], spec["deadline_s"], cwd=ROOT
                )
                validation.append(dict(receipt, **spec))
                if spec["name"] == "coverage_json" and (private / "coverage.json").is_file():
                    atomic_json(
                        raw / "coverage.json", json.loads((private / "coverage.json").read_text())
                    )
            progress("validation_after")
        atomic_json(raw / "plan.json", plan)
        atomic_json(raw / "validation.json", validation)
        value = artifact(plan, rows, raw, validation, fixture=bool(args.fixture_input))
        if not args.fixture_input:
            candidate = raw / "terminal_candidate.json"
            atomic_json(candidate, value)
            for name, script, extra in [
                ("adversarial", "adversarial_verify.py", []),
                ("strict_rows", "verdict_row_consistency_lint.py", ["--strict"]),
            ]:
                receipt = execute(
                    [
                        str(ROOT / ".venv/bin/python"),
                        str(ROOT / "scripts" / script),
                        *extra,
                        str(candidate),
                    ],
                    raw / "validation" / name,
                    120,
                )
                validation.append(dict(receipt, name=name, classification="required"))
            value = artifact(plan, rows, raw, validation, fixture=False)
            if not all(r["passed"] for r in validation if r.get("classification") == "required"):
                value.update(
                    diagnostic_ready_score=0,
                    process_isolation_ready_score=0,
                    verdict_class="disqualified",
                    honest_verdict="complete_disqualified_owned_validation",
                )
        progress("publication_before")
        publication = publish_primary(output, value, check_candidate)
        atomic_json(raw / "terminal_validation.json", publication)
        if not replay(output):
            raise ValueError("terminal_cold_replay")
        progress("publication_after", sum(r["status"] == "completed" for r in rows), 0)
    return 0
