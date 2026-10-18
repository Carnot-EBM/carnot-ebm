"""Exp 4319 off-ARC execution-verifier transfer accumulation.

Spec refs: REQ-VERIFY-4319, SCENARIO-VERIFY-4319.

This module replays checked-in MBPP/EvalPlus candidate evidence from the same
local Gemma 4 12B GGUF generator family and scores the model-free visible
demo-fit execution selector against vote@1 on hidden tests.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT = RESULTS_DIR / "experiment_4319_off_arc_execution_verifier_transfer_accumulate.json"
GGUF_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / (
    "models--unsloth--gemma-4-12B-it-GGUF"
)
MBPP_MANIFEST = REPO_ROOT / "data" / "eval_manifests" / "mbpp_20260522.jsonl"

RANDOM_SEED = 4319
BOOTSTRAP_RESAMPLES = 2000
SPEC_REFS = ["REQ-VERIFY-4319", "SCENARIO-VERIFY-4319"]
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

DEFAULT_PRIOR_PATHS = [
    RESULTS_DIR / "experiment_4032_offarc_exec_verifier_transfer_raw.json",
    RESULTS_DIR / "experiment_4051_verifier_registry_and_gaps_hygiene.json",
    RESULTS_DIR / "experiment_4063_verifier_registry_and_gaps_hygiene.json",
]
DEFAULT_WINDOW_PATHS = [
    RESULTS_DIR / "experiment_4068_offarc_transfer_power_sync.json",
]

REQUIRED_FIELDS = [
    "honest_verdict",
    "off_arc_demofit_beats_vote",
    "off_arc_demofit_minus_vote_delta",
    "off_arc_delta_ci95",
    "accumulated_n",
    "accumulation_window_added",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An off-ARC transfer win, a powered null, and an "
        "honest blocked_generator_not_cached are all complete."
    ),
    "off_arc_demofit_beats_vote": (
        "BARE bool: true iff hidden-test demo-fit selection - vote > 0 and "
        "CI95 excludes 0 at the accumulated n."
    ),
    "off_arc_demofit_minus_vote_delta": (
        "BARE float: hidden-test demo-fit accuracy - vote@1."
    ),
    "off_arc_delta_ci95": (
        "Bootstrap CI95 with at least 2000 resamples for the paired hidden-test "
        "demo-fit-minus-vote delta."
    ),
    "accumulated_n": (
        "BARE int: total accumulated task count across corpus+model-keyed "
        "sources."
    ),
    "accumulation_window_added": (
        "BARE int: tasks added by the current accumulation window."
    ),
    "verifier_is_oracle": (
        "BARE bool=true: the demo-fit verifier is the executable oracle; this "
        "is the cheap execution layer, not an oracle-distinct moat."
    ),
    "preconditions_checked": (
        "Records GGUF cache, corpus, prior checkpoint, sandbox, and TRM "
        "stand-down preconditions."
    ),
    "random_seed": "Determinism precondition for bootstrap and replay ordering.",
    "reproducibility_checksum": (
        "Hash of source artifact checksums, task outcomes, model specs, and seed."
    ),
    "model_specs": (
        "The Gemma 4 12B GGUF, visible demo-fit primitive, sandbox, corpus, "
        "source artifacts, and accumulated n."
    ),
}


@dataclass(frozen=True)
class TaskOutcome:
    """One accumulated hidden-test paired outcome."""

    unique_id: str
    task_id: str
    corpus: str
    source_path: str
    vote_pass: bool
    demofit_pass: bool
    oracle_pass: bool
    n_candidates: int
    n_visible_tests: int
    n_hidden_tests: int

    @property
    def paired_delta(self) -> int:
        return int(self.demofit_pass) - int(self.vote_pass)


def resolve_gemma_gguf(cache_dir: Path = GGUF_CACHE) -> Path | None:
    """Resolve the cached Gemma 4 12B GGUF file, following HF snapshot symlinks."""
    for path in sorted(cache_dir.glob("snapshots/**/*.gguf")):
        if path.exists():
            return path
    for path in sorted(cache_dir.rglob("*.gguf")):
        if path.exists():
            return path
    return None


def check_preconditions(
    *,
    prior_paths: list[Path] | None = None,
    window_paths: list[Path] | None = None,
    cache_dir: Path = GGUF_CACHE,
    mbpp_manifest: Path = MBPP_MANIFEST,
) -> list[dict[str, Any]]:
    """Check resources before any inference or accumulation extension."""
    prior = prior_paths if prior_paths is not None else list(DEFAULT_PRIOR_PATHS)
    window = window_paths if window_paths is not None else list(DEFAULT_WINDOW_PATHS)
    all_sources = [*prior, *window]
    model_path = resolve_gemma_gguf(cache_dir)
    source_status = [_source_readable(path) for path in all_sources]
    evalplus_rows = sum(
        status.get("evalplus_rows", 0)
        for status in source_status
        if isinstance(status.get("evalplus_rows"), int)
    )
    mbpp_ok, mbpp_detail = _mbpp_manifest_status(mbpp_manifest)
    evalplus_importable = importlib.util.find_spec("evalplus") is not None
    corpus_ok = mbpp_ok and (evalplus_importable or evalplus_rows > 0)
    return [
        {
            "resource": "generator_gguf_cached",
            "available": model_path is not None,
            "path": str(model_path) if model_path else None,
            "repo_id": "unsloth/gemma-4-12B-it-GGUF",
        },
        {
            "resource": "llama_cpp_importable",
            "available": importlib.util.find_spec("llama_cpp") is not None,
        },
        {
            "resource": "mbpp_evalplus_corpus_loadable",
            "available": corpus_ok,
            "detail": (
                f"{mbpp_detail}; evalplus_package_importable={evalplus_importable}; "
                f"evalplus_artifact_rows={evalplus_rows}"
            ),
        },
        {
            "resource": "prior_accumulation_checkpoint_readable",
            "available": all(bool(row["available"]) for row in source_status),
            "sources": source_status,
        },
        {
            "resource": "restricted_exec_sandbox_available",
            "available": _sandbox_importable(),
        },
        {
            "resource": "trm_training_stood_down",
            "available": _trm_training_stood_down(),
            "detail": "no active process command references results/trm_runs",
        },
    ]


def blocker_from_preconditions(preconditions: list[dict[str, Any]]) -> str | None:
    """Return the first terminal blocker name, if any."""
    for row in preconditions:
        if bool(row.get("available")):
            continue
        resource = str(row.get("resource", "resource"))
        if resource == "generator_gguf_cached":
            return "blocked_generator_not_cached"
        return f"blocked_{resource}"
    return None


def load_task_outcomes(paths: list[Path]) -> tuple[list[TaskOutcome], list[dict[str, Any]]]:
    """Load per-task vote/demo-fit outcomes from checked-in source artifacts."""
    outcomes: list[TaskOutcome] = []
    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        payload = _read_json(path)
        rows = payload.get("per_task") if isinstance(payload, dict) else None
        used = 0
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                outcome = _outcome_from_row(row, path)
                if outcome is None or outcome.unique_id in seen:
                    continue
                seen.add(outcome.unique_id)
                outcomes.append(outcome)
                used += 1
        summaries.append(
            {
                "path": str(path),
                "sha256": _file_sha256(path) if path.exists() else None,
                "experiment": payload.get("experiment") if isinstance(payload, dict) else None,
                "corpus": payload.get("corpus") if isinstance(payload, dict) else None,
                "tasks_used": used,
                "declared_n": _declared_n(payload),
            }
        )
    return outcomes, summaries


def bootstrap_ci95(
    values: list[int],
    *,
    seed: int = RANDOM_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[float]:
    """Bootstrap a paired mean-delta CI on the accuracy-fraction scale."""
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(draw) / len(draw))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return [round(lo, 6), round(hi, 6)]


def build_accumulation_artifact(
    *,
    prior_paths: list[Path],
    window_paths: list[Path],
    preconditions_checked: list[dict[str, Any]],
    model_specs: dict[str, Any],
    seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    started_s: float | None = None,
    ended_s: float | None = None,
) -> dict[str, Any]:
    """Build the terminal Exp 4319 accumulation artifact."""
    prior_outcomes, prior_summaries = load_task_outcomes(prior_paths)
    prior_ids = {outcome.unique_id for outcome in prior_outcomes}
    window_outcomes, window_summaries = load_task_outcomes(window_paths)
    added_window = [outcome for outcome in window_outcomes if outcome.unique_id not in prior_ids]
    outcomes = [*prior_outcomes, *added_window]

    accumulated_n = len(outcomes)
    vote_hits = sum(int(row.vote_pass) for row in outcomes)
    demofit_hits = sum(int(row.demofit_pass) for row in outcomes)
    oracle_hits = sum(int(row.oracle_pass) for row in outcomes)
    vote_rate = round(vote_hits / accumulated_n, 6) if accumulated_n else 0.0
    demofit_rate = round(demofit_hits / accumulated_n, 6) if accumulated_n else 0.0
    oracle_rate = round(oracle_hits / accumulated_n, 6) if accumulated_n else 0.0
    delta = round(demofit_rate - vote_rate, 6)
    paired = [row.paired_delta for row in outcomes]
    ci95 = bootstrap_ci95(paired, seed=seed, resamples=bootstrap_resamples)
    beats_vote = bool(delta > 0.0 and ci95[0] > 0.0)
    source_summaries = [*prior_summaries, *window_summaries]
    expanded_specs = {
        **model_specs,
        "generator_hf_id": model_specs.get("generator_hf_id", "unsloth/gemma-4-12B-it-GGUF"),
        "induction_primitive": (
            "GAP-4 visible-test demo-fit selector over generated Python programs"
        ),
        "restricted_exec_sandbox": "carnot.verify.sandbox.sandboxed_exec_function",
        "corpus": "MBPP plus EvalPlus hidden-test replay",
        "accumulated_n": accumulated_n,
        "source_artifacts": [row["path"] for row in source_summaries],
    }
    artifact: dict[str, Any] = {
        "experiment": "experiment_4319_off_arc_execution_verifier_transfer_accumulate",
        "schema": "carnot.experiment_4319.off_arc_execution_verifier_transfer_accumulate.v1",
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": _verdict(beats_vote=beats_vote, delta=delta, ci95=ci95),
        "off_arc_demofit_beats_vote": beats_vote,
        "off_arc_demofit_minus_vote_delta": delta,
        "off_arc_delta_ci95": ci95,
        "hidden_test_vote_at_1": vote_rate,
        "hidden_test_demofit_accuracy": demofit_rate,
        "oracle_hidden_passrate": oracle_rate,
        "accumulated_n": accumulated_n,
        "accumulation_window_added": len(added_window),
        "verifier_is_oracle": True,
        "preconditions_checked": preconditions_checked,
        "random_seed": seed,
        "bootstrap_resamples": bootstrap_resamples,
        "reproducibility_checksum": "",
        "model_specs": expanded_specs,
        "source_artifacts": source_summaries,
        "per_task": [_outcome_record(row) for row in outcomes],
        "missing_verifier_gaps": _missing_gap_record(outcomes, ci95),
        "gap_ledger_update_required": bool(delta <= 0.0 or ci95[0] <= 0.0),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _duration(started_s, ended_s),
        "field_principles": dict(FIELD_PRINCIPLES),
        "adversarial_verify": {"status": "pending"},
    }
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def blocked_artifact(
    verdict: str,
    *,
    preconditions_checked: list[dict[str, Any]],
    seed: int = RANDOM_SEED,
    started_s: float | None = None,
    ended_s: float | None = None,
) -> dict[str, Any]:
    """Build a terminal blocked artifact without inference or scoring."""
    artifact: dict[str, Any] = {
        "experiment": "experiment_4319_off_arc_execution_verifier_transfer_accumulate",
        "schema": "blocked.carnot.experiment_4319.off_arc_execution_verifier_transfer_accumulate.v1",
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": verdict,
        "off_arc_demofit_beats_vote": False,
        "off_arc_demofit_minus_vote_delta": 0.0,
        "off_arc_delta_ci95": [0.0, 0.0],
        "hidden_test_vote_at_1": 0.0,
        "hidden_test_demofit_accuracy": 0.0,
        "oracle_hidden_passrate": 0.0,
        "accumulated_n": 0,
        "accumulation_window_added": 0,
        "verifier_is_oracle": True,
        "preconditions_checked": preconditions_checked,
        "random_seed": seed,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "reproducibility_checksum": "",
        "model_specs": {
            "generator_hf_id": "unsloth/gemma-4-12B-it-GGUF",
            "generator_cache_status": verdict,
            "induction_primitive": "not_run_blocked_precondition",
            "restricted_exec_sandbox": "not_run_blocked_precondition",
            "corpus": "not_scored",
            "accumulated_n": 0,
        },
        "source_artifacts": [],
        "per_task": [],
        "missing_verifier_gaps": [],
        "gap_ledger_update_required": False,
        "inference_substrate": "precondition_check_no_inference",
        "duration_s": _duration(started_s, ended_s),
        "field_principles": dict(FIELD_PRINCIPLES),
        "adversarial_verify": {"status": "not_run_blocked_precondition"},
    }
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate the required Exp 4319 gate fields and bare types."""
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("success:")
        or verdict.startswith("complete:")
        or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if type(artifact["off_arc_demofit_beats_vote"]) is not bool:
        raise ValueError("off_arc_demofit_beats_vote must be a bare bool")
    if type(artifact["verifier_is_oracle"]) is not bool or artifact["verifier_is_oracle"] is not True:
        raise ValueError("verifier_is_oracle must be bare bool true")
    for field in ("accumulated_n", "accumulation_window_added", "random_seed"):
        if type(artifact[field]) is not int:
            raise ValueError(f"{field} must be a bare int")
    for field in (
        "off_arc_demofit_minus_vote_delta",
        "hidden_test_vote_at_1",
        "hidden_test_demofit_accuracy",
    ):
        if not isinstance(artifact[field], float) or isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare float")
    ci = artifact["off_arc_delta_ci95"]
    if not (
        isinstance(ci, list)
        and len(ci) == 2
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in ci)
    ):
        raise ValueError("off_arc_delta_ci95 must be a numeric two-element list")
    if int(artifact.get("bootstrap_resamples", 0)) < BOOTSTRAP_RESAMPLES:
        raise ValueError("bootstrap_resamples must be at least 2000")
    if not isinstance(artifact["preconditions_checked"], list):
        raise ValueError("preconditions_checked must be a list")
    if not isinstance(artifact["model_specs"], dict):
        raise ValueError("model_specs must be an object")
    if not isinstance(artifact["reproducibility_checksum"], str) or not artifact[
        "reproducibility_checksum"
    ]:
        raise ValueError("reproducibility_checksum must be a non-empty string")


def run(
    *,
    output_path: Path = OUTPUT,
    prior_paths: list[Path] | None = None,
    window_paths: list[Path] | None = None,
    cache_dir: Path = GGUF_CACHE,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Run Exp 4319 and write the terminal JSON artifact."""
    started = time.time()
    prior = prior_paths if prior_paths is not None else list(DEFAULT_PRIOR_PATHS)
    window = window_paths if window_paths is not None else list(DEFAULT_WINDOW_PATHS)
    preconditions = check_preconditions(prior_paths=prior, window_paths=window, cache_dir=cache_dir)
    blocker = blocker_from_preconditions(preconditions)
    if blocker:
        artifact = blocked_artifact(
            blocker,
            preconditions_checked=preconditions,
            seed=seed,
            started_s=started,
            ended_s=time.time(),
        )
        _write_json(output_path, artifact)
        return artifact

    model_path = resolve_gemma_gguf(cache_dir)
    artifact = build_accumulation_artifact(
        prior_paths=prior,
        window_paths=window,
        preconditions_checked=preconditions,
        model_specs={
            "generator_hf_id": "unsloth/gemma-4-12B-it-GGUF",
            "generator_gguf_path": str(model_path) if model_path else None,
            "generator_loading_rule": "load via .gguf path; do not use AutoTokenizer",
            "sandbox": "carnot.verify.sandbox.sandboxed_exec_function",
        },
        seed=seed,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        started_s=started,
        ended_s=time.time(),
    )
    _write_json(output_path, artifact)
    artifact["adversarial_verify"] = _adversarial_verify_summary(output_path)
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    validate_artifact(artifact)
    _write_json(output_path, artifact)
    print(
        "-> "
        f"{artifact['honest_verdict']} accumulated_n={artifact['accumulated_n']} "
        f"window_added={artifact['accumulation_window_added']} "
        f"vote={artifact['hidden_test_vote_at_1']} "
        f"demofit={artifact['hidden_test_demofit_accuracy']} "
        f"delta={artifact['off_arc_demofit_minus_vote_delta']} "
        f"ci95={artifact['off_arc_delta_ci95']} "
        f"beats={artifact['off_arc_demofit_beats_vote']}",
        flush=True,
    )
    return artifact


def _outcome_from_row(row: dict[str, Any], source_path: Path) -> TaskOutcome | None:
    task_id = row.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        return None
    if "armA_vote_pass1" not in row or "armB_demofit_pass1" not in row:
        return None
    corpus = str(row.get("corpus") or _corpus_from_task_id(task_id))
    return TaskOutcome(
        unique_id=f"{corpus}:{task_id}",
        task_id=task_id,
        corpus=corpus,
        source_path=str(source_path),
        vote_pass=bool(row.get("armA_vote_pass1")),
        demofit_pass=bool(row.get("armB_demofit_pass1")),
        oracle_pass=bool(row.get("oracle_hidden_pass")),
        n_candidates=int(row.get("n_candidates") or 0),
        n_visible_tests=int(row.get("n_visible_tests") or 0),
        n_hidden_tests=int(row.get("n_hidden_tests") or 0),
    )


def _outcome_record(row: TaskOutcome) -> dict[str, Any]:
    return {
        "task_id": row.task_id,
        "corpus": row.corpus,
        "source_path": row.source_path,
        "armA_vote_pass1": row.vote_pass,
        "armB_demofit_pass1": row.demofit_pass,
        "paired_delta": row.paired_delta,
        "oracle_hidden_pass": row.oracle_pass,
        "n_candidates": row.n_candidates,
        "n_visible_tests": row.n_visible_tests,
        "n_hidden_tests": row.n_hidden_tests,
    }


def _missing_gap_record(outcomes: list[TaskOutcome], ci95: list[float]) -> list[dict[str, Any]]:
    if ci95[0] > 0.0:
        return []
    residual = [
        row.task_id
        for row in outcomes
        if row.oracle_pass and not row.demofit_pass
    ]
    return [
        {
            "gap_id": "GAP-CODE-EXEC-DEMOFIT",
            "status": "open_accumulating" if ci95[1] >= 0.0 else "powered_negative",
            "failure_mode": "visible/demo-fit code tests do not yet decide hidden semantics",
            "missing_discriminator": (
                "code_demo_fit_visible_tests_do_not_discriminate_hidden_semantics"
            ),
            "residual_oracle_hit_demofit_miss_count": len(residual),
            "example_task_ids": residual[:20],
        }
    ]


def _verdict(*, beats_vote: bool, delta: float, ci95: list[float]) -> str:
    if beats_vote:
        return "success: off_arc_demofit_beats_vote_accumulated_ci_excludes_zero"
    if delta <= 0.0 and ci95[1] < 0.0:
        return "complete: off_arc_demofit_powered_negative_scope_boundary"
    return "complete: off_arc_demofit_accumulated_ci_includes_zero_gap_open"


def _reproducibility_checksum(artifact: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"reproducibility_checksum", "duration_s"}
    }
    encoded = json.dumps(payload, sort_keys=True, default=repr)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_readable(path: Path) -> dict[str, Any]:
    try:
        payload = _read_json(path)
    except Exception as exc:
        return {
            "path": str(path),
            "available": False,
            "detail": f"{type(exc).__name__}: {str(exc)[:160]}",
            "evalplus_rows": 0,
        }
    per_task = payload.get("per_task") if isinstance(payload, dict) else None
    evalplus_rows = 0
    if isinstance(per_task, list):
        evalplus_rows = sum(
            1
            for row in per_task
            if isinstance(row, dict)
            and "evalplus" in str(row.get("corpus") or payload.get("corpus", "")).lower()
        )
    return {
        "path": str(path),
        "available": True,
        "experiment": payload.get("experiment") if isinstance(payload, dict) else None,
        "per_task_rows": len(per_task) if isinstance(per_task, list) else 0,
        "evalplus_rows": evalplus_rows,
    }


def _mbpp_manifest_status(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"mbpp_manifest_missing={path}"
    try:
        n_rows = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)
    except Exception as exc:
        return False, f"mbpp_manifest_error={type(exc).__name__}: {exc}"
    return n_rows > 0, f"mbpp_manifest_rows={n_rows}"


def _sandbox_importable() -> bool:
    try:
        from carnot.verify import sandbox  # noqa: F401

        return True
    except Exception:
        return False


def _trm_training_stood_down() -> bool:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return False
    current_pid = os.getpid()
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_raw, _, command = stripped.partition(" ")
        try:
            pid = int(pid_raw)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        command_lower = command.lower()
        if "results/trm_runs" in command_lower:
            return False
        if "trm" in command_lower and "train" in command_lower:
            return False
    return True


def _declared_n(payload: dict[str, Any]) -> int | None:
    for key in ("accumulated_n_tasks", "accumulated_n", "n_tasks"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _corpus_from_task_id(task_id: str) -> str:
    if task_id.startswith("MBPP:") or task_id.lower().startswith("mbpp"):
        return "mbpp"
    if task_id.startswith("HumanEval/"):
        return "evalplus_humaneval"
    return "unknown"


def _duration(started_s: float | None, ended_s: float | None) -> float:
    if started_s is None or ended_s is None:
        return 0.0001
    return max(0.0001, round(ended_s - started_s, 6))


def _adversarial_verify_summary(path: Path) -> dict[str, Any]:
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        from adversarial_verify import verify_artifact

        report = verify_artifact(path)
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    flags = report.get("flags", []) if isinstance(report, dict) else []
    critical = [
        flag
        for flag in flags
        if isinstance(flag, dict) and flag.get("severity") == "critical"
    ]
    return {
        "status": "clean" if not flags else "flags_present",
        "critical_count": len(critical),
        "flag_count": len(flags),
        "flags": flags,
    }
