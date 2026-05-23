"""Exp 2946 AUPRC-gated continuation for SOTA code generation.

This module is deliberately a thin gate around the Exp 2910 code-generation
protocol. Exp 2940 decides whether a pass-rate continuation is scientifically
allowed: retain expands the same live protocol to 50 total tasks, narrow uses a
20-task limited continuation, and retract produces failure-mode analysis
without a pass-rate claim.

Spec: REQ-CODE-2946, SCENARIO-CODE-2946.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot.eval import sota_code_generation_corrigendum as exp2910


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
OUTPUT_FILENAME = "experiment_2946_sota_code_generation_continuation_v1.json"
NESTED_EXP2910_FILENAME = "experiment_2946_nested_exp2910_protocol.json"
EXP2940_REL_PATH = Path("results/experiment_2940_verifier_ensemble_auprc_code_corpora_v1.json")
EXP2910_REL_PATH = Path("results/experiment_2910_sota_code_generation_corrigendum_v2.json")
RAW_RESPONSE_REL_DIR = Path("results/raw/experiment_2946_sota_code_generation_continuation_v1")
ARTIFACT = "experiment_2946_sota_code_generation_continuation_v1"
SCHEMA = "carnot.sota_code_generation_continuation.v1"
INFERENCE_SUBSTRATE = "live_llm_inference"
DEFAULT_RANDOM_SEED = exp2910.DEFAULT_RANDOM_SEED
DEFAULT_K_CANDIDATES_PER_TASK = exp2910.DEFAULT_K_CANDIDATES_PER_TASK
DEFAULT_RETAIN_N_TASKS = 50
DEFAULT_NARROW_N_TASKS = 20

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "exp2940_recommendation_used",
    "protocol_executed",
    "pass_at_1",
    "pass_at_k",
    "failure_mode_analysis",
    "random_seeds_used",
    "reproducibility_checksum",
    "duration_s",
)


@dataclass(frozen=True)
class ContinuationPlan:
    """Concrete continuation choice derived from Exp 2940's AUPRC verdict."""

    recommendation: str
    protocol_executed: str
    n_tasks_total: int
    n_tasks_per_corpus: int
    k_candidates_per_task: int
    limitation_framing: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "recommendation": self.recommendation,
            "protocol_executed": self.protocol_executed,
            "n_tasks_total": self.n_tasks_total,
            "n_tasks_per_corpus": self.n_tasks_per_corpus,
            "k_candidates_per_task": self.k_candidates_per_task,
            "limitation_framing": self.limitation_framing,
        }


@dataclass(frozen=True)
class ContinuationConfig:
    """Runtime knobs for the Exp 2946 continuation artifact."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    raw_response_dir: Path | None = None
    exp2940_path: Path = EXP2940_REL_PATH
    exp2910_path: Path = EXP2910_REL_PATH
    retain_n_tasks: int = DEFAULT_RETAIN_N_TASKS
    narrow_n_tasks: int = DEFAULT_NARROW_N_TASKS
    k_candidates_per_task: int = DEFAULT_K_CANDIDATES_PER_TASK
    max_tokens: int = exp2910.DEFAULT_MAX_TOKENS
    temperature: float = exp2910.DEFAULT_TEMPERATURE
    random_seed: int = DEFAULT_RANDOM_SEED
    sandbox_timeout_s: float = exp2910.DEFAULT_SANDBOX_TIMEOUT_S
    duration_floor_s: float = exp2910.DEFAULT_DURATION_FLOOR_S
    tests_run: Sequence[str] = field(default_factory=tuple)
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME

    def raw_dir(self) -> Path:
        return self.raw_response_dir or self.repo_root / RAW_RESPONSE_REL_DIR

    def nested_protocol_path(self) -> Path:
        return self.repo_root / "results" / NESTED_EXP2910_FILENAME


ProtocolRunner = Callable[[ContinuationConfig, ContinuationPlan], dict[str, Any]]
CudaProbe = Callable[[], dict[str, Any]]


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_recommendation(payload: dict[str, Any]) -> str:
    raw: Any = payload.get("paper_v6_recommendation")
    if isinstance(raw, dict):
        raw = raw.get("value")
    value = str(raw or "").strip().lower()
    for recommendation in ("retain", "narrow", "retract"):
        if value == recommendation or value.startswith(f"{recommendation}:"):
            return recommendation
    return "unknown"


def continuation_plan(config: ContinuationConfig, recommendation: str) -> ContinuationPlan:
    if recommendation == "retain":
        return _plan(
            recommendation="retain",
            n_tasks_total=config.retain_n_tasks,
            protocol_executed=f"exp2910_protocol_n_tasks_{config.retain_n_tasks}_k8",
            k_candidates_per_task=config.k_candidates_per_task,
            limitation_framing=None,
        )
    if recommendation == "narrow":
        return _plan(
            recommendation="narrow",
            n_tasks_total=config.narrow_n_tasks,
            protocol_executed=(
                f"exp2910_protocol_n_tasks_{config.narrow_n_tasks}_k8_limitation_framing"
            ),
            k_candidates_per_task=config.k_candidates_per_task,
            limitation_framing=(
                "Limitation: Exp 2940 did not justify the full 50-task continuation, "
                "so this artifact reports only a bounded 20-task pass-rate slice."
            ),
        )
    return _plan(
        recommendation=recommendation,
        n_tasks_total=0,
        protocol_executed="failure_mode_analysis_no_pass_rate_claim",
        k_candidates_per_task=config.k_candidates_per_task,
        limitation_framing=None,
    )


def _plan(
    *,
    recommendation: str,
    n_tasks_total: int,
    protocol_executed: str,
    k_candidates_per_task: int,
    limitation_framing: str | None,
) -> ContinuationPlan:
    n_corpora = len(exp2910.CODE_CORPORA)
    n_tasks_per_corpus = n_tasks_total // n_corpora if n_tasks_total else 0
    return ContinuationPlan(
        recommendation=recommendation,
        protocol_executed=protocol_executed,
        n_tasks_total=n_tasks_total,
        n_tasks_per_corpus=n_tasks_per_corpus,
        k_candidates_per_task=k_candidates_per_task,
        limitation_framing=limitation_framing,
    )


def default_cuda_probe() -> dict[str, Any]:  # pragma: no cover - exercised by the live run.
    try:
        import torch
    except Exception as exc:
        return {"available": False, "device_count": 0, "error": repr(exc)}
    available = bool(torch.cuda.is_available())
    return {
        "available": available,
        "device_count": int(torch.cuda.device_count()) if available else 0,
        "device_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
        if available
        else [],
    }


def run_exp2910_protocol(
    config: ContinuationConfig,
    plan: ContinuationPlan,
) -> dict[str, Any]:
    """Delegate the live generation work to the existing Exp 2910 runner."""

    nested_config = exp2910.ExperimentConfig(
        repo_root=config.repo_root,
        output_path=config.nested_protocol_path(),
        raw_response_dir=config.raw_dir(),
        n_tasks_per_corpus=plan.n_tasks_per_corpus,
        k_candidates_per_task=plan.k_candidates_per_task,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        random_seed=config.random_seed,
        sandbox_timeout_s=config.sandbox_timeout_s,
        duration_floor_s=config.duration_floor_s,
        tests_run=config.tests_run,
        started_at=config.started_at,
        clock=config.clock,
    )
    return exp2910.run_experiment(nested_config)


def build_artifact(
    config: ContinuationConfig | None = None,
    *,
    protocol_runner: ProtocolRunner = run_exp2910_protocol,
    cuda_probe: CudaProbe = default_cuda_probe,
) -> dict[str, Any]:
    """Build the Exp 2946 artifact from Exp 2940's recommendation."""

    config = config or ContinuationConfig()
    started = config.start_time()
    exp2940_path = _repo_path(config.repo_root, config.exp2940_path)
    exp2940 = read_json_object(exp2940_path)
    exp2940_sha = sha256_file(exp2940_path)
    if not exp2940_path.is_file() or not exp2940:
        return _blocked_artifact(
            config=config,
            started=started,
            verdict="blocked_exp2940_artifact_missing",
            recommendation="missing",
            protocol_executed="blocked_preconditions",
            exp2940_sha=exp2940_sha,
            cuda_status=None,
        )

    recommendation = normalize_recommendation(exp2940)
    if recommendation == "unknown":
        return _blocked_artifact(
            config=config,
            started=started,
            verdict="blocked_unknown_exp2940_recommendation",
            recommendation=recommendation,
            protocol_executed="blocked_preconditions",
            exp2940_sha=exp2940_sha,
            cuda_status=None,
        )

    if recommendation == "retract":
        analysis = failure_mode_analysis(config, exp2940)
        duration_s = _elapsed(config, started)
        return _final_artifact(
            config=config,
            honest_verdict=(
                "complete: exp2940 recommended retract; failure-mode analysis "
                "executed without a pass-rate claim"
            ),
            recommendation=recommendation,
            protocol_executed="failure_mode_analysis_no_pass_rate_claim",
            pass_at_1=None,
            pass_at_k=None,
            failure_analysis=analysis,
            random_seeds_used=[],
            duration_s=duration_s,
            exp2940_sha=exp2940_sha,
            cuda_status=None,
            protocol_plan=None,
            protocol_artifact_sha=None,
            limitation_framing=None,
        )

    cuda_status = cuda_probe()
    if not cuda_status.get("available"):
        return _blocked_artifact(
            config=config,
            started=started,
            verdict="blocked_cuda_unavailable",
            recommendation=recommendation,
            protocol_executed="blocked_preconditions",
            exp2940_sha=exp2940_sha,
            cuda_status=cuda_status,
        )

    plan = continuation_plan(config, recommendation)
    protocol_artifact = protocol_runner(config, plan)
    ready = protocol_artifact.get("codegen_corrigendum_ready") is True
    random_seeds_used = _int_list(protocol_artifact.get("random_seeds_used"))
    protocol_duration = _number_or_none(protocol_artifact.get("duration_s"))
    duration_s = protocol_duration if protocol_duration is not None else _elapsed(config, started)
    pass_at_1 = _number_or_none(protocol_artifact.get("aggregate_pass_at_1")) if ready else None
    pass_at_k = _number_or_none(protocol_artifact.get("aggregate_pass_at_k")) if ready else None
    if ready:
        honest_verdict = _complete_verdict(plan, pass_at_1, pass_at_k)
    else:
        honest_verdict = str(protocol_artifact.get("honest_verdict") or "blocked_exp2910_protocol")

    return _final_artifact(
        config=config,
        honest_verdict=honest_verdict,
        recommendation=recommendation,
        protocol_executed=plan.protocol_executed,
        pass_at_1=pass_at_1,
        pass_at_k=pass_at_k,
        failure_analysis=None,
        random_seeds_used=random_seeds_used,
        duration_s=duration_s,
        exp2940_sha=exp2940_sha,
        cuda_status=cuda_status,
        protocol_plan=plan.as_dict(),
        protocol_artifact_sha=sha256_file(config.nested_protocol_path()),
        limitation_framing=plan.limitation_framing,
    )


def write_artifact(
    config: ContinuationConfig | None = None,
    *,
    protocol_runner: ProtocolRunner = run_exp2910_protocol,
    cuda_probe: CudaProbe = default_cuda_probe,
) -> dict[str, Any]:
    config = config or ContinuationConfig()
    artifact = build_artifact(config, protocol_runner=protocol_runner, cuda_probe=cuda_probe)
    out_path = config.artifact_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def failure_mode_analysis(config: ContinuationConfig, exp2940: dict[str, Any]) -> dict[str, Any]:
    exp2910_path = _repo_path(config.repo_root, config.exp2910_path)
    exp2910_payload = read_json_object(exp2910_path)
    candidates = exp2910_payload.get("candidate_results")
    rows = candidates if isinstance(candidates, list) else []
    counts = Counter(str(row.get("row_status") or "unknown") for row in rows if isinstance(row, dict))
    return {
        "reason": "exp2940_recommendation_retract",
        "pass_rate_claim_made": False,
        "code_corpus_auprc": exp2940.get("code_corpus_auprc"),
        "max_f1_operating_point": exp2940.get("max_f1_operating_point"),
        "source_exp2910_candidate_count": len(rows),
        "candidate_failure_counts": dict(sorted(counts.items())),
        "source_exp2910_sha256": sha256_file(exp2910_path),
    }


def _blocked_artifact(
    *,
    config: ContinuationConfig,
    started: float,
    verdict: str,
    recommendation: str,
    protocol_executed: str,
    exp2940_sha: str | None,
    cuda_status: dict[str, Any] | None,
) -> dict[str, Any]:
    duration_s = _elapsed(config, started)
    return _final_artifact(
        config=config,
        honest_verdict=verdict,
        recommendation=recommendation,
        protocol_executed=protocol_executed,
        pass_at_1=None,
        pass_at_k=None,
        failure_analysis=None,
        random_seeds_used=[],
        duration_s=duration_s,
        exp2940_sha=exp2940_sha,
        cuda_status=cuda_status,
        protocol_plan=None,
        protocol_artifact_sha=None,
        limitation_framing=None,
    )


def _final_artifact(
    *,
    config: ContinuationConfig,
    honest_verdict: str,
    recommendation: str,
    protocol_executed: str,
    pass_at_1: float | None,
    pass_at_k: float | None,
    failure_analysis: dict[str, Any] | None,
    random_seeds_used: list[int],
    duration_s: float,
    exp2940_sha: str | None,
    cuda_status: dict[str, Any] | None,
    protocol_plan: dict[str, Any] | None,
    protocol_artifact_sha: str | None,
    limitation_framing: str | None,
) -> dict[str, Any]:
    checksum = _reproducibility_checksum(
        recommendation=recommendation,
        protocol_executed=protocol_executed,
        pass_at_1=pass_at_1,
        pass_at_k=pass_at_k,
        failure_analysis=failure_analysis,
        random_seeds_used=random_seeds_used,
        exp2940_sha=exp2940_sha,
        protocol_artifact_sha=protocol_artifact_sha,
    )
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "exp2940_recommendation_used": recommendation,
        "protocol_executed": protocol_executed,
        "pass_at_1": pass_at_1,
        "pass_at_k": pass_at_k,
        "failure_mode_analysis": failure_analysis,
        "random_seeds_used": random_seeds_used,
        "reproducibility_checksum": checksum,
        "duration_s": float(duration_s),
        "exp2940_artifact_path": str(EXP2940_REL_PATH),
        "exp2940_sha256": exp2940_sha,
        "cuda_precondition": cuda_status,
        "protocol_plan": protocol_plan,
        "protocol_artifact_path": str(Path("results") / NESTED_EXP2910_FILENAME)
        if protocol_plan
        else None,
        "protocol_artifact_sha256": protocol_artifact_sha,
        "raw_response_dir": str(RAW_RESPONSE_REL_DIR),
        "limitation_framing": limitation_framing,
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "run_date": RUN_DATE,
    }


def _complete_verdict(
    plan: ContinuationPlan,
    pass_at_1: float | None,
    pass_at_k: float | None,
) -> str:
    if plan.recommendation == "narrow":
        return (
            "complete: narrow continuation executed under limitation framing with "
            f"n_tasks={plan.n_tasks_total}, pass@1={pass_at_1:.4f}, pass@k={pass_at_k:.4f}"
        )
    return (
        "complete: retain continuation executed with "
        f"n_tasks={plan.n_tasks_total}, pass@1={pass_at_1:.4f}, pass@k={pass_at_k:.4f}"
    )


def _reproducibility_checksum(
    *,
    recommendation: str,
    protocol_executed: str,
    pass_at_1: float | None,
    pass_at_k: float | None,
    failure_analysis: dict[str, Any] | None,
    random_seeds_used: list[int],
    exp2940_sha: str | None,
    protocol_artifact_sha: str | None,
) -> str:
    return stable_payload_sha256(
        {
            "exp2940_sha256": exp2940_sha,
            "failure_mode_analysis": failure_analysis,
            "pass_at_1": pass_at_1,
            "pass_at_k": pass_at_k,
            "protocol_artifact_sha256": protocol_artifact_sha,
            "protocol_executed": protocol_executed,
            "random_seeds_used": random_seeds_used,
            "recommendation": recommendation,
        }
    )


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _elapsed(config: ContinuationConfig, started: float) -> float:
    return round(max(0.0, config.clock() - started), 6)


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value if isinstance(item, int) and not isinstance(item, bool)]


def main() -> int:  # pragma: no cover
    artifact = write_artifact(
        ContinuationConfig(
            tests_run=(
                ".venv/bin/pytest tests/python/test_experiment_2946_sota_code_generation_continuation.py -q",
                ".venv/bin/pytest tests/python -q",
            )
        )
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARTIFACT",
    "ContinuationConfig",
    "ContinuationPlan",
    "DEFAULT_K_CANDIDATES_PER_TASK",
    "DEFAULT_RANDOM_SEED",
    "EXP2940_REL_PATH",
    "INFERENCE_SUBSTRATE",
    "NESTED_EXP2910_FILENAME",
    "OUTPUT_FILENAME",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_artifact",
    "continuation_plan",
    "default_cuda_probe",
    "exp2910",
    "failure_mode_analysis",
    "normalize_recommendation",
    "run_exp2910_protocol",
    "write_artifact",
]
