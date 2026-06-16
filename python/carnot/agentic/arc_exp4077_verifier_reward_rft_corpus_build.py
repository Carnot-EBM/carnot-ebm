"""Exp 4077 verifier-reward RFT corpus build gate.

Spec refs: REQ-LEARN-4077, SCENARIO-LEARN-4077,
SCENARIO-LEARN-4077-NMATCH.

The experiment is intentionally gated before any training. It measures whether
the existing ARC execution-verifier label, "demo-perfect program", is precise
enough to become an RFT correctness label. If the certification precision is
below the Phase-0 threshold, the runner writes a blocked artifact and does not
construct training corpora.
"""

from __future__ import annotations

import gzip
import importlib
import json
import os
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from carnot.agentic.arc_gap4_execution_verifier import (
    Gap4ExecutionVerifier,
    get_consistency_energy,
)
from carnot.verify.sandbox import sandboxed_exec_function


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_FILENAME = "experiment_4077_verifier_reward_rft_corpus_build.json"
DEFAULT_ARC1_POOL = Path("results/arc3_gap3_stage2_eval_pool.json.gz")
DEFAULT_ARC2_POOL = Path("results/arc3_gap4_arc2_eval_pool.json.gz")
DEFAULT_PROGRAM_CHECKPOINT = Path("results/experiment_4012_gap4_local_best_of_n.checkpoint.json")
INFERENCE_SUBSTRATE = "offline_arc_k8_execution_verifier_reward_lora_runner"
RANDOM_SEED = 4077
K_PROGRAMS_PER_TASK = 8
PRECISION_THRESHOLD = 0.85
RECALL_THRESHOLD = 0.20

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "certification_precision",
    "certification_recall",
    "n_rft_correct",
    "n_rft_ablation",
    "n_gold_sft",
    "n_heldout_tasks",
    "runner_ready",
    "trainer_smoke_passed",
    "preconditions_checked",
    "inference_substrate",
)
TERMINAL_PREFIXES = ("complete:", "success:", "blocked_")
_SANDBOX_NUMPY_UNAVAILABLE = False


@dataclass(frozen=True)
class PreconditionCheck:
    """One mandatory resource check recorded in the terminal artifact."""

    resource: str
    available: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArcTask:
    """ARC task row with public demos, test input, and candidate gold labels."""

    task_id: str
    pool_name: str
    demos: list[dict[str, Any]]
    test_input: Any
    candidates: list[dict[str, Any]]

    @property
    def gold_outputs(self) -> list[Any]:
        return [candidate.get("grid") for candidate in self.candidates if candidate.get("correct") is True]


@dataclass(frozen=True)
class CandidateProgram:
    """Generated transform program for one ARC task."""

    task_id: str
    program_id: str
    code: str
    source: str


@dataclass(frozen=True)
class ProgramEvaluation:
    """Post-hoc labels for one generated program."""

    task_id: str
    program_id: str
    code: str
    source: str
    split: str
    demo_perfect: bool
    test_gold: bool
    verifier_energy: float
    error: str


@dataclass(frozen=True)
class CertificationMetrics:
    """Phase-0 precision/recall for demo-perfect certification."""

    precision: float
    recall: float
    n_certified: int
    n_gold: int
    n_true_positive: int

    @property
    def gate_passed(self) -> bool:
        return self.precision >= PRECISION_THRESHOLD and self.recall >= RECALL_THRESHOLD


@dataclass(frozen=True)
class RftCorpora:
    """N-matched records for the three training arms."""

    rft_correct: list[dict[str, object]]
    rft_ablation: list[dict[str, object]]
    gold_sft: list[dict[str, object]]


def _resource_slug(text: str) -> str:
    return (
        text.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("__", "_")
    )


def check_hf_safetensors_model(
    model_id: str,
    *,
    cache_root: str | Path | None = None,
    trust_remote_code: bool = False,
) -> PreconditionCheck:
    """REQ-LEARN-4077: require cached HF safetensors, not GGUF-only weights."""

    root = Path(cache_root) if cache_root is not None else Path.home() / ".cache" / "huggingface" / "hub"
    resource = f"hf_safetensors_{_resource_slug(model_id)}"
    model_dir = root / f"models--{model_id.replace('/', '--')}"
    safetensors = sorted(model_dir.glob("snapshots/*/*.safetensors"))
    configs = sorted(model_dir.glob("snapshots/*/config.json"))
    ggufs = sorted(model_dir.glob("snapshots/*/*.gguf"))
    if not model_dir.exists():
        return PreconditionCheck(resource, False, f"missing cache directory {model_dir}")
    if not safetensors:
        suffix = "gguf_only" if ggufs else "no_safetensors"
        return PreconditionCheck(resource, False, f"{suffix}; LoRA requires trainable HF weights")
    if not configs:
        return PreconditionCheck(resource, False, "safetensors present but config.json missing")
    detail = f"{len(safetensors)} safetensors shard(s); trust_remote_code={trust_remote_code}"
    return PreconditionCheck(resource, True, detail)


def check_trainer_imports() -> PreconditionCheck:  # pragma: no cover
    """REQ-LEARN-4077: assert TRL and PEFT trainer imports before training."""

    try:
        import peft  # noqa: F401
        import trl  # noqa: F401
        from trl import GRPOTrainer, SFTTrainer  # noqa: F401

        return PreconditionCheck("trl_peft_trainers", True, "trl, peft, SFTTrainer, GRPOTrainer import")
    except Exception as exc:
        return PreconditionCheck("trl_peft_trainers", False, f"{type(exc).__name__}: {exc}")


def check_cuda_visible() -> PreconditionCheck:  # pragma: no cover
    """REQ-LEARN-4077: require CUDA before LoRA smoke training."""

    try:
        torch = importlib.import_module("torch")
        available = bool(torch.cuda.is_available())
        detail = "torch.cuda.is_available() is true" if available else "torch.cuda.is_available() is false"
        return PreconditionCheck("cuda_visible", available, detail)
    except Exception as exc:
        return PreconditionCheck("cuda_visible", False, f"{type(exc).__name__}: {exc}")


def _load_json_or_gzip(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def check_arc_pool(path: str | Path, resource: str) -> PreconditionCheck:
    """REQ-LEARN-4077: verify a cached ARC pool is readable and non-empty."""

    pool_path = Path(path)
    try:
        payload = _load_json_or_gzip(pool_path)
    except Exception as exc:
        return PreconditionCheck(resource, False, f"{type(exc).__name__}: {exc}")
    entries = payload.get("entries") if isinstance(payload, Mapping) else payload
    available = isinstance(entries, list) and len(entries) > 0
    detail = f"{len(entries) if isinstance(entries, list) else 0} entries at {pool_path}"
    return PreconditionCheck(resource, available, detail)


def check_preconditions(*, repo_root: str | Path = REPO_ROOT) -> list[PreconditionCheck]:  # pragma: no cover
    """REQ-LEARN-4077: collect every mandatory precondition before inference."""

    root = Path(repo_root)
    return [
        check_hf_safetensors_model("Qwen/Qwen3.5-0.8B"),
        check_hf_safetensors_model(
            "openbmb/MiniCPM5-1B",
            trust_remote_code=os.environ.get("CARNOT_TRUST_REMOTE_CODE", "") == "1",
        ),
        check_trainer_imports(),
        check_cuda_visible(),
        check_arc_pool(root / DEFAULT_ARC1_POOL, "arc1_pool"),
        check_arc_pool(root / DEFAULT_ARC2_POOL, "arc2_pool"),
    ]


def _first_missing(preconditions_checked: Sequence[PreconditionCheck]) -> PreconditionCheck | None:
    return next((check for check in preconditions_checked if not check.available), None)


def load_arc_pool(path: str | Path, *, pool_name: str) -> list[ArcTask]:
    """Load ARC task rows from a cached GAP-4 candidate pool."""

    payload = _load_json_or_gzip(Path(path))
    entries = payload.get("entries") if isinstance(payload, Mapping) else payload
    if not isinstance(entries, list):
        raise ValueError(f"{path} does not contain an entries list")
    tasks: list[ArcTask] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        task_id = str(entry.get("task") or entry.get("task_id") or "")
        if not task_id:
            continue
        tasks.append(
            ArcTask(
                task_id=task_id,
                pool_name=pool_name,
                demos=list(entry.get("demos") or []),
                test_input=entry.get("test_input"),
                candidates=list(entry.get("candidates") or []),
            )
        )
    if not tasks:
        raise ValueError(f"{path} contains no usable ARC tasks")
    return tasks


def load_program_checkpoint(path: str | Path, *, k_required: int = K_PROGRAMS_PER_TASK) -> dict[str, list[CandidateProgram]]:
    """Load cached k-sample transform programs from the local generator checkpoint."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = payload.get("tasks")
    if not isinstance(tasks, Mapping):
        raise ValueError(f"{path} does not contain a tasks mapping")
    programs: dict[str, list[CandidateProgram]] = {}
    for task_id, rows in tasks.items():
        if not isinstance(rows, list) or len(rows) < k_required:
            continue
        task_programs: list[CandidateProgram] = []
        for index, row in enumerate(rows[:k_required]):
            if not isinstance(row, Mapping) or not isinstance(row.get("code"), str):
                continue
            draw_index = row.get("draw_index", index)
            task_programs.append(
                CandidateProgram(
                    task_id=str(task_id),
                    program_id=f"{task_id}:draw{draw_index}",
                    code=str(row["code"]),
                    source=str(row.get("source") or "experiment_4012_local_k8"),
                )
            )
        if len(task_programs) >= k_required:
            programs[str(task_id)] = task_programs
    if not programs:
        raise ValueError(f"{path} contains no tasks with k>={k_required} programs")
    return programs


def load_default_task_splits(
    *,
    repo_root: str | Path = REPO_ROOT,
    k_required: int = K_PROGRAMS_PER_TASK,
) -> tuple[list[ArcTask], list[ArcTask], list[ArcTask]]:
    """Build disjoint precision, held-in train, and held-out eval ARC splits."""

    root = Path(repo_root)
    programs = load_program_checkpoint(root / DEFAULT_PROGRAM_CHECKPOINT, k_required=k_required)
    arc1_tasks = {task.task_id: task for task in load_arc_pool(root / DEFAULT_ARC1_POOL, pool_name="arc1")}
    task_ids = sorted(task_id for task_id in programs if task_id in arc1_tasks)
    if len(task_ids) < 3:
        raise ValueError("fewer than three ARC-1 k-program tasks are available")
    if len(task_ids) < 12:
        return ([arc1_tasks[task_id] for task_id in task_ids], [], [])
    n_precision = max(1, min(6, len(task_ids) // 3))
    n_eval = max(1, min(4, len(task_ids) - n_precision - 1))
    precision_ids = task_ids[:n_precision]
    eval_ids = task_ids[-n_eval:]
    heldin_ids = [task_id for task_id in task_ids if task_id not in set(precision_ids + eval_ids)]
    return (
        [arc1_tasks[task_id] for task_id in precision_ids],
        [arc1_tasks[task_id] for task_id in heldin_ids],
        [arc1_tasks[task_id] for task_id in eval_ids],
    )


def _normalize_grid(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(int).tolist()
    if isinstance(value, list):
        return value
    return np.asarray(value, dtype=int).tolist()


def _wrapped_transform_code(code: str) -> str:
    return (
        "import numpy as np\n"
        f"{code}\n\n"
        "def __carnot_transform(input_grid):\n"
        "    result = transform(np.array(input_grid, dtype=int))\n"
        "    return np.asarray(result, dtype=int).tolist()\n"
    )


def _execute_transform(code: str, grid: Any, *, timeout: float = 2.0) -> tuple[Any, str]:
    wrapped = _wrapped_transform_code(code)
    global _SANDBOX_NUMPY_UNAVAILABLE  # noqa: PLW0603 - cache one sandbox substrate fact
    from carnot.verify.python_types import safe_exec_function

    if _SANDBOX_NUMPY_UNAVAILABLE:
        result, error = safe_exec_function(wrapped, "__carnot_transform", (grid,), timeout=timeout)
    else:
        result, error = sandboxed_exec_function(
            wrapped,
            "__carnot_transform",
            (grid,),
            timeout=timeout,
            allow_fallback=True,
        )
        if error is not None and type(error).__name__ == "ModuleNotFoundError" and "numpy" in str(error):
            _SANDBOX_NUMPY_UNAVAILABLE = True
            result, error = safe_exec_function(wrapped, "__carnot_transform", (grid,), timeout=timeout)
    if error is not None:
        return None, f"{type(error).__name__}: {error}"
    return _normalize_grid(result), ""


def evaluate_program(
    task: ArcTask,
    program: CandidateProgram,
    *,
    split: str = "heldout",
) -> ProgramEvaluation:
    """REQ-LEARN-4077-1: execute one transform program and assign labels."""

    errors: list[str] = []
    demo_matches: list[bool] = []
    for index, demo in enumerate(task.demos):
        prediction, error = _execute_transform(program.code, demo.get("input"))
        if error:
            errors.append(f"demo{index}:{error}")
            demo_matches.append(False)
            continue
        demo_matches.append(prediction == demo.get("output"))

    test_prediction, test_error = _execute_transform(program.code, task.test_input)
    if test_error:
        errors.append(f"test:{test_error}")
    gold_outputs = task.gold_outputs
    test_gold = bool(not test_error and any(test_prediction == gold for gold in gold_outputs))

    verifier_energy = 1.0
    if test_prediction is not None:
        rule = Gap4ExecutionVerifier().induce_program(task.demos)
        if rule is not None:
            verifier_energy = float(get_consistency_energy(rule, task.test_input, test_prediction))

    return ProgramEvaluation(
        task_id=task.task_id,
        program_id=program.program_id,
        code=program.code,
        source=program.source,
        split=split,
        demo_perfect=bool(task.demos and all(demo_matches)),
        test_gold=test_gold,
        verifier_energy=float(verifier_energy),
        error="; ".join(errors),
    )


def evaluate_programs(
    tasks: Sequence[ArcTask],
    programs_by_task: Mapping[str, Sequence[CandidateProgram]],
    *,
    split: str,
    program_evaluator: Callable[[ArcTask, CandidateProgram], ProgramEvaluation] | None = None,
) -> list[ProgramEvaluation]:
    """Evaluate all available k-sample programs for a task split."""

    evaluator = program_evaluator or (lambda task, program: evaluate_program(task, program, split=split))
    rows: list[ProgramEvaluation] = []
    for task in tasks:
        for program in programs_by_task.get(task.task_id, ()):
            row = evaluator(task, program)
            if row.split != split:
                row = ProgramEvaluation(
                    task_id=row.task_id,
                    program_id=row.program_id,
                    code=row.code,
                    source=row.source,
                    split=split,
                    demo_perfect=row.demo_perfect,
                    test_gold=row.test_gold,
                    verifier_energy=row.verifier_energy,
                    error=row.error,
                )
            rows.append(row)
    return rows


def compute_certification_metrics(evaluations: Sequence[ProgramEvaluation]) -> CertificationMetrics:
    """REQ-LEARN-4077-2: compute P(test-gold | demo-perfect) and recall."""

    certified = [row for row in evaluations if row.demo_perfect]
    gold = [row for row in evaluations if row.test_gold]
    true_positive = [row for row in certified if row.test_gold]
    precision = len(true_positive) / len(certified) if certified else 0.0
    recall = len(true_positive) / len(gold) if gold else 0.0
    return CertificationMetrics(
        precision=float(precision),
        recall=float(recall),
        n_certified=len(certified),
        n_gold=len(gold),
        n_true_positive=len(true_positive),
    )


def _corpus_item(row: ProgramEvaluation, arm: str) -> dict[str, object]:
    return {
        "arm": arm,
        "task_id": row.task_id,
        "program_id": row.program_id,
        "source": row.source,
        "code": row.code,
        "demo_perfect": bool(row.demo_perfect),
        "test_gold": bool(row.test_gold),
        "verifier_energy": float(row.verifier_energy),
        "text": (
            f"ARC task {row.task_id}\n"
            f"Generator source: {row.source}\n"
            "Write a Python transform(grid) program.\n\n"
            f"{row.code}"
        ),
    }


def build_n_matched_corpora(evaluations: Sequence[ProgramEvaluation]) -> RftCorpora:
    """REQ-LEARN-4077-4: build three same-generator N-matched training arms."""

    by_task: dict[str, list[ProgramEvaluation]] = defaultdict(list)
    for row in evaluations:
        if row.split == "heldin":
            by_task[row.task_id].append(row)

    rft_correct: list[dict[str, object]] = []
    rft_ablation: list[dict[str, object]] = []
    gold_sft: list[dict[str, object]] = []
    for task_id in sorted(by_task):
        rows = sorted(by_task[task_id], key=lambda row: row.program_id)
        correct = [row for row in rows if row.demo_perfect]
        ablation = [row for row in rows if not row.demo_perfect]
        gold = [row for row in rows if row.test_gold]
        n_match = min(len(correct), len(ablation), len(gold))
        for row in correct[:n_match]:
            rft_correct.append(_corpus_item(row, "rft_correct"))
        for row in ablation[:n_match]:
            rft_ablation.append(_corpus_item(row, "rft_ablation"))
        for row in gold[:n_match]:
            gold_sft.append(_corpus_item(row, "gold_sft"))

    return RftCorpora(rft_correct=rft_correct, rft_ablation=rft_ablation, gold_sft=gold_sft)


def build_heldout_eval_manifest(evaluations: Sequence[ProgramEvaluation]) -> list[dict[str, object]]:
    """REQ-LEARN-4077-5: expose a disjoint held-out eval task manifest."""

    task_ids = sorted({row.task_id for row in evaluations if row.split == "heldout"})
    return [{"task_id": task_id, "split": "heldout"} for task_id in task_ids]


def _preconditions_payload(preconditions_checked: Sequence[PreconditionCheck]) -> list[dict[str, object]]:
    return [check.to_dict() for check in preconditions_checked]


def _field_principles() -> dict[str, str]:
    return {
        "certification_precision": "Phase-0 gate: demo-perfect labels must predict test-gold at high precision.",
        "certification_recall": "Usability gate: certification must retain a non-trivial share of gold programs.",
        "n_rft_correct": "RFT-correct corpus size after N matching; zero when the label is poisoned.",
        "n_rft_ablation": "Verifier-label ablation size matched to RFT-correct.",
        "n_gold_sft": "Oracle-label upper-bound corpus size matched where possible.",
        "n_heldout_tasks": "Disjoint tasks reserved for precision/eval accounting.",
        "runner_ready": "Whether the corpus/trainer harness can be invoked after preconditions.",
        "trainer_smoke_passed": "True only after the two-task LoRA smoke train runs.",
    }


def _base_artifact(
    *,
    honest_verdict: str,
    metrics: CertificationMetrics,
    n_rft_correct: int,
    n_rft_ablation: int,
    n_gold_sft: int,
    n_heldout_tasks: int,
    runner_ready: bool,
    trainer_smoke_passed: bool,
    preconditions_checked: Sequence[PreconditionCheck],
    duration_s: float,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "experiment": "experiment_4077_verifier_reward_rft_corpus_build",
        "schema": "carnot.experiment_4077_verifier_reward_rft_corpus_build.v1",
        "honest_verdict": honest_verdict,
        "certification_precision": round(float(metrics.precision), 4),
        "certification_recall": round(float(metrics.recall), 4),
        "n_rft_correct": int(n_rft_correct),
        "n_rft_ablation": int(n_rft_ablation),
        "n_gold_sft": int(n_gold_sft),
        "n_heldout_tasks": int(n_heldout_tasks),
        "runner_ready": bool(runner_ready),
        "trainer_smoke_passed": bool(trainer_smoke_passed),
        "preconditions_checked": _preconditions_payload(preconditions_checked),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": _field_principles(),
        "duration_s": float(duration_s),
        "random_seed": RANDOM_SEED,
        "spec_refs": ["REQ-LEARN-4077", "SCENARIO-LEARN-4077", "SCENARIO-LEARN-4077-NMATCH"],
    }
    errors = artifact_schema_errors(artifact)
    if errors:  # pragma: no cover - builders only pass schema-valid values
        raise ValueError("; ".join(errors))
    return artifact


def build_precondition_blocked_artifact(
    preconditions_checked: Sequence[PreconditionCheck],
    *,
    duration_s: float,
) -> dict[str, object]:
    """REQ-LEARN-4077: fail closed before inference when resources are missing."""

    missing = _first_missing(preconditions_checked)
    verdict = f"blocked_{missing.resource if missing else 'unknown_precondition'}"
    return _base_artifact(
        honest_verdict=verdict,
        metrics=CertificationMetrics(0.0, 0.0, 0, 0, 0),
        n_rft_correct=0,
        n_rft_ablation=0,
        n_gold_sft=0,
        n_heldout_tasks=0,
        runner_ready=False,
        trainer_smoke_passed=False,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
    )


def build_precision_blocked_artifact(
    metrics: CertificationMetrics,
    *,
    preconditions_checked: Sequence[PreconditionCheck],
    n_heldout_tasks: int,
    duration_s: float,
) -> dict[str, object]:
    """SCENARIO-LEARN-4077: block poisoned demo-perfect labels before corpus build."""

    verdict = f"blocked_precision_gate_unmet_{metrics.precision:.4f}_{metrics.recall:.4f}"
    return _base_artifact(
        honest_verdict=verdict,
        metrics=metrics,
        n_rft_correct=0,
        n_rft_ablation=0,
        n_gold_sft=0,
        n_heldout_tasks=n_heldout_tasks,
        runner_ready=True,
        trainer_smoke_passed=False,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
    )


def build_complete_artifact(
    metrics: CertificationMetrics,
    *,
    corpora: RftCorpora,
    heldout_manifest: Sequence[Mapping[str, object]],
    preconditions_checked: Sequence[PreconditionCheck],
    trainer_smoke_passed: bool,
    duration_s: float,
) -> dict[str, object]:
    """SCENARIO-LEARN-4077-NMATCH: report a completed three-arm corpus build."""

    n_a = len(corpora.rft_correct)
    n_b = len(corpora.rft_ablation)
    n_c = len(corpora.gold_sft)
    verdict = f"complete: rft_corpus_built_3arms_nA_{n_a}_nB_{n_b}_nC_{n_c}_precgate_PASS"
    return _base_artifact(
        honest_verdict=verdict,
        metrics=metrics,
        n_rft_correct=n_a,
        n_rft_ablation=n_b,
        n_gold_sft=n_c,
        n_heldout_tasks=len(heldout_manifest),
        runner_ready=True,
        trainer_smoke_passed=trainer_smoke_passed,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
    )


def artifact_schema_errors(artifact: Mapping[str, object]) -> list[str]:
    """Validate the required bare Exp 4077 artifact fields."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")

    for field in ("certification_precision", "certification_recall"):
        if field in artifact and not isinstance(artifact[field], int | float):
            errors.append(f"{field} must be numeric")
    for field in ("n_rft_correct", "n_rft_ablation", "n_gold_sft", "n_heldout_tasks"):
        if field in artifact and type(artifact[field]) is not int:
            errors.append(f"{field} must be a bare int")
    for field in ("runner_ready", "trainer_smoke_passed"):
        if field in artifact and type(artifact[field]) is not bool:
            errors.append(f"{field} must be a bare bool")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must declare the Exp 4077 substrate")

    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, list):
        errors.append("preconditions_checked must be a list")
    elif any(not isinstance(item, Mapping) or "resource" not in item or "available" not in item for item in preconditions):
        errors.append("preconditions_checked entries must include resource and available")

    if isinstance(verdict, str) and verdict.startswith("complete:"):
        counts = [int(artifact.get(field, -1)) for field in ("n_rft_correct", "n_rft_ablation", "n_gold_sft")]
        if len(set(counts)) != 1:
            errors.append("complete artifacts must be N-matched across all three corpora")
    if isinstance(verdict, str) and verdict.startswith("blocked_precision_gate_unmet"):
        if any(int(artifact.get(field, 0) or 0) for field in ("n_rft_correct", "n_rft_ablation", "n_gold_sft")):
            errors.append("precision-gate blocked artifacts must not include corpus rows")
        if artifact.get("trainer_smoke_passed") is not False:
            errors.append("precision-gate blocked artifacts must not smoke train")
    return errors


def write_corpus_jsonl(corpora: RftCorpora, output_dir: str | Path) -> dict[str, Path]:
    """Write the three corpora as JSONL sidecars for the later Exp 4078 runner."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    mapping = {
        "rft_correct": (root / "experiment_4077_rft_correct.jsonl", corpora.rft_correct),
        "rft_ablation": (root / "experiment_4077_rft_ablation.jsonl", corpora.rft_ablation),
        "gold_sft": (root / "experiment_4077_gold_sft.jsonl", corpora.gold_sft),
    }
    paths: dict[str, Path] = {}
    for name, (path, rows) in mapping.items():
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        paths[name] = path
    return paths


def smoke_train_two_tasks(
    corpora: RftCorpora,
    *,
    trainer_factory: Callable[[list[dict[str, object]], list[str]], bool] | None = None,
) -> bool:
    """REQ-LEARN-4077-5: run or delegate the two-task three-arm LoRA smoke."""

    arms = ["rft_correct", "rft_ablation", "gold_sft"]
    records = (corpora.rft_correct + corpora.rft_ablation + corpora.gold_sft)[:2]
    if len(records) < 2:
        return False
    if trainer_factory is not None:
        return bool(trainer_factory(records, arms))
    return _run_trl_lora_smoke(records, arms)  # pragma: no cover - real smoke is gated by Phase 0


def _run_trl_lora_smoke(records: list[dict[str, object]], arms: list[str]) -> bool:  # pragma: no cover
    """Minimal real TRL/PEFT smoke path; skipped unless the precision gate passes."""

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    del arms
    model_id = "Qwen/Qwen3.5-0.8B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=True,
        trust_remote_code=os.environ.get("CARNOT_TRUST_REMOTE_CODE", "") == "1",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=True,
        trust_remote_code=os.environ.get("CARNOT_TRUST_REMOTE_CODE", "") == "1",
        device_map="auto",
    )
    dataset = Dataset.from_list([{"text": str(row["text"])} for row in records[:2]])
    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    args = SFTConfig(
        output_dir=str(REPO_ROOT / "results" / "experiment_4077_lora_smoke"),
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=256,
        bf16=False,
        fp16=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    return True


def write_result_artifact(artifact: Mapping[str, object], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_path: str | Path | None = None,
    preconditions_checker: Callable[..., Sequence[PreconditionCheck]] = check_preconditions,
    task_loader: Callable[..., tuple[Sequence[ArcTask], Sequence[ArcTask]] | tuple[Sequence[ArcTask], Sequence[ArcTask], Sequence[ArcTask]]] = load_default_task_splits,
    program_loader: Callable[..., Mapping[str, Sequence[CandidateProgram]]] = load_program_checkpoint,
    program_evaluator: Callable[[ArcTask, CandidateProgram], ProgramEvaluation] | None = None,
    smoke_trainer: Callable[[RftCorpora], bool] | None = None,
) -> dict[str, object]:
    """REQ-LEARN-4077: run preconditions, precision gate, corpus build, and JSON write."""

    start = time.perf_counter()
    root = Path(repo_root)
    checks = list(preconditions_checker(repo_root=root))
    output = Path(output_path) if output_path is not None else root / "results" / RESULT_FILENAME
    if _first_missing(checks) is not None:
        artifact = build_precondition_blocked_artifact(checks, duration_s=time.perf_counter() - start)
        write_result_artifact(artifact, output)
        return artifact

    split_result = task_loader(repo_root=root)
    if len(split_result) == 2:
        precision_tasks, heldin_tasks = split_result
        heldout_tasks: Sequence[ArcTask] = ()
    else:
        precision_tasks, heldin_tasks, heldout_tasks = split_result

    try:
        programs_by_task = program_loader(root / DEFAULT_PROGRAM_CHECKPOINT, k_required=K_PROGRAMS_PER_TASK)
    except TypeError:
        programs_by_task = program_loader(repo_root=root)

    precision_rows = evaluate_programs(
        precision_tasks,
        programs_by_task,
        split="heldout",
        program_evaluator=program_evaluator,
    )
    metrics = compute_certification_metrics(precision_rows)
    if not metrics.gate_passed:
        artifact = build_precision_blocked_artifact(
            metrics,
            preconditions_checked=checks,
            n_heldout_tasks=len({task.task_id for task in precision_tasks}),
            duration_s=time.perf_counter() - start,
        )
        write_result_artifact(artifact, output)
        return artifact

    heldin_rows = evaluate_programs(
        heldin_tasks,
        programs_by_task,
        split="heldin",
        program_evaluator=program_evaluator,
    )
    heldout_rows = evaluate_programs(
        heldout_tasks,
        programs_by_task,
        split="heldout",
        program_evaluator=program_evaluator,
    )
    all_rows = heldin_rows + heldout_rows
    corpora = build_n_matched_corpora(all_rows)
    manifest = build_heldout_eval_manifest(all_rows)
    smoke_passed = smoke_trainer(corpora) if smoke_trainer is not None else smoke_train_two_tasks(corpora)
    if any((corpora.rft_correct, corpora.rft_ablation, corpora.gold_sft)):
        write_corpus_jsonl(corpora, output.parent)
    artifact = build_complete_artifact(
        metrics,
        corpora=corpora,
        heldout_manifest=manifest,
        preconditions_checked=checks,
        trainer_smoke_passed=bool(smoke_passed),
        duration_s=time.perf_counter() - start,
    )
    write_result_artifact(artifact, output)
    return artifact
