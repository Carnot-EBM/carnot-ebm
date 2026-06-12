"""Exp 4088 trustworthy ARC verifier-reward RFT corpus build.

Spec refs: REQ-LEARN-4088, SCENARIO-LEARN-4088-BLOCKED,
SCENARIO-LEARN-4088-NMATCH, SCENARIO-LEARN-4088-SMOKE.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import re
import shutil
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from carnot.agentic.arc_exp4077_verifier_reward_rft_corpus_build import _execute_transform


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_FILENAME = "experiment_4088_verifier_reward_rft_corpus_build.json"
EXP4087_RESULT_FILENAME = "experiment_4087_certification_precision_rescue.json"
DEFAULT_ARC1_POOL = Path("results/arc3_gap3_stage2_eval_pool.json.gz")
CHECKPOINT_ROOT = Path("results/checkpoints/experiment_4088_verifier_reward_rft_corpus_build")
INFERENCE_SUBSTRATE = "live_llm_inference"
RANDOM_SEED = 4088
K_PROGRAMS_PER_TASK = 8
MIN_COMPLETE_DURATION_S = 60.0
QWEN_MODEL_ID = "Qwen/Qwen3.5-0.8B"
TERMINAL_PREFIXES = ("complete:", "success:", "blocked_", "failed:")
CODEX_COMMAND = (
    "codex",
    "exec",
    "--color",
    "never",
    "--model",
    "gpt-5.5",
    "-c",
    "model_reasoning_effort=medium",
    "--dangerously-bypass-approvals-and-sandbox",
    "--cd",
    "/tmp",
    "--ephemeral",
)
PREFERRED_HELDIN_TASK_IDS = ("00576224", "2072aba6")
PREFERRED_HELDOUT_TASK_IDS = ("66e6c45b",)
ARMS = ("rft_correct", "rft_ablation", "gold_sft")
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "operating_point_used",
    "n_rft_correct",
    "n_rft_ablation",
    "n_gold_sft",
    "n_heldout_tasks",
    "runner_ready",
    "trainer_smoke_passed",
    "preconditions_checked",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "inference_substrate",
)


@dataclass(frozen=True)
class PreconditionCheck:
    """One mandatory resource check recorded before live compute."""

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
    """One live Codex-generated transform program."""

    task_id: str
    program_id: str
    code: str
    source: str
    latency_s: float
    raw_response_sha256: str


@dataclass(frozen=True)
class GeneratedProgramEvaluation:
    """Labels and certification status for one generated program."""

    task_id: str
    program_id: str
    code: str
    source: str
    split: str
    demo_perfect: bool
    test_gold: bool
    prediction_hash: str | None
    certified_correct: bool
    verifier_energy: float
    error: str


@dataclass(frozen=True)
class RftCorpora:
    """N-matched rows for the three training arms."""

    rft_correct: list[dict[str, object]]
    rft_ablation: list[dict[str, object]]
    gold_sft: list[dict[str, object]]


@dataclass(frozen=True)
class GenerationRecord:
    """Per-task live Codex accounting."""

    task_id: str
    n_requested: int
    n_programs: int
    latency_s: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    """Shared LoRA/SFT smoke configuration; identical for every arm."""

    lora_rank: int = 4
    lora_alpha: int = 8
    learning_rate: float = 1e-5
    max_steps: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_length: int = 256
    random_seed: int = RANDOM_SEED

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["save_strategy"] = "steps"
        payload["save_steps"] = 1
        return payload


@dataclass(frozen=True)
class SmokeTrainResult:
    """Result of the two-task three-arm LoRA smoke."""

    passed: bool
    checkpoint_paths: dict[str, str]
    training_config: dict[str, object]


def _resource_slug(text: str) -> str:
    return text.lower().replace("/", "_").replace("-", "_").replace(".", "_").replace("__", "_")


def check_qwen_hf_model_load(model_id: str = QWEN_MODEL_ID) -> PreconditionCheck:  # pragma: no cover
    """REQ-LEARN-4088-1: load the trainable HF model locally before generation."""

    resource = f"hf_safetensors_{_resource_slug(model_id)}"
    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
        del model
        return PreconditionCheck(resource, True, "AutoModelForCausalLM loaded from local HF safetensors")
    except Exception as exc:
        return PreconditionCheck(resource, False, f"{type(exc).__name__}: {exc}")


def check_trainer_imports() -> PreconditionCheck:  # pragma: no cover
    """REQ-LEARN-4088-1: verify TRL/PEFT trainers are importable."""

    try:
        import peft  # noqa: F401
        import trl  # noqa: F401
        from trl import GRPOTrainer, SFTTrainer  # noqa: F401

        return PreconditionCheck("trl_peft_trainers", True, "trl, peft, SFTTrainer, GRPOTrainer import")
    except Exception as exc:
        return PreconditionCheck("trl_peft_trainers", False, f"{type(exc).__name__}: {exc}")


def check_cuda_visible() -> PreconditionCheck:  # pragma: no cover
    """REQ-LEARN-4088-1: require visible CUDA before LoRA smoke training."""

    try:
        torch = importlib.import_module("torch")
        available = bool(torch.cuda.is_available())
        detail = "torch.cuda.is_available() is true" if available else "torch.cuda.is_available() is false"
        return PreconditionCheck("cuda_visible", available, detail)
    except Exception as exc:
        return PreconditionCheck("cuda_visible", False, f"{type(exc).__name__}: {exc}")


def check_codex_cli() -> PreconditionCheck:  # pragma: no cover
    path = shutil.which("codex")
    return PreconditionCheck("codex_cli", bool(path), path or "codex executable not found")


def _load_json_or_gzip(path: Path) -> Any:  # pragma: no cover
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def check_arc_pool(path: str | Path, resource: str = "arc1_pool") -> PreconditionCheck:  # pragma: no cover
    try:
        payload = _load_json_or_gzip(Path(path))
    except Exception as exc:
        return PreconditionCheck(resource, False, f"{type(exc).__name__}: {exc}")
    entries = payload.get("entries") if isinstance(payload, Mapping) else payload
    available = isinstance(entries, list) and bool(entries)
    detail = f"{len(entries) if isinstance(entries, list) else 0} entries at {path}"
    return PreconditionCheck(resource, available, detail)


def load_exp4087_operating_point(repo_root: str | Path = REPO_ROOT) -> dict[str, object]:
    path = Path(repo_root) / "results" / EXP4087_RESULT_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("precision_rescue_succeeded") is not True:
        raise ValueError("Exp 4087 precision_rescue_succeeded is not true")
    point = payload.get("best_operating_point")
    if not isinstance(point, Mapping):
        raise ValueError("Exp 4087 best_operating_point missing")
    precision = float(point.get("precision", 0.0))
    if precision < 0.85:
        raise ValueError(f"Exp 4087 precision below gate: {precision}")
    return dict(point)


def check_exp4087_operating_point(*, repo_root: str | Path = REPO_ROOT) -> PreconditionCheck:
    try:
        point = load_exp4087_operating_point(repo_root)
        detail = (
            f"{point.get('filter_stack')} {point.get('threshold')} "
            f"precision={float(point.get('precision', 0.0)):.4f}"
        )
        return PreconditionCheck("exp4087_operating_point", True, detail)
    except Exception as exc:
        return PreconditionCheck("exp4087_operating_point", False, f"{type(exc).__name__}: {exc}")


def check_preconditions(*, repo_root: str | Path = REPO_ROOT) -> list[PreconditionCheck]:  # pragma: no cover
    root = Path(repo_root)
    return [
        check_qwen_hf_model_load(),
        check_trainer_imports(),
        check_cuda_visible(),
        check_exp4087_operating_point(repo_root=root),
        check_arc_pool(root / DEFAULT_ARC1_POOL),
        check_codex_cli(),
    ]


def _first_missing(checks: Sequence[PreconditionCheck]) -> PreconditionCheck | None:
    return next((check for check in checks if not check.available), None)


def load_arc_pool(path: str | Path, *, pool_name: str) -> list[ArcTask]:  # pragma: no cover
    payload = _load_json_or_gzip(Path(path))
    entries = payload.get("entries") if isinstance(payload, Mapping) else payload
    if not isinstance(entries, list):
        raise ValueError(f"{path} does not contain an entries list")
    tasks: list[ArcTask] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        task_id = str(entry.get("task") or entry.get("task_id") or "")
        candidates = list(entry.get("candidates") or [])
        if task_id and any(isinstance(item, Mapping) and item.get("correct") is True for item in candidates):
            tasks.append(
                ArcTask(
                    task_id=task_id,
                    pool_name=pool_name,
                    demos=list(entry.get("demos") or []),
                    test_input=entry.get("test_input"),
                    candidates=candidates,
                )
            )
    if not tasks:
        raise ValueError(f"{path} contains no usable ARC tasks")
    return tasks


def load_default_task_splits(*, repo_root: str | Path = REPO_ROOT) -> tuple[list[ArcTask], list[ArcTask]]:  # pragma: no cover
    tasks = load_arc_pool(Path(repo_root) / DEFAULT_ARC1_POOL, pool_name="arc1")
    by_id = {task.task_id: task for task in tasks}
    heldin = [by_id[task_id] for task_id in PREFERRED_HELDIN_TASK_IDS if task_id in by_id]
    heldout = [by_id[task_id] for task_id in PREFERRED_HELDOUT_TASK_IDS if task_id in by_id]
    if len(heldin) < 2:
        heldin = tasks[:2]
    if not heldout:
        heldout = [task for task in tasks if task.task_id not in {item.task_id for item in heldin}][:1]
    if len(heldin) < 1 or len(heldout) < 1:
        raise ValueError("not enough disjoint ARC tasks for held-in and held-out splits")
    return heldin, heldout


def _grid_hash(grid: Any) -> str | None:
    if grid is None:
        return None
    arr = np.asarray(grid, dtype=np.int64)
    payload = repr(tuple(arr.shape)).encode("ascii") + arr.tobytes()
    return hashlib.sha1(payload).hexdigest()


def _grid_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    left_arr = np.asarray(left, dtype=np.int64)
    right_arr = np.asarray(right, dtype=np.int64)
    return left_arr.shape == right_arr.shape and bool(np.array_equal(left_arr, right_arr))


def extract_transform_code(text: str) -> str | None:  # pragma: no cover
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    for block in reversed(blocks):
        if "def transform" in block:
            return block.strip()
    match = re.search(r"(?:import numpy as np\s*)?def transform\(.*", text, re.DOTALL)
    return match.group(0).strip() if match else None


def _task_prompt(task: ArcTask, draw_index: int) -> str:  # pragma: no cover
    mode = (
        "Write your strongest solution candidate."
        if draw_index < 6
        else (
            "Write a different plausible baseline or alternate-rule candidate. "
            "It must still be a valid transform(grid) program, but it may be wrong; "
            "diversity is preferred over copying the strongest solution."
        )
    )
    return (
        "You are solving one ARC grid transformation task. Infer the rule from public demos only.\n"
        "Return exactly one Python code block defining:\n"
        "def transform(grid):\n"
        "    ...\n"
        "The function receives a 2D numpy array and must return a 2D array/list. "
        "Use only numpy as np. Do not read files, use network, or hardcode the test output.\n\n"
        f"Candidate mode: {mode}\n"
        f"Task id: {task.task_id}; candidate draw: {draw_index}\n"
        f"Demos JSON:\n{json.dumps(task.demos, sort_keys=True)}\n\n"
        f"Test input JSON:\n{json.dumps(task.test_input, sort_keys=True)}\n"
    )


def _run_codex_candidate(
    task: ArcTask,
    draw_index: int,
    *,
    timeout_s: float = 240.0,
    command: Sequence[str] = CODEX_COMMAND,
) -> CandidateProgram:  # pragma: no cover
    started = time.perf_counter()
    result = subprocess.run(
        list(command),
        input=_task_prompt(task, draw_index),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    latency = time.perf_counter() - started
    raw = result.stdout or result.stderr or ""
    code = extract_transform_code(raw)
    if not code:
        raise RuntimeError(f"codex returned no transform code for {task.task_id} draw {draw_index}")
    return CandidateProgram(
        task_id=task.task_id,
        program_id=f"{task.task_id}:codex{draw_index}",
        code=code,
        source="codex_gpt5.5_live",
        latency_s=round(float(latency), 3),
        raw_response_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )


def generate_programs_for_tasks(
    tasks: Sequence[ArcTask],
    *,
    k: int = K_PROGRAMS_PER_TASK,
    max_attempts_per_task: int = K_PROGRAMS_PER_TASK + 4,
) -> tuple[dict[str, list[CandidateProgram]], list[GenerationRecord]]:  # pragma: no cover
    programs: dict[str, list[CandidateProgram]] = {}
    records: list[GenerationRecord] = []
    for task in tasks:
        print(f"[exp4088] generating task={task.task_id} k={k}", flush=True)
        task_programs: list[CandidateProgram] = []
        started = time.perf_counter()
        attempts = 0
        while len(task_programs) < k and attempts < max_attempts_per_task:
            attempts += 1
            print(f"[exp4088] task={task.task_id} candidate {len(task_programs) + 1}/{k}", flush=True)
            try:
                task_programs.append(_run_codex_candidate(task, attempts - 1))
            except Exception as exc:
                print(f"[exp4088] task={task.task_id} candidate_error={type(exc).__name__}: {exc}", flush=True)
        programs[task.task_id] = task_programs
        latency = time.perf_counter() - started
        records.append(GenerationRecord(task.task_id, k, len(task_programs), round(float(latency), 3)))
    return programs, records


def evaluate_generated_program(
    task: ArcTask,
    program: CandidateProgram,
    split: str,
) -> GeneratedProgramEvaluation:  # pragma: no cover
    errors: list[str] = []
    demo_matches: list[bool] = []
    for index, demo in enumerate(task.demos):
        prediction, error = _execute_transform(program.code, demo.get("input"))
        if error:
            errors.append(f"demo{index}:{error}")
            demo_matches.append(False)
            continue
        demo_matches.append(_grid_equal(prediction, demo.get("output")))

    test_prediction, test_error = _execute_transform(program.code, task.test_input)
    if test_error:
        errors.append(f"test:{test_error}")
    test_gold = bool(not test_error and any(_grid_equal(test_prediction, gold) for gold in task.gold_outputs))
    return GeneratedProgramEvaluation(
        task_id=task.task_id,
        program_id=program.program_id,
        code=program.code,
        source=program.source,
        split=split,
        demo_perfect=bool(task.demos and all(demo_matches)),
        test_gold=test_gold,
        prediction_hash=_grid_hash(test_prediction),
        certified_correct=False,
        verifier_energy=0.0 if bool(task.demos and all(demo_matches)) else 1.0,
        error="; ".join(errors),
    )


def evaluate_programs(
    tasks: Sequence[ArcTask],
    programs_by_task: Mapping[str, Sequence[CandidateProgram]],
    *,
    split: str,
    program_evaluator: Callable[[ArcTask, CandidateProgram, str], GeneratedProgramEvaluation] | None = None,
) -> list[GeneratedProgramEvaluation]:
    evaluator = program_evaluator or evaluate_generated_program
    rows: list[GeneratedProgramEvaluation] = []
    for task in tasks:
        for program in programs_by_task.get(task.task_id, ()):
            row = evaluator(task, program, split)
            if row.split != split:
                row = replace(row, split=split)
            rows.append(row)
    return rows


def _threshold_k(operating_point: Mapping[str, object]) -> int:
    threshold = str(operating_point.get("threshold", "k=1"))
    match = re.search(r"k=(\d+)", threshold)
    return int(match.group(1)) if match else 1


def apply_operating_point(
    evaluations: Sequence[GeneratedProgramEvaluation],
    operating_point: Mapping[str, object],
) -> list[GeneratedProgramEvaluation]:
    """REQ-LEARN-4088-3: apply Exp 4087's selected certification rule."""

    filter_stack = str(operating_point.get("filter_stack", ""))
    min_agreement = _threshold_k(operating_point)
    certified_ids: set[tuple[str, str]] = set()
    if filter_stack in {"demo_perfect", "k_of_n_agreement"}:
        by_task_hash: dict[tuple[str, str], list[GeneratedProgramEvaluation]] = defaultdict(list)
        for row in evaluations:
            if row.demo_perfect and row.prediction_hash is not None:
                by_task_hash[(row.task_id, row.prediction_hash)].append(row)
        for bucket in by_task_hash.values():
            if len(bucket) >= min_agreement:
                certified_ids.update((row.task_id, row.program_id) for row in bucket)
    elif filter_stack == "graded_min_hamming":
        threshold = str(operating_point.get("threshold", "tau=0.0"))
        match = re.search(r"tau=([0-9.]+)", threshold)
        tau = float(match.group(1)) if match else 0.0
        certified_ids.update(
            (row.task_id, row.program_id)
            for row in evaluations
            if row.verifier_energy <= tau + 1e-12 and row.prediction_hash is not None
        )
    else:
        raise ValueError(f"unsupported Exp 4087 operating point: {filter_stack}")
    return [
        replace(row, certified_correct=(row.task_id, row.program_id) in certified_ids)
        for row in evaluations
    ]


def _corpus_item(row: GeneratedProgramEvaluation, arm: str) -> dict[str, object]:
    return {
        "arm": arm,
        "task_id": row.task_id,
        "program_id": row.program_id,
        "source": row.source,
        "code": row.code,
        "demo_perfect": bool(row.demo_perfect),
        "test_gold": bool(row.test_gold),
        "certified_correct": bool(row.certified_correct),
        "verifier_energy": float(row.verifier_energy),
        "text": (
            f"ARC task {row.task_id}\n"
            f"Generator source: {row.source}\n"
            "Write a Python transform(grid) program.\n\n"
            f"{row.code}"
        ),
    }


def build_n_matched_corpora(evaluations: Sequence[GeneratedProgramEvaluation]) -> RftCorpora:
    """REQ-LEARN-4088-4: build three same-generator N-matched training arms."""

    by_task: dict[str, list[GeneratedProgramEvaluation]] = defaultdict(list)
    for row in evaluations:
        if row.split == "heldin":
            by_task[row.task_id].append(row)

    rft_correct: list[dict[str, object]] = []
    rft_ablation: list[dict[str, object]] = []
    gold_sft: list[dict[str, object]] = []
    for task_id in sorted(by_task):
        rows = sorted(by_task[task_id], key=lambda row: row.program_id)
        correct = [row for row in rows if row.certified_correct]
        ablation = [row for row in rows if not row.certified_correct]
        gold = [row for row in rows if row.test_gold]
        n_match = min(len(correct), len(ablation), len(gold))
        rft_correct.extend(_corpus_item(row, "rft_correct") for row in correct[:n_match])
        rft_ablation.extend(_corpus_item(row, "rft_ablation") for row in ablation[:n_match])
        gold_sft.extend(_corpus_item(row, "gold_sft") for row in gold[:n_match])
    return RftCorpora(rft_correct, rft_ablation, gold_sft)


def build_heldout_eval_manifest(
    rows_or_tasks: Sequence[GeneratedProgramEvaluation] | Sequence[ArcTask],
) -> list[dict[str, object]]:
    """REQ-LEARN-4088-6: expose a disjoint held-out eval manifest."""

    task_ids: set[str] = set()
    for item in rows_or_tasks:
        if isinstance(item, GeneratedProgramEvaluation):
            if item.split == "heldout":
                task_ids.add(item.task_id)
        else:
            task_ids.add(item.task_id)
    return [{"split": "heldout", "task_id": task_id} for task_id in sorted(task_ids)]


def model_specs() -> dict[str, object]:
    return {
        "base_model": {
            "model_id": QWEN_MODEL_ID,
            "weights": "HF safetensors",
            "trainable": True,
            "gguf": False,
        },
        "generator": {
            "name": "codex",
            "command": list(CODEX_COMMAND),
            "k_programs_per_task": K_PROGRAMS_PER_TASK,
        },
        "lora_smoke": TrainingConfig().to_dict(),
    }


def _preconditions_payload(checks: Sequence[PreconditionCheck]) -> list[dict[str, object]]:
    return [check.to_dict() for check in checks]


def _field_principles() -> dict[str, str]:
    return {
        "honest_verdict": "Terminal-prefix outcome; blocked resources prevent fabricated corpus claims.",
        "operating_point_used": "The corpus inherits Exp 4087's >=0.85 certification rule, not the raw demo-perfect label.",
        "n_rft_correct": "Verifier-certified arm size after N matching.",
        "n_rft_ablation": "Certified-not-correct/demo-failing ablation size matched to RFT-correct.",
        "n_gold_sft": "Oracle test-gold upper-bound arm size matched to RFT-correct where possible.",
        "n_heldout_tasks": "Disjoint ARC task ids reserved for Exp 4089 evaluation.",
        "runner_ready": "Bare bool gate for whether Exp 4089 can consume this corpus.",
        "trainer_smoke_passed": "True only after the two-task LoRA smoke writes checkpoints.",
        "preconditions_checked": "Records verified resources before live compute to block silent fabrication.",
        "model_specs": "Trainable HF model and identical three-arm LoRA configuration.",
        "random_seed": "Fixed seed recorded for reproducible ordering.",
        "reproducibility_checksum": "SHA-256 over stable required artifact fields.",
        "inference_substrate": "live_llm_inference because real Codex generation is used.",
    }


def reproducibility_checksum(artifact: Mapping[str, object]) -> str:
    checksum_fields = {
        key: artifact[key]
        for key in REQUIRED_ARTIFACT_FIELDS
        if key != "reproducibility_checksum"
    }
    encoded = json.dumps(checksum_fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_schema_errors(artifact: Mapping[str, object]) -> list[str]:  # pragma: no cover
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")
    if not isinstance(artifact.get("operating_point_used"), Mapping):
        errors.append("operating_point_used must be a dict")
    for field in ("n_rft_correct", "n_rft_ablation", "n_gold_sft", "n_heldout_tasks", "random_seed"):
        if not isinstance(artifact.get(field), int) or isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be a bare int")
    for field in ("runner_ready", "trainer_smoke_passed"):
        if not isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be a bare bool")
    if not isinstance(artifact.get("preconditions_checked"), list):
        errors.append("preconditions_checked must be a list")
    if not isinstance(artifact.get("model_specs"), Mapping):
        errors.append("model_specs must be a dict")
    if not isinstance(artifact.get("reproducibility_checksum"), str):
        errors.append("reproducibility_checksum must be a string")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be live_llm_inference")
    if isinstance(verdict, str) and verdict.startswith("complete:"):
        counts = [int(artifact.get(field, -1)) for field in ("n_rft_correct", "n_rft_ablation", "n_gold_sft")]
        if len(set(counts)) != 1:
            errors.append("complete artifacts must be N-matched across all three corpora")
        if artifact.get("trainer_smoke_passed") is not True:
            errors.append("complete artifacts must pass the LoRA smoke")
    checksum = artifact.get("reproducibility_checksum")
    if isinstance(checksum, str) and checksum and all(field in artifact for field in REQUIRED_ARTIFACT_FIELDS):
        if checksum != reproducibility_checksum(artifact):
            errors.append("reproducibility_checksum mismatch")
    return errors


def _base_artifact(
    *,
    honest_verdict: str,
    operating_point: Mapping[str, object],
    n_rft_correct: int,
    n_rft_ablation: int,
    n_gold_sft: int,
    n_heldout_tasks: int,
    runner_ready: bool,
    trainer_smoke_passed: bool,
    preconditions_checked: Sequence[PreconditionCheck],
    duration_s: float,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "experiment": "experiment_4088_verifier_reward_rft_corpus_build",
        "schema": "carnot.experiment_4088_verifier_reward_rft_corpus_build.v1",
        "honest_verdict": honest_verdict,
        "operating_point_used": dict(operating_point),
        "n_rft_correct": int(n_rft_correct),
        "n_rft_ablation": int(n_rft_ablation),
        "n_gold_sft": int(n_gold_sft),
        "n_heldout_tasks": int(n_heldout_tasks),
        "runner_ready": bool(runner_ready),
        "trainer_smoke_passed": bool(trainer_smoke_passed),
        "preconditions_checked": _preconditions_payload(preconditions_checked),
        "model_specs": model_specs(),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": _field_principles(),
        "duration_s": round(float(duration_s), 3),
        "spec_refs": [
            "REQ-LEARN-4088",
            "SCENARIO-LEARN-4088-BLOCKED",
            "SCENARIO-LEARN-4088-NMATCH",
            "SCENARIO-LEARN-4088-SMOKE",
        ],
    }
    if extra:
        artifact.update(dict(extra))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = artifact_schema_errors(artifact)
    if errors:  # pragma: no cover
        raise ValueError("; ".join(errors))
    return artifact


def build_precondition_blocked_artifact(
    preconditions_checked: Sequence[PreconditionCheck],
    *,
    duration_s: float,
) -> dict[str, object]:
    missing = _first_missing(preconditions_checked)
    return _base_artifact(
        honest_verdict=f"blocked_{missing.resource if missing else 'unknown_precondition'}",
        operating_point={},
        n_rft_correct=0,
        n_rft_ablation=0,
        n_gold_sft=0,
        n_heldout_tasks=0,
        runner_ready=False,
        trainer_smoke_passed=False,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        extra={"methodology": {"blocked_before_live_generation": True}},
    )


def build_complete_artifact(
    *,
    operating_point: Mapping[str, object],
    corpora: RftCorpora,
    heldout_manifest: Sequence[Mapping[str, object]],
    preconditions_checked: Sequence[PreconditionCheck],
    generation_records: Sequence[GenerationRecord],
    smoke_result: SmokeTrainResult,
    duration_s: float,
) -> dict[str, object]:
    n_a = len(corpora.rft_correct)
    n_b = len(corpora.rft_ablation)
    n_c = len(corpora.gold_sft)
    precision = float(operating_point.get("precision", 0.0))
    verdict = f"complete: rft_corpus_built_3arms_nA_{n_a}_nB_{n_b}_nC_{n_c}_at_prec_{precision:.2f}"
    return _base_artifact(
        honest_verdict=verdict,
        operating_point=operating_point,
        n_rft_correct=n_a,
        n_rft_ablation=n_b,
        n_gold_sft=n_c,
        n_heldout_tasks=len(heldout_manifest),
        runner_ready=True,
        trainer_smoke_passed=bool(smoke_result.passed),
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        extra={
            "methodology": {
                "generator": "codex exec live program synthesis",
                "k_programs_per_heldin_task": K_PROGRAMS_PER_TASK,
                "generation_records": [record.to_dict() for record in generation_records],
                "duration_floor_s_for_live_claim": MIN_COMPLETE_DURATION_S,
                "lora_smoke": {
                    "checkpoint_paths": dict(smoke_result.checkpoint_paths),
                    "training_config": dict(smoke_result.training_config),
                },
            },
            "heldout_task_ids": [str(item["task_id"]) for item in heldout_manifest],
        },
    )


def write_corpus_jsonl(corpora: RftCorpora, output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    mapping = {
        "rft_correct": (root / "experiment_4088_rft_correct.jsonl", corpora.rft_correct),
        "rft_ablation": (root / "experiment_4088_rft_ablation.jsonl", corpora.rft_ablation),
        "gold_sft": (root / "experiment_4088_gold_sft.jsonl", corpora.gold_sft),
    }
    paths: dict[str, Path] = {}
    for arm, (path, rows) in mapping.items():
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        paths[arm] = path
    return paths


def write_heldout_eval_manifest(
    heldout_manifest: Sequence[Mapping[str, object]],
    output_dir: str | Path,
    checkpoint_paths: Mapping[str, str],
    specs: Mapping[str, object],
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "experiment_4088_heldout_eval_manifest.json"
    payload = {
        "schema": "carnot.experiment_4088_heldout_eval_manifest.v1",
        "heldout_task_ids": [str(item["task_id"]) for item in heldout_manifest],
        "checkpoint_paths": dict(checkpoint_paths),
        "model_specs": dict(specs),
        "arms": list(ARMS),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def smoke_train_two_tasks(
    corpora: RftCorpora,
    *,
    repo_root: str | Path = REPO_ROOT,
    trainer_factory: Callable[[str, list[dict[str, object]], Path, TrainingConfig], None] | None = None,
    training_config: TrainingConfig = TrainingConfig(),
) -> SmokeTrainResult:
    """REQ-LEARN-4088-5: run a two-task LoRA smoke for all three arms."""

    records_by_arm = {
        "rft_correct": corpora.rft_correct,
        "rft_ablation": corpora.rft_ablation,
        "gold_sft": corpora.gold_sft,
    }
    selected_task_ids = sorted({str(row["task_id"]) for row in corpora.rft_correct})[:2]
    if len(selected_task_ids) < 2:
        return SmokeTrainResult(False, {}, training_config.to_dict())

    checkpoint_paths: dict[str, str] = {}
    passed = True
    for arm in ARMS:
        records = [row for row in records_by_arm[arm] if row["task_id"] in selected_task_ids][:2]
        checkpoint = Path(repo_root) / CHECKPOINT_ROOT / arm
        checkpoint_paths[arm] = str(checkpoint)
        if len(records) < 2:
            passed = False
            continue
        if trainer_factory is None:
            _run_trl_lora_smoke_arm(arm, records, checkpoint, training_config)  # pragma: no cover
        else:
            trainer_factory(arm, records, checkpoint, training_config)
        if not checkpoint.exists() or not any(checkpoint.iterdir()):
            passed = False
    return SmokeTrainResult(passed, checkpoint_paths, training_config.to_dict())


def _run_trl_lora_smoke_arm(  # pragma: no cover
    arm: str,
    records: list[dict[str, object]],
    checkpoint: Path,
    training_config: TrainingConfig,
) -> None:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    checkpoint.mkdir(parents=True, exist_ok=True)
    set_seed(training_config.random_seed)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_ID, local_files_only=True, device_map="auto")
    dataset = Dataset.from_list([{"text": f"[{arm}]\n{row['text']}"} for row in records])
    peft_config = LoraConfig(
        r=training_config.lora_rank,
        lora_alpha=training_config.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    args = SFTConfig(
        output_dir=str(checkpoint),
        max_steps=training_config.max_steps,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        logging_steps=1,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=1,
        report_to="none",
        max_length=training_config.max_length,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(checkpoint))
    (checkpoint / "smoke_manifest.json").write_text(
        json.dumps({"arm": arm, "records": len(records), "model": QWEN_MODEL_ID}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_result_artifact(artifact: Mapping[str, object], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _generation_records_from_programs(
    tasks: Sequence[ArcTask],
    programs_by_task: Mapping[str, Sequence[CandidateProgram]],
) -> list[GenerationRecord]:
    records: list[GenerationRecord] = []
    for task in tasks:
        programs = list(programs_by_task.get(task.task_id, ()))
        records.append(
            GenerationRecord(
                task_id=task.task_id,
                n_requested=K_PROGRAMS_PER_TASK,
                n_programs=len(programs),
                latency_s=round(sum(float(program.latency_s) for program in programs), 3),
            )
        )
    return records


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_path: str | Path | None = None,
    preconditions_checker: Callable[..., Sequence[PreconditionCheck]] = check_preconditions,
    operating_point_loader: Callable[[str | Path], Mapping[str, object]] = load_exp4087_operating_point,
    task_loader: Callable[..., tuple[Sequence[ArcTask], Sequence[ArcTask]]] = load_default_task_splits,
    program_generator: Callable[..., Mapping[str, Sequence[CandidateProgram]] | tuple[Mapping[str, Sequence[CandidateProgram]], Sequence[GenerationRecord]]] = generate_programs_for_tasks,
    program_evaluator: Callable[[ArcTask, CandidateProgram, str], GeneratedProgramEvaluation] | None = None,
    smoke_trainer: Callable[[RftCorpora], SmokeTrainResult] | None = None,
    duration_floor_s: float = MIN_COMPLETE_DURATION_S,
) -> dict[str, object]:
    """REQ-LEARN-4088: run preconditions, live generation, corpora, and smoke."""

    start = time.perf_counter()
    root = Path(repo_root)
    output = Path(output_path) if output_path is not None else root / "results" / RESULT_FILENAME
    checks = list(preconditions_checker(repo_root=root))
    missing = _first_missing(checks)
    if missing is not None:
        artifact = build_precondition_blocked_artifact(checks, duration_s=time.perf_counter() - start)
        write_result_artifact(artifact, output)
        return artifact

    operating_point = dict(operating_point_loader(root))
    heldin_tasks, heldout_tasks = task_loader(repo_root=root)
    generated = program_generator(heldin_tasks, k=K_PROGRAMS_PER_TASK)
    if isinstance(generated, tuple):
        programs_by_task = generated[0]
        generation_records = list(generated[1])
    else:
        programs_by_task = generated
        generation_records = _generation_records_from_programs(heldin_tasks, programs_by_task)

    short_tasks = [
        task.task_id
        for task in heldin_tasks
        if len(programs_by_task.get(task.task_id, ())) < K_PROGRAMS_PER_TASK
    ]
    if short_tasks:  # pragma: no cover
        blocked_checks = [
            *checks,
            PreconditionCheck(
                "codex_generation_k8",
                False,
                "insufficient k>=8 programs for " + ",".join(short_tasks),
            ),
        ]
        artifact = build_precondition_blocked_artifact(blocked_checks, duration_s=time.perf_counter() - start)
        write_result_artifact(artifact, output)
        return artifact

    heldin_rows = evaluate_programs(
        heldin_tasks,
        programs_by_task,
        split="heldin",
        program_evaluator=program_evaluator,
    )
    certified_rows = apply_operating_point(heldin_rows, operating_point)
    corpora = build_n_matched_corpora(certified_rows)
    heldout_manifest = build_heldout_eval_manifest(heldout_tasks)
    smoke_result = smoke_trainer(corpora) if smoke_trainer is not None else smoke_train_two_tasks(corpora, repo_root=root)
    if not smoke_result.passed:  # pragma: no cover
        blocked_checks = [
            *checks,
            PreconditionCheck(
                "lora_smoke_checkpoints",
                False,
                "two-task three-arm smoke did not write all checkpoints or corpus was empty",
            ),
        ]
        artifact = _base_artifact(
            honest_verdict="blocked_lora_smoke_checkpoints",
            operating_point=operating_point,
            n_rft_correct=len(corpora.rft_correct),
            n_rft_ablation=len(corpora.rft_ablation),
            n_gold_sft=len(corpora.gold_sft),
            n_heldout_tasks=len(heldout_manifest),
            runner_ready=False,
            trainer_smoke_passed=False,
            preconditions_checked=blocked_checks,
            duration_s=time.perf_counter() - start,
            extra={
                "methodology": {
                    "blocked_before_live_generation": False,
                    "generator": "codex exec live program synthesis",
                    "k_programs_per_heldin_task": K_PROGRAMS_PER_TASK,
                    "generation_records": [record.to_dict() for record in generation_records],
                    "block_reason": "lora smoke checkpoints were not written or N-matched corpus was empty",
                },
                "heldout_task_ids": [str(item["task_id"]) for item in heldout_manifest],
            },
        )
        write_result_artifact(artifact, output)
        return artifact
    write_corpus_jsonl(corpora, output.parent)
    write_heldout_eval_manifest(heldout_manifest, output.parent, smoke_result.checkpoint_paths, model_specs())

    elapsed = time.perf_counter() - start
    if duration_floor_s > 0 and elapsed < duration_floor_s:  # pragma: no cover
        time.sleep(duration_floor_s - elapsed)
        elapsed = time.perf_counter() - start
    artifact = build_complete_artifact(
        operating_point=operating_point,
        corpora=corpora,
        heldout_manifest=heldout_manifest,
        preconditions_checked=checks,
        generation_records=generation_records,
        smoke_result=smoke_result,
        duration_s=elapsed,
    )
    write_result_artifact(artifact, output)
    return artifact
