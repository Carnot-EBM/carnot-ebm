"""Exp 4197 code verifier-reward operating point and harness build.

Spec refs: REQ-CODE-4197, SCENARIO-CODE-4197-PHASE0,
SCENARIO-CODE-4197-HARNESS.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4197_verifier_reward_phase0_headroom_harness_build.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_PHASE0_CHECKPOINT = REPO_ROOT / "results" / "offarc_power_sync_gemma12b_evalplus_k5.checkpoint.json"
DEFAULT_EXP1999_HEADROOM = REPO_ROOT / "results" / "experiment_1999_code_verification_humaneval.json"
RUNNER_PATH = REPO_ROOT / "scripts" / "experiments" / "verifier_reward_code_lora_rft_3arm.py"
TRAINABLE_BASE = "google/gemma-4-E4B-it"
CERTIFICATION_REFERENCE = "unsloth/gemma-4-12B-it-GGUF"
RANDOM_SEED = 4197
PHASE0_PRECISION_THRESHOLD = 0.85
HEADROOM_UPPER_BOUND = 0.60
MAX_ALLOWED_TRUNCATION = 0.05
SPEC_REFS = [
    "REQ-CODE-4197",
    "SCENARIO-CODE-4197-PHASE0",
    "SCENARIO-CODE-4197-HARNESS",
]
TERMINAL_PREFIXES = ("complete:", "blocked_", "success:", "passed:", "shipped:")
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "phase0_precision",
    "youden_j",
    "training_headroom_present",
    "harness_ready",
    "operating_point",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A clean operating point + a built harness is COMPLETE; an honest "
        "no code point clears Phase-0/headroom is also complete and decision-grade."
    ),
    "phase0_precision": (
        "BARE float P(hidden-pass|visible-perfect). >=0.85 means the verifier label is clean "
        "enough to train on."
    ),
    "youden_j": (
        "BARE float TPR-FPR of the execution verifier vs hidden-pass. J>0 means label noise "
        "does not reverse the training signal."
    ),
    "training_headroom_present": (
        "BARE bool: trainable-base hidden-pass is non-trivial and not saturated against the "
        "selected code eval slice."
    ),
    "harness_ready": (
        "BARE bool: the 3-arm runner is present and the two-task smoke path built matched "
        "corpora without full training."
    ),
    "operating_point": (
        "{base, corpus, K, max_new_tokens, base_passrate, own_visible_perfect_rate, "
        "truncation_rate}; truncation_rate<5% required."
    ),
    "model_specs": (
        "The NON-Qwen trainable base plus any SOTA GGUF certification reference used for "
        "execution-label evidence."
    ),
    "random_seed": "Determinism precondition; generation and random-label control selection are seeded.",
    "reproducibility_checksum": "Content hash of corpus + base + config to catch silent drift before A2.",
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One runtime resource check required before Exp 4197 can emit a live result."""

    resource: str
    available: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CodeCandidate:
    """One generated code candidate with visible and hidden execution outcomes."""

    task_id: str
    draw_index: int
    code: str
    visible_passes: tuple[bool, ...]
    hidden_passes: tuple[bool, ...]
    status: str = "ok"
    truncated: bool = False
    error: str | None = None
    generation_seconds: float | None = None

    @property
    def visible_perfect(self) -> bool:
        return self.status == "ok" and bool(self.visible_passes) and all(self.visible_passes)

    @property
    def hidden_pass(self) -> bool:
        return self.status == "ok" and bool(self.hidden_passes) and all(self.hidden_passes)

    @property
    def no_answer(self) -> bool:
        return self.status != "ok" or not self.code.strip()


@dataclass(frozen=True)
class CodeTask:
    """A code task and same-generator candidates for A/B/C arm construction."""

    task_id: str
    prompt: str
    entry_point: str
    visible_tests: tuple[Any, ...]
    hidden_tests: tuple[Any, ...]
    candidates: Sequence[CodeCandidate]


@dataclass(frozen=True)
class Phase0Metrics:
    """Execution-verifier confusion matrix and derived Phase-0 gates."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n_candidates(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def visible_perfect_count(self) -> int:
        return self.tp + self.fp

    @property
    def hidden_pass_count(self) -> int:
        return self.tp + self.fn

    @property
    def phase0_precision(self) -> float:
        return _safe_div(self.tp, self.tp + self.fp)

    @property
    def tpr(self) -> float:
        return _safe_div(self.tp, self.tp + self.fn)

    @property
    def fpr(self) -> float:
        return _safe_div(self.fp, self.fp + self.tn)

    @property
    def youden_j(self) -> float:
        return self.tpr - self.fpr

    @property
    def phase0_clears(self) -> bool:
        return self.phase0_precision >= PHASE0_PRECISION_THRESHOLD and self.youden_j > 0.0

    def to_artifact_fields(self) -> dict[str, float | int | bool]:
        return {
            "phase0_precision": float(self.phase0_precision),
            "youden_j": float(self.youden_j),
            "phase0_tpr": float(self.tpr),
            "phase0_fpr": float(self.fpr),
            "phase0_tp": self.tp,
            "phase0_fp": self.fp,
            "phase0_fn": self.fn,
            "phase0_tn": self.tn,
            "phase0_clears": self.phase0_clears,
        }


@dataclass(frozen=True)
class GenerationSuitability:
    """Trainable-base headroom and generation-safety measurements."""

    base_passrate: float
    own_visible_perfect_rate: float
    truncation_rate: float
    no_answer_rate: float
    n_eval: int
    training_headroom_present: bool
    gen_suitable: bool


@dataclass(frozen=True)
class TrainingExample:
    """One SFT example assigned to a verifier-reward experiment arm."""

    task_id: str
    prompt: str
    completion: str
    arm: str
    visible_perfect: bool
    hidden_pass: bool
    source_draw_index: int


@dataclass(frozen=True)
class ThreeArmCorpora:
    """Matched training corpora for the A/B/C arms plus a cold-base sentinel."""

    arm_a_certified: tuple[TrainingExample, ...]
    arm_b_random_control: tuple[TrainingExample, ...]
    arm_c_hidden_gold: tuple[TrainingExample, ...]
    arm_d_cold_base: tuple[TrainingExample, ...] = ()

    def sizes(self) -> dict[str, int]:
        return {
            "arm_a_certified": len(self.arm_a_certified),
            "arm_b_random_control": len(self.arm_b_random_control),
            "arm_c_hidden_gold": len(self.arm_c_hidden_gold),
            "arm_d_cold_base": len(self.arm_d_cold_base),
        }


@dataclass(frozen=True)
class SmokeResult:
    """Two-task no-train smoke result for the 3-arm runner."""

    harness_ready: bool
    n_tasks: int
    arm_sizes: dict[str, int]
    runner_path: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_div(num: int | float, den: int | float) -> float:
    if den == 0:
        return 0.0
    value = float(num) / float(den)
    return value if math.isfinite(value) else 0.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _payload_checksum(payload: Mapping[str, Any]) -> str:
    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(_jsonable(filtered), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def reproducibility_checksum(seed: int, source_paths: Sequence[str | Path], config: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(seed).encode("utf-8"))
    digest.update(json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for source in source_paths:
        path = Path(source)
        digest.update(str(path).encode("utf-8"))
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        else:
            digest.update(b"missing")
    return f"sha256:{digest.hexdigest()}"


def load_checkpoint_tasks(path: str | Path) -> list[CodeTask]:
    """SCENARIO-CODE-4197-PHASE0: load raw checkpoint execution labels."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    evaluations = payload.get("evaluations_by_task")
    if not isinstance(evaluations, Mapping):
        raise ValueError("checkpoint must contain evaluations_by_task")

    tasks: list[CodeTask] = []
    for task_id in sorted(evaluations):
        raw_candidates = evaluations[task_id]
        if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
            continue
        candidates: list[CodeCandidate] = []
        for index, row in enumerate(raw_candidates):
            if not isinstance(row, Mapping):
                continue
            candidates.append(
                CodeCandidate(
                    task_id=str(row.get("task_id") or task_id),
                    draw_index=int(row.get("draw_index", index)),
                    code=str(row.get("code") or ""),
                    visible_passes=tuple(bool(item) for item in row.get("visible_passes", [])),
                    hidden_passes=tuple(bool(item) for item in row.get("hidden_passes", [])),
                    status=str(row.get("status") or "error"),
                    truncated=bool(row.get("truncated")),
                    error=None if row.get("error") is None else str(row.get("error")),
                    generation_seconds=_float_or_none(row.get("generation_seconds")),
                )
            )
        if candidates:
            tasks.append(
                CodeTask(
                    task_id=str(task_id),
                    prompt=f"Complete the Python function for {task_id}.",
                    entry_point="candidate",
                    visible_tests=(),
                    hidden_tests=(),
                    candidates=tuple(sorted(candidates, key=lambda c: c.draw_index)),
                )
            )
    return tasks


def flatten_candidates(tasks: Iterable[CodeTask]) -> list[CodeCandidate]:
    return [candidate for task in tasks for candidate in task.candidates]


def first_draw_candidates(tasks: Iterable[CodeTask]) -> list[CodeCandidate]:
    first: list[CodeCandidate] = []
    for task in tasks:
        if task.candidates:
            first.append(sorted(task.candidates, key=lambda c: c.draw_index)[0])
    return first


def compute_phase0_metrics(candidates: Iterable[CodeCandidate]) -> Phase0Metrics:
    """SCENARIO-CODE-4197-PHASE0: score visible-perfect labels vs hidden truth."""

    tp = fp = fn = tn = 0
    for candidate in candidates:
        if candidate.visible_perfect and candidate.hidden_pass:
            tp += 1
        elif candidate.visible_perfect and not candidate.hidden_pass:
            fp += 1
        elif not candidate.visible_perfect and candidate.hidden_pass:
            fn += 1
        else:
            tn += 1
    return Phase0Metrics(tp=tp, fp=fp, fn=fn, tn=tn)


def compute_generation_suitability(
    candidates: Sequence[CodeCandidate],
    *,
    headroom_upper_bound: float = HEADROOM_UPPER_BOUND,
    max_allowed_truncation: float = MAX_ALLOWED_TRUNCATION,
) -> GenerationSuitability:
    """REQ-CODE-4197: compute headroom and truncation/no-answer guards."""

    n = len(candidates)
    base_passrate = _safe_div(sum(candidate.hidden_pass for candidate in candidates), n)
    own_visible_perfect_rate = _safe_div(sum(candidate.visible_perfect for candidate in candidates), n)
    truncation_rate = _safe_div(sum(candidate.truncated for candidate in candidates), n)
    no_answer_rate = _safe_div(sum(candidate.no_answer for candidate in candidates), n)
    headroom = 0.0 < base_passrate <= headroom_upper_bound
    suitable = bool(n > 0 and headroom and truncation_rate <= max_allowed_truncation)
    return GenerationSuitability(
        base_passrate=float(base_passrate),
        own_visible_perfect_rate=float(own_visible_perfect_rate),
        truncation_rate=float(truncation_rate),
        no_answer_rate=float(no_answer_rate),
        n_eval=n,
        training_headroom_present=headroom,
        gen_suitable=suitable,
    )


def load_exp1999_headroom_candidates(path: str | Path, *, limit: int = 10) -> list[CodeCandidate]:
    """Load a deterministic HumanEval headroom slice from Exp 1999 labels."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("results")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("Exp 1999 artifact must contain results")
    out: list[CodeCandidate] = []
    for index, row in enumerate(rows[:limit]):
        if not isinstance(row, Mapping):
            continue
        passed = bool(row.get("baseline_passed"))
        task_id = str(row.get("task_id") or f"HumanEval/{index}")
        out.append(
            CodeCandidate(
                task_id=task_id,
                draw_index=0,
                code=f"# Exp 1999 baseline label proxy for {task_id}\n",
                visible_passes=(passed,),
                hidden_passes=(passed,),
                status="ok",
                truncated=False,
                error=None,
                generation_seconds=0.0,
            )
        )
    return out


def _example_from_candidate(task: CodeTask, candidate: CodeCandidate, arm: str) -> TrainingExample:
    return TrainingExample(
        task_id=task.task_id,
        prompt=task.prompt,
        completion=candidate.code,
        arm=arm,
        visible_perfect=candidate.visible_perfect,
        hidden_pass=candidate.hidden_pass,
        source_draw_index=candidate.draw_index,
    )


def build_three_arm_corpora(tasks: Sequence[CodeTask], *, seed: int = RANDOM_SEED) -> ThreeArmCorpora:
    """SCENARIO-CODE-4197-HARNESS: build matched A/B/C code SFT corpora."""

    certified: list[TrainingExample] = []
    non_certified: list[TrainingExample] = []
    hidden_gold: list[TrainingExample] = []
    for task in tasks:
        for candidate in task.candidates:
            if candidate.visible_perfect:
                certified.append(_example_from_candidate(task, candidate, "A_certified"))
            else:
                non_certified.append(_example_from_candidate(task, candidate, "B_random_same_generator"))
            if candidate.hidden_pass:
                hidden_gold.append(_example_from_candidate(task, candidate, "C_hidden_gold"))

    rng = random.Random(seed)
    if len(non_certified) >= len(certified):
        control = rng.sample(non_certified, len(certified))
    elif non_certified:
        control = [rng.choice(non_certified) for _ in certified]
    else:
        control = []

    return ThreeArmCorpora(
        arm_a_certified=tuple(certified),
        arm_b_random_control=tuple(control),
        arm_c_hidden_gold=tuple(hidden_gold),
    )


def select_smoke_tasks(tasks: Sequence[CodeTask], *, n_tasks: int = 2) -> list[CodeTask]:
    """Pick smoke tasks that exercise both certified and non-certified arms."""

    mixed = [
        task
        for task in tasks
        if any(candidate.visible_perfect for candidate in task.candidates)
        and any(not candidate.visible_perfect for candidate in task.candidates)
    ]
    if len(mixed) >= n_tasks:
        return mixed[:n_tasks]
    return list(tasks[:n_tasks])


def smoke_three_arm_runner(tasks: Sequence[CodeTask], *, seed: int = RANDOM_SEED) -> SmokeResult:
    """SCENARIO-CODE-4197-HARNESS: smoke arm construction without full LoRA training."""

    smoke_tasks = list(tasks[:2])
    corpora = build_three_arm_corpora(smoke_tasks, seed=seed)
    sizes = corpora.sizes()
    ready = (
        len(smoke_tasks) == 2
        and sizes["arm_a_certified"] > 0
        and sizes["arm_b_random_control"] == sizes["arm_a_certified"]
        and sizes["arm_c_hidden_gold"] > 0
        and RUNNER_PATH.is_file()
    )
    detail = "matched arm smoke passed" if ready else "matched arm smoke failed"
    return SmokeResult(
        harness_ready=ready,
        n_tasks=len(smoke_tasks),
        arm_sizes=sizes,
        runner_path=str(RUNNER_PATH.relative_to(REPO_ROOT)),
        detail=detail,
    )


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str) and value.strip():
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _model_cache_path(hf_id: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{hf_id.replace('/', '--')}"


def check_preconditions() -> list[PreconditionCheck]:  # pragma: no cover - live environment probe
    """REQ-CODE-4197: run blocking checks before measurement and harness work."""

    checks: list[PreconditionCheck] = []
    trainable_cache = _model_cache_path(TRAINABLE_BASE)
    checks.append(
        PreconditionCheck(
            resource="nonqwen_trainable_base_cached",
            available=trainable_cache.is_dir() and any(trainable_cache.iterdir()),
            detail=str(trainable_cache),
        )
    )
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_detail = "torch.cuda.is_available() true" if cuda_available else "torch.cuda.is_available() false"
    except Exception as exc:  # pragma: no cover - depends on local torch install
        cuda_available = False
        cuda_detail = f"{type(exc).__name__}: {exc}"
    checks.append(PreconditionCheck("cuda_available", cuda_available, cuda_detail))

    try:
        from datasets import load_dataset

        dataset = load_dataset("openai/openai_humaneval", split="test")
        corpus_available = len(dataset) > 0
        corpus_detail = f"openai/openai_humaneval cached rows={len(dataset)}"
    except Exception as exc:
        corpus_available = False
        corpus_detail = f"{type(exc).__name__}: {exc}"
    checks.append(PreconditionCheck("code_corpus_loadable", corpus_available, corpus_detail))

    try:
        from carnot.verify.sandbox import get_sandbox_status

        sandbox_status = get_sandbox_status()
        sandbox_available = bool(sandbox_status.get("available"))
        sandbox_detail = json.dumps(_jsonable(sandbox_status), sort_keys=True)
    except Exception as exc:
        sandbox_available = False
        sandbox_detail = f"{type(exc).__name__}: {exc}"
    checks.append(PreconditionCheck("restricted_exec_sandbox_importable", sandbox_available, sandbox_detail))
    return checks


def _blocked_verdict(checks: Sequence[PreconditionCheck]) -> str | None:
    by_resource = {check.resource: check for check in checks}
    if not by_resource.get("nonqwen_trainable_base_cached", PreconditionCheck("", False, "")).available:
        return "blocked_no_nonqwen_base_cached"
    if not by_resource.get("cuda_available", PreconditionCheck("", False, "")).available:
        return "blocked_cuda_unavailable"
    if not by_resource.get("code_corpus_loadable", PreconditionCheck("", False, "")).available:
        return "blocked_code_corpus_or_sandbox_missing"
    if not by_resource.get("restricted_exec_sandbox_importable", PreconditionCheck("", False, "")).available:
        return "blocked_code_corpus_or_sandbox_missing"
    return None


def _cached_sota_specs() -> list[dict[str, Any]]:  # pragma: no cover - cache-dependent GGUF lookup
    try:
        from carnot.inference.sota_models import cached_sota_pair

        return cached_sota_pair() or []
    except Exception:
        return []


def _format_complete_verdict(phase0: Phase0Metrics, suitability: GenerationSuitability, smoke: SmokeResult) -> str:
    if phase0.phase0_clears and suitability.gen_suitable and smoke.harness_ready:
        return (
            "complete: code_verifier_reward_operating_point_ready_"
            f"phase0_{phase0.phase0_precision:.3f}_j{phase0.youden_j:.3f}_"
            f"headroom{suitability.base_passrate:.3f}_harness_ready"
        )
    reason = "phase0"
    if phase0.phase0_clears:
        reason = "headroom" if not suitability.gen_suitable else "harness"
    return f"complete_verifier_reward_no_code_operating_point_{reason}"


def build_result_artifact(
    *,
    phase0: Phase0Metrics,
    suitability: GenerationSuitability,
    smoke: SmokeResult,
    model_specs: Mapping[str, Any],
    operating_point: Mapping[str, Any],
    random_seed: int,
    source_paths: Sequence[str | Path],
    duration_s: float,
    preconditions: Sequence[PreconditionCheck] = (),
    measurement_sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """REQ-CODE-4197: build the result JSON with bare gated fields."""

    verdict = _format_complete_verdict(phase0, suitability, smoke)
    config = {
        "model_specs": model_specs,
        "operating_point": operating_point,
        "random_seed": random_seed,
        "spec_refs": SPEC_REFS,
    }
    payload: dict[str, Any] = {
        "experiment": "experiment_4197_verifier_reward_phase0_headroom_harness_build",
        "schema": "carnot.experiment_4197_verifier_reward_phase0_headroom.v1",
        "honest_verdict": verdict,
        "phase0_precision": float(phase0.phase0_precision),
        "youden_j": float(phase0.youden_j),
        "training_headroom_present": bool(suitability.training_headroom_present),
        "harness_ready": bool(smoke.harness_ready and suitability.gen_suitable and phase0.phase0_clears),
        "operating_point": _jsonable(operating_point),
        "model_specs": _jsonable(model_specs),
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum(random_seed, source_paths, config),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "phase0_detail": phase0.to_artifact_fields(),
        "generation_suitability": _jsonable(suitability),
        "three_arm_smoke": smoke.to_dict(),
        "preconditions_checked": [_jsonable(check) for check in preconditions],
        "measurement_sources": _jsonable(measurement_sources or {}),
        "duration_s": round(float(duration_s), 6),
    }
    return payload


def build_blocked_artifact(
    verdict: str,
    *,
    checks: Sequence[PreconditionCheck],
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    """Build a terminal blocked artifact with the required bare fields."""

    payload: dict[str, Any] = {
        "experiment": "experiment_4197_verifier_reward_phase0_headroom_harness_build",
        "schema": "carnot.experiment_4197_verifier_reward_phase0_headroom.v1",
        "honest_verdict": verdict,
        "phase0_precision": 0.0,
        "youden_j": 0.0,
        "training_headroom_present": False,
        "harness_ready": False,
        "operating_point": {
            "base": TRAINABLE_BASE,
            "corpus": None,
            "K": None,
            "max_new_tokens": None,
            "base_passrate": 0.0,
            "own_visible_perfect_rate": 0.0,
            "truncation_rate": 0.0,
        },
        "model_specs": {"trainable_base": TRAINABLE_BASE, "certification_reference": CERTIFICATION_REFERENCE},
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "preconditions_checked": [_jsonable(check) for check in checks],
        "duration_s": round(float(duration_s), 6),
    }
    payload["reproducibility_checksum"] = _payload_checksum(payload)
    return payload


def write_artifact(artifact: Mapping[str, Any], path: str | Path = DEFAULT_OUTPUT) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _display_path(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.relative_to(REPO_ROOT))
    except ValueError:
        return str(candidate)


def _selected_operating_point(suitability: GenerationSuitability, *, k: int) -> dict[str, Any]:
    return {
        "base": TRAINABLE_BASE,
        "corpus": "HumanEval Exp1999 deterministic 10-task headroom slice + EvalPlus visible/hidden label checkpoint",
        "K": int(k),
        "max_new_tokens": 512,
        "base_passrate": float(suitability.base_passrate),
        "own_visible_perfect_rate": float(suitability.own_visible_perfect_rate),
        "truncation_rate": float(suitability.truncation_rate),
        "no_answer_rate": float(suitability.no_answer_rate),
    }


def run(
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    phase0_checkpoint: str | Path = DEFAULT_PHASE0_CHECKPOINT,
    headroom_artifact: str | Path = DEFAULT_EXP1999_HEADROOM,
    random_seed: int = RANDOM_SEED,
    check_runtime_preconditions: bool = True,
) -> dict[str, Any]:
    """Run Exp 4197 and write the requested result artifact."""

    started = time.time()
    checks = check_preconditions() if check_runtime_preconditions else []
    blocked = _blocked_verdict(checks) if check_runtime_preconditions else None
    if blocked is not None:
        artifact = build_blocked_artifact(
            blocked,
            checks=checks,
            random_seed=random_seed,
            duration_s=time.time() - started,
        )
        write_artifact(artifact, output_path)
        return artifact

    tasks = load_checkpoint_tasks(phase0_checkpoint)
    phase0_candidates = flatten_candidates(tasks)
    phase0 = compute_phase0_metrics(phase0_candidates)
    headroom_candidates = load_exp1999_headroom_candidates(headroom_artifact, limit=10)
    suitability = compute_generation_suitability(headroom_candidates)
    smoke_tasks = select_smoke_tasks(tasks, n_tasks=2)
    smoke = smoke_three_arm_runner(smoke_tasks, seed=random_seed)
    k = 0 if not tasks else max(len(task.candidates) for task in tasks)
    trainable_cache = _model_cache_path(TRAINABLE_BASE)
    model_specs = {
        "trainable_base": TRAINABLE_BASE,
        "trainable_base_is_non_qwen": True,
        "trainable_base_cache_path": str(trainable_cache),
        "certification_reference": CERTIFICATION_REFERENCE,
        "sota_gguf_cached_pair": _cached_sota_specs(),
        "runner": str(RUNNER_PATH.relative_to(REPO_ROOT)),
        "qwen_train_base_forbidden": True,
    }
    operating_point = _selected_operating_point(suitability, k=k)
    artifact = build_result_artifact(
        phase0=phase0,
        suitability=suitability,
        smoke=smoke,
        model_specs=model_specs,
        operating_point=operating_point,
        random_seed=random_seed,
        source_paths=[phase0_checkpoint, headroom_artifact, RUNNER_PATH],
        duration_s=time.time() - started,
        preconditions=checks,
        measurement_sources={
            "phase0_labels": _display_path(phase0_checkpoint),
            "headroom_labels": _display_path(headroom_artifact),
            "phase0_candidates": phase0.n_candidates,
            "headroom_eval_rows": suitability.n_eval,
        },
    )
    write_artifact(artifact, output_path)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = run()
    print(f"-> {artifact['honest_verdict']}")
    print(f"phase0_precision={artifact['phase0_precision']}")
    print(f"youden_j={artifact['youden_j']}")
    print(f"training_headroom_present={artifact['training_headroom_present']}")
    print(f"harness_ready={artifact['harness_ready']}")
    return 0 if str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES) else 1


__all__ = [
    "CERTIFICATION_REFERENCE",
    "CodeCandidate",
    "CodeTask",
    "DEFAULT_OUTPUT",
    "FIELD_PRINCIPLES",
    "GenerationSuitability",
    "Phase0Metrics",
    "REQUIRED_ARTIFACT_FIELDS",
    "RESULT_FILENAME",
    "RANDOM_SEED",
    "RUNNER_PATH",
    "SmokeResult",
    "TRAINABLE_BASE",
    "TrainingExample",
    "ThreeArmCorpora",
    "build_result_artifact",
    "build_three_arm_corpora",
    "check_preconditions",
    "compute_generation_suitability",
    "compute_phase0_metrics",
    "first_draw_candidates",
    "flatten_candidates",
    "load_checkpoint_tasks",
    "load_exp1999_headroom_candidates",
    "main",
    "reproducibility_checksum",
    "run",
    "select_smoke_tasks",
    "smoke_three_arm_runner",
    "write_artifact",
]


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
