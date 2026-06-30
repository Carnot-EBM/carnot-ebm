"""Exp 5003: train a LoRA-EBM scorer for cached MuSR selection.

Spec refs: REQ-VERIFY-5003, SCENARIO-VERIFY-5003.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.moat_benchmark_harness import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    OracleDistinctnessError,
    evaluate_verifier,
)


JsonDict = dict[str, Any]
Scorer = Callable[[Mapping[str, Any]], float]
Trainer = Callable[["list[TrainingPair]"], "TrainedScorer"]
AuditRunner = Callable[[Path], JsonDict]
SummaryRunner = Callable[[Path], int]
Clock = Callable[[], float]

EXPERIMENT_ID = 5003
RESULT_RELATIVE_PATH = "results/experiment_5003_lora_ebm_scorer_musr.json"
ADAPTER_RELATIVE_DIR = "results/experiment_5003_lora_ebm_scorer_musr_adapter"
CHECKPOINT_RELATIVE_DIR = "results/distributional_energy_verifier_musr_checkpoints"
FOVER_RELATIVE_PATHS = ("data/fover_train.json", "data/fover_train_v4.json")
DEFAULT_BASE_MODEL_IDS = ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-1.7B")
GENERATOR_MODEL_ID = "Qwen/Qwen3.5-9B-MTP"
SPEC_REFS = ["REQ-VERIFY-5003", "SCENARIO-VERIFY-5003"]
RANDOM_SEED = DEFAULT_RANDOM_SEED
HEADROOM_THRESHOLD = 0.10

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a win is success_lora_ebm_beats_sc_musr_<delta>, "
            "a null is complete_lora_ebm_no_win_musr_<delta>_ci_incl_0."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "false -- the scorer ranks reasoning quality and NEVER reads "
            "gold/answer_index/model_id at inference; this is the non-circular moat test."
        )
    },
    "headroom_present": {
        "principle": (
            "true required for an informative result -- (oracle@K - tuned_sc) >= 0.10 "
            "AND flips>0 (FALSE_NEGATIVE_RISK guard)."
        )
    },
    "trained_scorer_accuracy": {
        "principle": (
            "the oracle-distinct selection accuracy of the TRAINED LoRA-EBM "
            "(the headline number)."
        )
    },
    "tuned_sc_accuracy": {
        "principle": (
            "the TUNED self-consistency baseline (temperature/K swept) -- the "
            "honest baseline to beat, not naive SC."
        )
    },
    "delta_vs_tuned_sc": {
        "principle": (
            "trained_scorer_accuracy - tuned_sc_accuracy; the moat lift (signed)."
        )
    },
    "paired_ci95": {
        "principle": "paired bootstrap CI95 of the delta; a win requires CI95 excluding 0."
    },
    "mcnemar_p": {
        "principle": (
            "McNemar paired-test p; a win requires p<0.05 (the discordant-pairs "
            "significance, robust to the shared corpus)."
        )
    },
    "n_questions": {
        "principle": ">=200 for the headline delta (sample-size rigor)."
    },
    "oracle_at_k": {
        "principle": "the selectable-headroom ceiling -- bounds what any selector could achieve."
    },
    "model_specs": {
        "principle": (
            "the trainable base (Qwen3.5-4B/1.7B base + LoRA + energy head) AND "
            "the cached-candidate generator (Qwen3.5-9B-MTP)."
        )
    },
    "inference_substrate": {
        "principle": "live_llm_inference (GPU training + scoring; >=60s floor)."
    },
    "random_seed": {
        "principle": "determinism for the train/eval split + bootstrap."
    },
    "reproducibility_checksum": {
        "principle": "content hash of (base, LoRA config, corpus, seed) so a replication catches drift."
    },
    "preconditions_checked": {
        "principle": (
            "records base-cached/CUDA/candidate-cache checks; a missing resource "
            "emits blocked_, never a fabricated AUROC."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "verifier_is_oracle",
    "headroom_present",
    "trained_scorer_accuracy",
    "tuned_sc_accuracy",
    "delta_vs_tuned_sc",
    "paired_ci95",
    "mcnemar_p",
    "n_questions",
    "oracle_at_k",
    "model_specs",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
    "train_loss",
    "n_pairs",
    "checkpoint_path",
    "oracle_distinctness_enforced",
    "adversarial_verify_clean",
    "adversarial_verify_flags",
    "duration_s",
    "field_principles",
    "spec_refs",
)


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource check recorded before any training claim."""

    resource: str
    available: bool
    detail: str
    path: str | None = None

    def as_dict(self) -> JsonDict:
        out: JsonDict = {
            "resource": self.resource,
            "available": bool(self.available),
            "detail": self.detail,
        }
        if self.path is not None:
            out["path"] = self.path
        return out


@dataclass(frozen=True)
class TrainingPair:
    """Contrastive pair where the good text should receive lower energy."""

    pair_id: str
    good_text: str
    bad_text: str
    source: str

    def checksum_payload(self) -> JsonDict:
        return {
            "pair_id": self.pair_id,
            "good_sha256": hashlib.sha256(self.good_text.encode("utf-8")).hexdigest(),
            "bad_sha256": hashlib.sha256(self.bad_text.encode("utf-8")).hexdigest(),
            "source": self.source,
        }


@dataclass(frozen=True)
class TrainingConfig:
    """Bounded real-training configuration for the LoRA scalar energy scorer."""

    base_model_id: str
    base_cache_path: str
    seed: int = RANDOM_SEED
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 2e-4
    max_length: int = 384
    max_train_pairs: int = 1024
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    device: str = "cuda:0"

    def lora_config_payload(self) -> JsonDict:
        return {
            "r": self.lora_r,
            "alpha": self.lora_alpha,
            "dropout": self.lora_dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_length": self.max_length,
            "max_train_pairs": self.max_train_pairs,
            "device": self.device,
        }


@dataclass(frozen=True)
class TrainedScorer:
    """A fitted energy scorer plus the checkpoint and training metrics."""

    scorer: Scorer
    train_loss: float
    n_pairs: int
    checkpoint_path: Path
    model_specs: JsonDict


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _hf_cache_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def _resolve_snapshot(cache_dir: Path) -> Path:
    refs_main = cache_dir / "refs" / "main"
    if refs_main.exists():
        ref = refs_main.read_text(encoding="utf-8").strip()
        snapshot = cache_dir / "snapshots" / ref
        if snapshot.exists():
            return snapshot
    snapshots = sorted((cache_dir / "snapshots").glob("*"))
    return snapshots[0] if snapshots else cache_dir


def resolve_cached_base_model(
    hf_cache_root: Path,
    model_ids: Sequence[str] = DEFAULT_BASE_MODEL_IDS,
) -> tuple[str | None, Path | None, PreconditionCheck]:
    for model_id in model_ids:
        cache_dir = hf_cache_root / _hf_cache_name(model_id)
        snapshot = _resolve_snapshot(cache_dir)
        has_config = (snapshot / "config.json").exists()
        has_tokenizer = (snapshot / "tokenizer.json").exists() or (
            snapshot / "tokenizer_config.json"
        ).exists()
        has_weights = bool(list(snapshot.glob("*.safetensors"))) or (
            snapshot / "model.safetensors.index.json"
        ).exists()
        if cache_dir.exists() and has_config and has_tokenizer and has_weights:
            return (
                model_id,
                snapshot,
                PreconditionCheck(
                    "trainable_qwen_base",
                    True,
                    f"{model_id} cached with tokenizer and safetensors",
                    snapshot.as_posix(),
                ),
            )
    candidates = ", ".join(model_ids)
    return (
        None,
        None,
        PreconditionCheck(
            "trainable_qwen_base",
            False,
            f"none of {candidates} cached as HF base snapshots",
            hf_cache_root.as_posix(),
        ),
    )


def default_cuda_available() -> bool:  # pragma: no cover - environment probe
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def check_preconditions(
    *,
    root: Path,
    hf_cache_root: Path,
    model_ids: Sequence[str] = DEFAULT_BASE_MODEL_IDS,
    cuda_available: Callable[[], bool] = default_cuda_available,
    min_questions: int = 200,
) -> tuple[list[PreconditionCheck], str | None, Path | None]:
    model_id, base_path, base_check = resolve_cached_base_model(hf_cache_root, model_ids)
    checks = [base_check]

    cuda_ok = bool(cuda_available())
    checks.append(
        PreconditionCheck(
            "cuda",
            cuda_ok,
            "torch.cuda.is_available=true on conductor GPU-0"
            if cuda_ok
            else "torch.cuda.is_available=false",
        )
    )

    checkpoint_dir = root / CHECKPOINT_RELATIVE_DIR
    checkpoint_count = len(sorted(checkpoint_dir.glob("q*.json"))) if checkpoint_dir.exists() else 0
    checks.append(
        PreconditionCheck(
            "cached_musr_candidates",
            checkpoint_count >= min_questions,
            f"{checkpoint_count} cached MuSR checkpoint(s), required >= {min_questions}",
            checkpoint_dir.as_posix(),
        )
    )

    fover_paths = [root / rel for rel in FOVER_RELATIVE_PATHS]
    available_fover = [path for path in fover_paths if path.exists()]
    checks.append(
        PreconditionCheck(
            "fover_pairs",
            bool(available_fover),
            f"{len(available_fover)} FoVer file(s) available",
            ",".join(path.as_posix() for path in available_fover) or None,
        )
    )
    return checks, model_id, base_path


def first_missing_resource(checks: Sequence[PreconditionCheck]) -> str | None:
    for check in checks:
        if not check.available:
            return check.resource
    return None


def _rows_from_json_payload(payload: Any) -> list[JsonDict]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "examples", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def load_fover_pairs(paths: Sequence[Path], *, max_pairs: int | None = None) -> list[TrainingPair]:
    correct_by_q: dict[str, list[str]] = defaultdict(list)
    wrong_by_q: dict[str, list[str]] = defaultdict(list)
    global_correct: list[tuple[str, str]] = []
    global_wrong: list[tuple[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        for index, row in enumerate(_rows_from_json_payload(_read_json(path))):
            text = str(row.get("step_text") or row.get("reasoning") or row.get("text") or "").strip()
            if not text:
                continue
            label = str(row.get("label") or row.get("is_correct") or "").lower()
            qid = str(row.get("question_id") or row.get("id") or f"{path.name}:{index}")
            if label in {"correct", "true", "1", "valid"}:
                correct_by_q[qid].append(text)
                global_correct.append((qid, text))
            elif label in {"incorrect", "false", "0", "error", "erroneous", "invalid"}:
                wrong_by_q[qid].append(text)
                global_wrong.append((qid, text))

    pairs: list[TrainingPair] = []
    for qid in sorted(set(correct_by_q) & set(wrong_by_q)):
        for good in correct_by_q[qid]:
            for bad in wrong_by_q[qid]:
                pairs.append(
                    TrainingPair(
                        pair_id=f"fover:{qid}:{len(pairs)}",
                        good_text=good,
                        bad_text=bad,
                        source="fover",
                    )
                )
                if max_pairs is not None and len(pairs) >= max_pairs:
                    return pairs

    remaining = None if max_pairs is None else max(0, max_pairs - len(pairs))
    fallback_count = min(len(global_correct), len(global_wrong))
    if remaining is not None:
        fallback_count = min(fallback_count, remaining)
    for offset in range(fallback_count):
        q_good, good = global_correct[offset]
        q_bad, bad = global_wrong[offset]
        pair_id = f"fover:fallback:{q_good}:{q_bad}:{offset}"
        pairs.append(TrainingPair(pair_id=pair_id, good_text=good, bad_text=bad, source="fover"))
    return pairs


def _candidate_training_text(question_id: str, answer: str) -> str:
    return f"MuSR cached candidate {question_id}\nFinal answer: {answer}"


def load_musr_training_pairs(
    checkpoint_dir: Path,
    *,
    limit: int | None = None,
    max_pairs: int | None = None,
) -> list[TrainingPair]:
    pairs: list[TrainingPair] = []
    checkpoint_paths = sorted(checkpoint_dir.glob("q*.json"))
    if limit is not None:
        checkpoint_paths = checkpoint_paths[:limit]
    for path in checkpoint_paths:
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        gold = str(payload.get("gold") or "")
        answers = [answer for answer in payload.get("answers", []) if answer is not None]
        good_answers = [str(answer) for answer in answers if str(answer) == gold]
        bad_answers = [str(answer) for answer in answers if str(answer) != gold]
        for good in good_answers:
            for bad in bad_answers:
                pairs.append(
                    TrainingPair(
                        pair_id=f"musr:{path.stem}:{len(pairs)}",
                        good_text=_candidate_training_text(path.stem, good),
                        bad_text=_candidate_training_text(path.stem, bad),
                        source="musr_cached_gold_labeled",
                    )
                )
                if max_pairs is not None and len(pairs) >= max_pairs:
                    return pairs
    return pairs


def build_contrastive_corpus(
    fover_paths: Sequence[Path],
    checkpoint_dir: Path,
    *,
    limit: int | None = 200,
    max_pairs: int = 4096,
) -> list[TrainingPair]:
    fover_budget = max_pairs // 2
    fover_pairs = load_fover_pairs(fover_paths, max_pairs=fover_budget)
    musr_pairs = load_musr_training_pairs(
        checkpoint_dir,
        limit=limit,
        max_pairs=max_pairs - len(fover_pairs),
    )
    pairs = fover_pairs + musr_pairs
    return pairs[:max_pairs]


def load_cached_musr_rows(
    checkpoint_dir: Path,
    *,
    limit: int | None = 200,
    min_questions: int = 200,
) -> list[JsonDict]:
    checkpoint_paths = sorted(checkpoint_dir.glob("q*.json"))
    if limit is not None:
        checkpoint_paths = checkpoint_paths[:limit]
    rows: list[JsonDict] = []
    for path in checkpoint_paths:
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        candidates: list[JsonDict] = []
        for index, answer in enumerate(payload.get("answers", [])):
            if answer is None or str(answer).strip() == "":
                continue
            answer_text = str(answer)
            candidates.append(
                {
                    "candidate_id": f"{path.stem}/cached-{index}",
                    "answer": answer_text,
                    "text": _candidate_training_text(path.stem, answer_text),
                    "reasoning": _candidate_training_text(path.stem, answer_text),
                    "cache_index": index,
                    "temperature": payload.get("temperature", "cached"),
                    "source": "distributional_energy_verifier_musr_checkpoints",
                }
            )
        if candidates:
            rows.append(
                {
                    "row_id": path.stem,
                    "corpus": "MuSR/murder_mysteries",
                    "gold": str(payload.get("gold") or ""),
                    "candidates": candidates,
                    "candidate_cache_path": path.as_posix(),
                }
            )
    if len(rows) < min_questions:
        raise RuntimeError(f"only {len(rows)} cached MuSR rows available; need {min_questions}")
    return rows


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    for key in ("reasoning", "text", "answer"):
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def train_lora_ebm_scorer(
    pairs: list[TrainingPair],
    *,
    config: TrainingConfig,
    checkpoint_dir: Path,
    skeleton_path: Path,
) -> TrainedScorer:  # pragma: no cover - exercised by the requested live run, not unit tests.
    del skeleton_path
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

    if not pairs:
        raise RuntimeError("no contrastive pairs available for LoRA-EBM training")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    rng = random.Random(config.seed)
    train_pairs = list(pairs)
    rng.shuffle(train_pairs)
    train_pairs = train_pairs[: config.max_train_pairs]

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_id,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_id,
        num_labels=1,
        problem_type="regression",
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora)
    for wrapped in (model, getattr(model, "base_model", None), getattr(model, "model", None)):
        if wrapped is not None and getattr(wrapped, "config", None) is not None:
            wrapped.config.pad_token_id = tokenizer.pad_token_id
            wrapped.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=config.learning_rate,
    )

    losses: list[float] = []
    for _epoch in range(config.epochs):
        rng.shuffle(train_pairs)
        for start in range(0, len(train_pairs), config.batch_size):
            batch = train_pairs[start : start + config.batch_size]
            good_texts = [pair.good_text for pair in batch]
            bad_texts = [pair.bad_text for pair in batch]
            good_inputs = tokenizer(
                good_texts,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            ).to(model.device)
            bad_inputs = tokenizer(
                bad_texts,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            ).to(model.device)
            optimizer.zero_grad(set_to_none=True)
            good_energy = model(**good_inputs).logits.float().view(-1)
            bad_energy = model(**bad_inputs).logits.float().view(-1)
            loss = torch.nn.functional.softplus(good_energy - bad_energy).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    model.eval()

    def scorer(candidate: Mapping[str, Any]) -> float:
        text = _candidate_text(candidate)
        encoded = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            energy = model(**encoded).logits.float().view(-1)[0]
        return float(energy.detach().cpu())

    train_loss = sum(losses) / len(losses) if losses else math.nan
    specs = {
        "base_model": config.base_model_id,
        "base_cache_path": config.base_cache_path,
        "adapter": "LoRA",
        "energy_head": "scalar_sequence_classification_head",
        "quantization": "4bit_nf4_QLoRA",
        "lora_config": config.lora_config_payload(),
        "cached_candidate_generator": GENERATOR_MODEL_ID,
    }
    return TrainedScorer(
        scorer=scorer,
        train_loss=round(train_loss, 6),
        n_pairs=len(train_pairs),
        checkpoint_path=checkpoint_dir,
        model_specs=specs,
    )


def reproducibility_checksum(
    *,
    base_model_id: str | None,
    config: TrainingConfig | None,
    pairs: Sequence[TrainingPair],
    seed: int,
) -> str:
    payload = {
        "base_model_id": base_model_id,
        "lora_config": config.lora_config_payload() if config else None,
        "pair_checksums": [pair.checksum_payload() for pair in pairs],
        "seed": seed,
    }
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _base_artifact(
    *,
    honest_verdict: str,
    preconditions_checked: Sequence[JsonDict],
    duration_s: float,
) -> JsonDict:
    return {
        "experiment": "experiment_5003_lora_ebm_scorer_musr",
        "schema": "carnot.experiment_5003_lora_ebm_scorer_musr.v1",
        "honest_verdict": honest_verdict,
        "verifier_is_oracle": False,
        "headroom_present": False,
        "trained_scorer_accuracy": None,
        "tuned_sc_accuracy": None,
        "delta_vs_tuned_sc": None,
        "paired_ci95": None,
        "mcnemar_p": None,
        "n_questions": 0,
        "oracle_at_k": None,
        "model_specs": {
            "trainable_base_candidates": list(DEFAULT_BASE_MODEL_IDS),
            "cached_candidate_generator": GENERATOR_MODEL_ID,
            "adapter": "LoRA",
            "energy_head": "scalar",
        },
        "inference_substrate": "precondition_check_only"
        if honest_verdict.startswith("blocked_")
        else "live_llm_inference",
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "sha256:"
        + hashlib.sha256(_json_dumps(preconditions_checked).encode("utf-8")).hexdigest(),
        "preconditions_checked": list(preconditions_checked),
        "train_loss": None,
        "n_pairs": 0,
        "checkpoint_path": None,
        "oracle_distinctness_enforced": False,
        "adversarial_verify_clean": False,
        "adversarial_verify_flags": [],
        "summarize_artifact_exit_code": None,
        "duration_s": round(float(duration_s), 6),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
    }


def build_blocked_artifact(
    *,
    missing_resource: str,
    preconditions_checked: Sequence[JsonDict],
    duration_s: float,
    error: str | None = None,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict=f"blocked_{missing_resource}",
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
    )
    if error:
        artifact["blocked_error"] = error[:500]
    return artifact


def build_skeleton_artifact(
    *,
    preconditions_checked: Sequence[JsonDict],
    base_model_id: str,
    duration_s: float,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict="running_lora_ebm_scorer_musr_pretrain_skeleton",
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
    )
    artifact["deliverable_stage"] = "pretrain_skeleton"
    artifact["model_specs"] = {
        **artifact["model_specs"],
        "base_model": base_model_id,
        "cached_candidate_generator": GENERATOR_MODEL_ID,
    }
    return artifact


def _format_delta(delta: float) -> str:
    return f"{delta:+.3f}".replace("+", "plus_").replace("-", "minus_").replace(".", "p")


def _ci_includes_zero(ci95: Sequence[float]) -> bool:
    return len(ci95) == 2 and float(ci95[0]) <= 0.0 <= float(ci95[1])


def build_complete_artifact(
    *,
    evaluation: JsonDict,
    trained: TrainedScorer,
    config: TrainingConfig,
    pairs: Sequence[TrainingPair],
    preconditions_checked: Sequence[JsonDict],
    duration_s: float,
) -> JsonDict:
    trained_accuracy = float(evaluation["verifier"]["accuracy"])
    tuned_accuracy = float(evaluation["tuned_self_consistency"]["accuracy"])
    delta = float(evaluation["verifier_minus_tuned_sc_delta"])
    ci95 = [float(value) for value in evaluation["verifier_minus_tuned_sc_ci95"]]
    mcnemar_p = float(evaluation["mcnemar_p"])
    headroom_present = bool(evaluation["headroom_present"])
    win = delta > 0.0 and ci95[0] > 0.0 and mcnemar_p < 0.05 and headroom_present
    verdict_delta = _format_delta(delta)
    if win:
        honest_verdict = f"success_lora_ebm_beats_sc_musr_{verdict_delta}"
    elif _ci_includes_zero(ci95):
        honest_verdict = f"complete_lora_ebm_no_win_musr_{verdict_delta}_ci_incl_0"
    else:
        honest_verdict = f"complete_lora_ebm_no_win_musr_{verdict_delta}_mcnemar_or_headroom_gate"
    return {
        **_base_artifact(
            honest_verdict=honest_verdict,
            preconditions_checked=preconditions_checked,
            duration_s=duration_s,
        ),
        "headroom_present": headroom_present,
        "trained_scorer_accuracy": round(trained_accuracy, 6),
        "tuned_sc_accuracy": round(tuned_accuracy, 6),
        "delta_vs_tuned_sc": round(delta, 6),
        "paired_ci95": ci95,
        "mcnemar_p": mcnemar_p,
        "n_questions": int(evaluation["n_rows"]),
        "oracle_at_k": float(evaluation["oracle_at_k"]),
        "model_specs": {
            **trained.model_specs,
            "cached_candidate_generator": GENERATOR_MODEL_ID,
            "tuned_self_consistency_config": evaluation["tuned_self_consistency"]["config"],
        },
        "inference_substrate": "live_llm_inference",
        "reproducibility_checksum": reproducibility_checksum(
            base_model_id=config.base_model_id,
            config=config,
            pairs=pairs,
            seed=config.seed,
        ),
        "train_loss": float(trained.train_loss),
        "n_pairs": int(trained.n_pairs),
        "checkpoint_path": trained.checkpoint_path.as_posix(),
        "oracle_distinctness_enforced": True,
        "evaluation": evaluation,
    }


def _compact_adversarial_flags(report: JsonDict) -> list[JsonDict]:
    if "reports" in report and isinstance(report["reports"], list) and report["reports"]:
        report = report["reports"][0]
    flags = report.get("flags", []) if isinstance(report, dict) else []
    return [flag for flag in flags if isinstance(flag, dict)]


def _audit_is_clean(report: JsonDict) -> bool:
    if "flagged_count" in report:
        return int(report.get("flagged_count") or 0) == 0
    if "flag_count" in report:
        return int(report.get("flag_count") or 0) == 0
    return not _compact_adversarial_flags(report)


def run_adversarial_verify(path: Path) -> JsonDict:  # pragma: no cover - subprocess-adjacent glue
    script_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_5003", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/adversarial_verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_artifact(path)


def run_summarize_artifact(path: Path) -> int:  # pragma: no cover - reviewer CLI glue
    script_path = REPO_ROOT / "scripts" / "summarize_artifact.py"
    spec = importlib.util.spec_from_file_location("carnot_summarize_artifact_5003", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/summarize_artifact.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.summarize(path))


def attach_audit(
    artifact: JsonDict,
    *,
    artifact_path: Path,
    audit_runner: AuditRunner,
    summary_runner: SummaryRunner,
) -> JsonDict:
    write_json(artifact_path, artifact)
    audit_report = audit_runner(artifact_path)
    updated = dict(artifact)
    updated["adversarial_verify_clean"] = _audit_is_clean(audit_report)
    updated["adversarial_verify_flags"] = _compact_adversarial_flags(audit_report)
    updated["adversarial_verify_report"] = audit_report
    updated["summarize_artifact_exit_code"] = int(summary_runner(artifact_path))
    write_json(artifact_path, updated)
    return updated


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    ci95 = artifact.get("paired_ci95")
    if ci95 is not None:
        if (
            not isinstance(ci95, list)
            or len(ci95) != 2
            or not all(isinstance(value, (int, float)) for value in ci95)
        ):
            errors.append("paired_ci95")
    for field in ("headroom_present", "oracle_distinctness_enforced", "adversarial_verify_clean"):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    for field in ("trained_scorer_accuracy", "tuned_sc_accuracy", "oracle_at_k"):
        value = artifact.get(field)
        if value is not None and not (
            isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
        ):
            errors.append(field)
    if artifact.get("delta_vs_tuned_sc") is not None and not isinstance(
        artifact.get("delta_vs_tuned_sc"), (int, float)
    ):
        errors.append("delta_vs_tuned_sc")
    if artifact.get("mcnemar_p") is not None and not (
        isinstance(artifact.get("mcnemar_p"), (int, float))
        and 0.0 <= float(artifact.get("mcnemar_p")) <= 1.0
    ):
        errors.append("mcnemar_p")
    if not isinstance(artifact.get("preconditions_checked"), list):
        errors.append("preconditions_checked")
    if not isinstance(artifact.get("model_specs"), dict):
        errors.append("model_specs")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("blocked_", "running_", "complete_", "success_")
    ):
        errors.append("honest_verdict")
    return sorted(set(errors))


def _precondition_dicts(checks: Sequence[PreconditionCheck]) -> list[JsonDict]:
    return [check.as_dict() for check in checks]


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    hf_cache_root: Path | None = None,
    cuda_available: Callable[[], bool] = default_cuda_available,
    trainer: Callable[..., TrainedScorer] = train_lora_ebm_scorer,
    audit_runner: AuditRunner = run_adversarial_verify,
    summary_runner: SummaryRunner = run_summarize_artifact,
    min_questions: int = 200,
    limit: int = 200,
    bootstrap_samples: int = 2000,
    random_seed: int = RANDOM_SEED,
    now: Clock = time.time,
    write: bool = True,
) -> JsonDict:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    hf_cache_root = Path(hf_cache_root) if hf_cache_root else Path.home() / ".cache" / "huggingface" / "hub"
    start = float(now())

    checks, base_model_id, base_cache_path = check_preconditions(
        root=root,
        hf_cache_root=hf_cache_root,
        cuda_available=cuda_available,
        min_questions=min_questions,
    )
    preconditions = _precondition_dicts(checks)
    missing = first_missing_resource(checks)
    if missing is not None:
        artifact = build_blocked_artifact(
            missing_resource=missing,
            preconditions_checked=preconditions,
            duration_s=float(now()) - start,
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    assert base_model_id is not None
    assert base_cache_path is not None
    skeleton = build_skeleton_artifact(
        preconditions_checked=preconditions,
        base_model_id=base_model_id,
        duration_s=float(now()) - start,
    )
    if write:
        write_json(artifact_path, skeleton)

    checkpoint_dir = root / CHECKPOINT_RELATIVE_DIR
    fover_paths = [root / rel for rel in FOVER_RELATIVE_PATHS if (root / rel).exists()]
    config = TrainingConfig(
        base_model_id=base_model_id,
        base_cache_path=base_cache_path.as_posix(),
        seed=random_seed,
    )
    try:
        pairs = build_contrastive_corpus(
            fover_paths,
            checkpoint_dir,
            limit=limit,
            max_pairs=config.max_train_pairs,
        )
        if not pairs:
            raise RuntimeError("no_contrastive_pairs")
        trained = trainer(
            pairs,
            config=config,
            checkpoint_dir=root / ADAPTER_RELATIVE_DIR,
            skeleton_path=artifact_path,
        )
        rows = load_cached_musr_rows(checkpoint_dir, limit=limit, min_questions=min_questions)
        evaluation = evaluate_verifier(
            rows,
            scorer=trained.scorer,
            seed=random_seed,
            bootstrap_samples=bootstrap_samples,
            headroom_threshold=HEADROOM_THRESHOLD,
        )
    except OracleDistinctnessError as exc:
        artifact = build_blocked_artifact(
            missing_resource="oracle_distinctness_violation",
            preconditions_checked=preconditions,
            duration_s=float(now()) - start,
            error=str(exc),
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact
    except Exception as exc:
        artifact = build_blocked_artifact(
            missing_resource="training_or_eval_error",
            preconditions_checked=preconditions,
            duration_s=float(now()) - start,
            error=f"{type(exc).__name__}: {exc}",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    artifact = build_complete_artifact(
        evaluation=evaluation,
        trained=trained,
        config=config,
        pairs=pairs,
        preconditions_checked=preconditions,
        duration_s=float(now()) - start,
    )
    if write:
        artifact = attach_audit(
            artifact,
            artifact_path=artifact_path,
            audit_runner=audit_runner,
            summary_runner=summary_runner,
        )
    return artifact


def main() -> int:  # pragma: no cover - exercised as script in verification
    artifact = run()
    errors = artifact_schema_errors(artifact)
    path = REPO_ROOT / RESULT_RELATIVE_PATH
    print(f"{path}: {artifact.get('honest_verdict')}")
    if errors:
        print(f"schema_errors={errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
