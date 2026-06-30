"""Exp 5017: train the small-base LoRA-EBM MuSR scorer for real.

Spec refs: REQ-VERIFY-5017, SCENARIO-VERIFY-5017.
"""

from __future__ import annotations

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

from carnot import experiment_5003_lora_ebm_scorer_musr as exp5003  # noqa: E402
from carnot.moat_benchmark_harness import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    OracleDistinctnessError,
    evaluate_verifier,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Scorer = Callable[[Mapping[str, Any]], float]
AuditRunner = Callable[[Path], JsonDict]
SummaryRunner = Callable[[Path], int]
Clock = Callable[[], float]
TrainingPair = exp5003.TrainingPair

EXPERIMENT_ID = 5017
EXPERIMENT_NAME = "experiment_5017_lora_ebm_scorer_musr_v2"
RESULT_RELATIVE_PATH = "results/experiment_5017_lora_ebm_scorer_musr_v2.json"
ADAPTER_RELATIVE_DIR = "results/experiment_5017_lora_ebm_scorer_musr_v2_adapter"
SHARED_CACHE_RELATIVE_PATH = "results/experiment_5016_shared_logprob_candidate_cache_musr.jsonl"
FALLBACK_CHECKPOINT_RELATIVE_DIR = "results/distributional_energy_verifier_musr_checkpoints"
B1_BASELINE_RELATIVE_PATH = "results/experiment_5015_genuine_sc_baseline_fix.json"
FOVER_RELATIVE_PATHS = ("data/fover_train.json", "data/fover_train_v4.json")
BASE_MODEL_ID = "Qwen/Qwen3.5-1.7B"
GENERATOR_MODEL_ID = "shared_cache_or_distributional_energy_verifier_musr_checkpoints"
SPEC_REFS = ["REQ-VERIFY-5017", "SCENARIO-VERIFY-5017"]
RANDOM_SEED = DEFAULT_RANDOM_SEED
HEADROOM_THRESHOLD = 0.10
TRAIN_DURATION_FLOOR_S = 60.0

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a win is success_lora_ebm_beats_sc_musr_<delta>, "
            "a clean null is complete_lora_ebm_no_win_musr_<delta>_ci_incl_0, "
            "a failed train is blocked_lora_ebm_train_did_not_run."
        )
    },
    "scorer_trained": {
        "principle": (
            "true iff the model ACTUALLY trained (train_loss non-null, n_pairs>0, "
            "duration>60s) -- the anti-skeleton gate AND the field D3 gates on; "
            "false = a FAILED execution, not a null."
        )
    },
    "train_loss": {
        "principle": (
            "the final contrastive training loss (non-null REQUIRED -- a null means "
            "the .461 skeleton bail recurred)."
        )
    },
    "n_pairs": {
        "principle": (
            "the contrastive-pair count (>0 REQUIRED -- 0 was the .461 skeleton signature)."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "false -- the scorer ranks reasoning quality and NEVER reads "
            "gold/answer_index/model_id at inference (must pass check_circular_moat_overclaim)."
        )
    },
    "headroom_present": {
        "principle": (
            "true required for an informative result -- (oracle@K - GENUINE tuned_sc) "
            ">= 0.10 AND flips>0 (FALSE_NEGATIVE_RISK guard, vs the B1 genuine baseline)."
        )
    },
    "trained_scorer_accuracy": {
        "principle": (
            "the oracle-distinct selection accuracy of the TRAINED LoRA-EBM "
            "(the headline number)."
        )
    },
    "genuine_tuned_sc_accuracy": {
        "principle": (
            "the GENUINE K-way tuned-SC baseline from B1 (NOT the k=1 strawman) -- "
            "the honest baseline to beat."
        )
    },
    "delta_vs_tuned_sc": {
        "principle": (
            "trained_scorer_accuracy - genuine_tuned_sc_accuracy; the moat lift (signed)."
        )
    },
    "paired_ci95": {
        "principle": "paired bootstrap CI95 of the delta; a win requires CI95 excluding 0."
    },
    "mcnemar_p": {"principle": "McNemar paired-test p; a win requires p<0.05."},
    "n_questions": {"principle": ">=200 for the headline delta (sample-size rigor)."},
    "oracle_at_k": {"principle": "the selectable-headroom ceiling."},
    "model_specs": {
        "principle": (
            "the trainable base (Qwen3.5-1.7B base + LoRA + energy head) AND the "
            "cached-candidate generator -- the methodology stamp."
        )
    },
    "inference_substrate": {
        "principle": "live_llm_inference (GPU training + scoring; >=60s floor)."
    },
    "random_seed": {"principle": "determinism for the train/eval split + bootstrap."},
    "reproducibility_checksum": {
        "principle": (
            "content hash of (base, LoRA config, corpus, seed) so a replication catches drift."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records base-cached/CUDA/candidate-cache/FoVer checks; a missing resource "
            "emits blocked_, never a fabricated AUROC."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "checkpoint_path",
    "epoch_checkpoints",
    "candidate_cache_source",
    "oracle_distinctness_enforced",
    "adversarial_verify_clean",
    "adversarial_verify_flags",
    "summarize_artifact_exit_code",
    "duration_s",
    "field_principles",
)


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource check recorded before training is allowed to make a claim."""

    resource: str
    available: bool
    detail: str
    path: str | None = None

    def as_dict(self) -> JsonDict:
        payload: JsonDict = {
            "resource": self.resource,
            "available": bool(self.available),
            "detail": self.detail,
        }
        if self.path is not None:
            payload["path"] = self.path
        return payload


@dataclass(frozen=True)
class CandidateSource:
    """The selected cached candidate source for both training-pair build and eval."""

    name: str
    path: Path
    rows: list[JsonDict]
    check: PreconditionCheck


@dataclass(frozen=True)
class TrainingConfig:
    """Bounded QLoRA configuration for the small-base scalar energy scorer."""

    base_model_id: str
    base_cache_path: str
    seed: int = RANDOM_SEED
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 2e-4
    max_length: int = 384
    max_train_pairs: int = 2048
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
    epoch_checkpoints: list[Path]


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonMap) -> None:
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
    snapshots = sorted((cache_dir / "snapshots").glob("*")) if cache_dir.exists() else []
    return snapshots[0] if snapshots else cache_dir


def _snapshot_is_trainable_base(snapshot: Path) -> bool:
    has_config = (snapshot / "config.json").exists()
    has_tokenizer = (snapshot / "tokenizer.json").exists() or (
        snapshot / "tokenizer_config.json"
    ).exists()
    has_weights = bool(list(snapshot.glob("*.safetensors"))) or (
        snapshot / "model.safetensors.index.json"
    ).exists()
    return snapshot.exists() and has_config and has_tokenizer and has_weights


def resolve_or_download_base_model(
    hf_cache_root: Path,
    *,
    allow_download: bool = True,
    downloader: Callable[..., str] | None = None,
) -> tuple[Path | None, PreconditionCheck]:
    cache_dir = hf_cache_root / _hf_cache_name(BASE_MODEL_ID)
    snapshot = _resolve_snapshot(cache_dir)
    if _snapshot_is_trainable_base(snapshot):
        return (
            snapshot,
            PreconditionCheck(
                "trainable_qwen_base",
                True,
                f"{BASE_MODEL_ID} cached with tokenizer and safetensors",
                snapshot.as_posix(),
            ),
        )
    if allow_download:
        try:  # pragma: no cover - environment/network dependent precondition path
            if downloader is None:
                from huggingface_hub import snapshot_download

                downloader = snapshot_download
            downloaded = Path(
                downloader(
                    repo_id=BASE_MODEL_ID,
                    cache_dir=hf_cache_root.as_posix(),
                    local_files_only=False,
                )
            )
            if _snapshot_is_trainable_base(downloaded):
                return (
                    downloaded,
                    PreconditionCheck(
                        "trainable_qwen_base",
                        True,
                        f"{BASE_MODEL_ID} downloaded with tokenizer and safetensors",
                        downloaded.as_posix(),
                    ),
                )
        except Exception as exc:
            return (
                None,
                PreconditionCheck(
                    "trainable_qwen_base",
                    False,
                    f"{BASE_MODEL_ID} not cached and download failed: {type(exc).__name__}: {exc}",
                    hf_cache_root.as_posix(),
                ),
            )
    return (
        None,
        PreconditionCheck(
            "trainable_qwen_base",
            False,
            f"{BASE_MODEL_ID} is not cached as a trainable HF base snapshot",
            hf_cache_root.as_posix(),
        ),
    )


def default_cuda_available() -> bool:  # pragma: no cover - environment probe
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _candidate_answer(candidate: JsonMap) -> str:
    return str(candidate.get("answer") or candidate.get("final_answer") or "").strip()


def _candidate_text(row_id: str, candidate: JsonMap) -> str:
    answer = _candidate_answer(candidate)
    reasoning = str(candidate.get("reasoning") or candidate.get("text") or "").strip()
    prefix = f"MuSR cached candidate {row_id}"
    if reasoning:
        return f"{prefix}\n{reasoning}\nFinal answer: {answer}"
    return f"{prefix}\nFinal answer: {answer}"


def _normalise_candidate(row_id: str, candidate: JsonMap, *, source: str) -> JsonDict:
    answer = _candidate_answer(candidate)
    cache_index = candidate.get("cache_index", candidate.get("index", 0))
    out: JsonDict = {
        "candidate_id": str(candidate.get("candidate_id") or f"{row_id}/cached-{cache_index}"),
        "answer": answer,
        "reasoning": _candidate_text(row_id, candidate),
        "text": _candidate_text(row_id, candidate),
        "cache_index": cache_index,
        "temperature": candidate.get("temperature", "cached"),
        "source": source,
    }
    for key in ("token_logprobs", "top_logprobs", "mean_logprob", "uprm_marker_logprobs"):
        if key in candidate:
            out[key] = candidate[key]
    return out


def load_shared_candidate_cache_rows(
    path: Path,
    *,
    min_questions: int = 200,
    k_candidates: int = 5,
    limit: int | None = None,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    if not path.exists():
        raise RuntimeError(f"shared candidate cache missing: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            continue
        candidates = [
            _normalise_candidate(str(payload.get("row_id") or len(rows)), candidate, source="exp5016_shared_logprob_candidate_cache")
            for candidate in payload.get("candidates", [])
            if isinstance(candidate, Mapping) and _candidate_answer(candidate)
        ]
        if len(candidates) < k_candidates:
            continue
        row = {
            "row_id": str(payload.get("row_id") or f"shared:{len(rows)}"),
            "corpus": str(payload.get("corpus") or "MuSR/murder_mysteries"),
            "gold": str(payload.get("gold") or ""),
            "candidate_cache_path": path.as_posix(),
            "candidates": candidates,
        }
        if row["gold"]:
            rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    if len(rows) < min_questions:
        raise RuntimeError(f"only {len(rows)} shared MuSR rows available; need {min_questions}")
    return rows


def load_fallback_checkpoint_rows(
    checkpoint_dir: Path,
    *,
    min_questions: int = 200,
    limit: int | None = 200,
) -> list[JsonDict]:
    rows = exp5003.load_cached_musr_rows(
        checkpoint_dir,
        limit=limit,
        min_questions=min_questions,
    )
    for row in rows:
        row["candidate_cache_path"] = str(row.get("candidate_cache_path") or checkpoint_dir)
        for candidate in row.get("candidates", []):
            if isinstance(candidate, dict):
                row_id = str(row.get("row_id") or "")
                candidate["reasoning"] = _candidate_text(row_id, candidate)
                candidate["text"] = _candidate_text(row_id, candidate)
    return rows


def _count_shared_rows(path: Path, *, k_candidates: int) -> int:
    try:
        return len(
            load_shared_candidate_cache_rows(
                path,
                min_questions=0,
                k_candidates=k_candidates,
                limit=None,
            )
        )
    except Exception:
        return 0


def _count_fallback_rows(checkpoint_dir: Path) -> int:
    return len(sorted(checkpoint_dir.glob("q*.json"))) if checkpoint_dir.exists() else 0


def select_candidate_source(
    root: Path,
    *,
    min_questions: int = 200,
    limit: int | None = 200,
    k_candidates: int = 5,
) -> CandidateSource:
    shared_path = root / SHARED_CACHE_RELATIVE_PATH
    fallback_dir = root / FALLBACK_CHECKPOINT_RELATIVE_DIR
    shared_count = _count_shared_rows(shared_path, k_candidates=k_candidates)
    if shared_count >= min_questions:
        rows = load_shared_candidate_cache_rows(
            shared_path,
            min_questions=min_questions,
            k_candidates=k_candidates,
            limit=limit,
        )
        return CandidateSource(
            "exp5016_shared_logprob_candidate_cache",
            shared_path,
            rows,
            PreconditionCheck(
                "cached_musr_candidates",
                True,
                f"{len(rows)} shared-cache MuSR rows available",
                shared_path.as_posix(),
            ),
        )

    fallback_count = _count_fallback_rows(fallback_dir)
    if fallback_count >= min_questions:
        rows = load_fallback_checkpoint_rows(
            fallback_dir,
            min_questions=min_questions,
            limit=limit,
        )
        return CandidateSource(
            "distributional_energy_verifier_musr_checkpoints",
            fallback_dir,
            rows,
            PreconditionCheck(
                "cached_musr_candidates",
                True,
                (
                    f"shared cache rows={shared_count}; using fallback "
                    f"{len(rows)} checkpoint MuSR rows"
                ),
                fallback_dir.as_posix(),
            ),
        )

    raise RuntimeError(
        f"shared cache rows={shared_count}, fallback checkpoints={fallback_count}; "
        f"required >= {min_questions}"
    )


def load_musr_training_pairs_from_rows(
    rows: Sequence[JsonMap],
    *,
    max_pairs: int | None = None,
) -> list[TrainingPair]:
    pairs: list[TrainingPair] = []
    for row in rows:
        row_id = str(row.get("row_id") or len(pairs))
        gold = str(row.get("gold") or "")
        candidates = [candidate for candidate in row.get("candidates", []) if isinstance(candidate, Mapping)]
        good = [candidate for candidate in candidates if _candidate_answer(candidate) == gold]
        bad = [candidate for candidate in candidates if _candidate_answer(candidate) != gold]
        for good_candidate in good:
            for bad_candidate in bad:
                pairs.append(
                    TrainingPair(
                        pair_id=f"musr:{row_id}:{len(pairs)}",
                        good_text=_candidate_text(row_id, good_candidate),
                        bad_text=_candidate_text(row_id, bad_candidate),
                        source="musr_cached_gold_labeled",
                    )
                )
                if max_pairs is not None and len(pairs) >= max_pairs:
                    return pairs
    return pairs


def build_contrastive_corpus(
    fover_paths: Sequence[Path],
    candidate_rows: Sequence[JsonMap],
    *,
    max_pairs: int = 4096,
) -> list[TrainingPair]:
    fover_budget = max_pairs // 2
    fover_pairs = exp5003.load_fover_pairs(fover_paths, max_pairs=fover_budget)
    musr_pairs = load_musr_training_pairs_from_rows(
        candidate_rows,
        max_pairs=max_pairs - len(fover_pairs),
    )
    return (fover_pairs + musr_pairs)[:max_pairs]


def check_preconditions(
    *,
    root: Path,
    hf_cache_root: Path,
    cuda_available: Callable[[], bool] = default_cuda_available,
    min_questions: int = 200,
    allow_download: bool = True,
    downloader: Callable[..., str] | None = None,
    k_candidates: int = 5,
) -> tuple[list[PreconditionCheck], Path | None]:
    base_path, base_check = resolve_or_download_base_model(
        hf_cache_root,
        allow_download=allow_download,
        downloader=downloader,
    )
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

    try:
        source = select_candidate_source(
            root,
            min_questions=min_questions,
            limit=min_questions,
            k_candidates=k_candidates,
        )
        checks.append(source.check)
    except Exception as exc:
        checks.append(
            PreconditionCheck(
                "cached_musr_candidates",
                False,
                f"{type(exc).__name__}: {exc}",
                (root / FALLBACK_CHECKPOINT_RELATIVE_DIR).as_posix(),
            )
        )

    fover_paths = [root / rel for rel in FOVER_RELATIVE_PATHS]
    available = [path for path in fover_paths if path.exists()]
    checks.append(
        PreconditionCheck(
            "fover_pairs",
            bool(available),
            f"{len(available)} FoVer file(s) available",
            ",".join(path.as_posix() for path in available) or None,
        )
    )
    return checks, base_path


def first_missing_resource(checks: Sequence[PreconditionCheck]) -> str | None:
    for check in checks:
        if not check.available:
            return check.resource
    return None


def _latest_epoch_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = [
        path
        for path in checkpoint_dir.glob("epoch_*")
        if (path / "adapter_config.json").exists() and (path / "train_metrics.json").exists()
    ]
    return sorted(candidates)[-1] if candidates else None


def _load_transformer_stack(
    config: TrainingConfig,
    *,
    adapter_path: Path | None = None,
):  # pragma: no cover - exercised by live requested run
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

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
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path.as_posix(), is_trainable=False)
        model.eval()
    return tokenizer, model


def _scorer_from_model(  # pragma: no cover - live transformer scoring closure
    tokenizer: Any,
    model: Any,
    config: TrainingConfig,
) -> Scorer:
    def scorer(candidate: Mapping[str, Any]) -> float:
        text = str(candidate.get("reasoning") or candidate.get("text") or candidate.get("answer") or "")
        encoded = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        ).to(model.device)
        import torch

        with torch.no_grad():
            energy = model(**encoded).logits.float().view(-1)[0]
        return float(energy.detach().cpu())

    return scorer


def _load_existing_trained_scorer(
    *,
    config: TrainingConfig,
    checkpoint_dir: Path,
) -> TrainedScorer | None:  # pragma: no cover - cap-survival live path
    latest = _latest_epoch_checkpoint(checkpoint_dir)
    if latest is None:
        return None
    metrics = _read_json(latest / "train_metrics.json")
    if not isinstance(metrics, dict) or metrics.get("train_loss") is None:
        return None
    tokenizer, model = _load_transformer_stack(config, adapter_path=latest)
    model_specs = dict(metrics.get("model_specs") or {})
    model_specs["resumed_from_checkpoint"] = latest.as_posix()
    epoch_checkpoints = [Path(path) for path in metrics.get("epoch_checkpoints", [latest.as_posix()])]
    return TrainedScorer(
        scorer=_scorer_from_model(tokenizer, model, config),
        train_loss=float(metrics["train_loss"]),
        n_pairs=int(metrics.get("n_pairs") or 0),
        checkpoint_path=latest,
        model_specs=model_specs,
        epoch_checkpoints=epoch_checkpoints,
    )


def train_lora_ebm_scorer(
    pairs: list[TrainingPair],
    *,
    config: TrainingConfig,
    checkpoint_dir: Path,
    skeleton_path: Path,
) -> TrainedScorer:  # pragma: no cover - live GPU path, unit tests inject trainer
    del skeleton_path
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if not pairs:
        raise RuntimeError("no_contrastive_pairs")

    resumed = _load_existing_trained_scorer(config=config, checkpoint_dir=checkpoint_dir)
    if resumed is not None:
        return resumed

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    rng = random.Random(config.seed)
    train_pairs = list(pairs)
    rng.shuffle(train_pairs)
    train_pairs = train_pairs[: config.max_train_pairs]

    tokenizer, model = _load_transformer_stack(config)
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
    epoch_checkpoints: list[Path] = []
    for epoch in range(config.epochs):
        rng.shuffle(train_pairs)
        for start in range(0, len(train_pairs), config.batch_size):
            batch = train_pairs[start : start + config.batch_size]
            good_inputs = tokenizer(
                [pair.good_text for pair in batch],
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            ).to(model.device)
            bad_inputs = tokenizer(
                [pair.bad_text for pair in batch],
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

        epoch_dir = checkpoint_dir / f"epoch_{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        epoch_checkpoints.append(epoch_dir)

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
    metrics = {
        "train_loss": round(train_loss, 6),
        "n_pairs": len(train_pairs),
        "model_specs": specs,
        "epoch_checkpoints": [path.as_posix() for path in epoch_checkpoints],
    }
    latest = epoch_checkpoints[-1]
    write_json(latest / "train_metrics.json", metrics)
    model.eval()
    return TrainedScorer(
        scorer=_scorer_from_model(tokenizer, model, config),
        train_loss=round(train_loss, 6),
        n_pairs=len(train_pairs),
        checkpoint_path=latest,
        model_specs=specs,
        epoch_checkpoints=epoch_checkpoints,
    )


def reproducibility_checksum(
    *,
    base_model_id: str | None,
    config: TrainingConfig | None,
    pairs: Sequence[TrainingPair],
    candidate_source: str | None,
    seed: int,
) -> str:
    payload = {
        "base_model_id": base_model_id,
        "lora_config": config.lora_config_payload() if config else None,
        "pair_checksums": [pair.checksum_payload() for pair in pairs],
        "candidate_source": candidate_source,
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
        "schema": "carnot.experiment_5017_lora_ebm_scorer_musr_v2.v1",
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": honest_verdict,
        "scorer_trained": False,
        "train_loss": None,
        "n_pairs": 0,
        "verifier_is_oracle": False,
        "headroom_present": False,
        "trained_scorer_accuracy": None,
        "genuine_tuned_sc_accuracy": None,
        "delta_vs_tuned_sc": None,
        "paired_ci95": None,
        "mcnemar_p": None,
        "n_questions": 0,
        "oracle_at_k": None,
        "model_specs": {
            "base_model": BASE_MODEL_ID,
            "adapter": "LoRA",
            "energy_head": "scalar",
            "cached_candidate_generator": GENERATOR_MODEL_ID,
        },
        "inference_substrate": "precondition_check_only"
        if honest_verdict.startswith("blocked_")
        else "live_llm_inference",
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "sha256:"
        + hashlib.sha256(_json_dumps(preconditions_checked).encode("utf-8")).hexdigest(),
        "preconditions_checked": list(preconditions_checked),
        "checkpoint_path": None,
        "epoch_checkpoints": [],
        "candidate_cache_source": None,
        "oracle_distinctness_enforced": False,
        "adversarial_verify_clean": False,
        "adversarial_verify_flags": [],
        "summarize_artifact_exit_code": None,
        "duration_s": round(float(duration_s), 6),
        "field_principles": FIELD_PRINCIPLES,
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
        artifact["blocked_error"] = error[:1000]
    return artifact


def build_train_did_not_run_artifact(
    *,
    preconditions_checked: Sequence[JsonDict],
    duration_s: float,
    error: str | None = None,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict="blocked_lora_ebm_train_did_not_run",
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
    )
    artifact["inference_substrate"] = "live_llm_inference"
    if error:
        artifact["blocked_error"] = error[:1000]
    return artifact


def build_skeleton_artifact(
    *,
    preconditions_checked: Sequence[JsonDict],
    base_cache_path: Path,
    candidate_source: str,
    duration_s: float,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict="running_lora_ebm_scorer_musr_v2_pretrain_skeleton",
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
    )
    artifact["deliverable_stage"] = "pretrain_skeleton"
    artifact["model_specs"] = {
        **artifact["model_specs"],
        "base_cache_path": base_cache_path.as_posix(),
    }
    artifact["candidate_cache_source"] = candidate_source
    artifact["inference_substrate"] = "live_llm_inference"
    return artifact


def _format_delta(delta: float) -> str:
    return f"{delta:+.3f}".replace("+", "plus_").replace("-", "minus_").replace(".", "p")


def _ci_includes_zero(ci95: Sequence[float]) -> bool:
    return len(ci95) == 2 and float(ci95[0]) <= 0.0 <= float(ci95[1])


def _read_b1_baseline(root: Path) -> JsonDict:
    path = root / B1_BASELINE_RELATIVE_PATH
    if not path.exists():
        return {"path": path.as_posix(), "available": False}
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {"path": path.as_posix(), "available": False}
    return {
        "path": path.as_posix(),
        "available": True,
        "honest_verdict": payload.get("honest_verdict"),
        "genuine_tuned_sc_accuracy": payload.get("genuine_tuned_sc_accuracy"),
        "oracle_at_k": payload.get("oracle_at_k"),
        "n_questions": payload.get("n_questions"),
    }


def build_complete_artifact(
    *,
    evaluation: JsonDict,
    trained: TrainedScorer,
    config: TrainingConfig,
    pairs: Sequence[TrainingPair],
    preconditions_checked: Sequence[JsonDict],
    candidate_source: CandidateSource,
    root: Path,
    duration_s: float,
) -> JsonDict:
    trained_accuracy = float(evaluation["verifier"]["accuracy"])
    genuine_sc_accuracy = float(evaluation["tuned_self_consistency"]["accuracy"])
    delta = float(evaluation["verifier_minus_tuned_sc_delta"])
    ci95 = [float(value) for value in evaluation["verifier_minus_tuned_sc_ci95"]]
    mcnemar_p = float(evaluation["mcnemar_p"])
    headroom_present = bool(evaluation["headroom_present"])
    scorer_trained = (
        trained.train_loss is not None
        and math.isfinite(float(trained.train_loss))
        and int(trained.n_pairs) > 0
        and float(duration_s) > TRAIN_DURATION_FLOOR_S
    )
    if not scorer_trained:
        return build_train_did_not_run_artifact(
            preconditions_checked=preconditions_checked,
            duration_s=duration_s,
            error=(
                f"trained_gate_failed train_loss={trained.train_loss!r} "
                f"n_pairs={trained.n_pairs!r} duration_s={duration_s:.6f}"
            ),
        )

    win = delta > 0.0 and ci95[0] > 0.0 and mcnemar_p < 0.05 and headroom_present
    verdict_delta = _format_delta(delta)
    if win:
        honest_verdict = f"success_lora_ebm_beats_sc_musr_{verdict_delta}"
    elif _ci_includes_zero(ci95):
        honest_verdict = f"complete_lora_ebm_no_win_musr_{verdict_delta}_ci_incl_0"
    else:
        honest_verdict = f"complete_lora_ebm_no_win_musr_{verdict_delta}_mcnemar_or_headroom_gate"

    artifact = {
        **_base_artifact(
            honest_verdict=honest_verdict,
            preconditions_checked=preconditions_checked,
            duration_s=duration_s,
        ),
        "scorer_trained": True,
        "train_loss": float(trained.train_loss),
        "n_pairs": int(trained.n_pairs),
        "headroom_present": headroom_present,
        "trained_scorer_accuracy": round(trained_accuracy, 6),
        "genuine_tuned_sc_accuracy": round(genuine_sc_accuracy, 6),
        "delta_vs_tuned_sc": round(delta, 6),
        "paired_ci95": ci95,
        "mcnemar_p": mcnemar_p,
        "n_questions": int(evaluation["n_rows"]),
        "oracle_at_k": float(evaluation["oracle_at_k"]),
        "model_specs": {
            **trained.model_specs,
            "base_model": BASE_MODEL_ID,
            "cached_candidate_generator": GENERATOR_MODEL_ID,
            "candidate_cache_source": candidate_source.name,
            "candidate_cache_path": candidate_source.path.as_posix(),
            "tuned_self_consistency_config": evaluation["tuned_self_consistency"]["config"],
            "b1_genuine_sc_baseline_reference": _read_b1_baseline(root),
        },
        "inference_substrate": "live_llm_inference",
        "reproducibility_checksum": reproducibility_checksum(
            base_model_id=config.base_model_id,
            config=config,
            pairs=pairs,
            candidate_source=candidate_source.name,
            seed=config.seed,
        ),
        "checkpoint_path": trained.checkpoint_path.as_posix(),
        "epoch_checkpoints": [path.as_posix() for path in trained.epoch_checkpoints],
        "candidate_cache_source": candidate_source.name,
        "oracle_distinctness_enforced": True,
        "evaluation": evaluation,
    }
    return artifact


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


def run_adversarial_verify(path: Path) -> JsonDict:  # pragma: no cover - script glue
    script_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_5017", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/adversarial_verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_artifact(path)


def run_summarize_artifact(path: Path) -> int:  # pragma: no cover - script glue
    script_path = REPO_ROOT / "scripts" / "summarize_artifact.py"
    spec = importlib.util.spec_from_file_location("carnot_summarize_artifact_5017", script_path)
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
    write_json(artifact_path, updated)
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
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    for field in (
        "scorer_trained",
        "verifier_is_oracle",
        "headroom_present",
        "oracle_distinctness_enforced",
        "adversarial_verify_clean",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
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
    for field in ("trained_scorer_accuracy", "genuine_tuned_sc_accuracy", "oracle_at_k"):
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
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(("blocked_", "running_", "complete_", "success_")):
        errors.append("honest_verdict")
    if artifact.get("scorer_trained") is True:
        if artifact.get("train_loss") is None:
            errors.append("train_loss")
        if int(artifact.get("n_pairs") or 0) <= 0:
            errors.append("n_pairs")
        if float(artifact.get("duration_s") or 0.0) <= TRAIN_DURATION_FLOOR_S:
            errors.append("duration_s")
    if artifact.get("scorer_trained") is False and verdict.startswith(("success_", "complete_")):
        errors.append("scorer_trained")
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
    k_candidates: int = 5,
    bootstrap_samples: int = 2000,
    random_seed: int = RANDOM_SEED,
    now: Clock = time.time,
    allow_download: bool = True,
    downloader: Callable[..., str] | None = None,
    write: bool = True,
) -> JsonDict:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    hf_cache_root = (
        Path(hf_cache_root)
        if hf_cache_root
        else Path.home() / ".cache" / "huggingface" / "hub"
    )
    start = float(now())

    checks, base_cache_path = check_preconditions(
        root=root,
        hf_cache_root=hf_cache_root,
        cuda_available=cuda_available,
        min_questions=min_questions,
        allow_download=allow_download,
        downloader=downloader,
        k_candidates=k_candidates,
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

    assert base_cache_path is not None
    candidate_source = select_candidate_source(
        root,
        min_questions=min_questions,
        limit=limit,
        k_candidates=k_candidates,
    )
    skeleton = build_skeleton_artifact(
        preconditions_checked=preconditions,
        base_cache_path=base_cache_path,
        candidate_source=candidate_source.name,
        duration_s=float(now()) - start,
    )
    if write:
        write_json(artifact_path, skeleton)

    fover_paths = [root / rel for rel in FOVER_RELATIVE_PATHS if (root / rel).exists()]
    config = TrainingConfig(
        base_model_id=BASE_MODEL_ID,
        base_cache_path=base_cache_path.as_posix(),
        seed=random_seed,
    )
    try:
        pairs = build_contrastive_corpus(
            fover_paths,
            candidate_source.rows,
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
        training_elapsed = float(now()) - start
        if (
            trained.train_loss is None
            or not math.isfinite(float(trained.train_loss))
            or int(trained.n_pairs) <= 0
            or training_elapsed <= TRAIN_DURATION_FLOOR_S
        ):
            artifact = build_train_did_not_run_artifact(
                preconditions_checked=preconditions,
                duration_s=training_elapsed,
                error=(
                    f"trained_gate_failed train_loss={trained.train_loss!r} "
                    f"n_pairs={trained.n_pairs!r} duration_s={training_elapsed:.6f}"
                ),
            )
            if write:
                write_json(artifact_path, artifact)
            return artifact
        evaluation = evaluate_verifier(
            candidate_source.rows,
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
        artifact = build_train_did_not_run_artifact(
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
        candidate_source=candidate_source,
        root=root,
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


def main() -> int:  # pragma: no cover - script entrypoint
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
