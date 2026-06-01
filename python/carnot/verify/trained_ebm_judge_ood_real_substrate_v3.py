"""Exp 3659 real-substrate trained EBM judge OOD retry.

Spec: REQ-VERIFY-3659, SCENARIO-VERIFY-3659.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Protocol

import numpy as np

from carnot.verify.trained_ebm_judge_ood_counterpoint_v2 import (
    FOVER_CORPUS_REL_PATH,
    JudgeExample,
    _example_checksum_payload,
    feature_vector,
    load_corpora,
    metric_object,
    shuffled_labels,
    stratified_train_eval_split,
    summarize_values,
    tie_aware_auroc,
    trainable_math_corpus,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3659_trained_ebm_judge_ood_real_substrate_v3.json")
EXP3640_REL_PATH = Path("results/experiment_3640_build_factual_corpus_v3.json")
EXP3641_REL_PATH = Path("results/experiment_3641_code_corpus_verifiers_fire_transfer_v3.json")
DEFAULT_SEEDS = (3659, 3660, 3661)
RANDOM_SEED = DEFAULT_SEEDS[0]
TOY_HEAD_OOD_REFERENCE_AUROC = 0.673554
CONFIDENCE_ONLY_EXP3646_REFERENCE_AUROC = 0.882162
REAL_SUBSTRATE_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SCALAR_FEATURE_NAMES = (
    "validity_signal",
    "confidence_error_signal",
    "log1p_char_count",
    "digit_fraction",
    "format_noise_signal",
)
INFERENCE_SUBSTRATE_PREFIX = "verifier_ensemble_against_cached_candidates"
INFERENCE_SUBSTRATE = INFERENCE_SUBSTRATE_PREFIX
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "judge_substrate",
    "judge_recipe",
    "in_domain_judge_auroc",
    "ood_judge_auroc",
    "confidence_only_baseline_auroc",
    "shuffled_label_control_auroc",
    "ood_domains_tested",
    "real_substrate_vs_toy_head_ood_delta",
    "trained_judge_transfers_ood",
    "n_examples_per_domain",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)
FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "Trains a head on real embeddings of cached labels + scores cached corpora; declares GPU/CPU + substrate."
    ),
    "judge_substrate": (
        "The REAL substrate used (GGUF embedding model / small transformer) -- the key difference from the .334 toy numpy head."
    ),
    "judge_recipe": "Features/loss/epochs -- reproducibility of the prototype.",
    "in_domain_judge_auroc": (
        "Sanity: the trained judge must work in-domain before any OOD claim is meaningful."
    ),
    "ood_judge_auroc": (
        "The core number: does the real-substrate trained judge detect errors OOD?"
    ),
    "confidence_only_baseline_auroc": (
        "The bar the judge must beat OOD (exp3646 found confidence 0.882 beat the toy judge 0.673)."
    ),
    "shuffled_label_control_auroc": (
        "Adversarial control: ~0.5 confirms the judge is not learning a degenerate length/format prior."
    ),
    "ood_domains_tested": "Which OOD domains (code/facts) were evaluated -- scopes the transfer claim.",
    "real_substrate_vs_toy_head_ood_delta": (
        "Did a real substrate buy OOD transfer the .334 toy head lacked? -- the resourcing-vs-method question."
    ),
    "trained_judge_transfers_ood": (
        "BARE bool. True iff the real-substrate judge beats BOTH confidence-only and shuffled-control OOD."
    ),
    "n_examples_per_domain": "Sample-size rigor.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor (a real train+eval takes minutes, not seconds).",
}


class EmbeddingProvider(Protocol):
    """Protocol for real embedding substrates used by the ranking head."""

    def encode_examples(self, examples: list[JudgeExample]) -> np.ndarray:
        """Return one real embedding vector per example."""
        ...

    def substrate_report(self) -> JsonDict:
        """Return a JSON-serializable substrate report."""
        ...


class TorchEnergyRanker:
    """Torch logistic ranking head trained to assign high energy to errors."""

    def __init__(
        self,
        *,
        input_dim: int,
        epochs: int = 80,
        lr: float = 0.05,
        l2: float = 1e-4,
        device: str | None = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.device = select_device(device)
        self.weights: np.ndarray | None = None
        self.bias = 0.0
        self._mu = np.zeros(self.input_dim, dtype=np.float32)
        self._sigma = np.ones(self.input_dim, dtype=np.float32)
        self._fitted = False

    @property
    def n_params(self) -> int:
        return self.input_dim + 1

    def fit(self, X: np.ndarray, y: Sequence[int], *, seed: int) -> "TorchEnergyRanker":
        import torch

        Xa = np.asarray(X, dtype=np.float32)
        ya = np.asarray(y, dtype=np.float32)
        if Xa.ndim != 2 or Xa.shape[1] != self.input_dim:
            raise ValueError(f"expected (_, {self.input_dim}) feature matrix, got {Xa.shape}")
        if Xa.shape[0] != ya.shape[0]:
            raise ValueError("feature matrix and labels must have the same row count")

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        self._mu = Xa.mean(axis=0)
        sigma = Xa.std(axis=0)
        sigma[sigma < 1e-8] = 1.0
        self._sigma = sigma.astype(np.float32)

        Z = self._standardise(Xa)
        X_tensor = torch.as_tensor(Z, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(ya.reshape(-1, 1), dtype=torch.float32, device=self.device)
        layer = torch.nn.Linear(self.input_dim, 1).to(self.device)
        torch.nn.init.normal_(layer.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(layer.bias)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=self.lr, weight_decay=self.l2)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(layer(X_tensor), y_tensor)
            loss.backward()
            optimizer.step()
        self.weights = layer.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
        self.bias = float(layer.bias.detach().cpu().numpy().reshape(-1)[0])
        self._fitted = True
        return self

    def predict_scores(self, X: np.ndarray) -> list[float]:
        if not self._fitted or self.weights is None:
            raise RuntimeError("torch energy ranker is not fitted")
        Xa = np.asarray(X, dtype=np.float32)
        if Xa.ndim != 2 or Xa.shape[1] != self.input_dim:
            raise ValueError(f"expected (_, {self.input_dim}) feature matrix, got {Xa.shape}")
        logits = np.clip(self._standardise(Xa) @ self.weights + self.bias, -50.0, 50.0)
        return [float(1.0 / (1.0 + math.exp(-float(logit)))) for logit in logits]

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mu) / self._sigma


class TransformerEmbeddingProvider:  # pragma: no cover - exercised by the experiment script.
    """Local pretrained transformer embedding substrate for the Exp 3659 run."""

    def __init__(
        self,
        *,
        model_id: str = REAL_SUBSTRATE_MODEL_ID,
        device: str | None = None,
        batch_size: int = 96,
        max_length: int = 192,
    ) -> None:
        self.model_id = model_id
        self.device = select_device(device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._num_params: int | None = None

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)
        self._model = AutoModel.from_pretrained(self.model_id, local_files_only=True).to(self.device)
        self._model.eval()
        self._num_params = int(sum(param.numel() for param in self._model.parameters()))

    def encode_examples(self, examples: list[JudgeExample]) -> np.ndarray:
        import torch

        self.ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        texts = [example.text for example in examples]
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                output = self._model(**encoded)
            hidden = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            batches.append(pooled.detach().cpu().numpy().astype(np.float32))
        if not batches:
            return np.zeros((0, 1), dtype=np.float32)
        return np.vstack(batches)

    def substrate_report(self) -> JsonDict:
        return {
            "kind": "small_transformer_embedding_plus_torch_rank_head",
            "model_id": self.model_id,
            "device": self.device,
            "local_files_only": True,
            "num_embedding_model_params": self._num_params,
            "embedding_batch_size": self.batch_size,
            "max_length": self.max_length,
        }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    math_examples: Sequence[JudgeExample] | None = None,
    ood_examples_by_domain: Mapping[str, Sequence[JudgeExample]] | None = None,
    feature_provider: EmbeddingProvider | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    epochs: int = 80,
    lr: float = 0.05,
    l2: float = 1e-4,
    max_math_examples: int = 2000,
    force_no_trainable_substrate: bool = False,
    tests_run: Sequence[str] | None = None,
    device: str | None = None,
) -> JsonDict:
    """Build the Exp 3659 terminal artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    if math_examples is None or ood_examples_by_domain is None:
        loaded_math, loaded_ood = load_corpora(root_path, max_math_examples=max_math_examples)
        if math_examples is None:
            math_examples = loaded_math
        if ood_examples_by_domain is None:
            ood_examples_by_domain = loaded_ood

    math_rows = list(math_examples)
    ood_rows_by_domain = {
        str(domain): list(rows)
        for domain, rows in dict(ood_examples_by_domain or {}).items()
        if rows
    }
    ood_domains = sorted(ood_rows_by_domain)
    provider = feature_provider
    if provider is None and not force_no_trainable_substrate:
        provider = default_feature_provider(device=device)
    no_substrate = bool(force_no_trainable_substrate or provider is None or not trainable_math_corpus(math_rows))
    if no_substrate:
        evaluation = empty_evaluation()
    else:
        assert provider is not None
        evaluation = evaluate_real_substrate_judge(
            math_rows,
            ood_rows_by_domain,
            feature_provider=provider,
            seeds=seeds,
            epochs=epochs,
            lr=lr,
            l2=l2,
            device=device,
        )

    trained_transfer = bool(
        evaluation["ood_judge_auroc"] is not None
        and evaluation["shuffled_label_control_auroc"] is not None
        and evaluation["confidence_only_baseline_auroc"] is not None
        and float(evaluation["ood_judge_auroc"])
        > float(evaluation["shuffled_label_control_auroc"])
        and float(evaluation["ood_judge_auroc"])
        > float(evaluation["confidence_only_baseline_auroc"])
    )
    finished = time.perf_counter() if now_s is None else float(now_s)
    substrate = blocked_substrate_report(device=device) if provider is None else provider.substrate_report()
    artifact: JsonDict = {
        "artifact": "experiment_3659_trained_ebm_judge_ood_real_substrate_v3",
        "schema": "carnot.trained_ebm_judge_ood_real_substrate.v3",
        "honest_verdict": terminal_verdict(
            no_trainable_substrate=no_substrate,
            ood_domains_tested=ood_domains,
            trained_judge_transfers_ood=trained_transfer,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "judge_substrate": substrate,
        "models_tested": [str(substrate.get("model_id") or "blocked")],
        "judge_recipe": judge_recipe(
            epochs=epochs,
            lr=lr,
            l2=l2,
            seeds=seeds,
            input_dim=evaluation["input_dim"],
            device=str(substrate.get("device") or select_device(device)),
        ),
        "in_domain_judge_auroc": evaluation["in_domain_judge_auroc"],
        "in_domain_judge_auroc_ci95": evaluation["in_domain_judge_auroc_ci95"],
        "ood_judge_auroc": evaluation["ood_judge_auroc"],
        "ood_judge_auroc_ci95": evaluation["ood_judge_auroc_ci95"],
        "ood_domains_tested": ood_domains,
        "ood_judge_auroc_by_domain": evaluation["ood_judge_auroc_by_domain"],
        "confidence_only_baseline_auroc": evaluation["confidence_only_baseline_auroc"],
        "confidence_only_baseline_auroc_ci95": evaluation["confidence_only_baseline_auroc_ci95"],
        "shuffled_label_control_auroc": evaluation["shuffled_label_control_auroc"],
        "shuffled_label_control_auroc_ci95": evaluation["shuffled_label_control_auroc_ci95"],
        "toy_head_ood_reference_auroc": TOY_HEAD_OOD_REFERENCE_AUROC,
        "exp3646_reference_metrics": {
            "toy_head_ood_reference_auroc": TOY_HEAD_OOD_REFERENCE_AUROC,
            "confidence_only_baseline_auroc": CONFIDENCE_ONLY_EXP3646_REFERENCE_AUROC,
        },
        "real_substrate_vs_toy_head_ood_delta": real_substrate_delta(evaluation["ood_judge_auroc"]),
        "real_substrate_vs_confidence_ood_delta": comparison_delta(
            evaluation["ood_judge_auroc"],
            evaluation["confidence_only_baseline_auroc"],
        ),
        "trained_judge_transfers_ood": trained_transfer,
        "n_examples_per_domain": {
            "math": len(math_rows),
            **{domain: len(rows) for domain, rows in ood_rows_by_domain.items()},
        },
        "random_seed": int(seeds[0]) if seeds else RANDOM_SEED,
        "random_seeds_used": [int(seed) for seed in seeds],
        "reproducibility_checksum": reproducibility_checksum(
            math_rows,
            ood_rows_by_domain,
            evaluation,
            seeds,
            substrate,
        ),
        "duration_s": round(max(0.0, finished - start), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": (
                "in_domain_judge_auroc present AND shuffled_label_control_auroc present "
                "AND ood_judge_auroc present"
            ),
            "passed": bool(
                evaluation["in_domain_judge_auroc"] is not None
                and evaluation["shuffled_label_control_auroc"] is not None
                and evaluation["ood_judge_auroc"] is not None
            ),
            "principle": (
                "An OOD-transfer claim requires an in-domain sanity pass AND a shuffled-label "
                "adversarial control -- without both the OOD number could be a degenerate-prior artifact."
            ),
        },
        "preconditions": {
            "cuda_available": cuda_available(),
            "fover_math_labels_loadable": trainable_math_corpus(math_rows),
            "ood_eval_corpus_loadable": bool(ood_domains),
            "real_trainable_substrate_available": not no_substrate,
        },
        "per_seed_results": evaluation["per_seed_results"],
        "source_artifacts": [str(EXP3640_REL_PATH), str(EXP3641_REL_PATH), str(FOVER_CORPUS_REL_PATH)],
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    tests_run: Sequence[str] | None = None,
    math_examples: Sequence[JudgeExample] | None = None,
    ood_examples_by_domain: Mapping[str, Sequence[JudgeExample]] | None = None,
    feature_provider: EmbeddingProvider | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    device: str | None = None,
) -> Path:
    """Build and persist the Exp 3659 artifact."""

    root_path = Path(root)
    output = root_path / Path(output_path)
    artifact = build_artifact(
        root_path,
        tests_run=tests_run,
        math_examples=math_examples,
        ood_examples_by_domain=ood_examples_by_domain,
        feature_provider=feature_provider,
        started_s=started_s,
        now_s=now_s,
        device=device,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def evaluate_real_substrate_judge(
    math_examples: Sequence[JudgeExample],
    ood_examples_by_domain: Mapping[str, Sequence[JudgeExample]],
    *,
    feature_provider: EmbeddingProvider,
    seeds: Sequence[int],
    epochs: int,
    lr: float,
    l2: float,
    device: str | None,
) -> JsonDict:
    """Train on math embeddings and evaluate held-out math plus available OOD rows."""

    ood_flat = [
        example
        for domain in sorted(ood_examples_by_domain)
        for example in ood_examples_by_domain[domain]
    ]
    all_examples = list(math_examples) + ood_flat
    all_features = feature_matrix(all_examples, feature_provider)
    math_features = all_features[: len(math_examples)]
    ood_features = all_features[len(math_examples) :]
    ood_slices = domain_slices(ood_examples_by_domain)
    per_seed: list[JsonDict] = []
    in_domain_values: list[float] = []
    ood_values: list[float] = []
    shuffled_values: list[float] = []
    confidence_values: list[float] = []
    ood_by_domain_values: dict[str, list[float]] = {domain: [] for domain in ood_examples_by_domain}
    labels = [example.error_label for example in math_examples]
    for seed in seeds:
        train_indices, heldout_indices = split_indices(math_examples, seed=int(seed))
        X_train = math_features[train_indices]
        y_train = [labels[idx] for idx in train_indices]
        input_dim = int(math_features.shape[1])
        ranker = TorchEnergyRanker(
            input_dim=input_dim,
            epochs=epochs,
            lr=lr,
            l2=l2,
            device=device,
        ).fit(X_train, y_train, seed=int(seed))
        heldout_labels = [labels[idx] for idx in heldout_indices]
        heldout_scores = ranker.predict_scores(math_features[heldout_indices])
        in_auc = tie_aware_auroc(heldout_labels, heldout_scores)
        in_domain_values.append(in_auc)
        seed_row: JsonDict = {"seed": int(seed), "in_domain_judge_auroc": round(in_auc, 6)}
        if ood_flat:
            ood_labels = [example.error_label for example in ood_flat]
            ood_scores = ranker.predict_scores(ood_features)
            ood_auc = tie_aware_auroc(ood_labels, ood_scores)
            shuffled_y = shuffled_labels(y_train, seed=int(seed))
            shuffled_ranker = TorchEnergyRanker(
                input_dim=input_dim,
                epochs=epochs,
                lr=lr,
                l2=l2,
                device=device,
            ).fit(X_train, shuffled_y, seed=int(seed) + 7919)
            shuffled_scores = shuffled_ranker.predict_scores(ood_features)
            shuffled_auc = tie_aware_auroc(ood_labels, shuffled_scores)
            confidence_auc = tie_aware_auroc(
                ood_labels,
                [example.confidence_error_signal for example in ood_flat],
            )
            ood_values.append(ood_auc)
            shuffled_values.append(shuffled_auc)
            confidence_values.append(confidence_auc)
            seed_row.update(
                {
                    "ood_judge_auroc": round(ood_auc, 6),
                    "shuffled_label_control_auroc": round(shuffled_auc, 6),
                    "confidence_only_baseline_auroc": round(confidence_auc, 6),
                }
            )
            for domain, (start, end) in ood_slices.items():
                rows = list(ood_examples_by_domain[domain])
                domain_auc = tie_aware_auroc(
                    [example.error_label for example in rows],
                    ranker.predict_scores(ood_features[start:end]),
                )
                ood_by_domain_values.setdefault(domain, []).append(domain_auc)
        per_seed.append(seed_row)

    in_point, in_ci = summarize_values(in_domain_values)
    ood_point, ood_ci = summarize_values(ood_values)
    shuffled_point, shuffled_ci = summarize_values(shuffled_values)
    confidence_point, confidence_ci = summarize_values(confidence_values)
    return {
        "input_dim": int(math_features.shape[1]),
        "in_domain_judge_auroc": in_point,
        "in_domain_judge_auroc_ci95": in_ci,
        "ood_judge_auroc": ood_point,
        "ood_judge_auroc_ci95": ood_ci,
        "ood_judge_auroc_by_domain": {
            domain: metric_object(values) for domain, values in sorted(ood_by_domain_values.items())
        },
        "confidence_only_baseline_auroc": confidence_point,
        "confidence_only_baseline_auroc_ci95": confidence_ci,
        "shuffled_label_control_auroc": shuffled_point,
        "shuffled_label_control_auroc_ci95": shuffled_ci,
        "per_seed_results": per_seed,
    }


def feature_matrix(examples: Sequence[JudgeExample], provider: EmbeddingProvider) -> np.ndarray:
    embeddings = np.asarray(provider.encode_examples(list(examples)), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(examples):
        raise ValueError("embedding provider must return a 2D matrix with one row per example")
    scalar = np.asarray([feature_vector(example) for example in examples], dtype=np.float32)
    return np.concatenate([embeddings, scalar], axis=1)


def split_indices(examples: Sequence[JudgeExample], *, seed: int) -> tuple[list[int], list[int]]:
    train_rows, heldout_rows = stratified_train_eval_split(examples, seed=seed)
    train_lookup = {id(example) for example in train_rows}
    heldout_lookup = {id(example) for example in heldout_rows}
    train_indices = [idx for idx, example in enumerate(examples) if id(example) in train_lookup]
    heldout_indices = [idx for idx, example in enumerate(examples) if id(example) in heldout_lookup]
    return train_indices, heldout_indices


def domain_slices(ood_examples_by_domain: Mapping[str, Sequence[JudgeExample]]) -> dict[str, tuple[int, int]]:
    slices: dict[str, tuple[int, int]] = {}
    cursor = 0
    for domain in sorted(ood_examples_by_domain):
        n_rows = len(ood_examples_by_domain[domain])
        slices[domain] = (cursor, cursor + n_rows)
        cursor += n_rows
    return slices


def empty_evaluation() -> JsonDict:
    return {
        "input_dim": None,
        "in_domain_judge_auroc": None,
        "in_domain_judge_auroc_ci95": None,
        "ood_judge_auroc": None,
        "ood_judge_auroc_ci95": None,
        "ood_judge_auroc_by_domain": {},
        "confidence_only_baseline_auroc": None,
        "confidence_only_baseline_auroc_ci95": None,
        "shuffled_label_control_auroc": None,
        "shuffled_label_control_auroc_ci95": None,
        "per_seed_results": [],
    }


def terminal_verdict(
    *,
    no_trainable_substrate: bool,
    ood_domains_tested: Sequence[str],
    trained_judge_transfers_ood: bool,
) -> str:
    if no_trainable_substrate:
        return "complete: blocked_no_trainable_substrate"
    if not ood_domains_tested:
        return "complete: blocked_no_ood_eval_corpus"
    if trained_judge_transfers_ood:
        return "complete: real_substrate_trained_judge_transfers_ood_resourcing_was_the_bottleneck"
    return "complete: real_substrate_trained_judge_also_math_only_trained_judge_not_the_cross_domain_fix"


def judge_recipe(
    *,
    epochs: int,
    lr: float,
    l2: float,
    seeds: Sequence[int],
    input_dim: int | None,
    device: str,
) -> JsonDict:
    return {
        "model": "torch_logistic_energy_rank_head_on_real_transformer_embeddings",
        "embedding_features": "frozen local transformer sentence embeddings",
        "scalar_features": list(SCALAR_FEATURE_NAMES),
        "input_dim": input_dim,
        "loss": "AdamW binary cross-entropy on math error_label; score is P(error).",
        "epochs": int(epochs),
        "learning_rate": float(lr),
        "l2": float(l2),
        "device": device,
        "train_eval_split": "80/20 stratified FoVer math split per seed; OOD labels are never used for training.",
        "random_seeds": [int(seed) for seed in seeds],
    }


def real_substrate_delta(ood_auroc: float | None) -> float | None:
    if ood_auroc is None:
        return None
    return round(float(ood_auroc) - TOY_HEAD_OOD_REFERENCE_AUROC, 6)


def comparison_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be present")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    if artifact.get("honest_verdict") not in {
        "complete: real_substrate_trained_judge_transfers_ood_resourcing_was_the_bottleneck",
        "complete: real_substrate_trained_judge_also_math_only_trained_judge_not_the_cross_domain_fix",
        "complete: blocked_no_ood_eval_corpus",
        "complete: blocked_no_trainable_substrate",
    }:
        raise ValueError("honest_verdict is not an allowed Exp 3659 terminal verdict")
    if type(artifact.get("trained_judge_transfers_ood")) is not bool:
        raise ValueError("trained_judge_transfers_ood must be a bare bool")
    if not isinstance(artifact.get("judge_substrate"), Mapping):
        raise ValueError("judge_substrate must be a mapping")
    if not isinstance(artifact.get("acceptance_gate"), Mapping):
        raise ValueError("acceptance_gate must be present")
    duration = artifact.get("duration_s")
    if not isinstance(duration, int | float) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")


def reproducibility_checksum(
    math_examples: Sequence[JudgeExample],
    ood_examples_by_domain: Mapping[str, Sequence[JudgeExample]],
    evaluation: Mapping[str, Any],
    seeds: Sequence[int],
    substrate: Mapping[str, Any],
) -> str:
    payload = {
        "math": _example_checksum_payload(math_examples),
        "ood": {
            domain: _example_checksum_payload(rows)
            for domain, rows in sorted(ood_examples_by_domain.items())
        },
        "evaluation": evaluation,
        "seeds": [int(seed) for seed in seeds],
        "substrate": substrate,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def default_feature_provider(*, device: str | None = None) -> EmbeddingProvider | None:  # pragma: no cover
    try:
        provider = TransformerEmbeddingProvider(device=device)
        provider.ensure_loaded()
    except Exception:
        return None
    return provider


def blocked_substrate_report(*, device: str | None = None) -> JsonDict:
    return {
        "kind": "blocked_no_real_trainable_substrate",
        "model_id": REAL_SUBSTRATE_MODEL_ID,
        "device": select_device(device),
        "available": False,
    }


def select_device(device: str | None = None) -> str:
    if device:
        return str(device)
    return "cuda" if cuda_available() else "cpu"


def cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())
