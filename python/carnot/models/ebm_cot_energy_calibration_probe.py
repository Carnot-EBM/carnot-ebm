"""EBM-CoT hinge calibration probe for Carnot's KAN energy tier.

The probe is intentionally CPU-only: it reads existing FoVer step labels,
warm-starts the two-layer KAN energy surface from a local JSON checkpoint when
one is shape-compatible, and applies the EBM-CoT objective:

    max(0, margin - (E_negative - E_positive))
    + lambda * |E_positive - E_positive_paraphrase|

Lower energy means a FoVer step is predicted correct; higher energy means the
step is predicted incorrect.  The module writes the experiment 1384 JSON
artifact and exposes small pure helpers so tests can pin the loss semantics.

Spec: REQ-KAN-1384, SCENARIO-KAN-1384
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from carnot.eval.metrics import auroc
from carnot.models.prompt_injection_kan import _injection_energy


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FOVER_PATH = REPO_ROOT / "data" / "fover_corpus.jsonl"
DEFAULT_MODELS_DIR = REPO_ROOT / "python" / "carnot" / "models"
DEFAULT_ARTIFACT_PATH = REPO_ROOT / "results" / "experiment_1384_ebm_cot_energy_calibration_probe.json"

DEFAULT_HINGE_MARGIN = 1.0
DEFAULT_CONSISTENCY_WEIGHT = 0.10
DEFAULT_N_EPOCHS = 20
DEFAULT_N_FEATURES = 32
DEFAULT_N_HIDDEN = 8
DEFAULT_N_KNOTS = 10
DEFAULT_DEGREE = 3

_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?|[+\-*/=<>%$]")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_EQUATION_RE = re.compile(
    r"(?P<a>-?\d+(?:\.\d+)?)\s*"
    r"(?P<op>[+\-*/xX×])\s*"
    r"(?P<b>-?\d+(?:\.\d+)?)\s*=\s*"
    r"(?P<c>-?\d+(?:\.\d+)?)"
)


@dataclass(frozen=True)
class FoVerStepCase:
    """One labeled FoVer reasoning step used by the calibration probe."""

    case_id: str
    question: str
    step_text: str
    label: int


@dataclass(frozen=True)
class FoVerSplit:
    """Balanced FoVer split with paired positives and negatives for training."""

    train_positive: list[FoVerStepCase]
    train_negative: list[FoVerStepCase]
    test_cases: list[FoVerStepCase]

    @property
    def corpus_cases_used(self) -> int:
        return len(self.train_positive) + len(self.train_negative) + len(self.test_cases)


@dataclass(frozen=True)
class KANCheckpointInfo:
    """Metadata about the local checkpoint warm-start used by the probe."""

    loaded: bool
    path: str | None
    schema: str
    reason: str


def _stable_hash_bucket(text: str, n_buckets: int) -> int:
    """Return a deterministic bucket index without relying on salted Python hash()."""

    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % n_buckets
    return value


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _normalize_label(row: dict[str, Any]) -> int | None:
    raw = row.get("label", row.get("step_correct", row.get("is_correct")))
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)) and raw in {0, 1}:
        return int(raw)
    text = str(raw).strip().lower()
    if text in {"correct", "true", "1", "yes"}:
        return 1
    if text in {"incorrect", "false", "0", "no"}:
        return 0
    return None


def _rows_from_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        payload = json.load(handle)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("pairs", "rows", "cases", "items", "examples", "data", "records"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    raise ValueError(f"unsupported FoVer corpus shape at {path}")


def load_fover_verified_cases(path: Path | str = DEFAULT_FOVER_PATH) -> list[FoVerStepCase]:
    """Load FoVer verified step pairs without invoking any LLM inference.

    Rows may come from JSONL or the repo's experiment JSON schemas.  Only rows
    with a usable correct/incorrect label and non-empty step text are retained.

    Spec: REQ-KAN-1384
    """

    source = Path(path)
    cases: list[FoVerStepCase] = []
    for index, row in enumerate(_rows_from_json_or_jsonl(source)):
        label = _normalize_label(row)
        step_text = str(row.get("step_text") or row.get("step") or row.get("response") or "")
        if label is None or not step_text.strip():
            continue
        question = str(row.get("question") or row.get("prompt") or "")
        case_id = str(row.get("question_id") or row.get("id") or f"fover_{index}")
        cases.append(
            FoVerStepCase(
                case_id=case_id,
                question=question,
                step_text=step_text,
                label=label,
            )
        )
    labels = {case.label for case in cases}
    if labels != {0, 1}:
        raise ValueError(f"FoVer corpus must contain both labels, found {sorted(labels)}")
    return cases


def make_balanced_split(
    cases: list[FoVerStepCase],
    *,
    test_fraction: float = 0.20,
    seed: int = 20260505,
    max_pairs_per_class: int | None = None,
) -> FoVerSplit:
    """Create a deterministic balanced train/test split for contrastive pairs."""

    positives = [case for case in cases if case.label == 1]
    negatives = [case for case in cases if case.label == 0]
    n_per_class = min(len(positives), len(negatives))
    if max_pairs_per_class is not None:
        n_per_class = min(n_per_class, max_pairs_per_class)
    if n_per_class < 4:
        raise ValueError("FoVer split requires at least four cases per class")

    rng = np.random.default_rng(seed)
    pos_idx = rng.permutation(len(positives))[:n_per_class]
    neg_idx = rng.permutation(len(negatives))[:n_per_class]
    positives = [positives[int(i)] for i in pos_idx]
    negatives = [negatives[int(i)] for i in neg_idx]

    n_test = max(1, int(round(n_per_class * test_fraction)))
    n_train = n_per_class - n_test
    return FoVerSplit(
        train_positive=positives[:n_train],
        train_negative=negatives[:n_train],
        test_cases=positives[n_train:] + negatives[n_train:],
    )


def _arithmetic_error_rate(text: str) -> float:
    mismatches = 0
    total = 0
    for match in _EQUATION_RE.finditer(text):
        total += 1
        a = float(match.group("a"))
        b = float(match.group("b"))
        c = float(match.group("c"))
        op = match.group("op")
        if op == "+":
            expected = a + b
        elif op == "-":
            expected = a - b
        elif op in {"*", "x", "X", "×"}:
            expected = a * b
        elif abs(b) > 1e-12:
            expected = a / b
        else:
            expected = math.inf
        if not math.isfinite(expected) or abs(expected - c) > 1e-6:
            mismatches += 1
    if total == 0:
        return 0.0
    return mismatches / total


def _question_number_overlap(question: str, step_text: str) -> float:
    question_numbers = set(_NUMBER_RE.findall(question))
    step_numbers = set(_NUMBER_RE.findall(step_text))
    if not step_numbers:
        return 0.0
    return len(question_numbers & step_numbers) / len(step_numbers)


def paraphrase_positive_step(step_text: str) -> str:
    """Create a deterministic paraphrase-like positive variant without LLM calls."""

    text = step_text.replace("**", "").replace("\\[", "").replace("\\]", "")
    replacements = (
        ("Therefore,", "Thus,"),
        ("Therefore", "Thus"),
        ("First,", "Initially,"),
        ("Next,", "Then,"),
        ("Now,", "At this point,"),
        ("we can", "we may"),
        ("the total", "the sum"),
        ("equals", "is equal to"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return f"In other words, {text.strip()}"


def encode_fover_features(case: FoVerStepCase, n_features: int = DEFAULT_N_FEATURES) -> np.ndarray:
    """Encode a FoVer question/step pair into KAN features in [0, 1].

    The first 16 features are interpretable arithmetic/text-shape probes.  The
    remaining buckets are deterministic token hashes so the KAN can still learn
    corpus-specific regularities without a tokenizer or external model.

    Spec: REQ-KAN-1384
    """

    if n_features < 16:
        raise ValueError(f"n_features must be >= 16, got {n_features}")

    text = case.step_text
    question = case.question
    tokens = _TOKEN_RE.findall(text.lower())
    words = [tok for tok in tokens if tok.isalpha()]
    numbers = [float(tok) for tok in _NUMBER_RE.findall(text)]
    token_count = max(len(tokens), 1)
    word_count = max(len(words), 1)
    operator_count = sum(1 for tok in tokens if tok in {"+", "-", "*", "/", "=", "<", ">"})
    unique_ratio = len(set(words)) / word_count
    answer_only = 1.0 if re.fullmatch(r"\s*the answer is\s+-?\d+(?:\.\d+)?\.?\s*", text.lower()) else 0.0
    arithmetic_error = _arithmetic_error_rate(text)
    number_overlap_error = 1.0 - _question_number_overlap(question, text)
    latex_density = min((text.count("\\") + text.count("{") + text.count("}")) / max(len(text), 1), 1.0)
    magnitude = max((abs(num) for num in numbers), default=0.0)

    base = np.zeros(n_features, dtype=np.float32)
    base[:16] = np.array(
        [
            _clip01(math.log1p(len(text)) / 8.0),
            _clip01(len(numbers) / token_count),
            _clip01(operator_count / token_count),
            _clip01(text.count("=") / 5.0),
            _clip01(arithmetic_error),
            _clip01(number_overlap_error),
            _clip01((text.lower().count("therefore") + text.lower().count("answer")) / 4.0),
            _clip01(latex_density * 20.0),
            _clip01((text.count("$") + text.count("%")) / 5.0),
            _clip01(1.0 - unique_ratio),
            _clip01(len(set(_TOKEN_RE.findall(question.lower())) & set(tokens)) / token_count),
            _clip01(math.log1p(magnitude) / 8.0),
            _clip01(sum(text.lower().count(cue) for cue in ("wrong", "error", "mistake")) / 3.0),
            _clip01(answer_only),
            _clip01(sum(text.lower().count(cue) for cue in ("but", "however", "although")) / 3.0),
            _clip01(abs(len(text) - len(question)) / max(len(question), len(text), 1)),
        ],
        dtype=np.float32,
    )

    n_hash = n_features - 16
    if n_hash > 0:
        hashed = np.zeros(n_hash, dtype=np.float32)
        for token in tokens:
            hashed[_stable_hash_bucket(token, n_hash)] += 1.0
        if hashed.sum() > 0:
            hashed = hashed / hashed.sum()
        base[16:] = hashed
    return np.clip(base, 0.0, 1.0)


class EBMCoTKANEnergyCalibrator:
    """Small KAN energy wrapper trained with the EBM-CoT hinge objective."""

    def __init__(
        self,
        *,
        edge_ctrl: np.ndarray,
        output_ctrl: np.ndarray,
        n_features: int,
        n_hidden: int,
        n_knots: int,
        degree: int,
        checkpoint_info: KANCheckpointInfo,
    ) -> None:
        self.edge_ctrl = np.asarray(edge_ctrl, dtype=np.float32)
        self.output_ctrl = np.asarray(output_ctrl, dtype=np.float32)
        self.n_features = int(n_features)
        self.n_hidden = int(n_hidden)
        self.n_knots = int(n_knots)
        self.degree = int(degree)
        self.checkpoint_info = checkpoint_info
        self.loss_history: list[dict[str, float]] = []

    @classmethod
    def random_init(
        cls,
        *,
        n_features: int = DEFAULT_N_FEATURES,
        n_hidden: int = DEFAULT_N_HIDDEN,
        n_knots: int = DEFAULT_N_KNOTS,
        degree: int = DEFAULT_DEGREE,
        seed: int = 20260505,
        reason: str = "no compatible checkpoint found",
    ) -> "EBMCoTKANEnergyCalibrator":
        rng = np.random.default_rng(seed)
        n_ctrl = n_knots + degree
        edge_ctrl = rng.uniform(-0.1, 0.1, (n_hidden, n_features, n_ctrl)).astype(np.float32)
        output_ctrl = rng.uniform(-0.1, 0.1, (n_hidden, n_ctrl)).astype(np.float32)
        return cls(
            edge_ctrl=edge_ctrl,
            output_ctrl=output_ctrl,
            n_features=n_features,
            n_hidden=n_hidden,
            n_knots=n_knots,
            degree=degree,
            checkpoint_info=KANCheckpointInfo(False, None, "random_init", reason),
        )

    @classmethod
    def load_current_checkpoint(
        cls,
        models_dir: Path | str = DEFAULT_MODELS_DIR,
        *,
        n_features: int = DEFAULT_N_FEATURES,
    ) -> "EBMCoTKANEnergyCalibrator":
        """Load the first compatible local KAN JSON checkpoint.

        The repo's prompt-injection KAN checkpoint has the same low-energy /
        high-energy tier semantics and a shape-compatible two-layer KAN energy
        surface.  If the checkpoint shape drifts, the probe falls back to a
        deterministic random KAN and records that in the artifact.

        Spec: REQ-KAN-1384
        """

        candidates = (
            "prompt_injection_kan_weights.json",
            "prompt_injection_kan_v1_weights.json",
            "privacy_filter_kan_v2.json",
        )
        for name in candidates:
            path = Path(models_dir) / name
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                edge_ctrl = np.asarray(data["edge_ctrl"], dtype=np.float32)
                output_ctrl = np.asarray(data["output_ctrl"], dtype=np.float32)
                ckpt_features = int(data["n_features"])
                n_hidden = int(data["n_hidden"])
                n_knots = int(data.get("n_knots", DEFAULT_N_KNOTS))
                degree = int(data.get("degree", DEFAULT_DEGREE))
            except Exception as exc:
                last_reason = f"{path.name} unreadable: {exc}"
                continue

            expected_edge_shape = (n_hidden, ckpt_features, n_knots + degree)
            expected_output_shape = (n_hidden, n_knots + degree)
            if ckpt_features != n_features:
                last_reason = f"{path.name} has n_features={ckpt_features}, expected {n_features}"
                continue
            if edge_ctrl.shape != expected_edge_shape or output_ctrl.shape != expected_output_shape:
                last_reason = f"{path.name} has incompatible control-point shapes"
                continue
            return cls(
                edge_ctrl=edge_ctrl,
                output_ctrl=output_ctrl,
                n_features=ckpt_features,
                n_hidden=n_hidden,
                n_knots=n_knots,
                degree=degree,
                checkpoint_info=KANCheckpointInfo(
                    True,
                    str(path),
                    str(data.get("schema", "unknown")),
                    "loaded compatible KAN checkpoint",
                ),
            )

        return cls.random_init(reason=locals().get("last_reason", "no KAN JSON checkpoint found"))

    def energy_from_features(self, features: np.ndarray) -> float:
        features_j = jnp.asarray(features, dtype=jnp.float32)
        return float(
            _injection_energy(
                features_j,
                jnp.asarray(self.edge_ctrl),
                jnp.asarray(self.output_ctrl),
                self.n_knots,
                self.degree,
                self.n_features,
                self.n_hidden,
            )
        )

    def energy(self, case: FoVerStepCase) -> float:
        return self.energy_from_features(encode_fover_features(case, self.n_features))

    def evaluate_auroc(self, cases: list[FoVerStepCase]) -> float:
        labels = np.array([case.label for case in cases], dtype=np.float64)
        scores = np.array([-self.energy(case) for case in cases], dtype=np.float64)
        return auroc(labels, scores)

    def train_ebm_cot(
        self,
        positive_cases: list[FoVerStepCase],
        negative_cases: list[FoVerStepCase],
        *,
        n_epochs: int = DEFAULT_N_EPOCHS,
        hinge_margin: float = DEFAULT_HINGE_MARGIN,
        consistency_weight: float = DEFAULT_CONSISTENCY_WEIGHT,
        learning_rate: float = 2e-2,
    ) -> list[dict[str, float]]:
        """Train the KAN control points with EBM-CoT hinge + consistency loss."""

        n_pairs = min(len(positive_cases), len(negative_cases))
        if n_pairs == 0:
            self.loss_history = []
            return []

        positive_cases = positive_cases[:n_pairs]
        negative_cases = negative_cases[:n_pairs]
        paraphrases = [
            FoVerStepCase(
                case_id=f"{case.case_id}:paraphrase",
                question=case.question,
                step_text=paraphrase_positive_step(case.step_text),
                label=1,
            )
            for case in positive_cases
        ]

        pos_arr = jnp.asarray(
            np.stack([encode_fover_features(case, self.n_features) for case in positive_cases])
        )
        neg_arr = jnp.asarray(
            np.stack([encode_fover_features(case, self.n_features) for case in negative_cases])
        )
        para_arr = jnp.asarray(
            np.stack([encode_fover_features(case, self.n_features) for case in paraphrases])
        )

        params = (jnp.asarray(self.edge_ctrl), jnp.asarray(self.output_ctrl))

        def single_energy(edge_ctrl: jax.Array, output_ctrl: jax.Array, feats: jax.Array) -> jax.Array:
            return _injection_energy(
                feats,
                edge_ctrl,
                output_ctrl,
                self.n_knots,
                self.degree,
                self.n_features,
                self.n_hidden,
            )

        def loss_fn(param_tuple: tuple[jax.Array, jax.Array]) -> jax.Array:
            edge_ctrl, output_ctrl = param_tuple
            energy_fn = lambda feats: single_energy(edge_ctrl, output_ctrl, feats)
            e_pos = jax.vmap(energy_fn)(pos_arr)
            e_neg = jax.vmap(energy_fn)(neg_arr)
            e_para = jax.vmap(energy_fn)(para_arr)
            contrastive = jnp.mean(jax.nn.relu(hinge_margin - (e_neg - e_pos)))
            consistency = jnp.mean(jnp.abs(e_pos - e_para))
            regularizer = 1e-5 * (jnp.mean(edge_ctrl**2) + jnp.mean(output_ctrl**2))
            return contrastive + consistency_weight * consistency + regularizer

        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)
        grad_fn = jax.jit(jax.value_and_grad(loss_fn))
        history: list[dict[str, float]] = []
        for _ in range(n_epochs):
            loss_value, grads = grad_fn(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            components = self._loss_components_for_params(
                params,
                pos_arr,
                neg_arr,
                para_arr,
                hinge_margin,
                consistency_weight,
            )
            components["loss"] = float(loss_value)
            history.append(components)

        self.edge_ctrl = np.asarray(params[0], dtype=np.float32)
        self.output_ctrl = np.asarray(params[1], dtype=np.float32)
        self.loss_history = history
        return history

    def _loss_components_for_params(
        self,
        params: tuple[jax.Array, jax.Array],
        pos_arr: jax.Array,
        neg_arr: jax.Array,
        para_arr: jax.Array,
        hinge_margin: float,
        consistency_weight: float,
    ) -> dict[str, float]:
        edge_ctrl, output_ctrl = params
        energy_fn = lambda feats: _injection_energy(
            feats,
            edge_ctrl,
            output_ctrl,
            self.n_knots,
            self.degree,
            self.n_features,
            self.n_hidden,
        )
        e_pos = np.asarray(jax.vmap(energy_fn)(pos_arr))
        e_neg = np.asarray(jax.vmap(energy_fn)(neg_arr))
        e_para = np.asarray(jax.vmap(energy_fn)(para_arr))
        return ebm_cot_loss_components(
            e_positive=e_pos,
            e_negative=e_neg,
            e_positive_paraphrase=e_para,
            hinge_margin=hinge_margin,
            consistency_weight=consistency_weight,
        )


def ebm_cot_loss_components(
    *,
    e_positive: np.ndarray,
    e_negative: np.ndarray,
    e_positive_paraphrase: np.ndarray,
    hinge_margin: float,
    consistency_weight: float,
) -> dict[str, float]:
    """Return EBM-CoT loss components from precomputed energy arrays.

    Spec: REQ-KAN-1384
    """

    e_pos = np.asarray(e_positive, dtype=np.float64)
    e_neg = np.asarray(e_negative, dtype=np.float64)
    e_para = np.asarray(e_positive_paraphrase, dtype=np.float64)
    contrastive = float(np.maximum(0.0, hinge_margin - (e_neg - e_pos)).mean())
    consistency = float(np.abs(e_pos - e_para).mean())
    return {
        "contrastive": contrastive,
        "consistency": consistency,
        "total": contrastive + float(consistency_weight) * consistency,
        "mean_energy_gap": float(np.mean(e_neg - e_pos)),
    }


def paraphrase_energy_delta_variance(
    calibrator: EBMCoTKANEnergyCalibrator,
    positive_cases: list[FoVerStepCase],
) -> float:
    """Variance of original-vs-paraphrase energy deltas for positive cases."""

    deltas = []
    for case in positive_cases:
        para = FoVerStepCase(
            case_id=f"{case.case_id}:paraphrase",
            question=case.question,
            step_text=paraphrase_positive_step(case.step_text),
            label=1,
        )
        deltas.append(calibrator.energy(case) - calibrator.energy(para))
    return float(np.var(np.asarray(deltas, dtype=np.float64))) if deltas else 0.0


def build_artifact(
    *,
    split: FoVerSplit,
    checkpoint_info: KANCheckpointInfo,
    baseline_auroc: float,
    ebm_cot_auroc: float,
    variance_before: float,
    variance_after: float,
    loss_history: list[dict[str, float]],
    started_at: float,
    hinge_margin: float = DEFAULT_HINGE_MARGIN,
    consistency_weight: float = DEFAULT_CONSISTENCY_WEIGHT,
    n_epochs: int = DEFAULT_N_EPOCHS,
) -> dict[str, Any]:
    """Build the experiment 1384 JSON payload with all required fields."""

    calibration_delta = float(ebm_cot_auroc - baseline_auroc)
    variance_reduction = float(variance_before - variance_after)
    viable = calibration_delta > 0.0
    if viable and variance_reduction >= 0.0:
        verdict = "implicit_cot_energy_viable"
    elif viable:
        verdict = "implicit_cot_energy_viable_consistency_variance_worsened"
    elif checkpoint_info.loaded:
        verdict = "no_calibration_improvement"
    else:
        verdict = "no_calibration_improvement_checkpoint_fallback"
    return {
        "status": "complete",
        "run_date": "20260505",
        "experiment": 1384,
        "title": "EBM-CoT KAN energy calibration probe on FoVer pairs",
        "corpus_cases_used": split.corpus_cases_used,
        "training_method": (
            "EBM-CoT contrastive hinge loss on balanced FoVer correct/incorrect "
            f"step pairs for {n_epochs} CPU epochs, plus deterministic positive "
            "paraphrase consistency regularization"
        ),
        "hinge_margin": float(hinge_margin),
        "consistency_regularization_weight": float(consistency_weight),
        "baseline_auroc": float(baseline_auroc),
        "ebm_cot_auroc": float(ebm_cot_auroc),
        "calibration_auroc_delta": calibration_delta,
        "consistency_regularization_effect": variance_reduction,
        "implicit_cot_energy_viable": viable,
        "honest_verdict": verdict,
        "checkpoint_loaded": checkpoint_info.loaded,
        "checkpoint_path": checkpoint_info.path,
        "checkpoint_schema": checkpoint_info.schema,
        "checkpoint_note": checkpoint_info.reason,
        "n_train_pairs": len(split.train_positive),
        "n_test_cases": len(split.test_cases),
        "n_epochs": int(n_epochs),
        "paraphrase_energy_delta_variance_before": float(variance_before),
        "paraphrase_energy_delta_variance_after": float(variance_after),
        "final_loss": loss_history[-1] if loss_history else {},
        "duration_s": round(time.time() - started_at, 3),
        "paper_reference": "arXiv:2511.07124",
        "source_reference": "research-references.md EBM-CoT entry and https://arxiv.org/abs/2511.07124",
    }


def run_probe(
    *,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    artifact_path: Path | str = DEFAULT_ARTIFACT_PATH,
    n_epochs: int = DEFAULT_N_EPOCHS,
    hinge_margin: float = DEFAULT_HINGE_MARGIN,
    consistency_weight: float = DEFAULT_CONSISTENCY_WEIGHT,
    max_pairs_per_class: int | None = None,
) -> dict[str, Any]:
    """Run the CPU-only EBM-CoT KAN calibration probe and write its artifact."""

    started_at = time.time()
    cases = load_fover_verified_cases(fover_path)
    split = make_balanced_split(cases, max_pairs_per_class=max_pairs_per_class)
    calibrator = EBMCoTKANEnergyCalibrator.load_current_checkpoint(models_dir)

    baseline_auroc = calibrator.evaluate_auroc(split.test_cases)
    test_positive = [case for case in split.test_cases if case.label == 1]
    variance_before = paraphrase_energy_delta_variance(calibrator, test_positive)
    loss_history = calibrator.train_ebm_cot(
        split.train_positive,
        split.train_negative,
        n_epochs=n_epochs,
        hinge_margin=hinge_margin,
        consistency_weight=consistency_weight,
    )
    ebm_cot_auroc = calibrator.evaluate_auroc(split.test_cases)
    variance_after = paraphrase_energy_delta_variance(calibrator, test_positive)

    artifact = build_artifact(
        split=split,
        checkpoint_info=calibrator.checkpoint_info,
        baseline_auroc=baseline_auroc,
        ebm_cot_auroc=ebm_cot_auroc,
        variance_before=variance_before,
        variance_after=variance_after,
        loss_history=loss_history,
        started_at=started_at,
        hinge_margin=hinge_margin,
        consistency_weight=consistency_weight,
        n_epochs=n_epochs,
    )

    target = Path(artifact_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fover-path", type=Path, default=DEFAULT_FOVER_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--epochs", type=int, default=DEFAULT_N_EPOCHS)
    parser.add_argument("--hinge-margin", type=float, default=DEFAULT_HINGE_MARGIN)
    parser.add_argument("--consistency-weight", type=float, default=DEFAULT_CONSISTENCY_WEIGHT)
    parser.add_argument("--max-pairs-per-class", type=int, default=None)
    args = parser.parse_args()

    artifact = run_probe(
        fover_path=args.fover_path,
        models_dir=args.models_dir,
        artifact_path=args.artifact_path,
        n_epochs=args.epochs,
        hinge_margin=args.hinge_margin,
        consistency_weight=args.consistency_weight,
        max_pairs_per_class=args.max_pairs_per_class,
    )
    print(
        artifact["calibration_auroc_delta"],
        artifact["implicit_cot_energy_viable"],
        artifact["honest_verdict"],
    )


if __name__ == "__main__":
    main()
