"""Typed chain-of-thought verifier for cached telemetry.

The verifier maps reasoning steps to a tiny Curry-Howard-inspired surface type
system: factual premises become propositions, connective-bearing steps become
inferences, and final deductive cue steps become conclusions.  It intentionally
uses deterministic text patterns instead of full proof search or SMT encoding.

Spec: REQ-TIER28-002, SCENARIO-TIER28-002, Exp 2396.
"""

from __future__ import annotations

import datetime as _datetime
import json
import math
import re
import time
from enum import Enum
from pathlib import Path
from typing import Any

from carnot.verify.semantic_energy import binary_auroc

JsonDict = dict[str, Any]

DEFAULT_MANIFEST_PATH = Path("results/live_sota_balanced_telemetry_manifest_1480.jsonl")
DEFAULT_OUTPUT_PATH = Path("results/experiment_2396_typed_cot.json")
DEFAULT_RANDOM_SEED = 42
SEMANTIC_ENERGY_BASELINE_AUROC = 0.685
TERMINAL_VERDICT_PREFIXES = ("complete:", "success:", "blocked:", "failed:")

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "typed_cot_validated",
    "typed_cot_auroc",
    "typed_cot_mean_score",
    "typed_cot_vs_semantic_energy_delta",
    "cot_fields_found",
    "n_eval_examples",
    "random_seed",
    "duration_s",
    "preconditions_checked",
}

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix required.",
    "typed_cot_validated": "True if TypedCoTVerifier ran on real data.",
    "typed_cot_auroc": (
        "Primary metric. May be low if telemetry lacks CoT fields — honest result."
    ),
    "typed_cot_vs_semantic_energy_delta": (
        "Delta vs baseline 0.685. Key improvement signal."
    ),
    "cot_fields_found": (
        "Which telemetry fields contained CoT content (empty list = no CoT in corpus)."
    ),
    "n_eval_examples": "Must be 36.",
    "random_seed": "Must be 42.",
    "duration_s": "Guards against fabrication.",
    "preconditions_checked": "Records telemetry manifest check.",
}

_COT_FIELDS = ("chain_of_thought", "reasoning", "response_text")
_THINK_RE = re.compile(r"<think>(?P<body>.*?)</think>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
_BOLD_RE = re.compile(r"[*_`]+")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+\s*[.)]\s*)")
_STEP_PREFIX_RE = re.compile(r"^\s*(?:step\s*)?\d+\s*[:.)-]\s*", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")

_CONCLUSION_START_RE = re.compile(
    r"^\s*(?:thus|therefore|so|hence|consequently|in conclusion|as a result)\b",
    re.IGNORECASE,
)
_CONCLUSION_ANY_RE = re.compile(
    r"\b(?:thus|therefore|hence|consequently|as a result)\b",
    re.IGNORECASE,
)
_INFERENCE_RE = re.compile(
    r"\b(?:because|since|if\b.+\bthen|then|implies?|entails?|follows from|"
    r"therefore|so|given that|as)\b",
    re.IGNORECASE,
)
_DETERMINER_RE = re.compile(r"^\s*(?:the|a|an|this|that|these|those)\b", re.IGNORECASE)
_NOUN_PHRASE_STOPWORDS = {
    "analyze",
    "answer",
    "because",
    "command",
    "consequently",
    "final",
    "first",
    "hence",
    "here",
    "if",
    "input",
    "instruction",
    "process",
    "reply",
    "return",
    "so",
    "step",
    "task",
    "then",
    "therefore",
    "thus",
}


class StepType(str, Enum):
    """Surface proof types used by `TypedCoTVerifier`."""

    PROPOSITION = "Proposition"
    INFERENCE = "Inference"
    CONCLUSION = "Conclusion"
    UNKNOWN = "Unknown"


class TypedCoTVerifier:
    """Deterministic Typed CoT checker for lightweight proof-structure scoring.

    The checker does not prove semantic entailment.  It verifies the local type
    discipline requested for the Tier 2.8 probe: inferences need an earlier
    proposition, conclusions need an earlier valid inference, and unsupported
    steps are counted as failed type checks.

    Spec: REQ-TIER28-002.
    """

    def split_steps(self, text: str) -> list[str]:
        """Split a free-form response or CoT field into candidate proof steps."""

        cleaned = _clean_cot_text(text)
        if not cleaned:
            return []

        steps: list[str] = []
        for raw_line in cleaned.splitlines():
            line = _normalize_step_text(raw_line)
            if not line:
                continue
            for part in _SENTENCE_SPLIT_RE.split(line):
                step = _normalize_step_text(part)
                if step:
                    steps.append(step)
        return steps

    def classify_step(self, text: str, index: int, total: int) -> StepType:
        """Assign one heuristic Curry-Howard proof type to a reasoning step."""

        normalized = _normalize_step_text(text)
        if not normalized:
            return StepType.UNKNOWN

        if _CONCLUSION_START_RE.search(normalized):
            return StepType.CONCLUSION
        if index == total - 1 and _CONCLUSION_ANY_RE.search(normalized):
            return StepType.CONCLUSION
        if _INFERENCE_RE.search(normalized):
            return StepType.INFERENCE
        if _looks_like_proposition(normalized):
            return StepType.PROPOSITION
        return StepType.UNKNOWN

    def verify_text(self, text: str) -> JsonDict:
        """Type-check all extracted steps and return a score in `[0, 1]`."""

        steps = self.split_steps(text)
        typed_steps: list[JsonDict] = []
        propositions_seen = 0
        valid_inferences_seen = 0
        passed = 0

        for index, step in enumerate(steps):
            step_type = self.classify_step(step, index, len(steps))
            type_checks, reason = self._check_step_type(
                step_type,
                propositions_seen=propositions_seen,
                valid_inferences_seen=valid_inferences_seen,
            )
            if type_checks:
                passed += 1
                if step_type is StepType.PROPOSITION:
                    propositions_seen += 1
                elif step_type is StepType.INFERENCE:
                    valid_inferences_seen += 1

            typed_steps.append(
                {
                    "index": index,
                    "text": step,
                    "type": step_type.value,
                    "type_checks": bool(type_checks),
                    "check_reason": reason,
                }
            )

        score = float(passed / len(steps)) if steps else 0.0
        return {
            "typed_cot_score": score,
            "n_steps": len(steps),
            "n_typechecked_steps": passed,
            "typed_steps": typed_steps,
        }

    @staticmethod
    def _check_step_type(
        step_type: StepType,
        *,
        propositions_seen: int,
        valid_inferences_seen: int,
    ) -> tuple[bool, str]:
        if step_type is StepType.PROPOSITION:
            return True, "proposition introduces a premise"
        if step_type is StepType.INFERENCE:
            if propositions_seen > 0:
                return True, "inference follows at least one proposition"
            return False, "inference has no preceding proposition"
        if step_type is StepType.CONCLUSION:
            if valid_inferences_seen > 0:
                return True, "conclusion follows at least one valid inference"
            return False, "conclusion has no preceding valid inference"
        return False, "step type is unknown"


def extract_cot_text(entry: JsonDict) -> tuple[str, str | None]:
    """Return the first available CoT-bearing telemetry field and its name."""

    for field_name in _COT_FIELDS:
        value = entry.get(field_name)
        if isinstance(value, str) and value.strip():
            return value, field_name
    return "", None


def label_from_entry(entry: JsonDict) -> int:
    """Return `1` for incorrect/hallucination-like rows and `0` for correct rows."""

    correctness = str(entry.get("correctness_label", "")).strip().lower()
    if correctness == "incorrect":
        return 1
    if correctness == "correct":
        return 0
    if entry.get("correct") is False:
        return 1
    if entry.get("correct") is True:
        return 0
    raise ValueError("entry does not contain a binary correctness label")


def build_experiment_artifact(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    n_eval_examples: int = 36,
    random_seed: int = DEFAULT_RANDOM_SEED,
    semantic_energy_baseline: float = SEMANTIC_ENERGY_BASELINE_AUROC,
) -> JsonDict:
    """Evaluate TypedCoTVerifier on cached live SOTA telemetry."""

    start = time.perf_counter()
    manifest = Path(manifest_path)
    preconditions = _preconditions(manifest)
    if not preconditions["telemetry_manifest_present"]:
        return _blocked_artifact(
            honest_verdict="blocked_telemetry_manifest_missing",
            random_seed=random_seed,
            duration_s=round(time.perf_counter() - start, 6),
            preconditions=preconditions,
        )
    if not preconditions["sklearn_importable"]:
        return _blocked_artifact(
            honest_verdict="blocked_sklearn_missing",
            random_seed=random_seed,
            duration_s=round(time.perf_counter() - start, 6),
            preconditions=preconditions,
        )

    entries = _read_jsonl(manifest, limit=n_eval_examples)
    verifier = TypedCoTVerifier()
    row_results: list[JsonDict] = []
    labels: list[int] = []
    typed_scores: list[float] = []
    risk_scores: list[float] = []
    cot_fields_seen: list[str] = []

    for entry in entries:
        label = label_from_entry(entry)
        cot_text, cot_field = extract_cot_text(entry)
        if cot_field is not None and cot_field not in cot_fields_seen:
            cot_fields_seen.append(cot_field)

        result = verifier.verify_text(cot_text)
        typed_score = float(result["typed_cot_score"])
        risk_score = float(1.0 - typed_score)
        labels.append(label)
        typed_scores.append(typed_score)
        risk_scores.append(risk_score)
        row_results.append(
            {
                "case_id": entry.get("case_id"),
                "label": int(label),
                "cot_field": cot_field,
                "typed_cot_score": typed_score,
                "typed_cot_risk_score": risk_score,
                "typed_step_count": int(result["n_steps"]),
                "typechecked_step_count": int(result["n_typechecked_steps"]),
                "typed_steps": result["typed_steps"][:5],
            }
        )

    auroc = _compute_auroc(labels, risk_scores)
    mean_score = float(sum(typed_scores) / len(typed_scores)) if typed_scores else 0.0
    validated = bool(len(entries) == n_eval_examples and all(math.isfinite(s) for s in risk_scores))
    duration_s = round(time.perf_counter() - start, 6)

    artifact: JsonDict = {
        "status": "complete" if validated else "failed",
        "experiment": 2396,
        "title": "Typed CoT verifier telemetry validation",
        "module_path": "python/carnot/verify/typed_cot.py",
        "run_date": _datetime.date.today().isoformat(),
        "spec_refs": ["REQ-TIER28-002", "SCENARIO-TIER28-002"],
        "candidate_stage": "Tier 2.8 after Tier 2 VERGE repair and before Tier 3 Ising",
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": _honest_verdict(validated, auroc, len(entries), cot_fields_seen),
        "typed_cot_validated": validated,
        "typed_cot_auroc": auroc,
        "typed_cot_mean_score": mean_score,
        "typed_cot_vs_semantic_energy_delta": float(auroc - semantic_energy_baseline),
        "semantic_energy_baseline_auroc": float(semantic_energy_baseline),
        "cot_fields_found": cot_fields_seen,
        "n_eval_examples": len(entries),
        "n_factual_examples": int(labels.count(0)),
        "n_hallucination_examples": int(labels.count(1)),
        "random_seed": int(random_seed),
        "duration_s": duration_s,
        "preconditions_checked": preconditions,
        "score_direction": "higher_typed_cot_risk_score_means_more_hallucination_like",
        "score_field": "typed_cot_risk_score = 1.0 - typed_cot_score",
        "evaluation_design": (
            "Load the first 36 live SOTA balanced telemetry rows, extract "
            "chain_of_thought/reasoning/response_text in priority order, type-check "
            "the extracted steps, and score incorrect rows as hallucination-like."
        ),
        "source_artifact": str(manifest),
        "per_entry_results": row_results,
        "test_command": (
            'PYTEST_ADDOPTS="" .venv/bin/python -m pytest tests/python/verify/ '
            '-k "typed_cot" -v --no-cov 2>&1 | tail -15'
        ),
        "acceptance_gates": {"typed_cot_validated": validated},
    }
    validate_experiment_artifact(artifact, expected_n_eval_examples=n_eval_examples)
    return artifact


def write_experiment_artifact(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> JsonDict:
    """Write `results/experiment_2396_typed_cot.json` and return the artifact."""

    artifact = build_experiment_artifact(manifest_path=manifest_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_experiment_artifact(
    artifact: JsonDict,
    *,
    expected_n_eval_examples: int = 36,
) -> None:
    """Validate required Exp 2396 artifact fields and acceptance invariants."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")

    verdict = str(artifact["honest_verdict"])
    if verdict != "blocked_telemetry_manifest_missing" and not verdict.startswith(
        TERMINAL_VERDICT_PREFIXES
    ):
        raise ValueError("honest_verdict must start with a terminal prefix")
    if int(artifact["random_seed"]) != DEFAULT_RANDOM_SEED:
        raise ValueError(f"random_seed must be {DEFAULT_RANDOM_SEED}")
    if int(artifact["n_eval_examples"]) != expected_n_eval_examples:
        raise ValueError(f"n_eval_examples must be {expected_n_eval_examples}")

    if artifact["typed_cot_auroc"] is not None:
        auroc = float(artifact["typed_cot_auroc"])
        if not 0.0 <= auroc <= 1.0:
            raise ValueError("typed_cot_auroc must be in [0, 1]")
        delta = float(artifact["typed_cot_vs_semantic_energy_delta"])
        expected_delta = auroc - float(artifact.get("semantic_energy_baseline_auroc", 0.685))
        if not math.isclose(delta, expected_delta, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("typed_cot_vs_semantic_energy_delta must match baseline delta")

    if artifact["status"] == "complete" and not bool(artifact["typed_cot_validated"]):
        raise ValueError("complete artifacts must set typed_cot_validated=true")


def _read_jsonl(path: Path, limit: int | None = None) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _preconditions(manifest_path: Path) -> JsonDict:
    checked: JsonDict = {
        "telemetry_manifest_present": manifest_path.is_file(),
        "telemetry_manifest_path": str(manifest_path),
    }
    try:
        import sklearn  # noqa: PLC0415
    except ModuleNotFoundError:
        checked.update({"sklearn_importable": False, "sklearn_version": None})
    else:
        checked.update({"sklearn_importable": True, "sklearn_version": sklearn.__version__})

    if manifest_path.is_file():
        checked["telemetry_rows_available"] = sum(1 for line in manifest_path.open() if line.strip())
    else:
        checked["telemetry_rows_available"] = 0
    return checked


def _compute_auroc(labels: list[int], scores: list[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score  # noqa: PLC0415
    except ModuleNotFoundError:
        return binary_auroc(labels, scores)
    return float(roc_auc_score(labels, scores))


def _blocked_artifact(
    *,
    honest_verdict: str,
    random_seed: int,
    duration_s: float,
    preconditions: JsonDict,
) -> JsonDict:
    return {
        "status": "blocked",
        "experiment": 2396,
        "title": "Typed CoT verifier telemetry validation",
        "module_path": "python/carnot/verify/typed_cot.py",
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": honest_verdict,
        "typed_cot_validated": False,
        "typed_cot_auroc": None,
        "typed_cot_mean_score": None,
        "typed_cot_vs_semantic_energy_delta": None,
        "cot_fields_found": [],
        "n_eval_examples": 0,
        "random_seed": int(random_seed),
        "duration_s": duration_s,
        "preconditions_checked": preconditions,
    }


def _honest_verdict(validated: bool, auroc: float, n_examples: int, fields: list[str]) -> str:
    field_text = ", ".join(fields) if fields else "no CoT fields"
    if validated:
        return (
            "complete: TypedCoTVerifier ran on "
            f"{n_examples} cached telemetry entries; AUROC={auroc:.6f}; "
            f"fields={field_text}."
        )
    return (
        "failed: TypedCoTVerifier did not complete the requested telemetry run "
        f"(n={n_examples}, AUROC={auroc:.6f}, fields={field_text})."
    )


def _clean_cot_text(text: str) -> str:
    if not text:
        return ""

    think_match = _THINK_RE.search(text)
    if think_match is not None and think_match.group("body").strip():
        text = think_match.group("body")

    cleaned = _TAG_RE.sub("\n", text)
    cleaned = _BOLD_RE.sub("", cleaned)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*", "\n", cleaned)
    return cleaned.strip()


def _normalize_step_text(text: str) -> str:
    stripped = text.strip()
    stripped = _LIST_PREFIX_RE.sub("", stripped)
    stripped = _STEP_PREFIX_RE.sub("", stripped)
    stripped = stripped.strip(" -:\t")
    return _SPACE_RE.sub(" ", stripped).strip()


def _looks_like_proposition(text: str) -> bool:
    if _DETERMINER_RE.search(text):
        return True

    tokens = _TOKEN_RE.findall(text)
    if len(tokens) < 2:
        return False
    first = tokens[0]
    if first.lower() in _NOUN_PHRASE_STOPWORDS:
        return False
    if not first[:1].isupper():
        return False
    if text.endswith(":"):
        return False
    return True


if __name__ == "__main__":
    print(json.dumps(write_experiment_artifact(), indent=2, sort_keys=True))
