"""Deterministic EBT/ARM sidecar schema and replay scoring.

The sidecar boundary is intentionally boring: it accepts only cached JSON rows
and recomputes inspectable energy terms from those rows. That gives future
EBT/ARM work a concrete data shape without implying model training, live model
inference, hidden weight access, benchmark speedup, or hardware acceleration.

Spec refs: REQ-VERIFY-3091, SCENARIO-VERIFY-3091.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


JsonDict = dict[str, Any]

SCHEMA_REL_PATH = Path("python/carnot/schemas/ebt_arm_sidecar_adapter_v1.json")
REQUIRED_SIDECAR_FIELDS = frozenset(
    {
        "record_id",
        "candidate",
        "constraints",
        "energy_terms",
        "verifier_feedback",
        "confidence",
        "exact_label_reference",
        "source_artifacts",
    }
)
REPLAY_INFERENCE_SUBSTRATE: JsonDict = {
    "kind": "deterministic_cached_sidecar_replay",
    "live_model_inference": False,
    "live_llm_inference": False,
    "model_weights_loaded": False,
    "generation_performed": False,
    "gpu_required": False,
}

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_VERIFIER_STATUSES = {"pass", "fail", "abstain", "not_run"}
_LABEL_AUTHORITIES = {"exact_solver", "deterministic_tests", "human_audited_fixture"}


@dataclass(frozen=True)
class ReplayWeights:
    """Weights for deterministic replay energy components.

    These defaults make exact labels and explicit violations dominate logprob
    confidence. That is deliberate: ARM logprobs are useful confidence
    telemetry, but they are not correctness authority.
    """

    arm_sequence: float = 0.1
    verifier_feedback: float = 1.0
    confidence: float = 1.0
    abstention: float = 5.0
    exact_label_mismatch: float = 20.0


@dataclass(frozen=True)
class SidecarReplayScore:
    """Replay result with decomposed terms and no-live-inference metadata."""

    record_id: str
    candidate_id: str
    total_energy: float
    energy_terms: list[JsonDict]
    confidence: float
    abstain: bool
    input_fingerprint: str
    inference_substrate: JsonDict

    def to_json(self) -> JsonDict:
        """Return a JSON-serializable copy for experiment artifacts."""

        return asdict(self)


class SidecarReplayScorer:
    """Score cached sidecar rows with deterministic local arithmetic only."""

    def __init__(
        self, weights: ReplayWeights | None = None, schema: JsonDict | None = None
    ) -> None:
        self.weights = weights or ReplayWeights()
        self.schema = schema

    def score(self, record: JsonDict) -> SidecarReplayScore:
        """Compute the canonical sidecar energy components for one cached row."""

        schema = self.schema or load_sidecar_schema()
        validate_sidecar_record(record, schema)

        candidate = record["candidate"]
        confidence = record["confidence"]
        exact_label = record["exact_label_reference"]

        terms = [
            self._term(
                "constraint_violation_energy",
                _constraint_violation_energy(record["constraints"]),
                1.0,
                "cached_constraints",
            ),
            self._term(
                "arm_sequence_energy",
                _arm_sequence_energy(candidate.get("token_logprobs", [])),
                self.weights.arm_sequence,
                "cached_token_logprobs",
            ),
            self._term(
                "verifier_feedback_energy",
                _verifier_feedback_energy(record["verifier_feedback"]),
                self.weights.verifier_feedback,
                "cached_verifier_feedback",
            ),
            self._term(
                "confidence_energy",
                1.0 - float(confidence["confidence"]),
                self.weights.confidence,
                "cached_confidence_metadata",
            ),
            self._term(
                "abstention_energy",
                1.0 if confidence["abstain"] else 0.0,
                self.weights.abstention,
                "cached_abstention_metadata",
            ),
            self._term(
                "exact_label_mismatch_energy",
                1.0 if candidate["candidate_label"] != exact_label["label"] else 0.0,
                self.weights.exact_label_mismatch,
                "cached_exact_label_reference",
            ),
        ]
        total = _stable_float(sum(float(term["weighted_value"]) for term in terms))
        return SidecarReplayScore(
            record_id=str(record["record_id"]),
            candidate_id=str(candidate["candidate_id"]),
            total_energy=total,
            energy_terms=terms,
            confidence=float(confidence["confidence"]),
            abstain=bool(confidence["abstain"]),
            input_fingerprint=canonical_fingerprint(record),
            inference_substrate=dict(REPLAY_INFERENCE_SUBSTRATE),
        )

    @staticmethod
    def _term(name: str, raw_value: float, weight: float, source: str) -> JsonDict:
        raw = _stable_float(raw_value)
        stable_weight = _stable_float(weight)
        return {
            "name": name,
            "raw_value": raw,
            "weight": stable_weight,
            "weighted_value": _stable_float(raw * stable_weight),
            "source": source,
        }


def load_sidecar_schema(repo_root: Path | None = None) -> JsonDict:
    """Load the packaged sidecar JSON schema from the repository tree."""

    root = repo_root or Path(__file__).resolve().parents[3]
    return json.loads((root / SCHEMA_REL_PATH).read_text(encoding="utf-8"))


def validate_sidecar_record(record: JsonDict, schema: JsonDict | None = None) -> None:
    """Validate the subset of JSON Schema needed by the sidecar fixture rows."""

    active_schema = schema or load_sidecar_schema()
    _require_mapping(record, "$")
    _require_fields(record, REQUIRED_SIDECAR_FIELDS, "$")
    _reject_extra_fields(record, set(active_schema["properties"]), "$")

    candidate = _require_mapping(record["candidate"], "$.candidate")
    _require_fields(
        candidate,
        {"candidate_id", "prompt_id", "candidate_text", "candidate_label", "model_id"},
        "$.candidate",
    )
    _reject_extra_fields(
        candidate, set(active_schema["properties"]["candidate"]["properties"]), "$.candidate"
    )
    for field in ("candidate_id", "prompt_id", "candidate_text", "candidate_label", "model_id"):
        _require_non_empty_string(candidate[field], f"$.candidate.{field}")
    _require_numbers(candidate.get("token_logprobs", []), "$.candidate.token_logprobs")

    _validate_constraints(record["constraints"])
    _validate_energy_terms(record["energy_terms"])
    _validate_verifier_feedback(record["verifier_feedback"])
    _validate_confidence(record["confidence"])
    _validate_exact_label(record["exact_label_reference"])
    _validate_source_artifacts(record["source_artifacts"])


def canonical_fingerprint(record: JsonDict) -> str:
    """Hash the canonical cached row so replay results can cite exact input."""

    payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def example_sidecar_records() -> list[JsonDict]:
    """Return deterministic fixture rows for schema validation and replay tests."""

    return [
        {
            "record_id": "exp3091-correct",
            "candidate": {
                "candidate_id": "sidecar-fixture-correct",
                "prompt_id": "exp3091-prompt-arithmetic",
                "candidate_text": "2 + 3 = 5",
                "candidate_label": "correct",
                "model_id": "cached-fixture/no-live-model",
                "token_logprobs": [-0.1, -0.2],
            },
            "constraints": [
                {
                    "constraint_id": "sum-equals-five",
                    "description": "The arithmetic sum must equal the exact label.",
                    "satisfied": True,
                    "weight": 2.0,
                    "violation_energy": 1.5,
                    "label_ref": "label-arithmetic-correct",
                }
            ],
            "energy_terms": [
                {
                    "name": "cached_constraint_violation_energy",
                    "source": "fixture_author",
                    "value": 0.0,
                    "weight": 1.0,
                }
            ],
            "verifier_feedback": [
                {
                    "verifier_id": "deterministic-arithmetic-check",
                    "status": "pass",
                    "energy": 0.0,
                    "message": "Exact arithmetic check passed.",
                    "violations": [],
                }
            ],
            "confidence": {
                "confidence": 0.95,
                "abstain": False,
                "abstention_reason": "",
                "calibration_ref": "exp3091-fixture",
            },
            "exact_label_reference": {
                "label_id": "label-arithmetic-correct",
                "label": "correct",
                "authority": "exact_solver",
                "source_artifact": "results/experiment_3073_ebt_arm_ebm_adapter_feasibility_audit_v1.json",
                "checksum": _sha256_text("label-arithmetic-correct:correct"),
            },
            "source_artifacts": [
                "results/experiment_3073_ebt_arm_ebm_adapter_feasibility_audit_v1.json"
            ],
        },
        {
            "record_id": "exp3091-incorrect",
            "candidate": {
                "candidate_id": "sidecar-fixture-incorrect",
                "prompt_id": "exp3091-prompt-arithmetic",
                "candidate_text": "2 + 3 = 6",
                "candidate_label": "incorrect",
                "model_id": "cached-fixture/no-live-model",
                "token_logprobs": [-1.0, -2.0],
            },
            "constraints": [
                {
                    "constraint_id": "sum-equals-five",
                    "description": "The arithmetic sum must equal the exact label.",
                    "satisfied": False,
                    "weight": 2.0,
                    "violation_energy": 1.5,
                    "label_ref": "label-arithmetic-correct",
                }
            ],
            "energy_terms": [
                {
                    "name": "cached_constraint_violation_energy",
                    "source": "fixture_author",
                    "value": 3.0,
                    "weight": 1.0,
                }
            ],
            "verifier_feedback": [
                {
                    "verifier_id": "deterministic-arithmetic-check",
                    "status": "fail",
                    "energy": 2.0,
                    "message": "Exact arithmetic check rejected the candidate.",
                    "violations": ["2 + 3 evaluated to 5, not 6"],
                }
            ],
            "confidence": {
                "confidence": 0.4,
                "abstain": True,
                "abstention_reason": "constraint_violation",
                "calibration_ref": "exp3091-fixture",
            },
            "exact_label_reference": {
                "label_id": "label-arithmetic-correct",
                "label": "correct",
                "authority": "exact_solver",
                "source_artifact": "results/experiment_3073_ebt_arm_ebm_adapter_feasibility_audit_v1.json",
                "checksum": _sha256_text("label-arithmetic-correct:correct"),
            },
            "source_artifacts": [
                "results/experiment_3073_ebt_arm_ebm_adapter_feasibility_audit_v1.json"
            ],
        },
    ]


def _constraint_violation_energy(constraints: list[JsonDict]) -> float:
    return sum(
        float(row["weight"]) * float(row["violation_energy"])
        for row in constraints
        if not bool(row["satisfied"])
    )


def _arm_sequence_energy(token_logprobs: list[float]) -> float:
    return -sum(float(value) for value in token_logprobs)


def _verifier_feedback_energy(feedback_rows: list[JsonDict]) -> float:
    return sum(float(row["energy"]) for row in feedback_rows)


def _validate_constraints(value: Any) -> None:
    rows = _require_non_empty_list(value, "$.constraints")
    for index, row in enumerate(rows):
        path = f"$.constraints[{index}]"
        item = _require_mapping(row, path)
        _require_fields(
            item,
            {
                "constraint_id",
                "description",
                "satisfied",
                "weight",
                "violation_energy",
                "label_ref",
            },
            path,
        )
        _reject_extra_fields(
            item,
            {
                "constraint_id",
                "description",
                "satisfied",
                "weight",
                "violation_energy",
                "label_ref",
            },
            path,
        )
        for field in ("constraint_id", "description", "label_ref"):
            _require_non_empty_string(item[field], f"{path}.{field}")
        _require_bool(item["satisfied"], f"{path}.satisfied")
        _require_nonnegative_number(item["weight"], f"{path}.weight")
        _require_nonnegative_number(item["violation_energy"], f"{path}.violation_energy")


def _validate_energy_terms(value: Any) -> None:
    rows = _require_list(value, "$.energy_terms")
    for index, row in enumerate(rows):
        path = f"$.energy_terms[{index}]"
        item = _require_mapping(row, path)
        _require_fields(item, {"name", "source", "value", "weight"}, path)
        _reject_extra_fields(item, {"name", "source", "value", "weight"}, path)
        _require_non_empty_string(item["name"], f"{path}.name")
        _require_non_empty_string(item["source"], f"{path}.source")
        _require_number(item["value"], f"{path}.value")
        _require_nonnegative_number(item["weight"], f"{path}.weight")


def _validate_verifier_feedback(value: Any) -> None:
    rows = _require_non_empty_list(value, "$.verifier_feedback")
    for index, row in enumerate(rows):
        path = f"$.verifier_feedback[{index}]"
        item = _require_mapping(row, path)
        _require_fields(item, {"verifier_id", "status", "energy", "message", "violations"}, path)
        _reject_extra_fields(
            item, {"verifier_id", "status", "energy", "message", "violations"}, path
        )
        _require_non_empty_string(item["verifier_id"], f"{path}.verifier_id")
        _require_enum(item["status"], _VERIFIER_STATUSES, f"{path}.status")
        _require_nonnegative_number(item["energy"], f"{path}.energy")
        _require_string(item["message"], f"{path}.message")
        for violation_index, violation in enumerate(
            _require_list(item["violations"], f"{path}.violations")
        ):
            _require_string(violation, f"{path}.violations[{violation_index}]")


def _validate_confidence(value: Any) -> None:
    item = _require_mapping(value, "$.confidence")
    _require_fields(
        item, {"confidence", "abstain", "abstention_reason", "calibration_ref"}, "$.confidence"
    )
    _reject_extra_fields(
        item,
        {"confidence", "abstain", "abstention_reason", "calibration_ref"},
        "$.confidence",
    )
    confidence = _require_number(item["confidence"], "$.confidence.confidence")
    if not 0.0 <= confidence <= 1.0:
        msg = "$.confidence.confidence must be between 0 and 1"
        raise ValueError(msg)
    _require_bool(item["abstain"], "$.confidence.abstain")
    _require_string(item["abstention_reason"], "$.confidence.abstention_reason")
    _require_non_empty_string(item["calibration_ref"], "$.confidence.calibration_ref")


def _validate_exact_label(value: Any) -> None:
    item = _require_mapping(value, "$.exact_label_reference")
    _require_fields(
        item,
        {"label_id", "label", "authority", "source_artifact", "checksum"},
        "$.exact_label_reference",
    )
    _reject_extra_fields(
        item,
        {"label_id", "label", "authority", "source_artifact", "checksum"},
        "$.exact_label_reference",
    )
    for field in ("label_id", "label", "source_artifact"):
        _require_non_empty_string(item[field], f"$.exact_label_reference.{field}")
    _require_enum(item["authority"], _LABEL_AUTHORITIES, "$.exact_label_reference.authority")
    checksum = _require_non_empty_string(item["checksum"], "$.exact_label_reference.checksum")
    if not _HEX64_RE.fullmatch(checksum):
        msg = "$.exact_label_reference.checksum must be a sha256 hex digest"
        raise ValueError(msg)


def _validate_source_artifacts(value: Any) -> None:
    rows = _require_non_empty_list(value, "$.source_artifacts")
    for index, row in enumerate(rows):
        _require_non_empty_string(row, f"$.source_artifacts[{index}]")


def _require_fields(value: JsonDict, required: set[str] | frozenset[str], path: str) -> None:
    missing = sorted(required - set(value))
    if missing:
        msg = f"{path} missing required field(s): {missing}"
        raise ValueError(msg)


def _reject_extra_fields(value: JsonDict, allowed: set[str], path: str) -> None:
    extra = sorted(set(value) - allowed)
    if extra:
        msg = f"{path} contains additional field(s): {extra}"
        raise ValueError(msg)


def _require_mapping(value: Any, path: str) -> JsonDict:
    if not isinstance(value, dict):
        msg = f"{path} must be an object"
        raise ValueError(msg)
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        msg = f"{path} must be an array"
        raise ValueError(msg)
    return value


def _require_non_empty_list(value: Any, path: str) -> list[Any]:
    rows = _require_list(value, path)
    if not rows:
        msg = f"{path} must not be empty"
        raise ValueError(msg)
    return rows


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        msg = f"{path} must be a string"
        raise ValueError(msg)
    return value


def _require_non_empty_string(value: Any, path: str) -> str:
    text = _require_string(value, path)
    if text == "":
        msg = f"{path} must not be empty"
        raise ValueError(msg)
    return text


def _require_bool(value: Any, path: str) -> None:
    if not isinstance(value, bool):
        msg = f"{path} must be a boolean"
        raise ValueError(msg)


def _require_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{path} must be a number"
        raise ValueError(msg)
    return float(value)


def _require_nonnegative_number(value: Any, path: str) -> float:
    number = _require_number(value, path)
    if number < 0.0:
        msg = f"{path} must be non-negative"
        raise ValueError(msg)
    return number


def _require_numbers(value: Any, path: str) -> None:
    for index, item in enumerate(_require_list(value, path)):
        _require_number(item, f"{path}[{index}]")


def _require_enum(value: Any, allowed: set[str], path: str) -> None:
    text = _require_non_empty_string(value, path)
    if text not in allowed:
        msg = f"{path} must be one of {sorted(allowed)}"
        raise ValueError(msg)


def _stable_float(value: float) -> float:
    return round(float(value), 10)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
