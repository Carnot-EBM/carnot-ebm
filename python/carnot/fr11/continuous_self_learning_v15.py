"""FR-11 continuous self-learning v15 for EBT stabilizer efficacy counters.

Spec: REQ-LEARN-3740, SCENARIO-LEARN-3740.

The v15 forward difference is Tier-1 online learning applied to EBT training
bring-up diagnostics.  It reads checked-in per-chunk training artifacts,
updates CPU-only counters for stabilizer/no-divergence observations, persists
the raw counters, and emits a preliminary recipe for the next bounded run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

OUTPUT_REL_PATH = Path("results/experiment_3740_fr11_self_learning_v15_stabilizer_tracker.json")
TRACKER_STATE_REL_PATH = Path("results/experiment_3740_fr11_v15_stabilizer_tracker_state.json")
DEFAULT_RANDOM_SEED = 3740
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: reads training diagnostics, no live model)."
)
SUCCESS_VERDICT_PREFIX = (
    "complete: "
    "fr11_v15_tier1_stabilizer_efficacy_tracker_recipe_recommended_state_persisted_"
    "preliminary_over_"
)
EMPTY_VERDICT = "complete: no training diagnostics to learn from -- tracker initialized empty"
TIER1_COUNTER_UPDATE = "cpu_counter_update_only_lt_1us_no_model_retrain"
EXPERIMENT_ID_LIMIT = 3735

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "stabilizer_efficacy_table",
    "recommended_recipe",
    "n_chunks_observed",
    "tier1_counter_update",
    "tracker_state_persisted",
    "is_preliminary_heuristic",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; the self-learning outcome.",
    "inference_substrate": "Reads training diagnostics, no live model.",
    "stabilizer_efficacy_table": (
        "Per-stabilizer no-divergence correlation -- the learned knowledge; "
        "the core deliverable."
    ),
    "recommended_recipe": (
        "The upweighted stabilizer set for the next bounded run -- the actionable "
        "self-learning output that feeds back into the loop."
    ),
    "n_chunks_observed": (
        "Sample-size hygiene -- how many training chunks the tracker learned from "
        "(small-sample heuristic, labeled preliminary)."
    ),
    "tier1_counter_update": (
        "Confirms the mechanic is a CPU counter update (<1us, the Tier-1 hardware "
        "path), not a model retrain -- aligns with the hardware-acceleration principle."
    ),
    "tracker_state_persisted": (
        "True iff the tracker state was saved so a future EBT-training milestone "
        "resumes it (continuous, not one-shot)."
    ),
    "is_preliminary_heuristic": (
        "Honest label: over a handful of chunks this is a heuristic, not a "
        "statistical claim -- pre-empts SAMPLE_SIZE_BELOW_CLAIM."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}


@dataclass(frozen=True)
class TrainingChunk:
    """One upstream EBT training diagnostic chunk."""

    source_path: str
    experiment_id: int | None
    stabilizers: tuple[str, ...]
    nan_or_divergence_events: bool
    ebt_converged: bool | None
    cumulative_steps_trained: int | None

    @property
    def no_divergence(self) -> bool:
        return not self.nan_or_divergence_events

    def to_json(self) -> JsonDict:
        return {
            "source_path": self.source_path,
            "experiment_id": self.experiment_id,
            "stabilizers_applied": list(self.stabilizers),
            "nan_or_divergence_events": self.nan_or_divergence_events,
            "ebt_converged": self.ebt_converged,
            "cumulative_steps_trained": self.cumulative_steps_trained,
        }


@dataclass
class StabilizerStats:
    """Raw enabled/disabled no-divergence counters for one stabilizer."""

    enabled_total: int = 0
    enabled_no_divergence: int = 0
    disabled_total: int = 0
    disabled_no_divergence: int = 0

    def record(self, *, enabled: bool, no_divergence: bool) -> None:
        if enabled:
            self.enabled_total += 1
            if no_divergence:
                self.enabled_no_divergence += 1
            return
        self.disabled_total += 1
        if no_divergence:
            self.disabled_no_divergence += 1

    @property
    def enabled_rate(self) -> float:
        return _rate(self.enabled_no_divergence, self.enabled_total)

    @property
    def disabled_rate(self) -> float:
        return _rate(self.disabled_no_divergence, self.disabled_total)

    @property
    def delta(self) -> float:
        return self.enabled_rate - self.disabled_rate

    def to_json(self) -> JsonDict:
        return {
            "enabled_total": self.enabled_total,
            "enabled_no_divergence": self.enabled_no_divergence,
            "disabled_total": self.disabled_total,
            "disabled_no_divergence": self.disabled_no_divergence,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> StabilizerStats:
        return cls(
            enabled_total=int(payload.get("enabled_total", 0)),
            enabled_no_divergence=int(payload.get("enabled_no_divergence", 0)),
            disabled_total=int(payload.get("disabled_total", 0)),
            disabled_no_divergence=int(payload.get("disabled_no_divergence", 0)),
        )


class StabilizerEfficacyTracker:
    """Tier-1 raw counter tracker for EBT stabilizer/no-divergence observations."""

    def __init__(self) -> None:
        self._stats: dict[str, StabilizerStats] = {}

    def record_chunk(
        self,
        enabled_stabilizers: Sequence[str],
        *,
        no_divergence: bool,
        observed_stabilizers: Sequence[str],
    ) -> None:
        enabled = set(enabled_stabilizers)
        for stabilizer in observed_stabilizers:
            stats = self._stats.setdefault(stabilizer, StabilizerStats())
            stats.record(enabled=stabilizer in enabled, no_divergence=no_divergence)

    def stats(self) -> dict[str, StabilizerStats]:
        return dict(self._stats)

    def to_json(self) -> JsonDict:
        return {
            "schema": "carnot.fr11_v15_stabilizer_efficacy_tracker_state",
            "version": 1,
            "stats": {name: self._stats[name].to_json() for name in sorted(self._stats)},
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> StabilizerEfficacyTracker:
        if payload.get("version") != 1:
            raise ValueError("unsupported stabilizer tracker state version")
        tracker = cls()
        raw_stats = payload.get("stats", {})
        if not isinstance(raw_stats, Mapping):
            raise ValueError("stabilizer tracker stats must be a mapping")
        for name, raw in raw_stats.items():
            if not isinstance(raw, Mapping):
                raise ValueError("stabilizer tracker stat entry must be a mapping")
            tracker._stats[str(name)] = StabilizerStats.from_json(raw)
        return tracker


def parse_stabilizers(value: Any) -> tuple[str, ...]:
    """Parse a checked-in stabilizer field into a stable de-duplicated tuple."""

    if value is None:
        return ()
    if isinstance(value, str):
        candidates = value.split(",")
    elif isinstance(value, Sequence) and not isinstance(value, bytes):
        candidates = [str(item) for item in value]
    else:
        candidates = []
    cleaned = {
        item.strip()
        for item in candidates
        if item.strip() and item.strip().lower() != "none"
    }
    return tuple(sorted(cleaned))


def load_training_chunks(results_dir: Path | str) -> list[TrainingChunk]:
    """Load EBT training diagnostics through Exp 3735 from checked-in JSON."""

    root = Path(results_dir)
    if not root.is_dir():
        return []
    chunks: list[TrainingChunk] = []
    for path in sorted(root.glob("experiment_*.json")):
        experiment_id = _experiment_id(path.name)
        if experiment_id is None or experiment_id > EXPERIMENT_ID_LIMIT:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not _looks_like_ebt_training_diagnostic(path.name, payload):
            continue
        chunks.append(_chunk_from_payload(path, payload, experiment_id))
    return chunks


def build_tracker_from_chunks(
    chunks: Sequence[TrainingChunk],
) -> tuple[StabilizerEfficacyTracker, tuple[str, ...]]:
    """Build raw enabled/disabled counters from upstream chunks."""

    observed = tuple(sorted({item for chunk in chunks for item in chunk.stabilizers}))
    tracker = StabilizerEfficacyTracker()
    for chunk in chunks:
        tracker.record_chunk(
            chunk.stabilizers,
            no_divergence=chunk.no_divergence,
            observed_stabilizers=observed,
        )
    return tracker, observed


def efficacy_table(
    tracker: StabilizerEfficacyTracker,
    observed_stabilizers: Sequence[str],
) -> list[JsonDict]:
    """Return derived no-divergence rates from raw tracker counters."""

    rows: list[JsonDict] = []
    stats_by_name = tracker.stats()
    for stabilizer in observed_stabilizers:
        stats = stats_by_name.get(stabilizer, StabilizerStats())
        rows.append(
            {
                "stabilizer": stabilizer,
                "enabled_total": stats.enabled_total,
                "enabled_no_divergence": stats.enabled_no_divergence,
                "enabled_no_divergence_rate": _round(stats.enabled_rate),
                "disabled_total": stats.disabled_total,
                "disabled_no_divergence": stats.disabled_no_divergence,
                "disabled_no_divergence_rate": _round(stats.disabled_rate),
                "no_divergence_rate_delta_enabled_minus_disabled": _round(stats.delta),
                "sample_note": "preliminary_counter_heuristic_not_statistical_claim",
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -float(row["no_divergence_rate_delta_enabled_minus_disabled"]),
            -float(row["enabled_no_divergence_rate"]),
            str(row["stabilizer"]),
        ),
    )


def recommended_recipe(table: Sequence[Mapping[str, Any]], *, n_chunks_observed: int) -> JsonDict:
    """Select every stabilizer tied for the best preliminary counter delta."""

    if not table:
        return {
            "stabilizers": [],
            "selection_basis": "no_observed_training_diagnostics",
            "max_delta": None,
            "is_preliminary_heuristic": True,
            "n_chunks_observed": int(n_chunks_observed),
        }
    max_delta = max(
        float(row["no_divergence_rate_delta_enabled_minus_disabled"]) for row in table
    )
    winners = [
        str(row["stabilizer"])
        for row in table
        if float(row["no_divergence_rate_delta_enabled_minus_disabled"]) == max_delta
    ]
    return {
        "stabilizers": sorted(winners),
        "selection_basis": (
            "max_enabled_minus_disabled_no_divergence_delta_with_ties_retained"
        ),
        "max_delta": _round(max_delta),
        "is_preliminary_heuristic": True,
        "n_chunks_observed": int(n_chunks_observed),
    }


def build_artifact(
    repo_root: Path | str,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    state_path: Path | str = TRACKER_STATE_REL_PATH,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> JsonDict:
    """Build the Exp 3740 artifact and persist tracker state."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    chunks = load_training_chunks(root / "results")
    tracker, observed = build_tracker_from_chunks(chunks)
    table = efficacy_table(tracker, observed)
    recipe = recommended_recipe(table, n_chunks_observed=len(chunks))
    state_output = _resolve_under_root(root, state_path)
    state_sha = persist_tracker_state(
        tracker,
        state_output,
        n_chunks_observed=len(chunks),
        observed_stabilizers=observed,
        random_seed=random_seed,
    )
    persisted = state_output.is_file()
    verdict = (
        f"{SUCCESS_VERDICT_PREFIX}{len(chunks)}_chunks" if chunks else EMPTY_VERDICT
    )
    checksum = reproducibility_checksum(
        chunks=chunks,
        table=table,
        recipe=recipe,
        tracker_state_sha256=state_sha,
        random_seed=random_seed,
    )
    artifact: JsonDict = {
        "artifact": "experiment_3740_fr11_self_learning_v15_stabilizer_tracker",
        "schema": "carnot.fr11_continuous_self_learning_v15",
        "continuous_self_learning_task": True,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "stabilizer_efficacy_table": table,
        "recommended_recipe": recipe,
        "n_chunks_observed": int(len(chunks)),
        "tier1_counter_update": TIER1_COUNTER_UPDATE,
        "tracker_state_persisted": bool(persisted),
        "is_preliminary_heuristic": True,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "duration_s": _round(_duration(start, now_s)),
        "tracker_state_path": _relative_path(state_output, root),
        "tracker_state_sha256": state_sha,
        "source_training_chunks": [chunk.to_json() for chunk in chunks],
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "passed": bool(persisted),
            "condition": "tracker_state_persisted == true",
            "principle": "Continuous self-learning requires resumable counter state.",
        },
    }
    validate_artifact(artifact)
    return artifact


def persist_tracker_state(
    tracker: StabilizerEfficacyTracker,
    path: Path | str,
    *,
    n_chunks_observed: int,
    observed_stabilizers: Sequence[str],
    random_seed: int,
) -> str:
    """Persist raw counters and return a checksum over the stored payload."""

    output = Path(path)
    payload = {
        **tracker.to_json(),
        "n_chunks_observed": int(n_chunks_observed),
        "observed_stabilizers": list(observed_stabilizers),
        "tier1_counter_update": TIER1_COUNTER_UPDATE,
        "random_seed": int(random_seed),
    }
    checksum = _json_sha256(payload)
    stored = {**payload, "sha256": checksum}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return checksum


def write_artifact(
    repo_root: Path | str,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    state_path: Path | str = TRACKER_STATE_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and write the Exp 3740 JSON artifact."""

    root = Path(repo_root)
    artifact = build_artifact(
        root,
        started_s=started_s,
        now_s=now_s,
        state_path=state_path,
    )
    output = _resolve_under_root(root, output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def reproducibility_checksum(
    *,
    chunks: Sequence[TrainingChunk],
    table: Sequence[Mapping[str, Any]],
    recipe: Mapping[str, Any],
    tracker_state_sha256: str,
    random_seed: int,
) -> str:
    """Hash the deterministic inputs and derived counter outputs."""

    payload = {
        "chunks": [chunk.to_json() for chunk in chunks],
        "stabilizer_efficacy_table": list(table),
        "recommended_recipe": dict(recipe),
        "tracker_state_sha256": tracker_state_sha256,
        "random_seed": int(random_seed),
    }
    return _json_sha256(payload)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3740 artifact schema before writing."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must declare aggregation-only diagnostics")
    serialized = json.dumps(artifact, sort_keys=True)
    for marker in ("GGUF", "CUDA", "cuda", "live_llm_inference", "torch.cuda"):
        if marker in serialized:
            raise ValueError("forbidden inference marker present")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be present")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    if artifact.get("tier1_counter_update") != TIER1_COUNTER_UPDATE:
        raise ValueError("tier1_counter_update must declare CPU counter update only")
    if not isinstance(artifact.get("tracker_state_persisted"), bool):
        raise ValueError("tracker_state_persisted must be boolean")
    if artifact.get("is_preliminary_heuristic") is not True:
        raise ValueError("is_preliminary_heuristic must be true")
    if not isinstance(artifact.get("duration_s"), int | float):
        raise ValueError("duration_s must be numeric")
    n_chunks = int(artifact.get("n_chunks_observed", -1))
    table = artifact.get("stabilizer_efficacy_table")
    recipe = artifact.get("recommended_recipe")
    if not isinstance(table, list):
        raise ValueError("stabilizer_efficacy_table must be a list")
    if not isinstance(recipe, Mapping) or not isinstance(recipe.get("stabilizers"), list):
        raise ValueError("recommended_recipe.stabilizers must be a list")
    verdict = artifact.get("honest_verdict")
    if n_chunks == 0:
        if verdict != EMPTY_VERDICT:
            raise ValueError("empty tracker verdict must note missing training diagnostics")
        if table or recipe.get("stabilizers"):
            raise ValueError("empty tracker artifact must not recommend stabilizers")
    elif not isinstance(verdict, str) or not verdict.startswith(SUCCESS_VERDICT_PREFIX):
        raise ValueError("non-empty tracker verdict must use the v15 success prefix")


def _chunk_from_payload(path: Path, payload: Mapping[str, Any], experiment_id: int) -> TrainingChunk:
    return TrainingChunk(
        source_path=path.name,
        experiment_id=experiment_id,
        stabilizers=parse_stabilizers(payload.get("stabilizers_applied")),
        nan_or_divergence_events=_parse_bool(payload.get("nan_or_divergence_events")),
        ebt_converged=_parse_optional_bool(payload.get("ebt_converged")),
        cumulative_steps_trained=_parse_optional_int(payload.get("cumulative_steps_trained")),
    )


def _looks_like_ebt_training_diagnostic(name: str, payload: Mapping[str, Any]) -> bool:
    if "stabilizers_applied" not in payload or "nan_or_divergence_events" not in payload:
        return False
    lowered = name.lower()
    if "ebt" in lowered or ("bounded" in lowered and "train" in lowered):
        return True
    specs = payload.get("model_specs")
    return isinstance(specs, Mapping) and "ebt_model" in specs


def _experiment_id(name: str) -> int | None:
    match = re.match(r"experiment_(\d+)_", name)
    return int(match.group(1)) if match else None


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"expected boolean diagnostic value, got {value!r}")


def _parse_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return _parse_bool(value)


def _parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.time() if now_s is None else float(now_s)
    return max(0.0001, end - float(started_s))


def _round(value: float) -> float:
    return round(float(value), 9)


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_under_root(root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
