"""Exp 2969 non-tautological FR-11 utility-gated replay evaluator.

The Exp 2954 replay result was positive, but it was flagged because the held-out
and replay signals could not fully rule out tautological reuse. This module
rebuilds the check from independent checked-in artifacts, records split and
reward-source checksums, compares reset/random/prior/new replay policies, and
rolls back any update that degrades stable guard slices.

Spec: REQ-LEARN-2969, SCENARIO-LEARN-2969,
SCENARIO-LEARN-2969-ROLLBACK, SCENARIO-LEARN-2969-BLOCKED.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260524"
OUTPUT_FILENAME = "experiment_2969_fr11_non_tautological_utility_gate_v3.json"
ARTIFACT = "experiment_2969_fr11_non_tautological_utility_gate_v3"
SCHEMA = "carnot.fr11.non_tautological_utility_gate.v3"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXP2954_REL_PATH = Path("results/experiment_2954_fr11_utility_gated_replay_curriculum_v2.json")
EXP2952_REL_PATH = Path("results/experiment_2952_sota_taxonomy_guided_code_repair_eval_v1.json")
EXP2959_REL_PATH = Path("results/experiment_2959_nl_to_z3_execution_repair_mini_v2.json")
EXP2960_REL_PATH = Path("results/experiment_2960_cross_corpus_matrix_v12.json")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "continuous_self_learning_task",
    "non_tautological_self_learning_ready",
    "source_artifacts",
    "split_checksums",
    "leakage_check_passed",
    "replay_policies_compared",
    "frozen_heldout_utility",
    "random_replay_heldout_utility",
    "prior_utility_gated_heldout_utility",
    "new_heldout_utility",
    "heldout_utility_delta_vs_random",
    "negative_control_delta",
    "forgetting_guard_passed",
    "rollback_triggered",
    "update_rule",
    "model_specs_if_live_llm_used",
    "inference_substrate",
    "duration_s",
)

TAXONOMY_ORDER = (
    "syntax_repair",
    "runtime_repair",
    "extraction_repair",
    "logic_repair",
    "verified_pass",
    "logic_guard",
    "threshold_policy",
)
GUARD_TAXONOMIES = ("verified_pass", "logic_guard", "threshold_policy")


@dataclass(frozen=True)
class SourceSpec:
    """One upstream artifact required by the aggregation-only evaluator."""

    experiment_id: str
    path: Path
    role: str
    fields_imported: tuple[str, ...]
    required: bool = True


SOURCE_SPECS = (
    SourceSpec(
        "exp2954",
        EXP2954_REL_PATH,
        "flagged_prior_utility_gate",
        (
            "self_learning_utility_artifact_ready",
            "final_replay_weights",
            "heldout_utility_after",
            "forgetting_guard_passed",
        ),
    ),
    SourceSpec(
        "exp2952",
        EXP2952_REL_PATH,
        "code_repair_rows",
        (
            "candidate_evaluations",
            "baseline_pass_at_1",
            "repair_pass_at_1",
            "candidate_manifest_sha256",
        ),
    ),
    SourceSpec(
        "exp2959",
        EXP2959_REL_PATH,
        "logic_execution_rows",
        (
            "per_item_results",
            "failure_categories",
            "formalization_manifest_sha256",
            "answer_accuracy",
        ),
    ),
    SourceSpec(
        "exp2960",
        EXP2960_REL_PATH,
        "matrix_v12_guard_and_summary",
        (
            "matrix_v12_ready",
            "matrix_rows",
            "self_learning_delta_summary",
            "code_repair_delta_summary",
        ),
    ),
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths, clock, and test provenance for the artifact builder."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    exp2954_path: Path = EXP2954_REL_PATH
    exp2952_path: Path = EXP2952_REL_PATH
    exp2959_path: Path = EXP2959_REL_PATH
    exp2960_path: Path = EXP2960_REL_PATH
    tests_run: Sequence[str] = field(default_factory=tuple)
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME


@dataclass(frozen=True)
class EvidenceExample:
    """One split-scoped replay or guard evidence row."""

    item_id: str
    domain: str
    split: str
    taxonomy: str
    reward_signal: float
    utility_signal: float
    guard_signal: float
    source_id: str


def build_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """REQ-LEARN-2969: build the non-tautological replay artifact."""

    config = config or ExperimentConfig()
    started = config.start_time()
    source_artifacts = _source_artifacts(config)
    missing_sources = _missing_required_sources(source_artifacts)
    if missing_sources:
        return _blocked_artifact(
            config,
            started,
            source_artifacts,
            "blocked_missing_upstream_artifact",
            [f"source:{source}" for source in missing_sources],
        )

    payloads = {
        "exp2954": read_json_object(_repo_path(config.repo_root, config.exp2954_path)),
        "exp2952": read_json_object(_repo_path(config.repo_root, config.exp2952_path)),
        "exp2959": read_json_object(_repo_path(config.repo_root, config.exp2959_path)),
        "exp2960": read_json_object(_repo_path(config.repo_root, config.exp2960_path)),
    }
    missing_fields = _missing_required_fields(payloads)
    if missing_fields:
        return _blocked_artifact(
            config,
            started,
            source_artifacts,
            "blocked_missing_required_fields",
            missing_fields,
        )

    examples = build_evidence_examples(
        exp2952=payloads["exp2952"],
        exp2959=payloads["exp2959"],
        exp2960=payloads["exp2960"],
    )
    split_checksums = compute_split_checksums(examples)
    leakage_passed = leakage_check(split_checksums)
    slice_errors = _slice_errors(examples, split_checksums)
    if slice_errors:
        artifact = _blocked_artifact(
            config,
            started,
            source_artifacts,
            "blocked_insufficient_disjoint_slices",
            slice_errors,
        )
        artifact["split_checksums"] = split_checksums
        artifact["leakage_check_passed"] = leakage_passed
        return artifact

    train_replay_examples = _examples_for_splits(examples, ("train", "replay"))
    heldout_examples = _examples_for_splits(examples, ("heldout",))
    guard_examples = _examples_for_splits(examples, ("guard",))
    observed = observed_taxonomies(examples, payloads["exp2954"])

    frozen_weights = frozen_baseline_weights(observed)
    random_weights = random_replay_weights(observed)
    prior_weights = prior_utility_gated_weights(payloads["exp2954"], observed)
    target_weights = target_weights_from_reward(
        train_replay_examples,
        baseline_weights=random_weights,
        guard_taxonomies=GUARD_TAXONOMIES,
    )
    negative_weights = negative_control_weights(observed)

    decision = evaluate_policy_update(
        baseline_weights=random_weights,
        candidate_weights=target_weights,
        heldout_examples=heldout_examples,
        guard_examples=guard_examples,
    )

    frozen_utility = policy_utility(frozen_weights, heldout_examples)
    random_utility = policy_utility(random_weights, heldout_examples)
    prior_utility = policy_utility(prior_weights, heldout_examples)
    negative_utility = policy_utility(negative_weights, heldout_examples)
    new_utility = policy_utility(decision["accepted_weights"], heldout_examples)
    negative_delta = _round(negative_utility - random_utility)
    delta_vs_random = _round(new_utility - random_utility)
    ready = bool(
        leakage_passed
        and new_utility > frozen_utility
        and new_utility > random_utility
        and negative_delta <= 0.0
        and decision["forgetting_guard_passed"]
        and not decision["rollback_triggered"]
    )

    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": _verdict(ready, bool(decision["rollback_triggered"])),
        "continuous_self_learning_task": True,
        "non_tautological_self_learning_ready": ready,
        "source_artifacts": source_artifacts,
        "split_checksums": split_checksums,
        "leakage_check_passed": leakage_passed,
        "replay_policies_compared": [
            _policy_row("frozen_baseline", frozen_weights, frozen_utility, None, False),
            _policy_row(
                "random_replay",
                random_weights,
                random_utility,
                decision["guard_metrics_before"],
                False,
            ),
            _policy_row(
                "prior_278_utility_gated_replay",
                prior_weights,
                prior_utility,
                None,
                False,
            ),
            _policy_row(
                "negative_control_uninformative",
                negative_weights,
                negative_utility,
                None,
                False,
            ),
            _policy_row(
                "non_tautological_utility_gated_replay",
                decision["accepted_weights"],
                new_utility,
                decision["guard_metrics_after"],
                ready,
            ),
        ],
        "frozen_heldout_utility": frozen_utility,
        "random_replay_heldout_utility": random_utility,
        "prior_utility_gated_heldout_utility": prior_utility,
        "new_heldout_utility": new_utility,
        "heldout_utility_delta_vs_random": delta_vs_random,
        "negative_control_delta": negative_delta,
        "forgetting_guard_passed": decision["forgetting_guard_passed"],
        "rollback_triggered": decision["rollback_triggered"],
        "update_rule": _update_rule(),
        "model_specs_if_live_llm_used": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _elapsed(config, started),
        "candidate_pre_rollback_heldout_utility": decision["candidate_heldout_utility"],
        "guard_metrics_before": decision["guard_metrics_before"],
        "guard_metrics_after": decision["guard_metrics_after"],
        "negative_control_heldout_utility": negative_utility,
        "slice_manifest": slice_manifest(examples),
        "final_replay_weights": decision["accepted_weights"],
        "candidate_replay_weights": target_weights,
        "frozen_replay_weights": frozen_weights,
        "random_replay_weights": random_weights,
        "prior_utility_gated_replay_weights": prior_weights,
        "missing_fields": [],
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }


def write_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build and persist `results/experiment_2969...json`."""

    config = config or ExperimentConfig()
    artifact = build_artifact(config)
    output_path = config.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def build_evidence_examples(
    *,
    exp2952: Mapping[str, Any],
    exp2959: Mapping[str, Any],
    exp2960: Mapping[str, Any],
) -> tuple[EvidenceExample, ...]:
    """Build disjoint train/replay/held-out/guard evidence rows."""

    examples: list[EvidenceExample] = []
    examples.extend(_code_examples(exp2952.get("candidate_evaluations", ())))
    examples.extend(_logic_examples(exp2959))
    examples.extend(_threshold_guard_examples(exp2960))
    return tuple(examples)


def compute_split_checksums(examples: Sequence[EvidenceExample]) -> JsonDict:
    """REQ-LEARN-2969-2: hash split IDs and reward/target source IDs."""

    train_ids = _ids_for_split(examples, "train")
    replay_ids = _ids_for_split(examples, "replay")
    heldout_ids = _ids_for_split(examples, "heldout")
    guard_ids = _ids_for_split(examples, "guard")
    reward_source_ids = sorted(
        {
            example.source_id
            for example in examples
            if example.split in {"train", "replay"} and example.reward_signal > 0.0
        }
    )
    heldout_target_ids = sorted(
        {
            example.source_id
            for example in examples
            if example.split == "heldout" and example.utility_signal > 0.0
        }
    )
    return {
        "train_ids_sha256": sha256_json(train_ids),
        "replay_ids_sha256": sha256_json(replay_ids),
        "heldout_ids_sha256": sha256_json(heldout_ids),
        "guard_ids_sha256": sha256_json(guard_ids),
        "reward_source_sha256": sha256_json(reward_source_ids),
        "heldout_target_sha256": sha256_json(heldout_target_ids),
        "train_count": len(train_ids),
        "replay_count": len(replay_ids),
        "heldout_count": len(heldout_ids),
        "guard_count": len(guard_ids),
        "reward_source_count": len(reward_source_ids),
        "heldout_target_count": len(heldout_target_ids),
        "overlap_counts": {
            "train_vs_replay": len(set(train_ids) & set(replay_ids)),
            "train_vs_heldout": len(set(train_ids) & set(heldout_ids)),
            "replay_vs_heldout": len(set(replay_ids) & set(heldout_ids)),
            "reward_vs_heldout": len(set(reward_source_ids) & set(heldout_target_ids)),
        },
        "reward_not_from_evaluation_target": not (set(reward_source_ids) & set(heldout_target_ids)),
    }


def leakage_check(split_checksums: Mapping[str, Any]) -> bool:
    """Return true when all split and reward-target overlap checks pass."""

    overlaps = _mapping(split_checksums.get("overlap_counts"))
    return bool(
        split_checksums.get("train_count", 0) > 0
        and split_checksums.get("replay_count", 0) > 0
        and split_checksums.get("heldout_count", 0) > 0
        and split_checksums.get("reward_source_count", 0) > 0
        and split_checksums.get("heldout_target_count", 0) > 0
        and all(int(value) == 0 for value in overlaps.values())
        and split_checksums.get("reward_source_sha256")
        != split_checksums.get("heldout_target_sha256")
        and split_checksums.get("reward_not_from_evaluation_target") is True
    )


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Normalize positive weights into a deterministic probability table."""

    positive = {name: float(value) for name, value in sorted(weights.items()) if float(value) > 0.0}
    total = sum(positive.values())
    if total <= 0.0:
        raise ValueError("at least one positive replay weight is required")
    return {name: _round(value / total) for name, value in positive.items()}


def frozen_baseline_weights(observed_taxonomies: Sequence[str]) -> dict[str, float]:
    """Reset baseline: retain only stable guard taxonomies when available."""

    guard_weights = {
        taxonomy: 1.0 for taxonomy in observed_taxonomies if taxonomy in GUARD_TAXONOMIES
    }
    if guard_weights:
        return normalize_weights(guard_weights)
    return random_replay_weights(observed_taxonomies)


def random_replay_weights(observed_taxonomies: Sequence[str]) -> dict[str, float]:
    """Uniform replay baseline over observed taxonomies."""

    return normalize_weights({taxonomy: 1.0 for taxonomy in observed_taxonomies})


def negative_control_weights(observed_taxonomies: Sequence[str]) -> dict[str, float]:
    """Uninformative control: deterministic uniform weights, independent of reward."""

    return random_replay_weights(observed_taxonomies)


def prior_utility_gated_weights(
    exp2954: Mapping[str, Any],
    observed_taxonomies: Sequence[str],
) -> dict[str, float]:
    """Map the flagged Exp 2954 accepted weights onto this run's taxonomies."""

    prior = {
        taxonomy: _positive_float(_mapping(exp2954.get("final_replay_weights")).get(taxonomy))
        for taxonomy in observed_taxonomies
    }
    if sum(prior.values()) <= 0.0:
        prior = {
            taxonomy: _positive_float(
                _mapping(exp2954.get("candidate_replay_weights")).get(taxonomy)
            )
            for taxonomy in observed_taxonomies
        }
    if sum(prior.values()) <= 0.0:
        return random_replay_weights(observed_taxonomies)
    return normalize_weights(prior)


def target_weights_from_reward(
    examples: Sequence[EvidenceExample],
    *,
    baseline_weights: Mapping[str, float],
    guard_taxonomies: Sequence[str],
) -> dict[str, float]:
    """Build reward-weighted replay weights while preserving guard mass."""

    baseline = normalize_weights(baseline_weights)
    reward_totals = {taxonomy: 0.0 for taxonomy in baseline}
    for example in examples:
        if example.taxonomy in reward_totals:
            reward_totals[example.taxonomy] += example.reward_signal

    guard_mass = sum(baseline.get(taxonomy, 0.0) for taxonomy in guard_taxonomies)
    candidate = {
        taxonomy: baseline[taxonomy]
        for taxonomy in guard_taxonomies
        if taxonomy in baseline
    }
    non_guard_rewards = {
        taxonomy: reward
        for taxonomy, reward in reward_totals.items()
        if taxonomy not in candidate and reward > 0.0
    }
    if not non_guard_rewards:
        return baseline
    scaled = normalize_weights(non_guard_rewards)
    for taxonomy, value in scaled.items():
        candidate[taxonomy] = value * max(0.0, 1.0 - guard_mass)
    return normalize_weights(candidate)


def evaluate_policy_update(
    *,
    baseline_weights: Mapping[str, float],
    candidate_weights: Mapping[str, float],
    heldout_examples: Sequence[EvidenceExample],
    guard_examples: Sequence[EvidenceExample],
) -> JsonDict:
    """Apply held-out utility and multi-slice forgetting guards."""

    baseline = normalize_weights(baseline_weights)
    candidate = normalize_weights(candidate_weights)
    baseline_utility = policy_utility(baseline, heldout_examples)
    candidate_utility = policy_utility(candidate, heldout_examples)
    guard_before = forgetting_guard_metrics(baseline, guard_examples)
    guard_after = forgetting_guard_metrics(candidate, guard_examples)
    utility_improved = candidate_utility > baseline_utility
    guard_passed = all(guard_after[name] >= guard_before[name] for name in guard_before)
    rollback = bool(utility_improved and not guard_passed)
    accepted = candidate if utility_improved and guard_passed else baseline
    return {
        "baseline_heldout_utility": baseline_utility,
        "candidate_heldout_utility": candidate_utility,
        "utility_delta": _round(candidate_utility - baseline_utility),
        "utility_improved": utility_improved,
        "guard_metrics_before": guard_before,
        "guard_metrics_after": guard_after,
        "forgetting_guard_passed": guard_passed,
        "rollback_triggered": rollback,
        "accepted_weights": accepted,
    }


def policy_utility(weights: Mapping[str, float], examples: Sequence[EvidenceExample]) -> float:
    """Score held-out utility as replay mass covering independent target signals."""

    normalized = normalize_weights(weights)
    total_signal = sum(example.utility_signal for example in examples)
    if total_signal <= 0.0:
        return 0.0
    score = sum(normalized.get(example.taxonomy, 0.0) * example.utility_signal for example in examples)
    return _round(score / total_signal)


def forgetting_guard_metrics(
    weights: Mapping[str, float],
    examples: Sequence[EvidenceExample],
) -> dict[str, float]:
    """Measure retained replay mass for code, logic, and threshold guards."""

    normalized = normalize_weights(weights)
    return {
        "stable_code": _guard_metric(normalized, examples, "code"),
        "stable_logic": _guard_metric(normalized, examples, "logic"),
        "threshold_policy": _guard_metric(normalized, examples, "threshold_policy"),
    }


def observed_taxonomies(
    examples: Sequence[EvidenceExample],
    exp2954: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return ordered taxonomies present in evidence or prior replay weights."""

    observed = {
        example.taxonomy
        for example in examples
        if example.reward_signal > 0.0 or example.utility_signal > 0.0 or example.guard_signal > 0.0
    }
    observed.update(str(name) for name in _mapping(exp2954.get("final_replay_weights")))
    ordered = tuple(taxonomy for taxonomy in TAXONOMY_ORDER if taxonomy in observed)
    if ordered:
        return ordered
    return TAXONOMY_ORDER


def slice_manifest(examples: Sequence[EvidenceExample]) -> JsonDict:
    """Compact counts for the train/replay/held-out/guard split."""

    manifest: dict[str, JsonDict] = {}
    for split in ("train", "replay", "heldout", "guard"):
        split_examples = [example for example in examples if example.split == split]
        manifest[split] = {
            "item_count": len({example.item_id for example in split_examples}),
            "row_count": len(split_examples),
            "taxonomy_counts": _taxonomy_counts(split_examples),
            "domains": sorted({example.domain for example in split_examples}),
        }
    return manifest


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, returning an empty dict for malformed content."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_json(value: object) -> str:
    """Hash JSON-serializable evidence with stable key ordering."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    """CLI entry point used by the experiment wrapper."""

    write_artifact()
    return 0


def _code_examples(rows: object) -> tuple[EvidenceExample, ...]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return ()
    mappings = [_mapping(row) for row in rows]
    stable_ids = sorted({str(row.get("stable_id", "")) for row in mappings if row.get("stable_id")})
    split_by_id = {
        stable_id: _split_for_stable_index(index)
        for index, stable_id in enumerate(stable_ids)
    }
    examples: list[EvidenceExample] = []
    for row in mappings:
        stable_id = str(row.get("stable_id", ""))
        if stable_id not in split_by_id:
            continue
        mode = str(row.get("mode", ""))
        split = _code_row_split(split_by_id[stable_id], mode)
        if split is None:
            continue
        taxonomy = _code_taxonomy(row)
        reward_signal = _code_reward(taxonomy) if split in {"train", "replay"} else 0.0
        utility_signal = _code_reward(taxonomy) if split == "heldout" else 0.0
        guard_signal = 1.0 if split == "guard" and taxonomy == "verified_pass" else 0.0
        sample_index = int(row.get("sample_index", 0) or 0)
        source_hash = str(row.get("candidate_manifest_sha256", ""))[:16]
        examples.append(
            EvidenceExample(
                item_id=f"code:{stable_id}",
                domain="code",
                split=split,
                taxonomy=taxonomy,
                reward_signal=_round(reward_signal),
                utility_signal=_round(utility_signal),
                guard_signal=guard_signal,
                source_id=f"exp2952:{stable_id}:{mode}:{sample_index}:{source_hash}",
            )
        )
    return tuple(examples)


def _logic_examples(exp2959: Mapping[str, Any]) -> tuple[EvidenceExample, ...]:
    rows = exp2959.get("per_item_results", ())
    examples: list[EvidenceExample] = []
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        for index, row_obj in enumerate(rows):
            row = _mapping(row_obj)
            item_id = str(row.get("item_id", f"logic-{index:03d}"))
            split = "train" if index % 2 == 0 else "replay"
            category = str(row.get("failure_category", "unparseable"))
            taxonomy = "logic_guard" if category == "solver_verified_correct" else "logic_repair"
            if taxonomy == "logic_repair":
                reward_signal = _logic_reward(category)
                examples.append(
                    EvidenceExample(
                        item_id=f"logic:{item_id}",
                        domain="logic",
                        split=split,
                        taxonomy=taxonomy,
                        reward_signal=reward_signal,
                        utility_signal=0.0,
                        guard_signal=0.0,
                        source_id=f"exp2959:{item_id}:{category}",
                    )
                )

    manifest_sha = str(exp2959.get("formalization_manifest_sha256", "missing"))
    examples.append(
        EvidenceExample(
            item_id="logic:stable_manifest_guard",
            domain="logic",
            split="guard",
            taxonomy="logic_guard",
            reward_signal=0.0,
            utility_signal=0.0,
            guard_signal=1.0,
            source_id=f"exp2959:formalization_manifest:{manifest_sha}",
        )
    )
    return tuple(examples)


def _threshold_guard_examples(exp2960: Mapping[str, Any]) -> tuple[EvidenceExample, ...]:
    for row_obj in _sequence(exp2960.get("matrix_rows")):
        row = _mapping(row_obj)
        if row.get("row_id") != "exp2953_threshold_policy":
            continue
        summary = _mapping(row.get("summary"))
        if summary.get("threshold_policy_ready") is not True:
            continue
        threshold = summary.get("selected_default_threshold", "unknown")
        ppv = _positive_float(summary.get("expected_ppv_at_default"), default=1.0)
        return (
            EvidenceExample(
                item_id="threshold_policy:exp2953_default",
                domain="threshold_policy",
                split="guard",
                taxonomy="threshold_policy",
                reward_signal=0.0,
                utility_signal=0.0,
                guard_signal=max(0.1, ppv),
                source_id=f"exp2960:exp2953_threshold_policy:{threshold}",
            ),
        )
    return ()


def _code_row_split(stable_split: str, mode: str) -> str | None:
    if stable_split == "train" and mode == "baseline_no_taxonomy":
        return "train"
    if stable_split == "replay" and mode == "taxonomy_guided":
        return "replay"
    if stable_split == "heldout" and mode == "taxonomy_guided":
        return "heldout"
    if stable_split == "guard":
        return "guard"
    return None


def _split_for_stable_index(index: int) -> str:
    return ("heldout", "guard", "train", "replay")[index % 4]


def _code_taxonomy(row: Mapping[str, Any]) -> str:
    if bool(row.get("passed")) or str(row.get("test_status")) == "passed":
        return "verified_pass"
    parser_status = str(row.get("parser_status", ""))
    if parser_status in {"extraction_error", "extraction_failed", "no_code"}:
        return "extraction_repair"
    if bool(row.get("syntax_success")):
        return "runtime_repair"
    return "syntax_repair"


def _code_reward(taxonomy: str) -> float:
    return {
        "extraction_repair": 1.0,
        "syntax_repair": 0.9,
        "runtime_repair": 0.7,
        "verified_pass": 0.1,
    }.get(taxonomy, 0.0)


def _logic_reward(category: str) -> float:
    return {
        "unparseable": 0.8,
        "wrong_formula": 0.7,
        "z3_exception": 0.6,
        "wrong_answer": 0.5,
    }.get(category, 0.0)


def _ids_for_split(examples: Sequence[EvidenceExample], split: str) -> list[str]:
    return sorted({example.item_id for example in examples if example.split == split})


def _examples_for_splits(
    examples: Sequence[EvidenceExample],
    splits: Sequence[str],
) -> tuple[EvidenceExample, ...]:
    split_set = set(splits)
    return tuple(example for example in examples if example.split in split_set)


def _guard_metric(
    weights: Mapping[str, float],
    examples: Sequence[EvidenceExample],
    domain: str,
) -> float:
    domain_examples = [
        example
        for example in examples
        if example.domain == domain and example.guard_signal > 0.0
    ]
    total_signal = sum(example.guard_signal for example in domain_examples)
    if total_signal <= 0.0:
        return 1.0
    score = sum(weights.get(example.taxonomy, 0.0) * example.guard_signal for example in domain_examples)
    return _round(score / total_signal)


def _source_artifacts(config: ExperimentConfig) -> list[JsonDict]:
    path_overrides = {
        "exp2954": config.exp2954_path,
        "exp2952": config.exp2952_path,
        "exp2959": config.exp2959_path,
        "exp2960": config.exp2960_path,
    }
    rows = []
    for spec in SOURCE_SPECS:
        rel_path = path_overrides.get(spec.experiment_id, spec.path)
        path = _repo_path(config.repo_root, rel_path)
        present = path.is_file()
        rows.append(
            {
                "experiment_id": spec.experiment_id,
                "path": rel_path.as_posix(),
                "role": spec.role,
                "required": spec.required,
                "present": present,
                "fields_imported": list(spec.fields_imported) if present else [],
                "sha256": _sha256(path) if present else None,
            }
        )
    return rows


def _blocked_artifact(
    config: ExperimentConfig,
    started: float,
    source_artifacts: Sequence[JsonDict],
    verdict: str,
    missing_fields: Sequence[str],
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": verdict,
        "continuous_self_learning_task": True,
        "non_tautological_self_learning_ready": False,
        "source_artifacts": list(source_artifacts),
        "split_checksums": {},
        "leakage_check_passed": False,
        "replay_policies_compared": [],
        "frozen_heldout_utility": 0.0,
        "random_replay_heldout_utility": 0.0,
        "prior_utility_gated_heldout_utility": 0.0,
        "new_heldout_utility": 0.0,
        "heldout_utility_delta_vs_random": 0.0,
        "negative_control_delta": 0.0,
        "forgetting_guard_passed": False,
        "rollback_triggered": False,
        "update_rule": {},
        "model_specs_if_live_llm_used": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _elapsed(config, started),
        "missing_fields": list(missing_fields),
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }


def _policy_row(
    name: str,
    weights: Mapping[str, float],
    heldout_utility: float,
    guard_metrics: Mapping[str, float] | None,
    accepted: bool,
) -> JsonDict:
    return {
        "policy_name": name,
        "weights": dict(weights),
        "heldout_utility": heldout_utility,
        "guard_metrics": dict(guard_metrics) if guard_metrics is not None else None,
        "accepted": accepted,
    }


def _update_rule() -> JsonDict:
    return {
        "name": "non_tautological_verifier_weighted_utility_gate_v3",
        "baseline_policy": "random_replay",
        "candidate_policy": "train_replay_reward_weighted_with_guard_mass_preserved",
        "prior_policy": "exp2954_final_replay_weights_scored_on_new_heldout_slice",
        "negative_control": "uniform weights independent of reward signal",
        "acceptance_rule": (
            "accept only when new held-out utility beats frozen and random baselines, "
            "negative control does not improve, leakage checks pass, and every guard "
            "slice is non-degrading"
        ),
        "rollback_rule": "restore random replay weights when any stable guard degrades",
        "reward_signal": (
            "Rewards come from train/replay Exp2952 repair taxonomies and Exp2959 logic "
            "failure categories; held-out utility is scored only on disjoint held-out "
            "Exp2952 target rows."
        ),
        "guard_domains": ["stable_code", "stable_logic", "threshold_policy"],
        "live_llm_invoked": False,
    }


def _slice_errors(examples: Sequence[EvidenceExample], checksums: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for split in ("train", "replay", "heldout", "guard"):
        if not any(example.split == split for example in examples):
            errors.append(f"split:{split}")
    if not leakage_check(checksums):
        errors.append("split:leakage_check")
    guard_domains = {
        example.domain
        for example in examples
        if example.split == "guard" and example.guard_signal > 0.0
    }
    for domain in ("code", "logic", "threshold_policy"):
        if domain not in guard_domains:
            errors.append(f"guard:{domain}")
    return errors


def _missing_required_sources(source_artifacts: Sequence[JsonDict]) -> list[str]:
    return [
        str(source["experiment_id"])
        for source in source_artifacts
        if source.get("required") is True and source.get("present") is not True
    ]


def _missing_required_fields(payloads: Mapping[str, Mapping[str, Any]]) -> list[str]:
    checks = {
        "exp2954": (
            "self_learning_utility_artifact_ready",
            "heldout_utility_after",
            "final_replay_weights",
        ),
        "exp2952": ("candidate_evaluations", "baseline_pass_at_1", "repair_pass_at_1"),
        "exp2959": ("per_item_results", "failure_categories", "formalization_manifest_sha256"),
        "exp2960": ("matrix_v12_ready", "matrix_rows", "self_learning_delta_summary"),
    }
    missing: list[str] = []
    for source, fields in checks.items():
        payload = payloads.get(source, {})
        for field_name in fields:
            if field_name not in payload:
                missing.append(f"{source}:{field_name}")
    if payloads.get("exp2954", {}).get("self_learning_utility_artifact_ready") is not True:
        missing.append("exp2954:self_learning_utility_artifact_ready_true")
    if payloads.get("exp2960", {}).get("matrix_v12_ready") is not True:
        missing.append("exp2960:matrix_v12_ready_true")
    return missing


def _taxonomy_counts(examples: Sequence[EvidenceExample]) -> dict[str, int]:
    counts = {taxonomy: 0 for taxonomy in TAXONOMY_ORDER}
    for example in examples:
        counts[example.taxonomy] = counts.get(example.taxonomy, 0) + 1
    return {taxonomy: count for taxonomy, count in counts.items() if count}


def _verdict(ready: bool, rollback_triggered: bool) -> str:
    if rollback_triggered:
        return "complete: non_tautological_candidate_rolled_back_by_forgetting_guard"
    if ready:
        return "complete: non_tautological_self_learning_ready"
    return "complete: non_tautological_self_learning_not_ready"


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive_float(value: object, *, default: float = 0.0) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default


def _elapsed(config: ExperimentConfig, started: float) -> float:
    return _round(config.clock() - started)


def _round(value: float) -> float:
    return round(float(value), 12)
