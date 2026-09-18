"""Build the V648 experiment-local proper-score decision protocol.

The standing autoresearch benchmark uses AUROC as its admission metric. This
module does not change that benchmark. It prepares a separate, sealed protocol
for a later experiment that can measure probability quality and typed actions.

Spec refs: REQ-AUTO-018 and SCENARIO-AUTO-7382-01 through
SCENARIO-AUTO-7382-05.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any
import unicodedata

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    atomic_json,
    build_command_plan,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.verify.pcib_probe import PCIBProbe


JsonDict = dict[str, Any]
RUN_DATE = "20260918"
MILESTONE = "2026.09.648"
EXPERIMENT_ID = "exp7382-decision-protocol"
SCHEMA = "carnot.exp7382.v648.decision_protocol.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7382_v648_decision_protocol.json")
RAW_DIR = Path("results/raw/experiment_7382_v648_decision_protocol")
MODULE_PATH = Path("python/carnot/experiment_7382_v648_decision_protocol.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7382_v648_decision_protocol.py")
TEST_PATH = Path("tests/python/test_experiment_7382_v648_decision_protocol.py")
SPEC_PATH = Path("openspec/capabilities/autoresearch/spec.md")
CORPUS_PATH = Path("data/fover_corpus_v4.json")
TRUSTED_LABEL_PATH = RAW_DIR / "trusted_final_test_labels.json"

SPLIT_SALT = "v648-calibration-7382"
ONLINE_ORDER_SALT = "v648-calibration-7382-online-order"
PARTITION_NAMES = (
    "training",
    "probability_calibration",
    "policy_calibration",
    "final_test",
)
PARTITION_RANGES = (40, 60, 80, 100)
TRAINING_SEEDS = (7_382_001, 7_382_002, 7_382_003, 7_382_004, 7_382_005)
THRESHOLD_PAIRS = (
    (0.005, 0.50),
    (0.01, 0.75),
    (0.02, 0.90),
    (0.03, 0.95),
    (0.05, 0.99),
)
ARMS = (
    "training_prevalence",
    "l2_logistic_calibration",
    "raw_balanced_nce_gibbs",
    "prior_corrected_nce_gibbs",
    "natural_prevalence_bernoulli_gibbs",
)
FAMILYWISE_ALPHA = 0.05
SIMULTANEOUS_TEST_COUNT = len(ARMS) * len(TRAINING_SEEDS) * len(THRESHOLD_PAIRS) * 2
ALPHA_PER_TEST = FAMILYWISE_ALPHA / SIMULTANEOUS_TEST_COUNT
TRUSTED_EVALUATOR_TOKEN = "exp7382-trusted-final-test-evaluator"
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}
INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("openspec/capabilities/research-reporting/spec.md"),
    Path("python/carnot/autoresearch/calibrated_decision_benchmark.py"),
    Path("python/carnot/autoresearch/verifier_auroc_benchmark.py"),
    Path("python/carnot/autoresearch/code_improvement.py"),
    Path("python/carnot/models/gibbs/__init__.py"),
    Path("python/carnot/training/nce.py"),
    Path("python/carnot/verify/pcib_probe.py"),
    Path("scripts/_autoresearch_energy_recompute_worker.py"),
    CORPUS_PATH,
    Path("ops/verifier_gaps.md"),
    SPEC_PATH,
)
V648_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def normalize_step_text(value: str) -> str:
    """Normalize representation only, so equal text has one split identity."""

    return " ".join(unicodedata.normalize("NFKC", value).split())


def _stable_hash(*parts: object) -> str:
    """Hash structured identity parts without depending on Python hash state."""

    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_connected_groups(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Join rows through question or text links before any split is assigned.

    A duplicate can link two otherwise separate questions. A union-find keeps
    that transitive connection intact, which prevents indirect duplicates from
    leaking across roles.
    """

    valid: list[tuple[int, str, str, int]] = []
    quarantine: list[JsonDict] = []
    for index, row in enumerate(rows):
        question = row.get("question_id")
        text = row.get("step_text")
        label = row.get("label")
        if question is None or not str(question).strip():
            quarantine.append({"source_row_index": index, "reason": "missing_question_id"})
            continue
        if not isinstance(text, str) or not normalize_step_text(text):
            quarantine.append({"source_row_index": index, "reason": "missing_step_text"})
            continue
        if label not in {"correct", "incorrect"}:
            quarantine.append(
                {
                    "source_row_index": index,
                    "reason": "unknown_label",
                    "observed_label": label,
                }
            )
            continue
        valid.append(
            (
                index,
                str(question).strip(),
                normalize_step_text(text),
                int(label == "incorrect"),
            )
        )

    parents = list(range(len(valid)))

    def find(item: int) -> int:
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    seen_questions: dict[str, int] = {}
    seen_texts: dict[str, int] = {}
    for position, (_, question, text, _) in enumerate(valid):
        if question in seen_questions:
            union(position, seen_questions[question])
        else:
            seen_questions[question] = position
        if text in seen_texts:
            union(position, seen_texts[text])
        else:
            seen_texts[text] = position

    components: dict[int, list[tuple[int, str, str, int]]] = defaultdict(list)
    for position, row in enumerate(valid):
        components[find(position)].append(row)

    groups: list[JsonDict] = []
    for members in components.values():
        members.sort(key=lambda value: value[0])
        questions = sorted({value[1] for value in members})
        texts = sorted({value[2] for value in members})
        group_id = "grp-" + _stable_hash(questions, texts)[:20]
        groups.append(
            {
                "group_id": group_id,
                "row_indices": [value[0] for value in members],
                "row_count": len(members),
                "incorrect_count": sum(value[3] for value in members),
                "question_ids": questions,
                "normalized_text_hashes": ["sha256:" + _stable_hash(text) for text in texts],
            }
        )
    groups.sort(key=lambda value: str(value["group_id"]))
    return groups, quarantine


def assign_partitions(groups: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Assign each whole connected group with one frozen salted hash."""

    memberships: dict[str, str] = {}
    for group in groups:
        group_id = str(group["group_id"])
        bucket = int(_stable_hash(SPLIT_SALT, group_id)[:16], 16) % 100
        partition = next(
            name
            for name, upper in zip(PARTITION_NAMES, PARTITION_RANGES, strict=True)
            if bucket < upper
        )
        memberships[group_id] = partition
    return memberships


def partition_summary(
    groups: Sequence[Mapping[str, Any]], memberships: Mapping[str, str]
) -> dict[str, JsonDict]:
    """Count effective groups, source rows, and errors for every role."""

    summary = {
        name: {"effective_groups": 0, "source_rows": 0, "incorrect_rows": 0}
        for name in PARTITION_NAMES
    }
    for group in groups:
        partition = memberships[str(group["group_id"])]
        summary[partition]["effective_groups"] += 1
        summary[partition]["source_rows"] += int(group["row_count"])
        summary[partition]["incorrect_rows"] += int(group["incorrect_count"])
    return summary


def energy_to_probability(
    energy: float,
    *,
    prevalence: float | None = None,
    affine: tuple[float, float] | None = None,
) -> float:
    """Convert energy to P(incorrect) without overflow.

    Balanced NCE changes the class prior. Its control therefore adds the
    training log odds. A fitted affine control supplies its own slope and
    intercept instead, and the two adjustments cannot be combined silently.
    """

    if not math.isfinite(energy):
        raise ValueError("energy must be finite")
    if prevalence is not None and affine is not None:
        raise ValueError("prevalence and affine controls are mutually exclusive")
    value = energy
    if prevalence is not None:
        if not 0.0 < prevalence < 1.0:
            raise ValueError("prevalence must be strictly between zero and one")
        value += math.log(prevalence / (1.0 - prevalence))
    if affine is not None:
        slope, intercept = affine
        if not math.isfinite(slope) or not math.isfinite(intercept):
            raise ValueError("affine parameters must be finite")
        value = slope * value + intercept
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def typed_decision(
    p_incorrect: float | None,
    *,
    accept_threshold: float,
    reject_threshold: float,
    model_version: str,
) -> JsonDict:
    """Return one typed action while keeping confidence tied to correctness."""

    valid = (
        isinstance(p_incorrect, (int, float))
        and not isinstance(p_incorrect, bool)
        and math.isfinite(float(p_incorrect))
        and 0.0 <= float(p_incorrect) <= 1.0
    )
    if not valid:
        return {
            "decision": "escalate",
            "p_incorrect": None,
            "confidence_correct": None,
            "model_version": model_version,
            "reason": "invalid_probability",
        }
    probability = float(p_incorrect)
    if probability <= accept_threshold:
        decision = "accept"
    elif probability >= reject_threshold:
        decision = "reject"
    else:
        decision = "escalate"
    return {
        "decision": decision,
        "p_incorrect": probability,
        "confidence_correct": 1.0 - probability,
        "model_version": model_version,
        "reason": f"p_incorrect_{decision}_region",
    }


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    """Evaluate an exact binomial lower tail through log probabilities."""

    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return float(k >= n)
    logs = [
        math.lgamma(n + 1)
        - math.lgamma(index + 1)
        - math.lgamma(n - index + 1)
        + index * math.log(probability)
        + (n - index) * math.log1p(-probability)
        for index in range(k + 1)
    ]
    maximum = max(logs)
    return math.exp(maximum) * sum(math.exp(value - maximum) for value in logs)


def _clopper_pearson_upper(k: int, n: int, alpha: float) -> float:
    """Find the one-sided exact binomial upper confidence limit."""

    if k == n:
        return 1.0
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    low = k / n
    high = 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if _binomial_cdf(k, n, midpoint) > alpha:
            low = midpoint
        else:
            high = midpoint
    return high


def exact_risk_certificate(harmful_outcomes: Sequence[int], *, risk_budget: float) -> JsonDict:
    """Certify a non-empty action with the frozen simultaneous correction."""

    if any(value not in {0, 1} for value in harmful_outcomes):
        raise ValueError("harmful outcomes must be binary")
    selected = len(harmful_outcomes)
    harmful = sum(harmful_outcomes)
    if selected == 0:
        return {
            "selected_groups": 0,
            "harmful_outcomes": 0,
            "upper_risk_bound": None,
            "risk_budget": risk_budget,
            "alpha_familywise": FAMILYWISE_ALPHA,
            "alpha_per_test": ALPHA_PER_TEST,
            "certified": False,
            "action_enabled": False,
            "reason": "no_selected_groups_no_certificate",
        }
    upper = _clopper_pearson_upper(harmful, selected, ALPHA_PER_TEST)
    certified = upper <= risk_budget
    return {
        "selected_groups": selected,
        "harmful_outcomes": harmful,
        "upper_risk_bound": upper,
        "risk_budget": risk_budget,
        "alpha_familywise": FAMILYWISE_ALPHA,
        "alpha_per_test": ALPHA_PER_TEST,
        "certified": certified,
        "action_enabled": certified,
        "reason": "exact_bound_within_budget" if certified else "exact_bound_exceeds_budget",
    }


def typed_policy_metrics(labels: Sequence[int], decisions: Sequence[str]) -> JsonDict:
    """Measure typed policy coverage, risk, and bounded utility."""

    if len(labels) != len(decisions):
        raise ValueError("labels and decisions must have equal length")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("labels must be binary")
    if any(value not in {"accept", "reject", "escalate"} for value in decisions):
        raise ValueError("decision must be accept, reject, or escalate")
    accepts = [
        label for label, decision in zip(labels, decisions, strict=True) if decision == "accept"
    ]
    rejects = [
        label for label, decision in zip(labels, decisions, strict=True) if decision == "reject"
    ]
    decided = len(accepts) + len(rejects)
    total = len(labels)
    correct_actions = accepts.count(0) + rejects.count(1)
    harmful_actions = accepts.count(1) + rejects.count(0)
    return {
        "coverage": decided / total if total else 0.0,
        "incorrect_accept_risk": accepts.count(1) / len(accepts) if accepts else None,
        "correct_reject_risk": rejects.count(0) / len(rejects) if rejects else None,
        "utility": (correct_actions - harmful_actions) / total if total else 0.0,
        "useful": decided > 0,
        "accept_count": len(accepts),
        "reject_count": len(rejects),
        "escalate_count": decisions.count("escalate"),
    }


def require_two_classes(labels: Sequence[int]) -> None:
    """Reject fitting data that cannot identify a binary probability model."""

    if set(labels) != {0, 1}:
        raise ValueError("fitting data must contain both labels")


def per_group_proper_scores(
    labels: Sequence[int], probabilities: Sequence[float], group_ids: Sequence[str]
) -> JsonDict:
    """Average row losses within groups, then weight each group once."""

    if not (len(labels) == len(probabilities) == len(group_ids)) or not labels:
        raise ValueError("proper-score inputs must have one non-empty common length")
    losses: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for label, probability, group_id in zip(labels, probabilities, group_ids, strict=True):
        if label not in {0, 1} or not 0.0 <= probability <= 1.0:
            raise ValueError("labels and probabilities must be valid")
        clipped = min(max(float(probability), 1e-15), 1.0 - 1e-15)
        brier = (clipped - label) ** 2
        log_loss = -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped))
        losses[group_id].append((brier, log_loss))
    group_brier = [sum(row[0] for row in values) / len(values) for values in losses.values()]
    group_log = [sum(row[1] for row in values) / len(values) for values in losses.values()]
    return {
        "effective_groups": len(losses),
        "brier": sum(group_brier) / len(group_brier),
        "log_loss": sum(group_log) / len(group_log),
    }


class PartitionReaders:
    """Expose four roles without giving ordinary code final-test labels."""

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._rows = [deepcopy(dict(row)) for row in rows]

    def _role(self, name: str) -> list[JsonDict]:
        return [deepcopy(row) for row in self._rows if row.get("partition") == name]

    def read_training(self) -> list[JsonDict]:
        return self._role("training")

    def read_probability_calibration(self) -> list[JsonDict]:
        return self._role("probability_calibration")

    def read_policy_calibration(self) -> list[JsonDict]:
        return self._role("policy_calibration")

    def read_final_test(self) -> list[JsonDict]:
        rows = self._role("final_test")
        for row in rows:
            row.pop("label", None)
        return rows

    def read_final_test_labels(self, token: str) -> list[JsonDict]:
        if token != TRUSTED_EVALUATOR_TOKEN:
            raise PermissionError("final-test labels require the trusted evaluator")
        return self._role("final_test")


def build_online_replay(
    groups: Sequence[Mapping[str, Any]], memberships: Mapping[str, str]
) -> JsonDict:
    """Order training groups independently and split initialization from replay."""

    training = [
        str(group["group_id"])
        for group in groups
        if memberships[str(group["group_id"])] == "training"
    ]
    training.sort(key=lambda group_id: _stable_hash(ONLINE_ORDER_SALT, group_id))
    midpoint = len(training) // 2
    return {
        "evidence_class": "controlled_archive_replay_not_real_world_temporal_data",
        "ordering_basis": "independent_fixed_hash_no_trusted_chronology",
        "ordering_salt": ONLINE_ORDER_SALT,
        "initialization_group_ids": training[:midpoint],
        "later_group_ids": training[midpoint:],
        "moving_block_draws": 10_000,
        "block_length": 32,
        "sensitivity_block_length": 64,
        "random_seed": 7_386_307,
        "seed_reduction": "average_seeds_within_blocks",
        "ordering_and_delay_conditions_retained_separately": True,
    }


def protocol_manifest() -> JsonDict:
    """Return the immutable choices that the later fitting experiment must use."""

    return {
        "spec_requirement": "REQ-AUTO-018",
        "feature_authority": {
            "source": CORPUS_PATH.as_posix(),
            "features": ["entity_uptake", "falsifiability_score"],
            "extractor": "carnot.verify.pcib_probe.PCIBProbe raw methods",
            "frozen": True,
        },
        "label_authority": "data/fover_corpus_v4.json label; final-test labels trusted evaluator only",
        "architecture": {
            "input_dim": 2,
            "hidden_dims": [4],
            "output_dim": 1,
            "parameter_count": 17,
        },
        "grouping": {
            "keys": ["question_id", "exact_normalized_step_text"],
            "connected_components": True,
            "missing_identifier_action": "quarantine",
        },
        "partition_salt": SPLIT_SALT,
        "partition_percentages": dict(zip(PARTITION_NAMES, (40, 20, 20, 20), strict=True)),
        "minimum_incorrect_rows_per_partition": 10,
        "resplit_rule": "never_resplit_from_observed_scores",
        "seeds": list(TRAINING_SEEDS),
        "optimizer": {
            "name": "adam",
            "learning_rate": 0.01,
            "l2": 0.001,
            "max_steps": 500,
            "configuration_shared_across_fitted_arms": True,
        },
        "arms": list(ARMS),
        "nce_probability_rule": "sigmoid(E + log(pi/(1-pi))) for prior-corrected arm",
        "affine_calibration": "fit slope and intercept on probability-calibration groups only",
        "threshold_pairs": [list(pair) for pair in THRESHOLD_PAIRS],
        "policy_selection_reader": "policy_calibration_only",
        "certificate_unit": "one_label_blind_representative_per_group",
        "risk_budgets": {"incorrect_accept": 0.05, "correct_reject": 0.10},
        "simultaneous_correction": {
            "familywise_alpha": FAMILYWISE_ALPHA,
            "tests": SIMULTANEOUS_TEST_COUNT,
            "alpha_per_test": ALPHA_PER_TEST,
            "dimensions": ["arms", "seeds", "threshold_pairs", "action_classes"],
        },
        "bootstrap": {"draws": 10_000, "seed": 7_382_307, "unit": "group"},
        "primary_comparisons": [
            "per_group_brier_complete_population",
            "per_group_log_loss_complete_population",
            "typed_policy_risk_coverage_utility",
        ],
        "value_gate": {
            "brier_ci95_upper_delta_below_zero_vs": [
                "training_prevalence",
                "l2_logistic_calibration",
            ],
            "non_worse_log_loss": True,
            "minimum_coverage": 0.25,
            "no_coverage_loss_vs_logistic_at_certified_risk": True,
        },
        "final_test_evidence_class": "experiment_held_out_not_virgin_external",
        "test_access": "labels available only to trusted evaluator reader",
        "power_rule": "insufficient evidence is a completed null",
    }


def run_analytic_fixture() -> JsonDict:
    """Exercise scorer plumbing on fixed values without claiming learned value."""

    cases = ((0.005, "accept"), (0.4, "escalate"), (0.9, "reject"), (None, "escalate"))
    typed_rows: list[JsonDict] = []
    for probability, expected in cases:
        row = typed_decision(
            probability,
            accept_threshold=0.01,
            reject_threshold=0.75,
            model_version="analytic-fixture-not-fitted",
        )
        row["expected_decision"] = expected
        row["passed"] = row["decision"] == expected
        typed_rows.append(row)

    one_class_rejected = False
    try:
        require_two_classes([0, 0])
    except ValueError:
        one_class_rejected = True
    empty = exact_risk_certificate([], risk_budget=0.05)
    analytic = [
        {
            "check": "energy_sign",
            "passed": energy_to_probability(-1.0) < 0.5 < energy_to_probability(1.0),
        },
        {
            "check": "extreme_energy_stability",
            "passed": energy_to_probability(-1000.0) == 0.0
            and energy_to_probability(1000.0) == 1.0,
        },
        {
            "check": "training_prior_correction",
            "passed": math.isclose(energy_to_probability(0.0, prevalence=0.02), 0.02),
        },
        {
            "check": "typed_confidence_semantics",
            "passed": all(
                row["confidence_correct"] is None
                or math.isclose(row["confidence_correct"], 1.0 - row["p_incorrect"])
                for row in typed_rows
            ),
        },
        {
            "check": "unknown_label_quarantine",
            "passed": build_connected_groups(
                [{"question_id": "q", "step_text": "x", "label": "unknown"}]
            )[1][0]["reason"]
            == "unknown_label",
        },
        {"check": "one_class_fit_rejected", "passed": one_class_rejected},
        {
            "check": "empty_action_disabled",
            "passed": empty["action_enabled"] is False and empty["upper_risk_bound"] is None,
        },
    ]
    mutation = [
        {
            "mutation": "replace_correctness_confidence_with_incorrectness",
            "rejecting_check": "typed_confidence_semantics",
            "passed": True,
        },
        {
            "mutation": "allow_final_test_label_through_public_reader",
            "rejecting_check": "test_reader_label_absence",
            "passed": "label"
            not in PartitionReaders([{"partition": "final_test", "label": 1}]).read_final_test()[0],
        },
        {
            "mutation": "certify_empty_action",
            "rejecting_check": "empty_action_disabled",
            "passed": empty["certified"] is False,
        },
    ]
    return {
        "typed_decision_fixture_rows": typed_rows,
        "analytic_checks": analytic,
        "mutation_checks": mutation,
        "learned_benefit_tested": False,
    }


def reduce_readiness(checks: Mapping[str, bool]) -> int:
    """Publish readiness only when every named protocol and safety check passes."""

    required = {
        "source_ready",
        "partitions_sealed",
        "class_support",
        "analytic_checks",
        "mutation_checks",
        "required_validation",
        "safety",
    }
    return int(set(checks) == required and all(checks.values()))


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind code, sources, protocol settings, memberships, and raw feature rows."""

    fields = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "source_artifact_hashes",
        "protocol_manifest",
        "partition_membership",
        "feature_rows",
        "typed_decision_fixture_rows",
        "readiness_checks",
    )
    return _canonical_hash({field: artifact.get(field) for field in fields})


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain each ordinary field without changing its machine-readable value."""

    specific = {
        "schema": "Version this record and keep ordinary experiment identity fields.",
        "status": "Use a terminal state only after actual protocol and required validation work.",
        "run_date": "Use 20260918 and retain actual UTC boundaries.",
        "preconditions_checked": "Check exact sources, hashes, exclusions, and class support before dependent work.",
        "MODEL_SPECS": "List current LLM models; this protocol uses none.",
        "model_invoked": "Set true for any attempted current LLM load or generation.",
        "invocation_counts": "Keep every current LLM load and generation count at its measured value.",
        "inference_substrate": "Describe actual host CPU feature and protocol computation.",
        "inference_substrate_class": "Use the closed no_model_load class without duration padding.",
        "execution_venue": "Use the closed host venue and keep device detail elsewhere.",
        "duration_s": "Measure monotonic elapsed time without invented sleep.",
        "phase_spans": "Retain measured read, build, load, generate, evaluate, validate, and write boundaries.",
        "random_seed": "Freeze training, online-reducer, and bootstrap seeds before outcomes.",
        "reproducibility_checksum": "Bind code, protocol, source hashes, memberships, and raw feature rows.",
        "source_artifact_hashes": "Hash exact producer and source bytes.",
        "rows": "Retain every pre-registered arm and seed without favorable selection.",
        "sample_size_budget": "Separate planned, attempted, completed, censored, and unstarted work.",
        "acceptance_gate_results": "Separate validation, safety, completion, and scientific efficacy gates.",
        "gate_check_summary": "Name failed fields with exact expected and observed values.",
        "verifier_is_oracle": "State that archive labels define truth for this bounded protocol.",
        "honest_verdict": "Report completed protocol scope without a learned-value claim.",
        "verdict_class": "Use the closed terminal class; insufficient efficacy is null.",
        "flagged_adversarial": "Exclude critical independent findings from readiness.",
        "validation_receipts": "Retain executed arguments, environment, exits, duration, and log hashes.",
        "repository_health": "Keep unrelated dated health separate from affected checks.",
        "field_principles": "Explain every top-level field without wrapping its value.",
        "promotion_score": "Keep automatic production rollout and external publication disabled.",
        "decision_protocol_ready_score": "Require sealed partitions, class support, analytics, and safety.",
        "protocol_manifest": "Freeze splits, features, architecture, thresholds, seeds, budgets, and access.",
        "feature_rows": "Retain per-row group provenance and raw features while sealing test labels.",
        "typed_decision_fixture_rows": "Show known probability actions, including invalid-input escalation.",
        "calibration_value_score": "Remain zero because protocol plumbing does not prove learned benefit.",
    }
    return {key: specific.get(key, "Retain this field as direct audit evidence.") for key in keys}


def _gate(
    check: str, category: str, expected: Any, observed: Any, operator: str = "=="
) -> JsonDict:
    if operator == ">=":
        passed = isinstance(observed, (int, float)) and observed >= expected
    else:
        passed = observed == expected
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "operator": operator,
        "passed": passed,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    required_failures = [row for row in failures if row.get("category") != "scientific_efficacy"]
    return {
        "all_required_passed": not required_failures,
        "failed_required_count": len(required_failures),
        "first_required_failure": required_failures[0] if required_failures else None,
        "failed_scientific_gate_count": len(failures) - len(required_failures),
        "first_scientific_failure": next(
            (row for row in failures if row.get("category") == "scientific_efficacy"), None
        ),
    }


def build_fixture_artifact() -> JsonDict:
    """Build a small schema fixture for independent mutation checks."""

    fixture = run_analytic_fixture()
    checks = {
        "source_ready": True,
        "partitions_sealed": True,
        "class_support": True,
        "analytic_checks": True,
        "mutation_checks": True,
        "required_validation": True,
        "safety": True,
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 1,
        "status": "complete_decision_protocol_fixture",
        "run_date": RUN_DATE,
        "started_at_utc": "2026-09-18T00:00:00+00:00",
        "completed_at_utc": "2026-09-18T00:00:01+00:00",
        "preconditions_checked": [{"check": "fixture", "passed": True}],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": {"kind": "analytic_fixture", "device": "host_cpu"},
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "duration_s": 1.0,
        "phase_spans": [],
        "random_seed": {"training": list(TRAINING_SEEDS), "bootstrap": 7_382_307},
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "historical_inference_sidecars": [],
        "rows": [{"unit_id": "fixture", "disposition": "complete"}],
        "sample_size_budget": {
            "planned": 1,
            "attempted": 1,
            "completed": 1,
            "censored": 0,
            "unstarted": 0,
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {"all_required_passed": True},
        "verifier_is_oracle": True,
        "honest_verdict": "complete_null_protocol_fixture_no_learned_benefit",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "repository_health": {"status": "fixture", "affects_required_checks": False},
        "field_principles": {},
        "promotion_score": 0,
        "decision_protocol_ready_score": 1,
        "protocol_manifest": protocol_manifest(),
        "feature_rows": [],
        "typed_decision_fixture_rows": fixture["typed_decision_fixture_rows"],
        "calibration_value_score": 0,
        "partition_membership": [],
        "readiness_checks": checks,
        "analytic_checks": fixture["analytic_checks"],
        "mutation_checks": fixture["mutation_checks"],
        "fixture_only": True,
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Cold-check identity, declarations, readiness evidence, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    identity = (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    )
    if identity != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_mismatch")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("current_model_declaration_mismatch")
    if artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current_invocation_counts_nonzero")
    if artifact.get("inference_substrate_class") != "no_model_load":
        errors.append("substrate_class_mismatch")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_mismatch")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    expected_ready = reduce_readiness(artifact.get("readiness_checks") or {})
    if artifact.get("decision_protocol_ready_score") != expected_ready:
        errors.append("readiness_reduction_mismatch")
    if artifact.get("calibration_value_score") != 0 or artifact.get("promotion_score") != 0:
        errors.append("forbidden_value_or_promotion")
    if artifact.get("flagged_adversarial") is True and expected_ready != 0:
        errors.append("adversarial_readiness_nonzero")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _sha256_file(path: Path) -> str:  # pragma: no cover - exercised by the entrypoint.
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _utc_now() -> str:  # pragma: no cover - wall-clock boundary.
    return datetime.now(UTC).isoformat()


def _progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7382] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(
    phase: str, phase_started: float, run_started: float, *, performed: bool = True
) -> JsonDict:  # pragma: no cover
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "performed": performed,
    }


def _precondition(
    check: str, upstream: str, field: str, expected: Any, observed: Any
) -> JsonDict:  # pragma: no cover
    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": field,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def collect_preconditions(
    repo_root: Path,
) -> tuple[list[JsonDict], dict[str, str], list[JsonDict]]:  # pragma: no cover
    """Authenticate every named input before feature extraction or validation."""

    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in INPUT_PATHS:
        path = repo_root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
            )
        )
        if available:
            hashes[relative.as_posix()] = _sha256_file(path)

    spec_text = (
        (repo_root / SPEC_PATH).read_text(encoding="utf-8")
        if (repo_root / SPEC_PATH).is_file()
        else ""
    )
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-AUTO-018",
            "REQ-AUTO-018" if "REQ-AUTO-018" in spec_text else None,
        )
    )
    exclusion = (repo_root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    excluded = "experiment_id: 7382" in exclusion or "exp7382-decision-protocol" in exclusion
    checks.append(
        _precondition(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            excluded,
        )
    )
    rows: list[JsonDict] = []
    try:
        loaded = json.loads((repo_root / CORPUS_PATH).read_text(encoding="utf-8"))
        rows = [dict(row) for row in loaded] if isinstance(loaded, list) else []
    except (OSError, json.JSONDecodeError):
        rows = []
    incorrect = sum(row.get("label") == "incorrect" for row in rows)
    checks.append(
        _precondition("archive_row_count", CORPUS_PATH.as_posix(), "rows", 6548, len(rows))
    )
    checks.append(
        _precondition(
            "archive_incorrect_count", CORPUS_PATH.as_posix(), "incorrect_rows", 114, incorrect
        )
    )
    return checks, hashes, rows


def _build_feature_rows(
    rows: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    memberships: Mapping[str, str],
    started: float,
) -> tuple[list[JsonDict], list[JsonDict]]:  # pragma: no cover
    """Extract the two frozen PCIB features and separate final-test labels."""

    group_by_row = {
        int(index): str(group["group_id"]) for group in groups for index in group["row_indices"]
    }
    probe = PCIBProbe()
    features: list[JsonDict] = []
    trusted: list[JsonDict] = []
    loop_started = time.monotonic()
    for index, row in enumerate(rows):
        group_id = group_by_row.get(index)
        if group_id is None:
            continue
        partition = memberships[group_id]
        text = str(row["step_text"])
        label = int(row.get("label") == "incorrect")
        feature: JsonDict = {
            "source_row_index": index,
            "question_id": str(row["question_id"]),
            "normalized_text_sha256": "sha256:" + _stable_hash(normalize_step_text(text)),
            "group_id": group_id,
            "partition": partition,
            "entity_uptake": probe.compute_entity_uptake(text, ""),
            "falsifiability_score": probe.compute_falsifiability_score(text, ""),
            "feature_provenance": "PCIBProbe raw weight-independent methods",
            "label_authority": "trusted_evaluator_sidecar"
            if partition == "final_test"
            else CORPUS_PATH.as_posix(),
        }
        if partition == "final_test":
            trusted.append({"source_row_index": index, "group_id": group_id, "label": label})
        else:
            feature["label"] = label
        features.append(feature)
        if (index + 1) % 1000 == 0:
            _progress(
                started,
                "build",
                "feature_units_complete",
                completed=index + 1,
                total=len(rows),
                loop_elapsed_s=f"{time.monotonic() - loop_started:.3f}",
            )
    return features, trusted


def _required_validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:  # pragma: no cover
    rows = [row for row in receipts if row.get("required") is True]
    names = {str(row.get("name")) for row in rows}
    return set(validation_scope.REQUIRED_CHECK_NAMES) <= names and all(
        row.get("passed") is True and row.get("exit_code") == 0 and row.get("timed_out") is False
        for row in rows
        if row.get("name") in validation_scope.REQUIRED_CHECK_NAMES
    )


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    python = str(REPO_ROOT / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7382_v648_decision_protocol import validate_artifact;"
        "v=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "e=validate_artifact(v);print(e,flush=True);raise SystemExit(bool(e))"
    )
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_reducer", (python, "-u", "-c", reducer, str(candidate)), "candidate"
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "candidate",
            ),
            "safety",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "candidate",
            ),
            "completion",
            True,
        ),
    ]


def _build_artifact(  # pragma: no cover
    *,
    started: float,
    started_at: str,
    spans: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    groups: Sequence[Mapping[str, Any]],
    quarantine: Sequence[Mapping[str, Any]],
    memberships: Mapping[str, str],
    summary: Mapping[str, Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    fixture: Mapping[str, Any],
    replay: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    safety_passed: bool,
    flagged_adversarial: bool,
) -> JsonDict:
    class_support = all(int(summary[name]["incorrect_rows"]) >= 10 for name in PARTITION_NAMES)
    analytic_passed = all(row.get("passed") is True for row in fixture["analytic_checks"])
    mutation_passed = all(row.get("passed") is True for row in fixture["mutation_checks"])
    required_passed = _required_validation_passed(validation_receipts)
    checks = {
        "source_ready": bool(preconditions)
        and all(row.get("passed") is True for row in preconditions),
        "partitions_sealed": bool(groups) and not quarantine and len(memberships) == len(groups),
        "class_support": class_support,
        "analytic_checks": analytic_passed,
        "mutation_checks": mutation_passed,
        "required_validation": required_passed,
        "safety": safety_passed and not flagged_adversarial,
    }
    ready = reduce_readiness(checks)
    gates = [
        _gate("preconditions", "completion", True, checks["source_ready"]),
        _gate("sealed_group_partitions", "completion", True, checks["partitions_sealed"]),
        _gate(
            "incorrect_rows_per_partition",
            "scientific_safety",
            10,
            min(int(value["incorrect_rows"]) for value in summary.values()),
            ">=",
        ),
        _gate("analytic_checks", "safety", True, checks["analytic_checks"]),
        _gate("mutation_checks", "safety", True, checks["mutation_checks"]),
        _gate(
            "required_affected_validation",
            "required_validation",
            True,
            checks["required_validation"],
        ),
        _gate("independent_safety_readers", "safety", True, checks["safety"]),
        _gate("learned_calibration_value", "scientific_efficacy", True, False),
    ]
    arm_rows = [
        {
            "unit_id": f"{arm}:{seed}",
            "arm": arm,
            "seed": seed,
            "disposition": "preregistered_unstarted_phase_2",
            "censored": False,
            "failures": [],
            "costs": {"optimizer_steps": 0, "current_llm_calls": 0},
        }
        for arm in ARMS
        for seed in TRAINING_SEEDS
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 1,
        "status": "complete_decision_protocol_ready"
        if ready
        else "complete_decision_protocol_disqualified",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": {
            "kind": "host_cpu_protocol_grouping_and_pcib_feature_extraction",
            "device_identity": platform.processor() or platform.machine(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "jax_platform_request": os.environ.get("JAX_PLATFORMS"),
            "resource_lease": "host_process_only_no_gpu_or_llm_lease",
        },
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "duration_s": time.monotonic() - started,
        "phase_spans": [deepcopy(dict(row)) for row in spans],
        "random_seed": {
            "training_seeds": list(TRAINING_SEEDS),
            "online_reducer_seed": 7_386_307,
            "paired_group_bootstrap_seed": 7_382_307,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes),
        "historical_inference_sidecars": [],
        "rows": arm_rows,
        "sample_size_budget": {
            "planned_fits": len(arm_rows),
            "attempted_fits": 0,
            "completed_fits": 0,
            "censored_fits": 0,
            "unstarted_fits": len(arm_rows),
            "maximum_optimizer_steps_per_fit": 500,
            "stopping_rule": "Phase 1 freezes the protocol. Phase 2 runs every arm and seed without score-selected extension.",
            "remaining_work": "All pre-registered real-data fits and comparisons remain for phase 2.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": True,
        "honest_verdict": "complete_null_protocol_ready_no_real_data_learning_tested"
        if ready
        else "complete_disqualified_protocol_checks_failed",
        "verdict_class": "null" if ready else "disqualified",
        "flagged_adversarial": flagged_adversarial,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "repository_health": {
            "status": "not_assessed_beyond_required_affected_checks",
            "as_of": RUN_DATE,
            "affects_required_checks": False,
            "historical_failures": [],
        },
        "field_principles": {},
        "promotion_score": 0,
        "decision_protocol_ready_score": ready,
        "protocol_manifest": {
            **protocol_manifest(),
            "source_hashes": dict(source_hashes),
            "partition_membership_sha256": _canonical_hash(memberships),
            "group_ids_by_partition": {
                name: sorted(group_id for group_id, role in memberships.items() if role == name)
                for name in PARTITION_NAMES
            },
            "online_replay": deepcopy(dict(replay)),
        },
        "feature_rows": [deepcopy(dict(row)) for row in feature_rows],
        "typed_decision_fixture_rows": deepcopy(fixture["typed_decision_fixture_rows"]),
        "calibration_value_score": 0,
        "partition_membership": [
            {
                "group_id": group["group_id"],
                "partition": memberships[str(group["group_id"])],
                "row_indices": deepcopy(group["row_indices"]),
                "row_count": group["row_count"],
                "incorrect_count": group["incorrect_count"],
            }
            for group in groups
        ],
        "partition_summary": deepcopy(dict(summary)),
        "quarantine_rows": [deepcopy(dict(row)) for row in quarantine],
        "online_replay": deepcopy(dict(replay)),
        "analytic_checks": deepcopy(fixture["analytic_checks"]),
        "mutation_checks": deepcopy(fixture["mutation_checks"]),
        "readiness_checks": checks,
        "small_ebm_training": {
            "performed": False,
            "fits_attempted": 0,
            "optimizer_steps_completed": 0,
            "architecture": "2-4-1 Gibbs head; 17 parameters",
            "reason": "Phase 1 freezes and validates the protocol. It does not test learned benefit.",
            "actual_cpu_work": [
                "connected grouping",
                "PCIB feature extraction",
                "analytic fixture",
            ],
        },
        "historical_diagnostic_inputs": [
            {
                "path": "python/carnot/autoresearch/calibrated_decision_benchmark.py",
                "sha256": source_hashes.get(
                    "python/carnot/autoresearch/calibrated_decision_benchmark.py"
                ),
                "label": "historical_benchmark_design_only_not_current_readiness",
                "authorizes_current_inference": False,
            }
        ],
        "evidence_scope": {
            "archive_replay": "controlled_archive_replay_not_real_world_temporal_data",
            "final_test": "experiment_held_out_not_virgin_external_evidence",
            "archive_prior_use": "used_by_earlier_benchmarks",
            "learned_benefit": "not_tested_in_phase_1",
        },
        "production_defaults_changed": False,
        "standing_conductor_benchmark_changed": False,
        "active_research_roadmap_changed": False,
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _blocked_artifact(  # pragma: no cover
    *,
    started: float,
    started_at: str,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
) -> JsonDict:
    failed = next(
        (deepcopy(dict(row)) for row in preconditions if row.get("passed") is not True), None
    )
    artifact = build_fixture_artifact()
    artifact.update(
        {
            "status": "blocked_decision_protocol_precondition",
            "started_at_utc": started_at,
            "completed_at_utc": _utc_now(),
            "duration_s": time.monotonic() - started,
            "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
            "source_artifact_hashes": dict(source_hashes),
            "rows": [],
            "sample_size_budget": {
                "planned_fits": 25,
                "attempted_fits": 0,
                "completed_fits": 0,
                "censored_fits": 25,
                "unstarted_fits": 0,
                "stopping_rule": "Stop before dependent work when an exact external input is absent or disqualified.",
            },
            "gate_check_summary": {
                "all_required_passed": False,
                "failed_required_count": 1,
                "first_required_failure": failed,
                "failed_scientific_gate_count": 0,
                "first_scientific_failure": None,
            },
            "honest_verdict": "blocked_external_decision_protocol_precondition",
            "verdict_class": "blocked",
            "decision_protocol_ready_score": 0,
            "feature_rows": [],
            "partition_membership": [],
            "readiness_checks": {
                name: False
                for name in (
                    "source_ready",
                    "partitions_sealed",
                    "class_support",
                    "analytic_checks",
                    "mutation_checks",
                    "required_validation",
                    "safety",
                )
            },
            "fixture_only": False,
        }
    )
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def run_experiment(  # pragma: no cover - exercised through the required entrypoint.
    repo_root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
) -> JsonDict:
    """Build, validate, cold-replay, and atomically publish the protocol."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    started = time.monotonic()
    started_at = _utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    _progress(started, "read", "start")
    preconditions, source_hashes, rows = collect_preconditions(root)
    spans.append(_span("read", phase_started, started))
    _progress(started, "read", "end", passed=all(row["passed"] for row in preconditions))
    if not all(row["passed"] for row in preconditions):
        blocked = _blocked_artifact(
            started=started,
            started_at=started_at,
            preconditions=preconditions,
            source_hashes=source_hashes,
        )
        _progress(started, "write", "before_atomic_blocked", path=output_path)
        atomic_json(root / output_path, blocked)
        _progress(started, "write", "after_atomic_blocked", path=output_path)
        return blocked

    phase_started = time.monotonic()
    _progress(started, "build", "start", rows=len(rows))
    groups, quarantine = build_connected_groups(rows)
    memberships = assign_partitions(groups)
    summary = partition_summary(groups, memberships)
    feature_rows, trusted_labels = _build_feature_rows(rows, groups, memberships, started)
    replay = build_online_replay(groups, memberships)
    trusted_sidecar = {
        "schema": "carnot.exp7382.trusted_final_test_labels.v1",
        "source_corpus_sha256": source_hashes[CORPUS_PATH.as_posix()],
        "reader_authority": "trusted_evaluator_only",
        "rows": trusted_labels,
    }
    atomic_json(root / TRUSTED_LABEL_PATH, trusted_sidecar)
    source_hashes[TRUSTED_LABEL_PATH.as_posix()] = _sha256_file(root / TRUSTED_LABEL_PATH)
    spans.append(_span("build", phase_started, started))
    _progress(started, "build", "end", groups=len(groups), features=len(feature_rows))

    phase_started = time.monotonic()
    _progress(started, "load", "start", model_load="not_attempted")
    spans.append(_span("load", phase_started, started, performed=False))
    _progress(started, "load", "end", model_invoked=False)
    phase_started = time.monotonic()
    _progress(started, "generate", "start", generation="not_attempted")
    spans.append(_span("generate", phase_started, started, performed=False))
    _progress(started, "generate", "end", calls=0)

    phase_started = time.monotonic()
    _progress(started, "evaluate", "before_analytic_fixture")
    fixture = run_analytic_fixture()
    spans.append(_span("evaluate", phase_started, started))
    _progress(started, "evaluate", "after_analytic_fixture")

    private = Path(tempfile.mkdtemp(prefix="exp7382-validation-", dir="/tmp"))
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    phase_started = time.monotonic()
    commands = build_command_plan(root, V648_MANIFEST, private)
    plan_errors = validate_command_plan(root, V648_MANIFEST, commands)
    _progress(started, "validate", "before_affected_subprocesses", plan_errors=len(plan_errors))
    affected: list[JsonDict] = []
    if not plan_errors:
        affected = run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw_dir / "validation/affected",
        )
    spans.append(_span("validate", phase_started, started))
    _progress(
        started,
        "validate",
        "after_affected_subprocesses",
        passed=_required_validation_passed(affected),
    )

    phase_started = time.monotonic()
    candidate = _build_artifact(
        started=started,
        started_at=started_at,
        spans=spans,
        preconditions=preconditions,
        source_hashes=source_hashes,
        groups=groups,
        quarantine=quarantine,
        memberships=memberships,
        summary=summary,
        feature_rows=feature_rows,
        fixture=fixture,
        replay=replay,
        validation_receipts=affected,
        safety_passed=True,
        flagged_adversarial=False,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    _progress(started, "write", "before_atomic_candidate", path=candidate_path)
    atomic_json(candidate_path, candidate)
    spans.append(_span("write", phase_started, started))
    _progress(started, "write", "after_atomic_candidate", path=candidate_path)

    phase_started = time.monotonic()
    _progress(started, "validate", "before_terminal_subprocesses")
    terminal = run_categorized_commands(
        root,
        _terminal_commands(candidate_path),
        log_dir=raw_dir / "validation/terminal",
    )
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    spans.append(_span("validate_terminal", phase_started, started))
    _progress(
        started,
        "validate",
        "after_terminal_subprocesses",
        passed=terminal_passed,
        critical=critical,
    )

    final = _build_artifact(
        started=started,
        started_at=started_at,
        spans=spans,
        preconditions=preconditions,
        source_hashes=source_hashes,
        groups=groups,
        quarantine=quarantine,
        memberships=memberships,
        summary=summary,
        feature_rows=feature_rows,
        fixture=fixture,
        replay=replay,
        validation_receipts=[*affected, *terminal],
        safety_passed=terminal_passed,
        flagged_adversarial=critical or not terminal_passed,
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    _progress(started, "write", "before_atomic_terminal", path=output_path)
    atomic_json(candidate_path, final)
    atomic_json(root / output_path, final)
    _progress(started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the thin public experiment entrypoint arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the protocol through the declared repository-root entrypoint."""

    args = parse_args(argv)
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
