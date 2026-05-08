"""Exp 1568 FR-11 v14 retained-policy mode-collapse audit.

Spec: REQ-LEARN-1568, SCENARIO-LEARN-1568, SCENARIO-LEARN-1569.

This module audits the v14 policy retentions produced by Exp 1555.  It does
not rerun RL or mutate weights.  It turns checked-in retained-policy evidence
into the four requested anti-exploration predictors and records unavailable
evidence explicitly, because the predecessor artifacts do not always contain a
pre-RL baseline, fresh k=8 reward groups, or adversarial OOD accuracy rows.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILE = "experiment_1568_fr11_v14_retained_mode_collapse_audit.json"
DEFAULT_OUTPUT_PATH = Path("results") / OUTPUT_FILE
DEFAULT_EXP1555_ARTIFACT_PATH = Path(
    "results/experiment_1555_fr11_positive_utility_or_retire_v14.json"
)
DEFAULT_SKILL_GRAPH_PATH = Path("results/fr11_positive_utility_skill_graph_1555.json")
DEFAULT_EXP1539_ARTIFACT_PATH = Path(
    "results/experiment_1539_fr11_external_feedback_skill_promotion_v13.json"
)
DEFAULT_REPAIR_ARTIFACT_PATH = Path("results/experiment_1552_residual_drift_repair_policy_v1.json")
DEFAULT_REPAIR_MANIFEST_PATH = Path("results/residual_drift_repair_policy_1552.jsonl")

SPEC_REFS: tuple[str, ...] = (
    "REQ-LEARN-1568",
    "SCENARIO-LEARN-1568",
    "SCENARIO-LEARN-1569",
)

RETAINED_POLICY_TARGET_COUNT = 5
ENTROPY_DROP_THRESHOLD_NATS = 0.5
BOILERPLATE_THRESHOLD = 0.30
REWARD_GROUP_SIZE = 8
VARIANCE_COLLAPSE_EPSILON = 1e-12

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "mode_collapse_audit_complete",
    "retained_policies_audited_count",
    "mode_collapse_confirmed_count",
    "reversal_recommended_count",
    "honest_verdict",
    "retained_policy_target_count",
    "retained_policy_target_met",
    "mode_collapse_confirmed_percent",
    "retention_reversal_recommended_policy_ids",
    "retained_policy_audits",
    "source_limitations",
    "spec",
)

REQUIRED_POLICY_AUDIT_FIELDS: tuple[str, ...] = (
    "policy_id",
    "source",
    "predictors",
    "confirmed_predictors",
    "confirmed_predictor_count",
    "mode_collapse_confirmed",
)

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[^\s]")


def audit_retained_policy(
    *,
    policy_id: str,
    source: str,
    generated_repairs: Sequence[str] = (),
    baseline_repairs: Sequence[str] = (),
    training_corpus: Sequence[str] = (),
    reward_groups: Sequence[Sequence[float]] = (),
    baseline_ood_accuracy: float | None = None,
    post_ood_accuracy: float | None = None,
    evidence_basis: str = "checked_in_artifact",
) -> JsonDict:
    """REQ-LEARN-1568-3/4: evaluate one retained policy's predictors."""

    predictors = {
        "token_entropy_drop": _token_entropy_predictor(generated_repairs, baseline_repairs),
        "boilerplate_fraction": _boilerplate_predictor(generated_repairs, training_corpus),
        "reward_variance_collapse": _reward_variance_predictor(reward_groups),
        "ood_adversarial_accuracy_regression": _ood_accuracy_predictor(
            baseline_ood_accuracy,
            post_ood_accuracy,
        ),
    }
    confirmed = [
        name
        for name, predictor in predictors.items()
        if predictor["available"] and predictor["confirmed"]
    ]
    return {
        "policy_id": str(policy_id),
        "source": str(source),
        "evidence_basis": evidence_basis,
        "predictors": predictors,
        "confirmed_predictors": confirmed,
        "confirmed_predictor_count": len(confirmed),
        "mode_collapse_confirmed": len(confirmed) >= 2,
    }


def build_artifact(
    *,
    retained_policy_audits: Sequence[Mapping[str, Any]],
    source_limitations: Sequence[str],
) -> JsonDict:
    """REQ-LEARN-1568-1/5/6/7: build the terminal conductor artifact."""

    audits = [dict(audit) for audit in retained_policy_audits]
    audited_count = len(audits)
    confirmed_policy_ids = [
        str(audit["policy_id"])
        for audit in audits
        if audit.get("mode_collapse_confirmed") is True
    ]
    confirmed_count = len(confirmed_policy_ids)
    confirmed_percent = _rate(confirmed_count, audited_count)
    reversal_ids = confirmed_policy_ids if confirmed_count > 0 and confirmed_percent >= 0.5 else []
    artifact = {
        "status": "complete",
        "mode_collapse_audit_complete": True,
        "retained_policies_audited_count": audited_count,
        "mode_collapse_confirmed_count": confirmed_count,
        "reversal_recommended_count": len(reversal_ids),
        "honest_verdict": _honest_verdict(
            audited_count=audited_count,
            confirmed_count=confirmed_count,
            reversal_count=len(reversal_ids),
        ),
        "retained_policy_target_count": RETAINED_POLICY_TARGET_COUNT,
        "retained_policy_target_met": audited_count >= RETAINED_POLICY_TARGET_COUNT,
        "mode_collapse_confirmed_percent": confirmed_percent,
        "retention_reversal_recommended_policy_ids": reversal_ids,
        "retained_policy_audits": audits,
        "source_limitations": list(source_limitations),
        "spec": list(SPEC_REFS),
    }
    validate_artifact(artifact)
    return artifact


def run_experiment(
    *,
    project_root: Path | str | None = None,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    exp1555_artifact_path: Path | str = DEFAULT_EXP1555_ARTIFACT_PATH,
    skill_graph_path: Path | str = DEFAULT_SKILL_GRAPH_PATH,
    exp1539_artifact_path: Path | str = DEFAULT_EXP1539_ARTIFACT_PATH,
    repair_artifact_path: Path | str = DEFAULT_REPAIR_ARTIFACT_PATH,
    repair_manifest_path: Path | str = DEFAULT_REPAIR_MANIFEST_PATH,
) -> JsonDict:
    """Run the Exp 1568 audit from checked-in predecessor artifacts."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    output = _resolve_under_root(root, Path(output_path))
    exp1555_path = _resolve_under_root(root, Path(exp1555_artifact_path))
    graph_path = _resolve_under_root(root, Path(skill_graph_path))
    exp1539_path = _resolve_under_root(root, Path(exp1539_artifact_path))
    repair_path = _resolve_under_root(root, Path(repair_artifact_path))
    manifest_path = _resolve_under_root(root, Path(repair_manifest_path))

    exp1555, limitations = _load_json_or_limitation(exp1555_path)
    skill_graph, graph_limitations = _load_json_or_limitation(graph_path)
    exp1539, exp1539_limitations = _load_json_or_limitation(exp1539_path)
    repair_artifact, repair_limitations = _load_json_or_limitation(repair_path)
    repair_rows, row_limitations = _read_jsonl_or_limitation(manifest_path)
    policies = snapshot_retained_policies(exp1555, skill_graph)
    audits = [
        audit_retained_policy(**_measurement_kwargs(policy, exp1539, repair_artifact, repair_rows))
        for policy in policies
    ]
    all_limitations = [
        *limitations,
        *graph_limitations,
        *exp1539_limitations,
        *repair_limitations,
        *row_limitations,
        *_snapshot_limitations(policies),
        *_measurement_limitations(audits),
    ]
    artifact = build_artifact(
        retained_policy_audits=audits,
        source_limitations=_dedupe(all_limitations),
    )
    _write_json(output, artifact)
    return artifact


def snapshot_retained_policies(
    exp1555_artifact: Mapping[str, Any],
    skill_graph: Mapping[str, Any],
) -> list[JsonDict]:
    """REQ-LEARN-1568-2: snapshot v14 policies retained by Exp 1555."""

    retained: list[JsonDict] = []
    seen: set[str] = set()
    for node in skill_graph.get("nodes", []):
        node_map = _mapping(node)
        update_id = str(node_map.get("update_id") or _nested(node_map, "promotion_decision").get("update_id") or "")
        if not update_id or update_id in seen:
            continue
        if _is_retained_node(node_map):
            retained.append(
                {
                    "policy_id": update_id,
                    "source": str(node_map.get("source") or "skill_graph"),
                    "node_id": str(node_map.get("node_id") or ""),
                }
            )
            seen.add(update_id)
    for update in exp1555_artifact.get("skill_updates_promoted", []):
        update_map = _mapping(update)
        update_id = str(update_map.get("update_id") or "")
        if update_id and update_id not in seen:
            retained.append(
                {
                    "policy_id": update_id,
                    "source": str(update_map.get("source") or "exp1555"),
                    "node_id": "",
                }
            )
            seen.add(update_id)
    return retained


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required conductor-facing fields and derived counts."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] != "complete":
        raise AssertionError("Exp 1568 terminal artifact must be complete")
    if not str(artifact["honest_verdict"]).startswith("complete:"):
        raise AssertionError("honest_verdict must be prefixed with complete:")
    audits = list(artifact["retained_policy_audits"])
    if int(artifact["retained_policies_audited_count"]) != len(audits):
        raise AssertionError("retained_policies_audited_count must match retained_policy_audits")
    for audit in audits:
        missing_policy_fields = [
            field for field in REQUIRED_POLICY_AUDIT_FIELDS if field not in audit
        ]
        if missing_policy_fields:
            raise AssertionError(f"missing policy audit fields: {missing_policy_fields}")
    confirmed = [audit for audit in audits if audit.get("mode_collapse_confirmed") is True]
    if int(artifact["mode_collapse_confirmed_count"]) != len(confirmed):
        raise AssertionError("mode_collapse_confirmed_count must match policy audits")
    expected_reversal_count = len(artifact["retention_reversal_recommended_policy_ids"])
    if int(artifact["reversal_recommended_count"]) != expected_reversal_count:
        raise AssertionError("reversal_recommended_count must match reversal policy ids")


def _measurement_kwargs(
    policy: Mapping[str, Any],
    exp1539_artifact: Mapping[str, Any],
    repair_artifact: Mapping[str, Any],
    repair_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    policy_id = str(policy["policy_id"])
    if policy_id == "policy:residual_drift_repair:1552":
        return _residual_repair_measurements(policy, repair_artifact, repair_rows)
    candidate = _find_exp1539_candidate(exp1539_artifact, policy_id)
    outputs = _nested(candidate, "model_outputs")
    return {
        "policy_id": policy_id,
        "source": str(policy.get("source") or "exp1539_external_feedback"),
        "generated_repairs": _present_strings([outputs.get("promoted_excerpt")]),
        "baseline_repairs": _present_strings([outputs.get("baseline_excerpt")]),
        "training_corpus": _present_strings([outputs.get("baseline_excerpt")]),
        "reward_groups": [[float(candidate.get("verifier_reward", 0.0))]],
        "evidence_basis": "exp1539_model_output_excerpts",
    }


def _residual_repair_measurements(
    policy: Mapping[str, Any],
    repair_artifact: Mapping[str, Any],
    repair_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    accepted_rows = [
        _mapping(row)
        for row in repair_rows
        if row.get("row_type") == "residual_drift_repair_case"
        and row.get("accepted") is True
        and row.get("replay_passed") is True
        and row.get("false_accept") is not True
    ]
    generated = _present_strings(
        _nested(row, "proposal").get("model_proposal_excerpt") for row in accepted_rows
    )
    holdout_start = min(max(1, len(generated) // 5), 8) if len(generated) > 1 else 0
    training_corpus = _present_strings(
        [
            _nested(repair_artifact, "model_probe").get("proposal_output_excerpt"),
            *generated[:holdout_start],
        ]
    )
    holdout_generated = generated[holdout_start:] or generated
    rewards = [1.0 if row.get("replay_passed") is True else 0.0 for row in accepted_rows]
    return {
        "policy_id": str(policy["policy_id"]),
        "source": str(policy.get("source") or "exp1552_residual_drift_repair"),
        "generated_repairs": holdout_generated,
        "training_corpus": training_corpus,
        "reward_groups": _chunk(rewards[holdout_start:] or rewards, REWARD_GROUP_SIZE),
        "evidence_basis": "exp1552_checked_in_replay_proxy",
    }


def _token_entropy_predictor(
    generated_repairs: Sequence[str],
    baseline_repairs: Sequence[str],
) -> JsonDict:
    if not generated_repairs or not baseline_repairs:
        return _unavailable("pre_rl_or_generated_repair_text_unavailable")
    generated_distribution = [_token_entropy(text) for text in generated_repairs]
    baseline_distribution = [_token_entropy(text) for text in baseline_repairs]
    generated_mean = _mean(generated_distribution)
    baseline_mean = _mean(baseline_distribution)
    drop = round(baseline_mean - generated_mean, 6)
    return {
        "available": True,
        "confirmed": drop >= ENTROPY_DROP_THRESHOLD_NATS,
        "drop_nats_per_token": drop,
        "threshold_nats_per_token": ENTROPY_DROP_THRESHOLD_NATS,
        "generated_distribution_nats_per_token": generated_distribution,
        "baseline_distribution_nats_per_token": baseline_distribution,
        "generated_mean_nats_per_token": generated_mean,
        "baseline_mean_nats_per_token": baseline_mean,
    }


def _boilerplate_predictor(
    generated_repairs: Sequence[str],
    training_corpus: Sequence[str],
) -> JsonDict:
    if not generated_repairs or not training_corpus:
        return _unavailable("training_corpus_or_generated_repair_text_unavailable")
    fraction = _ngram_overlap_fraction(
        _tokens_from_texts(generated_repairs),
        _tokens_from_texts(training_corpus),
        n=3,
    )
    return {
        "available": True,
        "confirmed": fraction >= BOILERPLATE_THRESHOLD,
        "boilerplate_fraction": fraction,
        "threshold": BOILERPLATE_THRESHOLD,
        "ngram_size": 3,
    }


def _reward_variance_predictor(reward_groups: Sequence[Sequence[float]]) -> JsonDict:
    full_groups = [list(group[:REWARD_GROUP_SIZE]) for group in reward_groups if len(group) >= REWARD_GROUP_SIZE]
    if not full_groups:
        return _unavailable("fresh_k8_reward_groups_unavailable")
    variances = [_variance(group) for group in full_groups]
    collapsed = all(variance <= VARIANCE_COLLAPSE_EPSILON for variance in variances)
    return {
        "available": True,
        "confirmed": collapsed,
        "group_size": REWARD_GROUP_SIZE,
        "group_variances": variances,
        "collapse_epsilon": VARIANCE_COLLAPSE_EPSILON,
        "single_mode": collapsed,
    }


def _ood_accuracy_predictor(
    baseline_ood_accuracy: float | None,
    post_ood_accuracy: float | None,
) -> JsonDict:
    if baseline_ood_accuracy is None or post_ood_accuracy is None:
        return _unavailable("ood_adversarial_accuracy_baseline_or_post_unavailable")
    delta = round(float(post_ood_accuracy) - float(baseline_ood_accuracy), 6)
    return {
        "available": True,
        "confirmed": delta < 0.0,
        "baseline_accuracy": float(baseline_ood_accuracy),
        "post_accuracy": float(post_ood_accuracy),
        "delta": delta,
    }


def _ngram_overlap_fraction(generated_tokens: Sequence[str], training_tokens: Sequence[str], *, n: int) -> float:
    generated_ngrams = _ngrams(generated_tokens, n)
    if not generated_ngrams:
        return 0.0
    training_ngrams = set(_ngrams(training_tokens, n))
    matched = sum(1 for ngram in generated_ngrams if ngram in training_ngrams)
    return round(matched / len(generated_ngrams), 6)


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index : index + n]) for index in range(0, max(0, len(tokens) - n + 1))]


def _token_entropy(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    total = len(tokens)
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return round(entropy, 6)


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def _tokens_from_texts(texts: Sequence[str]) -> list[str]:
    tokens: list[str] = []
    for text in texts:
        tokens.extend(_tokenize(text))
    return tokens


def _variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return round(sum((value - mean) ** 2 for value in values) / len(values), 12)


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else round(sum(values) / len(values), 6)


def _unavailable(reason: str) -> JsonDict:
    return {"available": False, "confirmed": False, "reason": reason}


def _is_retained_node(node: Mapping[str, Any]) -> bool:
    decision = node.get("promotion_decision")
    if isinstance(decision, str):
        return "reject" not in decision and "retire" not in decision
    return bool(decision)


def _find_exp1539_candidate(artifact: Mapping[str, Any], policy_id: str) -> JsonDict:
    for candidate in artifact.get("candidate_updates", []):
        candidate_map = _mapping(candidate)
        if str(candidate_map.get("policy_update_id")) == policy_id:
            return candidate_map
    return {}


def _measurement_limitations(audits: Sequence[Mapping[str, Any]]) -> list[str]:
    limitations = []
    for audit in audits:
        policy_id = str(audit["policy_id"])
        for name, predictor in audit["predictors"].items():
            if predictor.get("available") is not True:
                limitations.append(f"{name}_unavailable:{policy_id}:{predictor.get('reason')}")
    return limitations


def _snapshot_limitations(policies: Sequence[Mapping[str, Any]]) -> list[str]:
    count = len(policies)
    limitations = []
    if count < RETAINED_POLICY_TARGET_COUNT:
        limitations.append(f"retained_policy_target_not_met:{count}_of_{RETAINED_POLICY_TARGET_COUNT}")
    if count == 0:
        limitations.append("no_exp1555_retained_policies_found")
    return limitations


def _honest_verdict(*, audited_count: int, confirmed_count: int, reversal_count: int) -> str:
    retention_word = "retention" if reversal_count == 1 else "retentions"
    return (
        "complete: "
        f"audited {audited_count} retained v14 policies; "
        f"{confirmed_count} showed 2+ mode-collapse predictors; "
        f"{reversal_count} {retention_word} flagged for next-milestone reversal"
    )


def _load_json_or_limitation(path: Path) -> tuple[JsonDict, list[str]]:
    if not path.exists():
        return {}, [f"missing:{_display_path(path)}"]
    return _load_json(path), []


def _read_jsonl_or_limitation(path: Path) -> tuple[list[JsonDict], list[str]]:
    if not path.exists():
        return [], [f"missing:{_display_path(path)}"]
    return _read_jsonl(path), []


def _load_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _nested(mapping: Mapping[str, Any], key: str) -> JsonDict:
    return _mapping(mapping.get(key))


def _present_strings(values: Sequence[Any]) -> list[str]:
    return [str(value) for value in values if isinstance(value, str) and value.strip()]


def _chunk(values: Sequence[float], size: int) -> list[list[float]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(numerator / denominator, 6)


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _resolve_under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(path: Path | str, *, project_root: Path | str = REPO_ROOT) -> str:
    target = Path(path)
    try:
        return target.resolve().relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return target.as_posix()


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    artifact = run_experiment()
    print(
        "[exp1568] "
        f"audited={artifact['retained_policies_audited_count']} "
        f"confirmed={artifact['mode_collapse_confirmed_count']} "
        f"reversal={artifact['reversal_recommended_count']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())


__all__ = [
    "OUTPUT_FILE",
    "REQUIRED_ARTIFACT_FIELDS",
    "REQUIRED_POLICY_AUDIT_FIELDS",
    "audit_retained_policy",
    "build_artifact",
    "run_experiment",
    "snapshot_retained_policies",
    "validate_artifact",
]
