"""Exp 1581 FR-11 v15 lambda-GRPO retention-reversal gate.

Spec: REQ-LEARN-1581, SCENARIO-LEARN-1581, SCENARIO-LEARN-1582.

Exp 1568 found one retained v14 FR-11 policy with two mode-collapse
predictors.  This module reruns those predictor gates on held-out checked-in
repair rows, records the lambda-GRPO correction path, and reverses the retained
policy only when fresh replay still confirms collapse without soundness
mistakes.  It never mutates model weights.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.reporting.fr11_v14_retained_mode_collapse_audit import audit_retained_policy

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILE = "experiment_1581_fr11_v15_lambda_grpo_retention_reversal.json"
DEFAULT_OUTPUT_PATH = Path("results") / OUTPUT_FILE
DEFAULT_EXP1568_ARTIFACT_PATH = Path(
    "results/experiment_1568_fr11_v14_retained_mode_collapse_audit.json"
)
DEFAULT_REPAIR_ARTIFACT_PATH = Path("results/experiment_1552_residual_drift_repair_policy_v1.json")
DEFAULT_REPAIR_MANIFEST_PATH = Path("results/residual_drift_repair_policy_1552.jsonl")

FLAGGED_POLICY_ID = "policy:residual_drift_repair:1552"
REWARD_GROUP_SIZE = 8
TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[^\s]")

SPEC_REFS: tuple[str, ...] = (
    "REQ-LEARN-1581",
    "SCENARIO-LEARN-1581",
    "SCENARIO-LEARN-1582",
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "continuous_self_learning_task",
    "flagged_policy_replayed",
    "retention_reversal_applied",
    "lambda_grpo_patch_implemented",
    "lambda_grpo_simulated_only",
    "soundness_mistakes",
    "entropy_preservation_rate",
    "boilerplate_fraction_delta",
    "fr11_v15_decision_ready",
    "honest_verdict",
)


def write_in_progress_artifact(output_path: Path | str = DEFAULT_OUTPUT_PATH) -> JsonDict:
    """REQ-LEARN-1581-1: write the durable in-progress artifact."""

    artifact: JsonDict = {
        "status": "in_progress",
        "continuous_self_learning_task": True,
        "flagged_policy_replayed": False,
        "retention_reversal_applied": False,
        "lambda_grpo_patch_implemented": False,
        "lambda_grpo_simulated_only": False,
        "soundness_mistakes": 0,
        "entropy_preservation_rate": 0.0,
        "boilerplate_fraction_delta": 0.0,
        "fr11_v15_decision_ready": False,
        "honest_verdict": "in_progress",
    }
    validate_artifact(artifact)
    _write_json(Path(output_path), artifact)
    return artifact


def replay_flagged_policy(
    *,
    exp1568_artifact: Mapping[str, Any],
    repair_artifact: Mapping[str, Any],
    repair_rows: Sequence[Mapping[str, Any]],
    lambda_grpo_patch_available: bool,
) -> JsonDict:
    """REQ-LEARN-1581-2/3/4: replay the flagged retention and correction path."""

    recommended = FLAGGED_POLICY_ID in {
        str(policy_id)
        for policy_id in exp1568_artifact.get("retention_reversal_recommended_policy_ids", [])
    }
    patch_implemented = bool(lambda_grpo_patch_available)
    simulated_only = not patch_implemented
    if not recommended:
        return {
            "flagged_policy_id": FLAGGED_POLICY_ID,
            "flagged_policy_replayed": False,
            "blockers": [f"not recommended by exp1568: {FLAGGED_POLICY_ID}"],
            "lambda_grpo_patch_implemented": patch_implemented,
            "lambda_grpo_simulated_only": simulated_only,
            "implementation_deferred": simulated_only,
            "soundness_mistakes": 0,
            "entropy_preservation_rate": 0.0,
            "boilerplate_fraction_delta": 0.0,
            "ood_accuracy_proxy": 0.0,
            "replay_mode_collapse_confirmed": False,
            "replay_confirmed_predictors": [],
            "replay_confirmed_predictor_count": 0,
        }

    accepted_rows = [
        _mapping(row) for row in repair_rows if _is_repair_case(row) and row.get("accepted") is True
    ]
    training_rows, heldout_rows = _heldout_split(accepted_rows)
    generated_repairs = _present_strings(_proposal_excerpt(row) for row in heldout_rows)
    training_corpus = _training_corpus(repair_artifact, training_rows)
    rewards = [_row_reward(row) for row in heldout_rows]
    replay_audit = audit_retained_policy(
        policy_id=FLAGGED_POLICY_ID,
        source="exp1552_residual_drift_repair",
        generated_repairs=generated_repairs,
        training_corpus=training_corpus,
        reward_groups=_chunk(rewards, REWARD_GROUP_SIZE),
        evidence_basis="exp1581_heldout_replay",
    )
    simulator = build_lambda_grpo_simulator_evidence(
        heldout_rows=heldout_rows,
        training_corpus=training_corpus,
    )
    confirmed_predictors = list(replay_audit["confirmed_predictors"])
    replay_boilerplate = _predictor_float(
        replay_audit,
        "boilerplate_fraction",
        "boilerplate_fraction",
    )
    return {
        "flagged_policy_id": FLAGGED_POLICY_ID,
        "flagged_policy_replayed": True,
        "blockers": [],
        "heldout_replay_case_count": len(heldout_rows),
        "accepted_replay_case_count": len(accepted_rows),
        "lambda_grpo_patch_implemented": patch_implemented,
        "lambda_grpo_simulated_only": simulated_only,
        "implementation_deferred": simulated_only,
        "no_model_weight_mutation": True,
        "soundness_mistakes": _soundness_mistakes(heldout_rows),
        "entropy_preservation_rate": _text_distribution_entropy_rate(generated_repairs),
        "replay_boilerplate_fraction": replay_boilerplate,
        "boilerplate_fraction_delta": simulator["boilerplate_fraction_delta"],
        "ood_accuracy_proxy": _ood_accuracy_proxy(training_rows, heldout_rows),
        "replay_confirmed_predictors": confirmed_predictors,
        "replay_confirmed_predictor_count": len(confirmed_predictors),
        "replay_mode_collapse_confirmed": len(confirmed_predictors) >= 2,
        "replay_policy_audit": replay_audit,
        "lambda_grpo_simulator": simulator,
    }


def simulate_lambda_grpo_weights(
    cases: Sequence[Mapping[str, Any]],
    *,
    entropy_lambda: float = 0.35,
    boilerplate_lambda: float = 0.45,
    ood_lambda: float = 0.20,
    soundness_penalty: float = 1.0,
) -> list[JsonDict]:
    """REQ-LEARN-1581-4: normalize lambda-GRPO scores across equal rewards."""

    scored: list[JsonDict] = []
    for case in cases:
        reward = float(case.get("reward", 0.0))
        entropy = float(case.get("entropy_preservation", 0.0))
        boilerplate = float(case.get("boilerplate_fraction", 0.0))
        ood = float(case.get("ood_accuracy_proxy", 0.0))
        soundness = float(case.get("soundness_mistake", 0.0))
        score = max(
            0.0,
            reward
            + entropy_lambda * entropy
            + ood_lambda * ood
            - boilerplate_lambda * boilerplate
            - soundness_penalty * soundness,
        )
        scored.append({**dict(case), "corrected_score": round(score, 6)})
    total = sum(float(case["corrected_score"]) for case in scored)
    if total <= 0.0:  # pragma: no cover - defensive fallback for all-penalty fixtures.
        weight = 0.0 if not scored else round(1.0 / len(scored), 6)
        return [{**case, "normalized_weight": weight} for case in scored]
    return [
        {
            **case,
            "normalized_weight": round(float(case["corrected_score"]) / total, 6),
        }
        for case in scored
    ]


def build_lambda_grpo_simulator_evidence(
    *,
    heldout_rows: Sequence[Mapping[str, Any]],
    training_corpus: Sequence[str],
) -> JsonDict:
    """Build deterministic simulator evidence for the corrected weighting path."""

    options = _simulator_options(heldout_rows, training_corpus)
    normalized = simulate_lambda_grpo_weights(options)
    collapsed_options = [
        option for option in normalized if option["option_type"] == "collapsed_model_excerpt"
    ]
    before = _mean_float(option["boilerplate_fraction"] for option in collapsed_options)
    after = round(
        sum(
            float(option["normalized_weight"]) * float(option["boilerplate_fraction"])
            for option in normalized
        ),
        6,
    )
    collapsed_weight = round(
        sum(float(option["normalized_weight"]) for option in collapsed_options),
        6,
    )
    return {
        "simulator_ready": bool(normalized),
        "option_count": len(normalized),
        "collapsed_weight_total": collapsed_weight,
        "localized_weight_total": round(1.0 - collapsed_weight, 6) if normalized else 0.0,
        "boilerplate_fraction_before": before,
        "weighted_boilerplate_fraction_after": after,
        "boilerplate_fraction_delta": round(after - before, 6),
        "corrected_weight_examples": normalized[:6],
    }


def build_artifact(*, replay: Mapping[str, Any]) -> JsonDict:
    """REQ-LEARN-1581-5/6: build the terminal retention-reversal artifact."""

    replayed = replay.get("flagged_policy_replayed") is True
    patch_implemented = replay.get("lambda_grpo_patch_implemented") is True
    simulated_only = replay.get("lambda_grpo_simulated_only") is True
    correction_ready = patch_implemented or simulated_only
    soundness_mistakes = int(replay.get("soundness_mistakes", 0))
    collapse_confirmed = replay.get("replay_mode_collapse_confirmed") is True
    no_mutation = replay.get("no_model_weight_mutation", True) is True
    reversal = bool(
        replayed
        and correction_ready
        and no_mutation
        and soundness_mistakes == 0
        and collapse_confirmed
    )
    decision_ready = bool(replayed and correction_ready)
    status = "complete" if decision_ready else "blocked"
    artifact: JsonDict = {
        "status": status,
        "continuous_self_learning_task": True,
        "flagged_policy_id": str(replay.get("flagged_policy_id", FLAGGED_POLICY_ID)),
        "flagged_policy_replayed": replayed,
        "retention_reversal_applied": reversal,
        "lambda_grpo_patch_implemented": patch_implemented,
        "lambda_grpo_simulated_only": simulated_only,
        "implementation_deferred": simulated_only,
        "no_model_weight_mutation": no_mutation,
        "soundness_mistakes": soundness_mistakes,
        "entropy_preservation_rate": float(replay.get("entropy_preservation_rate", 0.0)),
        "replay_boilerplate_fraction": float(replay.get("replay_boilerplate_fraction", 0.0)),
        "boilerplate_fraction_delta": float(replay.get("boilerplate_fraction_delta", 0.0)),
        "ood_accuracy_proxy": float(replay.get("ood_accuracy_proxy", 0.0)),
        "fr11_v15_decision_ready": decision_ready,
        "replay_mode_collapse_confirmed": collapse_confirmed,
        "replay_confirmed_predictors": list(replay.get("replay_confirmed_predictors", [])),
        "replay_confirmed_predictor_count": int(replay.get("replay_confirmed_predictor_count", 0)),
        "heldout_replay_case_count": int(replay.get("heldout_replay_case_count", 0)),
        "accepted_replay_case_count": int(replay.get("accepted_replay_case_count", 0)),
        "lambda_grpo_simulator": dict(replay.get("lambda_grpo_simulator", {})),
        "retention_reversal_audit_note": _audit_note(reversal),
        "blockers": list(replay.get("blockers", [])),
        "spec": list(SPEC_REFS),
        "honest_verdict": _honest_verdict(
            status=status,
            replayed=replayed,
            reversal=reversal,
            collapse_confirmed=collapse_confirmed,
            blockers=list(replay.get("blockers", [])),
        ),
    }
    if "replay_policy_audit" in replay:
        artifact["replay_policy_audit"] = dict(replay["replay_policy_audit"])
    validate_artifact(artifact)
    return artifact


def run_experiment(
    *,
    project_root: Path | str | None = None,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    exp1568_artifact_path: Path | str = DEFAULT_EXP1568_ARTIFACT_PATH,
    repair_artifact_path: Path | str = DEFAULT_REPAIR_ARTIFACT_PATH,
    repair_manifest_path: Path | str = DEFAULT_REPAIR_MANIFEST_PATH,
    lambda_grpo_patch_available: bool | None = None,
) -> JsonDict:
    """Run Exp 1581 from checked-in predecessor artifacts."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    output = _resolve_under_root(root, Path(output_path))
    exp1568_path = _resolve_under_root(root, Path(exp1568_artifact_path))
    repair_path = _resolve_under_root(root, Path(repair_artifact_path))
    manifest_path = _resolve_under_root(root, Path(repair_manifest_path))

    write_in_progress_artifact(output)
    exp1568_artifact = _load_json(exp1568_path)
    repair_artifact = _load_json(repair_path)
    repair_rows = _read_jsonl(manifest_path)
    replay = replay_flagged_policy(
        exp1568_artifact=exp1568_artifact,
        repair_artifact=repair_artifact,
        repair_rows=repair_rows,
        lambda_grpo_patch_available=(
            _lambda_grpo_patch_available()
            if lambda_grpo_patch_available is None
            else bool(lambda_grpo_patch_available)
        ),
    )
    artifact = build_artifact(replay=replay)
    _write_json(output, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required Exp 1581 artifact fields and derived safety gates."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:
        raise AssertionError("status must be in_progress, complete, or blocked")
    if artifact["continuous_self_learning_task"] is not True:
        raise AssertionError("continuous_self_learning_task must be true")
    if (
        artifact["lambda_grpo_patch_implemented"] is True
        and artifact["lambda_grpo_simulated_only"] is True
    ):
        raise AssertionError("lambda_grpo_patch_implemented and simulated_only cannot both be true")
    if artifact["retention_reversal_applied"] is True:
        if artifact["flagged_policy_replayed"] is not True:
            raise AssertionError("reversal requires flagged_policy_replayed")
        if int(artifact["soundness_mistakes"]) != 0:
            raise AssertionError("reversal requires zero soundness mistakes")
        if artifact.get("no_model_weight_mutation") is not True:
            raise AssertionError("reversal requires no model weight mutation")
        if int(artifact.get("replay_confirmed_predictor_count", 0)) < 2:
            raise AssertionError("reversal requires at least two replay predictors")


def _simulator_options(
    heldout_rows: Sequence[Mapping[str, Any]],
    training_corpus: Sequence[str],
) -> list[JsonDict]:
    raw_options: list[JsonDict] = []
    for row in heldout_rows:
        collapsed_text = _proposal_excerpt(row)
        localized_text = _localized_replacement_text(row)
        for option_type, text in (
            ("collapsed_model_excerpt", collapsed_text),
            ("localized_repair_view", localized_text),
        ):
            raw_options.append(
                {
                    "case_id": f"{row.get('case_id')}:{option_type}",
                    "option_type": option_type,
                    "reward": _row_reward(row),
                    "boilerplate_fraction": _boilerplate_fraction([text], training_corpus),
                    "ood_accuracy_proxy": 1.0 if _row_reward(row) == 1.0 else 0.0,
                    "soundness_mistake": 1 if _row_soundness_mistake(row) else 0,
                    "_text_entropy": _token_entropy(text),
                }
            )
    max_entropy = max((float(option["_text_entropy"]) for option in raw_options), default=0.0)
    options: list[JsonDict] = []
    for option in raw_options:
        entropy = (
            0.0 if max_entropy <= 0.0 else round(float(option["_text_entropy"]) / max_entropy, 6)
        )
        cleaned = {key: value for key, value in option.items() if key != "_text_entropy"}
        options.append({**cleaned, "entropy_preservation": entropy})
    return options


def _is_repair_case(row: Mapping[str, Any]) -> bool:
    return row.get("row_type") == "residual_drift_repair_case"


def _heldout_split(rows: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], list[JsonDict]]:
    copied = [_mapping(row) for row in rows]
    holdout_start = min(max(1, len(copied) // 5), 8) if len(copied) > 1 else 0
    return copied[:holdout_start], copied[holdout_start:] or copied


def _training_corpus(
    repair_artifact: Mapping[str, Any], training_rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    model_probe = _mapping(repair_artifact.get("model_probe"))
    return _present_strings(
        [
            model_probe.get("proposal_output_excerpt"),
            *(_proposal_excerpt(row) for row in training_rows),
        ]
    )


def _proposal_excerpt(row: Mapping[str, Any]) -> str:
    return str(_mapping(row.get("proposal")).get("model_proposal_excerpt") or "")


def _localized_replacement_text(row: Mapping[str, Any]) -> str:
    proposal = _mapping(row.get("proposal"))
    payload = {
        "case_id": row.get("case_id"),
        "localized_span": proposal.get("localized_span"),
        "replacement": proposal.get("replacement"),
        "source_domain": row.get("source_domain"),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _row_reward(row: Mapping[str, Any]) -> float:
    return 1.0 if row.get("replay_passed") is True and not _row_soundness_mistake(row) else 0.0


def _row_soundness_mistake(row: Mapping[str, Any]) -> bool:
    replay = _mapping(row.get("replay"))
    return (
        row.get("false_accept") is True
        or replay.get("false_accept") is True
        or row.get("replay_passed") is not True
    )


def _soundness_mistakes(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for row in rows if _row_soundness_mistake(row))


def _text_distribution_entropy_rate(texts: Sequence[str]) -> float:
    normalized = [_normalize_text(text) for text in texts if text.strip()]
    if not normalized:
        return 0.0
    if len(normalized) == 1:
        return 1.0
    counts: dict[str, int] = {}
    for text in normalized:
        counts[text] = counts.get(text, 0) + 1
    entropy = -sum(
        (count / len(normalized)) * math.log(count / len(normalized)) for count in counts.values()
    )
    rate = entropy / math.log(len(normalized))
    return 0.0 if abs(rate) < 1e-12 else round(rate, 6)


def _boilerplate_fraction(texts: Sequence[str], training_corpus: Sequence[str]) -> float:
    generated_tokens = _tokens_from_texts(texts)
    generated_ngrams = _ngrams(generated_tokens, 3)
    if not generated_ngrams:
        return 0.0
    training_ngrams = set(_ngrams(_tokens_from_texts(training_corpus), 3))
    matched = sum(1 for ngram in generated_ngrams if ngram in training_ngrams)
    return round(matched / len(generated_ngrams), 6)


def _ood_accuracy_proxy(
    training_rows: Sequence[Mapping[str, Any]], heldout_rows: Sequence[Mapping[str, Any]]
) -> float:
    training_domains = {str(row.get("source_domain")) for row in training_rows}
    ood_rows = [
        row for row in heldout_rows if str(row.get("source_domain")) not in training_domains
    ]
    target_rows = ood_rows or list(heldout_rows)
    return _rate(sum(1 for row in target_rows if _row_reward(row) == 1.0), len(target_rows))


def _predictor_float(audit: Mapping[str, Any], predictor_name: str, field_name: str) -> float:
    predictor = _mapping(_mapping(audit.get("predictors")).get(predictor_name))
    return float(predictor.get(field_name, 0.0))


def _token_entropy(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    total = len(tokens)
    return round(-sum((count / total) * math.log(count / total) for count in counts.values()), 6)


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def _tokens_from_texts(texts: Sequence[str]) -> list[str]:
    tokens: list[str] = []
    for text in texts:
        tokens.extend(_tokenize(text))
    return tokens


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index : index + n]) for index in range(0, max(0, len(tokens) - n + 1))]


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _audit_note(reversal: bool) -> JsonDict:
    return {
        "policy_id": FLAGGED_POLICY_ID,
        "action": "v14_retention_reversed" if reversal else "v14_retention_preserved",
        "reason": (
            "held-out replay reconfirmed mode collapse after lambda-GRPO correction evidence"
            if reversal
            else "held-out replay did not satisfy all reversal gates"
        ),
    }


def _honest_verdict(
    *,
    status: str,
    replayed: bool,
    reversal: bool,
    collapse_confirmed: bool,
    blockers: Sequence[str],
) -> str:
    if status == "blocked" and not replayed:
        reason = blockers[0] if blockers else "missing flagged-policy replay"
        return f"blocked: {reason}"
    if reversal:
        return "complete: replay reconfirmed mode collapse; v14 retention reversed"
    if not collapse_confirmed:
        return "complete: retention reversal blocked by replay; mode collapse not reconfirmed"
    return "complete: retention reversal blocked by replay safety gates"


def _load_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _present_strings(values: Sequence[Any]) -> list[str]:
    return [str(value) for value in values if isinstance(value, str) and value.strip()]


def _chunk(values: Sequence[float], size: int) -> list[list[float]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(numerator / denominator, 6)


def _mean_float(values: Sequence[Any]) -> float:
    numbers = [float(value) for value in values]
    return 0.0 if not numbers else round(sum(numbers) / len(numbers), 6)


def _resolve_under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _lambda_grpo_patch_available() -> bool:  # pragma: no cover - environment probe.
    return importlib.util.find_spec("carnot.training.lambda_grpo") is not None


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    artifact = run_experiment()
    print(
        "[exp1581] "
        f"replayed={artifact['flagged_policy_replayed']} "
        f"reversed={artifact['retention_reversal_applied']} "
        f"decision_ready={artifact['fr11_v15_decision_ready']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())


__all__ = [
    "FLAGGED_POLICY_ID",
    "OUTPUT_FILE",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_artifact",
    "build_lambda_grpo_simulator_evidence",
    "replay_flagged_policy",
    "run_experiment",
    "simulate_lambda_grpo_weights",
    "validate_artifact",
    "write_in_progress_artifact",
]
