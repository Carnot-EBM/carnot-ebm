"""Exp 4421 registry/gaps hygiene and GAP-4 regression guard.

Spec refs: REQ-VERIFY-4421, SCENARIO-VERIFY-4421.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

import yaml

from carnot.reporting import capstone_aggregate_available
from carnot.reporting import capstone_v407_4412
from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base


REPO_ROOT = Path(__file__).resolve().parents[2]
RANDOM_SEED = 4421
INFERENCE_SUBSTRATE = "cached_registry_reconciliation_gap4_guard_and_capstone_stamp_audit"
SPEC_REFS = ["REQ-VERIFY-4421", "SCENARIO-VERIFY-4421"]

EXP4421_ARTIFACT_PATH = "results/experiment_4421_registry_gaps_hygiene_gap4_guard.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH
ARC_REGISTRY_PATH = "ops/arc_solve_registry.yaml"
GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
FOVER_VERIFIER_ID = "fover_production_ensemble"

EXP4414_PATH = "results/experiment_4414_config_rule_induction_solve.json"
EXP4415_PATH = "results/experiment_4415_agent2world_adaptive_e3_repair.json"
EXP4416_PATH = "results/experiment_4416_hidden_state_localizer_falsification_audit.json"
EXP4417_PATH = "results/experiment_4417_gap4_local_generator_sovereign_arm.json"
EXP4418_PATH = "results/experiment_4418_config_rule_vocabulary_transfer.json"
EXP4419_PATH = "results/experiment_4419_steerconf_code_detection_calibration_repair.json"
CAPSTONE_PATH = "results/experiment_4412_capstone_v407.json"
GAP4_EXECUTION_RESULT_PATH = "results/arc3_gap4_rule_exec_verifier.json"

GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED = "GAP-FOVER-BIPRM-LOCALIZATION-untyped"
GAP_4414_CONFIG_RULE_KA59 = "GAP-4414-KA59-CONFIG-RULE-GROUNDED"
GAP_4418_CONFIG_RULE_VOCAB_BLOCKED = "GAP-4418-CONFIG-RULE-VOCABULARY-LOCAL-MODEL-UNAVAILABLE"
GAP_4417_SOVEREIGN_ZERO_FIRES = "GAP-4417-SOVEREIGN-GAP4-LOCAL-GENERATOR-ZERO-FIRES"

V408_ROLE_ID = "oracle_distinct_v408_registry_gaps_hygiene_4421"
V408_STATE = (
    "config_rule_partial__adaptive_e3_zero_new__hidden_state_position_null__"
    "sovereign_gap4_holds_zero_fires__vocabulary_blocked__code_detection_chance"
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "registry_reconciliation",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "v408_outcomes",
    "availability_report",
    "gap4_regression_guard",
    "capstone_stamp_fix",
    "field_principles",
    "spec_refs",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "Terminal-prefixed (complete: registry_gaps_reconciled_guard_<passed|failed>)."
    },
    "regression_guard_passed": {
        "principle": (
            "BARE bool: the GAP-4 execution result did not regress AND the "
            "verifier_is_oracle stamping fix is durable -- the capstone reads this."
        )
    },
    "registry_reconciliation": {
        "principle": (
            "dict: which gaps moved to filled / sharpened / newly-logged from the "
            ".408 outcomes (the never-prune audit trail)."
        )
    },
    "preconditions_checked": {
        "principle": (
            "Records the registries + the .408 artifacts loaded (robust "
            "aggregate-available) + TRM-stand-down; pre-empts the "
            "silent-missing-resource fabrication mode."
        )
    },
    "random_seed": {"principle": "Determinism precondition for the guard checks."},
    "reproducibility_checksum": {
        "principle": (
            "Hash of the reconciliation + the guard inputs; lets a third party re-run."
        )
    },
}

Gap4GuardRunner = Callable[[Path], dict[str, Any]]
CapstoneStampRunner = Callable[[Path], dict[str, Any]]


def _json_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_optional_json(repo_root: Path, rel_path: str) -> tuple[dict[str, Any] | None, str]:
    path = repo_root / rel_path
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(loaded, dict):
        return None, "top-level JSON is not an object"
    return loaded, ""


def _bool(payload: Mapping[str, Any] | None, key: str) -> bool | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(key)
    return value if isinstance(value, bool) else None


def _int(payload: Mapping[str, Any] | None, key: str) -> int:
    if not isinstance(payload, Mapping):
        return 0
    value = payload.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _float(payload: Mapping[str, Any] | None, key: str) -> float | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _str(payload: Mapping[str, Any] | None, key: str) -> str:
    if not isinstance(payload, Mapping):
        return ""
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def _list(payload: Mapping[str, Any] | None, key: str) -> list[Any]:
    if not isinstance(payload, Mapping):
        return []
    value = payload.get(key)
    return value if isinstance(value, list) else []


def _yaml_parse_check(repo_root: Path, key: str, rel_path: str) -> dict[str, Any]:
    path = repo_root / rel_path
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        parsed = True
        error = ""
    except (OSError, yaml.YAMLError) as exc:
        loaded = None
        parsed = False
        error = f"{type(exc).__name__}: {exc}"
    ok = parsed and isinstance(loaded, dict)
    if parsed and not isinstance(loaded, dict):
        error = "top-level YAML is not a mapping"
    return {
        "key": key,
        "path": rel_path,
        "yaml_safe_load": parsed,
        "top_level_type": type(loaded).__name__ if parsed else None,
        "readable": ok,
        "error": error,
    }


def _markdown_text_check(repo_root: Path, key: str, rel_path: str) -> dict[str, Any]:
    path = repo_root / rel_path
    try:
        text = path.read_text(encoding="utf-8")
        ok = True
        error = ""
    except OSError as exc:
        text = ""
        ok = False
        error = f"{type(exc).__name__}: {exc}"
    return {
        "key": key,
        "path": rel_path,
        "readable": ok,
        "markdown_text": True,
        "bytes": len(text.encode("utf-8")),
        "error": error,
    }


def check_preconditions(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4421: YAML registries parse; gaps ledger loads as markdown text."""
    checks = {
        "verifier_registry": _yaml_parse_check(repo_root, "verifier_registry", REGISTRY_PATH),
        "verifier_gaps": _markdown_text_check(repo_root, "verifier_gaps", GAPS_PATH),
        "arc_solve_registry": _yaml_parse_check(repo_root, "arc_solve_registry", ARC_REGISTRY_PATH),
    }
    blocked_file = next((key for key, row in checks.items() if not row["readable"]), None)
    return {"ok": blocked_file is None, "blocked_file": blocked_file, "files": checks}


def _artifact_status(payload: dict[str, Any] | None, error: str, rel_path: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": rel_path, "available": False, "error": error}
    return {
        "artifact_path": rel_path,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "random_seed": payload.get("random_seed"),
        "reproducibility_checksum": payload.get("reproducibility_checksum"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
    }


def _read_config_rule(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    row = _artifact_status(payload, error, EXP4414_PATH)
    if payload is None:
        return row
    row.update(
        {
            "new_levels_reproduced": _int(payload, "new_levels_reproduced"),
            "reproducible_total_levels": _int(payload, "reproducible_total_levels"),
            "config_win_rules_grounded": _list(payload, "config_win_rules_grounded"),
            "per_target_scorecard": _list(payload, "per_target_scorecard"),
            "preconditions_checked": payload.get("preconditions_checked", {}),
        }
    )
    return row


def _read_adaptive_e3(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    row = _artifact_status(payload, error, EXP4415_PATH)
    if payload is None:
        return row
    row.update(
        {
            "new_levels_reproduced": _int(payload, "new_levels_reproduced"),
            "reproducible_total_levels": _int(payload, "reproducible_total_levels"),
            "per_target_scorecard": _list(payload, "per_target_scorecard"),
            "preconditions_checked": payload.get("preconditions_checked", {}),
        }
    )
    return row


def _read_hidden_state(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    row = _artifact_status(payload, error, EXP4416_PATH)
    if payload is None:
        return row
    row.update(
        {
            "hidden_state_localizer_has_nonposition_signal": _bool(
                payload, "hidden_state_localizer_has_nonposition_signal"
            ),
            "position_only_baseline_f1": _float(payload, "position_only_baseline_f1"),
            "missing_verifier_gaps": _list(payload, "missing_verifier_gaps"),
            "preconditions_checked": payload.get("preconditions_checked", []),
        }
    )
    return row


def _read_sovereign_gap4(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    row = _artifact_status(payload, error, EXP4417_PATH)
    if payload is None:
        return row
    pass2 = payload.get("pass2_vs_vote", {})
    row.update(
        {
            "sovereign_gap4_gate_holds": _bool(payload, "sovereign_gap4_gate_holds"),
            "local_generator_coverage": _float(payload, "local_generator_coverage"),
            "pass2_vs_vote": dict(pass2) if isinstance(pass2, Mapping) else {},
            "preconditions_checked": payload.get("preconditions_checked", []),
        }
    )
    return row


def _read_vocab(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    row = _artifact_status(payload, error, EXP4418_PATH)
    if payload is None:
        return row
    preconditions = payload.get("preconditions_checked", {})
    row.update(
        {
            "config_rule_vocabulary": _list(payload, "config_rule_vocabulary"),
            "config_rule_vocabulary_transfers": (
                _bool(payload, "config_rule_vocabulary_transfers") is True
            ),
            "preconditions_checked": preconditions,
            "local_model_status": (
                preconditions.get("local_model_server", {}).get("status")
                if isinstance(preconditions, Mapping)
                else None
            ),
        }
    )
    return row


def _read_steerconf(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    row = _artifact_status(payload, error, EXP4419_PATH)
    if payload is None:
        return row
    row.update(
        {
            "detection_calibrated_multi_domain": (
                _bool(payload, "detection_calibrated_multi_domain") is True
            ),
            "domains_at_chance": _list(payload, "domains_at_chance"),
            "detection_by_domain": _list(payload, "detection_by_domain"),
            "missing_verifier_gaps": _list(payload, "missing_verifier_gaps"),
            "positive_control_passed": _bool(payload, "positive_control_passed"),
            "preconditions_checked": payload.get("preconditions_checked", []),
        }
    )
    return row


def _axis_specs() -> list[capstone_aggregate_available.AxisSpec]:
    return [
        capstone_aggregate_available.AxisSpec(
            name="config_rule",
            required_keys=("4414_config_rule",),
            verdict_fn=lambda present: bool(
                present.get("4414_config_rule", {}).get("config_win_rules_grounded")
            ),
        ),
        capstone_aggregate_available.AxisSpec(
            name="adaptive_e3",
            required_keys=("4415_adaptive_e3",),
            verdict_fn=lambda present: int(
                present.get("4415_adaptive_e3", {}).get("new_levels_reproduced") or 0
            ),
        ),
        capstone_aggregate_available.AxisSpec(
            name="hidden_state_localizer",
            required_keys=("4416_hidden_state",),
            verdict_fn=lambda present: present.get("4416_hidden_state", {}).get(
                "hidden_state_localizer_has_nonposition_signal"
            )
            is True,
        ),
        capstone_aggregate_available.AxisSpec(
            name="sovereign_gap4",
            required_keys=("4417_sovereign_gap4",),
            verdict_fn=lambda present: present.get("4417_sovereign_gap4", {}).get(
                "sovereign_gap4_gate_holds"
            )
            is True,
        ),
        capstone_aggregate_available.AxisSpec(
            name="config_rule_vocabulary",
            required_keys=("4418_vocab_transfer",),
            verdict_fn=lambda present: present.get("4418_vocab_transfer", {}).get(
                "config_rule_vocabulary_transfers"
            )
            is True,
        ),
        capstone_aggregate_available.AxisSpec(
            name="code_detection",
            required_keys=("4419_steerconf_code",),
            verdict_fn=lambda present: present.get("4419_steerconf_code", {}).get(
                "detection_calibrated_multi_domain"
            )
            is True,
        ),
    ]


def load_v408_outcomes(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4421: read .408 outcomes without fabricating missing artifacts."""
    config_payload, config_error = _load_optional_json(repo_root, EXP4414_PATH)
    e3_payload, e3_error = _load_optional_json(repo_root, EXP4415_PATH)
    hidden_payload, hidden_error = _load_optional_json(repo_root, EXP4416_PATH)
    sovereign_payload, sovereign_error = _load_optional_json(repo_root, EXP4417_PATH)
    vocab_payload, vocab_error = _load_optional_json(repo_root, EXP4418_PATH)
    steer_payload, steer_error = _load_optional_json(repo_root, EXP4419_PATH)
    raw_artifacts = {
        "4414_config_rule": config_payload,
        "4415_adaptive_e3": e3_payload,
        "4416_hidden_state": hidden_payload,
        "4417_sovereign_gap4": sovereign_payload,
        "4418_vocab_transfer": vocab_payload,
        "4419_steerconf_code": steer_payload,
    }
    outcomes = {
        "config_rule_induction": _read_config_rule(config_payload, config_error),
        "adaptive_e3_repair": _read_adaptive_e3(e3_payload, e3_error),
        "hidden_state_localizer": _read_hidden_state(hidden_payload, hidden_error),
        "sovereign_gap4": _read_sovereign_gap4(sovereign_payload, sovereign_error),
        "config_rule_vocabulary": _read_vocab(vocab_payload, vocab_error),
        "steerconf_code_detection": _read_steerconf(steer_payload, steer_error),
        "availability_report": capstone_aggregate_available.aggregate_available_report_gaps(
            raw_artifacts,
            _axis_specs(),
            artifact_experiment_ids={
                "4414_config_rule": 4414,
                "4415_adaptive_e3": 4415,
                "4416_hidden_state": 4416,
                "4417_sovereign_gap4": 4417,
                "4418_vocab_transfer": 4418,
                "4419_steerconf_code": 4419,
            },
        ),
    }
    outcomes["trm_training_stood_down"] = _trm_training_stood_down(outcomes)
    return outcomes


def _trm_training_stood_down(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "trm_training_stood_down" and item is True:
                return True
            if item == "trm_training_stand_down":
                return bool(value.get("available", True))
            if _trm_training_stood_down(item):
                return True
    if isinstance(value, list):
        return any(_trm_training_stood_down(item) for item in value)
    return False


def _gap_entry(
    gap_id: str,
    *,
    status: str,
    evidence: str,
    failure_mode: str,
    missing_discriminator: str,
    candidate_design: str,
    priority: str = "high",
    movement: str = "newly_logged",
) -> dict[str, Any]:
    return {
        "gap_id": gap_id,
        "status": status,
        "evidence": evidence,
        "failure_mode": failure_mode,
        "missing_discriminator": missing_discriminator,
        "candidate_design": candidate_design,
        "priority": priority,
        "movement": movement,
    }


def _add_upstream_gap(
    entries: dict[str, dict[str, Any]],
    gap: Mapping[str, Any],
    evidence: str,
    *,
    movement: str = "newly_logged",
) -> None:
    gap_id = str(gap.get("gap_id", ""))
    if not gap_id:
        return
    entries[gap_id] = _gap_entry(
        gap_id,
        status=str(gap.get("status", "open")),
        evidence=evidence,
        failure_mode=str(gap.get("failure_mode") or "residual verifier gap"),
        missing_discriminator=str(gap.get("missing_discriminator", "")),
        candidate_design=str(gap.get("candidate_design", "")),
        priority=str(gap.get("priority", "high")),
        movement=movement,
    )


def _missing_upstream_gap(exp_id: int, rel_path: str, error: str) -> dict[str, Any]:
    return _gap_entry(
        f"GAP-4421-MISSING-UPSTREAM-{exp_id}",
        status="open",
        evidence=f"{rel_path}; missing_or_unreadable={error}",
        failure_mode="required .408 upstream artifact was missing or unreadable",
        missing_discriminator="landed upstream evidence before registry reconciliation",
        candidate_design=f"rerun or recover Exp {exp_id}, then rerun Exp 4421",
    )


def build_gap_entries(outcomes: Mapping[str, Any]) -> list[dict[str, Any]]:
    """REQ-VERIFY-4421: collect filled, sharpened, and residual .408 gaps."""
    entries: dict[str, dict[str, Any]] = {}
    config = outcomes["config_rule_induction"]
    adaptive = outcomes["adaptive_e3_repair"]
    hidden = outcomes["hidden_state_localizer"]
    sovereign = outcomes["sovereign_gap4"]
    vocab = outcomes["config_rule_vocabulary"]
    steer = outcomes["steerconf_code_detection"]

    for exp_id, section, path in (
        (4414, config, EXP4414_PATH),
        (4415, adaptive, EXP4415_PATH),
        (4416, hidden, EXP4416_PATH),
        (4417, sovereign, EXP4417_PATH),
        (4418, vocab, EXP4418_PATH),
        (4419, steer, EXP4419_PATH),
    ):
        if section.get("available") is not True:
            gap = _missing_upstream_gap(exp_id, path, str(section.get("error", "")))
            entries[gap["gap_id"]] = gap

    for rule in config.get("config_win_rules_grounded", []):
        if not isinstance(rule, Mapping):
            continue
        if rule.get("game") == "ka59" and rule.get("fires_on_win") is True:
            entries[GAP_4414_CONFIG_RULE_KA59] = _gap_entry(
                GAP_4414_CONFIG_RULE_KA59,
                status="filled (ka59_config_win_rule_predicate)",
                evidence=(
                    f"{EXP4414_PATH}; predicate={rule.get('predicate')}; "
                    f"tier={rule.get('tier')}; false_positive_rate={rule.get('false_positive_rate')}"
                ),
                failure_mode="ka59 had an ungrounded config-game win-rule predicate",
                missing_discriminator="grounded config win rule checked against the offline state",
                candidate_design="registry-backed config-rule predicate reused for future ka59 targets",
                priority="medium",
                movement="filled",
            )

    for row in config.get("per_target_scorecard", []):
        if not isinstance(row, Mapping):
            continue
        if row.get("offline_reproduced") is True or row.get("game") == "ka59":
            continue
        game = str(row.get("game", "unknown")).upper()
        gap_id = f"GAP-4414-CONFIG-RULE-INDUCTION-{game}"
        entries[gap_id] = _gap_entry(
            gap_id,
            status="open",
            evidence=f"{EXP4414_PATH}; game={row.get('game')}; search_blocker={row.get('search_blocker')}",
            failure_mode="fresh config-rule induction did not run or did not reproduce a new level",
            missing_discriminator=f"grounded reusable config win-rule for {row.get('game')}",
            candidate_design="run local symbolic/config-rule induction once the local proposer is available",
        )

    for row in adaptive.get("per_target_scorecard", []):
        if not isinstance(row, Mapping) or row.get("offline_reproduced") is True:
            continue
        game = str(row.get("game", "unknown"))
        target = row.get("target_level", "unknown")
        gap_id = f"GAP-4415-ADAPTIVE-E3-{game.upper()}-L{target}"
        entries[gap_id] = _gap_entry(
            gap_id,
            status="open",
            evidence=(
                f"{EXP4415_PATH}; game={game}; target_level={target}; "
                f"adaptive_tests={row.get('adaptive_tests_passed')}/{row.get('adaptive_tests_total')}; "
                f"verifier_accuracy={row.get('verifier_accuracy')}; "
                f"lookahead_fidelity={row.get('lookahead_fidelity')}; "
                f"residual={row.get('residual_failing_behavior')}"
            ),
            failure_mode=f"{game} L{target} remains unreproduced after adaptive E3 repair",
            missing_discriminator=f"state-grounded executable rule for {row.get('residual_failing_behavior')}",
            candidate_design="convert the residual behavior test into an offline reproduce() plan",
        )

    if hidden.get("hidden_state_localizer_has_nonposition_signal") is False:
        entries[GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED] = _gap_entry(
            GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED,
            status="open (sharpened by exp4416 hidden-state null)",
            evidence=(
                f"{EXP4416_PATH}; hidden_state_localizer_has_nonposition_signal=False; "
                f"position_only_baseline_f1={hidden.get('position_only_baseline_f1')}"
            ),
            failure_mode="hidden-state localizer tied the content-blind position baseline",
            missing_discriminator="non-position earliest causal error signal under non-degenerate traces",
            candidate_design="collect typed multi-step first-error traces before reviving localization",
            priority="medium",
            movement="sharpened",
        )
    for gap in hidden.get("missing_verifier_gaps", []):
        if isinstance(gap, Mapping):
            _add_upstream_gap(entries, gap, f"{EXP4416_PATH}; hidden-state localizer null", movement="sharpened")

    pass2 = sovereign.get("pass2_vs_vote", {})
    if isinstance(pass2, Mapping) and int(pass2.get("graded_gate_fires") or 0) == 0:
        entries[GAP_4417_SOVEREIGN_ZERO_FIRES] = _gap_entry(
            GAP_4417_SOVEREIGN_ZERO_FIRES,
            status="open",
            evidence=(
                f"{EXP4417_PATH}; sovereign_gap4_gate_holds={sovereign.get('sovereign_gap4_gate_holds')}; "
                f"graded_gate_fires={pass2.get('graded_gate_fires')}; "
                f"gated_pass2={pass2.get('gated_pass2')}; vote_pass2={pass2.get('vote_pass2')}"
            ),
            failure_mode="local-generator sovereign arm holds the safety gate but fires zero graded wins",
            missing_discriminator="local open-weight generator proposal that creates verifier-actionable GAP-4 candidates",
            candidate_design="separate reusable symbolic induction from another static local-generator replay",
        )

    if vocab.get("available") is True and vocab.get("config_rule_vocabulary_transfers") is not True:
        entries[GAP_4418_CONFIG_RULE_VOCAB_BLOCKED] = _gap_entry(
            GAP_4418_CONFIG_RULE_VOCAB_BLOCKED,
            status="open",
            evidence=(
                f"{EXP4418_PATH}; transfers={vocab.get('config_rule_vocabulary_transfers')}; "
                f"local_model_status={vocab.get('local_model_status')}"
            ),
            failure_mode="config-rule vocabulary transfer was blocked by local model unavailability",
            missing_discriminator="local config-rule proposer that transfers grounded vocabulary to unsolved games",
            candidate_design="cache or start the declared local iGPU proposer and rerun vocabulary transfer",
        )

    for gap in steer.get("missing_verifier_gaps", []):
        if isinstance(gap, Mapping):
            _add_upstream_gap(entries, gap, f"{EXP4419_PATH}; SteerConf code detection chance")
    known = set(entries)
    for domain in steer.get("domains_at_chance", []):
        gap_id = f"GAP-4419-{str(domain).upper().replace('_', '-')}-STEERCONF-DETECTOR-CHANCE"
        if gap_id in known:
            continue
        row = _domain_row(steer, str(domain))
        entries[gap_id] = _gap_entry(
            gap_id,
            status="open",
            evidence=(
                f"{EXP4419_PATH}; domain={domain}; auroc={row.get('detection_auroc')}; "
                f"ci95={row.get('auroc_ci95')}; n={row.get('n')}"
            ),
            failure_mode=f"{domain} SteerConf detector CI includes chance",
            missing_discriminator=f"domain-native oracle-distinct verifier feature for {domain}",
            candidate_design="build a domain-specific verifier feature and rerun the calibration gate",
        )
    return list(entries.values())


def _domain_row(section: Mapping[str, Any], domain: str) -> dict[str, Any]:
    for row in section.get("detection_by_domain", []):
        if isinstance(row, Mapping) and row.get("domain") == domain:
            return dict(row)
    return {}


def _gap_entry_block(gap: Mapping[str, Any]) -> str:
    return (
        f"### {gap['gap_id']}: Exp 4421 .408 verifier gap update\n"
        f"- status: {gap.get('status', 'open')}\n"
        f"- evidence: {gap.get('evidence', '')}.\n"
        f"- failure mode: {gap.get('failure_mode', '')}\n"
        f"- missing discriminator: {gap.get('missing_discriminator', '')}\n"
        f"- candidate design: {gap.get('candidate_design', '')}\n"
        f"- priority: {gap.get('priority', 'high')}\n"
        f"- movement: {gap.get('movement', 'newly_logged')}\n"
    )


def _replace_marked_block(text: str, marker: str, block: str) -> str:
    start = f"<!-- {marker}:start -->"
    end = f"<!-- {marker}:end -->"
    replacement = f"{start}\n{block.rstrip()}\n{end}"
    if start in text and end in text:
        prefix, rest = text.split(start, 1)
        _, suffix = rest.split(end, 1)
        return f"{prefix}{replacement}{suffix}"
    return text.rstrip() + "\n\n" + replacement + "\n"


def _arc_total(outcomes: Mapping[str, Any]) -> int:
    return max(
        int(outcomes["config_rule_induction"].get("reproducible_total_levels") or 0),
        int(outcomes["adaptive_e3_repair"].get("reproducible_total_levels") or 0),
    )


def _arc_new_levels(outcomes: Mapping[str, Any]) -> int:
    return int(outcomes["config_rule_induction"].get("new_levels_reproduced") or 0) + int(
        outcomes["adaptive_e3_repair"].get("new_levels_reproduced") or 0
    )


def _ensure_gap4_eval(
    registry: dict[str, Any],
    guard: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    gap_entries: list[dict[str, Any]],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    sovereign = outcomes["sovereign_gap4"]
    pass2 = sovereign.get("pass2_vs_vote", {})
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4421": EXP4421_ARTIFACT_PATH,
            "exp4421_regression_guard_passed": bool(guard.get("regression_guard_passed")),
            "exp4421_v408_state": V408_STATE,
            "exp4421_arc_reproducible_total_levels": _arc_total(outcomes),
            "exp4421_new_levels_reproduced": _arc_new_levels(outcomes),
            "exp4421_config_win_rules_grounded": outcomes["config_rule_induction"].get(
                "config_win_rules_grounded"
            ),
            "exp4421_sovereign_gap4_gate_holds": sovereign.get("sovereign_gap4_gate_holds"),
            "exp4421_graded_gate_fires": pass2.get("graded_gate_fires") if isinstance(pass2, Mapping) else None,
            "exp4421_local_generator_coverage": sovereign.get("local_generator_coverage"),
            "exp4421_gaps_reconciled": [gap["gap_id"] for gap in gap_entries],
        }
    )
    role = {
        "role_id": V408_ROLE_ID,
        "experiment": EXP4421_ARTIFACT_PATH,
        "role": "registry_gaps_arc_hygiene_v408",
        "status": "v408_outcomes_recorded_with_gap4_guard_and_stamp_durability",
        "v408_state": V408_STATE,
        "arc_reproducible_total_levels": _arc_total(outcomes),
        "new_levels_reproduced": _arc_new_levels(outcomes),
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [
        old for old in old_roles if old.get("role_id") != V408_ROLE_ID
    ] + [role]


def _ensure_fover_eval(registry: dict[str, Any], outcomes: Mapping[str, Any]) -> None:
    entry = base._find_verifier(registry, FOVER_VERIFIER_ID)
    if entry is None:
        entry = {
            "verifier_id": FOVER_VERIFIER_ID,
            "domain": "math_reasoning",
            "kind": "ensemble",
            "eval": {},
            "status": "active",
        }
        registry.setdefault("verifiers", []).append(entry)
    hidden = outcomes["hidden_state_localizer"]
    steer = outcomes["steerconf_code_detection"]
    code = _domain_row(steer, "code_humaneval")
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4421": EXP4421_ARTIFACT_PATH,
            "exp4421_hidden_state_localizer_has_nonposition_signal": hidden.get(
                "hidden_state_localizer_has_nonposition_signal"
            ),
            "exp4421_position_only_baseline_f1": hidden.get("position_only_baseline_f1"),
            "exp4421_detection_calibrated_multi_domain": steer.get(
                "detection_calibrated_multi_domain"
            ),
            "exp4421_code_humaneval_detection_auroc": code.get("detection_auroc"),
            "exp4421_code_humaneval_detection_ci95": code.get("auroc_ci95"),
            "exp4421_domains_at_chance": steer.get("domains_at_chance"),
            "exp4421_verifier_is_oracle": steer.get("verifier_is_oracle"),
        }
    )


def _ensure_arc_registry(arc_registry: dict[str, Any], outcomes: Mapping[str, Any]) -> None:
    arc_registry["updated"] = "2026-06-19"
    arc_registry["reproducible_total_levels"] = max(
        int(arc_registry.get("reproducible_total_levels") or 0),
        _arc_total(outcomes),
    )
    arc_registry["latest_hygiene_4421"] = {
        "artifact": EXP4421_ARTIFACT_PATH,
        "reproducible_total_levels": _arc_total(outcomes),
        "new_levels_reproduced": _arc_new_levels(outcomes),
        "exp4414_new_levels_reproduced": outcomes["config_rule_induction"].get(
            "new_levels_reproduced"
        ),
        "exp4415_new_levels_reproduced": outcomes["adaptive_e3_repair"].get(
            "new_levels_reproduced"
        ),
        "config_rule_grounded_games": [
            row.get("game")
            for row in outcomes["config_rule_induction"].get("config_win_rules_grounded", [])
            if isinstance(row, Mapping)
        ],
        "note": ".408 grounded rules and adaptive E3 sharpened gaps but added zero reproduced ARC levels.",
    }


def registry_contains_v408(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    fover = base._find_verifier(registry, FOVER_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4421") == EXP4421_ARTIFACT_PATH
        and gap4.get("eval", {}).get("exp4421_v408_state") == V408_STATE
        and any(role.get("role_id") == V408_ROLE_ID for role in gap4.get("registry_roles", []))
        and fover
        and fover.get("eval", {}).get("eval_exp_4421") == EXP4421_ARTIFACT_PATH
    )


def arc_registry_contains_v408(arc_registry: dict[str, Any]) -> bool:
    latest = arc_registry.get("latest_hygiene_4421", {})
    return bool(
        int(arc_registry.get("reproducible_total_levels") or 0) >= 34
        and isinstance(latest, Mapping)
        and latest.get("artifact") == EXP4421_ARTIFACT_PATH
        and latest.get("new_levels_reproduced") == 0
    )


def gaps_contain_v408(gaps_text: str, gap_entries: list[dict[str, Any]]) -> bool:
    return all(f"<!-- exp4421-{gap['gap_id'].lower()}:start -->" in gaps_text for gap in gap_entries)


def ensure_ledgers_record_v408(
    registry: dict[str, Any],
    gaps_text: str,
    arc_registry: dict[str, Any],
    regression_guard: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    gap_entries: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Return ledgers with .408 truth represented idempotently."""
    updated_registry = deepcopy(registry)
    updated_arc = deepcopy(arc_registry)
    _ensure_gap4_eval(updated_registry, regression_guard, outcomes, gap_entries)
    _ensure_fover_eval(updated_registry, outcomes)
    _ensure_arc_registry(updated_arc, outcomes)
    for gap in gap_entries:
        marker = f"exp4421-{gap['gap_id'].lower()}"
        gaps_text = _replace_marked_block(gaps_text, marker, _gap_entry_block(gap))

    registry_ok = registry_contains_v408(updated_registry)
    arc_ok = arc_registry_contains_v408(updated_arc)
    gaps_ok = gaps_contain_v408(gaps_text, gap_entries)
    filled = [gap["gap_id"] for gap in gap_entries if gap.get("movement") == "filled"]
    sharpened = [gap["gap_id"] for gap in gap_entries if gap.get("movement") == "sharpened"]
    newly_logged = [gap["gap_id"] for gap in gap_entries if gap.get("movement") == "newly_logged"]
    return (
        updated_registry,
        gaps_text,
        updated_arc,
        {
            "verifier_registry_reconciled": registry_ok,
            "verifier_gaps_reconciled": gaps_ok,
            "arc_solve_registry_reconciled": arc_ok,
            "registries_reconciled": registry_ok and gaps_ok and arc_ok,
            "filled_gap_ids": filled,
            "sharpened_gap_ids": sharpened,
            "newly_logged_gap_ids": newly_logged,
            "gaps_moved": {
                "filled": filled,
                "sharpened": sharpened,
                "newly_logged": newly_logged,
            },
        },
    )


def _flags_from_report(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for row in report.get("reports", []):
        if isinstance(row, Mapping):
            for flag in row.get("flags", []):
                if isinstance(flag, Mapping):
                    flags.append(dict(flag))
    return flags


def _adversarial_report(repo_root: Path, rel_path: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(repo_root / "scripts" / "adversarial_verify.py"),
        "--json",
        str(repo_root / rel_path),
    ]
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError:
        parsed = {"reports": [], "parse_error": completed.stdout[-500:]}
    parsed["returncode"] = completed.returncode
    parsed["stdout_tail"] = completed.stdout[-1000:]
    parsed["stderr_tail"] = completed.stderr[-1000:]
    parsed["command"] = command
    return parsed


def _ranker_pass2(payload: Mapping[str, Any], ranker: str) -> float | None:
    rankers = payload.get("rankers", {})
    if not isinstance(rankers, Mapping):
        return None
    row = rankers.get(ranker, {})
    if not isinstance(row, Mapping):
        return None
    value = row.get("pass@2")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def run_gap4_regression_guard(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4421: protect the checked-in GAP-4 execution win."""
    payload, error = _load_optional_json(repo_root, GAP4_EXECUTION_RESULT_PATH)
    if payload is None:
        return {
            "regression_guard_passed": False,
            "gap4_execution_guard_passed": False,
            "artifact_path": GAP4_EXECUTION_RESULT_PATH,
            "error": error,
        }
    gated_pass2 = _ranker_pass2(payload, "GAP4_GATED")
    vote_pass2 = _ranker_pass2(payload, "TRM_VOTE")
    gates = payload.get("gates", {})
    if not isinstance(gates, Mapping):
        gates = {}
    report = _adversarial_report(repo_root, GAP4_EXECUTION_RESULT_PATH)
    flags = _flags_from_report(report)
    execution_ok = (
        gated_pass2 is not None
        and vote_pass2 is not None
        and gated_pass2 > vote_pass2
        and gates.get("selection_beats_vote") is True
        and int(gates.get("vote_wins_lost") or 0) == 0
        and not flags
        and report.get("returncode") == 0
    )
    return {
        "regression_guard_passed": execution_ok,
        "gap4_execution_guard_passed": execution_ok,
        "arc_oracle_distinct_verifier_beats_vote": gated_pass2 is not None
        and vote_pass2 is not None
        and gated_pass2 > vote_pass2,
        "artifact_path": GAP4_EXECUTION_RESULT_PATH,
        "current": {
            "gated_pass2": gated_pass2,
            "vote_pass2": vote_pass2,
            "selection_beats_vote": gates.get("selection_beats_vote"),
            "vote_wins_lost": gates.get("vote_wins_lost"),
            "headroom_recovered": gates.get("headroom_recovered"),
        },
        "adversarial_verify": {
            "returncode": report.get("returncode"),
            "flag_count": len(flags),
            "flags": flags,
        },
    }


def _capstone_aggregation_propagates_oracle_stamp() -> bool:
    return (
        "verifier_is_oracle" in capstone_v407_4412.REQUIRED_ARTIFACT_FIELDS
        and "verifier_is_oracle" in capstone_v407_4412.FIELD_PRINCIPLES
    )


def _capstone_aggregation_uses_available_helper() -> bool:
    return (
        capstone_v407_4412.aggregate.aggregate_available_report_gaps
        is capstone_aggregate_available.aggregate_available_report_gaps
    )


def verify_capstone_stamp_fix_durable(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4421: audit the latest capstone stamp path."""
    propagates = _capstone_aggregation_propagates_oracle_stamp()
    uses_helper = _capstone_aggregation_uses_available_helper()
    capstone_path = repo_root / CAPSTONE_PATH
    try:
        capstone = json.loads(capstone_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "capstone_stamp_fix_durable": False,
            "capstone_path": CAPSTONE_PATH,
            "error": f"{type(exc).__name__}: {exc}",
            "capstone_verifier_is_oracle": None,
            "capstone_verifier_is_oracle_honored": None,
            "capstone_aggregation_propagates_oracle_stamp": propagates,
            "capstone_aggregation_uses_available_helper": uses_helper,
            "circular_moat_overclaim_fired": False,
            "flag_count": 0,
            "flags": [],
            "returncode": None,
        }
    report = _adversarial_report(repo_root, CAPSTONE_PATH)
    flags = _flags_from_report(report)
    circular = [flag for flag in flags if flag.get("kind") == "CIRCULAR_MOAT_OVERCLAIM"]
    durable = (
        capstone.get("verifier_is_oracle") is False
        and capstone.get("verifier_is_oracle_honored") is True
        and propagates
        and uses_helper
        and not circular
        and not flags
        and report.get("returncode") == 0
    )
    return {
        "capstone_stamp_fix_durable": durable,
        "capstone_path": CAPSTONE_PATH,
        "capstone_verifier_is_oracle": capstone.get("verifier_is_oracle"),
        "capstone_verifier_is_oracle_honored": capstone.get("verifier_is_oracle_honored"),
        "capstone_aggregation_propagates_oracle_stamp": propagates,
        "capstone_aggregation_uses_available_helper": uses_helper,
        "capstone_aggregation_source": "carnot.reporting.capstone_v407_4412",
        "circular_moat_overclaim_fired": bool(circular),
        "flag_count": len(flags),
        "flags": flags,
        "returncode": report.get("returncode"),
        "stdout_tail": report.get("stdout_tail"),
        "stderr_tail": report.get("stderr_tail"),
        "command": report.get("command"),
    }


def _patch_arc_registry_text(text: str, outcomes: Mapping[str, Any]) -> str:
    if "latest_hygiene_4421:" in text:
        return text
    block = (
        "latest_hygiene_4421:\n"
        f"  artifact: {EXP4421_ARTIFACT_PATH}\n"
        f"  reproducible_total_levels: {_arc_total(outcomes)}\n"
        f"  new_levels_reproduced: {_arc_new_levels(outcomes)}\n"
        "  exp4414_new_levels_reproduced: "
        f"{outcomes['config_rule_induction'].get('new_levels_reproduced')}\n"
        "  exp4415_new_levels_reproduced: "
        f"{outcomes['adaptive_e3_repair'].get('new_levels_reproduced')}\n"
        '  note: ".408 grounded rules and adaptive E3 sharpened gaps but added zero reproduced ARC levels."\n'
    )
    return text.rstrip() + "\n\n" + block


def model_specs() -> dict[str, Any]:
    return {
        "method": "cached_v408_ledger_reconciliation_gap4_guard_and_stamp_audit",
        "upstream_artifacts": [
            EXP4414_PATH,
            EXP4415_PATH,
            EXP4416_PATH,
            EXP4417_PATH,
            EXP4418_PATH,
            EXP4419_PATH,
            CAPSTONE_PATH,
            GAP4_EXECUTION_RESULT_PATH,
        ],
        "codex_calls": 0,
        "live_model_inference": False,
        "gguf_inference": False,
        "gpu_inference": False,
    }


def _combined_regression_guard_passed(
    gap4_regression_guard: Mapping[str, Any],
    capstone_stamp_fix: Mapping[str, Any],
) -> bool:
    return bool(gap4_regression_guard.get("regression_guard_passed")) and bool(
        capstone_stamp_fix.get("capstone_stamp_fix_durable")
    )


def build_artifact(
    *,
    preconditions_checked: dict[str, Any],
    gap4_regression_guard: dict[str, Any],
    capstone_stamp_fix: dict[str, Any],
    v408_outcomes: dict[str, Any],
    registry_reconciliation: dict[str, Any],
    availability_report: dict[str, Any],
    reproducibility_checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    guard_ok = _combined_regression_guard_passed(gap4_regression_guard, capstone_stamp_fix)
    reconciled = bool(registry_reconciliation.get("registries_reconciled"))
    artifact = {
        "experiment": "experiment_4421_registry_gaps_hygiene_gap4_guard",
        "schema": "carnot.experiment_4421_registry_gaps_hygiene_gap4_guard.v1",
        "honest_verdict": (
            "complete: registry_gaps_reconciled_guard_passed"
            if guard_ok and reconciled
            else "complete: registry_gaps_reconciled_guard_failed"
        ),
        "regression_guard_passed": guard_ok,
        "registry_reconciliation": registry_reconciliation,
        "preconditions_checked": preconditions_checked,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum,
        "v408_outcomes": v408_outcomes,
        "availability_report": availability_report,
        "gap4_regression_guard": gap4_regression_guard,
        "capstone_stamp_fix": capstone_stamp_fix,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": model_specs(),
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "arc_registry_path": ARC_REGISTRY_PATH,
    }
    validate_artifact(artifact)
    return artifact


def _blocked_artifact(preconditions_checked: dict[str, Any], duration_s: float) -> dict[str, Any]:
    artifact = {
        "experiment": "experiment_4421_registry_gaps_hygiene_gap4_guard",
        "schema": "carnot.experiment_4421_registry_gaps_hygiene_gap4_guard.v1",
        "honest_verdict": "blocked_registry_gaps_reconciliation_unavailable",
        "regression_guard_passed": False,
        "registry_reconciliation": {},
        "preconditions_checked": preconditions_checked,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "blocked:registry_gaps_reconciliation_unavailable",
        "v408_outcomes": {},
        "availability_report": {},
        "gap4_regression_guard": {},
        "capstone_stamp_fix": {},
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": model_specs(),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 4421 terminal artifact before writing."""
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required artifact field: {field}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(("complete:", "blocked_")):
        raise ValueError("honest_verdict must use a terminal prefix")
    if type(artifact["regression_guard_passed"]) is not bool:
        raise ValueError("regression_guard_passed must be a BARE bool")
    for field in (
        "registry_reconciliation",
        "preconditions_checked",
        "v408_outcomes",
        "availability_report",
        "gap4_regression_guard",
        "capstone_stamp_fix",
    ):
        if not isinstance(artifact[field], dict):
            raise ValueError(f"{field} must be an object")
    if isinstance(artifact["random_seed"], bool) or not isinstance(artifact["random_seed"], int):
        raise ValueError("random_seed must be a bare int")
    if (
        not isinstance(artifact["reproducibility_checksum"], str)
        or not artifact["reproducibility_checksum"]
    ):
        raise ValueError("reproducibility_checksum must be a non-empty string")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required Exp 4421 principles")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs must cite REQ-VERIFY-4421 and SCENARIO-VERIFY-4421")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def run_hygiene(
    repo_root: Path = REPO_ROOT,
    *,
    gap4_guard_runner: Gap4GuardRunner | None = None,
    capstone_stamp_runner: CapstoneStampRunner | None = None,
) -> dict[str, Any]:
    """Run Exp 4421 and write the terminal artifact plus reconciled ledgers."""
    if gap4_guard_runner is None:
        gap4_guard_runner = run_gap4_regression_guard
    if capstone_stamp_runner is None:
        capstone_stamp_runner = verify_capstone_stamp_fix_durable
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4421_ARTIFACT_PATH
    if not preflight["ok"]:
        artifact = _blocked_artifact(preflight, time.time() - started)
        base._write_json(out_path, artifact)
        return artifact

    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH
    arc_path = repo_root / ARC_REGISTRY_PATH
    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    arc_registry = yaml.safe_load(arc_path.read_text(encoding="utf-8"))
    if not isinstance(arc_registry, dict):
        arc_registry = {}

    guard = gap4_guard_runner(repo_root)
    stamp = capstone_stamp_runner(repo_root)
    outcomes = load_v408_outcomes(repo_root)
    availability_report = dict(outcomes.get("availability_report", {}))
    preconditions_checked = {
        **preflight,
        "v408_artifacts_loaded": {
            key: section.get("available")
            for key, section in outcomes.items()
            if isinstance(section, Mapping) and "available" in section
        },
        "availability_report": availability_report,
        "trm_training_stood_down": outcomes.get("trm_training_stood_down") is True,
    }
    gap_entries = build_gap_entries(outcomes)
    registry, gaps_text, arc_registry, summary = ensure_ledgers_record_v408(
        registry,
        gaps_text,
        arc_registry,
        guard,
        outcomes,
        gap_entries,
    )

    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")
    original_arc_text = arc_path.read_text(encoding="utf-8")
    patched_arc_text = _patch_arc_registry_text(original_arc_text, outcomes)
    if patched_arc_text != original_arc_text:
        arc_path.write_text(patched_arc_text, encoding="utf-8")
    elif not arc_registry_contains_v408(yaml.safe_load(original_arc_text) or {}):
        arc_path.write_text(yaml.safe_dump(arc_registry, sort_keys=False), encoding="utf-8")

    checksum = _json_hash(
        {
            "registry_reconciliation": summary,
            "gap_ids": [gap["gap_id"] for gap in gap_entries],
            "gap4_regression_guard": guard,
            "capstone_stamp_fix": stamp,
            "availability_report": availability_report,
        }
    )
    artifact = build_artifact(
        preconditions_checked=preconditions_checked,
        gap4_regression_guard=guard,
        capstone_stamp_fix=stamp,
        v408_outcomes=outcomes,
        registry_reconciliation=summary,
        availability_report=availability_report,
        reproducibility_checksum=checksum,
        duration_s=time.time() - started,
    )
    base._write_json(out_path, artifact)
    return artifact


def main() -> None:
    artifact = run_hygiene(REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
