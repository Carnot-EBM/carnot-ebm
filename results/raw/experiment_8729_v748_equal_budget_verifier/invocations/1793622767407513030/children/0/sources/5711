"""Exp 4410 registry/gaps hygiene, GAP-4 guard, and stamp durability.

Spec refs: REQ-VERIFY-4410, SCENARIO-VERIFY-4410.
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
from carnot.reporting import capstone_v406_4401
from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_gap4_guard_4399 as exp4399


REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 4410
INFERENCE_SUBSTRATE = "cached_registry_reconciliation_gap4_guard_and_capstone_stamp_audit"

EXP4410_ARTIFACT_PATH = "results/experiment_4410_registry_gaps_hygiene_gap4_guard.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH
ARC_REGISTRY_PATH = "ops/arc_solve_registry.yaml"
GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
FOVER_VERIFIER_ID = exp4399.FOVER_VERIFIER_ID

CAPSTONE_V406_PATH = "results/experiment_4401_capstone_v406.json"
EXP4403_PATH = "results/experiment_4403_real_intervention_localizer_deconfound.json"
EXP4404_PATH = "results/experiment_4404_localizer_typed_taxonomy_cross_domain.json"
EXP4405_PATH = "results/experiment_4405_e3_deeper_mechanic_unit_tests.json"
EXP4406_PATH = "results/experiment_4406_e3_blocked_mechanic_tails_unit_tests.json"
EXP4407_PATH = "results/experiment_4407_active_learning_self_learning_compounds.json"
EXP4408_PATH = "results/experiment_4408_cross_domain_detection_calibration_repair.json"

V407_ROLE_ID = "oracle_distinct_v407_registry_gaps_hygiene_4410"
V407_STATE = (
    "real_intervention_localizer_position_bound__active_learning_null__"
    "calibration_code_chance__arc_total_34_no_new_levels"
)

GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED = exp4399.GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED
GAP_4404_LOCALIZER_TAXONOMY_BLOCKED = "GAP-4404-LOCALIZER-TYPED-TAXONOMY-BLOCKED"
GAP_UPSTREAM_MISSING_4404 = "GAP-4410-MISSING-UPSTREAM-4404"

SPEC_REFS = ["REQ-VERIFY-4410", "SCENARIO-VERIFY-4410"]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "gaps_reconciled",
    "capstone_stamp_fix_durable",
    "preconditions_checked",
    "random_seed",
    "v407_outcomes",
    "registry_reconciliation",
    "gap4_regression_guard",
    "capstone_stamp_fix",
    "availability_report",
    "reproducibility_checksum",
    "model_specs",
    "field_principles",
    "spec_refs",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "Terminal-prefixed (complete: registry_gaps_arc_reconciled_to_v407_truth_...) "
            "-- the reconciliation + guard landed."
        )
    },
    "regression_guard_passed": {
        "principle": (
            "BARE bool: the capstone reads this; true iff the GAP-4 execution result "
            "did NOT regress vs .406 -- the ARC oracle-distinct verifier-beats-vote "
            "result is protected."
        )
    },
    "gaps_reconciled": {
        "principle": (
            "list/int: the verifier_gaps.md entries moved/sharpened/filled this milestone "
            "(per Missing-Verifier Gap Logging -- the verifier improves monotonically as "
            "gaps are built against)."
        )
    },
    "capstone_stamp_fix_durable": {
        "principle": (
            "BARE bool: the .407 capstone path inherits the exp4355 verifier_is_oracle "
            "stamp fix (so a circular execution-grounded ARC solve is not over-claimed "
            "as a moat)."
        )
    },
    "preconditions_checked": {
        "principle": (
            "Records the registries parse; pre-empts the silent-missing-resource "
            "fabrication mode."
        )
    },
    "random_seed": {
        "principle": "Determinism precondition for any sampled audit ordering."
    },
}

Gap4GuardRunner = Callable[[Path], dict[str, Any]]
CapstoneStampRunner = Callable[[Path], dict[str, Any]]

_json_hash = exp4399._json_hash
_load_optional_json = exp4399._load_optional_json
_bool = exp4399._bool
_int = exp4399._int
_float = exp4399._float
_str = exp4399._str
_list = exp4399._list
_flags_from_report = exp4399._flags_from_report
_scorecard_map = exp4399._scorecard_map


def _yaml_parse_check(repo_root: Path, key: str, rel_path: str, require_mapping: bool) -> dict[str, Any]:
    path = repo_root / rel_path
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        parsed = True
        error = ""
    except (OSError, yaml.YAMLError) as exc:
        loaded = None
        parsed = False
        error = f"{type(exc).__name__}: {exc}"
    top_level_type = type(loaded).__name__ if parsed else None
    ok = parsed and (not require_mapping or isinstance(loaded, dict))
    if parsed and require_mapping and not isinstance(loaded, dict):
        error = "top-level YAML is not a mapping"
    return {
        "key": key,
        "path": rel_path,
        "yaml_safe_load": parsed,
        "top_level_type": top_level_type,
        "readable": ok,
        "error": error,
    }


def check_preconditions(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4410: parse all target ledgers before any mutation."""
    checks = {
        "verifier_registry": _yaml_parse_check(repo_root, "verifier_registry", REGISTRY_PATH, True),
        "verifier_gaps": _yaml_parse_check(repo_root, "verifier_gaps", GAPS_PATH, False),
        "arc_solve_registry": _yaml_parse_check(repo_root, "arc_solve_registry", ARC_REGISTRY_PATH, True),
    }
    blocked_file = next((key for key, row in checks.items() if not row["readable"]), None)
    return {
        "ok": blocked_file is None,
        "blocked_file": blocked_file,
        "files": checks,
    }


def run_gap4_regression_guard(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4410: reuse the Exp 4399 GAP-4 execution guard."""
    return exp4399.run_gap4_regression_guard(repo_root)


def _read_localizer(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4403_PATH, "available": False, "error": error}
    return {
        "artifact_path": EXP4403_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "localizer_genuinely_beats_position_only": (
            _bool(payload, "localizer_genuinely_beats_position_only") is True
        ),
        "beats_position_only_baseline": _bool(payload, "beats_position_only_baseline"),
        "position_only_baseline_f1": _float(payload, "position_only_baseline_f1"),
        "template_family_holdout_drop": _float(payload, "template_family_holdout_drop"),
        "localization_f1_by_domain": dict(payload.get("localization_f1_by_domain", {})),
        "missing_verifier_gaps": _list(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "n_traces": _int(payload, "n_traces"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def _read_taxonomy(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4404_PATH, "available": False, "error": error}
    return {
        "artifact_path": EXP4404_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "status": _str(payload, "status"),
        "blocked_at_layer": _str(payload, "blocked_at_layer"),
        "gate_check_summary": _str(payload, "gate_check_summary"),
        "gates_evaluated": _list(payload, "gates_evaluated"),
    }


def _read_e3_partial(
    payload: dict[str, Any] | None,
    error: str,
    artifact_path: str,
    rows_key: str,
    residual_key: str,
) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": artifact_path, "available": False, "error": error, "rows": {}}
    return {
        "artifact_path": artifact_path,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "new_levels_reproduced": _int(payload, "new_levels_reproduced"),
        "reproducible_total_levels": _int(payload, "reproducible_total_levels"),
        "rows": _scorecard_map(_list(payload, rows_key), residual_key),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def _read_compounds(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4407_PATH, "available": False, "error": error}
    return {
        "artifact_path": EXP4407_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "localizer_compounds": _bool(payload, "localizer_compounds") is True,
        "active_vs_random_learning_curve": _list(payload, "active_vs_random_learning_curve"),
        "compounding_delta_ci95": _list(payload, "compounding_delta_ci95"),
        "positive_control_passed": _bool(payload, "positive_control_passed"),
        "missing_verifier_gaps": _list(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def _read_calibration(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4408_PATH, "available": False, "error": error}
    return {
        "artifact_path": EXP4408_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "detection_calibrated_multi_domain": (
            _bool(payload, "detection_calibrated_multi_domain") is True
        ),
        "detection_by_domain": _list(payload, "detection_by_domain"),
        "domains_at_chance": _list(payload, "domains_at_chance"),
        "missing_verifier_gaps": _list(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def _axis_specs() -> list[capstone_aggregate_available.AxisSpec]:
    return [
        capstone_aggregate_available.AxisSpec(
            name="localizer",
            required_keys=("4403_localizer_deconfound", "4404_localizer_taxonomy"),
            verdict_fn=lambda present: bool(
                present.get("4403_localizer_deconfound", {}).get(
                    "localizer_genuinely_beats_position_only"
                )
            ),
        ),
        capstone_aggregate_available.AxisSpec(
            name="compounding",
            required_keys=("4407_active_learning",),
            verdict_fn=lambda present: bool(
                present.get("4407_active_learning", {}).get("localizer_compounds")
            ),
        ),
        capstone_aggregate_available.AxisSpec(
            name="calibration",
            required_keys=("4408_calibration",),
            verdict_fn=lambda present: bool(
                present.get("4408_calibration", {}).get("detection_calibrated_multi_domain")
            ),
        ),
        capstone_aggregate_available.AxisSpec(
            name="arc",
            required_keys=("4405_e3_deeper", "4406_e3_blocked"),
            verdict_fn=lambda present: int(
                present.get("4405_e3_deeper", {}).get("new_levels_reproduced") or 0
            )
            + int(present.get("4406_e3_blocked", {}).get("new_levels_reproduced") or 0),
        ),
    ]


def load_v407_outcomes(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4410: read .407 outcomes without fabricating missing artifacts."""
    localizer_payload, localizer_error = _load_optional_json(repo_root, EXP4403_PATH)
    taxonomy_payload, taxonomy_error = _load_optional_json(repo_root, EXP4404_PATH)
    deeper_payload, deeper_error = _load_optional_json(repo_root, EXP4405_PATH)
    blocked_payload, blocked_error = _load_optional_json(repo_root, EXP4406_PATH)
    compounds_payload, compounds_error = _load_optional_json(repo_root, EXP4407_PATH)
    calibration_payload, calibration_error = _load_optional_json(repo_root, EXP4408_PATH)

    raw_artifacts = {
        "4403_localizer_deconfound": localizer_payload,
        "4404_localizer_taxonomy": taxonomy_payload,
        "4405_e3_deeper": deeper_payload,
        "4406_e3_blocked": blocked_payload,
        "4407_active_learning": compounds_payload,
        "4408_calibration": calibration_payload,
    }
    return {
        "localizer_deconfound": _read_localizer(localizer_payload, localizer_error),
        "localizer_taxonomy": _read_taxonomy(taxonomy_payload, taxonomy_error),
        "arc_e3": {
            "deeper_mechanics": _read_e3_partial(
                deeper_payload,
                deeper_error,
                EXP4405_PATH,
                "per_target_scorecard",
                "residual_failing_mechanic",
            ),
            "blocked_mechanics": _read_e3_partial(
                blocked_payload,
                blocked_error,
                EXP4406_PATH,
                "per_game_scorecard",
                "residual_gap_class",
            ),
        },
        "active_learning_compounding": _read_compounds(compounds_payload, compounds_error),
        "calibration_repair": _read_calibration(calibration_payload, calibration_error),
        "availability_report": capstone_aggregate_available.aggregate_available_report_gaps(
            raw_artifacts,
            _axis_specs(),
            artifact_experiment_ids={
                "4403_localizer_deconfound": 4403,
                "4404_localizer_taxonomy": 4404,
                "4405_e3_deeper": 4405,
                "4406_e3_blocked": 4406,
                "4407_active_learning": 4407,
                "4408_calibration": 4408,
            },
        ),
    }


def _gap_entry(
    gap_id: str,
    *,
    status: str,
    evidence: str,
    failure_mode: str,
    missing_discriminator: str,
    candidate_design: str,
    priority: str = "high",
) -> dict[str, Any]:
    return {
        "gap_id": gap_id,
        "status": status,
        "evidence": evidence,
        "failure_mode": failure_mode,
        "missing_discriminator": missing_discriminator,
        "candidate_design": candidate_design,
        "priority": priority,
    }


def _add_upstream_gap(entries: dict[str, dict[str, Any]], gap: Mapping[str, Any], evidence: str) -> None:
    gap_id = str(gap.get("gap_id", ""))
    if not gap_id:
        return
    entries[gap_id] = _gap_entry(
        gap_id,
        status=str(gap.get("status", "open")),
        evidence=evidence,
        failure_mode=str(gap.get("failure_mode") or gap.get("confounder") or "residual gap"),
        missing_discriminator=str(gap.get("missing_discriminator", "")),
        candidate_design=str(gap.get("candidate_design", "")),
        priority=str(gap.get("priority", "high")),
    )


def _missing_upstream_gap(gap_id: str, artifact_path: str, error: str) -> dict[str, Any]:
    return _gap_entry(
        gap_id,
        status="open",
        evidence=f"{artifact_path}; missing_or_unreadable={error}",
        failure_mode="required .407 upstream artifact was missing or unreadable",
        missing_discriminator="landed upstream evidence before registry reconciliation",
        candidate_design="rerun or recover the upstream artifact, then rerun Exp 4410",
    )


def build_gap_entries(outcomes: Mapping[str, Any]) -> list[dict[str, Any]]:
    """REQ-VERIFY-4410: collect .407 residual missing-verifier gaps."""
    entries: dict[str, dict[str, Any]] = {}
    localizer = outcomes["localizer_deconfound"]
    taxonomy = outcomes["localizer_taxonomy"]
    compounds = outcomes["active_learning_compounding"]
    calibration = outcomes["calibration_repair"]
    genuine = localizer.get("localizer_genuinely_beats_position_only") is True
    fover_status = (
        "filled (exp4403_real_intervention_localizer)"
        if genuine
        else "retired (exp4403_position_bound)"
    )
    entries[GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED] = _gap_entry(
        GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED,
        status=fover_status,
        evidence=(
            f"{EXP4403_PATH}; localizer_genuinely_beats_position_only={genuine}; "
            f"position_only_baseline_f1={localizer.get('position_only_baseline_f1')}; "
            f"{EXP4407_PATH}; localizer_compounds={compounds.get('localizer_compounds')}"
        ),
        failure_mode=(
            "real-intervention localizer beat position/content controls"
            if genuine
            else "real-intervention localizer remains position-bound or template-bound"
        ),
        missing_discriminator="content-dependent first-error signal beyond position-only controls",
        candidate_design=(
            "keep only real intervention labels with varied first-error positions and non-empty "
            "suffix redirects before reviving the localizer as a headline"
        ),
        priority="medium" if genuine else "high",
    )
    for gap in localizer.get("missing_verifier_gaps", []):
        if isinstance(gap, Mapping):
            _add_upstream_gap(entries, gap, f"{EXP4403_PATH}; real-intervention localizer gap")
    if taxonomy.get("available") is not True:
        entries[GAP_UPSTREAM_MISSING_4404] = _missing_upstream_gap(
            GAP_UPSTREAM_MISSING_4404,
            EXP4404_PATH,
            str(taxonomy.get("error", "")),
        )
    elif str(taxonomy.get("honest_verdict", "")).startswith("blocked"):
        entries[GAP_4404_LOCALIZER_TAXONOMY_BLOCKED] = _gap_entry(
            GAP_4404_LOCALIZER_TAXONOMY_BLOCKED,
            status="open",
            evidence=f"{EXP4404_PATH}; {taxonomy.get('gate_check_summary')}",
            failure_mode="typed taxonomy localizer cross-domain gate blocked",
            missing_discriminator="typed first-error taxonomy that survives gate checks",
            candidate_design="fix the typed taxonomy gate before marking localization sharpened",
        )
    for gap in compounds.get("missing_verifier_gaps", []):
        if isinstance(gap, Mapping):
            _add_upstream_gap(entries, gap, f"{EXP4407_PATH}; active-learning compounding gap")
    for gap in calibration.get("missing_verifier_gaps", []):
        if isinstance(gap, Mapping):
            _add_upstream_gap(entries, gap, f"{EXP4408_PATH}; calibration repair gap")
    known_gap_ids = set(entries)
    for domain in calibration.get("domains_at_chance", []):
        gap_id = f"GAP-4408-{str(domain).upper().replace('_', '-')}-DETECTOR-CHANCE"
        if gap_id in known_gap_ids:
            continue
        row = _domain_row(calibration, str(domain))
        entries[gap_id] = _gap_entry(
            gap_id,
            status="open",
            evidence=(
                f"{EXP4408_PATH}; domain={domain}; auroc={row.get('detection_auroc')}; "
                f"ci95={row.get('auroc_ci95')}; n={row.get('n')}"
            ),
            failure_mode=f"{domain} detector CI includes chance after deconfounding",
            missing_discriminator=f"domain-native oracle-distinct verifier feature for {domain}",
            candidate_design="add a residual wrong-mode verifier score and rerun calibration repair",
        )
    _add_arc_gap_entries(entries, outcomes["arc_e3"]["deeper_mechanics"], "4405", EXP4405_PATH)
    _add_arc_gap_entries(entries, outcomes["arc_e3"]["blocked_mechanics"], "4406", EXP4406_PATH)
    return list(entries.values())


def _add_arc_gap_entries(
    entries: dict[str, dict[str, Any]],
    section: Mapping[str, Any],
    exp_id: str,
    artifact_path: str,
) -> None:
    if section.get("available") is not True:
        entries[f"GAP-4410-MISSING-UPSTREAM-{exp_id}"] = _missing_upstream_gap(
            f"GAP-4410-MISSING-UPSTREAM-{exp_id}",
            artifact_path,
            str(section.get("error", "")),
        )
        return
    for game, row in section.get("rows", {}).items():
        if not isinstance(row, Mapping):
            continue
        residual = str(row.get("residual_failing_mechanic") or row.get("residual_gap_class") or "")
        if row.get("offline_reproduced") is True or not residual:
            continue
        target = row.get("target_level", "unknown")
        gap_id = f"GAP-{exp_id}-E3-MECHANIC-{str(game).upper()}-L{target}"
        entries[gap_id] = _gap_entry(
            gap_id,
            status="open",
            evidence=(
                f"{artifact_path}; game={game}; target_level={target}; "
                f"offline_reproduced=False; verifier_accuracy={row.get('verifier_accuracy')}; "
                f"lookahead_fidelity={row.get('lookahead_fidelity')}; residual={residual}"
            ),
            failure_mode=f"{game} L{target} remains unreproduced after mechanic tests",
            missing_discriminator=f"{game} executable reproduction rule for {residual}",
            candidate_design=(
                "convert the passing mechanic/register unit into an offline reproduce() plan "
                "and count only reproduction-gated progress"
            ),
        )


def _gap_entry_block(gap: Mapping[str, Any]) -> str:
    return (
        f"### {gap['gap_id']}: Exp 4410 .407 verifier gap update\n"
        f"# - status: {gap.get('status', 'open')}\n"
        f"# - evidence: {gap.get('evidence', '')}.\n"
        f"# - failure mode: {gap.get('failure_mode', '')}\n"
        f"# - missing discriminator: {gap.get('missing_discriminator', '')}\n"
        f"# - candidate design: {gap.get('candidate_design', '')}\n"
        f"# - priority: {gap.get('priority', 'high')}\n"
    )


def _replace_yaml_safe_marked_block(text: str, marker: str, block: str) -> str:
    start = f"# {marker}:start"
    end = f"# {marker}:end"
    replacement = f"{start}\n{block.rstrip()}\n{end}"
    if start in text and end in text:
        prefix, rest = text.split(start, 1)
        _, suffix = rest.split(end, 1)
        return f"{prefix}{replacement}{suffix}"
    return text.rstrip() + "\n\n" + replacement + "\n"


def _domain_row(cross_domain: Mapping[str, Any], domain: str) -> dict[str, Any]:
    for row in cross_domain.get("detection_by_domain", []):
        if isinstance(row, Mapping) and row.get("domain") == domain:
            return dict(row)
    return {}


def _arc_new_levels(outcomes: Mapping[str, Any]) -> int:
    return int(outcomes["arc_e3"]["deeper_mechanics"].get("new_levels_reproduced") or 0) + int(
        outcomes["arc_e3"]["blocked_mechanics"].get("new_levels_reproduced") or 0
    )


def _arc_total(outcomes: Mapping[str, Any]) -> int:
    return max(
        int(outcomes["arc_e3"]["deeper_mechanics"].get("reproducible_total_levels") or 0),
        int(outcomes["arc_e3"]["blocked_mechanics"].get("reproducible_total_levels") or 0),
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
    replay = guard.get("replayed_arc1_rule_exec", {})
    calibration = outcomes["calibration_repair"]
    code = _domain_row(calibration, "code_humaneval")
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4410": EXP4410_ARTIFACT_PATH,
            "exp4410_regression_guard_passed": bool(guard.get("regression_guard_passed")),
            "exp4410_arc1_rule_exec_vote_pass2": replay.get("vote_pass2"),
            "exp4410_arc1_rule_exec_gated_pass2": replay.get("gated_pass2"),
            "exp4410_arc1_headroom_recovered": replay.get("headroom_recovered"),
            "exp4410_arc1_vote_wins_lost": replay.get("vote_wins_lost"),
            "exp4410_v407_state": V407_STATE,
            "exp4410_localizer_genuinely_beats_position_only": outcomes[
                "localizer_deconfound"
            ].get("localizer_genuinely_beats_position_only"),
            "exp4410_localizer_compounds": outcomes["active_learning_compounding"].get(
                "localizer_compounds"
            ),
            "exp4410_detection_calibrated_multi_domain": calibration.get(
                "detection_calibrated_multi_domain"
            ),
            "exp4410_code_humaneval_detection_auroc": code.get("detection_auroc"),
            "exp4410_code_humaneval_detection_ci95": code.get("auroc_ci95"),
            "exp4410_arc_reproducible_total_levels": _arc_total(outcomes),
            "exp4410_new_levels_reproduced": _arc_new_levels(outcomes),
            "exp4410_gaps_reconciled": [gap["gap_id"] for gap in gap_entries],
        }
    )


def _ensure_v407_role(registry: dict[str, Any], outcomes: Mapping[str, Any], gap_entries: list[dict[str, Any]]) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        return
    role = {
        "role_id": V407_ROLE_ID,
        "experiment": EXP4410_ARTIFACT_PATH,
        "role": "registry_gaps_arc_hygiene_v407",
        "status": "v407_outcomes_recorded_with_gap4_guard_and_stamp_durability",
        "v407_state": V407_STATE,
        "localizer_genuinely_beats_position_only": outcomes["localizer_deconfound"].get(
            "localizer_genuinely_beats_position_only"
        ),
        "localizer_compounds": outcomes["active_learning_compounding"].get(
            "localizer_compounds"
        ),
        "detection_calibrated_multi_domain": outcomes["calibration_repair"].get(
            "detection_calibrated_multi_domain"
        ),
        "arc_reproducible_total_levels": _arc_total(outcomes),
        "new_levels_reproduced": _arc_new_levels(outcomes),
        "gaps_reconciled": [gap["gap_id"] for gap in gap_entries],
        "eval_exp_4410": EXP4410_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [
        old for old in old_roles if old.get("role_id") != V407_ROLE_ID
    ] + [role]


def _ensure_fover_detector(registry: dict[str, Any], outcomes: Mapping[str, Any]) -> None:
    entry = base._find_verifier(registry, FOVER_VERIFIER_ID)
    if entry is None:
        entry = {
            "verifier_id": FOVER_VERIFIER_ID,
            "domain": "math_reasoning",
            "version": 4,
            "kind": "ensemble",
            "eval": {},
            "status": "active",
        }
        registry.setdefault("verifiers", []).append(entry)
    localizer = outcomes["localizer_deconfound"]
    compounds = outcomes["active_learning_compounding"]
    calibration = outcomes["calibration_repair"]
    fover = _domain_row(calibration, "fover")
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4403": EXP4403_PATH,
            "eval_exp_4407": EXP4407_PATH,
            "eval_exp_4408": EXP4408_PATH,
            "eval_exp_4410": EXP4410_ARTIFACT_PATH,
            "exp4410_localizer_genuinely_beats_position_only": localizer.get(
                "localizer_genuinely_beats_position_only"
            ),
            "exp4410_beats_position_only_baseline": localizer.get(
                "beats_position_only_baseline"
            ),
            "exp4410_template_family_holdout_drop": localizer.get(
                "template_family_holdout_drop"
            ),
            "exp4410_localizer_compounds": compounds.get("localizer_compounds"),
            "exp4410_compounding_delta_ci95": compounds.get("compounding_delta_ci95"),
            "exp4410_detection_calibrated_multi_domain": calibration.get(
                "detection_calibrated_multi_domain"
            ),
            "exp4410_fover_detection_auroc": fover.get("detection_auroc"),
            "exp4410_domains_at_chance": calibration.get("domains_at_chance"),
            "exp4410_verifier_is_oracle": calibration.get("verifier_is_oracle"),
        }
    )


def _ensure_arc_registry(arc_registry: dict[str, Any], outcomes: Mapping[str, Any]) -> None:
    arc_registry["updated"] = "2026-06-18"
    arc_registry["reproducible_total_levels"] = max(
        int(arc_registry.get("reproducible_total_levels") or 0),
        _arc_total(outcomes),
    )
    arc_registry["latest_hygiene_4410"] = {
        "artifact": EXP4410_ARTIFACT_PATH,
        "reproducible_total_levels": _arc_total(outcomes),
        "new_levels_reproduced": _arc_new_levels(outcomes),
        "exp4405_new_levels_reproduced": outcomes["arc_e3"]["deeper_mechanics"].get(
            "new_levels_reproduced"
        ),
        "exp4406_new_levels_reproduced": outcomes["arc_e3"]["blocked_mechanics"].get(
            "new_levels_reproduced"
        ),
        "note": ".407 mechanic tests sharpened residual gaps but did not add reproduced ARC levels.",
    }


def registry_contains_v407(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    fover = base._find_verifier(registry, FOVER_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4410") == EXP4410_ARTIFACT_PATH
        and gap4.get("eval", {}).get("exp4410_v407_state") == V407_STATE
        and any(role.get("role_id") == V407_ROLE_ID for role in gap4.get("registry_roles", []))
        and fover
        and fover.get("eval", {}).get("eval_exp_4410") == EXP4410_ARTIFACT_PATH
    )


def arc_registry_contains_v407(arc_registry: dict[str, Any]) -> bool:
    latest = arc_registry.get("latest_hygiene_4410", {})
    return bool(
        int(arc_registry.get("reproducible_total_levels") or 0) >= 34
        and isinstance(latest, Mapping)
        and latest.get("artifact") == EXP4410_ARTIFACT_PATH
        and latest.get("new_levels_reproduced") == 0
    )


def gaps_contain_v407(gaps_text: str, gap_entries: list[dict[str, Any]]) -> bool:
    return all(f"# exp4410-{gap['gap_id'].lower()}:start" in gaps_text for gap in gap_entries)


def ensure_ledgers_record_v407(
    registry: dict[str, Any],
    gaps_text: str,
    arc_registry: dict[str, Any],
    regression_guard: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    gap_entries: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Return ledgers with .407 truth represented idempotently."""
    updated_registry = deepcopy(registry)
    updated_arc = deepcopy(arc_registry)
    _ensure_gap4_eval(updated_registry, regression_guard, outcomes, gap_entries)
    _ensure_v407_role(updated_registry, outcomes, gap_entries)
    _ensure_fover_detector(updated_registry, outcomes)
    _ensure_arc_registry(updated_arc, outcomes)
    for gap in gap_entries:
        marker = f"exp4410-{gap['gap_id'].lower()}"
        gaps_text = _replace_yaml_safe_marked_block(gaps_text, marker, _gap_entry_block(gap))

    registry_ok = registry_contains_v407(updated_registry)
    arc_ok = arc_registry_contains_v407(updated_arc)
    gaps_ok = gaps_contain_v407(gaps_text, gap_entries)
    reconciled = [gap["gap_id"] for gap in gap_entries]
    return (
        updated_registry,
        gaps_text,
        updated_arc,
        {
            "verifier_registry_reconciled": registry_ok,
            "arc_solve_registry_reconciled": arc_ok,
            "verifier_gaps_reconciled": gaps_ok,
            "registries_reconciled": registry_ok and arc_ok and gaps_ok,
            "gaps_reconciled": reconciled,
            "filled_gap_ids": [
                gap["gap_id"]
                for gap in gap_entries
                if str(gap.get("status", "")).startswith("filled")
            ],
        },
    )


def _capstone_aggregation_propagates_oracle_stamp() -> bool:
    return (
        "verifier_is_oracle" in capstone_v406_4401.REQUIRED_ARTIFACT_FIELDS
        and "verifier_is_oracle" in capstone_v406_4401.FIELD_PRINCIPLES
    )


def _capstone_aggregation_uses_available_helper() -> bool:
    return (
        capstone_v406_4401.aggregate.aggregate_available_report_gaps
        is capstone_aggregate_available.aggregate_available_report_gaps
    )


def verify_capstone_stamp_fix_durable(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4410: scan the current capstone and inspect the capstone helper."""
    capstone_path = repo_root / CAPSTONE_V406_PATH
    propagates = _capstone_aggregation_propagates_oracle_stamp()
    uses_helper = _capstone_aggregation_uses_available_helper()
    try:
        capstone = json.loads(capstone_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "capstone_stamp_fix_durable": False,
            "capstone_path": CAPSTONE_V406_PATH,
            "error": f"{type(exc).__name__}: {exc}",
            "capstone_verifier_is_oracle": None,
            "capstone_aggregation_propagates_oracle_stamp": propagates,
            "capstone_aggregation_uses_available_helper": uses_helper,
            "circular_moat_overclaim_fired": False,
            "flag_count": 0,
            "flags": [],
            "returncode": None,
        }
    command = [
        sys.executable,
        str(repo_root / "scripts" / "adversarial_verify.py"),
        "--json",
        str(capstone_path),
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
    flags = _flags_from_report(parsed)
    circular = [flag for flag in flags if flag.get("kind") == "CIRCULAR_MOAT_OVERCLAIM"]
    durable = (
        capstone.get("verifier_is_oracle") is False
        and capstone.get("verifier_is_oracle_honored") is True
        and propagates
        and uses_helper
        and not circular
        and not flags
        and completed.returncode == 0
    )
    return {
        "capstone_stamp_fix_durable": durable,
        "capstone_path": CAPSTONE_V406_PATH,
        "capstone_verifier_is_oracle": capstone.get("verifier_is_oracle"),
        "capstone_verifier_is_oracle_honored": capstone.get("verifier_is_oracle_honored"),
        "capstone_aggregation_propagates_oracle_stamp": propagates,
        "capstone_aggregation_uses_available_helper": uses_helper,
        "capstone_aggregation_source": "carnot.reporting.capstone_v406_4401",
        "circular_moat_overclaim_fired": bool(circular),
        "flag_count": len(flags),
        "flags": flags,
        "returncode": completed.returncode,
        "command": command,
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    }


def model_specs() -> dict[str, Any]:
    return {
        "method": "cached_v407_ledger_reconciliation_gap4_guard_and_stamp_audit",
        "upstream_artifacts": [
            EXP4403_PATH,
            EXP4404_PATH,
            EXP4405_PATH,
            EXP4406_PATH,
            EXP4407_PATH,
            EXP4408_PATH,
            CAPSTONE_V406_PATH,
        ],
        "gap4_guard_source": "carnot.reporting.verifier_registry_gaps_hygiene_gap4_guard_4399",
        "capstone_stamp_source": CAPSTONE_V406_PATH,
        "codex_calls": 0,
        "live_model_inference": False,
        "gguf_inference": False,
        "gpu_inference": False,
    }


def build_artifact(
    *,
    preconditions_checked: dict[str, Any],
    gap4_regression_guard: dict[str, Any],
    capstone_stamp_fix: dict[str, Any],
    v407_outcomes: dict[str, Any],
    registry_reconciliation: dict[str, Any],
    availability_report: dict[str, Any],
    reproducibility_checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    guard_ok = bool(gap4_regression_guard.get("regression_guard_passed"))
    stamp_ok = bool(capstone_stamp_fix.get("capstone_stamp_fix_durable"))
    gaps_reconciled = list(registry_reconciliation.get("gaps_reconciled", []))
    reconciled = bool(registry_reconciliation.get("registries_reconciled"))
    complete = guard_ok and stamp_ok and reconciled and bool(gaps_reconciled)
    artifact = {
        "experiment": "experiment_4410_registry_gaps_hygiene_gap4_guard",
        "schema": "carnot.experiment_4410_registry_gaps_hygiene_gap4_guard.v1",
        "honest_verdict": (
            "complete: registry_gaps_arc_reconciled_to_v407_truth_"
            f"gap4_guard_passed_{guard_ok}_capstone_stamp_fix_durable_{stamp_ok}"
            if complete
            else "blocked_v407_hygiene_incomplete"
        ),
        "regression_guard_passed": guard_ok,
        "gaps_reconciled": gaps_reconciled,
        "capstone_stamp_fix_durable": stamp_ok,
        "preconditions_checked": preconditions_checked,
        "random_seed": RANDOM_SEED,
        "v407_outcomes": v407_outcomes,
        "registry_reconciliation": registry_reconciliation,
        "gap4_regression_guard": gap4_regression_guard,
        "capstone_stamp_fix": capstone_stamp_fix,
        "availability_report": availability_report,
        "reproducibility_checksum": reproducibility_checksum,
        "model_specs": model_specs(),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "arc_registry_path": ARC_REGISTRY_PATH,
    }
    validate_artifact(artifact)
    return artifact


def _blocked_artifact(preflight: dict[str, Any], duration_s: float) -> dict[str, Any]:
    artifact = {
        "experiment": "experiment_4410_registry_gaps_hygiene_gap4_guard",
        "schema": "carnot.experiment_4410_registry_gaps_hygiene_gap4_guard.v1",
        "honest_verdict": "blocked_registry_unreadable",
        "regression_guard_passed": False,
        "gaps_reconciled": [],
        "capstone_stamp_fix_durable": False,
        "preconditions_checked": preflight,
        "random_seed": RANDOM_SEED,
        "v407_outcomes": {},
        "registry_reconciliation": {},
        "gap4_regression_guard": {},
        "capstone_stamp_fix": {},
        "availability_report": {},
        "reproducibility_checksum": "blocked:registry_unreadable",
        "model_specs": model_specs(),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 4410 terminal artifact before writing."""
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required artifact field: {field}")
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked_")):
        raise ValueError("honest_verdict must use a terminal prefix")
    for field in ("regression_guard_passed", "capstone_stamp_fix_durable"):
        if type(artifact[field]) is not bool:
            raise ValueError(f"{field} must be a BARE bool")
    if not (
        isinstance(artifact["gaps_reconciled"], list)
        or (isinstance(artifact["gaps_reconciled"], int) and not isinstance(artifact["gaps_reconciled"], bool))
    ):
        raise ValueError("gaps_reconciled must be a list or bare int")
    for field in (
        "preconditions_checked",
        "v407_outcomes",
        "registry_reconciliation",
        "gap4_regression_guard",
        "capstone_stamp_fix",
        "availability_report",
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
        raise ValueError("field_principles must match the required Exp 4410 principles")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs must cite REQ-VERIFY-4410 and SCENARIO-VERIFY-4410")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def _patch_arc_registry_text(text: str, outcomes: Mapping[str, Any]) -> str:
    if "latest_hygiene_4410:" in text:
        return text
    block = (
        "latest_hygiene_4410:\n"
        f"  artifact: {EXP4410_ARTIFACT_PATH}\n"
        f"  reproducible_total_levels: {_arc_total(outcomes)}\n"
        f"  new_levels_reproduced: {_arc_new_levels(outcomes)}\n"
        "  exp4405_new_levels_reproduced: "
        f"{outcomes['arc_e3']['deeper_mechanics'].get('new_levels_reproduced')}\n"
        "  exp4406_new_levels_reproduced: "
        f"{outcomes['arc_e3']['blocked_mechanics'].get('new_levels_reproduced')}\n"
        '  note: ".407 mechanic tests sharpened residual gaps but did not add reproduced ARC levels."\n'
    )
    return text.rstrip() + "\n\n" + block


def run_hygiene(
    repo_root: Path = REPO_ROOT,
    *,
    gap4_guard_runner: Gap4GuardRunner | None = None,
    capstone_stamp_runner: CapstoneStampRunner | None = None,
) -> dict[str, Any]:
    """Run Exp 4410 and write the terminal artifact plus reconciled ledgers."""
    if gap4_guard_runner is None:
        gap4_guard_runner = run_gap4_regression_guard
    if capstone_stamp_runner is None:
        capstone_stamp_runner = verify_capstone_stamp_fix_durable
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4410_ARTIFACT_PATH
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
    outcomes = load_v407_outcomes(repo_root)
    availability_report = dict(outcomes.get("availability_report", {}))
    gap_entries = build_gap_entries(outcomes)
    registry, gaps_text, arc_registry, summary = ensure_ledgers_record_v407(
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
    elif not arc_registry_contains_v407(yaml.safe_load(original_arc_text) or {}):
        arc_path.write_text(yaml.safe_dump(arc_registry, sort_keys=False), encoding="utf-8")

    checksum = _json_hash(
        {
            "registry": registry,
            "gaps_text_sha256": hashlib.sha256(gaps_text.encode("utf-8")).hexdigest(),
            "arc_registry": arc_registry,
        }
    )
    artifact = build_artifact(
        preconditions_checked=preflight,
        gap4_regression_guard=guard,
        capstone_stamp_fix=stamp,
        v407_outcomes=outcomes,
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
