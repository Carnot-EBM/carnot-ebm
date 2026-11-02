"""Build the Exp 4423 v408 verifier-grounded config-rule capstone.

Spec refs: REQ-CAPSTONE-4423, SCENARIO-CAPSTONE-4423.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot.reporting import capstone_aggregate_available as aggregate
from carnot.reporting import capstone_v400_4335 as base
from carnot.reporting import capstone_v405_4390 as v405


JsonDict = dict[str, Any]
LiveFlagRunner = Callable[[Path], list[dict[str, Any]]]
SummarizeRunner = Callable[[Path, Path], int]
PublicationGateRunner = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_4423_capstone_v408.json")
EXPERIMENT_ID = 4423
RANDOM_SEED = 4423
SCHEMA = "carnot.capstone_v408_4423.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4423", "SCENARIO-CAPSTONE-4423"]
REGISTRY_REL_PATH = v405.REGISTRY_REL_PATH
PUBLICATION_GATE_REL_PATH = v405.PUBLICATION_GATE_REL_PATH
PRIOR_REPRODUCIBLE_TOTAL_LEVELS = 34
FROZEN_FOVER_AUROC = 0.9131

ARC_CONFIG_RULE_STATES = {
    "new_reproducible_levels_added",
    "grounded_config_rules_no_new_reproducible_levels",
    "config_toggle_class_blocked",
}
LOCALIZER_PROGRAM_STATES = {
    "closed_position_bound_text_and_hidden",
    "off_text_signal_logged_gap",
    "localizer_program_missing_or_excluded",
}
SOVEREIGN_VERIFIER_STATES = {
    "sovereign_gap4_local_gate_holds_execution_grounded",
    "sovereign_gap4_local_gate_null",
    "sovereign_gap4_missing_or_excluded",
}


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4412_prior_capstone": Upstream(4412, Path("results/experiment_4412_capstone_v407.json")),
    "4414_config_rule": Upstream(
        4414,
        Path("results/experiment_4414_config_rule_induction_solve.json"),
    ),
    "4415_agent2world": Upstream(
        4415,
        Path("results/experiment_4415_agent2world_adaptive_e3_repair.json"),
    ),
    "4416_hidden_state": Upstream(
        4416,
        Path("results/experiment_4416_hidden_state_localizer_falsification_audit.json"),
    ),
    "4417_sovereign_gap4": Upstream(
        4417,
        Path("results/experiment_4417_gap4_local_generator_sovereign_arm.json"),
    ),
    "4418_vocabulary": Upstream(
        4418,
        Path("results/experiment_4418_config_rule_vocabulary_transfer.json"),
    ),
    "4419_detection": Upstream(
        4419,
        Path("results/experiment_4419_steerconf_code_detection_calibration_repair.json"),
    ),
}

ARTIFACT_EXPERIMENT_IDS = {
    key: upstream.experiment_id for key, upstream in DEFAULT_UPSTREAMS.items()
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "verifier_thesis_state",
    "arc_config_rule_state",
    "localizer_program_state",
    "sovereign_verifier_state",
    "config_rule_vocabulary_transfers",
    "detection_calibrated_multi_domain",
    "reproducible_total_levels",
    "publication_gate",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "cited_upstream_artifacts",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed; one honest summary string of the .408 outcome.",
    "verifier_thesis_state": (
        "One honest string: where the verifier thesis stands after .408 (config-rule "
        "grounding moved/did-not-move ARC; localizer program closed/open; sovereign "
        "GAP-4 holds/null; vocabulary compounds/null; detection calibrated/domain-bound)."
    ),
    "arc_config_rule_state": (
        "One honest string: did verifier-grounded config-rule induction + Agent2World "
        "adaptive E3 add reproducible levels / ground new config win-rules, or did the "
        "config/toggle class stay blocked (the logged search/world-model gaps)?"
    ),
    "localizer_program_state": (
        "One honest string (closed_position_bound_text_and_hidden / off_text_signal_logged_gap): "
        "whether exp4416's hidden-state audit conclusively closed the first-error-localizer "
        "program or found an off-text signal."
    ),
    "sovereign_verifier_state": (
        "One honest string: did a LOCAL open-weight generator hold the GAP-4 execution gate "
        "(the decentralization tier), or is the moat still closed-generator-bound?"
    ),
    "config_rule_vocabulary_transfers": (
        "BARE bool: the exp4418 self-learning result (does the config-rule vocabulary "
        "compound via transfer)."
    ),
    "detection_calibrated_multi_domain": (
        "BARE bool: the exp4419 result (did SteerConf rescue cross-domain code detection)."
    ),
    "reproducible_total_levels": (
        "BARE int: the ARC reproducible-level count after .408 (>= the prior 34) -- the "
        "monotonic north-star accuracy signal."
    ),
    "publication_gate": (
        "The G1-G4 publication_gate.py output (paper_ready + unmet_gates) -- the FROZEN "
        "FoVer headline gate; do NOT redefine it to show progress."
    ),
    "verifier_is_oracle": (
        "BARE bool: carried correctly so the capstone does NOT trip CIRCULAR_MOAT_OVERCLAIM "
        "(execution-grounded ARC/config/GAP-4 solves are NOT moat headlines)."
    ),
    "preconditions_checked": (
        "Records the .408 artifacts loaded (robust aggregate-available) + TRM-stand-down; "
        "pre-empts the silent-missing-resource fabrication mode."
    ),
    "random_seed": "Determinism precondition for the aggregation.",
    "reproducibility_checksum": (
        "Hash of the aggregated upstream artifacts + the gate output; lets a third party "
        "re-derive the scorecard."
    ),
    "cited_upstream_artifacts": (
        "list of {experiment_id, fields_imported, sha256} -- the audit trail that the "
        "capstone's numbers trace to real upstream measurements (G4 discipline)."
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts -- this capstone reads upstream JSON, the ARC "
        "registry, and publication_gate.py output."
    ),
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "4412_prior_capstone": [
        "localizer_state",
        "localizer_compounds",
        "detection_calibrated_multi_domain",
        "reproducible_total_levels",
        "publication_gate",
        "verifier_is_oracle",
    ],
    "4414_config_rule": [
        "new_levels_reproduced",
        "reproducible_total_levels",
        "config_win_rules_grounded",
        "per_target_scorecard",
        "preconditions_checked",
        "verifier_is_oracle",
    ],
    "4415_agent2world": [
        "new_levels_reproduced",
        "reproducible_total_levels",
        "per_target_scorecard",
        "preconditions_checked",
        "verifier_is_oracle",
    ],
    "4416_hidden_state": [
        "hidden_state_localizer_has_nonposition_signal",
        "position_only_baseline_f1",
        "localization_f1_comparison",
        "missing_verifier_gaps",
        "preconditions_checked",
        "verifier_is_oracle",
    ],
    "4417_sovereign_gap4": [
        "sovereign_gap4_gate_holds",
        "pass2_vs_vote",
        "local_generator_coverage",
        "k_consistency_details",
        "preconditions_checked",
        "verifier_is_oracle",
    ],
    "4418_vocabulary": [
        "config_rule_vocabulary_transfers",
        "preconditions_checked",
        "verifier_is_oracle",
    ],
    "4419_detection": [
        "detection_calibrated_multi_domain",
        "detection_by_domain",
        "domains_at_chance",
        "preconditions_checked",
        "verifier_is_oracle",
    ],
}

read_registry_progress = v405.read_registry_progress


def _selected_paths(root: Path) -> dict[str, Path]:
    return {key: root / upstream.path for key, upstream in DEFAULT_UPSTREAMS.items()}


def _fields_for_payload(key: str, skipped: bool) -> list[str]:
    return [] if skipped else list(IMPORTED_FIELDS[key])


def _skipped_payload(payload: JsonDict) -> JsonDict:
    skipped = dict(payload)
    skipped["flagged_adversarial"] = True
    return skipped


def _read_inputs(
    root: Path,
    live_flag_runner: LiveFlagRunner,
    summarize_runner: SummarizeRunner,
) -> tuple[dict[str, Any], list[JsonDict], list[JsonDict]]:
    raw_artifacts: dict[str, Any] = {}
    provenance: list[JsonDict] = []
    exclusions: list[JsonDict] = []

    for key, path in _selected_paths(root).items():
        upstream = DEFAULT_UPSTREAMS[key]
        if not path.exists():
            raw_artifacts[key] = None
            continue
        sha = base.sha256_file(path)
        summarize_exit_code, summarize_error = base._safe_summarize(  # noqa: SLF001
            path,
            root,
            summarize_runner,
        )
        live_flags = base._safe_live_flags(path, live_flag_runner)  # noqa: SLF001
        critical = base.live_has_critical(live_flags)
        payload: JsonDict | None = None
        parse_error = ""
        try:
            payload = base.read_json_object(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:  # pragma: no cover
            parse_error = f"{type(exc).__name__}: {exc}"

        stamped = payload.get("flagged_adversarial") is True if payload is not None else False
        skipped = stamped or critical or payload is None
        raw_artifacts[key] = (
            _skipped_payload(payload) if payload is not None and skipped else payload
        )
        provenance_row = {
            "artifact_key": key,
            "experiment_id": upstream.experiment_id,
            "path": str(upstream.path),
            "sha256": sha,
            "payload_reproducibility_checksum": base.sha_from_payload_checksum(payload or {}),
            "summarize_exit_code": summarize_exit_code,
            "summarize_error": summarize_error,
            "live_adversarial_flags": live_flags,
            "stamped_flagged_adversarial": stamped,
            "live_critical": critical,
            "parse_error": parse_error,
            "skipped": skipped,
            "fields_imported": _fields_for_payload(key, skipped),
        }
        provenance.append(provenance_row)
        if skipped:
            exclusions.append(
                {
                    "artifact_key": key,
                    "experiment_id": upstream.experiment_id,
                    "path": str(upstream.path),
                    "sha256": sha,
                    "stamped_flagged_adversarial": stamped,
                    "live_critical": critical,
                    "parse_error": parse_error,
                    "live_critical_flags": [
                        flag
                        for flag in live_flags
                        if str(flag.get("severity", "")).lower() == "critical"
                    ],
                    "reason": base._exclusion_reason(stamped, critical, parse_error),  # noqa: SLF001
                }
            )
    return raw_artifacts, provenance, exclusions


def _summary_rows(rows: list[Any]) -> list[JsonDict]:
    summarized: list[JsonDict] = []
    keys = (
        "game",
        "grounding_tier",
        "offline_reproduced",
        "new_reproduced_level",
        "prior_best_level",
        "win_rule_predicate",
        "search_blocker",
        "adaptive_tests_passed",
        "adaptive_tests_total",
        "held_out_mechanic_test_pass",
        "checkpoint_status",
        "residual_failing_behavior",
    )
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        summarized.append({key: row[key] for key in keys if key in row})
    return summarized


def _grounded_win_rules(payload: JsonDict | None) -> list[JsonDict]:
    grounded: list[JsonDict] = []
    for row in base.list_metric(payload, "config_win_rules_grounded"):
        if not isinstance(row, Mapping):
            continue
        tier = row.get("tier")
        false_positive_rate = row.get("false_positive_rate")
        if (
            isinstance(tier, int)
            and not isinstance(tier, bool)
            and tier >= 2
            and row.get("fires_on_win") is True
            and false_positive_rate == 0.0
            and row.get("literal_hardcode") is False
        ):
            grounded.append(dict(row))
    return grounded


def config_rule_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:  # pragma: no cover
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:  # pragma: no cover
        return {"status": "missing_or_excluded"}
    rules = _grounded_win_rules(payload)
    new_levels = base.int_metric(payload, "new_levels_reproduced")
    return {
        "status": "grounded" if rules else "blocked",
        "new_levels_reproduced": new_levels,
        "reproducible_total_levels_reported": base.int_metric(
            payload,
            "reproducible_total_levels",
        ),
        "grounded_win_rules": rules,
        "grounded_win_rules_count": len(rules),
        "per_target_scorecard": _summary_rows(base.list_metric(payload, "per_target_scorecard")),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def agent2world_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:  # pragma: no cover
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:  # pragma: no cover
        return {"status": "missing_or_excluded"}
    rows = _summary_rows(base.list_metric(payload, "per_target_scorecard"))
    new_levels = base.int_metric(payload, "new_levels_reproduced")
    return {
        "status": "reproduced" if new_levels > 0 else "partial",
        "new_levels_reproduced": new_levels,
        "reproducible_total_levels_reported": base.int_metric(
            payload,
            "reproducible_total_levels",
        ),
        "per_target_scorecard": rows,
        "targets_with_residual_gaps": [
            row.get("game")
            for row in rows
            if row.get("offline_reproduced") is not True and isinstance(row.get("game"), str)
        ],
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def arc_config_rule_summary(
    config_rule: Mapping[str, Any],
    agent2world: Mapping[str, Any],
) -> JsonDict:
    new_levels = int(config_rule.get("new_levels_reproduced") or 0) + int(
        agent2world.get("new_levels_reproduced") or 0
    )
    grounded_count = int(config_rule.get("grounded_win_rules_count") or 0)
    return {
        "status": "advanced" if new_levels > 0 else ("grounded" if grounded_count > 0 else "blocked"),
        "new_levels_reproduced_from_artifacts": new_levels,
        "grounded_win_rules_count": grounded_count,
        "grounded_win_rules": list(config_rule.get("grounded_win_rules") or []),
        "config_rule_induction": dict(config_rule),
        "agent2world_adaptive_e3": dict(agent2world),
        "execution_grounded": any(
            read.get("verifier_is_oracle") is True for read in (config_rule, agent2world)
        ),
    }


def decide_arc_config_rule_state(
    arc_config_rule: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> str:
    if int(arc_config_rule.get("new_levels_reproduced_from_artifacts") or 0) > 0:
        return "new_reproducible_levels_added"
    if int(registry.get("new_levels_since_prior") or 0) > 0:
        return "new_reproducible_levels_added"
    if int(arc_config_rule.get("grounded_win_rules_count") or 0) > 0:
        return "grounded_config_rules_no_new_reproducible_levels"
    return "config_toggle_class_blocked"


def localizer_program_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:  # pragma: no cover
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:  # pragma: no cover
        return {"status": "missing_or_excluded"}
    signal = base.bool_metric(payload, "hidden_state_localizer_has_nonposition_signal")
    return {
        "status": "off_text_signal" if signal is True else "closed_position_bound",
        "hidden_state_localizer_has_nonposition_signal": signal is True,
        "reported_hidden_state_localizer_has_nonposition_signal": signal,
        "position_only_baseline_f1": base.float_metric(payload, "position_only_baseline_f1"),
        "localization_f1_comparison": dict(payload.get("localization_f1_comparison", {}))
        if isinstance(payload.get("localization_f1_comparison"), Mapping)
        else {},
        "missing_verifier_gaps": base.list_metric(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def decide_localizer_program_state(read: Mapping[str, Any]) -> str:
    status = read.get("status")
    if status in {"excluded_flagged_adversarial", "missing_or_excluded"}:
        return "localizer_program_missing_or_excluded"
    if read.get("hidden_state_localizer_has_nonposition_signal") is True:
        return "off_text_signal_logged_gap"
    return "closed_position_bound_text_and_hidden"


def sovereign_verifier_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:  # pragma: no cover
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:  # pragma: no cover
        return {"status": "missing_or_excluded"}
    holds = base.bool_metric(payload, "sovereign_gap4_gate_holds")
    pass2 = payload.get("pass2_vs_vote")
    return {
        "status": "holds" if holds is True else "null",
        "sovereign_gap4_gate_holds": holds is True,
        "reported_sovereign_gap4_gate_holds": holds,
        "pass2_vs_vote": dict(pass2) if isinstance(pass2, Mapping) else {},
        "local_generator_coverage": base.float_metric(payload, "local_generator_coverage"),
        "k_consistency_details": dict(payload.get("k_consistency_details", {}))
        if isinstance(payload.get("k_consistency_details"), Mapping)
        else {},
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def decide_sovereign_verifier_state(read: Mapping[str, Any]) -> str:
    status = read.get("status")
    if status in {"excluded_flagged_adversarial", "missing_or_excluded"}:
        return "sovereign_gap4_missing_or_excluded"
    if read.get("sovereign_gap4_gate_holds") is True:
        return "sovereign_gap4_local_gate_holds_execution_grounded"
    return "sovereign_gap4_local_gate_null"


def vocabulary_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:  # pragma: no cover
        return {"status": "excluded_flagged_adversarial", "config_rule_vocabulary_transfers": False}
    if payload is None:
        return {"status": "missing_or_excluded", "config_rule_vocabulary_transfers": False}
    transfers = (
        base.bool_metric(payload, "config_rule_vocabulary_transfers") is True
        and base.bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "transfers" if transfers else "no_transfer",
        "config_rule_vocabulary_transfers": transfers,
        "reported_config_rule_vocabulary_transfers": base.bool_metric(
            payload,
            "config_rule_vocabulary_transfers",
        ),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def detection_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial", "detection_calibrated_multi_domain": False}
    if payload is None:  # pragma: no cover
        return {"status": "missing_or_excluded", "detection_calibrated_multi_domain": False}
    calibrated = (
        base.bool_metric(payload, "detection_calibrated_multi_domain") is True
        and base.bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "calibrated_multi_domain" if calibrated else "domain_bound",
        "detection_calibrated_multi_domain": calibrated,
        "reported_detection_calibrated_multi_domain": base.bool_metric(
            payload,
            "detection_calibrated_multi_domain",
        ),
        "detection_by_domain": base.list_metric(payload, "detection_by_domain"),
        "domains_at_chance": base.list_metric(payload, "domains_at_chance"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def _axis_specs() -> list[aggregate.AxisSpec]:
    return [
        aggregate.AxisSpec(
            name="arc_config_rule",
            required_keys=("4414_config_rule", "4415_agent2world"),
            verdict_fn=lambda present: arc_config_rule_summary(
                config_rule_read(present.get("4414_config_rule"), False),
                agent2world_read(present.get("4415_agent2world"), False),
            )["status"],
        ),
        aggregate.AxisSpec(
            name="localizer_program",
            required_keys=("4416_hidden_state",),
            verdict_fn=lambda present: decide_localizer_program_state(
                localizer_program_read(present.get("4416_hidden_state"), False)
            ),
        ),
        aggregate.AxisSpec(
            name="sovereign_verifier",
            required_keys=("4417_sovereign_gap4",),
            verdict_fn=lambda present: decide_sovereign_verifier_state(
                sovereign_verifier_read(present.get("4417_sovereign_gap4"), False)
            ),
        ),
        aggregate.AxisSpec(
            name="vocabulary",
            required_keys=("4418_vocabulary",),
            verdict_fn=lambda present: vocabulary_read(
                present.get("4418_vocabulary"),
                False,
            )["config_rule_vocabulary_transfers"],
        ),
        aggregate.AxisSpec(
            name="detection",
            required_keys=("4419_detection",),
            verdict_fn=lambda present: detection_read(
                present.get("4419_detection"),
                False,
            )["detection_calibrated_multi_domain"],
        ),
        aggregate.AxisSpec(
            name="prior_capstone",
            required_keys=("4412_prior_capstone",),
            verdict_fn=lambda present: base.int_metric(
                present.get("4412_prior_capstone"),
                "reproducible_total_levels",
            ),
        ),
    ]


def _publication_gate_or_gap(
    root: Path,
    runner: PublicationGateRunner,
) -> tuple[JsonDict, JsonDict, list[JsonDict]]:
    publication_gate, check = v405._publication_gate_check(root, runner)  # noqa: SLF001
    if publication_gate is not None:
        return publication_gate, check, []
    return (  # pragma: no cover
        {
            "paper_ready": False,
            "gates": {},
            "unmet_gates": ["publication_gate_unrunnable"],
            "error": str(check.get("error", "unrunnable")),
        },
        check,
        [
            {
                "axis": "publication_gate",
                "artifact_key": "publication_gate",
                "reason": "unrunnable",
            }
        ],
    )


def verifier_thesis_state(
    arc_config_rule_state: str,
    localizer_program_state: str,
    sovereign_verifier_state: str,
    config_rule_vocabulary_transfers: bool,
    detection_calibrated_multi_domain: bool,
    reproducible_total_levels: int,
) -> str:
    arc = {
        "new_reproducible_levels_added": "config_rule_new_levels",
        "grounded_config_rules_no_new_reproducible_levels": "config_rule_grounded_no_new_levels",
        "config_toggle_class_blocked": "config_toggle_blocked",
    }.get(arc_config_rule_state, "config_toggle_blocked")
    localizer = (
        "localizer_off_text_signal"
        if localizer_program_state == "off_text_signal_logged_gap"
        else "localizer_closed"
    )
    sovereign = (
        "sovereign_gap4_holds"
        if sovereign_verifier_state == "sovereign_gap4_local_gate_holds_execution_grounded"
        else "sovereign_gap4_null"
    )
    vocabulary = (
        "vocab_transfers" if config_rule_vocabulary_transfers else "vocab_no_transfer"
    )
    detection = (
        "detection_calibrated"
        if detection_calibrated_multi_domain
        else "detection_domain_bound"
    )
    return (
        f"{arc}_{localizer}_{sovereign}_{vocabulary}_{detection}_"
        f"arc_levels_{reproducible_total_levels}"
    )


def _honest_verdict(
    arc_config_rule_state: str,
    localizer_program_state: str,
    sovereign_verifier_state: str,
    config_rule_vocabulary_transfers: bool,
    detection_calibrated_multi_domain: bool,
    total_levels: int,
    publication_gate_available: bool,
    paper_ready: bool,
) -> str:
    arc = {
        "new_reproducible_levels_added": "config_rule_new_levels",
        "grounded_config_rules_no_new_reproducible_levels": "config_rule_grounded_no_new_levels",
        "config_toggle_class_blocked": "config_rule_blocked",
    }.get(arc_config_rule_state, "config_rule_blocked")
    localizer = (
        "localizer_off_text_signal"
        if localizer_program_state == "off_text_signal_logged_gap"
        else "localizer_closed"
    )
    sovereign = (
        "sovereign_gap4_holds"
        if sovereign_verifier_state == "sovereign_gap4_local_gate_holds_execution_grounded"
        else "sovereign_gap4_null"
    )
    paper = (
        "publication_ready"
        if publication_gate_available and paper_ready
        else ("publication_not_ready" if publication_gate_available else "publication_gate_gap")
    )
    vocabulary = "true" if config_rule_vocabulary_transfers else "false"
    detection = "true" if detection_calibrated_multi_domain else "false"
    return (
        f"complete: v408_{arc}_{localizer}_{sovereign}_vocab_{vocabulary}_"
        f"detection_{detection}_arc_levels_{total_levels}_{paper}"
    )


def checksum_from_inputs(
    provenance: list[Mapping[str, Any]],
    publication_gate: Mapping[str, Any],
) -> str:
    payload = {
        "publication_gate": publication_gate,
        "upstream_sha256_set": sorted(str(row["sha256"]) for row in provenance),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _field_provenance(satisfied_by: str) -> dict[str, JsonDict]:
    return {
        field: {"principle": principle, "satisfied_by": satisfied_by}
        for field, principle in FIELD_PRINCIPLES.items()
    }


def _cited_upstream_artifacts(provenance: list[JsonDict]) -> list[JsonDict]:
    cited: list[JsonDict] = []
    for row in provenance:
        if row.get("skipped") is True:
            continue
        fields = row.get("fields_imported")
        if not isinstance(fields, list) or not fields:
            continue
        cited.append(
            {
                "artifact_key": row["artifact_key"],
                "experiment_id": row["experiment_id"],
                "path": row["path"],
                "sha256": row["sha256"],
                "fields_imported": fields,
            }
        )
    return cited


def _trm_stood_down(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_s = str(key).lower()
            if key_s in {"trm_training_stood_down", "no_trm_training"} and value is True:
                return True
            if key_s == "resource" and isinstance(value, str) and "trm_training" in value:
                if payload.get("available") is True:
                    return True
            if isinstance(value, str) and "no trm training" in value.lower():
                return True
            if _trm_stood_down(value):
                return True
    if isinstance(payload, list):
        return any(_trm_stood_down(item) for item in payload)
    return False


def _trm_preconditions(clean: Mapping[str, JsonDict | None]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for key in (
        "4414_config_rule",
        "4415_agent2world",
        "4416_hidden_state",
        "4417_sovereign_gap4",
        "4418_vocabulary",
        "4419_detection",
    ):
        payload = clean.get(key)
        rows.append(
            {
                "artifact_key": key,
                "experiment_id": DEFAULT_UPSTREAMS[key].experiment_id,
                "trm_training_stood_down": _trm_stood_down(
                    payload.get("preconditions_checked") if isinstance(payload, Mapping) else None
                )
                or _trm_stood_down(payload.get("model_specs") if isinstance(payload, Mapping) else None),
            }
        )
    return rows


def _preconditions_checked(
    root: Path,
    publication_gate_check: Mapping[str, Any],
    provenance: list[JsonDict],
    registry: Mapping[str, Any],
    clean: Mapping[str, JsonDict | None],
) -> JsonDict:
    provenance_by_key = {row["artifact_key"]: row for row in provenance}
    upstreams: list[JsonDict] = []
    for key, path in _selected_paths(root).items():
        upstream = DEFAULT_UPSTREAMS[key]
        row = provenance_by_key.get(key)
        upstreams.append(
            {
                "artifact_key": key,
                "experiment_id": upstream.experiment_id,
                "path": str(upstream.path),
                "exists": path.exists(),
                "summarize_exit_code": row.get("summarize_exit_code") if row else None,
                "skipped": row.get("skipped") if row else None,
            }
        )
    trm_rows = _trm_preconditions(clean)
    return {
        "publication_gate": dict(publication_gate_check),
        "upstream_artifacts": upstreams,
        "arc_registry": dict(registry),
        "robust_aggregate_available_helper": (
            "capstone_aggregate_available.aggregate_available_report_gaps"
        ),
        "trm_training_by_artifact": trm_rows,
        "trm_training_stood_down": all(row["trm_training_stood_down"] for row in trm_rows),
    }


def _oracle_declarations(
    provenance: list[JsonDict],
    clean: Mapping[str, JsonDict | None],
) -> list[JsonDict]:
    declarations: list[JsonDict] = []
    for row in provenance:
        key = str(row["artifact_key"])
        payload = clean.get(key)
        declarations.append(
            {
                "artifact_key": key,
                "experiment_id": row["experiment_id"],
                "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
                "skipped": row["skipped"],
            }
        )
    return declarations


def _capstone_recheck_status(flags: list[dict[str, Any]]) -> JsonDict:
    circular = any(flag.get("kind") == "CIRCULAR_MOAT_OVERCLAIM" for flag in flags)
    critical = base.live_has_critical(flags)
    return {
        "status": "critical_flags" if critical else "clean",
        "flags": flags,
        "circular_moat_overclaim": circular,
    }


def build_artifact(
    root: Path = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = base.run_live_flags,
    summarize_runner: SummarizeRunner = base.run_summarize_artifact,
    publication_gate_runner: PublicationGateRunner = base.run_publication_gate,
) -> JsonDict:
    start = time.time() if started_s is None else started_s
    publication_gate, publication_gate_check, publication_gate_gaps = _publication_gate_or_gap(
        root,
        publication_gate_runner,
    )
    raw_artifacts, provenance, exclusions = _read_inputs(root, live_flag_runner, summarize_runner)
    availability_report = aggregate.aggregate_available_report_gaps(
        raw_artifacts,
        _axis_specs(),
        artifact_experiment_ids=ARTIFACT_EXPERIMENT_IDS,
    )
    skipped = {row["artifact_key"]: bool(row["skipped"]) for row in provenance}
    clean = {
        key: base.clean_payload(
            raw_artifacts.get(key) if isinstance(raw_artifacts.get(key), dict) else None,
            skipped.get(key, False),
        )
        for key in DEFAULT_UPSTREAMS
    }

    config_rule = config_rule_read(
        clean["4414_config_rule"],
        skipped.get("4414_config_rule", False),
    )
    agent2world = agent2world_read(
        clean["4415_agent2world"],
        skipped.get("4415_agent2world", False),
    )
    arc_config_rule = arc_config_rule_summary(config_rule, agent2world)
    registry = read_registry_progress(root)
    arc_state = decide_arc_config_rule_state(arc_config_rule, registry)
    localizer = localizer_program_read(
        clean["4416_hidden_state"],
        skipped.get("4416_hidden_state", False),
    )
    localizer_state = decide_localizer_program_state(localizer)
    sovereign = sovereign_verifier_read(
        clean["4417_sovereign_gap4"],
        skipped.get("4417_sovereign_gap4", False),
    )
    sovereign_state = decide_sovereign_verifier_state(sovereign)
    vocabulary = vocabulary_read(
        clean["4418_vocabulary"],
        skipped.get("4418_vocabulary", False),
    )
    detection = detection_read(
        clean["4419_detection"],
        skipped.get("4419_detection", False),
    )
    vocabulary_transfers = vocabulary.get("config_rule_vocabulary_transfers") is True
    calibrated = detection.get("detection_calibrated_multi_domain") is True
    total_levels = int(registry.get("reproducible_total_levels") or 0)
    paper_ready = bool(publication_gate.get("paper_ready"))
    publication_available = bool(publication_gate_check.get("runnable"))
    thesis = verifier_thesis_state(
        arc_state,
        localizer_state,
        sovereign_state,
        vocabulary_transfers,
        calibrated,
        total_levels,
    )
    end = time.time() if now_s is None else now_s
    per_axis_gaps = list(availability_report.get("missing_upstream_artifacts", []))
    per_axis_gaps.extend(publication_gate_gaps)

    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "random_seed": RANDOM_SEED,
        "duration_s": round(end - start, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(
            arc_state,
            localizer_state,
            sovereign_state,
            vocabulary_transfers,
            calibrated,
            total_levels,
            publication_available,
            paper_ready,
        ),
        "verifier_thesis_state": thesis,
        "arc_config_rule_state": arc_state,
        "arc_config_rule": arc_config_rule,
        "localizer_program_state": localizer_state,
        "localizer_program": localizer,
        "sovereign_verifier_state": sovereign_state,
        "sovereign_verifier": sovereign,
        "config_rule_vocabulary_transfers": vocabulary_transfers,
        "config_rule_vocabulary": vocabulary,
        "detection_calibrated_multi_domain": calibrated,
        "detection_calibration": detection,
        "reproducible_total_levels": total_levels,
        "arc_reproducible_progress": registry,
        "publication_gate": publication_gate,
        "paper_ready": paper_ready,
        "unmet_gates": base.list_metric(publication_gate, "unmet_gates"),
        "frozen_fover_auroc": FROZEN_FOVER_AUROC,
        "verifier_is_oracle": False,
        "verifier_is_oracle_honored": True,
        "upstream_oracle_declarations": _oracle_declarations(provenance, clean),
        "cited_upstream_artifacts": _cited_upstream_artifacts(provenance),
        "preconditions_checked": _preconditions_checked(
            root,
            publication_gate_check,
            provenance,
            registry,
            clean,
        ),
        "per_axis_gaps": per_axis_gaps,
        "flagged_artifacts_excluded": exclusions,
        "availability_report": availability_report,
        "upstream_provenance": provenance,
        "upstream_sha256_set": sorted(str(row["sha256"]) for row in provenance),
        "publication_gate_checksum": hashlib.sha256(
            json.dumps(publication_gate, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "reproducibility_checksum": checksum_from_inputs(provenance, publication_gate),
        "capstone_live_adversarial_recheck": {"status": "not_run_until_write"},
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance("aggregation logic"),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:  # pragma: no cover
            raise ValueError(f"missing required field: {field}")  # pragma: no cover
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict must be terminal-prefixed")  # pragma: no cover
    if artifact.get("arc_config_rule_state") not in ARC_CONFIG_RULE_STATES:
        raise ValueError("arc_config_rule_state is not recognized")  # pragma: no cover
    if artifact.get("localizer_program_state") not in LOCALIZER_PROGRAM_STATES:
        raise ValueError("localizer_program_state is not recognized")  # pragma: no cover
    if artifact.get("sovereign_verifier_state") not in SOVEREIGN_VERIFIER_STATES:
        raise ValueError("sovereign_verifier_state is not recognized")  # pragma: no cover
    if not isinstance(artifact.get("config_rule_vocabulary_transfers"), bool):
        raise ValueError("config_rule_vocabulary_transfers must be a bare bool")  # pragma: no cover
    if not isinstance(artifact.get("detection_calibrated_multi_domain"), bool):
        raise ValueError("detection_calibrated_multi_domain must be a bare bool")  # pragma: no cover
    total = artifact.get("reproducible_total_levels")
    if not isinstance(total, int) or isinstance(total, bool):
        raise ValueError("reproducible_total_levels must be a bare int")  # pragma: no cover
    if not isinstance(artifact.get("verifier_thesis_state"), str):
        raise ValueError("verifier_thesis_state must be a string")  # pragma: no cover
    if not isinstance(artifact.get("publication_gate"), Mapping):
        raise ValueError("publication_gate must be an object")  # pragma: no cover
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle must be bare false")  # pragma: no cover
    if not isinstance(artifact.get("cited_upstream_artifacts"), list):
        raise ValueError("cited_upstream_artifacts must be a list")  # pragma: no cover
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        raise ValueError("preconditions_checked must be an object")  # pragma: no cover
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed does not match experiment")  # pragma: no cover
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")  # pragma: no cover
    checksum = str(artifact.get("reproducibility_checksum", "")).removeprefix("sha256:")
    if not base.is_sha256(checksum):
        raise ValueError("reproducibility_checksum must be sha256-prefixed")  # pragma: no cover
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match")  # pragma: no cover
    provenance = artifact.get("upstream_provenance")
    if not isinstance(provenance, list):
        raise ValueError("upstream_provenance must be a list")  # pragma: no cover
    for row in provenance:
        if not isinstance(row, Mapping):  # pragma: no cover
            raise ValueError("upstream provenance row must be an object")  # pragma: no cover
        if not base.is_sha256(row.get("sha256")):
            raise ValueError("upstream provenance row has invalid sha256")  # pragma: no cover
        if row.get("skipped") is True and row.get("fields_imported") != []:
            raise ValueError("skipped upstreams must not import fields")  # pragma: no cover
    expected = checksum_from_inputs(provenance, artifact["publication_gate"])
    if artifact.get("reproducibility_checksum") != expected:
        raise ValueError("reproducibility_checksum does not match inputs")  # pragma: no cover


def write_artifact(
    root: Path = REPO_ROOT,
    *,
    output_path: Path = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = base.run_live_flags,
    summarize_runner: SummarizeRunner = base.run_summarize_artifact,
    publication_gate_runner: PublicationGateRunner = base.run_publication_gate,
    capstone_live_flag_runner: LiveFlagRunner = base.run_live_flags,
) -> Path:
    artifact = build_artifact(
        root,
        started_s=started_s,
        now_s=now_s,
        live_flag_runner=live_flag_runner,
        summarize_runner=summarize_runner,
        publication_gate_runner=publication_gate_runner,
    )
    validate_artifact(artifact)
    path = root / output_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact["capstone_live_adversarial_recheck"] = _capstone_recheck_status(
        capstone_live_flag_runner(path)
    )
    validate_artifact(artifact)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parse_args() -> JsonDict:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_REL_PATH)
    args = parser.parse_args()
    return {"output": args.output}


def main() -> int:  # pragma: no cover
    args = _parse_args()
    output = write_artifact(REPO_ROOT, output_path=args["output"])
    print(output.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
