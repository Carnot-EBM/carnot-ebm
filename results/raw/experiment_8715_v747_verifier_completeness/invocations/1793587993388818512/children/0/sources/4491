"""Build the Exp 4390 v405 verifier detector scorecard capstone.

Spec refs: REQ-CAPSTONE-4390, SCENARIO-CAPSTONE-4390.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting import capstone_aggregate_available as aggregate
from carnot.reporting import capstone_v400_4335 as base


JsonDict = dict[str, Any]
LiveFlagRunner = Callable[[Path], list[dict[str, Any]]]
SummarizeRunner = Callable[[Path, Path], int]
PublicationGateRunner = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_4390_capstone_v405.json")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
PUBLICATION_GATE_REL_PATH = Path("scripts/publication_gate.py")
EXPERIMENT_ID = 4390
RANDOM_SEED = 4390
SCHEMA = "carnot.capstone_v405_4390.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4390", "SCENARIO-CAPSTONE-4390"]
PRIOR_REPRODUCIBLE_TOTAL_LEVELS = 34
PRIOR_REPRODUCIBLE_TOTAL_GAMES = 17
BLOCKED_PUBLICATION_GATE_CHECKSUM = hashlib.sha256(
    b"blocked_publication_gate_unrunnable_v405"
).hexdigest()
EMPTY_UPSTREAM_CHECKSUM = hashlib.sha256(b"no_v405_upstream_artifacts").hexdigest()

ACTIONABLE_STATES = {
    "actionable_localization_and_abstention",
    "detects_but_not_actionable",
    "open",
}


def _thesis_states() -> set[str]:
    action_parts = {
        "actionable_localization_and_abstention": "detector_actionable",
        "detects_but_not_actionable": "detector_detects_but_not_actionable",
        "open": "detector_actionability_open",
    }
    compound_parts = {
        True: "detector_compounds",
        False: "detector_compounding_open",
    }
    generalize_parts = {
        True: "detection_generalizes",
        False: "detection_domain_bound",
    }
    states = {"blocked_publication_gate_unrunnable"}
    for action in action_parts.values():
        for compound in compound_parts.values():
            for generalize in generalize_parts.values():
                states.add(f"{action}_{compound}_{generalize}")
    return states


THESIS_STATES = _thesis_states()


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4379_capstone_v404": Upstream(4379, Path("results/experiment_4379_capstone_v404.json")),
    "4381_localization": Upstream(
        4381,
        Path("results/experiment_4381_biprm_detector_localization_abstention.json"),
    ),
    "4382_skeptic": Upstream(
        4382,
        Path("results/experiment_4382_detector_localization_skeptic_proof.json"),
    ),
    "4383_e3_deeper": Upstream(
        4383,
        Path("results/experiment_4383_e3_deeper_high_headroom_lookahead.json"),
    ),
    "4384_e3_blocked": Upstream(
        4384,
        Path("results/experiment_4384_e3_blocked_mechanic_tails_ar25_ka59_ft09.json"),
    ),
    "4385_compounds": Upstream(
        4385,
        Path("results/experiment_4385_detector_self_learning_compounds.json"),
    ),
    "4386_generalization": Upstream(
        4386,
        Path("results/experiment_4386_cross_domain_detection_generalization.json"),
    ),
}

ARTIFACT_EXPERIMENT_IDS = {
    key: upstream.experiment_id for key, upstream in DEFAULT_UPSTREAMS.items()
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "detector_actionable_state",
    "detector_compounds",
    "detector_generalizes_cross_domain",
    "reproducible_total_levels",
    "verifier_thesis_state",
    "publication_gate",
    "verifier_is_oracle",
    "cited_upstream_artifacts",
    "preconditions_checked",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. The .405 scorecard string (whether the detector "
        "became actionable, whether it compounds, whether detection generalizes, "
        "the ARC reproducible-total)."
    ),
    "detector_actionable_state": (
        "One of actionable_localization_and_abstention / detects_but_not_actionable "
        "/ open -- the headline decision: did the one ALIVE oracle-distinct vehicle "
        "graduate from 'beats chance' (exp4375) to 'localizes the earliest error + "
        "abstains usefully', genuinely (not position/leak/overfit)?"
    ),
    "detector_compounds": (
        "BARE bool: did the detector self-improve as labeled traces accumulate "
        "(exp4385) -- the mandated continuous-self-learning reading on the live "
        "vehicle?"
    ),
    "detector_generalizes_cross_domain": (
        "BARE bool: did detection beat chance beyond FoVer (exp4386) -- the "
        "verifier-domain-expansion reading -- or is it domain-bound (logged gaps)?"
    ),
    "reproducible_total_levels": (
        "BARE int: the ARC reproducible-level count after .405 (>= the prior 34) "
        "-- the monotonic north-star accuracy signal."
    ),
    "verifier_thesis_state": (
        "One honest string summarizing where the verifier-moat thesis stands "
        "after .405 (detector-actionable / detector-detects-but-not-actionable / "
        "detector-compounds / detection-generalizes / detection-domain-bound / etc.)."
    ),
    "publication_gate": (
        "G1-G4 via publication_gate.py (paper_ready + unmet_gates) -- the stable "
        "finish line (north-star \u00a72)."
    ),
    "verifier_is_oracle": (
        "BARE bool=false for the oracle-distinct detector reads (the durable exp4355 "
        "stamp fix) -- so this capstone does NOT trip CIRCULAR_MOAT_OVERCLAIM."
    ),
    "cited_upstream_artifacts": (
        "list of {experiment_id, fields_imported} -- the audit trail so the "
        "capstone numbers trace to real measurements."
    ),
    "preconditions_checked": (
        "Records the upstream-artifact + publication_gate availability; "
        "pre-empts the silent-missing-resource fabrication mode."
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts -- this capstone reads upstream JSON, "
        "the ARC registry, and publication_gate.py output."
    ),
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "4379_capstone_v404": [
        "detector_beats_chance",
        "detector",
        "reproducible_total_levels",
        "verifier_is_oracle",
    ],
    "4381_localization": [
        "detector_localization_actionable",
        "localization_f1_by_direction",
        "localization_delta_ci95",
        "abstention_curve",
        "n_traces",
        "n_error_traces",
        "verifier_is_oracle",
    ],
    "4382_skeptic": [
        "localization_win_is_genuine",
        "status",
        "blocked_at_layer",
        "gate_check_summary",
        "gates_evaluated",
    ],
    "4383_e3_deeper": [
        "new_levels_reproduced",
        "reproducible_total_levels",
        "per_target_scorecard",
        "verifier_is_oracle",
    ],
    "4384_e3_blocked": [
        "new_levels_reproduced",
        "reproducible_total_levels",
        "per_game_scorecard",
        "verifier_is_oracle",
    ],
    "4385_compounds": [
        "detector_compounds",
        "learning_curve",
        "compounding_delta_ci95",
        "fresh_headroom_direction",
        "no_learning_baseline",
        "positive_control_passed",
        "verifier_is_oracle",
    ],
    "4386_generalization": [
        "detector_generalizes_cross_domain",
        "detection_by_domain",
        "domains_at_chance",
        "unavailable_domains",
        "verifier_is_oracle",
    ],
}


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
            path, root, summarize_runner
        )
        live_flags = base._safe_live_flags(path, live_flag_runner)  # noqa: SLF001
        critical = base.live_has_critical(live_flags)
        payload: JsonDict | None = None
        parse_error = ""
        try:
            payload = base.read_json_object(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
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


def prior_detector_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    detector = payload.get("detector")
    detector_positive = (
        base.bool_metric(payload, "detector_beats_chance") is True
        and base.bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "detector_positive" if detector_positive else "detector_null",
        "detector_beats_chance": detector_positive,
        "reported_detector_beats_chance": base.bool_metric(payload, "detector_beats_chance"),
        "detector": dict(detector) if isinstance(detector, Mapping) else {},
        "reproducible_total_levels_reported": base.int_metric(
            payload,
            "reproducible_total_levels",
        ),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def localization_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    actionable = (
        base.bool_metric(payload, "detector_localization_actionable") is True
        and base.bool_metric(payload, "verifier_is_oracle") is False
    )
    abstention = payload.get("abstention_curve")
    return {
        "status": "actionable" if actionable else "not_actionable",
        "detector_localization_actionable": actionable,
        "reported_detector_localization_actionable": base.bool_metric(
            payload,
            "detector_localization_actionable",
        ),
        "localization_f1_by_direction": dict(
            payload.get("localization_f1_by_direction", {})
        ),
        "localization_delta_ci95": base.list_metric(payload, "localization_delta_ci95"),
        "abstention_curve": dict(abstention) if isinstance(abstention, Mapping) else {},
        "n_traces": base.int_metric(payload, "n_traces"),
        "n_error_traces": base.int_metric(payload, "n_error_traces"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def skeptic_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    gates = base.list_metric(payload, "gates_evaluated")
    all_gates_passed = bool(gates) and all(
        isinstance(row, Mapping) and row.get("passed") is True for row in gates
    )
    status = base.str_metric(payload, "status")
    inferred_genuine = status in {"success", "complete", "passed"} and all_gates_passed
    genuine = base.bool_metric(payload, "localization_win_is_genuine") is True or inferred_genuine
    return {
        "status": "genuine" if genuine else "not_genuine",
        "localization_win_is_genuine": genuine,
        "reported_localization_win_is_genuine": base.bool_metric(
            payload,
            "localization_win_is_genuine",
        ),
        "artifact_status": status,
        "blocked_at_layer": base.str_metric(payload, "blocked_at_layer"),
        "gate_check_summary": base.str_metric(payload, "gate_check_summary"),
        "gates_evaluated": gates,
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def decide_detector_actionable_state(
    prior_detector: Mapping[str, Any],
    localization: Mapping[str, Any],
    skeptic: Mapping[str, Any],
) -> str:
    actionable = (
        localization.get("detector_localization_actionable") is True
        and skeptic.get("localization_win_is_genuine") is True
    )
    if actionable:
        return "actionable_localization_and_abstention"
    localization_evaluated = localization.get("status") in {"actionable", "not_actionable"}
    skeptic_evaluated = skeptic.get("status") in {"genuine", "not_genuine"}
    if (
        prior_detector.get("detector_beats_chance") is True
        and localization_evaluated
        and skeptic_evaluated
    ):
        return "detects_but_not_actionable"
    return "open"


def self_learning_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    compounds = (
        base.bool_metric(payload, "detector_compounds") is True
        and base.bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "compounds" if compounds else "open",
        "detector_compounds": compounds,
        "reported_detector_compounds": base.bool_metric(payload, "detector_compounds"),
        "learning_curve": base.list_metric(payload, "learning_curve"),
        "compounding_delta_ci95": base.list_metric(payload, "compounding_delta_ci95"),
        "fresh_headroom_direction": base.str_metric(payload, "fresh_headroom_direction"),
        "no_learning_baseline": base.float_metric(payload, "no_learning_baseline"),
        "positive_control_passed": base.bool_metric(payload, "positive_control_passed"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def generalization_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    generalizes = (
        base.bool_metric(payload, "detector_generalizes_cross_domain") is True
        and base.bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "generalizes" if generalizes else "domain_bound",
        "detector_generalizes_cross_domain": generalizes,
        "reported_detector_generalizes_cross_domain": base.bool_metric(
            payload,
            "detector_generalizes_cross_domain",
        ),
        "detection_by_domain": base.list_metric(payload, "detection_by_domain"),
        "domains_at_chance": base.list_metric(payload, "domains_at_chance"),
        "unavailable_domains": base.list_metric(payload, "unavailable_domains"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def arc_progress_read(payload: JsonDict | None, skipped: bool, rows_field: str) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    rows = base.list_metric(payload, rows_field)
    cleaned: list[JsonDict] = []
    games: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        item = dict(row)
        game = item.get("game")
        level = item.get("new_reproduced_level")
        if (
            item.get("offline_reproduced") is True
            and isinstance(game, str)
            and isinstance(level, int)
            and not isinstance(level, bool)
            and level > 0
        ):
            games.append(game)
        cleaned.append(item)
    new_levels = base.int_metric(payload, "new_levels_reproduced")
    return {
        "status": "reproduced" if new_levels > 0 else "partial",
        "new_levels_reproduced": new_levels,
        "games_with_new_reproducible_levels": games,
        "reproducible_total_levels_reported": base.int_metric(
            payload,
            "reproducible_total_levels",
        ),
        rows_field: cleaned,
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def arc_e3_summary(deeper: Mapping[str, Any], blocked: Mapping[str, Any]) -> JsonDict:
    reads = [deeper, blocked]
    new_levels = sum(int(read.get("new_levels_reproduced") or 0) for read in reads)
    games: set[str] = set()
    for read in reads:
        for game in read.get("games_with_new_reproducible_levels") or []:
            if isinstance(game, str):
                games.add(game)
    return {
        "status": "advanced" if new_levels > 0 else "partial",
        "new_levels_reproduced_from_artifacts": new_levels,
        "games_with_new_reproducible_levels": sorted(games),
        "execution_grounded": any(read.get("verifier_is_oracle") is True for read in reads),
        "deeper_high_headroom": dict(deeper),
        "blocked_mechanics": dict(blocked),
    }


def read_registry_progress(root: Path) -> JsonDict:
    path = root / REGISTRY_REL_PATH
    if not path.exists():
        return {
            "status": "missing",
            "path": str(REGISTRY_REL_PATH),
            "reproducible_total_levels": 0,
            "reproducible_total_games": 0,
            "prior_reproducible_total_levels": PRIOR_REPRODUCIBLE_TOTAL_LEVELS,
            "prior_reproducible_total_games": PRIOR_REPRODUCIBLE_TOTAL_GAMES,
            "new_levels_since_prior": 0,
            "new_games_since_prior": 0,
        }
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return {
            "status": "unparseable",
            "path": str(REGISTRY_REL_PATH),
            "reproducible_total_levels": 0,
            "reproducible_total_games": 0,
            "prior_reproducible_total_levels": PRIOR_REPRODUCIBLE_TOTAL_LEVELS,
            "prior_reproducible_total_games": PRIOR_REPRODUCIBLE_TOTAL_GAMES,
            "new_levels_since_prior": 0,
            "new_games_since_prior": 0,
            "error": str(exc),
        }
    if not isinstance(payload, Mapping):
        return {
            "status": "unparseable",
            "path": str(REGISTRY_REL_PATH),
            "reproducible_total_levels": 0,
            "reproducible_total_games": 0,
            "prior_reproducible_total_levels": PRIOR_REPRODUCIBLE_TOTAL_LEVELS,
            "prior_reproducible_total_games": PRIOR_REPRODUCIBLE_TOTAL_GAMES,
            "new_levels_since_prior": 0,
            "new_games_since_prior": 0,
            "error": "non-mapping registry",
        }
    levels = payload.get("reproducible_total_levels")
    games = payload.get("reproducible_total_games")
    if not isinstance(levels, int) or isinstance(levels, bool):
        levels = 0
    if not isinstance(games, int) or isinstance(games, bool):
        games = 0
    return {
        "status": "loaded",
        "path": str(REGISTRY_REL_PATH),
        "reproducible_total_levels": levels,
        "reproducible_total_games": games,
        "prior_reproducible_total_levels": PRIOR_REPRODUCIBLE_TOTAL_LEVELS,
        "prior_reproducible_total_games": PRIOR_REPRODUCIBLE_TOTAL_GAMES,
        "new_levels_since_prior": max(0, levels - PRIOR_REPRODUCIBLE_TOTAL_LEVELS),
        "new_games_since_prior": max(0, games - PRIOR_REPRODUCIBLE_TOTAL_GAMES),
    }


def _publication_gate_check(
    root: Path,
    runner: PublicationGateRunner,
) -> tuple[JsonDict | None, JsonDict]:
    path = root / PUBLICATION_GATE_REL_PATH
    check: JsonDict = {
        "path": str(PUBLICATION_GATE_REL_PATH),
        "exists": path.exists(),
        "runnable": False,
    }
    if not path.exists():
        check["error"] = "missing"
        return None, check
    try:
        payload = runner(root)
    except Exception as exc:
        check["error"] = f"{type(exc).__name__}: {exc}"
        return None, check
    if not isinstance(payload, dict):
        check["error"] = "publication_gate returned non-object"
        return None, check
    check["runnable"] = True
    check["paper_ready"] = bool(payload.get("paper_ready"))
    check["unmet_gates"] = base.list_metric(payload, "unmet_gates")
    return payload, check


def _axis_specs() -> list[aggregate.AxisSpec]:
    return [
        aggregate.AxisSpec(
            name="detector_actionability",
            required_keys=("4379_capstone_v404", "4381_localization", "4382_skeptic"),
            verdict_fn=lambda present: decide_detector_actionable_state(
                prior_detector_read(present.get("4379_capstone_v404"), False),
                localization_read(present.get("4381_localization"), False),
                skeptic_read(present.get("4382_skeptic"), False),
            ),
        ),
        aggregate.AxisSpec(
            name="self_learning",
            required_keys=("4385_compounds",),
            verdict_fn=lambda present: self_learning_read(
                present.get("4385_compounds"),
                False,
            )["detector_compounds"],
        ),
        aggregate.AxisSpec(
            name="generalization",
            required_keys=("4386_generalization",),
            verdict_fn=lambda present: generalization_read(
                present.get("4386_generalization"),
                False,
            )["detector_generalizes_cross_domain"],
        ),
        aggregate.AxisSpec(
            name="arc",
            required_keys=("4383_e3_deeper", "4384_e3_blocked"),
            verdict_fn=lambda present: arc_e3_summary(
                arc_progress_read(
                    present.get("4383_e3_deeper"),
                    False,
                    "per_target_scorecard",
                ),
                arc_progress_read(
                    present.get("4384_e3_blocked"),
                    False,
                    "per_game_scorecard",
                ),
            )["new_levels_reproduced_from_artifacts"],
        ),
    ]


def verifier_thesis_state(
    detector_actionable_state: str,
    detector_compounds: bool,
    detector_generalizes_cross_domain: bool,
) -> str:
    action_parts = {
        "actionable_localization_and_abstention": "detector_actionable",
        "detects_but_not_actionable": "detector_detects_but_not_actionable",
        "open": "detector_actionability_open",
    }
    compound = "detector_compounds" if detector_compounds else "detector_compounding_open"
    generalize = (
        "detection_generalizes"
        if detector_generalizes_cross_domain
        else "detection_domain_bound"
    )
    return f"{action_parts.get(detector_actionable_state, 'detector_actionability_open')}_{compound}_{generalize}"


def _honest_verdict(
    detector_actionable_state: str,
    detector_compounds: bool,
    detector_generalizes_cross_domain: bool,
    total_levels: int,
    paper_ready: bool,
) -> str:
    paper = "publication_ready" if paper_ready else "publication_not_ready"
    compounds = "true" if detector_compounds else "false"
    generalizes = "true" if detector_generalizes_cross_domain else "false"
    return (
        f"complete: v405_detector_{detector_actionable_state}_compounds_{compounds}_"
        f"generalizes_{generalizes}_arc_levels_{total_levels}_{paper}"
    )


def checksum_from_provenance(provenance: list[Mapping[str, Any]]) -> str:
    if not provenance:
        return EMPTY_UPSTREAM_CHECKSUM
    shas = sorted(str(row["sha256"]) for row in provenance)
    return hashlib.sha256("\n".join(shas).encode("utf-8")).hexdigest()


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


def _preconditions_checked(
    root: Path,
    publication_gate_check: Mapping[str, Any],
    provenance: list[JsonDict],
    registry: Mapping[str, Any],
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
    return {
        "publication_gate": dict(publication_gate_check),
        "upstream_artifacts": upstreams,
        "arc_registry": dict(registry),
    }


def _blocked_publication_gate_artifact(
    started_s: float,
    now_s: float | None,
    publication_gate_check: Mapping[str, Any],
) -> JsonDict:
    end = time.time() if now_s is None else now_s
    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "random_seed": RANDOM_SEED,
        "duration_s": round(end - started_s, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": "blocked_publication_gate_unrunnable",
        "detector_actionable_state": "open",
        "detector_actionability": {"status": "blocked_publication_gate_unrunnable"},
        "detector_compounds": False,
        "self_learning": {"status": "blocked_publication_gate_unrunnable"},
        "detector_generalizes_cross_domain": False,
        "generalization": {"status": "blocked_publication_gate_unrunnable"},
        "reproducible_total_levels": 0,
        "arc_reproducible_progress": {"status": "not_checked", "path": str(REGISTRY_REL_PATH)},
        "arc_e3_outcomes": {"status": "not_checked"},
        "verifier_thesis_state": "blocked_publication_gate_unrunnable",
        "publication_gate": {
            "paper_ready": False,
            "unmet_gates": ["publication_gate_unrunnable"],
            "error": str(publication_gate_check.get("error", "unrunnable")),
        },
        "paper_ready": False,
        "unmet_gates": ["publication_gate_unrunnable"],
        "verifier_is_oracle": False,
        "verifier_is_oracle_honored": True,
        "cited_upstream_artifacts": [],
        "preconditions_checked": {
            "publication_gate": dict(publication_gate_check),
            "upstream_artifacts": [],
            "arc_registry": {"status": "not_checked", "path": str(REGISTRY_REL_PATH)},
        },
        "per_axis_gaps": [],
        "flagged_artifacts_excluded": [],
        "availability_report": {},
        "upstream_provenance": [],
        "upstream_sha256_set": [],
        "reproducibility_checksum": BLOCKED_PUBLICATION_GATE_CHECKSUM,
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance("blocked precondition"),
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
    publication_gate, publication_gate_check = _publication_gate_check(
        root,
        publication_gate_runner,
    )
    if publication_gate is None:
        return _blocked_publication_gate_artifact(start, now_s, publication_gate_check)

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

    prior = prior_detector_read(
        clean["4379_capstone_v404"],
        skipped.get("4379_capstone_v404", False),
    )
    localization = localization_read(
        clean["4381_localization"],
        skipped.get("4381_localization", False),
    )
    skeptic = skeptic_read(clean["4382_skeptic"], skipped.get("4382_skeptic", False))
    actionable_state = decide_detector_actionable_state(prior, localization, skeptic)
    actionability = {
        "status": actionable_state,
        "prior_detector": prior,
        "localization": localization,
        "skeptic_validation": skeptic,
    }

    self_learning = self_learning_read(
        clean["4385_compounds"],
        skipped.get("4385_compounds", False),
    )
    generalization = generalization_read(
        clean["4386_generalization"],
        skipped.get("4386_generalization", False),
    )
    deeper = arc_progress_read(
        clean["4383_e3_deeper"],
        skipped.get("4383_e3_deeper", False),
        "per_target_scorecard",
    )
    blocked = arc_progress_read(
        clean["4384_e3_blocked"],
        skipped.get("4384_e3_blocked", False),
        "per_game_scorecard",
    )
    arc_e3 = arc_e3_summary(deeper, blocked)
    registry = read_registry_progress(root)
    detector_compounds = self_learning.get("detector_compounds") is True
    detector_generalizes = generalization.get("detector_generalizes_cross_domain") is True
    total_levels = int(registry.get("reproducible_total_levels") or 0)
    paper_ready = bool(publication_gate.get("paper_ready"))
    thesis = verifier_thesis_state(actionable_state, detector_compounds, detector_generalizes)
    end = time.time() if now_s is None else now_s

    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "random_seed": RANDOM_SEED,
        "duration_s": round(end - start, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(
            actionable_state,
            detector_compounds,
            detector_generalizes,
            total_levels,
            paper_ready,
        ),
        "detector_actionable_state": actionable_state,
        "detector_actionability": actionability,
        "detector_compounds": detector_compounds,
        "self_learning": self_learning,
        "detector_generalizes_cross_domain": detector_generalizes,
        "generalization": generalization,
        "reproducible_total_levels": total_levels,
        "arc_reproducible_progress": registry,
        "arc_e3_outcomes": arc_e3,
        "verifier_thesis_state": thesis,
        "publication_gate": publication_gate,
        "paper_ready": paper_ready,
        "unmet_gates": base.list_metric(publication_gate, "unmet_gates"),
        "verifier_is_oracle": False,
        "verifier_is_oracle_honored": True,
        "cited_upstream_artifacts": _cited_upstream_artifacts(provenance),
        "preconditions_checked": _preconditions_checked(
            root,
            publication_gate_check,
            provenance,
            registry,
        ),
        "per_axis_gaps": list(availability_report.get("missing_upstream_artifacts", [])),
        "flagged_artifacts_excluded": exclusions,
        "availability_report": availability_report,
        "upstream_provenance": provenance,
        "upstream_sha256_set": sorted(str(row["sha256"]) for row in provenance),
        "reproducibility_checksum": checksum_from_provenance(provenance),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance("aggregation logic"),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact.get("honest_verdict")
    if verdict != "blocked_publication_gate_unrunnable":
        if not isinstance(verdict, str) or not verdict.startswith(
            ("complete:", "success:", "passed:", "shipped:", "blocked:")
        ):
            raise ValueError("honest_verdict must be terminal-prefixed")
    if artifact.get("detector_actionable_state") not in ACTIONABLE_STATES:
        raise ValueError("detector_actionable_state is not recognized")
    if not isinstance(artifact.get("detector_compounds"), bool):
        raise ValueError("detector_compounds must be a bare bool")
    if not isinstance(artifact.get("detector_generalizes_cross_domain"), bool):
        raise ValueError("detector_generalizes_cross_domain must be a bare bool")
    if not isinstance(artifact.get("reproducible_total_levels"), int) or isinstance(
        artifact.get("reproducible_total_levels"),
        bool,
    ):
        raise ValueError("reproducible_total_levels must be a bare int")
    if artifact.get("verifier_thesis_state") not in THESIS_STATES:
        raise ValueError("verifier_thesis_state is not recognized")
    if not isinstance(artifact.get("publication_gate"), Mapping):
        raise ValueError("publication_gate must be an object")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle must be bare false")
    if not isinstance(artifact.get("cited_upstream_artifacts"), list):
        raise ValueError("cited_upstream_artifacts must be a list")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        raise ValueError("preconditions_checked must be an object")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if not base.is_sha256(artifact.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be a sha256 hex string")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required principles")
    provenance = artifact.get("upstream_provenance")
    if not isinstance(provenance, list):
        raise ValueError("upstream_provenance must be a list")
    for row in provenance:
        if not isinstance(row, Mapping):
            raise ValueError("upstream provenance row must be an object")
        if not base.is_sha256(row.get("sha256")):
            raise ValueError("upstream provenance row has invalid sha256")
        if row.get("skipped") is True and row.get("fields_imported") != []:
            raise ValueError("skipped upstreams must not import fields")
    expected = (
        BLOCKED_PUBLICATION_GATE_CHECKSUM
        if artifact.get("honest_verdict") == "blocked_publication_gate_unrunnable"
        else checksum_from_provenance(provenance)
    )
    if artifact.get("reproducibility_checksum") != expected:
        raise ValueError("reproducibility_checksum does not match upstream sha256 set")


def write_artifact(
    root: Path = REPO_ROOT,
    *,
    output_path: Path = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = base.run_live_flags,
    summarize_runner: SummarizeRunner = base.run_summarize_artifact,
    publication_gate_runner: PublicationGateRunner = base.run_publication_gate,
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
