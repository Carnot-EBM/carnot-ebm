"""Build the Exp 4357 v402 verifier scorecard capstone.

Spec refs: REQ-CAPSTONE-4357, SCENARIO-CAPSTONE-4357.
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
OUTPUT_REL_PATH = Path("results/experiment_4357_capstone_v402.json")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
PUBLICATION_GATE_REL_PATH = Path("scripts/publication_gate.py")
EXPERIMENT_ID = 4357
RANDOM_SEED = 4357
SCHEMA = "carnot.capstone_v402_4357.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4357", "SCENARIO-CAPSTONE-4357"]
PRIOR_REPRODUCIBLE_TOTAL_LEVELS = 21
PRIOR_REPRODUCIBLE_TOTAL_GAMES = 13
BLOCKED_PUBLICATION_GATE_CHECKSUM = hashlib.sha256(
    b"blocked_publication_gate_unrunnable"
).hexdigest()
EMPTY_UPSTREAM_CHECKSUM = hashlib.sha256(b"no_v402_upstream_artifacts").hexdigest()

S3_MOAT_UTILITIES = {"useful_generation_gain", "proven_but_not_useful", "open"}
THESIS_STATES = {
    "moat_proven_useful",
    "moat_proven_not_useful",
    "moat_proven_leak_robust_but_s3_utility_open",
    "blocked_publication_gate_unrunnable",
}


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4348_s3_search": Upstream(
        4348, Path("results/experiment_4348_s3_stratified_verifier_guided_search.json")
    ),
    "4349_reward_state_alignment": Upstream(
        4349, Path("results/experiment_4349_papo_reward_state_alignment_diagnostic.json")
    ),
    "4350_e3_ka59": Upstream(
        4350, Path("results/experiment_4350_e3_explore_verify_plan_ka59.json")
    ),
    "4351_e3_deeper": Upstream(4351, Path("results/experiment_4351_e3_deeper_solved_games.json")),
    "4352_e3_tr87_ft09": Upstream(
        4352, Path("results/experiment_4352_e3_explore_verify_plan_tr87_ft09.json")
    ),
    "4353_action_efficiency": Upstream(
        4353, Path("results/experiment_4353_learned_action_cost_heuristic_efficiency.json")
    ),
}

ARTIFACT_EXPERIMENT_IDS = {
    key: upstream.experiment_id for key, upstream in DEFAULT_UPSTREAMS.items()
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "s3_moat_utility",
    "reproducible_total_levels",
    "action_efficiency_improves",
    "verifier_thesis_state",
    "publication_gate",
    "verifier_is_oracle",
    "cited_upstream_artifacts",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. The .402 scorecard string (whether S3 converted the "
        "moat, the ARC reproducible-total, the self-learning result)."
    ),
    "s3_moat_utility": (
        "One of useful_generation_gain / proven_but_not_useful / open -- the "
        "headline decision: did the proven oracle-distinct moat become a fixed-NFE "
        "generation gain (Pareto-improvement), validated by reward-state alignment?"
    ),
    "reproducible_total_levels": (
        "BARE int: the ARC reproducible-level count after .402 (>= the prior 21) "
        "-- the monotonic north-star accuracy signal."
    ),
    "action_efficiency_improves": (
        "BARE bool: did the learned action-cost heuristic lower held-out "
        "env-actions-to-solve (north-star efficiency axis)?"
    ),
    "verifier_thesis_state": (
        "One honest string summarizing where the verifier-moat thesis stands "
        "after .402 (moat proven+useful / proven-not-useful / etc.)."
    ),
    "publication_gate": (
        "G1-G4 via publication_gate.py (paper_ready + unmet_gates) -- the stable "
        "finish line (north-star \u00a72)."
    ),
    "verifier_is_oracle": (
        "BARE bool=false for the oracle-distinct moat read (the exp4355 stamp fix) "
        "-- so this capstone does NOT trip CIRCULAR_MOAT_OVERCLAIM."
    ),
    "cited_upstream_artifacts": (
        "list of {experiment_id, fields_imported} -- the audit trail so the "
        "capstone numbers trace to real measurements."
    ),
    "preconditions_checked": (
        "Records the upstream-artifact + publication_gate availability; "
        "pre-empts the silent-missing-resource fabrication mode."
    ),
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "4348_s3_search": [
        "s3_guided_beats_control",
        "controls_differentiated",
        "scorer_leak_recheck_passed",
        "nfe_budget",
        "s3_gain_ci95",
        "s3_minus_best_of_k_delta",
        "s3_minus_self_reward_smc_delta",
        "s3_minus_unguided_delta",
        "verifier_is_oracle",
    ],
    "4349_reward_state_alignment": [
        "reward_state_alignment_passed",
        "gate_check_summary",
        "gates_evaluated",
        "status",
    ],
    "4350_e3_ka59": [
        "game",
        "offline_reproduced",
        "reproduced_levels",
        "residual_mismatch_class",
        "verifier_best_accuracy",
        "verifier_is_oracle",
    ],
    "4351_e3_deeper": [
        "new_levels_reproduced",
        "reproducible_total_levels",
        "per_target_scorecard",
        "verifier_is_oracle",
    ],
    "4352_e3_tr87_ft09": [
        "games",
        "new_levels_reproduced",
        "per_game_scorecard",
        "verifier_is_oracle",
    ],
    "4353_action_efficiency": [
        "action_efficiency_improves",
        "held_out_actions_baseline",
        "held_out_actions_learned",
        "positive_control_passed",
        "reproduction_gated",
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


def _positive_float(payload: Mapping[str, Any], field: str) -> bool:
    value = base.float_metric(payload, field)
    return value is not None and value > 0.0


def s3_search_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    ci95 = base.list_metric(payload, "s3_gain_ci95")
    nfe_budget = base.int_metric(payload, "nfe_budget")
    reported = base.bool_metric(payload, "s3_guided_beats_control")
    verifier_is_oracle = base.bool_metric(payload, "verifier_is_oracle")
    deltas_positive = all(
        _positive_float(payload, field)
        for field in (
            "s3_minus_best_of_k_delta",
            "s3_minus_self_reward_smc_delta",
            "s3_minus_unguided_delta",
        )
    )
    beats_controls = (
        reported is True
        and base.bool_metric(payload, "controls_differentiated") is True
        and base.bool_metric(payload, "scorer_leak_recheck_passed") is True
        and base.ci95_excludes_zero(ci95)
        and nfe_budget > 0
        and deltas_positive
        and verifier_is_oracle is False
    )
    null_clean = (
        reported is False
        and base.bool_metric(payload, "controls_differentiated") is True
        and verifier_is_oracle is False
    )
    if beats_controls:
        status = "beats_controls"
    elif null_clean:
        status = "null"
    else:
        status = "measured_unresolved"
    return {
        "status": status,
        "beats_compute_matched_and_intrinsic_controls": beats_controls,
        "reported_s3_guided_beats_control": reported,
        "controls_differentiated": base.bool_metric(payload, "controls_differentiated"),
        "scorer_leak_recheck_passed": base.bool_metric(payload, "scorer_leak_recheck_passed"),
        "nfe_budget": nfe_budget,
        "s3_gain_ci95": ci95,
        "s3_gain_ci95_excludes_zero": base.ci95_excludes_zero(ci95),
        "s3_minus_best_of_k_delta": base.float_metric(payload, "s3_minus_best_of_k_delta"),
        "s3_minus_self_reward_smc_delta": base.float_metric(
            payload, "s3_minus_self_reward_smc_delta"
        ),
        "s3_minus_unguided_delta": base.float_metric(payload, "s3_minus_unguided_delta"),
        "verifier_is_oracle": verifier_is_oracle,
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def reward_alignment_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    passed = base.bool_metric(payload, "reward_state_alignment_passed")
    artifact_status = base.str_metric(payload, "status")
    verdict = base.str_metric(payload, "honest_verdict")
    if passed is True:
        status = "passed"
    elif artifact_status == "blocked" or verdict.startswith("blocked"):
        status = "blocked"
    elif passed is False:
        status = "failed"
    else:
        status = "missing_measurement"
    return {
        "status": status,
        "reward_state_alignment_passed": passed,
        "gate_check_summary": base.str_metric(payload, "gate_check_summary"),
        "gates_evaluated": base.list_metric(payload, "gates_evaluated"),
        "honest_verdict": verdict,
    }


def decide_s3_moat_utility(
    s3_search: Mapping[str, Any],
    reward_alignment: Mapping[str, Any],
) -> str:
    if (
        s3_search.get("beats_compute_matched_and_intrinsic_controls") is True
        and reward_alignment.get("reward_state_alignment_passed") is True
    ):
        return "useful_generation_gain"
    if s3_search.get("status") == "null":
        return "proven_but_not_useful"
    return "open"


def s3_utility_read(
    s3_search: Mapping[str, Any],
    reward_alignment: Mapping[str, Any],
) -> JsonDict:
    utility = decide_s3_moat_utility(s3_search, reward_alignment)
    if utility != "open":
        status = utility
    elif s3_search.get("status") in {"excluded_flagged_adversarial", "missing_or_excluded"}:
        status = "open_flagged_or_missing_s3"
    elif reward_alignment.get("reward_state_alignment_passed") is not True:
        status = "open_alignment_not_validated"
    else:
        status = "open"
    return {
        "status": status,
        "s3_moat_utility": utility,
        "requires_clean_s3_gain": True,
        "requires_reward_state_alignment": True,
    }


def arc_ka59_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    reproduced = base.bool_metric(payload, "offline_reproduced") is True
    levels = base.int_metric(payload, "reproduced_levels") if reproduced else 0
    return {
        "status": "reproduced" if reproduced else "partial",
        "game": base.str_metric(payload, "game") or "ka59",
        "offline_reproduced": reproduced,
        "new_levels_reproduced": levels,
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "residual_mismatch_class": base.str_metric(payload, "residual_mismatch_class"),
        "verifier_best_accuracy": base.float_metric(payload, "verifier_best_accuracy"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def _clean_scorecard_rows(
    rows: list[Any], level_field: str
) -> tuple[int, list[str], list[JsonDict]]:
    total = 0
    games: list[str] = []
    cleaned: list[JsonDict] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        item = dict(row)
        game = item.get("game")
        level = item.get(level_field)
        if (
            item.get("offline_reproduced") is True
            and isinstance(game, str)
            and isinstance(level, int)
            and not isinstance(level, bool)
        ):
            games.append(game)
        cleaned.append(item)
    return total, games, cleaned


def arc_deeper_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    rows = base.list_metric(payload, "per_target_scorecard")
    _, games, cleaned = _clean_scorecard_rows(rows, "new_reproduced_level")
    new_levels = base.int_metric(payload, "new_levels_reproduced")
    return {
        "status": "reproduced" if new_levels > 0 else "partial",
        "new_levels_reproduced": new_levels,
        "games_with_new_reproducible_levels": games,
        "reproducible_total_levels_reported": base.int_metric(payload, "reproducible_total_levels"),
        "per_target_scorecard": cleaned,
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def arc_tr87_ft09_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    rows = base.list_metric(payload, "per_game_scorecard")
    _, games, cleaned = _clean_scorecard_rows(rows, "reproduced_levels")
    new_levels = base.int_metric(payload, "new_levels_reproduced")
    return {
        "status": "reproduced" if new_levels > 0 else "partial",
        "new_levels_reproduced": new_levels,
        "games_with_new_reproducible_levels": games,
        "per_game_scorecard": cleaned,
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def arc_e3_summary(
    ka59: Mapping[str, Any],
    deeper: Mapping[str, Any],
    tr87_ft09: Mapping[str, Any],
) -> JsonDict:
    reads = [ka59, deeper, tr87_ft09]
    new_levels = sum(int(read.get("new_levels_reproduced") or 0) for read in reads)
    games: set[str] = set()
    for read in reads:
        game = read.get("game")
        if read.get("new_levels_reproduced") and isinstance(game, str):
            games.add(game)
        for nested in read.get("games_with_new_reproducible_levels") or []:
            if isinstance(nested, str):
                games.add(nested)
    return {
        "status": "advanced" if new_levels > 0 else "partial",
        "new_levels_reproduced_from_artifacts": new_levels,
        "games_with_new_reproducible_levels": sorted(games),
        "execution_grounded": any(read.get("verifier_is_oracle") is True for read in reads),
        "ka59": dict(ka59),
        "deeper": dict(deeper),
        "tr87_ft09": dict(tr87_ft09),
    }


def action_efficiency_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    baseline = base.int_metric(payload, "held_out_actions_baseline")
    learned = base.int_metric(payload, "held_out_actions_learned")
    reported = base.bool_metric(payload, "action_efficiency_improves")
    improves = (
        reported is True
        and baseline > learned
        and learned > 0
        and base.bool_metric(payload, "positive_control_passed") is True
        and base.bool_metric(payload, "reproduction_gated") is True
        and base.bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "improves" if improves else "open",
        "action_efficiency_improves": improves,
        "reported_action_efficiency_improves": reported,
        "held_out_actions_baseline": baseline,
        "held_out_actions_learned": learned,
        "action_reduction": baseline - learned if baseline and learned else 0,
        "positive_control_passed": base.bool_metric(payload, "positive_control_passed"),
        "reproduction_gated": base.bool_metric(payload, "reproduction_gated"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def read_registry_progress(root: Path) -> JsonDict:
    path = root / REGISTRY_REL_PATH
    if not path.exists():
        return {
            "status": "missing",
            "path": str(REGISTRY_REL_PATH),
            "reproducible_total_levels": 0,
            "reproducible_total_games": 0,
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
            name="s3_utility",
            required_keys=("4348_s3_search", "4349_reward_state_alignment"),
            verdict_fn=lambda present: decide_s3_moat_utility(
                s3_search_read(present.get("4348_s3_search"), False),
                reward_alignment_read(present.get("4349_reward_state_alignment"), False),
            ),
        ),
        aggregate.AxisSpec(
            name="arc",
            required_keys=("4350_e3_ka59", "4351_e3_deeper", "4352_e3_tr87_ft09"),
            verdict_fn=lambda present: (
                arc_e3_summary(
                    arc_ka59_read(present.get("4350_e3_ka59"), False),
                    arc_deeper_read(present.get("4351_e3_deeper"), False),
                    arc_tr87_ft09_read(present.get("4352_e3_tr87_ft09"), False),
                )["new_levels_reproduced_from_artifacts"]
                > 0
            ),
        ),
        aggregate.AxisSpec(
            name="action_efficiency",
            required_keys=("4353_action_efficiency",),
            verdict_fn=lambda present: (
                action_efficiency_read(present.get("4353_action_efficiency"), False)[
                    "action_efficiency_improves"
                ]
                is True
            ),
        ),
    ]


def verifier_thesis_state(s3_moat_utility: str) -> str:
    if s3_moat_utility == "useful_generation_gain":
        return "moat_proven_useful"
    if s3_moat_utility == "proven_but_not_useful":
        return "moat_proven_not_useful"
    return "moat_proven_leak_robust_but_s3_utility_open"


def _honest_verdict(
    s3_moat_utility: str,
    total_levels: int,
    action_efficiency_improves: bool,
    paper_ready: bool,
) -> str:
    action = "improves" if action_efficiency_improves else "open"
    paper = "publication_ready" if paper_ready else "publication_not_ready"
    return (
        f"complete: v402_s3_{s3_moat_utility}_arc_levels_{total_levels}_"
        f"action_efficiency_{action}_{paper}"
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
        "s3_moat_utility": "open",
        "reproducible_total_levels": 0,
        "action_efficiency_improves": False,
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
        root, publication_gate_runner
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

    s3_search = s3_search_read(clean["4348_s3_search"], skipped.get("4348_s3_search", False))
    reward_alignment = reward_alignment_read(
        clean["4349_reward_state_alignment"],
        skipped.get("4349_reward_state_alignment", False),
    )
    s3_utility = s3_utility_read(s3_search, reward_alignment)
    ka59 = arc_ka59_read(clean["4350_e3_ka59"], skipped.get("4350_e3_ka59", False))
    deeper = arc_deeper_read(clean["4351_e3_deeper"], skipped.get("4351_e3_deeper", False))
    tr87_ft09 = arc_tr87_ft09_read(
        clean["4352_e3_tr87_ft09"], skipped.get("4352_e3_tr87_ft09", False)
    )
    arc_e3 = arc_e3_summary(ka59, deeper, tr87_ft09)
    action_efficiency = action_efficiency_read(
        clean["4353_action_efficiency"], skipped.get("4353_action_efficiency", False)
    )
    registry = read_registry_progress(root)
    paper_ready = bool(publication_gate.get("paper_ready"))
    utility = str(s3_utility["s3_moat_utility"])
    thesis = verifier_thesis_state(utility)
    action_improves = action_efficiency.get("action_efficiency_improves") is True
    total_levels = int(registry.get("reproducible_total_levels") or 0)
    end = time.time() if now_s is None else now_s

    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "random_seed": RANDOM_SEED,
        "duration_s": round(end - start, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(utility, total_levels, action_improves, paper_ready),
        "s3_moat_utility": utility,
        "s3_utility": s3_utility,
        "s3_search": s3_search,
        "reward_state_alignment": reward_alignment,
        "reproducible_total_levels": total_levels,
        "arc_reproducible_progress": registry,
        "arc_e3_outcomes": arc_e3,
        "action_efficiency_improves": action_improves,
        "action_efficiency": action_efficiency,
        "verifier_thesis_state": thesis,
        "publication_gate": publication_gate,
        "paper_ready": paper_ready,
        "unmet_gates": base.list_metric(publication_gate, "unmet_gates"),
        "verifier_is_oracle": False,
        "verifier_is_oracle_honored": True,
        "cited_upstream_artifacts": _cited_upstream_artifacts(provenance),
        "preconditions_checked": _preconditions_checked(
            root, publication_gate_check, provenance, registry
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
    if artifact.get("s3_moat_utility") not in S3_MOAT_UTILITIES:
        raise ValueError("s3_moat_utility is not recognized")
    if not isinstance(artifact.get("reproducible_total_levels"), int) or isinstance(
        artifact.get("reproducible_total_levels"), bool
    ):
        raise ValueError("reproducible_total_levels must be a bare int")
    if not isinstance(artifact.get("action_efficiency_improves"), bool):
        raise ValueError("action_efficiency_improves must be a bare bool")
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
