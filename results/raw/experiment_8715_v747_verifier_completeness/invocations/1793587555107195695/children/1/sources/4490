"""Build the Exp 4379 v404 verifier scorecard capstone.

Spec refs: REQ-CAPSTONE-4379, SCENARIO-CAPSTONE-4379.
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
OUTPUT_REL_PATH = Path("results/experiment_4379_capstone_v404.json")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
PUBLICATION_GATE_REL_PATH = Path("scripts/publication_gate.py")
EXPERIMENT_ID = 4379
RANDOM_SEED = 4379
SCHEMA = "carnot.capstone_v404_4379.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4379", "SCENARIO-CAPSTONE-4379"]
PRIOR_REPRODUCIBLE_TOTAL_LEVELS = 33
PRIOR_REPRODUCIBLE_TOTAL_GAMES = 17
BLOCKED_PUBLICATION_GATE_CHECKSUM = hashlib.sha256(
    b"blocked_publication_gate_unrunnable_v404"
).hexdigest()
EMPTY_UPSTREAM_CHECKSUM = hashlib.sha256(b"no_v404_upstream_artifacts").hexdigest()

EFFICIENCY_MOAT_STATES = {"deepened_stronger_class", "linear_is_settled", "open"}
S3_MOAT_UTILITIES = {"useful_generation_gain", "proven_but_not_useful", "retired", "open"}


def _thesis_states() -> set[str]:
    efficiency_parts = {
        "deepened_stronger_class": "efficiency_moat_deepened",
        "linear_is_settled": "linear_settled",
        "open": "efficiency_open",
    }
    s3_parts = {
        "useful_generation_gain": "in_generation_converted",
        "proven_but_not_useful": "proven_not_useful",
        "retired": "in_generation_retired",
        "open": "in_generation_open",
    }
    states = {"blocked_publication_gate_unrunnable"}
    for efficiency in efficiency_parts.values():
        for s3 in s3_parts.values():
            states.add(f"{efficiency}_{s3}")
            states.add(f"{efficiency}_{s3}_detector_positive")
    return states


THESIS_STATES = _thesis_states()


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4370_llm_heuristic": Upstream(
        4370, Path("results/experiment_4370_llm_generated_action_cost_heuristics.json")
    ),
    "4371_contamination_skeptic": Upstream(
        4371,
        Path("results/experiment_4371_llm_heuristic_contamination_skeptic_proof.json"),
    ),
    "4372_e3_deeper": Upstream(
        4372, Path("results/experiment_4372_e3_deeper_high_headroom_games.json")
    ),
    "4373_e3_blocked": Upstream(
        4373, Path("results/experiment_4373_e3_blocked_mechanic_levels_ar25_ka59_ft09.json")
    ),
    "4374_diffusiongemma": Upstream(
        4374, Path("results/experiment_4374_diffusiongemma_scorer_repair_or_retire.json")
    ),
    "4375_detector": Upstream(
        4375, Path("results/experiment_4375_verifier_as_detector_measurement.json")
    ),
}

ARTIFACT_EXPERIMENT_IDS = {
    key: upstream.experiment_id for key, upstream in DEFAULT_UPSTREAMS.items()
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "efficiency_moat_state",
    "s3_moat_utility",
    "reproducible_total_levels",
    "detector_beats_chance",
    "verifier_thesis_state",
    "publication_gate",
    "verifier_is_oracle",
    "cited_upstream_artifacts",
    "preconditions_checked",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. The .404 scorecard string (whether the stronger "
        "learned-heuristic class deepened the efficiency moat, the ARC "
        "reproducible-total, the DiffusionGemma convert/null/retire decision, "
        "the detector reading)."
    ),
    "efficiency_moat_state": (
        "One of deepened_stronger_class / linear_is_settled / open -- the "
        "headline decision: did a stronger learned-heuristic class "
        "(LLM-generated Python heuristics) deepen the oracle-distinct efficiency "
        "moat beyond the linear cost, leakage-clean + contamination-free?"
    ),
    "s3_moat_utility": (
        "One of useful_generation_gain / proven_but_not_useful / retired / open "
        "-- the DiffusionGemma in-generation decision: did the scorer-repair + "
        "CoDiLA control convert the moat, null cleanly, or retire the direction "
        "(4th block)?"
    ),
    "reproducible_total_levels": (
        "BARE int: the ARC reproducible-level count after .404 (>= the prior 33) "
        "-- the monotonic north-star accuracy signal."
    ),
    "detector_beats_chance": (
        "BARE bool: did the verifier-as-detector beat chance where selection "
        "headroom is ~0 (the oracle-distinct ACCURACY complementary probe)?"
    ),
    "verifier_thesis_state": (
        "One honest string summarizing where the verifier-moat thesis stands "
        "after .404 (efficiency-moat deepened / linear-settled / "
        "in-generation-converted / in-generation-retired / detector-positive / "
        "etc.)."
    ),
    "publication_gate": (
        "G1-G4 via publication_gate.py (paper_ready + unmet_gates) -- the stable "
        "finish line (north-star \u00a72)."
    ),
    "verifier_is_oracle": (
        "BARE bool=false for the oracle-distinct moat reads (the durable exp4355 "
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
    "4370_llm_heuristic": [
        "llm_heuristic_beats_linear",
        "static_leakage_clean",
        "reproduction_gated",
        "n_held_out_levels",
        "held_out_actions_by_heuristic",
        "verifier_is_oracle",
    ],
    "4371_contamination_skeptic": [
        "win_is_contamination_free",
        "gate_check_summary",
        "gates_evaluated",
        "status",
    ],
    "4372_e3_deeper": [
        "new_levels_reproduced",
        "reproducible_total_levels",
        "per_target_scorecard",
        "verifier_is_oracle",
    ],
    "4373_e3_blocked": [
        "new_levels_reproduced",
        "reproducible_total_levels",
        "per_game_scorecard",
        "verifier_is_oracle",
    ],
    "4374_diffusiongemma": [
        "s3_guided_beats_control",
        "scorer_requalified_leak_clean",
        "codila_control_differentiates",
        "retirement_gate",
        "s3_gain_ci95",
        "s3_minus_best_of_n_delta",
        "benchmark_n",
        "verifier_is_oracle",
    ],
    "4375_detector": [
        "detector_auroc",
        "detector_beats_chance",
        "detector_auroc_ci95",
        "selection_headroom",
        "n_candidates",
        "per_verifier_auroc",
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


def efficiency_moat_read(
    llm_payload: JsonDict | None,
    skeptic_payload: JsonDict | None,
    llm_skipped: bool,
    skeptic_skipped: bool,
) -> JsonDict:
    if llm_skipped:
        return {"status": "excluded_flagged_adversarial"}
    if llm_payload is None:
        return {"status": "missing_or_excluded"}
    reported_win = base.bool_metric(llm_payload, "llm_heuristic_beats_linear")
    static_clean = base.bool_metric(llm_payload, "static_leakage_clean") is True
    reproduction_gated = base.bool_metric(llm_payload, "reproduction_gated") is True
    verifier_is_oracle = base.bool_metric(llm_payload, "verifier_is_oracle")
    contamination_free = (
        False
        if skeptic_skipped
        else base.bool_metric(skeptic_payload, "win_is_contamination_free") is True
    )
    deepened = (
        reported_win is True
        and contamination_free
        and static_clean
        and reproduction_gated
        and verifier_is_oracle is False
    )
    clean_null = (
        reported_win is False
        and base.bool_metric(llm_payload, "acceptance_gate_passed") is True
        and static_clean
        and reproduction_gated
        and verifier_is_oracle is False
    )
    if deepened:
        status = "deepened_stronger_class"
    elif clean_null:
        status = "clean_powered_null"
    else:
        status = "open"
    return {
        "status": status,
        "efficiency_moat_state": decide_efficiency_moat_state(status),
        "llm_heuristic_beats_linear": reported_win is True,
        "reported_llm_heuristic_beats_linear": reported_win,
        "win_is_contamination_free": contamination_free,
        "static_leakage_clean": static_clean,
        "reproduction_gated": reproduction_gated,
        "n_held_out_levels": base.int_metric(llm_payload, "n_held_out_levels"),
        "held_out_actions_by_heuristic": dict(
            llm_payload.get("held_out_actions_by_heuristic", {})
        ),
        "verifier_is_oracle": verifier_is_oracle,
        "honest_verdict": base.str_metric(llm_payload, "honest_verdict"),
        "skeptic_honest_verdict": base.str_metric(skeptic_payload, "honest_verdict"),
        "skeptic_gate_check_summary": base.str_metric(
            skeptic_payload,
            "gate_check_summary",
        ),
        "skeptic_gates_evaluated": base.list_metric(skeptic_payload, "gates_evaluated"),
        "skeptic_skipped": skeptic_skipped,
    }


def decide_efficiency_moat_state(status: str) -> str:
    if status == "deepened_stronger_class":
        return "deepened_stronger_class"
    if status == "clean_powered_null":
        return "linear_is_settled"
    return "open"


def diffusiongemma_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    s3_win = base.bool_metric(payload, "s3_guided_beats_control")
    leak_clean = base.bool_metric(payload, "scorer_requalified_leak_clean") is True
    codila_differentiates = base.bool_metric(payload, "codila_control_differentiates") is True
    verifier_is_oracle = base.bool_metric(payload, "verifier_is_oracle")
    retirement_gate = payload.get("retirement_gate")
    retired = isinstance(retirement_gate, Mapping) and retirement_gate.get("retired") is True
    converted = (
        s3_win is True
        and leak_clean
        and codila_differentiates
        and verifier_is_oracle is False
    )
    clean_null = (
        s3_win is False
        and leak_clean
        and codila_differentiates
        and verifier_is_oracle is False
    )
    if converted:
        status = "useful_generation_gain"
    elif clean_null:
        status = "proven_but_not_useful"
    elif retired:
        status = "retired"
    else:
        status = "open"
    return {
        "status": status,
        "s3_moat_utility": status if status in S3_MOAT_UTILITIES else "open",
        "s3_guided_beats_control": s3_win is True,
        "reported_s3_guided_beats_control": s3_win,
        "scorer_requalified_leak_clean": leak_clean,
        "codila_control_differentiates": codila_differentiates,
        "retirement_gate": dict(retirement_gate) if isinstance(retirement_gate, Mapping) else {},
        "s3_gain_ci95": base.list_metric(payload, "s3_gain_ci95"),
        "s3_minus_best_of_n_delta": base.float_metric(payload, "s3_minus_best_of_n_delta"),
        "benchmark_n": base.int_metric(payload, "benchmark_n"),
        "verifier_is_oracle": verifier_is_oracle,
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def detector_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    detector_positive = (
        base.bool_metric(payload, "detector_beats_chance") is True
        and base.bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "detector_positive" if detector_positive else "detector_null",
        "detector_beats_chance": detector_positive,
        "reported_detector_beats_chance": base.bool_metric(payload, "detector_beats_chance"),
        "detector_auroc": base.float_metric(payload, "detector_auroc"),
        "detector_auroc_ci95": base.list_metric(payload, "detector_auroc_ci95"),
        "selection_headroom": dict(payload.get("selection_headroom", {})),
        "n_candidates": base.int_metric(payload, "n_candidates"),
        "per_verifier_auroc": dict(payload.get("per_verifier_auroc", {})),
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
            payload, "reproducible_total_levels"
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
            name="efficiency",
            required_keys=("4370_llm_heuristic", "4371_contamination_skeptic"),
            verdict_fn=lambda present: decide_efficiency_moat_state(
                efficiency_moat_read(
                    present.get("4370_llm_heuristic"),
                    present.get("4371_contamination_skeptic"),
                    False,
                    False,
                )["status"]
            ),
        ),
        aggregate.AxisSpec(
            name="arc",
            required_keys=("4372_e3_deeper", "4373_e3_blocked"),
            verdict_fn=lambda present: (
                arc_e3_summary(
                    arc_progress_read(
                        present.get("4372_e3_deeper"),
                        False,
                        "per_target_scorecard",
                    ),
                    arc_progress_read(
                        present.get("4373_e3_blocked"),
                        False,
                        "per_game_scorecard",
                    ),
                )["new_levels_reproduced_from_artifacts"]
                > 0
            ),
        ),
        aggregate.AxisSpec(
            name="diffusiongemma",
            required_keys=("4374_diffusiongemma",),
            verdict_fn=lambda present: diffusiongemma_read(
                present.get("4374_diffusiongemma"),
                False,
            )["s3_moat_utility"],
        ),
        aggregate.AxisSpec(
            name="detector",
            required_keys=("4375_detector",),
            verdict_fn=lambda present: detector_read(
                present.get("4375_detector"),
                False,
            )["detector_beats_chance"],
        ),
    ]


def verifier_thesis_state(
    efficiency_moat_state: str,
    s3_moat_utility: str,
    detector_beats: bool,
) -> str:
    efficiency_parts = {
        "deepened_stronger_class": "efficiency_moat_deepened",
        "linear_is_settled": "linear_settled",
        "open": "efficiency_open",
    }
    s3_parts = {
        "useful_generation_gain": "in_generation_converted",
        "proven_but_not_useful": "proven_not_useful",
        "retired": "in_generation_retired",
        "open": "in_generation_open",
    }
    state = (
        f"{efficiency_parts.get(efficiency_moat_state, 'efficiency_open')}_"
        f"{s3_parts.get(s3_moat_utility, 'in_generation_open')}"
    )
    if detector_beats:
        state = f"{state}_detector_positive"
    return state


def _honest_verdict(
    efficiency_moat_state: str,
    total_levels: int,
    s3_moat_utility: str,
    detector_beats: bool,
    paper_ready: bool,
) -> str:
    detector = "positive" if detector_beats else "null"
    paper = "publication_ready" if paper_ready else "publication_not_ready"
    return (
        f"complete: v404_efficiency_{efficiency_moat_state}_arc_levels_{total_levels}_"
        f"s3_{s3_moat_utility}_detector_{detector}_{paper}"
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
        "efficiency_moat_state": "open",
        "efficiency_moat": {"status": "blocked_publication_gate_unrunnable"},
        "s3_moat_utility": "open",
        "diffusiongemma": {"status": "blocked_publication_gate_unrunnable"},
        "reproducible_total_levels": 0,
        "arc_reproducible_progress": {"status": "not_checked", "path": str(REGISTRY_REL_PATH)},
        "arc_e3_outcomes": {"status": "not_checked"},
        "detector_beats_chance": False,
        "detector": {"status": "blocked_publication_gate_unrunnable"},
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

    efficiency = efficiency_moat_read(
        clean["4370_llm_heuristic"],
        clean["4371_contamination_skeptic"],
        skipped.get("4370_llm_heuristic", False),
        skipped.get("4371_contamination_skeptic", False),
    )
    diffusion = diffusiongemma_read(
        clean["4374_diffusiongemma"],
        skipped.get("4374_diffusiongemma", False),
    )
    detector = detector_read(clean["4375_detector"], skipped.get("4375_detector", False))
    deeper = arc_progress_read(
        clean["4372_e3_deeper"],
        skipped.get("4372_e3_deeper", False),
        "per_target_scorecard",
    )
    blocked = arc_progress_read(
        clean["4373_e3_blocked"],
        skipped.get("4373_e3_blocked", False),
        "per_game_scorecard",
    )
    arc_e3 = arc_e3_summary(deeper, blocked)
    registry = read_registry_progress(root)
    paper_ready = bool(publication_gate.get("paper_ready"))
    efficiency_state = str(efficiency.get("efficiency_moat_state", "open"))
    s3_utility = str(diffusion.get("s3_moat_utility", "open"))
    detector_beats = detector.get("detector_beats_chance") is True
    total_levels = int(registry.get("reproducible_total_levels") or 0)
    thesis = verifier_thesis_state(efficiency_state, s3_utility, detector_beats)
    end = time.time() if now_s is None else now_s

    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "random_seed": RANDOM_SEED,
        "duration_s": round(end - start, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(
            efficiency_state,
            total_levels,
            s3_utility,
            detector_beats,
            paper_ready,
        ),
        "efficiency_moat_state": efficiency_state,
        "efficiency_moat": efficiency,
        "s3_moat_utility": s3_utility,
        "diffusiongemma": diffusion,
        "reproducible_total_levels": total_levels,
        "arc_reproducible_progress": registry,
        "arc_e3_outcomes": arc_e3,
        "detector_beats_chance": detector_beats,
        "detector": detector,
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
    if artifact.get("efficiency_moat_state") not in EFFICIENCY_MOAT_STATES:
        raise ValueError("efficiency_moat_state is not recognized")
    if artifact.get("s3_moat_utility") not in S3_MOAT_UTILITIES:
        raise ValueError("s3_moat_utility is not recognized")
    if not isinstance(artifact.get("reproducible_total_levels"), int) or isinstance(
        artifact.get("reproducible_total_levels"), bool
    ):
        raise ValueError("reproducible_total_levels must be a bare int")
    if not isinstance(artifact.get("detector_beats_chance"), bool):
        raise ValueError("detector_beats_chance must be a bare bool")
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
