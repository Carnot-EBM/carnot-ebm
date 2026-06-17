"""Build the Exp 4346 v401 verifier scorecard capstone.

Spec refs: REQ-CAPSTONE-4346, SCENARIO-CAPSTONE-4346.
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
OUTPUT_REL_PATH = Path("results/experiment_4346_capstone_v401.json")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
EXPERIMENT_ID = 4346
RANDOM_SEED = 4346
SCHEMA = "carnot.capstone_v401_4346.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4346", "SCENARIO-CAPSTONE-4346"]
GATE_MET = "MET_oracle_distinct_leak_robust_replicated"
BLOCKED_PUBLICATION_GATE_CHECKSUM = hashlib.sha256(
    b"blocked_publication_gate_missing"
).hexdigest()
EMPTY_UPSTREAM_CHECKSUM = hashlib.sha256(b"no_v401_upstream_artifacts").hexdigest()

THESIS_STATES = {
    "in_generation_moat_replicated_leak_robust",
    "in_generation_moat_retired_corpus_specific",
    "first_e3_arc_solve",
    "verifier_domain_bound_self_learning_open",
    "blocked_publication_gate_missing",
}


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4337_leak_robust_scorer": Upstream(
        4337, Path("results/experiment_4337_leak_robust_partial_state_scorer_build.json")
    ),
    "4338_in_generation_moat": Upstream(
        4338, Path("results/experiment_4338_in_generation_moat_replicate_leak_robust.json")
    ),
    "4339_e3_ar25": Upstream(
        4339, Path("results/experiment_4339_e3_explore_verify_plan_ar25.json")
    ),
    "4340_e3_ka59": Upstream(
        4340, Path("results/experiment_4340_e3_explore_verify_plan_ka59.json")
    ),
    "4341_e3_sc25": Upstream(
        4341, Path("results/experiment_4341_e3_sc25_reproduction.json")
    ),
    "4342_self_learning": Upstream(
        4342, Path("results/experiment_4342_self_learning_action_role_cross_game_encoder.json")
    ),
    "4344_hygiene": Upstream(
        4344, Path("results/experiment_4344_verifier_registry_gaps_hygiene.json")
    ),
}

ARTIFACT_EXPERIMENT_IDS = {
    key: upstream.experiment_id for key, upstream in DEFAULT_UPSTREAMS.items()
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "in_generation_moat_replicates_headline",
    "diffusiongemma_gate_status",
    "arc_reproducible_total_levels",
    "verifier_thesis_state",
    "verifier_is_oracle_honored",
    "per_axis_gaps",
    "flagged_artifacts_excluded",
    "paper_ready",
    "upstream_provenance",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. The .401 close-state -- whether the in-generation moat "
        "replicated leak-robust (gate flips) or retired (corpus-specific), whether "
        "E3 landed the first ARC solve, whether the action-role encoder transferred."
    ),
    "in_generation_moat_replicates_headline": (
        "BARE bool: did the in-generation oracle-distinct moat REPLICATE with the "
        "LEAK-ROBUST scorer on the 2nd corpus (exp4338, CI95-excl-0, "
        "controls_differentiated, verifier_is_oracle=false) -- the SETTLED headline."
    ),
    "diffusiongemma_gate_status": (
        "One honest string: 'MET_oracle_distinct_leak_robust_replicated' iff the "
        "LEAK-ROBUST scorer (exp4337) replicated the moat (exp4338) with matched "
        "controls + CI95-excl-0; 'RETIRED_corpus_specific_<reason>' iff a powered "
        "non-replication even with a leak-robust scorer; else 'STILL_PENDING_<reason>'. "
        "The operator was twice-burned -> only a leak-robust REPLICATED win flips MET."
    ),
    "arc_reproducible_total_levels": (
        "BARE int: the cumulative OFFLINE-REPRODUCED ARC solved-level count (the "
        "north-star accuracy metric; only reproduced levels count) after the .401 "
        "E3 advances (target >=14)."
    ),
    "verifier_thesis_state": (
        "One honest string for the verifier thesis state -- the framing the .402 "
        "planner inherits (e.g. in_generation_moat_replicated_leak_robust / "
        "in_generation_moat_retired_corpus_specific / first_e3_arc_solve)."
    ),
    "verifier_is_oracle_honored": (
        "BARE bool=true -- confirms every cited MOAT/headline result carried "
        "verifier_is_oracle=false (the E3 solves are reported as execution-grounded "
        "ARC progress, NOT moats)."
    ),
    "per_axis_gaps": (
        "List of .401 axes whose artifact was MISSING (reported as a gap, NOT "
        "defaulted False) -- the robust-aggregator fix (no exp4301-style all-False)."
    ),
    "flagged_artifacts_excluded": (
        "List of .401 artifacts excluded for flagged_adversarial -- the fabrication "
        "gate (their numbers are NOT aggregated)."
    ),
    "paper_ready": (
        "From publication_gate.py --json -- the G1-G4 status (FoVer 0.9131 stays "
        "the publication target; a replicated leak-robust in-generation moat would "
        "be a new headline-grade supporting result)."
    ),
    "upstream_provenance": (
        "Each cited upstream artifact {experiment_id, fields_imported, sha256} -- "
        "the audit trail that the capstone synthesizes nothing from nothing."
    ),
    "reproducibility_checksum": (
        "Hash of the aggregated upstream sha256 set; lets a third party re-derive "
        "the capstone."
    ),
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "4337_leak_robust_scorer": [
        "scorer_leak_audit_passed",
        "masked_answer_recovery_auroc",
        "process_ranking_auroc",
        "scorer_module_path",
        "verifier_is_oracle",
    ],
    "4338_in_generation_moat": [
        "in_generation_moat_replicates",
        "replication_ci95",
        "controls_differentiated",
        "scorer_leak_recheck_passed",
        "benchmark_n",
        "carnot_minus_best_control_delta",
        "carnot_minus_self_reward_smc_delta",
        "verifier_is_oracle",
    ],
    "4339_e3_ar25": [
        "game",
        "offline_reproduced",
        "reproduced_levels",
        "plan_executed",
        "residual_mismatch_class",
        "verifier_accuracy_per_round",
        "verifier_best_accuracy",
        "verifier_is_oracle",
    ],
    "4340_e3_ka59": [
        "game",
        "offline_reproduced",
        "reproduced_levels",
        "plan_executed",
        "residual_mismatch_class",
        "verifier_accuracy_per_round",
        "verifier_best_accuracy",
        "verifier_is_oracle",
    ],
    "4341_e3_sc25": [
        "game",
        "offline_reproduced",
        "reproduced_levels",
        "plan_executed",
        "win_mechanic_cracked",
        "verifier_accuracy_per_round",
        "verifier_best_accuracy",
        "verifier_is_oracle",
    ],
    "4342_self_learning": [
        "learned_encoder_transfer_helps",
        "cross_game_state_reduction",
        "cross_game_state_reduction_ci95",
        "n_held_out_levels",
        "n_held_out_games",
        "positive_control_passed",
        "verifier_is_oracle",
    ],
    "4344_hygiene": [
        "regression_guard_passed",
        "registry_reconciled",
        "manifest_reconciled",
        "gaps_logged",
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
        raw_artifacts[key] = _skipped_payload(payload) if payload is not None and skipped else payload
        provenance.append(
            {
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
        )
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


def scorer_leak_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    passed = base.bool_metric(payload, "scorer_leak_audit_passed") is True
    return {
        "status": "passed" if passed else "failed",
        "scorer_leak_audit_passed": passed,
        "masked_answer_recovery_auroc": base.float_metric(payload, "masked_answer_recovery_auroc"),
        "process_ranking_auroc": base.float_metric(payload, "process_ranking_auroc"),
        "scorer_module_path": base.str_metric(payload, "scorer_module_path"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def in_generation_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    ci95 = base.list_metric(payload, "replication_ci95")
    ci95_excludes = base.ci95_excludes_zero(ci95)
    reported = base.bool_metric(payload, "in_generation_moat_replicates")
    verifier_is_oracle = base.bool_metric(payload, "verifier_is_oracle")
    headline = (
        reported is True
        and base.bool_metric(payload, "controls_differentiated") is True
        and base.bool_metric(payload, "scorer_leak_recheck_passed") is True
        and ci95_excludes
        and verifier_is_oracle is False
    )
    verdict = base.str_metric(payload, "honest_verdict")
    retired = (
        base.bool_metric(payload, "retire_if_same_verdict") is True
        or "retired" in verdict
        or "corpus_specific" in verdict
    ) and reported is False
    status = "replicated" if headline else ("retired_corpus_specific" if retired else "measured")
    return {
        "status": status,
        "in_generation_moat_replicates_headline": headline,
        "reported_in_generation_moat_replicates": reported,
        "replication_ci95": ci95,
        "replication_ci95_excludes_zero": ci95_excludes,
        "controls_differentiated": base.bool_metric(payload, "controls_differentiated"),
        "scorer_leak_recheck_passed": base.bool_metric(payload, "scorer_leak_recheck_passed"),
        "benchmark_n": base.int_metric(payload, "benchmark_n"),
        "carnot_minus_best_control_delta": base.float_metric(
            payload, "carnot_minus_best_control_delta"
        ),
        "carnot_minus_self_reward_smc_delta": base.float_metric(
            payload, "carnot_minus_self_reward_smc_delta"
        ),
        "verifier_is_oracle": verifier_is_oracle,
        "honest_verdict": verdict,
    }


def e3_single_read(payload: JsonDict | None, skipped: bool, fallback_game: str) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    reproduced = base.bool_metric(payload, "offline_reproduced") is True
    return {
        "status": "reproduced" if reproduced else "partial",
        "game": base.str_metric(payload, "game") or fallback_game,
        "offline_reproduced": base.bool_metric(payload, "offline_reproduced"),
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "plan_executed": base.bool_metric(payload, "plan_executed"),
        "residual_mismatch_class": base.str_metric(payload, "residual_mismatch_class"),
        "win_mechanic_cracked": base.bool_metric(payload, "win_mechanic_cracked"),
        "verifier_accuracy_per_round": base.list_metric(payload, "verifier_accuracy_per_round"),
        "verifier_best_accuracy": base.float_metric(payload, "verifier_best_accuracy"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def _e3_summary(*reads: Mapping[str, Any]) -> JsonDict:
    games: dict[str, JsonDict] = {}
    total = 0
    for read in reads:
        game = read.get("game")
        if isinstance(game, str) and game:
            games[game] = dict(read)
        if read.get("offline_reproduced") is True:
            total += int(read.get("reproduced_levels") or 0)
    return {
        "status": "reproduced" if total > 0 else "partial",
        "reproduced_levels_total": total,
        "games": games,
        "execution_grounded": any(
            row.get("verifier_is_oracle") is True
            for row in games.values()
            if isinstance(row, Mapping)
        ),
    }


def self_learning_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    reported = base.bool_metric(payload, "learned_encoder_transfer_helps")
    helps = reported is True and base.bool_metric(payload, "verifier_is_oracle") is False
    return {
        "status": "helps" if helps else "open",
        "learned_encoder_transfer_helps": helps,
        "reported_learned_encoder_transfer_helps": reported,
        "cross_game_state_reduction": base.float_metric(payload, "cross_game_state_reduction"),
        "cross_game_state_reduction_ci95": base.list_metric(
            payload, "cross_game_state_reduction_ci95"
        ),
        "n_held_out_levels": base.int_metric(payload, "n_held_out_levels"),
        "n_held_out_games": base.int_metric(payload, "n_held_out_games"),
        "positive_control_passed": base.bool_metric(payload, "positive_control_passed"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def hygiene_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    gaps = payload.get("gaps_logged")
    gaps_logged_count = gaps if isinstance(gaps, int) and not isinstance(gaps, bool) else 0
    if isinstance(gaps, list):
        gaps_logged_count = len(gaps)
    passed = base.bool_metric(payload, "regression_guard_passed") is True
    return {
        "status": "passed" if passed else "open",
        "regression_guard_passed": passed,
        "registry_reconciled": base.bool_metric(payload, "registry_reconciled"),
        "manifest_reconciled": base.bool_metric(payload, "manifest_reconciled"),
        "gaps_logged_count": gaps_logged_count,
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def read_registry_total_levels(root: Path) -> JsonDict:
    path = root / REGISTRY_REL_PATH
    if not path.exists():
        return {
            "status": "missing",
            "reproducible_total_levels": 0,
            "path": str(REGISTRY_REL_PATH),
        }
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return {
            "status": "unparseable",
            "reproducible_total_levels": 0,
            "path": str(REGISTRY_REL_PATH),
            "error": str(exc),
        }
    if not isinstance(payload, Mapping):
        return {
            "status": "unparseable",
            "reproducible_total_levels": 0,
            "path": str(REGISTRY_REL_PATH),
            "error": "non-mapping registry",
        }
    total = payload.get("reproducible_total_levels")
    if not isinstance(total, int) or isinstance(total, bool):
        return {
            "status": "unparseable",
            "reproducible_total_levels": 0,
            "path": str(REGISTRY_REL_PATH),
            "error": "reproducible_total_levels missing or non-int",
        }
    return {
        "status": "loaded",
        "reproducible_total_levels": total,
        "path": str(REGISTRY_REL_PATH),
    }


def _axis_specs() -> list[aggregate.AxisSpec]:
    return [
        aggregate.AxisSpec(
            name="scorer_leak_audit",
            required_keys=("4337_leak_robust_scorer",),
            verdict_fn=lambda present: scorer_leak_read(
                present.get("4337_leak_robust_scorer"), False
            )["scorer_leak_audit_passed"]
            is True,
        ),
        aggregate.AxisSpec(
            name="in_generation_moat",
            required_keys=("4338_in_generation_moat",),
            verdict_fn=lambda present: in_generation_read(
                present.get("4338_in_generation_moat"), False
            )["in_generation_moat_replicates_headline"]
            is True,
        ),
        aggregate.AxisSpec(
            name="arc_e3",
            required_keys=("4339_e3_ar25", "4340_e3_ka59", "4341_e3_sc25"),
            verdict_fn=lambda present: (
                base.int_metric(present.get("4339_e3_ar25"), "reproduced_levels")
                + base.int_metric(present.get("4340_e3_ka59"), "reproduced_levels")
                + base.int_metric(present.get("4341_e3_sc25"), "reproduced_levels")
            )
            > 0,
        ),
        aggregate.AxisSpec(
            name="self_learning",
            required_keys=("4342_self_learning",),
            verdict_fn=lambda present: self_learning_read(
                present.get("4342_self_learning"), False
            )["learned_encoder_transfer_helps"]
            is True,
        ),
        aggregate.AxisSpec(
            name="hygiene",
            required_keys=("4344_hygiene",),
            verdict_fn=lambda present: hygiene_read(present.get("4344_hygiene"), False)[
                "regression_guard_passed"
            ]
            is True,
        ),
    ]


def diffusiongemma_gate_status(
    scorer: Mapping[str, Any],
    in_generation: Mapping[str, Any],
) -> str:
    if scorer.get("status") in {"missing_or_excluded", "excluded_flagged_adversarial"}:
        return "STILL_PENDING_leak_robust_scorer_unavailable"
    if scorer.get("scorer_leak_audit_passed") is not True:
        return "STILL_PENDING_leak_robust_scorer_failed"
    if scorer.get("verifier_is_oracle") is not False:
        return "STILL_PENDING_leak_robust_scorer_oracle_not_distinct"
    if in_generation.get("status") in {"missing_or_excluded", "excluded_flagged_adversarial"}:
        return "STILL_PENDING_second_corpus_replication_unavailable"
    if in_generation.get("verifier_is_oracle") is not False:
        return "STILL_PENDING_verifier_oracle_not_distinct"
    if in_generation.get("controls_differentiated") is not True:
        return "STILL_PENDING_controls_not_differentiated"
    if in_generation.get("scorer_leak_recheck_passed") is not True:
        return "STILL_PENDING_second_corpus_scorer_leak_recheck_failed"
    if in_generation.get("replication_ci95_excludes_zero") is not True:
        return "STILL_PENDING_ci95_includes_zero"
    if in_generation.get("in_generation_moat_replicates_headline") is True:
        return GATE_MET
    if in_generation.get("status") == "retired_corpus_specific":
        return "RETIRED_corpus_specific_powered_non_replication"
    return "STILL_PENDING_second_corpus_replication_false"


def verifier_thesis_state(
    gate_status: str,
    e3_reproduced_levels: int,
    self_learning_helps: bool,
) -> str:
    if gate_status == GATE_MET:
        return "in_generation_moat_replicated_leak_robust"
    if gate_status.startswith("RETIRED_corpus_specific_"):
        return "in_generation_moat_retired_corpus_specific"
    if e3_reproduced_levels > 0:
        return "first_e3_arc_solve"
    return "verifier_domain_bound_self_learning_open"


def _oracle_violations(
    scorer: Mapping[str, Any],
    in_generation: Mapping[str, Any],
    self_learning: Mapping[str, Any],
) -> list[str]:
    violations: list[str] = []
    if (
        scorer.get("scorer_leak_audit_passed") is True
        and scorer.get("verifier_is_oracle") is not False
    ):
        violations.append("4337_leak_robust_scorer:scorer_leak_audit")
    if (
        in_generation.get("reported_in_generation_moat_replicates") is True
        and in_generation.get("verifier_is_oracle") is not False
    ):
        violations.append("4338_in_generation_moat:in_generation_moat")
    if (
        self_learning.get("reported_learned_encoder_transfer_helps") is True
        and self_learning.get("verifier_is_oracle") is not False
    ):
        violations.append("4342_self_learning:action_role_encoder")
    return violations


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


def _status_part(read: Mapping[str, Any], true_key: str, true_part: str, false_part: str) -> str:
    if read.get(true_key) is True:
        return true_part
    status = str(read.get("status"))
    if status == "excluded_flagged_adversarial":
        return "excluded"
    if status == "missing_or_excluded":
        return "missing"
    if status == "retired_corpus_specific":
        return "retired_corpus_specific"
    return false_part


def _honest_verdict(
    in_generation: Mapping[str, Any],
    gate_status: str,
    registry: Mapping[str, Any],
    e3: Mapping[str, Any],
    self_learning: Mapping[str, Any],
    hygiene: Mapping[str, Any],
) -> str:
    in_generation_part = _status_part(
        in_generation,
        "in_generation_moat_replicates_headline",
        "replicated_leak_robust",
        "pending",
    )
    self_part = _status_part(
        self_learning, "learned_encoder_transfer_helps", "helps", "open"
    )
    hygiene_part = _status_part(hygiene, "regression_guard_passed", "passed", "open")
    return (
        f"complete: v401_in_generation_{in_generation_part}_gate_{gate_status}_"
        f"arc_levels_{int(registry.get('reproducible_total_levels') or 0)}_"
        f"e3_reproduced_{int(e3.get('reproduced_levels_total') or 0)}_"
        f"self_learning_{self_part}_hygiene_{hygiene_part}"
    )


def _headline_outcome(
    gate_status: str,
    registry: Mapping[str, Any],
    e3: Mapping[str, Any],
    self_learning: Mapping[str, Any],
    paper_ready: bool,
) -> str:
    paper = "paper_ready" if paper_ready else "paper_not_ready"
    self_part = _status_part(
        self_learning, "learned_encoder_transfer_helps", "helps", "open"
    )
    return (
        f"gate_{gate_status}__arc_levels_"
        f"{int(registry.get('reproducible_total_levels') or 0)}_e3_"
        f"{int(e3.get('reproduced_levels_total') or 0)}__"
        f"self_learning_{self_part}__{paper}"
    )


def _blocked_publication_gate_artifact(
    started_s: float,
    now_s: float | None,
) -> JsonDict:
    end = time.time() if now_s is None else now_s
    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "random_seed": RANDOM_SEED,
        "duration_s": round(end - started_s, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": "blocked_publication_gate_missing",
        "headline_outcome": "blocked_publication_gate_missing",
        "in_generation_moat_replicates_headline": False,
        "diffusiongemma_gate_status": "STILL_PENDING_publication_gate_missing",
        "arc_reproducible_total_levels": 0,
        "verifier_thesis_state": "blocked_publication_gate_missing",
        "verifier_is_oracle_honored": True,
        "per_axis_gaps": [],
        "flagged_artifacts_excluded": [],
        "paper_ready": False,
        "unmet_gates": ["publication_gate_missing"],
        "publication_gate": {"error": "scripts/publication_gate.py missing"},
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
    if not (root / "scripts" / "publication_gate.py").exists():
        return _blocked_publication_gate_artifact(start, now_s)

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

    scorer = scorer_leak_read(
        clean["4337_leak_robust_scorer"],
        skipped.get("4337_leak_robust_scorer", False),
    )
    in_generation = in_generation_read(
        clean["4338_in_generation_moat"],
        skipped.get("4338_in_generation_moat", False),
    )
    ar25 = e3_single_read(clean["4339_e3_ar25"], skipped.get("4339_e3_ar25", False), "ar25")
    ka59 = e3_single_read(clean["4340_e3_ka59"], skipped.get("4340_e3_ka59", False), "ka59")
    sc25 = e3_single_read(clean["4341_e3_sc25"], skipped.get("4341_e3_sc25", False), "sc25")
    e3 = _e3_summary(ar25, ka59, sc25)
    self_learning = self_learning_read(
        clean["4342_self_learning"], skipped.get("4342_self_learning", False)
    )
    hygiene = hygiene_read(clean["4344_hygiene"], skipped.get("4344_hygiene", False))
    registry = read_registry_total_levels(root)
    publication_gate = publication_gate_runner(root)
    paper_ready = bool(publication_gate.get("paper_ready"))
    gate_status = diffusiongemma_gate_status(scorer, in_generation)
    violations = _oracle_violations(scorer, in_generation, self_learning)
    thesis = verifier_thesis_state(
        gate_status,
        int(e3.get("reproduced_levels_total") or 0),
        self_learning.get("learned_encoder_transfer_helps") is True,
    )
    end = time.time() if now_s is None else now_s

    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "random_seed": RANDOM_SEED,
        "duration_s": round(end - start, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(
            in_generation, gate_status, registry, e3, self_learning, hygiene
        ),
        "headline_outcome": _headline_outcome(
            gate_status, registry, e3, self_learning, paper_ready
        ),
        "in_generation_moat_replicates_headline": in_generation.get(
            "in_generation_moat_replicates_headline"
        )
        is True
        and scorer.get("scorer_leak_audit_passed") is True,
        "diffusiongemma_gate_status": gate_status,
        "arc_reproducible_total_levels": int(registry.get("reproducible_total_levels") or 0),
        "arc_registry": registry,
        "verifier_thesis_state": thesis,
        "verifier_is_oracle_honored": not violations,
        "oracle_distinct_violations": violations,
        "per_axis_gaps": list(availability_report.get("missing_upstream_artifacts", [])),
        "flagged_artifacts_excluded": exclusions,
        "paper_ready": paper_ready,
        "unmet_gates": base.list_metric(publication_gate, "unmet_gates"),
        "publication_gate": publication_gate,
        "scorer_leak_audit": scorer,
        "in_generation_moat": in_generation,
        "e3_arc_progress": e3,
        "self_learning": self_learning,
        "hygiene": hygiene,
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
    if verdict != "blocked_publication_gate_missing":
        if not isinstance(verdict, str) or not verdict.startswith(
            ("complete:", "success:", "passed:", "shipped:", "blocked:")
        ):
            raise ValueError("honest_verdict must be terminal-prefixed")
    for field in (
        "in_generation_moat_replicates_headline",
        "verifier_is_oracle_honored",
        "paper_ready",
    ):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare bool")
    if not isinstance(artifact.get("arc_reproducible_total_levels"), int) or isinstance(
        artifact.get("arc_reproducible_total_levels"), bool
    ):
        raise ValueError("arc_reproducible_total_levels must be a bare int")
    gate_status = artifact.get("diffusiongemma_gate_status")
    if not isinstance(gate_status, str) or not (
        gate_status == GATE_MET
        or gate_status.startswith("STILL_PENDING_")
        or gate_status.startswith("RETIRED_corpus_specific_")
    ):
        raise ValueError("diffusiongemma_gate_status is not recognized")
    if artifact.get("verifier_thesis_state") not in THESIS_STATES:
        raise ValueError("verifier_thesis_state is not recognized")
    if not isinstance(artifact.get("per_axis_gaps"), list):
        raise ValueError("per_axis_gaps must be a list")
    if not isinstance(artifact.get("flagged_artifacts_excluded"), list):
        raise ValueError("flagged_artifacts_excluded must be a list")
    if not isinstance(artifact.get("upstream_provenance"), list):
        raise ValueError("upstream_provenance must be a list")
    if not base.is_sha256(artifact.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be a sha256 hex string")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required principles")
    for row in artifact["upstream_provenance"]:
        if not base.is_sha256(row.get("sha256")):
            raise ValueError("upstream provenance row has invalid sha256")
        if row.get("skipped") is True and row.get("fields_imported") != []:
            raise ValueError("skipped upstreams must not import fields")
    expected = (
        BLOCKED_PUBLICATION_GATE_CHECKSUM
        if artifact.get("honest_verdict") == "blocked_publication_gate_missing"
        else checksum_from_provenance(artifact["upstream_provenance"])
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
