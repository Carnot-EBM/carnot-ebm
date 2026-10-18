"""Build the Exp 4335 v400 verifier scorecard capstone.

Spec refs: REQ-CAPSTONE-4335, SCENARIO-CAPSTONE-4335.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot.reporting import capstone_aggregate_available as aggregate


JsonDict = dict[str, Any]
LiveFlagRunner = Callable[[Path], list[dict[str, Any]]]
SummarizeRunner = Callable[[Path, Path], int]
PublicationGateRunner = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import adversarial_verify as av  # noqa: E402


OUTPUT_REL_PATH = Path("results/experiment_4335_capstone_v400.json")
EXPERIMENT_ID = 4335
RANDOM_SEED = 4335
SCHEMA = "carnot.capstone_v400_4335.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4335", "SCENARIO-CAPSTONE-4335"]
BLOCKED_CHECKSUM = hashlib.sha256(b"blocked_no_v400_artifacts").hexdigest()
GATE_MET = "MET_oracle_distinct_replicated"

THESIS_STATES = {
    "in_generation_moat_replicated",
    "in_generation_moat_corpus_specific",
    "arc_deep_tail_e3_solving",
    "verifier_domain_bound_self_learning_open",
}


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4325_in_generation_replication": Upstream(
        4325, Path("results/experiment_4325_in_generation_moat_replicate_second_corpus.json")
    ),
    "4326_adaptive": Upstream(
        4326, Path("results/experiment_4326_adaptive_guided_generation_scaleup.json")
    ),
    "4327_e3_ar25": Upstream(
        4327, Path("results/experiment_4327_e3_executable_world_model_ar25.json")
    ),
    "4328_e3_ka59": Upstream(
        4328, Path("results/experiment_4328_e3_executable_world_model_ka59.json")
    ),
    "4329_e3_tr87_ft09": Upstream(
        4329, Path("results/experiment_4329_e3_executable_world_model_tr87_ft09.json")
    ),
    "4330_shallow": Upstream(
        4330, Path("results/experiment_4330_arc_adapter_free_discovery_sweep_shallow_tail.json")
    ),
    "4331_self_learning": Upstream(
        4331,
        Path("results/experiment_4331_self_learning_learned_frame_encoder_cross_game_transfer.json"),
    ),
    "4333_hygiene": Upstream(
        4333, Path("results/experiment_4333_verifier_registry_gaps_hygiene.json")
    ),
}

ARTIFACT_EXPERIMENT_IDS = {
    key: upstream.experiment_id for key, upstream in DEFAULT_UPSTREAMS.items()
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "headline_outcome",
    "in_generation_moat_replicates_headline",
    "diffusiongemma_gate_status",
    "arc_reproducible_total_levels",
    "verifier_thesis_state",
    "flagged_artifacts_excluded",
    "per_axis_gaps",
    "paper_ready",
    "verifier_is_oracle_honored",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. The .400 close-state -- whether the in-generation moat "
        "replicated + the gate flipped, whether E3 advanced ARC, whether the learned "
        "encoder transferred."
    ),
    "headline_outcome": (
        "One honest string aggregating the in-generation + ARC + self-learning reads; "
        "the single line the .401 planner frames from."
    ),
    "in_generation_moat_replicates_headline": (
        "BARE bool: did the in-generation oracle-distinct moat REPLICATE on a 2nd "
        "corpus (exp4325, CI95-excl-0, controls_differentiated, "
        "verifier_is_oracle=false) -- the hardened headline."
    ),
    "diffusiongemma_gate_status": (
        "One honest string: 'MET_oracle_distinct_replicated' iff the oracle-distinct "
        "in-generation win replicated across >=2 corpora with matched controls + "
        "CI95-excl-0; else 'STILL_PENDING_<reason>'. The operator was twice-burned "
        "-> only a REPLICATED oracle-distinct win flips the gate."
    ),
    "arc_reproducible_total_levels": (
        "BARE int: the cumulative OFFLINE-REPRODUCED ARC solved-level count (the "
        "north-star accuracy metric; only reproduced levels count) after the .400 "
        "E3 + sweep advances."
    ),
    "verifier_thesis_state": (
        "One honest string for the verifier thesis state -- the framing the .401 "
        "planner inherits."
    ),
    "flagged_artifacts_excluded": (
        "List of .400 artifacts excluded for flagged_adversarial -- the fabrication "
        "gate (their numbers are NOT aggregated)."
    ),
    "per_axis_gaps": (
        "List of .400 axes whose artifact was MISSING (reported as a gap, NOT "
        "defaulted False) -- the robust-aggregator fix."
    ),
    "paper_ready": (
        "From publication_gate.py --json -- the G1-G4 status (FoVer headline stays "
        "the publication target; a replicated in-generation moat would be a new "
        "headline-grade supporting result)."
    ),
    "verifier_is_oracle_honored": (
        "BARE bool=true -- confirms every cited MOAT/headline result carried "
        "verifier_is_oracle=false (the E3 + sweep execution-grounded solves are "
        "reported as ARC progress, NOT moats)."
    ),
    "reproducibility_checksum": (
        "Hash of the aggregated upstream sha256 set; lets a third party re-derive "
        "the capstone."
    ),
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "4325_in_generation_replication": [
        "in_generation_moat_replicates",
        "replication_ci95",
        "controls_differentiated",
        "scorer_leak_recheck_passed",
        "carnot_minus_best_control_delta",
        "carnot_minus_self_reward_smc_delta",
        "verifier_is_oracle",
    ],
    "4326_adaptive": [
        "adaptive_guidance_beats_control",
        "adaptive_ci95",
        "controls_differentiated",
        "scorer_leak_recheck_passed",
        "carnot_minus_best_control_delta",
        "domain_used",
        "verifier_is_oracle",
    ],
    "4327_e3_ar25": [
        "game",
        "offline_reproduced",
        "reproduced_levels",
        "verifier_accuracy_per_round",
        "verifier_best_accuracy",
        "verifier_is_oracle",
    ],
    "4328_e3_ka59": [
        "game",
        "offline_reproduced",
        "reproduced_levels",
        "verifier_accuracy_per_round",
        "verifier_best_accuracy",
        "verifier_is_oracle",
    ],
    "4329_e3_tr87_ft09": [
        "games",
        "reproduced_levels_total",
        "per_game_scorecard",
        "verifier_is_oracle",
    ],
    "4330_shallow": [
        "offline_reproduced",
        "reproducible_total_levels",
        "games_advanced",
        "prior_reproducible_total_levels",
        "verifier_is_oracle",
    ],
    "4331_self_learning": [
        "learned_encoder_transfer_helps",
        "cross_game_state_reduction",
        "cross_game_state_reduction_ci95",
        "n_held_out_levels",
        "verifier_is_oracle",
    ],
    "4333_hygiene": [
        "regression_guard_passed",
        "registry_reconciled",
        "manifest_reconciled",
        "gaps_logged",
    ],
}


def bool_metric(payload: Mapping[str, Any] | None, field: str) -> bool | None:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, bool) else None


def int_metric(payload: Mapping[str, Any] | None, field: str) -> int:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def float_metric(payload: Mapping[str, Any] | None, field: str) -> float | None:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def str_metric(payload: Mapping[str, Any] | None, field: str) -> str:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def list_metric(payload: Mapping[str, Any] | None, field: str) -> list[Any]:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return list(value) if isinstance(value, list) else []


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def sha_from_payload_checksum(payload: Mapping[str, Any]) -> str:
    value = payload.get("reproducibility_checksum")
    if not isinstance(value, str):
        return ""
    normalized = value.removeprefix("sha256:")
    return normalized if is_sha256(normalized) else ""


def live_has_critical(flags: list[dict[str, Any]]) -> bool:
    return any(str(flag.get("severity", "")).lower() == "critical" for flag in flags)


def run_live_flags(path: Path) -> list[dict[str, Any]]:  # pragma: no cover
    return list(av.verify_artifact(path).get("flags", []))


def run_summarize_artifact(path: Path, root: Path) -> int:  # pragma: no cover
    proc = subprocess.run(
        [sys.executable, "scripts/summarize_artifact.py", str(path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return int(proc.returncode)


def run_publication_gate(root: Path) -> JsonDict:  # pragma: no cover
    proc = subprocess.run(
        [sys.executable, "scripts/publication_gate.py", "--json"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    if not isinstance(payload, dict):
        raise ValueError("publication_gate.py --json returned a non-object payload")
    return payload


def read_json_object(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("non-object")
    return payload


def clean_payload(payload: JsonDict | None, skipped: bool) -> JsonDict | None:
    return None if skipped or payload is None else payload


def ci95_excludes_zero(value: list[Any]) -> bool:
    if len(value) != 2:
        return False
    low, high = value
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        return False
    return (float(low) > 0.0 and float(high) > 0.0) or (
        float(low) < 0.0 and float(high) < 0.0
    )


def _selected_paths(root: Path) -> dict[str, Path]:
    return {key: root / upstream.path for key, upstream in DEFAULT_UPSTREAMS.items()}


def _fields_for_payload(key: str, skipped: bool) -> list[str]:
    return [] if skipped else list(IMPORTED_FIELDS[key])


def _safe_summarize(path: Path, root: Path, runner: SummarizeRunner) -> tuple[int | None, str]:
    try:
        return runner(path, root), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _safe_live_flags(path: Path, runner: LiveFlagRunner) -> list[dict[str, Any]]:
    try:
        return runner(path)
    except Exception as exc:
        return [{"kind": "VERIFY_ERROR", "severity": "warn", "detail": str(exc)}]


def _skipped_payload(payload: JsonDict) -> JsonDict:
    skipped = dict(payload)
    skipped["flagged_adversarial"] = True
    return skipped


def _exclusion_reason(stamped: bool, critical: bool, parse_error: str) -> str:
    if stamped:
        return "flagged_adversarial"
    if critical:
        return "live_critical_adversarial"
    if parse_error:
        return "unparsable_or_non_object"
    return "excluded"


def _read_inputs(
    root: Path,
    live_flag_runner: LiveFlagRunner,
    summarize_runner: SummarizeRunner,
) -> tuple[dict[str, Any], list[JsonDict], list[JsonDict], int]:
    raw_artifacts: dict[str, Any] = {}
    provenance: list[JsonDict] = []
    exclusions: list[JsonDict] = []
    present_count = 0

    for key, path in _selected_paths(root).items():
        upstream = DEFAULT_UPSTREAMS[key]
        if not path.exists():
            raw_artifacts[key] = None
            continue
        present_count += 1
        sha = sha256_file(path)
        summarize_exit_code, summarize_error = _safe_summarize(path, root, summarize_runner)
        live_flags = _safe_live_flags(path, live_flag_runner)
        critical = live_has_critical(live_flags)
        payload: JsonDict | None = None
        parse_error = ""
        try:
            payload = read_json_object(path)
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
                "payload_reproducibility_checksum": sha_from_payload_checksum(payload or {}),
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
                    "reason": _exclusion_reason(stamped, critical, parse_error),
                }
            )
    return raw_artifacts, provenance, exclusions, present_count


def in_generation_replication_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    ci95 = list_metric(payload, "replication_ci95")
    reported = bool_metric(payload, "in_generation_moat_replicates")
    scorer_passed = bool_metric(payload, "scorer_leak_recheck_passed")
    headline = (
        reported is True
        and bool_metric(payload, "controls_differentiated") is True
        and scorer_passed is True
        and ci95_excludes_zero(ci95)
        and bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "replicated" if headline else "corpus_specific",
        "in_generation_moat_replicates_headline": headline,
        "reported_in_generation_moat_replicates": reported,
        "replication_ci95": ci95,
        "replication_ci95_excludes_zero": ci95_excludes_zero(ci95),
        "controls_differentiated": bool_metric(payload, "controls_differentiated"),
        "scorer_leak_recheck_passed": scorer_passed,
        "carnot_minus_best_control_delta": float_metric(
            payload, "carnot_minus_best_control_delta"
        ),
        "carnot_minus_self_reward_smc_delta": float_metric(
            payload, "carnot_minus_self_reward_smc_delta"
        ),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def adaptive_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    ci95 = list_metric(payload, "adaptive_ci95")
    beats = (
        bool_metric(payload, "adaptive_guidance_beats_control") is True
        and bool_metric(payload, "controls_differentiated") is True
        and bool_metric(payload, "scorer_leak_recheck_passed") is True
        and ci95_excludes_zero(ci95)
        and bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "beats_control" if beats else "open",
        "adaptive_guidance_beats_control": beats,
        "reported_adaptive_guidance_beats_control": bool_metric(
            payload, "adaptive_guidance_beats_control"
        ),
        "adaptive_ci95": ci95,
        "adaptive_ci95_excludes_zero": ci95_excludes_zero(ci95),
        "controls_differentiated": bool_metric(payload, "controls_differentiated"),
        "scorer_leak_recheck_passed": bool_metric(payload, "scorer_leak_recheck_passed"),
        "carnot_minus_best_control_delta": float_metric(
            payload, "carnot_minus_best_control_delta"
        ),
        "domain_used": str_metric(payload, "domain_used"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def e3_single_read(payload: JsonDict | None, skipped: bool, fallback_game: str) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    return {
        "status": "reproduced" if bool_metric(payload, "offline_reproduced") is True else "partial",
        "game": str_metric(payload, "game") or fallback_game,
        "offline_reproduced": bool_metric(payload, "offline_reproduced"),
        "reproduced_levels": int_metric(payload, "reproduced_levels"),
        "verifier_accuracy_per_round": list_metric(payload, "verifier_accuracy_per_round"),
        "verifier_best_accuracy": float_metric(payload, "verifier_best_accuracy"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def e3_multi_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    scorecard = payload.get("per_game_scorecard")
    games: dict[str, JsonDict] = {}
    if isinstance(scorecard, Mapping):
        for game, row in scorecard.items():
            if not isinstance(row, Mapping):
                continue
            game_name = str(row.get("game") or game)
            verifier_best = float_metric(row, "best_verifier_accuracy")
            if verifier_best is None:
                verifier_best = float_metric(row, "verifier_best_accuracy")
            games[game_name] = {
                "status": (
                    "reproduced"
                    if bool_metric(row, "offline_reproduced") is True
                    else "partial"
                ),
                "game": game_name,
                "offline_reproduced": bool_metric(row, "offline_reproduced"),
                "reproduced_levels": int_metric(row, "reproduced_levels"),
                "verifier_accuracy_per_round": list_metric(row, "verifier_accuracy_per_round"),
                "verifier_best_accuracy": verifier_best,
                "verifier_is_oracle": bool_metric(row, "verifier_is_oracle")
                if bool_metric(row, "verifier_is_oracle") is not None
                else bool_metric(payload, "verifier_is_oracle"),
                "honest_verdict": str_metric(row, "honest_verdict")
                or str_metric(row, "status"),
            }
    total = int_metric(payload, "reproduced_levels_total") or sum(
        int(row.get("reproduced_levels") or 0) for row in games.values()
    )
    return {
        "status": "reproduced" if total > 0 else "partial",
        "games": games,
        "reproduced_levels_total": total,
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def shallow_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    total = int_metric(payload, "reproducible_total_levels")
    return {
        "status": "included",
        "offline_reproduced": bool_metric(payload, "offline_reproduced"),
        "reproducible_total_levels": total,
        "prior_reproducible_total_levels": int_metric(payload, "prior_reproducible_total_levels"),
        "games_advanced": list_metric(payload, "games_advanced"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def self_learning_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    helps = (
        bool_metric(payload, "learned_encoder_transfer_helps") is True
        and bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "helps" if helps else "open",
        "learned_encoder_transfer_helps": helps,
        "reported_learned_encoder_transfer_helps": bool_metric(
            payload, "learned_encoder_transfer_helps"
        ),
        "cross_game_state_reduction": float_metric(payload, "cross_game_state_reduction"),
        "cross_game_state_reduction_ci95": list_metric(
            payload, "cross_game_state_reduction_ci95"
        ),
        "n_held_out_levels": int_metric(payload, "n_held_out_levels"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def hygiene_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    passed = bool_metric(payload, "regression_guard_passed") is True
    return {
        "status": "passed" if passed else "open",
        "regression_guard_passed": passed,
        "registry_reconciled": bool_metric(payload, "registry_reconciled"),
        "manifest_reconciled": bool_metric(payload, "manifest_reconciled"),
        "gaps_logged_count": len(list_metric(payload, "gaps_logged")),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def _axis_specs() -> list[aggregate.AxisSpec]:
    return [
        aggregate.AxisSpec(
            name="in_generation",
            required_keys=("4325_in_generation_replication", "4326_adaptive"),
            verdict_fn=lambda present: in_generation_replication_read(
                present.get("4325_in_generation_replication"), False
            )["in_generation_moat_replicates_headline"]
            is True,
        ),
        aggregate.AxisSpec(
            name="arc_deep_tail",
            required_keys=("4327_e3_ar25", "4328_e3_ka59", "4329_e3_tr87_ft09"),
            verdict_fn=lambda present: (
                int_metric(present.get("4327_e3_ar25"), "reproduced_levels")
                + int_metric(present.get("4328_e3_ka59"), "reproduced_levels")
                + int_metric(present.get("4329_e3_tr87_ft09"), "reproduced_levels_total")
            )
            > 0,
        ),
        aggregate.AxisSpec(
            name="arc_shallow",
            required_keys=("4330_shallow",),
            verdict_fn=lambda present: int_metric(
                present.get("4330_shallow"), "reproducible_total_levels"
            )
            > 0,
        ),
        aggregate.AxisSpec(
            name="self_learning",
            required_keys=("4331_self_learning",),
            verdict_fn=lambda present: self_learning_read(
                present.get("4331_self_learning"), False
            )["learned_encoder_transfer_helps"]
            is True,
        ),
        aggregate.AxisSpec(
            name="hygiene",
            required_keys=("4333_hygiene",),
            verdict_fn=lambda present: hygiene_read(present.get("4333_hygiene"), False)[
                "regression_guard_passed"
            ]
            is True,
        ),
    ]


def diffusiongemma_gate_status(replication: Mapping[str, Any], blocked: bool) -> str:
    if blocked:
        return "STILL_PENDING_no_v400_artifacts"
    if replication.get("status") in {"missing_or_excluded", "excluded_flagged_adversarial"}:
        return "STILL_PENDING_second_corpus_replication_unavailable"
    if replication.get("verifier_is_oracle") is not False:
        return "STILL_PENDING_verifier_oracle_not_distinct"
    if replication.get("scorer_leak_recheck_passed") is False:
        return "STILL_PENDING_second_corpus_scorer_leaky"
    if replication.get("controls_differentiated") is not True:
        return "STILL_PENDING_controls_not_differentiated"
    if replication.get("replication_ci95_excludes_zero") is not True:
        return "STILL_PENDING_ci95_includes_zero"
    if replication.get("reported_in_generation_moat_replicates") is not True:
        return "STILL_PENDING_second_corpus_replication_false"
    if replication.get("in_generation_moat_replicates_headline") is True:
        return GATE_MET
    return "STILL_PENDING_second_corpus_replication_false"


def verifier_thesis_state(
    in_generation_replicated: bool,
    in_generation_axis_available: bool,
    e3_reproduced: bool,
    self_learning_helps: bool,
) -> str:
    if in_generation_replicated:
        return "in_generation_moat_replicated"
    if e3_reproduced:
        return "arc_deep_tail_e3_solving"
    if in_generation_axis_available:
        return "in_generation_moat_corpus_specific"
    return (
        "verifier_domain_bound_self_learning_open"
        if not self_learning_helps
        else "verifier_domain_bound_self_learning_open"
    )


def _read_status_part(read: Mapping[str, Any], true_key: str, true_part: str, false_part: str) -> str:
    if read.get(true_key) is True:
        return true_part
    status = str(read.get("status"))
    if status == "excluded_flagged_adversarial":
        return "excluded"
    if status == "missing_or_excluded":
        return "missing"
    return false_part


def _e3_total(*reads: Mapping[str, Any]) -> int:
    total = 0
    for read in reads:
        total += int(read.get("reproduced_levels") or 0)
        total += int(read.get("reproduced_levels_total") or 0)
    return total


def _e3_summary(
    ar25: Mapping[str, Any],
    ka59: Mapping[str, Any],
    tr87_ft09: Mapping[str, Any],
) -> JsonDict:
    games: dict[str, Any] = {}
    for read in (ar25, ka59):
        game = read.get("game")
        if isinstance(game, str) and game:
            games[game] = dict(read)
    multi_games = tr87_ft09.get("games")
    if isinstance(multi_games, Mapping):
        games.update({str(key): dict(value) for key, value in multi_games.items()})
    total = _e3_total(ar25, ka59, tr87_ft09)
    return {
        "status": "reproduced" if total > 0 else "partial",
        "reproduced_levels_total": total,
        "games": games,
        "execution_grounded": any(
            row.get("verifier_is_oracle") is True for row in games.values() if isinstance(row, Mapping)
        ),
    }


def _headline_outcome(
    replication: Mapping[str, Any],
    adaptive: Mapping[str, Any],
    e3: Mapping[str, Any],
    shallow: Mapping[str, Any],
    self_learning: Mapping[str, Any],
    paper_ready: bool,
) -> str:
    paper = "paper_ready" if paper_ready else "paper_not_ready"
    in_generation_part = _read_status_part(
        replication,
        "in_generation_moat_replicates_headline",
        "replicated",
        "corpus_specific",
    )
    adaptive_status = str(adaptive.get("status"))
    if adaptive_status in {"missing_or_excluded", "excluded_flagged_adversarial"}:
        adaptive_part = "adaptive_" + ("missing" if adaptive_status == "missing_or_excluded" else "excluded")
    else:
        adaptive_part = (
            "adaptive_beats_control"
            if adaptive.get("adaptive_guidance_beats_control") is True
            else f"adaptive_open_{adaptive.get('domain_used') or 'unknown_domain'}"
        )
    arc_levels = int(shallow.get("reproducible_total_levels") or 0)
    e3_total = int(e3.get("reproduced_levels_total") or 0)
    self_part = _read_status_part(
        self_learning, "learned_encoder_transfer_helps", "helps", "open"
    )
    return (
        f"in_generation_{in_generation_part}__"
        f"{adaptive_part}__"
        f"arc_levels_{arc_levels}_e3_{e3_total}__"
        f"self_learning_{self_part}__{paper}"
    )


def _honest_verdict(
    replication: Mapping[str, Any],
    gate_status: str,
    e3: Mapping[str, Any],
    shallow: Mapping[str, Any],
    self_learning: Mapping[str, Any],
    hygiene: Mapping[str, Any],
) -> str:
    in_generation_part = _read_status_part(
        replication,
        "in_generation_moat_replicates_headline",
        "replicated",
        "corpus_specific",
    )
    self_part = _read_status_part(
        self_learning, "learned_encoder_transfer_helps", "helps", "open"
    )
    hygiene_part = _read_status_part(hygiene, "regression_guard_passed", "passed", "open")
    return (
        f"complete: v400_in_generation_{in_generation_part}_gate_{gate_status}_"
        f"arc_levels_{int(shallow.get('reproducible_total_levels') or 0)}_"
        f"e3_reproduced_{int(e3.get('reproduced_levels_total') or 0)}_"
        f"self_learning_{self_part}_hygiene_{hygiene_part}"
    )


def _oracle_violations(
    replication: Mapping[str, Any],
    adaptive: Mapping[str, Any],
) -> list[str]:
    violations: list[str] = []
    if (
        replication.get("reported_in_generation_moat_replicates") is True
        and replication.get("verifier_is_oracle") is not False
    ):
        violations.append("4325_in_generation_replication:in_generation_replication")
    if (
        adaptive.get("reported_adaptive_guidance_beats_control") is True
        and adaptive.get("verifier_is_oracle") is not False
    ):
        violations.append("4326_adaptive:adaptive_guidance")
    return violations


def checksum_from_provenance(provenance: list[Mapping[str, Any]]) -> str:
    if not provenance:
        return BLOCKED_CHECKSUM
    shas = sorted(str(row["sha256"]) for row in provenance)
    return hashlib.sha256("\n".join(shas).encode("utf-8")).hexdigest()


def _field_provenance(satisfied_by: str) -> dict[str, JsonDict]:
    return {
        field: {"principle": principle, "satisfied_by": satisfied_by}
        for field, principle in FIELD_PRINCIPLES.items()
    }


def _available(read: Mapping[str, Any]) -> bool:
    return read.get("status") not in {"missing_or_excluded", "excluded_flagged_adversarial"}


def build_artifact(
    root: Path = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = run_live_flags,
    summarize_runner: SummarizeRunner = run_summarize_artifact,
    publication_gate_runner: PublicationGateRunner = run_publication_gate,
) -> JsonDict:
    start = time.time() if started_s is None else started_s
    raw_artifacts, provenance, exclusions, present_count = _read_inputs(
        root, live_flag_runner, summarize_runner
    )
    availability_report = aggregate.aggregate_available_report_gaps(
        raw_artifacts,
        _axis_specs(),
        artifact_experiment_ids=ARTIFACT_EXPERIMENT_IDS,
    )
    skipped = {row["artifact_key"]: bool(row["skipped"]) for row in provenance}
    clean = {
        key: clean_payload(
            raw_artifacts.get(key) if isinstance(raw_artifacts.get(key), dict) else None,
            skipped.get(key, False),
        )
        for key in DEFAULT_UPSTREAMS
    }

    replication = in_generation_replication_read(
        clean["4325_in_generation_replication"],
        skipped.get("4325_in_generation_replication", False),
    )
    adaptive = adaptive_read(clean["4326_adaptive"], skipped.get("4326_adaptive", False))
    ar25 = e3_single_read(clean["4327_e3_ar25"], skipped.get("4327_e3_ar25", False), "ar25")
    ka59 = e3_single_read(clean["4328_e3_ka59"], skipped.get("4328_e3_ka59", False), "ka59")
    tr87_ft09 = e3_multi_read(
        clean["4329_e3_tr87_ft09"], skipped.get("4329_e3_tr87_ft09", False)
    )
    e3 = _e3_summary(ar25, ka59, tr87_ft09)
    shallow = shallow_read(clean["4330_shallow"], skipped.get("4330_shallow", False))
    self_learning = self_learning_read(
        clean["4331_self_learning"], skipped.get("4331_self_learning", False)
    )
    hygiene = hygiene_read(clean["4333_hygiene"], skipped.get("4333_hygiene", False))

    publication_gate = publication_gate_runner(root)
    paper_ready = bool(publication_gate.get("paper_ready"))
    blocked = present_count == 0
    gate_status = diffusiongemma_gate_status(replication, blocked)
    violations = _oracle_violations(replication, adaptive)
    thesis = verifier_thesis_state(
        replication.get("in_generation_moat_replicates_headline") is True,
        _available(replication),
        int(e3.get("reproduced_levels_total") or 0) > 0,
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
        "honest_verdict": (
            "blocked_no_v400_artifacts"
            if blocked
            else _honest_verdict(replication, gate_status, e3, shallow, self_learning, hygiene)
        ),
        "headline_outcome": (
            "blocked_no_v400_artifacts"
            if blocked
            else _headline_outcome(replication, adaptive, e3, shallow, self_learning, paper_ready)
        ),
        "in_generation_moat_replicates_headline": replication.get(
            "in_generation_moat_replicates_headline"
        )
        is True,
        "diffusiongemma_gate_status": gate_status,
        "arc_reproducible_total_levels": int(shallow.get("reproducible_total_levels") or 0),
        "verifier_thesis_state": (
            "verifier_domain_bound_self_learning_open" if blocked else thesis
        ),
        "flagged_artifacts_excluded": exclusions,
        "per_axis_gaps": list(availability_report.get("missing_upstream_artifacts", [])),
        "paper_ready": paper_ready,
        "unmet_gates": list_metric(publication_gate, "unmet_gates"),
        "publication_gate": publication_gate,
        "verifier_is_oracle_honored": not violations,
        "oracle_distinct_violations": violations,
        "in_generation_replication": replication,
        "adaptive_scaleup": adaptive,
        "e3_deep_tail": e3,
        "arc_shallow": shallow,
        "self_learning": self_learning,
        "hygiene": hygiene,
        "availability_report": availability_report,
        "upstream_provenance": provenance,
        "upstream_sha256_set": sorted(str(row["sha256"]) for row in provenance),
        "reproducibility_checksum": checksum_from_provenance([] if blocked else provenance),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance(
            "blocked precondition" if blocked else "aggregation logic"
        ),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact.get("honest_verdict")
    if verdict != "blocked_no_v400_artifacts":
        if not isinstance(verdict, str) or not verdict.startswith(
            ("complete:", "success:", "passed:", "shipped:", "blocked:")
        ):
            raise ValueError("honest_verdict must be terminal-prefixed")
    headline = artifact.get("headline_outcome")
    if not isinstance(headline, str) or not headline:
        raise ValueError("headline_outcome must be a non-empty string")
    for field in (
        "in_generation_moat_replicates_headline",
        "paper_ready",
        "verifier_is_oracle_honored",
    ):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare bool")
    if not isinstance(artifact.get("arc_reproducible_total_levels"), int) or isinstance(
        artifact.get("arc_reproducible_total_levels"), bool
    ):
        raise ValueError("arc_reproducible_total_levels must be a bare int")
    gate_status = artifact.get("diffusiongemma_gate_status")
    if not isinstance(gate_status, str) or not (
        gate_status == GATE_MET or gate_status.startswith("STILL_PENDING_")
    ):
        raise ValueError("diffusiongemma_gate_status is not recognized")
    if artifact.get("verifier_thesis_state") not in THESIS_STATES:
        raise ValueError("verifier_thesis_state is not recognized")
    if not isinstance(artifact.get("flagged_artifacts_excluded"), list):
        raise ValueError("flagged_artifacts_excluded must be a list")
    if not isinstance(artifact.get("per_axis_gaps"), list):
        raise ValueError("per_axis_gaps must be a list")
    if not is_sha256(artifact.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be a sha256 hex string")
    if not isinstance(artifact.get("upstream_provenance"), list):
        raise ValueError("upstream_provenance must be a list")
    principles = artifact.get("field_principles")
    if principles != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required principles")
    for row in artifact["upstream_provenance"]:
        if not is_sha256(row.get("sha256")):
            raise ValueError("upstream provenance row has invalid sha256")
        if row.get("skipped") is True and row.get("fields_imported") != []:
            raise ValueError("skipped upstreams must not import fields")
    expected = checksum_from_provenance(artifact["upstream_provenance"])
    if artifact.get("reproducibility_checksum") != expected:
        raise ValueError("reproducibility_checksum does not match upstream sha256 set")


def write_artifact(
    root: Path = REPO_ROOT,
    *,
    output_path: Path = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = run_live_flags,
    summarize_runner: SummarizeRunner = run_summarize_artifact,
    publication_gate_runner: PublicationGateRunner = run_publication_gate,
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
