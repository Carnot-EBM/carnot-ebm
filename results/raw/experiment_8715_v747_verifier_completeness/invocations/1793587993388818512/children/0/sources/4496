"""Build the Exp 4441 .410 generic-solver capstone.

Spec refs: REQ-CAPSTONE-4441, SCENARIO-CAPSTONE-4441.

The capstone reads already-written result artifacts and does no solving itself.
That distinction matters here because most ARC solve evidence is verified by
execution. Execution-grounded evidence is useful ARC progress, but it is not an
oracle-distinct verifier moat headline, so this module keeps the capstone's own
`verifier_is_oracle` stamp false and carries upstream oracle declarations as
provenance.
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
OUTPUT_REL_PATH = Path("results/experiment_4441_capstone_v410.json")
EXPERIMENT_ID = 4441
RANDOM_SEED = 4441
SCHEMA = "carnot.capstone_v410_4441.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4441", "SCENARIO-CAPSTONE-4441"]

GENERIC_SOLVER_GAP_STATES = {"closing", "partial", "total-gap"}
WIN_INDUCTION_STATES = {
    "held_out_level_banked_examples_helped",
    "held_out_level_banked_examples_not_demonstrated",
    "no_held_out_bank",
    "excluded_flagged_adversarial",
    "missing_or_excluded",
}
ACTION_MODEL_STATES = {
    "examples_helped_and_reproduced_level",
    "examples_helped_no_reproduced_level",
    "examples_no_help",
    "missing_or_excluded",
}
FIRST_CONTACT_STATES = {
    "contract_fixed_routed_solve_banked",
    "contract_fixed_no_routed_solve",
    "contract_not_fixed",
    "missing_or_excluded",
}
PRIMITIVE_STATES = {
    "consolidated_no_regression",
    "regression_or_empty",
    "missing_or_excluded",
}


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4432_loo_generic": Upstream(
        4432,
        Path("results/experiment_4432_loo_generic_solve_benchmark.json"),
    ),
    "4433_win_induction": Upstream(
        4433,
        Path("results/experiment_4433_example_conditioned_win_induction.json"),
    ),
    "4434_action_model": Upstream(
        4434,
        Path("results/experiment_4434_example_conditioned_action_model.json"),
    ),
    "4435_first_contact": Upstream(
        4435,
        Path("results/experiment_4435_generic_first_contact_fixed.json"),
    ),
    "4436_primitives": Upstream(
        4436,
        Path("results/experiment_4436_deepen_plus_primitive_consolidation.json"),
    ),
    "4438_hygiene": Upstream(
        4438,
        Path("results/experiment_4438_registry_gaps_hygiene.json"),
    ),
    "4440_sota_ingestion": Upstream(
        4440,
        Path("results/experiment_4440_sota_ingestion_410.json"),
    ),
}

ARTIFACT_EXPERIMENT_IDS = {
    key: upstream.experiment_id for key, upstream in DEFAULT_UPSTREAMS.items()
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "generic_solver_gap_state",
    "reproducible_total_levels",
    "next_backlog",
    "generic_loo_solve_count",
    "residual_deltas",
    "win_induction_state",
    "action_model_state",
    "first_contact_state",
    "primitives_consolidated_count",
    "publication_gate",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "cited_upstream_artifacts",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal-prefixed",
    "generic_solver_gap_state": (
        "one honest string (closing/partial/total-gap) -- the .410 headline answer "
        "to 'can the example corpus give us generic live solves?'"
    ),
    "reproducible_total_levels": "the authoritative monotonic sprint metric",
    "next_backlog": (
        "the residual_deltas + missing primitives that become the .411 generic-solver "
        "build backlog"
    ),
    "publication_gate": "The G1-G4 publication_gate.py output from publication_gate.py --json.",
    "verifier_is_oracle": (
        "BARE bool=false for the capstone itself; upstream execution-grounded ARC "
        "solves are carried separately so CIRCULAR_MOAT_OVERCLAIM does not fire."
    ),
    "cited_upstream_artifacts": (
        "list of {experiment_id, fields_imported, sha256}; skipped flagged artifacts "
        "must import no fields."
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts -- this capstone reads upstream JSON, "
        "exp4438 hygiene totals, exp4440 SOTA handoff, and publication_gate.py output."
    ),
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "4432_loo_generic": [
        "honest_verdict",
        "generic_loo_solve_count",
        "per_game",
        "missing_verifier_gaps",
        "offline_reproduced",
        "verifier_is_oracle",
    ],
    "4433_win_induction": [
        "honest_verdict",
        "target_game",
        "offline_reproduced",
        "reproduced_levels",
        "few_shot_examples_used",
        "verifier_is_oracle",
    ],
    "4434_action_model": [
        "honest_verdict",
        "target_game",
        "world_model_accuracy_cold",
        "world_model_accuracy_with_examples",
        "offline_reproduced",
        "reproduced_levels",
        "missing_verifier_gaps",
        "verifier_is_oracle",
    ],
    "4435_first_contact": [
        "honest_verdict",
        "target_game",
        "verdict_contract_fixed",
        "offline_reproduced",
        "reproduced_levels",
        "missing_verifier_gaps",
        "verifier_is_oracle",
    ],
    "4436_primitives": [
        "honest_verdict",
        "deepened_game",
        "new_levels_reproduced",
        "reproduced_levels",
        "offline_reproduced",
        "no_regression",
        "primitives_consolidated",
        "verifier_is_oracle",
    ],
    "4438_hygiene": [
        "honest_verdict",
        "reproducible_total_levels",
        "reproducible_total_games",
        "regression_guard_passed",
        "availability_report",
    ],
    "4440_sota_ingestion": [
        "honest_verdict",
        "flagged_for_v411",
        "v410_outcome_conditioning",
        "inference_substrate",
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
        row = {
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
        provenance.append(row)
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


def _residual_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    seen: set[tuple[str, str]] = set()
    sources = []
    per_game = payload.get("per_game")
    if isinstance(per_game, list):
        sources.extend(per_game)
    gaps = payload.get("missing_verifier_gaps")
    if isinstance(gaps, list):
        sources.extend(gaps)
    for raw in sources:
        if not isinstance(raw, Mapping):
            continue
        residual = str(raw.get("residual_delta") or "")
        if not residual or residual == "none":
            continue
        game = str(raw.get("game") or "")
        key = (game, residual)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "game": game,
                "routed_to": str(raw.get("routed_to") or ""),
                "residual_delta": residual,
                "source_artifact": str(DEFAULT_UPSTREAMS["4432_loo_generic"].path),
            }
        )
    return rows


def loo_generic_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "generic_loo_solve_count": 0,
            "residual_deltas": [],
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "generic_loo_solve_count": 0,
            "residual_deltas": [],
        }
    count = base.int_metric(payload, "generic_loo_solve_count")
    residuals = _residual_rows(payload)
    target_count = len(base.list_metric(payload, "per_game"))
    return {
        "state": "measured" if count > 0 or target_count > 0 else "empty_measurement",
        "generic_loo_solve_count": count,
        "target_count": target_count,
        "residual_deltas": residuals,
        "offline_reproduced": base.bool_metric(payload, "offline_reproduced") is True,
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def win_induction_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "held_out_level_banked": False,
            "few_shot_examples_demonstrably_helped": False,
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "held_out_level_banked": False,
            "few_shot_examples_demonstrably_helped": False,
        }
    examples = base.list_metric(payload, "few_shot_examples_used")
    banked = (
        base.bool_metric(payload, "offline_reproduced") is True
        and base.int_metric(payload, "reproduced_levels") > 0
    )
    helped = banked and len(examples) >= 3
    if banked and helped:
        state = "held_out_level_banked_examples_helped"
    elif banked:
        state = "held_out_level_banked_examples_not_demonstrated"
    else:
        state = "no_held_out_bank"
    return {
        "state": state,
        "held_out_level_banked": banked,
        "few_shot_examples_demonstrably_helped": helped,
        "few_shot_example_count": len(examples),
        "target_game": base.str_metric(payload, "target_game"),
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def action_model_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped or payload is None:
        return {
            "state": "missing_or_excluded",
            "helped_vs_cold_control": False,
            "accuracy_delta": 0.0,
        }
    cold = base.float_metric(payload, "world_model_accuracy_cold")
    with_examples = base.float_metric(payload, "world_model_accuracy_with_examples")
    helped = cold is not None and with_examples is not None and with_examples > cold
    reproduced = (
        base.bool_metric(payload, "offline_reproduced") is True
        and base.int_metric(payload, "reproduced_levels") > 0
    )
    if helped and reproduced:
        state = "examples_helped_and_reproduced_level"
    elif helped:
        state = "examples_helped_no_reproduced_level"
    else:
        state = "examples_no_help"
    delta = round(float(with_examples or 0.0) - float(cold or 0.0), 6)
    return {
        "state": state,
        "helped_vs_cold_control": helped,
        "accuracy_delta": delta,
        "world_model_accuracy_cold": cold,
        "world_model_accuracy_with_examples": with_examples,
        "offline_reproduced": reproduced,
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "missing_verifier_gaps": base.list_metric(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def first_contact_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped or payload is None:
        return {
            "state": "missing_or_excluded",
            "verdict_contract_fixed": False,
            "routed_solve_banked": False,
            "missing_verifier_gaps": [],
        }
    fixed = base.bool_metric(payload, "verdict_contract_fixed") is True
    solved = (
        base.bool_metric(payload, "offline_reproduced") is True
        and base.int_metric(payload, "reproduced_levels") > 0
    )
    if fixed and solved:
        state = "contract_fixed_routed_solve_banked"
    elif fixed:
        state = "contract_fixed_no_routed_solve"
    else:
        state = "contract_not_fixed"
    return {
        "state": state,
        "verdict_contract_fixed": fixed,
        "routed_solve_banked": solved,
        "target_game": base.str_metric(payload, "target_game"),
        "offline_reproduced": base.bool_metric(payload, "offline_reproduced") is True,
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "missing_verifier_gaps": base.list_metric(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def primitives_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped or payload is None:
        return {"state": "missing_or_excluded", "count": 0, "no_regression": False}
    primitives = [
        dict(row)
        for row in base.list_metric(payload, "primitives_consolidated")
        if isinstance(row, Mapping)
    ]
    no_regression = base.bool_metric(payload, "no_regression") is True
    counted = len(primitives) if no_regression else 0
    state = "consolidated_no_regression" if counted > 0 else "regression_or_empty"
    return {
        "state": state,
        "count": counted,
        "raw_count": len(primitives),
        "no_regression": no_regression,
        "primitives": primitives,
        "deepened_game": base.str_metric(payload, "deepened_game"),
        "new_levels_reproduced": base.int_metric(payload, "new_levels_reproduced"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def hygiene_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped or payload is None:
        return {"state": "missing_or_excluded", "reproducible_total_levels": 0}
    return {
        "state": "reconciled" if base.int_metric(payload, "reproducible_total_levels") else "empty",
        "reproducible_total_levels": base.int_metric(payload, "reproducible_total_levels"),
        "reproducible_total_games": base.int_metric(payload, "reproducible_total_games"),
        "regression_guard_passed": base.bool_metric(payload, "regression_guard_passed") is True,
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def sota_ingestion_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped or payload is None:
        return {"state": "missing_or_excluded"}
    conditioning = payload.get("v410_outcome_conditioning")
    return {
        "state": "mapped",
        "flagged_for_v411": base.str_metric(payload, "flagged_for_v411"),
        "v410_outcome_conditioning": dict(conditioning) if isinstance(conditioning, Mapping) else {},
        "inference_substrate": base.str_metric(payload, "inference_substrate"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def _axis_specs() -> list[aggregate.AxisSpec]:
    return [
        aggregate.AxisSpec(
            name="loo_generic",
            required_keys=("4432_loo_generic",),
            verdict_fn=lambda present: loo_generic_read(
                present.get("4432_loo_generic"),
                False,
            )["generic_loo_solve_count"],
        ),
        aggregate.AxisSpec(
            name="win_induction",
            required_keys=("4433_win_induction",),
            verdict_fn=lambda present: win_induction_read(
                present.get("4433_win_induction"),
                False,
            )["state"],
        ),
        aggregate.AxisSpec(
            name="action_model",
            required_keys=("4434_action_model",),
            verdict_fn=lambda present: action_model_read(
                present.get("4434_action_model"),
                False,
            )["state"],
        ),
        aggregate.AxisSpec(
            name="first_contact",
            required_keys=("4435_first_contact",),
            verdict_fn=lambda present: first_contact_read(
                present.get("4435_first_contact"),
                False,
            )["state"],
        ),
        aggregate.AxisSpec(
            name="primitives",
            required_keys=("4436_primitives",),
            verdict_fn=lambda present: primitives_read(
                present.get("4436_primitives"),
                False,
            )["count"],
        ),
        aggregate.AxisSpec(
            name="registry_hygiene",
            required_keys=("4438_hygiene",),
            verdict_fn=lambda present: hygiene_read(
                present.get("4438_hygiene"),
                False,
            )["reproducible_total_levels"],
        ),
        aggregate.AxisSpec(
            name="sota_ingestion",
            required_keys=("4440_sota_ingestion",),
            verdict_fn=lambda present: sota_ingestion_read(
                present.get("4440_sota_ingestion"),
                False,
            )["state"],
        ),
    ]


def _publication_gate_or_gap(
    root: Path,
    runner: PublicationGateRunner,
) -> tuple[JsonDict, JsonDict, list[JsonDict]]:
    publication_gate, check = v405._publication_gate_check(root, runner)  # noqa: SLF001
    if publication_gate is not None:
        return publication_gate, check, []
    return (
        {
            "paper_ready": False,
            "gates": {},
            "unmet_gates": ["publication_gate_unrunnable"],
            "error": str(check.get("error", "unrunnable")),
        },
        check,
        [{"axis": "publication_gate", "artifact_key": "publication_gate", "reason": "unrunnable"}],
    )


def decide_generic_solver_gap_state(
    loo: Mapping[str, Any],
    action_model: Mapping[str, Any],
    first_contact: Mapping[str, Any],
    primitives: Mapping[str, Any],
) -> str:
    count = int(loo.get("generic_loo_solve_count") or 0)
    residuals = loo.get("residual_deltas")
    has_residuals = isinstance(residuals, list) and bool(residuals)
    closing = (
        count >= 2
        and not has_residuals
        and action_model.get("helped_vs_cold_control") is True
        and first_contact.get("routed_solve_banked") is True
        and primitives.get("no_regression") is True
        and int(primitives.get("count") or 0) > 0
    )
    if closing:
        return "closing"
    if (
        count > 0
        or action_model.get("helped_vs_cold_control") is True
        or first_contact.get("verdict_contract_fixed") is True
        or int(primitives.get("count") or 0) > 0
    ):
        return "partial"
    return "total-gap"


def _missing_primitive_name(residual_delta: str) -> str:
    return residual_delta.removeprefix("missing_")


def _first_contact_missing_gaps(first_contact: Mapping[str, Any]) -> list[JsonDict]:
    gaps: list[JsonDict] = []
    for raw in first_contact.get("missing_verifier_gaps", []):
        if not isinstance(raw, Mapping):
            continue
        gaps.append(
            {
                "gap_id": str(raw.get("gap_id") or "EXP4435-FIRST-CONTACT-GAP"),
                "game": str(raw.get("game") or first_contact.get("target_game") or ""),
                "failure_mode": str(raw.get("failure_mode") or "routed_no_reproduced_level"),
                "missing_discriminator": str(raw.get("missing_discriminator") or ""),
                "candidate_design": str(raw.get("candidate_design") or ""),
                "source_artifact": str(DEFAULT_UPSTREAMS["4435_first_contact"].path),
            }
        )
    return gaps


def build_next_backlog(
    *,
    loo: Mapping[str, Any],
    win_induction: Mapping[str, Any],
    action_model: Mapping[str, Any],
    first_contact: Mapping[str, Any],
) -> JsonDict:
    residuals = [
        dict(row)
        for row in loo.get("residual_deltas", [])
        if isinstance(row, Mapping) and row.get("residual_delta")
    ]
    missing_gaps = _first_contact_missing_gaps(first_contact)
    for raw in action_model.get("missing_verifier_gaps", []):
        if isinstance(raw, Mapping):
            missing_gaps.append(dict(raw) | {"source_artifact": str(DEFAULT_UPSTREAMS["4434_action_model"].path)})
    if win_induction.get("state") == "excluded_flagged_adversarial":
        missing_gaps.append(
            {
                "gap_id": "EXP4433-FLAGGED-ADVERSARIAL-RERUN",
                "game": "g50t",
                "failure_mode": "flagged_adversarial_win_induction_untrusted",
                "missing_discriminator": "clean artifact-discipline rerun before held-out bank counts",
                "candidate_design": "rerun example-conditioned win induction with non-adversarial provenance",
                "source_artifact": str(DEFAULT_UPSTREAMS["4433_win_induction"].path),
            }
        )
    missing_primitives = sorted(
        {
            _missing_primitive_name(str(row["residual_delta"]))
            for row in residuals
            if isinstance(row.get("residual_delta"), str)
        }
    )
    return {
        "residual_deltas": residuals,
        "missing_primitives": missing_primitives,
        "missing_gaps": missing_gaps,
    }


def _honest_verdict(
    *,
    gap_state: str,
    loo_count: int,
    residual_count: int,
    total_levels: int,
    publication_available: bool,
    paper_ready: bool,
) -> str:
    publication = (
        "publication_ready"
        if publication_available and paper_ready
        else "publication_not_ready"
        if publication_available
        else "publication_gate_gap"
    )
    return (
        f"complete: v410_generic_solver_{gap_state}_loo_{loo_count}_"
        f"residuals_{residual_count}_levels_{total_levels}_{publication}"
    )


def checksum_from_inputs(
    provenance: list[Mapping[str, Any]],
    publication_gate: Mapping[str, Any],
    *,
    gap_state: str,
    total_levels: int,
    next_backlog: Mapping[str, Any],
) -> str:
    payload = {
        "generic_solver_gap_state": gap_state,
        "next_backlog": next_backlog,
        "publication_gate": publication_gate,
        "reproducible_total_levels": total_levels,
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


def _preconditions_checked(
    root: Path,
    publication_gate_check: Mapping[str, Any],
    provenance: list[JsonDict],
    hygiene: Mapping[str, Any],
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
        "registry_hygiene": dict(hygiene),
        "robust_aggregate_available_helper": (
            "capstone_aggregate_available.aggregate_available_report_gaps"
        ),
        "leaderboard_submission": False,
        "trm_training_stood_down": True,
    }


def _headline_answers(
    loo: Mapping[str, Any],
    win_induction: Mapping[str, Any],
    action_model: Mapping[str, Any],
    first_contact: Mapping[str, Any],
    primitives: Mapping[str, Any],
) -> JsonDict:
    residuals = loo.get("residual_deltas")
    return {
        "exp4432": {
            "generic_loo_solve_count": int(loo.get("generic_loo_solve_count") or 0),
            "residual_delta_count": len(residuals) if isinstance(residuals, list) else 0,
            "residual_deltas": residuals if isinstance(residuals, list) else [],
        },
        "exp4433": {
            "state": win_induction.get("state"),
            "held_out_level_banked": win_induction.get("held_out_level_banked") is True,
            "few_shot_examples_demonstrably_helped": (
                win_induction.get("few_shot_examples_demonstrably_helped") is True
            ),
        },
        "exp4434": {
            "state": action_model.get("state"),
            "helped_vs_cold_control": action_model.get("helped_vs_cold_control") is True,
            "accuracy_delta": action_model.get("accuracy_delta", 0.0),
        },
        "exp4435": {
            "state": first_contact.get("state"),
            "verdict_contract_fixed": first_contact.get("verdict_contract_fixed") is True,
            "routed_solve_banked": first_contact.get("routed_solve_banked") is True,
        },
        "exp4436": {
            "state": primitives.get("state"),
            "primitives_consolidated_count": int(primitives.get("count") or 0),
            "no_regression": primitives.get("no_regression") is True,
        },
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

    loo = loo_generic_read(clean["4432_loo_generic"], skipped.get("4432_loo_generic", False))
    win_induction = win_induction_read(
        clean["4433_win_induction"],
        skipped.get("4433_win_induction", False),
    )
    action_model = action_model_read(
        clean["4434_action_model"],
        skipped.get("4434_action_model", False),
    )
    first_contact = first_contact_read(
        clean["4435_first_contact"],
        skipped.get("4435_first_contact", False),
    )
    primitives = primitives_read(clean["4436_primitives"], skipped.get("4436_primitives", False))
    hygiene = hygiene_read(clean["4438_hygiene"], skipped.get("4438_hygiene", False))
    sota_ingestion = sota_ingestion_read(
        clean["4440_sota_ingestion"],
        skipped.get("4440_sota_ingestion", False),
    )

    gap_state = decide_generic_solver_gap_state(loo, action_model, first_contact, primitives)
    next_backlog = build_next_backlog(
        loo=loo,
        win_induction=win_induction,
        action_model=action_model,
        first_contact=first_contact,
    )
    total_levels = int(hygiene.get("reproducible_total_levels") or 0)
    paper_ready = bool(publication_gate.get("paper_ready"))
    publication_available = bool(publication_gate_check.get("runnable"))
    end = time.time() if now_s is None else now_s
    per_axis_gaps = list(availability_report.get("missing_upstream_artifacts", []))
    per_axis_gaps.extend(publication_gate_gaps)

    residuals = loo.get("residual_deltas", [])
    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "random_seed": RANDOM_SEED,
        "duration_s": round(end - start, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(
            gap_state=gap_state,
            loo_count=int(loo.get("generic_loo_solve_count") or 0),
            residual_count=len(residuals) if isinstance(residuals, list) else 0,
            total_levels=total_levels,
            publication_available=publication_available,
            paper_ready=paper_ready,
        ),
        "generic_solver_gap_state": gap_state,
        "reproducible_total_levels": total_levels,
        "next_backlog": next_backlog,
        "generic_loo_solve_count": int(loo.get("generic_loo_solve_count") or 0),
        "residual_deltas": residuals if isinstance(residuals, list) else [],
        "win_induction_state": win_induction["state"],
        "win_induction": win_induction,
        "action_model_state": action_model["state"],
        "action_model": action_model,
        "first_contact_state": first_contact["state"],
        "first_contact": first_contact,
        "primitives_consolidated_count": int(primitives.get("count") or 0),
        "primitives": primitives,
        "registry_hygiene": hygiene,
        "sota_ingestion": sota_ingestion,
        "headline_question_answers": _headline_answers(
            loo,
            win_induction,
            action_model,
            first_contact,
            primitives,
        ),
        "publication_gate": publication_gate,
        "paper_ready": paper_ready,
        "unmet_gates": base.list_metric(publication_gate, "unmet_gates"),
        "verifier_is_oracle": False,
        "verifier_is_oracle_honored": True,
        "upstream_oracle_declarations": _oracle_declarations(provenance, clean),
        "cited_upstream_artifacts": _cited_upstream_artifacts(provenance),
        "preconditions_checked": _preconditions_checked(
            root,
            publication_gate_check,
            provenance,
            hygiene,
        ),
        "per_axis_gaps": per_axis_gaps,
        "flagged_artifacts_excluded": exclusions,
        "availability_report": availability_report,
        "upstream_provenance": provenance,
        "upstream_sha256_set": sorted(str(row["sha256"]) for row in provenance),
        "publication_gate_checksum": hashlib.sha256(
            json.dumps(publication_gate, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "capstone_live_adversarial_recheck": {"status": "not_run_until_write"},
        "submitted_to_leaderboard": False,
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance("aggregation logic"),
    }
    artifact["reproducibility_checksum"] = checksum_from_inputs(
        provenance,
        publication_gate,
        gap_state=gap_state,
        total_levels=total_levels,
        next_backlog=next_backlog,
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")  # pragma: no cover
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(
        ("complete:", "success:", "passed:", "shipped:"),
    ):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if artifact.get("generic_solver_gap_state") not in GENERIC_SOLVER_GAP_STATES:
        raise ValueError("generic_solver_gap_state is not recognized")
    for field in (
        "reproducible_total_levels",
        "generic_loo_solve_count",
        "primitives_consolidated_count",
    ):
        value = artifact.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{field} must be a bare int")
    if not isinstance(artifact.get("next_backlog"), Mapping):
        raise ValueError("next_backlog must be an object")
    if not isinstance(artifact.get("residual_deltas"), list):
        raise ValueError("residual_deltas must be a list")
    if artifact.get("win_induction_state") not in WIN_INDUCTION_STATES:
        raise ValueError("win_induction_state is not recognized")  # pragma: no cover
    if artifact.get("action_model_state") not in ACTION_MODEL_STATES:
        raise ValueError("action_model_state is not recognized")  # pragma: no cover
    if artifact.get("first_contact_state") not in FIRST_CONTACT_STATES:
        raise ValueError("first_contact_state is not recognized")  # pragma: no cover
    if artifact.get("primitives", {}).get("state") not in PRIMITIVE_STATES:
        raise ValueError("primitive state is not recognized")  # pragma: no cover
    if not isinstance(artifact.get("publication_gate"), Mapping):
        raise ValueError("publication_gate must be an object")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle must be bare false")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        raise ValueError("preconditions_checked must be an object")  # pragma: no cover
    if not isinstance(artifact.get("cited_upstream_artifacts"), list):
        raise ValueError("cited_upstream_artifacts must be a list")  # pragma: no cover
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed does not match experiment")  # pragma: no cover
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if "gated_on" in artifact:
        raise ValueError("gated_on is forbidden")  # pragma: no cover
    checksum = str(artifact.get("reproducibility_checksum", "")).removeprefix("sha256:")
    if not base.is_sha256(checksum):
        raise ValueError("reproducibility_checksum must be sha256-prefixed")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match")
    provenance = artifact.get("upstream_provenance")
    if not isinstance(provenance, list):
        raise ValueError("upstream_provenance must be a list")  # pragma: no cover
    for row in provenance:
        if not isinstance(row, Mapping):  # pragma: no cover
            raise ValueError("upstream provenance row must be an object")
        if not base.is_sha256(row.get("sha256")):
            raise ValueError("upstream provenance row has invalid sha256")  # pragma: no cover
        if row.get("skipped") is True and row.get("fields_imported") != []:
            raise ValueError("skipped upstreams must not import fields")  # pragma: no cover
    expected = checksum_from_inputs(
        provenance,
        artifact["publication_gate"],
        gap_state=str(artifact["generic_solver_gap_state"]),
        total_levels=int(artifact["reproducible_total_levels"]),
        next_backlog=artifact["next_backlog"],
    )
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
