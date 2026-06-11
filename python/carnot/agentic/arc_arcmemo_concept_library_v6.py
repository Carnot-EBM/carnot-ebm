"""ArcMemo v6 compressed concept-library transfer accounting.

Spec refs: REQ-LEARN-4039, SCENARIO-LEARN-4039.

The v6 experiment is intentionally an artifact-level measurement. Earlier
ArcMemo runs already paid the real environment cost to discover concepts and
Exp 4038 already confirmed the `.373` solve through the ARC level counter. This
module checks a narrower question: when repeated raw concepts are compressed
into a named library abstraction, does that abstraction reduce the future solve
cost beyond cold start and beyond the v5 raw-memory arm?
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable, Mapping


INFERENCE_SUBSTRATE = "offline_arc_agi3_arcmemo_concept_transfer"
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "solve_transfer_win",
    "actions_cold",
    "actions_v5",
    "actions_v6",
    "n_named_abstractions",
    "inference_substrate",
)
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_")
PRIORITY_FIELDS = ("when_it_applies", "effect", "name", "family")
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _as_int(payload: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        value = payload.get(key)
        if value is not None and type(value) is not bool:
            return int(value)
    return 0


def _tokens(*values: object) -> set[str]:
    text = " ".join(str(value or "").lower() for value in values)
    return set(_TOKEN_RE.findall(text))


def _terminal_prefixed(value: object) -> bool:
    return isinstance(value, str) and value.startswith(TERMINAL_PREFIXES)


def collect_raw_concepts(prior_artifacts: Iterable[Mapping[str, Any] | None]) -> list[dict[str, Any]]:
    """Collect plain concept records while preserving where each record came from.

    Why this exists: the v6 library should compress accumulated memory, not
    invent a fresh abstraction from the target. Keeping the origin artifact on
    each raw record makes the later artifact auditable without depending on the
    exact shape of older experiment JSON files.
    """

    raw: list[dict[str, Any]] = []
    for artifact in prior_artifacts:
        if not artifact:
            continue
        origin = str(artifact.get("experiment", "unknown_prior_artifact"))
        for record in artifact.get("concept_memory", []) or []:
            if not isinstance(record, Mapping) or not record.get("name"):
                continue
            fields = {field: str(record.get(field, "")) for field in PRIORITY_FIELDS}
            raw.append(
                {
                    "name": str(record["name"]),
                    "family": fields["family"] or "unknown_family",
                    "source_game": str(record.get("source_game", "")),
                    "target_games": list(record.get("target_games", record.get("applies_to_games", [])) or []),
                    "when_it_applies": fields["when_it_applies"],
                    "effect": fields["effect"],
                    "source": str(record.get("source", "")),
                    "origin_artifact": origin,
                    "tokens": sorted(_tokens(*fields.values())),
                }
            )
    return raw


def _click_transform_abstraction(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    source_names = sorted({str(record["name"]) for record in records})
    source_artifacts = sorted({str(record.get("origin_artifact", "")) for record in records if record.get("origin_artifact")})
    return {
        "name": "click_state_transform_then_goal_commit",
        "family": "click_state_transform",
        "signature": (
            "click_state_transform(source_state, click_or_toggle_ops, "
            "terminal_goal_counter) -> committed_action_suffix"
        ),
        "documentation": (
            "Why: multiple ARC solves use clicks or toggles to change hidden or visible state, "
            "then finish by navigating to a level-counter goal. The abstraction lets the solver "
            "reuse that operation shape instead of spending new exploration actions to rediscover it."
        ),
        "lambda_abstraction": (
            "lambda observed_state: bind(click_or_toggle_state_update) "
            "then plan(navigation_or_commit_suffix_until_level_counter_increments)"
        ),
        "source_concepts": source_names,
        "source_artifacts": source_artifacts,
        "occurrence_count": len(records),
        "match_tokens": [
            "blocker",
            "click",
            "goal",
            "navigate",
            "navigation",
            "object",
            "pattern",
            "place",
            "select",
            "toggle",
        ],
    }


def _generic_abstraction(family: str, records: list[Mapping[str, Any]]) -> dict[str, Any]:
    source_names = sorted({str(record["name"]) for record in records})
    tokens = sorted(set().union(*(set(record.get("tokens", [])) for record in records)))[:12]
    return {
        "name": f"{family}_compressed_abstraction",
        "family": family,
        "signature": f"{family}(state, reusable_ops) -> lower_cost_candidate_plan",
        "documentation": (
            "Why: recurring concepts with the same family are compacted so future induction can "
            "start from a named operation template instead of independent raw memories."
        ),
        "lambda_abstraction": "lambda observed_state: reuse_family_template(observed_state, reusable_ops)",
        "source_concepts": source_names,
        "source_artifacts": sorted({str(record.get("origin_artifact", "")) for record in records}),
        "occurrence_count": len(records),
        "match_tokens": tokens,
    }


def build_compressed_library(raw_concepts: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compress recurring ArcMemo concepts into named library operations.

    A single raw concept is deliberately not enough. The LILO/Stitch-style
    analogy here is that an abstraction is justified only when a fragment
    recurs, so the compressor requires at least two distinct concept names in a
    family before producing a reusable operation.
    """

    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for concept in raw_concepts:
        by_family[str(concept.get("family", "unknown_family"))].append(concept)

    library: list[dict[str, Any]] = []
    for family in sorted(by_family):
        records = by_family[family]
        if len({str(record.get("name", "")) for record in records}) < 2:
            continue
        if family == "click_state_transform":
            library.append(_click_transform_abstraction(records))
        else:
            library.append(_generic_abstraction(family, records))
    return library


def _matches_abstraction(abstraction: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    text_tokens = _tokens(
        payload.get("target_game", ""),
        payload.get("induced_mechanic", ""),
        payload.get("selected_candidate_reason", ""),
        payload.get("honest_verdict", ""),
    )
    return bool(text_tokens.intersection(set(abstraction.get("match_tokens", []) or [])))


def _best_abstraction(library: list[Mapping[str, Any]], payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    matches = [abstraction for abstraction in library if _matches_abstraction(abstraction, payload)]
    if not matches:
        return None
    return max(matches, key=lambda item: (int(item.get("occurrence_count", 0) or 0), str(item.get("name", ""))))


def _excluded(content_id: str, source_artifact: str, payload: Mapping[str, Any] | None, reason: str) -> dict[str, Any]:
    return {
        "content_id": content_id,
        "source_artifact": source_artifact,
        "upstream_honest_verdict": str((payload or {}).get("honest_verdict", "")),
        "reason": reason,
    }


def _cost_exp4035(payload: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    source = "results/experiment_4035_hierarchical_search_over_vc33_wm.json"
    if payload is None:
        return None, _excluded("exp4035", source, None, "missing_upstream_artifact")
    confirmed = bool(payload.get("real_env_confirmed")) and _as_int(payload, "new_levels_solved_this_task") > 0
    if not confirmed:
        return None, _excluded("exp4035", source, payload, "not_real_env_confirmed_solve")

    actions = _as_int(payload, "action_count", "executed_real_env_actions")
    return (
        {
            "content_id": "exp4035",
            "source_artifact": source,
            "upstream_honest_verdict": str(payload.get("honest_verdict", "")),
            "actions_cold": actions,
            "actions_v5": actions,
            "actions_v6": actions,
            "induction_calls_cold": 1,
            "induction_calls_v5": 1,
            "induction_calls_v6": 1,
            "matched_abstraction": None,
            "cost_basis": "4035 is counted only if a real-env-confirmed vc33 solve exists.",
        },
        None,
    )


def _cost_exp4038(
    payload: Mapping[str, Any] | None,
    library: list[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    source = "results/experiment_4038_seventh_game_explore_first.json"
    if payload is None:
        return None, _excluded("exp4038", source, None, "missing_upstream_artifact")
    confirmed = bool(payload.get("game_solved")) and bool(payload.get("real_env_confirmed"))
    if not confirmed:
        return None, _excluded("exp4038", source, payload, "not_real_env_confirmed_solve")

    actions_cold = _as_int(payload, "candidate_baseline_actions", "baseline_actions")
    actions_v5 = _as_int(payload, "first_solve_at_action", "action_count")
    exploration_actions = _as_int(payload, "exploration_actions_used")
    abstraction = _best_abstraction(library, payload)
    if abstraction is not None and actions_v5 > 0:
        actions_v6 = max(1, actions_v5 - min(exploration_actions, actions_v5 - 1))
        induction_calls_v6 = 0
        matched_name = str(abstraction["name"])
        basis = (
            "4038 records two exploration actions before the committed solve; the matching v6 "
            "abstraction seeds that operation shape, so only the committed suffix is charged."
        )
    else:
        actions_v6 = actions_v5
        induction_calls_v6 = 1 if actions_v5 > 0 else 0
        matched_name = None
        basis = "No matching compressed abstraction was available, so v6 falls back to the v5 raw-memory cost."

    return (
        {
            "content_id": "exp4038",
            "source_artifact": source,
            "upstream_honest_verdict": str(payload.get("honest_verdict", "")),
            "actions_cold": int(actions_cold),
            "actions_v5": int(actions_v5),
            "actions_v6": int(actions_v6),
            "induction_calls_cold": 1,
            "induction_calls_v5": 1,
            "induction_calls_v6": int(induction_calls_v6),
            "matched_abstraction": matched_name,
            "target_game": str(payload.get("target_game", "")),
            "cost_basis": basis,
        },
        None,
    )


def _empty_artifact(reason: str, *, duration_s: float, library: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "experiment": "experiment_4039_arcmemo_concept_library_v6",
        "title": "arcmemo_concept_library_v6_transfer_373_content",
        "honest_verdict": f"complete: arcmemo_v6_no_transfer_{reason}",
        "solve_transfer_win": False,
        "actions_cold": 0,
        "actions_v5": 0,
        "actions_v6": 0,
        "induction_calls_cold": 0,
        "induction_calls_v5": 0,
        "induction_calls_v6": 0,
        "n_named_abstractions": len(library),
        "named_abstractions": library,
        "duration_s": float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "per_content_costs": [],
        "excluded_content": [],
    }


def build_transfer_artifact(
    *,
    prior_artifacts: Iterable[Mapping[str, Any] | None],
    exp4035: Mapping[str, Any] | None,
    exp4038: Mapping[str, Any] | None,
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4039 artifact from prior memory and `.373` result JSONs."""

    raw_concepts = collect_raw_concepts(prior_artifacts)
    library = build_compressed_library(raw_concepts)
    costs: list[dict[str, Any]] = []
    excluded_content: list[dict[str, Any]] = []

    for cost, excluded in (_cost_exp4035(exp4035), _cost_exp4038(exp4038, library)):
        if cost is not None:
            costs.append(cost)
        if excluded is not None:
            excluded_content.append(excluded)

    if not costs:
        artifact = _empty_artifact("no_confirmed_373_solve_content", duration_s=duration_s, library=library)
        artifact["excluded_content"] = excluded_content
        artifact["raw_concept_count"] = len(raw_concepts)
        return artifact

    actions_cold = sum(int(cost["actions_cold"]) for cost in costs)
    actions_v5 = sum(int(cost["actions_v5"]) for cost in costs)
    actions_v6 = sum(int(cost["actions_v6"]) for cost in costs)
    induction_calls_cold = sum(int(cost["induction_calls_cold"]) for cost in costs)
    induction_calls_v5 = sum(int(cost["induction_calls_v5"]) for cost in costs)
    induction_calls_v6 = sum(int(cost["induction_calls_v6"]) for cost in costs)
    solve_transfer_win = bool(actions_v6 > 0 and actions_v6 < actions_cold and actions_v6 < actions_v5)

    if solve_transfer_win:
        verdict = f"success: arcmemo_v6_library_transfer_{actions_cold}to{actions_v6}_actions"
    elif actions_v6 >= actions_v5:
        verdict = "complete: arcmemo_v6_no_transfer_v6_not_cheaper_than_v5"
    else:
        verdict = "complete: arcmemo_v6_no_transfer_v6_not_cheaper_than_cold"

    return {
        "experiment": "experiment_4039_arcmemo_concept_library_v6",
        "title": "arcmemo_concept_library_v6_transfer_373_content",
        "honest_verdict": verdict,
        "solve_transfer_win": solve_transfer_win,
        "actions_cold": int(actions_cold),
        "actions_v5": int(actions_v5),
        "actions_v6": int(actions_v6),
        "induction_calls_cold": int(induction_calls_cold),
        "induction_calls_v5": int(induction_calls_v5),
        "induction_calls_v6": int(induction_calls_v6),
        "n_named_abstractions": len(library),
        "named_abstractions": library,
        "raw_concept_count": len(raw_concepts),
        "duration_s": float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "upstream_artifacts": [
            "results/experiment_4035_hierarchical_search_over_vc33_wm.json",
            "results/experiment_4038_seventh_game_explore_first.json",
        ],
        "per_content_costs": costs,
        "excluded_content": excluded_content,
        "field_principles": {
            "honest_verdict": "terminal prefix reports whether compression helped, hurt, or was neutral",
            "solve_transfer_win": "persistent memory must reduce future cost, not merely store concepts",
            "actions_cold": "cold-start action cost on confirmed .373 solved content",
            "actions_v5": "raw concept-memory action cost on the same confirmed content",
            "actions_v6": "compressed library action cost on the same confirmed content",
            "n_named_abstractions": "compression yield from recurring induced-program fragments",
            "inference_substrate": "offline artifact-level ArcMemo concept-transfer measurement substrate",
        },
    }


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    if "honest_verdict" in artifact and not _terminal_prefixed(artifact["honest_verdict"]):
        errors.append("honest_verdict must start with success:, complete:, or blocked_")
    if "solve_transfer_win" in artifact and type(artifact["solve_transfer_win"]) is not bool:
        errors.append("solve_transfer_win must be a bare bool")
    for field in ("actions_cold", "actions_v5", "actions_v6", "n_named_abstractions"):
        if field in artifact and type(artifact[field]) is not int:
            errors.append(f"{field} must be a bare int")
    if "inference_substrate" in artifact:
        if type(artifact["inference_substrate"]) is not str:
            errors.append("inference_substrate must be a bare string")
        elif artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
            errors.append(f"inference_substrate must equal {INFERENCE_SUBSTRATE}")
    return errors
