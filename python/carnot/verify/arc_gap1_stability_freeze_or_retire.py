"""Exp 5237: GAP-1 stability freeze-or-retire decision.

Spec refs: REQ-VERIFY-5237, SCENARIO-VERIFY-5237.

This module makes the final stability decision for the current GAP-1 registry
promotion path. It does not search for new invariants. It reads the existing
Exp 5209 hardening evidence and Exp 5222 registry decision, applies a
predeclared stability rule, and writes a terminal artifact that either freezes
the deterministic subset or blocks/retires the current path with a concrete
criterion for future reopening.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = "experiment_5237_gap1_stability_freeze_or_retire_v479"
EXPERIMENT_ID = 5237
SCHEMA = "carnot.arc_gap1_stability_freeze_or_retire_5237.v1"
RUN_DATE = "2026-07-04"
RESULT_RELATIVE_PATH = "results/experiment_5237_gap1_stability_freeze_or_retire_v479.json"
EXP5209_RELATIVE_PATH = "results/experiment_5209_gap1_set_search_holdout_hardening_v477.json"
EXP5222_RELATIVE_PATH = "results/experiment_5222_gap1_gate_field_registry_promotion_v478.json"
VERIFIER_GAPS_RELATIVE_PATH = "ops/verifier_gaps.md"
VERIFIER_REGISTRY_RELATIVE_PATH = "ops/verifier_registry.yaml"
REFUTED_SINGLE_INVARIANT = "directional_adjacency_refuted_20260609"
INFERENCE_SUBSTRATE = "deterministic_gap1_stability_analysis"
SPEC_REFS = ("REQ-VERIFY-5237", "SCENARIO-VERIFY-5237")
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")
VALID_DECISIONS = {
    "frozen_promoted",
    "blocked_instability",
    "retired_current_path",
    "blocked_missing_evidence",
}
DECISION_PATH_FEATURES = [
    "exp5209.gap1_hardened_positive.value",
    "exp5209.leakage_audit_passed.value",
    "exp5209.leakage_audit.no_subset_selection_on_heldout_rows",
    "exp5209.subset_stability.selection_counts",
    "exp5222.gap1_registry_promoted.value",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "gap1_stability_decision": (
        "Final GAP-1 stability decision enum: frozen_promoted, blocked_instability, "
        "retired_current_path, or blocked_missing_evidence."
    ),
    "gap1_registry_promoted": (
        "BARE top-level boolean. True only when a deterministic non-leaky stable subset is frozen."
    ),
    "frozen_subset": "List of frozen invariant names, or null when GAP-1 is not promoted.",
    "stability_rule_predeclared": (
        "True only when the stability rule is declared before any new performance result."
    ),
    "no_new_broad_search": (
        "True only when Exp 5237 evaluates existing Exp 5209 subset evidence without a new search."
    ),
    "refuted_single_invariant_excluded": (
        "True when directional_adjacency_refuted_20260609 is absent from the frozen/current subset."
    ),
    "tests_run": "List of verification commands with pass/fail booleans for this decision.",
    "ops_verifier_gaps_updated": "True when ops/verifier_gaps.md records the block/freeze decision.",
    "inference_substrate": "Must be deterministic_gap1_stability_analysis.",
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and state whether GAP-1 was "
        "frozen, blocked, or retired."
    ),
}
REQUIRED_PRINCIPLED_FIELDS = tuple(FIELD_PRINCIPLES)


def _value(artifact: Mapping[str, Any], field: str, default: Any = None) -> Any:
    raw = artifact.get(field, default)
    if isinstance(raw, Mapping) and "value" in raw:
        return raw["value"]
    return raw


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _subset_list(value: Any) -> list[str] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return None


def _selection_counts(exp5209: Mapping[str, Any]) -> list[JsonDict]:
    stability = exp5209.get("subset_stability")
    if not isinstance(stability, Mapping):
        return []
    rows = stability.get("selection_counts")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    parsed: list[JsonDict] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        subset = _subset_list(row.get("subset"))
        count = row.get("count")
        if subset is None or not isinstance(count, int) or count < 1:
            continue
        parsed.append({"subset": subset, "count": count})
    return parsed


def _top_subset(exp5209: Mapping[str, Any]) -> list[str] | None:
    stability = exp5209.get("subset_stability")
    if not isinstance(stability, Mapping):
        return None
    return _subset_list(stability.get("top_subset"))


def _top_subset_count(exp5209: Mapping[str, Any], top_subset: Sequence[str] | None) -> int:
    stability = exp5209.get("subset_stability")
    if isinstance(stability, Mapping) and isinstance(stability.get("top_subset_count"), int):
        return int(stability["top_subset_count"])
    if top_subset is None:
        return 0
    return sum(
        int(row["count"]) for row in _selection_counts(exp5209) if row["subset"] == list(top_subset)
    )


def _n_grouped_splits(exp5209: Mapping[str, Any], counts: Sequence[Mapping[str, Any]]) -> int:
    raw = _value(exp5209, "n_grouped_splits", None)
    if isinstance(raw, int) and raw > 0:
        return raw
    return sum(int(row["count"]) for row in counts)


def _ceil_half(value: int) -> int:
    if value <= 0:
        return 0
    return (value + 1) // 2


def _frequency(count: int, total: int) -> float:
    return round(count / total, 6) if total > 0 else 0.0


def _invariant_inclusion_frequencies(
    *,
    top_subset: Sequence[str] | None,
    selection_counts: Sequence[Mapping[str, Any]],
    n_grouped_splits: int,
) -> dict[str, float]:
    if top_subset is None:
        return {}
    frequencies: dict[str, float] = {}
    for invariant in top_subset:
        count = sum(
            int(row["count"]) for row in selection_counts if invariant in row.get("subset", [])
        )
        frequencies[invariant] = _frequency(count, n_grouped_splits)
    return frequencies


def _exp5209_gate_parsed_from_value(exp5209: Mapping[str, Any]) -> bool:
    raw = exp5209.get("gap1_hardened_positive")
    return isinstance(raw, Mapping) and raw.get("value") is True


def _exp5222_evidence_present(exp5222: Mapping[str, Any]) -> bool:
    promoted = exp5222.get("gap1_registry_promoted")
    decision = exp5222.get("gap1_registry_decision")
    return (
        isinstance(promoted, Mapping)
        and isinstance(promoted.get("value"), bool)
        and isinstance(decision, Mapping)
        and isinstance(decision.get("value"), str)
    )


def _leakage_audit_passed(exp5209: Mapping[str, Any]) -> bool:
    audit = exp5209.get("leakage_audit")
    if not isinstance(audit, Mapping):
        return False
    return (
        _value(exp5209, "leakage_audit_passed", False) is True
        and audit.get("passed") is not False
        and audit.get("no_duplicate_task_ids_across_train_eval") is True
        and audit.get("no_subset_selection_on_heldout_rows") is True
        and audit.get("no_test_gold_in_scoring") is True
        and audit.get("no_test_output_derived_features") is True
    )


def _predeclared_stability_audit(
    exp5209: Mapping[str, Any], exp5222: Mapping[str, Any]
) -> JsonDict:
    top_subset = _top_subset(exp5209)
    counts = _selection_counts(exp5209)
    n_grouped_splits = _n_grouped_splits(exp5209, counts)
    top_count = _top_subset_count(exp5209, top_subset)
    min_count = _ceil_half(n_grouped_splits)
    inclusion_frequencies = _invariant_inclusion_frequencies(
        top_subset=top_subset,
        selection_counts=counts,
        n_grouped_splits=n_grouped_splits,
    )
    inclusion_passed = bool(inclusion_frequencies) and all(
        frequency >= 0.5 for frequency in inclusion_frequencies.values()
    )
    exact_subset_passed = (
        _value(exp5209, "best_subset_stable", False) is True
        and top_subset is not None
        and top_count >= min_count
        and n_grouped_splits > 0
    )
    audit = exp5209.get("leakage_audit")
    no_heldout_tuning = (
        isinstance(audit, Mapping) and audit.get("no_subset_selection_on_heldout_rows") is True
    )
    return {
        "rule": (
            "Freeze only if the Exp 5209 positive gate is parsed from its value field, leakage "
            "guards pass, no held-out rows select the subset, one exact subset wins at least half "
            "of grouped splits, each frozen invariant appears in at least half of grouped splits, "
            f"and {REFUTED_SINGLE_INVARIANT} is excluded."
        ),
        "source_subset_evidence": "exp5209.subset_stability.selection_counts",
        "heldout_rows_used_for_freeze": False,
        "no_heldout_tuning": no_heldout_tuning,
        "top_subset": top_subset,
        "top_subset_count": top_count,
        "n_grouped_splits": n_grouped_splits,
        "exact_subset_min_count": min_count,
        "exact_subset_selection_frequency": _frequency(top_count, n_grouped_splits),
        "exact_subset_stability_passed": exact_subset_passed,
        "invariant_inclusion_min_frequency": 0.5,
        "invariant_inclusion_frequencies": inclusion_frequencies,
        "invariant_inclusion_stability_passed": inclusion_passed,
        "exp5209_gate_parsed_from_value": _exp5209_gate_parsed_from_value(exp5209),
        "leakage_audit_passed": _leakage_audit_passed(exp5209),
        "exp5222_registry_promoted": _value(exp5222, "gap1_registry_promoted", None),
        "exp5222_registry_decision": _value(exp5222, "gap1_registry_decision", None),
        "refuted_single_invariant_excluded": (
            top_subset is None or REFUTED_SINGLE_INVARIANT not in top_subset
        ),
    }


def _required_evidence_present(
    exp5209: Mapping[str, Any],
    exp5222: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> bool:
    return (
        isinstance(exp5209.get("gap1_hardened_positive"), Mapping)
        and isinstance(exp5209.get("leakage_audit_passed"), Mapping)
        and isinstance(exp5209.get("best_subset_stable"), Mapping)
        and audit.get("top_subset") is not None
        and bool(_selection_counts(exp5209))
        and _exp5222_evidence_present(exp5222)
    )


def _decision(
    exp5209: Mapping[str, Any], exp5222: Mapping[str, Any], audit: Mapping[str, Any]
) -> str:
    if not _required_evidence_present(exp5209, exp5222, audit):
        return "blocked_missing_evidence"
    if audit.get("refuted_single_invariant_excluded") is False:
        return "retired_current_path"
    if (
        audit.get("exp5209_gate_parsed_from_value") is not True
        or audit.get("leakage_audit_passed") is not True
        or audit.get("no_heldout_tuning") is not True
    ):
        return "blocked_missing_evidence"
    if (
        audit.get("exact_subset_stability_passed") is not True
        or audit.get("invariant_inclusion_stability_passed") is not True
    ):
        return "blocked_instability"
    return "frozen_promoted"


def _future_reopen_criterion(audit: Mapping[str, Any]) -> str:
    n_grouped_splits = audit.get("n_grouped_splits")
    min_count = audit.get("exact_subset_min_count")
    if isinstance(n_grouped_splits, int) and n_grouped_splits > 0 and isinstance(min_count, int):
        exact_clause = f"at least {min_count} of {n_grouped_splits} grouped splits"
    else:
        exact_clause = "at least half of predeclared grouped splits"
    return (
        "Minimum evidence to reopen: choose one frozen subset from training evidence alone, before "
        f"held-out scoring, and have the exact subset win {exact_clause}; every frozen invariant must "
        "appear in at least half of those splits, all leakage/no-held-out-tuning guards must pass, "
        f"and {REFUTED_SINGLE_INVARIANT} must remain excluded."
    )


def _verdict(decision: str) -> str:
    if decision == "frozen_promoted":
        return "complete: GAP-1 was frozen and promoted with a deterministic stable subset."
    if decision == "retired_current_path":
        return (
            "complete: GAP-1 current promotion path retired_current_path because the refuted "
            "single directional-adjacency invariant remains on the freeze path."
        )
    if decision == "blocked_instability":
        return (
            "complete: GAP-1 blocked_instability; the existing Exp 5209 positive result is "
            "non-leaky, but the exact selected subset is not stable enough to freeze."
        )
    return "complete: GAP-1 blocked_missing_evidence; decision-grade stability evidence is absent."


def _tests_run_shape_ok(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(row, Mapping)
        and isinstance(row.get("command"), str)
        and isinstance(row.get("passed"), bool)
        for row in value
    )


def _registry_audit(root: Path | str | None) -> JsonDict:
    path = (
        Path(root) / VERIFIER_REGISTRY_RELATIVE_PATH
        if root is not None
        else Path(VERIFIER_REGISTRY_RELATIVE_PATH)
    )
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    verifier_ids = re.findall(r"verifier_id:\s*([^\s#]+)", text)
    promoted_gap1_ids = [
        verifier_id for verifier_id in verifier_ids if "gap1" in verifier_id.lower()
    ]
    return {
        "path": VERIFIER_REGISTRY_RELATIVE_PATH,
        "exists": path.exists(),
        "verifier_ids_loaded": verifier_ids,
        "promoted_gap1_registry_entry_present": bool(promoted_gap1_ids),
        "promoted_gap1_verifier_ids": promoted_gap1_ids,
    }


def build_artifact(
    *,
    exp5209: Mapping[str, Any],
    exp5222: Mapping[str, Any],
    root: Path | str | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
    ops_docs_updated: bool = False,
    duration_s: float = 0.0,
) -> JsonDict:
    stability_audit = _predeclared_stability_audit(exp5209, exp5222)
    decision = _decision(exp5209, exp5222, stability_audit)
    promoted = decision == "frozen_promoted"
    frozen_subset = stability_audit["top_subset"] if promoted else None
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "run_date": RUN_DATE,
        "result_path": RESULT_RELATIVE_PATH,
        "source_artifacts": [EXP5209_RELATIVE_PATH, EXP5222_RELATIVE_PATH],
        "gap1_stability_decision": _wrap("gap1_stability_decision", decision),
        "gap1_registry_promoted": _wrap("gap1_registry_promoted", promoted),
        "frozen_subset": _wrap("frozen_subset", frozen_subset),
        "stability_rule_predeclared": _wrap("stability_rule_predeclared", True),
        "no_new_broad_search": _wrap("no_new_broad_search", True),
        "refuted_single_invariant_excluded": _wrap(
            "refuted_single_invariant_excluded",
            bool(stability_audit["refuted_single_invariant_excluded"]),
        ),
        "tests_run": _wrap("tests_run", [dict(row) for row in tests_run]),
        "ops_verifier_gaps_updated": _wrap("ops_verifier_gaps_updated", bool(ops_docs_updated)),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "honest_verdict": _wrap("honest_verdict", _verdict(decision)),
        "stability_audit": stability_audit,
        "registry_audit": _registry_audit(root),
        "decision_path_features": list(DECISION_PATH_FEATURES),
        "future_reopen_criterion": _future_reopen_criterion(stability_audit),
        "retirement_or_block_condition": (
            "Freeze is blocked unless the exact-subset and invariant-inclusion stability checks both "
            "pass under the no-held-out-tuning leakage guard."
        ),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(float(duration_s), 3),
    }
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_PRINCIPLED_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    for field in REQUIRED_PRINCIPLED_FIELDS:
        raw = artifact.get(field)
        if not isinstance(raw, Mapping) or "value" not in raw or "principle" not in raw:
            errors.append(f"{field} must be principle-wrapped")
            continue
        if raw.get("principle") != FIELD_PRINCIPLES[field]:
            errors.append(f"{field} principle mismatch")
    decision = _value(artifact, "gap1_stability_decision")
    promoted = _value(artifact, "gap1_registry_promoted")
    frozen_subset = _value(artifact, "frozen_subset")
    if decision not in VALID_DECISIONS:
        errors.append("gap1_stability_decision must be a valid decision enum")
    if not isinstance(promoted, bool):
        errors.append("gap1_registry_promoted must be bool")
    if decision == "frozen_promoted":
        if promoted is not True or not isinstance(frozen_subset, list) or not frozen_subset:
            errors.append("frozen_promoted requires true gap1_registry_promoted and frozen_subset")
    elif promoted is not False or frozen_subset is not None:
        errors.append(
            "non-promoted decisions require false gap1_registry_promoted and null frozen_subset"
        )
    if _value(artifact, "stability_rule_predeclared") is not True:
        errors.append("stability_rule_predeclared must be true")
    if _value(artifact, "no_new_broad_search") is not True:
        errors.append("no_new_broad_search must be true")
    if not isinstance(_value(artifact, "refuted_single_invariant_excluded"), bool):
        errors.append("refuted_single_invariant_excluded must be bool")
    if not isinstance(_value(artifact, "ops_verifier_gaps_updated"), bool):
        errors.append("ops_verifier_gaps_updated must be bool")
    if not _tests_run_shape_ok(_value(artifact, "tests_run")):
        errors.append("tests_run must list commands with boolean passed fields")
    if _value(artifact, "inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be deterministic_gap1_stability_analysis")
    verdict = _value(artifact, "honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must use a terminal complete/success prefix")
    elif not any(word in verdict for word in ("frozen", "blocked", "retired")):
        errors.append("honest_verdict must state whether GAP-1 was frozen, blocked, or retired")
    return errors


def _gap_update_block(artifact: Mapping[str, Any]) -> str:
    return (
        "<!-- experiment_5237_gap1_stability_freeze_or_retire_v479 start -->\n"
        "- experiment_5237 GAP-1 stability freeze-or-retire decision (2026-07-04): "
        f"decision={_value(artifact, 'gap1_stability_decision')}, "
        f"gap1_registry_promoted={_value(artifact, 'gap1_registry_promoted')}, "
        f"frozen_subset={_value(artifact, 'frozen_subset')}, "
        f"stability_rule_predeclared={_value(artifact, 'stability_rule_predeclared')}, "
        f"no_new_broad_search={_value(artifact, 'no_new_broad_search')}. "
        f"Block condition: {artifact['retirement_or_block_condition']} "
        f"Minimum evidence to reopen: {artifact['future_reopen_criterion']} "
        f"Verdict: {_value(artifact, 'honest_verdict')}\n"
        "<!-- experiment_5237_gap1_stability_freeze_or_retire_v479 end -->\n"
    )


def update_verifier_gap_doc(root: Path | str, artifact: Mapping[str, Any]) -> bool:
    path = Path(root) / VERIFIER_GAPS_RELATIVE_PATH
    if not path.exists():  # pragma: no cover - temp or partial repos can omit ops docs.
        return False
    text = path.read_text(encoding="utf-8")
    start = "<!-- experiment_5237_gap1_stability_freeze_or_retire_v479 start -->"
    end = "<!-- experiment_5237_gap1_stability_freeze_or_retire_v479 end -->"
    block = _gap_update_block(artifact)
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _old, after = rest.split(end, 1)
        path.write_text(before + block + after, encoding="utf-8")
        return True
    prior_end = "<!-- experiment_5222_gap1_gate_field_registry_promotion_v478 end -->"
    if prior_end in text:
        text = text.replace(prior_end, prior_end + "\n" + block.rstrip("\n"), 1)
        path.write_text(text, encoding="utf-8")
        return True
    path.write_text(text.rstrip() + "\n" + block, encoding="utf-8")
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str | None = None,
    update_gap_doc: bool = True,
    tests_run: Sequence[Mapping[str, Any]] = (),
    duration_s: float | None = None,
) -> JsonDict:
    started = time.time()
    root_path = Path(root)
    exp5209 = _read_json(root_path / EXP5209_RELATIVE_PATH)
    exp5222 = _read_json(root_path / EXP5222_RELATIVE_PATH)
    elapsed = time.time() - started if duration_s is None else duration_s
    ops_docs_will_update = bool(
        update_gap_doc and (root_path / VERIFIER_GAPS_RELATIVE_PATH).exists()
    )
    artifact = build_artifact(
        exp5209=exp5209,
        exp5222=exp5222,
        root=root_path,
        tests_run=tests_run,
        ops_docs_updated=ops_docs_will_update,
        duration_s=elapsed,
    )
    output = Path(result_path) if result_path is not None else root_path / RESULT_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if update_gap_doc:
        update_verifier_gap_doc(root_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover - direct experiment entrypoint.
    artifact = run()
    print(_value(artifact, "honest_verdict"))
    print(f"wrote {RESULT_RELATIVE_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
