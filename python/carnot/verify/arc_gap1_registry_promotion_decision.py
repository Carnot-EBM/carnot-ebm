"""Exp 5222: GAP-1 registry promotion decision.

Spec refs: REQ-VERIFY-5222, SCENARIO-VERIFY-5222.

This module is intentionally evidence-only. It reads the existing Exp 5209
hardening artifact, parses the upstream gate from
``gap1_hardened_positive.value``, and decides whether the GAP-1 set verifier is
ready to become a frozen registry entry. A positive held-out gate is necessary
but not sufficient: the selected subset must also be stable enough to freeze
without held-out tuning.
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
EXPERIMENT = "experiment_5222_gap1_gate_field_registry_promotion_v478"
EXPERIMENT_ID = 5222
SCHEMA = "carnot.arc_gap1_registry_promotion_decision_5222.v1"
RUN_DATE = "2026-07-04"
RESULT_RELATIVE_PATH = "results/experiment_5222_gap1_gate_field_registry_promotion_v478.json"
EXP5209_RELATIVE_PATH = "results/experiment_5209_gap1_set_search_holdout_hardening_v477.json"
VERIFIER_GAPS_RELATIVE_PATH = "ops/verifier_gaps.md"
VERIFIER_REGISTRY_RELATIVE_PATH = "ops/verifier_registry.yaml"
REFUTED_SINGLE_INVARIANT = "directional_adjacency_refuted_20260609"
INFERENCE_SUBSTRATE = "deterministic_verifier_registry"
SPEC_REFS = ("REQ-VERIFY-5222", "SCENARIO-VERIFY-5222")
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")
VALID_DECISIONS = {
    "promoted",
    "blocked_instability",
    "blocked_leakage",
    "blocked_missing_evidence",
}
DECISION_PATH_FEATURES = [
    "gap1_hardened_positive.value",
    "leakage_audit_passed.value",
    "leakage_audit.no_test_output_derived_features",
    "best_subset_stable.value",
    "subset_stability.top_subset",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "gap1_registry_promoted": (
        "BARE top-level boolean. True only if implementation, tests, and docs support registry promotion."
    ),
    "gap1_registry_decision": (
        "Registry decision enum: promoted, blocked_instability, blocked_leakage, or blocked_missing_evidence."
    ),
    "promoted_registry_path": "Path to the registry modified for promotion, or null when GAP-1 is blocked.",
    "frozen_subset": "Frozen invariant subset promoted into the registry, or null when GAP-1 is blocked.",
    "exp5209_gate_parsed_from_value": (
        "True only when the upstream scientific gate was parsed from gap1_hardened_positive.value."
    ),
    "refuted_single_invariant_excluded": (
        "True when the refuted directional-adjacency singleton is not promoted as the frozen GAP-1 verifier."
    ),
    "tests_run": "List of verification commands with pass/fail booleans for this decision.",
    "ops_docs_updated": "True when the GAP-1 verifier-gaps ledger records this promotion decision.",
    "inference_substrate": "Must be deterministic_verifier_registry.",
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and state whether GAP-1 was promoted or "
        "explicitly blocked."
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
        subset = [str(item) for item in value]
        return subset
    return None


def _exp5209_gate_parsed_from_value(exp5209: Mapping[str, Any]) -> bool:
    raw = exp5209.get("gap1_hardened_positive")
    return isinstance(raw, Mapping) and raw.get("value") is True


def _leakage_audit_passed(exp5209: Mapping[str, Any]) -> bool:
    audit = exp5209.get("leakage_audit")
    if isinstance(audit, Mapping):
        no_test_output = audit.get("no_test_output_derived_features") is not False
        no_test_gold = audit.get("no_test_gold_in_scoring") is not False
        no_heldout_selection = audit.get("no_subset_selection_on_heldout_rows") is not False
    else:
        no_test_output = False
        no_test_gold = False
        no_heldout_selection = False
    return (
        _value(exp5209, "leakage_audit_passed", False) is True
        and no_test_output
        and no_test_gold
        and no_heldout_selection
    )


def _subset_freeze_audit(exp5209: Mapping[str, Any]) -> JsonDict:
    stability = exp5209.get("subset_stability")
    stability_map = stability if isinstance(stability, Mapping) else {}
    top_subset = _subset_list(stability_map.get("top_subset"))
    best_subset_stable = _value(exp5209, "best_subset_stable", False) is True
    return {
        "can_freeze_without_heldout_tuning": best_subset_stable and top_subset is not None,
        "best_subset_stable": best_subset_stable,
        "top_subset": top_subset,
        "top_subset_count": stability_map.get("top_subset_count"),
        "top_subset_fraction": stability_map.get("top_subset_fraction"),
        "stability_rule": stability_map.get("stability_rule"),
        "heldout_rows_used_for_freeze": False,
        "selection_count_source": "exp5209.subset_stability",
    }


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


def _decision(exp5209: Mapping[str, Any]) -> str:
    if _value(exp5209, "gap1_hardened_positive", False) is not True:
        return "blocked_missing_evidence"
    if not _leakage_audit_passed(exp5209):
        return "blocked_leakage"
    if _value(exp5209, "best_subset_stable", False) is not True:
        return "blocked_instability"
    return "blocked_missing_evidence"


def _refuted_single_excluded(
    frozen_subset: Sequence[str] | None, top_subset: Sequence[str] | None
) -> bool:
    subset = list(frozen_subset or top_subset or [])
    return REFUTED_SINGLE_INVARIANT not in subset or frozen_subset is None


def _verdict(decision: str, gate_from_value: bool) -> str:
    if decision == "blocked_instability":
        return (
            "complete: GAP-1 registry promotion blocked_instability; exp5209 gate parsed from "
            f"gap1_hardened_positive.value={gate_from_value}, but the selected subset is not stable enough "
            "to freeze without held-out tuning; this is not the exp5210 gate-shape failure alone."
        )
    if decision == "blocked_leakage":
        return "complete: GAP-1 registry promotion blocked_leakage; the deterministic leakage guard failed."
    if decision == "promoted":
        return "complete: GAP-1 registry promotion promoted with a frozen deterministic verifier subset."
    return "complete: GAP-1 registry promotion blocked_missing_evidence; decision-grade evidence is absent."


def build_artifact(
    *,
    exp5209: Mapping[str, Any],
    root: Path | str | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
    ops_docs_updated: bool = False,
    duration_s: float = 0.0,
) -> JsonDict:
    freeze_audit = _subset_freeze_audit(exp5209)
    decision = _decision(exp5209)
    promoted = decision == "promoted"
    frozen_subset = freeze_audit["top_subset"] if promoted else None
    promoted_registry_path = VERIFIER_REGISTRY_RELATIVE_PATH if promoted else None
    gate_from_value = _exp5209_gate_parsed_from_value(exp5209)
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "run_date": RUN_DATE,
        "result_path": RESULT_RELATIVE_PATH,
        "source_artifacts": [EXP5209_RELATIVE_PATH],
        "gap1_registry_promoted": _wrap("gap1_registry_promoted", promoted),
        "gap1_registry_decision": _wrap("gap1_registry_decision", decision),
        "promoted_registry_path": _wrap("promoted_registry_path", promoted_registry_path),
        "frozen_subset": _wrap("frozen_subset", frozen_subset),
        "exp5209_gate_parsed_from_value": _wrap("exp5209_gate_parsed_from_value", gate_from_value),
        "refuted_single_invariant_excluded": _wrap(
            "refuted_single_invariant_excluded",
            _refuted_single_excluded(frozen_subset, freeze_audit["top_subset"]),
        ),
        "tests_run": _wrap("tests_run", [dict(row) for row in tests_run]),
        "ops_docs_updated": _wrap("ops_docs_updated", bool(ops_docs_updated)),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "honest_verdict": _wrap("honest_verdict", _verdict(decision, gate_from_value)),
        "subset_freeze_audit": freeze_audit,
        "registry_audit": _registry_audit(root),
        "decision_path_features": list(DECISION_PATH_FEATURES),
        "upstream_scientific_gate": {
            "gap1_hardened_positive": _value(exp5209, "gap1_hardened_positive", None),
            "heldout_pass_at_2_mean": _value(exp5209, "heldout_pass_at_2_mean", None),
            "baseline_always_on_pass_at_2_mean": _value(
                exp5209, "baseline_always_on_pass_at_2_mean", None
            ),
            "single_refuted_directional_pass_at_2_mean": _value(
                exp5209,
                "single_refuted_directional_pass_at_2_mean",
                None,
            ),
            "paired_delta_ci95": _value(exp5209, "paired_delta_ci95", None),
            "leakage_audit_passed": _value(exp5209, "leakage_audit_passed", None),
            "best_subset_stable": _value(exp5209, "best_subset_stable", None),
        },
        "follow_up_criterion": (
            "Reconsider registry promotion only after a predeclared frozen subset is selected from training "
            "evidence alone, one exact subset wins at least half of grouped splits, leakage guards pass, and "
            f"{REFUTED_SINGLE_INVARIANT} remains excluded from the promoted frozen verifier."
        ),
        "duration_s": round(float(duration_s), 3),
        "field_principles": FIELD_PRINCIPLES,
    }
    return artifact


def _tests_run_shape_ok(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(row, Mapping)
        and isinstance(row.get("command"), str)
        and isinstance(row.get("passed"), bool)
        for row in value
    )


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
    promoted = _value(artifact, "gap1_registry_promoted")
    decision = _value(artifact, "gap1_registry_decision")
    if not isinstance(promoted, bool):
        errors.append("gap1_registry_promoted must be bool")
    if decision not in VALID_DECISIONS:
        errors.append("gap1_registry_decision must be a valid decision enum")
    if promoted is False and (
        _value(artifact, "promoted_registry_path") is not None
        or _value(artifact, "frozen_subset") is not None
    ):
        errors.append("blocked decisions require null promoted_registry_path and frozen_subset")
    if _value(artifact, "inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be deterministic_verifier_registry")
    if not isinstance(_value(artifact, "exp5209_gate_parsed_from_value"), bool):
        errors.append("exp5209_gate_parsed_from_value must be bool")
    if not isinstance(_value(artifact, "refuted_single_invariant_excluded"), bool):
        errors.append("refuted_single_invariant_excluded must be bool")
    if not isinstance(_value(artifact, "ops_docs_updated"), bool):
        errors.append("ops_docs_updated must be bool")
    if not _tests_run_shape_ok(_value(artifact, "tests_run")):
        errors.append("tests_run must list commands with boolean passed fields")
    verdict = _value(artifact, "honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must use a terminal complete/success prefix")
    return errors


def _gap_update_block(artifact: Mapping[str, Any]) -> str:
    return (
        "<!-- experiment_5222_gap1_gate_field_registry_promotion_v478 start -->\n"
        "- experiment_5222 GAP-1 registry promotion decision (2026-07-04): "
        f"decision={_value(artifact, 'gap1_registry_decision')}, "
        f"gap1_registry_promoted={_value(artifact, 'gap1_registry_promoted')}, "
        f"exp5209_gate_parsed_from_value={_value(artifact, 'exp5209_gate_parsed_from_value')}, "
        f"frozen_subset={_value(artifact, 'frozen_subset')}. "
        f"Follow-up criterion: {artifact['follow_up_criterion']} Verdict: {_value(artifact, 'honest_verdict')}\n"
        "<!-- experiment_5222_gap1_gate_field_registry_promotion_v478 end -->\n"
    )


def update_verifier_gap_doc(root: Path | str, artifact: Mapping[str, Any]) -> bool:
    path = Path(root) / VERIFIER_GAPS_RELATIVE_PATH
    if not path.exists():  # pragma: no cover - defensive for temp roots that omit ops docs.
        return False
    text = path.read_text(encoding="utf-8")
    start = "<!-- experiment_5222_gap1_gate_field_registry_promotion_v478 start -->"
    end = "<!-- experiment_5222_gap1_gate_field_registry_promotion_v478 end -->"
    block = _gap_update_block(artifact)
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _old, after = rest.split(end, 1)
        path.write_text(before + block + after, encoding="utf-8")
        return True
    prior_end = "<!-- experiment_5209_gap1_set_search_holdout_hardening_v477 end -->"
    text = text.replace(prior_end, prior_end + "\n" + block.rstrip("\n"), 1)
    path.write_text(text, encoding="utf-8")
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
    elapsed = time.time() - started if duration_s is None else duration_s
    ops_docs_will_update = bool(
        update_gap_doc and (root_path / VERIFIER_GAPS_RELATIVE_PATH).exists()
    )
    artifact = build_artifact(
        exp5209=exp5209,
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
