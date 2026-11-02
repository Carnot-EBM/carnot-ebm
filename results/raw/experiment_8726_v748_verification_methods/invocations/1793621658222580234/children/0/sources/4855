"""Reduce V669 raw evidence without asking producers for their conclusions.

REQ-REPORT-7679 and REQ-CONTINUOUS-7679. A missing field is a custody
failure; the caller may still audit other available sources.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import sha256_file


PRODUCERS = {
    7672: "results/experiment_7672_v669_bound_relations.json",
    7673: "results/experiment_7673_v669_fresh_relation_cohort.json",
    7674: "results/experiment_7674_relation_energy.json",
    7675: "results/experiment_7675_v669_static_decision.json",
    7676: "results/experiment_7676_v669_qwen_quote_relations.json",
    7677: "results/experiment_7677_v669_online_learning.json",
    7678: "results/experiment_7678_v669_continuous_learning.json",
}
FIXTURE_ARMS = {"membership_only", "bound_relation"}
COHORT_ARMS = {"original_source", "evidence_erasure", "within_role_derangement"}
QUOTE_ARMS = {"numeric_offset", "exact_quote"}


def failed_check(
    check: str,
    upstream: str,
    path: Path,
    field: str,
    expected: Any,
    observed: Any,
    operator: str = "==",
) -> dict[str, Any]:
    """Keep the operand that makes a blocked source observable and replayable."""
    return {
        "check": check,
        "upstream": upstream,
        "path": str(path.resolve()),
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
    }


def _receipt_passed(value: Mapping[str, Any]) -> bool:
    """Use the source's actual check receipt, not its optimistic headline."""
    receipts = value.get("validation_receipts")
    if isinstance(receipts, dict):
        return receipts.get("required_checks_passed") is True
    if isinstance(receipts, list):
        return bool(receipts) and all(r.get("passed") is True for r in receipts)
    return False


def inventory(root: Path) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Authenticate each terminal source while retaining invalid pre-gate bytes."""
    found: dict[int, dict[str, Any]] = {}
    checks: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {
        "producers": {},
        "pre_gate_receipts": {},
        "raw_stores": {},
        "missing_evidence": [],
    }
    for number, label in PRODUCERS.items():
        path = root / label
        name = f"Exp{number}"
        if not path.is_file():
            checks.append(failed_check("producer_exists", name, path, "exists", True, False))
            hashes["missing_evidence"].append(label)
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            checks.append(
                failed_check("producer_json", name, path, "json_valid", True, type(error).__name__)
            )
            hashes["pre_gate_receipts"][label] = sha256_file(path)
            continue
        bad = False
        for field, expected, observed in (
            ("honest_verdict", "complete_*", value.get("honest_verdict")),
            ("verdict_class", "positive|circular_positive|null", value.get("verdict_class")),
            ("acceptance_gate_results", "present", value.get("acceptance_gate_results")),
            ("validation_receipts.required_checks_passed", True, _receipt_passed(value)),
        ):
            valid = (
                (isinstance(observed, str) and observed.startswith("complete_"))
                if field == "honest_verdict"
                else (
                    observed in {"positive", "circular_positive", "null"}
                    if field == "verdict_class"
                    else (
                        observed is not None
                        if field == "acceptance_gate_results"
                        else observed is True
                    )
                )
            )
            if not valid:
                checks.append(
                    failed_check("producer_contract", name, path, field, expected, observed)
                )
                bad = True
        bucket = "pre_gate_receipts" if bad else "producers"
        hashes[bucket][label] = sha256_file(path)
        found[number] = value
    return found, checks, hashes


def _required(row: Mapping[str, Any], fields: Sequence[str]) -> None:
    """Absent metrics cannot quietly become zero or a valid null."""
    for field in fields:
        if field not in row:
            raise ValueError(f"missing field {field}")


def _paired(
    rows: Sequence[Mapping[str, Any]], arms: set[str], fields: Sequence[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    """One source group has exactly one row for each declared arm."""
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for raw in rows:
        _required(
            raw,
            ("unit_id", "arm", "source_sha256", "answer_sha256", "excluded", "censored", *fields),
        )
        row = dict(raw)
        unit, arm = str(row["unit_id"]), str(row["arm"])
        if arm not in arms:
            raise ValueError(f"unexpected arm {arm}")
        if arm in groups[unit]:
            raise ValueError(f"duplicate family arm {unit}/{arm}")
        groups[unit][arm] = row
    for unit, group in groups.items():
        if set(group) != arms:
            raise ValueError(f"missing arm for {unit}")
        if len({row["answer_sha256"] for row in group.values()}) != 1:
            raise ValueError(f"answer changed between arms for {unit}")
    return groups


def audit_fixture(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Fixture oracle agreement is useful protocol evidence, not learned quality."""
    groups = _paired(
        rows, FIXTURE_ARMS, ("truth", "observed", "population", "provenance", "raw_metrics")
    )
    out: list[dict[str, Any]] = []
    summary = Counter(independent_groups=len(groups))
    for group in groups.values():
        bound = group["bound_relation"]
        summary["fixture_correct_groups"] += int(
            bound["population"] == "fixture" and bound["truth"] == bound["observed"]
        )
        summary["unknown_groups"] += int(bound["truth"] == "unknown")
        summary["excluded_groups"] += int(any(r["excluded"] for r in group.values()))
        summary["censored_groups"] += int(any(r["censored"] for r in group.values()))
        for row in group.values():
            out.append(
                {
                    **row,
                    "evidence_stage": "fixture_protocol",
                    "independent_unit": row["unit_id"],
                    "claim_limit": "exact fixture oracle only",
                }
            )
    return out, dict(summary)


def audit_cohort(
    by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    rosters: Mapping[str, Sequence[str]],
    model_inputs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Check family, role, and label boundaries before counting source views."""
    out: list[dict[str, Any]] = []
    summary = Counter()
    all_units: set[str] = set()
    for role, rows in by_role.items():
        groups = _paired(
            rows,
            COHORT_ARMS,
            (
                "role",
                "source_group_id",
                "denominator",
                "numerator",
                "unknown_claims",
                "checked_relations",
                "whole_answer_certified",
            ),
        )
        if set(groups) != set(rosters[role]):
            raise ValueError(f"roster membership mismatch for {role}")
        if all_units.intersection(groups):
            raise ValueError("duplicate family across roles")
        all_units.update(groups)
        if role == "online_admission" and any(
            item.get("labels_accessible") is not False for item in model_inputs[role]
        ):
            raise ValueError("admission label exposed before selection")
        summary["independent_groups"] += len(groups)
        for unit, group in groups.items():
            if (
                any(row["role"] != role for row in group.values())
                or group["original_source"]["source_group_id"] != unit
                or group["evidence_erasure"]["source_group_id"] is not None
                or group["within_role_derangement"]["source_group_id"] == unit
            ):
                raise ValueError(f"role or source family changed for {unit}")
            original = group["original_source"]
            if original["source_sha256"] != original["original_source_sha256"]:
                raise ValueError(f"original source hash changed for {unit}")
            if any(row["whole_answer_certified"] is True for row in group.values()):
                raise ValueError(f"partial atom claimed answer truth for {unit}")
            for row in group.values():
                if (
                    not isinstance(row["denominator"], int)
                    or not 0 <= row["numerator"] <= row["denominator"]
                    or row["unknown_claims"] < 0
                ):
                    raise ValueError(f"invalid relation denominator for {unit}")
                out.append(
                    {
                        **row,
                        "evidence_stage": "cohort_source_control",
                        "independent_unit": unit,
                        "claim_limit": "injected-error cohort; no natural-answer accuracy",
                    }
                )
            summary["unknown_groups"] += int(original["unknown_claims"] > 0)
            summary["excluded_groups"] += int(any(r["excluded"] for r in group.values()))
            summary["censored_groups"] += int(any(r["censored"] for r in group.values()))
    return out, dict(summary)


def audit_quotes(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Count every proposal attempt; an exact quote alone proves no relation."""
    groups = _paired(
        rows,
        QUOTE_ARMS,
        (
            "population",
            "truth",
            "fixture_truth",
            "prior_exposure",
            "raw_metrics",
            "output_tokens",
            "generation_s",
        ),
    )
    out: list[dict[str, Any]] = []
    summary = Counter(independent_groups=len(groups))
    for unit, group in groups.items():
        if len({row["source_sha256"] for row in group.values()}) != 1:
            raise ValueError(f"source changed between arms for {unit}")
        summary["fixture_groups"] += int(group["exact_quote"]["population"] == "fixture")
        summary["exposed_groups"] += int(any(r["prior_exposure"] for r in group.values()))
        summary["censored_groups"] += int(any(r["censored"] for r in group.values()))
        for row in group.values():
            metrics = row["raw_metrics"]
            _required(
                metrics, ("proposal_count", "full_proposition_supported", "unknown_remainder")
            )
            summary["proposal_count"] += metrics["proposal_count"]
            summary["supported_relations"] += metrics["full_proposition_supported"]
            summary["output_tokens"] += row["output_tokens"]
            out.append(
                {
                    **row,
                    "evidence_stage": "qwen_quote_diagnostic",
                    "independent_unit": unit,
                    "claim_limit": "quote binding does not certify relation content",
                }
            )
    return out, dict(summary)


def check_private_corruptions(
    fixture: list[dict[str, Any]], cohort: list[dict[str, Any]], quotes: list[dict[str, Any]]
) -> dict[str, bool]:
    """Private copies prove that five important corruptions fail closed."""
    bad_fixture = deepcopy(fixture)
    bad_cohort = deepcopy(cohort)
    cases = {
        "remove_row": (audit_fixture, (bad_fixture[:-1],)),
        "duplicate_family": (audit_fixture, (bad_fixture + bad_fixture[:1],)),
    }
    swapped = deepcopy(bad_cohort)
    swapped[0]["role"] = "fit"
    exposed = {"online_admission": [{"labels_accessible": True}]}
    roster = {"online_admission": list(dict.fromkeys(r["unit_id"] for r in bad_cohort))}
    cases["swap_role"] = (
        audit_cohort,
        (
            {"online_admission": swapped},
            roster,
            {"online_admission": [{"labels_accessible": False}]},
        ),
    )
    cases["early_admission_label"] = (
        audit_cohort,
        ({"online_admission": bad_cohort}, roster, exposed),
    )
    missing_gate = {
        "honest_verdict": "complete_null",
        "verdict_class": "null",
        "acceptance_gate_results": {},
    }
    del missing_gate["acceptance_gate_results"]
    cases["remove_gate_field"] = (
        lambda value: _required(value, ("acceptance_gate_results",)),
        (missing_gate,),
    )
    verdicts = {}
    for name, (reducer, args) in cases.items():
        try:
            reducer(*args)
        except ValueError:
            verdicts[name] = True
        else:
            verdicts[name] = False
    if quotes:
        try:
            audit_quotes(quotes[:-1])
        except ValueError:
            verdicts["remove_quote_row"] = True
        else:
            verdicts["remove_quote_row"] = False
    return verdicts
