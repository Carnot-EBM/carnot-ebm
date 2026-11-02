"""Independent byte custody and family reduction for REQ-REPORT-7721."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import sha256_file


def failed_check(
    check: str, upstream_id: str, path: Path, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Give a failed gate its exact upstream, byte location, and operands."""
    return {
        "check": check,
        "upstream_id": upstream_id,
        "artifact_path": str(path.resolve()),
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
    }


def inspect_sources(
    root: Path, plan: dict[int, str], required: set[int]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Hash all planned input bytes and classify absent, valid, and pre-gate files."""
    custody: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {
        "valid_producers": {},
        "flagged_historical_evidence": {},
        "pre_gate_receipts": {},
        "missing_custody": [],
    }
    failures: list[dict[str, Any]] = []
    for number, relative in plan.items():
        path = root / relative
        source = f"Exp{number}"
        if not path.is_file():
            hashes["missing_custody"].append(relative)
            custody.append({"upstream_id": source, "artifact_path": relative, "state": "missing"})
            if number in required:
                failures.append(
                    failed_check("required_science_exists", source, path, "exists", True, False)
                )
            continue
        digest = sha256_file(path)
        try:
            value = json.loads(path.read_bytes())
        except (ValueError, UnicodeDecodeError):
            value = None
        if not isinstance(value, dict):
            state = "pre_gate"
            field, expected, observed = "json_object", True, False
        else:
            verdict = value.get("honest_verdict")
            valid = (
                isinstance(verdict, str)
                and verdict.startswith("complete_")
                and value.get("verdict_class") in {"positive", "null", "circular_positive"}
                and value.get("flagged_adversarial") is False
            )
            state = "valid" if valid else "pre_gate"
            field, expected, observed = (
                "verdict_class",
                "eligible_terminal",
                value.get("verdict_class"),
            )
            if not isinstance(verdict, str) or not verdict.startswith("complete_"):
                field, observed = "honest_verdict", verdict
            elif value.get("flagged_adversarial") is not False:
                field, expected, observed = "flagged_adversarial", False, value.get(field)
        bucket = (
            "valid_producers"
            if state == "valid"
            else "flagged_historical_evidence"
            if number < 7713
            else "pre_gate_receipts"
        )
        hashes[bucket][relative] = digest
        custody.append(
            {
                "upstream_id": source,
                "artifact_path": relative,
                "state": state,
                "sha256": digest,
                "verdict_class": value.get("verdict_class") if isinstance(value, dict) else None,
            }
        )
        if number in required and state != "valid":
            failures.append(
                failed_check("required_science_eligible", source, path, field, expected, observed)
            )
    return custody, hashes, failures


def reduce_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Recompute raw source-family, chronology, loss, action, and retention facts."""
    roster = bundle["family_roster"]
    registry = bundle["source_registry"]
    rows = bundle["rows"]
    failures: list[dict[str, Any]] = []

    def fail(check: str, field: str, expected: Any, observed: Any) -> None:
        failures.append(
            {
                "check": check,
                "field": field,
                "operator": "==",
                "expected": expected,
                "observed": observed,
            }
        )

    seen: set[tuple[str, str]] = set()
    families: set[str] = set()
    if len(roster) != len(set(roster)) or set(registry) != set(roster):
        fail("family_roster", "family_roster", "unique_registry_match", roster)
    by_arm: dict[str, list[float]] = defaultdict(list)
    reduced_rows: list[dict[str, Any]] = []
    for row in rows:
        family = row["family_id"]
        if family not in registry or row["source_sha256"] != registry.get(family):
            fail(
                "source_identity",
                "family_id/source_sha256",
                registry.get(family),
                [family, row["source_sha256"]],
            )
        unit_arm = (family, row["arm"])
        if unit_arm in seen:
            fail("duplicate_family_arm", "family_id/arm", "unique", list(unit_arm))
        seen.add(unit_arm)
        families.add(family)
        if row["label_release_tick"] <= row["prediction_tick"]:
            fail(
                "future_label", "label_release_tick", "> prediction_tick", row["label_release_tick"]
            )
        if row["annotation_origin"] != "human":
            fail("annotation_origin", "annotation_origin", "human", row["annotation_origin"])
        if row["static_bank_features"] <= 0:
            fail("static_closure", "static_bank_features", "> 0", row["static_bank_features"])
        if row["candidate_hash_at_prediction"] != row["candidate_hash_at_admission"]:
            fail(
                "frozen_candidate",
                "candidate_hash_at_admission",
                row["candidate_hash_at_prediction"],
                row["candidate_hash_at_admission"],
            )
        if row["typed_action"] not in {"accept", "reject", "escalate"}:
            fail("typed_action", "typed_action", "accept|reject|escalate", row["typed_action"])
        probability = row["probability_error"]
        label = row["label"]
        covered = row["coverage"]
        if covered and (
            type(label) is not int
            or label not in (0, 1)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            fail(
                "probability",
                "label/probability_error",
                "binary/finitely_bounded",
                [label, probability],
            )
        brier = (
            (probability - label) ** 2
            if covered
            and type(label) is int
            and label in (0, 1)
            and isinstance(probability, (int, float))
            and math.isfinite(probability)
            else None
        )
        if brier is not None:
            by_arm[row["arm"]].append(brier)
        retention = row["retention_probability_error"]
        retention_label = row["retention_label"]
        retention_loss = (
            (retention - retention_label) ** 2
            if isinstance(retention, (int, float))
            and type(retention_label) is int
            and retention_label in (0, 1)
            else None
        )
        reduced_rows.append(
            {
                "family_id": family,
                "arm": row["arm"],
                "coverage": covered,
                "brier": brier,
                "typed_action": row["typed_action"],
                "retention_brier": retention_loss,
                "future_template_firings": row["future_template_firings"],
                "provenance": {"source_sha256": row["source_sha256"]},
            }
        )
    if families != set(roster):
        fail("family_coverage", "family_ids", sorted(roster), sorted(families))
    for event in bundle["events"]:
        if event["released_tick"] >= event["use_tick"]:
            fail("event_chronology", "released_tick", "< use_tick", event["released_tick"])
        if event["kind"] == "proposal" and event["charged"] is not True:
            fail("proposal_charge", "charged", True, event["charged"])
    arm_summary = {
        arm: {"covered_families": len(losses), "brier_mean": sum(losses) / len(losses)}
        for arm, losses in by_arm.items()
    }
    return {
        "failed_checks": failures,
        "rows": reduced_rows,
        "sample_size": {
            "intended_families": len(roster),
            "observed_families": len(families),
            "covered_families": len({row["family_id"] for row in rows if row["coverage"]}),
            "zero_coverage_families": len(
                set(roster) - {row["family_id"] for row in rows if row["coverage"]}
            ),
        },
        "by_arm": arm_summary,
        "frozen_best_arm": bundle["frozen_best_arm"],
        "latent_vs_pooled_brier": (
            arm_summary["latent"]["brier_mean"] - arm_summary["pooled"]["brier_mean"]
            if "latent" in arm_summary and "pooled" in arm_summary
            else None
        ),
    }
