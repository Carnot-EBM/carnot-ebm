"""Read current V679 evidence from source bytes (REQ-REPORT-7821)."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

PLAN = {
    7813: "results/experiment_7813_v679_decision_measurement.json",
    7815: "results/experiment_7815_v679_qwen_counter_evidence.json",
    7816: "results/experiment_7816_v679_continuous_acquisition.json",
}
PRE_GATE = {7815: "results/experiment_7815_qwen_counter_evidence.json"}
MANIFEST = (
    "results/raw/experiment_7821_v679_independent_evidence_audit/validation_command_manifest.json"
)
GATES = (
    "validity",
    "readiness",
    "probability_quality",
    "decision_benefit",
    "retention",
    "efficiency",
)


def failure(
    number: int, path: Path, field: str, expected: Any, observed: Any, operator: str = "=="
) -> dict[str, Any]:
    """Name the exact failed operand so a missing producer remains auditable."""
    return {
        "upstream_id": f"Exp{number}",
        "artifact_path": str(path.resolve()),
        "artifact_hash": sha256_file(path) if path.is_file() else None,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
    }


def inspect_sources(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Check current science paths and give conductor receipts explanation-only status."""
    sources: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for number, relative in PLAN.items():
        path = root / relative
        source: dict[str, Any] = {
            "upstream_id": f"Exp{number}",
            "path": relative,
            "sha256": sha256_file(path) if path.is_file() else None,
            "date": None,
            "imported_fields": {},
            "raw_paths": {},
            "state": "missing",
            "eligibility": False,
            "pre_gate_receipt": None,
        }
        if number in PRE_GATE:
            receipt = root / PRE_GATE[number]
            if receipt.is_file():
                source["pre_gate_receipt"] = {
                    "path": PRE_GATE[number],
                    "sha256": sha256_file(receipt),
                    "role": "explanation_only",
                }
                try:
                    data = json.loads(receipt.read_bytes())
                    for gate in data["gates_evaluated"]:
                        if gate["passed"] is False:
                            gate_path = Path(gate["artifact_path"])
                            if not gate_path.is_absolute():
                                gate_path = root / gate_path
                            row = failure(
                                number,
                                gate_path,
                                gate["artifact_field"],
                                gate["expected"],
                                gate.get("actual"),
                                gate["op"],
                            )
                            row["gate_upstream"] = gate["upstream"]
                            failures.append(row)
                except (ValueError, TypeError, KeyError):
                    failures.append(
                        failure(
                            number, receipt, "pre_gate_schema", "JSON gates_evaluated", "invalid"
                        )
                    )
        if not path.is_file():
            failures.append(
                failure(
                    number, path, "producer_path", "existing declared science producer", "missing"
                )
            )
            sources.append(source)
            continue
        try:
            data = json.loads(path.read_bytes())
            if not isinstance(data, dict):
                raise ValueError("non-object")
        except (ValueError, UnicodeError):
            source["state"] = "disqualified"
            failures.append(failure(number, path, "schema", "JSON object", "invalid"))
            sources.append(source)
            continue
        before = len(failures)
        expected = {
            "experiment_id": number,
            "milestone": "2026.09.679",
            "run_date": "20260928",
            "flagged_adversarial": False,
        }
        for field, wanted in expected.items():
            if data.get(field) != wanted:
                failures.append(failure(number, path, field, wanted, data.get(field)))
        if data.get("verdict_class") not in ("positive", "circular_positive", "null"):
            failures.append(
                failure(
                    number,
                    path,
                    "verdict_class",
                    ["positive", "circular_positive", "null"],
                    data.get("verdict_class"),
                    "in",
                )
            )
        if not str(data.get("honest_verdict", "")).startswith("complete_"):
            failures.append(
                failure(number, path, "honest_verdict", "complete_*", data.get("honest_verdict"))
            )
        raw = data.get("raw_rows_path")
        safe = (
            isinstance(raw, str)
            and bool(raw)
            and not Path(raw).is_absolute()
            and ".." not in Path(raw).parts
        )
        raw_path = root / raw if safe else path
        digest = sha256_file(raw_path) if safe and raw_path.is_file() else None
        source["raw_paths"] = {"raw_rows_path": {"path": raw, "sha256": digest}}
        if digest is None:
            failures.append(
                failure(number, raw_path, "raw_rows_path", "existing safe declared bytes", raw)
            )
        if data.get("raw_rows_sha256") != digest:
            failures.append(
                failure(number, raw_path, "raw_rows_sha256", data.get("raw_rows_sha256"), digest)
            )
        source["date"] = data.get("run_date")
        source["imported_fields"] = {
            key: data.get(key)
            for key in (
                "experiment_id",
                "milestone",
                "run_date",
                "verdict_class",
                "honest_verdict",
                "flagged_adversarial",
                "raw_rows_path",
            )
        }
        source["state"] = "eligible" if len(failures) == before else "disqualified"
        source["eligibility"] = source["state"] == "eligible"
        sources.append(source)
    return sources, failures


def worst_window(values: list[int]) -> dict[str, Any]:
    """Enumerate every contiguous burst; recovery cannot erase an earlier peak."""
    best = 0
    window: list[int] | None = None
    for start in range(len(values)):
        total = 0
        for end in range(start, len(values)):
            total += values[end]
            if total > best:
                best, window = total, [start, end]
    running = peak = 0
    for value in values:
        running = max(0, running + value)
        peak = max(peak, running)
    return {"peak": best, "running_peak": peak, "final_net": sum(values), "window": window}


def check_debt_claim(values: list[int], claimed: int) -> list[str]:
    """Reject a saved final balance presented as the worst historical burst."""
    result = worst_window(values)
    return [] if result["peak"] == result["running_peak"] == claimed else ["peak_debt"]


def check_interval_units(rows: list[dict[str, Any]], claimed_n: int) -> list[str]:
    """Count source families once, regardless of seeds and intervention views."""
    return [] if len({r.get("family_id") for r in rows}) == claimed_n else ["seed_as_sample"]


def check_events(events: list[dict[str, Any]]) -> list[str]:
    """Replay released labels and commits by clock, including pending restarts."""
    errors: set[str] = set()
    predicted: dict[tuple[Any, Any], float] = {}
    queued: set[tuple[Any, Any]] = set()
    admitted: set[tuple[Any, Any]] = set()
    for event in sorted(events, key=lambda item: (item.get("tick", -1), item.get("sequence", 0))):
        key = (event.get("arm"), event.get("family_id"))
        tick = event.get("tick", -1)
        kind = event.get("kind")
        if kind == "prediction":
            predicted[key] = tick
        elif kind == "feedback":
            if key not in predicted or tick <= predicted[key]:
                errors.add("future_feedback")
            queued.add(key)
        elif kind == "admission":
            if key in admitted or key not in queued:
                errors.add("duplicate_or_early_admission")
            admitted.add(key)
        elif kind == "commit":
            if key not in queued or key not in admitted:
                errors.add("unreleased_commit")
            queued.discard(key)
        elif kind == "restart" and {tuple(pair) for pair in event.get("queued", [])} != queued:
            errors.add("restart_queue_mismatch")
        if kind == "shuffle" and event.get("source_arm") != event.get("arm"):
            errors.add("cross_arm_shuffle")
    return sorted(errors)


def reduce_rows(rows: list[dict[str, Any]], expected_families: int) -> dict[str, Any]:
    """Reprice original examples and leave unknown sentence labels unscored."""
    errors: set[str] = set()
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    output = []
    for row in rows:
        family = row.get("family_id")
        grouped[family].append(row)
        features = row.get("feature_names", [])
        if any("label" in str(name) or "confidence" in str(name) for name in features):
            errors.add("feature_leakage")
        if row.get("label_origin") not in ("independent_annotation", "unknown"):
            errors.add("self_label")
        if row.get("label_join") != family:
            errors.add("label_join")
        if row.get("prediction_tick", -1) >= row.get("label_tick", math.inf):
            errors.add("future_feedback")
        for field in ("source_bytes", "answer_bytes"):
            if field in row and row.get(field.replace("bytes", "sha256")):
                import hashlib

                exact = "sha256:" + hashlib.sha256(row[field].encode()).hexdigest()
                if row[field.replace("bytes", "sha256")] != exact:
                    errors.add("source_answer_bytes")
        params = row.get("checkpoint_parameters", [])
        if (
            not isinstance(params, list)
            or len(params) > 4096
            or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in params)
        ):
            errors.add("checkpoint_parameters")
        span = row.get("target_sentence_span")
        if span is not None:
            answer = row.get("answer_bytes", "").encode()
            if (
                not isinstance(span, list)
                or len(span) != 2
                or span[0] != 0
                or not all(isinstance(x, int) for x in span)
                or not 0 <= span[0] < span[1] <= len(answer)
                or answer[span[0] : span[1]].decode(errors="replace") != row.get("target_sentence")
            ):
                errors.add("target_sentence_span")
            if row.get("label") in (0, 1) and row.get("annotation_sentence_span") != span:
                errors.add("sentence_label_join")
        if row.get("arm") in ("intact", "witness_removed", "control_removed"):
            witness, control = row.get("witness_token_count"), row.get("control_token_count")
            if row.get("token_counter") != "gguf" or not all(
                isinstance(x, int) and x > 0 for x in (witness, control)
            ):
                errors.add("gguf_token_receipt")
            elif abs(witness - control) > 0.25 * witness:
                errors.add("control_token_match")
            if row.get("arm") != "intact" and row.get("removed_source_label") is not None:
                errors.add("modified_source_label")
        label = row.get("label")
        probability = row.get("unsupported_probability", row.get("probability"))
        if (
            label not in (0, 1, None)
            or not isinstance(probability, (int, float))
            or not 0 <= probability <= 1
        ):
            errors.add("primitive_schema")
            metrics = {"brier": None, "cost": None, "fallback": None}
        else:
            brier = None if label is None else round((probability - label) ** 2, 12)
            action = row.get("action")
            cost = (
                None
                if label is None
                else (
                    0.25
                    if action == "escalate"
                    else 5.0
                    if action == "accept" and label == 1
                    else 1.0
                    if action == "reject" and label == 0
                    else 0.0
                )
            )
            metrics = {"brier": brier, "cost": cost, "fallback": int(action == "escalate")}
            if row.get("brier") is not None and not math.isclose(
                row["brier"], brier or 0, abs_tol=1e-9
            ):
                errors.add("saved_metric")
        output.append(
            {
                "family_id": family,
                "seed": row.get("seed"),
                "arm": row.get("arm"),
                "role": row.get("role"),
                "metrics": metrics,
                "raw_provenance": {
                    "source_sha256": row.get("source_sha256"),
                    "answer_sha256": row.get("answer_sha256"),
                },
                "target_sentence_span": span,
                "aligned_label": label if row.get("annotation_sentence_span") == span else None,
            }
        )
    if len(grouped) != expected_families:
        errors.add("family_roster")
    return {"failed_checks": sorted(errors), "independent_n": len(grouped), "rows": output}


def reduce_qwen_pairs(rows: list[dict[str, Any]], expected_families: int) -> dict[str, Any]:
    """Compare matched original families; altered sources never inherit labels."""
    import random

    base = reduce_rows(rows, expected_families)
    errors = set(base["failed_checks"])
    by_family: dict[Any, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["arm"] in by_family[row["family_id"]]:
            errors.add("paired_roster")
        by_family[row["family_id"]][row["arm"]] = row
        if row["arm"] != "intact" and row.get("label") is not None:
            errors.add("modified_source_label")
    shifts = []
    aligned = 0
    for family, arms in by_family.items():
        if set(arms) != {"intact", "witness_removed", "control_removed"}:
            errors.add("paired_roster")
            continue
        if arms["intact"].get("label") in (0, 1) and (
            arms["intact"].get("annotation_sentence_span")
            == arms["intact"].get("target_sentence_span")
        ):
            aligned += 1
        shifts.append(
            (
                family,
                arms["witness_removed"]["unsupported_probability"]
                - arms["control_removed"]["unsupported_probability"],
            )
        )
    values = [value for _, value in shifts]
    rng = random.Random(67821)
    draws = (
        sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(10000))
        if values
        else []
    )
    interval = [draws[250], draws[9749]] if draws else None
    return {
        "failed_checks": sorted(errors),
        "independent_n": len(by_family),
        "aligned_label_count": aligned,
        "mean_shift": round(sum(values) / len(values), 12) if values else None,
        "paired_interval95": interval,
        "rows": [
            {"family_id": family, "shift": round(shift, 12), "modified_source_label": None}
            for family, shift in shifts
        ],
    }


def reduce_feedback_debt(values: list[int], claimed_peak: int) -> dict[str, Any]:
    """Compare saved peak with both complete-window enumeration and recurrence."""
    result = worst_window(values)
    return {
        **result,
        "failed_checks": check_debt_claim(values, claimed_peak),
        "independent_n": len(values),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Correct the full frozen comparison family, including unfavorable tests."""
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, p_values[index] * (len(order) - rank)))
        adjusted[index] = running
    return adjusted


def read_branches(
    root: Path, sources: list[dict[str, Any]]
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Open qualified raw bytes and keep a bad branch separate from good ones."""
    branches: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for source in sources:
        if source["state"] != "eligible":
            continue
        number = int(source["upstream_id"][3:])
        raw = root / source["raw_paths"]["raw_rows_path"]["path"]
        try:
            data = json.loads(raw.read_bytes())
            if not isinstance(data["rowsets"], dict):
                raise ValueError("rowsets must be an object")
            sections = {}
            for role, rows in data["rowsets"].items():
                expected = data.get("expected_families", {}).get(
                    role, len({r["family_id"] for r in rows})
                )
                sections[role] = (
                    reduce_qwen_pairs(rows, expected)
                    if number == 7815
                    else reduce_rows(rows, expected)
                )
                if sections[role]["failed_checks"]:
                    raise ValueError(",".join(sections[role]["failed_checks"]))
            if number == 7816:
                event_errors = check_events(data.get("events", []))
                if event_errors:
                    raise ValueError(",".join(event_errors))
                for key, debt in data.get("feedback_debt", {}).items():
                    sections[key] = reduce_feedback_debt(debt["g_t"], debt["peak"])
                    if sections[key]["failed_checks"]:
                        raise ValueError(",".join(sections[key]["failed_checks"]))
            branches[number] = sections
        except (OSError, KeyError, TypeError, ValueError) as exc:
            source["state"] = "disqualified"
            source["eligibility"] = False
            failures.append(
                failure(
                    number,
                    raw,
                    "raw_reduction",
                    "valid primitive rows",
                    type(exc).__name__ + ":" + str(exc),
                )
            )
    return branches, failures


def private_mutations() -> list[dict[str, Any]]:
    """Exercise the same raw checker against known private defects."""
    import copy
    import hashlib

    row = {
        "family_id": "fixture",
        "seed": 1,
        "arm": "candidate",
        "role": "evaluation64",
        "label": 1,
        "label_origin": "independent_annotation",
        "label_join": "fixture",
        "feature_names": ["public_source_length"],
        "source_bytes": "Source.",
        "answer_bytes": "First. Second.",
        "target_sentence_span": [0, 6],
        "target_sentence": "First.",
        "annotation_sentence_span": [0, 6],
        "prediction_tick": 1,
        "label_tick": 2,
        "probability": 0.8,
        "action": "accept",
        "brier": 0.04,
        "checkpoint_parameters": [0.1],
    }
    for prefix in ("source", "answer"):
        row[prefix + "_sha256"] = (
            "sha256:" + hashlib.sha256(row[prefix + "_bytes"].encode()).hexdigest()
        )
    challenges = [
        ("leaked_label", {"feature_names": ["private_label"]}, "feature_leakage"),
        ("leaked_confidence", {"feature_names": ["gold_confidence"]}, "feature_leakage"),
        ("self_label", {"label_origin": "self_label"}, "self_label"),
        ("altered_prediction", {"probability": 0.1}, "saved_metric"),
        ("future_feedback", {"prediction_tick": 3}, "future_feedback"),
        ("wrong_sentence_join", {"annotation_sentence_span": [7, 14]}, "sentence_label_join"),
    ]
    outcomes = []
    for name, changes, expected in challenges:
        changed = copy.deepcopy(row)
        changed.update(changes)
        observed = reduce_rows([changed], 1)["failed_checks"]
        outcomes.append(
            {
                "mutation": name,
                "expected_check": expected,
                "observed_checks": observed,
                "rejected": expected in observed,
            }
        )
    for name, observed, expected in (
        ("removed_unfavorable_row", reduce_rows([], 1)["failed_checks"], "family_roster"),
        ("seed_as_sample", check_interval_units([row], 2), "seed_as_sample"),
        ("final_net_debt", check_debt_claim([1, 1, -1, -1], 0), "peak_debt"),
    ):
        outcomes.append(
            {
                "mutation": name,
                "expected_check": expected,
                "observed_checks": observed,
                "rejected": expected in observed,
            }
        )
    return outcomes


def build_artifact(
    root: Path,
    date: str,
    sources: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    branches: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Keep branch measurements while closing benefit gates on absent science."""
    blocked = any(source["state"] == "missing" for source in sources)
    disqualified = any(source["state"] == "disqualified" for source in sources)
    ready = int(not blocked and not disqualified and not failures and len(branches) == 3)
    rows = [
        {
            "upstream_id": source["upstream_id"],
            "state": source["state"],
            "role": "science_source",
            "family_id": None,
            "seed": None,
            "arm": None,
            "metrics": None,
            "raw_provenance": {"path": source["path"], "sha256": source["sha256"]},
            "excluded": source["state"] != "eligible",
            "censored": False,
        }
        for source in sources
    ]
    for number, sections in branches.items():
        for role, section in sections.items():
            if "rows" in section:
                rows.extend({**row, "upstream_id": f"Exp{number}"} for row in section["rows"])
            else:
                rows.append(
                    {
                        "upstream_id": f"Exp{number}",
                        "role": role,
                        "family_id": None,
                        "seed": None,
                        "arm": None,
                        "metrics": {
                            "peak_debt": section["peak"],
                            "final_net_debt": section["final_net"],
                        },
                        "raw_provenance": sources[
                            [s["upstream_id"] for s in sources].index(f"Exp{number}")
                        ]["raw_paths"],
                    }
                )
    aligned = [
        {
            "upstream_id": row["upstream_id"],
            "family_id": row["family_id"],
            "seed": row["seed"],
            "label": row["aligned_label"],
            "target_sentence_span": row["target_sentence_span"],
        }
        for row in rows
        if row.get("aligned_label") in (0, 1)
    ]
    independent = sum(
        section["independent_n"]
        for sections in branches.values()
        for section in sections.values()
        if "rows" in section
    )
    artifact: dict[str, Any] = {
        "schema": "independent_evidence_audit_v2",
        "experiment_id": 7821,
        "milestone": "2026.09.679",
        "run_date": date,
        "honest_verdict": "complete_blocked_required_v679_evidence"
        if blocked
        else "complete_disqualified_source_custody"
        if disqualified
        else "complete_null_exposed_development",
        "verdict_class": "blocked" if blocked else "disqualified" if disqualified else "null",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": rows,
        "discrepancy_rows": failures,
        "branch_dispositions": [
            {
                "upstream_id": source["upstream_id"],
                "state": source["state"],
                "reason": "current_science_missing"
                if source["state"] == "missing"
                else "raw_or_custody_disqualified"
                if source["state"] == "disqualified"
                else "cold_recomputed",
            }
            for source in sources
        ],
        "mutation_results": private_mutations(),
        "acceptance_gate_results": {
            "validity": not disqualified,
            "readiness": ready,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "independent_evidence_ready_score": ready,
        "sample_size_budget": {
            "intended": {"Exp7813": 64, "Exp7815": 48, "Exp7816": 64},
            "eligible": independent,
            "started": len(rows) - 3,
            "completed": len(rows) - 3,
            "excluded": sum(s["state"] != "eligible" for s in sources),
            "censored": 0,
            "independent_n": independent,
        },
        "source_artifact_hashes": sources,
        "preconditions_checked": {
            "root": str(root.resolve()),
            "backend": "host aggregation",
            "declared_paths": list(PLAN.values()),
            "resource_check": "CPU and local file access",
            "source_count": len(sources),
        },
        "target_event": "unsupported_probability for first original answer sentence",
        "target_sentence_spans": [row["target_sentence_span"] for row in aligned],
        "aligned_label_rows": aligned,
        "aligned_label_count": len(aligned),
        "claim_scope": {
            "natural_annotations": "exposed_development_only",
            "all_640_source_families_exposed": True,
            "fresh_generalization_eligible": False,
            "fixtures": "circular_positive",
        },
        "verifier_is_oracle": False,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "actual_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {
            key: 0
            for key in (
                "loads",
                "forwards",
                "generations",
                "input_tokens",
                "output_tokens",
                "failures",
                "cancellations",
            )
        },
        "random_seed": {"bootstrap": 67821, "permutation": 67821},
        "duration_s": 0.0,
        "phase_spans": [],
        "validation_receipts": {},
        "validation_command_manifest_path": MANIFEST,
        "validation_command_manifest_sha256": sha256_file(root / MANIFEST)
        if (root / MANIFEST).is_file()
        else None,
        "observed_child_commands": [],
        "repository_health": None,
    }
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "code": sha256_file(Path(__file__)),
            "sources": sources,
            "roles": ["evaluation64", "retention32", "qwen48"],
            "configuration": {"cost": [5, 1, 0.25], "seed": 67821},
        }
    )
    artifact["field_principles"] = {
        key: "Bind this field to current exact bytes and its measured scope." for key in artifact
    }
    artifact["gate_principles"] = {
        key: "Missing science and fixtures cannot prove benefit." for key in GATES
    }
    return artifact


def cold_replay(candidate: Path) -> list[str]:
    """Reopen source and sealed log bytes before accepting saved claims."""
    value = json.loads(candidate.read_bytes())
    root = Path(value["preconditions_checked"]["root"])
    sources, failures = inspect_sources(root)
    branches, raw_failures = read_branches(root, sources)
    expected = build_artifact(root, value["run_date"], sources, failures + raw_failures, branches)
    keys = (
        "source_artifact_hashes",
        "gate_check_summary",
        "rows",
        "discrepancy_rows",
        "branch_dispositions",
        "mutation_results",
        "sample_size_budget",
        "aligned_label_rows",
        "aligned_label_count",
        "independent_evidence_ready_score",
        "reproducibility_checksum",
        "validation_command_manifest_sha256",
    )
    errors = [key + "_changed" for key in keys if value.get(key) != expected[key]]
    for receipt in value.get("observed_child_commands", []):
        path = Path(receipt["log_path"])
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or sha256_file(path) != receipt["log_sha256"]:
            errors.append("validation_log_changed")
    return sorted(set(errors))
