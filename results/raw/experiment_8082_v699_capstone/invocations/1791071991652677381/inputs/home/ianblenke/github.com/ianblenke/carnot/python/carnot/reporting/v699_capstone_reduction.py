"""REQ-REPORT-8082: immutable primitive equations determine development findings."""

from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any
from unittest.mock import patch

from carnot import experiment_8074_v699_interaction_decision_audit as source
from carnot import experiment_8077_v699_projected_learning_audit as online
from carnot import experiment_8078_v699_feature_cache_core as core
from carnot import experiment_8079_v699_feature_cache_lifecycle as lifecycle
from carnot.reporting import hardware_workload_8081 as hardware
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.evidence_features_custody_7980 import checked

Json = dict[str, Any]
HISTORICAL_SPEC = (
    source.ROOT
    / "results/raw/experiment_8082_v699_capstone/historical/research-reporting-6010583.md"
)
HISTORICAL_SPEC_HASH = "sha256:e5a06e3cc4a06a952f191cd8dca511de91a97814a43ea6f9817d83b85280de5e"
SHARED_CODE = (
    "verify/guarded_transaction_8053.py",
    "verify/evidence_features_7980.py",
    "verify/source_alignment.py",
    "experiment_8027_v695_native_update_cost.py",
)


def original_reference(ref: Json) -> Path:
    """Resolve one archived administrative spec without changing its sealed identity."""
    if (
        ref["path"].endswith("openspec/capabilities/research-reporting/spec.md")
        and ref["sha256"] == HISTORICAL_SPEC_HASH
    ):
        return checked(dict(path=str(HISTORICAL_SPEC), sha256=HISTORICAL_SPEC_HASH))
    return checked(ref)


def independent(data: Json, number: int) -> Json:
    """Use shipped independent equations without rewriting upstream evidence."""
    if data.get("verifier_is_oracle") or not data.get("rows"):
        return dict(measurement_available=False, reason="absent_or_fixture_primitives")
    raw = Path(data["terminal_validation_sidecar_path"]).parent
    if number == 8074:
        with tempfile.TemporaryDirectory(prefix="capstone8082-source-") as temp:
            scratch = Path(temp)
            shutil.copyfile(raw / "predictions.json", scratch / "predictions.json")
            # The helper still verifies the original seal and every scientific input.
            # Only the archived administrative spec gets a byte-identical location.
            with patch.object(source, "checked", original_reference):
                observed = source.evaluate(scratch)
        original = json.loads((raw / "work.json").read_text())["measurement"]
        for key, value in observed.items():
            if key != "label_access_receipt" and value != original[key]:
                raise ValueError("H1_primitive_reduction_drift." + key)
        ledger = data["label_access_receipt"]
        if not ledger["opened_after_seal"] or ledger["selected_role"] != "evaluation96":
            raise ValueError("H1_label_access_order")
        observed["label_access_receipt"] = ledger
        return dict(measurement_available=True, **observed)
    if number == 8077:
        work = json.loads((raw / "work.json").read_text())
        plan = work["data"]
        rebuilt = online.a.reconstruct(
            raw / "trajectory", online.prior.labels(plan, "labels"), budget_s=500
        )
        with tempfile.TemporaryDirectory(prefix="capstone8082-retention-") as temp:
            retained = [
                dict(r, condition="sealed_retention")
                for r in online.independent.retention(
                    plan,
                    rebuilt["final_head_seals"],
                    Path(temp),
                    lambda: online.prior.labels(plan, "retention_labels"),
                )
            ]
        rebuilt.update(
            retention_rows=retained, **online.a.comparisons(rebuilt["later_source_rows"], retained)
        )
        cells = {
            (r["source"], r["seed"], r["arm"]): r
            for r in rebuilt["later_source_rows"]
            if r["denominator"]
        }
        changed = {
            r["source"]
            for r in rebuilt["later_source_rows"]
            if r["denominator"]
            and r["arm"] == "projected_fresh"
            and r["action"] != cells[r["source"], r["seed"], "ray_fresh"]["action"]
        }
        rebuilt["unchanged_sources"] = sorted(
            {r["source"] for r in rebuilt["later_source_rows"] if r["denominator"]} - changed
        )
        for key, value in rebuilt.items():
            if value != work["evidence"][key]:
                raise ValueError("H2_primitive_reduction_drift." + key)
        return dict(measurement_available=True, **rebuilt)
    if number in (8078, 8079):
        observed = json.loads((raw / "observations.json").read_text())
        if observed["rows"] != data["rows"]:
            raise ValueError("cache_primitive_row_drift")
        inputs = json.loads((raw / "inputs.json").read_text())
        cases = {w["identity"]: w for w in inputs["cases"]}
        groups: dict[tuple[Any, ...], list[Json]] = {}
        for row in observed["rows"]:
            if row["status"] in ("completed", "excluded") and row.get("checkpoint"):
                groups.setdefault(
                    (row["mode"], row["condition"], row["transaction_class"], row["repetition"]), []
                ).append(row)
        parity = {p["unit"]: p for p in observed["parity_rows"]}
        for (_, condition, kind, _), pair in groups.items():
            if len(pair) != 4:
                raise ValueError("cache_incomplete_quartet")
            for p in core.quartet_parity(pair, cases[pair[0]["identity"]]):
                if p != parity[p["unit"]]:
                    raise ValueError("cache_checkpoint_parity_drift")
        if observed["population_rows"] != data["population_rows"]:
            raise ValueError("cache_population_drift")
        ratios = core.reduce_rows(observed["rows"])
        if ratios != data["complete_workload_ratios"]:
            raise ValueError("cache_ratio_drift")
        return dict(
            measurement_available=True,
            **observed,
            ratios=ratios,
            independently_recomputed_parity=True,
        )
    if number == 8081:
        inputs = json.loads(Path(data["replay_input_reference"]["path"]).read_text())
        observed = hardware.reduce(inputs)
        if any(value != data[key] for key, value in observed.items()):
            raise ValueError("hardware_primitive_reduction_drift")
        return dict(measurement_available=True, **observed)
    return dict(
        measurement_available=False,
        reason="custody_only_not_a_primary_test",
        primitive_row_count=len(data["rows"]),
        primitive_rows_sha256=canonical_hash(data["rows"]),
    )


def holm(h1: Json | None, h2: Json | None, valid: tuple[bool, bool]) -> list[Json]:
    """Keep exactly the registered two-test family, even when a branch fails."""
    rows = []
    for index, h in enumerate((h1, h2)):
        h = h or {}
        test = h.get("tests", [{}])[0]
        raw = h.get("raw_p_value", test.get("raw_p", 1.0))
        gain = h.get("observed_gain", test.get("gain"))
        qualified = bool(
            valid[index]
            and h.get("support_passed")
            and h.get("safety_passed")
            and type(raw) in (int, float)
            and math.isfinite(raw)
            and 0 <= raw <= 1
        )
        rows.append(
            dict(
                hypothesis=f"H{index + 1}",
                margin=0.02,
                raw_p=raw,
                family_p=float(raw) if qualified else 1.0,
                qualified=qualified,
                observed_gain=gain,
                beneficial_changed_sources=h.get("beneficial_changed_sources", 0),
                primitive_result=h,
                positive_claim=False,
            )
        )
    adjusted = 0.0
    for rank, index in enumerate(sorted(range(2), key=lambda i: (rows[i]["family_p"], i))):
        row = rows[index]
        adjusted = max(adjusted, min(1.0, row["family_p"] * (2 - rank)))
        row.update(
            holm_adjusted_p=adjusted,
            positive_claim=bool(
                row["qualified"]
                and row["observed_gain"] is not None
                and row["observed_gain"] >= 0.02
                and row["beneficial_changed_sources"] >= 5
                and adjusted <= 0.05
            ),
            multiplicity="Holm .05 across exactly H1/H2",
            uncertainty_scope="conditional exposed development; seeds add no source groups",
        )
    return rows


def service_join(core_data: Json, lifecycle_data: Json, valid: tuple[bool, bool]) -> Json:
    """Exact shared identities prevent timings from different workloads being spliced."""
    parts = [core_data, lifecycle_data]
    identities = []
    for data in parts:
        raw = Path(data.get("raw_directory", "/absent"))
        inputs = (
            json.loads((raw / "inputs.json").read_text()) if (raw / "inputs.json").is_file() else {}
        )
        codes = data.get("code_config_hashes", [])
        identities.append(
            dict(
                workload_sha256=canonical_hash(
                    {k: inputs.get(k) for k in ("cases", "head", "sources")}
                ),
                code_hashes={p["path"]: p["sha256"] for p in codes},
                native_build_sha256=data.get("loaded_library_receipt", {}).get("sha256"),
                configuration_sha256=canonical_hash(data.get("config")),
            )
        )
    shared = set(identities[0]["code_hashes"]) & set(identities[1]["code_hashes"])
    required = {str(source.ROOT / "python/carnot" / p) for p in SHARED_CODE}
    exact = bool(
        all(valid)
        and all(
            p.get("verdict_class") in ("positive", "null") and not p.get("verifier_is_oracle")
            for p in parts
        )
        and required <= shared
        and identities[0]["workload_sha256"] == identities[1]["workload_sha256"]
        and identities[0]["native_build_sha256"] is not None
        and identities[0]["native_build_sha256"] == identities[1]["native_build_sha256"]
        and identities[0]["configuration_sha256"] == identities[1]["configuration_sha256"]
        and all(identities[0]["code_hashes"][p] == identities[1]["code_hashes"][p] for p in shared)
    )
    cells, parity, population, rows = [], [], [], []
    for index, (eid, names) in enumerate(hardware.MODES.items()):
        data = parts[index]
        for mode in names:
            for condition, kind in hardware.CELLS:
                selected = [
                    r
                    for r in data.get("rows", [])
                    if (r["mode"], r["condition"], r["transaction_class"])
                    == (mode, condition, kind)
                    and r["status"] == "completed"
                    and r["repetition"] >= 0
                ]
                roster = {(r["arm"], r["repetition"]) for r in selected}
                wanted = {(arm, rep) for arm in hardware.ARMS for rep in range(30)}
                ready = bool(exact and roster == wanted and len(selected) == 120)
                cells.append(
                    dict(
                        mode=mode,
                        condition=condition,
                        transaction_class=kind,
                        source=f"exp{eid}",
                        observed_count=len(selected),
                        intended_count=120,
                        qualified=ready,
                        status="completed" if ready else "excluded",
                        exclusion_reason=None
                        if ready
                        else "missing_or_disqualified_or_identity_mismatch",
                    )
                )
        if valid[index]:
            rows.extend(data.get("rows", []))
            population.extend(data.get("population_rows", []))
            for p in data.get("parity_rows", []):
                parity.append(dict(p, partition=f"exp{eid}"))
    parity_passed = bool(parity and all(p.get("passed") is True for p in parity))
    complete = bool(exact and all(c["qualified"] for c in cells) and parity_passed)
    ratios = core.reduce_rows(rows) if complete else []
    service = lifecycle.service_reduce(ratios, population) if complete else []
    breaks = []
    for cell in ratios:
        if cell["mode"] != "warm":
            continue
        population_ns = sum(
            p["elapsed_ns"]
            for p in population
            if (p["mode"], p["condition"], p["transaction_class"], p["arm"])
            == ("warm", cell["condition"], cell["transaction_class"], cell["arm"])
        )
        saving = (cell["numerator"] - cell["denominator"]) / cell["paired_repetitions"]
        breaks.append(
            dict(
                arm=cell["arm"],
                condition=cell["condition"],
                transaction_class=cell["transaction_class"],
                population_ns=population_ns,
                warm_saving_ns=saving,
                requests=math.ceil(population_ns / saving) if saving > 0 else None,
            )
        )
    return dict(
        join_identities=identities,
        shared_code_paths=sorted(shared),
        exact_identity_match=exact,
        complete_six_mode_credit=complete,
        mode_condition_rows=cells,
        parity_rows=parity,
        parity_passed=parity_passed,
        complete_workload_ratios=ratios,
        population_inclusive_ratios=service,
        break_even_requests=breaks,
        complete_service_measured=False,
        native_10x_requirement_closed=False,
        unknown_costs=["source acquisition", "original Qwen inference", "external feedback"],
        claim_scope="one exposed host workload; cache transactions exclude acquisition",
    )


def retirements(tasks: list[Json], rows: list[Json], history: list[Json]) -> list[Json]:
    """Retire authenticated exact failed repeats while preserving continuity obligations."""
    lookup = {r["task_id"]: r for r in history}
    result = []
    for task, row in zip(tasks, rows, strict=True):
        for old in task.get("prior_failures", []):
            historical = lookup.get(old["experiment_id"], {})
            same = bool(
                historical.get("sha256")
                and historical.get("honest_verdict") == old["verdict"]
                and row["honest_verdict"] == old["verdict"]
            )
            monitoring = task["id"].startswith("exp8080-") and row["verdict_class"] == "null"
            retire = bool(
                same
                and old["retire_if_same_verdict"]
                and not monitoring
                and row["verdict_class"] in ("blocked", "disqualified", "null")
            )
            result.append(
                dict(
                    task_id=task["id"],
                    prior_failure=old,
                    exact_repeat=same,
                    retire=retire,
                    prior_evidence_sha256=historical.get("sha256"),
                    current_evidence_sha256=row.get("sha256"),
                    mandatory_monitoring=monitoring,
                    rationale="Retire only the exact repeated failure under this declared mechanism."
                    if retire
                    else "Verdict or mechanism changed, prior custody is absent, or valid monitoring remains mandatory.",
                    mechanism_scope=old["addressed_by"],
                    reopen_condition=old["addressed_by"],
                )
            )
    return result
