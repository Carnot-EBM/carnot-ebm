"""REQ-REPORT-8318: old outcomes remain evidence even when their science failed."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

from carnot.reporting import v717_capstone_evidence as old
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v710_contract_replay import require_reference, snapshot
from carnot.verify.cached_sentence_custody_8305 import check_predictor

Json = dict[str, Any]
OLD_DESIGN = "openspec/change-proposals/research-roadmap-v716-preserved-20261008.md"
PRIOR_DESIGN = "openspec/change-proposals/research-roadmap-v717-preserved-20261008.md"
CUSTODY = "results/experiment_8305_v717_cached_sentence_custody.json"
POLICY: Json = dict(
    version="8318-v1",
    allowed_exit_codes=[0, 1],
    allowed_info_kinds=["IMPLAUSIBLE_PERFECT"],
    required_binding=["candidate_sha256", "verifier_sha256"],
    info_resolution="independent primitives and deliberate error rejection",
    blocked=["process_error", "missing_report", "unknown", "warn", "critical"],
)


def design(root: Path, milestone: str) -> Path:
    """An explicit version selects preserved bytes instead of mutable future plans."""
    return (
        root
        / {
            "2026.10.716": OLD_DESIGN,
            "2026.10.717": PRIOR_DESIGN,
            "2026.10.718": "openspec/change-proposals/research-roadmap-vNEXT.md",
        }[milestone]
    )


def first_difference(expected: Any, observed: Any, field: str = "") -> Json | None:
    """Retain the first actual differing operand instead of a generic drift label."""
    if expected == observed:
        return None
    if (
        isinstance(expected, dict)
        and isinstance(observed, dict)
        and expected.keys() == observed.keys()
    ):
        for key in expected:
            found = first_difference(
                expected[key], observed[key], field + ("." if field else "") + key
            )
            if found:
                return found
    if isinstance(expected, list) and isinstance(observed, list) and len(expected) == len(observed):
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            found = first_difference(left, right, f"{field}[{index}]")
            if found:
                return found
    return dict(field=field, expected=expected, observed=observed)


def historical(root: Path, raw: Path) -> Json:
    """Preserve failed original bytes and prove that serialization order caused drift."""
    primary = json.loads((root / "results/experiment_8317_v717_capstone.json").read_bytes())
    source = Path(primary["work_reference"]["path"]).parent / "valid.json"
    candidate = snapshot(source, raw / "history", "original")
    bound = next(r for r in primary["raw_shard_hashes"] if r["path"] == str(source))
    require_reference(dict(candidate, sha256=bound["sha256"]))
    value = json.loads(Path(candidate["snapshot_path"]).read_bytes())
    work_ref = snapshot(Path(value["work_reference"]["path"]), raw / "history", "work")
    require_reference(dict(work_ref, sha256=value["work_reference"]["sha256"]))
    work = json.loads(Path(work_ref["snapshot_path"]).read_bytes())
    closure: dict[str, str] = {}

    def retain(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("exists") and value.get("snapshot_path"):
                original = value["snapshot_path"]
                if original not in closure:
                    private = (
                        raw
                        / "history"
                        / "closure"
                        / (value["sha256"][7:] + "-" + str(len(closure)))
                    )
                    private.parent.mkdir(parents=True, exist_ok=True)
                    if not private.exists():
                        shutil.copyfile(original, private)
                        private.chmod(0o400)
                    require_reference(dict(path=str(private), sha256=value["sha256"]))
                    closure[original] = str(private)
                value["snapshot_path"] = closure[original]
            for item in value.values():
                retain(item)
        elif isinstance(value, list):
            for item in value:
                retain(item)

    retain(work)
    private_work = raw / "history" / "private_work.json"
    atomic_json(private_work, work)
    original_work_ref = work_ref
    work_ref = snapshot(private_work, raw / "history", "serialized_work")
    work = json.loads(Path(work_ref["snapshot_path"]).read_bytes())
    reduced = restore_locations(old.reduce(work, value["validation_receipts"]), closure)
    expected = {k: value[k] for k in reduced}
    mismatch = first_difference(expected, reduced)
    restored = deepcopy(work)
    order = [8303, 8259, 8302, 8289]
    restored["history"] = {
        k: v for i in order for k, v in work["history"].items() if int(k.split("_")[1]) == i
    }
    explained = (
        restore_locations(old.reduce(restored, value["validation_receipts"]), closure) == expected
    )
    stable = restore_locations(
        old.reduce(json.loads(json.dumps(work, sort_keys=True)), value["validation_receipts"]),
        closure,
    )
    return dict(
        original_candidate=candidate,
        work_reference=work_ref,
        original_work_reference=original_work_ref,
        relocation_map=closure,
        first_reduction_mismatch=mismatch,
        recomputed=reduced,
        reduction_sha256=canonical_hash(reduced),
        deterministic=bool(mismatch and explained and stable == reduced),
        cause="serialized_history_key_order" if explained else "unknown",
        historical_verdict_class=primary["verdict_class"],
        historical_honest_verdict=primary["honest_verdict"],
        historical_receipts=value["validation_receipts"],
    )


def restore_locations(value: Any, relocation: dict[str, str]) -> Any:
    """Reversible path relocation preserves every compared field's original identity."""
    reverse = {copied: original for original, copied in relocation.items()}
    if isinstance(value, dict):
        return {k: restore_locations(v, relocation) for k, v in value.items()}
    if isinstance(value, list):
        return [restore_locations(v, relocation) for v in value]
    return reverse.get(value, value) if isinstance(value, str) else value


def support(source: Json, raw: Path, refs: list[Json]) -> Json:
    """Count original predictor/target pairs; byte custody does not imply fresh data."""
    predictors: Json = {}
    targets: Json = {}
    for name, destination in [("predictor_shards", predictors), ("evaluator_shards", targets)]:
        for role, operand in source[name].items():
            ref = snapshot(Path(operand["path"]), raw / "shards", name + role)
            require_reference(dict(ref, sha256=operand["sha256"]))
            refs.append(ref)
            destination[role] = json.loads(Path(ref["snapshot_path"]).read_bytes())["rows"]
    manifest = []
    for role, rows in predictors.items():
        for p in rows:
            check_predictor(p)
            manifest.append(
                {
                    k: p[k]
                    for k in (
                        "unit_id",
                        "source_cluster_id",
                        "role",
                        "slot",
                        "source_sha256",
                        "answer_sha256",
                    )
                }
            )
    if canonical_hash(sorted(manifest, key=lambda p: (p["role"], p["slot"]))) != canonical_hash(
        sorted(source["source_role_manifest"], key=lambda p: (p["role"], p["slot"]))
    ):
        raise ValueError("source_role_manifest")
    counts = {}
    for role in [*predictors, "stream", "later", "retention"]:
        origin = role if role in predictors else "reserved"
        pairs = [
            (p, t)
            for p, t in zip(predictors[origin], targets[origin], strict=True)
            if role in predictors
            or (
                p["slot"] <= 96
                if role == "stream"
                else 9 <= p["slot"] <= 96
                if role == "later"
                else p["slot"] >= 97
            )
        ]
        if any(
            p["unit_id"] != t["unit_id"] or p["source_cluster_id"] != t["source_cluster_id"]
            for p, t in pairs
        ):
            raise ValueError("predictor_target_identity")
        counts[role] = dict(
            intended=len(pairs),
            feature_rows=sum(p["x"] is not None for p, _ in pairs),
            usable=sum(p["x"] is not None and t["y"] in (0, 1) for p, t in pairs),
            supported=sum(p["x"] is not None and t["y"] == 0 for p, t in pairs),
            unsupported=sum(p["x"] is not None and t["y"] == 1 for p, t in pairs),
            missing_labels=sum(p["x"] is not None and t["y"] is None for p, t in pairs),
        )
    ready = counts == source["class_support_by_role"] and all(
        counts[r]["usable"] >= n and min(counts[r]["supported"], counts[r]["unsupported"]) >= c
        for r, n, c in [("fit", 96, 12), ("calibration", 24, 4), ("comparator_selection", 24, 4)]
    )
    return dict(
        ready=ready,
        counts=counts,
        predictor_shards=source["predictor_shards"],
        evaluator_shards=source["evaluator_shards"],
        manifest_sha256=canonical_hash(manifest),
    )


def verifier_hash() -> str:
    """Bind the consumer to the unchanged on-disk verifier, not an assumed version."""
    return str(sha256_file(old.ROOT / "scripts/adversarial_verify.py"))


def consume(report: Json, candidate: Path, exit_code: int, proof: Json) -> Json:
    """Only verified arithmetic may resolve a known informational finding."""
    reports = report.get("reports", [])
    if not isinstance(reports, list) or any(not isinstance(r, dict) for r in reports):
        return dict(passed=False, findings=[], dispositions=[], report=report)
    if reports and (
        not isinstance(reports[0].get("flags"), list)
        or any(not isinstance(f, dict) for f in reports[0]["flags"])
    ):
        return dict(passed=False, findings=[], dispositions=[], report=report)
    bound = (
        exit_code in (0, 1)
        and report.get("candidate_sha256") == sha256_file(candidate)
        and report.get("verifier_sha256") == verifier_hash()
        and len(reports) == 1
        and reports[0].get("loaded") is True
        and reports[0].get("artifact") == str(candidate)
    )
    flags = reports[0].get("flags", []) if reports else []
    dispositions = [
        dict(
            finding=f,
            resolved=bool(
                f.get("severity") == "info"
                and f.get("kind") in POLICY["allowed_info_kinds"]
                and "dense_sparse_error_max" in f.get("detail", "")
                and proof.get("recomputed") is True
                and proof.get("deliberate_error_rejected") is True
            ),
            evidence=proof,
        )
        for f in flags
    ]
    passed = bool(
        bound
        and reports[0].get("flag_count") == len(flags)
        and exit_code == int(bool(flags))
        and all(d["resolved"] for d in dispositions)
    )
    return dict(passed=passed, findings=flags, dispositions=dispositions, report=report)


def parity(work: Json, protocol: Json) -> float:
    """Independent recursive basis arithmetic checks every stored dense/sparse release."""
    from carnot.verify.local_update_isolation_8306 import scalar_design

    errors: list[float] = []
    for stored, trajectory in zip(work["states"], protocol["trajectories"], strict=True):
        for left, right in zip(
            stored["arms"]["full"]["releases"], stored["arms"]["indexed"]["releases"], strict=True
        ):
            errors.extend(
                abs(a - b) for a, b in zip(left["coefficients"], right["coefficients"], strict=True)
            )
            for x in trajectory["cache_x"]:
                basis = scalar_design(x)
                errors.append(
                    abs(
                        sum(a * b for a, b in zip(left["coefficients"], basis, strict=True))
                        - sum(a * b for a, b in zip(right["coefficients"], basis, strict=True))
                    )
                )
    return max(errors)


def numeric(root: Path, raw: Path, refs: list[Json]) -> Json:
    """A corrupted coefficient must fail before any exact-zero finding is resolved."""
    primary = root / "results/experiment_8306_v717_local_update_isolation.json"
    value = json.loads(primary.read_bytes())
    operands = []
    for operand in [value["measurement_reference"], *value["raw_shard_hashes"]]:
        ref = snapshot(Path(operand["path"]), raw / "numeric", str(len(refs)))
        require_reference(dict(ref, sha256=operand["sha256"]))
        refs.append(ref)
        operands.append(json.loads(Path(ref["snapshot_path"]).read_bytes()))
    work, protocol = operands[0], operands[-1]
    measured = parity(work, protocol)
    altered = deepcopy(work)
    altered["states"][0]["arms"]["indexed"]["releases"][0]["coefficients"][0] += 0.125
    wrong = parity(altered, protocol)
    negative = raw / "corrupted_measurement.json"
    atomic_json(negative, altered)
    return dict(
        recomputed=measured == value["dense_sparse_error_max"],
        deliberate_error_rejected=wrong != value["dense_sparse_error_max"],
        observed_error=measured,
        deliberate_error=wrong,
        primary_path=str(primary),
        primary_sha256=sha256_file(primary),
        corrupted_measurement_reference=dict(path=str(negative), sha256=sha256_file(negative)),
    )
