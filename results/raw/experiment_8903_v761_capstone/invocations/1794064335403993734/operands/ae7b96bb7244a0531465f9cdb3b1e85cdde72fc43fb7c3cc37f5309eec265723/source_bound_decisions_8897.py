"""REQ-REPORT-8897: constructed paired worlds qualify three separate source claims."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.reporting import evidence_loss_boundary_8883 as old
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v748_collection_contract import namespace
from carnot.verify.size_aware_witness import verify_size_aware_witness
from carnot.verify.source_bound_witness import verify_source_bound_witness

Json = dict[str, Any]
__all__ = ["old"]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8897_v761_source_bound_decisions", "exp8897-source-bound-decisions"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/verify/source_bound_witness.py",
    "python/carnot/reporting/source_bound_decisions_8897.py",
    "python/carnot/reporting/source_bound_runner_8897.py",
    "scripts/experiments/source_bound_worker_8897.py",
    CLI,
]
TESTS = ["tests/python/test_source_bound_decisions_8897.py"]
SEED = 7618897
CORRUPTIONS = [
    "duplicate_ids",
    "missing_ids",
    "altered_sign",
    "bad_hash",
    "source_bytes",
    "ledger_duplicate",
    "ledger_missing_id",
    "ledger_disagreement",
]
ARMS = ["view_only", "absent", "valid", *CORRUPTIONS, "order", "rename"]
FIELDS = [
    "record_decision",
    "source_coverage_status",
    "source_semantics_status",
    "ledger_errors",
    "budget_exhausted",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so parent supervision can distinguish progress from a stall."""
    print(f"[exp8897] phase={phase} completed={completed} pending={pending}", flush=True)


def freeze(exposed: set[str], *, count: int = 30) -> list[Json]:
    """Reuse the paired graph recipe with a new seed and exclude earlier source structures."""
    generator = namespace(old, "source8897_freeze")
    generator.SEED = SEED
    worlds: list[Json] = generator.freeze(exposed, count=count)
    for world in worlds:
        world["cluster_id"] = world["cluster_id"].replace("v760/8883", "v761/8897")
        for pair in world["pairs"]:
            pair["unit_id"] = pair["unit_id"].replace("v760/8883", "v761/8897")
            pair["cluster_id"] = world["cluster_id"]
            records = [
                dict(record_id=f"record-{i}", fact=fact)
                for i, fact in enumerate(pair["source"]["facts"])
            ]
            pools: dict[str, list[str]] = {}
            for record in records:
                pools.setdefault(canonical_hash(record["fact"]), []).append(record["record_id"])
            ids = []
            for fact in pair["input"]["facts"]:
                pool = pools.get(canonical_hash(fact), [])
                ids.append(pool.pop(0) if pool else "inserted-conflict")
            source = json.dumps(
                dict(records=records), sort_keys=True, separators=(",", ":")
            ).encode()
            pair.update(
                record_ids=ids,
                source_hex=source.hex(),
                ledger=dict(
                    records=records, source_sha256="sha256:" + hashlib.sha256(source).hexdigest()
                ),
            )
            pair["ledger_sha256"] = canonical_hash(pair["ledger"])
    return worlds


def requests(worlds: list[Json]) -> list[Json]:
    """Serving gets visible facts and optional caller provenance, never evaluator labels."""
    result = []
    for world in worlds:
        for pair in world["pairs"]:
            for arm in ARMS:
                request = deepcopy(
                    dict(
                        payload=pair["input"],
                        record_ids=pair["record_ids"],
                        source_ledger=pair["ledger"],
                        source_hex=pair["source_hex"],
                    )
                )
                if arm in {"view_only", "absent"}:
                    request.update(source_ledger=None, source_hex=None, record_ids=None)
                elif arm == "duplicate_ids":
                    request["record_ids"][1] = request["record_ids"][0]
                elif arm == "missing_ids":
                    request["record_ids"] = None
                elif arm == "altered_sign":
                    request["source_ledger"]["records"][0]["fact"][3] = False
                elif arm == "bad_hash":
                    request["source_ledger"]["source_sha256"] = "sha256:corrupt"
                elif arm == "source_bytes":
                    request["source_hex"] += "20"
                elif arm == "ledger_duplicate":
                    request["source_ledger"]["records"].append(
                        request["source_ledger"]["records"][0]
                    )
                elif arm == "ledger_missing_id":
                    request["source_ledger"]["records"][0]["record_id"] = ""
                elif arm == "ledger_disagreement":
                    request["source_ledger"]["records"].pop()
                elif arm == "order":
                    request["payload"]["facts"].reverse()
                    request["record_ids"].reverse()
                    request["source_ledger"]["records"].reverse()
                elif arm == "rename":

                    def rename(fact: list[Any]) -> list[Any]:
                        return ["renamed_" + fact[0], fact[1], "renamed_" + fact[2], *fact[3:]]

                    request["payload"]["facts"] = [rename(f) for f in request["payload"]["facts"]]
                    request["payload"]["query"] = [
                        [t if t.startswith("?") else "renamed_" + t for t in (q[0], q[2])]
                        for q in pair["input"]["query"]
                    ]
                    request["payload"]["query"] = [
                        [q[0], original[1], q[1]]
                        for q, original in zip(
                            request["payload"]["query"], pair["input"]["query"], strict=True
                        )
                    ]
                    for row in request["source_ledger"]["records"]:
                        row["fact"] = rename(row["fact"])
                    source = json.dumps(
                        dict(records=request["source_ledger"]["records"]),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                    request["source_hex"] = source.hex()
                    request["source_ledger"]["source_sha256"] = (
                        "sha256:" + hashlib.sha256(source).hexdigest()
                    )
                result.append(dict(request, arm=arm))
    return result


def observe(request: Json) -> Json:
    """Keep only deterministic claims; timing noise cannot change replay equality."""
    if request["arm"] == "view_only":
        exact = verify_size_aware_witness(request["payload"])
        return dict(
            record_decision=exact["decision"],
            budget_exhausted=exact["budget_exhausted"],
            source_coverage_status="unknown",
            source_semantics_status="unknown",
            ledger_errors=[],
        )
    result = verify_source_bound_witness(
        request["payload"],
        record_ids=request["record_ids"],
        source_ledger=request["source_ledger"],
        source_bytes=bytes.fromhex(request["source_hex"])
        if request["source_hex"] is not None
        else None,
    )
    return {key: result[key] for key in FIELDS}


def scheduled(worlds: list[Json], reason: str, status: str) -> list[Json]:
    """Retain every arm and view when external absence or an owned failure stops work."""
    return [
        dict(
            unit_id=pair["unit_id"],
            cluster_id=world["cluster_id"],
            arm=arm,
            seed=SEED,
            view=pair["view"],
            metric="source_boundary_contract",
            denominator=1,
            status=status,
            missing_reason=reason,
            record_decision=None,
            source_coverage_status="unknown",
            source_semantics_status="unknown",
        )
        for world in worlds
        for pair in world["pairs"]
        for arm in ARMS
    ]


def evaluate(worlds: list[Json], observations: list[Json]) -> Json:
    """Open independent source labels only after all visible decisions have been returned."""
    roster = requests(worlds)
    if len(observations) != len(roster):
        raise ValueError("owned_observation_roster")
    rows, ledgers, errors, source_pairs = [], [], [], []
    parity, rejection, invariant, unsupported = True, True, True, 0
    offset = 0
    for world in worlds:
        world_errors = []
        for pair in world["pairs"]:
            visible_truth, source_truth = old.truth(pair["input"]), old.truth(pair["source"])
            source_pairs.append(
                dict(
                    unit_id=pair["unit_id"],
                    cluster_id=world["cluster_id"],
                    view=pair["view"],
                    source_sha256=pair["source_sha256"],
                    view_sha256=pair["view_sha256"],
                    ledger_sha256=pair["ledger_sha256"],
                    typed_source_sha256=pair["ledger"]["source_sha256"],
                    evaluator_source_truth=source_truth,
                    evaluator_visible_truth=visible_truth,
                )
            )
            baseline = observations[offset + ARMS.index("valid")]
            for arm in ARMS:
                result, request = observations[offset], roster[offset]
                offset += 1
                expected = (
                    "unknown"
                    if arm in {"absent", "view_only"}
                    else "invalid"
                    if arm in CORRUPTIONS or pair["view"] == "insert_conflict"
                    else "complete"
                    if pair["view"] == "complete"
                    else "incomplete"
                )
                semantic_ok = (
                    result["record_decision"] == visible_truth and not result["budget_exhausted"]
                )
                coverage_ok = result["source_coverage_status"] == expected
                parity &= semantic_ok
                rejection &= coverage_ok and (
                    expected != "invalid" or bool(result["ledger_errors"])
                )
                unsupported += int(
                    (result["source_coverage_status"] == "complete" and expected != "complete")
                    or result["source_semantics_status"] != "unknown"
                )
                if arm in {"order", "rename"}:
                    invariant &= all(result[key] == baseline[key] for key in FIELDS)
                if (
                    not semantic_ok
                    or not coverage_ok
                    or result["source_semantics_status"] != "unknown"
                ):
                    world_errors.append(
                        dict(unit_id=pair["unit_id"], arm=arm, expected=expected, observed=result)
                    )
                row = dict(
                    unit_id=pair["unit_id"],
                    cluster_id=world["cluster_id"],
                    arm=arm,
                    seed=SEED,
                    view=pair["view"],
                    metric="source_boundary_contract",
                    denominator=1,
                    status="completed",
                    missing_reason=None,
                    **result,
                )
                rows.append(row)
                ledgers.append(
                    dict(
                        unit_id=pair["unit_id"],
                        cluster_id=world["cluster_id"],
                        arm=arm,
                        source_sha256="sha256:"
                        + hashlib.sha256(bytes.fromhex(request["source_hex"])).hexdigest()
                        if request["source_hex"] is not None
                        else None,
                        view_sha256=canonical_hash(request["payload"]),
                        ledger_sha256=canonical_hash(request["source_ledger"]),
                        expected_coverage=expected,
                        observed_coverage=result["source_coverage_status"],
                        passed=coverage_ok,
                        errors=result["ledger_errors"],
                    )
                )
        errors.append(dict(cluster_id=world["cluster_id"], errors=world_errors))
        progress("scored_worlds", len(errors), len(worlds) - len(errors))
    return dict(
        rows=rows,
        source_view_pairs=source_pairs,
        ledger_validation_rows=ledgers,
        per_world_errors=errors,
        unsupported_claim_count=unsupported,
        controls=dict(
            semantic_parity=parity,
            provenance_rejection=rejection,
            order_rename_invariance=invariant,
            no_unsupported_claims=unsupported == 0,
        ),
    )
