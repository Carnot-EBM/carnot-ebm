"""REQ-REPORT-8346: reuse pinned natural inputs without fitting or reserved targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import v718_contract_replay as custody
from carnot.reporting import sentence_spline_fit_8334 as heads
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.v710_contract_replay import require_reference, snapshot
from carnot.verify.reserved_prediction_seal_8335 import validate_bundle, ARMS

Json = dict[str, Any]
PIN = heads.PINS[heads.PROTOCOL]
SOURCE = heads.SOURCE
HEAD = "results/experiment_8334_v719_sentence_spline_fit.json"
PRED = "results/experiment_8335_v719_reserved_prediction_seal.json"
PINS = dict(
    heads.PINS,
    **{
        HEAD: "sha256:a63db57cb7f735a1171ce16b87d09aa02abe69c7f7abb6fc6ca5fa6240d59208",
        PRED: "sha256:e0a274e9da2053f4f4997d8d52ae53b13857c2c3e39719b530bc8c4ff9339da4",
    },
)
USABLE = dict(
    fit=104, calibration=25, comparator_selection=28, reserved=97, stream=74, later=67, retention=23
)


class CustodyError(ValueError):
    """Carry the exact missing or drifted operand into the terminal gate summary."""

    def __init__(self, gate: Json) -> None:
        super().__init__(gate["artifact_field"])
        self.gate = gate


def bind(ref: Json, raw: Path, refs: list[Json], *, parse: bool = True) -> Json:
    """The reserved target operand is copied and hashed without JSON decoding."""
    saved = snapshot(Path(ref["path"]), raw / "inputs", str(len(refs)))
    refs.append(saved)
    if not saved["exists"] or saved["sha256"] != ref["sha256"]:
        raise CustodyError(
            custody.failure(Path(ref["path"]), "sha256", ref["sha256"], saved["sha256"])
        )
    require_reference(dict(saved, sha256=ref["sha256"]))
    return dict(json.loads(Path(saved["snapshot_path"]).read_bytes())) if parse else {}


def primary(root: Path, name: str, field: str, raw: Path, refs: list[Json]) -> Json:
    """Pinned bytes and passed terminal reports authorize input custody, not benefit."""
    path = root / name
    bind(dict(path=str(path), sha256=PINS[name]), raw, refs, parse=False)
    problems: list[Json] = []
    value = custody.authenticate(path, raw, refs, problems)
    if problems:
        raise ValueError(str(problems))
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    side = read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))
    if (
        side["report"]["passed"] is not True
        or value.get(field) != 1
        or value.get("required_checks_passed") is not True
        or value.get("flagged_adversarial") is not False
        or value.get("verdict_class") in ("disqualified", "blocked")
    ):
        raise ValueError("qualified_terminal_" + field)
    return dict(value)


def recount(source: Json, protocol: Json, raw: Path, refs: list[Json]) -> Json:
    """Fit/tune targets are public to fitting; reserved usable totals remain imported."""
    predictors: Json = {}
    evaluators: Json = {}
    for role in [*heads.ROLES, "reserved"]:
        predictors[role] = bind(source["predictor_shards"][role], raw, refs)["rows"]
        operand = bind(source["evaluator_shards"][role], raw, refs, parse=role != "reserved")
        if role != "reserved":
            evaluators[role] = operand["rows"]
    counts = heads.validate_bundle(
        dict(
            protocol=protocol,
            manifest=source["source_role_manifest"],
            predictors=predictors,
            evaluators=evaluators,
        )
    )
    for role, minimum, per_class in [
        ("fit", 96, 12),
        ("calibration", 24, 4),
        ("comparator_selection", 24, 4),
    ]:
        s = counts[role]
        if (
            s != source["class_support_by_role"][role]
            or s["usable"] < minimum
            or min(s["supported"], s["unsupported"]) < per_class
        ):
            raise ValueError(role + "_support")
    slots = predictors["reserved"]
    for role in ["reserved", "stream", "later", "retention"]:
        selected = [
            p
            for p in slots
            if role == "reserved"
            or (
                p["slot"] <= 96
                if role == "stream"
                else 9 <= p["slot"] <= 96
                if role == "later"
                else p["slot"] >= 97
            )
        ]
        s = dict(source["class_support_by_role"][role])
        if s["intended"] != len(selected) or s["feature_rows"] != sum(
            p["x"] is not None for p in selected
        ):
            raise ValueError(role + "_public_count")
        counts[role] = s
    if {k: v["usable"] for k, v in counts.items()} != USABLE:
        raise ValueError("frozen_usable_counts")
    for role, s in counts.items():
        origin = role if role in predictors else "reserved"
        selected = [
            p
            for p in predictors[origin]
            if role in predictors
            or (
                p["slot"] <= 96
                if role == "stream"
                else 9 <= p["slot"] <= 96
                if role == "later"
                else p["slot"] >= 97
            )
        ]
        s["transport_completed"] = sum(p["status"] == "completed" for p in selected)
        s["missing_features"] = s["intended"] - s["feature_rows"]
        s["usable_count_authority"] = (
            "recounted_fit_tune_targets"
            if role in heads.ROLES
            else "authenticated_8305_summary_targets_unopened"
        )
    return dict(counts=counts, slots=slots)


def frozen_heads(
    head: Json, source: Json, protocol: Json, support: Json, raw: Path, refs: list[Json]
) -> Json:
    """Compare the checkpoint to its original seal; no optimizer executes here."""
    seal = head["frozen_head_manifest"]
    checkpoint = bind(seal["checkpoint_reference"], raw, refs)
    projected = [
        {k: h[k] for k in ["arm", "coefficients", "temperature", "geometry"]}
        for h in checkpoint["heads"]
    ]
    if (
        projected != seal["heads"]
        or checkpoint["selected_comparator"] != seal["selected_comparator"]
    ):
        raise ValueError("checkpoint_seal")
    bundle = dict(
        slots=support["slots"],
        roster=protocol["original_roles"]["evaluation"],
        head_manifest=seal,
        head_hash=seal["whole_model_sha256"],
        protocol_hash=PIN,
    )
    validate_bundle(bundle)
    if seal["source_manifest_reference"]["sha256"] != PINS[SOURCE]:
        raise ValueError("head_source_hash")
    for ref in head["raw_shard_hashes"] + head["code_config_hashes"]:
        bind(ref, raw, refs, parse=False)
    return dict(manifest=seal, bundle=bundle, optimizer_config=head["optimizer_config"])


def frozen_predictions(pred: Json, head: Json, raw: Path, refs: list[Json]) -> Json:
    """Verify original sealed rows and policy links without issuing new predictions."""
    seal = bind(pred["prediction_manifest"], raw, refs)
    bundle = bind(seal["input_reference"], raw, refs)
    rows = []
    for ref in seal["files"]:
        shard = bind(ref, raw, refs)
        if shard["evaluator_labels_opened"] is not False:
            raise ValueError("sealed_label_barrier")
        rows.extend(shard["rows"])
    if (
        bundle != head["bundle"]
        or seal["head_hash"] != head["manifest"]["whole_model_sha256"]
        or seal["protocol_hash"] != PIN
        or seal["evaluator_labels_opened"] is not False
        or seal["intended_slots"] != 128
        or seal["accounted_rows"] != 128 * len(ARMS)
        or rows != pred["prediction_rows"]
    ):
        raise ValueError("prediction_seal")
    for ref in pred["code_config_hashes"]:
        bind(ref, raw, refs, parse=False)
    return dict(
        manifest=seal,
        reference=pred["prediction_manifest"],
        row_count=len(rows),
        label_access_ledger=pred["label_access_ledger"],
    )


def load(root: Path, raw: Path) -> Json:
    """Keep each authenticated input's readiness separate from historical consumers."""
    refs: list[Json] = []
    result: Json = dict(
        refs=refs,
        failures=[],
        support={},
        heads={},
        predictions={},
        historical=[],
        protocol={},
        protocol_sha256=None,
    )
    source: Json = {}
    for phase, path in [("source", SOURCE), ("heads", HEAD), ("predictions", PRED)]:
        try:
            if phase == "source":
                source = primary(root, SOURCE, "fit_support_ready_score", raw, refs)
                protocol = bind(dict(path=str(root / heads.PROTOCOL), sha256=PIN), raw, refs)
                result.update(
                    protocol=protocol,
                    protocol_sha256=PIN,
                    historical=source["historical_model_provenance"],
                )
                result["support"] = recount(source, protocol, raw, refs)
            elif phase == "heads":
                value = primary(root, HEAD, "heads_ready_score", raw, refs)
                result["heads"] = frozen_heads(
                    value, source, result["protocol"], result["support"], raw, refs
                )
            else:
                value = primary(root, PRED, "predictions_ready_score", raw, refs)
                result["predictions"] = frozen_predictions(value, result["heads"], raw, refs)
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            result["failures"].append(
                error.gate
                if isinstance(error, CustodyError)
                else custody.failure(
                    root / path,
                    phase + "_authenticated_reuse",
                    True,
                    str(error) if (root / path).exists() else None,
                )
            )
        print(
            f"[exp8346] phase={phase}_bound completed={len(refs)} pending={3 - ['source', 'heads', 'predictions'].index(phase) - 1}",
            flush=True,
        )
    return result
