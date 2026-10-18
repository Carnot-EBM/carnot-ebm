"""REQ-REPORT-8533: sealed V726 operands remain independent of new source fitting."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from carnot.reporting import v733_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = methods.ROOT
UPSTREAM = "results/experiment_8526_v733_contract_methods.json"
PIN = "sha256:5185ba15b6e23c5a8185553fe4c878921255768bd7d12d88ebf16202d3f5c004"
HEAD = "results/experiment_8431_v726_criterion_energy_fit.json"
HEAD_PIN = "sha256:38db38433d5e2803a7d4b008c88fba5a1a91138abceb42e52f3dadf5b2f10b7f"
PANEL = "results/experiment_8430_v726_criterion_feature_panel.json"
PANEL_PIN = "sha256:c5783590d946a9500c335a7d39e0fb154dd5943c0efc2c1355dde885c3712989"


def reference(path: Path) -> Json:
    """Bind a real file to its exact bytes for cold consumers."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def read(ref: Json) -> Json:
    """Never trust a reference after its bytes have changed."""
    if sha256_file(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("source_hash")
    return dict(json.loads(Path(ref["path"]).read_bytes()))


def operands(root: Path, raw: Path) -> tuple[Json, Json, list[Json], Json]:
    """Read original producer seals and authority without rerunning historical validation."""
    upstream = read(dict(path=str(root / UPSTREAM), sha256=PIN))
    for field in ["methods_ready_score", "nine_input_inputs_ready_score"]:
        if upstream.get(field) != 1 or upstream.get("required_checks_passed") is not True:
            raise ValueError(field)
    terminal = json.loads(Path(upstream["terminal_validation_sidecar_path"]).read_bytes())
    sidecar = read_bound_sidecar(root / UPSTREAM, Path(terminal["publication"]["sidecar_path"]))
    if sidecar.get("report", {}).get("passed") is not True:
        raise ValueError("upstream_terminal_publication")
    protocol = methods.protocol()
    authority = methods.authority(root, raw / "authority")
    if not authority["activated"]:
        raise ValueError("current_task_authority")
    task = next(
        t for t in authority["tasks"] if t["id"] == "exp8533-nine-input-native-qualification"
    )
    if task != next(t for t in protocol["task_map_v733"] if t["id"] == task["id"]):
        raise ValueError("current_task_object")
    originals = {r["experiment_id"]: r for r in upstream["historical_family_receipts"]}
    for number, pin in [(8430, PANEL_PIN), (8431, HEAD_PIN)]:
        if not originals[number]["qualified"] or originals[number]["primary_sha256"] != pin:
            raise ValueError("original_family_qualification")
    fitted = read(dict(path=str(root / HEAD), sha256=HEAD_PIN))
    panel = read(dict(path=str(root / PANEL), sha256=PANEL_PIN))
    heads_ref = dict(path=fitted["frozen_heads_path"], sha256=fitted["frozen_heads_sha256"])
    frozen = read(heads_ref)
    head = frozen["heads"]["spline9"][0]
    if len(head["parameters"]) != 109 or head["seed"] != 17 or head["scaler"] != frozen["scaler"]:
        raise ValueError("original_nine_input_head")
    features_ref = dict(path=panel["feature_panel_path"], sha256=panel["feature_panel_sha256"])
    features = read(features_ref)
    primitive = read(panel["primitive_reference"])
    master_ref = next(
        r
        for r in panel["source_artifact_hashes"]
        if r["path"].endswith("human_judge_release_master.csv")
    )
    if sha256_file(Path(master_ref["path"])) != master_ref["sha256"]:
        raise ValueError("original_text_hash")
    with Path(master_ref["path"]).open(encoding="utf-8-sig") as stream:
        questions = {r["item_id"]: r["question"] for r in csv.DictReader(stream)}
    rows = [
        dict(
            unit_id=r["unit_id"],
            question_hash=r["question_hash"],
            role=r["role"],
            question=questions[r["unit_id"]],
            q=r["q"],
            cached_p2=r["factorial_outputs"]["p2"],
            public=p,
            original_features=r["x"],
            missing_reason=r["missing_reason"],
        )
        for r, p in zip(features["rows"], primitive["public_rows"], strict=True)
    ]
    if len(rows) != 900 or any(r["unit_id"] != r["public"]["family_id"] for r in rows):
        raise ValueError("complete_request_tape")
    tape = dict(
        rows=rows,
        timed_ids=[r["unit_id"] for r in rows if r["role"] == "test"][:64],
        order="original released master order",
        labels_read=0,
        historical_dates=dict(head=fitted["run_date"], panel=panel["run_date"]),
    )
    refs = [
        dict(path=str(root / UPSTREAM), sha256=PIN),
        dict(path=str(root / HEAD), sha256=HEAD_PIN),
        dict(path=str(root / PANEL), sha256=PANEL_PIN),
        heads_ref,
        features_ref,
        panel["primitive_reference"],
        master_ref,
        upstream["methods_protocol_reference"],
    ]
    atomic_json(raw / "head.json", head)
    atomic_json(raw / "request_tape.json", tape)
    return head, tape, refs, upstream
