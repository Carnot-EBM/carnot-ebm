"""REQ-VERIFY-8368: schema fields, rather than suffix guesses, identify operands."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
from unittest.mock import patch

from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting import v718_contract_replay as contract
from carnot.reporting import v718_replay_history as history
from carnot.reporting.v710_contract_replay import snapshot

Json = dict[str, Any]
RECORDS = {
    "primitive_reference",
    "work_reference",
    "current_reference",
    "baseline_reference",
    "baseline_source_reference",
    "execution_manifest_reference",
    "owned_coverage_reference",
}
COLLECTIONS = {
    "refs",
    "source_artifact_hashes",
    "code_config_hashes",
    "raw_shard_hashes",
    "historical_fixture_manifest",
    "historical_fixture_hashes",
    "cited_upstream_artifacts",
}
PAIRS = ("runtime_binding", "stdout", "stderr", "log", "binary")


def references(value: Any) -> list[Json]:
    """Only documented records create dependencies; prose still remains in its parent hash.

    Nested containers are traversed, but a bare path/hash pair under an unknown
    field has no schema meaning. Declared records must contain both valid types.
    """
    rows: list[Json] = []

    def record(item: Any) -> None:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not item["path"]
            or not isinstance(item.get("sha256"), str)
            or re.fullmatch(r"sha256:[a-f0-9]{64}", item["sha256"]) is None
            or ("snapshot_path" in item and not isinstance(item["snapshot_path"], str))
        ):
            raise ValueError("typed_reference_schema")
        rows.append(item)

    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"historical", "prior_validation_attempts"}:
                continue
            if key == "field_principles":
                if not isinstance(item, dict) or not all(isinstance(v, str) for v in item.values()):
                    raise ValueError("typed_reference_annotation_schema")
            elif key in RECORDS:
                record(item)
            elif key in COLLECTIONS:
                if not isinstance(item, list):
                    raise ValueError("typed_reference_collection")
                for ref in item:
                    if not (
                        isinstance(ref, dict)
                        and ref.get("exists") is False
                        and ref.get("sha256") is None
                    ):
                        record(ref)
            elif key == "authority_snapshots":
                for ref in item.values():
                    if ref.get("exists") is not False:
                        record(dict(ref, path=ref["source_path"]))
            else:
                rows.extend(references(item))
        for prefix in PAIRS:
            if prefix + "_sha256" in value:
                record(
                    dict(
                        path=value.get(prefix + "_path", value.get(prefix)),
                        sha256=value[prefix + "_sha256"],
                    )
                )
    elif isinstance(value, list):
        for item in value:
            rows.extend(references(item))
    return rows


def authenticate(ref: Json) -> Path:
    """Absence is reported before digest syntax so a missing operand is never measured zero."""
    operand = ref.get("snapshot_path", ref["path"])
    if not isinstance(operand, str):
        raise ValueError("typed_reference_path_type")
    path = Path(operand)
    custody = (
        Path(__file__).resolve().parents[3]
        / "results/raw/experiment_8368_v721_typed_runtime_closure/recovered_source"
        / str(ref.get("sha256", "")).removeprefix("sha256:")
    )
    if path.is_file() and sha256_file(path) != ref.get("sha256") and custody.is_file():
        path = custody
    if not path.is_file():
        raise FileNotFoundError(str(path))
    references(dict(primitive_reference=ref))
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("typed_reference_hash:" + str(path))
    return path


def terminal(primary: Path) -> list[Path]:
    """A byte-bound publication report can authenticate a failed scientific disposition."""
    digest = sha256_file(primary)
    side = primary.parent / "raw" / primary.stem / "validators" / (digest[7:] + ".json")
    if not read_bound_sidecar(primary, side)["report"]["passed"]:
        raise ValueError("historical_terminal_rejected:" + str(primary))
    value = json.loads(primary.read_bytes())
    end = Path(value["terminal_validation_sidecar_path"])
    receipt = json.loads(end.read_bytes())
    if receipt["publication"]["primary_sha256"] != digest or not (
        receipt.get("normal_process_exit") or receipt.get("normal_process_completion")
    ):
        raise ValueError("historical_terminal_binding:" + str(primary))
    return [side, end]


def closure(root: Path, raw: Path, primaries: list[str]) -> list[Json]:
    """Retain declared source and authority copies; never execute patched historical code."""
    refs: list[Json] = []
    seen: dict[str, str] = {}

    def retain(path: Path, expected: str | None = None) -> None:
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError("typed_reference_hash:" + str(path))
        if str(path) in seen:
            return
        seen[str(path)] = digest
        refs.append(snapshot(path, raw / "historical_closure", "source"))

    for name in primaries:
        primary = root / name
        if not primary.is_file():
            continue
        retain(primary)
        value = json.loads(primary.read_bytes())
        if value.get("primitive_reference"):
            authenticate(value["primitive_reference"])
        for ref in references(value):
            retain(authenticate(ref), ref["sha256"])
        primitive = value.get("primitive_reference")
        if primitive:
            for ref in references(json.loads(authenticate(primitive).read_bytes())):
                retain(authenticate(ref), ref["sha256"])
        if value.get("terminal_validation_sidecar_path"):
            for path in terminal(primary):
                retain(path)
                for ref in references(json.loads(path.read_bytes())):
                    retain(authenticate(ref), ref["sha256"])
    return refs


def historical_authority(raw: Path) -> Json:
    """Read the original V720 authority snapshots instead of today's activated plan."""
    from carnot.verify import runtime_reader_8353 as old

    primary = old.ROOT / "results" / (old.NAME + ".json")
    terminal(primary)
    value = json.loads(primary.read_bytes())
    work = json.loads(authenticate(value["primitive_reference"]).read_bytes())
    snapshots = work["authority"]["authority_snapshots"]
    root = raw / "private_original_authority"
    design = root / "openspec/change-proposals/research-roadmap-vNEXT.md"
    design.parent.mkdir(parents=True, exist_ok=True)
    design.write_bytes(
        authenticate(
            dict(snapshots["design"], path=snapshots["design"]["source_path"])
        ).read_bytes()
    )
    (root / "research-roadmap.yaml").write_bytes(
        authenticate(
            dict(snapshots["active"], path=snapshots["active"]["source_path"])
        ).read_bytes()
    )
    with patch.object(
        history,
        "design",
        lambda base, milestone: base / "openspec/change-proposals/research-roadmap-vNEXT.md",
    ):
        return dict(contract.authority(root, raw / "assessment", old.MILESTONE))
