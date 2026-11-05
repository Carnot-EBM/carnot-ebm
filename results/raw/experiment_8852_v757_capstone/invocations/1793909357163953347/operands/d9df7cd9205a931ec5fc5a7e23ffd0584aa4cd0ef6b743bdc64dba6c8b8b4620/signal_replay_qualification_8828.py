"""REQ-REPORT-8828: qualify original arithmetic with invocation-owned primitives.

Only deterministic primitive observations are memoized. The original audit,
optimizer, supervision, sample order and numerical comparisons remain unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterator
from unittest.mock import patch

import jax
import numpy as np

from carnot.reporting import learning_signal_diagnosis_8815 as d
from carnot.reporting import learning_signal_runner_8815 as old
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_8828_v756_signal_replay_qualification"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/signal_replay_qualification_8828.py",
    "python/carnot/reporting/signal_replay_runner_8828.py",
    CLI,
]
TESTS = ["tests/python/test_signal_replay_qualification_8828.py"]
HISTORICAL = ROOT / "results/experiment_8815_v755_learning_signal_diagnosis.json"
HISTORICAL_PIN = "sha256:80e449d811cbe975ff442fd89b3c6f8da4080d193ee96265bfdecd2c1090779a"
DESIGN = "openspec/change-proposals/research-roadmap-v755-preserved-20261104.md"


def inventory() -> list[Json]:
    """Schedule every original seed, objective and rate before reading outcomes."""
    return [
        dict(seed=s, objective=o, lr=lr, updates=400)
        for s in d.SEEDS
        for o in d.OBJECTIVES
        for lr in d.LRS
    ]


def bindings(refs: list[Json], reason: str) -> None:
    """Changed bytes cannot inherit authentication from a previous snapshot."""
    began = time.monotonic()
    for i, ref in enumerate(refs):
        d.require(sha256_file(Path(ref["path"])) == ref["sha256"], reason + ":" + ref["path"])
        if time.monotonic() - began >= 30:
            d.progress(reason, i + 1, len(refs) - i - 1)
            began = time.monotonic()


@contextmanager
def primitives() -> Iterator[list[Json]]:
    """Cache pure calls by effective input, returning copies to prevent mutation.

    Observation uses only membership of this query's normalized template. Other
    templates and their order cannot affect the unchanged primitive's result.
    """
    calls: list[Json] = []
    cache: dict[str, Any] = {}
    originals = {name: getattr(d.m, name) for name in ("normalize", "decide", "observe")}

    def wrapped(name: str, *args: Any) -> Any:
        if name == "observe":
            template = originals["normalize"](args[0])
            args = (args[0], [template] if template in args[1] else [])
        key = canonical_hash([name, args])
        if key not in cache:
            value = originals[name](*args)
            cache[key] = deepcopy(value)
            calls.append(dict(name=name, args=deepcopy(list(args)), value=deepcopy(value)))
        return deepcopy(cache[key])

    with (
        patch.object(d.m, "normalize", lambda *a: wrapped("normalize", *a)),
        patch.object(d.m, "decide", lambda *a: wrapped("decide", *a)),
        patch.object(d.m, "observe", lambda *a: wrapped("observe", *a)),
    ):
        yield calls


def verify_primitives(calls: list[Json]) -> None:
    """Recompute unique primitive values rather than accepting self-reported truth."""
    began = time.monotonic()
    with primitives():
        for i, row in enumerate(calls):
            if time.monotonic() - began >= 30:
                d.progress("verify_primitives", i, len(calls) - i)
                began = time.monotonic()
            d.require(getattr(d.m, row["name"])(*row["args"]) == row["value"], "primitive_drift")


def references(value: Any) -> list[Json]:
    """Enumerate every nested operand so removing a cache entry cannot hide drift."""
    refs = []
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            refs.append(dict(path=value["path"], sha256=value["sha256"]))
        for item in value.values():
            refs.extend(references(item))
    elif isinstance(value, list):
        for item in value:
            refs.extend(references(item))
    return refs


def preconditions(raw: Path) -> Json:
    """Check resources and every historical reference before any scientific work."""
    raw.mkdir(parents=True, exist_ok=True)
    raw.chmod(0o700)
    probe = raw / "write_probe"
    probe.write_bytes(b"private")
    probe.unlink()
    free = shutil.disk_usage(raw).free
    d.require(free >= 2**30, "private_disk_capacity")
    d.require(Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(), "repository_virtualenv")
    d.require(jax.default_backend() == "cpu" and np.isfinite(np.ones(1)).all(), "cpu_dependencies")
    prior = d.load(dict(path=str(HISTORICAL), sha256=HISTORICAL_PIN))
    bindings(prior["code_config_hashes"], "historical_code_hash")
    evidence = d.load(prior["raw_shard_hashes"][0])
    audit = d.load(dict(path=str(ROOT / old.AUDIT), sha256=old.AUDIT_PIN))
    d.require(audit["required_checks_passed"] is True, "rejected_producer_8810")
    source = d.load(audit["raw_shard_hashes"][0])
    refs: dict[str, Json] = {}
    for manifest in (
        source,
        audit["raw_shard_hashes"],
        prior["raw_shard_hashes"],
        evidence["source_artifact_hashes"],
        prior["code_config_hashes"],
        audit["code_config_hashes"],
    ):
        for ref in references(manifest):
            refs[ref["path"]] = ref
    code = [d.reference(ROOT / p) for p in old.CODE + old.TESTS + OWNED + TESTS + [DESIGN]]
    inputs = list(refs.values()) + [d.reference(HISTORICAL)]
    bindings(inputs, "operand_hash")
    result = dict(
        private_output=str(raw),
        free_bytes=free,
        minimum_free_bytes=2**30,
        virtualenv=sys.prefix,
        cpu_backend=jax.default_backend(),
        numpy=np.__version__,
        jax=jax.__version__,
        inputs=inputs,
        code=code,
    )
    atomic_json(raw / "preconditions.json", result)
    return result


def authenticate(raw: Path) -> Json:
    """Run the unchanged complete audit once, then seal its verified primitives."""
    checked = d.load(d.reference(raw / "preconditions.json"))
    bindings(checked["inputs"], "operand_hash")
    began = time.monotonic()
    reduced: Json = {}
    original_reduce = old.audit.reduce

    def capture(source: Json, frozen: Json) -> Json:
        d.progress("unchanged_audit_before", 0, 30600)
        result = original_reduce(source, frozen)
        reduced.update(result)
        d.progress("unchanged_audit_after", len(result["rows"]), 0)
        return result

    with primitives() as calls, patch.object(old.audit, "reduce", capture):
        diagnostic, refs, excluded = old.authenticate()
        d.progress("snapshot_development_before", 0, 60)
        development, split = d.development(excluded)
    prior = d.load(dict(path=str(HISTORICAL), sha256=HISTORICAL_PIN))
    evidence = d.load(prior["raw_shard_hashes"][0])
    for key, value in dict(
        diagnostic_rows=diagnostic,
        source_artifact_hashes=refs,
        development_rows=development,
        split_manifest=split,
    ).items():
        d.require(evidence[key] == value, "historical_evidence_drift:" + key)
    snapshot = dict(
        owner=str(raw),
        code=checked["code"],
        inputs=checked["inputs"],
        primitive_calls=calls,
        audit_reduction=reduced,
        diagnostic_rows=diagnostic,
        source_artifact_hashes=refs,
        excluded=sorted(excluded),
        development_rows=development,
        split_manifest=split,
        audit_duration_s=time.monotonic() - began,
    )
    path = raw / "snapshot.json"
    d.require(not path.exists(), "snapshot_already_exists")
    atomic_json(path, snapshot)
    path.chmod(0o400)
    d.progress("snapshot_authenticated", len(calls), 0)
    return snapshot


def load_snapshot(ref: Json, *, verify: bool = False) -> Json:
    """A child accepts only its invocation's exact snapshot and source bytes."""
    path = Path(ref["path"])
    d.require(sha256_file(path) == ref["sha256"], "snapshot_hash")
    value: Json = json.loads(path.read_bytes())
    d.require(Path(value["owner"]) == path.parent, "snapshot_owner")
    bindings(value["code"], "code_hash")
    expected_code = {str(ROOT / p) for p in old.CODE + old.TESTS + OWNED + TESTS + [DESIGN]}
    d.require({r["path"] for r in value["code"]} == expected_code, "code_inventory")
    if verify:
        prior = d.load(dict(path=str(HISTORICAL), sha256=HISTORICAL_PIN))
        admitted = d.load(prior["raw_shard_hashes"][0])
        d.require(
            value["source_artifact_hashes"] == admitted["source_artifact_hashes"],
            "snapshot_sources",
        )
        audited = d.load(dict(path=str(ROOT / old.AUDIT), sha256=old.AUDIT_PIN))
        d.require(
            value["audit_reduction"] == {k: audited[k] for k in ("rows", "leakage_checks")},
            "audit_reduction_drift",
        )
        bindings(value["inputs"], "operand_hash")
        preflight = d.load(d.reference(path.parent / "preconditions.json"))
        d.require(
            value["inputs"] == preflight["inputs"] and value["code"] == preflight["code"],
            "snapshot_manifest",
        )
        verify_primitives(value["primitive_calls"])
        # Rebuild diagnostic features and split from their admitted public operands.
        source = (
            d.load(value["source_artifact_hashes"][1]) if value["source_artifact_hashes"] else None
        )
        if source is not None:
            d.require(
                all(ref in value["inputs"] for ref in references(source)), "operand_inventory"
            )
            frozen = d.load(value["source_artifact_hashes"][2])
            diagnostic, excluded = [], set()
            with primitives():
                for stream, public_ref in zip(frozen["streams"], source["input_refs"], strict=True):
                    public = d.load(public_ref)
                    for split_name, key in [("online", "worlds"), ("retention", "retention")]:
                        for recipe, payload in zip(stream[key], public[split_name], strict=True):
                            structure = d.structure(payload)
                            excluded.add(structure)
                            for mode in ("off", "on"):
                                diagnostic.append(
                                    dict(
                                        unit_id=recipe["unit_id"],
                                        cluster_id=stream["stream_id"],
                                        structure=structure,
                                        template_mode=mode,
                                        counts=d.m.observe(
                                            payload,
                                            [] if mode == "off" else [d.m.normalize(payload)],
                                        )["counts"],
                                        truth=d.m.decide(payload)["decision"],
                                        full_structure=recipe["structural_hash"],
                                    )
                                )
                development, split = d.development(excluded)
            refs = value["source_artifact_hashes"]
            d.require(
                [diagnostic, refs, sorted(excluded), development, split]
                == [
                    value["diagnostic_rows"],
                    value["source_artifact_hashes"],
                    value["excluded"],
                    value["development_rows"],
                    value["split_manifest"],
                ],
                "snapshot_evidence_drift",
            )
    return value


def fit(snapshot: Json, index: int) -> Json:
    """Each child recomputes one complete original fit, including every update."""
    cell = inventory()[index]
    rows = snapshot["development_rows"]
    train = [r for r in rows if r["split"] == "fit"]
    valid = [r for r in rows if r["split"] == "validation"]
    d.progress("partition_fit_before", index, 45 - index)
    value = d.fit(train, valid, cell["seed"], cell["objective"], cell["lr"])
    d.progress("partition_fit_after", index + 1, 44 - index)
    prior = d.load(dict(path=str(HISTORICAL), sha256=HISTORICAL_PIN))
    expected = d.load(prior["raw_shard_hashes"][0])["fits"][index]
    d.require(value == expected, "fit_drift")
    return value


def projection(authentication_s: float, pilot_s: list[float]) -> Json:
    """Reserve validation time and two full fit passes under the frozen budget."""
    full = 45 * max(pilot_s)
    total = authentication_s + 2 * full + 900
    cold = authentication_s + full + 60
    return dict(
        authentication_s=authentication_s,
        pilot_s=pilot_s,
        projected_full_fit_s=full,
        projected_cold_replay_s=cold,
        validation_reserve_s=900,
        projected_total_s=total,
        budget_s=4200,
        child_deadline_s=600,
        feasible=total <= 4200 and cold <= 600 and authentication_s < 600 and max(pilot_s) < 600,
    )


def child(raw: Path, mode: str, ref: Json | None = None, index: int = 0) -> None:
    """Write results only after a bounded subprocess finishes its owned work."""
    if mode == "authenticate":
        authenticate(raw)
    else:
        d.require(ref is not None, "missing_snapshot")
        snapshot = load_snapshot(ref or {})
        if mode == "controls":
            d.progress("controls_before", 0, 1)
            value: Any = d.controls()
            expected = d.load(json.loads(HISTORICAL.read_bytes())["raw_shard_hashes"][0])[
                "controls"
            ]
            d.require(value == expected and all(r["passed"] for r in value), "control_drift")
            d.progress("controls_after", len(value), 0)
        else:
            value = fit(snapshot, index)
        atomic_json(raw / (mode + ".json"), dict(value=value))
