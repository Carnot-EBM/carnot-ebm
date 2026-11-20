"""REQ-REPORT-9097-V776: resolve old authority without repinning old evidence.

The historical mapper and its imported project dependencies run in a separate
Python process. Independent author bytes and original roles must agree with it
before any downstream feature or evaluator file becomes usable.
"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import shutil
from typing import Any

from carnot.reporting import human_methods_9084 as methods
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.natural_label_eligibility_8975 import ref as ref
from carnot.reporting.natural_label_runner_8975 import read_ref
from carnot.reporting.spline_basis_methods_8986 import gate as gate
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK = "experiment_9097_v776_human_authority", "exp9097-human-authority"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_human_authority_9097.py"
WORKER = "python/carnot/reporting/human_authority_replay_9097.py"
OWNED = [
    "python/carnot/reporting/human_authority_9097.py",
    "python/carnot/reporting/human_authority_runner_9097.py",
    WORKER,
    CLI,
]
DEFAULT_INPUT = Path("/tmp/carnot-9097-private")
MODEL_SPECS: list[Json] = []
COMMIT = "8e8b08277f98f6ef3a3680e5a9d2653d7761c69e"
PRODUCER = "python/carnot/reporting/human_label_custody_8389.py"
SNAPSHOT_PIN = "sha256:52cc5fe1903f0c991ccb416903d873f14d1d01785329e4d005fe77fb2268d128"
DEPENDENCIES = {
    "human_label_custody_8389": SNAPSHOT_PIN,
    "current_work_receipt": "sha256:428d1696eef0820ce9df519532743aeccb8ec1156eb8c1b52fa0c8f72bf16a10",
    "external_evidence_8375": "sha256:ec1c58150e244454c137a2ea68ef68f3e23a15dc673a3a3fbc97e0f0ce1de85a",
}
PINS = dict(
    methods.PINS,
    experiment_9084_v775_human_methods="4249a4e9b12e62f5638c161b546049946f30cf2f82a64be9525707e4b672f4df",
)
ROLE_PIN = "sha256:bb868af1d6f173c5b98155498e5daadd286e2641ef4aa3c20b7a0671b1b95117"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so the supervisor can distinguish an unavailable input from a stall."""
    print(f"[exp9097] phase={phase} completed={completed} pending={pending}", flush=True)


def seal(path: Path, value: Json) -> Json:
    """Keep identical private bytes on replay and refuse replacement of frozen evidence."""
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise ValueError("sealed_drift:" + path.name)
    else:
        atomic_json(path, value)
        path.chmod(0o400)
    return ref(path)


def read_historical(path: Path) -> bytes:
    """Current-source bytes cannot inherit the original producer's authority."""
    if ref(path)["sha256"] != SNAPSHOT_PIN:
        raise ValueError("historical_snapshot")
    return path.read_bytes()


def compare_projection(expected: list[Json], observed: list[Json]) -> None:
    """Every identity, label and original role must match, including row order."""
    keys = ["unit_id", "qid", "group_id", "annotator1", "annotator2", "role"]
    if [{k: r[k] for k in keys} for r in expected] != observed:
        raise ValueError("historical_projection")


def bound(reference: Json, checks: list[Json], references: list[Json], check: str) -> bytes:
    """Report absence separately from a zero operand before decoding dependent fields."""
    path = Path(reference["path"])
    actual = ref(path)
    references.append(actual)
    verdict = gate(
        path,
        "sha256",
        reference["sha256"],
        actual["sha256"] or "missing_file",
        check,
        "original_authority",
    )
    checks.append(verdict)
    if not verdict["passed"]:
        raise ValueError(check)
    return path.read_bytes()


def snapshots(root: Path, raw: Path, checks: list[Json], references: list[Json]) -> Json:
    """Recover the complete imported project closure from one immutable Git tree."""
    package = raw / "snapshot"
    for name in ["carnot", "carnot/reporting"]:
        folder = package / name
        folder.mkdir(parents=True, exist_ok=True)
        initializer = folder / "__init__.py"
        if not initializer.exists():
            initializer.write_bytes(b"")
            initializer.chmod(0o400)
        if initializer.read_bytes() != b"":
            raise ValueError("snapshot_package")
    resolved = {}
    for name, pin in DEPENDENCIES.items():
        path = package / "carnot/reporting" / (name + ".py")
        if not path.exists():
            call = child(
                "git_" + name,
                [
                    "git",
                    "-C",
                    str(root),
                    "show",
                    COMMIT + ":python/carnot/reporting/" + name + ".py",
                ],
                raw / "git",
                deadline=60,
                heartbeat=20,
            )
            if not call["passed"]:
                raise ValueError("historical_git_blob")
            shutil.copyfile(call["stdout_path"], path)
            path.chmod(0o400)
        bound(dict(path=str(path), sha256=pin), checks, references, "historical_dependency_" + name)
        resolved[name] = ref(path)
    read_historical(Path(resolved["human_label_custody_8389"]["path"]))
    return resolved


def replay_mapper(raw: Path, manifest: Json, roles: Json, resolved: Json) -> Json:
    """A fresh isolated interpreter imports only the authenticated historical package."""
    job = seal(
        raw / "replay_job.json",
        dict(snapshot_root=str(raw / "snapshot"), manifest=manifest, roles=roles, modules=resolved),
    )
    argv = [str(ROOT / ".venv/bin/python"), "-I"]
    config = os.environ.get("COVERAGE_RCFILE")
    if config:
        argv += ["-m", "coverage", "run", "--rcfile=" + config]
    argv += [str(ROOT / WORKER), job["path"]]
    call = child(
        "historical_replay",
        ["env", "-u", "PYTHONPATH", "-C", str(raw), *argv],
        raw / "child",
        deadline=120,
        heartbeat=20,
    )
    if not call["passed"]:
        raise ValueError("historical_replay")
    if not (raw / "replay_execution.json").exists():
        seal(raw / "replay_execution.json", call)
    return dict(read_ref(ref(raw / "historical_projection.json")))


def inspect(root: Path, raw: Path) -> Json:
    """Authenticate closure before accepting exposed source inventory or sealing shards."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    progress("authentication_before")
    checks: list[Json] = []
    references: list[Json] = []
    value: Json = dict(
        rows=[],
        question_count=0,
        role_counts={},
        exposure_ledger=[],
        annotator1_observed_count=0,
        annotator2_observed_count=0,
        annotator2_question_count=0,
        historical_producer_snapshot_reference={},
        replay_receipt={},
        label_authority_reference={},
        original_group_partition_reference={},
        source_manifest_reference={},
        shard_a_roster_reference={},
        shard_b_roster_reference={},
    )
    try:
        primaries = {}
        for name, pin in PINS.items():
            path = root / "results" / (name + ".json")
            primaries[name.split("_")[1]] = json.loads(
                bound(
                    dict(path=str(path), sha256="sha256:" + pin),
                    checks,
                    references,
                    "primary_bytes",
                )
            )
        for n, field in [
            ("8403", "release_criterion_ready_score"),
            ("8429", "methods_ready_score"),
            ("8435", "learning_audit_ready_score"),
        ]:
            upstream_path = (
                root / "results" / next(name + ".json" for name in PINS if name.split("_")[1] == n)
            )
            check = gate(
                upstream_path,
                field,
                1,
                primaries[n].get(field, "missing_field"),
                "upstream_" + n,
                n,
            )
            checks.append(check)
            if not check["passed"]:
                raise ValueError(check["check"])
        manifest_ref = primaries["8389"]["manifest_reference"]
        manifest = json.loads(bound(manifest_ref, checks, references, "source_manifest"))
        if (
            manifest["release_commit"] != methods.old.COMMIT
            or manifest["target_mapping"] != methods.old.TARGET
            or set(manifest["release_files"]) != set(methods.old.RELEASE_FILES)
        ):
            raise ValueError("original_manifest_lineage")
        for name, reference in manifest["release_files"].items():
            bound(reference, checks, references, "author_bytes")
            methods.old.authenticate(name, reference)
        for reference in manifest["overlap_inventory"]:
            bound(reference, checks, references, "original_overlap_manifest")
            bound(
                dict(path=reference["source_path"], sha256=reference["sha256"]),
                checks,
                references,
                "original_overlap_source",
            )
        resolved = snapshots(root, raw, checks, references)
        for reference in primaries["8389"]["code_config_hashes"]:
            selected = (
                resolved["human_label_custody_8389"]
                if reference["path"].endswith("/human_label_custody_8389.py")
                else reference
            )
            bound(
                dict(path=selected["path"], sha256=reference["sha256"]),
                checks,
                references,
                "original_code_config",
            )
        roles_ref = dict(
            path=str(root / "openspec/change-proposals/v726-methods-protocol.json"), sha256=ROLE_PIN
        )
        roles = json.loads(bound(roles_ref, checks, references, "original_roles"))
        original_static = json.loads(
            bound(
                primaries["9084"]["human_protocol_reference"],
                checks,
                references,
                "original_static_protocol",
            )
        )
        if original_static != methods.protocol():
            raise ValueError("static_protocol_drift")
        master = list(
            csv.DictReader(
                io.StringIO(
                    methods.old.authenticate(
                        methods.old.MASTER, manifest["release_files"][methods.old.MASTER]
                    ).decode("utf-8-sig")
                )
            )
        )
        reduced = methods.inventory(master, roles["partition"]["roles"])
        historical = replay_mapper(raw, manifest_ref, roles_ref, resolved)
        compare_projection(reduced["rows"], historical["rows"])
        if historical["imported_modules"] != resolved:
            raise ValueError("imported_dependency_closure")
        counts = [
            len(reduced["rows"]),
            reduced["question_count"],
            reduced["annotator1_observed_count"],
            reduced["annotator2_observed_count"],
            reduced["annotator2_question_count"],
            reduced["role_counts"],
        ]
        expected = [900, 300, 900, 300, 100, dict(TRAIN=450, CAL=150, TEST=300)]
        checks.append(gate(raw, "inventory", expected, counts, "complete_original_panel", "LPHB"))
        if counts != expected:
            raise ValueError("complete_original_panel")
        value.update(
            reduced,
            historical_producer_snapshot_reference=dict(
                resolved["human_label_custody_8389"],
                git_commit=COMMIT,
                original_path=PRODUCER,
            ),
            replay_receipt=dict(
                passed=True,
                compared_rows=len(historical["rows"]),
                result_reference=ref(raw / "historical_projection.json"),
                execution_receipt_reference=ref(raw / "replay_execution.json"),
                imported_modules=resolved,
                worker_reference=ref(ROOT / WORKER),
                isolated=True,
            ),
            label_authority_reference=dict(
                commit=methods.old.COMMIT,
                files=manifest["release_files"],
                target_mapping=manifest["target_mapping"],
            ),
            original_group_partition_reference=roles_ref,
            source_manifest_reference=manifest_ref,
            static_protocol=original_static,
        )
        value["original_capacity_null_reference"] = ref(
            root / "results/experiment_8435_v726_capacity_benefit_audit.json"
        )
        value["exposure_ledger"] = [
            dict(
                experiment_id=int(n),
                honest_verdict=p["honest_verdict"],
                verdict_class=p["verdict_class"],
                exposure_scope=p.get("exposure_scope"),
            )
            for n, p in primaries.items()
        ]
        value.update(seal_shards(raw, master, reduced["rows"]))
    except (OSError, ValueError, KeyError, TypeError) as error:
        checks.append(
            gate(root, "authenticated_closure", True, str(error), "source_closure", "LPHB_V726")
        )
    value.update(
        gate_check_summary=checks,
        source_references=references,
        blocker=next((c["check"] for c in checks if not c["passed"]), None),
    )
    progress("authentication_after", len(value["rows"]), 0)
    return value


def seal_shards(raw: Path, master: list[Json], rows: list[Json]) -> Json:
    """Keep all siblings together while separating learner text from evaluator labels."""
    progress("seal_shards_before")
    qids = sorted({r["qid"] for r in rows})
    index = {r["item_id"]: r for r in master}
    labels = seal(
        raw / "evaluator_labels.json",
        dict(
            rows=[
                {
                    k: r[k]
                    for k in ["unit_id", "qid", "group_id", "role", "annotator1", "annotator2"]
                }
                for r in rows
            ]
        ),
    )
    references = dict(evaluator_label_reference=labels)
    for name, groups in [("a", qids[:150]), ("b", qids[150:])]:
        selected = [r for r in rows if r["qid"] in groups]
        features = seal(
            raw / ("features_" + name + ".json"),
            dict(views=[methods.feature_view(index[r["unit_id"]]) for r in selected]),
        )
        roster = seal(
            raw / ("shard_" + name + ".json"),
            dict(
                qids=groups,
                rows=[
                    {
                        k: r[k]
                        for k in ["unit_id", "qid", "group_id", "role", "feature_input_sha256"]
                    }
                    for r in selected
                ],
                feature_reference=features,
                evaluator_label_reference=labels,
            ),
        )
        references["shard_" + name + "_roster_reference"] = roster
    progress("seal_shards_after", len(rows), 0)
    return references
