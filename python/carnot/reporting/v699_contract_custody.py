"""REQ-REPORT-8070: preserve input provenance without converting custody to science.

The two historical branches share sealed identities but qualify separately.
Every reused scalar is tied to its original call, never to failed Exp8059 NLLs.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting import v685_authority_lifecycle as authority
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v698_fixture_consumer_contract import failure, gate_matrix, resolve
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8070_v699_contract_custody"
TASK = "exp8070-contract-custody"
CLI = f"scripts/experiments/{NAME}.py"
DESIGN = "openspec/change-proposals/research-roadmap-vNEXT.md"
MODULE = "python/carnot/reporting/v699_contract_custody.py"
RUNNER = "python/carnot/reporting/v699_custody_execution.py"
TEST = "tests/python/test_contract_custody_8070.py"
MILESTONE = "2026.10.699"
CAPTURE_HASHES = {
    7969: "3f25b2e4b43d50536525e64ace321db55e9d6889aab97cc2581e4665014f2f51",
    7995: "df6eb8b559181455348b1d806f23c36d13c7f49202c2d49b2233aa80edebe02d",
}


class InputFailure(ValueError):
    """External evidence errors retain exact operands rather than look unfinished."""


class Binder:
    """Freeze exact evidence bytes so replay cannot quietly read newer inputs."""

    def __init__(self, raw: Path):
        self.raw = raw
        self.refs: list[Json] = []
        self.failures: list[Json] = []

    def require(self, path: Path, field: str, expected: Any, observed: Any) -> None:
        """Keep zero and absence distinct because they require different remedies."""
        if expected != observed:
            self.failures.append(failure(path, field, expected, observed, TASK))
            raise InputFailure(field)

    def bind(self, path: Path, digest: str | None = None) -> Json:
        """Copy only authenticated bytes; never overwrite a prior snapshot."""
        self.require(path, "resource_exists", True, path.is_file())
        actual = sha256_file(path)
        self.require(path, "sha256", digest or actual, actual)
        frozen = authority._snapshot(path, path.read_bytes(), self.raw, "input")
        ref = dict(path=str(path), sha256=actual, snapshot_path=frozen["snapshot_path"])
        if ref not in self.refs:
            self.refs.append(ref)
        return ref

    def read(self, path: Path, digest: str | None = None) -> Json:
        """Read the saved copy so later path changes cannot affect this observation."""
        return dict(json.loads(Path(self.bind(path, digest)["snapshot_path"]).read_text()))

    def terminal(self, path: Path) -> Json:
        """Readiness integers alone cannot authenticate a completed publication."""
        try:
            value = json.loads(path.read_text())
            terminal = Path(value["terminal_validation_sidecar_path"])
            binding = self.read(terminal)
            binding = binding.get("publication", binding)
            self.require(
                terminal, "primary_sha256", sha256_file(path), binding.get("primary_sha256")
            )
            sidecar = Path(
                binding.get("sidecar_path", binding.get("validator", "/absent_validator"))
            )
            report = read_bound_sidecar(path, sidecar)
            self.require(sidecar, "report.passed", True, report["report"].get("passed"))
            self.require(path, "flagged_adversarial", False, value.get("flagged_adversarial"))
            self.bind(sidecar)
        except (KeyError, OSError, ValueError) as error:
            self.failures.append(failure(path, "clean_terminal", True, str(error), TASK))
            raise InputFailure("clean_terminal") from error
        return dict(passed=True, primary_sha256=sha256_file(path))


def assess(design: Path, staged: Path, active: Path, snapshots: Path) -> Json:
    """Use the existing reader and separately check the complete embedded task list."""
    try:
        value = authority.assess_authorities(
            design, staged, active, snapshots, milestone=MILESTONE, first_id=8070, count=13
        )
        _, tasks = parse_design(design.read_text(), milestone=MILESTONE)
        digest = authority.tasks_digest(tasks)
        if digest != value["canonical_tasks_sha256"]:
            value["gate_check_summary"].append(
                failure(design, "design_tasks_sha256", value["canonical_tasks_sha256"], digest)
            )
        value["gate_check_summary"] = [
            failure(
                Path(r.get("artifact_path", r.get("path"))),
                r.get("artifact_field", r.get("field")),
                r["expected"],
                r["observed"],
                "V699_authority",
            )
            for r in value["gate_check_summary"]
        ]
        value["activated"] = not value["gate_check_summary"]
        return value
    except (OSError, ValueError, KeyError, TypeError, IndexError, yaml.YAMLError) as error:
        return dict(
            activated=False,
            contract_rows=[],
            canonical_tasks_sha256=None,
            authority_snapshots={},
            gate_check_summary=[failure(design, "authority_readable", True, str(error))],
        )


def qualify(root: Path, raw: Path) -> Json:
    """Qualify each branch independently; historical model hashes stay historical."""
    binder = Binder(raw)
    result: Json = dict(
        source_ready=False,
        learning_ready=False,
        failures=binder.failures,
        refs=binder.refs,
        feature_rows=[],
        qualified_head={},
        head_sha256=None,
        historical_model_receipts=[],
    )
    try:
        sealed_path = root / "results/experiment_8058_v698_sealed_evidence_methods.json"
        sealed = binder.read(sealed_path)
        binder.terminal(sealed_path)
        binder.require(
            sealed_path, "task_id", "exp8058-sealed-evidence-methods", sealed.get("task_id")
        )
        binder.require(
            sealed_path, "eligible_class", True, sealed.get("verdict_class") in {"positive", "null"}
        )
        refs = {Path(r["path"]).name: r for r in sealed["source_artifact_hashes"]}
        target_ref = refs["experiment_8019_v695_eligible_targets.json"]
        targets_path = Path(target_ref["path"])
        targets = binder.read(targets_path, target_ref["sha256"])
        binder.terminal(targets_path)
    except (InputFailure, OSError, KeyError, ValueError) as error:
        if not binder.failures:
            binder.failures.append(failure(sealed_path, "sealed_inputs_readable", True, str(error)))
        return result
    readiness = {"source": True, "learning": True}
    try:
        parent_ref = refs["experiment_8046_v697_branch_protocols.json"]
        parent = binder.read(Path(parent_ref["path"]), parent_ref["sha256"])
        binder.terminal(Path(parent_ref["path"]))
        binder.require(
            sealed_path,
            "qualified_head_sha256",
            sealed["qualified_head_sha256"],
            canonical_hash(sealed["qualified_head"]),
        )
        binder.require(
            sealed_path,
            "qualified_head_original",
            parent["qualified_head"],
            sealed["qualified_head"],
        )
        binder.require(
            sealed_path,
            "qualified_head_finite_converged",
            True,
            sealed["qualified_head"].get("finite") is True
            and sealed["qualified_head"].get("converged") is True,
        )
        result.update(
            qualified_head=sealed["qualified_head"], head_sha256=sealed["qualified_head_sha256"]
        )
    except (InputFailure, KeyError, ValueError, OSError) as error:
        readiness["learning"] = False
        binder.failures.append(
            failure(sealed_path, "learning_head_authentication", True, str(error))
        )
    custody = {r["sha256"]: r for r in targets["cited_upstream_artifacts"]}
    for branch, roles, producer in [
        ("source", ["fit", "tune"], 7969),
        ("source", ["evaluation"], 7995),
        ("learning", ["stream", "retention"], 7995),
    ]:
        try:
            capture_ref = custody["sha256:" + CAPTURE_HASHES[producer]]
            capture = binder.read(Path(capture_ref["path"]), capture_ref["sha256"])
            binder.require(
                Path(capture_ref["path"]),
                "capture_producer",
                producer,
                capture.get("experiment_id"),
            )
            binder.require(
                Path(capture_ref["path"]),
                "capture_qualified",
                True,
                capture.get("verdict_class") in {"positive", "null"}
                and capture.get("flagged_adversarial") is False,
            )
            original = (
                root
                / "results"
                / (
                    "experiment_7969_v691_qwen_calibration_capture.json"
                    if producer == 7969
                    else "experiment_7995_v693_qwen_development_capture.json"
                )
            )
            binder.bind(original, capture_ref["sha256"])
            binder.terminal(original)
            result["historical_model_receipts"].append(
                dict(
                    producer=producer,
                    original_primary_sha256=capture_ref["sha256"],
                    provenance={
                        k: capture.get(k)
                        for k in [
                            "MODEL_SPECS",
                            "model_specs",
                            "model_identity_receipt",
                            "gguf_sha256",
                            "model_revision",
                            "code_config_hashes",
                            "runtime_receipts",
                            "capture_identity",
                        ]
                    },
                )
            )
            calls = {
                r["family_id"]: (r, ref)
                for r, ref in zip(capture["rows"], capture["raw_response_shards"], strict=True)
            }
            public: dict[str, Json] = {}
            public_roles = (
                ["fit", "tune"] if producer == 7969 else ["stream"] if branch == "source" else roles
            )
            for role in public_roles:
                ref = targets["public_manifests"][role]
                for row in binder.read(Path(ref["path"]), ref["sha256"])["rows"]:
                    binder.require(
                        Path(ref["path"]), "unique_family", False, row["family_id"] in public
                    )
                    public[row["family_id"]] = row
            for role in roles:
                print(
                    f"[exp8070] role={role} before authentication monotonic_s={time.monotonic():.3f}",
                    flush=True,
                )
                ref = sealed["role_manifests"][role]
                path = Path(ref["path"])
                rows = binder.read(path, ref["sha256"])["rows"]
                expected = dict(fit=64, tune=32, evaluation=96, stream=256, retention=64)[role]
                binder.require(
                    path, "original_slots", list(range(expected)), [r["slot"] for r in rows]
                )
                for index, row in enumerate(rows):
                    if index % 32 == 0:
                        print(
                            f"[exp8070] role={role} completed={index} pending={len(rows) - index}",
                            flush=True,
                        )
                    pub = public[row["family_id"]]
                    call, call_ref = calls[row["family_id"]]
                    saved_ref = custody.get(call_ref["sha256"], call_ref)
                    saved = binder.read(Path(saved_ref["path"]), call_ref["sha256"])
                    binder.require(Path(saved_ref["path"]), "raw_call_equals_recorded", call, saved)
                    for key in ("source_bytes", "answer_bytes"):
                        binder.require(path, key, pub[key], row[key])
                    binder.require(
                        path, "capture_identity", call["capture_identity"], pub["capture_identity"]
                    )
                    if pub["public_eligible"]:
                        binder.require(
                            path, "original_qwen_scalar", call["parsed"]["probability"], pub["q"]
                        )
                    if "features" in row:
                        binder.require(path, "original_features", pub["features"], row["features"])
                        binder.require(path, "original_q", pub["q"], row["q"])
                    result["feature_rows"].append(
                        dict(
                            unit=f"{role}/{row['slot']}",
                            role=role,
                            slot=row["slot"],
                            source=row["source_cluster_id"],
                            family_id=row["family_id"],
                            arm="historical_features",
                            seed=None,
                            condition=branch,
                            features=pub["features"],
                            q=pub["q"],
                            capture_identity=pub["capture_identity"],
                            capture_producer=producer,
                            raw_call_sha256=call_ref["sha256"],
                            numerator=int(pub["public_eligible"]),
                            denominator=1,
                            status="completed" if pub["public_eligible"] else "excluded",
                            exclusion_reason=pub["exclusion_reason"],
                        )
                    )
                print(
                    f"[exp8070] role={role} after authentication completed={len(rows)} pending=0",
                    flush=True,
                )
        except (InputFailure, KeyError, ValueError, OSError) as error:
            readiness[branch] = False
            binder.failures.append(
                dict(
                    failure(sealed_path, branch + "_authentication", True, str(error)),
                    branch=branch,
                )
            )
    result.update(source_ready=readiness["source"], learning_ready=readiness["learning"])
    return result


def historical(root: Path, raw: Path) -> Json:
    """Reconstruct all thirteen historical dispositions without inventing primaries."""
    binder = Binder(raw)
    capstone_path = root / "results/experiment_8069_v698_capstone.json"
    capstone = binder.read(capstone_path)
    binder.terminal(capstone_path)
    prior = binder.read(root / "results/experiment_8057_v698_fixture_consumer_contract.json")
    binder.terminal(root / "results/experiment_8057_v698_fixture_consumer_contract.json")
    snapshots = prior["authority_snapshots"]
    for ref in snapshots.values():
        if ref["exists"]:
            binder.bind(Path(ref["snapshot_path"]), ref["sha256"])
    _, tasks = parse_design(
        Path(snapshots["design"]["snapshot_path"]).read_text(), milestone="2026.10.698"
    )
    active = yaml.safe_load(Path(snapshots["active"]["snapshot_path"]).read_text())
    binder.require(capstone_path, "historical_complete_tasks", tasks, active["tasks"])
    binder.require(
        capstone_path,
        "historical_tasks_sha256",
        prior["canonical_tasks_sha256"],
        authority.tasks_digest(tasks),
    )
    old_rows = {r["task_id"]: r for r in capstone["task_dispositions"]}
    rows = []
    for task in tasks:
        declared = root / task["deliverable"]
        old = old_rows[task["id"]]
        path = declared
        kind = "primary"
        if declared.is_file():
            path = resolve(root, task)
            value = binder.read(path)
            binder.terminal(path)
            if task["id"] != "exp8069-capstone":
                binder.require(path, "historical_primary_sha256", old["sha256"], sha256_file(path))
        elif task["id"] == "exp8060-source-energy-training":
            path = root / "results/experiment_8060_source_energy_training.json"
            value = binder.read(path, old["sha256"])
            binder.require(path, "skip_schema", "blocked_gate_check_v1", value.get("schema"))
            binder.require(path, "skip_experiment", 8060, value.get("experiment"))
            binder.bind(Path(value["failed_evidence_path"]), value["failed_evidence_sha256"])
            kind = "actual_skip_receipt"
        else:
            binder.require(declared, "historical_primary_present", False, old["primary_present"])
            value = {}
            kind = "absent_primary"
        rows.append(
            dict(
                task_id=task["id"],
                declared_path=str(declared),
                path=str(path),
                sha256=sha256_file(path) if path.is_file() else None,
                primary_present=declared.is_file(),
                evidence_kind=kind,
                verdict_class=value.get("verdict_class", old["verdict_class"]),
                honest_verdict=value.get("honest_verdict", old["honest_verdict"]),
                original_disposition=old,
                censored_count=value.get("censored_count", 0),
                producer_gate_failures=value.get("gate_check_summary", []),
                no_retry_unchanged_outcome=True,
            )
        )
    return dict(
        rows=rows, refs=binder.refs, authority_snapshots=snapshots, failures=binder.failures
    )
