"""REQ-REPORT-8368: qualify custody before comparing the real CUDA substrate."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import v718_contract_replay as contract
from carnot.reporting import v721_contract_methods as current_contract
from carnot.reporting import v718_replay_history as history
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, sha256_file
from carnot.reporting.evidence_features_custody_7980 import reference as reference
from carnot.reporting.v710_contract_replay import snapshot, require_reference
from carnot.verify import runtime_reader_8340 as legacy
from carnot.verify import runtime_reader_8353 as prior
from carnot.verify import runtime_reader_report_8340 as report
from carnot.verify import typed_runtime_8368 as typed

Json = dict[str, Any]
ROOT, UPSTREAM, old = legacy.ROOT, legacy.UPSTREAM, legacy.old
NAME, TASK, MILESTONE = (
    "experiment_8368_v721_typed_runtime_closure",
    "exp8368-typed-runtime-closure",
    "2026.10.721",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_runtime_closure_8368.py"
OWNED = [
    "python/carnot/verify/typed_runtime_8368.py",
    "python/carnot/verify/runtime_closure_8368.py",
    "python/carnot/verify/runtime_closure_execution_8368.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
EXPERIMENT_ID = 8368
checksum, parse_design, changes, reduce = (
    legacy.checksum,
    legacy.parse_design,
    legacy.changes,
    legacy.reduce,
)
PINS = {
    "results/experiment_8353_v720_runtime_reader_qualification.json": "sha256:0f50c66b383bdaa2f533407886ad9f908097907625680cbd9322c7eb7a96a36d",
    UPSTREAM: "sha256:dc64000d6025f1339a4130c8527e83419861dfef1d0dea548f8e85bacd647fbd",
    "results/experiment_8340_v719_runtime_reader_qualification.json": "sha256:50889d14b946affdc8476a55fd851fa88faead1c3ad2fd62067dc0566ca06bd2",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed phase counts distinguish completed work from a supervised wait."""
    print(f"[exp8368] phase={phase} completed={completed} pending={pending}", flush=True)


def design(root: Path, milestone: str) -> Path:
    """Current and preserved plans have distinct filenames, preventing historical drift."""
    return root / (
        "openspec/change-proposals/research-roadmap-vNEXT.md"
        if milestone == MILESTONE
        else history.PRIOR_DESIGN
        if milestone == "2026.10.717"
        else "openspec/change-proposals/research-roadmap-v716-preserved-20261008.md"
    )


def private_authority(root: Path, milestone: str) -> Path:
    """Reuse the full-task private control without editing activated authority."""
    with patch.object(prior, "design", design):
        return prior.private_authority(root, milestone)


def authority(root: Path, raw: Path, milestone: str = MILESTONE) -> Json:
    """Full task objects, including prompts, must match the selected versioned plan."""
    try:
        if milestone == MILESTONE:
            return dict(current_contract.authority(root, raw, milestone))
        with patch.object(history, "design", design):
            return dict(contract.authority(root, raw, milestone))
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(
            activated=False,
            refs=[],
            gate_check_summary=[
                old.gate(root / "research-roadmap.yaml", "execution_authority", True, str(error))
            ],
        )


@contextmanager
def bindings() -> Iterator[None]:
    """Version adapters preserve existing runtime thresholds and child supervision."""
    with ExitStack() as stack:
        for name, value in dict(
            NAME=NAME,
            TASK=TASK,
            MILESTONE=MILESTONE,
            CLI=CLI,
            TEST=TEST,
            OWNED=OWNED,
            EXPERIMENT_ID=8368,
            design=design,
            authority=authority,
            progress=progress,
        ).items():
            stack.enter_context(patch.object(legacy, name, value))
        yield


def measure(root: Path, raw: Path, *, reader_checks_passed: bool = True) -> Json:
    """Authenticate historical operands first; only qualified readers may authorize a probe."""
    progress("historical_closure_before", 0, len(PINS))
    failures, retained = [], []
    retained = [
        snapshot(root / name, raw / "historical_primaries", Path(name).stem)
        for name in PINS
        if (root / name).is_file()
    ]
    try:
        retained.extend(typed.closure(root, raw, list(PINS)))
        for name, pin in PINS.items():
            if (root / name).is_file() and sha256_file(root / name) != pin:
                raise ValueError("historical_primary_substitution:" + name)
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(old.gate(root / UPSTREAM, "historical_source_closure", True, str(error)))
    progress("historical_closure_after", len(retained), 0)
    with bindings():
        work = legacy.measure(root, raw, authorize_probe=reader_checks_passed and not failures)
    work["checks"].extend(failures)
    work["historical_fixture_manifest"] = retained
    work["root"] = str(root)
    work["refs"].extend(retained)
    work["first_failed_operand"] = failures[0] if failures else None
    work["historical_dispositions"] = [
        dict(
            path=str(root / name),
            sha256=pin,
            honest_verdict=json.loads((root / name).read_bytes())["honest_verdict"],
            verdict_class=json.loads((root / name).read_bytes())["verdict_class"],
        )
        for name, pin in PINS.items()
        if (root / name).is_file()
    ]
    work["development_validation_history"] = [
        snapshot(path, raw / "development_history", path.parent.name + "_" + path.stem)
        for path in (ROOT / "results/raw" / NAME).glob("development*/*.receipt.json")
    ]
    work["refs"].extend(work["development_validation_history"])
    for path in (
        ROOT / "results/raw" / NAME / "development/existing_exclusion_source.json",
        ROOT / "results/raw" / NAME / "recovered_source/recovery_manifest.json",
    ):
        if path.is_file():
            work["refs"].append(snapshot(path, raw / "reader_provenance", path.stem))
    return work


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Reader qualification, causal change and actual context completion remain separate scores."""
    with patch.object(report, "q", sys.modules[__name__]):
        value = report.build(work, raw, receipts)
    value.update(
        run_date="20261010",
        random_seed=7218368,
        historical_fixture_manifest=work["historical_fixture_manifest"],
        typed_reference_rows=work["historical_fixture_manifest"],
        historical_dispositions=work["historical_dispositions"],
        first_failed_operand=work["first_failed_operand"],
        runtime_binding_path=work["current_reference"]["path"],
        runtime_binding_sha256=work["current_reference"]["sha256"],
        current_probe_count=len(work["diagnostic"]["rows"]),
        development_validation_history=work["development_validation_history"],
    )
    value["field_principles"].update(
        {
            k: "Bind typed runtime custody and preserve original failure dispositions; reader success is separate from CUDA execution."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def check_receipt(receipt: Json, work: Json) -> bool:
    """A reported pass must agree with the owned child's durable exit receipt.

    The two arithmetic finding controls use the existing typed consumer, whose
    proof is recomputed here rather than trusting a changed pass flag.
    """
    recorded = json.loads(Path(receipt["stdout_path"]).with_suffix(".receipt.json").read_bytes())
    names = {"legitimate_info_qualified": 0, "false_zero_rejected": 1}
    if receipt["name"] not in names:
        return bool(
            receipt == recorded
            and receipt["passed"]
            == (
                receipt["exit_code"] == receipt["expected_exit"]
                and receipt["normal_exit"]
                and not receipt["timed_out"]
            )
        )
    index = names[receipt["name"]]
    if any(
        receipt[k] != recorded[k] for k in recorded if k not in {"name", "passed", "expected_exit"}
    ):
        return False
    from carnot.verify.local_update_isolation_8306 import design as basis, scalar_design

    x = [0.0, 0.0, 1.0, 0.0, 1.0]
    direct, independent = basis(x), scalar_design(x)
    deliberate_error_rejected = abs(direct[2] + 0.125 - independent[2]) > max(
        abs(a - b) for a, b in zip(direct, independent, strict=True)
    )
    direct[2] += 0.125 * index
    error = max(abs(a - b) for a, b in zip(direct, independent, strict=True))
    raw_report = json.loads(Path(receipt["stdout_path"]).read_bytes())
    candidate = Path(raw_report["reports"][0]["artifact"])
    report_value = dict(
        raw_report, candidate_sha256=sha256_file(candidate), verifier_sha256=history.verifier_hash()
    )
    actual = history.consume(
        report_value,
        candidate,
        recorded["exit_code"],
        dict(
            recomputed=error == json.loads(candidate.read_bytes())["dense_sparse_error_max"],
            deliberate_error_rejected=deliberate_error_rejected,
        ),
    )
    return bool(
        receipt["passed"]
        == (actual["passed"] if index == 0 else bool(not actual["passed"] and actual["findings"]))
    )


def replay(path: Path) -> bool:
    """Cold replay binds pinned historical operands and independently reduces current scores.

    Rehashing a substituted reference does not change the original producer pins.
    Recorded inventory streams and child exits remain required primitive evidence.
    """
    try:
        value = json.loads(path.read_bytes())
        if value.get("reproducibility_checksum") != checksum(value) or [
            value.get(k) for k in ("experiment_id", "task_id", "milestone", "run_date")
        ] != [8368, TASK, MILESTONE, "20261010"]:
            return False
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            require_reference(ref)
        work = json.loads(typed.authenticate(value["primitive_reference"]).read_bytes())
        baseline_required = bool(
            work["previous"] or work["historical"] or value["runtime_reader_ready_score"]
        )
        if baseline_required and not (
            work.get("baseline_reference") and work.get("baseline_source_reference")
        ):
            return False
        for ref in work["historical_fixture_manifest"]:
            typed.authenticate(ref)
        if work.get("baseline_reference") and work["root"] != str(
            Path(work["baseline_source_reference"]["path"]).parents[1]
        ):
            return False
        dispositions = []
        for name, pin in PINS.items():
            original = next(
                (
                    r
                    for r in work["historical_fixture_manifest"]
                    if r["path"] == str(Path(work["root"]) / name)
                ),
                None,
            )
            if work.get("baseline_reference") and original is None:
                return False
            if original and original["sha256"] != pin:
                return False
            if original:
                source_value = json.loads(typed.authenticate(original).read_bytes())
                dispositions.append(
                    dict(
                        path=original["path"],
                        sha256=pin,
                        honest_verdict=source_value["honest_verdict"],
                        verdict_class=source_value["verdict_class"],
                    )
                )
        if dispositions != work["historical_dispositions"]:
            return False
        if work.get("baseline_reference"):
            baseline = json.loads(typed.authenticate(work["baseline_reference"]).read_bytes())
            source = json.loads(typed.authenticate(work["baseline_source_reference"]).read_bytes())
            if (
                baseline != work["previous"]
                or work["baseline_reference"]["sha256"] != source["runtime_binding_sha256"]
                or work["baseline_source_reference"]["sha256"] != PINS[UPSTREAM]
            ):
                return False
        if (
            json.loads(typed.authenticate(work["current_reference"]).read_bytes())
            != work["current"]
        ):
            return False
        auth = work.get("authority", {})
        if auth.get("activated"):
            active = auth["authority_snapshots"]["active"]
            design_ref = auth["authority_snapshots"]["design"]
            import yaml  # type: ignore[import-untyped]

            actual = yaml.safe_load(
                typed.authenticate(dict(active, path=active["source_path"])).read_bytes()
            )
            tasks = parse_design(
                typed.authenticate(dict(design_ref, path=design_ref["source_path"])).read_text(),
                milestone=MILESTONE,
            )[1]
            if (
                actual["tasks"] != auth["tasks"]
                or actual["tasks"] != tasks
                or actual["milestone"] != MILESTONE
            ):
                return False
        for receipt in value["validation_receipts"] + work["observation_receipts"]:
            if not check_receipt(receipt, work):
                return False
            for stream in ("stdout", "stderr"):
                typed.authenticate(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                )
        if work["observation_receipts"]:
            text = Path(work["observation_receipts"][0]["stdout_path"]).read_text()
            devices = [
                dict(
                    zip(
                        ("index", "uuid", "name", "driver_version"),
                        map(str.strip, cols),
                        strict=True,
                    )
                )
                for line in text.splitlines()
                if len(cols := line.split(",")) == 4
            ]
            if devices != work["current"].get("devices"):
                return False
        for row in work["diagnostic"]["rows"]:
            receipt = row["receipt"]
            for stream in ("stdout", "stderr"):
                typed.authenticate(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                )
            if (
                receipt["passed"]
                and json.loads(Path(receipt["stdout_path"]).read_text().splitlines()[-1])
                != row["primitive"]
            ):
                return False
        passed = bool(value["validation_receipts"]) and all(
            r["passed"]
            for r in value["validation_receipts"]
            if r.get("scope") != "external_preconditions"
        )
        return (
            value["required_checks_passed"] == passed
            and all(value[k] == v for k, v in reduce(work, passed).items())
            and value["typed_reference_rows"] == work["historical_fixture_manifest"]
            and value["historical_dispositions"] == work["historical_dispositions"]
            and value["first_failed_operand"] == work["first_failed_operand"]
            and value["runtime_binding_path"] == work["current_reference"]["path"]
            and value["runtime_binding_sha256"] == work["current_reference"]["sha256"]
            and value["current_probe_count"] == len(work["diagnostic"]["rows"])
            and not value["MODEL_SPECS"]
            and value["model_invocation_counts"] == ZERO_INVOCATION_COUNTS
            and value["independent_generalization_score"]
            == value["generalized_learning_benefit_score"]
            == 0
            and value["inference_substrate_class"] == "no_model_load"
            and value["no_model_load"] is True
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
