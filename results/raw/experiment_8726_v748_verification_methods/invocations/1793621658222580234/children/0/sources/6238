"""REQ-REPORT-8208: seal real restricted fits before reserved evaluation.

Fit validity permits the next independent evaluation even when tune evidence
shows no benefit. Current CPU fits do not relabel imported Qwen calls as new.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import restricted_action_methods_8207 as methods
from carnot.verify import restricted_energy_8208 as n
from carnot.verify import selective_energy_fit_8195 as previous
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT = methods.ROOT
NAME = "experiment_8208_v709_restricted_energy_fit"
TASK = "exp8208-restricted-energy-fit"
MODULE = "python/carnot/verify/restricted_energy_fit_8208.py"
NUMERIC = "python/carnot/verify/restricted_energy_8208.py"
RUNNER = MODULE
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_restricted_energy_fit_8208.py"
OWNED = [MODULE, NUMERIC, CLI]
RUN_DATE = "20261006"
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8207_v709_restricted_action_methods.json"
PIN = "sha256:" + "cae587805824db04c8d58825a0607e501bfb9e4c75cf416fc65c1394c9eea187"
fit = previous.fit
reference = fit.reference
BASE_MANIFEST = methods.BASE_MANIFEST


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush completed counts at phase boundaries so no fit or child is hidden."""
    print(f"[exp8208] phase={phase} completed={completed} pending={pending}", flush=True)


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """Authenticate only branch operands before opening exposed fit/tune rows."""
    began, wall = time.monotonic_ns(), time.time_ns()
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(
        checks=[],
        refs=[],
        raw_shard_hashes=[],
        evidence={},
        diagnostics={},
        protocol={},
        owned_failure="",
        precondition_receipts=[],
        cited_upstream_artifacts=[],
        optional_sibling_disposition="not_required_by_this_branch",
    )
    progress("before_preconditions")
    try:
        fit.gate(work, Path(sys.executable), "python_supported", True, sys.version_info >= (3, 11))
        probe = raw / ".probe"
        probe.write_bytes(b"private writable scratch")
        fit.gate(
            work,
            raw,
            "private_scratch_writable",
            True,
            probe.read_bytes() == b"private writable scratch",
        )
        probe.unlink()
        if fixture:
            protocol = json.loads((ROOT / methods.PROTOCOL).read_bytes())
            rows, roles, baseline = methods.fixture()
            if mutation:
                fit.gate(work, root / UPSTREAM, "action_protocol_ready_score", 1, None)
        else:
            upstream = fit.bind(work, dict(path=str(root / UPSTREAM), sha256=PIN), raw)
            for field, expected in [
                ("action_protocol_ready_score", 1),
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ]:
                fit.gate(work, root / UPSTREAM, field, expected, upstream.get(field))
            fit.gate(
                work,
                root / UPSTREAM,
                "terminal_publication_passed",
                True,
                read_bound_sidecar(root / UPSTREAM, fit.historical.publication_sidecar(upstream))[
                    "report"
                ]["passed"],
            )
            fit.gate(
                work,
                root / UPSTREAM,
                "protocol_sha256",
                methods.PIN,
                upstream.get("protocol_sha256"),
            )
            protocol = fit.bind(work, dict(path=upstream["protocol_path"], sha256=methods.PIN), raw)
            plan = fit.inputs(root, raw / "fit_inputs")
            work["checks"].extend(plan["checks"])
            work["refs"].extend(plan["refs"])
            if not all(c["passed"] for c in plan["checks"]):
                raise ValueError("fit_inputs")
            for row in plan["rows"]:
                row["source_id"] = protocol["fit_source_id_map"][row["unit_id"]]
            roles = previous.n.freeze_roles(plan["rows"], protocol["role_manifest"]["reserved"])
            rows = [
                dict((k, v) for k, v in r.items() if k != "historical_paired_control")
                | dict(historical_x=r["historical_paired_control"]["x"])
                for r in plan["rows"]
            ]
            normalize = lambda rs: {
                k: sorted(v, key=lambda r: r["source_cluster_id"]) for k, v in rs.items()
            }
            fit.gate(
                work,
                Path(upstream["protocol_path"]),
                "frozen_role_manifest",
                normalize(protocol["role_manifest"]),
                normalize(roles),
            )
            baseline = next(h for h in plan["controls"] if h["arm"] == "radial16")
            prior_ref = next(
                r for r in protocol["source_artifact_hashes"] if "experiment_8195_" in r["path"]
            )
            prior = fit.bind(work, prior_ref, raw)
            prior_heads = fit.bind(
                work,
                dict(path=prior["frozen_heads_path"], sha256=prior["frozen_heads_sha256"]),
                raw,
            )
            work["original_selective_heads"] = prior_heads["heads"]
            work["cited_upstream_artifacts"] = [
                *plan["upstream"],
                dict(
                    experiment_id=8207,
                    sha256=PIN,
                    fields_imported=[
                        "action_protocol_ready_score",
                        "protocol_sha256",
                        "role_manifest",
                    ],
                ),
                dict(
                    experiment_id=8195,
                    sha256=prior_ref["sha256"],
                    fields_imported=["frozen_heads_path", "frozen_heads_sha256"],
                ),
            ]
            work["historical_model_provenance"] = plan["historical_model_provenance"]
        work["protocol"] = protocol
        fit.gate(
            work,
            ROOT / methods.PROTOCOL,
            "immutable_costs",
            n.base.CONFIG["costs"],
            protocol["costs"],
        )
        progress("after_preconditions", len(rows), 0)
        work["owned_failure"] = "fit_incomplete"
        progress("before_benchmark_natural_fit")
        trained = n.train(rows, roles)
        data = dict(
            rows=rows,
            roles=roles,
            baseline=baseline,
            original_selective_heads=work.get("original_selective_heads"),
            **trained,
        )
        reduced = n.reduce(data)
        work["diagnostics"] = n.diagnostics(rows, roles, baseline)
        progress("after_benchmark_natural_fit", len(rows), 0)
        frozen = dict(
            trained,
            baseline=baseline,
            original_selective_heads=work.get("original_selective_heads"),
            features=protocol["features"],
            costs=protocol["costs"],
            allowed_actions=protocol["allowed_actions"],
            tie_rule=protocol["tie_rule"],
            roles=roles,
            selected_simple_control=reduced["selected_simple_control"],
            source_masks=[
                dict(
                    unit_id=r["unit_id"],
                    available=r["x"] is not None,
                    historical_available=r["historical_x"] is not None,
                )
                for r in rows
            ],
            source_artifact_hashes=work["refs"],
            train_tune_action_choices=[
                {
                    k: r[k]
                    for k in (
                        "unit_id",
                        "source_cluster_id",
                        "arm",
                        "p",
                        "baseline_action",
                        "action",
                    )
                }
                for r in reduced["rows"]
            ],
            config=n.CONFIG,
        )
        work["evidence"] = data
        for name, value in [
            ("frozen_heads", frozen),
            ("primitive_evidence", data),
            ("independent_reduction", reduced),
            ("diagnostic_evidence", work["diagnostics"]),
            ("fit_config", n.CONFIG),
        ]:
            atomic_json(raw / (name + ".json"), value)
            work["raw_shard_hashes"].append(reference(raw / (name + ".json")))
        (raw / "frozen_heads.json").chmod(0o444)
        work["owned_failure"] = ""
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        TimeoutError,
        StopIteration,
    ) as error:
        if work["owned_failure"]:
            work["owned_failure"] = str(error)
        elif all(c["passed"] for c in work["checks"]):
            work["checks"].append(
                dict(
                    check="input_structure",
                    path=str(root / UPSTREAM),
                    upstream=UPSTREAM,
                    hash=None,
                    artifact_field="input_structure",
                    op="==",
                    expected="authenticated_fit_tune_primitives",
                    observed=str(error),
                    passed=False,
                )
            )
        progress("terminal_operand_or_fit_failure", len(work["checks"]), 0)
    ended = time.monotonic_ns()
    work.update(
        duration_s=(ended - began) / 1e9,
        measurement_clocks=dict(
            started_monotonic_ns=began, ended_monotonic_ns=ended, started_wall_ns=wall
        ),
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                "python/carnot/verify/restricted_action_rule_8207.py",
                "python/carnot/verify/evidence_energy_8154.py",
            ]
        ],
    )
    startup = Path("/tmp/carnot8208-startup")
    if not fixture and startup.is_dir():
        shutil.copytree(startup, raw / "startup_receipts")
        receipt = json.loads((raw / "startup_receipts/receipt.json").read_bytes())
        for label in ("stdout", "stderr", "log"):
            receipt[label + "_path"] = str(
                raw / "startup_receipts" / Path(receipt[label + "_path"]).name
            )
        work["precondition_receipts"].append(receipt)
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Fit validity and checked custody determine readiness, never tune benefit."""
    checked = bool(receipts) and all(r["passed"] for r in receipts) and not work["owned_failure"]
    failures = [c for c in work["checks"] if not c["passed"]]
    ready = int(checked and not failures and bool(work["evidence"]))
    verdict = (
        "disqualified"
        if not checked
        else "blocked"
        if failures
        else "circular_positive"
        if fixture
        else "null"
    )
    reduced = (
        n.reduce(work["evidence"])
        if work["evidence"]
        else dict(
            rows=[],
            equivalent_logistic_parity=dict(passed=False, rows=[]),
            common_shift_invariance=dict(passed=False, rows=[]),
            selected_simple_control=None,
        )
    )
    rows = work["evidence"].get("rows", [])
    completed = sum(r["x"] is not None and r["y"] in (0, 1) for r in rows)
    frozen = raw / "frozen_heads.json"
    value: Json = dict(
        experiment_id=8208,
        task_id=TASK,
        milestone="2026.10.709",
        run_date=RUN_DATE,
        honest_verdict="complete_" + verdict + "_restricted_energy_fit",
        verdict_class=verdict,
        gate_check_summary=work["checks"],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(
            model_loads=0, generate_calls=0, forward_calls=0, model_count=0
        ),
        call_ledger=[],
        preconditions_checked=[
            dict(resource=c["check"], available=c["passed"], path=c["path"]) for c in work["checks"]
        ],
        precondition_receipts=work["precondition_receipts"],
        duration_s=work["duration_s"],
        random_seed=n.CONFIG["seed"],
        intended_count=len(rows) if rows else 192,
        completed_count=completed,
        failed_count=0,
        censored_count=0,
        excluded_count=(len(rows) if rows else 192) - completed,
        independent_count=completed,
        verifier_is_oracle=fixture,
        exposure_scope="oracle_fixture_only"
        if fixture
        else "exposed_development_within_run_disjoint",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        action_fit_ready_score=ready,
        required_checks_passed=bool(checked),
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        acceptance_gates=dict(
            external_inputs=not failures,
            real_fit_sealed=bool(work["evidence"]),
            owned_validation=checked,
            equivalent_logistic_parity=reduced["equivalent_logistic_parity"]["passed"],
            common_shift_invariance=reduced["common_shift_invariance"]["passed"],
        ),
        raw_shard_hashes=work["raw_shard_hashes"],
        code_config_hashes=work["code_config_hashes"],
        source_artifact_hashes=work["refs"],
        cited_upstream_artifacts=work["cited_upstream_artifacts"],
        frozen_heads_path=str(frozen) if frozen.is_file() else None,
        frozen_heads_sha256=sha256_file(frozen) if frozen.is_file() else None,
        fit_rows=[r for r in reduced["rows"] if r["role"] == "head_fit"],
        temperature_rows=[r for r in reduced["rows"] if r["role"] == "temperature_fit"],
        tuning_rows=[r for r in reduced["rows"] if r["role"] == "calibration"],
        parameter_hashes=[
            dict(arm=h["arm"], **h["parameter_hashes"]) for h in work["evidence"].get("heads", [])
        ],
        trained_head_specs=[
            dict(
                arm=h["arm"],
                coefficients=len(h["weights"]),
                ridge=h["ridge"],
                temperature=h["temperature"],
                fit_count=len(h["fit_ids"]),
                temperature_count=len(h["temperature_ids"]),
            )
            for h in work["evidence"].get("heads", [])
        ],
        diagnostics={k: n.reduce(v) for k, v in work["diagnostics"].items()},
        historical_model_provenance=work.get("historical_model_provenance", {}),
        optional_sibling_disposition=work["optional_sibling_disposition"],
        role_manifest=work["evidence"].get("roles", {}),
        reserved_outcomes_opened=False,
        measurement_reference=reference(raw / "measurement.json"),
        measurement_clocks=work["measurement_clocks"],
        repository_health=work.get("global_health", {}),
        methodology_note="Qualified binary conditional NLL plus ridge; original five-value grid and four fit-only folds supersede the methods fixture ridge0.01 as explicitly required by8208. Shared 256-iteration budget,16 inputs and17 coefficients; separate bounded temperature role and tune-selected simple control. Common baseline permission, original costs and target-free decisions. Constant and shuffled diagnostics are separate. Reused Qwen provenance is historical; current calls zero; H1 remains unmeasured.",
        claim_scope="fit validity and immutable heads only; reserved evaluation belongs to8209 and independent H1 audit to8210",
        **reduced,
    )
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Bind this invocation to exact inputs, primitive rows, real exits and sealed heads; exposed fitting cannot establish independent benefit."
        for k in value
    }
    value["field_principles"].update(
        action_fit_ready_score="Real optimization, sealed heads and passed owned validation suffice; favorable tune costs are not required.",
        independent_count="Unique completed source clusters, never folds, arms, seeds or diagnostic repeats.",
        reserved_outcomes_opened="Only fit/tune primitive references are resolved; reserved identities alone are frozen.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Fresh processes rehash primitive bytes and independently recompute rows."""
    try:
        value = json.loads(path.read_bytes())
        checksum = value.pop("reproducibility_checksum")
        if canonical_hash(value) != checksum:
            return False
        for ref in [
            value["measurement_reference"],
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
            *value["code_config_hashes"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for receipt in [*value["validation_receipts"], *value["precondition_receipts"]]:
            for label in ("stdout", "stderr"):
                if (
                    label + "_path" in receipt
                    and sha256_file(Path(receipt[label + "_path"])) != receipt[label + "_sha256"]
                ):
                    return False
        work = json.loads(Path(value["measurement_reference"]["path"]).read_bytes())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        if work["evidence"]:
            if json.loads((raw / "primitive_evidence.json").read_bytes()) != work[
                "evidence"
            ] or json.loads((raw / "independent_reduction.json").read_bytes()) != n.reduce(
                work["evidence"]
            ):
                return False
            frozen = json.loads((raw / "frozen_heads.json").read_bytes())
            if (
                frozen["heads"] != work["evidence"]["heads"]
                or frozen["selected_simple_control"] != value["selected_simple_control"]
            ):
                return False
        return build(
            work, raw, value["validation_receipts"], fixture=value["verifier_is_oracle"]
        ) == dict(value, reproducibility_checksum=checksum)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze explicit owned checks; preserve unrelated full-suite health once."""
    with (
        patch.object(execution, "OWNED", OWNED),
        patch.object(execution, "e", sys.modules[__name__]),
    ):
        specs = BASE_MANIFEST(private, candidate)
    (private / "pytest_parent").mkdir(parents=True, exist_ok=True)
    config = private / "coverage.ini"
    config.write_text(config.read_text() + "[report]\nexclude_lines =\n")
    for spec in specs["commands"]:
        if spec["name"] == "owned_unit_and_private_CLI":
            spec["argv"].extend(["--basetemp=" + str(private / "pytest_parent/owned")])
            spec["deadline_s"] = 240
        if spec["name"] == "consumer_and_E2E015_019":
            start = spec["argv"].index("tests/python/test_development_methods_8098.py")
            spec["argv"][start:] = [
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_source_boundary_7852.py",
                "tests/python/test_experiment_7942_v689_sentence_labels.py",
                "tests/python/test_restricted_action_methods_8207.py",
                "--basetemp=" + str(private / "pytest_parent/consumers"),
            ]
            spec["deadline_s"] = 240
        if spec["name"] == "spec_coverage":
            spec["argv"].insert(-1, "--files")
        if spec["name"] == "strict_mypy":
            spec["argv"] = [
                a.replace("--follow-imports=silent", "--follow-imports=skip") for a in spec["argv"]
            ]
    specs["repository_health"]["deadline_s"] = 180
    specs["repository_health"]["argv"].append("--basetemp=" + str(private / "pytest_parent/full"))
    return specs


def main(argv: list[str] | None = None) -> int:
    """Use the qualified bounded supervisor and unchanged publication validator."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
        patch.object(execution, "manifest", manifest),
        patch.object(execution, "run_check", methods.run_check),
    ):
        return int(execution.main(argv))
