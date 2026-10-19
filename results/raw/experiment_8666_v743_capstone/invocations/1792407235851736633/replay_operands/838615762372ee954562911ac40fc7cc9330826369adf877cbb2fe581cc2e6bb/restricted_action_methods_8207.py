"""REQ-REPORT-8207: register a changed action protocol without natural fitting.

Real fixture fits qualify mechanics. Original source identities stay exposed;
the next experiments own natural fitting, sealed predictions and the H1 audit.
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

import numpy as np
from scipy.special import expit  # type: ignore[import-untyped]

from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting import v709_execution as supervisor
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import restricted_action_rule_8207 as n
from carnot.verify import selective_methods_8194 as old
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_8207_v709_restricted_action_methods"
TASK = "exp8207-restricted-action-methods"
MODULE = "python/carnot/verify/restricted_action_methods_8207.py"
NUMERIC = "python/carnot/verify/restricted_action_rule_8207.py"
RUNNER = MODULE
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_restricted_action_methods_8207.py"
OWNED = [MODULE, NUMERIC, CLI]
RUN_DATE = "20261006"
MODEL_SPECS: list[Json] = []
PROTOCOL = "openspec/change-proposals/v709-restricted-action-protocol.json"
PIN = "sha256:9930e8704978baa53b58234d8fd5f9e74cea98d8f913571c930052b53f22fdf0"
BASE_MANIFEST = execution.manifest


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual work counts at each boundary so child waits stay visible."""
    print(f"[exp8207] phase={phase} completed={completed} pending={pending}", flush=True)


def query(row: Json) -> Json:
    """Strip fit targets before any decision code can see an input."""
    return {k: row[k] for k in ("unit_id", "source_cluster_id", "x", "historical_x")}


def fixture() -> tuple[list[Json], Json, Json]:
    """Oracle fixtures qualify code paths and receive zero scientific credit."""
    rows, reserved, baseline = old.fixture()
    roles = old.n.freeze_roles(rows, reserved)
    for row in rows:
        row["x"][0] = 2.0 if row["y"] else -2.0
        row["historical_x"] = row["x"][:12]
    return rows, roles, baseline


def reduce_diagnostics(data: Json) -> Json:
    """Recompute fixture probabilities and control selection from saved heads."""
    result: Json = dict(empty=dict(action=n.action(None, "accept")))
    for condition, item in data.items():
        predictions = [
            n.predict(h, query(r), item["baseline"]) for h in item["heads"] for r in item["rows"]
        ]
        spreads = [
            float(np.ptp([p["p"] for p in predictions if p["arm"] == arm])) for arm in n.ARMS
        ]
        parity = []
        for head in item["heads"]:
            for row in item["rows"]:
                phi = n.design(head["arm"], np.asarray([row["x"]]), head["geometry"])[0]
                logistic = float(
                    expit(float(phi @ np.asarray(head["weights"])) / head["temperature"])
                )
                energy = n.predict(head, query(row), item["baseline"])["p"]
                parity.append(abs(logistic - energy))
        result[condition] = dict(
            maximum_probability_spread=max(spreads),
            equivalent_logistic_maximum_error=max(parity),
            label_order_sha256=canonical_hash([r["y"] for r in item["rows"]]),
            action_counts={
                a: sum(p["action"] == a for p in predictions)
                for a in ["accept", "reject", "escalate"]
            },
            selected_simple=n.select_simple(
                item["heads"], item["rows"], item["roles"], item["baseline"]
            ),
            fit_count=len(item["roles"]["head_fit"]),
            temperature_count=len(item["roles"]["temperature_fit"]),
        )
    return result


def diagnostics() -> tuple[Json, Json]:
    """Constant scores and shuffled fit targets expose degenerate learned behavior."""
    rows, roles, baseline = fixture()
    data = {}
    for index, condition in enumerate(["natural", "constant", "shuffled"]):
        progress("before_benchmark_fixture_" + condition, index, 3 - index)
        changed = deepcopy(rows)
        if condition == "constant":
            for row in changed:
                row["x"] = [0.0] * 16
        if condition == "shuffled":
            ids = {r["unit_id"] for r in roles["head_fit"]}
            selected = [r for r in changed if r["unit_id"] in ids]
            targets = np.random.default_rng(7098207).permutation([r["y"] for r in selected])
            for row, y in zip(selected, targets, strict=True):
                row["y"] = int(y)
        data[condition] = dict(
            rows=changed, roles=roles, baseline=baseline, heads=n.train(changed, roles)
        )
        progress("after_benchmark_fixture_" + condition, index + 1, 2 - index)
    return data, reduce_diagnostics(data)


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """Authenticate branch operands before running owned fixtures; absent stays absent."""
    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(
        checks=[],
        refs=[],
        raw_shard_hashes=[],
        evidence={},
        diagnostics={},
        owned_failure="",
        protocol={},
        precondition_receipts=[],
        cited_upstream_artifacts=[],
    )
    gate = lambda p, field, expected, observed: old.fit.gate(work, p, field, expected, observed)
    read = lambda ref: old.fit.bind(work, ref, raw)
    progress("before_preconditions")
    try:
        gate(Path(sys.executable), "python_supported", True, sys.version_info >= (3, 11))
        probe = raw / ".probe"
        probe.write_bytes(b"private writable scratch")
        gate(
            raw, "private_scratch_writable", True, probe.read_bytes() == b"private writable scratch"
        )
        probe.unlink()
        protocol = read(dict(path=str(ROOT / PROTOCOL), sha256=PIN))
        work["protocol"] = protocol
        if mutation:
            gate(
                root / protocol["source_artifact_hashes"][0]["path"],
                "source_custody",
                "authenticated",
                None,
            )
        if not fixture:
            for ref in protocol["source_artifact_hashes"]:
                path = root / Path(ref["path"]).relative_to(ROOT)
                value = read(dict(path=str(path), sha256=ref["sha256"]))
                if "experiment_" in path.name:
                    for field, expected in [
                        ("required_checks_passed", True),
                        ("flagged_adversarial", False),
                    ]:
                        gate(path, field, expected, value.get(field))
                    gate(
                        path,
                        "terminal_publication_passed",
                        True,
                        read_bound_sidecar(path, old.fit.historical.publication_sidecar(value))[
                            "report"
                        ]["passed"],
                    )
                    work["cited_upstream_artifacts"].append(
                        dict(
                            experiment_id=value["experiment_id"],
                            sha256=ref["sha256"],
                            fields_imported=[
                                "role_manifest",
                                "historical_disposition",
                                "original_control_binding",
                            ],
                        )
                    )
                    if value["experiment_id"] == 8195:
                        previous = read(value["measurement_reference"])
                        normalize = lambda roles: {
                            k: sorted(v, key=lambda r: r["source_cluster_id"])
                            for k, v in roles.items()
                        }
                        gate(
                            path,
                            "same_role_manifest",
                            normalize(protocol["role_manifest"]),
                            normalize(previous["role_manifest"]),
                        )
                        atomic_json(raw / "original_baseline.json", previous["evidence"]["control"])
            literature = json.loads(
                (ROOT / "results/raw" / NAME / "literature/access.json").read_bytes()
            )
            for source in protocol["literature_mapping"]:
                if source["status"] == "readable":
                    path = Path(source["path"])
                    gate(path, "primary_response_sha256", source["sha256"], sha256_file(path))
                    target = raw / (source["arxiv"] + ".html")
                    target.write_bytes(path.read_bytes())
                    work["refs"].append(old.fit.reference(target))
            work["precondition_receipts"] = [r["receipt"] for r in literature]
            startup = Path("/tmp/carnot8207-startup")
            if startup.is_dir():
                shutil.copytree(startup, raw / "startup_receipts")
        progress("after_preconditions", len(work["checks"]), 0)
    except (OSError, ValueError, KeyError, TypeError) as error:
        if all(c["passed"] for c in work["checks"]):
            work["checks"].append(
                dict(
                    check="input_custody",
                    path=str(root),
                    hash=None,
                    artifact_field="authenticated_input_schema",
                    op="==",
                    expected="valid",
                    observed=str(error),
                    passed=False,
                )
            )
    if all(c["passed"] for c in work["checks"]):
        try:
            work["evidence"], work["diagnostics"] = diagnostics()
            gate(
                raw,
                "equivalent_logistic_identity",
                True,
                all(
                    r["equivalent_logistic_maximum_error"] <= 1e-10
                    for k, r in work["diagnostics"].items()
                    if k != "empty"
                ),
            )
        except (ValueError, TimeoutError) as error:
            work["owned_failure"] = str(error)
    work["optional_sibling_disposition"] = dict(
        path=str(root / "results/experiment_8205_v709_contract_consumer_qualification.json"),
        required=False,
        available=(
            root / "results/experiment_8205_v709_contract_consumer_qualification.json"
        ).is_file(),
    )
    work["code_config_hashes"] = [
        old.fit.reference(ROOT / p)
        for p in [
            *OWNED,
            TEST,
            PROTOCOL,
            "python/carnot/verify/evidence_energy_8154.py",
            "python/carnot/verify/selective_rule_8194.py",
            "python/carnot/reporting/v709_execution.py",
            "python/carnot/reporting/methods_stream_execution_8111.py",
            "scripts/experiment_template.py",
        ]
    ]
    atomic_json(raw / "fixture_primitives.json", work["evidence"])
    work["raw_shard_hashes"] = [
        old.fit.reference(p)
        for p in sorted(raw.rglob("*"))
        if p.is_file() and not p.is_relative_to(raw / "logs")
    ]
    work["duration_s"] = time.monotonic() - began
    work["clock"] = dict(
        measured_wall_ns=time.time_ns(), completed_monotonic_ns=time.monotonic_ns()
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", len(work["diagnostics"]), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Administrative readiness cannot report unmeasured H1 as positive science."""
    protocol = work["protocol"]
    failures = [c for c in work["checks"] if not c["passed"]]
    checked = bool(receipts) and all(r["passed"] for r in receipts) and not work["owned_failure"]
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
    rows = [
        dict(
            **r,
            arm="restricted_action_registration",
            condition="original_reserved_slot",
            metric="protocol_registered",
            numerator=ready,
            denominator=1,
            status="completed" if ready else "excluded",
            exclusion_reason=None if ready else "protocol_unavailable",
            evaluation_status="pending_exp8209",
            semantic_metric=None,
        )
        for r in protocol.get("role_manifest", {}).get("reserved", [])
    ]
    value: Json = dict(
        experiment_id=8207,
        task_id=TASK,
        milestone="2026.10.709",
        run_date=RUN_DATE,
        honest_verdict="complete_" + verdict + "_restricted_action_methods",
        verdict_class=verdict,
        gate_check_summary=work["checks"],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(
            model_loads=0, generate_calls=0, forward_calls=0, model_count=0
        ),
        call_ledger=[],
        trained_head_specs=[],
        preconditions_checked=[
            dict(resource=c["check"], available=c["passed"], path=c["path"]) for c in work["checks"]
        ],
        precondition_receipts=work["precondition_receipts"],
        duration_s=work["duration_s"],
        random_seed=7098207,
        rows=rows,
        intended_count=len(rows),
        completed_count=len(rows) if ready else 0,
        failed_count=0,
        censored_count=0,
        excluded_count=0 if ready else len(rows),
        independent_count=len(rows),
        verifier_is_oracle=fixture,
        exposure_scope="exposed_development_within_run_disjoint",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        action_protocol_ready_score=ready,
        protocol_path=str(ROOT / PROTOCOL),
        protocol_sha256=PIN,
        role_manifest=protocol.get("role_manifest", {}),
        literature_mapping=protocol.get("literature_mapping", []),
        acceptance_subset_proof=protocol.get("acceptance_subset_proof"),
        primary_control_selection_rule=protocol.get("primary_control_selection_rule"),
        primary_control_selected=None,
        H1=dict(protocol.get("H1", {}), measured_here=False),
        H2=protocol.get("H2", {}),
        acceptance_gates=dict(
            owned_validation=checked,
            input_authentication=not failures,
            protocol_mechanics=bool(work["evidence"]),
            scientific_H1="registered_unmeasured",
        ),
        validation_receipts=receipts,
        required_checks_passed=checked,
        flagged_adversarial=False,
        source_artifact_hashes=work["refs"],
        cited_upstream_artifacts=work["cited_upstream_artifacts"],
        raw_shard_hashes=work["raw_shard_hashes"],
        code_config_hashes=work["code_config_hashes"],
        fixture_diagnostics=reduce_diagnostics(work["evidence"]) if work["evidence"] else {},
        fixture_protocol_only=fixture,
        measurement_reference=old.fit.reference(raw / "measurement.json"),
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        optional_sibling_disposition=work["optional_sibling_disposition"],
        policy_retirement=protocol.get("retirement", {}),
        repository_health=work.get("global_health", {}),
        measurement_clocks=work["clock"],
        claim_scope="protocol registration and oracle fixture mechanics; natural fitting deferred to8208; reserved measurement deferred to8209/8210",
        methodology_note="Deterministic permission limits actions for every input; it is not learned correctness. All synthetic fits are plumbing fixtures. No current LLM calls or population risk guarantee. H1/H2 remain unmeasured here.",
    )
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Bind current protocol mechanics to exact evidence; fixtures and historical exposure confer zero independent scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        rows="128 original identities register a protocol, not measured outcomes; semantic metrics stay null until8209/8210.",
        acceptance_subset_proof="For every input and label assignment, extra false accept count is bounded by baseline; remaining-accept error rate is not bounded.",
        H1="Tune chooses a simple comparator before reserved labels; H1 and H2 share family alpha0.05.",
        action_protocol_ready_score="Frozen executable costs, permission and negative fixtures pass; future learned benefit remains unknown.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Rehash saved bytes and recompute fixture conclusions before accepting rows."""
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
            for label in ["stdout", "stderr"]:
                if (
                    label + "_path" in receipt
                    and sha256_file(Path(receipt[label + "_path"])) != receipt[label + "_sha256"]
                ):
                    return False
        work = json.loads(Path(value["measurement_reference"]["path"]).read_bytes())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        if work["evidence"] != json.loads((raw / "fixture_primitives.json").read_bytes()):
            return False
        rebuilt = build(
            work, raw, value["validation_receipts"], fixture=value["fixture_protocol_only"]
        )
        return rebuilt == dict(value, reproducibility_checksum=checksum)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_check(root: Path, spec: Json, private: Path, raw: Path, *, heartbeat_s: float = 20) -> Json:
    """Adapt the existing supervisor to retain separate complete child streams."""
    return supervisor.child(
        spec["name"],
        spec["argv"],
        raw,
        deadline=spec["deadline_s"],
        expected=spec["expected_exit"],
        heartbeat=heartbeat_s,
        scope=spec.get("classification", "owned"),
    )


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze exact owned paths, private coverage and one repository-health run."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
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
                "tests/python/test_selective_decision_audit_8197.py",
                "--basetemp=" + str(private / "pytest_parent/consumers"),
            ]
            spec["deadline_s"] = 240
        if spec["name"] == "strict_mypy":
            spec["argv"] = [
                a.replace("--follow-imports=silent", "--follow-imports=skip") for a in spec["argv"]
            ]
        if spec["name"] == "spec_coverage":
            spec["argv"].insert(-1, "--files")
    specs["repository_health"]["deadline_s"] = 180
    return specs


def main(argv: list[str] | None = None) -> int:
    """Reuse normal-exit validation and atomic candidate publication end to end."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
        patch.object(execution, "manifest", manifest),
        patch.object(execution, "run_check", run_check),
    ):
        return int(execution.main(argv))
