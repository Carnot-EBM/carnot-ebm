"""REQ-REPORT-8609: qualify mechanics while preserving original scientific authority."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import lean_resident_8608 as prototype
from carnot.reporting import lean_peer_8609 as peer
from carnot.reporting import resident_complete_cost_8584 as original
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.current_validator_8580 import xml_nodes
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.resident_worker_8572 import evaluate
from carnot.reporting.v709_execution import progress
from carnot.verify.native_nine_input_8533 import load

Json = dict[str, Any]
ROOT = prototype.ROOT
NAME = "experiment_8609_v739_lean_resident_qualification"
TASK = "exp8609-lean-resident-qualification"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_lean_qualification_8609.py", "tests/python/test_lean_runner_8609.py"]
OWNED = [
    f"python/carnot/reporting/lean_{s}_8609.py" for s in ["peer", "qualification", "runner"]
] + ["python/carnot/reporting/lean_original_8609.py", CLI]
ARMS = ["legacy_python", "legacy_native", "lean_python", "lean_native"]
KINDS = [
    "expired",
    "duplicate",
    "malformed",
    "nonfinite",
    "empty",
    "long_text",
    "stale",
    "partial",
    "timeout",
    "idle",
    "abnormal",
    "backpressure",
]
MODEL_SPECS: list[Json] = []
REQUIRED = {
    "exact_disjoint_node_union",
    "new_owned_statement_coverage_100",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "cold",
    "missing",
    "owned_failure",
    "rehashed_tamper",
    "private_publication_readers",
}
reference, authenticate, checksum = prototype.reference, prototype.authenticate, prototype.checksum


def upstream(root: Path, name: str) -> tuple[Json, list[Json]]:
    """Only exact bytes with an accepted bound publication can supply authority."""
    path = root / "results" / (name + ".json")
    ref = reference(path)
    value = json.loads(path.read_bytes())
    sidecar = path.parent / "raw" / name / "validators" / (ref["sha256"].split(":")[1] + ".json")
    if read_bound_sidecar(path, sidecar)["report"]["passed"] is not True:
        raise ValueError("upstream_publication")
    return value, [ref, reference(sidecar)]


def summarize(work: Json) -> list[Json]:
    """Reconstruct every response from original arithmetic, never from success totals."""
    rows: list[Json] = []
    if not work["sessions"]:
        return rows
    config = json.loads(authenticate(work["config_reference"]).read_bytes())
    native = load(authenticate(config["extension"]))
    for session in work["sessions"]:
        config = json.loads(authenticate(session["config_reference"]).read_bytes())
        label = session["arm"]
        for i, (operand, observed) in enumerate(
            zip(config["rows"], session["observations"], strict=True)
        ):
            actual = observed["response"]
            expected, _ = evaluate(
                config["head"], operand, native if label.endswith("native") else None
            )
            reference_response, _ = evaluate(config["head"], operand, None)
            response = actual.get("response", {})
            fields = set(response) == set(expected) and all(
                type(response[k]) is type(expected[k]) for k in expected
            )
            exact = all(
                response.get(k) == v
                for k, v in expected.items()
                if k not in {"probability", "residual"}
            )
            errors = {
                k: abs(response.get(k, -999) - v)
                for k, v in reference_response.items()
                if k in {"probability", "residual"}
            }
            finite = all(
                error <= 1e-12 * (1 + abs(reference_response[k])) for k, error in errors.items()
            )
            passed = fields and exact and finite and actual.get("status") == "ok"
            passed = (
                passed
                and actual.get("request_id") == f"request-{i}"
                and actual.get("config_hash") == session["config_reference"]["sha256"]
            )
            rows.append(
                dict(
                    unit_id=label + ":" + operand["unit_id"],
                    arm=label,
                    kind=config["case_kinds"][i],
                    response=response,
                    expected_response=expected,
                    errors=errors,
                    passed=bool(passed),
                    decision_mismatch=response.get("decision") != expected["decision"],
                    missing_reason=None if passed else "parity_or_response_contract",
                )
            )
    return rows


def configurations(bundle: Json) -> tuple[Json, Json]:
    """Derive every fixture configuration from original operands during cold replay."""
    requests = [r for r in bundle["answers"] if r["role"] == "test"][:64]
    config = dict(
        head=bundle["head"],
        extension={k: bundle["extension"][k] for k in ["path", "sha256"]},
        rows=requests + bundle["controls"],
    )
    long_row = deepcopy(requests[0])
    long_row.update(
        unit_id="long-text",
        public=dict(
            family_id="long-text",
            source_bytes=(b"There are 12 apples. " * 60).hex(),
            answer_bytes=b"There are 12 apples.".hex(),
        ),
    )
    config["rows"].append(long_row)
    kinds = ["original64"] * 64 + ["knot_decay"] * len(bundle["controls"]) + ["long_text"]
    config["case_kinds"] = kinds
    import numpy as np

    threshold = deepcopy(config)
    threshold["head"] = dict(
        config["head"], parameters=[0.0] * 109, temperature=1.0, decay_scale=1.0
    )
    threshold["rows"] = [
        dict(requests[0], unit_id=f"threshold-{i}-{j}", q=float(q))
        for i, p in enumerate([0.2, 0.8])
        for j, q in enumerate([np.nextafter(p, 0), p, np.nextafter(p, 1)])
    ]
    threshold["case_kinds"] = ["threshold"] * 6
    return config, threshold


def reserve_lifecycle(deadline_ns: int) -> None:
    """Keep enough time to reap a private peer before the measurement cutoff."""
    if time.monotonic_ns() + 90_000_000_000 >= deadline_ns:
        raise TimeoutError("lifecycle_deadline_reserved_cleanup")


def measure(raw: Path, run_date: str, *, root: Path = ROOT) -> Json:
    """Run independent CPU mechanics even when current scheduling authority is blocked."""
    progress("8609_start_no_model_load_MODEL_SPECS=[]_current_llm_calls=0")
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(
        raw=str(raw),
        run_date=run_date,
        started_ns=time.monotonic_ns(),
        gate_check_summary=[],
        sources=[],
        sessions=[],
        lifecycle_rows=[],
        rejection_rows=[],
        parity_rows=[],
        case_kinds=[],
        target_model=None,
        loaded_extension_identity=[],
    )
    with TemporaryDirectory(prefix="carnot8609-", dir="/var/tmp") as scratch:
        path = Path(scratch) / "probe"
        path.write_bytes(b"disk-backed qualification scratch")
        work["preconditions_checked"] = dict(
            scratch=str(path.parent),
            writable=path.read_bytes() == b"disk-backed qualification scratch",
            disk_free_bytes=shutil.disk_usage(scratch).free,
            memory=Path("/proc/meminfo").read_text(),
            tools={n: shutil.which(n) for n in ["python", "findmnt"]},
            python=sys.version,
            scratch_mount=subprocess.check_output(
                ["findmnt", "-T", scratch, "-n", "-o", "FSTYPE"], text=True, timeout=5
            ).strip(),
            optional_gpu="not required",
            optional_board="not required",
        )
    admitted = {}
    for name, field in [
        (prototype.NAME, "lean_prototype_ready_score"),
        ("experiment_8602_v739_contract_methods", "methods_ready_score"),
    ]:
        gate: Json = dict(
            check="qualified_input",
            upstream=name,
            path=str(root / "results" / (name + ".json")),
            sha256=None,
            field=field,
            operator="==",
            expected=1,
            observed="missing",
        )
        try:
            value, refs = upstream(root, name)
            work["sources"].extend(refs)
            gate.update(sha256=refs[0]["sha256"], observed=value.get(field, "missing"))
            admitted[name] = value
            if value.get(field) != 1 or value.get("required_checks_passed") is not True:
                gate["upstream_findings"] = value.get("gate_check_summary", [])
                work["gate_check_summary"].append(gate)
        except (OSError, ValueError, KeyError) as error:
            gate["error"] = str(error)
            work["gate_check_summary"].append(gate)
    if prototype.NAME in admitted and admitted[prototype.NAME]["lean_prototype_ready_score"] == 1:
        value = admitted[prototype.NAME]
        prototype.replay(root / "results" / (prototype.NAME + ".json"))
        prior = json.loads(authenticate(value["work_reference"]).read_bytes())
        bundle = original.admit(root)
        work["sources"].extend(
            [value["work_reference"], *value["source_artifact_hashes"], *bundle["refs"]]
        )
        work.update(
            target_model=prior["target_model"],
            historical_model_provenance=prior["historical_model_provenance"],
            preserved_quarantines=prior["preserved_quarantines"],
        )
        requests = [r for r in bundle["answers"] if r["role"] == "test"][:64]
        assert len(requests) == 64 and prior["head"] == bundle["head"]
        config, threshold = configurations(bundle)
        work["case_kinds"] = config["case_kinds"]
        atomic_json(raw / "configuration.json", config)
        work["config_reference"] = reference(raw / "configuration.json")
        progress("8609_before_loaded_parity")
        deadline_ns = work["started_ns"] + 3600_000_000_000
        try:
            for label in ARMS:
                for kind, operands in [("original", config), ("threshold", threshold)]:
                    session = peer.execute(
                        operands, raw / label / kind, label, deadline_ns=deadline_ns
                    )
                    work["sessions"].append(session)
                    work["loaded_extension_identity"].append(
                        dict(arm=label, loaded=session["ready"]["loaded_extension"])
                    )
                for kind in KINDS:
                    reserve_lifecycle(deadline_ns)
                    work["lifecycle_rows"].append(
                        peer.lifecycle(config, raw / label / kind, label, kind)
                    )
                progress("8609_completed_arms", ARMS.index(label) + 1, 3 - ARMS.index(label))
            work["parity_rows"] = summarize(work)
        except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
            work["owned_failure"] = str(error)
            work["parity_rows"] = [
                dict(
                    unit_id="owned-measurement",
                    arm="owned",
                    kind="owned_failure",
                    passed=False,
                    missing_reason=str(error),
                )
            ]
        work["rejection_rows"] = [r for r in work["lifecycle_rows"] if r["response"] is not None]
        work["descriptor_qualification"] = dict(
            original_descriptor=prior["target_model"],
            changed=False,
            affected_operands_byte_identical=True,
            original_producer=8593,
        )
    methods = admitted.get("experiment_8602_v739_contract_methods")
    if methods is not None:
        ref = methods["methods_protocol_reference"]
        protocol = json.loads(authenticate(ref).read_bytes())
        assert protocol["lean_worker_2x2"] == methods["lean_worker_2x2"]
        work["sources"].append(ref)
        work["registered_lean_methods"] = protocol["lean_worker_2x2"]
    work["code_config_hashes"] = [
        reference(ROOT / p) for p in OWNED + TESTS + prototype.BASELINE + prototype.OWNED
    ]
    work["ended_ns"] = time.monotonic_ns()
    atomic_json(raw / "work.json", work)
    progress("8609_primitives_complete")
    return work


def artifact(work: Json, receipts: list[Json], output: Path) -> Json:
    """Qualification is separate from speed, and owned failure takes precedence."""
    value: Json = prototype.artifact(dict(work, sessions=[]), receipts, output)
    value.pop("lean_prototype_ready_score")
    rows = deepcopy(work["parity_rows"] + work["lifecycle_rows"])
    passed = all(r["passed"] for r in receipts + rows) and all(
        s["partition"]["passed"] for s in work["sessions"]
    )
    ready = int(
        passed
        and not work["gate_check_summary"]
        and bool(rows)
        and REQUIRED <= {r.get("name") for r in receipts if r["passed"]}
    )
    verdict = (
        "disqualified"
        if not passed
        else "blocked"
        if work["gate_check_summary"]
        else "null"
        if ready
        else "disqualified"
    )
    value.update(
        experiment_id=8609,
        task_id=TASK,
        honest_verdict=f"complete_{verdict}_v739_lean_resident_qualification",
        verdict_class=verdict,
        required_checks_passed=passed,
        rows=rows,
        intended_count=len(rows) if rows else 4,
        completed_count=len(rows),
        failed_count=sum(not r["passed"] for r in rows),
        censored_count=0 if rows else 4,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            original_requests=64,
            arms=4,
            independent_questions=0,
            interpretation="exposed mechanics only",
        ),
        exposure_scope="original exposed requests, numerical fixtures and private lifecycle controls",
        lean_measurement_ready_score=ready,
        parity_rows=deepcopy(work["parity_rows"]),
        rejection_rows=deepcopy(work["rejection_rows"]),
        lifecycle_rows=deepcopy(work["lifecycle_rows"]),
        loaded_extension_identity=work["loaded_extension_identity"],
        original_node_union=work.get("original_node_union", {}),
        child_statement_coverage=work.get("child_statement_coverage", {}),
        complete_operation_contract=dict(
            boundary="immutable configuration through child reap",
            sessions=[dict(arm=s["arm"], **s["partition"]) for s in work["sessions"]],
            benchmark_skipped=True,
            speed_inferred=False,
        ),
        phase_spans=[
            dict(
                phase="authentication_loaded_parity_and_private_lifecycle",
                start_ns=work["started_ns"],
                end_ns=work["ended_ns"],
            )
        ],
        descriptor_qualification=work.get("descriptor_qualification"),
        registered_lean_methods=work.get("registered_lean_methods"),
        work_reference=reference(Path(work["raw"]) / "work.json"),
        preserved_history=dict(
            native_ci95=[1.0740525, 1.1057774],
            startup_reuse_separate=True,
            NFR01_met=False,
            exp8592="second deadline-clock read unqualified",
            exp8597="no usable MedExpert release evidence",
            exp8599="zero resolved ARC outcomes",
        ),
    )
    value["field_principles"] = {
        k: "Invocation-bound mechanics evidence; no independent learning or speed benefit."
        for k in value
    }
    value["field_principles"].update(
        lean_measurement_ready_score="One requires parity, isolation, unchanged head, cold replay, original node union and 100 percent owned coverage; any external gate blocks.",
        complete_operation_contract="Disjoint actual launch, transfer, compute, serialization and shutdown clocks retain explicit unattributed time without speed inference.",
        gate_check_summary="Failed check, upstream, path/hash, field, operator, expected and observed preserve missing separately from zero.",
        child_statement_coverage="Combined genuine subprocess and unit statement coverage; only newly owned files control the 100 percent acceptance gate.",
        loaded_extension_identity="Each real ready frame binds the loaded native binary to the original descriptor.",
        parity_rows="Every arm and immutable operand retains exact typed fields and original 1e-12*(1+abs(reference)) tolerance.",
        rejection_rows="Errors have no successful payload; the next accepted request retains its own identity.",
        lifecycle_rows="Real private process and pipe failures retain exits, recovery, cleanup and hashed traces.",
        target_model="Original Exp8593 descriptor authenticates the small numerical head; no language model loads occur.",
        original_node_union="Exact original collected identities equal the disjoint passed union, with original fixtures and dates.",
        descriptor_qualification="Reuse Exp8593 only for the same checkpoint, parameters, scaler, temperature, policy and native binary bytes.",
        registered_lean_methods="Exp8602 registers the four resident arms and original statistical rules; its authority failure remains a separate blocking gate.",
        preserved_history="Original confidence interval, startup distinction, deadline failure, evidence gaps and unresolved ARC outcomes are unchanged.",
        rows="Each exposed operand and arm or controlled lifecycle case is retained once; repetitions confer no independent-question credit.",
        independent_count="Zero because this invocation qualifies exposed execution mechanics rather than evaluating new scientific questions.",
        sample_size_budget="The original64 request panel crosses four arms; knot, threshold and failure fixtures add mechanics checks only.",
        required_checks_passed="All owned tests, response checks, operation partitions and bounded validation receipts pass; external authority blockage is reported separately.",
        acceptance_gates="Every required gate retains its actual result; any owned failure disqualifies and all failed gates forbid positive verdicts.",
        preconditions_checked="Actual tools, host memory, writable disk-backed scratch and authenticated input bytes are recorded before CPU work.",
        duration_s="Measured monotonic time for authentication and real private children, without padding or a speed conclusion.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> None:
    """Fresh reconstruction rejects rehashed summaries and replaced scientific operands."""
    value = json.loads(path.read_bytes())
    if value.get("reproducibility_checksum") != checksum(value):
        raise ValueError("artifact_checksum")
    work = json.loads(authenticate(value["work_reference"]).read_bytes())
    for ref in work["sources"] + work["code_config_hashes"]:
        authenticate(ref)
    if "original_node_union" in work:
        union = work["original_node_union"]
        outcomes = [node for ref in union["junit_shards"] for node in xml_nodes(authenticate(ref))]
        if sorted(outcomes) != sorted(union["collected"]) or sorted(outcomes) != sorted(
            union["passed_node_ids"]
        ):
            raise ValueError("original_node_union_drift")
    if work["sessions"] and not work.get("owned_failure"):
        methods, _ = upstream(ROOT, "experiment_8602_v739_contract_methods")
        if methods["methods_ready_score"] != 1 and not any(
            gate["upstream"] == "experiment_8602_v739_contract_methods"
            and gate["field"] == "methods_ready_score"
            and gate["observed"] == methods["methods_ready_score"]
            for gate in work["gate_check_summary"]
        ):
            raise ValueError("external_authority_block_erased")
        producer, _ = upstream(ROOT, prototype.NAME)
        prototype.replay(ROOT / "results" / (prototype.NAME + ".json"))
        bundle = original.admit(ROOT)
        if work["target_model"] != producer["target_model"]:
            raise ValueError("descriptor_authority")
        config = json.loads(authenticate(work["config_reference"]).read_bytes())
        requests = [r for r in bundle["answers"] if r["role"] == "test"][:64]
        if (
            config["head"] != bundle["head"]
            or config["rows"][:64] != requests
            or config["rows"][64:-1] != bundle["controls"]
        ):
            raise ValueError("original_operands")
        if [s["arm"] for s in work["sessions"]] != [arm for arm in ARMS for _ in range(2)]:
            raise ValueError("session_roster")
        expected_configs = configurations(bundle)
        for index, session in enumerate(work["sessions"]):
            if (
                json.loads(authenticate(session["config_reference"]).read_bytes())
                != expected_configs[index % 2]
            ):
                raise ValueError("fixture_configuration_authority")
            for key in ["config_reference", "wrapper_reference", "stderr_reference"]:
                authenticate(session[key])
            modules = json.loads(authenticate(session["modules_reference"]).read_bytes())
            for ref in modules:
                authenticate(ref)
            if session["arm"].startswith("lean") and any(
                r["name"].startswith(("carnot.reporting", "carnot.training", "sklearn"))
                for r in modules
            ):
                raise ValueError("lean_import_graph")
            loaded = config["extension"] if session["arm"].endswith("native") else None
            if (
                session["ready"]["loaded_extension"] != loaded
                or session["exit_code"] != 0
                or session["forced_cleanup"]
            ):
                raise ValueError("loaded_child_contract")
            if peer.partition(session, "resident") != session["partition"]:
                raise ValueError("operation_partition")
        if summarize(work) != work["parity_rows"]:
            raise ValueError("reconstructed_parity")
        if [(r["arm"], r["kind"]) for r in work["lifecycle_rows"]] != [
            (a, k) for a in ARMS for k in KINDS
        ]:
            raise ValueError("lifecycle_roster")
        for row in work["lifecycle_rows"]:
            receipt = row["lifecycle"]
            authenticate(receipt["stderr_reference"])
            authenticate(receipt["config_reference"])
            if row["recovery"] is not None:
                expected, _ = evaluate(config["head"], config["rows"][1], None)
                actual = row["recovery"]["response"]
                if (
                    actual["unit_id"] != expected["unit_id"]
                    or actual["decision"] != expected["decision"]
                    or abs(actual["probability"] - expected["probability"])
                    > 1e-12 * (1 + abs(expected["probability"]))
                ):
                    raise ValueError("next_request_contamination")
            if row["kind"] in {
                "expired",
                "duplicate",
                "malformed",
                "nonfinite",
                "empty",
                "long_text",
            } and (row["response"]["status"] != "error" or "response" in row["response"]):
                raise ValueError("stale_successful_payload")
            if receipt["forced_cleanup"] or not receipt["child_reaped"] or not row["passed"]:
                raise ValueError("failed_lifecycle")
    expected = artifact(work, value["validation_receipts"], Path(value["publication_output"]))
    if value != expected:
        raise ValueError("reduction_drift")
    progress("8609_cold_replay_passed")
