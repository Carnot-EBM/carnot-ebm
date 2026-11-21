"""REQ-REPORT-9083-V775: enforce a finite supervised boundary before fresh collection.

The parent owns secrets and the append-only ledger. Workers receive numeric JSON
through a pipe in an OS namespace. This cannot establish past or host-wide access.
"""

from collections.abc import Callable
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
import uuid

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_9083_v775_prospective_custody", "exp9083-prospective-custody"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_prospective_custody_9083.py"
OWNED = [
    "python/carnot/reporting/prospective_custody_9083.py",
    "python/carnot/reporting/prospective_custody_runner_9083.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
SEED = 775083
REQUEST_FIELDS = {"slot", "graph", "shard", "role", "split", "prompt", "model", "seed"}
CONTROLS = (
    "direct_paths",
    "symlink_traversal",
    "inherited_descriptors",
    "alternate_entrypoint",
    "network",
    "missing_request_start",
    "duplicate_slot",
    "orphan_response",
    "reordered_ledger",
    "rehashed_ledger",
    "early_label_reads",
)
THREAT_MODEL = dict(
    trusted="parent custodian and immutable worker code",
    untrusted="scorer/trainer workers; no raw responses, source graphs, future labels or other streams",
    excluded="privileged host/operator compromise",
    scope="only invocations routed through this supervisor; no global completeness claim",
    isolation="OS mount/user/PID/network namespaces; allowlisted mounts; closed inherited descriptors",
    allow_fallback=False,
    in_process_guardrail_is_security_boundary=False,
)
WORKER = """import json, os, socket, sys
v = json.load(sys.stdin)
r = {"score": sum(v["features"]), "uid": os.getuid(),
     "namespaces": {k: os.readlink("/proc/self/ns/"+k) for k in ("mnt","user","pid","net")}}
denied = []
for p in v.get("paths", []):
    try:
        with open(p, "rb") as f: f.read(1)
        denied.append(False)
    except OSError: denied.append(True)
r["paths_denied"] = all(denied)
r["path_count"] = len(denied)
try:
    os.symlink(v.get("secret", "/absent"), "/tmp/link")
    with open("/tmp/link", "rb") as f: f.read(1)
    r["symlink_denied"] = False
except OSError: r["symlink_denied"] = True
try:
    os.read(v.get("fd", 9999), 1)
    r["fd_denied"] = False
except OSError: r["fd_denied"] = True
s = socket.socket(); s.settimeout(0.2)
try:
    s.connect(("192.0.2.1", 9)); r["network_denied"] = False
except OSError: r["network_denied"] = True
s.close()
print(json.dumps(r), flush=True)
"""


def progress(phase: str) -> None:
    """Flush each phase so supervisors can distinguish bounded work from a stall."""
    print(f"[exp9083] phase={phase}", flush=True)


def reference(path: Path) -> Json:
    """A byte identity, rather than an authority string, binds retained evidence."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path) if path.is_file() else None)


def private(path: Path) -> None:
    """Reject public existing directories before writing any retained response."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or path.stat().st_mode & 0o077:
        raise PermissionError("private_storage")


class Ledger:
    """Fsync intent before transport; crashes spend a slot instead of enabling a retry."""

    def __init__(self, root: Path):
        private(root)
        self.path = root / "ledger.jsonl"
        self.root = root

    def events(self) -> list[Json]:
        """Check every chain link before using the ledger as request authority."""
        rows = (
            [json.loads(s) for s in self.path.read_text().splitlines()]
            if self.path.exists()
            else []
        )
        previous = canonical_hash([])
        for index, row in enumerate(rows):
            body = {k: v for k, v in row.items() if k != "sha256"}
            if (
                row["index"] != index
                or row["previous"] != previous
                or canonical_hash(body) != row["sha256"]
            ):
                raise ValueError("ledger_chain")
            previous = row["sha256"]
        return rows

    def append(self, event: str, slot: str, payload: Json) -> None:
        """Append durable bytes; a seal prevents later application-level additions."""
        if (self.root / "seal.json").exists():
            raise ValueError("sealed_ledger")
        rows = self.events()
        body = dict(
            index=len(rows),
            previous=rows[-1]["sha256"] if rows else canonical_hash([]),
            event=event,
            slot=slot,
            payload=payload,
            monotonic_ns=time.monotonic_ns(),
        )
        body["sha256"] = canonical_hash(body)
        with self.path.open("a") as stream:
            stream.write(json.dumps(body, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        fd = os.open(self.root, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def start(self, request: Json) -> None:
        """Bind all scientific request fields before calling any collector."""
        if (
            set(request) != REQUEST_FIELDS
            or type(request["seed"]) is not int
            or any(
                not isinstance(request[k], str) or not request[k] for k in REQUEST_FIELDS - {"seed"}
            )
            or request["shard"] not in {"A", "B"}
        ):
            raise ValueError("request_schema")
        if any(r["slot"] == request["slot"] for r in self.events()):
            self.append("rejected", request["slot"], dict(reason="duplicate_slot", request=request))
            raise ValueError("duplicate_slot")
        self.append("start", request["slot"], dict(request, request_id=str(uuid.uuid4())))

    def finish(self, slot: str, response: Json, *, failed: bool = False) -> None:
        """Retain orphan outcomes as rejected attempts rather than dropping their bytes."""
        rows = self.events()
        if not any(r["event"] == "start" and r["slot"] == slot for r in rows) or any(
            r["event"] in {"response", "error"} and r["slot"] == slot for r in rows
        ):
            self.append("rejected", slot, dict(reason="orphan_response", response=response))
            raise ValueError("missing_request_start")
        self.append("error" if failed else "response", slot, response)

    def collect(self, request: Json, collector: Callable[[], Json]) -> Json:
        """A transport exception receives a durable terminal record and propagates."""
        self.start(request)
        try:
            response = collector()
        except Exception as error:
            self.finish(
                request["slot"],
                dict(error_type=type(error).__name__, error=str(error)),
                failed=True,
            )
            raise
        self.finish(request["slot"], response)
        return response

    def summary(self) -> Json:
        """Pending durable starts are censored; repeats never add independent units."""
        rows = self.events()
        attempted = len(
            {
                r["slot"]
                for r in rows
                if r["event"] == "start"
                or (r["event"] == "rejected" and r["payload"]["reason"] == "orphan_response")
            }
        )
        completed = sum(r["event"] == "response" for r in rows)
        failed = sum(r["event"] == "error" for r in rows)
        return dict(
            attempted=attempted,
            completed=completed,
            failed=failed,
            censored=attempted - completed - failed,
        )

    def seal(self) -> Json:
        """Bind replay to a separate immutable seal owned by the trusted parent."""
        target = self.root / "seal.json"
        value = dict(
            ledger=reference(self.path),
            summary=self.summary(),
            producer=reference(Path(__file__)),
            threat_model=THREAT_MODEL,
        )
        with target.open("x") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        return reference(target)

    @staticmethod
    def replay(ref: Json) -> Json:
        """An external byte pin defeats rewriting both events and their local hashes."""
        path = Path(ref["path"])
        if reference(path) != ref:
            raise ValueError("seal_bytes")
        value = json.loads(path.read_bytes())
        if reference(Path(value["ledger"]["path"])) != value["ledger"]:
            raise ValueError("ledger_bytes")
        ledger = Ledger(path.parent)
        if ledger.summary() != value["summary"]:
            raise ValueError("ledger_summary")
        return ledger.summary()


def namespace(
    raw: Path, view: Json, *, entrypoint: str = "worker.py", probes: Json | None = None
) -> Json:
    """Run only sealed worker code with numeric stdin and no host data mounts.

    Bubblewrap creates real namespaces. A failed launch is terminal for this
    invocation; neither the in-process guard nor a container fallback is used.
    """
    if entrypoint != "worker.py":
        raise ValueError("entrypoint")
    if (
        set(view) != {"features"}
        or not isinstance(view["features"], list)
        or not all(type(n) in (int, float) and math.isfinite(n) for n in view["features"])
    ):
        raise ValueError("numeric_view")
    private(raw)
    executable = shutil.which("bwrap")
    if executable is None:
        return dict(passed=False, reason="missing_namespace_runtime")
    worker = raw / "worker.py"
    with worker.open("x") as stream:
        stream.write(WORKER)
        stream.flush()
        os.fsync(stream.fileno())
    worker.chmod(0o444)
    argv = [executable, "--unshare-all", "--die-with-parent", "--new-session", "--clearenv"]
    for mount in ("/usr", "/lib", "/lib64"):
        argv += ["--ro-bind", mount, mount]
    argv += [
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/work",
        "--ro-bind",
        str(worker.absolute()),
        "/work/worker.py",
        "--chdir",
        "/work",
        "--uid",
        "65534",
        "--gid",
        "65534",
        "/usr/bin/python3",
        "-I",
        "/work/worker.py",
    ]
    invocation = Ledger(raw / "invocation")
    payload = json.dumps(dict(view, **(probes or {})), sort_keys=True)
    request = dict(
        slot="namespace",
        graph="fixture",
        shard="A",
        role="test",
        split="test",
        prompt=payload,
        model="scripted_no_model",
        seed=SEED,
    )
    invocation.start(request)
    progress("namespace_before")
    try:
        result = subprocess.run(
            argv,
            input=payload,
            text=True,
            capture_output=True,
            timeout=10,
            close_fds=True,
            cwd="/",
            env={},
        )
        response = dict(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        response = dict(exit_code=-1, error=str(error))
    invocation.finish("namespace", response, failed=response["exit_code"] != 0)
    sealed = invocation.seal()
    progress("namespace_after")
    observed = json.loads(response["stdout"]) if response["exit_code"] == 0 else {}
    parent = {k: os.readlink("/proc/self/ns/" + k) for k in ("mnt", "user", "pid", "net")}
    passed = (
        response["exit_code"] == 0
        and observed.get("uid") == 65534
        and set(observed.get("namespaces", {})) == set(parent)
        and all(observed.get("namespaces", {}).get(k) != v for k, v in parent.items())
    )
    receipt = dict(
        passed=passed,
        reason="namespace_enforced" if passed else "namespace_failed",
        observed=observed,
        parent_namespaces=parent,
        argv=argv,
        worker_reference=reference(worker),
        runtime_reference=reference(Path(executable)),
        ledger_reference=sealed,
        allow_fallback=False,
        mount_allowlist=["/usr", "/lib", "/lib64", "/proc", "/dev", "/tmp", "/work/worker.py"],
        environment={},
        close_fds=True,
    )
    atomic_json(raw / "enforcement.json", receipt)
    return receipt


def refused(call: Callable[[], Any]) -> bool:
    """Controls must observe a refusal rather than copy a claimed readiness field."""
    try:
        call()
    except ValueError:
        return True
    return False


def qualify(raw: Path) -> Json:
    """Exercise a real isolated scorer plus durable custody without any model call."""
    private(raw)
    progress("positive_fixture_before")
    secrets = raw / "secrets"
    private(secrets)
    paths = []
    for name in ("raw_response", "source_graph", "future_labels", "other_stream"):
        path = secrets / (name + ".json")
        atomic_json(path, dict(secret=name))
        paths.append(str(path.absolute()))
    ledger = Ledger(raw / "positive_ledger")
    request = dict(
        slot="private-A",
        graph="private-graph",
        shard="A",
        role="test",
        split="test",
        prompt="scripted fixture; no truth",
        model="scripted_no_model",
        seed=SEED,
    )
    response = ledger.collect(request, lambda: dict(raw_response="scripted", features=[1.0, 2.0]))
    with Path(paths[0]).open("rb") as secret:
        os.set_inheritable(secret.fileno(), True)
        receipt = namespace(
            raw / "namespace",
            dict(features=response["features"]),
            probes=dict(paths=paths, secret=paths[0], fd=secret.fileno()),
        )
    observed = receipt.get("observed", {})
    controls = dict(
        direct_paths=receipt["passed"]
        and observed.get("paths_denied", False)
        and observed.get("path_count") == len(paths),
        symlink_traversal=receipt["passed"] and observed.get("symlink_denied", False),
        inherited_descriptors=receipt["passed"] and observed.get("fd_denied", False),
        alternate_entrypoint=refused(
            lambda: namespace(raw / "alternate", dict(features=[1]), entrypoint="other")
        ),
        network=receipt["passed"] and observed.get("network_denied", False),
        missing_request_start=refused(lambda: ledger.finish("missing", dict(response="rejected"))),
        duplicate_slot=refused(lambda: ledger.start(request)),
        orphan_response=refused(lambda: ledger.finish("orphan", dict(response="retained orphan"))),
        early_label_reads=refused(lambda: namespace(raw / "early", dict(features=[1], truth=[0]))),
    )
    sealed = ledger.seal()
    saved = ledger.path.read_bytes()
    rows = ledger.events()
    ledger.path.write_text("\n".join(json.dumps(r) for r in reversed(rows)) + "\n")
    controls["reordered_ledger"] = refused(lambda: Ledger.replay(sealed))
    rows[0]["payload"]["seed"] += 1
    previous = canonical_hash([])
    for row in rows:
        row["previous"] = previous
        row["sha256"] = canonical_hash({k: v for k, v in row.items() if k != "sha256"})
        previous = row["sha256"]
    ledger.path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    controls["rehashed_ledger"] = refused(lambda: Ledger.replay(sealed))
    ledger.path.write_bytes(saved)
    replayed = Ledger.replay(sealed)
    inventory = dict(
        expected_worker_invocations=1,
        invocations=[receipt],
        observed_seals=sorted(str(p.absolute()) for p in raw.glob("*/invocation/seal.json")),
        observed_ledgers=sorted(str(p.absolute()) for p in raw.glob("*/invocation/ledger.jsonl")),
        complete_access_authority_scope="supervised_fixture_invocations_only",
    )
    expected_seals = [receipt["ledger_reference"]["path"]] if receipt["passed"] else []
    expected_ledgers = [str(Path(p).with_name("ledger.jsonl")) for p in expected_seals]
    inventory["passed"] = (
        inventory["observed_seals"] == expected_seals
        and inventory["observed_ledgers"] == expected_ledgers
        and receipt["passed"]
    )
    atomic_json(raw / "invocation_inventory.json", inventory)
    policy = dict(
        threat_model=THREAT_MODEL,
        request_fields=sorted(REQUEST_FIELDS),
        worker_reference=receipt.get("worker_reference"),
        namespace_policy=receipt.get("argv"),
        gate_field_schemas=dict(
            custody_ready_score="integer 0/1; actual checks",
            roster_ready_score="integer 0/1; authenticated complete exclusions and sealed roster",
        ),
        numeric_view_schema=dict(features="finite numeric vector; no paths or labels"),
        feedback_policy="Only protocol-scheduled parent release at query8,12,...,32; labels29..32 remain pending. No feedback is released in this fixture.",
    )
    atomic_json(raw / "policy.json", policy)
    result = dict(
        custody_ready_score=int(
            receipt["passed"]
            and all(controls.values())
            and replayed["completed"] == 1
            and inventory["passed"]
        ),
        ledger_reference=sealed,
        policy_reference=reference(raw / "policy.json"),
        invocation_inventory_reference=reference(raw / "invocation_inventory.json"),
        enforcement_receipts=[receipt],
        bypass_control_rows=[dict(name=k, passed=controls[k], fixture_only=True) for k in CONTROLS],
        complete_access_authority_scope="supervised_fixture_invocations_only",
        threat_model=THREAT_MODEL,
        current_llm_invocations=0,
    )
    atomic_json(raw / "qualification.json", result)
    progress("positive_fixture_after")
    return result


PINS = {
    "results/experiment_9069_v774_source_custody.json": "sha256:21a8e31edb572b34ea638fb78591b993e8c40e0855bfa4695ae3cbd9c4c295ee",
    "results/experiment_9055_v773_prospective_roster.json": "sha256:80aa684f1cc271897823983b66bc2d673073223413233438ad135a81edfbc867",
}


def gate(path: Path, field: str, expected: Any, observed: Any, check: str) -> Json:
    """Distinguish missing files, missing fields and measured zero in terminal gates."""
    return dict(
        check=check,
        upstream=path.stem,
        **reference(path),
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=type(observed) is type(expected) and observed == expected,
    )


def authenticate(root: Path, raw: Path) -> Json:
    """Authenticate original bytes before inspecting historical exclusion coverage.

    The original observed-path inventory is not a complete topology inventory.
    Excluding all V773 B is sufficient for uncertain attempts on that roster;
    it does not close unidentified attempts elsewhere. Missing source authority
    blocks roster construction rather than adding a completeness field today.
    """
    private(raw)
    progress("authentication_before")
    checks: list[Json] = []
    refs: list[Json] = []
    values: Json = {}
    for name, pin in PINS.items():
        path = root / name
        observed = reference(path)["sha256"] or "missing_file"
        checks.append(gate(path, "sha256", pin, observed, "declared_source_bytes"))
        if observed != pin:
            continue
        target = raw / path.name
        shutil.copyfile(path, target)
        refs.append(dict(reference(target), source_path=str(path.absolute()), producer=path.stem))
        value = json.loads(target.read_bytes())
        values[name] = value
        for ref in value["code_config_hashes"]:
            producer = root / Path(ref["path"]).relative_to(ROOT)
            checks.append(
                gate(
                    producer,
                    "sha256",
                    ref["sha256"],
                    reference(producer)["sha256"] or "missing_file",
                    "historical_producer_bytes",
                )
            )
    if len(values) == len(PINS) and all(r["passed"] for r in checks):
        roster = values[next(n for n in PINS if "9055" in n)]
        audit = values[next(n for n in PINS if "9069" in n)]
        checks.append(
            gate(
                root / next(n for n in PINS if "9055" in n),
                "fresh_source_ready_score",
                1,
                roster.get("fresh_source_ready_score", "missing_field"),
                "historical_roster_gate",
            )
        )
        inventory = dict(
            frozen_rosters=[],
            retained_proposal_count=0,
            excluded_v773_shards=["A", "B"],
            uncertain_v773_B_does_not_require_nonattempt=True,
            coverage="authenticated_observed_sources_only",
            global_completeness_claim=False,
        )
        for ref in roster["raw_shard_hashes"]:
            label = Path(ref.get("source_path", ref["path"])).name
            if label not in {
                "fresh_role_manifest.json",
                "historical_inventory.json",
                "construction.json",
            }:
                continue
            path = root / Path(ref["path"]).relative_to(ROOT)
            observed = reference(path)["sha256"] or "missing_file"
            checks.append(
                gate(path, "sha256", ref["sha256"], observed, "historical_roster_topology_bytes")
            )
            if observed != ref["sha256"]:
                continue
            target = raw / ("retained_" + label)
            shutil.copyfile(path, target)
            refs.append(
                dict(
                    reference(target),
                    source_path=str(path.absolute()),
                    producer="exp9055-prospective-roster",
                )
            )
            data = json.loads(target.read_bytes())
            if label == "fresh_role_manifest.json":
                inventory["frozen_rosters"].append(
                    dict(
                        reference=reference(target),
                        identities=[u["unit_id"] for u in data["graph_manifest"]],
                        excluded=True,
                    )
                )
            elif label == "historical_inventory.json":
                inventory["prior_graph_count"] = len(data["old_graphs"])
                inventory["prior_seed_count"] = len(data["old_seeds"])
            else:
                inventory["retained_proposal_count"] = len(data["proposal_rows"])
        inventory["observed_attempt_roots"] = audit.get("b_attempt_inventory", {}).get("roots", [])
        inventory["unresolved_operand"] = (
            "complete topology membership for attempts outside all excluded historical rosters"
        )
        atomic_json(raw / "historical_exclusion_observation.json", inventory)
        refs.append(reference(raw / "historical_exclusion_observation.json"))
        checks.append(
            gate(
                root / next(n for n in PINS if "9069" in n),
                "b_attempt_inventory.historical_attempt_topology_exclusion_reference",
                True,
                "missing_field",
                "historical_attempt_topology_coverage",
            )
        )
    progress("authentication_after")
    return dict(
        gate_check_summary=checks,
        source_references=refs,
        historical_attempt_exclusion_reference=refs[-1] if refs else None,
        historical_model_provenance=[
            dict(
                reference=reference(root / name),
                scope="historical_only; zero current invocation credit",
            )
            for name in PINS
        ],
    )


def build(work: Json, validation: Json) -> Json:
    """Recompute readiness from primitives; fixture success cannot close an external gap."""
    evidence = work["evidence"]
    checks = evidence["gate_check_summary"]
    failures = [r for r in checks if not r["passed"]]
    qualification = work["qualification"]
    passed = validation["passed"]
    cls = "disqualified" if not passed else "blocked" if failures else "null"
    rows = [
        dict(
            unit_id=f"gate-{i}",
            arm="precondition",
            condition=r["check"],
            family="custody",
            group_id=r["path"],
            seed=SEED,
            metric="precondition_authenticated",
            value=int(r["passed"]),
            absolute_metric=int(r["passed"]),
            numerator=int(r["passed"]),
            denominator=1,
            status="completed",
            exclusion_reason=None if r["passed"] else r["observed"],
        )
        for i, r in enumerate(checks)
    ]
    value = dict(
        experiment_id=9083,
        task_id=TASK,
        milestone="2026.11.775",
        run_date="20261115",
        honest_verdict="complete_"
        + cls
        + "_"
        + (failures[0]["check"] if failures else "prospective_custody"),
        verdict_class=cls,
        status="error" if cls == "disqualified" else "blocked" if cls == "blocked" else "success",
        verifier_is_oracle=True,
        independent_generalization_score=0,
        claim_scope="Finite supervised custody qualification only. Exact graph truth is an oracle; any graph success would be circular_positive.",
        exposure_scope="Private scripted fixture and historical metadata only. Human LPHB remains exposed development; no independent/generalized claim.",
        MODEL_SPECS=[],
        model_specs=[],
        inference_substrate="metadata_and_scripted_private_fixture",
        inference_substrate_class="no_model_load",
        current_llm_invocations=0,
        model_invoked=False,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        minimum_duration_s=0,
        rows=rows,
        intended_count=len(rows),
        attempted_count=len(rows),
        completed_count=len(rows),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        effective_independent_groups=0,
        collection_counts=dict(
            intended=1024, attempted=0, completed=0, failed=0, censored=0, unattempted=1024
        ),
        custody_ready_score=0,
        roster_ready_score=0,
        complete_access_authority_scope="supervised_fixture_invocations_only; historical completeness unavailable",
        threat_model=THREAT_MODEL,
        enforcement_receipts=qualification.get("enforcement_receipts", []),
        bypass_control_rows=qualification.get("bypass_control_rows", []),
        fixture_custody_ready_score=qualification.get("custody_ready_score", 0),
        authenticated_access_manifest_reference=qualification.get("ledger_reference"),
        invocation_inventory_reference=qualification.get("invocation_inventory_reference"),
        worker_code_namespace_policy_reference=qualification.get("policy_reference"),
        collection_manifest_reference=qualification.get("policy_reference"),
        roster_reference=None,
        split_hash=None,
        online_order_hash=None,
        proposal_count=0,
        proposal_rejection_rows=[],
        historical_attempt_exclusion_reference=evidence["historical_attempt_exclusion_reference"],
        proposal_contract=dict(
            seed=SEED,
            maximum_proposals=100000,
            target_identities=1024,
            shards=dict(A=512, B=512),
            per_shard=dict(calibration=64, test=64, online=384),
            independent_streams=24,
            queries_per_stream=32,
            strata="Exp9055 unchanged graph source/cardinality",
            replacement_allowed=False,
            construction_started=False,
        ),
        gate_check_summary=checks,
        code_config_hashes=work["code_config_hashes"],
        source_references=evidence["source_references"],
        source_artifact_hashes=evidence["source_references"],
        historical_model_provenance=evidence["historical_model_provenance"],
        current_work_receipts=work["current_work_receipts"],
        raw_shard_hashes=work["raw_references"],
        duration_s=work["duration_s"],
        random_seed=SEED,
        validation_receipts=validation["receipts"],
        validation_scope_manifest=validation.get("scope", {}),
        owned_statement_coverage=validation.get("coverage", {}),
        repository_health=work.get("repository_health", {}),
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        acceptance_gates=dict(
            owned_checks=passed,
            enforced_fixture=qualification.get("custody_ready_score") == 1,
            historical_exclusions=not failures,
            fresh_roster=False,
        ),
        methodology_note="Authenticate source and producer bytes; qualify actual OS denial controls and fsynced chained request custody. Missing out-of-roster historical attempt topology authority blocks all new proposals and collection. No LLM is loaded and V773 A/B are never reused.",
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Byte-authenticated primitives and current validation govern this finite custody claim; external gaps block readiness.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value
