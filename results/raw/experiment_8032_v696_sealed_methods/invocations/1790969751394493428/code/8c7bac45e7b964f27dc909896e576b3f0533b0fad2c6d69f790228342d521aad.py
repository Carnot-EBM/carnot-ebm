"""REQ-REPORT-8032: bind current custody without claiming unseen public labels.

This entry point seals methods for later experiments. It does not score a
language model or train a head. Private workers test the actual access rules.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import random
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from carnot.experiment_artifacts import artifact_output_root
from carnot import experiment_8019_v695_eligible_targets as prior
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.verify.evidence_features_7980 import normalized
from carnot.verify.learning_retention_audit_8026 import action, design

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8032_v696_sealed_methods"
SCRIPT = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_sealed_methods_8032.py"
OWNED = [f"python/carnot/{NAME}.py", SCRIPT]
SLOTS = prior.SLOTS
LEARNING_ARMS = ("recent64", "cumulative", "newest16", "frozen_no_write")
PUBLIC = {
    "family_id",
    "role",
    "slot",
    "source_bytes",
    "answer_bytes",
    "features",
    "q",
    "public_eligible",
    "exclusion_reason",
}
METHODS: Json = dict(
    likelihood=dict(
        roles=dict(fit=64, tune=32, evaluation=96),
        views=["full", "duplicate", "no_source"],
        input_tokens=6000,
        answer_tokens=384,
        truncation=False,
        replacements=False,
        target="original_full_source_answer",
        duplicate_tolerance=1e-6,
        qualification_before_calibration=True,
    ),
    learning=dict(
        delay=20,
        block=16,
        updates_per_block=4,
        maximum_updates=64,
        seeds=list(range(101, 121)),
        learning_rate=0.01,
        l2=0.001,
        arms=list(LEARNING_ARMS),
        selection="uniform_without_replacement_within_block",
        repeated_use="count IDs reused across later blocks",
        terminal_flush=False,
        unknown_targets="exclude_without_slot_compression",
        new_capture_gate=False,
    ),
    costs=dict(
        unsupported_accept=5,
        supported_reject=1,
        escalation=0.5,
        correct=0,
        probability="unsupported",
        decision="minimum_expected_cost",
        ties="escalate",
    ),
    statistics=dict(
        primary_comparisons=[
            "recent64_vs_cumulative",
            "recent64_vs_newest16",
            "recent64_vs_frozen_no_write",
        ],
        holm_family=3,
        alpha=0.05,
        draws=10000,
        blocks=[32, 16, 64],
        later_start=36,
        gain=0.02,
        no_false_accept_increase=True,
        cost_drift=0.01,
        brier_drift=0.005,
        seed_averaging="within original slot before paired block bootstrap",
        independent_n="source groups; seeds and duplicates add zero",
    ),
    support_floors={r: dict(minimum=m, per_class=c) for r, (m, c) in prior.FLOORS.items()},
    overlap=dict(
        fit_only=True,
        quantiles=[0.25, 0.5, 0.75],
        excluded_columns=[0, 1],
        method="mean fit activation prevalence on active cubic spline columns",
        quartile_ties="searchsorted_right",
        descriptive_only=True,
    ),
    access=dict(
        scoring="public bytes only",
        fitter="fit feedback only",
        learner="released stream feedback only",
        evaluator="vault only after immutable protocol",
        retention="evaluator only",
        release="original slot+20 after durable issue",
    ),
)
SOURCE_MAP = [
    dict(
        source="CoRun",
        version="2608.14376v1",
        url="https://arxiv.org/html/2608.14376v1",
        code="likelihood.duplicate_tolerance",
        limit="Another serving stack; no llama.cpp parity or drift cause established.",
    ),
    dict(
        source="LLM-42",
        version="2601.17768; abstract checked 20261002",
        url="https://arxiv.org/abs/2601.17768",
        code="carnot.inference.fixed_answer_likelihood_8022.target_likelihood",
        limit="Token alignment check; no speculative-generation reproduction.",
    ),
    dict(
        source="GASP",
        version="2607.04223v1",
        url="https://arxiv.org/html/2607.04223v1",
        code="likelihood.views",
        limit="Source dependence does not establish factual correctness.",
    ),
    dict(
        source="To Retain or to Adapt?",
        version="2607.05609v1",
        url="https://arxiv.org/html/2607.05609v1",
        code="select; learning.arms",
        limit="Finite exposed development stream; no transferred theorem or deployment claim.",
    ),
]
UPSTREAM = {
    8019: prior.NAME,
    8020: "experiment_8020_v695_qualified_energy_fit",
    8022: "experiment_8022_v695_likelihood_protocol",
    8025: "experiment_8025_v695_causal_online_updates",
    8026: "experiment_8026_v695_learning_retention_audit",
}
BEGAN = time.monotonic()


def progress(phase: str, units: int = 0, pending: int = 0) -> None:
    """Show completed work so a quiet child is never mistaken for a finished run."""
    print(
        f"[exp8032] phase={phase} elapsed_s={time.monotonic() - BEGAN:.3f} completed={units} pending={pending}",
        flush=True,
    )


def immutable(path: Path, value: Json) -> Json:
    """Refuse different bytes at a frozen identity instead of overwriting history."""
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("immutable_bytes")
    atomic_json(path, value)
    path.chmod(0o400)
    return dict(path=str(path), sha256=sha256_file(path))


def select(released: list[Json], arm: str, seed: int, block: int) -> list[Json]:
    """Uniform sampling has a fresh block seed; later blocks may reuse an ID."""
    if arm not in LEARNING_ARMS:
        raise ValueError("arm_contract")
    if arm == "frozen_no_write":
        return []
    pool = (
        released[-64:] if arm == "recent64" else released[-16:] if arm == "newest16" else released
    )
    if len(pool) < 4 or len({r["family_id"] for r in pool}) != len(pool):
        raise ValueError("selection_support")
    rng = random.Random(int(canonical_hash(dict(seed=seed, block=block)).split(":")[1], 16))
    return rng.sample(pool, 4)


def public_worker(rows: list[Json]) -> Json:
    """Only source and answer bytes enter this public admission interface."""
    for row in rows:
        if set(row) != PUBLIC:
            raise ValueError("public_worker_fields")
        if row["public_eligible"] and (
            not bytes.fromhex(row["source_bytes"]) or not bytes.fromhex(row["answer_bytes"])
        ):
            raise ValueError("incomplete_public_bytes")
    return dict(
        completed=len(rows),
        labels_available=False,
        scope="public admission only; no scoring or fitting",
    )


def evaluator(
    protocol: Path, vault: Path, mode: str, *, issue: Path | None = None, now: int = -1
) -> Json:
    """Validate protocol before touching a vault; retention never enters feedback."""
    if (
        not protocol.is_file()
        or json.loads(protocol.read_text()).get("methods") != METHODS
        or not json.loads(protocol.read_text()).get("frozen")
    ):
        raise ValueError("protocol_before_vault")
    data = json.loads(vault.read_text())
    rows = data["rows"]
    if any(
        "eligible_y" not in r
        or (
            r["eligible_y"] is not None
            and (type(r["eligible_y"]) is not int or r["eligible_y"] not in (0, 1))
        )
        for r in rows
    ):
        raise ValueError("target_contract")
    if mode == "support":
        return dict(
            role=data["role"],
            class_counts={str(y): sum(r["eligible_y"] == y for r in rows) for y in (0, 1)},
            unknown_target=sum(r["eligible_y"] is None for r in rows),
        )
    if data["role"] != "stream":
        raise ValueError("retention_or_fit_feedback_forbidden")
    if issue is None or not issue.is_file() or not json.loads(issue.read_text()).get("durable"):
        raise ValueError("durable_issue")
    issued = json.loads(issue.read_text())
    row = next(r for r in rows if r["family_id"] == issued["family_id"])
    if now != issued["slot"] + 20 or row["slot"] != issued["slot"]:
        raise ValueError("release_slot")
    return dict(
        family_id=row["family_id"],
        y=row["eligible_y"],
        original_slot=issued["slot"],
        released_at=now,
        excluded=row["eligible_y"] is None,
    )


class Contract(ValueError):
    """A failed external operand is terminal evidence, not an unfinished task."""

    def __init__(self, path: Path, field: str, expected: Any, observed: Any):
        self.gate = dict(
            upstream_id=path.stem,
            path=str(path),
            sha256=sha256_file(path) if path.is_file() else None,
            artifact_field=field,
            expected=expected,
            observed=observed,
            passed=False,
            check_name=field,
        )
        super().__init__(field)


def require(path: Path, field: str, expected: Any, observed: Any) -> None:
    """Keep both operands so downstream readers can diagnose a blocked branch."""
    if expected != observed:
        raise Contract(path, field, expected, observed)


def copy_bound(ref: Json, raw: Path, scope: str = "custody") -> Json:
    """Copy original bytes into durable custody only after their hash agrees."""
    path = Path(ref["path"])
    require(
        path,
        "sha256",
        ref["sha256"],
        sha256_file(path) if path.is_file() else "MISSING_CONTRACT_FIELD",
    )
    target = raw / scope / (ref["sha256"].split(":")[-1] + path.suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        require(target, "sha256", ref["sha256"], sha256_file(target))
    else:
        with target.open("wb") as stream:
            stream.write(path.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        target.chmod(0o400)
    return dict(path=str(target), sha256=sha256_file(target), original_path=str(path))


def primary(root: Path, identity: int, raw: Path, refs: list[Json]) -> Json:
    """A current terminal receipt must bind the primary and a passed validator."""
    path = root / "results" / (UPSTREAM[identity] + ".json")
    require(path, "primary_exists", True, path.is_file())
    value: Json = json.loads(path.read_text())
    require(path, "experiment_id", identity, value.get("experiment_id", "MISSING_CONTRACT_FIELD"))
    require(
        path, "terminal_validation_sidecar_path", True, "terminal_validation_sidecar_path" in value
    )
    terminal_path = Path(value["terminal_validation_sidecar_path"])
    require(terminal_path, "terminal_exists", True, terminal_path.is_file())
    terminal = json.loads(terminal_path.read_text())
    binding = terminal.get("publication", terminal)
    sidecar = Path(binding["sidecar_path"])
    require(sidecar, "sidecar_exists", True, sidecar.is_file())
    report = json.loads(sidecar.read_text())
    for source, obj in ((terminal_path, binding), (sidecar, report)):
        require(source, "primary_sha256", sha256_file(path), obj.get("primary_sha256"))
        require(source, "primary_path", str(path), obj.get("primary_path"))
    require(sidecar, "report.passed", True, report.get("report", {}).get("passed"))
    for p in (path, terminal_path, sidecar):
        refs.append(copy_bound(dict(path=str(p), sha256=sha256_file(p)), raw))
    for ref in value.get("code_config_hashes", []):
        refs.append(copy_bound(ref, raw))
    if identity == 8022:
        for ref in value.get("raw_shard_hashes", []):
            if "/forwards/" in ref["path"] or Path(ref["path"]).name == "capture.json":
                refs.append(copy_bound(ref, raw))
    return value


def panel_rows(panel: Json) -> list[Json]:
    """Keep existing admitted identities; no truth label can select a replacement."""
    rows, groups, counts = [], set(), {r: 0 for r in METHODS["likelihood"]["roles"]}
    for item in panel["rows"]:
        row = {
            k: item[k]
            for k in ("family_id", "role", "source_bytes", "answer_bytes", "target_tokens")
        }
        if (
            not row["source_bytes"]
            or not row["answer_bytes"]
            or not 1 <= len(row["target_tokens"]) <= 384
        ):
            raise ValueError("panel_complete_bytes")
        group = normalized(bytes.fromhex(row["source_bytes"]))
        if group in groups:
            raise ValueError("source_role_overlap")
        groups.add(group)
        views = {a: item["views"][a] for a in METHODS["likelihood"]["views"]}
        if (
            views["duplicate"] != views["full"]
            or views["full"]["source_bytes"] != row["source_bytes"]
            or views["no_source"]["source_bytes"] != ""
        ):
            raise ValueError("duplicate_source_views")
        for view in views.values():
            if (
                view["input_tokens"] > 6000
                or view["tokens"][view["response_start"] :] != row["target_tokens"]
            ):
                raise ValueError("panel_token_admission")
        counts[row["role"]] += 1
        rows.append(
            dict(
                row,
                views=views,
                normalized_source=group,
                exact_byte_hash=canonical_hash(row["source_bytes"]),
            )
        )
    if counts != METHODS["likelihood"]["roles"]:
        raise ValueError("panel_role_counts")
    return rows


def overlap(public: Json, head: Json) -> Json:
    """Only public fit activations determine the quartiles and source masks."""
    masks = {
        role: [
            dict(
                family_id=r["family_id"],
                slot=r["slot"],
                columns=np.flatnonzero(design(head, r)[2:] != 0).tolist(),
            )
            for r in rows
            if r["public_eligible"]
        ]
        for role, rows in public.items()
    }
    fit = np.zeros((len(masks["fit"]), 108))
    for i, row in enumerate(masks["fit"]):
        fit[i, row["columns"]] = 1
    prevalence = fit.mean(axis=0)
    cuts = np.quantile(
        [float(np.mean(prevalence[r["columns"]])) for r in masks["fit"]], [0.25, 0.5, 0.75]
    ).tolist()
    return dict(
        cutpoints=cuts,
        masks=masks,
        fit_prevalence=prevalence.tolist(),
        excluded_shared_columns=[0, 1],
    )


def seal(root: Path, raw: Path) -> Json:
    """Seal public methods before the evaluator opens any current annotation shard."""
    progress("authenticate_inputs")
    refs: list[Json] = []
    failures: list[Json] = []
    values: dict[int, Json] = {}
    for n in UPSTREAM:
        try:
            values[n] = primary(root, n, raw, refs)
        except Contract as error:
            error.gate["branch"] = (
                "likelihood" if n == 8022 else "historical" if n == 8026 else "learning"
            )
            failures.append(error.gate)
    public, role_refs, access, support = {}, {}, [], {}
    likelihood, head, ov = [], {}, dict(cutpoints=[], masks={})
    try:
        if 8019 in values:
            for role in SLOTS:
                ref = copy_bound(values[8019]["public_manifests"][role], raw, "public")
                refs.append(ref)
                original = json.loads(Path(ref["path"]).read_text())["rows"]
                public[role] = [{k: r[k] for k in PUBLIC} for r in original]
                require(
                    Path(ref["path"]),
                    "original_slots",
                    list(range(SLOTS[role])),
                    [r["slot"] for r in public[role]],
                )
                public_worker(public[role])
                role_refs[role] = immutable(
                    raw / "public" / (role + ".json"), dict(rows=public[role], role=role)
                )
        if 8022 in values:
            try:
                likelihood = panel_rows(values[8022]["public_panel_manifest"])
            except (ValueError, KeyError) as error:
                gate = Contract(
                    root / "results" / (UPSTREAM[8022] + ".json"),
                    str(error),
                    "valid frozen likelihood panel",
                    str(error),
                ).gate
                gate["branch"] = "likelihood"
                failures.append(gate)
        if 8020 in values:
            require(
                root / "results" / (UPSTREAM[8020] + ".json"),
                "energy_fit_ready_score",
                1,
                values[8020].get("energy_fit_ready_score"),
            )
            progress("before_small_head_load")
            for r in values[8020]["head_checkpoints"]:
                ref = copy_bound(r, raw)
                h = json.loads(Path(ref["path"]).read_text())
                refs.append(ref)
                if h["arm"] == "conditioned_energy" and h["seed"] == 17:
                    head = h
            require(
                raw,
                "qualified_head",
                True,
                bool(head) and head["converged"] and len(head["parameters"]) == 110,
            )
            calibration_ref = copy_bound(values[8020]["calibration_checkpoint"], raw)
            refs.append(calibration_ref)
            calibration = json.loads(Path(calibration_ref["path"]).read_text())["maps"][
                "conditioned_energy-17"
            ]
            require(
                Path(calibration_ref["path"]),
                "calibration.converged",
                True,
                calibration["converged"],
            )
            head["calibration"] = calibration["parameters"]
            progress("after_small_head_load")
            if public:
                ov = overlap(public, head)
        if 8025 in values and 8019 in values:
            expected = values[8019]["public_manifests"]["stream"]["sha256"]
            require(
                root / "results" / (UPSTREAM[8025] + ".json"),
                "cited_upstream_artifacts.stream.sha256",
                True,
                any(r["sha256"] == expected for r in values[8025]["cited_upstream_artifacts"]),
            )
    except Contract as error:
        failures.append(error.gate)
    except (ValueError, KeyError) as error:
        failures.append(
            Contract(
                root / "results" / (UPSTREAM[8022] + ".json"),
                str(error),
                "valid frozen public contract",
                str(error),
            ).gate
        )
    code = [
        copy_bound(dict(path=str(ROOT / p), sha256=sha256_file(ROOT / p)), raw, "code")
        for p in OWNED
    ]
    protocol = dict(
        methods=METHODS,
        original_role_order=list(SLOTS),
        head=head,
        frozen=True,
        frozen_at_ns=time.time_ns(),
        frozen_support_by_role=values.get(8019, {}).get("support_by_role", {}),
        public=role_refs,
        likelihood=likelihood,
        overlap=ov,
        code=code,
        upstream=refs,
        target_exclusions="inherited complete-target masks; unknown never supported",
    )
    methods_ref = immutable(raw / "methods.json", protocol)
    progress("protocol_frozen", sum(map(len, public.values())), 0)
    evaluator_refs = {}
    if 8019 in values:
        for role in SLOTS:
            if role not in public:
                continue
            try:
                ref = copy_bound(values[8019]["evaluator_manifests"][role], raw, "evaluator")
                evaluator_refs[role] = ref
                progress("before_evaluator_support_" + role)
                opened_at_ns = time.time_ns()
                summary = evaluator(Path(methods_ref["path"]), Path(ref["path"]), "support")
                summary.update(
                    intended=SLOTS[role],
                    eligible=sum(summary["class_counts"].values()),
                    public_eligible=sum(r["public_eligible"] for r in public[role]),
                )
                floor = METHODS["support_floors"][role]
                summary["passed"] = (
                    summary["eligible"] >= floor["minimum"]
                    and min(summary["class_counts"].values()) >= floor["per_class"]
                )
                support[role] = summary
                access.append(
                    dict(
                        sequence=len(access) + 1,
                        role=role,
                        scope="support_only",
                        protocol_sha256=methods_ref["sha256"],
                        vault=ref,
                        protocol_frozen_before_access=opened_at_ns > protocol["frozen_at_ns"],
                        opened_at_ns=opened_at_ns,
                        labels_returned_to_workers=False,
                    )
                )
                require(Path(ref["path"]), "support_by_role." + role, True, summary["passed"])
                progress("after_evaluator_support_" + role)
            except Contract as error:
                failures.append(error.gate)
    rows = [
        dict(
            family_id=r["family_id"],
            role=role,
            original_slot=r["slot"],
            normalized_source=normalized(bytes.fromhex(r["source_bytes"])),
            exact_byte_hash=canonical_hash([r["source_bytes"], r["answer_bytes"]]),
            eligible=r["public_eligible"],
            status="sealed" if r["public_eligible"] else "excluded",
            exclusion_reason=r["exclusion_reason"],
            numerator=None,
            denominator=None,
            measured=False,
        )
        for role, records in public.items()
        for r in records
    ]
    row_ref = immutable(raw / "public_rows.json", dict(rows=rows))
    historic = [
        dict(
            experiment_id=n,
            run_date=v["run_date"],
            honest_verdict=v["honest_verdict"],
            verdict_class=v["verdict_class"],
            sha256=sha256_file(root / "results" / (UPSTREAM[n] + ".json")),
            gate_check_summary=v.get("gate_check_summary", []),
        )
        for n, v in values.items()
    ]
    for n in values:
        previous = root / "results" / "raw" / UPSTREAM[n] / "failure_logs" / "first-attempt.json"
        if previous.is_file():
            ref = copy_bound(dict(path=str(previous), sha256=sha256_file(previous)), raw)
            old = json.loads(Path(ref["path"]).read_text())
            historic.append(
                dict(
                    experiment_id=n,
                    run_date=old["run_date"],
                    honest_verdict=old["honest_verdict"],
                    verdict_class=old["verdict_class"],
                    reference=ref,
                    previous_attempt=True,
                )
            )
    result = dict(
        methods_reference=methods_ref,
        row_reference=row_ref,
        rows=rows,
        gate_check_summary=failures,
        role_manifests=dict(public=role_refs, evaluator=evaluator_refs),
        evaluator_access_log=access,
        historical_exposure=dict(
            all_roles_development_exposed=True,
            newly_unseen=False,
            prior_verdicts=historic,
            earlier_failure_logs=values.get(8019, {}).get("historical_failure_logs", []),
            earlier_exposure=values.get(8019, {}).get("exposure_rows", []),
            pretraining_exposure_unknown=True,
        ),
        overlap_cutpoints=ov["cutpoints"],
        support_by_role=support,
        method_source_map=SOURCE_MAP,
        cited_upstream_artifacts=refs,
        code_config_hashes=code,
        likelihood_panel_ready_score=int(
            len(likelihood) == 192 and not any(g.get("branch") != "learning" for g in failures)
        ),
        learning_inputs_ready_score=int(
            bool(head)
            and set(support) == set(SLOTS)
            and all(r["passed"] for r in support.values())
            and not any(g.get("branch") != "likelihood" for g in failures)
        ),
    )
    result["raw_shard_hashes"] = [
        dict(path=str(p), sha256=sha256_file(p)) for p in sorted(raw.rglob("*")) if p.is_file()
    ]
    return result


def replay(path: Path) -> Json:
    """Recompute public rows and fit cutpoints from durable frozen input bytes."""
    value = json.loads(path.read_text())
    for ref in value["raw_shard_hashes"]:
        require(Path(ref["path"]), "sha256", ref["sha256"], sha256_file(Path(ref["path"])))
    protocol = json.loads(Path(value["methods_reference"]["path"]).read_text())
    require(path, "methods", METHODS, protocol["methods"])
    saved = json.loads(Path(value["row_reference"]["path"]).read_text())
    if (
        saved["rows"] != value["rows"]
        or protocol["overlap"]["cutpoints"] != value["overlap_cutpoints"]
    ):
        raise ValueError("reduction_drift")
    if protocol["head"] and protocol["public"]:
        public = {
            role: json.loads(Path(ref["path"]).read_text())["rows"]
            for role, ref in protocol["public"].items()
        }
        require(path, "overlap_reduction", overlap(public, protocol["head"]), protocol["overlap"])
    if protocol["likelihood"]:
        panel_rows(dict(rows=protocol["likelihood"]))
    for event in value["evaluator_access_log"]:
        require(
            path, "protocol_sha256", value["methods_reference"]["sha256"], event["protocol_sha256"]
        )
        require(path, "protocol_frozen_before_access", True, event["protocol_frozen_before_access"])
        require(
            path,
            "chronological_vault_access",
            True,
            event["opened_at_ns"] > protocol["frozen_at_ns"],
        )
        role = event["role"]
        summary = evaluator(
            Path(value["methods_reference"]["path"]), Path(event["vault"]["path"]), "support"
        )
        require(
            path,
            "support_by_role." + role + ".class_counts",
            summary["class_counts"],
            value["support_by_role"][role]["class_counts"],
        )
    if value.get("protocol_ready_score", 0) and (
        value["gate_check_summary"] or value["verdict_class"] not in {"null", "positive"}
    ):
        raise ValueError("unsafe_readiness")
    return dict(
        passed=True,
        completed=len(saved["rows"]),
        scope="current custody; no scientific measurement",
    )


def validation_plan(scratch: Path) -> list[CommandSpec]:
    """Reuse the bounded runner and existing E2E checks with added-code coverage."""
    commands = prior.validation_plan(scratch)
    (scratch / "coverage.ini").write_text(
        "[run]\nparallel = True\ndata_file = "
        + str(scratch / ".coverage")
        + "\ninclude =\n    "
        + "\n    ".join(str(ROOT / p) for p in OWNED)
        + "\n"
    )
    return [
        replace(
            c,
            argv=tuple(a.replace(prior.NAME, NAME).replace(prior.TEST, TEST) for a in c.argv),
            timeout_s=180 if c.scope == "repository_health" else c.timeout_s,
        )
        for c in commands
    ]


def terminal(path: Path) -> Json:
    """Cold replay and the shared validators cover both candidate and published bytes."""
    py = str(ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "cold_reduction",
            (py, "-u", str(ROOT / SCRIPT), "--cold-replay", str(path)),
            "terminal",
            120,
        ),
        CommandSpec(
            "adversarial",
            (py, "scripts/adversarial_verify.py", str(path), "--json"),
            "terminal",
            120,
        ),
        CommandSpec(
            "strict_rows",
            (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)),
            "terminal",
            60,
        ),
    ]
    raw = path.parent if path.parent.name != "results" else path.parent / "raw" / NAME
    receipts = run_commands(
        ROOT,
        commands,
        log_dir=raw / "terminal_logs" / sha256_file(path).split(":")[-1],
        heartbeat_s=30,
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def finish(value: Json, raw: Path, receipts: list[Json], counts: Json, duration: float) -> Json:
    """Only owned checks qualify custody; historical exposure always bars benefit."""
    owned = [r for r in receipts if r["scope"] == "owned"]
    good = (
        bool(owned)
        and all(r["passed"] for r in owned)
        and set(counts) == set(OWNED)
        and all(r["num_statements"] > 0 and r["missing_lines"] == 0 for r in counts.values())
    )
    blocked = bool(value["gate_check_summary"])
    ready = int(
        good
        and not blocked
        and value["likelihood_panel_ready_score"]
        and value["learning_inputs_ready_score"]
    )
    kind = "blocked" if blocked else "null" if good else "disqualified"
    n = len(value["rows"])
    eligible = sum(r["eligible"] for r in value["rows"])
    value.update(
        experiment_id=8032,
        task_id="exp8032-sealed-methods",
        milestone="2026.10.696",
        schema="carnot.v696.sealed_methods.v1",
        run_date="20261002",
        claim_scope="This invocation seals current custody and future methods on historically exposed development evidence. No model scoring, head fitting, learning trajectory or deployment benefit measured.",
        honest_verdict="complete_" + kind + "_sealed_methods",
        verdict_class=kind,
        protocol_ready_score=ready,
        random_seed=69632,
        reproducibility_checksum=canonical_hash(
            dict(
                methods=value["methods_reference"],
                code=value["code_config_hashes"],
                inputs=value["cited_upstream_artifacts"],
            )
        ),
        sample_size_budget=dict(
            unit="original public source slot custody",
            intended=704,
            eligible=eligible,
            completed=n,
            excluded=n - eligible,
            failed=0,
            censored=704 - n,
            independent=len({r["normalized_source"] for r in value["rows"] if r["eligible"]}),
        ),
        intended_count=704,
        eligible_count=eligible,
        completed_count=n,
        excluded_count=n - eligible,
        failed_count=0,
        censored_count=704 - n,
        independent_count=len({r["normalized_source"] for r in value["rows"] if r["eligible"]}),
        checkpoint_references=[value["methods_reference"], value["row_reference"]],
        validation_receipts=owned,
        repository_health=[r for r in receipts if r["scope"] == "repository_health"],
        coverage_statement_counts=counts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        flagged_adversarial=False,
        acceptance_gate_results=dict(
            owned_checks=good, protocol_sealed=True, scientific_benefit=False
        ),
        verifier_is_oracle=False,
        genuine_headroom=dict(measured=False),
        positive_control_results=dict(
            sentinel_rejected=True, premature_vault_rejected=True, scope="private access controls"
        ),
        generalized_learning_benefit_score=0,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        substrate_declaration=dict(
            custody="aggregation_from_upstream_artifacts",
            numerical_work="verifier_scoring",
            inference_substrate_class="no_model_load",
            pretrained_model_calls=0,
        ),
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        duration_s=duration,
        phase_spans=[dict(name="owned_validation_and_current_custody", duration_s=duration)],
    )
    value["field_principles"] = {
        k: "Bind this field to current durable custody; imported activity and exposed development labels give no current model or benefit credit."
        for k in value
    }
    return value


def main(argv: list[str] | None = None) -> int:
    """Run current custody or a separate public/evaluator interface from any directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261002")
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--public-worker", choices=["scoring", "fitter", "learner"])
    parser.add_argument("--public-input", type=Path)
    parser.add_argument("--evaluator-worker", choices=["support", "release"])
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--vault", type=Path)
    parser.add_argument("--issue", type=Path)
    parser.add_argument("--now", type=int, default=-1)
    args = parser.parse_args(argv)
    progress("start")
    started = time.monotonic()
    try:
        if args.date != "20261002":
            raise ValueError("run_date")
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        if args.public_worker:
            print(
                json.dumps(public_worker(json.loads(args.public_input.read_text())["rows"])),
                flush=True,
            )
            return 0
        if args.evaluator_worker:
            print(
                json.dumps(
                    evaluator(
                        args.protocol,
                        args.vault,
                        args.evaluator_worker,
                        issue=args.issue,
                        now=args.now,
                    )
                ),
                flush=True,
            )
            return 0
        output = artifact_output_root(root=ROOT) / (NAME + ".json")
        raw = output.parent / "raw" / NAME
        invocation = raw / "invocations" / str(time.time_ns())
        with tempfile.TemporaryDirectory(prefix="carnot-8032-") as temporary:
            scratch = Path(temporary)
            progress("private_negative_controls")
            for operation in (
                lambda: public_worker([dict(y="PRIVATE_SENTINEL")]),
                lambda: evaluator(
                    scratch / "absent-protocol", scratch / "unopened-vault", "support"
                ),
            ):
                rejected = False
                try:
                    operation()
                except ValueError:
                    rejected = True
                assert rejected, "negative_access_control"
            receipts = (
                []
                if args.fixture_root
                else run_commands(
                    ROOT,
                    validation_plan(scratch),
                    log_dir=raw / "validation_logs",
                    extra_env={
                        "CARNOT_8032_COVERAGE_CONFIG": str(scratch / "coverage.ini"),
                        "COVERAGE_FILE": str(scratch / ".coverage-health"),
                    },
                    heartbeat_s=30,
                )
            )
            counts = (
                {
                    p: r["summary"]
                    for p, r in json.loads((scratch / "coverage.json").read_text())
                    .get("files", {})
                    .items()
                    if p in OWNED
                }
                if (scratch / "coverage.json").is_file()
                else {}
            )
            value = seal(args.fixture_root or ROOT, invocation)
            value = finish(value, raw, receipts, counts, time.monotonic() - started)
            progress("before_publication", len(value["rows"]), 0)
            validator = replay if args.fixture_root else terminal
            publication = publish_primary(output, value, validator)
            post = validator(output)
            if not post["passed"]:
                raise ValueError("published_validation")
            atomic_json(
                raw / "terminal_validation.json", dict(publication=publication, published=post)
            )
            progress("complete", len(value["rows"]), 0)
            print(json.dumps(publication), flush=True)
        return 0
    except (ValueError, KeyError, OSError) as error:
        progress("rejected_" + str(error))
        return 1
