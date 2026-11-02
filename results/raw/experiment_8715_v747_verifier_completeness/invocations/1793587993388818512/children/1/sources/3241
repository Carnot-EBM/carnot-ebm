"""REQ-REPORT-8019: separate public source admission from complete human targets.

The manifests prepare historically exposed development evidence. No model or
small head runs, and readiness gives no credit for natural learning benefit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify.evidence_features_7980 import normalized
from carnot.verify.sentence_labels_7942 import digest, index_unique

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8019_v695_eligible_targets"
TASK = "exp8019-eligible-targets"
MODEL_SPECS: list[str] = []
SLOTS = dict(fit=256, tune=64, calibration=64, stream=256, retention=64)
FLOORS = dict(fit=(128, 16), tune=(32, 4), calibration=(48, 8), stream=(192, 20), retention=(48, 8))
COSTS = dict(accept_unsupported=5, reject_supported=1, escalate=0.5, correct_accept_reject=0)
OWNED = [f"python/carnot/{NAME}.py", f"scripts/experiments/{NAME}.py"]
TEST = "tests/python/test_eligible_targets_8019.py"
PUBLIC_KEYS = set(
    [
        "family_id",
        "role",
        "source_cluster_id",
        "slot",
        "source_bytes",
        "answer_bytes",
        "q",
        "features",
        "status",
        "exclusion_reason",
        "capture_identity",
        "response_id",
    ]
)
BEGAN = time.monotonic()


def progress(phase: str, units: int = 0, pending: int = 0) -> None:
    """Flush actual work counts so quiet children cannot look like finished work."""
    print(
        f"[exp8019] phase={phase} elapsed_s={time.monotonic() - BEGAN:.3f} completed={units} pending={pending}",
        flush=True,
    )


class InputBlock(ValueError):
    """Keep missing and changed external operands for terminal blocked publication."""

    def __init__(self, path: Path, field: str, expected: Any, observed: Any):
        self.check = dict(
            upstream_id="exp8008_custody",
            path=str(path),
            hash=sha256_file(path) if path.is_file() else None,
            artifact_field=field,
            expected=expected,
            observed=observed,
            passed=False,
        )
        super().__init__(field)


def shard(raw: Path, scope: str, value: Json) -> Json:
    """Content addresses refuse overwrite so readers can retain original evidence."""
    encoded = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path = raw / scope / (hashlib.sha256(encoded).hexdigest() + ".json")
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError("immutable_shard_changed")
    atomic_json(path, value)
    if scope in {"evaluator", "custody"}:
        path.chmod(0o600)
    return reference(path)


def load_inputs(root: Path, raw: Path) -> Json:
    """Use the bound custody snapshot rather than reopening mutable role artifacts."""
    refs: list[Json] = []

    def copy(ref: Json) -> Path:
        path = Path(ref["path"])
        observed = sha256_file(path) if path.is_file() else None
        if observed != ref["sha256"]:
            raise InputBlock(path, "sha256", ref["sha256"], observed)
        target = raw / "custody" / (ref["sha256"].split(":")[-1] + path.suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256_file(target) != ref["sha256"]:
            raise InputBlock(target, "sha256", ref["sha256"], sha256_file(target))
        target.write_bytes(path.read_bytes())
        target.chmod(0o600)
        refs.append(dict(reference(target), original_path=str(path)))
        return target

    def read(ref: Json) -> Json:
        return dict(json.loads(copy(ref).read_text()))

    path = root / "results" / "experiment_8008_v694_conditioned_energy_fit.json"
    if not path.is_file():
        raise InputBlock(path, "checkpoints.bundle", "bound custody snapshot", None)
    primary = read(reference(path))
    if "bundle" not in primary.get("checkpoints", {}):
        raise InputBlock(
            path, "checkpoints.bundle", "bound custody snapshot", "MISSING_CONTRACT_FIELD"
        )
    bundle = read(primary["checkpoints"]["bundle"])
    imported = [read(r) for r in bundle["references"][:4]]
    cohort, capture, fit_history = imported[:3]
    if (cohort["experiment_id"], capture["experiment_id"], fit_history["experiment_id"]) != (
        7994,
        7995,
        7996,
    ):
        raise InputBlock(
            path,
            "bundle.references.producer_identity",
            [7994, 7995, 7996],
            [v.get("experiment_id") for v in imported[:3]],
        )
    public, evaluator, roster = {}, {}, []
    fit_refs = fit_history["cited_upstream_artifacts"]
    snapshot_refs = {r["sha256"]: r for r in bundle["references"]}
    fit_capture = read(next(r for r in fit_refs if r.get("producer_id") == 7969))
    for role in SLOTS:
        progress("recover_role_before_" + role, len(roster), 704 - len(roster))
        if role in {"fit", "tune"}:
            pub = read(
                next(
                    r for r in fit_refs if "/public/" in r["path"] and Path(r["path"]).stem == role
                )
            )
            ev = read(
                next(
                    r
                    for r in fit_refs
                    if "/evaluator/" in r["path"] and Path(r["path"]).stem == role
                )
            )
            slots = bundle["data"][role]
            captures = [r for r in fit_capture["rows"] if r["role"] == role]
            raw_refs = {r["sha256"]: r for r in fit_capture["raw_response_shards"]}
            for i, r in enumerate(captures):
                # Raw bytes authenticate each recorded judgment; the producer's rows alone do not.
                if r["capture_identity"] != fit_capture["capture_identity"]:
                    raise ValueError("capture_identity")
                raw_ref = fit_capture["raw_response_shards"][fit_capture["rows"].index(r)]
                if read(raw_refs[raw_ref["sha256"]]) != r:
                    raise ValueError("capture_bytes")
                if i % 32 == 0:
                    progress("recover_capture_" + role, i + 1, len(captures) - i - 1)
        else:
            pub = read(bundle["role_manifests"]["public"][role])
            ev = read(bundle["role_manifests"]["evaluator"][role])
            slots = [r for r in bundle["slots"] if r["role"] == role]
            captures = [r for r in capture["rows"] if r["role"] == role]
            for i, r in enumerate(captures):
                capture_ref = capture["raw_response_shards"][capture["rows"].index(r)]
                saved = read(snapshot_refs[capture_ref["sha256"]])
                if saved != r:
                    raise ValueError("capture_bytes")
                if i % 32 == 0:
                    progress("recover_capture_" + role, i + 1, len(captures) - i - 1)
        requests = index_unique(pub["request_rows"], "family_id")
        labels = index_unique(ev["rows"], "family_id")
        features = index_unique(pub["features"], "family_id")
        calls = index_unique(captures, "family_id")
        public[role], evaluator[role] = [], ev
        for i, slot in enumerate(slots):
            fid = slot["family_id"]
            request, call = requests[fid], calls[fid]
            original = dict(
                family_id=fid, role=role, source_cluster_id=slot["source_cluster_id"], slot=i
            )
            roster.append(original)
            public[role].append(
                dict(
                    original,
                    **{k: request[k] for k in ("source_bytes", "answer_bytes")},
                    q=slot["q"],
                    features=features[fid]["values"],
                    status=slot["status"],
                    exclusion_reason=call.get("exclusion_reason"),
                    capture_identity=call["capture_identity"],
                    response_id=labels[fid]["response_id"],
                )
            )
        progress("recover_role_after_" + role, len(roster), 704 - len(roster))
    exposure = [
        dict(r, **reference(copy(r)), original_path=r["path"])
        for r in bundle["original_exposure_rows"]
    ]
    feature_history = read(next(r for r in fit_refs if r.get("producer_id") == 7980))
    exposure.append(
        dict(
            shard(raw, "custody", feature_history["exposure_audit"]),
            scope="prior_frozen_discovery_inventory",
            roles=list(SLOTS),
            complete_history_custody=False,
        )
    )
    replay_path = root / "results/experiment_8006_v694_independent_replay.json"
    read(reference(replay_path))
    for r in primary["raw_shard_hashes"]:
        if "/failure_logs/" in r["path"]:
            copy(r)
    return dict(
        public=public,
        evaluator=evaluator,
        original_slot_roster=roster,
        exclusions=cohort["exclusion_manifest"],
        exposure_rows=exposure,
        feedback_schedule=cohort["feedback_schedule"],
        references=refs,
        historical_failure_logs=primary.get("gate_check_summary", []),
    )


def reduce(data: Json) -> Json:
    """Freeze completeness before counting classes; unknown evidence stays visible."""
    if set(data["public"]) != set(SLOTS) or set(data["evaluator"]) != set(SLOTS):
        raise ValueError("role_roster")
    originals = index_unique(data["original_slot_roster"], "family_id")
    seen: dict[str, str] = {}
    rows, private, support, public = [], {}, {}, {}
    for role, intended in SLOTS.items():
        slots = data["public"][role]
        if len(slots) != intended:
            raise ValueError("intended_slots:" + role)
        labels = index_unique(data["evaluator"][role]["rows"], "family_id")
        if set(labels) != {r["family_id"] for r in slots}:
            raise ValueError("target_roster")
        annotations = data["evaluator"][role]["annotation_rows"]
        private[role], public[role] = [], []
        selected = []
        for i, p in enumerate(slots):
            if set(p) != PUBLIC_KEYS or p["role"] != role:
                raise ValueError("public_role_or_fields")
            fid, target = p["family_id"], labels[p["family_id"]]
            y = target["y"]  # Missing fields are contract errors, never negative labels.
            if target["role"] != role or (
                y is not None and (type(y) is not int or y not in (0, 1))
            ):
                raise ValueError("target_contract")
            source, answer = bytes.fromhex(p["source_bytes"]), bytes.fromhex(p["answer_bytes"])
            group = normalized(source)
            if group in seen and seen[group] != role:
                raise ValueError("source_overlap")
            seen[group] = role
            identity = dict(family_id=fid, role=role, source_cluster_id=group, slot=i)
            if identity != originals[fid] or group != p["source_cluster_id"] or p["slot"] != i:
                raise ValueError("original_role_drift")
            if (
                target["response_sha256"] != digest(answer)
                or target["response_id"] != p["response_id"]
            ):
                raise ValueError("response_bytes_changed")
            spans = [a for a in annotations if a["family_id"] == fid]
            complete = (
                target["custody_passed"] is True
                and target["completely_annotated"] is True
                and target["quality"] == "good"
                and target["annotation_count"] == len(spans)
            )
            for a in spans:
                start, end = a["start_byte"], a["end_byte"]
                if (
                    a["response_sha256"] != digest(answer)
                    or a["response_id"] != target["response_id"]
                    or a["text_equal"] is not True
                    or not 0 <= start < end <= len(answer)
                    or answer[start:end].decode() != a["text"]
                ):
                    complete = False
            if complete and y is not None and y != int(bool(spans)):
                raise ValueError("original_target_drift")
            features, q = p["features"], p["q"]
            valid = (
                bool(source.strip() and answer.strip())
                and p["exclusion_reason"] is None
                and p["status"] in {"generated", "completed"}
                and type(q) in {int, float}
                and math.isfinite(q)
                and 0 <= q <= 1
                and isinstance(features, list)
                and len(features) == 8
                and all(type(v) in {int, float} and math.isfinite(v) for v in features)
            )
            eligible = valid and complete and y is not None
            reason = (
                (p["exclusion_reason"] or "invalid_public_input")
                if not valid
                else (
                    "incomplete_original_annotation"
                    if not complete
                    else "unknown_original_target"
                    if y is None
                    else None
                )
            )
            row = dict(
                identity,
                public_eligible=valid,
                complete_annotation=complete,
                target_eligible=eligible,
                eligibility=eligible,
                numerator=int(eligible),
                denominator=1,
                exclusion_reason=reason,
                failure_status=False,
                censor_status=role == "stream" and i + 20 >= 256,
                status="completed" if eligible else "excluded",
                arm="target_eligibility",
                seed=69519,
                reveal_at_slot=i + 20 if role == "stream" else None,
                historical_development_exposure=True,
                metric="eligible_target",
            )
            rows.append(row)
            selected.append(dict(row, y=y))
            public[role].append(dict(p, public_eligible=valid, target_eligible=eligible))
            private[role].append(
                dict(
                    target,
                    eligible_y=y if eligible else None,
                    target_eligible=eligible,
                    eligibility_reason=reason,
                )
            )
        eligible_rows = [r for r in selected if r["target_eligible"]]
        classes = {
            str(y): len({r["source_cluster_id"] for r in eligible_rows if r["y"] == y})
            for y in (0, 1)
        }
        independent = len({r["source_cluster_id"] for r in eligible_rows})
        minimum, per_class = FLOORS[role]
        support[role] = dict(
            intended=intended,
            eligible=len(eligible_rows),
            independent=independent,
            public_eligible=sum(r["public_eligible"] for r in selected),
            class_counts=classes,
            unknown_target=sum(r["public_eligible"] and not r["target_eligible"] for r in selected),
            minimum=minimum,
            per_class=per_class,
            passed=independent >= minimum and min(classes.values()) >= per_class,
            excluded=sum(not r["public_eligible"] for r in selected),
        )
        progress("mask_frozen_" + role, len(rows), 704 - len(rows))
    if set(originals) != {r["family_id"] for r in rows}:
        raise ValueError("original_slot_roster")
    return dict(
        eligibility_rows=rows,
        rows=rows,
        support_by_role=support,
        public_rows=public,
        evaluator_rows=private,
        sample_size_budget=dict(
            unit="normalized_source_group",
            intended=704,
            eligible=sum(r["target_eligible"] for r in rows),
            started=704,
            completed=704,
            excluded=sum(not r["public_eligible"] for r in rows),
            failed=0,
            censored=sum(r["censor_status"] for r in rows),
            independent=len({r["source_cluster_id"] for r in rows if r["target_eligible"]}),
        ),
    )


def seal(data: Json, raw: Path) -> Json:
    """Write public projections separately so acquisition never receives labels."""
    schema_refs = {}
    public_fields = PUBLIC_KEYS | {"public_eligible", "target_eligible"}
    schemas = dict(
        public=dict(
            type="object",
            required=["schema", "role", "rows"],
            properties=dict(
                schema=dict(const="carnot.v695.public_source_role.v1"),
                role=dict(enum=list(SLOTS)),
                rows=dict(
                    type="array",
                    items=dict(
                        type="object",
                        required=sorted(public_fields),
                        additionalProperties=False,
                        properties={k: {} for k in public_fields},
                    ),
                ),
            ),
        ),
        evaluator=dict(
            type="object",
            required=["schema", "role", "rows", "annotation_rows", "access_policy"],
            properties=dict(
                access_policy=dict(const="evaluator_only"),
                rows=dict(
                    type="array",
                    items=dict(
                        type="object",
                        required=["family_id", "role", "y", "eligible_y", "target_eligible"],
                        properties=dict(
                            y=dict(enum=[0, 1, None]), eligible_y=dict(enum=[0, 1, None])
                        ),
                    ),
                ),
                annotation_rows=dict(type="array"),
            ),
        ),
        exclusions=dict(type="object", required=["original_exclusions", "slot_exclusions"]),
    )
    for scope, schema in schemas.items():
        schema_refs[scope] = shard(
            raw,
            "schemas",
            dict(schema, **{"$schema": "https://json-schema.org/draft/2020-12/schema"}),
        )
    reduced = reduce(data)
    public, evaluator = {}, {}
    for role in SLOTS:
        public[role] = shard(
            raw,
            "public",
            dict(
                schema="carnot.v695.public_source_role.v1",
                role=role,
                rows=reduced["public_rows"][role],
            ),
        )
        evaluator[role] = shard(
            raw,
            "evaluator",
            dict(
                schema="carnot.v695.target_eligibility.v1",
                role=role,
                rows=reduced["evaluator_rows"][role],
                annotation_rows=data["evaluator"][role]["annotation_rows"],
                access_policy="evaluator_only",
            ),
        )
    reduced.pop("public_rows")
    reduced.pop("evaluator_rows")
    reduced.update(
        schema_manifests=schema_refs,
        public_manifests=public,
        evaluator_manifests=evaluator,
        original_slot_roster=data["original_slot_roster"],
        exposure_rows=data["exposure_rows"],
        exclusion_manifest=shard(
            raw,
            "exclusions",
            dict(
                original_exclusions=data["exclusions"],
                slot_exclusions=[r for r in reduced["rows"] if r["exclusion_reason"] is not None],
            ),
        ),
        checkpoint_references=[shard(raw, "evaluator", data)],
    )
    return reduced


def replay(path: Path) -> Json:
    """Cold reduction detects changed shards and edited aggregates without fitting."""
    value = json.loads(path.read_text())
    for ref in value["raw_shard_hashes"] + value["code_config_hashes"]:
        p = Path(ref["path"])
        if not p.is_file() or sha256_file(p) != ref["sha256"]:
            raise ValueError("bound_bytes_changed")
    if value["checkpoint_references"]:
        data = json.loads(Path(value["checkpoint_references"][0]["path"]).read_text())
        reduced = reduce(data)
        for key in ("rows", "eligibility_rows", "support_by_role", "sample_size_budget"):
            if reduced[key] != value[key]:
                raise ValueError("reduction_drift:" + key)
        for role in SLOTS:
            public = json.loads(Path(value["public_manifests"][role]["path"]).read_text())
            evaluator = json.loads(Path(value["evaluator_manifests"][role]["path"]).read_text())
            if (
                public["rows"] != reduced["public_rows"][role]
                or evaluator["rows"] != reduced["evaluator_rows"][role]
            ):
                raise ValueError("manifest_drift")
    if value["target_roles_ready_score"] and (
        value["verdict_class"] in {"blocked", "disqualified", "circular_positive"}
        or not value["acceptance_gate_results"]["owned_checks"]
        or not all(r["passed"] for r in value["support_by_role"].values())
    ):
        raise ValueError("unsafe_readiness")
    return dict(passed=True)


def validation_plan(scratch: Path) -> list[CommandSpec]:
    """Bound owned commands independently of the single broad health diagnostic."""
    py = str(ROOT / ".venv/bin/python")
    cov = str(ROOT / ".venv/bin/coverage")
    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel = True\ndata_file = "
        + str(scratch / ".coverage")
        + "\ninclude =\n    "
        + "\n    ".join(str(ROOT / p) for p in OWNED)
        + "\n"
    )
    tests = [
        TEST,
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_response_targets_7955.py",
        "tests/python/test_source_boundary_7852.py",
        "tests/python/test_experiment_7942_v689_sentence_labels.py",
    ]
    return [
        CommandSpec(
            "unit_consumers_e2e015_019",
            (
                py,
                "-m",
                "coverage",
                "run",
                "--data-file=" + str(scratch / ".coverage"),
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(scratch / "pytest"),
                *tests,
                "-q",
            ),
            "owned",
            300,
        ),
        CommandSpec(
            "coverage_combine",
            (cov, "combine", "--data-file=" + str(scratch / ".coverage"), str(scratch)),
            "owned",
            60,
        ),
        CommandSpec(
            "coverage_report",
            (
                cov,
                "report",
                "--data-file=" + str(scratch / ".coverage"),
                "--fail-under=100",
                "--show-missing",
            ),
            "owned",
            60,
        ),
        CommandSpec(
            "coverage_json",
            (
                cov,
                "json",
                "--data-file=" + str(scratch / ".coverage"),
                "-o",
                str(scratch / "coverage.json"),
            ),
            "owned",
            60,
        ),
        CommandSpec(
            "ruff_check", (str(ROOT / ".venv/bin/ruff"), "check", *OWNED, TEST), "owned", 60
        ),
        CommandSpec(
            "ruff_format",
            (str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, TEST),
            "owned",
            60,
        ),
        CommandSpec(
            "strict_mypy",
            (str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED),
            "owned",
            120,
        ),
        CommandSpec("spec_coverage", (py, "scripts/check_spec_coverage.py", TEST), "owned", 60),
        CommandSpec(
            "repository_health",
            (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
            "repository_health",
            180,
        ),
    ]


def terminal(path: Path) -> Json:
    """Check exact candidate bytes with fresh reduction and existing terminal tools."""
    py = str(ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "cold_reduction",
            (py, "-u", str(ROOT / OWNED[-1]), "--cold-replay", str(path)),
            "terminal",
            60,
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
    receipts = run_commands(
        ROOT,
        commands,
        log_dir=path.parent / "terminal_logs" / sha256_file(path).split(":")[-1],
        heartbeat_s=30,
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Publish one terminal result after real checks; fixtures cannot claim readiness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261002"], default="20261002")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    began = time.monotonic()
    progress("begin_no_model_load_generation_or_training")
    try:
        if args.cold_replay:
            replay(args.cold_replay)
            progress("cold_replay_passed")
            return 0
        output = args.output.absolute()
        raw = output.parent / "raw" / output.stem
        scratch = Path(tempfile.mkdtemp(prefix="carnot-8019-"))
        commands = validation_plan(scratch)
        freeze = shard(
            raw,
            "methods",
            dict(
                schema="carnot.v695.eligibility_methods.v1",
                slots=SLOTS,
                floors=FLOORS,
                policy_cost_matrix=COSTS,
                replacement_allowed=False,
                synthetic_fill=False,
                mask_selection="complete_original_annotation_and_valid_public_inputs_only",
                measurement_budget_s=300,
                model_calls=0,
                training_steps=0,
                random_seed=69519,
                feedback_delay_slots=20,
                commands=[asdict(c) for c in commands],
                schemas=dict(
                    public=sorted(PUBLIC_KEYS),
                    evaluator="original rows and annotations; eligible_y is null unless complete",
                ),
            ),
        )
        progress("methods_roles_exclusions_budgets_gates_frozen")
        failures: list[Json] = []
        try:
            data = (
                json.loads(args.fixture_input.read_text())
                if args.fixture_input
                else load_inputs(args.root, raw)
            )
        except InputBlock as error:
            failures.append(error.check)
            data = {}
        frozen = time.monotonic()
        role_freeze = shard(
            raw,
            "methods",
            dict(
                original_slot_roster=data.get("original_slot_roster", []),
                original_exclusions=data.get("exclusions", {}),
                source_references=data.get("references", []),
                methods_sha256=freeze["sha256"],
                eligibility_uses_correctness=False,
                fitting_allowed=False,
            ),
        )
        measured = (
            seal(data, raw)
            if data
            else dict(
                rows=[],
                eligibility_rows=[],
                support_by_role={},
                public_manifests={},
                evaluator_manifests={},
                exposure_rows=[],
                original_slot_roster=[],
                checkpoint_references=[],
                sample_size_budget=dict(
                    intended=704,
                    eligible=0,
                    started=0,
                    completed=0,
                    excluded=0,
                    failed=0,
                    censored=0,
                    independent=0,
                ),
            )
        )
        for role, support in measured["support_by_role"].items():
            if not support["passed"]:
                failures.append(
                    dict(
                        upstream_id="exp8019_" + role,
                        path=measured["evaluator_manifests"][role]["path"],
                        hash=measured["evaluator_manifests"][role]["sha256"],
                        artifact_field="support_by_role." + role,
                        expected=dict(minimum=FLOORS[role][0], per_class=FLOORS[role][1]),
                        observed=support,
                        passed=False,
                    )
                )
        progress("eligibility_and_support_measured", len(measured["rows"]))
        health_path = raw / "repository_health_receipt.json"
        selected_commands = (
            [c for c in commands if c.scope == "owned"] if health_path.is_file() else commands
        )
        receipts = (
            []
            if args.fixture_input
            else run_commands(
                ROOT,
                selected_commands,
                log_dir=raw / "validation_logs",
                heartbeat_s=30,
                extra_env=dict(
                    JAX_PLATFORMS="cpu",
                    COVERAGE_FILE=str(scratch / ".coverage-health"),
                    CARNOT_8019_COVERAGE_CONFIG=str(scratch / "coverage.ini"),
                    OPENBLAS_NUM_THREADS="1",
                ),
            )
        )
        if not args.fixture_input:
            if health_path.is_file():
                receipts.extend(json.loads(health_path.read_text())["rows"])
                progress("reuse_single_bounded_health_diagnostic")
            else:
                atomic_json(
                    health_path,
                    dict(
                        rows=[r for r in receipts if r["scope"] == "repository_health"],
                        evidence_scope="earlier_owned_task_attempt_if_reused; health never determines owned readiness",
                    ),
                )
        coverage = (
            json.loads((scratch / "coverage.json").read_text())["files"]
            if (scratch / "coverage.json").is_file()
            else {}
        )
        counts = {k: v["summary"] for k, v in coverage.items()}
        owned = [r for r in receipts if r["scope"] == "owned"]
        checked = (
            bool(owned)
            and all(r["passed"] for r in owned)
            and len(counts) == len(OWNED)
            and all(r["missing_lines"] == 0 and r["num_statements"] > 0 for r in counts.values())
        )
        verdict = (
            "blocked"
            if failures
            else "circular_positive"
            if args.fixture_input
            else "null"
            if checked
            else "disqualified"
        )
        ready = int(checked and not failures)
        value = dict(
            measured,
            experiment_id=8019,
            task_id=TASK,
            milestone="2026.10.695",
            run_date=args.date,
            schema="carnot.v695.eligible_targets.v1",
            honest_verdict="complete_" + verdict + "_eligible_targets",
            verdict_class=verdict,
            gate_check_summary=failures,
            target_roles_ready_score=ready,
            fit_targets_ready_score=int(
                checked
                and all(
                    measured["support_by_role"].get(r, {}).get("passed", False)
                    for r in ("fit", "tune")
                )
            ),
            calibration_targets_ready_score=int(
                checked and measured["support_by_role"].get("calibration", {}).get("passed", False)
            ),
            stream_targets_ready_score=int(
                checked and measured["support_by_role"].get("stream", {}).get("passed", False)
            ),
            retention_targets_ready_score=int(
                checked and measured["support_by_role"].get("retention", {}).get("passed", False)
            ),
            claim_scope="Immutable manifests for historically exposed development; no prospective deployment, fitting or scientific benefit measured.",
            exposure_scope="all roles have historical development exposure; unknown unrecorded access and pretraining",
            policy_cost_matrix=COSTS,
            feedback_schedule=data.get("feedback_schedule", {}),
            inference_substrate="aggregation_from_upstream_artifacts",
            inference_substrate_class="no_model_load",
            MODEL_SPECS=[],
            model_specs=[],
            model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
            trained_head_specs=[],
            random_seed=69519,
            verifier_is_oracle=False,
            generalized_learning_benefit_score=0,
            acceptance_gate_results=dict(
                measurement=bool(data),
                support=not failures,
                owned_checks=checked,
                natural_benefit=False,
            ),
            genuine_headroom=dict(
                assessed=False, reason="eligibility only; no decision predictions"
            ),
            positive_control_results=dict(
                scope="circular_annotation_and_cli_fixtures_only", natural_benefit_credit=False
            ),
            cited_upstream_artifacts=data.get("references", []),
            validation_receipts=owned,
            repository_health=[r for r in receipts if r["scope"] == "repository_health"],
            coverage_statement_counts=counts,
            historical_failure_logs=data.get("historical_failure_logs", []),
            code_config_hashes=[reference(ROOT / p) for p in OWNED + [TEST]],
            flagged_adversarial=False,
            terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
            duration_s=time.monotonic() - began,
            phase_spans=[
                dict(phase="freeze_and_recover", duration_s=frozen - began),
                dict(phase="eligibility_and_validation", duration_s=time.monotonic() - frozen),
            ],
        )
        value["raw_shard_hashes"] = (
            [freeze, role_freeze]
            + value["checkpoint_references"]
            + value["cited_upstream_artifacts"]
            + list(value["public_manifests"].values())
            + list(value["evaluator_manifests"].values())
            + list(measured.get("schema_manifests", {}).values())
            + [reference(ROOT / r["log_path"]) for r in receipts if "log_path" in r]
            + [reference(p) for p in sorted((raw / "failure_logs").glob("*.log"))]
            + ([reference(health_path)] if health_path.is_file() else [])
        )
        if "exclusion_manifest" in measured:
            value["raw_shard_hashes"].append(measured["exclusion_manifest"])
        value["reproducibility_checksum"] = canonical_hash(
            dict(methods=freeze, code=value["code_config_hashes"], raw=value["raw_shard_hashes"])
        )
        value["field_principles"] = {
            k: "Bind current eligibility work to original bytes and intended denominators; unknown evidence and exposed development give no natural benefit credit."
            for k in value
        }
        progress("publish_before")
        receipt = publish_primary(output, value, replay if args.fixture_input else terminal)
        atomic_json(raw / "terminal_validation.json", receipt)
        reader = reader_receipt(
            TASK, output.parent, field="target_roles_ready_score", expected=ready
        )
        atomic_json(raw / "reader_receipt.json", reader)
        if not reader["passed"] or reader["gate_sha256"] != sha256_file(output):
            raise ValueError("primary_reader")
        replay(output)
        progress("publish_after", len(value["rows"]))
        return 0
    except (ValueError, OSError, KeyError, TypeError, StopIteration) as error:
        print(f"[exp8019] failed={error}", flush=True)
        return 1
