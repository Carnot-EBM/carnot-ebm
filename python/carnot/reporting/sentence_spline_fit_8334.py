"""REQ-VERIFY-8334 / REQ-REPORT-8334: custody precedes static sentence fitting."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time
from typing import Any
import yaml

from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.verify.cached_sentence_custody_8305 import check_predictor
from carnot.verify import sentence_spline_fit_8334 as n

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8334_v719_sentence_spline_fit"
TASK = "exp8334-sentence-spline-fit"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_sentence_spline_fit_8334.py"
OWNED = [
    "python/carnot/verify/sentence_spline_fit_8334.py",
    "python/carnot/reporting/sentence_spline_fit_8334.py",
    "python/carnot/reporting/sentence_spline_execution_8334.py",
    CLI,
]
ROLES = ["fit", "calibration", "comparator_selection"]
SOURCE = "results/experiment_8305_v717_cached_sentence_custody.json"
PROTOCOL = "openspec/change-proposals/v717-local-learning-protocol.json"
PINS = {
    SOURCE: "sha256:84564c8702db3a3665c84e9db9f62d6a2bffe00729cc19a19bb4c3a6c2ea13ba",
    PROTOCOL: "sha256:853709123024de763e96dd688e819f0430205ae6d97d6561a2b95cca23b81c6f",
}
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed boundaries let a supervisor see real completed work."""
    print(f"[exp8334] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Hashes bind every conclusion to its actual bytes."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def validate_bundle(bundle: Json) -> Json:
    """Original roster identity binds labels separately from public features."""
    support = {}
    seen = set()
    for role, count in zip(ROLES, [128, 32, 32], strict=True):
        ps, ts = bundle["predictors"][role], bundle["evaluators"][role]
        if len(ps) != count or len(ts) != count:
            raise ValueError("intended_roster")
        classes = [0, 0]
        features = 0
        missing = 0
        for index, (p, t) in enumerate(zip(ps, ts, strict=True)):
            check_predictor(p)
            original = bundle["protocol"]["original_roles"]["fit" if role == "fit" else "tune"][
                index + (32 if role == "comparator_selection" else 0)
            ]
            keys = ["unit_id", "source_cluster_id", "role", "slot"]
            if (
                any(p[k] != t[k] for k in keys)
                or p["role"] != role
                or p["slot"] != index + 1 + (32 if role == "comparator_selection" else 0)
            ):
                raise ValueError("label_identity")
            manifest = next(r for r in bundle["manifest"] if r["unit_id"] == p["unit_id"])
            if (
                any(p[k] != manifest[k] for k in manifest)
                or any(p[k] != original[k] for k in ["unit_id", "source_cluster_id"])
                or p["source_sha256"] != original["original_source_sha256"]
                or p["source_cluster_id"] in seen
            ):
                raise ValueError("source_identity")
            seen.add(p["source_cluster_id"])
            if t["y"] is not None and (type(t["y"]) is not int or t["y"] not in (0, 1)):
                raise ValueError("binary_target")
            features += int(p["x"] is not None)
            missing += int(p["x"] is not None and t["y"] is None)
            if p["x"] is not None and t["y"] is not None:
                classes[t["y"]] += 1
        support[role] = dict(
            intended=count,
            feature_rows=features,
            missing_labels=missing,
            usable=sum(classes),
            supported=classes[0],
            unsupported=classes[1],
        )
    return support


def measure(root: Path, raw: Path) -> Json:
    """Authenticate only intended fit/tune bytes; reserved label shards stay closed."""
    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    raw.chmod(0o700)
    work: Json = dict(failures=[], refs=[], bundle={}, support={}, trained={}, phase_spans=[])

    def require(path: Path, field: str, expected: Any, observed: Any) -> None:
        if expected != observed:
            work["failures"].append(
                dict(
                    upstream=path.stem,
                    path=str(path),
                    hash=sha256_file(path) if path.is_file() else None,
                    artifact_field=field,
                    op="==",
                    expected=expected,
                    observed=observed,
                    passed=False,
                )
            )
            raise ValueError(field)

    def bind(ref: Json) -> Json:
        path = Path(ref["path"])
        require(path, "sha256", ref["sha256"], sha256_file(path) if path.is_file() else None)
        dest = raw / "inputs" / (ref["sha256"][7:] + "-" + path.name)
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(path.read_bytes())
        dest.chmod(0o600)
        work["refs"].append(reference(dest))
        return dict(json.loads(dest.read_bytes()))

    progress("before_authentication")
    try:
        require(
            raw,
            "private_resources",
            True,
            raw.stat().st_mode & 0o077 == 0 and shutil.disk_usage(raw).free > 1_000_000_000,
        )
        for tool in ["python", "pytest", "coverage", "ruff", "mypy"]:
            require(
                ROOT / ".venv/bin" / tool,
                "executable",
                True,
                os.access(ROOT / ".venv/bin" / tool, os.X_OK),
            )
        source, protocol = [
            bind(dict(path=str(root / p), sha256=PINS[p])) for p in [SOURCE, PROTOCOL]
        ]
        validate_primary(source, root / SOURCE)
        terminal = bind(reference(Path(source["terminal_validation_sidecar_path"])))
        side = Path(terminal["publication"]["sidecar_path"])
        attestation = read_bound_sidecar(root / SOURCE, side)
        bind(reference(side))
        require(side, "terminal_passed", True, attestation["report"]["passed"])
        require(
            root / SOURCE,
            "terminal_primary_sha256",
            PINS[SOURCE],
            terminal["publication"]["primary_sha256"],
        )
        require(root / SOURCE, "required_checks_passed", True, source["required_checks_passed"])
        require(root / SOURCE, "flagged_adversarial", False, source["flagged_adversarial"])
        design = root / "openspec/change-proposals/research-roadmap-vNEXT.md"
        active = root / "research-roadmap.yaml"
        text = design.read_text()
        tasks = parse_design(text, milestone="2026.10.719")[1]
        activated = yaml.safe_load(active.read_text())
        authority = compare_contract(
            text, activated, activated, milestone="2026.10.719", first_id=8332
        )
        require(
            active,
            "fourteen_task_agreement",
            True,
            authority["passed"] and activated["tasks"] == tasks,
        )
        work["refs"] += [reference(design), reference(active)]
        bundle = dict(
            protocol=protocol, manifest=source["source_role_manifest"], predictors={}, evaluators={}
        )
        for role in ROLES:
            for kind, field in [
                ("predictors", "predictor_shards"),
                ("evaluators", "evaluator_shards"),
            ]:
                bundle[kind][role] = bind(source[field][role])["rows"]
        work.update(
            bundle=bundle,
            support=validate_bundle(bundle),
            historical=source["historical_model_provenance"],
            protocol_reference=reference(root / PROTOCOL),
            source_manifest_reference=dict(reference(root / SOURCE), field="source_role_manifest"),
        )
        for role, minimum, per_class in [
            ("fit", 96, 12),
            ("calibration", 24, 4),
            ("comparator_selection", 24, 4),
        ]:
            s = work["support"][role]
            require(
                root / SOURCE,
                role + "_support",
                True,
                s["usable"] >= minimum and min(s["supported"], s["unsupported"]) >= per_class,
            )
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        if not work["failures"]:
            work["failures"].append(
                dict(
                    upstream="custody",
                    path=str(root),
                    hash=None,
                    artifact_field="structure",
                    op="==",
                    expected="authenticated_inputs",
                    observed=str(error),
                    passed=False,
                )
            )
    progress("after_authentication")
    progress("before_benchmark_numerics")
    work["numeric_audit"] = n.numeric_audit()
    work["optimizer_control"] = n.optimizer_control()
    progress("after_benchmark_numerics")
    if (
        not work["failures"]
        and work["numeric_audit"]["passed"]
        and work["optimizer_control"]["passed"]
    ):
        try:
            work["trained"] = n.train(work["bundle"])
        except ValueError as error:
            work["failures"].append(
                dict(
                    upstream="fit_geometry",
                    path=str(root / SOURCE),
                    hash=PINS[SOURCE],
                    artifact_field="distinct_fit_vectors",
                    op=">=",
                    expected=32,
                    observed=str(error),
                    passed=False,
                )
            )
    manifest = raw / "source_manifest.json"
    atomic_json(
        manifest,
        dict(rows=work["bundle"].get("manifest", []), source=work.get("source_manifest_reference")),
    )
    work["manifest_reference"] = reference(manifest)
    checkpoint = raw / "frozen_heads.json"
    atomic_json(checkpoint, work["trained"])
    work["checkpoint_reference"] = reference(checkpoint)
    api = raw / "predictor.py"
    api.write_text(
        '"""Feature-only prediction; targets and IDs are rejected."""\nfrom carnot.verify.sentence_spline_fit_8334 import predict\n'
    )
    work["predictor_api_reference"] = reference(api)
    work.update(
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(
                phase="authentication_numerics_fit_seal",
                start_s=0.0,
                duration_s=time.monotonic() - began,
            )
        ],
        code_refs=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                "python/carnot/verify/local_update_isolation_8306.py",
                "python/carnot/reporting/v718_replay_history.py",
                "scripts/adversarial_verify.py",
            ]
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress("heads_sealed", len(work["trained"].get("heads", [])), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Execution readiness does not depend on a favorable natural score."""
    coverage = work.get("owned_coverage_reference")
    coverage_ok = True
    if coverage:
        covered = json.loads(Path(coverage["path"]).read_bytes())
        coverage_ok = covered["totals"]["percent_covered"] == 100 and all(
            p in covered["files"] and covered["files"][p]["summary"]["missing_lines"] == 0
            for p in OWNED
        )
    owned = bool(receipts) and all(r["passed"] for r in receipts) and coverage_ok
    trained = work["trained"]
    heads = trained.get("heads", [])
    numeric = work["numeric_audit"]["passed"]
    control = work["optimizer_control"]["passed"]
    klass = (
        "disqualified" if not owned or not numeric else "blocked" if work["failures"] else "null"
    )
    ready = int(
        owned
        and numeric
        and control
        and not work["failures"]
        and len(heads) == 4
        and all(np_finite(h["coefficients"]) for h in heads)
    )
    verdict = (
        "complete_null_optimizer_control"
        if klass == "null" and not control
        else "complete_" + klass + "_sentence_spline_fit"
    )
    if klass == "blocked":
        verdict = "complete_blocked_" + work["failures"][0]["upstream"]
    predictors = work["bundle"].get("predictors", {})
    source_rows = [dict(p, x=None) for role in ROLES for p in predictors.get(role, [])]
    counts = {
        status: sum(p["status"] == status for p in source_rows)
        for status in ["completed", "failed", "censored", "excluded"]
    }
    seal = dict(
        heads=[
            {k: h[k] for k in ["arm", "coefficients", "temperature", "geometry"]} for h in heads
        ],
        selected_comparator=trained.get("selected_comparator"),
        role_hashes={
            r: canonical_hash(
                dict(predictors=predictors[r], evaluators=work["bundle"]["evaluators"][r])
            )
            for r in predictors
        },
        source_manifest_reference=work.get("source_manifest_reference"),
        protocol_reference=work.get("protocol_reference"),
        checkpoint_reference=work["checkpoint_reference"],
        reserved_labels_opened=False,
    )
    seal["whole_model_sha256"] = canonical_hash(seal)
    value = dict(
        experiment_id=8334,
        task_id=TASK,
        milestone="2026.10.719",
        run_date="20261009",
        honest_verdict=verdict,
        verdict_class=klass,
        heads_ready_score=ready,
        gate_check_summary=work["failures"],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=ZERO_INVOCATION_COUNTS,
        historical_model_provenance=work.get("historical", []),
        rows=source_rows,
        intended_count=192,
        completed_count=counts["completed"],
        failed_count=counts["failed"],
        censored_count=counts["censored"],
        excluded_count=counts["excluded"],
        independent_count=len(source_rows),
        sample_size_budget=dict(
            fit=128, calibration=32, comparator_selection=32, reserved_opened=0
        ),
        verifier_is_oracle=False,
        exposure_scope="exposed_cached_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        acceptance_gates=dict(
            custody=not work["failures"], numerical=numeric, optimizer_control=control, owned=owned
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=work.get("finding_audits", []),
        preconditions_checked=True,
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7178308,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=[
            work["checkpoint_reference"],
            work["manifest_reference"],
            work["predictor_api_reference"],
        ],
        cited_upstream_artifacts=[
            dict(
                ref,
                fields_imported=[
                    "source custody",
                    "fit/tune public inputs and sealed targets",
                    "protocol",
                ],
            )
            for ref in work["refs"]
        ],
        frozen_head_manifest=seal,
        selected_comparator=trained.get("selected_comparator"),
        fit_loss_rows=[
            dict(arm=h["arm"], initial=h["initial"], final=h["final"], losses=h["loss_rows"])
            for h in heads
        ],
        calibration_rows=trained.get("calibration_rows", []),
        trained_head_specs=[
            dict(arm=h["arm"], coefficient_count=n.COUNTS[h["arm"]], generator_weight_updates=False)
            for h in heads
        ],
        coefficient_counts=n.COUNTS,
        sigmoid_equivalence_error=trained.get("sigmoid_equivalence_error"),
        feature_information_parity=dict(
            spline34="holistic logit and four local inputs",
            RBF34="same five inputs",
            linear6="same five inputs",
            scalar2="holistic logit",
            sigmoid34="exact spline34 re-expression",
        ),
        checkpoint_hashes=[work["checkpoint_reference"]],
        source_manifest_reference=work.get("source_manifest_reference"),
        protocol_reference=work.get("protocol_reference"),
        optimizer_control=work["optimizer_control"],
        numerical_audit=work["numeric_audit"],
        class_support_by_role=work["support"],
        development_rows=trained.get("rows", []),
        methodology_note="Small heads fit original cached development sources only. Missing rows escalate at cost .5. Fixed thresholds p<.25 / p>.75 are conservative, not optimal under this cost. Unconverged arm differences describe frozen training procedures, not geometry alone. Reserved labels remain unopened.",
        optimizer_config=n.CONFIG,
        predictor_api_reference=work["predictor_api_reference"],
        work_reference=reference(raw / "measurement.json"),
        owned_coverage_reference=work.get("owned_coverage_reference"),
        execution_manifest_reference=work.get("execution_manifest_reference"),
        invocation_argv=work.get("invocation_argv", []),
    )
    value["field_principles"] = {
        k: "Bind "
        + k
        + " to authenticated primitives, separate source roles, actual execution and exposed-development claim scope."
        for k in value
    }
    value["field_principles"]["reproducibility_checksum"] = (
        "Hash the complete semantic artifact except this checksum."
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def np_finite(values: list[float]) -> bool:
    """Nonfinite coefficients cannot become ready predictor state."""
    import math

    return all(math.isfinite(v) for v in values)


def replay(path: Path) -> bool:
    """Cold replay rebuilds trained state; rehashing a forgery cannot authorize it."""
    try:
        value = json.loads(path.read_bytes())
        ref = value["work_reference"]
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            return False
        work = json.loads(Path(ref["path"]).read_bytes())
        for r in [
            *work["refs"],
            *work["code_refs"],
            work["checkpoint_reference"],
            work["manifest_reference"],
            work["predictor_api_reference"],
        ]:
            if sha256_file(Path(r["path"])) != r["sha256"]:
                return False
        if work["bundle"]:
            copied = {r["sha256"]: Path(r["path"]) for r in work["refs"]}
            source = json.loads(copied[PINS[SOURCE]].read_bytes())
            protocol = json.loads(copied[PINS[PROTOCOL]].read_bytes())
            expected = dict(
                protocol=protocol,
                manifest=source["source_role_manifest"],
                predictors={},
                evaluators={},
            )
            for role in ROLES:
                for kind, field in [
                    ("predictors", "predictor_shards"),
                    ("evaluators", "evaluator_shards"),
                ]:
                    operand = source[field][role]
                    expected[kind][role] = json.loads(copied[operand["sha256"]].read_bytes())[
                        "rows"
                    ]
            if expected != work["bundle"] or validate_bundle(expected) != work["support"]:
                return False
            if work["trained"] and canonical_hash(n.train(expected)) != canonical_hash(
                work["trained"]
            ):
                return False
        if canonical_hash(n.numeric_audit()) != canonical_hash(
            work["numeric_audit"]
        ) or canonical_hash(n.optimizer_control()) != canonical_hash(work["optimizer_control"]):
            return False
        if canonical_hash(
            json.loads(Path(work["checkpoint_reference"]["path"]).read_bytes())
        ) != canonical_hash(work["trained"]):
            return False
        rebuilt = build(work, Path(ref["path"]).parent, value["validation_receipts"])
        return canonical_hash(value) == canonical_hash(rebuilt)
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False
