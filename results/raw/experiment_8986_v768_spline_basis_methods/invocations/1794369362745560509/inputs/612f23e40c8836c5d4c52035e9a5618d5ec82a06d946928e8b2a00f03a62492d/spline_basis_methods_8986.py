"""REQ-REPORT-8986: freeze convex mechanics without claiming independent benefit."""

import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import urllib.request

import numpy as np
from scipy.special import expit

from carnot.autoresearch import spline_basis_8986 as n
from carnot.autoresearch import relational_graph_encoder as g
from carnot.reporting import shared_feature_methods_8973 as prior
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash

Json = dict[str, Any]
ROOT = prior.ROOT
NAME, TASK, SEED = (
    "experiment_8986_v768_spline_basis_methods",
    "exp8986-spline-basis-methods",
    768086,
)
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/autoresearch/spline_basis_8986.py",
    "python/carnot/reporting/spline_basis_methods_8986.py",
    "python/carnot/reporting/spline_basis_runner_8986.py",
    CLI,
]
TESTS = ["tests/python/test_spline_basis_methods_8986.py"]
PROTOCOL = "openspec/change-proposals/v768-spline-protocol.json"
PROTOCOL_PIN = "sha256:dd0eb38a797fd3e1ddbe172b27c4b504ac8229251eed6cce73bd3ce6114f251b"
METHODS = f"results/raw/{NAME}/methods/method_map.json"
METHOD_PIN = "sha256:2486c9863665a84edbe0c98d6d8ce928cef588ab6e3e8f39f5600bba569372b8"
CONTEXT = "results/experiment_8973_v767_shared_feature_methods.json"
CONTEXT_PIN = "sha256:3870b7055fbff26bc4a35316f60555abe99c0f2326c14156540af78a2e5b8fa6"
ARMS = ["spline", "converged_linear", "shared_logistic", "confidence_logistic", "prevalence"]
reference, retain, seal, checksum = prior.reference, prior.retain, prior.seal, prior.checksum


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts expose real progress without extending a measurement deadline."""
    print(f"[exp8986] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(
    path: Path, field: str, expected: Any, observed: Any, check: str, upstream: str = "environment"
) -> Json:
    """Every external refusal names its real operand and frozen comparison."""
    return dict(
        reference(path),
        check=check,
        upstream=upstream,
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=type(expected) is type(observed) and expected == observed,
    )


def authenticate(root: Path, raw: Path) -> Json:
    """Historical failure is context; only the declared diagnostic gate is required."""
    evidence: Json = prior.authenticate(root, raw)
    checks = [
        dict(
            r,
            op=r.get("operator", "=="),
            passed=False,
            observed="missing_field" if r["observed"] is None else r["observed"],
        )
        for r in evidence["failures"]
    ]
    evidence["checks"], evidence["method_source_rows"], evidence["protocol"] = checks, [], {}
    for name, pin, field, expected in [
        (CONTEXT, CONTEXT_PIN, "required_checks_passed", True),
        (PROTOCOL, PROTOCOL_PIN, "experiment_id", 8986),
        (METHODS, METHOD_PIN, None, None),
    ]:
        path = root / name
        digest = reference(path)["sha256"]
        row = gate(path, "sha256", pin, digest or "missing_file", "input_bytes", name)
        checks.append(row)
        if not row["passed"]:
            continue
        saved = retain(reference(path), raw)
        evidence["references"].append(saved)
        value = json.loads(Path(saved["path"]).read_text())
        if field:
            checks.append(
                gate(
                    path, field, expected, value.get(field, "missing_field"), "declared_gate", name
                )
            )
        if name == CONTEXT:
            evidence["context"] = value
        if name == PROTOCOL:
            evidence["protocol"] = value
        if name == METHODS:
            evidence["method_source_rows"] = value["rows"]
            for item in value["rows"]:
                actual = reference(Path(item["path"]))["sha256"]
                checks.append(
                    gate(
                        Path(item["path"]),
                        "sha256",
                        item["sha256"],
                        actual or "missing_file",
                        "method_bytes",
                        item["source_id"],
                    )
                )
                if actual == item["sha256"]:
                    evidence["references"].append(retain(item, raw))
    if evidence.get("producer"):
        checks += [
            gate(
                root / prior.INPUT,
                field,
                expected,
                evidence["producer"].get(field, "missing_field"),
                "declared_gate",
                "exp8972",
            )
            for field, expected in [("diagnosis_ready_score", 1), ("required_checks_passed", True)]
        ]
    evidence["failures"] = [r for r in checks if not r["passed"]]
    return evidence


def preflight(raw: Path, evidence: Json) -> Json:
    """Check resources and bounded network access before numerical work starts."""
    failures = prior.resources(raw)
    checks = [
        gate(
            ROOT / ".venv", "sys.prefix", str(ROOT / ".venv"), sys.prefix, "repository_virtualenv"
        ),
        gate(raw, "scipy", True, importlib.util.find_spec("scipy") is not None, "dependency"),
    ]
    for i, row in enumerate(evidence["method_source_rows"]):
        progress("before_network_check", i, 4 - i)
        try:
            with urllib.request.urlopen(row["url"], timeout=5) as response:
                response.read(1)
                observed: Any = response.status
        except OSError as error:
            observed = str(error)
        checks.append(
            gate(
                Path(row["path"]),
                "http_status",
                200,
                observed,
                "required_network",
                row["source_id"],
            )
        )
        progress("after_network_check", i + 1, 3 - i)
    evidence["checks"].extend(checks)
    evidence["failures"].extend(
        [dict(r, passed=False, op=r.get("operator", "==")) for r in failures]
        + [r for r in checks if not r["passed"]]
    )
    return dict(
        repository_virtualenv=sys.prefix,
        resources_checked=True,
        dependencies=["jax", "numpy", "scipy", "sklearn", "coverage", "pytest", "ruff", "mypy"],
        writable_disk_checked=True,
        free_memory_checked=True,
        disk_free_bytes=shutil.disk_usage(raw).free,
        memory_available_bytes=os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
        exclusion_manifest_reference=reference(ROOT / "ops/exclusion_manifest.yaml"),
        required_network_checks=checks,
        resource_failures=failures,
    )


def evaluate(evidence: Json, raw: Path, replay: bool = False) -> Json:
    """Fit only original graph labels and independently reduce retained per-graph scores."""
    original = evidence["work"]["evidence"]
    views, states = original["views"], original["states"]
    authority = original["producers"]["8959"]
    if any(
        len(views[role]["rows"]) != count
        or views[role]["unit_ids"] != authority[f"development_{role}_ids"]
        for role, count in [("fit", 256), ("guard", 64)]
    ):
        raise ValueError("development_roles")
    probabilities: Json = {role: {arm: [] for arm in ARMS[:2]} for role in ("fit", "guard")}
    heads: list[Json] = []
    encoders, invariants, clipping, feature_refs = [], [], [], []
    candidate = prior.load(evidence, evidence["producer"]["candidate_reference"])
    progress("before_cached_benchmark", 0, 6)
    for i, state in enumerate(states["relational_logistic"]):
        seed = state["seed"]
        fixed = next(s for s in candidate["encoder_states"] if s["seed"] == seed)
        if state["params"]["encoder"] != fixed["encoder"] or state["scaling"] != fixed["scaling"]:
            raise ValueError("encoder_bytes")
        encoders.append(
            dict(
                seed=seed,
                sha256=canonical_hash(fixed["encoder"]),
                scaling_hash=canonical_hash(fixed["scaling"]),
            )
        )
        arrays = {}
        for role in ("fit", "guard"):
            x = n.features(state, views[role]["rows"])
            ref = next(
                r
                for r in evidence["producer"]["feature_hashes"]
                if r["encoder"] == "logistic" and r["seed"] == seed and r["role"] == role
            )
            delta = float(np.max(np.abs(x - np.asarray(prior.load(evidence, ref)["features"]))))
            if delta > 1e-12:
                raise ValueError("historical_features")
            invariants.append(
                dict(
                    seed=seed,
                    role=role,
                    metric="historical_feature_delta",
                    value=delta,
                    passed=True,
                )
            )
            arrays[role] = x
        scaling = n.fit_scaling(arrays["fit"])
        designs = {}
        for role, x in arrays.items():
            scaled, counts = n.scale(x, scaling)
            clipping.append(dict(seed=seed, role=role, **counts))
            designs[role] = scaled
            path = raw / "features" / f"{seed}-{role}.json"
            seal(
                path,
                dict(features=x.tolist(), scaled=scaled.tolist(), unit_ids=views[role]["unit_ids"]),
                replay,
            )
            feature_refs.append(dict(reference(path), seed=seed, role=role))
        for arm in ARMS[:2]:
            progress(f"before_fit_{seed}_{arm}", len(heads), 6 - len(heads))
            fitted = n.fit(
                n.design(designs["fit"], arm == "spline"),
                np.asarray(views["fit"]["labels"], dtype=float),
            )
            heads.append(
                dict(
                    fitted,
                    seed=seed,
                    arm=arm,
                    scaling=scaling,
                    encoder_sha256=encoders[-1]["sha256"],
                )
            )
            for role in ("fit", "guard"):
                probabilities[role][arm].append(
                    expit(
                        n.design(designs[role], arm == "spline")
                        @ np.asarray(fitted["coefficients"])
                    ).tolist()
                )
            progress(f"after_fit_{seed}_{arm}", len(heads), 6 - len(heads))
        shuffled = []
        for row in views["guard"]["rows"]:
            names = {v: f"anonymous-{j}" for j, v in enumerate(row["nodes"])}
            shuffled.append(
                dict(
                    row,
                    nodes=[names[v] for v in row["nodes"][::-1]],
                    edges=[[names[b], names[a]] for a, b in row["edges"][::-1]],
                    query_node=names[row["query_node"]],
                    proposed_node=names[row["proposed_node"]]
                    if row["proposed_node"] is not None
                    else None,
                )
            )
        exact = bool(np.array_equal(arrays["guard"], n.features(state, shuffled)))
        invariants.append(
            dict(seed=seed, metric="exact_node_permutation", value=exact, passed=exact)
        )
    rows = []
    old_shared = [
        prior.load(evidence, r)
        for r in evidence["producer"]["head_state_references"]
        if r["arm"] == "logistic_encoder_linear_head"
    ]
    for role in ("fit", "guard"):
        view = views[role]
        predictions = {arm: np.mean(probabilities[role][arm], axis=0) for arm in ARMS[:2]}
        predictions["shared_logistic"] = g.ensemble(old_shared, view["rows"])
        predictions["confidence_logistic"] = g.ensemble(states["confidence_logistic"], view["rows"])
        predictions["prevalence"] = np.full(len(view["rows"]), np.mean(views["fit"]["labels"]))
        for arm, values in predictions.items():
            for j, unit in enumerate(view["unit_ids"]):
                loss = float((values[j] - view["labels"][j]) ** 2)
                rows.append(
                    dict(
                        unit_id=unit,
                        group_id=unit,
                        cluster_id=unit,
                        arm=arm,
                        condition=f"exposed_development_{role}",
                        metric="brier",
                        value=loss,
                        numerator=loss,
                        denominator=1,
                        p_error=float(values[j]),
                        error=int(view["labels"][j]),
                        status="completed",
                        exclusion_reason=None,
                    )
                )
    audit = n.numeric_audit()
    result: Json = dict(
        rows=rows,
        head_rows=heads,
        encoder_hashes=encoders,
        invariant_rows=invariants,
        clipping_rows=clipping,
        feature_references=feature_refs,
        basis_fixture_rows=[audit],
        optimizer_fixture_rows=[dict(audit, metric="finite_difference_and_convexity")],
        development_metrics={
            arm: float(
                np.mean(
                    [
                        r["value"]
                        for r in rows
                        if r["arm"] == arm and r["condition"] == "exposed_development_guard"
                    ]
                )
            )
            for arm in ARMS
        },
    )
    independent = {
        arm: math.fsum(
            r["numerator"]
            for r in rows
            if r["arm"] == arm and r["condition"] == "exposed_development_guard"
        )
        / 64
        for arm in ARMS
    }
    result["summary_recomputation"] = dict(
        method="independent stdlib fsum of primitive per-graph Brier rows",
        guard_brier=independent,
        passed=all(abs(independent[a] - result["development_metrics"][a]) <= 1e-15 for a in ARMS),
    )
    seal(raw / "independent_summary.json", result["summary_recomputation"], replay)
    seal(
        raw / "states.json",
        dict(
            heads=heads,
            encoders=states["relational_logistic"],
            old_shared_linear=old_shared,
            confidence_linear=states["confidence_logistic"],
            prevalence=float(np.mean(views["fit"]["labels"])),
        ),
        replay,
    )
    result["trained_state_reference"] = reference(raw / "states.json")
    seal(raw / "reduction.json", result, replay)
    progress("after_cached_benchmark", 6, 0)
    return result


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Mechanics readiness never grants a new signal gate or an oracle-distinct win."""
    from carnot.reporting.matched_corpus_a_8949 import qualified

    evidence, result = work["evidence"], work["result"]
    owned = qualified(validation, OWNED) if validation["receipts"] else False
    failures = evidence["failures"]
    mechanics = (
        result.get("summary_recomputation", {}).get("passed", False)
        and len(result.get("head_rows", [])) == 6
        and all(
            r["passed"]
            for r in result.get("invariant_rows", []) + result.get("basis_fixture_rows", [])
        )
    )
    disqualified = (
        bool(validation["receipts"])
        and not owned
        or not failures
        and not fixture
        and not owned
        or bool(result)
        and not mechanics
        or bool(work.get("owned_failure"))
    )
    cls = "disqualified" if disqualified else "blocked" if failures else "null"
    reason = (
        failures[0]["check"]
        if failures and not disqualified
        else "owned_checks"
        if disqualified
        else "convex_spline_mechanics_frozen"
    )
    rows = result.get("rows", [])
    completed = 320 if rows else 0
    if not rows:
        rows = [
            dict(
                unit_id=f"unavailable-{role}-{i}",
                group_id=f"unavailable-{role}-{i}",
                arm=arm,
                condition=f"exposed_development_{role}",
                metric="brier",
                value=None,
                numerator=None,
                denominator=1,
                status="censored",
                exclusion_reason="external_precondition"
                if failures
                else "owned_fit_or_validation_failure",
            )
            for role, count in [("fit", 256), ("guard", 64)]
            for i in range(count)
            for arm in ARMS
        ]
    protocol = evidence["protocol"]
    value: Json = dict(
        experiment_id=8986,
        task_id=TASK,
        milestone="2026.11.768",
        run_date="20261111",
        random_seed=SEED,
        honest_verdict=f"complete_{cls}_{reason}",
        verdict_class=cls,
        status="error" if disqualified else "blocked" if failures else "success",
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if result
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invoked=False,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[
            dict(
                arm=arm,
                parameter_count=count,
                seeds=[71, 72, 73],
                objective=protocol.get("objective"),
                optimizer=protocol.get("optimizer"),
            )
            for arm, count in [("spline", 217), ("converged_linear", 28)]
        ]
        if result
        else [],
        execution_venue="host",
        numerical_training_calls=6 if result else 0,
        rows=rows,
        intended_count=320,
        independent_count=completed,
        completed_count=completed,
        excluded_count=0,
        censored_count=320 - completed,
        failed_count=0,
        verifier_is_oracle=False,
        claim_scope="fixed-basis numerical mechanics only; no new scientific benefit; graph truth has a cheap BFS evaluator",
        exposure_scope="all V765/V766 labels are exposed development; original256 fit and64 guard graphs only",
        independent_generalization_score=0,
        spline_methods_ready_score=int(owned and mechanics and not failures and not fixture),
        fixture_only=fixture,
        fresh_collection_permitted=False,
        development_signal_gate="unmeasured; owned by Exp8987",
        required_checks_passed=owned,
        flagged_adversarial=False,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        gate_check_summary=evidence["checks"],
        preconditions_checked=work["preconditions"],
        protocol_path=str(ROOT / PROTOCOL),
        protocol_sha256=PROTOCOL_PIN,
        frozen_protocol=protocol,
        seven_bound_contract=protocol.get("seven_bound_contract", {}),
        convergence_rule=protocol.get("convergence_rule"),
        method_source_rows=evidence["method_source_rows"],
        basis_fixture_rows=result.get("basis_fixture_rows", []),
        optimizer_fixture_rows=result.get("optimizer_fixture_rows", []),
        encoder_hashes=result.get("encoder_hashes", []),
        feature_schema=dict(
            feature_count=27,
            allowed_keys=sorted(g.KEYS),
            readout=protocol.get("feature_readout", []),
            forbidden=protocol.get("forbidden_features", []),
        ),
        parameter_counts=dict(spline=217, linear=28),
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        source_artifact_hashes=evidence["references"],
        raw_shard_hashes=work["raw_references"],
        code_config_hashes=work["sources"],
        cited_upstream_artifacts=[
            dict(
                experiment_id=8972,
                sha256=prior.PIN,
                fields_imported=[
                    "diagnosis_ready_score",
                    "work_reference",
                    "candidate_reference",
                    "feature_hashes",
                    "head_state_references",
                ],
            ),
            dict(
                experiment_id=8973,
                sha256=CONTEXT_PIN,
                fields_imported=[
                    "required_checks_passed",
                    "shared_methods_ready_score",
                    "frozen_protocol",
                    "development_metrics",
                ],
            ),
        ],
        historical_context=dict(
            shared_methods_ready_score=evidence.get("context", {}).get(
                "shared_methods_ready_score"
            ),
            required_as_gate=False,
            reason="changed convex objective; preserve failed hidden-head result",
        ),
        global_test_debt=work["global_test_debt"],
        acceptance_gates=dict(
            authenticated_inputs=not failures,
            numerical_mechanics=mechanics,
            owned_checks=owned,
            later_scientific_benefit=False,
        ),
        methodology_note="An additive binary energy is a ridge logistic spline classifier. Frozen coefficients remove hidden-layer optimization as a variable. They do not diagnose a causal bug in the prior optimizer or guarantee benefit. Seven scientific bounds are frozen for later fresh observations; fixtures and exposed development cannot establish independent generalization.",
    )
    value = result | value
    value["field_principles"] = {
        key: "Bind authentic operands, graph units, numerical convergence and terminal scope; mechanics are not scientific evidence."
        for key in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value
