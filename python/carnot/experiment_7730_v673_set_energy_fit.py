"""Fit exposed finite set energies, with no generator load (REQ-REPORT-7730)."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import cache
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import build_scoped_commands, run_commands
from carnot.verify import source_alignment as alignment
from carnot.verify import source_set_energy as energy

ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7730_v673_set_energy_fit"
RAW = Path("results/raw") / NAME
OUTPUT = Path("results") / f"{NAME}.json"
UPSTREAM = ("experiment_7727_v673_development_corpus", "experiment_7728_v673_set_energy_protocol")
ROLES = ("fit", "tune", "policy")
ARMS = (
    "set_energy",
    "shared_location",
    "pooled_logistic",
    "pooled_mlp",
    "source_erased",
    "complete_static",
)
SEEDS = (67301, 67302, 67303, 67304, 67305)
CONFIGS = ((0.01, 0.0), (0.01, 0.001), (0.05, 0.0), (0.05, 0.001))
EPOCHS = 12
FEATURE_DICTIONARY = (
    "low_overlap",
    "negation_mismatch",
    "number_mismatch",
    "missing_evidence",
    "low_overlap_and_negation",
    "low_overlap_and_number",
    "low_overlap_and_missing",
    "negation_and_number",
    "negation_and_missing",
    "number_and_missing",
    "low_overlap_and_negation_and_number",
    "low_overlap_and_negation_and_missing",
    "low_overlap_and_number_and_missing",
    "negation_and_number_and_missing",
    "all_four",
    "zero_overlap_and_missing",
)
PRINCIPLE = "Measured evidence bounds the claim and downstream use."
_DATA_CACHE: dict[tuple[int, str], tuple[Any, Any]] = {}


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Flush phase boundaries and real completed units for the conductor."""
    print(
        f"[exp7730] {phase} {event} elapsed_s={time.monotonic() - start:.3f} completed={units}",
        flush=True,
    )


def digest(value: bytes) -> str:
    """Hash exact source or saved evidence bytes."""
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _view(example: Any) -> dict[str, Any]:
    """Use a cached public feature view when one exists."""
    if isinstance(example, dict):
        return example["view"]
    return energy.prepare(example[0], example[1])


def _advisory(view: dict[str, Any]) -> list[float]:
    """Evaluate the fixed advisory grammar without any human label."""
    units = view["answer_units"]
    pairs = [
        alignment.pair_features(window, [unit]) for unit in units for window in view["windows"]
    ]
    overlap = max((row[128] for row in pairs), default=0.0)
    negation = any(row[129] > 0 for row in pairs)
    number = any(row[130] > 0 for row in pairs)
    missing = not view["windows"] or overlap < 0.1
    a, b, c, d = overlap < 0.25, negation, number, missing
    return [
        float(value)
        for value in (
            a,
            b,
            c,
            d,
            a and b,
            a and c,
            a and d,
            b and c,
            b and d,
            c and d,
            a and b and c,
            a and b and d,
            a and c and d,
            b and c and d,
            a and b and c and d,
            overlap == 0 and d,
        )
    ]


def advisory_features(source: bytes, answer: bytes) -> list[float]:
    """Expose only label-free conjunctions from original source bytes."""
    return _advisory(energy.prepare(source, answer))


def policy_action(probability: float) -> str:
    """Minimize the fixed costs and favor escalation on exact ties."""
    costs = {"accept": 5 * probability, "reject": 1 - probability, "escalate": 0.25}
    return min(("escalate", "accept", "reject"), key=lambda action: costs[action])


def verified_role_file(raw: Path, role: dict[str, Any], kind: str) -> Path:
    """Reject absent or changed role files before reading any row."""
    path = raw / role[f"{kind}_path"]
    if not path.is_file() or sha256_file(path) != role[f"{kind}_sha256"]:
        raise ValueError(f"{kind}_hash_mismatch")
    return path


def _dataset(examples: list[Any], arm: str) -> tuple[Any, Any]:
    """Pad only public features; masked locations have zero probability."""
    key = (id(examples), arm)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    views = [_view(row) for row in examples]
    labels = jnp.asarray(
        [row["label"] if isinstance(row, dict) else row[2] for row in examples], dtype=jnp.float32
    )
    if arm in ("set_energy", "source_erased"):
        sets = []
        priors = []
        for view in views:
            if arm == "source_erased":
                view = energy.prepare(b"", view["answer_bytes"])
            groups = view["group_ids"]
            count = len(set(groups))
            prior = [1 / ((count + 1) * groups.count(g)) for g in groups] + [1 / (count + 1)]
            sets.append(
                [
                    [alignment.pair_features(window, [unit]) for window in [*view["windows"], b""]]
                    for unit in view["answer_units"]
                ]
            )
            priors.append(prior)
        max_units = max(map(len, sets))
        max_places = max(map(len, priors))
        x = np.zeros((len(sets), max_units, max_places, alignment.FEATURE_DIM), dtype=np.float32)
        log_prior = np.full((len(sets), max_places), -1e6, dtype=np.float32)
        unit_mask = np.zeros((len(sets), max_units), dtype=np.float32)
        for i, (units, prior) in enumerate(zip(sets, priors, strict=True)):
            x[i, : len(units), : len(prior)] = units
            log_prior[i, : len(prior)] = np.log(prior)
            unit_mask[i, : len(units)] = 1
        data: Any = (jnp.asarray(x), jnp.asarray(log_prior), jnp.asarray(unit_mask))
    elif arm == "shared_location":
        data = jnp.asarray([alignment.pooled_features(view) for view in views], dtype=jnp.float32)
        # Shared location is trained from the exact latent mixture below, not this pooled control.
        sets = []
        priors = []
        for view in views:
            groups = view["group_ids"]
            count = len(set(groups))
            priors.append([1 / ((count + 1) * groups.count(g)) for g in groups] + [1 / (count + 1)])
            sets.append(
                [*view["pair_features"], alignment.pair_features(b"", view["answer_units"])]
            )
        width = max(map(len, sets))
        x = np.zeros((len(sets), width, alignment.FEATURE_DIM), dtype=np.float32)
        lp = np.full((len(sets), width), -1e6, dtype=np.float32)
        for i, (rows, prior) in enumerate(zip(sets, priors, strict=True)):
            x[i, : len(rows)] = rows
            lp[i, : len(prior)] = np.log(prior)
        data = (jnp.asarray(x), jnp.asarray(lp))
    elif arm == "complete_static":
        data = jnp.asarray([_advisory(view) for view in views], dtype=jnp.float32)
    else:
        data = jnp.asarray([alignment.pooled_features(view) for view in views], dtype=jnp.float32)
    _DATA_CACHE[key] = (data, labels)
    return data, labels


@cache
def _trainer(arm: str) -> Any:
    """Compile one exact normalized binary objective per arm and input shape."""

    def objective(params: Any, data: Any, y: Any, ridge: float) -> Any:
        if arm in ("set_energy", "source_erased"):
            x, log_prior, unit_mask = data
            scores = (
                log_prior[:, None, :, None]
                - jnp.einsum("buld,cd->bulc", x, params["w"])
                - params["b"]
            )
            support = jnp.exp(
                jax.nn.logsumexp(scores[..., 1], axis=2) - jax.nn.logsumexp(scores, axis=(2, 3))
            )
            log_support = jnp.sum(jnp.log(jnp.clip(support, 1e-7, 1.0)) * unit_mask, axis=1)
            p = -jnp.expm1(log_support)
        elif arm == "shared_location":
            x, log_prior = data
            scores = log_prior[..., None] - jnp.einsum("bld,cd->blc", x, params["w"]) - params["b"]
            p = jnp.exp(
                jax.nn.logsumexp(scores[..., 0], axis=1) - jax.nn.logsumexp(scores, axis=(1, 2))
            )
        elif arm == "pooled_mlp":
            hidden = jnp.tanh(data @ params["w1"] + params["b1"])
            scores = hidden @ params["w2"] + params["b2"]
            p = jax.nn.softmax(scores, axis=1)[:, 1]
        else:
            scores = -data @ params["w"].T - params["b"]
            p = jax.nn.softmax(scores, axis=1)[:, 0]
        p = jnp.clip(p, 1e-7, 1 - 1e-7)
        nll = -jnp.mean(y * jnp.log(p) + (1 - y) * jnp.log1p(-p))
        penalty = sum(jnp.sum(value * value) for value in jax.tree_util.tree_leaves(params))
        return nll + ridge * penalty

    return jax.jit(jax.value_and_grad(objective))


def fit_head(
    arm: str, examples: list[Any], seed: int, learning_rate: float, ridge: float, epochs: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit binary NLL and record every honest epoch and gradient norm."""
    if (
        arm not in ARMS
        or not 1 <= epochs <= 200
        or seed not in SEEDS
        or (learning_rate, ridge) not in CONFIGS
    ):
        raise ValueError("unregistered_training_budget")
    data, labels = _dataset(examples, arm)
    width = 16 if arm == "complete_static" else alignment.FEATURE_DIM
    rng = np.random.default_rng(seed)
    if arm == "pooled_mlp":
        params = {
            "w1": jnp.asarray(rng.normal(0, 0.02, (width, energy.MLP_WIDTH)), dtype=jnp.float32),
            "b1": jnp.zeros(energy.MLP_WIDTH),
            "w2": jnp.asarray(rng.normal(0, 0.02, (energy.MLP_WIDTH, 2)), dtype=jnp.float32),
            "b2": jnp.zeros(2),
        }
    else:
        params = {
            "w": jnp.asarray(rng.normal(0, 0.01, (2, width)), dtype=jnp.float32),
            "b": jnp.zeros(2),
        }
    curve = []
    trainer = _trainer(arm)
    for epoch in range(1, epochs + 1):
        loss, grads = trainer(params, data, labels, ridge)
        norm = math.sqrt(sum(float(jnp.sum(g * g)) for g in jax.tree_util.tree_leaves(grads)))
        params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
        curve.append({"epoch": epoch, "fit_nll_plus_penalty": float(loss), "gradient_norm": norm})
        if epoch % 4 == 0:
            print(
                f"[exp7730] training {arm} seed={seed} epoch={epoch}/{epochs} loss={float(loss):.6f}",
                flush=True,
            )
    saved = {key: np.asarray(value).tolist() for key, value in params.items()}
    count = sum(np.asarray(value).size for value in params.values())
    return {
        "arm": arm,
        "seed": seed,
        "learning_rate": learning_rate,
        "ridge": ridge,
        "epochs": epochs,
        "parameter_count": count,
        "parameters": saved,
        "temperature": 1.0,
        "policy_costs": {"accept": "5*p", "reject": "1-p", "escalate": 0.25},
    }, curve


def _raw_probability(view: dict[str, Any], head: dict[str, Any]) -> float:
    """Run the saved finite energy through its original typed path."""
    arm = head["arm"]
    p = head["parameters"]
    if arm == "complete_static":
        x = _advisory(view)
        scores = [
            -sum(a * b for a, b in zip(row, x, strict=True)) - bias
            for row, bias in zip(p["w"], p["b"], strict=True)
        ]
        return 1 / (1 + math.exp(max(-60.0, min(60.0, scores[1] - scores[0]))))
    if arm == "pooled_mlp":
        params = {
            "w1": p["w1"],
            "b1": p["b1"],
            "w2": p["w2"],
            "b2": p["b2"],
            "parameter_count": head["parameter_count"],
        }
        return energy.pooled_control(view, params, "mlp")[1]
    params = {"weights": p["w"], "bias": p["b"]}
    if arm == "set_energy":
        return energy.distribution(view, params)["response_unsupported"]
    if arm == "source_erased":
        erased = energy.prepare(b"", view["answer_bytes"])
        return energy.distribution(erased, params)["response_unsupported"]
    if arm == "shared_location":
        return energy.shared_control(view, params)[0]
    return energy.pooled_control(view, params, "logistic")[0]


def _temperature_probability(p: float, temperature: float) -> float:
    """Scale final binary odds without changing the learned location model."""
    clipped = min(1 - 1e-10, max(1e-10, p))
    logit = math.log(clipped / (1 - clipped)) / temperature
    return 1 / (1 + math.exp(-logit))


def fit_temperature(head: dict[str, Any], examples: list[Any]) -> float:
    """Minimize tune NLL on a predeclared bounded scalar grid."""
    probabilities = [_raw_probability(_view(row), head) for row in examples]
    labels = [row["label"] if isinstance(row, dict) else row[2] for row in examples]
    grid = np.geomspace(0.25, 4.0, 65)
    scores = [
        sum(
            -y * math.log(max(1e-10, _temperature_probability(p, float(t))))
            - (1 - y) * math.log(max(1e-10, 1 - _temperature_probability(p, float(t))))
            for p, y in zip(probabilities, labels, strict=True)
        )
        / len(labels)
        for t in grid
    ]
    return float(grid[int(np.argmin(scores))])


def typed_decision(source: bytes, answer: bytes, head: dict[str, Any]) -> dict[str, Any]:
    """Reload-safe probability and action from original source and answer bytes."""
    view = energy.prepare(source, answer)
    if view["abstention"]:
        return {
            "probability_unsupported": None,
            "action": "escalate",
            "censored": view["abstention"],
        }
    p = _temperature_probability(_raw_probability(view, head), head["temperature"])
    return {"probability_unsupported": p, "action": policy_action(p), "censored": None}


def _metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain denominators, NLL, Brier, calibration bins and policy cost."""
    eligible = [row for row in predictions if row["probability_unsupported"] is not None]
    n = len(eligible)
    bins = []
    for index in range(10):
        members = [
            row for row in eligible if min(9, int(row["probability_unsupported"] * 10)) == index
        ]
        bins.append(
            {
                "bin": index,
                "count": len(members),
                "mean_probability": sum(r["probability_unsupported"] for r in members)
                / len(members)
                if members
                else None,
                "observed_fraction": sum(r["label"] for r in members) / len(members)
                if members
                else None,
            }
        )
    return {
        "denominator": len(predictions),
        "eligible": n,
        "censored": len(predictions) - n,
        "brier": sum((r["probability_unsupported"] - r["label"]) ** 2 for r in eligible) / n
        if n
        else None,
        "nll": sum(
            -r["label"] * math.log(max(1e-10, r["probability_unsupported"]))
            - (1 - r["label"]) * math.log(max(1e-10, 1 - r["probability_unsupported"]))
            for r in eligible
        )
        / n
        if n
        else None,
        "decision_cost": sum(
            0.25
            if r["action"] == "escalate"
            else 5 * r["label"]
            if r["action"] == "accept"
            else 1 - r["label"]
            for r in eligible
        )
        / n
        if n
        else None,
        "always_escalate_cost": 0.25 if n else None,
        "action_counts": dict(Counter(r["action"] for r in predictions)),
        "calibration_bins": bins,
    }


def check_preconditions(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    """Check exact producer and role schemas before opening permitted labels."""
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {
        "eligible_producers": [],
        "flagged_historical_inputs": [],
        "pre_gate_receipts": [],
        "absent_sources": [],
    }
    expected_fields = ("development_cohort_ready_score", "set_protocol_ready_score")
    for name, field in zip(UPSTREAM, expected_fields, strict=True):
        path = root / "results" / f"{name}.json"
        value = json.loads(path.read_text()) if path.is_file() else {}
        observed = value.get(field)
        passed = (
            observed == 1
            and value.get("flagged_adversarial") is False
            and str(value.get("honest_verdict", "")).startswith("complete_")
        )
        checks.append(
            {
                "path": str(path.relative_to(root)),
                "exists": path.is_file(),
                "schema_valid": isinstance(value, dict),
                "field": field,
                "operator": "==",
                "expected": 1,
                "observed": observed,
                "sha256": sha256_file(path) if path.is_file() else None,
                "passed": passed,
            }
        )
        if passed:
            hashes["eligible_producers"].append(
                {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
            )
        else:
            failures.append(
                {
                    "check": "producer_gate",
                    "upstream_id": name,
                    "artifact_path": str(path.relative_to(root)),
                    "field": field,
                    "operator": "==",
                    "expected": 1,
                    "observed": observed,
                }
            )
            hashes["absent_sources"].append(str(path.relative_to(root)))
    manifest_path = (
        root / "results/raw/experiment_7727_v673_development_corpus/development_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    valid = bool(
        manifest
        and manifest.get("schema") == "carnot.exp7727.development_manifest.v1"
        and all(
            manifest["counts"].get(role) == count
            for role, count in (("fit", 256), ("tune", 64), ("policy", 64))
        )
    )
    checks.append(
        {
            "path": str(manifest_path.relative_to(root)),
            "exists": manifest_path.is_file(),
            "schema_valid": valid,
            "sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
            "passed": valid,
        }
    )
    if not valid:
        failures.append(
            {
                "check": "development_manifest_schema",
                "upstream_id": UPSTREAM[0],
                "artifact_path": str(manifest_path.relative_to(root)),
                "field": "schema_and_counts",
                "operator": "==",
                "expected": "v1; fit256 tune64 policy64",
                "observed": manifest.get("schema") if manifest else None,
            }
        )
    if valid:
        from carnot.experiment_7727_v673_development_corpus import validate_public

        raw = manifest_path.parent
        for role in ROLES:
            meta = manifest["roles"][role]
            for kind in ("public", "evaluator"):
                try:
                    path = verified_role_file(raw, meta, kind)
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    if len(rows) != meta["count"]:
                        raise ValueError("row_count_mismatch")
                    if kind == "public":
                        for row in rows:
                            validate_public(row, role)
                    elif any(
                        set(row) != {"annotations", "family_id", "label", "response_id"}
                        or row["label"] not in (0, 1)
                        for row in rows
                    ):
                        raise ValueError("evaluator_schema_mismatch")
                    checks.append(
                        {
                            "path": str(path.relative_to(root)),
                            "exists": True,
                            "schema_valid": True,
                            "sha256": sha256_file(path),
                            "count": len(rows),
                            "passed": True,
                        }
                    )
                    hashes["eligible_producers"].append(
                        {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
                    )
                except (OSError, ValueError, KeyError, TypeError) as error:
                    relative = f"results/raw/{UPSTREAM[0]}/{meta.get(f'{kind}_path', 'missing')}"
                    failures.append(
                        {
                            "check": "role_file",
                            "upstream_id": UPSTREAM[0],
                            "artifact_path": relative,
                            "field": f"{kind}_sha256_and_schema",
                            "operator": "==",
                            "expected": meta.get(f"{kind}_sha256"),
                            "observed": str(error),
                        }
                    )
                    checks.append({"path": relative, "passed": False, "observed": str(error)})
    checks.append(
        {
            "resource": "repo_root",
            "path": str(root),
            "exists": root.is_dir(),
            "effective_coding_backend": os.environ.get("CODEX_MODEL", "gpt-6 (session)"),
            "jax_backend": jax.default_backend(),
            "cpu_count": os.cpu_count(),
        }
    )
    return checks, failures, hashes, manifest


def load_roles(raw: Path, manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Join only fit, tune and policy labels to authenticated public bytes."""
    result = {}
    families: set[str] = set()
    for role in ROLES:
        meta = manifest["roles"][role]
        public = [
            json.loads(line)
            for line in verified_role_file(raw, meta, "public").read_text().splitlines()
        ]
        labels = {
            row["family_id"]: row
            for row in (
                json.loads(line)
                for line in verified_role_file(raw, meta, "evaluator").read_text().splitlines()
            )
        }
        if len(public) != meta["count"] or len(labels) != len(public):
            raise ValueError("role_count_mismatch")
        rows = []
        for row in public:
            family = row["family_id"]
            if family in families or labels[family]["response_id"] != row["response_id"]:
                raise ValueError("family_or_response_mismatch")
            families.add(family)
            source = row["complete_source"].encode()
            answer = row["complete_response"].encode()
            rows.append(
                {
                    "family_id": family,
                    "role": role,
                    "source": source,
                    "answer": answer,
                    "source_sha256": row["source_sha256"],
                    "answer_sha256": row["response_sha256"],
                    "label": labels[family]["label"],
                    "view": energy.prepare(source, answer),
                }
            )
        result[role] = rows
    return result


def _predictions(rows: list[dict[str, Any]], head: dict[str, Any]) -> list[dict[str, Any]]:
    """Retain one auditable family row, including abstentions."""
    result = []
    for row in rows:
        decision = typed_decision(row["source"], row["answer"], head)
        result.append(
            {
                "family_id": row["family_id"],
                "role": row["role"],
                "arm": head["arm"],
                "seed": head["seed"],
                "label": row["label"],
                "source_sha256": row["source_sha256"],
                "answer_sha256": row["answer_sha256"],
                "denominator": 1,
                "excluded": False,
                "exclusions": [],
                **decision,
            }
        )
    return result


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a complete deterministic raw table before publication."""
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def train_and_checkpoint(
    root: Path, roles: dict[str, list[dict[str, Any]]], start: float
) -> dict[str, Any]:
    """Fit every declared trial, select by tune NLL and save all parameters."""
    raw = root / RAW
    heads_dir = raw / "heads"
    heads_dir.mkdir(parents=True, exist_ok=True)
    feature_path = raw / "feature_dictionary.json"
    atomic_json(
        feature_path,
        {
            "schema": "carnot.exp7730.advisory.v1",
            "features": list(FEATURE_DICTIONARY),
            "complete_static_closure": list(FEATURE_DICTIONARY),
            "advisory_only": True,
        },
    )
    all_trials = []
    training_rows = []
    selected = {}
    eligible_fit = [r for r in roles["fit"] if not r["view"]["abstention"]]
    eligible_tune = [r for r in roles["tune"] if not r["view"]["abstention"]]
    total = len(ARMS) * len(SEEDS) * len(CONFIGS)
    completed = 0
    for arm in ARMS:
        candidates = []
        for seed in SEEDS:
            for learning_rate, ridge in CONFIGS:
                progress(
                    start, "training", f"before_{arm}_{seed}_{learning_rate}_{ridge}", completed
                )
                head, curve = fit_head(arm, eligible_fit, seed, learning_rate, ridge, EPOCHS)
                head["temperature"] = fit_temperature(head, eligible_tune)
                predictions = _predictions(eligible_tune, head)
                score = _metrics(predictions)["nll"]
                head["tune_nll"] = score
                filename = f"{arm}_{seed}_{learning_rate}_{ridge}.json"
                path = heads_dir / filename
                atomic_json(path, head)
                reloaded = json.loads(path.read_text())
                original = typed_decision(
                    eligible_tune[0]["source"], eligible_tune[0]["answer"], head
                )
                replay = typed_decision(
                    eligible_tune[0]["source"], eligible_tune[0]["answer"], reloaded
                )
                if (
                    abs(original["probability_unsupported"] - replay["probability_unsupported"])
                    > 1e-8
                    or original["action"] != replay["action"]
                ):
                    raise ValueError("head_reload_mismatch")
                reference = {
                    "arm": arm,
                    "seed": seed,
                    "learning_rate": learning_rate,
                    "ridge": ridge,
                    "epochs": EPOCHS,
                    "temperature": head["temperature"],
                    "tune_nll": score,
                    "parameter_count": head["parameter_count"],
                    "path": str(path.relative_to(root)),
                    "sha256": sha256_file(path),
                    "policy_costs": head["policy_costs"],
                }
                candidates.append((score, head, reference))
                all_trials.append(reference)
                training_rows.extend(
                    {
                        "kind": "epoch",
                        "arm": arm,
                        "seed": seed,
                        "learning_rate": learning_rate,
                        "ridge": ridge,
                        **epoch,
                    }
                    for epoch in curve
                )
                completed += 1
                progress(
                    start, "training", f"after_{arm}_{seed}_{learning_rate}_{ridge}", completed
                )
        best = min(
            candidates,
            key=lambda candidate: (
                candidate[0],
                candidate[2]["seed"],
                candidate[2]["learning_rate"],
                candidate[2]["ridge"],
            ),
        )
        selected[arm] = best[2]
        progress(start, "selection", arm, len(selected))
    manifest = {
        "schema": "carnot.exp7730.head_manifest.v1",
        "seeds": list(SEEDS),
        "configurations": [list(x) for x in CONFIGS],
        "epochs": EPOCHS,
        "selection_metric": "tune_nll",
        "trials": all_trials,
        "selected": selected,
        "policy_costs": {"accept": "5*p", "reject": "1-p", "escalate": 0.25},
    }
    model_path = raw / "model_manifest.json"
    atomic_json(model_path, manifest)
    return {
        "manifest": manifest,
        "training_rows": training_rows,
        "model_path": model_path,
        "feature_path": feature_path,
        "completed": completed,
        "total": total,
    }


def reduce_raw(root: Path, raw: Path, candidate: Path | None = None) -> dict[str, Any]:
    """Reopen raw heads and original public bytes in a fresh-process replay."""
    manifest = json.loads((raw / "model_manifest.json").read_text())
    rows = [json.loads(line) for line in (raw / "rows.jsonl").read_text().splitlines()]
    development = json.loads(
        (root / f"results/raw/{UPSTREAM[0]}/development_manifest.json").read_text()
    )
    public_root = root / "results/raw" / UPSTREAM[0]
    public = {}
    for role in ROLES:
        for row in (
            json.loads(line)
            for line in verified_role_file(public_root, development["roles"][role], "public")
            .read_text()
            .splitlines()
        ):
            public[row["family_id"]] = row
    if len(public) != 384 or len(rows) != 384 * len(ARMS):
        raise ValueError("raw_family_or_arm_count_mismatch")
    for arm, reference in manifest["selected"].items():
        path = root / reference["path"]
        if sha256_file(path) != reference["sha256"]:
            raise ValueError("head_hash_mismatch")
        head = json.loads(path.read_text())
        arm_rows = [row for row in rows if row["arm"] == arm]
        if len(arm_rows) != 384:
            raise ValueError("selected_arm_count_mismatch")
        for row in arm_rows:
            source = public[row["family_id"]]["complete_source"].encode()
            answer = public[row["family_id"]]["complete_response"].encode()
            replay = typed_decision(source, answer, head)
            if (
                row["source_sha256"] != digest(source)
                or row["answer_sha256"] != digest(answer)
                or row["action"] != replay["action"]
                or row["censored"] != replay["censored"]
                or (
                    row["probability_unsupported"] is not None
                    and abs(row["probability_unsupported"] - replay["probability_unsupported"])
                    > 1e-8
                )
            ):
                raise ValueError("raw_prediction_mismatch")
    if candidate is not None:
        value = json.loads(candidate.read_text())
        if value["rows"] != [row for row in rows if row["role"] == "policy"]:
            raise ValueError("candidate_rows_mismatch")
        if value["model_manifest_path"]["sha256"] != sha256_file(raw / "model_manifest.json"):
            raise ValueError("candidate_manifest_mismatch")
    return {"rows": rows, "families": len(public), "arms": len(manifest["selected"]), "ready": True}


def _span(
    name: str, started: float, ended: float, units: int, checkpoint: str | None, date: str
) -> dict[str, Any]:
    """Record one nonoverlapping measured monotonic phase."""
    return {
        "phase": name,
        "start_monotonic_s": started,
        "end_monotonic_s": ended,
        "duration_s": ended - started,
        "run_date": date,
        "heartbeat_times": [started, ended],
        "completed_units": units,
        "checkpoint_hash": checkpoint,
    }


def build_artifact(
    root: Path,
    date: str,
    checks: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    hashes: dict[str, Any],
    trained: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    scope: dict[str, Any],
) -> dict[str, Any]:
    """State only measured development readiness and terminal limitations."""
    raw = root / RAW
    required = scope["required_names"]
    checks_pass = len(receipts) == len(required) and all(r["passed"] for r in receipts)
    ready = bool(
        trained
        and len(trained["manifest"]["selected"]) == len(ARMS)
        and checks_pass
        and not failures
    )
    verdict = "blocked" if failures else "null" if ready else "disqualified"
    policy = (
        {
            arm: _metrics([r for r in rows if r["role"] == "policy" and r["arm"] == arm])
            for arm in ARMS
        }
        if rows
        else {}
    )
    tune = (
        {
            arm: _metrics([r for r in rows if r["role"] == "tune" and r["arm"] == arm])
            for arm in ARMS
        }
        if rows
        else {}
    )
    gates = {
        "validity": bool(trained and not failures),
        "readiness": int(ready) if trained else None,
        "brier_score": {arm: metrics["brier"] for arm, metrics in policy.items()}
        if policy
        else None,
        "decision_cost": {arm: metrics["decision_cost"] for arm, metrics in policy.items()}
        if policy
        else None,
        "coverage": {
            arm: (
                metrics["action_counts"].get("accept", 0)
                + metrics["action_counts"].get("reject", 0)
            )
            / metrics["denominator"]
            for arm, metrics in policy.items()
        }
        if policy
        else None,
        "retention": None,
        "efficiency": None,
    }
    inputs = hashes["eligible_producers"]
    checksum = digest(
        json.dumps(
            {
                "configuration": [ARMS, SEEDS, CONFIGS, EPOCHS],
                "inputs": inputs,
                "reducer_sha256": sha256_file(
                    root / "python/carnot/experiment_7730_v673_set_energy_fit.py"
                ),
            },
            sort_keys=True,
        ).encode()
    )
    result: dict[str, Any] = {
        "experiment_id": "exp7730-set-energy-fit",
        "milestone": "2026.09.673",
        "run_date": date,
        "honest_verdict": f"complete_{verdict}_set_energy_fit",
        "verdict_class": verdict,
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "acceptance_gate_results": gates,
        "rows": [row for row in rows if row["role"] == "policy"],
        "sample_size_budget": {
            "intended": 384,
            "observed": 384 if rows else 0,
            "eligible": sum(row["censored"] is None for row in rows if row["arm"] == ARMS[0])
            if rows
            else 0,
            "excluded": 0,
            "censored": sum(row["censored"] is not None for row in rows if row["arm"] == ARMS[0])
            if rows
            else 0,
            "effective_independent_families": 384 if rows else 0,
            "arms_do_not_increase_n": True,
        },
        "claim_scope": {
            "value": "development_only",
            "fresh_generalization_eligible": False,
            "fixture_reference": "circular_positive",
            "adapter_withheld_public": None,
        },
        "inference_substrate": "cpu_jax_finite_energy_training_no_llm",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "model_invocation_counts": {
            key: 0
            for key in (
                "loads",
                "forwards",
                "generations",
                "input_tokens",
                "output_tokens",
                "failures",
                "cancellations",
            )
        },
        "execution_venue": "host",
        "execution_venue_details": {"host": platform.node(), "pid": os.getpid(), "gpu_uuid": None},
        "phase_spans": spans,
        "random_seed": {
            "fit_seeds": list(SEEDS),
            "purpose": "head_initialization",
            "selection": "tune_nll_only",
        },
        "reproducibility_checksum": checksum,
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": {
            "affected_scope": scope,
            "commands": receipts,
            "cold_replay": None,
            "terminal_checks_path": str((RAW / "terminal_checks.json")),
            "global_suite_debt": "tracked separately; not a science readiness gate",
        },
        "verifier_is_oracle": False,
        "set_heads_ready_score": int(ready),
        "model_manifest_path": {
            "path": str(RAW / "model_manifest.json"),
            "sha256": sha256_file(raw / "model_manifest.json") if trained else None,
        },
        "feature_dictionary_path": {
            "path": str(RAW / "feature_dictionary.json"),
            "sha256": sha256_file(raw / "feature_dictionary.json") if trained else None,
        },
        "training_rows": {
            "path": str(RAW / "training_rows.jsonl"),
            "sha256": sha256_file(raw / "training_rows.jsonl") if trained else None,
            "count": len(trained["training_rows"]) if trained else 0,
        },
        "tune_metrics": tune,
        "policy_metrics": policy,
        "always_escalate_cost": 0.25 if policy else None,
        "field_principles": {},
    }
    result["field_principles"] = {name: PRINCIPLE for name in [*result, *gates]}
    return result


def run_experiment(root: Path, date: str) -> dict[str, Any]:
    """Run the registered fit, scoped checks, cold replay and terminal readers."""
    root = root.resolve(strict=True)
    start = time.monotonic()
    progress(start, "preconditions", "start")
    raw = root / RAW
    (raw / "tmp").mkdir(parents=True, exist_ok=True)
    spans = []
    phase = time.monotonic()
    checks, failures, hashes, manifest = check_preconditions(root)
    commands = build_scoped_commands(
        root,
        [f"tests/python/test_{NAME}.py"],
        [f"python/carnot/{NAME}.py"],
        static_paths=[f"scripts/experiments/{NAME}.py"],
        basetemp=raw / "tmp",
        coverage_file=raw / ".coverage",
    )
    scope = {
        "test_paths": [f"tests/python/test_{NAME}.py"],
        "changed_modules": [f"python/carnot/{NAME}.py"],
        "static_paths": [f"scripts/experiments/{NAME}.py"],
        "required_names": [item.name for item in commands],
    }
    atomic_json(raw / "frozen_affected_scope.json", scope)
    spans.append(
        _span(
            "preconditions",
            phase,
            time.monotonic(),
            len(checks),
            sha256_file(raw / "frozen_affected_scope.json"),
            date,
        )
    )
    progress(start, "preconditions", "end", len(checks))
    trained = None
    rows: list[dict[str, Any]] = []
    if not failures and manifest is not None:
        phase = time.monotonic()
        progress(start, "feature_extraction", "start")
        roles = load_roles(root / "results/raw" / UPSTREAM[0], manifest)
        counts = {role: len(roles[role]) for role in ROLES}
        atomic_json(raw / "role_counts.json", counts)
        spans.append(
            _span(
                "feature_extraction",
                phase,
                time.monotonic(),
                sum(counts.values()),
                sha256_file(raw / "role_counts.json"),
                date,
            )
        )
        progress(start, "feature_extraction", "end", sum(counts.values()))
        phase = time.monotonic()
        progress(start, "training", "start")
        trained = train_and_checkpoint(root, roles, start)
        spans.append(
            _span(
                "training",
                phase,
                time.monotonic(),
                trained["completed"],
                sha256_file(trained["model_path"]),
                date,
            )
        )
        progress(start, "training", "end", trained["completed"])
        phase = time.monotonic()
        progress(start, "prediction", "start")
        for arm, reference in trained["manifest"]["selected"].items():
            head = json.loads((root / reference["path"]).read_text())
            for role in ROLES:
                predictions = _predictions(roles[role], head)
                rows.extend(predictions)
                trained["training_rows"].extend(
                    {"kind": "family_prediction", **row} for row in predictions
                )
            progress(start, "prediction", arm, len(rows))
        _write_jsonl(raw / "rows.jsonl", rows)
        _write_jsonl(raw / "training_rows.jsonl", trained["training_rows"])
        spans.append(
            _span(
                "prediction",
                phase,
                time.monotonic(),
                len(rows),
                sha256_file(raw / "rows.jsonl"),
                date,
            )
        )
        progress(start, "prediction", "end", len(rows))
    progress(start, "validation", "start")
    phase = time.monotonic()
    receipts = run_commands(
        root,
        commands,
        log_dir=raw / "validation_logs",
        extra_env={"COVERAGE_FILE": str(raw / ".coverage"), "JAX_PLATFORMS": "cpu"},
        heartbeat_s=30,
    )
    spans.append(
        _span(
            "validation",
            phase,
            time.monotonic(),
            len(receipts),
            digest(json.dumps(receipts, sort_keys=True).encode()),
            date,
        )
    )
    progress(start, "validation", "end", len(receipts))
    for path in (
        "model_manifest.json",
        "feature_dictionary.json",
        "rows.jsonl",
        "training_rows.jsonl",
        "frozen_affected_scope.json",
    ):
        file = raw / path
        if file.is_file():
            hashes["pre_gate_receipts"].append(
                {"path": str((RAW / path)), "sha256": sha256_file(file)}
            )
    artifact = build_artifact(
        root, date, checks, failures, hashes, trained, rows, receipts, spans, scope
    )
    candidate = raw / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec

    terminal = [
        CommandSpec(
            "cold_replay",
            (
                str(root / ".venv/bin/python"),
                "-u",
                "-m",
                f"carnot.{NAME}",
                "--cold-reduce",
                str(raw),
                "--candidate",
                str(candidate),
            ),
            "exact_candidate",
            600,
        ),
        CommandSpec(
            "adversarial_verify",
            (
                str(root / ".venv/bin/python"),
                "-u",
                "scripts/adversarial_verify.py",
                "--json",
                str(candidate),
            ),
            "exact_candidate",
            600,
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                str(root / ".venv/bin/python"),
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact_candidate",
            600,
        ),
    ]
    phase = time.monotonic()
    progress(start, "terminal", "start")
    terminal_receipts = run_commands(
        root,
        terminal,
        log_dir=raw / "terminal_logs",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=30,
    )
    if not all(item["passed"] for item in terminal_receipts):
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
        artifact["set_heads_ready_score"] = 0
        artifact["flagged_adversarial"] = not terminal_receipts[1]["passed"]
        artifact["gate_check_summary"] = [
            {
                "check": item["name"],
                "upstream_id": NAME,
                "artifact_path": str(candidate.relative_to(root)),
                "field": "exit_code",
                "operator": "==",
                "expected": 0,
                "observed": item["exit_code"],
            }
            for item in terminal_receipts
            if not item["passed"]
        ]
        atomic_json(candidate, artifact)
        terminal_receipts = run_commands(
            root,
            terminal,
            log_dir=raw / "terminal_logs",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
    atomic_json(
        raw / "terminal_checks.json",
        {"candidate_sha256": sha256_file(candidate), "commands": terminal_receipts},
    )
    spans.append(
        _span(
            "terminal",
            phase,
            time.monotonic(),
            len(terminal_receipts),
            sha256_file(raw / "terminal_checks.json"),
            date,
        )
    )
    progress(start, "terminal", "end", len(terminal_receipts))
    destination = root / OUTPUT
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_bytes(candidate.read_bytes())
    os.replace(temporary, destination)
    progress(start, "publish", "end", 1)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Dispatch a live fit or the fresh-process cold reducer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260926")
    parser.add_argument("--cold-reduce", type=Path)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce is not None:
        if args.candidate is None:
            parser.error("--candidate required")
        reduce_raw(ROOT, args.cold_reduce, args.candidate)
        print("cold reduction passed", flush=True)
        return 0
    artifact = run_experiment(ROOT, args.date)
    return 0 if artifact["verdict_class"] in {"null", "blocked", "circular_positive"} else 1


if __name__ == "__main__":
    sys.exit(main())
