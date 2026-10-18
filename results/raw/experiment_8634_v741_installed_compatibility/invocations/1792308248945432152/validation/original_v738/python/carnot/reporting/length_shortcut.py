"""Measure a length shortcut on the exposed human-labelled source cohort.

REQ-REPORT-7848. Public bytes enter the predictor; evaluator labels enter only
after prediction files have been sealed. This module loads no language model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file


Json = dict[str, Any]
TEMPERATURE_GRID = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
ROLE_COUNTS = {
    "fit": 256,
    "tune": 64,
    "evaluation": 64,
    "policy": 64,
    "retention": 32,
    "online_admission": 64,
    "online_update": 96,
}
QUALIFICATION = "results/experiment_7810_v679_source_view_qualification.json"
ENERGY = "results/experiment_7840_energy_fit.json"


class CustodyError(ValueError):
    """A required science input was absent, changed, or unqualified."""

    def __init__(self, failures: list[Json]) -> None:
        self.failures = failures
        super().__init__(str(failures))


def failure(upstream: str, path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Name an exact failed gate operand and the bytes that carried it."""
    return {
        "upstream_id": upstream,
        "path": str(path),
        "sha256": sha256_file(path) if path.is_file() else None,
        "artifact_field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
    }


def digest_text(value: str) -> str:
    """Hash original UTF-8 bytes rather than a tokenizer's encoding."""
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def verify_raw_row(row: Mapping[str, Any]) -> None:
    """Require both declared hashes to match the public original strings."""
    if digest_text(row["complete_source"]) != row["source_sha256"]:
        raise ValueError("source_hash_mismatch")
    if digest_text(row["complete_response"]) != row["response_sha256"]:
        raise ValueError("response_hash_mismatch")


def _jsonl(path: Path) -> list[Json]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def authenticate(root: Path, qualification_path: Path | None = None) -> Json:
    """Cold-check Exp7810, its canonical roster, and all 640 public/evaluator joins."""
    root = root.resolve()
    qpath = qualification_path or root / QUALIFICATION
    if not qpath.is_file():
        raise CustodyError([failure("exp7810", qpath, "is_file", True, False)])
    q = json.loads(qpath.read_text(encoding="utf-8"))
    failures = []
    for field, expected in (
        ("verdict_class", "circular_positive"),
        ("evidence_view_ready_score", 1),
        ("flagged_adversarial", False),
    ):
        if q.get(field) != expected:
            failures.append(failure("exp7810", qpath, field, expected, q.get(field)))
    mpath = Path(q.get("source_view_manifest_path", ""))
    if not mpath.is_file():
        failures.append(failure("exp7810", mpath, "is_file", True, False))
    if failures:
        raise CustodyError(failures)
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("role_hashes") != q.get("role_hashes"):
        raise CustodyError(
            [
                failure(
                    "exp7810",
                    mpath,
                    "role_hashes",
                    q.get("role_hashes"),
                    manifest.get("role_hashes"),
                )
            ]
        )
    rows_path = Path(manifest["rows_path"])
    observed_rows_hash = sha256_file(rows_path) if rows_path.is_file() else None
    if observed_rows_hash != manifest["rows_sha256"]:
        raise CustodyError(
            [
                failure(
                    "exp7810", rows_path, "rows_sha256", manifest["rows_sha256"], observed_rows_hash
                )
            ]
        )
    dpath = Path(manifest["development_manifest_path"])
    expected_dev_hash = manifest["development_manifest_sha256"]
    if not dpath.is_file() or sha256_file(dpath) != expected_dev_hash:
        raise CustodyError(
            [
                failure(
                    "exp7727",
                    dpath,
                    "development_manifest_sha256",
                    expected_dev_hash,
                    sha256_file(dpath) if dpath.is_file() else None,
                )
            ]
        )
    dev = json.loads(dpath.read_text(encoding="utf-8"))
    if dev.get("counts") != ROLE_COUNTS or manifest.get("role_counts") != ROLE_COUNTS:
        raise CustodyError(
            [failure("exp7810", mpath, "role_counts", ROLE_COUNTS, manifest.get("role_counts"))]
        )
    public: dict[str, list[Json]] = {}
    evaluator_paths: dict[str, Path] = {}
    seen_families: set[str] = set()
    sources = [
        dict(
            upstream_id="exp7810",
            path=str(qpath),
            sha256=sha256_file(qpath),
            eligibility="qualified_development",
            date=q.get("run_date"),
        ),
        dict(
            upstream_id="exp7810",
            path=str(mpath),
            sha256=sha256_file(mpath),
            eligibility="canonical_manifest",
            date=q.get("run_date"),
        ),
        dict(
            upstream_id="exp7727",
            path=str(dpath),
            sha256=sha256_file(dpath),
            eligibility="exposed_development",
            date="20260926",
        ),
    ]
    for role, count in ROLE_COUNTS.items():
        info = dev["roles"][role]
        ppath = dpath.parent / info["public_path"]
        epath = dpath.parent / info["evaluator_path"]
        for kind, path in (("public", ppath), ("evaluator", epath)):
            expected = info[f"{kind}_sha256"]
            observed = sha256_file(path) if path.is_file() else None
            if observed != expected or q["role_hashes"][role][kind] != expected:
                raise CustodyError(
                    [failure("exp7727", path, f"{role}.{kind}_sha256", expected, observed)]
                )
            sources.append(
                dict(
                    upstream_id="exp7727",
                    path=str(path),
                    sha256=observed,
                    role=role,
                    kind=kind,
                    eligibility="exposed_development",
                    date="20260926",
                )
            )
        rows = _jsonl(ppath)
        if len(rows) != count or [r["family_id"] for r in rows] != info["families"]:
            raise CustodyError(
                [
                    failure(
                        "exp7727",
                        ppath,
                        f"{role}.families",
                        info["families"],
                        [r["family_id"] for r in rows],
                    )
                ]
            )
        for row in rows:
            verify_raw_row(row)
            if row["family_id"] in seen_families or row["role"] != role:
                raise CustodyError(
                    [failure("exp7727", ppath, "unique_family_and_role", True, False)]
                )
            seen_families.add(row["family_id"])
        labels = _jsonl(epath)
        if len(labels) != count or any(
            label["family_id"] != row["family_id"]
            or label["response_id"] != row["response_id"]
            or label["label"] not in (0, 1)
            for label, row in zip(labels, rows, strict=True)
        ):
            raise CustodyError([failure("exp7727", epath, "evaluator_join", count, len(labels))])
        public[role], evaluator_paths[role] = rows, epath
    if len(seen_families) != 640:
        raise CustodyError([failure("exp7727", dpath, "family_count", 640, len(seen_families))])
    return dict(
        public=public,
        evaluator_paths=evaluator_paths,
        role_counts=ROLE_COUNTS,
        source_artifact_hashes=sources,
        preconditions_checked=[
            dict(
                upstream_id=s["upstream_id"],
                path=s["path"],
                sha256=s["sha256"],
                artifact_field="sha256",
                op="==",
                expected=s["sha256"],
                observed=s["sha256"],
                passed=True,
            )
            for s in sources
        ],
        canonical_manifest_path=str(mpath),
    )


def public_features(row: Mapping[str, Any]) -> tuple[float, float, float]:
    """Use three registered public length inputs, measured in UTF-8 bytes."""
    answer = math.log1p(len(row["complete_response"].encode("utf-8")))
    source = math.log1p(len(row["complete_source"].encode("utf-8")))
    return answer, source, answer / source if source else 0.0


def _labels(path: Path, public: Sequence[Json]) -> dict[str, int]:
    """Open evaluator-only labels after the caller has sealed public predictions."""
    labels = _jsonl(path)
    if len(labels) != len(public):
        raise ValueError("evaluator_count_mismatch")
    result = {}
    for item, row in zip(labels, public, strict=True):
        if item["family_id"] != row["family_id"] or item["label"] not in (0, 1):
            raise ValueError("evaluator_join_mismatch")
        result[item["family_id"]] = item["label"]
    return result


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


def fit_model(public: Sequence[Json], labels: Mapping[str, int]) -> Json:
    """Fit fixed-batch ridge logistic regression with 200 deterministic steps."""
    if not public or set(labels) != {r["family_id"] for r in public}:
        raise ValueError("fit_roster_mismatch")
    features = [public_features(row) for row in public]
    means = [sum(x[j] for x in features) / len(features) for j in range(3)]
    scales = [
        max((sum((x[j] - means[j]) ** 2 for x in features) / len(features)) ** 0.5, 1e-12)
        for j in range(3)
    ]
    scaled = [[(x[j] - means[j]) / scales[j] for j in range(3)] for x in features]
    weights = [0.0, 0.0, 0.0]
    bias = 0.0
    for _ in range(200):
        gradients = [0.0, 0.0, 0.0]
        bias_gradient = 0.0
        for row, x in zip(public, scaled, strict=True):
            error = (
                _sigmoid(bias + sum(w * value for w, value in zip(weights, x, strict=True)))
                - labels[row["family_id"]]
            )
            bias_gradient += error
            for j in range(3):
                gradients[j] += error * x[j]
        bias -= 0.25 * bias_gradient / len(public)
        weights = [
            weights[j] - 0.25 * (gradients[j] / len(public) + weights[j] / len(public))
            for j in range(3)
        ]
    return dict(
        weights=weights,
        bias=bias,
        means=means,
        scales=scales,
        regularization=1.0,
        steps=200,
        learning_rate=0.25,
        feature_names=[
            "log1p_answer_utf8_bytes",
            "log1p_source_utf8_bytes",
            "answer_log_length_over_source_log_length",
        ],
    )


def predict(model: Mapping[str, Any], row: Mapping[str, Any], temperature: float = 1.0) -> float:
    """Return calibrated risk using only the registered public features."""
    x = public_features(row)
    score = model["bias"] + sum(
        model["weights"][j] * (x[j] - model["means"][j]) / model["scales"][j] for j in range(3)
    )
    return _sigmoid(score / temperature)


def choose_temperature(model: Json, public: Sequence[Json], labels: Mapping[str, int]) -> float:
    """Pick the first minimum-Brier grid point on tune labels only."""
    return min(
        TEMPERATURE_GRID,
        key=lambda t: (
            sum((predict(model, row, t) - labels[row["family_id"]]) ** 2 for row in public)
            / len(public)
        ),
    )


def fit_strata(public: Sequence[Json]) -> list[int]:
    """Freeze four answer-byte bands from fit-only empirical quartiles."""
    sizes = sorted(len(row["complete_response"].encode("utf-8")) for row in public)
    return [sizes[len(sizes) * k // 4 - 1] for k in (1, 2, 3)]


def stratum(answer_bytes: int, edges: Sequence[int]) -> int:
    """Assign an answer to its predeclared fit quartile band."""
    return sum(answer_bytes > edge for edge in edges)


def action(risk: float) -> str:
    """Apply the fixed 5p, 1-p, 0.25 expected-cost policy; ties escalate."""
    candidates = {"accept": 5 * risk, "reject": 1 - risk, "escalate": 0.25}
    return min(("escalate", "accept", "reject"), key=lambda name: candidates[name])


def realized_cost(chosen: str, label: int) -> float:
    """Score the chosen action against the intact human response label."""
    return {"accept": 5.0 * label, "reject": 1.0 - label, "escalate": 0.25}[chosen]


def paired_bootstrap(differences: Sequence[float], draws: int, seed: int) -> Json:
    """Resample independent family differences with a fixed local generator."""
    rng = random.Random(seed)
    means = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(draws)
    )
    return dict(
        draws=draws,
        seed=seed,
        independent_n=len(differences),
        mean=sum(differences) / len(differences),
        lower95=means[int(draws * 0.025)],
        upper95=means[int(draws * 0.975)],
    )


def _freeze(path: Path, value: Json) -> str:
    """Keep a completed checkpoint immutable across retries."""
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != value:
            raise ValueError(f"checkpoint_mismatch:{path}")
    else:
        atomic_json(path, value)
    return sha256_file(path)


def _metrics(rows: Sequence[Json], arm: str) -> Json:
    if not rows:
        return dict(count=0, prevalence=None, brier=None, cost=None)
    return dict(
        count=len(rows),
        prevalence=sum(r["label"] for r in rows) / len(rows),
        brier=sum(r["arms"][arm]["brier"] for r in rows) / len(rows),
        cost=sum(r["arms"][arm]["cost"] for r in rows) / len(rows),
    )


def run_baseline(custody: Json, output: Path, seed: int = 7848) -> Json:
    """Fit on 256, tune on 64, seal 64 predictions, then open evaluation labels."""
    output.mkdir(parents=True, exist_ok=True)
    public = custody["public"]
    paths = custody["evaluator_paths"]
    fit_labels = _labels(paths["fit"], public["fit"])
    model = fit_model(public["fit"], fit_labels)
    model["prevalence"] = sum(fit_labels.values()) / 256
    model["temperature_grid"] = list(TEMPERATURE_GRID)
    edges = fit_strata(public["fit"])
    tune_labels = _labels(paths["tune"], public["tune"])
    model["temperature"] = choose_temperature(model, public["tune"], tune_labels)
    config = dict(
        seed=seed,
        fit_sha256=sha256_file(paths["fit"]),
        tune_sha256=sha256_file(paths["tune"]),
        public_hashes=[
            s["sha256"] for s in custody["source_artifact_hashes"] if s.get("kind") == "public"
        ],
        steps=200,
        regularization=1.0,
        temperature_grid=list(TEMPERATURE_GRID),
        costs={"accept": "5p", "reject": "1-p", "escalate": 0.25},
    )
    config_hash = (
        "sha256:" + hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    )
    model_path = output / f"fit-{config_hash[7:]}.json"
    _freeze(model_path, model)
    predictions = {
        "config_sha256": config_hash,
        "family_order": [r["family_id"] for r in public["evaluation"]],
        "rows": [
            {
                "family_id": r["family_id"],
                "answer_bytes": len(r["complete_response"].encode("utf-8")),
                "source_bytes": len(r["complete_source"].encode("utf-8")),
                "stratum": stratum(len(r["complete_response"].encode("utf-8")), edges),
                "length_risk": predict(model, r, model["temperature"]),
                "prevalence_risk": model["prevalence"],
            }
            for r in public["evaluation"]
        ],
    }
    prediction_path = output / "evaluation_predictions.json"
    prediction_hash = _freeze(prediction_path, predictions)
    evaluation_labels = _labels(paths["evaluation"], public["evaluation"])
    rows = []
    for prediction in predictions["rows"]:
        family = prediction["family_id"]
        label = evaluation_labels[family]
        arms = {}
        for name, risk in (
            ("length", prediction["length_risk"]),
            ("prevalence", prediction["prevalence_risk"]),
        ):
            chosen = action(risk)
            arms[name] = dict(
                risk=risk,
                action=chosen,
                brier=(risk - label) ** 2,
                cost=realized_cost(chosen, label),
            )
        arms["always_escalate"] = dict(risk=None, action="escalate", brier=None, cost=0.25)
        rows.append(
            dict(
                family_id=family,
                arm="paired_intact_family",
                role="evaluation",
                seed=seed,
                status="completed",
                label=label,
                source_path=str(paths["evaluation"].parent / "evaluation_public.jsonl"),
                evaluator_path=str(paths["evaluation"]),
                prediction_path=str(prediction_path),
                answer_bytes=prediction["answer_bytes"],
                source_bytes=prediction["source_bytes"],
                stratum=prediction["stratum"],
                arms=arms,
            )
        )
    diffs = [r["arms"]["prevalence"]["cost"] - r["arms"]["length"]["cost"] for r in rows]
    brier_diffs = [r["arms"]["prevalence"]["brier"] - r["arms"]["length"]["brier"] for r in rows]
    strata = [
        dict(
            index=k,
            answer_bytes_lower_exclusive=edges[k - 1] if k else None,
            answer_bytes_upper_inclusive=edges[k] if k < 3 else None,
            **_metrics([r for r in rows if r["stratum"] == k], "length"),
        )
        for k in range(4)
    ]
    return dict(
        rows=rows,
        length_model=model,
        strata_definition=dict(source="fit256_only", feature="answer_utf8_bytes", edges=edges),
        strata_metrics=strata,
        baseline_rows={
            name: _metrics(rows, name)
            if name != "always_escalate"
            else dict(
                count=64, prevalence=sum(r["label"] for r in rows) / 64, brier=None, cost=0.25
            )
            for name in ("length", "prevalence", "always_escalate")
        },
        bootstrap={
            "cost_gain_over_prevalence": paired_bootstrap(diffs, 10000, seed),
            "brier_gain_over_prevalence": paired_bootstrap(brier_diffs, 10000, seed + 1),
            "draws": 10000,
            "independent_n": 64,
        },
        sample_size_budget=dict(
            intended=64,
            eligible=64,
            started=64,
            completed=64,
            censored=0,
            excluded=0,
            independent_n=64,
        ),
        model_checkpoint_path=str(model_path),
        model_checkpoint_sha256=sha256_file(model_path),
        prediction_path=str(prediction_path),
        prediction_seal_sha256=prediction_hash,
        config_sha256=config_hash,
    )


def cold_replay(result: Json, custody: Json, output: Path) -> Json:
    """Reopen sealed predictions and independently recalculate every scored row."""
    path = output / "evaluation_predictions.json"
    if not path.is_file() or sha256_file(path) != result["prediction_seal_sha256"]:
        return dict(passed=False, reason="prediction_seal_mismatch")
    predictions = json.loads(path.read_text(encoding="utf-8"))
    if predictions["config_sha256"] != result["config_sha256"]:
        return dict(passed=False, reason="config_mismatch")
    if predictions["family_order"] != [r["family_id"] for r in custody["public"]["evaluation"]]:
        return dict(passed=False, reason="family_order_mismatch")
    labels = _labels(custody["evaluator_paths"]["evaluation"], custody["public"]["evaluation"])
    if len(result["rows"]) != 64 or len(predictions["rows"]) != 64:
        return dict(passed=False, reason="row_count_mismatch")
    for prediction, row in zip(predictions["rows"], result["rows"], strict=True):
        family = prediction["family_id"]
        if row["family_id"] != family or row["label"] != labels[family]:
            return dict(passed=False, reason="label_join_mismatch")
        for arm in ("length", "prevalence"):
            risk = prediction[f"{arm}_risk"]
            expected = dict(
                risk=risk,
                action=action(risk),
                brier=(risk - labels[family]) ** 2,
                cost=realized_cost(action(risk), labels[family]),
            )
            if row["arms"][arm] != expected:
                return dict(passed=False, reason=f"{arm}_metric_mismatch")
        if row["arms"]["always_escalate"]["cost"] != 0.25:
            return dict(passed=False, reason="escalation_mismatch")
    return dict(passed=True, checked_families=64, prediction_sha256=sha256_file(path))


def energy_gate(root: Path) -> Json:
    """Reject current conductor diagnostics as a source of energy heads."""
    path = root / ENERGY
    if not path.is_file():
        return dict(
            qualified=False,
            failures=[failure("exp7840", path, "is_file", True, False)],
            permutation_rows=[],
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    checks = [
        ("status", "completed"),
        ("blocked_at_layer", None),
        ("flagged_adversarial", False),
        ("energy_fit_ready_score", 1),
    ]
    failures = [
        failure("exp7840", path, key, expected, value.get(key))
        for key, expected in checks
        if value.get(key) != expected
    ]
    if not failures and not value.get("constrained_set_heads"):
        failures.append(failure("exp7840", path, "constrained_set_heads", "nonempty", None))
    return dict(
        qualified=not failures,
        failures=failures,
        permutation_rows=[],
        source_path=str(path),
        source_sha256=sha256_file(path),
        observed_status=value.get("status"),
        observed_layer=value.get("blocked_at_layer"),
    )
