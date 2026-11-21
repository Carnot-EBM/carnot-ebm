"""REQ-REPORT-9107-V776: frozen judgments remain predictions about human labels.

Exact source bytes and complete continuation logits prevent an administrative
pass or a partial vocabulary score from becoming a scientific result.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.natural_label_eligibility_8975 import ref as ref
from carnot.reporting.natural_label_runner_8975 import read_ref
from carnot.reporting.spline_basis_methods_8986 import gate

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_9107_v776_qwen_judge", "exp9107-qwen-judge"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_qwen_judge_9107.py"
OWNED = [
    "python/carnot/reporting/qwen_judge_9107.py",
    "python/carnot/inference/qwen_judge_9107.py",
    "python/carnot/reporting/qwen_judge_runner_9107.py",
    CLI,
]
MODEL_SPECS = [dict(hf_id="unsloth/Qwen3.8-27B-GGUF", quantization="Q4_K_M")]
AUTHORITY = "experiment_9097_v776_human_authority.json"
RUNTIME = "experiment_9098_v776_embedding_runtime.json"
PINS = {
    AUTHORITY: "sha256:24c9e0983d9a4eb0aa5a488b5467386e57d6f13b79447f0f3a2f3d1c68e25532",
    RUNTIME: "sha256:3eb28ae1a30060138bc99a7b46f85f5ac5a94a840a2c51996ac1c4a0e3f92dff",
}
BUDGET = dict(cal_qids=50, test_qids=100, rows=450, bootstrap_draws=10000)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real boundaries so native work cannot silently exceed its budget."""
    print(f"[exp9107] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Fix decisions before outcomes so favorable cohorts cannot replace failures."""
    return dict(
        prompt="Question: {question}\nAnswer: {answer}\nDoes the answer contain any factually false specific claim? Reply Yes or No.",
        data_delimiters="JSON quoted strings; ensure_ascii=False",
        chat_template="cached GGUF embedded template; add_generation_prompt=True",
        text_tokens="embedded tokenizer, add_bos=False, special=False",
        prompt_tokens="embedded tokenizer, add_bos=False, special=True",
        right_truncation=dict(question=256, answer_text=768),
        likelihood="full raw vocabulary; reset state before each literal Yes and No; sum every continuation token",
        temperatures=[0.5, 1, 2, 4],
        selection="CAL logloss; ascending ties",
        accept_lte=0.1,
        reject_gte=0.9,
        bootstrap_draws=10000,
        bootstrap_seed=775092,
        random_seed=7769107,
        sample_size_budget=BUDGET,
        half_coverage="ascending p_error, stable original order; floor(N/2); rerank each draw",
        measurement_cap_s=3300,
        validation_reserve_s=1200,
        hard_cap_s=4800,
        generated_tokens=0,
        frozen_weights=True,
        exposure_scope="LPHB exposed development",
        comparison="qualified Exp9102 only; absence does not gate baseline",
    )


def authenticate(root: Path) -> Json:
    """Bind producer fields and consumed private files before any dependent model work."""
    checks: list[Json] = []
    references: list[Json] = []
    values: Json = {}
    for name, field in [(AUTHORITY, "source_ready_score"), (RUNTIME, "logits_ready_score")]:
        path = root / "results" / name
        actual = ref(path)
        references.append(actual)
        check = gate(
            path, "sha256", PINS[name], actual["sha256"] or "missing_file", name + "_sha256", name
        )
        checks.append(check)
        if check["passed"]:
            value = json.loads(path.read_bytes())
            observed = value.get(field, "missing_field")
            checks.append(gate(path, field, 1, observed, field, name))
            if observed == 1:
                values[name] = value
    rows, views = [], []
    if len(values) == 2:
        authority = values[AUTHORITY]
        reference = dict(path=str(root), sha256=None)
        try:
            for key in [
                "original_group_partition_reference",
                "static_protocol_reference",
                "evaluator_label_reference",
                "historical_producer_snapshot_reference",
            ]:
                reference = authority[key]
                read_path(reference)
                references.append(reference)
            for key in ["shard_a_roster_reference", "shard_b_roster_reference"]:
                reference = authority[key]
                shard = json.loads(read_path(reference).read_bytes())
                references.append(reference)
                feature = read_ref(shard["feature_reference"])
                references.append(shard["feature_reference"])
                if len(shard["rows"]) != len(feature["views"]):
                    raise ValueError("feature_alignment")
                for row, view in zip(shard["rows"], feature["views"], strict=True):
                    if canonical_hash(view) != row["feature_input_sha256"]:
                        raise ValueError("feature_input_drift")
                    if row["role"] in {"CAL", "TEST"}:
                        rows.append(row)
                        views.append({k: view[k] for k in ["question", "answer_text"]})
            counts = Counter(r["role"] for r in rows)
            groups = Counter(r["group_id"] for r in rows)
            if (
                counts != dict(CAL=150, TEST=300)
                or len(groups) != 150
                or set(groups.values()) != {3}
            ):
                raise ValueError("original_panel_counts")
        except (OSError, KeyError, ValueError) as error:
            checks.append(
                gate(
                    Path(reference["path"]),
                    "sha256" if "reference_drift" in str(error) else "immutable_inputs",
                    reference.get("sha256")
                    if "reference_drift" in str(error)
                    else "authenticated complete original panel",
                    "missing_field"
                    if isinstance(error, KeyError)
                    else ref(Path(reference["path"]))["sha256"]
                    if "reference_drift" in str(error)
                    else str(error),
                    "immutable_inputs",
                    "exp9097-human-authority",
                )
            )
    failed = [c for c in checks if not c["passed"]]
    return dict(
        checks=checks,
        references=references,
        rows=rows,
        views=views,
        blocker=failed[0]["check"] if failed else None,
        label_reference=values.get(AUTHORITY, {}).get("evaluator_label_reference"),
    )


def prompt(model: Any, view: Json) -> tuple[str, Json]:
    """Quote only question and answer after the same independent embedding text caps."""
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    texts, counts = {}, {}
    for key, short, cap in [("question", "question", 256), ("answer_text", "answer", 768)]:
        tokens = model.tokenize(view[key].encode(), add_bos=False, special=False)
        retained = tokens[:cap]
        texts[key] = model.detokenize(retained).decode("utf-8", errors="replace")
        counts["full_" + short + "_tokens"] = len(tokens)
        counts["retained_" + short + "_tokens"] = len(retained)
    content = protocol()["prompt"].format(
        question=json.dumps(texts["question"], ensure_ascii=False),
        answer=json.dumps(texts["answer_text"], ensure_ascii=False),
    )
    formatter = Jinja2ChatFormatter(
        template=model.metadata["tokenizer.chat_template"],
        eos_token=model.detokenize([model.token_eos()], special=True).decode(),
        bos_token=model.detokenize([model.token_bos()], special=True).decode(),
    )
    text = formatter(messages=[dict(role="user", content=content)]).prompt
    counts["prompt_tokens"] = len(model.tokenize(text.encode(), add_bos=False, special=True))
    counts["untemplated_embedding_tokens"] = sum(
        counts["retained_" + k + "_tokens"] for k in ["question", "answer"]
    )
    counts["template_and_quoting_token_difference"] = (
        counts["prompt_tokens"] - counts["untemplated_embedding_tokens"]
    )
    return text, counts


def verify_likelihood(value: Json) -> None:
    """Recompute each token score from retained full logits, never a top-k summary."""
    logits = np.load(read_path(value["raw_logits_reference"]), allow_pickle=False)
    tokens, lp = value["tokens"], value["token_logprobs"]
    if (
        logits.shape != (len(tokens), value["raw_logits_width"])
        or not tokens
        or len(lp) != len(tokens)
        or not np.isfinite(logits).all()
        or not value["raw_logits_unmasked"]
        or not value["raw_logits_finite"]
    ):
        raise ValueError("invalid_likelihood")
    observed = [
        float(row[t] - np.logaddexp.reduce(row.astype(np.float64)))
        for row, t in zip(logits, tokens, strict=True)
    ]
    if not np.allclose(observed, lp, atol=1e-9, rtol=0) or value["log_probability"] != sum(lp):
        raise ValueError("likelihood_reduction")


def read_path(reference: Json) -> Path:
    """A rehashed aggregate cannot authorize changed primitive bytes."""
    path = Path(reference["path"])
    if ref(path)["sha256"] != reference["sha256"]:
        raise ValueError("reference_drift:" + str(path))
    return path


def score(runtime: Any, identity: Json, view: Json, raw: Path) -> Json:
    """Time tokenization, both fresh continuation passes and durable serialization."""
    began = time.monotonic_ns()
    text, counts = prompt(runtime.model, view)
    checks = {word: runtime.likelihood(text, word) for word in ["Yes", "No"]}
    for value in checks.values():
        verify_likelihood(value)
    yes, no = (checks[w]["log_probability"] for w in ["Yes", "No"])
    row = dict(
        identity,
        **counts,
        likelihoods=checks,
        yes_logprob=yes,
        no_logprob=no,
        p_error=float(np.exp(yes - np.logaddexp(yes, no))),
        prompt_sha256=canonical_hash(text),
        feature_input_sha256=canonical_hash(view),
        continuation_token_evaluations=sum(len(v["tokens"]) for v in checks.values()),
        prompt_token_evaluations=2 * counts["prompt_tokens"],
        status="completed",
        arm="frozen_qwen_judge",
        condition="exposed_development",
        seed=7769107,
        absolute_metric=1,
        numerator=1,
        denominator=1,
        original_qid=identity["qid"],
        exclusion_reason=None,
        censor_reason=None,
    )
    atomic_json(raw / (identity["unit_id"] + ".json"), row)
    row["outer_duration_s"] = (time.monotonic_ns() - began) / 1e9
    row["started_monotonic_ns"] = began
    row["ended_monotonic_ns"] = time.monotonic_ns()
    return row


def metrics(p: Any, y: Any) -> Json:
    """Keep undefined selective errors null instead of counting them as correct."""
    accept = p <= 0.1
    half = np.argsort(p, kind="stable")[: len(p) // 2]
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return dict(
        Brier=float(np.mean((p - y) ** 2)),
        logloss=float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        coverage=float(np.mean(accept)),
        accepted_error=float(np.mean(y[accept])) if accept.any() else None,
        half_coverage_error=float(np.mean(y[half])) if len(half) else None,
        accepted_numerator=int(y[accept].sum()),
        accepted_denominator=int(accept.sum()),
        half_numerator=int(y[half].sum()),
        half_denominator=len(half),
        denominator=len(p),
    )


def reduce_scores(rows: list[Json], labels: Json, *, draws: int = 10000) -> Json:
    """Choose temperature only on CAL, then resample original TEST questions with siblings."""
    if not rows or any(r["status"] != "completed" for r in rows):
        raise ValueError("incomplete_scores")
    logits = np.array([r["yes_logprob"] - r["no_logprob"] for r in rows])
    y = np.array([labels[r["unit_id"]] for r in rows])
    cal = np.array([r["role"] == "CAL" for r in rows])
    if not cal.any() or cal.all() or not np.isfinite(logits).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("incomplete_labels_or_roles")
    candidates = []
    for temperature in protocol()["temperatures"]:
        z = logits[cal] / temperature
        candidates.append(float(np.mean(np.logaddexp(0, z) - y[cal] * z)))
    temperature = protocol()["temperatures"][int(np.argmin(candidates))]
    p = np.exp(-np.logaddexp(0, -logits[~cal] / temperature))
    target = y[~cal]
    test = [r for r in rows if r["role"] == "TEST"]
    groups = sorted({r["group_id"] for r in test})
    indices = [np.array([i for i, r in enumerate(test) if r["group_id"] == g]) for g in groups]
    qids = [
        dict(qid=test[ix[0]]["qid"], group_id=g, **metrics(p[ix], target[ix]))
        for g, ix in zip(groups, indices, strict=True)
    ]
    rng, samples = np.random.default_rng(775092), []
    for draw in range(draws):
        if draw % 1000 == 0:
            progress("bootstrap", draw, draws - draw)
        ix = np.concatenate([indices[i] for i in rng.integers(0, len(groups), len(groups))])
        samples.append(metrics(p[ix], target[ix]))
    intervals = {}
    for key in ["Brier", "logloss", "coverage", "accepted_error", "half_coverage_error"]:
        finite = [r[key] for r in samples if r[key] is not None]
        intervals[key] = dict(
            ci95=np.quantile(finite, [0.025, 0.975]).tolist() if finite else None,
            undefined_draws=draws - len(finite),
        )
    return dict(
        temperature=temperature,
        cal_logloss_by_temperature=candidates,
        metrics=metrics(p, target),
        qid_rows=qids,
        test_predictions=[
            dict(
                unit_id=r["unit_id"],
                qid=r["qid"],
                group_id=r["group_id"],
                p_error=float(prob),
                target=int(t),
            )
            for r, prob, t in zip(test, p, target, strict=True)
        ],
        uncertainty=dict(
            draws=draws,
            seed=775092,
            unit="original TEST qid with all siblings",
            intervals=intervals,
            scope="conditional descriptive exposed development",
        ),
    )
