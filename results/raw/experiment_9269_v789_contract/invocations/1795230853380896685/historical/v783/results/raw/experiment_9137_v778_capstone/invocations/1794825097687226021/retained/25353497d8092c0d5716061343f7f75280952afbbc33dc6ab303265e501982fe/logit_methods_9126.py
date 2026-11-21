"""REQ-REPORT-9126-V778: freeze whole-answer methods without borrowing model evidence.

Full-vocabulary reductions keep the probability controls comparable to raw energy.
Fixtures qualify formulas only; the exposed human panel cannot prove generalization.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_9126_v778_logit_methods", "exp9126-logit-methods"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_logit_methods_9126.py"
OWNED = [
    "python/carnot/reporting/logit_methods_9126.py",
    "python/carnot/reporting/logit_methods_runner_9126.py",
    CLI,
]
UPSTREAM = "results/experiment_9124_v778_contract.json"
PIN = "sha256:a93fa7d34b6b6f151bcf5a7bc8c3fe9fcc1813395d68f9c825fba6b40ed0a4e2"
MODEL_SPECS: list[Json] = []


def progress(phase: str) -> None:
    """Flush phase boundaries so a refused input cannot look like a model stall."""
    print(f"[exp9126] phase={phase}", flush=True)


def ref(path: Path) -> Json:
    """Bind exact bytes and keep file absence distinct from a measured zero."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path) if path.is_file() else None)


def protocol() -> Json:
    """Seal the inherited choices before token statistics or evaluator labels are read."""
    return dict(
        version="v778-whole-answer-v1",
        target="original A1:1=error,0=correct,-1=missing; A2 observed-only sensitivity",
        roles=dict(TRAIN=150, CAL=50, TEST=100, siblings=3),
        features=[
            "log1p_answer_length",
            "mean_nll",
            "mean_entropy",
            "max_entropy",
            "mean_marginal_energy",
            "mean_absolute_spill",
            "no_adjacent_pair",
        ],
        arms={
            "prevalence": [],
            "probability_length": [0, 1, 2, 3],
            "raw_energy": [0, 4, 5, 6],
            "combined": list(range(7)),
            "combined_donor": list(range(7)),
        },
        tokenizer="GGUF embedded; question/answer separately; add_bos=False; no special tokens/chat template",
        caps=dict(question=256, answer=768),
        truncation="right",
        prefix_bytes_hex=[b"Question:\n".hex(), b"\nAnswer:\n".hex()],
        detokenization="embedded tokenizer detokenize retained question IDs to exact UTF-8 bytes; concatenate literal prefixes; tokenize prefix without BOS; append retained answer IDs, never retokenize answer",
        indexing="zero-based answer i scored by prefix_length+i-1; no terminal next-step logit; D only for i<length-1",
        chunk_size=128,
        state="reset native state and reducer before every answer; retain previous scalar across chunks",
        formulas=dict(
            L="logsumexp(z_i)",
            NLL="L_i-z_i[a_i]",
            H="-sum(p_i*log(p_i))",
            M="-L_i",
            D="-z_i[a_i]+L_(i+1)",
        ),
        schema=[
            "answer_position",
            "prediction_position",
            "token_id",
            "vocabulary_size",
            "selected_logit",
            "logsumexp",
            "nll",
            "entropy",
            "max_probability",
            "marginal_energy",
            "full_vector_sha256",
        ],
        vector_hash="SHA256 contiguous little-endian float64 full vector; raw sample stores exact reducer inputs",
        raw_sample=dict(
            qids=20,
            siblings="all3",
            positions=list(range(8)),
            order="lexical original qid",
            disk_budget_bytes=2 * 1024**3,
        ),
        missing="TRAIN-only median per feature; wholly missing TRAIN feature blocks; one-token spill missing; empty escalates",
        support="predeclared nonempty answers for shared losses/risks; coverage all300 TEST responses; no label filtering",
        fit=dict(
            ridge=0.01,
            objective="mean binary logloss+0.5*0.01*sum(nonintercept theta^2)",
            scaling="TRAIN population mean/std ddof0; constant scale1",
            optimizer="L-BFGS-B from zeros",
            maxiter=2000,
            gtol=1e-7,
            ftol=1e-12,
            gradient_infinity_norm_max=1e-6,
            finite_required=True,
        ),
        binary_energy="E(0)=0; E(1)=-theta.phi; p(error)=sigmoid(theta.phi)",
        donor="sorted whole TRAIN qids cyclic successor (nonidentity), sibling positions fixed; report actual changed labels",
        calibration=dict(
            temperatures=[0.5, 1, 2, 4],
            selection="CAL A1 logloss; ascending ties",
            accept_lte=0.1,
            reject_gte=0.9,
            other="escalate",
            seal="TEST predictions before evaluator access",
        ),
        metrics=["Brier", "logloss", "coverage", "accepted_error", "exact_half_coverage_error"],
        bootstrap=dict(
            draws=10000,
            seed=777121,
            unit="paired TEST qid with all siblings",
            tail=0.05 / 2,
            half_coverage="rerank each draw; floor(N/2); original-order ties",
            undefined="retain null risk and support; cannot pass",
        ),
        bounds=[
            "combined Brier improvement lower>0 vs probability_length",
            "combined Brier improvement lower>0 vs combined_donor",
        ],
        accepted_risk="combined empirical accepted error <= probability_length; undefined cannot pass",
        gauges=dict(
            constant="c_i=3",
            alternating="c_i=3*(-1)^i; zero-based i",
            probability="invariant",
            nll_entropy="invariant",
            M_shift="-c_i",
            D_shift="c_(i+1)-c_i",
            interpretation="representation sensitivity, never probability-chain violation; no favorable gauge selection; later freeze heads for susceptibility",
        ),
        scope="whole-answer cross-model archived-answer adaptation; not paper localization or CES; no new Qwen answers",
        retired_null="Exp6980 complete_null_spilled_energy_requalification_retired remains unchanged; exact-oracle scope",
        exclusions=[],
        measurement_cap_s=3300,
        validation_reserve_s=1200,
        hard_cap_s=4800,
    )


class Stream:
    """Keep one answer's scalars and a bounded raw sample, never its full logit matrix."""

    def __init__(
        self, vocabulary: int, prefix_length: int, *, raw_positions: Iterable[int] = ()
    ) -> None:
        self.raw_positions = set(raw_positions)
        if vocabulary < 1 or prefix_length < 1 or not self.raw_positions <= set(range(8)):
            raise ValueError("vocabulary_prefix_or_sample")
        self.vocabulary, self.prefix_length = vocabulary, prefix_length
        self.reset()

    def reset(self) -> None:
        """Discard the previous answer so its last token cannot create a false pair."""
        self.rows: list[Json] = []
        self.raw: list[Json] = []
        self.spill: list[float] = []

    def add(self, logits: Any, token_ids: list[int], prediction_positions: list[int]) -> None:
        """Validate the complete vocabulary and preceding-prefix alignment before reduction."""
        z = np.asarray(logits, dtype=np.float64)
        if (
            z.ndim != 2
            or z.shape[0] != len(token_ids)
            or len(token_ids) != len(prediction_positions)
        ):
            raise ValueError("shape")
        if z.shape[1] != self.vocabulary:
            raise ValueError("vocabulary")
        if len(z) > 128:
            raise ValueError("chunk")
        if len(self.rows) + len(z) > 768:
            raise ValueError("cap")
        if not np.isfinite(z).all():
            raise ValueError("nonfinite")
        for vector, token, position in zip(z, token_ids, prediction_positions, strict=True):
            i = len(self.rows)
            if type(token) is not int or not 0 <= token < self.vocabulary:
                raise ValueError("token")
            if type(position) is not int or position != self.prefix_length + i - 1:
                raise ValueError("index")
            centered = vector - np.max(vector)
            log_norm = math.log(float(np.exp(centered).sum()))
            logp = centered - log_norm
            p = np.exp(logp)
            entropy = -float(np.sum(p * logp))
            lse = float(np.max(vector)) + log_norm
            selected = float(vector[token])
            row = dict(
                answer_position=i,
                prediction_position=position,
                token_id=token,
                vocabulary_size=self.vocabulary,
                selected_logit=selected,
                logsumexp=lse,
                nll=float(-logp[token]),
                entropy=entropy,
                max_probability=float(p.max()),
                marginal_energy=-lse,
                full_vector_sha256="sha256:"
                + hashlib.sha256(vector.astype("<f8").tobytes()).hexdigest(),
            )
            if not all(math.isfinite(row[k]) for k in ["nll", "entropy", "logsumexp"]):
                raise ValueError("nonfinite_reduction")
            if self.rows:
                self.spill.append(-self.rows[-1]["selected_logit"] + lse)
            self.rows.append(row)
            if i in self.raw_positions:
                self.raw.append(
                    dict(
                        answer_position=i, logits=vector.tolist(), sha256=row["full_vector_sha256"]
                    )
                )

    def finish(self) -> Json:
        """Leave unsupported quantities missing so later fitting cannot invent support."""
        n = len(self.rows)
        features = (
            None
            if not n
            else [
                math.log1p(n),
                float(np.mean([r["nll"] for r in self.rows])),
                float(np.mean([r["entropy"] for r in self.rows])),
                max(r["entropy"] for r in self.rows),
                float(np.mean([r["marginal_energy"] for r in self.rows])),
                float(np.mean(np.abs(self.spill))) if self.spill else None,
                int(not self.spill),
            ]
        )
        return dict(
            status="completed" if n else "escalate_empty",
            token_statistics=self.rows.copy(),
            spill=self.spill.copy(),
            features=features,
            raw_samples=self.raw.copy(),
            token_count=n,
        )


def reduce_answer(logits: Any, token_ids: list[int], *, prefix_length: int = 1) -> Json:
    """Convenience for bounded references; production callers feed Stream chunk by chunk."""
    if len(logits) != len(token_ids):
        raise ValueError("shape")
    stream = Stream(len(logits[0]) if len(logits) else 1, prefix_length, raw_positions=range(8))
    for start in range(0, len(logits), 128):
        end = min(start + 128, len(logits))
        stream.add(
            logits[start:end],
            token_ids[start:end],
            list(range(prefix_length + start - 1, prefix_length + end - 1)),
        )
    return stream.finish()


def check(path: Path, field: str, expected: Any, observed: Any, name: str) -> Json:
    """Expose gate operands without conflating missing evidence and an observed zero."""
    return dict(
        check=name,
        upstream="exp9124-contract",
        **ref(path),
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=type(expected) is type(observed) and expected == observed,
    )


def byte_check(path: Path, pin: str) -> Json:
    """Authenticate producer bytes before trusting fields or transitive references."""
    return check(path, "sha256", pin, ref(path)["sha256"] or "missing_file", "immutable_bytes")


def field_check(path: Path, field: str, expected: Any) -> Json:
    """Read a declared field only after the caller has authenticated its producer."""
    value = json.loads(path.read_bytes()) if path.is_file() else {}
    observed = value.get(field, "missing_field") if path.is_file() else "missing_file"
    return check(path, field, expected, observed, "upstream_gate")


def inspect(root: Path) -> Json:
    """Use authenticated methods and label-free identities, independent of model readiness."""
    path = root / UPSTREAM
    checks, references, roster = [byte_check(path, PIN)], [ref(path)], []
    if checks[0]["passed"]:
        checks.extend(
            field_check(path, k, 1) for k in ["contract_ready_score", "source_ready_score"]
        )
        source = json.loads(path.read_bytes())
        for key in [
            "energy_protocol_reference",
            "static_protocol_reference",
            "inference_view_reference",
        ]:
            reference = source.get(key)
            checks.append(
                check(
                    path,
                    key,
                    "present",
                    "present" if isinstance(reference, dict) else "missing_field",
                    "reference_field",
                )
            )
            if isinstance(reference, dict):
                target = Path(reference["path"])
                references.append(ref(target))
                checks.append(byte_check(target, reference["sha256"]))
        if all(c["passed"] for c in checks):
            rows = json.loads(Path(source["inference_view_reference"]["path"]).read_bytes())["rows"]
            qids = sorted({row["qid"] for row in rows})[:20]
            roster = [
                dict(
                    unit_id=row["unit_id"],
                    qid=row["qid"],
                    sibling=row["sibling"],
                    answer_positions=list(range(8)),
                )
                for row in rows
                if row["qid"] in qids
            ]
            checks.append(check(path, "raw_sample_units", 60, len(roster), "sample_roster"))
    return dict(checks=checks, source_references=references, raw_roster=roster)


def controls() -> list[Json]:
    """Retain independent formula references and both offset gauges without human labels."""
    z = np.array([[0.0, math.log(2), math.log(3)], [2.0, -1.0, 0.0], [1.0, 3.0, -2.0]])
    tokens = [2, 0, 1]
    base = reduce_answer(z, tokens, prefix_length=7)
    rows: list[Json] = []
    for i, vector in enumerate(z):
        total = math.fsum(math.exp(float(v)) for v in vector)
        p = [math.exp(float(v)) / total for v in vector]
        expected = dict(
            logsumexp=math.log(total),
            nll=math.log(total) - float(vector[tokens[i]]),
            entropy=-math.fsum(v * math.log(v) for v in p),
        )
        observed = {k: base["token_statistics"][i][k] for k in expected}
        passed = all(
            math.isclose(expected[k], observed[k], abs_tol=1e-12, rel_tol=1e-12) for k in expected
        )
        rows.append(
            dict(
                unit_id=f"formula-{i}",
                condition="independent_formula",
                passed=passed,
                expected=expected,
                observed=observed,
                raw=base["raw_samples"][i],
                token_id=tokens[i],
                prediction_position=6 + i,
            )
        )
    for name, offsets in [
        ("constant", np.full(3, 3.0)),
        ("alternating", 3.0 * (-1.0) ** np.arange(3)),
    ]:
        shifted = reduce_answer(z + offsets[:, None], tokens, prefix_length=7)
        raw_p = np.exp(z - np.logaddexp.reduce(z, axis=1)[:, None])
        shifted_z = z + offsets[:, None]
        shifted_p = np.exp(shifted_z - np.logaddexp.reduce(shifted_z, axis=1)[:, None])
        errors = [float(np.max(np.abs(raw_p - shifted_p)))]
        for i, (a, b) in enumerate(
            zip(base["token_statistics"], shifted["token_statistics"], strict=True)
        ):
            errors.extend(abs(b[k] - a[k]) for k in ["nll", "entropy", "max_probability"])
            errors.append(abs(b["marginal_energy"] - a["marginal_energy"] + float(offsets[i])))
        errors.extend(
            abs(b - a - float(offsets[i + 1] - offsets[i]))
            for i, (a, b) in enumerate(zip(base["spill"], shifted["spill"], strict=True))
        )
        rows.append(
            dict(
                unit_id="gauge-" + name,
                condition=name,
                passed=max(errors) < 1e-12,
                expected_max_error=1e-12,
                observed_max_error=max(errors),
                offsets=offsets.tolist(),
                base=base,
                shifted=shifted,
                probabilities=raw_p.tolist(),
                shifted_probabilities=shifted_p.tolist(),
                probability_chain_violation=False,
            )
        )
    for name, vector, token, position in [
        ("shifted_index", [[0.0, 1.0, 2.0]], 0, 7),
        ("truncated_vocabulary", [[0.0, 1.0]], 0, 6),
        ("nonfinite", [[0.0, float("inf"), 2.0]], 0, 6),
    ]:
        rejected = False
        try:
            Stream(3, 7).add(vector, [token], [position])
        except ValueError:
            rejected = True
        rows.append(
            dict(
                unit_id=name,
                condition="indexing_rejection",
                passed=rejected,
                expected="rejected",
                observed="rejected" if rejected else "accepted",
            )
        )
    zlong = np.arange(130 * 3, dtype=float).reshape(130, 3) / 100
    edge = reduce_answer(zlong, [1] * 130, prefix_length=5)
    expected_edge = -float(zlong[127, 1]) + math.log(
        math.fsum(math.exp(float(x)) for x in zlong[128])
    )
    rows.append(
        dict(
            unit_id="chunk-edge",
            condition="chunk128",
            passed=math.isclose(edge["spill"][127], expected_edge, abs_tol=1e-12),
            expected=expected_edge,
            observed=edge["spill"][127],
            pair=[127, 128],
        )
    )
    for row in rows:
        row.update(
            qid=None,
            group_id=row["unit_id"],
            sibling=None,
            arm="method_control",
            seed=778126,
            absolute_metric=int(row["passed"]),
            numerator=int(row["passed"]),
            denominator=1,
            status="completed" if row["passed"] else "failed",
            exclusion_reason=None,
            censor_reason=None,
        )
    return rows
