"""REQ-REPORT-9084-V775: freeze exposed methods without running an embedding model.

Original author blobs establish label custody. They cannot repair historical
producer drift or turn an exposed question partition into independent evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path
from typing import Any

from carnot.reporting import human_label_custody_8389 as old
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.natural_label_eligibility_8975 import ref as ref
from carnot.reporting.natural_label_runner_8975 import read_ref
from carnot.reporting.spline_basis_methods_8986 import gate as gate

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_9084_v775_human_methods", "exp9084-human-methods"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_human_methods_9084.py"
OWNED = [
    "python/carnot/reporting/human_methods_9084.py",
    "python/carnot/reporting/human_methods_runner_9084.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
DEFAULT_INPUT = Path("/tmp/carnot-9084-private")
PINS = {
    "experiment_8389_v723_human_label_custody": "3cc8b7659d145f5ace20e850b4dd0abda2074b8e9c88d61091431722ec618e0e",
    "experiment_8403_v724_released_question_identity": "13afe888c9f7f516d63adff8eab9eea25e011745db6530d41cd836925661235b",
    "experiment_8409_v724_retrospective_label_criterion": "0929b927f547d028f05ad466aba9575a797d25a146a671b453db9de5a6eef2e8",
    "experiment_8429_v726_contract_methods": "dfad99775207ce8073b35c7d5efbc0cbefa70ad1ddadadafa584157add3a3095",
    "experiment_8430_v726_criterion_feature_panel": "c5783590d946a9500c335a7d39e0fb154dd5943c0efc2c1355dde885c3712989",
    "experiment_8431_v726_criterion_energy_fit": "38db38433d5e2803a7d4b008c88fba5a1a91138abceb42e52f3dadf5b2f10b7f",
    "experiment_8432_v726_criterion_decision_audit": "100efbb17df04e60271df4f8c7e574e1ad03948d1b0d4888625fd6b35993e4b2",
    "experiment_8433_v726_block_checkpoint_qualification": "5cbdfe3e9b9498c1430fa125e53b635c72b0979bed8522d4fa3183b7991a704c",
    "experiment_8434_v726_capacity_online_learning": "e3bced9f078ce48912394c9047c2d94d44c7cefdc5df1519530b25924caba51c",
    "experiment_8435_v726_capacity_benefit_audit": "e2e58cbf680becad2f1572629337a58d747911492c24fa43697a8d867f4b0a51",
    "experiment_9070_v774_adjudication_methods": "aee3a477ea80a3ec5b0d7187b1f3f0b92001c34e1fb5a732a470bdfd2ffe8e73",
}
PROHIBITED = [
    "ref_str",
    "label_*",
    "annotator1_label",
    "annotator2_label",
    "annotator_tags",
    "item_id",
    "qid",
    "dataset",
    "model",
    "source_id",
    "oracle_derived_signals",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so a blocked input cannot resemble a running model."""
    print(f"[exp9084] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Seal later extraction and scoring rules before any learned representation exists."""
    return dict(
        version="v775-human-embedding-v1",
        scope="exposed_development_only",
        extraction_performed=False,
        head_training_performed=False,
        target="original annotator1 factuality: 1=error, 0=correct, -1=missing",
        annotator2="descriptive sensitivity on actual observed subset; no forced consensus",
        references="author ref_str and annotations are evaluator/context records only",
        partition="original V726 fit=TRAIN, tune=CAL, test=TEST; no repartition or exclusions for exposure",
        embedding=dict(
            model="unsloth/Qwen3.8-27B-GGUF Q4_K_M; actual frozen llama.cpp runtime",
            runtime_binding="before extraction seal GGUF SHA256, embedded tokenizer metadata and runtime binary/build SHA256; missing or unsupported hidden states block",
            input_columns=["question", "answer_text"],
            tokenizer="GGUF embedded tokenizer only",
            special_tokens="add_bos=False; no chat template, EOS, separator or dataset tags",
            token_limits=dict(question=256, answer_text=768),
            truncation="right",
            retain=["full_token_count", "retained_token_count", "truncated", "padding_mask"],
            pooling="mean_last_layer_nonpadding",
            normalization="independent question and answer L2, then concatenate in that order",
            empty="zero vector of runtime hidden width; no synthetic token; flag empty",
            padding="exclude padding from mean; all-padding becomes flagged zero vector",
            zero_norm="leave zero vector unchanged",
            frozen=True,
        ),
        projection=dict(
            center_scale="TRAIN-only population mean/std (ddof=0); constants use scale1",
            distribution="numpy default_rng(seed).normal(0, 1/sqrt(27), (concat_width,27))",
            dimensions=27,
            seed=775084,
            label_dependent=False,
            scaling="TRAIN-only projected coordinate min/max to[-1,1]; constants0; future clip[-1,1]",
            freeze=["TRAIN means/std", "Gaussian matrix", "projected TRAIN min/max"],
            shared_by=["projected27_linear", "projected27_spline", "donor_projected27_spline"],
        ),
        arms=[
            "prevalence",
            "text_length_logistic",
            "projected27_linear",
            "full_embedding_regularized_linear",
            "projected27_spline",
            "donor_projected27_spline",
        ],
        length_features="log1p(full embedded-tokenizer question count), log1p(full answer count); TRAIN-only center/std",
        spline=dict(
            implementation="carnot.autoresearch.spline_basis_8986.design",
            degree=3,
            knots=[-1, -1, -1, -1, -0.5, 0, 0.5, 1, 1, 1, 1],
            columns="intercept +27 linear +189 fixed basis; identical donor architecture",
        ),
        fit=dict(
            loss="mean proper binary logloss +0.5*0.01*sum(nonintercept coefficient squares)",
            l2=0.01,
            penalize_intercept=False,
            nce=False,
            architecture_search=False,
            optimizer="L-BFGS-B from zeros; maxiter2000, gtol1e-7, ftol1e-12",
            gradient_infinity_norm_max=1e-6,
            convergence="all six arms require verified finite objective and gradient bound; otherwise disqualify",
            training="TRAIN A1 only; no CAL/TEST targets or A2 used in fitting",
            coefficient_hashes="hash actual before/after trained coefficients; require real changes; no metadata-only changes",
        ),
        donor=dict(
            seed=775084,
            unit="whole TRAIN qid",
            assignment="sort original TRAIN group hashes, seeded permutation then one cyclic successor; at least2 groups",
            positions="keep generator positions in original release order; donor labels from matching generator",
            fixed_points=0,
            report="donor map and actual changed-label numerator/denominator/fraction; zero change is retained as degenerate",
            forbidden="shuffling only the three siblings within one question",
        ),
        calibration=dict(
            temperatures=[0.5, 1, 2, 4],
            selection="CAL mean A1 binary logloss",
            ties="ascending",
            probability="sigmoid(logit/temperature)",
            accept_lte=0.1,
            reject_gte=0.9,
            otherwise="escalate",
            empty_cal="block; never tune on TEST",
        ),
        metrics=[
            "Brier",
            "binary_logloss",
            "coverage",
            "accepted_error",
            "exact_half_coverage_error",
        ],
        strata="question-level A2 unobserved, observed with any disagreement, observed with no disagreement; retain actual A2 support and missingness",
        uncertainty=dict(
            draws=10000,
            seed=775092,
            paired=True,
            unit="original TEST qid with all siblings",
            ci95=[0.025, 0.975],
            rerank_half_coverage_each_draw=True,
            ranking="ascending p(error), tie original item order, accept floor(N/2) occurrences per draw",
            zero_denominator="null metric; report undefined draws and actual support",
            scope="conditional descriptive exposed-development intervals; no confirmatory significance, independent generalization or training-variability coverage",
            retain="all arm predictions, numerators, denominators, draw counts and paired contrasts against every control",
        ),
    )


def feature_view(value: Json) -> dict[str, str]:
    """Project an allowlist before a tokenizer or learner can encounter labels or metadata."""
    if not all(isinstance(value.get(k), str) for k in ("question", "answer_text")):
        raise ValueError("text_fields")
    return {k: value[k] for k in ("question", "answer_text")}


def label(value: str) -> int | None:
    """Missing annotations have no binary target and cannot become consensus votes."""
    if value not in {"-1", "0", "1"}:
        raise ValueError("label_encoding")
    return None if value == "-1" else int(value)


def inventory(master: list[Json], roles: Json) -> Json:
    """Count every original sibling before exclusions and reject incompatible group identities."""
    role_index = {
        q: {"fit": "TRAIN", "tune": "CAL", "test": "TEST"}[role]
        for role, groups in roles.items()
        for q in groups
    }
    if len(role_index) != sum(len(groups) for groups in roles.values()):
        raise ValueError("partition_overlap")
    rows: list[Json] = []
    groups: dict[str, list[Json]] = defaultdict(list)
    seen: set[str] = set()
    for index, rec in enumerate(master):
        if index % 100 == 0:
            progress("inventory", index, len(master) - index)
        view = feature_view(rec)
        group = old.text_hash(old.normalized(view["question"]))
        if rec["item_id"] in seen or not view["question"].strip() or group not in role_index:
            raise ValueError("identity_or_partition")
        seen.add(rec["item_id"])
        a1, a2 = label(rec["annotator1_label"]), label(rec["annotator2_label"])
        row = dict(
            unit_id=rec["item_id"],
            group_id=group,
            family=group,
            qid=rec["qid"],
            dataset=rec["dataset"],
            generator=rec["model"],
            original_row=index + 2,
            arm="source_inventory",
            condition="exposed_development",
            seed=None,
            role=role_index[group],
            question_sha256=old.text_hash(view["question"]),
            answer_sha256=old.text_hash(view["answer_text"]),
            feature_input_sha256=canonical_hash(view),
            annotator1=a1,
            annotator2=a2,
            disagreement=a1 is not None and a2 is not None and a1 != a2,
            absolute_metric=1,
            numerator=1,
            denominator=1,
            raw_numerator=1,
            raw_denominator=1,
            status="completed",
            censored=False,
            excluded=False,
            exclusion_reason=None,
        )
        rows.append(row)
        groups[group].append(row)
    if set(groups) != set(role_index) or any(
        len(g) != 3
        or {r["generator"] for r in g} != set(old.GENERATORS)
        or len({(r["dataset"], r["qid"]) for r in g}) != 1
        for g in groups.values()
    ):
        raise ValueError("whole_qid_partition")
    return dict(
        rows=rows,
        question_count=len(groups),
        annotator1_observed_count=sum(r["annotator1"] is not None for r in rows),
        annotator2_observed_count=sum(r["annotator2"] is not None for r in rows),
        annotator2_missing_count=sum(r["annotator2"] is None for r in rows),
        annotator2_question_count=sum(
            any(r["annotator2"] is not None for r in g) for g in groups.values()
        ),
        disagreement_count=sum(r["disagreement"] for r in rows),
        role_counts=dict(Counter(r["role"] for r in rows)),
        excluded_count=0,
    )


def inspect(root: Path) -> Json:
    """Authenticate separate source and producer gates; retain independent inventory on producer drift."""
    progress("authentication_before")
    checks: list[Json] = []
    references: list[Json] = []
    primaries: Json = {}
    for name, pin in PINS.items():
        path = root / "results" / (name + ".json")
        reference = ref(path)
        references.append(reference)
        check = gate(
            path,
            "sha256",
            "sha256:" + pin,
            reference["sha256"] or "missing_file",
            "primary_bytes",
            name,
        )
        checks.append(check)
        if check["passed"]:
            primaries[name.split("_")[1]] = json.loads(path.read_bytes())
    empty = dict(
        rows=[],
        question_count=0,
        annotator1_observed_count=0,
        annotator2_observed_count=0,
        annotator2_missing_count=0,
        annotator2_question_count=0,
        disagreement_count=0,
        role_counts={},
        excluded_count=0,
    )
    result: Json = dict(
        empty,
        release_commit=old.COMMIT,
        exposure_ledger=[],
        label_authority_reference={},
        original_group_partition_reference={},
        original_capacity_null_reference={},
        historical_failures=[],
    )
    source_checks = len(checks)
    try:
        if len(primaries) != len(PINS):
            raise ValueError("required_primaries")
        for n, field, expected in [
            ("8403", "release_criterion_ready_score", 1),
            ("8429", "methods_ready_score", 1),
            ("8435", "learning_audit_ready_score", 1),
        ]:
            path = (
                root / "results" / next(name + ".json" for name in PINS if name.split("_")[1] == n)
            )
            checks.append(
                gate(
                    path,
                    field,
                    expected,
                    primaries[n].get(field, "missing_field"),
                    "upstream_gate",
                    n,
                )
            )
        manifest_ref = primaries["8389"]["manifest_reference"]
        manifest = read_ref(manifest_ref)
        references.append(manifest_ref)
        if manifest["release_commit"] != old.COMMIT or manifest["target_mapping"] != old.TARGET:
            raise ValueError("original_label_lineage")
        files = manifest["release_files"]
        if set(files) != set(old.RELEASE_FILES):
            raise ValueError("release_file_roster")
        operands = {name: old.authenticate(name, reference) for name, reference in files.items()}
        references.extend(dict(path=r["path"], sha256=r["sha256"]) for r in files.values())
        protocol_ref = dict(
            path=str(root / "openspec/change-proposals/v726-methods-protocol.json"),
            sha256="sha256:bb868af1d6f173c5b98155498e5daadd286e2641ef4aa3c20b7a0671b1b95117",
        )
        inherited = read_ref(protocol_ref)
        references.append(protocol_ref)
        master = list(csv.DictReader(io.StringIO(operands[old.MASTER].decode("utf-8-sig"))))
        reduced = inventory(master, inherited["partition"]["roles"])
        checks.append(
            gate(
                Path(files[old.MASTER]["path"]),
                "response_and_question_counts",
                [900, 300],
                [len(reduced["rows"]), reduced["question_count"]],
                "complete_inventory",
                "LPHB",
            )
        )
        result.update(
            reduced,
            label_authority_reference=dict(
                commit=old.COMMIT,
                files=files,
                target_mapping=manifest["target_mapping"],
                dictionary=files["data/human_judge_release_data_dictionary.csv"],
            ),
            original_group_partition_reference=protocol_ref,
            original_capacity_null_reference=references[
                list(PINS).index("experiment_8435_v726_capacity_benefit_audit")
            ],
            exposure_ledger=[
                dict(
                    experiment_id=int(n),
                    honest_verdict=p["honest_verdict"],
                    verdict_class=p["verdict_class"],
                    exposure_scope=p.get("exposure_scope"),
                    ledger=p.get("exposure_ledger"),
                    gate_check_summary=p.get("gate_check_summary", []),
                )
                for n, p in primaries.items()
            ],
            historical_failures=[
                dict(
                    experiment_id=int(n),
                    honest_verdict=p["honest_verdict"],
                    gate_check_summary=p.get("gate_check_summary", []),
                )
                for n, p in primaries.items()
                if p["verdict_class"] in {"blocked", "null"}
            ],
        )
        # A reviewed author blob supports inventory even if its old Carnot producer has drifted.
        for reference in primaries["8389"]["code_config_hashes"]:
            path = Path(reference["path"])
            references.append(ref(path))
            checks.append(
                gate(
                    path,
                    "sha256",
                    reference["sha256"],
                    ref(path)["sha256"] or "missing_file",
                    "historical_producer_hash",
                    "exp8389-human-label-custody",
                )
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        checks.append(
            gate(
                root,
                "authenticated_release_and_partition",
                True,
                str(error),
                "source_authority",
                "LPHB_V726",
            )
        )
    result.update(
        gate_check_summary=checks,
        source_references=references,
        source_checks_count=source_checks,
        blocker=next((c["check"] for c in checks if not c["passed"]), None),
    )
    progress("authentication_after", len(result["rows"]), 0)
    return result
