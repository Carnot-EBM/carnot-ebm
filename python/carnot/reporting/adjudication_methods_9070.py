"""REQ-REPORT-9070-V774: preserve original targets until source families are known.

The five files share response and section identities. That join does not prove
which question-generation variants have the same parent. Unknown ancestry must
stop learning before labels can influence grouping or a favorable panel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.human_source_authority_9001 import normalized
from carnot.reporting.natural_label_eligibility_8975 import ref as ref
from carnot.reporting.spline_basis_methods_8986 import gate as gate

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_9070_v774_adjudication_methods"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_adjudication_methods_9070.py"
OWNED = [
    "python/carnot/reporting/adjudication_methods_9070.py",
    "python/carnot/reporting/adjudication_methods_runner_9070.py",
    CLI,
]
REVISION = "30d9349a2a8d6a7395681cfbd23cc8116af5a2da"
DEFAULT_INPUT = Path("/tmp/carnot-9070-private")
MODEL_SPECS: list[Json] = []
ANNOTATION_FILES = [
    "data/mdhjudgments-" + name + ".jsonl"
    for name in [
        "firstpassannotations",
        "firstmedicalexpertadjudication",
        "secondmedicalexpertadjudication",
        "firstfactcheckeradjudication",
        "secondfactcheckeradjudication",
    ]
]
PINS = {
    "literature/adjudication.html": "sha256:4df42dcadbc16041c896f87985201548bf7ac452373185f9cc78cecce9b68844",
    "literature/taxonomy.html": "sha256:34a1eb35d84eb4ad797447be1c19887bf1e73840e428ebf64019635d284cc5c7",
    "literature/weak-supervision.html": "sha256:651c49ef68d1aa22af0fa0d4cf8eb851050345839edc4661ac1e2e0c43d8c0ee",
    "literature/calibration.html": "sha256:29d50da773226b552d4cd7db03a1fdca18538aea9dd4fd6adf59b890d3d1d20c",
    "github_tree.json": "sha256:1bd5a5b165d950518b806b22a66b924b6f0f14872d8978ff3d34f559429dd212",
    "README.md": "sha256:edbe42e3adf44b4b81de6685e65b93967c45f11a1b0e413d98eebcb8ffb9302f",
    "DATA_LICENSE": "sha256:7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661",
    "LICENSE": "sha256:1acdfe98c9b301fa0f7e028fa59e5cc8eefd4dce8a2b192bb847f98064e0281e",
    "mdhjudgments/enhanced_responses_to_csv.py": "sha256:2ce288c302db9efc5d544c53715e5517174b0a788a2157f21a2dcd712a693e73",
    "mdhjudgments/model.py": "sha256:12add3b494f4cfd9f06cbb0df59995f28b508e9b662c7cda5a14c98bd420d0cf",
    "mdhjudgments/score_p_r_f1.py": "sha256:abb866027fdf90417a0dd75958689b6b3c6bb2f4453f76e9d2da55d832ec838c",
    "mdhjudgments/show_adjudication_summary.py": "sha256:94dcbf6e0bf7459c9c7b13d55db4164b5e91e5c15f3583d955d28b1f965614cb",
    "mdhjudgments/show_annotation_summary.py": "sha256:f7231b271426ab3a30bfc4c321abfc7cd995ed23badf15c868ec238d4c1ad828",
    "data/mdhjudgments-firstfactcheckeradjudication.jsonl": "sha256:b13a693878690dc6ff5d3d36bd2a77833f856c58cf92dd5d631628b020593c05",
    "data/mdhjudgments-firstmedicalexpertadjudication.jsonl": "sha256:ae087ffe60773c9ad0e1aaf5cd52a8ace655d4c761134f444fdcaf23e5e7e443",
    "data/mdhjudgments-firstpassannotations.jsonl": "sha256:d4000c3dd43c44621aac32b4e93d0b084c2cf9ed4a1d5867d3f1c079195cbe48",
    "data/mdhjudgments-secondfactcheckeradjudication.jsonl": "sha256:299ad10431449d7a8d44ba7c490d5775cd8dd340023494af609a69a084f0e1d5",
    "data/mdhjudgments-secondmedicalexpertadjudication.jsonl": "sha256:6e0b7980a129eea74075e217793156870c3cd1669f3bd8e3f238cc82831be713",
}
HISTORY = {
    "results/experiment_9056_v773_human_release_authority.json": "sha256:5a0f0e0fd0831b61fd30f03a311c50327f46c6676f4a77fe33b431eacc86224a",
    "results/raw/experiment_9015_v770_human_disagreement_source/release/exposure_inventory.json": "sha256:ab488648f73f763d0c2ddaf5d1f6388dbb24572d0ab5691598ec49691bbe8c0e",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real phase boundaries so a blocked source cannot resemble computation."""
    print(f"[exp9070] phase={phase} completed={completed} pending={pending}", flush=True)


def label(value: str) -> int | None:
    """Keep unresolved original judgments unknown instead of importing a composite label."""
    return {"major": 1, "minor": 1, "no_hallucination": 0}.get(value)


def protocol() -> Json:
    """Freeze the later human estimand without loading an embedding model or a head."""
    return dict(
        panel="adjudication-conditioned sections with two definitive original medical passes",
        mapping=dict(major=1, minor=1, no_hallucination=0, needs_research=None, unresolved=None),
        judgment_field="q1_response; needs_research remains unknown even if composite resolves it",
        people="anonymous pass IDs do not establish distinct people",
        folds=dict(
            salt="v774-mdhj-v1",
            outer=5,
            minimum_families=60,
            minimum_test_families=12,
            minimum_class_groups_fit_cal=5,
            assignment="SHA256(salt + ':' + authenticated base-question ID), sorted, round-robin",
            inner="every fifth hash-ordered non-TEST group is CAL; remainder FIT",
            outcome_reshuffle=False,
            insufficient="terminal_null",
        ),
        features=dict(
            model="Qwen Q4_K_M",
            pooling="mean final layer",
            input="question plus full response; explicit target-section character offsets",
            maximum_tokens=4096,
            truncation=False,
            projection="Gaussian",
            projection_dimensions=32,
            projection_seed=774070,
            scaling="FIT-only coordinate scaling to [-3,3]; constant coordinates zero",
            denied=["reference", "judge_id", "explanation", "evidence", "label", "generator_id"],
        ),
        heads=dict(
            arms=[
                "prevalence",
                "full_embedding_ridge_logistic",
                "projected_ridge_logistic",
                "additive_cubic_spline_energy",
            ],
            shared_projected_features=True,
            ridge=0.01,
            penalize_intercept=False,
            knots=[-3, -2, -1, 0, 1, 2, 3],
            gradient_infinity_norm_max=1e-6,
            maximum_iterations=2000,
            loss="soft-target log loss",
            temperature_grid=[0.5, 1, 2, 4],
            temperature_ties="ascending",
            temperature_metric="CAL group-balanced per-original-judgment Brier",
            accept_threshold=0.1,
            reject_threshold=0.9,
        ),
        bound=dict(
            metric="mean of two squared errors to original medical judgments, then group balanced",
            forbidden_metric="squared error to averaged label",
            bootstrap_unit="base-question family",
            paired=True,
            bootstrap_draws=10000,
            bootstrap_seed=774077,
            one_sided_tail=0.05 / 3,
            required_improvements=["prevalence", "projected_linear", "full_linear"],
            scoped_benefit="all three lower bounds strictly exceed zero",
            ci95="separate two-sided percentile [0.025,0.975]",
            sensitivity="first-pass/fact-checker Brier and ranking reversals on identical overlap; descriptive",
            limitations="conditional on frozen OOF predictions; no training variability or population-wide medical truth",
        ),
    )


def feature_view(value: Json) -> Json:
    """Reject extra fields before a future learner can see judgment or generator shortcuts."""
    if set(value) != {"question", "response", "section_start", "section_end"}:
        raise ValueError("feature_fields")
    a, b = value["section_start"], value["section_end"]
    if not (type(a) is int and type(b) is int and 0 <= a < b <= len(value["response"])):
        raise ValueError("section_offsets")
    return dict(value)


def join(releases: dict[str, list[Json]]) -> Json:
    """Match all five exact source identities before looking inside any adjudication."""
    if set(releases) != set(ANNOTATION_FILES):
        raise ValueError("five_releases")
    identity: dict[str, Any] = {}
    rows: list[Json] = []
    parents: list[Json] = []
    owners: dict[str, str] = {}
    for file_index, name in enumerate(ANNOTATION_FILES):
        seen: set[str] = set()
        for record in releases[name]:
            ident = record["id"]
            sections = record["annotations"]["sections_with_annotations"]
            pairs = [(s["id"], s["section"]) for s in sections]
            if ident in seen or len({p[0] for p in pairs}) != len(pairs):
                raise ValueError("duplicate_identity")
            seen.add(ident)
            signature = (record["question"], record["answer"], pairs)
            if file_index == 0:
                identity[ident] = signature
                parents.append(
                    dict(
                        response_id=ident,
                        candidate_question_key=canonical_hash(normalized(record["question"])),
                        authenticated_base_question_id=None,
                        authority="missing_construction_lineage",
                    )
                )
                for section_id, text in pairs:
                    if section_id in owners:
                        raise ValueError("duplicate_section")
                    owners[section_id] = ident
                    rows.append(
                        dict(
                            unit_id=section_id,
                            group_id="unresolved-family:" + ident,
                            arm="original_medical_passes",
                            condition="authority_before_targets",
                            status="excluded",
                            metric="eligible_human_target",
                            value=None,
                            numerator=0,
                            denominator=1,
                            exclusion_reason="unknown_family_and_exposure_authority",
                            response_id=ident,
                            section_sha256=canonical_hash(text),
                        )
                    )
            elif identity.get(ident) != signature:
                raise ValueError("mismatched_identity")
        if seen != set(identity):
            raise ValueError("mismatched_response_set")
        progress("release_identity_join", file_index + 1, len(ANNOTATION_FILES) - file_index - 1)
    return dict(
        rows=rows,
        parent_group_manifest=parents,
        targets_opened=False,
        independent_count=0,
        response_count=len(identity),
        candidate_question_count=len({p["candidate_question_key"] for p in parents}),
    )


def inspect(directory: Path) -> Json:
    """Authenticate release and retained failure bytes before metadata-only projection."""
    progress("authenticate_before")
    checks: list[Json] = []
    references: list[Json] = []
    for base, pins in [(directory, PINS), (ROOT, HISTORY)]:
        for name, pin in pins.items():
            reference = ref(base / name)
            references.append(reference)
            checks.append(
                gate(
                    base / name,
                    "sha256",
                    pin,
                    reference["sha256"] or "missing_file",
                    "release_bytes:" + name,
                    "MDHJudgments_and_retained_history",
                )
            )
    reduced: Json = dict(
        rows=[],
        parent_group_manifest=[],
        targets_opened=False,
        independent_count=0,
        response_count=0,
        candidate_question_count=0,
    )
    exposure_rows: list[Json] = []
    if all(c["passed"] for c in checks):
        try:
            releases = {
                name: [json.loads(s) for s in (directory / name).read_text().splitlines()]
                for name in ANNOTATION_FILES
            }
            reduced = join(releases)
        except (KeyError, TypeError, ValueError) as error:
            checks.append(
                gate(
                    directory / ANNOTATION_FILES[0],
                    "identity_join",
                    "matching_five_releases",
                    str(error),
                    "release_identity",
                )
            )
        inventory_path = ROOT / next(
            (p for p in HISTORY if "exposure_inventory" in p), "missing_inventory"
        )
        inventory = json.loads(inventory_path.read_bytes()) if inventory_path.is_file() else {}
        for index, reference in enumerate(inventory.get("rows", [])):
            path = Path(reference["path"])
            references.append(dict(path=str(path), sha256=reference["sha256"]))
            checks.append(
                gate(
                    path,
                    "sha256",
                    reference["sha256"],
                    ref(path)["sha256"] or "missing_file",
                    "exposure_ledger_integrity",
                    "retained_exposure_ledger",
                )
            )
            if index % 50 == 0:
                progress("exposure_ledger_hashes", index + 1, len(inventory["rows"]) - index - 1)
        exposed = set(inventory.get("exposed_keys", []))
        exposure_rows = [
            dict(
                response_id=p["response_id"],
                candidate_question_key=p["candidate_question_key"],
                exact_prompt_match="prompt:" + p["candidate_question_key"] in exposed,
                family_overlap="unknown",
                eligible=False,
            )
            for p in reduced["parent_group_manifest"]
        ]
        checks += [
            gate(
                directory / "mdhjudgments/model.py",
                "author_base_question_construction_lineage",
                "authenticated_parent_for_all_template_and_generator_variants",
                "missing_field",
                "base_question_family_authority",
                "pinned_author_release",
            ),
            gate(
                inventory_path,
                "complete_family_exposure_join",
                True,
                "unknown",
                "prior_exposure_closure",
                "retained_exposure_ledger",
            ),
        ]
    failures = [c for c in checks if not c["passed"]]
    progress("authenticate_after", reduced["response_count"], 0)
    return dict(
        reduced,
        gate_check_summary=checks,
        blocker=failures[0]["check"],
        source_artifact_hashes=references,
        exposure_rows=exposure_rows,
        release_hashes=[r for r in references if Path(r["path"]).is_relative_to(directory)],
        eligibility_rows=reduced["rows"],
        fold_manifest=[],
        construction_evidence=dict(
            released_response_fields=["id", "question", "answer", "annotations"],
            missing="author mapping from source-version/question/template IDs to common base families",
            grouping_not_proven_by="ID prefixes or equal normalized question strings",
        ),
    )
