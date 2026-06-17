"""Tests for Exp 4314 IR3DE+CASCAL cross-domain selector rerun.

Spec refs: REQ-VERIFY-4314, SCENARIO-VERIFY-4314.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4314_cross_domain_selector_ir3de_cascal as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _adversarial_clean(_path: Path) -> dict[str, Any]:
    return {"returncode": 0, "reports": [{"flag_count": 0, "flags": [], "max_severity": 0}]}


def _candidate(
    *,
    task_id: str,
    domain_id: str,
    family_id: str,
    index: int,
    correct: bool,
    vote_weight: float,
    quality: float,
    candidate_count: int = 4,
) -> mod.CandidateRow:
    return mod.CandidateRow(
        task_id=task_id,
        candidate_id=f"{task_id}::candidate{index}",
        candidate_index=index,
        domain_id=domain_id,
        family_id=family_id,
        target_hash=f"sha256:{domain_id}:{task_id}",
        is_correct=correct,
        vote_weight=vote_weight,
        features=mod.common_feature_payload(
            vote_weight=vote_weight,
            quality=quality,
            candidate_count=candidate_count,
            entropy=0.5 + index * 0.1,
        ),
    )


def _domain_pool(domain_id: str, task_n: int = 20) -> mod.DomainPool:
    rows: list[mod.CandidateRow] = []
    for task_index in range(task_n):
        task_id = f"{domain_id}:task{task_index:02d}"
        family_id = f"{domain_id}:family{task_index % 4}"
        mode = task_index % 4
        oracle_present = mode != 3
        vote_correct = mode == 0
        if oracle_present:
            rows.append(
                _candidate(
                    task_id=task_id,
                    domain_id=domain_id,
                    family_id=family_id,
                    index=0,
                    correct=True,
                    vote_weight=12.0 if vote_correct else 2.0,
                    quality=0.96,
                )
            )
        else:
            rows.append(
                _candidate(
                    task_id=task_id,
                    domain_id=domain_id,
                    family_id=family_id,
                    index=0,
                    correct=False,
                    vote_weight=2.0,
                    quality=0.24,
                )
            )
        for index in range(1, 4):
            rows.append(
                _candidate(
                    task_id=task_id,
                    domain_id=domain_id,
                    family_id=family_id,
                    index=index,
                    correct=False,
                    vote_weight=3.0 if vote_correct else 10.0 - index,
                    quality=0.18 + index * 0.03,
                )
            )
    return mod.DomainPool(
        domain_id=domain_id,
        rows=rows,
        source_path=f"fixture/{domain_id}.json",
        source_sha256="sha256:" + domain_id * 8,
        provenance={"fixture": True, "task_n": task_n},
    )


def _three_domains() -> dict[str, mod.DomainPool]:
    return {
        "arc": _domain_pool("arc"),
        "arcgen": _domain_pool("arcgen"),
        "fover": _domain_pool("fover"),
    }


def test_req_verify_4314_spec_declares_ir3de_cascal_contextprm_gate() -> None:
    """REQ-VERIFY-4314: OpenSpec declares the upgraded held-out-domain contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4314",
        "SCENARIO-VERIFY-4314",
        "python/carnot/experiment_4314_cross_domain_selector_ir3de_cascal.py",
        "results/experiment_4314_cross_domain_selector_ir3de_cascal.py",
        "IR3DE",
        "CASCAL",
        "ContextPRM",
        "blocked_insufficient_domains",
        "cross_domain_delta_ci95",
        "vote_at_1 > 0.05",
        "oracle_at_k < 1.0",
        "label_ablation_robust",
        "GAP-CROSS-DOMAIN-FAMILY-INVARIANT-SELECTION-4305",
        "verifier_is_oracle=false",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_4314_ir3de_cascal_gate_and_label_ablation_survive(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4314: held-out FoVer lift survives label removal."""

    domains = _three_domains()
    artifact = mod.run(
        tmp_path,
        domain_loaders={name: (lambda pool=pool: pool) for name, pool in domains.items()},
        set_encoder_loader=lambda _root: {"verifier_is_oracle": False, "status": "fixture_loaded"},
        held_out_domain="fover",
        adversarial_runner=_adversarial_clean,
        bootstrap_resamples=200,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: cross_domain_selection_survives_ir3de_cascal"
    assert artifact["cross_domain_selection_holds"] is True
    assert artifact["cross_domain_delta"] == pytest.approx(0.5)
    assert artifact["cross_domain_delta_ci95"][0] > 0.0
    assert artifact["vote_at_1"] == pytest.approx(0.25)
    assert artifact["oracle_at_k"] == pytest.approx(0.75)
    assert artifact["label_ablation_robust"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["adversarial_verify"]["status"] == "clean"
    assert artifact["adversarial_verify"]["degenerate_separation_clean"] is True
    assert artifact["model_specs"]["router"]["kind"] == "ir3de_linear_domain_expert_ridge"
    assert artifact["model_specs"]["cascal_pretraining"]["held_out_target_labels_used"] is False
    assert artifact["model_specs"]["contextprm_features"]["domain_agnostic"] is True
    assert set(artifact["per_domain_delta"]) == {"arc", "arcgen", "fover"}
    assert artifact["per_domain_delta"]["fover"]["cross_domain_delta"] == pytest.approx(0.5)
    assert Path(artifact["cross_domain_pool_path"]).exists()
    assert Path(artifact["domain_manifest_path"]).exists()
    saved = json.loads((tmp_path / mod.OUTPUT_REL).read_text(encoding="utf-8"))
    assert saved == artifact


def test_req_verify_4314_contextprm_features_ignore_labels() -> None:
    """REQ-VERIFY-4314: ContextPRM features are task-structure features, not labels."""

    task_a = [
        _candidate(
            task_id="arc:task00",
            domain_id="arc",
            family_id="arc:family0",
            index=0,
            correct=True,
            vote_weight=2.0,
            quality=0.96,
        ),
        _candidate(
            task_id="arc:task00",
            domain_id="arc",
            family_id="arc:family0",
            index=1,
            correct=False,
            vote_weight=9.0,
            quality=0.21,
        ),
    ]
    task_b = [
        mod.CandidateRow(
            task_id=row.task_id.replace("arc", "fover"),
            candidate_id=row.candidate_id.replace("arc", "fover"),
            candidate_index=row.candidate_index,
            domain_id="fover",
            family_id="fover:family9",
            target_hash=row.target_hash,
            is_correct=row.is_correct,
            vote_weight=row.vote_weight,
            features=dict(row.features),
        )
        for row in task_a
    ]

    assert mod.contextprm_feature_vector(task_a, task_a[0]) == mod.contextprm_feature_vector(task_b, task_b[0])
    assert "domain_id" not in mod.CONTEXTPRM_FEATURE_NAMES
    assert "family_id" not in mod.CONTEXTPRM_FEATURE_NAMES


def test_req_verify_4314_holds_requires_guards_and_ablation() -> None:
    """REQ-VERIFY-4314: the bare hold bool is gated by every non-degenerate guard."""

    passing = {
        "cross_domain_delta": 0.5,
        "cross_domain_delta_ci95": [0.25, 0.75],
        "vote_at_1": 0.25,
        "oracle_at_k": 0.75,
        "label_ablation_robust": True,
    }

    assert mod.cross_domain_selection_holds_from_metrics(passing) is True
    assert mod.cross_domain_selection_holds_from_metrics(passing | {"cross_domain_delta": 0.0}) is False
    assert mod.cross_domain_selection_holds_from_metrics(passing | {"cross_domain_delta_ci95": [-0.1, 0.4]}) is False
    assert mod.cross_domain_selection_holds_from_metrics(passing | {"vote_at_1": 0.0}) is False
    assert mod.cross_domain_selection_holds_from_metrics(passing | {"oracle_at_k": 1.0}) is False
    assert mod.cross_domain_selection_holds_from_metrics(passing | {"cross_domain_delta": 0.99}) is False
    assert mod.cross_domain_selection_holds_from_metrics(passing | {"label_ablation_robust": False}) is False


def test_scenario_4314_label_ablation_failure_is_complete_not_a_win(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4314: a label-ablation collapse blocks the headline."""

    domains = _three_domains()
    artifact = mod.run(
        tmp_path,
        domain_loaders={name: (lambda pool=pool: pool) for name, pool in domains.items()},
        set_encoder_loader=lambda _root: {"verifier_is_oracle": False, "status": "fixture_loaded"},
        held_out_domain="fover",
        adversarial_runner=_adversarial_clean,
        bootstrap_resamples=200,
        label_ablation_tolerance=-0.6,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: label_ablation_failure_router_read_label"
    assert artifact["cross_domain_selection_holds"] is False
    assert artifact["cross_domain_delta"] == pytest.approx(0.5)
    assert artifact["label_ablation_robust"] is False
    assert artifact["missing_verifier_gaps"][0]["failure_mode"] == "label_ablation_failure_router_read_label"


def test_scenario_4314_blocks_when_less_than_three_domains_load(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4314: fewer than three domains stops honestly."""

    artifact = mod.run(
        tmp_path,
        domain_loaders={
            "arc": lambda: _domain_pool("arc"),
            "arcgen": lambda: _domain_pool("arcgen"),
            "fover": lambda: (_ for _ in ()).throw(FileNotFoundError("missing fover")),
        },
        set_encoder_loader=lambda _root: {"verifier_is_oracle": False, "status": "fixture_loaded"},
        adversarial_runner=_adversarial_clean,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_insufficient_domains"
    assert artifact["cross_domain_selection_holds"] is False
    assert artifact["cross_domain_delta"] == 0.0
    assert artifact["cross_domain_delta_ci95"] == [0.0, 0.0]
    assert artifact["vote_at_1"] == 0.0
    assert artifact["oracle_at_k"] == 0.0
    assert artifact["label_ablation_robust"] is False
    assert artifact["model_specs"]["available_domains"] == ["arc", "arcgen"]
    assert artifact["model_specs"]["missing_domains"][0]["domain_id"] == "fover"


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"honest_verdict": "draft"}, "terminal-prefixed"),
        ({"cross_domain_selection_holds": 1}, "cross_domain_selection_holds"),
        ({"cross_domain_delta": {"value": 0.5}}, "cross_domain_delta"),
        ({"cross_domain_delta_ci95": [0.1]}, "cross_domain_delta_ci95"),
        ({"vote_at_1": True}, "vote_at_1"),
        ({"oracle_at_k": "0.75"}, "oracle_at_k"),
        ({"label_ablation_robust": 1}, "label_ablation_robust"),
        ({"verifier_is_oracle": True}, "verifier_is_oracle"),
        ({"field_principles": {}}, "field_principles"),
        ({"spec_refs": ["REQ-VERIFY-4314"]}, "spec_refs"),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_req_4314(
    tmp_path: Path, patch: dict[str, Any], message: str
) -> None:
    """REQ-VERIFY-4314: required gate fields remain bare and exact."""

    artifact = mod.run(
        tmp_path,
        domain_loaders={name: (lambda pool=pool: pool) for name, pool in _three_domains().items()},
        set_encoder_loader=lambda _root: {"verifier_is_oracle": False, "status": "fixture_loaded"},
        adversarial_runner=_adversarial_clean,
        bootstrap_resamples=50,
    )
    bad = artifact | patch
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad)
