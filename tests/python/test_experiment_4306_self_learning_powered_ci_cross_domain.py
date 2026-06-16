"""Tests for Exp 4306 powered cross-domain self-learning CI.

Spec refs: REQ-VERIFY-4306, SCENARIO-VERIFY-4306.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4306_self_learning_powered_ci_cross_domain as mod
from carnot.reporting import arc_cross_family_transfer_existing_pool_4271 as exp4271


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _adversarial_clean(_path: Path) -> dict[str, Any]:
    return {"returncode": 0, "reports": [{"flag_count": 0, "flags": [], "max_severity": 0}]}


def _row(
    task_id: str,
    family_id: str,
    candidate_index: int,
    *,
    correct: bool,
    vote_weight: float,
    good_feature: float,
) -> exp4271.FamilyAnnotatedRow:
    return exp4271.FamilyAnnotatedRow(
        task_id=task_id,
        family_id=family_id,
        fold=0,
        candidate_id=f"{task_id}::candidate{candidate_index}",
        candidate_index=candidate_index,
        correct=correct,
        features={"good_feature": good_feature},
        vote_weight=vote_weight,
    )


def _fixture_inputs(
    tmp_path: Path,
    *,
    n_families: int = 60,
    vote_has_headroom: bool = True,
) -> mod.ExperimentInputs:
    rows: list[exp4271.FamilyAnnotatedRow] = []
    static_rows: list[dict[str, Any]] = []
    task_family_ids: dict[str, str] = {}
    task_folds: dict[str, int] = {}
    domains = ("arc", "arcgen", "fover")
    for index in range(n_families):
        domain_id = domains[index % len(domains)]
        family_id = f"{domain_id}:family-{index:02d}"
        task_id = f"{domain_id}:task-{index:02d}"
        vote_correct = not vote_has_headroom
        wrong = _row(
            task_id,
            family_id,
            0,
            correct=vote_correct,
            vote_weight=0.9,
            good_feature=0.1,
        )
        correct = _row(
            task_id,
            family_id,
            1,
            correct=True,
            vote_weight=0.1,
            good_feature=0.9,
        )
        rows.extend([wrong, correct])
        task_family_ids[task_id] = family_id
        task_folds[task_id] = index % 5
        static_rows.append(
            {
                "task_id": task_id,
                "family_id": family_id,
                "fold": index % 5,
                "vote_candidate_id": wrong.candidate_id,
                "vote_correct": vote_correct,
                "set_encoder_candidate_id": wrong.candidate_id,
                "set_encoder_correct": vote_correct,
            }
        )
    corpus = exp4271.FamilyAnnotatedCorpus(
        rows=rows,
        task_family_ids=task_family_ids,
        task_folds=task_folds,
        manifest_path=tmp_path / "cross_domain_manifest.json",
        manifest_sha256="manifest-sha",
        pool_artifact_path=tmp_path / "cross_domain_pool.json.gz",
        pool_artifact_sha256="pool-sha",
        upstream_checksum="upstream-sha",
        held_out_family_n=n_families,
        held_out_task_n=n_families,
        candidate_n=len(rows),
    )
    return mod.ExperimentInputs(
        corpus=corpus,
        static_task_rows=static_rows,
        feature_names=["good_feature"],
        cross_domain_pool_path=tmp_path / "cross_domain_pool.json.gz",
        cross_domain_pool_sha256="sha256:" + "a" * 64,
        domain_manifest_path=tmp_path / "cross_domain_manifest.json",
        domain_manifest_sha256="sha256:" + "b" * 64,
        domain_sources={
            "arc": {"source_path": "fixture/arc", "source_sha256": "sha256:" + "1" * 64},
            "arcgen": {"source_path": "fixture/arcgen", "source_sha256": "sha256:" + "2" * 64},
            "fover": {"source_path": "fixture/fover", "source_sha256": "sha256:" + "3" * 64},
        },
        upstream_artifacts={
            "experiment_4305": {"reproducibility_checksum": "sha256:" + "4" * 64},
            "experiment_4295": {"reproducibility_checksum": "sha256:" + "5" * 64},
        },
        input_notes=["fixture_cross_domain_family_stream"],
    )


def _write_loader_fixture(root: Path) -> None:
    results = root / "results"
    sources = root / "sources"
    results.mkdir(parents=True)
    sources.mkdir(parents=True)
    for domain_id in mod.exp4305.DOMAIN_ORDER:
        (sources / f"{domain_id}.json").write_text("{}\n", encoding="utf-8")

    tasks: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for domain_index, domain_id in enumerate(mod.exp4305.DOMAIN_ORDER):
        for task_index in range(4):
            task_id = f"{domain_id}:load:{task_index}"
            family_id = f"{domain_id}_family:{task_index}"
            candidates = []
            for candidate_index, correct in enumerate((False, True)):
                quality = 0.15 if candidate_index == 0 else 0.92
                vote_weight = 12.0 if candidate_index == 0 else 2.0
                features = mod.exp4305.common_feature_payload(
                    vote_weight=vote_weight,
                    quality=quality,
                    candidate_count=2,
                    entropy=0.4 + 0.1 * candidate_index + domain_index * 0.01,
                )
                candidates.append(
                    {
                        "task_id": task_id,
                        "candidate_id": f"{task_id}::candidate{candidate_index}",
                        "candidate_index": candidate_index,
                        "domain_id": domain_id,
                        "family_id": family_id,
                        "target_hash": f"sha256:{domain_id}:{task_index}",
                        "is_correct": correct,
                        "vote_weight": vote_weight,
                        "features": features,
                    }
                )
            tasks.append(
                {
                    "task_id": task_id,
                    "domain_id": domain_id,
                    "family_id": family_id,
                    "target_hash": f"sha256:{domain_id}:{task_index}",
                    "candidate_count": 2,
                    "oracle_present": True,
                    "vote_top_correct": False,
                    "candidates": candidates,
                }
            )
            manifest_rows.append(
                {
                    "task_id": task_id,
                    "domain_id": domain_id,
                    "family_id": family_id,
                    "target_hash": f"sha256:{domain_id}:{task_index}",
                    "candidate_count": 2,
                    "oracle_present": True,
                    "vote_top_correct": False,
                }
            )

    with gzip.open(root / mod.CROSS_DOMAIN_POOL_REL, "wt", encoding="utf-8") as handle:
        json.dump({"schema": "fixture", "tasks": tasks}, handle)
    (root / mod.CROSS_DOMAIN_MANIFEST_REL).write_text(
        json.dumps(
            {
                "schema": "fixture",
                "rows": manifest_rows,
                "domain_sources": {
                    domain_id: {
                        "source_path": str(sources / f"{domain_id}.json"),
                        "source_sha256": "sha256:" + str(index + 1) * 64,
                        "provenance": {"domain_index": index},
                    }
                    for index, domain_id in enumerate(mod.exp4305.DOMAIN_ORDER)
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / mod.CROSS_DOMAIN_ARTIFACT_REL).write_text(
        json.dumps({"reproducibility_checksum": "sha256:" + "6" * 64}) + "\n",
        encoding="utf-8",
    )


def test_req_4306_spec_declares_powered_cross_domain_contract() -> None:
    """REQ-VERIFY-4306: OpenSpec declares the powered cross-domain contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4306",
        "SCENARIO-VERIFY-4306",
        "python/carnot/experiment_4306_self_learning_powered_ci_cross_domain.py",
        "results/experiment_4306_self_learning_powered_ci_cross_domain.py",
        "results/experiment_4306_self_learning_powered_ci_cross_domain.json",
        "blocked_pools_missing",
        "best_adaptive_minus_static_delta",
        "best_adaptive_minus_static_ci95",
        "arm_deltas",
        "positive_control_headroom",
        "verifier_is_oracle=false",
        "without model weight mutation",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_4306_load_inputs_normalizes_persisted_cross_domain_pool(tmp_path: Path) -> None:
    """REQ-VERIFY-4306: loader requires all domain pools and qualifies families."""

    _write_loader_fixture(tmp_path)

    loaded = mod.load_inputs(tmp_path)

    assert loaded.cross_domain_pool_path == (tmp_path / mod.CROSS_DOMAIN_POOL_REL).resolve()
    assert loaded.domain_manifest_path == (tmp_path / mod.CROSS_DOMAIN_MANIFEST_REL).resolve()
    assert set(loaded.domain_sources) == set(mod.exp4305.DOMAIN_ORDER)
    assert loaded.upstream_artifacts["experiment_4305"]["reproducibility_checksum"].startswith("sha256:")
    assert loaded.upstream_artifacts["experiment_4295"]["status"] == "missing"
    assert loaded.corpus.held_out_task_n == 12
    assert loaded.corpus.held_out_family_n == 12
    assert len(loaded.static_task_rows) == 12
    assert all(":" in family for family in loaded.corpus.task_family_ids.values())


def test_req_4306_loader_edge_helpers_report_missing_inputs(tmp_path: Path) -> None:
    """REQ-VERIFY-4306: malformed or missing pools become blocked preconditions."""

    assert mod.BlockedRun("missing fixture").missing_pools == [{"reason": "missing fixture"}]
    assert mod._domain_sources({}) == {}
    assert mod._missing_source_paths(tmp_path, {})[0]["domain_id"] == "arc"
    assert mod._missing_source_paths(
        tmp_path,
        {"arc": {"source_path": "missing.json"}, "arcgen": {}, "fover": {}},
    )[0]["path"].endswith("missing.json")
    empty_pools = mod._domain_pools_from_payload(
        {"tasks": [None, {"candidates": [None]}, {"candidates": [{"domain_id": ""}]}]},
        {},
    )
    assert empty_pools == {}
    empty_corpus = mod._corpus_from_domain_pools(
        {},
        pool_path=tmp_path / "pool.json.gz",
        pool_sha256="sha256:" + "7" * 64,
        manifest_path=tmp_path / "manifest.json",
        manifest_sha256="sha256:" + "8" * 64,
    )
    assert empty_corpus.rows == []
    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        mod._read_json_object(list_json)
    list_gz = tmp_path / "list.json.gz"
    with gzip.open(list_gz, "wt", encoding="utf-8") as handle:
        json.dump([], handle)
    with pytest.raises(ValueError, match="JSON object"):
        mod._read_json_gz_object(list_gz)
    with pytest.raises(mod.BlockedRun):
        mod.load_inputs(tmp_path)


def test_req_4306_load_inputs_blocks_malformed_missing_and_empty_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-4306: loader failure modes are honest blocked preconditions."""

    (tmp_path / "results").mkdir()
    (tmp_path / mod.CROSS_DOMAIN_POOL_REL).write_text("not gzip", encoding="utf-8")
    (tmp_path / mod.CROSS_DOMAIN_MANIFEST_REL).write_text("{}\n", encoding="utf-8")
    with pytest.raises(mod.BlockedRun):
        mod.load_inputs(tmp_path)

    tmp_missing_source = tmp_path / "missing_source"
    _write_loader_fixture(tmp_missing_source)
    (tmp_missing_source / "sources" / "arc.json").unlink()
    with pytest.raises(mod.BlockedRun):
        mod.load_inputs(tmp_missing_source)

    tmp_empty_static = tmp_path / "empty_static"
    _write_loader_fixture(tmp_empty_static)
    monkeypatch.setattr(mod, "_static_rows_from_cross_domain_reports", lambda *_args, **_kwargs: [])
    with pytest.raises(mod.BlockedRun):
        mod.load_inputs(tmp_empty_static)


def test_req_4306_static_rows_skip_missing_reports_and_bad_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-4306: static-row extraction ignores malformed report rows."""

    monkeypatch.setattr(
        mod.exp4305,
        "_per_domain_reports",
        lambda *_args, **_kwargs: {"arc": {"task_rows": [None]}},
    )
    assert mod._static_rows_from_cross_domain_reports({}, random_seed=4306, bootstrap_resamples=1) == []


def test_scenario_4306_powered_ci_reports_best_adaptive_gain(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4306: powered bootstrap gates best-adaptive minus static."""

    inputs = _fixture_inputs(tmp_path)
    metrics = mod.measure_powered_adaptation(
        inputs,
        random_seed=4306,
        bootstrap_resamples=2000,
        retrieval_k=3,
    )

    assert metrics["arm_deltas"]["static"] == pytest.approx(0.0)
    assert metrics["arm_deltas"]["online"] == pytest.approx(59 / 60)
    assert metrics["arm_deltas"]["tier2_memory"] == pytest.approx(59 / 60)
    assert metrics["arm_deltas"]["tier2_retrieval"] == pytest.approx(59 / 60)
    assert metrics["best_adaptive_arm"] in {"online", "tier2_memory", "tier2_retrieval"}
    assert metrics["best_adaptive_minus_static_delta"] == pytest.approx(59 / 60)
    assert metrics["best_adaptive_minus_static_ci95"][0] > 0.0
    assert metrics["online_adaptation_helps"] is True
    assert metrics["positive_control_headroom"]["passed"] is True
    assert metrics["bootstrap_resamples"] == 2000


def test_scenario_4306_run_writes_required_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-4306: run emits required fields and clean verifier metadata."""

    inputs = _fixture_inputs(tmp_path)
    monkeypatch.setattr(mod, "load_inputs", lambda _root: inputs)

    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean, bootstrap_resamples=2000)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: powered_cross_domain_online_adaptation_helps"
    assert artifact["online_adaptation_helps"] is True
    assert artifact["best_adaptive_minus_static_delta"] == pytest.approx(59 / 60)
    assert artifact["best_adaptive_minus_static_ci95"][0] > 0.0
    assert artifact["arm_deltas"]["tier2_memory"] == pytest.approx(59 / 60)
    assert artifact["verifier_is_oracle"] is False
    assert artifact["model_specs"]["arms"]["static"]["model_training"] is False
    assert artifact["model_specs"]["arms"]["tier1_online"]["counter_update_only"] is True
    assert artifact["model_specs"]["arms"]["tier2_retrieval"]["weight_mutation"] is False
    assert artifact["model_specs"]["bootstrap_protocol"]["resamples"] == 2000
    assert artifact["adversarial_verify"]["status"] == "clean"
    assert artifact["adversarial_verify"]["circular_moat_overclaim_clean"] is True
    assert json.loads((tmp_path / mod.OUTPUT_REL).read_text(encoding="utf-8")) == artifact


def test_scenario_4306_blocks_when_pools_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-4306: missing pools stop with blocked_pools_missing."""

    def _blocked(_root: Path) -> mod.ExperimentInputs:
        raise mod.BlockedRun([{"path": "results/experiment_4305_cross_domain_pool.json.gz"}])

    monkeypatch.setattr(mod, "load_inputs", _blocked)
    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean, bootstrap_resamples=2000)

    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_pools_missing"
    assert artifact["online_adaptation_helps"] is False
    assert artifact["best_adaptive_minus_static_delta"] == 0.0
    assert artifact["best_adaptive_minus_static_ci95"] == [0.0, 0.0]
    assert artifact["arm_deltas"] == {
        "static": 0.0,
        "online": 0.0,
        "tier2_memory": 0.0,
        "tier2_retrieval": 0.0,
    }
    assert artifact["model_specs"]["missing_pools"][0]["path"].endswith("cross_domain_pool.json.gz")


def test_req_4306_complete_artifact_verdict_variants(tmp_path: Path) -> None:
    """REQ-VERIFY-4306: complete artifacts distinguish help, powered null, and no headroom."""

    inputs = _fixture_inputs(tmp_path)
    metrics = mod.measure_powered_adaptation(inputs, bootstrap_resamples=2000)

    powered_null = dict(metrics)
    powered_null.update(
        {
            "online_adaptation_helps": False,
            "best_adaptive_minus_static_delta": 0.0,
            "best_adaptive_minus_static_ci95": [0.0, 0.0],
        }
    )
    artifact = mod._complete_artifact(
        inputs=inputs,
        metrics=powered_null,
        checksum="sha256:" + "9" * 64,
        random_seed=4306,
        duration_s=0.01,
    )
    assert artifact["honest_verdict"] == "complete: powered_cross_domain_static_is_the_ceiling"

    no_headroom = dict(powered_null)
    no_headroom["positive_control_headroom"] = dict(
        powered_null["positive_control_headroom"],
        passed=False,
    )
    artifact = mod._complete_artifact(
        inputs=inputs,
        metrics=no_headroom,
        checksum="sha256:" + "0" * 64,
        random_seed=4306,
        duration_s=0.01,
    )
    assert artifact["honest_verdict"] == "complete: cross_domain_positive_control_headroom_missing"


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"__remove__": "online_adaptation_helps"}, "missing"),
        ({"honest_verdict": "pending"}, "terminal-prefixed"),
        ({"online_adaptation_helps": 1}, "online_adaptation_helps"),
        ({"best_adaptive_minus_static_delta": True}, "best_adaptive_minus_static_delta"),
        ({"best_adaptive_minus_static_ci95": [0.1]}, "best_adaptive_minus_static_ci95"),
        ({"arm_deltas": {"static": 0.0}}, "arm_deltas"),
        ({"positive_control_headroom": []}, "positive_control_headroom"),
        ({"verifier_is_oracle": True}, "verifier_is_oracle"),
        ({"random_seed": "4306"}, "random_seed"),
        ({"model_specs": []}, "model_specs"),
        ({"field_principles": {}}, "field_principles"),
        ({"spec_refs": ["REQ-VERIFY-4306"]}, "spec_refs"),
        (
            {"online_adaptation_helps": False, "best_adaptive_minus_static_delta": 1.0},
            "online_adaptation_helps",
        ),
    ],
)
def test_validate_artifact_rejects_req_4306_schema_violations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch: dict[str, Any],
    message: str,
) -> None:
    """REQ-VERIFY-4306: required gate fields stay bare and exact."""

    monkeypatch.setattr(mod, "load_inputs", lambda _root: _fixture_inputs(tmp_path))
    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean, bootstrap_resamples=2000)
    if "__remove__" in patch:
        bad = dict(artifact)
        bad.pop(str(patch["__remove__"]))
    else:
        bad = artifact | patch
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad)


def test_req_4306_positive_control_null_and_entrypoint(tmp_path: Path) -> None:
    """REQ-VERIFY-4306: nulls record headroom state and the entrypoint delegates."""

    inputs = _fixture_inputs(tmp_path, vote_has_headroom=False)
    metrics = mod.measure_powered_adaptation(
        inputs,
        random_seed=4306,
        bootstrap_resamples=2000,
        retrieval_k=3,
    )
    assert metrics["positive_control_headroom"]["passed"] is False
    assert metrics["online_adaptation_helps"] is False

    entrypoint = REPO / "results" / "experiment_4306_self_learning_powered_ci_cross_domain.py"
    text = entrypoint.read_text(encoding="utf-8") if entrypoint.exists() else ""
    assert "experiment_4306_self_learning_powered_ci_cross_domain" in text
