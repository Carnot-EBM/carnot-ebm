"""Tests for Exp 5171 Set-Encoder cross-corpus n>=30 hardening.

Spec refs: REQ-VERIFY-5171, SCENARIO-VERIFY-5171,
SCENARIO-VERIFY-5171-UPSTREAM-BLOCKED, SCENARIO-VERIFY-5171-INSUFFICIENT-POOL.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244
from carnot.reporting import harden_set_encoder_cross_corpus_n30_5171 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _adversarial_clean(_path: Path) -> dict[str, Any]:
    return {"returncode": 0, "reports": [{"flags": [], "flag_count": 0}]}


def _features() -> dict[str, float]:
    return {name: 0.0 for name in exp4244.FEATURE_NAMES} | {
        "vote_weight": 1.0,
        "grid_height": 1.0,
        "grid_width": 1.0,
        "grid_cells": 1.0,
        "set_candidate_count": 2.0,
    }


def _original_task(task_id: str = "gap3_stage2:50a16a69") -> dict[str, Any]:
    return {
        "candidate_count": 2,
        "candidates": [
            {
                "candidate_id": f"{task_id}::candidate0",
                "candidate_index": 0,
                "features": _features(),
                "grid": [[9]],
                "is_correct": True,
                "votes": 1.0,
            },
            {
                "candidate_id": f"{task_id}::candidate1",
                "candidate_index": 1,
                "features": _features(),
                "grid": [[8]],
                "is_correct": False,
                "votes": 0.0,
            },
        ],
        "oracle_present": True,
        "raw_task_id": task_id.split(":")[-1],
        "source_id": task_id.split(":")[0],
        "task_id": task_id,
        "vote_top_candidate_id": f"{task_id}::candidate0",
        "wrong_majority": False,
    }


def _write_5171_inputs(root: Path, *, second_pool_source: Path | None = None) -> None:
    source = second_pool_source or mod.EXP5160_SECOND_POOL_REL
    _write_json(
        root / mod.EXP5160_REL,
        {
            "experiment": "experiment_5160_oracle_distinct_cross_corpus_closure_v473",
            "honest_verdict": "success_arc_set_encoder_win_survives_cross_corpus_replication",
            "second_pool_source": str(source),
            "second_pool_leak_audit_passed": True,
            "cross_corpus_delta": 0.5,
            "cross_corpus_delta_ci95": [0.5, 0.5],
            "held_out_task_n": 24,
            "verifier_is_oracle": False,
        },
    )
    _write_gzip_json(
        root / source,
        {
            "schema": "carnot.arcgen_cross_generator_pool_4291.v1",
            "source_kind": "arcgen",
            "tasks": [_original_task("arcgen:prior:000")],
        },
    )
    _write_gzip_json(
        root / mod.ORIGINAL_POOL_REL,
        {
            "candidate_n": 2,
            "positive_candidate_n": 1,
            "schema": "carnot.arc_candidate_pool_grow.v1",
            "task_n": 1,
            "tasks": [_original_task()],
            "wrong_majority_n": 0,
        },
    )


def _fake_catalog(generator_n: int = 10) -> dict[str, tuple[Any, Any]]:
    catalog: dict[str, tuple[Any, Any]] = {}
    for index in range(generator_n):
        task_id = f"{index:08x}"

        def generator(index: int = index) -> dict[str, Any]:
            value = index % 9
            return {"input": [[value]], "output": [[(value + 1) % 10]]}

        catalog[task_id] = (generator, None)
    return catalog


def _patch_seed_results(
    monkeypatch: pytest.MonkeyPatch,
    *,
    leak_training_collision: bool = False,
    seed_deltas: list[float] | None = None,
) -> None:
    deltas = seed_deltas or [0.48, 0.5, 0.52, 0.49, 0.51]

    def fake_seed(
        corpus: exp4244.GrownPoolCorpus,
        *,
        random_seed: int,
        **_kwargs: object,
    ) -> mod.SeedReplicationResult:
        index = mod.DEFAULT_RANDOM_SEEDS.index(random_seed)
        task_ids = sorted({row.task_id for row in corpus.rows})
        task_deltas = [1.0 if task_index % 2 == 0 else 0.0 for task_index, _ in enumerate(task_ids)]
        first_row = corpus.rows[0]
        train_task_ids = (first_row.task_id,) if leak_training_collision else ("not-held-out",)
        return mod.SeedReplicationResult(
            random_seed=random_seed,
            auroc=0.9 + index * 0.01,
            delta=deltas[index],
            held_out_task_n=corpus.held_out_task_n,
            candidate_count=len(corpus.rows),
            vote_at_1=0.25,
            set_encoder_at_1=0.25 + deltas[index],
            oracle_at_k=0.75,
            task_deltas=task_deltas,
            oof_rows=[
                exp4244.OOFRow(
                    task_id=first_row.task_id,
                    candidate_id=first_row.candidate_id,
                    correct=first_row.correct,
                    score=0.9,
                    fold=0,
                    train_task_ids=train_task_ids,
                )
            ],
        )

    monkeypatch.setattr(mod, "_train_seed_replication", fake_seed)


def test_req_5171_spec_declares_n30_contract() -> None:
    """REQ-VERIFY-5171: OpenSpec declares the n>=30 hardening contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-5171",
        "SCENARIO-VERIFY-5171",
        "SCENARIO-VERIFY-5171-UPSTREAM-BLOCKED",
        "SCENARIO-VERIFY-5171-INSUFFICIENT-POOL",
        "python/carnot/reporting/harden_set_encoder_cross_corpus_n30_5171.py",
        "results/experiment_5171_harden_set_encoder_cross_corpus_n30_v474.json",
        "held_out_task_n>=30",
        "cross_corpus_delta_ci95_n30",
        "variance_is_genuine",
        "blocked_insufficient_disjoint_pool_size",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_5171_expanded_arcgen_gate_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-5171: expanded same-source ARC-GEN set clears n>=30."""

    _write_5171_inputs(tmp_path)
    _patch_seed_results(monkeypatch)

    artifact = mod.run(
        tmp_path,
        arcgen_task_catalog=_fake_catalog(10),
        adversarial_runner=_adversarial_clean,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("success_")
    assert artifact["held_out_task_n"] == 30
    assert artifact["expanded_pool"]["source_pool_task_n"] == 24
    assert artifact["expanded_pool"]["generator_limit"] == 10
    assert artifact["expanded_pool"]["tasks_per_generator"] == 3
    assert artifact["cross_corpus_delta_n30"] == pytest.approx(0.5)
    assert artifact["cross_corpus_delta_ci95_n30"][0] > 0.0
    assert artifact["cross_corpus_delta_ci95_n30"][1] > artifact["cross_corpus_delta_ci95_n30"][0]
    assert artifact["variance_is_genuine"] is True
    assert artifact["seed_delta_variance_is_zero"] is False
    assert artifact["leak_audit_passed_on_expanded_set"] is True
    assert artifact["gate_passed"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["random_seeds_used"] == mod.DEFAULT_RANDOM_SEEDS
    assert artifact["adversarial_verify"]["status"] == "clean"
    assert (tmp_path / mod.OUTPUT_REL).exists()


def test_scenario_5171_upstream_missing_blocks_honestly(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5171-UPSTREAM-BLOCKED: missing Exp5160 stops scoring."""

    artifact = mod.run(tmp_path, arcgen_task_catalog=_fake_catalog(), adversarial_runner=_adversarial_clean)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.BLOCKED_UPSTREAM_VERDICT
    assert artifact["held_out_task_n"] == 0
    assert artifact["cross_corpus_delta_n30"] == 0.0
    assert artifact["cross_corpus_delta_ci95_n30"] == [0.0, 0.0]
    assert artifact["variance_is_genuine"] is False
    assert artifact["leak_audit_passed_on_expanded_set"] is False
    assert artifact["gate_passed"] is False
    assert artifact["random_seeds_used"] == []

    bad_source_root = tmp_path / "bad-source"
    _write_5171_inputs(bad_source_root, second_pool_source=Path("results/missing.json.gz"))
    (bad_source_root / "results/missing.json.gz").unlink()
    missing_source = mod.run(
        bad_source_root,
        arcgen_task_catalog=_fake_catalog(),
        adversarial_runner=_adversarial_clean,
    )
    assert missing_source["honest_verdict"] == mod.BLOCKED_UPSTREAM_VERDICT

    oracle_root = tmp_path / "oracle-upstream"
    _write_5171_inputs(oracle_root)
    payload = json.loads((oracle_root / mod.EXP5160_REL).read_text(encoding="utf-8"))
    payload["verifier_is_oracle"] = True
    _write_json(oracle_root / mod.EXP5160_REL, payload)
    oracle_blocked = mod.run(
        oracle_root,
        arcgen_task_catalog=_fake_catalog(),
        adversarial_runner=_adversarial_clean,
    )
    assert oracle_blocked["honest_verdict"] == mod.BLOCKED_UPSTREAM_VERDICT

    fallback_n_root = tmp_path / "fallback-source-n"
    _write_5171_inputs(fallback_n_root)
    payload = json.loads((fallback_n_root / mod.EXP5160_REL).read_text(encoding="utf-8"))
    payload.pop("held_out_task_n")
    _write_json(fallback_n_root / mod.EXP5160_REL, payload)
    assert mod._load_preconditions(fallback_n_root).source_pool_task_n == 1


def test_scenario_5171_insufficient_pool_blocks_without_padding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-5171-INSUFFICIENT-POOL: fewer than 30 usable tasks blocks."""

    _write_5171_inputs(tmp_path)
    _patch_seed_results(monkeypatch)

    artifact = mod.run(
        tmp_path,
        arcgen_task_catalog=_fake_catalog(2),
        adversarial_runner=_adversarial_clean,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.BLOCKED_INSUFFICIENT_POOL_VERDICT
    assert artifact["held_out_task_n"] == 6
    assert artifact["expanded_pool"]["usable_task_n"] == 6
    assert artifact["gate_passed"] is False

    overlap_selection = mod.build_expanded_arcgen_selection(
        tmp_path,
        original_payload={
            "tasks": [
                {
                    "task_id": "arcgen:00000000:000",
                    "raw_task_id": "00000000:000",
                    "candidates": [],
                }
            ]
        },
        source_pool_task_n=24,
        source_pool_sha256="source",
        arcgen_task_catalog=_fake_catalog(1),
        generator_limit=1,
        tasks_per_generator=1,
    )
    assert overlap_selection.dropped_overlap_task_n == 1

    def drop_to_too_small(**_kwargs: Any) -> tuple[None, list[Any], int]:
        return None, [], 0

    monkeypatch.setattr(mod, "_filtered_task_payload", drop_to_too_small)
    too_small_selection = mod.build_expanded_arcgen_selection(
        tmp_path,
        original_payload={"tasks": []},
        source_pool_task_n=24,
        source_pool_sha256="source",
        arcgen_task_catalog=_fake_catalog(1),
        generator_limit=1,
        tasks_per_generator=1,
    )
    assert too_small_selection.dropped_too_small_task_n == 1


def test_req_5171_expanded_leak_audit_failure_keeps_gate_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-5171: expanded-set training collisions fail the leak audit."""

    _write_5171_inputs(tmp_path)
    _patch_seed_results(monkeypatch, leak_training_collision=True)

    artifact = mod.run(
        tmp_path,
        arcgen_task_catalog=_fake_catalog(10),
        adversarial_runner=_adversarial_clean,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["held_out_task_n"] == 30
    assert artifact["leak_audit_passed_on_expanded_set"] is False
    assert artifact["expanded_leak_audit"]["heldout_training_task_collision_count"] == 5
    assert artifact["gate_passed"] is False


def test_req_5171_validation_and_statistical_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-5171: validation rejects non-actionable fields and degenerate CI."""

    _write_5171_inputs(tmp_path)
    _patch_seed_results(monkeypatch, seed_deltas=[0.5, 0.5, 0.5, 0.5, 0.5])
    artifact = mod.run(
        tmp_path,
        arcgen_task_catalog=_fake_catalog(10),
        adversarial_runner=_adversarial_clean,
    )
    assert artifact["seed_delta_variance_is_zero"] is True
    assert artifact["variance_is_genuine"] is True

    invalid_cases = [
        ({key: value for key, value in artifact.items() if key != "held_out_task_n"}, "missing"),
        ({**artifact, "honest_verdict": "pending"}, "terminal-prefixed"),
        ({**artifact, "held_out_task_n": 30.0}, "held_out_task_n"),
        ({**artifact, "cross_corpus_delta_n30": True}, "bare float"),
        ({**artifact, "cross_corpus_delta_ci95_n30": [0.1]}, "ci95"),
        ({**artifact, "variance_is_genuine": 1}, "bare bool"),
        ({**artifact, "leak_audit_passed_on_expanded_set": "yes"}, "bare bool"),
        ({**artifact, "gate_passed": "yes"}, "bare bool"),
        ({**artifact, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**artifact, "solve_provenance": "live"}, "solve_provenance"),
        ({**artifact, "random_seeds_used": [5171.0]}, "bare ints"),
        ({**artifact, "reproducibility_checksum": "nope"}, "sha256"),
        ({**artifact, "field_principles": {}}, "field_principles"),
        ({**artifact, "spec_refs": []}, "spec_refs"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)

    assert mod._task_bootstrap_ci95([], random_seed=1, resamples=10) == [0.0, 0.0]
    assert mod._task_bootstrap_ci95([0.5], random_seed=1, resamples=10) == [0.5, 0.5]
    assert mod._task_delta_means([]) == []
    assert mod._variance_is_genuine([0.5] * 3, [0.5, 0.5]) is False
    assert mod._ci_excludes_zero([0.1, 0.2]) is True
    assert mod._ci_excludes_zero([-0.1, 0.2]) is False
    flagged = mod._clean_adversarial_report(
        {"returncode": 1, "reports": [{"flags": [{"kind": "CIRCULAR_MOAT_OVERCLAIM"}]}]}
    )
    assert flagged["status"] == "flagged"
    assert flagged["circular_moat_overclaim_clean"] is False

    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(mod.BlockedRun):
        mod._read_json_object(list_json)
    list_gzip = tmp_path / "list.json.gz"
    _write_gzip_json(list_gzip, [])
    with pytest.raises(mod.BlockedRun):
        mod._read_gzip_json_object(list_gzip)
    with pytest.raises(mod.BlockedRun):
        mod._resolve_source_path(tmp_path, "")

    original_signature = mod.exp5160._pool_signature(
        {
            "tasks": [
                {
                    "task_id": "original",
                    "raw_task_id": "original",
                    "candidates": [
                        {"candidate_id": "a", "grid": [[1]], "is_correct": True},
                        {"candidate_id": "b", "grid": [[2]], "is_correct": False},
                    ],
                }
            ]
        }
    )
    filtered, filtered_rows, dropped = mod._filtered_task_payload(
        task_payload={
            "task_id": "second",
            "raw_task_id": "second",
            "candidates": [
                {"candidate_id": "second::candidate0", "grid": [[1]], "features": _features()},
                {"candidate_id": "second::candidate1", "grid": [[2]], "features": _features()},
            ],
        },
        rows=[
            SimpleNamespace(
                candidate_id="second::candidate0",
                candidate_index=0,
                correct=True,
                task_id="second",
                features=_features(),
                vote_weight=1.0,
            ),
            SimpleNamespace(
                candidate_id="second::candidate1",
                candidate_index=1,
                correct=False,
                task_id="second",
                features=_features(),
                vote_weight=0.5,
            ),
        ],
        original_signature=original_signature,
    )
    assert filtered is None
    assert filtered_rows == []
    assert dropped == 2
