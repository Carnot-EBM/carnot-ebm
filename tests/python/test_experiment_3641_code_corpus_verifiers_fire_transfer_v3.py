"""Tests for Exp 3641 code-corpus verifier transfer.

Spec: REQ-CODE-3641, SCENARIO-CODE-3641.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import code_corpus_verifiers_fire_transfer_v3 as exp


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_rows(n_rows: int = 60) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(n_rows):
        passed = idx % 4 == 0
        body = "return x + 1" if passed else "return x - 1"
        rows.append(
            {
                "candidate_index": idx % 8,
                "corpus": "MBPP",
                "extracted_code": f"def f_{idx}(x):\n    {body}\n",
                "generation_duration_s": 0.1 + (idx % 7) * 0.01,
                "passed": passed,
                "raw_response": f"def f_{idx}(x):\n    {body}\n",
                "row_status": "candidate_passed" if passed else "candidate_failed_tests",
                "stable_id": f"mbpp-{idx // 8}",
                "tokens_generated": 8 + (idx % 5),
            }
        )
    return rows


def _write_exp2910_fixture(root: Path, rows: list[dict[str, Any]]) -> None:
    _write_json(
        root / exp.EXP2910_REL_PATH,
        {
            "artifact": "experiment_2910_sota_code_generation_corrigendum_v2",
            "candidate_results": rows,
            "honest_verdict": "complete: synthetic fixture",
        },
    )


def _ranked_scores(rows: list[dict[str, Any]], *, transfer: bool) -> list[float]:
    scores = []
    for row in rows:
        is_error = not bool(row["passed"])
        if transfer:
            scores.append(0.9 if is_error else 0.1)
        else:
            scores.append(0.45 if is_error else 0.55)
    return scores


@pytest.mark.parametrize(
    ("case_name", "write_corpus", "score_overrides", "expected_verdict", "expected_fire"),
    [
        (
            "blocked",
            False,
            {},
            "complete: blocked_no_labeled_code_corpus",
            False,
        ),
        (
            "inert",
            True,
            {"execution_scores": [], "math_scores": "transfer", "confidence_scores": "weak"},
            "complete: code_corpus_built_but_execution_verifiers_inert_diagnosed",
            False,
        ),
        (
            "no_transfer",
            True,
            {
                "execution_scores": "transfer",
                "math_scores": "no_transfer",
                "confidence_scores": "weak",
            },
            "complete: code_corpus_built_verifiers_fire_math_signal_does_not_transfer_discriminative_fragility",
            True,
        ),
        (
            "transfer",
            True,
            {"execution_scores": "transfer", "math_scores": "transfer", "confidence_scores": "weak"},
            "complete: code_corpus_built_verifiers_fire_math_signal_transfers_to_code",
            True,
        ),
    ],
)
def test_scenario_code_3641_parametrized_honest_verdicts(
    tmp_path: Path,
    case_name: str,
    write_corpus: bool,
    score_overrides: dict[str, Any],
    expected_verdict: str,
    expected_fire: bool,
) -> None:
    """SCENARIO-CODE-3641: fixture verdicts cover blocked/inert/no-transfer/transfer."""

    rows = _fixture_rows()
    if write_corpus:
        _write_exp2910_fixture(tmp_path, rows)

    overrides = dict(score_overrides)
    for key in ("execution_scores", "math_scores"):
        if isinstance(overrides.get(key), str) and overrides[key] in {"transfer", "no_transfer"}:
            overrides[key] = _ranked_scores(rows, transfer=overrides[key] == "transfer")
    if overrides.get("confidence_scores") == "weak":
        overrides["confidence_scores"] = [0.4 + 0.01 * (idx % 3) for idx in range(len(rows))]

    artifact = exp.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=14.5,
        score_overrides=overrides,
        tests_run=[f"SCENARIO-CODE-3641 {case_name}"],
    )

    assert artifact["honest_verdict"] == expected_verdict
    assert type(artifact["code_verifiers_fire"]) is bool
    assert artifact["code_verifiers_fire"] is expected_fire
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["duration_s"] == pytest.approx(4.5)
    assert artifact["inference_substrate"].startswith("verifier_ensemble_against_cached_candidates")
    assert artifact["field_principles"]["code_verifiers_fire"].startswith("BARE bool")
    if not write_corpus:
        assert artifact["n_examples"] == 0
        assert artifact["code_corpus_path"] is None
    else:
        assert artifact["n_examples"] == len(rows)
        assert artifact["code_corpus_name"].startswith("experiment_2910")
        assert artifact["code_corpus_path"] == str(exp.CORPUS_REL_PATH)
        saved_rows = [
            json.loads(line)
            for line in (tmp_path / exp.CORPUS_REL_PATH).read_text(encoding="utf-8").splitlines()
        ]
        assert len(saved_rows) == len(rows)
        assert {"candidate_code", "label", "test_outcome"} <= set(saved_rows[0])
        assert artifact["math_signal_code_auroc"]["ci95"]
        assert len(artifact["math_signal_code_auroc"]["bootstrap_seeds"]) >= 3


def test_req_code_3641_prefers_exp1999_only_when_candidate_source_exists(tmp_path: Path) -> None:
    """REQ-CODE-3641: Exp 1999 labels without source do not masquerade as a corpus."""

    _write_json(
        tmp_path / exp.EXP1999_REL_PATH,
        {
            "dataset_size": 50,
            "results": [
                {"baseline_passed": idx % 2 == 0, "task_id": f"HumanEval/{idx}"}
                for idx in range(50)
            ],
        },
    )
    _write_exp2910_fixture(tmp_path, _fixture_rows())

    artifact = exp.build_artifact(
        tmp_path,
        score_overrides={
            "execution_scores": _ranked_scores(_fixture_rows(), transfer=True),
            "math_scores": _ranked_scores(_fixture_rows(), transfer=True),
            "confidence_scores": [0.5 for _ in _fixture_rows()],
        },
    )

    assert artifact["code_corpus_name"].startswith("experiment_2910")
    assert artifact["exp1999_corpus_status"] == "labels_without_candidate_code"


def test_req_code_3641_imports_configured_verifier_modules() -> None:
    """REQ-CODE-3641: configured verifier modules are import-checked separately."""

    statuses = exp.import_configured_verifiers()

    assert set(statuses) == set(exp.CONFIGURED_VERIFIER_MODULES)
    assert statuses["ast_structure_verifier"]["importable"] is True
    assert "diagnosis" in statuses["controlled_invariance_executor_v2"]


def test_req_code_3641_validate_artifact_rejects_non_bare_fire_bool(tmp_path: Path) -> None:
    """REQ-CODE-3641: code_verifiers_fire must remain a bare top-level bool."""

    artifact = exp.blocked_artifact(
        root=tmp_path,
        started_s=1.0,
        now_s=2.0,
        reason="blocked_no_labeled_code_corpus",
    )
    artifact["code_verifiers_fire"] = {"value": False}

    with pytest.raises(ValueError, match="code_verifiers_fire"):
        exp.validate_artifact(artifact)


def test_req_code_3641_exp1999_source_rows_and_write_artifact(tmp_path: Path) -> None:
    """REQ-CODE-3641: Exp 1999 is accepted only when source rows are real."""

    rows = [
        {
            "candidate_code": (
                f"def f_{idx}(x):\n    return x + {idx % 3}\n"
                if idx % 2 == 0
                else f"def f_{idx}(:\n    return x\n"
            ),
            "label": idx % 2 == 0,
            "task_id": f"HumanEval/{idx}",
            "extracted_constraints": idx % 4,
        }
        for idx in range(60)
    ]
    _write_json(
        tmp_path / exp.EXP1999_REL_PATH,
        {"dataset_size": 60, "results": [*rows, "ignored-non-mapping"]},
    )

    output = exp.write_artifact(
        tmp_path,
        output_path=tmp_path / "result.json",
        tests_run=["REQ-CODE-3641 write_artifact"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    corpus_rows, status = exp.assemble_labeled_corpus(tmp_path)

    assert output == tmp_path / "result.json"
    assert artifact["code_corpus_name"] == "experiment_1999_code_verification_humaneval"
    assert artifact["exp1999_corpus_status"] == "labels_and_candidate_code"
    assert artifact["code_verifiers_fire"] is True
    assert len(corpus_rows) == 60
    assert status["fallback_status"] == "not_needed"


def test_req_code_3641_actual_scorers_parse_candidate_code_and_manifests(tmp_path: Path) -> None:
    """REQ-CODE-3641: real scorers parse code instead of returning constants."""

    manifest_path = tmp_path / "data" / "eval_manifests" / "mbpp_20260522.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "canonical_code": "def add_one(x):\n    return x + 1\n",
                "dataset": "MBPP",
                "prompt": "add one",
                "stable_id": "mbpp-1",
                "tests": ["assert add_one(1) == 2"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = [
        exp.corpus_row(
            candidate_code="def add_one(x):\n    return x + 1\n",
            label=True,
            test_outcome="candidate_passed",
            source="fixture",
            task_id="mbpp-1",
            metadata={
                "candidate_index": 0,
                "corpus": "MBPP",
                "manifest_path": str(manifest_path),
                "stable_id": "mbpp-1",
            },
        ),
        exp.corpus_row(
            candidate_code="def add_one(y):\n    return x - 1\n",
            label=False,
            test_outcome="candidate_failed_tests",
            source="fixture",
            task_id="mbpp-1",
            metadata={
                "candidate_index": 1,
                "corpus": "MBPP",
                "manifest_path": str(manifest_path),
                "stable_id": "mbpp-1",
            },
        ),
        exp.corpus_row(
            candidate_code="def broken(:\n    return 0\n",
            label=False,
            test_outcome="candidate_syntax_failed",
            source="fixture",
            task_id="mbpp-missing",
            metadata={"candidate_index": "bad-index", "corpus": "other"},
        ),
    ]
    imports = exp.import_configured_verifiers()

    execution = exp.score_execution_verifiers(
        rows,
        tmp_path,
        verifier_imports=imports,
        score_overrides={},
    )
    math_scores = exp.score_math_signal(rows, score_overrides={})
    confidence_scores = exp.score_confidence_baseline(rows, score_overrides={})

    assert execution["n_scored"] == 3
    assert execution["score_variance"] > 0.0
    assert any(row["name"] == "code_structural_dependency_verifier" for row in execution["per_verifier"])
    assert len(math_scores) == 3
    assert len(confidence_scores) == 3
    assert exp.structural_dependency_scores(rows, tmp_path)
    assert exp.load_manifest_lookup(rows, tmp_path)[("mbpp", "mbpp-1")]


def test_req_code_3641_helper_branches_and_validation_errors(tmp_path: Path, monkeypatch) -> None:
    """REQ-CODE-3641: helper branches keep edge cases explicit."""

    assert exp.exp1999_corpus_status({}, []) == "missing"
    assert exp.exp1999_corpus_status({"results": []}, []) == "no_results"
    assert exp.exp1999_corpus_status({"results": [{"candidate_code": "def f(): pass"}]}, []) == (
        "candidate_code_without_labels"
    )
    assert exp.exp1999_corpus_status({"results": [{"note": "none"}]}, []) == "insufficient_rows"
    assert exp.corpus_from_exp2910({"candidate_results": ["ignored", {"passed": True}]}) == []
    assert exp.metric_bundle([0, 1], [float("nan"), 0.2])["n"] == 1
    assert exp.terminal_verdict(
        n_examples=1,
        code_verifiers_fire=True,
        hypothesis="transfer",
    ) == "complete: blocked_no_labeled_code_corpus"
    assert exp.minmax_normalize([]) == []
    assert exp.minmax_normalize([2.0, 2.0]) == [0.5, 0.5]
    assert exp.normalize_code("# comment\n def f( x ):\n    return   x\n") == "def f( x ):\nreturn x"
    assert exp.normalize_corpus("Human Eval") == "humaneval"
    assert exp.normalize_corpus("MBPP") == "mbpp"
    assert exp.normalize_corpus("custom") == "custom"
    assert exp._label_from_row({"x": 1}, ("x",)) is True
    assert exp._label_from_row({"x": "passed"}, ("x",)) is True
    assert exp._label_from_row({"x": "failed"}, ("x",)) is False
    assert exp._label_from_row({"x": "unknown"}, ("x",)) is None
    assert exp._coerce_float("bad", 3.5) == 3.5
    assert exp._repo_path(tmp_path, Path("/tmp/absolute")) == Path("/tmp/absolute")

    malformed = tmp_path / "bad.jsonl"
    malformed.write_text("\nnot-json\n{\"ok\": true}\n[]\n", encoding="utf-8")
    assert exp._read_jsonl(malformed) == [{"ok": True}]
    assert exp._read_jsonl(tmp_path / "missing.jsonl") == []
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("not-json", encoding="utf-8")
    assert exp._read_json_object(bad_json) == {}

    real_import = exp.importlib.import_module

    def fake_import(name: str):
        if name.endswith("ast_structure_verifier"):
            raise RuntimeError("boom")
        return real_import(name)

    monkeypatch.setattr(exp.importlib, "import_module", fake_import)
    statuses = exp.import_configured_verifiers()
    assert statuses["ast_structure_verifier"]["importable"] is False
    assert "import_failed:RuntimeError" in statuses["ast_structure_verifier"]["diagnosis"]

    artifact = exp.blocked_artifact(
        root=tmp_path,
        started_s=1.0,
        now_s=2.0,
        reason="blocked_no_labeled_code_corpus",
    )
    for field, value, message in [
        ("honest_verdict", "blocked", "honest_verdict"),
        ("n_examples", "0", "n_examples"),
        ("duration_s", -1.0, "duration_s"),
    ]:
        broken = dict(artifact)
        broken[field] = value
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(broken)
    missing = dict(artifact)
    missing.pop("random_seed")
    with pytest.raises(ValueError, match="missing required"):
        exp.validate_artifact(missing)
