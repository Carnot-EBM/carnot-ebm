"""Exp6186 LiveCodeBench bank preregistration tests.

Spec refs:
  REQ-CODE-6186
  SCENARIO-CODE-6186-CACHED-SNAPSHOT-FAIL-CLOSED
  SCENARIO-CODE-6186-DISJOINT-DETERMINISTIC-SPLITS
  SCENARIO-CODE-6186-PRIVATE-TEST-NONINTERFERENCE
  SCENARIO-CODE-6186-EXECUTOR-FIXTURE-ONLY
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

from carnot import experiment_6186_livecodebench_bank_preregistration as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/code-verification/spec.md"


def _synthetic_rows(n: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    platforms = ["atcoder", "leetcode", "codeforces"]
    difficulties = ["easy", "medium", "hard"]
    for index in range(n):
        platform = platforms[index % len(platforms)]
        starter_code = (
            "class Solution:\n    def solve(self, nums: list[int]) -> int:\n        pass\n"
            if platform == "leetcode"
            else ""
        )
        metadata = {"func_name": "solve"} if starter_code else {}
        rows.append(
            {
                "question_title": f"Task {index}",
                "question_content": (
                    f"Solve synthetic task {index}. " + ("x" * (80 + (index % 7) * 120))
                ),
                "platform": platform,
                "question_id": f"synthetic_{index:03d}",
                "contest_id": f"contest_{index // 10:03d}",
                "contest_date": f"2024-{(index % 12) + 1:02d}-01T00:00:00",
                "starter_code": starter_code,
                "difficulty": difficulties[index % len(difficulties)],
                "public_test_cases": json.dumps(
                    [{"input": f"{index}\n", "output": f"{index}\n", "testtype": "stdin"}]
                ),
                "private_test_cases": json.dumps(
                    [
                        {
                            "input": f"PRIVATE_SENTINEL_{index}\n",
                            "output": f"{index}\n",
                            "testtype": "stdin",
                        }
                    ]
                ),
                "metadata": json.dumps(metadata),
                "_cache_coordinate": {
                    "shard": "synthetic.arrow",
                    "shard_index": index,
                    "global_index": index,
                },
            }
        )
    return rows


def test_req_code_6186_spec_declares_bank_contract() -> None:
    """REQ-CODE-6186: OpenSpec declares the cache, split, and isolation gates."""
    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-CODE-6186") :]

    for required in [
        "already cached LiveCodeBench snapshot",
        "fail closed",
        "platform, contest date, difficulty, metadata",
        "candidate correctness",
        "36 calibration",
        "36 held-selector",
        "18 continuous-learning seed",
        "30 continuous-learning prospective",
        "private-test vault",
        "candidate_and_model_access_count",
        "deterministic_cached_livecodebench_bank_preregistration",
        "SCENARIO-CODE-6186-CACHED-SNAPSHOT-FAIL-CLOSED",
        "SCENARIO-CODE-6186-DISJOINT-DETERMINISTIC-SPLITS",
        "SCENARIO-CODE-6186-PRIVATE-TEST-NONINTERFERENCE",
        "SCENARIO-CODE-6186-EXECUTOR-FIXTURE-ONLY",
    ]:
        assert required in section


def test_scenario_code_6186_selection_is_disjoint_deterministic_and_private_blind() -> None:
    """SCENARIO-CODE-6186-DISJOINT-DETERMINISTIC-SPLITS: metadata fixes splits."""
    rows = _synthetic_rows(144)
    records = mod.metadata_records_from_rows(rows)
    splits = mod.freeze_task_splits(records)

    assert {split: len(tasks) for split, tasks in splits.items()} == mod.SPLIT_SIZES
    assert mod.split_overlap_matrix(splits) == {
        left: {right: (len(splits[left]) if left == right else 0) for right in mod.SPLIT_ORDER}
        for left in mod.SPLIT_ORDER
    }
    assert len({record["task_id"] for tasks in splits.values() for record in tasks}) == 120

    private_mutated = []
    for row in rows:
        changed = dict(row)
        changed["private_test_cases"] = "PRIVATE_SENTINEL_REPLACED"
        private_mutated.append(changed)
    mutated_splits = mod.freeze_task_splits(mod.metadata_records_from_rows(private_mutated))
    assert {
        split: [record["task_id"] for record in tasks] for split, tasks in mutated_splits.items()
    } == {split: [record["task_id"] for record in tasks] for split, tasks in splits.items()}

    for record in records:
        feature_blob = json.dumps(record["selector_features"], sort_keys=True)
        assert "private" not in feature_blob.lower()
        assert "candidate" not in feature_blob.lower()
        assert "hidden" not in feature_blob.lower()


def test_scenario_code_6186_public_artifacts_do_not_expose_private_tests(tmp_path: Path) -> None:
    """SCENARIO-CODE-6186-PRIVATE-TEST-NONINTERFERENCE: public files stay clean."""
    artifact = mod.build_artifact_from_rows(
        REPO,
        rows=_synthetic_rows(144),
        data_dir=tmp_path / "data" / "research",
        result_path=tmp_path / "results" / "experiment_6186.json",
        dataset_receipt={
            "dataset_name": "synthetic/livecodebench",
            "revision": "synthetic-revision",
            "cache_path": str(tmp_path / "cache"),
            "cache_sha256": "sha256:synthetic",
            "task_count": 144,
            "download_attempted": False,
            "cache_unchanged_during_run": True,
        },
        command_receipts=[{"name": "focused", "command": "pytest exp6186", "exit_code": 0}],
        duration_s=0.25,
    )

    assert artifact["status"] == "complete_ready"
    assert artifact["candidate_and_model_access_count"] == 0
    assert artifact["bank_ready_score"] == 1
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.validate_artifact(artifact) == []

    public_path = Path(
        artifact["public_prompt_and_private_test_vault_paths_and_hashes"]["public_prompt_bank"][
            "path"
        ]
    )
    vault_path = Path(
        artifact["public_prompt_and_private_test_vault_paths_and_hashes"]["private_test_vault"][
            "path"
        ]
    )
    bank_path = Path(artifact["frozen_bank_path_and_hash"]["path"])
    result_path = tmp_path / "results" / "experiment_6186.json"

    public_text = public_path.read_text(encoding="utf-8")
    bank_text = bank_path.read_text(encoding="utf-8")
    result_text = result_path.read_text(encoding="utf-8")
    vault_text = vault_path.read_text(encoding="utf-8")
    for text in [public_text, bank_text, result_text, vault_text]:
        assert "PRIVATE_SENTINEL" not in text
        assert 'private_test_cases"' not in text
        assert "expected output" not in text.lower()
        assert "oracle_trace" not in text
        assert "candidate_code" not in text
        assert "hidden_state" not in text

    assert stat.S_IMODE(vault_path.stat().st_mode) & 0o077 == 0
    assert json.loads(public_text.splitlines()[0])["split"] in mod.SPLIT_ORDER


def test_scenario_code_6186_blocked_when_exact_split_counts_cannot_be_met(tmp_path: Path) -> None:
    """SCENARIO-CODE-6186-CACHED-SNAPSHOT-FAIL-CLOSED: insufficient cache blocks."""
    artifact = mod.build_artifact_from_rows(
        REPO,
        rows=_synthetic_rows(119),
        data_dir=tmp_path / "data" / "research",
        result_path=tmp_path / "results" / "experiment_6186.json",
        dataset_receipt={
            "dataset_name": "synthetic/livecodebench",
            "revision": "synthetic-revision",
            "cache_path": str(tmp_path / "cache"),
            "cache_sha256": "sha256:synthetic",
            "task_count": 119,
            "download_attempted": False,
            "cache_unchanged_during_run": True,
        },
        command_receipts=[{"name": "focused", "command": "pytest exp6186", "exit_code": 0}],
        duration_s=0.25,
    )

    assert artifact["status"] == "blocked"
    assert artifact["bank_ready_score"] == 0
    assert artifact["eligible_task_count"] == 119
    assert artifact["honest_verdict"].startswith("blocked:")
    assert "calibration=36" not in artifact["honest_verdict"]
    assert mod.validate_artifact(artifact) == []


def test_scenario_code_6186_executor_dry_run_uses_fixtures_only() -> None:
    """SCENARIO-CODE-6186-EXECUTOR-FIXTURE-ONLY: no generated candidates run."""
    receipt = mod.dry_run_executor_fixtures(timeout_s=0.5)

    assert receipt["candidate_solution_count"] == 0
    assert receipt["model_call_count"] == 0
    assert receipt["fixtures"][0]["kind"] == "maintainer_reference_fixture"
    assert receipt["fixtures"][0]["passed"] is True
    assert receipt["fixtures"][1]["kind"] == "timeout_fixture"
    assert receipt["fixtures"][1]["classification"] == "timeout_enforced"
    for policy in [
        "timeout_policy",
        "process_policy",
        "filesystem_policy",
        "network_policy",
        "resource_policy",
        "nondeterminism_policy",
        "unsupported_task_policy",
    ]:
        assert policy in receipt


def test_req_code_6186_metadata_normalization_edge_cases() -> None:
    """REQ-CODE-6186: stratification helpers normalize metadata deterministically."""
    assert mod._parse_metadata({"tags": ["graphs"]}) == {"tags": ["graphs"]}
    assert mod._parse_metadata("") == {}
    assert mod._parse_metadata("{bad json") == {}
    assert mod._stable_date(None) == "unknown"
    assert mod._date_bucket("not-a-date") == "date_unknown"
    assert mod._prompt_size_bucket(1501) == "medium"
    assert mod._prompt_size_bucket(3501) == "long"
    assert mod._metadata_tags({"tags": "dynamic-programming"}) == ["dynamic-programming"]
    assert mod._metadata_tags({"tags": object()}) == []
    assert mod._optional_version("definitely_missing_6186_module") == "unavailable"
