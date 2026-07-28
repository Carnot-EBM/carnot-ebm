import os
import json
import importlib.util
from pathlib import Path
import sys

import pytest

# Repo root, derived from this file's location rather than a hardcoded absolute
# path, so the test works from either of this environment's two working-directory
# aliases (one is a symlink to the other).
_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_experiment_3611_archive_v331_activate_v332(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    REQ-REPORT-3611: Archive V331 Activate V332
    SCENARIO-REPORT-3611: Exp 3611 Archives .331 and Activates .332
    """
    script_path = str(_REPO_ROOT / "scripts/experiment_3611_archive_v331_activate_v332.py")

    # SANDBOX THE SCRIPT'S WRITES.
    #
    # WHY: the script resolves BOTH its input (`research-complete.yaml`) and its
    # output (`results/experiment_3611_...json`) relative to the CURRENT WORKING
    # DIRECTORY. Run from the repo root -- which is how pytest runs -- `module.main()`
    # OVERWRITES the committed historical artifact with a fresh `duration_s` and
    # `reproducibility_checksum`. Worse, the original version of this test called
    # `os.remove()` on that artifact FIRST, so a failure mid-run would have deleted
    # a piece of the research record outright. Running the suite is not running the
    # experiment; the committed artifact records the ORIGINAL 2026-06 run and must
    # stay frozen (CLAUDE.md: "never rewrite a historical artifact").
    #
    # HOW: chdir into pytest's tmp_path, symlink the read-only input the script
    # needs, and give it an empty results/ to write into. Every assertion below is
    # unchanged and still exercises the real `main()` -- only the write LOCATION
    # moves, so no test coverage is lost.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    os.symlink(_REPO_ROOT / "research-complete.yaml", tmp_path / "research-complete.yaml")

    output_path = tmp_path / "results/experiment_3611_archive_v331_activate_v332.json"

    # Import the script
    spec = importlib.util.spec_from_file_location("experiment_3611", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["experiment_3611"] = module
    spec.loader.exec_module(module)

    # Run the function
    module.main()

    # Check the output file
    assert output_path.exists(), "JSON output file was not created"

    with open(output_path) as f:
        data = json.load(f)

    # Verify the required fields (principle-annotated fields)
    assert (
        data["honest_verdict"]
        == "complete: archived_v331_unfinished_decontamination_facts_code_blocked_not_measured_v332_active_paper_ready_true"
    )
    assert data["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert (
        data["v331_outcome_recorded_as"]
        == "UNFINISHED de-contamination (facts/code rows BLOCKED not measured)"
    )
    assert data["false_negative_risk_recorded"] == "asserted a null with no valid positive control"
    assert data["facts_corpus_exists_for_332"] is True
    assert data["paper_ready_preserved"] is True
    assert data["n_tasks_archived"] > 0, "n_tasks_archived should be greater than 0"
    assert "random_seed" in data
    assert "duration_s" in data
    assert "reproducibility_checksum" in data
