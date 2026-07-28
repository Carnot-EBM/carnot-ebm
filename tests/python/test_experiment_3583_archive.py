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


def test_experiment_3583_archive_v329_activate_v330(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    REQ-REPORT-3583: Archive V329 Activate V330
    SCENARIO-REPORT-3583: Exp 3583 Archives .329 and Activates .330
    """
    script_path = str(_REPO_ROOT / "scripts/experiment_3583_archive_v329_activate_v330.py")

    # SANDBOX THE SCRIPT'S WRITES -- see the long rationale in
    # tests/python/test_experiment_3611.py, which has the identical defect.
    # In short: `run()` resolves its inputs (research-complete.yaml,
    # research-roadmap.yaml) and its output (results/experiment_3583_...json)
    # relative to the CURRENT WORKING DIRECTORY, so running the test suite from
    # the repo root silently OVERWROTE a committed historical artifact with a
    # fresh duration_s and reproducibility_checksum. Running the suite is not
    # running the experiment. chdir into tmp_path, symlink the read-only inputs,
    # and let the script write there instead. Assertions are unchanged.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()

    # SEED THE INPUTS DETERMINISTICALLY instead of symlinking the live repo files.
    #
    # WHY: `run()` counts the tasks recorded under milestone 2026.05.329 in
    # research-complete.yaml. That file is ROTATED as the project advances -- it no
    # longer contains .329 at all (its oldest entry is now 2026.07.5xx), so against
    # the live file the count is 0 and the historical `n_tasks_archived > 0`
    # assertion below could never pass again. That made this test a standing
    # failure whose only observable effect was rewriting the artifact in results/.
    # Seeding a fixture pins the input, so the assertion tests the script's COUNTING
    # LOGIC (the thing this test is actually for) rather than the mutable contents of
    # a rotating ops file. The committed artifact records the real run's value (0,
    # from the already-rotated file) and is left untouched.
    (tmp_path / "research-complete.yaml").write_text(
        "milestones:\n"
        "- id: '2026.05.328'\n"
        "  tasks:\n"
        "  - id: exp3570\n"
        "- id: '2026.05.329'\n"
        "  tasks:\n"
        "  - id: exp3571\n"
        "  - id: exp3572\n"
        "  - id: exp3573\n"
        "- id: '2026.05.330'\n"
        "  tasks:\n"
        "  - id: exp3580\n"
    )
    (tmp_path / "research-roadmap.yaml").write_text("milestone: '2026.05.330'\n")
    expected_n_tasks_archived = 3

    output_path = tmp_path / "results/experiment_3583_archive_v329_activate_v330.json"

    # Import the script
    spec = importlib.util.spec_from_file_location("experiment_3583", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["experiment_3583"] = module
    spec.loader.exec_module(module)

    # Run the function
    module.run()

    # Check the output file
    assert output_path.exists(), "JSON output file was not created"

    with open(output_path) as f:
        data = json.load(f)

    # Verify the required fields (principle-annotated fields)
    assert (
        data["honest_verdict"]
        == "complete: archived_v329_contaminated_null_recorded_v330_decontamination_pivot_active"
    )
    assert data["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert data["v329_headline_recorded_as"] == "contaminated_null_not_clean_math_only"
    assert data["paper_ready_preserved"] is True
    assert data["n_tasks_archived"] > 0, "n_tasks_archived should be greater than 0"
    assert data["n_tasks_archived"] == expected_n_tasks_archived, (
        "run() must count exactly the tasks under milestone 2026.05.329 in the "
        "seeded research-complete.yaml -- not those of the neighbouring milestones"
    )
    assert "random_seed" in data
    assert "duration_s" in data
    assert "reproducibility_checksum" in data
