"""
Tests for REQ-PUBLISH-026: HuggingFace Publish Retry.

WHY THESE TESTS RUN IN A TEMPORARY DIRECTORY (do not remove the `_isolated_cwd` fixture)
----------------------------------------------------------------------------------------
`scripts/experiment_1750.py` resolves every path it writes RELATIVE TO THE CURRENT WORKING
DIRECTORY, not relative to the repository:

    models_dir      = Path("python/carnot/models")   # read
    model_card_path = Path("README.md")              # WRITTEN  <- fixed at source
    out_dir         = Path("results")                # WRITTEN

pytest's working directory is the repo root, so before 2026-07-29 both tests below silently
overwrote the operator's hand-written `README.md` with a six-line HuggingFace model card, and
overwrote the committed `results/experiment_1750_huggingface_retry.json`, on every suite run --
while passing. `README.md` is OPERATOR-CURATED under CLAUDE.md's "Public Documentation
Discipline", which forbids the autonomous loop from editing it at all.

TWO layers fix this, deliberately:

1. AT SOURCE -- the script now stages its model card in a temp directory, so it cannot
   write a README.md relative to the working directory no matter who runs it or from
   where. This is the real fix, and it also protects an operator running the script by
   hand, where no test fixture would be in play. Two sibling scripts had the identical
   bug and got the identical fix: `python/carnot/pipeline/hf_publisher.py` and
   `scripts/publish_huggingface.py`.
2. HERE -- the `_isolated_cwd` fixture still redirects the script's OTHER CWD-relative
   write (`results/`), and keeps this test honest if anyone reintroduces the pattern.

`monkeypatch.chdir(tmp_path)` fixes this WITHOUT weakening the tests: the real script is still
imported and executed end-to-end, every branch taken is the same branch as before, and only the
destination of its writes changes. The temporary `python/carnot/models` tree is created so the
script takes its PRIMARY path-discovery branch -- exactly the branch it took when the working
directory was the repo root -- rather than silently falling through to its `/tmp/models_mock`
fallback, which would have been a quiet loss of coverage.

`tests/python/conftest.py` installs an audit hook that now makes this class of bug fail loudly
instead of passing silently; see `python/carnot/testing/operator_curated_doc_guard.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import patch

from scripts.experiment_1750 import run_experiment


@pytest.fixture
def _isolated_cwd(tmp_path, monkeypatch):
    """Run the experiment against a throwaway tree instead of the real repository.

    Yields the temporary root so a test can assert on what the script wrote.
    """
    # Mirror the one input the script looks for, so it takes the same branch it takes when the
    # working directory is the repo root (which really does contain python/carnot/models).
    models_dir = tmp_path / "python" / "carnot" / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "smallest.pt").write_text("weights")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_experiment_1750_success(_isolated_cwd):
    with (
        patch("scripts.experiment_1750.HfApi") as MockApi,
        patch("scripts.experiment_1750.create_repo") as mock_create_repo,
    ):
        instance = MockApi.return_value
        instance.whoami.return_value = {"id": "mock_id"}

        deliverable = run_experiment()

        assert deliverable["hf_upload_succeeded"] is True
        assert deliverable["honest_verdict"] == "OK: Model published"
        mock_create_repo.assert_called_once()
        assert instance.upload_file.call_count == 2


def test_experiment_1750_blocked(_isolated_cwd):
    with patch("scripts.experiment_1750.HfApi") as MockApi:
        instance = MockApi.return_value
        instance.whoami.side_effect = Exception("Blocked credentials mock")

        deliverable = run_experiment()

        assert deliverable["hf_upload_succeeded"] is False
        assert deliverable["honest_verdict"] == "blocked_credentials"


def test_experiment_1750_never_writes_a_readme_next_to_the_working_directory(
    _isolated_cwd,
):
    """The regression test for the incident itself.

    The original bug was `model_card_path = Path("README.md")` -- CWD-relative -- so with
    pytest's working directory at the repo root, the script overwrote the operator's
    README.md on every run. The script now stages the card in a temp directory instead.

    This asserts the STRONGER post-fix property: no README.md appears next to the working
    directory at all. That is a better regression test than "a README.md appears in the
    sandbox", because the latter would still pass if someone reintroduced a CWD-relative
    write -- it only checked that the sandbox absorbed the damage, not that the damage
    stopped happening.
    """
    with (
        patch("scripts.experiment_1750.HfApi") as MockApi,
        patch("scripts.experiment_1750.create_repo"),
    ):
        MockApi.return_value.whoami.return_value = {"id": "mock_id"}
        run_experiment()

    assert not (_isolated_cwd / "README.md").exists(), (
        "the model card must be staged outside the working directory; a CWD-relative "
        "write here is what destroyed the operator's README.md"
    )
    # The result artifact IS legitimately repo/CWD-relative output, and must still land
    # in the sandbox rather than over the committed artifact.
    assert (_isolated_cwd / "results" / "experiment_1750_huggingface_retry.json").exists()


def test_experiment_1750_still_uploads_the_model_card(_isolated_cwd):
    """The fix must not have silently dropped the upload it was protecting.

    Moving the staging location is only safe if the same bytes still reach HuggingFace
    under the same name. This pins that: a file named README.md, containing the model
    card, uploaded to `path_in_repo="README.md"` -- from somewhere that is NOT the
    working directory.
    """
    with (
        patch("scripts.experiment_1750.HfApi") as MockApi,
        patch("scripts.experiment_1750.create_repo"),
    ):
        instance = MockApi.return_value
        instance.whoami.return_value = {"id": "mock_id"}
        run_experiment()

    card_uploads = [
        call
        for call in instance.upload_file.call_args_list
        if call.kwargs.get("path_in_repo") == "README.md"
    ]
    assert len(card_uploads) == 1, "the model card upload must still happen exactly once"

    staged = Path(card_uploads[0].kwargs["path_or_fileobj"])
    assert staged.name == "README.md"
    assert "Carnot Smallest Test Model" in staged.read_text()
    assert _isolated_cwd not in staged.parents, (
        f"model card staged at {staged}, which is inside the working directory"
    )
