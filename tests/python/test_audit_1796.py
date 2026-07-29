import json
import os
import shutil

import pytest
from pathlib import Path

from carnot.paths import REPO_ROOT_ENV, repo_root
from scripts.audit_1796 import run_audit

# REQ-REPORT-1796: Findings Audit 1796
# The repository shall provide an audit script that verifies experiments in the .186 and .187 ranges.
# SCENARIO-REPORT-1796: Generating Audit Artifact

# The .186/.187-range artifacts whose corrigenda this audit is asserted to append.
# Named explicitly (rather than globbed) so that if one of them ever stops being
# copied into the sandbox the test fails loudly instead of silently asserting nothing.
#: The key ``run_audit()`` is expected to append to each target artifact.
_CORRIGENDUM_KEY = "corrigendum_2026_05_187_audit"

_CORRIGENDUM_TARGETS = (
    "experiment_1861_equivalence.json",
    "experiment_1862_e2e.json",
    "experiment_1864_roce.json",
    "experiment_1876_146_completion_147_gate_contract.json",
    "experiment_1877_artifact_contract_normalization.json",
)


@pytest.fixture()
def audit_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the audit against a COPY of the record, never the record itself.

    WHY THIS FIXTURE EXISTS
    -----------------------
    ``run_audit()`` does not merely read the .186/.187-range artifacts -- it appends a
    ``corrigendum_2026_05_187_audit`` key to each one and writes them back IN PLACE,
    then writes its own summary artifact. That is correct behaviour for an audit the
    operator runs deliberately. It is not acceptable behaviour for a unit test: every
    run of this test mutated tracked files under ``results/``, which are the committed
    research record rather than test output.

    Because the audit now resolves ``results/`` through ``carnot.paths`` instead of the
    current working directory, redirecting it is a single environment variable. We copy
    the real artifacts into a sandbox, point ``$CARNOT_REPO_ROOT`` at it, and let the
    audit mutate the copies -- so the test still exercises the genuine end-to-end
    behaviour (read, classify, append corrigenda, write summary) with none of the
    collateral damage.
    """
    real_results = repo_root(start=__file__) / "results"
    sandbox_results = tmp_path / "results"
    sandbox_results.mkdir(parents=True)

    # Copy exactly the input set the audit globs for, so the sandbox is a faithful
    # stand-in rather than an empty directory.
    copied = 0
    for pattern in ("experiment_186*.json", "experiment_187*.json"):
        for src in real_results.glob(pattern):
            shutil.copy2(src, sandbox_results / src.name)
            copied += 1
    assert copied, "precondition: the .186/.187 artifacts must exist to be audited"

    # Strip the corrigendum key from the COPIES so the test can actually fail.
    #
    # This audit has been run for real, so all five target artifacts on disk ALREADY
    # contain ``corrigendum_2026_05_187_audit``. Copying them verbatim meant the
    # copies inherited the key and ``assert key in payload`` passed no matter what --
    # it would have passed with run_audit() replaced by a no-op. Removing the key
    # first turns that assertion into a real test of "the audit ADDS the corrigendum".
    #
    # Only the sandbox copies are modified. The committed artifacts under results/ are
    # never written by this fixture (never-prune: the research record is read-only to
    # the test suite).
    for name in _CORRIGENDUM_TARGETS:
        target = sandbox_results / name
        payload = json.loads(target.read_text())
        payload.pop(_CORRIGENDUM_KEY, None)
        target.write_text(json.dumps(payload, indent=2))
        assert _CORRIGENDUM_KEY not in json.loads(target.read_text()), (
            f"precondition: {name} must start WITHOUT the corrigendum key, otherwise "
            "the post-run assertion cannot distinguish a working audit from a no-op"
        )

    monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))
    return sandbox_results


def test_audit_1796(audit_sandbox: Path):
    """
    Test that the audit_1796 script generates the correct output artifact and appends corrigenda.
    """
    run_audit()

    summary = audit_sandbox / "experiment_1796_findings_audit_186_187.json"
    assert os.path.exists(summary)
    with open(summary) as fp:
        data = json.load(fp)

    assert data["schema"] == "carnot.findings_audit_corrigenda.v3"
    assert data["experiment"] == 1796
    assert "audit_outcomes" in data
    assert "corrigenda_added" in data
    assert data["acceptance_gate_passed"] is True
    assert data["honest_verdict"].startswith("complete:")

    # Check that corrigenda was added. The fixture removed this key from every copy
    # beforehand, so its presence here is genuinely attributable to run_audit().
    for name in _CORRIGENDUM_TARGETS:
        with open(audit_sandbox / name) as fp:
            payload = json.load(fp)
        assert _CORRIGENDUM_KEY in payload, name


def test_audit_does_not_touch_the_real_results_directory(audit_sandbox: Path):
    """The audit must write ONLY inside the sandbox while the override is set.

    This is the regression test for the defect itself: without it, a future change that
    reintroduced a working-directory-relative or hardcoded path would go unnoticed
    because the test above would still pass on the sandbox copies.
    """
    # Derive the REAL results/ directory WITHOUT going through repo_root().
    #
    # This deliberately bypasses the resolver. The fixture has already set
    # $CARNOT_REPO_ROOT to the sandbox, and the override is absolute by design, so
    # repo_root() here would hand back the sandbox -- making this test compare the
    # sandbox against itself and pass no matter how badly the audit escaped. (That is
    # exactly what happened when this test was first written.) Walking up from this
    # file's own location is the one way to name the real tree while the override is
    # in force.
    real_results = Path(__file__).resolve().parents[2] / "results"
    assert real_results.is_dir(), "sanity: the real results/ directory must be found"

    watched = {
        name: (real_results / name).read_bytes()
        for name in _CORRIGENDUM_TARGETS
        if (real_results / name).is_file()
    }
    assert watched, "precondition: there must be real artifacts to protect"

    run_audit()

    for name, before in watched.items():
        assert (real_results / name).read_bytes() == before, (
            f"{name} in the real results/ directory was modified by run_audit(); "
            "the audit is escaping its sandbox again"
        )
