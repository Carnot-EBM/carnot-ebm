"""Tests for carnot.eval.fover_g2_package (exp 3476).

Spec: REQ-PUBLISH-038, SCENARIO-PUBLISH-038, SCENARIO-PUBLISH-038B

These tests exercise the self-contained-package builder + clean-room verifier.
Heavy external dependencies (Docker, IPFS) are stubbed via injected ``runner``
callables, so the suite is fast and hermetic and never shells out. The
package-tree build and tar round-trip run against a synthetic mini-repo fixture.
"""

from __future__ import annotations

import tarfile
import types
from pathlib import Path

import pytest

import carnot.eval.fover_g2_package as pkg
from carnot.eval.fover_g2_package import (
    CONDITION_A_CI_HIGH,
    CONDITION_A_CI_LOW,
    LEARNING_CONTRIB_CI_HIGH,
    LEARNING_CONTRIB_CI_LOW,
    PACKAGE_NAME,
    _dep_name,
    _safe_extractall,
    build_artifact,
    build_package_readme,
    build_package_tree,
    build_requirements_txt,
    build_run_sh,
    check_preconditions,
    classify_ci,
    determine_verdict_and_status,
    docker_is_available,
    make_tarball,
    maybe_ipfs_add,
    parse_harness_numbers,
    read_pyproject_dependencies,
    sha256_of_file,
    verify_package_in_docker,
)


# ---------------------------------------------------------------------------
# Fixtures: a synthetic mini-repo with exactly what build_package_tree needs
# ---------------------------------------------------------------------------


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    """Create a minimal repo tree: pyproject, package source, harness, corpus."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "carnot-ebm"\n'
        'dependencies = [\n  "numpy>=1.26",\n  "z3-solver>=4.16",\n'
        '  "scikit-learn>=1.4",\n]\n',
        encoding="utf-8",
    )
    (tmp_path / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
    # Package source with a .pyc that must be excluded.
    src = tmp_path / "python" / "carnot"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("# carnot\n", encoding="utf-8")
    (src / "stale.pyc").write_text("x", encoding="utf-8")
    # Harness + corpus.
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "reproduce_fover_headline.py").write_text(
        "print('harness')\n", encoding="utf-8"
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "fover_corpus.jsonl").write_text(
        '{"a": 1}\n{"a": 2}\n', encoding="utf-8"
    )
    # One FR-11 state file (matches the data/fr11_*.jsonl glob).
    (tmp_path / "data" / "fr11_state.jsonl").write_text('{"s": 1}\n', encoding="utf-8")
    # A directory matching the same glob — exercises the non-file skip branch.
    (tmp_path / "data" / "fr11_dir.jsonl").mkdir()
    return tmp_path


def _fake_proc(returncode: int = 0, stdout: str = "", stderr: str = ""):
    """A minimal stand-in for subprocess.CompletedProcess."""
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# _dep_name + pyproject parsing + requirements pinning
# ---------------------------------------------------------------------------


def test_dep_name_strips_specifiers_and_extras():
    """SCENARIO-PUBLISH-038: dependency name extraction handles extras/markers."""
    assert _dep_name("scikit-learn>=1.4") == "scikit-learn"
    assert _dep_name("jax[cuda]>=0.4.30") == "jax"
    assert _dep_name("numpy>=1.26 ; python_version>='3.11'") == "numpy"
    assert _dep_name("z3-solver") == "z3-solver"


def test_read_pyproject_dependencies(mini_repo: Path):
    """SCENARIO-PUBLISH-038: dependencies are read verbatim from pyproject."""
    deps = read_pyproject_dependencies(mini_repo)
    assert "numpy>=1.26" in deps
    assert "scikit-learn>=1.4" in deps


def test_build_requirements_txt_pins_installed_and_falls_back(mini_repo: Path):
    """SCENARIO-PUBLISH-038: exact pins when installed; pyproject range otherwise."""

    def fake_version(name: str) -> str:
        if name == "numpy":
            return "1.26.4"
        if name == "scikit-learn":
            return "1.5.0"
        raise pkg.importlib_metadata.PackageNotFoundError(name)

    txt = build_requirements_txt(mini_repo, version_lookup=fake_version)
    assert "numpy==1.26.4" in txt
    assert "scikit-learn==1.5.0" in txt
    # z3-solver not "installed" -> fall back to the pyproject specifier verbatim.
    assert "z3-solver>=4.16" in txt
    # Header comments present.
    assert txt.startswith("# Pinned dependency set")


def test_build_requirements_txt_default_lookup(mini_repo: Path):
    """SCENARIO-PUBLISH-038: default version_lookup path runs without error."""
    txt = build_requirements_txt(mini_repo)
    # numpy is installed in the test env, so it should pin (== present somewhere).
    assert "numpy" in txt


# ---------------------------------------------------------------------------
# run.sh + README content
# ---------------------------------------------------------------------------


def test_build_run_sh_installs_pinned_and_runs_harness():
    """SCENARIO-PUBLISH-038: run.sh pins deps, no-deps installs pkg, runs harness."""
    sh = build_run_sh()
    assert sh.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in sh
    assert "-r requirements.txt" in sh
    assert "--no-deps -e ." in sh
    assert "scripts/reproduce_fover_headline.py" in sh
    assert "JAX_PLATFORMS=cpu" in sh


def test_build_package_readme_includes_command_and_cis():
    """SCENARIO-PUBLISH-038: README states the one command + the published CIs."""
    readme = build_package_readme("deadbeef", package_sha256="cafef00d")
    assert "bash run.sh" in readme
    assert "[0.9027, 0.9235]" in readme
    assert "[0.0125, 0.0245]" in readme
    assert "deadbeef" in readme
    assert "cafef00d" in readme


def test_build_package_readme_without_package_sha():
    """SCENARIO-PUBLISH-038: README omits the package-sha line when absent."""
    readme = build_package_readme("deadbeef")
    assert "This package sha256" not in readme
    assert "deadbeef" in readme


# ---------------------------------------------------------------------------
# sha256 + package-tree build + tar round-trip
# ---------------------------------------------------------------------------


def test_sha256_of_file(tmp_path: Path):
    """SCENARIO-PUBLISH-038: sha256 is the 64-hex digest of the file content."""
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    digest = sha256_of_file(f)
    # sha256("abc")
    assert digest == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_build_package_tree_assembles_all_parts(mini_repo: Path):
    """SCENARIO-PUBLISH-038: the package tree contains every stranger-facing part."""
    pkg_dir = mini_repo / "dist" / PACKAGE_NAME

    def fake_version(name: str) -> str:
        return "9.9.9"

    manifest = build_package_tree(mini_repo, pkg_dir, version_lookup=fake_version)

    assert (pkg_dir / "pyproject.toml").exists()
    assert (pkg_dir / "LICENSE").exists()
    assert (pkg_dir / "python" / "carnot" / "__init__.py").exists()
    # .pyc excluded by ignore_patterns
    assert not (pkg_dir / "python" / "carnot" / "stale.pyc").exists()
    assert (pkg_dir / "scripts" / "reproduce_fover_headline.py").exists()
    assert (pkg_dir / "data" / "fover_corpus.jsonl").exists()
    assert (pkg_dir / "requirements.txt").exists()
    assert (pkg_dir / "README.md").exists()
    run_sh = pkg_dir / "run.sh"
    assert run_sh.exists()
    # run.sh must be executable.
    assert run_sh.stat().st_mode & 0o111
    # FR-11 state file packaged for condition A.
    assert (pkg_dir / "data" / "fr11_state.jsonl").exists()
    assert manifest["state_files_copied"] == 1
    assert len(manifest["corpus_sha256"]) == 64


def test_make_tarball_and_safe_extract_round_trip(mini_repo: Path):
    """SCENARIO-PUBLISH-038: tar round-trips under the package-name top dir."""
    pkg_dir = mini_repo / "dist" / PACKAGE_NAME
    build_package_tree(mini_repo, pkg_dir, version_lookup=lambda n: "1.0.0")
    tar_path = mini_repo / "dist" / "g2-fover-repro.tar.gz"
    make_tarball(pkg_dir, tar_path)
    assert tar_path.exists()

    out = mini_repo / "extracted"
    out.mkdir()
    with tarfile.open(tar_path, "r:gz") as tar:
        _safe_extractall(tar, out)
    assert (out / PACKAGE_NAME / "run.sh").exists()
    assert (out / PACKAGE_NAME / "data" / "fover_corpus.jsonl").exists()


def test_safe_extractall_rejects_path_traversal(tmp_path: Path):
    """SCENARIO-PUBLISH-038: a tar member escaping dest is refused."""
    evil_tar = tmp_path / "evil.tar.gz"
    payload = tmp_path / "payload"
    payload.write_text("x", encoding="utf-8")
    with tarfile.open(evil_tar, "w:gz") as tar:
        tar.add(payload, arcname="../escape.txt")
    dest = tmp_path / "dest"
    dest.mkdir()
    with tarfile.open(evil_tar, "r:gz") as tar:
        with pytest.raises(ValueError, match="unsafe tar member"):
            _safe_extractall(tar, dest)


# ---------------------------------------------------------------------------
# IPFS (stubbed runner)
# ---------------------------------------------------------------------------


def test_maybe_ipfs_add_no_binary(monkeypatch, tmp_path: Path):
    """SCENARIO-PUBLISH-038B: no ipfs binary -> ipfs_available False, no failure."""
    monkeypatch.setattr(pkg.shutil, "which", lambda name: None)
    res = maybe_ipfs_add(tmp_path / "x.tar.gz")
    assert res == {"ipfs_available": False, "package_cid": None}


def test_maybe_ipfs_add_real_add_succeeds(monkeypatch, tmp_path: Path):
    """SCENARIO-PUBLISH-038: a working ipfs add records the CID."""
    monkeypatch.setattr(pkg.shutil, "which", lambda name: "/usr/bin/ipfs")

    def runner(args, **kwargs):
        assert args[:3] == ["ipfs", "add", "-Q"]
        return _fake_proc(0, stdout="QmFakeCID123\n")

    res = maybe_ipfs_add(tmp_path / "x.tar.gz", runner=runner)
    assert res == {"ipfs_available": True, "package_cid": "QmFakeCID123"}


def test_maybe_ipfs_add_falls_back_to_only_hash(monkeypatch, tmp_path: Path):
    """SCENARIO-PUBLISH-038: real add fails -> --only-hash computes the CID."""
    monkeypatch.setattr(pkg.shutil, "which", lambda name: "/usr/bin/ipfs")
    calls = []

    def runner(args, **kwargs):
        calls.append(args)
        if "--only-hash" in args:
            return _fake_proc(0, stdout="QmOnlyHash\n")
        return _fake_proc(1, stderr="no daemon")

    res = maybe_ipfs_add(tmp_path / "x.tar.gz", runner=runner)
    assert res == {"ipfs_available": True, "package_cid": "QmOnlyHash"}
    assert len(calls) == 2


def test_maybe_ipfs_add_all_fail(monkeypatch, tmp_path: Path):
    """SCENARIO-PUBLISH-038B: both ipfs attempts fail -> available False."""
    monkeypatch.setattr(pkg.shutil, "which", lambda name: "/usr/bin/ipfs")

    def runner(args, **kwargs):
        raise OSError("boom")

    res = maybe_ipfs_add(tmp_path / "x.tar.gz", runner=runner)
    assert res == {"ipfs_available": False, "package_cid": None}


# ---------------------------------------------------------------------------
# docker availability + harness parsing + CI classification
# ---------------------------------------------------------------------------


def test_docker_is_available_true(monkeypatch):
    """SCENARIO-PUBLISH-038: docker present + info exits 0 -> available."""
    monkeypatch.setattr(pkg.shutil, "which", lambda name: "/usr/bin/docker")
    assert docker_is_available(runner=lambda *a, **k: _fake_proc(0)) is True


def test_docker_is_available_no_binary(monkeypatch):
    """SCENARIO-PUBLISH-038B: no docker binary -> not available."""
    monkeypatch.setattr(pkg.shutil, "which", lambda name: None)
    assert docker_is_available() is False


def test_docker_is_available_daemon_down(monkeypatch):
    """SCENARIO-PUBLISH-038B: client present but daemon down -> not available."""
    monkeypatch.setattr(pkg.shutil, "which", lambda name: "/usr/bin/docker")

    def runner(*a, **k):
        raise OSError("no daemon")

    assert docker_is_available(runner=runner) is False


def test_parse_harness_numbers():
    """SCENARIO-PUBLISH-038: harness stdout floats are parsed correctly."""
    stdout = (
        "condition A (production)        mean AUROC: 0.9131\n"
        "condition B (architecture-only) mean AUROC: 0.8947\n"
        "learning contribution:                      0.0185\n"
        "reproducibility_checksum:                   abc123\n"
    )
    cond_a, lc = parse_harness_numbers(stdout)
    assert cond_a == 0.9131
    assert lc == 0.0185


def test_parse_harness_numbers_missing():
    """SCENARIO-PUBLISH-038B: absent lines yield None (e.g. on error stdout)."""
    cond_a, lc = parse_harness_numbers("some unrelated error output\n")
    assert cond_a is None
    assert lc is None


def test_classify_ci_both_in():
    """SCENARIO-PUBLISH-038: both numbers in CI -> reproduced True."""
    a_in, lc_in, repro = classify_ci(0.9131, 0.0185)
    assert (a_in, lc_in, repro) == (True, True, True)


def test_classify_ci_one_out():
    """SCENARIO-PUBLISH-038B: one number out of CI -> reproduced False."""
    _, _, repro = classify_ci(0.9131, 0.05)  # lc above the band
    assert repro is False
    _, _, repro2 = classify_ci(0.80, 0.0185)  # cond_a below the band
    assert repro2 is False


def test_classify_ci_none():
    """SCENARIO-PUBLISH-038B: None (error) is never in any CI."""
    a_in, lc_in, repro = classify_ci(None, None)
    assert (a_in, lc_in, repro) == (False, False, False)


def test_ci_constants_sane():
    """Guard: the CI bands are ordered low < high."""
    assert CONDITION_A_CI_LOW < CONDITION_A_CI_HIGH
    assert LEARNING_CONTRIB_CI_LOW < LEARNING_CONTRIB_CI_HIGH


# ---------------------------------------------------------------------------
# verify_package_in_docker (real tar extract + stubbed docker runner)
# ---------------------------------------------------------------------------


def test_verify_package_in_docker_success(mini_repo: Path):
    """SCENARIO-PUBLISH-038: extract + docker run exit 0 surfaces the numbers."""
    pkg_dir = mini_repo / "dist" / PACKAGE_NAME
    build_package_tree(mini_repo, pkg_dir, version_lookup=lambda n: "1.0.0")
    tar_path = mini_repo / "dist" / "g2-fover-repro.tar.gz"
    make_tarball(pkg_dir, tar_path)

    captured = {}

    def runner(args, **kwargs):
        captured["args"] = args
        return _fake_proc(
            0,
            stdout=(
                "condition A (production)        mean AUROC: 0.9131\n"
                "learning contribution:                      0.0185\n"
            ),
        )

    result = verify_package_in_docker(tar_path, runner=runner)
    assert result["exit_code"] == 0
    assert "docker" in captured["args"][0]
    assert "python:3.12-slim" in captured["args"]
    cond_a, lc = parse_harness_numbers(result["stdout"])
    assert cond_a == 0.9131 and lc == 0.0185


def test_verify_package_in_docker_missing_run_sh(tmp_path: Path):
    """SCENARIO-PUBLISH-038B: a tarball lacking run.sh reports an error."""
    # Build a tarball whose top dir lacks run.sh.
    bogus = tmp_path / PACKAGE_NAME
    bogus.mkdir()
    (bogus / "other.txt").write_text("x", encoding="utf-8")
    tar_path = tmp_path / "g2-fover-repro.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(bogus, arcname=PACKAGE_NAME)

    result = verify_package_in_docker(tar_path, runner=lambda *a, **k: _fake_proc(0))
    assert result["error"] == "package_missing_run_sh"
    assert result["exit_code"] is None


# ---------------------------------------------------------------------------
# verdict/status mapping + artifact assembly
# ---------------------------------------------------------------------------


def test_determine_verdict_reproduced():
    """SCENARIO-PUBLISH-038: verified package -> verified verdict + status."""
    verdict, status = determine_verdict_and_status(True, "docker", True, True)
    assert verdict.startswith("complete:")
    assert status == "self_contained_package_verified_external_run_pending"


def test_determine_verdict_built_unverified():
    """SCENARIO-PUBLISH-038B: built but no clean env -> built-unverified."""
    verdict, status = determine_verdict_and_status(True, None, False, False)
    assert status == "package_built_verification_unavailable"
    assert verdict.startswith("complete:")


def test_determine_verdict_clean_env_out_of_ci():
    """SCENARIO-PUBLISH-038B: verification attempted but out of CI -> failing."""
    verdict, status = determine_verdict_and_status(True, "docker", False, True)
    assert status == "still_failing_clean_env_out_of_ci"


def test_determine_verdict_build_failed():
    """SCENARIO-PUBLISH-038B: package not built -> build-failed verdict."""
    verdict, status = determine_verdict_and_status(False, None, False, False)
    assert status == "still_failing_build_failed"


def _artifact_kwargs(**over):
    base = dict(
        start_time=0.0,
        preconditions={"ok": True},
        package_path="dist/g2-fover-repro.tar.gz",
        package_sha256="abc",
        ipfs_result={"ipfs_available": True, "package_cid": "QmX"},
        clean_env_method="docker",
        cond_a=0.9131,
        lc=0.0185,
        verification_attempted=True,
        isolated_checksum="chk",
        manifest={"corpus_sha256": "c" * 64, "state_files_copied": 3},
        clock=lambda: 12.0,
    )
    base.update(over)
    return base


def test_build_artifact_verified_has_all_required_fields():
    """SCENARIO-PUBLISH-038: artifact carries every required schema field."""
    art = build_artifact(**_artifact_kwargs())
    required = {
        "honest_verdict", "inference_substrate", "package_path", "package_sha256",
        "package_cid", "ipfs_available", "one_command_repro", "clean_env_method",
        "condition_a_auroc_isolated", "learning_contribution_isolated",
        "package_verified_reproduces", "g2_status", "g2_independent_reproducer",
        "operator_action_required", "reproducibility_checksum", "random_seed",
        "duration_s",
    }
    assert required.issubset(art.keys())
    assert art["honest_verdict"].startswith("complete:")
    assert art["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert art["package_verified_reproduces"] is True
    assert art["g2_independent_reproducer"] is False
    assert art["package_cid"] == "QmX"
    assert art["duration_s"] == 12.0
    assert art["random_seed"] == [42, 137, 271, 314, 1729]
    # Every required field carries a principle annotation.
    for key in ("package_sha256", "g2_independent_reproducer", "one_command_repro"):
        assert key in art["field_principles"]


def test_build_artifact_never_sets_independent_reproducer():
    """SCENARIO-PUBLISH-038: even a reproduced package keeps the gate flag false."""
    art = build_artifact(**_artifact_kwargs(cond_a=0.9131, lc=0.0185))
    assert art["g2_independent_reproducer"] is False


def test_build_artifact_built_unverified_status():
    """SCENARIO-PUBLISH-038B: no clean env -> built-unverified status, repro False."""
    art = build_artifact(
        **_artifact_kwargs(
            clean_env_method=None,
            cond_a=None,
            lc=None,
            verification_attempted=False,
            isolated_checksum=None,
        )
    )
    assert art["package_verified_reproduces"] is False
    assert art["g2_status"] == "package_built_verification_unavailable"
    # reproducibility_checksum falls back to the package sha256.
    assert art["reproducibility_checksum"] == "abc"


# ---------------------------------------------------------------------------
# preconditions
# ---------------------------------------------------------------------------


def test_check_preconditions_ok(mini_repo: Path):
    """SCENARIO-PUBLISH-038: harness + corpus present -> ok."""
    res = check_preconditions(mini_repo)
    assert res["ok"] is True


def test_check_preconditions_missing(tmp_path: Path):
    """SCENARIO-PUBLISH-038B: missing harness/corpus -> blocked reason."""
    res = check_preconditions(tmp_path)
    assert res["ok"] is False
    assert res["blocked_reason"] == "blocked_fover_harness_or_corpus_missing"
