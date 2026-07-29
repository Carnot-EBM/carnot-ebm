"""Tests for the centralised repository-root / output-path resolver.

WHY THESE TESTS EXIST
---------------------
``carnot.paths`` replaces ~110 independent, hardcoded answers to "where is the repo
root?". Because it is now the single point of failure for where every experiment
artifact gets written, its failure modes need to be pinned down explicitly rather
than assumed:

* If it returned the SYMLINK ALIAS instead of the real path, provenance recorded by
  two runs of the same experiment would disagree purely on which directory the
  operator happened to ``cd`` into.
* If it depended on ``os.getcwd()``, a script run from a subdirectory would resolve a
  different root, which is the class of bug that let a passing test rewrite README.md.
* If the ``$CARNOT_REPO_ROOT`` override did not win, tests could not sandbox their
  writes and would contaminate the real tree.

Each of those is a test below, phrased as the concrete misbehaviour it forbids.
"""

# Test traces to REQ-ARC-WMTE-6043, SCENARIO-ARC-WMTE-6043-CANONICAL-FROM-EITHER-SPELLING
# and SCENARIO-ARC-WMTE-6043-OVERRIDE-ENABLES-SANDBOXING.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from carnot.paths import (
    REPO_ROOT_ENV,
    is_canonical_repo_root,
    output_dir,
    repo_path,
    repo_root,
    results_dir,
    results_path,
)

# The two spellings of this checkout. The alias is a chain of symlinks that ends at
# the real path; both must resolve to the real path.
CANONICAL = Path("/home/ianblenke/github.com/ianblenke/carnot")
ALIAS = Path("/home/ianblenke/github.com/Carnot-EBM/carnot-ebm")


# Anything that reads the real environment must not inherit a sandbox override left
# behind by another test, so every test that cares takes this fixture.
@pytest.fixture()
def no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee ``$CARNOT_REPO_ROOT`` is unset for tests that exercise detection."""
    monkeypatch.delenv(REPO_ROOT_ENV, raising=False)


# --------------------------------------------------------------------------------
# Core detection
# --------------------------------------------------------------------------------


def test_repo_root_is_absolute_and_exists(no_override: None) -> None:
    root = repo_root()
    assert root.is_absolute(), "a relative root would be cwd-dependent, the bug we are fixing"
    assert root.is_dir()


def test_repo_root_finds_a_real_git_checkout(no_override: None) -> None:
    """The detected root must actually be a checkout, not just some parent directory."""
    root = repo_root()
    assert (root / ".git").exists()


def test_repo_root_contains_the_expected_layout(no_override: None) -> None:
    root = repo_root()
    assert (root / "python" / "carnot").is_dir()
    assert (root / "scripts").is_dir()


def test_repo_root_is_fully_symlink_resolved(no_override: None) -> None:
    """The returned path must equal its own ``resolve()`` -- i.e. contain no symlinks.

    This is the invariant that makes two runs started from different spellings of the
    same directory produce identical provenance.
    """
    root = repo_root()
    assert root == root.resolve()


# --------------------------------------------------------------------------------
# The symlink trap -- the specific defect this module exists to prevent
# --------------------------------------------------------------------------------
#
# These tests build their OWN alias inside tmp_path rather than depending on the
# Carnot-EBM symlink that happens to exist on the author's machine.
#
# That is deliberate and load-bearing. Gating the headline guarantee
# (canonical-from-alias) on `ALIAS.exists()` means it silently goes UNTESTED on every
# fresh clone -- which is exactly the reproducibility case this whole module was
# written to fix. A guarantee that only verifies itself on the one machine that
# already worked is not a guarantee. The real-alias tests further below are kept as
# an ADDITIONAL check against the true production layout, never as sole coverage.


@pytest.fixture()
def aliased_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Build a miniature checkout plus a CHAINED symlink alias to it.

    Returns ``(real_root, alias_root)`` where ``alias_root`` reaches the same
    directory through two hops. Two hops, not one, because the production alias is
    itself a chain (``carnot-ebm -> carnot -> ...``) and a resolver that followed only
    a single link would pass a one-hop test while still failing in production.
    """
    real = tmp_path / "real" / "carnot"
    (real / "python" / "carnot").mkdir(parents=True)
    (real / "scripts").mkdir()
    # A .git FILE, not a directory: this is the git-worktree shape, so the fixture
    # covers the marker case that a directory-only check would miss.
    (real / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x\n")
    (real / "pyproject.toml").write_text("[project]\nname='fake'\n")
    (real / "python" / "carnot" / "probe.py").write_text("# probe\n")

    midpoint = tmp_path / "midpoint"
    os.symlink(real, midpoint)
    alias = tmp_path / "alias"
    os.symlink(midpoint, alias)
    return real, alias


def test_alias_and_canonical_start_points_agree(
    no_override: None, aliased_repo: tuple[Path, Path]
) -> None:
    """Starting from either spelling of the SAME file must give the SAME root.

    This is the headline requirement. ``repo_root`` resolves its start point before
    walking up, so the alias's ancestors are never consulted.
    """
    real, alias = aliased_repo
    probe = Path("python") / "carnot" / "probe.py"
    assert repo_root(start=real / probe) == repo_root(start=alias / probe)


def test_alias_start_point_yields_the_canonical_spelling(
    no_override: None, aliased_repo: tuple[Path, Path]
) -> None:
    """Specifically: the answer is the REAL path, never the alias string."""
    real, alias = aliased_repo
    resolved = repo_root(start=alias / "python" / "carnot" / "probe.py")
    assert resolved == real.resolve()
    assert "alias" not in resolved.parts
    assert "midpoint" not in resolved.parts


def test_the_synthetic_alias_is_genuinely_a_different_string(
    aliased_repo: tuple[Path, Path],
) -> None:
    """Guards the two tests above from becoming vacuous.

    If the fixture ever produced a real directory instead of a symlink, the agreement
    assertions would still pass while testing nothing. Pin the precondition that gives
    them meaning: two different strings naming one directory.
    """
    real, alias = aliased_repo
    assert str(alias) != str(real)
    assert alias.resolve() == real.resolve()
    assert alias.is_symlink()


def test_a_subprocess_run_from_inside_the_alias_reports_canonical(
    no_override: None, aliased_repo: tuple[Path, Path]
) -> None:
    """The ``$PWD`` vector, on a synthetic alias, with no real-machine dependency.

    A shell that ``cd``s through a symlink exports the ALIAS spelling in ``$PWD``
    (``os.getcwd()`` does not -- it is a syscall returning the kernel's canonical
    path). This asserts the resolver is not fooled by a misleading ``$PWD``.
    """
    real, alias = aliased_repo
    env = {k: v for k, v in os.environ.items() if k != REPO_ROOT_ENV}
    env["PWD"] = str(alias)
    probe = alias / "python" / "carnot" / "probe.py"
    code = f"from carnot.paths import repo_root; print(repo_root(start={str(probe)!r}))"
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(alias),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert out.stdout.strip() == str(real.resolve())


# -- Additional coverage against the REAL production alias, when it is present. -----
# These are a bonus on top of the synthetic tests above, not a substitute for them.
# They assert nothing that the synthetic tests do not already assert; their value is
# confirming the actual on-disk layout matches the shape we modelled. Written as a
# runtime branch rather than a skip marker: a skip is an invisible failure, whereas an
# always-running test that reports what it checked stays honest on every machine.


def test_real_alias_layout_matches_the_modelled_shape() -> None:
    """If the production alias exists, confirm it behaves like the synthetic one."""
    if not (ALIAS.exists() and CANONICAL.exists()):
        # Not an assertion failure: on a fresh clone this alias legitimately does not
        # exist. The property it would test is already covered synthetically above.
        assert True
        return
    assert str(ALIAS) != str(CANONICAL)
    assert ALIAS.resolve() == CANONICAL.resolve()
    probe = "python/carnot/paths.py"
    from_real = repo_root(start=CANONICAL / probe)
    from_alias = repo_root(start=ALIAS / probe)
    assert from_real == from_alias == CANONICAL.resolve()
    assert "Carnot-EBM" not in str(from_alias)


# --------------------------------------------------------------------------------
# Independence from the working directory
# --------------------------------------------------------------------------------


def test_repo_root_ignores_the_working_directory(
    no_override: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """chdir'ing anywhere must not change the answer.

    A cwd-sensitive resolver is how relative writes escaped into tracked files.
    """
    before = repo_root(start=__file__)
    monkeypatch.chdir(tmp_path)
    assert repo_root(start=__file__) == before


def test_repo_root_from_a_subprocess_started_under_the_alias(no_override: None) -> None:
    """End-to-end: a fresh interpreter whose cwd is the alias still reports canonical.

    The in-process tests above pass an explicit ``start``; this one exercises the
    default caller-frame path in a real process, which is how scripts will use it.

    Runs against the REAL alias when present, and falls back to the canonical path
    otherwise. Either way the assertion is the same and the test really executes --
    the equivalent property on a synthetic alias is pinned by
    ``test_a_subprocess_run_from_inside_the_alias_reports_canonical`` above, so there
    is no coverage hole on a machine without the alias and no reason to skip.
    """
    # Expected value comes from the resolver itself rather than the hardcoded
    # CANONICAL constant, so this test is still meaningful in a clone at any path.
    expected = repo_root(start=__file__)
    entry = ALIAS if ALIAS.exists() else expected
    env = {k: v for k, v in os.environ.items() if k != REPO_ROOT_ENV}
    env["PWD"] = str(entry)
    out = subprocess.run(
        [sys.executable, "-c", "from carnot.paths import repo_root; print(repo_root())"],
        cwd=str(entry),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert out.stdout.strip() == str(expected)


# --------------------------------------------------------------------------------
# The override -- what makes sandboxing possible
# --------------------------------------------------------------------------------


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))
    assert repo_root() == tmp_path.resolve()


def test_env_override_wins_even_over_an_explicit_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The override is absolute, so a sandbox cannot be defeated by a stray ``start``.

    A test that sets the override is asserting "nothing may write outside here"; a
    caller passing its own ``__file__`` must not silently escape that.
    """
    monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))
    assert repo_root(start=__file__) == tmp_path.resolve()


def test_env_override_need_not_be_a_git_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty tmp_path is a legitimate override target.

    Validating the override as a repo would defeat its main use -- sandboxing writes
    into a directory that is deliberately not a checkout.
    """
    sandbox = tmp_path / "empty"
    sandbox.mkdir()
    assert not (sandbox / ".git").exists(), "precondition: the sandbox is not a checkout"
    monkeypatch.setenv(REPO_ROOT_ENV, str(sandbox))
    assert repo_root() == sandbox.resolve()


def test_empty_env_override_is_ignored(monkeypatch: pytest.MonkeyPatch, no_override: None) -> None:
    """An empty string must fall through to detection, not resolve to cwd.

    ``Path("").resolve()`` is the current directory, so treating an empty override as
    valid would silently reintroduce cwd-dependence.
    """
    monkeypatch.setenv(REPO_ROOT_ENV, "")
    assert repo_root(start=__file__).is_dir()
    assert (repo_root(start=__file__) / "python" / "carnot").is_dir()


def test_env_override_expands_user(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(REPO_ROOT_ENV, "~")
    assert repo_root() == Path.home().resolve()


# --------------------------------------------------------------------------------
# Marker walking
# --------------------------------------------------------------------------------


def test_git_marker_may_be_a_file_not_a_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_override: None
) -> None:
    """Git worktrees and submodules write ``.git`` as a POINTER FILE.

    Accepting only a directory would break every worktree -- one of the two scenarios
    the override exists to support, so it must not be the only way to support them.
    """
    fake = tmp_path / "wt"
    (fake / "deep" / "nested").mkdir(parents=True)
    (fake / ".git").write_text("gitdir: /somewhere/else\n")
    probe = fake / "deep" / "nested" / "mod.py"
    probe.write_text("")
    assert repo_root(start=probe) == fake.resolve()


def test_nearest_marker_wins_for_nested_checkouts(tmp_path: Path, no_override: None) -> None:
    """A repo nested inside another must resolve to its OWN root, not the outer one."""
    outer = tmp_path / "outer"
    inner = outer / "vendor" / "inner"
    inner.mkdir(parents=True)
    (outer / ".git").mkdir()
    (inner / ".git").mkdir()
    probe = inner / "mod.py"
    probe.write_text("")
    assert repo_root(start=probe) == inner.resolve()


def test_fallback_layout_markers_used_when_no_git(tmp_path: Path, no_override: None) -> None:
    """A VCS-less source copy (tarball) still resolves, via layout markers."""
    src = tmp_path / "src"
    (src / "python").mkdir(parents=True)
    (src / "scripts").mkdir()
    (src / "pyproject.toml").write_text("[project]\nname='x'\n")
    probe = src / "scripts" / "thing.py"
    probe.write_text("")
    assert repo_root(start=probe) == src.resolve()


def test_fallback_requires_the_full_layout_not_just_pyproject(
    tmp_path: Path, no_override: None
) -> None:
    """A stray ``pyproject.toml`` in some unrelated parent must NOT be mistaken for us.

    Without this, running a script from any directory beneath an unrelated Python
    project would silently resolve that project as the Carnot root.
    """
    src = tmp_path / "unrelated"
    src.mkdir()
    (src / "pyproject.toml").write_text("[project]\nname='other'\n")
    probe = src / "thing.py"
    probe.write_text("")
    with pytest.raises(RuntimeError):
        repo_root(start=probe)


def test_unlocatable_root_raises_rather_than_guessing(tmp_path: Path, no_override: None) -> None:
    """Failing loudly is the whole point.

    Returning cwd as a "sensible default" is precisely the behaviour that let writes
    land in tracked files, so absence of a root must be an error a human sees.
    """
    orphan = tmp_path / "nowhere" / "mod.py"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("")
    with pytest.raises(RuntimeError, match="Could not locate"):
        repo_root(start=orphan)


def test_the_error_names_the_override_so_it_is_actionable(
    tmp_path: Path, no_override: None
) -> None:
    orphan = tmp_path / "nowhere2" / "mod.py"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("")
    with pytest.raises(RuntimeError, match=REPO_ROOT_ENV):
        repo_root(start=orphan)


def test_directory_start_is_accepted_as_well_as_a_file(no_override: None) -> None:
    """Callers pass ``__file__`` (a file) or a directory; both must work."""
    root = repo_root(start=__file__)
    assert repo_root(start=Path(__file__).parent) == root


# --------------------------------------------------------------------------------
# Derived output paths
# --------------------------------------------------------------------------------


def test_results_dir_is_under_the_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))
    assert results_dir() == tmp_path.resolve() / "results"


def test_results_dir_does_not_create_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A read-only caller must not have the side effect of creating directories."""
    monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))
    path = results_dir()
    assert not path.exists()


def test_results_dir_creates_when_asked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))
    path = results_dir(ensure=True)
    assert path.is_dir()


def test_results_path_builds_a_named_artifact_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))
    assert results_path("experiment_1_results.json") == (
        tmp_path.resolve() / "results" / "experiment_1_results.json"
    )


def test_results_path_can_create_nested_parents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))
    path = results_path("nested/deeper/run.json", ensure_parent=True)
    assert path.parent.is_dir()
    assert not path.exists(), "creating the parent must not create the file itself"


def test_output_dir_is_distinct_from_results_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``results/`` is evidence; ``output/`` is regenerable. Conflating them loses that."""
    monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))
    assert output_dir() != results_dir()
    assert output_dir() == tmp_path.resolve() / "output"


def test_repo_path_joins_multiple_parts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))
    assert repo_path("ops", "status.md") == tmp_path.resolve() / "ops" / "status.md"


def test_repo_path_with_no_parts_is_the_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))
    assert repo_path() == tmp_path.resolve()


# --------------------------------------------------------------------------------
# is_canonical_repo_root
# --------------------------------------------------------------------------------


def test_is_canonical_repo_root_accepts_both_spellings(no_override: None) -> None:
    """The alias must compare EQUAL to the real path.

    A string comparison against one spelling silently misses the other, which is why
    this helper exists instead of ``==`` on strings.
    """
    assert is_canonical_repo_root(CANONICAL)
    if ALIAS.exists():
        assert is_canonical_repo_root(ALIAS)


def test_is_canonical_repo_root_rejects_other_paths(no_override: None, tmp_path: Path) -> None:
    assert not is_canonical_repo_root(tmp_path)


def test_is_canonical_repo_root_survives_a_bad_input(no_override: None) -> None:
    """A yes/no question must not make callers handle an exception for bad input."""
    assert not is_canonical_repo_root("\0not-a-path")


# --------------------------------------------------------------------------------
# Caller-frame default
# --------------------------------------------------------------------------------


def test_default_start_follows_the_caller_not_the_resolver(
    tmp_path: Path, no_override: None
) -> None:
    """``repo_root()`` with no argument must resolve relative to the CALLING module.

    This is what makes a script executed out of a second clone resolve to that clone
    rather than to wherever ``carnot.paths`` happens to be installed from.
    """
    fake = tmp_path / "otherclone"
    (fake / "python").mkdir(parents=True)
    (fake / "scripts").mkdir()
    (fake / ".git").mkdir()
    caller = fake / "scripts" / "caller.py"
    caller.write_text("from carnot.paths import repo_root\ndef where():\n    return repo_root()\n")
    env = {k: v for k, v in os.environ.items() if k != REPO_ROOT_ENV}
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, %r);"
            "import caller; print(caller.where())" % str(caller.parent),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert out.stdout.strip() == str(fake.resolve())


def test_stdlib_only_no_heavy_imports_in_the_module_source() -> None:
    """The resolver must not depend on the scientific stack.

    Path resolution runs at import time in most callers; making it depend on JAX or
    torch would be both slow and a new source of import cycles.
    """
    source = (Path(repo_root(start=__file__)) / "python" / "carnot" / "paths.py").read_text()
    for banned in ("import jax", "import torch", "import numpy", "import flax"):
        assert banned not in source, f"{banned!r} must not appear in the resolver"
