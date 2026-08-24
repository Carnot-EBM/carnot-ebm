"""Tests for the child-process tracked-``results/`` guard.

Spec coverage: REQ-REPORT-6157 (repo-wide artifact isolation closure).

The incident these are named for: `tests/python/test_experiment_1736_kanele_synth.py`
shells out to `scripts/experiment_1736_kanele_synth.py`, which rebuilds its artifact from a
hardcoded template. The two in-process guards cannot see a child process, so every run
rewrote the committed artifact -- dropping `flagged_adversarial`, `corrigendum_note` and
`corrigendum_pending` -- and passed while doing it.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import textwrap

from carnot.testing import child_results_guard


@contextlib.contextmanager
def _unpatched_popen():
    """Run a spawn through the REAL `Popen.__init__`.

    `conftest.py` installs this guard for the whole session, and its wrapper forces the
    session's redirect root onto every child -- correct in production, but it would override
    the fixture root these tests point at. Restoring the original for the duration keeps each
    test measuring the shim itself rather than the ambient installation.
    """
    patched = subprocess.Popen.__init__
    subprocess.Popen.__init__ = child_results_guard._ORIGINAL_POPEN_INIT
    try:
        yield
    finally:
        subprocess.Popen.__init__ = patched


def _writer_script(tmp_path, relative_target: str) -> str:
    """A child script that writes a repo-relative `results/...` path, like the real ones."""
    script = tmp_path / "writer.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import json, os
            path = {relative_target!r}
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as handle:
                json.dump({{"experiment": "fake", "status": "success"}}, handle)
            """
        ),
        encoding="utf-8",
    )
    return str(script)


def test_child_env_is_a_noop_without_a_redirect_root():
    """No redirect root means no change, so a non-test process spawns children as before."""
    result = child_results_guard.child_env({"PATH": "/usr/bin"}, redirect_root="")
    assert result == {"PATH": "/usr/bin"}
    assert "PYTHONPATH" not in result


def test_child_env_prepends_the_shim_and_sets_both_variables(tmp_path):
    result = child_results_guard.child_env({"PYTHONPATH": "/existing"}, redirect_root=str(tmp_path))
    shim = str(child_results_guard.shim_dir())
    assert result["PYTHONPATH"].split(":")[0] == shim
    assert "/existing" in result["PYTHONPATH"]
    assert result[child_results_guard.CHILD_REDIRECT_ROOT_ENV] == str(tmp_path)
    assert result[child_results_guard.CHILD_REPO_ROOT_ENV]


def test_child_env_does_not_duplicate_the_shim_on_repeated_calls(tmp_path):
    """A child that spawns a grandchild must not grow PYTHONPATH without bound."""
    once = child_results_guard.child_env({}, redirect_root=str(tmp_path))
    twice = child_results_guard.child_env(once, redirect_root=str(tmp_path))
    shim = str(child_results_guard.shim_dir())
    assert twice["PYTHONPATH"].split(":").count(shim) == 1


def test_shim_redirects_a_child_write_away_from_the_tracked_results_tree(tmp_path):
    """The incident itself: a child writing `results/...` must not touch the real tree.

    Driven through the generated shim directly rather than through `install()`, so the test
    never patches the interpreter running the suite.
    """
    repo = tmp_path / "repo"
    (repo / "results").mkdir(parents=True)
    evidence = repo / "results" / "experiment_fake.json"
    evidence.write_text(json.dumps({"kept": True, "flagged_adversarial": True}), encoding="utf-8")

    redirect = tmp_path / "redirect"
    redirect.mkdir()

    script = _writer_script(tmp_path, "results/experiment_fake.json")
    env = child_results_guard.child_env({"PATH": "/usr/bin:/bin"}, redirect_root=str(redirect))
    env[child_results_guard.CHILD_REPO_ROOT_ENV] = str(repo)

    with _unpatched_popen():
        completed = subprocess.run(
            [sys.executable, script], cwd=repo, env=env, capture_output=True, text=True
        )

    assert completed.returncode == 0, completed.stderr
    # The committed evidence keeps every key, including the conductor's determination.
    assert json.loads(evidence.read_text()) == {"kept": True, "flagged_adversarial": True}
    # And the child's write really did happen, just somewhere harmless.
    assert json.loads((redirect / "experiment_fake.json").read_text())["status"] == "success"


def test_shim_leaves_writes_outside_the_results_tree_alone(tmp_path):
    """Only `results/` is redirected; an ordinary child write must still land normally."""
    repo = tmp_path / "repo"
    (repo / "elsewhere").mkdir(parents=True)
    redirect = tmp_path / "redirect"
    redirect.mkdir()

    script = _writer_script(tmp_path, "elsewhere/note.json")
    env = child_results_guard.child_env({"PATH": "/usr/bin:/bin"}, redirect_root=str(redirect))
    env[child_results_guard.CHILD_REPO_ROOT_ENV] = str(repo)

    with _unpatched_popen():
        completed = subprocess.run(
            [sys.executable, script], cwd=repo, env=env, capture_output=True, text=True
        )

    assert completed.returncode == 0, completed.stderr
    assert (repo / "elsewhere" / "note.json").exists()
    assert not (redirect / "note.json").exists()


def test_env_positional_index_matches_the_live_popen_signature():
    """`env` is passed positionally by some callers; rewriting the wrong slot breaks them."""
    import inspect

    # Read the ORIGINAL signature. `subprocess.Popen.__init__` may already be wrapped by the
    # guard itself under the running suite, and the wrapper's `(self, args, *rest, **kwargs)`
    # signature would make this assertion measure the wrapper rather than the real thing.
    names = list(inspect.signature(child_results_guard._ORIGINAL_POPEN_INIT).parameters)
    assert names[child_results_guard._ENV_POSITIONAL_INDEX + 2] == "env"
