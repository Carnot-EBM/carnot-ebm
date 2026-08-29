"""Shared test fixtures for Carnot Python tests."""

import contextlib
import os
import resource
import sys
import tempfile
import warnings
from pathlib import Path

# Must be set before any JAX import to prevent CUDA backend probing.
# jax.config.update("jax_platform_name", "cpu") is insufficient — JAX still
# probes all backends at startup, which fails with CUDA_ERROR_OUT_OF_MEMORY
# when GPUs are occupied by other processes.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import pytest

from carnot.paths import repo_root as _resolve_repo_root
from carnot.testing.pytest_basetemp_isolation import install_isolated_basetemp
from carnot.testing.pytest_memory_watchdog import MemoryLeakDetected, PytestMemoryWatchdog

# Add repo root to sys.path so tests can import from scripts/
#
# Routed through the central resolver (`carnot.paths`) rather than an ad-hoc
# `Path(__file__).parent.parent.parent`. Two reasons, both about the SPELLING of the path
# rather than which directory it names:
#
#  * The old expression never called `.resolve()`. Entered through the
#    `Carnot-EBM/carnot-ebm` symlink alias, it inserted the ALIAS spelling on `sys.path`, so
#    modules imported from it recorded alias-flavoured `__file__` values and any provenance
#    derived from them disagreed with a run started from the real path -- exactly the
#    inconsistency `carnot.paths` exists to remove.
#  * Using the resolver here means there is ONE definition of "the repo root" in the test
#    suite, so a future fix to root detection reaches this insert too.
#
# This was never a path-POISONING bug: the old expression was relative to the conftest that
# is actually running, so it could not point at a different clone. It is corrected for
# consistency and de-aliasing, not because it selected the wrong tree.
#
# `carnot` is already importable at this point (see the import above), so this introduces no
# bootstrap cycle.
repo_root = _resolve_repo_root(start=__file__)
sys.path.insert(0, str(repo_root))

# Disable JAX GPU for testing (CPU only)
jax.config.update("jax_platform_name", "cpu")
try:
    # Initialise the CPU backend before pytest_configure installs RLIMIT_AS.
    # XLA can abort the interpreter if its first CPU-client allocation happens
    # after the 32 GB virtual-memory cap is active.
    jax.devices("cpu")
except Exception as exc:  # pragma: no cover - surfaced by tests that use JAX.
    warnings.warn(f"Could not pre-initialise JAX CPU backend: {exc}", stacklevel=2)

# Pytest collection exclusions:
# - quarantine/ — tests we've explicitly removed from default discovery.
# - test_conductor_supervisor.py — spawns subprocess.Popen of the real
#   supervisor, which then SIGTERMs the running conductor as "orphan"
#   (observed 4× on 2026-04-28). Quarantined version stays in
#   tests/python/quarantine/. Conductor self-heal sometimes regenerates
#   the file at the un-quarantined path; this ignore stops pytest
#   discovering it.
collect_ignore_glob = [
    "quarantine/**",
]
collect_ignore = [
    "test_conductor_supervisor.py",
    # test_experiment_772_semantic_energy_probe.py imports SemanticCluster,
    # _compute_tfidf_matrix, _cosine_similarity from semantic_energy_probe —
    # none of those exist in the shipped module. Test was written against a
    # TF-IDF+cosine design that was never landed; the actual implementation
    # uses random-projection embedding + Gaussian kernels. Quarantining
    # until the test is rewritten to match the shipped SemanticEnergyProbe.
    #
    # Observed: this collection error blocked .81 milestone activation
    # 2026-04-29 — the conductor's pre-test self-heal looped indefinitely
    # (1 failed / 302 passed) because the failure is a structural
    # ImportError, not a fixable runtime assertion.
    "test_experiment_772_semantic_energy_probe.py",
    # test_experiment_{1029,1031,1043}_*.py reference scripts that exist
    # transiently in the conductor's workflow but were never committed
    # (batching-check hook rejected them; see scripts/batching_precommit_check.py).
    # The scripts were removed from disk during a stash cycle; the tests
    # remain orphaned. Collection errors block the conductor's pre-test
    # self-heal, which in turn blocks .81 milestone retro from completing.
    # Quarantining until either the scripts are re-authored or the tests
    # are deleted as part of the conductor's clean-up pass.
    #
    # Observed: 2026-04-29 — exp1049 milestone-retro-81 SKIPPED because
    # 3 collection errors caused self-heal to loop (2 failed / 381 passed).
    "test_experiment_1029_fover_expansion_v2.py",
    "test_experiment_1031_energy_ssd_v3.py",
    "test_experiment_1043_fover_expansion_v3.py",
    # 2026-06-29 RESOLVED + UN-QUARANTINED: test_experiment_4940/4941/4943/4944 were quarantined as an
    # emergency unblock when the milestone .456/.457 archive/activate TRANSITION gate (then
    # PRETEST_COMMAND=`pytest tests/python -q`, WITH pyproject coverage addopts) ran codex past its 4801s
    # cap. ROOT CAUSE (diagnosed by the outer-loop watchdog): these are HEAVY-compute tests and COVERAGE
    # INSTRUMENTATION made them ~15x slower (6s each with --no-cov vs 90s+ with coverage); the full
    # coverage-instrumented suite then blew the cap. They were NEVER hanging on real I/O. FIXED at the
    # gate: the transition modules now use `pytest <own-test> -q --no-cov` (exp4968), and the conductor's
    # own full-suite gate already uses `--no-cov -o addopts=` (research_conductor.py:1354). With coverage
    # off in every conductor gate, these tests pass in ~6s and no longer threaten any cap, so they are
    # un-quarantined here (verified: all 4 pass rc=0 in ~6s with --no-cov). The remaining slowness only
    # appears in full-suite-WITH-coverage runs (CI / manual), which have no codex cap.
]


def _get_memory_watchdog(config) -> PytestMemoryWatchdog:
    watchdog = getattr(config, "_carnot_memory_watchdog", None)
    if watchdog is None:
        watchdog = PytestMemoryWatchdog()
        config._carnot_memory_watchdog = watchdog
    return watchdog


def _set_process_address_space_limit(limit_bytes: int = 32 * 1024**3) -> bool:
    """Set a hard address-space cap before tests can load oversized models.

    Cap raised from 8GB to 32GB on 2026-05-03: 8GB was below JAX's post-init
    virtual-memory baseline (~10-12GB on this dev rig), causing pytest
    collection to hang indefinitely as allocations thrashed against the cap.
    32GB gives JAX + ~16GB of test/model overhead while still preventing
    the runaway 35GB-per-worker pattern we were targeting (a single worker
    cannot grow to 35GB if it's capped at 32GB; multi-worker runs are
    capped per-process so total stays bounded by N × 32GB).
    """
    try:
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_soft = min(limit_bytes, hard) if hard != resource.RLIM_INFINITY else limit_bytes
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
    except (OSError, ValueError) as exc:
        warnings.warn(f"Could not set RLIMIT_AS: {exc}", RuntimeWarning, stacklevel=2)
        return False
    return True


def _mutation_check_module():
    """Import the mutation detector, or None if it is unavailable for any reason.

    Never raises. A guard that can break the test suite it guards will be deleted by the first
    person it inconveniences, which leaves the record unprotected -- the exact outcome it exists
    to prevent.
    """
    try:
        sys.path.insert(0, str(repo_root / "scripts"))
        import test_suite_mutation_check as _m

        return _m
    except Exception:  # noqa: BLE001 - deliberately total
        return None


def _operator_curated_doc_guard():
    """Import the operator-curated-doc guard, or None if unavailable.

    Never raises, for the same reason `_mutation_check_module` never raises: a guard that can
    break the suite it guards gets deleted by the first person it inconveniences.
    """
    try:
        from carnot.testing import operator_curated_doc_guard as _g

        return _g
    except Exception:  # noqa: BLE001 - deliberately total
        return None


def _tracked_results_guard():
    """Import the tracked-results guard, or None if unavailable."""
    try:
        from carnot.testing import tracked_results_guard as _g

        return _g
    except Exception:  # noqa: BLE001 - deliberately total
        return None


def _install_experiment_artifact_root(config) -> None:
    """Set a validated artifact-output temp root before tests collect modules."""
    try:
        from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV, validate_artifact_output_root
    except Exception:  # noqa: BLE001 - a missing guard should not hide collection failures
        return

    previous = os.environ.get(ARTIFACT_ROOT_ENV)
    config._carnot_artifact_root_previous = previous
    config._carnot_artifact_root_owned = False

    if previous:
        validate_artifact_output_root(previous)
        config._carnot_artifact_root = previous
        return

    root = tempfile.mkdtemp(prefix="carnot-pytest-artifacts-")
    os.environ[ARTIFACT_ROOT_ENV] = root
    validate_artifact_output_root(root)
    config._carnot_artifact_root = root
    config._carnot_artifact_root_owned = True


def _restore_experiment_artifact_root(config) -> None:
    """Restore the artifact-output env var to its pre-pytest value."""
    try:
        from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV
    except Exception:  # noqa: BLE001
        return
    previous = getattr(config, "_carnot_artifact_root_previous", None)
    if previous is None:
        os.environ.pop(ARTIFACT_ROOT_ENV, None)
    else:
        os.environ[ARTIFACT_ROOT_ENV] = previous


def _install_operator_curated_doc_guard() -> None:
    """Install the audit hook that refuses writes to operator-curated documents."""
    guard = _operator_curated_doc_guard()
    if guard is not None:
        # Deliberately total: a guard that can break the suite it guards gets deleted by the
        # first person it inconveniences, which leaves the record unprotected.
        with contextlib.suppress(Exception):
            guard.install()


def _install_tracked_results_guard() -> None:
    """Install the audit hook that refuses tracked results writes in tests."""
    guard = _tracked_results_guard()
    if guard is not None:
        with contextlib.suppress(Exception):
            guard.install()


def _install_legacy_results_write_compat() -> None:
    """Install the narrow legacy ``results/...`` redirector for pytest writes."""
    guard = _tracked_results_guard()
    if guard is not None:
        with contextlib.suppress(Exception):
            guard.install_legacy_results_write_compat()


def _child_results_guard():
    """Import the child-process results guard, or None if unavailable.

    Never raises, for the same reason its siblings never raise: a guard that can break the
    suite it guards gets deleted by the first person it inconveniences.
    """
    try:
        from carnot.testing import child_results_guard as _g

        return _g
    except Exception:  # noqa: BLE001 - deliberately total
        return None


def _install_child_results_guard(config) -> None:
    """Carry the ``results/`` redirect into child processes.

    The two in-process guards above cannot see a subprocess: a PEP 578 audit hook belongs to
    the interpreter that added it, and a monkeypatched `builtins.open` belongs to that
    interpreter's memory. Measured 2026-08-24 on `test_experiment_3361_*`, which exercises one
    writer both ways -- the in-process half left the tree clean, the subprocess half rewrote
    the committed artifact and still reported green. See `child_results_guard` for the full
    mechanism and for what it still does not catch.
    """
    guard = _child_results_guard()
    root = getattr(config, "_carnot_artifact_root", None)
    if guard is None or not root:
        return
    with contextlib.suppress(Exception):
        guard.install(str(root))


def _is_xdist_worker(config) -> bool:
    """True inside an xdist worker process, where this must NOT run.

    `-n 4` is in `addopts`, so conftest is imported in the controller AND in every worker.
    Without this check, four workers would each take a baseline and each write the marker, and a
    worker's "session end" is not the run's end -- the marker would be armed from a partial view
    while other workers were still writing.
    """
    return hasattr(config, "workerinput")


def pytest_configure(config) -> None:
    """Set hard address-space limit and keep the RSS watchdog installed."""
    # PRIVATE TMP BASE PER INVOCATION (2026-08-28). With the shared default tmp root and
    # `tmp_path_retention_count = 1`, any concurrent pytest pruned the OTHER run's live tmp
    # base. Two ~8h full-suite runs were destroyed that way, and mid-run deletion turned
    # failures into setup errors -- suppressing the very count being measured. An explicit
    # --basetemp is respected untouched; xdist workers already arrive with one assigned.
    install_isolated_basetemp(config)
    _install_experiment_artifact_root(config)
    config._carnot_rlimit_as_set = _set_process_address_space_limit()
    config._carnot_memory_watchdog = PytestMemoryWatchdog()

    # REFUSE ANY TEST WRITE TO AN OPERATOR-CURATED DOCUMENT (2026-07-29).
    #
    # `scripts/experiment_1750.py` writes `Path("README.md")` -- CWD-relative -- and pytest's
    # working directory is the repo root, so two tests silently replaced the operator's
    # hand-written README with a HuggingFace model card on every suite run, and passed while
    # doing it. CLAUDE.md's "Public Documentation Discipline" forbids the autonomous loop from
    # editing that file at all.
    #
    # This is installed in BOTH the controller and every xdist worker -- unlike the mutation
    # baseline below, which must run only in the controller. The distinction matters: the
    # baseline is a whole-session before/after diff and a worker's view of "session end" is
    # partial, whereas this hook must be present in whichever process actually executes the
    # offending test, and under `-n 4` that is always a worker.
    #
    # It only ARMS on writes INSIDE the repository, so a test that copies a doc into `tmp_path`
    # and rewrites the copy is unaffected (see test_experiment_209_cleanup.py, which does
    # exactly that and is correct).
    _install_operator_curated_doc_guard()
    _install_legacy_results_write_compat()
    _install_tracked_results_guard()
    _install_child_results_guard(config)

    # ARM THE RECORD-REWRITE INTERLOCK, HOWEVER PYTEST WAS INVOKED (2026-07-29).
    #
    # Running this suite REWRITES TRACKED FILES: tests that re-execute a real experiment script
    # cause it to overwrite its own committed artifact (see
    # docs/research-notes/test-suite-rewrites-the-record-survey-2026-07-29.md -- 41 tracked files,
    # including README.md and a paper-v6 section). The pre-commit interlock
    # (`test_suite_mutation_check.py --gate`) refuses a commit while a pending marker exists, but
    # ONLY `--run` ever wrote that marker -- so a bare `pytest tests/python/test_arc_*.py ...`,
    # the invocation behind both recorded incidents, left the interlock disarmed.
    #
    # Taking the baseline HERE closes that: the marker is armed from inside pytest, so opting out
    # requires opting out of pytest. Cost is one `git status --porcelain` per session.
    #
    # AND OBSERVE WHAT THIS RUN WRITES (2026-07-30).
    #
    # The baseline says WHETHER the tree moved. It cannot say WHO moved it, and the difference
    # matters because the advice ("git checkout -- <paths>") is unrecoverable: it has been aimed
    # at a concurrent agent's authored work three times now. So alongside the baseline we install
    # an audit hook recording what the run writes -- `runpy.run_path` on a real experiment script,
    # the documented damage mechanism, is an in-process event and therefore directly observable.
    # See `test_suite_mutation_check.classify()`.
    #
    # THE OBSERVER MUST RUN IN THE WORKERS, NOT JUST THE CONTROLLER. This suite defaults to
    # `-n 4` (pyproject addopts): the controller collects and the WORKERS execute. Installing only
    # in the controller would watch the one process that never calls `runpy.run_path`, record
    # nothing, and silently degrade every mutation to UNATTRIBUTED -- attribution switched off
    # under exactly the default invocation, which is the invocation behind all three incidents.
    # The controller pins the run id into the environment before spawning, so every worker
    # resolves the same id and appends to the SAME log; `read_observed` takes the union.
    config._carnot_mutation_baseline = None
    config._carnot_mutation_run_id = None
    config._carnot_mutation_flush = None
    mod = _mutation_check_module()
    if mod is None:
        return
    is_worker = _is_xdist_worker(config)
    try:
        if not is_worker:
            # Pin BEFORE the workers are spawned, so every one of them resolves to this id.
            os.environ.setdefault(mod.RUN_ID_ENV, mod.resolve_run_id())
            # This session is a new observation window. The log is append-only (so four workers
            # can share it without locking), and this path never calls `snapshot()` -- it holds
            # its baseline in memory -- so nothing else would clear it. Under a pinned
            # $CARNOT_MUTATION_RUN_ID, session 20 would otherwise still see session 1's writes.
            # Cleared in the CONTROLLER only, and before any worker is spawned, so a worker can
            # never truncate the log a sibling worker is appending to.
            mod.reset_writes(mod.resolve_run_id())
        run_id = mod.resolve_run_id()
        # Held so sessionfinish can flush BEFORE it reads: atexit is far too late.
        config._carnot_mutation_flush = mod.install_write_observer(mod._writes_path(run_id))
        config._carnot_mutation_run_id = run_id
    except Exception:  # noqa: BLE001 - observation is a diagnostic, never a blocker
        config._carnot_mutation_run_id = None

    # The BASELINE and the marker stay controller-only: they must be taken once for the session,
    # and four workers each arming their own marker would quadruple the noise for one event.
    if not is_worker:
        try:
            config._carnot_mutation_baseline = mod.dirty_tracked()
        except Exception:  # noqa: BLE001 - a diagnostic must never break the suite
            config._carnot_mutation_baseline = None


def _clear_guard_violations() -> None:
    guard = _operator_curated_doc_guard()
    if guard is not None:
        guard.clear_violations()
    tracked_guard = _tracked_results_guard()
    if tracked_guard is not None:
        tracked_guard.clear_violations()


def _fail_if_guard_violations() -> None:
    """Fail the current test if it wrote an operator-curated document.

    This is the ANTI-SWALLOW half of the guard. The audit hook raises at the write, which fails
    an ordinary test on the spot -- but a test that wraps the call in `except Exception: pass`
    would otherwise report green, and a guard a careless test can silence is not a guard. The
    hook records every violation to a ledger regardless of what the test body does with the
    exception; this reads that ledger and fails the test on a non-empty result. The only way to
    pass is to not perform the write.
    """
    guard = _operator_curated_doc_guard()
    if guard is not None:
        violations = guard.recorded_violations()
        if violations:
            guard.clear_violations()
            detail = "\n\n".join(f"  {v['event']} {v['path']}\n{v['stack']}" for v in violations)
            raise pytest.fail.Exception(
                "Test wrote (or attempted to write) an operator-curated document.\n"
                "CLAUDE.md 'Public Documentation Discipline' forbids the autonomous loop from\n"
                "editing these files. Redirect the write to tmp_path -- do not delete the test.\n\n"
                f"{detail}"
            )
    tracked_guard = _tracked_results_guard()
    if tracked_guard is None:
        return
    tracked_violations = tracked_guard.recorded_violations()
    if not tracked_violations:
        return
    tracked_guard.clear_violations()
    detail = "\n\n".join(f"  {v['event']} {v['path']}\n{v['stack']}" for v in tracked_violations)
    raise pytest.fail.Exception(
        "Test wrote (or attempted to write) tracked result evidence.\n"
        "CLAUDE.md 'Test-Run Record Integrity Discipline' forbids tests from\n"
        "rewriting results/**. Use the artifact-output resolver or tmp_path.\n\n"
        f"{detail}"
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item) -> None:
    # Reset the guard's ledger so a violation is attributed to exactly one test.
    _clear_guard_violations()
    _get_memory_watchdog(item.config).record_setup(item)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Convert a swallowed guard violation into a real call-phase FAILURE.

    Doing this in the CALL phase rather than teardown matters for how the result reads. A
    `pytest.fail` raised in teardown produces `1 passed, 1 error` -- the exit code is non-zero,
    so nothing ships, but the summary line still says "passed" next to the offending test, which
    is exactly the kind of ambiguous signal that lets a real problem get waved through. Failing
    in the call phase reports it as `1 failed`, which is what actually happened.
    """
    try:
        result = yield
    except BaseException:
        # The test already failed -- most likely on the guard's own exception, which carries the
        # better message and the write's stack. Drop our ledger entry so teardown does not
        # report the same violation a second time, and let the original propagate.
        _clear_guard_violations()
        raise
    _fail_if_guard_violations()
    return result


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem) -> None:
    # Safety net for violations that happen OUTSIDE the call phase -- a fixture's setup or
    # teardown. The call-phase wrapper above has already cleared anything it handled, so this
    # cannot double-report.
    _fail_if_guard_violations()

    try:
        _get_memory_watchdog(item.config).record_teardown(item)
    except MemoryLeakDetected as exc:
        get_closest_marker = getattr(item, "get_closest_marker", None)
        if get_closest_marker is not None and get_closest_marker("memory_watchdog_skip"):
            return
        pytest.fail(str(exc), pytrace=False)


def pytest_sessionfinish(session, exitstatus) -> None:
    report = _get_memory_watchdog(session.config).finish_session(Path(session.config.rootpath))
    if report is not None:
        warnings.warn(pytest.PytestWarning(report.warning), stacklevel=2)

    # Close the interlock armed in pytest_configure. Any tracked file that is dirty NOW but was
    # clean at session start was rewritten by this run; record it so the pre-commit gate refuses
    # to publish it. This only ARMS a marker -- it never reverts a file: arming is safe and
    # reverting is not (see test_suite_mutation_check.backup()).
    config = session.config

    # FLUSH FIRST, AND IN EVERY PROCESS, before the worker early-return below.
    # `install_write_observer` buffers in memory and writes at exit, which is too late here twice
    # over. In a WORKER: the controller may read the log before the worker's interpreter has
    # exited, so a worker that flushed only at exit would contribute nothing and its writes --
    # which under `-n 4` are ALL of the real writes -- would come out UNATTRIBUTED. In the
    # CONTROLLER: sessionfinish runs long before atexit, so reading first would attribute nothing.
    flush = getattr(config, "_carnot_mutation_flush", None)
    if flush is not None:
        # Suppressed deliberately: a diagnostic must never break the suite it guards.
        with contextlib.suppress(Exception):
            flush()

    baseline = getattr(config, "_carnot_mutation_baseline", None)
    if baseline is None or _is_xdist_worker(config):
        _restore_experiment_artifact_root(config)
        return
    mod = _mutation_check_module()
    if mod is None:
        _restore_experiment_artifact_root(config)
        return

    try:
        run_id = getattr(config, "_carnot_mutation_run_id", None)
        muts = mod.arm_from_pytest(baseline, ["pytest", *sys.argv[1:]], run_id)
        # Only files this run was OBSERVED writing may carry the `git checkout --` suggestion.
        # The rest changed in the same window but not by this run's hand -- most often a
        # concurrent agent editing the same tree -- and telling the operator to revert those is
        # how this guard destroyed authored work three times.
        attributed = mod.attributed_from_pytest(baseline, run_id)
    except Exception:  # noqa: BLE001 - a diagnostic must never break the suite
        _restore_experiment_artifact_root(config)
        return
    if muts:
        other = [p for p in muts if p not in attributed]
        parts = [
            f"This test run modified {len(muts)} tracked file(s) that were clean before it. "
            f"Commits are blocked until this is resolved -- run "
            f"`python3 scripts/test_suite_mutation_check.py --gate` to list every affected file."
        ]
        if attributed:
            parts.append(
                f"ATTRIBUTED TO THIS RUN ({len(attributed)}, observed being written): "
                f"{attributed[:5]}{' ...' if len(attributed) > 5 else ''}. These are the committed "
                f"research record, not test output -- `git checkout -- <paths>` restores them and "
                f"the marker retires itself once the tree shows the damage undone."
            )
        if other:
            parts.append(
                f"NOT ATTRIBUTED ({len(other)}): "
                f"{other[:5]}{' ...' if len(other) > 5 else ''}. This run was not observed writing "
                f"these; they may be a concurrent agent's in-flight work. Do NOT blanket "
                f"`git checkout --` them -- check authorship first."
            )
        parts.append(
            "Do NOT reach for `--check` here: it needs a baseline taken before the run under a "
            "matching --run-id, and this hook keeps its baseline in memory rather than on disk, "
            "so `--check` would refuse."
        )
        warnings.warn(pytest.PytestWarning(" ".join(parts)), stacklevel=2)
    _restore_experiment_artifact_root(config)
