"""Tests for the no-LLM substrate alias evidence lint.

REQ: REQ-SUBSTRATE-ALIAS-1 (openspec/capabilities/research-harnesses/spec.md).
SCENARIOs: SCENARIO-SUBSTRATE-ALIAS-1 (no evidence -> refuse), -2 (test names
the alias -> pass), -3 (ack entry -> pass), -4 (removal / unrelated edit ->
pass), -5 (git failure -> refuse, never pass).

Origin: on 2026-08-23 the fabrication gate's NO_LLM_SUBSTRATE_ALIASES tuple
held 38 names, 19 added in two days, and every sampled addition landed in the
same conductor commit as the artifact it exempted -- commit 59c8f8602d
registered an alias at 22:24Z and wrote experiment_6520 at 22:25Z. An
experiment could clear the gate by naming its own substrate in the gate.

No test writes tracked state; the git-facing paths are monkeypatched.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pytest  # noqa: E402

import substrate_alias_evidence_lint as lint  # noqa: E402

# A realistic pair of added lines: the module constant and the tuple entry,
# which is how a real alias addition looks in the diff.
DIFF_ADDING_ALIAS = """\
diff --git a/scripts/adversarial_verify.py b/scripts/adversarial_verify.py
--- a/scripts/adversarial_verify.py
+++ b/scripts/adversarial_verify.py
@@ -418,0 +419,3 @@
+LOCAL_COMPACT_ROUTER_SUBSTRATE = (
+    "local_compact_router_plus_exact_exception_table_and_native_solver_no_llm"
+)
@@ -440,0 +444 @@
+    LOCAL_COMPACT_ROUTER_SUBSTRATE,
"""

NEW_ALIAS = "local_compact_router_plus_exact_exception_table_and_native_solver_no_llm"

HEAD_WITHOUT_ALIAS = """\
NO_LLM_SUBSTRATE_ALIASES = (
    "aggregation_from_upstream_artifacts_no_llm",
    "deterministic_automaton_no_llm",
)
"""

HEAD_WITH_ALIAS = HEAD_WITHOUT_ALIAS.replace(
    '    "deterministic_automaton_no_llm",',
    f'    "deterministic_automaton_no_llm",\n    "{NEW_ALIAS}",',
)


class TestNewAliasDetection:
    def test_added_alias_is_detected_once(self) -> None:
        # SCENARIO-SUBSTRATE-ALIAS-1, detection half. The literal appears on
        # one added line and the constant name on another; the alias must be
        # reported exactly once, not twice.
        assert lint.new_aliases(DIFF_ADDING_ALIAS, HEAD_WITHOUT_ALIAS) == [NEW_ALIAS]

    def test_alias_already_in_head_is_not_new(self) -> None:
        # Reformatting can re-add a line for an alias that already existed.
        # Widening the gate is the trigger, not touching the file.
        assert lint.new_aliases(DIFF_ADDING_ALIAS, HEAD_WITH_ALIAS) == []

    def test_removal_is_not_an_addition(self) -> None:
        # SCENARIO-SUBSTRATE-ALIAS-4. Narrowing the gate needs no evidence.
        removal = DIFF_ADDING_ALIAS.replace("\n+", "\n-")
        assert lint.new_aliases(removal, HEAD_WITH_ALIAS) == []

    def test_removal_is_not_an_addition_even_with_an_empty_head(self) -> None:
        # The test above passes even without the '+' prefix check, because a
        # removed alias is by definition already in HEAD and the `already`
        # subtraction filters it. Mutation testing found that: deleting the
        # prefix check left the suite green, i.e. the check was decorative.
        #
        # It stops being decorative the moment head_text is empty or stale --
        # a caller that reads HEAD from the wrong ref, or a future
        # diff-from-file mode. Then every '-' line would parse as an addition
        # and REMOVING an alias would demand evidence. Pin that here so the
        # prefix check is load-bearing under test.
        removal = DIFF_ADDING_ALIAS.replace("\n+", "\n-")
        assert lint.new_aliases(removal, "") == []

    def test_unrelated_added_lines_are_ignored(self) -> None:
        unrelated = (
            "--- a/scripts/adversarial_verify.py\n"
            "+++ b/scripts/adversarial_verify.py\n"
            "+SOME_OTHER_CONSTANT = 3\n"
            '+SUBSTRATE_KIND_LIVE = "live_llm_inference"\n'
        )
        assert lint.new_aliases(unrelated, HEAD_WITHOUT_ALIAS) == []

    def test_diff_header_plusplusplus_is_not_an_added_line(self) -> None:
        # `+++ b/...` starts with '+' and must not be parsed as content.
        header_only = f"+++ b/x_{NEW_ALIAS}.py\n"
        assert lint.new_aliases(header_only, HEAD_WITHOUT_ALIAS) == []


class TestEvidence:
    def test_no_evidence_anywhere(self) -> None:
        assert lint.find_evidence(NEW_ALIAS, {}, "") is None

    def test_test_file_naming_the_alias_counts(self) -> None:
        # SCENARIO-SUBSTRATE-ALIAS-2.
        tests = {"tests/python/test_x.py": f'assert sub == "{NEW_ALIAS}"'}
        assert lint.find_evidence(NEW_ALIAS, tests, "") == "tests/python/test_x.py"

    def test_ack_entry_counts(self) -> None:
        # SCENARIO-SUBSTRATE-ALIAS-3.
        ack = f"- 2026-08-23 `{NEW_ALIAS}` — runs Z3 over a prebuilt instance."
        assert lint.find_evidence(NEW_ALIAS, {}, ack) == lint.ACK_FILE

    def test_evidence_for_a_different_alias_does_not_count(self) -> None:
        # The check must be per-alias. A neighbouring alias's ack line must not
        # launder an unrelated addition through.
        ack = "- 2026-08-23 `some_other_thing_no_llm` — deterministic replay."
        assert lint.find_evidence(NEW_ALIAS, {}, ack) is None


class TestMainExitCodes:
    @staticmethod
    def _patch_git(monkeypatch, diff: str, head: str) -> None:
        def fake(args: list[str]) -> str:
            if args[:2] == ["diff", "--cached"]:
                return diff
            if args[0] == "show":
                return head
            raise AssertionError(f"unexpected git call: {args}")

        monkeypatch.setattr(lint, "_run_git", fake)

    def test_refuses_when_alias_added_without_evidence(self, monkeypatch, capsys) -> None:
        # SCENARIO-SUBSTRATE-ALIAS-1.
        self._patch_git(monkeypatch, DIFF_ADDING_ALIAS, HEAD_WITHOUT_ALIAS)
        monkeypatch.setattr(lint, "collect_evidence_texts", lambda: ({}, ""))
        assert lint.main([]) == 1
        assert NEW_ALIAS in capsys.readouterr().err

    def test_passes_when_evidence_present(self, monkeypatch) -> None:
        self._patch_git(monkeypatch, DIFF_ADDING_ALIAS, HEAD_WITHOUT_ALIAS)
        monkeypatch.setattr(
            lint,
            "collect_evidence_texts",
            lambda: ({}, f"- 2026-08-23 `{NEW_ALIAS}` — exact solver only."),
        )
        assert lint.main([]) == 0

    def test_passes_when_gate_file_untouched(self, monkeypatch) -> None:
        self._patch_git(monkeypatch, "", HEAD_WITHOUT_ALIAS)
        monkeypatch.setattr(lint, "collect_evidence_texts", lambda: ({}, ""))
        assert lint.main([]) == 0

    def test_refuses_when_git_fails(self, monkeypatch, capsys) -> None:
        # SCENARIO-SUBSTRATE-ALIAS-5. Fail closed: a guard that answers "clean"
        # when it could not look is worse than no guard.
        def boom(args: list[str]) -> str:
            raise lint.GitUnavailable("git diff failed rc=128")

        monkeypatch.setattr(lint, "_run_git", boom)
        assert lint.main([]) == 1
        assert "REFUSING" in capsys.readouterr().err

    def test_reports_each_unsupported_alias(self, monkeypatch, capsys) -> None:
        second = "another_made_up_substrate_no_llm"
        diff = DIFF_ADDING_ALIAS + f'+    "{second}",\n'
        self._patch_git(monkeypatch, diff, HEAD_WITHOUT_ALIAS)
        monkeypatch.setattr(
            lint,
            "collect_evidence_texts",
            lambda: ({"tests/python/test_a.py": NEW_ALIAS}, ""),
        )
        assert lint.main([]) == 1
        err = capsys.readouterr().err
        assert second in err
        # The supported one must NOT appear in the refusal list.
        assert err.count(NEW_ALIAS) == 0


class TestWiring:
    def test_hook_is_registered(self) -> None:
        # A lint nothing calls is the bug class this project keeps hitting.
        config = (REPO_ROOT / ".pre-commit-config.yaml").read_text()
        assert "substrate-alias-evidence-lint" in config

    def test_ack_file_exists(self) -> None:
        assert (REPO_ROOT / lint.ACK_FILE).is_file()

    @pytest.mark.parametrize("alias", ["x_no_llm", "deterministic_automaton_no_llm"])
    def test_alias_regex_matches_the_real_shape(self, alias: str) -> None:
        assert lint.ALIAS_RE.findall(f'    "{alias}",') == [alias]
