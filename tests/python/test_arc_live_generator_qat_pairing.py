"""The live ARC generator is pinned, and the legacy Gemma QAT MTP drafter must stay scoped.

Spec coverage: REQ-ARC-WMTE-5717

Origin: 2026-07-31 operator directive, after a 20-game x 3-trial head-to-head found QAT and
the shipped Q4_K_M INDISTINGUISHABLE on induction quality (mean-B 6-6-8, sign test p = 1.0
over 12 discordant pairs). The switch is therefore justified ONLY on the non-quality axes:
~1 GB less VRAM (20430 vs 21418 MiB resident, measured across 60 cells per arm) and 1 GB less
on disk. `test_switch_is_not_justified_on_quality` exists so nobody later cites this switch as
a quality win -- an earlier 13-game read DID have QAT ahead 5-2, and that lead inverted when
the sample was extended.

THE HAZARD THIS FILE GUARDS. Both `unsloth/gemma-4-31B-it-GGUF` and
`unsloth/gemma-4-31B-it-qat-GGUF` ship a drafter named `mtp-gemma-4-31B-it-Q8_0.gguf`. The
filename cannot tell them apart, and `_resolve_mtp_head` globs every `models--*GGUF` root and
takes `sorted(...)[0]`. With a non-QAT head already on this box from the previous generator,
switching the target to QAT would have paired a NON-QAT drafter with a QAT target -- accepted
by llama.cpp, forbidden by Google's card ("the assistant model must also be a QAT checkpoint
with the same precision"), and invisible except as degraded draft acceptance.
"""

import pytest

from carnot.agentic import arc_executable_world_model as wm


class TestGeneratorIsQwen38:
    """The pinned live generator."""

    def test_model_id_is_the_qwen38_repo(self) -> None:
        assert wm.ARC_LIVE_GENERATOR_MODEL_ID == "unsloth/Qwen3.8-27B-GGUF"

    def test_model_filename_is_the_qwen38_quant(self) -> None:
        assert wm.ARC_LIVE_GENERATOR_MODEL_FILENAME == "Qwen3.8-27B-Q4_K_M.gguf"

    def test_repo_substr_matches_the_current_generator(self) -> None:
        """The main generator resolver must search for the current Qwen3.8 repo."""
        substr = wm.ARC_LIVE_GENERATOR_REPO_SUBSTR
        assert substr == "Qwen3.8-27B"
        assert substr in wm.ARC_LIVE_GENERATOR_MODEL_ID
        assert substr in wm.ARC_LIVE_GENERATOR_MODEL_FILENAME


class TestDrafterMustMatchTheTarget:
    """Google's card: a QAT target requires a QAT assistant at the same precision."""

    def test_head_repo_substr_is_pinned_to_qat(self) -> None:
        assert "qat" in wm.ARC_LIVE_GENERATOR_MTP_HEAD_REPO_SUBSTR

    def test_resolver_prefers_a_same_repo_head(self, tmp_path, monkeypatch) -> None:
        """With BOTH repos' heads on disk, the QAT one must win.

        Reproduces the exact on-disk situation at switch time: a non-QAT head left over from
        the previous generator, plus the newly downloaded QAT one. Sorted alphabetically the
        non-QAT repo comes FIRST, so an unscoped `sorted(...)[0]` picks the wrong one -- this
        test fails if the repo scoping is ever removed.
        """
        hub = tmp_path / ".cache" / "huggingface" / "hub"
        for repo in (
            "models--unsloth--gemma-4-31B-it-GGUF",
            "models--unsloth--gemma-4-31B-it-qat-GGUF",
        ):
            d = hub / repo / "snapshots" / "abc" / "MTP"
            d.mkdir(parents=True)
            (d / "mtp-gemma-4-31B-it-Q8_0.gguf").write_bytes(b"x")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("CARNOT_ARC_MTP_GGUF_PATH", raising=False)

        resolved = wm._resolve_mtp_head()
        assert resolved is not None, "both heads present; resolver returned none"
        assert "qat" in resolved, (
            f"resolver bound the NON-QAT drafter to a QAT target: {resolved}. Alphabetically "
            "the non-QAT repo sorts first, so this is what an unscoped search does."
        )

    def test_returning_none_is_still_allowed(self, tmp_path, monkeypatch) -> None:
        """No head on disk must remain a first-class 'MTP off, loudly' answer, not an error.

        The one thing it must never do is fall back to the main weights: `--model-draft <main
        gguf>` is accepted by llama.cpp and then serves with speculation silently disabled.
        """
        (tmp_path / ".cache" / "huggingface" / "hub").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("CARNOT_ARC_MTP_GGUF_PATH", raising=False)
        assert wm._resolve_mtp_head() is None


class TestProvenanceOfTheSwitch:
    """Keep the reason attached to the decision."""

    def test_switch_is_not_justified_on_quality(self) -> None:
        """The constants block must record that quality was INDISTINGUISHABLE.

        An earlier 13-game read had QAT ahead 5-2 (p = 0.453). That lead did not survive the
        pre-registered extension to 20 games -- it inverted to 6-6, p = 1.0. Without this note
        the switch reads as a quality win, and the 13-game number is the one someone would
        find first.
        """
        import inspect

        src = inspect.getsource(wm)
        head = src[: src.index("ARC_LIVE_GENERATOR_MTP_DEFAULT")]
        assert "INDISTINGUISHABLE" in head.upper()
        assert "p = 1.0" in head or "p=1.0" in head
