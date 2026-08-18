"""REQ-ARC-WMTE-6490: self-drafting (baked-in) MTP models launch without a separate draft head.

Scenarios covered here: SCENARIO-ARC-WMTE-6490-DETECT (the metadata marker is recognised),
SCENARIO-ARC-WMTE-6490-ABSENT (a head-requiring model is not mistaken for self-drafting),
SCENARIO-ARC-WMTE-6490-FAILCLOSED (an unreadable file forfeits MTP rather than claiming it),
SCENARIO-ARC-WMTE-6490-BOUNDED (the metadata scan is bounded, not a 23 GB read), and
SCENARIO-ARC-WMTE-6490-NOTAHEAD (a self-drafting model is never classified as a draft head).

WHY THIS EXISTS. Two different things are both called "MTP" in this project. gemma-4-31B-it has
no MTP layers in its main weights, so speculative decoding there needs `--model-draft <a separate
head file>`. The NVFP4 Qwen3.8-27B conversion has the head baked in and declares
`nextn_predict_layers` in its own GGUF metadata, so it takes `--spec-type draft-mtp` alone.

Before this change the launcher recognised only the first kind. A self-drafting model resolved no
head, so MTP was dropped entirely -- shipping the model for a feature that was then switched off,
with the only symptom a slower tok/s. That is the same silent-degradation class the surrounding
code already guards against, arriving from the other direction.
"""

from __future__ import annotations

from carnot.agentic.arc_executable_world_model import (
    _BAKED_MTP_METADATA_KEY,
    _gguf_declares_baked_mtp,
    _is_mtp_head_file,
)


def _write(tmp_path, name: str, body: bytes):
    p = tmp_path / name
    p.write_bytes(body)
    return p


# SCENARIO-ARC-WMTE-6490-DETECT
def test_detects_the_baked_in_marker(tmp_path) -> None:
    """The real file carries `qwen35.nextn_predict_layers`; the arch prefix varies by model, so
    the check must match the SUFFIX and never a whole prefixed key."""
    f = _write(tmp_path, "m.gguf", b"GGUF" + b"\x00" * 64 + b"qwen35.nextn_predict_layers")
    assert _gguf_declares_baked_mtp(f) is True


# SCENARIO-ARC-WMTE-6490-ABSENT
def test_absent_marker_is_false(tmp_path) -> None:
    """gemma-4-31B-it declares no such key. It must NOT be treated as self-drafting, or it would
    launch speculative decoding with no draft at all."""
    f = _write(tmp_path, "m.gguf", b"GGUF" + b"\x00" * 4096 + b"general.architecture")
    assert _gguf_declares_baked_mtp(f) is False


# SCENARIO-ARC-WMTE-6490-FAILCLOSED
def test_missing_file_fails_closed(tmp_path) -> None:
    """Fail CLOSED: an unreadable path costs the speedup, it never claims speculation a model
    cannot do. The opposite default would enable MTP on an unverified file."""
    assert _gguf_declares_baked_mtp(tmp_path / "nope.gguf") is False


# SCENARIO-ARC-WMTE-6490-BOUNDED
def test_scan_is_bounded_not_whole_file(tmp_path) -> None:
    """GGUF metadata sits at the head of the file, so the read is bounded -- the real model is
    23 GB and this check runs on every server launch. A marker past the window reads as absent,
    which is the fail-closed direction."""
    from carnot.agentic.arc_executable_world_model import _GGUF_METADATA_SCAN_BYTES

    f = _write(
        tmp_path, "m.gguf", b"\x00" * (_GGUF_METADATA_SCAN_BYTES + 16) + _BAKED_MTP_METADATA_KEY
    )
    assert _gguf_declares_baked_mtp(f) is False


# SCENARIO-ARC-WMTE-6490-NOTAHEAD
def test_baked_in_model_is_not_classified_as_a_draft_head() -> None:
    """The two checks must not collide. `Qwen3.8-27B-NVFP4-MTP-HIGHEST.gguf` contains "-MTP-" and
    IS the main model; only an `mtp-` PREFIX marks a head. If this inverted, the launcher would
    bind the 23 GB generator as its own draft."""
    assert _is_mtp_head_file("Qwen3.8-27B-NVFP4-MTP-HIGHEST.gguf") is False
    assert _is_mtp_head_file("mtp-gemma-4-31B-it-Q8_0.gguf") is True
