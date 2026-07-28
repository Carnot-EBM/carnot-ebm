"""The submission kernel requests a GPU big enough for the generator it ships.

REQ-ARC-WMTE-6022 / SCENARIO-ARC-WMTE-6022-MACHINE-SHAPE-MATCHES-THE-MODEL

WHY THIS FILE EXISTS. `machine_shape` in kernel-metadata.json is a free-form string that nothing
validates locally, and it silently decided how much VRAM the scored run got. It read "NvidiaL4"
(24 GB) while the generator was a 5.9 GB Qwen3.5-9B, which fitted. The 2026-07-28 generator switch
moved the model to an 18.3 GB gemma-4-31B-it whose q8 KV pool at n_ctx 81920 needs another ~5 GB.
That does NOT fit 24 GB. Reverting this string -- or copying the old value into a new kernel --
would produce a cudaMalloc failure, which the agent handles by degrading to the CPU
graph-explore cascade: a scored run that looks like it worked and scores like it did not.

WHAT WE CAN AND CANNOT ASSERT. We can assert what WE request. We cannot assert what Kaggle
ALLOCATES -- availability is not allocation, and no scored run of ours has confirmed the mapping.
The value "NvidiaRtxPro6000" rests on two independent pieces of evidence:

  1. docs.arcprize.org/arc-prize-2026's starter kit names an `rtx6000` accelerator, described as
     "Nvidia RTX 6000 (g4-standard-48) -- Heavy ML; ARC-AGI-3 exclusive".
  2. external/arc-m1-3rd-forge/kernel-metadata.json -- a real, Kaggle-pulled, SCORED 3rd-place
     kernel in this same competition (server-assigned id_no 124697453) -- requests exactly that
     string, alongside a `google/gemma-4/.../gemma-4-31b-it` model source. It corroborates both
     the machine shape and the model choice at once.

The local kagglesdk cannot adjudicate: its `kernels_api_service.py` documents only
NvidiaTeslaT4 / NvidiaTeslaP100 / Tpu1VmV38 and omits even NvidiaL4, which we have been using
successfully -- so the SDK's silence is stale documentation, not counter-evidence. The kernel
prints an `LLM GPU HARDWARE:` nvidia-smi line at runtime; the operator's next submission log is
what actually settles it. Nothing here submits anything to confirm it.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUBMISSION_META = REPO / "scripts" / "kaggle" / "submission_kernel" / "kernel-metadata.json"
PEER_META = REPO / "external" / "arc-m1-3rd-forge" / "kernel-metadata.json"

# 24 GB class. Too small for an 18.3 GB model plus an 81920-cell q8 KV pool.
UNDERSIZED_SHAPES = {"NvidiaL4", "NvidiaTeslaT4", "NvidiaTeslaP100"}
EXPECTED_SHAPE = "NvidiaRtxPro6000"


def _meta() -> dict:
    return json.loads(SUBMISSION_META.read_text(encoding="utf-8"))


def test_machine_shape_is_not_a_24gb_class_card() -> None:
    """The regression this file is named for. An 18.3 GB Q4 model does not fit 24 GB with an
    81920-cell KV pool, and the failure is SILENT: the agent catches the server failure and runs
    the CPU cascade while still reporting itself as the LLM-on scored path."""
    shape = _meta()["machine_shape"]
    assert shape not in UNDERSIZED_SHAPES, (
        f"machine_shape={shape!r} is a 24GB-class card, but the shipped generator is an 18.3GB "
        "gemma-4-31B-it Q4_K_M plus a ~5GB q8 KV pool. Either request a larger shape or shrink "
        "the model / CARNOT_ARC_INDUCE_N_CTX -- do not leave them contradicting each other."
    )


def test_machine_shape_matches_the_shape_a_scored_peer_kernel_requested() -> None:
    """Corroboration, not proof. If the peer metadata we copied this from is ever removed or
    changed, this test says so rather than leaving an unexplained magic string behind."""
    assert _meta()["machine_shape"] == EXPECTED_SHAPE
    assert PEER_META.exists(), (
        "the corroborating evidence for the machine_shape value has disappeared from the tree; "
        f"re-verify {EXPECTED_SHAPE!r} before the next submission"
    )
    peer = json.loads(PEER_META.read_text(encoding="utf-8"))
    assert peer["machine_shape"] == EXPECTED_SHAPE
    assert peer["competition_sources"] == ["arc-prize-2026-arc-agi-3"], (
        "the peer kernel must be from OUR competition for its machine_shape to be evidence "
        "about what our competition offers"
    )


def test_the_attached_model_dataset_is_the_gemma_one() -> None:
    """The kernel loads whatever .gguf the attached dataset contains. Leaving the Qwen dataset
    attached would run the retired 9B no matter what the agent code says."""
    sources = _meta()["dataset_sources"]
    assert any("gemma4-31b" in s for s in sources), sources
    assert not any("qwen" in s.lower() for s in sources), (
        f"a retired Qwen GGUF dataset is still attached: {sources}"
    )


def test_the_kernel_still_does_not_enable_internet() -> None:
    """Unrelated to the switch, and exactly why it is worth pinning: the offline contract is easy
    to break while editing this file for another reason."""
    meta = _meta()
    assert meta["enable_internet"] is False
    assert meta["enable_gpu"] is True
