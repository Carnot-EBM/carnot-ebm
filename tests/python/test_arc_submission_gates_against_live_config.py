"""The submission-readiness gates must pass against the LIVE config, not a synthetic fixture.

REQ-ARC-WMTE-6022 / SCENARIO-ARC-WMTE-6022-GATES-READ-THE-SHIPPED-CONFIG

WHY THIS FILE EXISTS. On 2026-07-28 the operator re-pinned the ARC generator from Qwen3.5-9B-MTP
to gemma-4-31B-it. Five submission-readiness / hardening modules asserted the retired model by
string literal, and TWO of them went red in production the instant the pin moved:

  * `experiment_4744`'s `frozen_generator_config_from_submitted(SUBMITTED_AGENT_CONFIG)` returned
    `confirmed=False` -- `model_is_pinned_generator`, `mtp_enabled` and `no_think` were all False, because
    the shipped config now (correctly) declares gemma, mtp off, and an empty think-prefix.
  * `experiment_4756`'s `REQUIRED_DATASETS` still demanded `iancblenke/carnot-qwen35-9b-mtp-gguf`
    while `kernel-metadata.json` requests `iancblenke/carnot-gemma4-31b-it-gguf`, so
    `datasets_attached` -- a subset check -- was False.

NOT ONE EXISTING TEST CAUGHT EITHER, and the reason is the interesting part: every test for those
modules builds a SYNTHETIC fixture dict shaped like `frozen_generator` and asserts the function
reads it correctly. That tests the parser. It cannot test the PIN, because the fixture carries
whatever model the test author typed -- so the fixtures were updated alongside the assertions and
stayed green while the real configuration diverged.

The fix is this file: every assertion below reads the ACTUAL shipped objects
(`SUBMITTED_AGENT_CONFIG`, `kernel-metadata.json`, the canonical `ARC_LIVE_GENERATOR_*`
constants). A fixture cannot make these pass.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from carnot.agentic import arc_executable_world_model as wm  # noqa: E402
from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
KERNEL_METADATA = REPO / "scripts" / "kaggle" / "submission_kernel" / "kernel-metadata.json"


@pytest.fixture(scope="module")
def kernel_metadata() -> dict:
    return json.loads(KERNEL_METADATA.read_text(encoding="utf-8"))


def test_exp4744_frozen_generator_gate_passes_on_the_shipped_config() -> None:
    """The exact call the readiness experiment makes, on the exact object it makes it against."""
    from carnot.experiment_4744_submission_package_readiness import (
        frozen_generator_config_from_submitted,
    )

    result = frozen_generator_config_from_submitted(SUBMITTED_AGENT_CONFIG)
    failed = sorted(k for k, v in result["checks"].items() if not v)
    assert result["confirmed"] is True, (
        "the frozen-generator readiness gate is RED against the configuration we actually ship; "
        f"failing checks: {failed}"
    )
    # ...and it is green for the RIGHT reason -- reading the canonical pin, not because someone
    # deleted the assertions.
    assert result["model_id"] == wm.ARC_LIVE_GENERATOR_MODEL_ID
    assert result["repo_substr"] == wm.ARC_LIVE_GENERATOR_REPO_SUBSTR
    assert result["model_filename"] == wm.ARC_LIVE_GENERATOR_MODEL_FILENAME


def test_exp4756_required_datasets_match_what_the_kernel_actually_attaches(
    kernel_metadata,
) -> None:
    """`datasets_attached` is `REQUIRED_DATASETS.issubset(dataset_sources)`. If the two drift, the
    readiness gate blocks the submission on a dataset the kernel deliberately does not attach."""
    from carnot.experiment_4756_submission_package_readiness import REQUIRED_DATASETS

    attached = set(kernel_metadata["dataset_sources"])
    missing = sorted(REQUIRED_DATASETS - attached)
    assert not missing, (
        "the readiness gate requires Kaggle datasets the kernel does not request: "
        f"{missing}; kernel requests {sorted(attached)}"
    )


def test_no_submission_gate_still_names_a_retired_generator_dataset(kernel_metadata) -> None:
    """The dataset slug is the one place the model identity leaks into Kaggle-side configuration,
    and it is not covered by the source-level pin sweep in test_arc_live_generator_pin.py (that
    sweep looks at `repo_substr=` / `_resolve_gguf(` / the canonical constants, none of which
    appear in a dataset slug)."""
    from carnot.experiment_4756_submission_package_readiness import REQUIRED_DATASETS

    for slug in sorted(REQUIRED_DATASETS) + sorted(kernel_metadata["dataset_sources"]):
        assert "qwen" not in slug.lower(), f"retired generator dataset still referenced: {slug}"


def test_the_submitted_config_and_the_kernel_agree_on_the_model_dataset(kernel_metadata) -> None:
    """`SUBMITTED_AGENT_CONFIG` records which dataset slug carries the weights, and the kernel
    metadata requests it. These are written in two different files by two different concerns and
    have no mechanical link -- so they get one here."""
    slug = SUBMITTED_AGENT_CONFIG["frozen_generator"]["kaggle_dataset_slug"]
    assert slug in set(kernel_metadata["dataset_sources"]), (
        f"the config names dataset {slug!r} but the kernel requests "
        f"{sorted(kernel_metadata['dataset_sources'])}"
    )


def test_the_kernel_requests_a_machine_shape_that_can_hold_the_new_generator(
    kernel_metadata,
) -> None:
    """AVAILABILITY IS NOT ALLOCATION. Kaggle grants the shape the metadata REQUESTS. The kernel
    used to request `NvidiaL4` (24 GB), which was correct for a 5.9 GB Qwen3.5-9B and is not
    correct for an 18.3 GB gemma-4-31B plus its KV pool -- the same arithmetic that makes the
    local 3090 need an FFN offload.

    CAVEAT, recorded here because it is load-bearing and UNVERIFIED: the exact string
    `NvidiaRtxPro6000` could not be confirmed against an authoritative enum. Kaggle's own
    `kernels_metadata.md` documents the field only by example (`NvidiaTeslaT4`, `NvidiaTeslaP100`,
    `Tpu1VmV38`) and the installed `kagglesdk` carries no machine-shape enum to check against. The
    evidence it rests on is `external/arc-m1-3rd-forge/kernel-metadata.json`, which is a
    server-serialized pull of a real published kernel (it carries a server-assigned `id_no`) and
    uses that exact spelling. That is good corroboration, not proof. It MUST be confirmed on the
    operator's next submission.
    """
    shape = kernel_metadata.get("machine_shape")
    assert shape == "NvidiaRtxPro6000", shape
    assert kernel_metadata.get("enable_gpu") is True
    assert kernel_metadata.get("enable_internet") is False, (
        "the scored ARC agent must run offline -- internet access would also break the "
        "decentralization contract the local-GGUF generator exists to satisfy"
    )
