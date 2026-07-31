"""Guarded loaders for the two deserialization formats that execute code.

**Researcher summary:**
    ``pickle.load`` and ``torch.load(weights_only=False)`` run arbitrary Python
    while parsing a file.  This module centralises those calls behind loaders
    that default to the safe mode and, when a caller genuinely needs the unsafe
    mode, force it to be explicit and confine it to files inside this checkout.

**Detailed explanation for engineers:**

    **Why this exists.**  The 2026-07-31 security audit found four
    ``pickle.load`` sites in ``python/carnot/verify/`` -- library code wired into
    the live verifier ensemble -- and eight ``torch.load(..., weights_only=False)``
    sites.  Both formats are *code execution disguised as data loading*: a
    pickle stream contains opcodes that call arbitrary constructors, so the
    payload runs during parsing.

    **The isinstance() trap.**  Two of the audited call sites did::

        loaded = pickle.load(handle)
        if not isinstance(loaded, cls):
            raise TypeError(...)

    That check is not a mitigation.  By the time ``pickle.load`` returns, any
    embedded payload has *already executed* -- the type check runs strictly
    after the damage.  It is a correctness guard, not a security one, and it was
    reasonable to mistake it for the latter, which is exactly why the guarded
    loaders below exist rather than a code-review rule.

    **What the trusted-root check actually buys, stated honestly.**  It is NOT
    protection against an attacker who can already write into this repository --
    such an attacker can edit the source directly and needs no pickle.  Its real
    value is narrower and worth naming precisely:

    1. It blocks the *supply-chain* vector.  Per the project's decentralization
       rules, model artifacts are mirrored to HuggingFace and IPFS.  A ``.pkl``
       or ``.pt`` fetched from a mirror into ``~/.cache`` and loaded by path is
       the realistic RCE route, and that path is outside the repo, so it is
       refused here.
    2. It makes every unsafe deserialization *greppable and centralised*.  The
       previous state -- twelve scattered call sites, some with a misleading
       isinstance check -- could not be audited without re-deriving the analysis
       each time.

    **Why not just convert everything to safetensors?**  safetensors stores
    tensors only.  ``tier0e_eorm`` and ``tier0f_semantic_calibration`` persist
    fitted scikit-learn estimators (a vectoriser plus a classifier), which are
    Python object graphs, not tensor dicts.  Converting them is a real project
    with a retraining step, not a security patch.  Until then, the honest
    position is: keep pickle for those, make the risk explicit, and confine it.

    **torch.load is mostly a non-issue already, and this is worth recording.**
    The audit's first pass reported "30 of 31 torch.load calls lack
    weights_only=True".  That framing overstated the exposure.  PyTorch 2.6
    flipped the *default* to ``weights_only=True``, and this project runs torch
    2.11, so the 22 bare calls are already safe -- verified empirically: a
    crafted payload raised ``UnpicklingError`` rather than executing.  Only the
    8 sites that pass ``weights_only=False`` explicitly were ever exposed.

Spec: REQ-SEC-001
"""

from __future__ import annotations

import pickle  # noqa: S403 - the point of this module is to confine its use
import warnings
from pathlib import Path
from typing import Any, TypeVar

from carnot.paths import repo_root

__all__ = [
    "UntrustedArtifactError",
    "safe_pickle_load",
    "safe_torch_load",
    "is_trusted_artifact_path",
]

_T = TypeVar("_T")


class UntrustedArtifactError(Exception):
    """Raised when a code-executing load is attempted on a file outside the repo.

    Deliberately NOT a subclass of ``OSError`` or ``ValueError``: several call
    sites wrap loads in broad ``except Exception`` handlers that downgrade a
    failure to "checkpoint absent, carry on".  A distinct type lets a caller
    re-raise this specific case instead of silently treating a refused load as a
    missing file.
    """


def is_trusted_artifact_path(path: str | Path) -> bool:
    """Return True if ``path`` resolves to a location inside this checkout.

    Symlinks are resolved BEFORE the comparison, so a symlink placed inside the
    repo but pointing at ``/tmp`` does not inherit trust -- that indirection is
    the obvious way to defeat a naive prefix check.

    Args:
        path: Filesystem path to test.  Need not exist.

    Returns:
        True when the resolved path is the repo root or beneath it.
    """
    try:
        resolved = Path(path).resolve()
        root = repo_root().resolve()
    except (OSError, RuntimeError):
        # An unresolvable path or an unlocatable repo root is treated as
        # untrusted. Fail CLOSED: the alternative is answering "trusted" when we
        # could not determine the answer, which is the failure mode this whole
        # module exists to prevent.
        return False
    return resolved == root or root in resolved.parents


def _require_trusted(path: str | Path, fmt: str) -> Path:
    """Refuse a code-executing load on a path outside the checkout."""
    if not is_trusted_artifact_path(path):
        raise UntrustedArtifactError(
            f"Refusing to {fmt}-load {path!r}: it resolves outside the repository. "
            f"{fmt} deserialization executes arbitrary code, so it is restricted to "
            "artifacts tracked in this checkout. If this file is a legitimately "
            "downloaded artifact, verify it out-of-band and copy it into the repo, or "
            "convert it to safetensors."
        )
    return Path(path)


def safe_torch_load(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    allow_unsafe_pickle: bool = False,
    **kwargs: Any,
) -> Any:
    """Load a torch checkpoint, defaulting to the tensors-only reader.

    **Researcher summary:**
        Same as ``torch.load`` but ``weights_only=True`` unless the caller
        explicitly opts into arbitrary-object loading, which additionally
        requires the file to live inside this checkout.

    **Detailed explanation for engineers:**
        With ``weights_only=True`` the unpickler accepts only tensors and plain
        containers (dict/list/tuple/str/int/float/bool), which covers the shape
        every audited call site actually consumes -- they all treat the payload
        as a Mapping.

        When ``allow_unsafe_pickle=True`` the safe read is still ATTEMPTED
        FIRST.  Only if it raises do we fall back, and the fallback warns.  That
        ordering matters: several of these checkpoints predate the audit and
        nobody knows offhand whether they need the unsafe reader, so trying the
        safe path first converts an assumption into a measurement, and the
        warning tells you which files genuinely still need it.

    Args:
        path: Checkpoint path.
        map_location: Passed to ``torch.load``; defaults to "cpu" because every
            audited call site used that.
        allow_unsafe_pickle: Permit falling back to ``weights_only=False``.
            Requires ``path`` to be inside the repo.
        **kwargs: Forwarded to ``torch.load``.

    Returns:
        The deserialized checkpoint.

    Raises:
        UntrustedArtifactError: Unsafe load requested for a path outside the repo.

    Spec: REQ-SEC-001
    """
    import torch  # local import: torch is heavy and not every caller needs it

    try:
        return torch.load(path, map_location=map_location, weights_only=True, **kwargs)
    except Exception as safe_error:  # noqa: BLE001 - deliberately broad, see below
        # Broad by design: torch raises UnpicklingError for a disallowed global,
        # but also RuntimeError/AttributeError depending on the payload and
        # version. Narrowing this would silently route some checkpoints down the
        # unsafe path for reasons unrelated to weights_only.
        if not allow_unsafe_pickle:
            raise
        _require_trusted(path, "pickle")
        warnings.warn(
            f"{path}: falling back to torch.load(weights_only=False), which executes "
            f"arbitrary code from the checkpoint. Safe read failed with: "
            f"{type(safe_error).__name__}: {safe_error}. This file should be "
            "re-saved as tensors-only or converted to safetensors.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(path, map_location=map_location, weights_only=False, **kwargs)


def safe_pickle_load(
    path: str | Path,
    *,
    expected_type: type[_T] | None = None,
    on_untrusted: str = "refuse",
) -> Any:
    """Load a pickle, refusing (or warning about) paths outside the checkout.

    **Researcher summary:**
        ``pickle.load`` with the trusted-path check applied BEFORE the file is
        opened, plus an optional post-hoc type assertion.

    **Detailed explanation for engineers:**
        There is no safe mode for pickle -- unlike torch there is no
        ``weights_only`` equivalent -- so the confinement IS the mitigation.
        Read this module's docstring for an honest account of what that does and
        does not buy.

        ``expected_type`` reproduces the ``isinstance`` check two call sites
        already had.  It is retained because it catches genuine corruption and
        wrong-file mistakes, but note it runs AFTER any embedded payload has
        executed, so it is a correctness guard only.  It is deliberately not
        described as a security control anywhere in this module.

        **Why ``on_untrusted`` exists, and why the default is not universal.**
        The first version of this module refused every out-of-repo path
        unconditionally.  That broke seven existing tests, and the breakage was
        informative rather than incidental: the failing pattern was
        ``scorer.save(tmp_path / "s.pkl")`` immediately followed by
        ``Scorer.load(tmp_path / "s.pkl")`` -- a round-trip of a file the SAME
        process had just written.  No threat model covers that, so refusing it
        was the control being wrong, not the callers.

        The two modes therefore track a real distinction in exposure:

        ``refuse``
            For loaders with a FIXED in-repo default path that feed the live
            verifier ensemble (``tier0e_eorm``, ``tier0f_semantic_calibration``).
            A path outside the repo means someone redirected a live-ensemble
            model to a downloaded file -- exactly the mirror-supply-chain vector
            -- and that should stop.

        ``warn``
            For general-purpose ``save``/``load`` round-trip APIs
            (``PartialStateDiffusionScorer``, ``DinaLRMPartialStateScorer``),
            where a scratch path is ordinary usage.  This is a genuinely weaker
            control and is not pretended otherwise: it buys visibility, not
            prevention.

    Args:
        path: Pickle path.
        expected_type: If given, the loaded object must be an instance of it.
        on_untrusted: ``"refuse"`` (default) raises for a path outside the
            checkout; ``"warn"`` emits a RuntimeWarning and proceeds.

    Returns:
        The deserialized object.

    Raises:
        UntrustedArtifactError: ``path`` is outside the repo and mode is refuse.
        TypeError: ``expected_type`` given and the payload does not match.
        ValueError: ``on_untrusted`` is not a recognised mode.

    Spec: REQ-SEC-001
    """
    if on_untrusted not in {"refuse", "warn"}:
        raise ValueError(f"on_untrusted must be 'refuse' or 'warn', got {on_untrusted!r}")

    if on_untrusted == "refuse":
        target = _require_trusted(path, "pickle")
    else:
        target = Path(path)
        if not is_trusted_artifact_path(target):
            warnings.warn(
                f"{path}: unpickling a file outside the repository. pickle executes "
                "arbitrary code while parsing, so this is only safe if the file was "
                "produced by this run or is otherwise known-good.",
                RuntimeWarning,
                stacklevel=2,
            )

    with target.open("rb") as handle:
        loaded = pickle.load(handle)  # noqa: S301 - gated by the trust check above
    if expected_type is not None and not isinstance(loaded, expected_type):
        raise TypeError(f"{path}: expected {expected_type.__name__}, got {type(loaded).__name__}")
    return loaded
