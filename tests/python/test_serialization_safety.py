"""Tests for the guarded pickle / torch loaders.

Spec coverage: REQ-SEC-001

Origin: 2026-07-31 security audit.  Four `pickle.load` sites in
`python/carnot/verify/` (library code wired into the live verifier ensemble) and
eight `torch.load(..., weights_only=False)` sites deserialized attacker-
influenceable files with code execution enabled.

The tests below encode two things that are easy to get wrong later:

1. The trusted-root check must resolve symlinks, or a symlink planted inside the
   repo trivially launders an outside path into "trusted".
2. `isinstance` after `pickle.load` is NOT a security control -- the payload has
   already executed.  `test_isinstance_check_runs_after_payload_execution` proves
   that with a real side effect, so nobody re-introduces the old pattern
   believing the type check made it safe.
"""

import pickle

import pytest

from carnot.serialization_safety import (
    UntrustedArtifactError,
    is_trusted_artifact_path,
    safe_pickle_load,
    safe_torch_load,
)

torch = pytest.importorskip("torch")


class _Marker:
    """Plain payload used to check a successful round-trip."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Marker) and other.value == self.value


class TestTrustedPathCheck:
    """REQ-SEC-001: code-executing loads are confined to this checkout."""

    def test_in_repo_path_is_trusted(self) -> None:
        assert is_trusted_artifact_path("results/tier0e_model.pkl")

    def test_outside_path_is_untrusted(self, tmp_path) -> None:
        """tmp_path stands in for ~/.cache -- the HuggingFace/IPFS mirror vector."""
        assert not is_trusted_artifact_path(tmp_path / "downloaded.pkl")

    def test_symlink_into_the_repo_does_not_launder_trust(self, tmp_path) -> None:
        """A symlink placed in-repo pointing outside must NOT inherit trust.

        This is the obvious way to defeat a naive string-prefix check, so it gets
        an explicit test rather than relying on `.resolve()` staying in place.
        """
        outside = tmp_path / "evil.pkl"
        outside.write_bytes(b"x")
        from carnot.paths import repo_root

        link = repo_root() / "_test_symlink_trust_probe.pkl"
        try:
            link.symlink_to(outside)
            assert not is_trusted_artifact_path(link)
        finally:
            link.unlink(missing_ok=True)


class TestSafePickleLoad:
    """REQ-SEC-001: pickle is confined, because pickle has no safe mode."""

    def test_refuses_out_of_repo_path(self, tmp_path) -> None:
        target = tmp_path / "payload.pkl"
        target.write_bytes(pickle.dumps({"a": 1}))
        with pytest.raises(UntrustedArtifactError, match="outside the repository"):
            safe_pickle_load(target)

    def test_loads_in_repo_path(self, tmp_path, monkeypatch) -> None:
        """Point the repo root at tmp_path so the load is legitimately trusted."""
        monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
        target = tmp_path / "payload.pkl"
        target.write_bytes(pickle.dumps(_Marker(7)))
        assert safe_pickle_load(target) == _Marker(7)

    def test_expected_type_mismatch_raises(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
        target = tmp_path / "payload.pkl"
        target.write_bytes(pickle.dumps({"not": "a marker"}))
        with pytest.raises(TypeError, match="expected _Marker"):
            safe_pickle_load(target, expected_type=_Marker)

    def test_untrusted_error_is_not_an_oserror(self) -> None:
        """Call sites wrap loads in broad handlers that treat OSError as 'absent'.

        A refused load must not be silently reclassified as a missing checkpoint,
        so the exception type is deliberately outside the OSError/ValueError tree.
        """
        assert not issubclass(UntrustedArtifactError, (OSError, ValueError))

    def test_isinstance_check_runs_after_payload_execution(self, tmp_path, monkeypatch) -> None:
        """Prove the old `pickle.load` + `isinstance` pattern was not a control.

        The payload's `__reduce__` fires during parsing.  The type check that the
        original call sites performed afterwards therefore ran strictly too late.
        This test asserts the side effect happens even though the type check
        ultimately rejects the object.
        """
        monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
        witness = tmp_path / "payload_executed"

        class _Exploit:
            def __reduce__(self):  # type: ignore[no-untyped-def]
                return (_touch, (str(witness),))

        target = tmp_path / "exploit.pkl"
        target.write_bytes(pickle.dumps(_Exploit()))

        with pytest.raises(TypeError):
            safe_pickle_load(target, expected_type=_Marker)

        assert witness.exists(), (
            "expected the pickle payload to execute during parsing -- if this ever "
            "stops being true, the isinstance-is-not-a-control rationale in "
            "serialization_safety.py needs revisiting"
        )


def _touch(path: str) -> str:
    """Module-level helper so the exploit payload is picklable."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("executed")
    return path


class TestSafeTorchLoad:
    """REQ-SEC-001: torch defaults to tensors-only; unsafe mode is opt-in + confined."""

    def test_tensor_checkpoint_round_trips(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
        target = tmp_path / "ckpt.pt"
        torch.save({"w": torch.ones(3), "step": 5}, target)
        loaded = safe_torch_load(target)
        assert loaded["step"] == 5
        assert torch.equal(loaded["w"], torch.ones(3))

    def test_arbitrary_object_payload_is_refused_by_default(self, tmp_path, monkeypatch) -> None:
        """weights_only=True must reject a non-tensor payload rather than run it."""
        monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
        target = tmp_path / "obj.pt"
        torch.save({"payload": _Marker(1)}, target)
        with pytest.raises(Exception) as excinfo:
            safe_torch_load(target)
        assert "UntrustedArtifactError" not in type(excinfo.value).__name__

    def test_unsafe_fallback_requires_trusted_path(self, tmp_path) -> None:
        """Opting into the unsafe reader is not enough -- the path must be in-repo."""
        target = tmp_path / "obj.pt"
        torch.save({"payload": _Marker(1)}, target)
        with pytest.raises(UntrustedArtifactError):
            safe_torch_load(target, allow_unsafe_pickle=True)

    def test_unsafe_fallback_warns_when_permitted(self, tmp_path, monkeypatch) -> None:
        """The fallback must be loud: a silent unsafe load is the failure mode."""
        monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
        target = tmp_path / "obj.pt"
        torch.save({"payload": _Marker(1)}, target)
        with pytest.warns(RuntimeWarning, match="weights_only=False"):
            loaded = safe_torch_load(target, allow_unsafe_pickle=True)
        assert loaded["payload"] == _Marker(1)

    def test_safe_read_is_attempted_first(self, tmp_path, monkeypatch) -> None:
        """A tensors-only checkpoint must NOT trip the unsafe path or warn.

        This is what makes `allow_unsafe_pickle=True` at the call sites honest:
        it permits a fallback, it does not force one, so the warning identifies
        exactly which checkpoints still genuinely need the unsafe reader.
        """
        monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
        target = tmp_path / "ckpt.pt"
        torch.save({"w": torch.zeros(2)}, target)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            loaded = safe_torch_load(target, allow_unsafe_pickle=True)
        assert torch.equal(loaded["w"], torch.zeros(2))
