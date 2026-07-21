"""Tests for the local ERNIE-Image OpenAI-Images-API-compatible shim.

Spec coverage: REQ-PUBLISH-042, SCENARIO-PUBLISH-042

None of these tests load the real 8B-parameter model or touch a GPU/network
-- ErniePipelineSingleton.get is monkeypatched to a deterministic stub, and
ernie_image_cached is exercised both for real (no network call, pure local
cache scan) and monkeypatched for the "server not yet started" precondition
path.
"""

from __future__ import annotations

import base64
import io

from PIL import Image

from carnot.imagegen.ernie_image_server import (
    ErniePipelineSingleton,
    build_app,
    ernie_image_cached,
    parse_openai_size,
)


class _StubResult:
    def __init__(self, image: Image.Image) -> None:
        self.images = [image]


class _StubPipeline:
    """Deterministic stand-in for the real Diffusers pipeline."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, prompt, width, height, guidance_scale, num_inference_steps):
        self.calls.append(
            {
                "prompt": prompt,
                "width": width,
                "height": height,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
            }
        )
        return _StubResult(Image.new("RGB", (width, height), color=(10, 20, 30)))


class TestParseOpenAISize:
    """REQ-PUBLISH-042: size-string parsing matches paperbanana's OpenAIImageGen contract."""

    def test_native_square(self):
        assert parse_openai_size("1024x1024") == (1024, 1024)

    def test_landscape(self):
        assert parse_openai_size("1536x1024") == (1536, 1024)

    def test_portrait(self):
        assert parse_openai_size("1024x1536") == (1024, 1536)

    def test_unknown_size_falls_back_to_native(self):
        """SCENARIO-PUBLISH-042: a malformed/unexpected size degrades to a working image."""
        assert parse_openai_size("garbage") == (1024, 1024)
        assert parse_openai_size("") == (1024, 1024)
        assert parse_openai_size("9999x9999") == (1024, 1024)


class TestErnieImageCached:
    """REQ-PUBLISH-042: the precondition check never raises, even off-network."""

    def test_returns_bool(self):
        # Real call against the local HF cache -- no network required by
        # scan_cache_dir, and the function must never raise regardless of
        # whether the model happens to be cached on this machine.
        assert isinstance(ernie_image_cached(), bool)

    def test_survives_scan_cache_dir_failure(self, monkeypatch):
        import huggingface_hub

        def _boom():
            raise RuntimeError("cache dir unreadable")

        monkeypatch.setattr(huggingface_hub, "scan_cache_dir", _boom)
        assert ernie_image_cached() is False


class TestGenerationsEndpoint:
    """REQ-PUBLISH-042: the shim matches paperbanana's OpenAIImageGen request/response contract.

    Contract (from external/paperbanana/paperbanana/providers/image_gen/openai_imagen.py):
    POST /v1/images/generations {model, prompt, n=1, size, quality?}
        -> {"data": [{"b64_json": <base64 PNG>}]}
    """

    def setup_method(self):
        ErniePipelineSingleton.reset()

    def teardown_method(self):
        ErniePipelineSingleton.reset()

    def _client(self, monkeypatch, stub: _StubPipeline):
        from fastapi.testclient import TestClient

        monkeypatch.setattr(ErniePipelineSingleton, "get", classmethod(lambda cls, gpu=None: stub))
        app = build_app(gpu=None)
        return TestClient(app)

    def test_returns_b64_json_matching_requested_size(self, monkeypatch):
        stub = _StubPipeline()
        client = self._client(monkeypatch, stub)

        resp = client.post(
            "/v1/images/generations",
            json={
                "model": "ernie-image",
                "prompt": "a methodology diagram",
                "n": 1,
                "size": "1024x1536",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body and len(body["data"]) == 1
        b64 = body["data"][0]["b64_json"]
        image = Image.open(io.BytesIO(base64.b64decode(b64)))
        assert image.size == (1024, 1536)
        assert image.format == "PNG"

    def test_forwards_prompt_and_size_to_pipeline(self, monkeypatch):
        stub = _StubPipeline()
        client = self._client(monkeypatch, stub)

        client.post(
            "/v1/images/generations",
            json={
                "model": "ernie-image",
                "prompt": "distinctive-prompt-xyz",
                "n": 1,
                "size": "1536x1024",
            },
        )

        assert len(stub.calls) == 1
        assert stub.calls[0]["prompt"] == "distinctive-prompt-xyz"
        assert stub.calls[0]["width"] == 1536
        assert stub.calls[0]["height"] == 1024

    def test_n_greater_than_one_is_rejected(self, monkeypatch):
        """paperbanana's OpenAIImageGen always sends n=1; a different n means an
        unexpected caller, so refuse rather than silently return only one image."""
        stub = _StubPipeline()
        client = self._client(monkeypatch, stub)

        resp = client.post(
            "/v1/images/generations",
            json={"model": "ernie-image", "prompt": "x", "n": 2, "size": "1024x1024"},
        )

        assert resp.status_code == 400
        assert stub.calls == []

    def test_healthz_reports_cache_state(self, monkeypatch):
        stub = _StubPipeline()
        client = self._client(monkeypatch, stub)
        monkeypatch.setattr("carnot.imagegen.ernie_image_server.ernie_image_cached", lambda: True)

        resp = client.get("/healthz")

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "model_cached": True}


class TestErniePipelineSingletonPrecondition:
    """REQ-PUBLISH-042: attempting to load an uncached model fails honestly, never silently."""

    def setup_method(self):
        ErniePipelineSingleton.reset()

    def teardown_method(self):
        ErniePipelineSingleton.reset()

    def test_raises_runtime_error_when_not_cached(self, monkeypatch):
        monkeypatch.setattr("carnot.imagegen.ernie_image_server.ernie_image_cached", lambda: False)

        try:
            ErniePipelineSingleton.get(gpu=None)
            raised = False
        except RuntimeError as exc:
            raised = True
            assert "baidu/ERNIE-Image" in str(exc)
            assert "huggingface-cli download" in str(exc)

        assert raised, "expected RuntimeError when the model is not cached"
