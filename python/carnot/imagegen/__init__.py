"""Local, open-weight image-generation adapters (REQ-PUBLISH-042).

**Researcher summary:**
    This package exists so that generating a figure for a paper never
    requires a paid closed-weight API call (Gemini image-gen, GPT-Image,
    Claude, etc.). It contains a local HTTP shim that speaks just enough of
    the OpenAI Images API surface for the vendored ``external/paperbanana``
    tool to treat it as a drop-in image-gen provider, backed by a genuinely
    local, open-weight diffusion model (``baidu/ERNIE-Image``, Apache-2.0).

**Detailed explanation for engineers:**
    ``paperbanana`` (the diagram-generation tool this project vendors under
    ``external/paperbanana``, see CLAUDE.md "Audit untrusted code") already
    supports pointing its ``openai_imagen`` provider at an arbitrary
    ``OPENAI_BASE_URL``. ERNIE-Image itself has no such HTTP surface — it is
    a plain Diffusers pipeline. ``ernie_image_server`` is the missing piece:
    a small FastAPI process that loads the Diffusers pipeline once and
    exposes ``POST /v1/images/generations`` in the exact shape paperbanana's
    OpenAI SDK client expects. paperbanana's own source is never patched;
    the adapter lives entirely on our side of the network boundary, which is
    the same "vendor adapter through abstract protocol" shape CLAUDE.md's
    Decentralization-Respecting Design Constraints rule 7 already mandates
    for closed-weight LLM integrations (``SamplerBackend``, ``LLMComponent``).

Spec: REQ-PUBLISH-042
"""
