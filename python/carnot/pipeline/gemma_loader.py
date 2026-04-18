"""GemmaTransformersLoader — loads Gemma4 models via HuggingFace transformers.

**Why this module exists (RETRO-028):**
    Experiment 439 reported Gemma4-E4B-it accuracy of 0.0% on GSM8K.  Root cause:
    llama.cpp tokenizer bug (GitHub issue llama.cpp#21516) causes the model to emit
    infinite ``<unused8>`` tokens (token_id=14 in the Gemma4 tokenizer) instead of
    valid text.  The model *never* actually ran — every generated token was the
    ``<unused8>`` placeholder, producing garbage responses that scored 0%.

    Published Gemma4 accuracy on GSM8K is 75–80%.  This false-negative result blocked
    downstream experiments for an entire milestone.

**The fix:**
    Load Gemma4 models via ``AutoModelForCausalLM.from_pretrained`` (HuggingFace
    transformers), which uses the correct tokenizer implementation.  The llama.cpp
    backend is explicitly avoided — it is NOT a fallback path in this module.

**Why we validate output (REQ-LOADER-002):**
    The llama.cpp#21516 bug produces *silent* failure.  The generation call returns
    without error; every token is just ``<unused8>`` (token_id=14).  Without explicit
    validation, a caller would see "generation succeeded" but score 0% on every
    benchmark question.  ``is_valid_output`` catches this at the model-output layer so
    callers get an honest failure signal instead of misleading silence.

**What ``<unused8>`` is:**
    In the Gemma4 tokenizer vocabulary, token_id=14 corresponds to the ``<unused8>``
    placeholder.  Gemma4's vocabulary pre-allocates a block of ``<unusedN>`` tokens
    for future use.  When llama.cpp misroutes the tokenizer, the model's logit
    distribution collapses onto this placeholder.  A correct transformers-loaded model
    never generates ``<unusedN>`` in normal text.

Spec: REQ-LOADER-001, REQ-LOADER-002,
      SCENARIO-LOADER-001, SCENARIO-LOADER-002
"""

from __future__ import annotations

import re
from typing import Optional

# Pattern matching any ``<unusedN>`` token (any digits, including bare ``<unused>``).
# Gemma4's token_id=14 renders as ``<unused8>`` but we reject any all-unused string
# to be robust against future tokenizer variants.
_UNUSED_TOKEN_RE = re.compile(r"^(\s*<unused\d*>\s*)+$", re.IGNORECASE)


class GemmaTransformersLoader:
    """Load and run inference on Gemma4 models via HuggingFace transformers.

    This class is the RETRO-028 fix.  It replaces any llama.cpp-based loader for
    Gemma4 models.  llama.cpp has a confirmed tokenizer bug (issue #21516) that
    causes Gemma4 to emit only ``<unused8>`` tokens — the model appears to run but
    produces zero valid text.  Transformers does not have this bug.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. ``'google/gemma-4-E4B-it'``.
        Must contain ``'gemma'`` (case-insensitive) — this loader is intentionally
        scoped to Gemma models only.  Pass a Qwen or Llama model ID and you get
        a ``ValueError`` rather than silent mis-loading.
    device : str
        Device placement string passed to ``from_pretrained``.
        ``'auto'`` lets transformers choose GPU vs CPU automatically.
        ``'cpu'`` forces CPU (useful for diagnostic runs without GPU).

    Raises
    ------
    ValueError
        If ``model_id`` does not contain ``'gemma'`` (case-insensitive).

    Spec: REQ-LOADER-001, REQ-LOADER-002
    """

    def __init__(self, model_id: str, device: str = "auto") -> None:
        # Enforce Gemma-only scope.  This loader is specifically for Gemma4; loading
        # other model families would bypass important Gemma-specific validation.
        if "gemma" not in model_id.lower():
            raise ValueError(
                f"GemmaTransformersLoader only supports Gemma models, got: {model_id!r}. "
                "Use a model_id containing 'gemma' or 'Gemma'."
            )
        self.model_id = model_id
        self.device = device
        self._model: Optional[object] = None
        self._tokenizer: Optional[object] = None

    def load(self) -> None:
        """Load model and tokenizer via AutoModelForCausalLM.from_pretrained.

        Uses HuggingFace transformers, NOT llama.cpp.  The llama.cpp tokenizer
        bug (issue #21516) is avoided by construction — this method never touches
        any llama.cpp binding.

        Sets ``self._model`` and ``self._tokenizer`` in-place.

        Raises
        ------
        ImportError
            If ``transformers`` is not installed.
        OSError / EnvironmentError
            If the model download fails (e.g. no internet, invalid model_id).

        Spec: REQ-LOADER-001
        """
        # Late import so the module is importable even without transformers installed
        # (unit tests mock this path).
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self.device,
        )

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Run inference on ``prompt`` and return the generated text.

        The model must have been loaded via ``load()`` before calling this method.

        Parameters
        ----------
        prompt : str
            The input prompt text.
        max_new_tokens : int
            Maximum number of new tokens to generate.

        Returns
        -------
        str
            Decoded generated text (excluding the input prompt tokens).

        Raises
        ------
        RuntimeError
            If ``load()`` has not been called.

        Spec: REQ-LOADER-001
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "Model not loaded. Call GemmaTransformersLoader.load() first."
            )

        # Encode input on the model's device so we don't get device-mismatch errors.
        inputs = self._tokenizer(prompt, return_tensors="pt")

        # Move inputs to the same device as the model.  ``device_map='auto'`` may
        # place model layers across multiple devices; using model.device (first device)
        # is the standard transformers pattern.
        device = next(iter(self._model.parameters())).device  # type: ignore[union-attr]
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output_ids = self._model.generate(  # type: ignore[union-attr]
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # Decode only the newly generated tokens (not the prompt).
        prompt_len = inputs["input_ids"].shape[1]
        new_ids = output_ids[0][prompt_len:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True)  # type: ignore[union-attr]

    @staticmethod
    def is_valid_output(text: str) -> bool:
        """Return False if text consists entirely of ``<unusedN>`` tokens.

        The llama.cpp#21516 bug causes Gemma4 to emit nothing but ``<unused8>``
        (token_id=14) placeholder tokens.  This method detects that failure mode.
        A correct transformers-based model never emits ``<unusedN>`` tokens in
        normal inference.

        Parameters
        ----------
        text : str
            The generated text to validate.

        Returns
        -------
        bool
            ``False`` if ``text`` is non-empty AND consists entirely of
            ``<unusedN>`` token strings.
            ``True`` for all other non-empty strings (normal output).
            ``False`` for an empty string (no valid output produced).

        Spec: REQ-LOADER-002, SCENARIO-LOADER-001, SCENARIO-LOADER-002
        """
        if not text or not text.strip():
            # Empty output is not valid — no answer was generated.
            return False
        return not bool(_UNUSED_TOKEN_RE.match(text.strip()))
