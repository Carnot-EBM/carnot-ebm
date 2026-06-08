"""GeneratorSpec — decouple the RFT harness from any single base model.

WHY (operator concern 2026-06-08): the verifier-as-self-improvement-reward experiments
used MiniCPM5-1B as a convenient small/fast generator, and the generation config grew
MiniCPM-specific quirks inline in two scripts (stop on <|im_end|> not just </s>;
enable_thinking=False because MiniCPM5 over-thinks GSM8K). That is fine for a TEST
generator, but the project must NOT be coupled to it — the generator is the swappable
commodity part (decentralization rule 1; hybrid-pragmatic-architecture); the VERIFIER is
the product. This module makes the base model a ONE-LINE swap: pick a GeneratorSpec.

A GeneratorSpec captures the *only* things that differ between transformers-loadable
chat generators for our harness: the model id, whether to enable "thinking", and which
turn-ender token(s) to stop on (beyond the tokenizer eos). resolve_stop_ids() and
format_prompt() take a live tokenizer so the same spec works regardless of vocab.

NOTE: this is the TRANSFORMERS path (AutoModelForCausalLM + AutoTokenizer), used by the
LoRA-SFT RFT harness. The SOTA *-GGUF repos (Qwen3.6-35B-GGUF, gemma-4-*-GGUF) ship NO
HF tokenizer and load via llama.cpp (see CLAUDE.md GGUF rule) — they are a separate
loader path, not a GeneratorSpec target here. For THIS harness, swap among
transformers-loadable models (MiniCPM5-1B, small Qwen/Gemma base safetensors, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GeneratorSpec:
    model_id: str
    # MiniCPM5 (and other reasoning models) over-think short GSM8K and ramble past the
    # answer -> 100% truncation. Most chat models ignore this kwarg harmlessly.
    enable_thinking: bool = False
    # turn-ender tokens the model emits to end a reply, BEYOND tok.eos_token. The classic
    # bug: a chat model ends on <|im_end|> but tok.eos_token is </s>, so passing only eos
    # means generation never stops. Listing the chat turn-enders here fixes that per-model.
    extra_stop_tokens: tuple[str, ...] = ("<|im_end|>", "<|endoftext|>", "<end_of_turn>")
    # plain-text fallback prompt if the tokenizer has no chat template. {q} = question.
    fallback_template: str = "Question: {q}\nSolution:"
    # the instruction wrapped around the question in the chat 'user' turn.
    user_instruction: str = "Solve the problem. End with the final number.\n\n{q}"

    def resolve_stop_ids(self, tok) -> list[int]:
        """All token ids that should halt generation, for THIS tokenizer."""
        ids: list[int] = []
        if tok.eos_token_id is not None:
            ids.append(int(tok.eos_token_id))
        unk = getattr(tok, "unk_token_id", None)
        for t in self.extra_stop_tokens:
            tid = tok.convert_tokens_to_ids(t)
            # convert_tokens_to_ids returns unk_id for unknown tokens -> skip those.
            if isinstance(tid, int) and tid >= 0 and tid != unk:
                ids.append(tid)
        return sorted(set(ids))

    def format_prompt(self, tok, question: str) -> str:
        """Render the question into the model's chat format (or fallback)."""
        msgs = [{"role": "user", "content": self.user_instruction.format(q=question)}]
        try:
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking)
        except Exception:
            return self.fallback_template.format(q=question)


# Known specs. Add an entry only when a model needs NON-default config; the default
# (im_end/endoftext/end_of_turn stop set + no-think) already covers most chat models,
# so an UNLISTED model still swaps in cleanly via for_model().
KNOWN_SPECS: dict[str, GeneratorSpec] = {
    "openbmb/MiniCPM5-1B": GeneratorSpec(
        model_id="openbmb/MiniCPM5-1B", enable_thinking=False,
        extra_stop_tokens=("<|im_end|>",)),
}


def for_model(model_id: str) -> GeneratorSpec:
    """Return the spec for a model id; unlisted models get the safe default spec.

    Swapping the base model is therefore a one-liner: change MODEL_ID. If the new model
    needs special handling (an unusual turn-ender, thinking on), add a KNOWN_SPECS entry.
    """
    return KNOWN_SPECS.get(model_id, GeneratorSpec(model_id=model_id))
