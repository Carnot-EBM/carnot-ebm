# RAGognize annotation authority audit, 2026-11-11

The public release supports a completed pipeline audit. Its released authority is
machine annotation. Natural response errors do not establish independent human
truth. The terminal scientific verdict is `complete_null_machine_annotation_only`
when owned validation passes. This is not a source-access block.

The audit pins [author code](https://github.com/F4biian/RAGognizer/tree/1d3e8fdd6de638dc2d06265829a4d9ced387e0ba)
and the [dataset release](https://huggingface.co/datasets/F4biian/RAGognize/tree/aab54518c2a7c0d25fff8bffbf5337d0321de142).
The retained manifest records every inspected file's SHA256. Both parquet files
fit the bounded read-only acquisition: 76,699,735 compressed bytes. The final
retained acquisition, including code and metadata, transferred 76,964,000 bytes.
Earlier metadata probes also fit the total256MiB and600-second limits. No author
script was executed. No new access agreement was accepted. No model was loaded,
called, trained or used to replace a released answer or label.

The population is all1,842 train rows and2,781 test rows. The4 released response
models yield18,492 inspected responses over2,359 article/question groups. These
are sampling groups, not independent human decisions. There are13,193 responses
without a released hallucination span and5,299 with one. There are50 responses
with token-start bounds failures; the audit retains them. All6,996 released
character spans match their response text. No original article or question
crosses train/test; exact response and reference overlaps are retained separately. A failure does not
change the released label or claim that the answer is factually wrong.

The code license is MIT. The dataset card declares CC BY-SA4.0 for the dataset
and its Wikipedia basis. Article URLs and Wikipedia revision IDs remain in each
source row. Those declarations support local release inspection with attribution.
They do not establish blanket permission to train on or redistribute every
response model's outputs. Separate model terms remain unresolved for those
future uses. No new permission is inferred from public download access.

The annotation chain is explicit in the pinned code:

- `ragognize/02_qa_generation/generate_user_prompts.py` configures
  `gemini-2.5-pro`. `user_prompt_generator.py` requests the question, golden
  answer and cluelessness answer from a Wikipedia passage. The dataset card
  separately identifies Gemini as the golden-answer generator.
- `ragognize/04_llm_responses/annotate_outputs.py` configures
  `gemini-2.5-flash` at temperature0. It supplies the released RAG prompt and
  generated response to `SHD_SimpleChatTokenLevel`. It does not supply the
  golden answer as an independent adjudication source.
- `ragognize/utils/shd_SimpleChatTokenLevel.py` asks for closed-context
  faithfulness, analysis text, whole/partial hallucination flags, cluelessness,
  request adherence and quoted spans. All those annotation decisions are
  generated. The result fields and span-validity flags are deterministic
  projections or substring checks over generated annotations.
- `ragognize/04_llm_responses/merge.py` copies those annotations into the release
  and filters records whose annotation checks fail. Released rows therefore
  follow a machine-dependent selection process. Historical rejected examples
  are not reconstructed or inferred from the released subset.

No item-level human decisions, reviewer identities, disagreement votes or human
adjudication record were found in the inspected card and author pipeline. This
is missing evidence. It is not proof that no human ever inspected any item.
Machine-generated reference answers and annotations cannot satisfy that missing
human authority. No labels were independently reassessed in this task.

Span `start` and `end` use Python character indices, with an exclusive end. Each
span is checked against the exact response string and its UTF8 encoding; derived
byte offsets and response-byte hashes are retained. These are not token indices.
`utils/llm.py` derives `token_starts` from lengths of decoded prefixes with special
tokens, while the released response omits special tokens. The audit checks
bounds and ordering without loading a tokenizer. It does not certify token IDs
or normalize mismatches away. `all_valid` describes the author substring checks;
it is not evidence of factual correctness.

Eligible uses are release inspection, offset and duplicate audits, pipeline
reproducibility, and machine-supervised development after checking applicable
terms. An annotation-relative experiment must name the machine annotator and
shared source dependence. Forbidden headline uses are independent human
factuality accuracy, an oracle-distinct verifier win, a blind evaluation, or
claims about newly generated Carnot/Qwen answers. No verifier was evaluated.

Before future fitting, group all model and answerability variants by original
article and question. Merge clusters that share articles, questions, retrieved
contexts, reference material or exact responses across splits. Assign whole
clusters before any fit or threshold choice. Retain dates and source revisions;
exclude golden answers, hallucination spans, annotation analyses, answerability,
answer quotes and other label-bearing construction metadata from inference.
Public test exposure remains exposure after a new random split.

Independent evaluation needs documented human decisions and adjudication,
usable original text, verified permission for the intended use, independent
factual evidence, and at least200 independent question groups supporting both
labels. The independent-human group count here is0. Source-access readiness,
annotation-audit readiness and independent-label readiness are separate fields.

LPHB exposure limits and HalluTruthQA construction limits remain unchanged.
AuthenHallu remains blocked on its unchanged original-source access. This task
does not retry that join. It has no downstream natural-model task and cannot
cascade-block graph science. Existing operator-curated pages remain untouched.

The full repository suite was invoked once. It collected84,293 tests, displayed
failures and reached its300-second bound without completion. Its receipt and
logs are retained separately from owned validation. The owned validation report
records actual unit and child statement coverage, Ruff, strict mypy, explicit
spec tracing, private E2E-018/021, cold replay and unmodified terminal scanners.
