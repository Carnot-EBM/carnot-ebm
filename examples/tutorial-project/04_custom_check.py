#!/usr/bin/env python3
"""04 — Adding a domain-specific check.

Carnot ships with checks for arithmetic, logic, and several other
built-in domains. Real applications usually have rules the built-ins
don't cover. The pattern for extending the pipeline is to implement
the ConstraintExtractor protocol and register your extractor with
the pipeline's AutoExtractor.

This script demonstrates the pattern with a small example: a
"homework-format" check that fires when the student's answer is
shorter than 20 characters (i.e., not showing their work). This is
a silly rule but the SHAPE is what matters — the same pattern works
for any domain-specific constraint your application needs.

Run it:

    JAX_PLATFORMS=cpu python 04_custom_check.py
"""

from carnot.pipeline import AutoExtractor, ConstraintResult, VerifyRepairPipeline


class HomeworkFormatExtractor:
    """A custom extractor that flags answers shorter than 20 characters.

    Implements the ConstraintExtractor protocol:
      - `supported_domains: list[str]` — which domain labels this
        extractor responds to. The pipeline routes only matching domains
        to your extractor.
      - `extract(text: str, domain: str | None = None) -> list[ConstraintResult]`
        — parse the response and emit one ConstraintResult per logical
        claim. Each carries a constraint_type, a description, and a
        metadata dict with at minimum `satisfied: bool`.
    """

    @property
    def supported_domains(self) -> list[str]:
        return ["homework_format"]

    def extract(self, text: str, domain: str | None = None) -> list[ConstraintResult]:
        # Standard guard: if a specific domain was requested and it's
        # not ours, return nothing.
        if domain is not None and domain not in self.supported_domains:
            return []

        stripped = text.strip()
        satisfied = len(stripped) >= 20
        return [
            ConstraintResult(
                constraint_type="homework_format",
                description=f"answer length >= 20 chars (got {len(stripped)})",
                metadata={"satisfied": satisfied, "length": len(stripped)},
            )
        ]


def main() -> None:
    # Build a pipeline that includes both the built-in arithmetic check
    # AND our custom homework-format check. The AutoExtractor pattern
    # lets us register a custom extractor alongside the built-ins.
    extractor = AutoExtractor()
    extractor.add_extractor(HomeworkFormatExtractor())

    pipeline = VerifyRepairPipeline(extractor=extractor, domains=["arithmetic", "homework_format"])

    question = "What is 47 + 28?"

    # Example A: terse correct answer (passes arithmetic, fails format).
    print("=" * 60)
    print("  A: Terse but correct answer")
    print("=" * 60)
    short = "75."
    result = pipeline.verify(question, short)
    print(f"  Q: {question}")
    print(f"  A: {short}")
    print(f"  verified: {result.verified}")
    for c in result.constraints:
        satisfied = c.metadata.get("satisfied", "n/a")
        print(f"     [{satisfied}] {c.constraint_type}: {c.description}")

    # Example B: full answer (passes both).
    print()
    print("=" * 60)
    print("  B: Full answer showing work")
    print("=" * 60)
    full = "47 + 28 = 75. Adding the tens places gives 60, then 7 + 8 = 15, so 60 + 15 = 75."
    result = pipeline.verify(question, full)
    print(f"  Q: {question}")
    print(f"  A: {full}")
    print(f"  verified: {result.verified}")
    for c in result.constraints:
        satisfied = c.metadata.get("satisfied", "n/a")
        print(f"     [{satisfied}] {c.constraint_type}: {c.description}")

    print()
    print("Notice: the homework_format extractor catches the terse answer (case A)")
    print("even though the arithmetic is correct. The full answer (case B) passes")
    print("both checks. Real applications add custom extractors for whatever the")
    print("domain requires — units consistency, schema compliance, attribution,")
    print("style guidelines, etc. The protocol is small and the registration")
    print("pattern is the same as above.")


if __name__ == "__main__":
    main()
