#!/usr/bin/env python3
"""Create a domain-specific constraint extractor for units-of-measure checking.

This example shows how to build a custom ConstraintExtractor that plugs
into Carnot's verification pipeline. The example extractor checks whether
LLM responses use consistent units of measurement (e.g., mixing kg and
pounds, or meters and feet without conversion).

Use case: You have domain-specific rules that the built-in extractors
don't cover. Implement the ConstraintExtractor protocol and register your
extractor with AutoExtractor to extend the pipeline.

Usage:
    JAX_PLATFORMS=cpu python examples/custom_extractor.py
"""

from __future__ import annotations

import re
import sys


def main() -> int:
    try:
        from carnot.pipeline import (
            AutoExtractor,
            ConstraintResult,
            VerifyRepairPipeline,
        )
    except ImportError:
        print("ERROR: carnot is not installed. Run: pip install -e '.[dev]'")
        return 1

    # --- Step 1: Define a custom extractor ---

    # Groups of units that should not be mixed without explicit conversion.
    UNIT_GROUPS = {
        "length_metric": {"m", "cm", "mm", "km", "meter", "meters", "centimeter", "centimeters", "kilometer", "kilometers"},
        "length_imperial": {"ft", "in", "mi", "yard", "yards", "foot", "feet", "inch", "inches", "mile", "miles"},
        "mass_metric": {"kg", "g", "mg", "gram", "grams", "kilogram", "kilograms"},
        "mass_imperial": {"lb", "lbs", "oz", "pound", "pounds", "ounce", "ounces"},
        "temp_celsius": {"celsius", "°c"},
        "temp_fahrenheit": {"fahrenheit", "°f"},
    }

    # Which groups conflict with each other.
    CONFLICTS = [
        ("length_metric", "length_imperial"),
        ("mass_metric", "mass_imperial"),
        ("temp_celsius", "temp_fahrenheit"),
    ]

    class UnitsExtractor:
        """Check for mixed unit systems in LLM responses.

        Implements the ConstraintExtractor protocol so it plugs directly
        into Carnot's AutoExtractor and VerifyRepairPipeline.
        """

        @property
        def supported_domains(self) -> list[str]:
            return ["units"]

        def extract(
            self, text: str, domain: str | None = None
        ) -> list[ConstraintResult]:
            if domain is not None and domain not in self.supported_domains:
                return []

            # Find all units mentioned in the text.
            text_lower = text.lower()
            words = set(re.findall(r"[a-z°]+", text_lower))

            found_groups: dict[str, set[str]] = {}
            for group_name, group_units in UNIT_GROUPS.items():
                matched = words & group_units
                if matched:
                    found_groups[group_name] = matched

            results: list[ConstraintResult] = []

            # Check for conflicting unit groups.
            for group_a, group_b in CONFLICTS:
                if group_a in found_groups and group_b in found_groups:
                    # Check if "convert" or "equivalent" appears, suggesting
                    # intentional cross-system usage.
                    conversion_words = {"convert", "conversion", "equivalent", "equals", "approximately"}
                    has_conversion = bool(words & conversion_words)

                    results.append(
                        ConstraintResult(
                            constraint_type="unit_consistency",
                            description=(
                                f"Mixed units: {', '.join(sorted(found_groups[group_a]))} "
                                f"({group_a}) with {', '.join(sorted(found_groups[group_b]))} "
                                f"({group_b})"
                            ),
                            metadata={
                                "group_a": group_a,
                                "group_b": group_b,
                                "units_a": sorted(found_groups[group_a]),
                                "units_b": sorted(found_groups[group_b]),
                                "has_conversion_context": has_conversion,
                                "satisfied": has_conversion,
                            },
                        )
                    )

            # If no conflicts found, report consistency as a passing constraint.
            if not results and found_groups:
                results.append(
                    ConstraintResult(
                        constraint_type="unit_consistency",
                        description=f"Consistent units: {', '.join(sorted(found_groups.keys()))}",
                        metadata={"satisfied": True, "groups": sorted(found_groups.keys())},
                    )
                )

            return results

    # --- Step 2: Use the custom extractor standalone ---
    print("=" * 60)
    print("Example 1: Custom extractor standalone")
    print("=" * 60)

    extractor = UnitsExtractor()

    text_ok = "The building is 50 meters tall and 200 meters from the river."
    constraints = extractor.extract(text_ok)
    print(f"  Text: {text_ok}")
    for c in constraints:
        print(f"    [{c.metadata.get('satisfied')}] {c.description}")

    print()
    text_mixed = "The room is 10 feet wide and 3 meters long."
    constraints = extractor.extract(text_mixed)
    print(f"  Text: {text_mixed}")
    for c in constraints:
        print(f"    [{c.metadata.get('satisfied')}] {c.description}")

    print()
    text_conversion = "10 feet is approximately equivalent to 3.048 meters."
    constraints = extractor.extract(text_conversion)
    print(f"  Text: {text_conversion}")
    for c in constraints:
        print(f"    [{c.metadata.get('satisfied')}] {c.description}")

    # --- Step 3: Register with AutoExtractor ---
    print()
    print("=" * 60)
    print("Example 2: Registered with AutoExtractor")
    print("=" * 60)

    auto = AutoExtractor()
    print(f"  Default domains: {auto.supported_domains}")

    auto.add_extractor(UnitsExtractor())
    print(f"  After registration: {auto.supported_domains}")

    text = "The package weighs 5 kg and measures 2 feet by 3 feet."
    all_constraints = auto.extract(text)
    print(f"  Text: {text}")
    print(f"  Total constraints: {len(all_constraints)}")
    for c in all_constraints:
        print(f"    [{c.constraint_type}] {c.description}")

    # --- Step 4: Plug into VerifyRepairPipeline ---
    print()
    print("=" * 60)
    print("Example 3: Full pipeline with custom extractor")
    print("=" * 60)

    custom_auto = AutoExtractor()
    custom_auto.add_extractor(UnitsExtractor())

    pipeline = VerifyRepairPipeline(extractor=custom_auto)

    question = "How tall is the Eiffel Tower?"
    response = "The Eiffel Tower is 330 meters tall, which is about 1083 feet."
    result = pipeline.verify(question, response)

    print(f"  Question: {question}")
    print(f"  Response: {response}")
    print(f"  Verified: {result.verified}")
    for c in result.constraints:
        satisfied = c.metadata.get("satisfied", "N/A")
        print(f"    [{satisfied}] {c.description}")

    print()
    print("Done. All examples completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
