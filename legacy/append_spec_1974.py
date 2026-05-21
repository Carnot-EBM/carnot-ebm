import sys

with open("openspec/capabilities/verifiable-reasoning/spec.md", "a") as f:
    f.write("\n### REQ-VERIFY-1974: Logic Extraction With Gemma\n\n")
    f.write("The system shall provide a logic extractor that maps unstructured prompts to continuous constraints.\n")
    f.write("The logic extractor MUST use the `unsloth/gemma-4-26B-A4B-it-GGUF` model to generate JSON constraints.\n\n")
    f.write("### SCENARIO-VERIFY-1974: Logic Extractor Parses Unstructured Text\n\n")
    f.write("**Given** an unstructured prompt containing implicit continuous constraints\n")
    f.write("**When** the LogicExtractor is invoked\n")
    f.write("**Then** it returns a list of ContinuousConstraint objects with type, target, and float values.\n")
