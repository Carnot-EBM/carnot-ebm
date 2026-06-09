import sys
with open("openspec/capabilities/research-reporting/spec.md", "a") as f:
    f.write("\n### REQ-REPORT-3841: External Research Refresh .353\n")
    f.write("The system shall confirm the '.353 additions' section is intact in research-references.md.\n")
    f.write("The system shall append further genuinely-new 2026 papers for the .353 tracks.\n")
    f.write("The system shall output a valid artifact with honest_verdict='complete: external_research_refresh_353_section_intact_references_appended_numbers_as_reported'.\n")
    f.write("\n### SCENARIO-REPORT-3841: Refresh Appends Papers\n")
    f.write("**Given** the .353 section exists\n")
    f.write("**When** the refresh script is run\n")
    f.write("**Then** it appends the new papers and generates the JSON artifact.\n")
print("Appended to spec.md")
