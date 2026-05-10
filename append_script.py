import re

# Append to docs/roadmap.md
with open('docs/roadmap.md', 'r') as f:
    roadmap = f.read()

roadmap_new = roadmap + "\n| 2026.05.133 | Operational Efficiency Analysis | 1603-1711+ | 1483 min wall time; GPUs correctly idle; Pre-gate blocks bottleneck; fail-fast needed |\n"

with open('docs/roadmap.md', 'w') as f:
    f.write(roadmap_new)

# Prepend to ops/changelog.md
with open('ops/changelog.md', 'r') as f:
    changelog = f.read()

new_changelog_entry = """# Carnot — Changelog

## 2026-05-10 (Milestone 2026.05.133 Operational Retrospective)

- 2026-05-10 20:58 UTC: Milestone 2026.05.133 operational retrospective complete. Analyzed 1483 min wall time / 280 experiments (avg 5 min). Slowest paths: Exp 1603 (88 min), Exp 1657 (57 min), Pre-gate block Exp 1711 (56 min), Exp 1642 (54 min). Doomed-rerun blocks correctly failed fast. Both RTX 3090s were completely idle at 0% utilization throughout, which is correct behavior as there were no compute-bound tasks. Estimated savings: implement fail-fast for pre-gate blocks.
"""

changelog_new = changelog.replace("# Carnot — Changelog\n", new_changelog_entry)

with open('ops/changelog.md', 'w') as f:
    f.write(changelog_new)
