import re

with open('docs/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Update the results card for the latest milestone
content = content.replace("28/28", "23/23")
content = content.replace("experiments completed in .160", "experiments completed in .162")

# Update the desc
old_desc = "Milestone .160 completed 28 experiments in 92.5 minutes. GPU utilization on the 3 compute-bound tasks was efficient, and no anomalous idling was flagged. Synthesis-only tasks remain the primary bottleneck."
new_desc = "Milestone .162 completed 23 experiments in 45.2 minutes. GPU utilization on the 3 compute-bound tasks was efficient, and no anomalous idling was flagged. Doomed-rerun blocks successfully saved time."
content = content.replace(old_desc, new_desc)

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated HTML card.")
