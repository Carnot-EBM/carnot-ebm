import re

with open('docs/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Update the results card for the latest milestone
content = content.replace("13/13", "28/28")
content = content.replace("experiments completed in .158", "experiments completed in .160")

# Update the desc
content = content.replace(
    "Milestone .158 completed 13 experiments in 26.5 minutes. All experiments were synthesis-only (0 GPU usage), resulting in an average duration of 2 minutes per experiment. The doomed-rerun block successfully saved time.",
    "Milestone .160 completed 28 experiments in 92.5 minutes. GPU utilization on the 3 compute-bound tasks was efficient, and no anomalous idling was flagged. Synthesis-only tasks remain the primary bottleneck."
)

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(content)
