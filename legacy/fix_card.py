with open("docs/index.html", "r") as f:
    html = f.read()

old_card = """<h3 class="r-title">Milestone 2026.05.166 Operational Retrospective</h3>
        <p class="r-desc">Milestone .172 completed 6 experiments in 41 minutes. All 10 tasks were synthesis-only, so GPUs correctly idled at 0% utilization throughout. The slowest paths were purely synthesis tasks, remaining the primary bottleneck for optimization.</p>
        <div class="r-stats"><span class="r-before">Analyzed 41 min wall time</span> <span class="r-after">Exp 2114</span></div>"""

new_card = """<h3 class="r-title">Milestone 2026.05.176 Operational Retrospective</h3>
        <p class="r-desc">Milestone .176 completed 10 experiments in 19.3 minutes. GPU utilization on the single compute-bound task was efficient, and no anomalous idling was flagged. The slowest path was a synthesis task, remaining the primary bottleneck for optimization.</p>
        <div class="r-stats"><span class="r-before">Analyzed 19.3 min wall time</span> <span class="r-after">Exp 2114</span></div>"""

if old_card in html:
    html = html.replace(old_card, new_card)
    with open("docs/index.html", "w") as f:
        f.write(html)
    print("Card updated successfully.")
else:
    print("Card NOT FOUND!")
