import re

def update_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements.items():
        content = content.replace(old, new)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

replacements_global = {
    "174 artifact-backed completed milestone records through 2026.05.160": "175 artifact-backed completed milestone records through 2026.05.161",
    "milestone 2026.05.160 on 2026-05-13": "milestone 2026.05.161 on 2026-05-13",
    "174 Archived Milestone Records": "175 Archived Milestone Records",
    "174\nartifact-backed": "175\nartifact-backed",
    "2,213 task records in 174": "2,223 task records in 175",
    "2,213** task records across **174** artifact-backed completed milestone records through 2026.05.160": "2,223** task records across **175** artifact-backed completed milestone records through 2026.05.161",
    "extend through milestone .160": "extend through milestone .161",
}

replacements_index = {
    '<div class="stat"><div class="stat-num">174</div><div class="stat-label">archived records through .160</div></div>': '<div class="stat"><div class="stat-num">175</div><div class="stat-label">archived records through .161</div></div>',
    '<div class="stat"><div class="stat-num">28/28</div><div class="stat-label">experiments completed in .160</div></div>': '<div class="stat"><div class="stat-num">25/25</div><div class="stat-label">experiments completed in .161</div></div>',
    '<h3 class="r-title">Milestone 2026.05.160 Operational Retrospective</h3>': '<h3 class="r-title">Milestone 2026.05.161 Operational Retrospective</h3>',
    '<p class="r-desc">Milestone .160 completed 28 experiments in 92.5 minutes. GPU utilization on the 3 compute-bound tasks was efficient, and no anomalous idling was flagged. Synthesis-only tasks remain the primary bottleneck.</p>': '<p class="r-desc">Milestone .161 completed 25 experiments in 118.9 minutes. GPU utilization on the 3 compute-bound tasks was efficient, and no anomalous idling was flagged. Synthesis-only tasks remain the primary bottleneck.</p>'
}

replacements_tr = {
    'Milestone .160 completed\n**13** synthesis-only experiments': 'Milestone .161 completed\n**25** experiments',
    'archived through 2026.05.160': 'archived through 2026.05.161',
    'extend through Exp 2038, the .159': 'extend through Exp 2065, the .161',
    '### 4.18 Recent Additions (Milestone .160)': '### 4.18 Recent Additions (Milestone .161)',
    'The Milestone .160 operational retrospective measured 92.5 minutes of wall time across 28 experiments. GPU utilization on the 3 compute-bound tasks was efficient, and no anomalous idling was flagged. Synthesis-only tasks remain the primary bottleneck for optimization.': 'The Milestone .161 operational retrospective measured 118.9 minutes of wall time across 25 experiments. GPU utilization on the 3 compute-bound tasks was efficient, and no anomalous idling was flagged. Synthesis-only tasks remain the primary bottleneck for optimization.'
}

update_file("README.md", replacements_global)

full_index_replacements = replacements_global.copy()
full_index_replacements.update(replacements_index)
update_file("docs/index.html", full_index_replacements)

full_tr_replacements = replacements_global.copy()
full_tr_replacements.update(replacements_tr)
update_file("docs/technical-report.md", full_tr_replacements)

with open("docs/technical-report.md", "r", encoding="utf-8") as f:
    tr_content = f.read()

new_findings = """
## Milestone 161 — DTM Thermodynamic Model and Soft Bellman Equation Solver (Exps 2053–2065, May 2026)

**Soft Bellman Equation Solver**
Experiment 2056 implemented a soft Bellman equation solver.

**DTM Thermodynamic Model**
Experiment 2060 explored the DTM Thermodynamic Model.

**Unsupervised System 2 Pretraining**
Experiment 2062 evaluated Unsupervised System 2 pretraining.

**Kona-Style Reasoning Benchmark**
Experiment 2063 ran a Kona-style reasoning benchmark.
"""

if "## Milestone 161" not in tr_content:
    tr_content = tr_content.replace(
        "## Milestones 159–160 — Continuous Execution and Architecture Audits (Exps 2028–2052, May 2026)",
        new_findings + "\n## Milestones 159–160 — Continuous Execution and Architecture Audits (Exps 2028–2052, May 2026)"
    )
    with open("docs/technical-report.md", "w", encoding="utf-8") as f:
        f.write(tr_content)

print("Updates applied.")
