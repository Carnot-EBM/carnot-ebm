import sys

def update_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

index_reps = [
    ('2,448</div><div class="stat-label">Experiment records through Exp 2089', '2,496</div><div class="stat-label">Experiment records through Exp 2109'),
    ('2,443</div><div class="stat-label">Experiment records through Exp 2109', '2,496</div><div class="stat-label">Experiment records through Exp 2109'),
    ('176</div><div class="stat-label">archived records through .163', '186</div><div class="stat-label">archived records through .172'),
    ('179</div><div class="stat-label">archived records through .166', '186</div><div class="stat-label">archived records through .172'),
    ('22/22</div><div class="stat-label">experiments completed in .166', '6/6</div><div class="stat-label">experiments completed in .172'),
    ('10/10</div><div class="stat-label">experiments completed in .166', '6/6</div><div class="stat-label">experiments completed in .172'),
    ('Milestone .166 completed 10 experiments', 'Milestone .172 completed 6 experiments'),
    ('24,472</div><div class="stat-label">Python test items collected', '24,678</div><div class="stat-label">Python test items collected'),
    ('24,584</div><div class="stat-label">Python test items collected', '24,678</div><div class="stat-label">Python test items collected'),
]

update_file('docs/index.html', index_reps)

with open('docs/index.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

new_card = """
      <div class="r-card">
        <span class="r-tag">Continuous Learning &amp; Verification</span>
        <h3 class="r-title">EBFT &amp; CASAL Sampler Integration</h3>
        <p class="r-desc">EBFT continuous self-learning loop and CASAL Primal-Dual sampler implementation on Phase 4 substrate provide empirical grounding for near-critical verification scaling.</p>
        <div class="r-stats"><span class="r-before">Prior empirical bounds extended</span> <span class="r-after">Exp 1688&ndash;1698</span></div>
      </div>
"""

target = '      <div class="r-card">\n        <span class="r-tag">Hardware Profiling</span>'
if "EBFT &amp; CASAL Sampler Integration" not in html_content:
    html_content = html_content.replace(target, new_card + target, 1)

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Updated docs/index.html successfully.")
