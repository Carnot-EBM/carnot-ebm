import re

with open('docs/index.html', 'r') as f:
    content = f.read()

new_cards = """      <div class="r-card">
        <span class="r-tag">Capstone &mdash; E2E Pipeline</span>
        <h3 class="r-title">Qwen and Gemma evaluated</h3>
        <div class="r-meter"><div class="r-bar" style="width: 100%; background: #22c55e;"></div></div>
        <div class="r-stats"><span class="r-before">Capstone E2E evaluation finished</span> <span class="r-after">Exps 1782/1783</span></div>
      </div>
      <div class="r-card">
        <span class="r-tag">Latent Modeling &mdash; Constraint Optimization</span>
        <h3 class="r-title">100% new code coverage</h3>
        <div class="r-meter"><div class="r-bar" style="width: 100%; background: #22c55e;"></div></div>
        <div class="r-stats"><span class="r-before">Targeted tests passed</span> <span class="r-after">Exp 1771</span></div>
      </div>
      <div class="r-card">
        <span class="r-tag">Milestone .137 &mdash; Phase 4 operations</span>
        <h3 class="r-title">Retrospective complete</h3>
        <div class="r-meter"><div class="r-bar" style="width: 100%; background: #f59e0b;"></div></div>
        <div class="r-stats"><span class="r-before">Phase 4 operations aggregated</span> <span class="r-after">Exp 1784</span></div>
      </div>
"""

# Let's find the closing of the cards section.
# The grid has id #results
pattern = r'(    </div>\n  </div>\n</section>\n\n<section id="preprint">)'
if re.search(pattern, content):
    content = re.sub(pattern, new_cards + r'\1', content)
    with open('docs/index.html', 'w') as f:
        f.write(content)
    print("Cards added successfully.")
else:
    print("Failed to find insertion point.")
