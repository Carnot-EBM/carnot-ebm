import re

with open('docs/index.html', 'r') as f:
    content = f.read()

cards = """      <div class="r-card">
        <span class="r-tag">Hardware &mdash; KV260 Vivado Synthesis</span>
        <h3 class="r-title">Potts sim and RTL export complete</h3>
        <div class="r-meter"><div class="r-bar" style="width: 100%; background: #0ea5e9;"></div></div>
        <div class="r-stats"><span class="r-before">KV260 synthesizable Verilog export for Potts model</span> <span class="r-after">Exps 1692/1693</span></div>
      </div>
      <div class="r-card">
        <span class="r-tag">Full Pipeline &mdash; SOTA Integration</span>
        <h3 class="r-title">GloroKAN + Eidoku + FR11 verified</h3>
        <div class="r-meter"><div class="r-bar" style="width: 100%; background: #22c55e;"></div></div>
        <div class="r-stats"><span class="r-before">End-to-end evaluation with Continual Learning and Dynamic Extract</span> <span class="r-after">Exps 1707/1720</span></div>
      </div>
      <div class="r-card">
        <span class="r-tag">CIKAN &mdash; Constraint-Informed KAN</span>
        <h3 class="r-title">FourierCSP constraint compiled</h3>
        <div class="r-meter"><div class="r-bar" style="width: 100%; background: #22c55e;"></div></div>
        <div class="r-stats"><span class="r-before">Preserved through toy residual training</span> <span class="r-after">Exp 1723</span></div>
      </div>
      <div class="r-card">
        <span class="r-tag">EqM Sampler &mdash; GPU Integration</span>
        <h3 class="r-title">Converged faster on GPU</h3>
        <div class="r-meter"><div class="r-bar" style="width: 100%; background: #22c55e;"></div></div>
        <div class="r-stats"><span class="r-before">Equilibrium Matching (EqM) Gradient Sampler evaluated</span> <span class="r-after">Exp 1740</span></div>
      </div>
      <div class="r-card">
        <span class="r-tag">Milestone .134 &mdash; Phase 4 synthesis</span>
        <h3 class="r-title">All tasks completed; RTX 3090s idle</h3>
        <div class="r-meter"><div class="r-bar" style="width: 100%; background: #f59e0b;"></div></div>
        <div class="r-stats"><span class="r-before">Analyzed 1388 min wall time / 253 experiments</span> <span class="r-after">Exp 1745</span></div>
      </div>
"""

pattern = r'(    </div>\n  </div>\n</section>\n\n<section id="preprint">)'
if re.search(pattern, content):
    content = re.sub(pattern, cards + r'\1', content)
    with open('docs/index.html', 'w') as f:
        f.write(content)
    print("Success")
else:
    print("Failed to find insertion point")