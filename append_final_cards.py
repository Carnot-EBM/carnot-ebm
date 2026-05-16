import re

with open('docs/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

new_cards = """      <div class="r-card">
        <span class="r-tag">Continual Learning &mdash; FR-11</span>
        <h3 class="r-title">Dynamic Resolution Prototype Evaluated</h3>
        <p class="r-desc">Implemented Dynamic Resolution Continual EBM Learning with Live Data Evaluation for FR-11, later verified through a comprehensive Continuous Self-Learning Retention Audit.</p>
        <div class="r-meter"><div class="r-bar" style="width: 100%; background: #22c55e;"></div></div>
        <div class="r-stats"><span class="r-before">Retention audit completed successfully</span> <span class="r-after">Exps 1915-1979</span></div>
      </div>
      <div class="r-card">
        <span class="r-tag">Architecture &mdash; CEM</span>
        <h3 class="r-title">CEM on 3-SAT (Local SOTA)</h3>
        <p class="r-desc">Introduced the Compositional Energy Minimization (CEM) Architecture Design, validating logic via a Proof of Concept on 3-SAT employing a Local SOTA.</p>
        <div class="r-meter"><div class="r-bar" style="width: 100%; background: #22c55e;"></div></div>
        <div class="r-stats"><span class="r-before">Proof of Concept functional</span> <span class="r-after">Exps 1922-1923</span></div>
      </div>
      <div class="r-card">
        <span class="r-tag">System-2 Decoding &mdash; THRML Hookup</span>
        <h3 class="r-title">THRML vs CPU Gibbs Latency Audit</h3>
        <p class="r-desc">Linked Phase 1 THRML Hybrid Thermodynamic Abstraction Hookup with Phase 2 EBT System-2 Energy Decoding Baseline and Inference Scaling on GSM8K Subset.</p>
        <div class="r-meter"><div class="r-bar" style="width: 100%; background: #22c55e;"></div></div>
        <div class="r-stats"><span class="r-before">Latency audit completed</span> <span class="r-after">Exps 1970-1973</span></div>
      </div>
"""

pattern = r'(    </div>\n  </div>\n</section>\n\n<section id="preprint">)'
if re.search(pattern, content):
    content = re.sub(pattern, new_cards + r'\1', content)
    with open('docs/index.html', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Cards added successfully.")
else:
    print("Failed to find insertion point.")
