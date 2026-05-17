import re

with open('docs/index.html', 'r') as f:
    content = f.read()

cards = """      <div class="r-card">
        <span class="r-tag">Optimization &mdash; ALPS Module</span>
        <h3 class="r-title">300x speedup over Langevin</h3>
        <div class="r-meter"><div class="r-bar" style="width: 100%; background: #22c55e;"></div></div>
        <div class="r-stats"><span class="r-before">Terminal energy -0.842</span> <span class="r-after">Exp 2109</span></div>
      </div>
      <div class="r-card">
        <span class="r-tag">Verification &mdash; CARM</span>
        <h3 class="r-title">Constraint-Aware Retrieval Module</h3>
        <div class="r-meter"><div class="r-bar" style="width: 100%; background: #0ea5e9;"></div></div>
        <div class="r-stats"><span class="r-before">Integrated retrieval with constraints</span> <span class="r-after">Exp 2121</span></div>
      </div>
"""

pattern = r'(</div>\n</section>\n\n<section id="preprint">)'
if re.search(pattern, content):
    content = re.sub(pattern, cards + r'\1', content)
    with open('docs/index.html', 'w') as f:
        f.write(content)
    print("Success")
else:
    print("Failed to find insertion point")
