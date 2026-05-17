import re

with open('docs/index.html', 'r') as f:
    content = f.read()

card = """      <div class="r-card">
        <span class="r-tag">Test-Time Compute &mdash; PREM</span>
        <h3 class="r-title">PREM Controller implemented</h3>
        <div class="r-meter"><div class="r-bar" style="width: 100%; background: #22c55e;"></div></div>
        <div class="r-stats"><span class="r-before">Dynamic budget controller scales TTC based on variance</span> <span class="r-after">Exps 2144-2150</span></div>
      </div>
"""

pattern = r'(    </div>\n  </div>\n</section>\n\n<section id="preprint">)'
if re.search(pattern, content):
    content = re.sub(pattern, card + r'\1', content)
    with open('docs/index.html', 'w') as f:
        f.write(content)
    print("Success inserting card")
else:
    print("Failed to find insertion point")
