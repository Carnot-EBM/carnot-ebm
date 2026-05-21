import re

with open('docs/index.html', 'r') as f:
    content = f.read()

content = content.replace("Six capabilities, one framework", "Seven capabilities, one framework")

capability_html = """      <div class="bento-card span-2">
        <div class="bento-icon">P</div>
        <h3 class="bento-title">Test-Time Compute (TTC) &amp; PREM</h3>
        <p class="bento-text">
          A dynamic budget controller that scales Test-Time Compute (TTC) based on Process-Reward Energy Model (PREM) variance. This provides intrinsic motivation for continuous self-learning.
        </p>
      </div>
"""

pattern = r'(    </div>\n  </div>\n</section>\n\n<section id="results">)'
if re.search(pattern, content):
    content = re.sub(pattern, capability_html + r'\1', content)
    with open('docs/index.html', 'w') as f:
        f.write(content)
    print("Success inserting capability")
else:
    print("Failed to find insertion point")
