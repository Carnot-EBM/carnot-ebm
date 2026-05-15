import sys
import re

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

# Find the end of the bento-grid or the previous r-card
if "EBFT &amp; CASAL Sampler Integration" not in html_content:
    html_content = html_content.replace('      <div class="r-card">\n        <span class="r-tag">Hardware Profiling</span>', new_card + '      <div class="r-card">\n        <span class="r-tag">Hardware Profiling</span>')

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Added new card to index.html")
