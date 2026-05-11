import re

with open('docs/index.html', 'r') as f:
    content = f.read()

# Find the end of the Multi-step reasoning paragraph
pattern = r'(<h3 class="bento-title">Multi-step reasoning</h3>.*?)(</p>)'

def repl(m):
    return m.group(1) + " Milestone .134 evaluated the EqM Gradient Sampler on System-2 reasoning benchmarks (MATH and GSM8K), successfully deployed CIKAN verification on FPGA hardware, and completed the full E2E pipeline with live telemetry and continual learning.\n        </p>"

new_content = re.sub(pattern, repl, content, flags=re.DOTALL)

with open('docs/index.html', 'w') as f:
    f.write(new_content)

print("Appended summary to index.html")
