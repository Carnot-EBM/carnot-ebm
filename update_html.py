import re
import sys

try:
    import markdown
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown"])
    import markdown

html_path = "docs/technical-report.html"
md_path = "docs/technical-report.md"

with open(html_path, "r", encoding="utf-8") as f:
    html_content = f.read()

with open(md_path, "r", encoding="utf-8") as f:
    md_content = f.read()

# Convert markdown to html
md_html = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

# Extract header string from markdown
m = re.search(r'^## A Technical Report[^\n]*', md_content, re.MULTILINE)
if m:
    header_str = m.group(0).replace('## ', '')
else:
    header_str = "A Technical Report — 1979 Experiments Across the Public Record, 140 Archived Milestone Records, 23,597 Python Test Items Collected (Artifacts Tracked Through Exp 1664)"

# Find everything before <article> and after </article>
pre_article = html_content.split("<article>")[0]
post_article = html_content.split("</article>")[1]

# Update the title and description in pre_article
# Replaces <title>...</title>
pre_article = re.sub(
    r'<title>.*?</title>',
    f'<title>Technical Report - {header_str}</title>',
    pre_article
)
# Replaces content in description
pre_article = re.sub(
    r'<meta name="description" content="Carnot technical report: .*?\. Live GPU benchmarks',
    f'<meta name="description" content="Carnot technical report: {header_str}. Live GPU benchmarks',
    pre_article
)

new_html = pre_article + "<article>\n" + md_html + "\n</article>" + post_article

with open(html_path, "w", encoding="utf-8") as f:
    f.write(new_html)

print("Done generating docs/technical-report.html")