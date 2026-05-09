import sys
import re

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

# Find everything before <article> and after </article>
pre_article = html_content.split("<article>")[0]
post_article = html_content.split("</article>")[1]

# Update the title and description in pre_article
pre_article = pre_article.replace(
    "1888 Experiments", "1941 Experiments"
).replace(
    "137 Archived Milestone Records", "138 Archived Milestone Records"
).replace(
    "23,714 Python Test Items", "23,749 Python Test Items"
)

new_html = pre_article + "<article>\n" + md_html + "\n</article>" + post_article

with open(html_path, "w", encoding="utf-8") as f:
    f.write(new_html)

print("Done generating docs/technical-report.html")