import sys
import subprocess

try:
    import markdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown"])
    import markdown

with open("docs/technical-report.html", "r") as f:
    html = f.read()

header = html.split('<article class="markdown-body">')[0] + '<article class="markdown-body">\n'
footer = '\n</article>' + html.split('</article>')[-1]

with open("docs/technical-report.md", "r") as f:
    md_content = f.read()

# Try to add slug IDs to headers like in the original html, we can use the 'toc' extension for this
md_html = markdown.markdown(md_content, extensions=['fenced_code', 'tables', 'toc'])

with open("docs/technical-report.html", "w") as f:
    f.write(header + md_html + footer)

print("Rendered HTML successfully!")
