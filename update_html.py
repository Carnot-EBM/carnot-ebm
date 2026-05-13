import sys
import re
import markdown

with open('docs/technical-report.md', 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert markdown to html
html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

with open('docs/technical-report.html', 'r', encoding='utf-8') as f:
    full_html = f.read()

# Replace everything between <article> and </article>
pattern = re.compile(r'(<article>)(.*?)(</article>)', re.DOTALL)
new_full_html = pattern.sub(r'\1\n' + html_content + r'\n\3', full_html)

with open('docs/technical-report.html', 'w', encoding='utf-8') as f:
    f.write(new_full_html)

print("Updated docs/technical-report.html successfully.")
