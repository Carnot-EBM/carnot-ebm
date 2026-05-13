import re
import os

def update_file_regex(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update task records count (e.g. 2,213 or 2,223 -> 2,238)
    content = re.sub(r'(\*\*|)\b2,2[0-9]{2}\b(\*\*|) task records', r'\g<1>2,238\g<2> task records', content)
    
    # Update archived milestone records count (174, 175 -> 176)
    content = re.sub(r'(\*\*|)\b17[1-5]\b(\*\*|) artifact-backed', r'\g<1>176\g<2> artifact-backed', content)
    content = re.sub(r'in \b17[1-5]\b artifact-backed', r'in 176 artifact-backed', content)
    content = re.sub(r'(\*\*|)\b17[1-5]\b(\*\*|) completed milestone', r'\g<1>176\g<2> completed milestone', content)
    
    # Update HTML specific counters
    content = re.sub(r'\b17[1-5]\b</div><div class="stat-label">archived records', r'176</div><div class="stat-label">archived records', content)

    # Update milestone versions (e.g. .161 -> .163)
    content = re.sub(r'2026\.05\.16[0-1]', r'2026.05.163', content)
    content = re.sub(r'milestone \.16[0-1]', r'milestone .163', content)
    content = re.sub(r'through \.16[0-1]', r'through .163', content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

update_file_regex("README.md")
update_file_regex("docs/technical-report.md")
update_file_regex("docs/index.html")

print("Regex updates complete.")
