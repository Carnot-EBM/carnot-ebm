import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time

queries = [
    'all:"Energy-Based Models" AND all:reasoning',
    'all:"Energy-Based Models" AND all:verification',
    'all:"constraint satisfaction" AND all:"neural networks"',
    'all:"Kolmogorov-Arnold Networks"',
    'all:"Energy-guided decoding"',
    'all:"continual learning" AND all:"constraint"',
    'all:"thermodynamic computing"'
]

results = []

for q in queries:
    # Adding date range might be tricky with arxiv API, so we fetch latest and filter.
    url = f"http://export.arxiv.org/api/query?search_query={urllib.parse.quote(q)}&sortBy=submittedDate&sortOrder=descending&max_results=5"
    try:
        response = urllib.request.urlopen(url)
        data = response.read().decode('utf-8')
        root = ET.fromstring(data)
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.replace('\n', ' ').strip()
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.replace('\n', ' ').strip()
            published = entry.find('{http://www.w3.org/2005/Atom}published').text
            url_link = entry.find('{http://www.w3.org/2005/Atom}id').text
            if published.startswith('2025') or published.startswith('2026'):
                results.append(f"Title: {title}\nDate: {published}\nURL: {url_link}\nSummary: {summary}\n")
    except Exception as e:
        results.append(f"Error for query {q}: {e}")
    time.sleep(3)

with open('my_arxiv_results.txt', 'w') as f:
    f.write("\n---\n".join(results))
