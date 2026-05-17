import urllib.request
import xml.etree.ElementTree as ET
import urllib.parse
from datetime import datetime

queries = [
    'all:"energy-based model" AND all:reasoning',
    'all:"constraint satisfaction" AND all:"neural network"',
    'all:"Kolmogorov-Arnold Network"',
    'all:"guided decoding" AND all:energy',
]

print("Recent arXiv papers (2025-2026):")
for q in queries:
    print(f"\n--- Query: {q} ---")
    query_encoded = urllib.parse.quote(q)
    url = f"http://export.arxiv.org/api/query?search_query={query_encoded}&sortBy=submittedDate&sortOrder=desc&max_results=3"
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        if not entries:
            print("No entries found.")
        for entry in entries:
            title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
            published = entry.find('atom:published', ns).text
            if published.startswith('2025') or published.startswith('2026'):
                summary = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()[:200]
                print(f"- {published[:10]} | {title}\n  {summary}...")
    except Exception as e:
        print(f"Error fetching {q}: {e}")
