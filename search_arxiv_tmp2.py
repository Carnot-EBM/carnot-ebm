import urllib.request
import urllib.parse
import json
import xml.etree.ElementTree as ET
import datetime

queries = [
    'all:"Energy-Based Model" AND all:"verification"',
    'all:"Energy-Based Model" AND all:"reasoning"',
    'all:"Constraint satisfaction" AND all:"neural network"',
    'all:"Kolmogorov-Arnold Network"',
    'all:"Energy-guided decoding"'
]

for q in queries:
    url = f'http://export.arxiv.org/api/query?search_query={urllib.parse.quote(q)}&sortBy=submittedDate&sortOrder=desc&max_results=3'
    try:
        data = urllib.request.urlopen(url).read()
        root = ET.fromstring(data)
        print(f"--- Results for: {q} ---")
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.replace('\n', ' ')
            published = entry.find('{http://www.w3.org/2005/Atom}published').text
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.replace('\n', ' ')[:200]
            print(f"[{published}] {title}\nSummary: {summary}...\n")
    except Exception as e:
        print(f"Error fetching {q}: {e}")
