import urllib.request
import xml.etree.ElementTree as ET

queries = [
    'all:"Energy-Based Model" AND all:reasoning',
    'all:"Constraint satisfaction" AND all:"neural networks"',
    'all:"Energy-guided decoding"',
    'all:"Hardware-accelerated sampling"',
    'all:"Kolmogorov-Arnold Network"',
]

for query in queries:
    url = f'http://export.arxiv.org/api/query?search_query={urllib.parse.quote(query)}&sortBy=submittedDate&sortOrder=desc&max_results=3'
    try:
        response = urllib.request.urlopen(url)
        data = response.read()
        root = ET.fromstring(data)
        print(f"--- Query: {query} ---")
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip().replace('\n', ' ')
            published = entry.find('{http://www.w3.org/2005/Atom}published').text
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip().replace('\n', ' ')[:200]
            print(f"Title: {title}")
            print(f"Published: {published}")
            print(f"Summary: {summary}...")
            print()
    except Exception as e:
        print(f"Error querying {query}: {e}")
