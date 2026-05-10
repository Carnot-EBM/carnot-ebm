import urllib.request
import xml.etree.ElementTree as ET
import urllib.parse
from datetime import datetime, timedelta

def search(query, max_results=5):
    # Fixed query format for arXiv API
    safe_query = urllib.parse.quote(query)
    url = f'http://export.arxiv.org/api/query?search_query=all:{safe_query}&max_results={max_results}&sortBy=submittedDate&sortOrder=desc'
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        
        # arXiv API uses Atom namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        
        if not entries:
            print("No results found.")
            return
            
        for entry in entries:
            title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
            published = entry.find('atom:published', ns).text[:10]
            authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
            author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            summary = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()[:200] + "..."
            
            print(f"- [{published}] {title} ({author_str})")
            print(f"  {summary}")
            print()
            
    except Exception as e:
        print(f"Error searching {query}: {e}")

print("SEARCHING: Energy-Based Models verification reasoning")
search("Energy-Based Models verification reasoning", 3)

print("SEARCHING: Kolmogorov-Arnold Networks")
search("Kolmogorov-Arnold Networks", 3)

print("SEARCHING: constraint satisfaction neural networks LLM")
search("constraint satisfaction neural networks LLM", 3)

print("SEARCHING: guided decoding energy")
search("guided decoding energy", 3)

print("SEARCHING: continual learning online constraint")
search("continual learning online constraint", 3)
