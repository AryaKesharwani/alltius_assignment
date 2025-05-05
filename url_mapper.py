import requests
import re
import urllib.parse
import time
import os
from html import unescape
from urllib.request import Request, urlopen

def get_html_content(url):
    """Get HTML content from a URL with proper headers."""
    try:
        print(f"Fetching URL: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        req = Request(url, headers=headers)
        with urlopen(req) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        print(f"Error getting content from {url}: {e}")
        return None

def extract_urls(html_content, base_url):
    """Extract all URLs from HTML content that are related to support pages."""
    urls = []
    
    # Find all links in the HTML
    link_pattern = re.compile(r'<a[^>]*href="([^"]+)"[^>]*>', re.DOTALL)
    for match in link_pattern.finditer(html_content):
        url = match.group(1)
        
        # Skip anchor links, javascript, and other non-HTTP links
        if url.startswith('#') or url.startswith('javascript:') or url.startswith('mailto:'):
            continue
        
        # Make sure it's an absolute URL
        if not url.startswith(('http://', 'https://')):
            url = urllib.parse.urljoin(base_url, url)
        
        # Only include Angel One support URLs
        if 'angelone.in/support' in url:
            # Clean up the URL (remove fragments and query parameters)
            url = url.split('#')[0]
            url = url.split('?')[0]
            
            urls.append(url)
    
    return urls

def map_support_urls(start_url="https://www.angelone.in/support", max_pages=500):
    """Map all URLs under the Angel One support section."""
    all_urls = set()
    to_visit = [start_url]
    visited = set()
    
    pages_processed = 0
    
    while to_visit and pages_processed < max_pages:
        url = to_visit.pop(0)
        
        if url in visited:
            continue
        
        visited.add(url)
        html_content = get_html_content(url)
        
        if not html_content:
            continue
        
        # Add this URL to our collection
        all_urls.add(url)
        pages_processed += 1
        
        # Extract more URLs to visit
        new_urls = extract_urls(html_content, url)
        for new_url in new_urls:
            if new_url not in visited and new_url not in to_visit:
                to_visit.append(new_url)
        
        print(f"Processed {pages_processed} pages. Found {len(all_urls)} unique URLs. Queue: {len(to_visit)}")
        
        # Be nice to the server
        time.sleep(1)
    
    # Save all collected URLs
    if all_urls:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Save to text file
        file_path = "data/angelone_support_urls.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            for url in sorted(all_urls):
                f.write(f"{url}\n")
        
        print(f"URL mapping complete. Processed {pages_processed} pages and found {len(all_urls)} unique URLs")
        print(f"Saved to {file_path}")
        
        return all_urls
    
    print("No URLs found")
    return set()

if __name__ == "__main__":
    print("Starting URL mapping for Angel One support pages...")
    map_support_urls()
