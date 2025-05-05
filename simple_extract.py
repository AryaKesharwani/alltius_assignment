import os
import re
import json
import urllib.request
import urllib.parse
from html import unescape
import time

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

def get_html_content(source, is_url=False):
    """Get HTML content from a file or URL."""
    try:
        if is_url:
            print(f"Fetching URL: {source}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            req = urllib.request.Request(source, headers=headers)
            with urllib.request.urlopen(req) as response:
                return response.read().decode('utf-8')
        else:
            print(f"Reading from file: {source}")
            with open(source, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        print(f"Error getting content from {source}: {e}")
        return None

def extract_title_and_description(html_content):
    """Extract title and description from HTML."""
    # Extract title
    title_match = re.search(r'<meta property="og:title" content="([^"]+)"', html_content)
    if not title_match:
        title_match = re.search(r'<title>(.*?)</title>', html_content)
    title = title_match.group(1) if title_match else "Angel One Support"
    
    # Extract description
    desc_match = re.search(r'<meta\s+(?:name="description"|property="og:description")\s+content="([^"]+)"', html_content, re.IGNORECASE)
    description = desc_match.group(1) if desc_match else None
    
    return title, description

def extract_faq_accordion(html_content, source_url=None):
    """Extract FAQ content from accordion-style layout."""
    # Try to find the accordion container
    accordion_patterns = [
        r'<div class="list-content">(.*?)<div class="raise-ticket">',
        r'<div class="faqlist">(.*?)</div>\s*</div>\s*</section>',
        r'<div class="tab-content">(.*?)</div>\s*</div>\s*</div>',
        r'<div class="accordion">(.*?)</div>\s*</section>'
    ]
    
    accordion_content = None
    for pattern in accordion_patterns:
        match = re.search(pattern, html_content, re.DOTALL)
        if match:
            accordion_content = match.group(1)
            break
    
    if not accordion_content:
        print("Could not find accordion content")
        return []
    
    # Extract questions and answers from tabs
    tab_pattern = re.compile(r'<div class="tab">\s*<input[^>]*>\s*<label class="tab-label"[^>]*>\s*<span>\s*([^<]+)\s*</span>.*?<div class="content">(.*?)</div>\s*</div>\s*</div>', re.DOTALL)
    
    faq_items = []
    for match in tab_pattern.finditer(accordion_content):
        question = match.group(1).strip()
        raw_answer = match.group(2).strip()
        
        # Clean up the answer HTML
        answer = clean_html(raw_answer)
        
        if question and answer:
            faq_items.append({
                "question": question,
                "answer": answer,
                "source": source_url
            })
    
    # If no items found with first pattern, try alternative patterns
    if not faq_items:
        # Try another common accordion pattern
        alt_pattern = re.compile(r'<h[2-4][^>]*>\s*(.*?)\s*</h[2-4]>.*?<div[^>]*>(.*?)</div>', re.DOTALL)
        
        for match in alt_pattern.finditer(html_content):
            question = match.group(1).strip()
            raw_answer = match.group(2).strip()
            
            # Only use if it looks like a question
            if question and ('?' in question or len(question.split()) < 12):
                answer = clean_html(raw_answer)
                if answer:
                    faq_items.append({
                        "question": question,
                        "answer": answer,
                        "source": source_url
                    })
    
    return faq_items

def extract_related_links(html_content, base_url):
    """Extract related support article links."""
    related_links = []
    
    # Find all links in the sidebar navigation
    sidebar_pattern = re.compile(r'<div class="list-item">.*?<ul.*?>(.*?)</ul>', re.DOTALL)
    sidebar_match = sidebar_pattern.search(html_content)
    
    if sidebar_match:
        sidebar_content = sidebar_match.group(1)
        link_pattern = re.compile(r'<a href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
        
        for match in link_pattern.finditer(sidebar_content):
            url = match.group(1)
            link_text = clean_html(match.group(2))
            
            # Make sure it's an absolute URL
            if not url.startswith(('http://', 'https://')):
                url = urllib.parse.urljoin(base_url, url)
            
            if "/support/" in url and link_text:
                related_links.append((url, link_text))
    
    return related_links

def clean_html(html_text):
    """Clean HTML text to plain text with some formatting preserved."""
    if not html_text:
        return ""
        
    # Replace <p> tags with newlines
    text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n', html_text)
    
    # Replace <br /> with newlines
    text = re.sub(r'<br\s*/>', '\n', text)
    
    # Replace <li> tags with bullet points
    text = re.sub(r'<li[^>]*>(.*?)</li>', r'• \1\n', text)
    
    # Replace <a> tags with their text content plus the URL
    def replace_link(match):
        link_text = match.group(1)
        link_url = re.search(r'href="([^"]+)"', match.group(0))
        if link_url:
            return f"{link_text} ({link_url.group(1)})"
        return link_text
    
    text = re.sub(r'<a[^>]*>(.*?)</a>', replace_link, text)
    
    # Replace headings with formatted headings
    text = re.sub(r'<h1[^>]*>(.*?)</h1>', r'\n# \1\n', text)
    text = re.sub(r'<h2[^>]*>(.*?)</h2>', r'\n## \1\n', text)
    text = re.sub(r'<h3[^>]*>(.*?)</h3>', r'\n### \1\n', text)
    text = re.sub(r'<h4[^>]*>(.*?)</h4>', r'\n#### \1\n', text)
    
    # Remove all other HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Fix any HTML entities
    text = unescape(text)
    
    # Clean up excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'^\s+', '', text)
    text = re.sub(r'\s+$', '', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    return text

def extract_all_support_pages(start_file=None, start_url=None, max_pages=20):
    """Extract content from all related support pages."""
    if not start_file and not start_url:
        start_file = 'raw.html'
    
    all_faqs = []
    visited_urls = set()
    to_visit = []
    
    # Initialize with the start page
    if start_file and os.path.exists(start_file):
        html_content = get_html_content(start_file)
        source_url = start_url or "Local File"
    elif start_url:
        html_content = get_html_content(start_url, is_url=True)
        source_url = start_url
    else:
        print("Error: Must provide either start_file or start_url")
        return None
    
    if not html_content:
        print("Could not get content from starting point")
        return None
    
    # Extract FAQs from the start page
    title, description = extract_title_and_description(html_content)
    faq_items = extract_faq_accordion(html_content, source_url)
    all_faqs.extend(faq_items)
    
    # Get related links to visit next
    if start_url:
        related_links = extract_related_links(html_content, start_url)
        for url, link_text in related_links:
            if url not in visited_urls:
                to_visit.append((url, link_text))
                visited_urls.add(url)
    
    # Process remaining pages
    pages_processed = 1
    while to_visit and pages_processed < max_pages:
        url, page_title = to_visit.pop(0)
        
        # Don't visit URLs we've already processed
        if url in visited_urls:
            continue
        
        visited_urls.add(url)
        html_content = get_html_content(url, is_url=True)
        
        if not html_content:
            continue
        
        # Extract FAQs from this page
        _, _ = extract_title_and_description(html_content)
        page_faqs = extract_faq_accordion(html_content, url)
        
        if page_faqs:
            print(f"Found {len(page_faqs)} FAQs on page: {url} ({page_title})")
            all_faqs.extend(page_faqs)
        
        # Get more links to visit
        more_links = extract_related_links(html_content, url)
        for new_url, new_link_text in more_links:
            if new_url not in visited_urls and new_url not in [u for u, _ in to_visit]:
                to_visit.append((new_url, new_link_text))
        
        pages_processed += 1
        
        # Be nice to the server
        time.sleep(1)
    
    # Save all collected FAQs
    if all_faqs:
        # Save to JSON file
        output = {
            "title": title,
            "description": description,
            "faqs": all_faqs,
            "pages_processed": pages_processed,
            "total_faqs": len(all_faqs)
        }
        
        file_path = "data/angelone_support_faqs.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        # Save to plain text file (better for RAG)
        text_path = "data/angelone_support_faqs.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            if description:
                f.write(f"{description}\n\n")
            
            f.write(f"## Angel One Support FAQs ({len(all_faqs)} items)\n\n")
            
            # Group FAQs by source URL
            faqs_by_source = {}
            for faq in all_faqs:
                source = faq.get("source", "Unknown")
                if source not in faqs_by_source:
                    faqs_by_source[source] = []
                faqs_by_source[source].append(faq)
            
            # Write FAQs grouped by source
            for source, faqs in faqs_by_source.items():
                if source != "Local File" and source != "Unknown":
                    f.write(f"### Source: {source}\n\n")
                
                for faq in faqs:
                    f.write(f"#### Q: {faq['question']}\n\n{faq['answer']}\n\n---\n\n")
        
        print(f"Extraction complete. Processed {pages_processed} pages and found {len(all_faqs)} FAQs")
        print(f"Saved to {file_path} and {text_path}")
        
        return output
    
    print("No FAQ content found")
    return None

def extract_single_page(html_file=None, url=None):
    """Extract content from a single page without following links."""
    if not html_file and not url:
        html_file = 'raw.html'
    
    if html_file and os.path.exists(html_file):
        html_content = get_html_content(html_file)
        source = "Local File"
    elif url:
        html_content = get_html_content(url, is_url=True)
        source = url
    else:
        print("Error: Must provide either html_file or url")
        return None
    
    if not html_content:
        print("Could not get content")
        return None
    
    # Extract content from the page
    title, description = extract_title_and_description(html_content)
    faq_items = extract_faq_accordion(html_content, source)
    
    if faq_items:
        # Save to JSON file
        output = {
            "title": title,
            "description": description,
            "faqs": faq_items
        }
        
        file_path = "data/angel_one_page.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        # Save to plain text file
        text_path = "data/angel_one_page.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            if description:
                f.write(f"{description}\n\n")
            
            f.write("## Frequently Asked Questions\n\n")
            for faq in faq_items:
                f.write(f"### {faq['question']}\n\n{faq['answer']}\n\n")
        
        print(f"Extraction complete. Found {len(faq_items)} FAQ items")
        print(f"Saved to {file_path} and {text_path}")
        
        return output
    
    print("No FAQ content found")
    return None

if __name__ == "__main__":
    import sys
    
    # Check if raw.html exists
    if os.path.exists('raw.html'):
        if len(sys.argv) > 1 and sys.argv[1] == '--all':
            # Extract all related pages
            base_url = "https://www.angelone.in/support/portfolio-and-corporate-actions/dividend"
            extract_all_support_pages('raw.html', base_url)
        else:
            # Extract just the current page
            extract_single_page('raw.html')
    else:
        print("Error: raw.html not found")
        url = "https://www.angelone.in/support/portfolio-and-corporate-actions/dividend"
        response = input(f"Do you want to fetch content from {url}? (y/n): ")
        if response.lower() == 'y':
            if len(sys.argv) > 1 and sys.argv[1] == '--all':
                extract_all_support_pages(start_url=url)
            else:
                extract_single_page(url=url) 