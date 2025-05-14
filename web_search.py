import re
from bs4 import BeautifulSoup
import requests

# Web search function for contact info with improved extraction
def web_search_contact_info(query, max_results=3):
    url = f"https://html.duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Scrape contact information, such as emails or phone numbers
        contact_results = []
        
        # Looking for contact links such as mailto or tel
        for result in soup.find_all('a', href=True):
            href = result['href']
            if 'mailto:' in href:
                contact_results.append(f"Email: {result.get_text(strip=True)} ({href})")
            elif 'tel:' in href:
                contact_results.append(f"Phone: {result.get_text(strip=True)} ({href})")

        # If no contact links are found, fallback to general search results
        if not contact_results:
            contact_results = [result.text.strip() for result in soup.select(".result__snippet", limit=max_results)]
        
        # If results were found, return them
        return contact_results

    except requests.RequestException as e:
        return [f"Error during search: {str(e)}"]

