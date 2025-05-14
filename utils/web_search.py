# web_search.py

import requests
from bs4 import BeautifulSoup
import urllib.parse

def search_web(query, language='fr'):
    """
    Effectue une recherche Google simulée (via scraping simple) et retourne le premier résultat.
    """
    try:
        # URL encodée
        query_encoded = urllib.parse.quote_plus(query)
        url = f"https://www.google.com/search?q={query_encoded}&hl={language}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/90.0.4430.93 Safari/537.36"
        }

        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code != 200:
            return "Je n'ai trouvé aucune information pertinente."

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extrait le texte d’un des premiers blocs de réponse
        snippet = soup.find('div', class_='BNeawe s3v9rd AP7Wnd')
        if snippet:
            return snippet.text.strip()

        return "Je n'ai pas trouvé de réponse pertinente sur le web."

    except Exception as e:
        return f"Erreur lors de la recherche : {str(e)}"
