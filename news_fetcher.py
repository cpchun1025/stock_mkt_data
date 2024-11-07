import requests
import os
from requests.status_codes import codes
from database import Session, NewsApiSource, NewsSources, NewsArticles

API_KEY = os.getenv('NEWS_API_KEY')
TOP_HEADLINES_URL = 'https://newsapi.org/v2/top-headlines'
NEWS_API_URL = 'https://newsapi.org/v2/everything'
SOURCES_URL = 'https://newsapi.org/v2/top-headlines/sources'
EVERYTHING_URL = 'https://newsapi.org/v2/everything'

# Fetch available news sources from NewsAPI
def fetch_news_sources():
    url = f"{SOURCES_URL}?apiKey={API_KEY}"
    response = requests.get(url)

    if response.status_code == codes.ok:
        return response.json().get('sources', [])
    return None

# Populate NewsApiSource table
def populate_news_sources(session):
    sources = fetch_news_sources()
    if sources:
        for source in sources:
            existing_source = session.query(NewsApiSource).filter_by(api_source_id=source['id']).first()
            if not existing_source:
                new_source = NewsApiSource(
                    api_source_id=source['id'],
                    api_source_name=source['name'],
                    description=source.get('description', '')
                )
                session.add(new_source)
        session.commit()

# Fetch headlines by country code
def fetch_top_headlines(country_code):
    url = f"{TOP_HEADLINES_URL}?country={country_code}&apiKey={API_KEY}"
    response = requests.get(url)

    if response.status_code == codes.ok:
        return response.json().get('articles', [])
    return None

# Fetch news by stock symbol
def fetch_news(stock_symbol):
    url = f"{NEWS_API_URL}?q={stock_symbol}&apiKey={API_KEY}"
    response = requests.get(url)

    if response.status_code == codes.ok:
        return response.json().get('articles', [])
    return None

# Fetch news articles by keyword (q) and optional filters
def fetch_everything(query, language='en', from_date=None, to_date=None, domains=None):
    """
    Fetch articles from the /everything endpoint with optional filters.
    
    Args:
        query (str): The keyword to search for in the news articles.
        language (str): The language code for filtering results (default is 'en' for English).
        from_date (str): The start date (ISO format: YYYY-MM-DD) for the articles (optional).
        to_date (str): The end date (ISO format: YYYY-MM-DD) for the articles (optional).
        domains (str): A comma-separated list of domains to restrict the search (optional).
    
    Returns:
        list: A list of articles that match the query and filters.
    """
    
    # Build the query parameters
    params = {
        'q': query,
        'language': language,
        'apiKey': API_KEY
    }
    
    # Add optional parameters if provided
    if from_date:
        params['from'] = from_date
    if to_date:
        params['to'] = to_date
    if domains:
        params['domains'] = domains
    
    # Make the request to the /everything endpoint
    response = requests.get(EVERYTHING_URL, params=params)

    # Check if the request was successful
    if response.status_code == codes.ok:
        return response.json().get('articles', [])
    
    # Handle errors or no results
    return None