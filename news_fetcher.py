import requests
import os
from requests.status_codes import codes
from database import Session, NewsApiSource, NewsSources, NewsArticles

API_KEY = os.getenv('NEWS_API_KEY')
TOP_HEADLINES_URL = 'https://newsapi.org/v2/top-headlines'
NEWS_API_URL = 'https://newsapi.org/v2/everything'
SOURCES_URL = 'https://newsapi.org/v2/top-headlines/sources'

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