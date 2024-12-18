import logging
import os
from datetime import datetime

from newsapi import NewsApiClient
from sqlalchemy.orm import Session

from database import (NewsApiSource, NewsArticles, NewsSources,
                      SentimentAnalysis, Session, create_tables)
from sentiment_analysis import analyze_sentiment

API_KEY = os.getenv('NEWS_API_KEY')

# Initialize the NewsApiClient with your API key
newsapi = NewsApiClient(api_key=API_KEY)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('NewsAPI')


def store_news_article(session, article, source_code=None, source_type=None):
    """
    Store a fetched article in the NewsArticles table.
    Handles cases where the source ID is missing by using the source name as a fallback.
    """
    # Extract source information from the article
    api_source_id = article.get('source', {}).get('id', None)
    api_source_name = article.get('source', {}).get('name', None)

    if not api_source_name:
        logger.error(f"Source name is missing from article: {article.get('title', 'No Title')}")
        return None

    # Step 1: Check for source by ID if available, otherwise fallback to source name
    if api_source_id:
        existing_api_source = session.query(NewsApiSource).filter_by(api_source_id=api_source_id).first()
    else:
        # If no source ID, use the name as a unique identifier
        existing_api_source = session.query(NewsApiSource).filter_by(api_source_name=api_source_name).first()

    if not existing_api_source:
        # If the API source does not exist, create a new entry in NewsApiSource
        new_api_source = NewsApiSource(
            api_source_id=api_source_id,  # This will be None if not provided
            api_source_name=api_source_name,
            description=article.get('source', {}).get('description', '')  # Optional description from API
        )
        session.add(new_api_source)
        session.commit()  # Commit to get the new ID
        existing_api_source = new_api_source  # Use the newly created source

    # Step 2: Check if the source exists in NewsSources (linked to NewsApiSource)
    existing_source = session.query(NewsSources).filter_by(api_source_id=existing_api_source.id).first()

    if not existing_source:
        # If the source doesn't exist in NewsSources, create a new entry
        new_source = NewsSources(
            api_source_id=existing_api_source.id,  # Link to NewsApiSource
            source_code=source_code,  # Stock symbol or country code (if provided)
            source_type=source_type   # 'stock' or 'country' (if provided)
        )
        session.add(new_source)
        session.commit()  # Commit to get the new source ID
        existing_source = new_source

    # Step 3: Create and store the new article in NewsArticles
    try:
        new_article = NewsArticles(
            source_id=existing_source.id,  # Link to the NewsSources entry
            author=article.get('author', 'Unknown'),
            title=article.get('title', ''),
            description=article.get('description', ''),
            url=article.get('url', ''),
            published_at=datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),
            content=article.get('content', '')
        )
        session.add(new_article)
        session.commit()
        logger.info(f"Stored article: {article.get('title', 'No Title')}")
        return new_article.id

    except Exception as e:
        session.rollback()
        logger.error(f"Failed to store article: {article.get('title', 'No Title')}. Error: {str(e)}")
        return None


# Store sentiment analysis results in the SentimentAnalysis table
def store_sentiment_analysis(session, article_id, model_name, sentiment_label, confidence_score=None, polarity=None, subjectivity=None):
    """
    Store sentiment analysis results of an article in the SentimentAnalysis table.
    """
    new_sentiment = SentimentAnalysis(
        article_id=article_id,
        model_name=model_name,
        sentiment_label=sentiment_label,
        confidence_score=confidence_score,
        polarity=polarity,
        subjectivity=subjectivity
    )
    session.add(new_sentiment)
    session.commit()
    logger.info(f"Stored sentiment analysis for article_id: {article_id} using {model_name}")


# Function to fetch top headlines using the NewsAPI Python client
def fetch_top_headlines(country_code):
    """
    Fetch top headlines from NewsAPI for a given country code.
    """
    response = newsapi.get_top_headlines(country=country_code)
    return response.get('articles', [])


# Function to fetch news for specific stock symbols using the NewsAPI Python client
def fetch_news(stock_symbol):
    """
    Fetch news articles from NewsAPI related to a specific stock symbol.
    """
    response = newsapi.get_everything(q=stock_symbol)
    return response.get('articles', [])

# Function to fetch news using the NewsAPI /everything endpoint
def fetch_news_by_country(keyword):
    """
    Fetch news articles from NewsAPI using the /everything endpoint with a specific keyword (e.g., country name).
    """
    response = newsapi.get_everything(q=keyword)
    return response.get('articles', [])

# Main function to fetch, analyze, and store news articles
def main():
    # Create a new session
    session = Session()

    # Step 1: Fetch and process articles for various countries using keywords
    country_keywords = ['Japan', 'United Kingdom']  # Example keywords related to countries
    for country_keyword in country_keywords:
        logger.info(f"Fetching news for country: {country_keyword}...")
        articles = fetch_news_by_country(country_keyword)

        if articles:
            for article in articles:
                # Perform sentiment analysis on the article content or description
                content = article.get('content', '') or article.get('description', '')

                # Get BERT and TextBlob sentiment results
                bert_label, bert_score, blob_label, blob_polarity, blob_subjectivity = analyze_sentiment(content)

                # Step 2: Store the article
                article_id = store_news_article(session, article, source_code=country_keyword, source_type='country')

                # If the article was successfully stored, store the sentiment analysis as well
                if article_id:
                    store_sentiment_analysis(session, article_id, "BERT", bert_label, bert_score)
                    store_sentiment_analysis(session, article_id, "TextBlob", blob_label, polarity=blob_polarity, subjectivity=blob_subjectivity)
    # Step 2: Fetch and process articles for specific stock symbols
    stock_symbols = ['AAPL', 'GOOGL', 'MSFT']  # Example stock symbols
    for stock_symbol in stock_symbols:
        logger.info(f"Fetching news for stock symbol: {stock_symbol}...")
        articles = fetch_news(stock_symbol)

        if articles:
            for article in articles:
                # Perform sentiment analysis on the article content or description
                content = article.get('content', '') or article.get('description', '')

                # Get BERT and TextBlob sentiment results
                bert_label, bert_score, blob_label, blob_polarity, blob_subjectivity = analyze_sentiment(content)

                # Step 3: Store the article
                article_id = store_news_article(session, article, source_code=stock_symbol, source_type='stock')

                # If the article was successfully stored, store the sentiment analysis as well
                if article_id:
                    store_sentiment_analysis(session, article_id, "BERT", bert_label, bert_score)
                    store_sentiment_analysis(session, article_id, "TextBlob", blob_label, polarity=blob_polarity, subjectivity=blob_subjectivity)

    # Close the session once everything is done
    session.close()


if __name__ == '__main__':
    # Ensure tables are created before running the script
    create_tables()
    main()