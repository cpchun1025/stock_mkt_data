import logging
from datetime import datetime
from sqlalchemy.orm import Session

from database import create_tables, Session, NewsArticles, SentimentAnalysis, NewsSources
from news_fetcher import populate_news_sources, fetch_top_headlines, fetch_news
from sentiment_analysis import analyze_sentiment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('NewsAPI')

# Store the news article in the database
def store_news_article(session, article, source_code=None, source_type=None):
    """
    Store a fetched article in the NewsArticles table.
    """
    # Extract the source info
    api_source_id = article.get('source', {}).get('id', None)
    api_source_name = article.get('source', {}).get('name', None)

    # Check if the source exists
    source = session.query(NewsSources).filter_by(api_source_id=api_source_id).first()

    if not source:
        logger.error(f"Source not found for article: {article.get('title', 'No Title')}")
        return None

    # Create and store the article
    new_article = NewsArticles(
        source_id=source.id,
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

# Main function to fetch, analyze, and store news articles
def main():
    # Create a new session
    session = Session()

    # Step 1: Populate NewsApiSource table
    logger.info("Populating NewsApiSource table...")
    populate_news_sources(session)

    # Step 2: Fetch and process articles for various countries
    country_codes = ['us', 'gb', 'jp']  # Example country codes for headlines
    for country_code in country_codes:
        logger.info(f"Fetching top headlines for {country_code}...")
        articles = fetch_top_headlines(country_code)

        if articles:
            for article in articles:
                # Perform sentiment analysis on the article content or description
                content = article.get('content', '') or article.get('description', '')

                # Get BERT and TextBlob sentiment results
                bert_label, bert_score, blob_label, blob_polarity, blob_subjectivity = analyze_sentiment(content)

                # Step 3: Store the article
                article_id = store_news_article(session, article, source_code=country_code, source_type='country')

                # If the article was successfully stored, store the sentiment analysis as well
                if article_id:
                    store_sentiment_analysis(session, article_id, "BERT", bert_label, bert_score)
                    store_sentiment_analysis(session, article_id, "TextBlob", blob_label, polarity=blob_polarity, subjectivity=blob_subjectivity)

    # Step 3: Fetch and process articles for specific stock symbols
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