import logging
from logging.handlers import RotatingFileHandler
import requests
from transformers import pipeline
from textblob import TextBlob
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
import json
from dotenv import load_dotenv
from requests.status_codes import codes
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Constants for DB connection
DB_SERVER = os.getenv('DB_SERVER')
DB_NAME = os.getenv('DB_NAME')
API_KEY = os.getenv('NEWS_API_KEY')

# API URLs
TOP_HEADLINES_URL = 'https://newsapi.org/v2/top-headlines'
NEWS_API_URL = 'https://newsapi.org/v2/everything'
SOURCES_URL = 'https://newsapi.org/v2/top-headlines/sources'

# SQLAlchemy setup
Base = declarative_base()

# Engine and session setup
DATABASE_URL = f"mssql+pyodbc://{DB_SERVER}/{DB_NAME}?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# Configure rolling log
log_file = 'news_api.log'
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Configure rotating file handler (5 files, 1MB each)
log_handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=5)
log_handler.setFormatter(log_formatter)

# Get the logger and apply the handler
logger = logging.getLogger('NewsAPI')
logger.setLevel(logging.INFO)  # Set log level to INFO
logger.addHandler(log_handler)

# Utility to save API response to a JSON file
def save_json_response(response_data, filename):
    # Create a folder structure with the current date (e.g., \data\YYYY-MM-DD)
    current_date = datetime.now().strftime('%Y-%m-%d')
    folder_path = Path(f"data/{current_date}")
    folder_path.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist

    # Define the full path for the output JSON file
    file_path = folder_path / filename

    # Save the response data as a JSON file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(response_data, f, ensure_ascii=False, indent=4)

    logger.info(f"Saved API response to {file_path}")

# Define the NewsSources model
class NewsSources(Base):
    __tablename__ = 'NewsSources'

    id = Column(Integer, primary_key=True, autoincrement=True)
    api_source_id = Column(String(50), nullable=False, unique=True)  # API returned source ID (e.g., 'the-irish-times')
    api_source_name = Column(String(255), nullable=False)  # API returned source name (e.g., 'The Irish Times')
    source_code = Column(String(10))  # Stock symbol or country code (optional, for filtering)
    source_type = Column(String(50))  # Type of source: 'stock' or 'country' (optional)
    description = Column(String(255))  # Optional description

# Define the NewsArticles model
class NewsArticles(Base):
    __tablename__ = 'NewsArticles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(Integer, ForeignKey('NewsSources.id'), nullable=False)
    author = Column(String(255))
    title = Column(String(255))
    description = Column(Text)
    url = Column(String(500))
    published_at = Column(DateTime)
    content = Column(Text)
    bert_sentiment_label = Column(String(50))
    bert_confidence_score = Column(Float)
    textblob_sentiment_label = Column(String(50))
    textblob_polarity = Column(Float)  # TextBlob polarity (-1 to 1)
    textblob_subjectivity = Column(Float)  # TextBlob subjectivity (0 to 1)

    source = relationship('NewsSources')

# Initialize the Hugging Face sentiment analysis pipeline (using a multilingual BERT-based model)
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Define country codes for each region
APAC_COUNTRIES = ['cn', 'hk', 'id', 'in', 'jp', 'kr', 'my', 'nz', 'ph', 'sg', 'th', 'tw', 'vn']
EMEA_COUNTRIES = ['ae', 'at', 'bg', 'ch', 'cz', 'de', 'eg', 'es', 'fr', 'gb', 'gr', 'hu', 'ie', 'il', 'it', 'lt', 'lv', 'ma', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'sa', 'se', 'tr', 'ua', 'za']
AMERICAS_COUNTRIES = ['ar', 'br', 'ca', 'co', 'cu', 'mx', 'us', 've']

# Fetch available news sources from NewsAPI
def fetch_news_sources():
    url = f"{SOURCES_URL}?apiKey={API_KEY}"
    response = requests.get(url)

    if response.status_code == codes.ok:
        sources_data = response.json()
        logger.info(f"Fetched news sources from NewsAPI: {len(sources_data.get('sources', []))} sources retrieved.")
        save_json_response(sources_data, 'news_sources.json')  # Save JSON response
        return sources_data.get('sources', [])  # List of sources
    else:
        logger.error(f"Error fetching news sources: {response.status_code}")
        return None

# Populate the NewsSources table with sources from NewsAPI
def populate_news_sources(session):
    sources = fetch_news_sources()

    if sources:
        for source in sources:
            # Check if the source already exists in the database
            existing_source = session.query(NewsSources).filter_by(api_source_id=source['id']).first()

            if not existing_source:
                # If the source doesn't exist, create a new entry
                new_source = NewsSources(
                    api_source_id=source['id'],
                    api_source_name=source['name'],
                    description=source.get('description', '')
                )
                session.add(new_source)

        # Commit all new sources to the database
        session.commit()
        logger.info(f"Populated {len(sources)} sources into NewsSources table.")

# Fetch top headlines from NewsAPI for a specific country
def fetch_top_headlines(country_code):
    url = f"{TOP_HEADLINES_URL}?country={country_code}&apiKey={API_KEY}"
    response = requests.get(url)

    if response.status_code == codes.ok:
        headlines_data = response.json()
        logger.info(f"Fetched top headlines for country: {country_code}, Articles: {len(headlines_data.get('articles', []))}")
        save_json_response(headlines_data, f'top_headlines_{country_code}.json')  # Save JSON response
        return headlines_data.get('articles', [])
    else:
        logger.error(f"Error fetching top headlines for country {country_code}: {response.status_code}")
        return None

# Fetch news articles from NewsAPI for a specific stock symbol
def fetch_news(stock_symbol):
    url = f"{NEWS_API_URL}?q={stock_symbol}&sortBy=publishedAt&apiKey={API_KEY}"
    response = requests.get(url)

    if response.status_code == codes.ok:
        news_data = response.json()
        logger.info(f"Fetched news for stock symbol: {stock_symbol}, Articles: {len(news_data.get('articles', []))}")
        save_json_response(news_data, f'news_{stock_symbol}.json')  # Save JSON response
        return news_data.get('articles', [])
    else:
        logger.error(f"Error fetching news for stock symbol {stock_symbol}: {response.status_code}")
        return None

# Analyze sentiment using both BERT-based and TextBlob models
def analyze_sentiment(text):
    # If the text is empty or None, return a default neutral sentiment
    if not text or text.strip() == "":
        logger.warning("Empty text provided for sentiment analysis. Returning default values.")
        return "neutral", 0.0, "neutral", 0.0, 0.0

    # BERT-based sentiment analysis
    bert_result = sentiment_pipeline(text[:512])  # Limit input to 512 tokens (BERT max length)
    bert_sentiment_label = bert_result[0]['label']  # Extract the predicted label
    bert_confidence_score = bert_result[0]['score']  # Extract the confidence score

    # TextBlob sentiment analysis
    textblob_analysis = TextBlob(text)
    textblob_polarity = textblob_analysis.sentiment.polarity  # Polarity ranges from -1 to 1
    textblob_subjectivity = textblob_analysis.sentiment.subjectivity  # Subjectivity ranges from 0 to 1

    # Determine TextBlob sentiment label based on polarity
    if textblob_polarity > 0:
        textblob_sentiment_label = "positive"
    elif textblob_polarity < 0:
        textblob_sentiment_label = "negative"
    else:
        textblob_sentiment_label = "neutral"

    logger.info(f"BERT Sentiment: {bert_sentiment_label}, Confidence: {bert_confidence_score}")
    logger.info(f"TextBlob Sentiment: {textblob_sentiment_label}, Polarity: {textblob_polarity}, Subjectivity: {textblob_subjectivity}")

    return bert_sentiment_label, bert_confidence_score, textblob_sentiment_label, textblob_polarity, textblob_subjectivity

# Get or create the source in the NewsSources table
def get_or_create_source(session, api_source_id, api_source_name, source_code=None, source_type=None):
    # Check if the source already exists by api_source_id
    source = session.query(NewsSources).filter_by(api_source_id=api_source_id).first()

    if source:
        return source.id  # Source exists, return its ID
    else:
        # Create a new source and commit it to the database
        new_source = NewsSources(
            api_source_id=api_source_id,
            api_source_name=api_source_name,
            source_code=source_code,  # Optional stock or country code
            source_type=source_type  # Optional source type
        )
        session.add(new_source)
        session.commit()  # Commit the new source to the database
        session.refresh(new_source)  # Refresh to get the new source's ID
        logger.info(f"Created new source: {api_source_name} (ID: {new_source.id})")
        return new_source.id  # Return the ID of the newly inserted source

# Store news article with sentiment in the database
def store_news_article(session, article, bert_sentiment_label, bert_confidence_score, textblob_sentiment_label, textblob_polarity, textblob_subjectivity, source_code=None, source_type=None):
    # Get the actual source info from the article (returned by NewsAPI)
    api_source_id = article.get('source', {}).get('id', None)  # e.g., 'the-irish-times'
    api_source_name = article.get('source', {}).get('name', None)  # e.g., 'The Irish Times'

    if api_source_id and api_source_name:
        # Get or create the source using the standardized API data
        source_id = get_or_create_source(session, api_source_id, api_source_name, source_code, source_type)
    else:
        # If no source info is provided, handle this case (e.g., log an error or skip the article)
        logger.error(f"Article missing source info: {article.get('title', 'Unknown title')}")
        return

    # Create a new NewsArticles object
    new_article = NewsArticles(
        source_id=source_id,
        author=article.get('author', 'Unknown'),
        title=article.get('title', ''),
        description=article.get('description', ''),
        url=article.get('url', ''),
        published_at=datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),  # Ensure correct datetime format
        content=article.get('content', ''),
        bert_sentiment_label=bert_sentiment_label,
        bert_confidence_score=bert_confidence_score,
        textblob_sentiment_label=textblob_sentiment_label,
        textblob_polarity=textblob_polarity,
        textblob_subjectivity=textblob_subjectivity  # Store subjectivity
    )

    # Add and commit the article to the database
    session.add(new_article)
    session.commit()
    logger.info(f"Stored article: {article.get('title', 'No Title')}")

# Main function to fetch, analyze, and store news articles
def main(source_code, source_type, fetch_function):
    # Step 1: Fetch news articles (could be top headlines or stock-related news)
    articles = fetch_function(source_code)

    if articles:
        # Step 2: Create a session
        session = Session()

        for article in articles:
            # Perform sentiment analysis on the article content
            content = article.get('content', '') or article.get('description', '')

            # Get BERT and TextBlob sentiment results, including subjectivity
            bert_sentiment_label, bert_confidence_score, textblob_sentiment_label, textblob_polarity, textblob_subjectivity = analyze_sentiment(content)

            # Step 3: Store the article and its sentiment (including confidence and subjectivity) in the database
            store_news_article(session, article, bert_sentiment_label, bert_confidence_score, textblob_sentiment_label, textblob_polarity, textblob_subjectivity, source_code, source_type)

        # Step 4: Close the session
        session.close()

# Run the script to fetch headlines for each region or stock
if __name__ == '__main__':
    session = Session()
    populate_news_sources(session)  # Populate sources from NewsAPI once
    session.close()

    # Fetch top headlines for APAC countries
    for country_code in APAC_COUNTRIES:
        logger.info(f"Fetching top headlines for {country_code} (APAC)...")
        main(country_code, 'country', fetch_top_headlines)
        logger.info(f"Finished processing headlines for {country_code} (APAC).")

    # Fetch top headlines for EMEA countries
    for country_code in EMEA_COUNTRIES:
        logger.info(f"Fetching top headlines for {country_code} (EMEA)...")
        main(country_code, 'country', fetch_top_headlines)
        logger.info(f"Finished processing headlines for {country_code} (EMEA).")

    # Fetch top headlines for Americas countries
    for country_code in AMERICAS_COUNTRIES:
        logger.info(f"Fetching top headlines for {country_code} (Americas)...")
        main(country_code, 'country', fetch_top_headlines)
        logger.info(f"Finished processing headlines for {country_code} (Americas).")

    # Define stock symbols for each sector
    banks = ['JPM', 'BAC', 'BCS']  # JPMorgan, Bank of America, Barclays    
    tech = ['AAPL', 'MSFT', 'GOOGL']  # Apple, Microsoft, Alphabet (Google)
    traditional_industry = ['XOM', 'CVX', 'MMM']  # ExxonMobil, Chevron, 3M    
    oils = ['XOM', 'CVX', 'BP']  # ExxonMobil, Chevron, BP
    chinese_stocks = ['BABA', 'TCEHY', 'JD']  # Alibaba, Tencent, JD.com

    # Fetch news for stock symbols
    for stock_symbol in banks + tech + traditional_industry + oils + chinese_stocks:
        logger.info(f"Fetching news for {stock_symbol}...")
        main(stock_symbol, 'stock', fetch_news)
        logger.info(f"Finished processing news for {stock_symbol}.")