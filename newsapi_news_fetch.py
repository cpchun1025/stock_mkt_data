import requests
from transformers import pipeline
from textblob import TextBlob
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv
from requests.status_codes import codes

# Load environment variables from .env file
load_dotenv()

# Constants for DB connection
DB_SERVER = os.getenv('DB_SERVER')
DB_NAME = os.getenv('DB_NAME')
API_KEY = os.getenv('NEWS_API_KEY')

# API URLs
TOP_HEADLINES_URL = 'https://newsapi.org/v2/top-headlines'
NEWS_API_URL = 'https://newsapi.org/v2/everything'

# SQLAlchemy setup
Base = declarative_base()

# Engine and session setup
DATABASE_URL = f"mssql+pyodbc://{DB_SERVER}/{DB_NAME}?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# Define the NewsSources model
class NewsSources(Base):
    __tablename__ = 'NewsSources'

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_code = Column(String(10), nullable=False)  # Stock symbol or country code
    source_type = Column(String(50), nullable=False)  # Type of source: 'stock' or 'country'
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

# Fetch top headlines from NewsAPI for a specific country
def fetch_top_headlines(country_code):
    url = f"{TOP_HEADLINES_URL}?country={country_code}&apiKey={API_KEY}"
    response = requests.get(url)

    if response.status_code == codes.ok:
        headlines_data = response.json()
        articles = headlines_data.get('articles', [])
        return articles  # List of articles
    else:
        print(f"Error fetching top headlines: {response.status_code}")
        return None

# Fetch news articles from NewsAPI for a specific stock symbol
def fetch_news(stock_symbol):
    url = f"{NEWS_API_URL}?q={stock_symbol}&sortBy=publishedAt&apiKey={API_KEY}"
    response = requests.get(url)

    if response.status_code == codes.ok:
        news_data = response.json()
        articles = news_data.get('articles', [])
        return articles  # List of articles
    else:
        print(f"Error fetching news: {response.status_code}")
        return None

# Analyze sentiment using both BERT-based and TextBlob models
def analyze_sentiment(text):
    # If the text is empty or None, return a default neutral sentiment
    if not text or text.strip() == "":
        print("Empty text provided for sentiment analysis. Returning default values.")
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

    # Print the results for debugging or inspection
    print(f"BERT Sentiment: {bert_sentiment_label}, Confidence: {bert_confidence_score}")
    print(f"TextBlob Sentiment: {textblob_sentiment_label}, Polarity: {textblob_polarity}, Subjectivity: {textblob_subjectivity}")

    return bert_sentiment_label, bert_confidence_score, textblob_sentiment_label, textblob_polarity, textblob_subjectivity

# Get or create the source in the NewsSources table
def get_or_create_source(session, source_code, source_type):
    # Check if the source already exists
    source = session.query(NewsSources).filter_by(source_code=source_code, source_type=source_type).first()

    if source:
        return source.id  # Source exists, return its ID
    else:
        # Create a new source and commit it to the database
        new_source = NewsSources(source_code=source_code, source_type=source_type)
        session.add(new_source)
        session.commit()  # Commit the new source to the database
        session.refresh(new_source)  # Refresh to get the new source's ID
        return new_source.id  # Return the ID of the newly inserted source

# Store news article with sentiment in the database
def store_news_article(session, article, bert_sentiment_label, bert_confidence_score, textblob_sentiment_label, textblob_polarity, textblob_subjectivity, source_code, source_type):
    # Get or create the source
    source_id = get_or_create_source(session, source_code, source_type)

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
    # Fetch top headlines for APAC countries
    for country_code in APAC_COUNTRIES:
        print(f"Fetching top headlines for {country_code} (APAC)...")
        main(country_code, 'country', fetch_top_headlines)
        print(f"Finished processing headlines for {country_code} (APAC).")

    # Fetch top headlines for EMEA countries
    for country_code in EMEA_COUNTRIES:
        print(f"Fetching top headlines for {country_code} (EMEA)...")
        main(country_code, 'country', fetch_top_headlines)
        print(f"Finished processing headlines for {country_code} (EMEA).")

    # Fetch top headlines for Americas countries
    for country_code in AMERICAS_COUNTRIES:
        print(f"Fetching top headlines for {country_code} (Americas)...")
        main(country_code, 'country', fetch_top_headlines)
        print(f"Finished processing headlines for {country_code} (Americas).")

    # Define stock symbols for each sector
    banks = ['JPM', 'BAC', 'BCS']  # JPMorgan, Bank of America, Barclays    
    tech = ['AAPL', 'MSFT', 'GOOGL']  # Apple, Microsoft, Alphabet (Google)
    traditional_industry = ['XOM', 'CVX', 'MMM']  # ExxonMobil, Chevron, 3M    
    oils = ['XOM', 'CVX', 'BP']  # ExxonMobil, Chevron, BP
    chinese_stocks = ['BABA', 'TCEHY', 'JD']  # Alibaba, Tencent, JD.com

    # Fetch news for stock symbols
    for stock_symbol in banks + tech + traditional_industry + oils + chinese_stocks:
        print(f"Fetching news for {stock_symbol}...")
        main(stock_symbol, 'stock', fetch_news)
        print(f"Finished processing news for {stock_symbol}.")