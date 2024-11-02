import requests
from transformers import pipeline
import pyodbc
from datetime import datetime
import os
from dotenv import load_dotenv
from requests.status_codes import codes

# Load environment variables from .env file
load_dotenv()

# Constants
API_KEY = os.getenv('NEWS_API_KEY')  # Get API key from environment variables
TOP_HEADLINES_URL = 'https://newsapi.org/v2/top-headlines'
DB_SERVER = os.getenv('DB_SERVER')  # Get DB server from environment variables
DB_NAME = os.getenv('DB_NAME')  # Get DB name from environment variables

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

# Analyze sentiment using the BERT-based model from Hugging Face
def analyze_sentiment(text):
    # If the text is empty or None, return a default neutral sentiment
    if not text or text.strip() == "":
        print("Empty text provided for sentiment analysis. Returning default values.")
        return "neutral", 0.0

    # Otherwise, process the text with the sentiment analysis model
    result = sentiment_pipeline(text[:512])  # Limit input to 512 tokens (BERT max length)
    sentiment_label = result[0]['label']  # Extract the predicted label
    confidence_score = result[0]['score']  # Extract the confidence score

    # Print the result for debugging or inspection
    print(f"Sentiment Label: {sentiment_label}, Confidence Score: {confidence_score}")

    return sentiment_label, confidence_score  # Return both label and confidence score

# Connect to MSSQL database using Windows Authentication
def connect_db():
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={DB_SERVER};"
        f"DATABASE={DB_NAME};"
        f"Trusted_Connection=yes;"  # Use Windows Authentication
    )
    conn = pyodbc.connect(conn_str)
    return conn

# Store news article with sentiment in the database
def store_news_article(conn, article, sentiment_label, confidence_score, source):
    with conn.cursor() as cursor:
        insert_query = """
            INSERT INTO NewsArticles (
                stock_symbol, source_name, author, title, description, url, published_at, content, sentiment_label, confidence_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(insert_query, (
            source,  # The source could be a stock symbol, country, or region
            article['source']['name'],
            article.get('author', 'Unknown'),
            article.get('title', ''),
            article.get('description', ''),
            article.get('url', ''),
            datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),  # Ensure correct datetime format
            article.get('content', ''),
            sentiment_label,  # Store the sentiment label (e.g., '5 stars')
            confidence_score  # Store the confidence score
        ))
        conn.commit()

# Main function to fetch, analyze, and store news articles
def main(source, fetch_function):
    # Step 1: Fetch news articles (could be top headlines or stock-related news)
    articles = fetch_function(source)

    if articles:
        # Step 2: Connect to the database
        conn = connect_db()

        for article in articles:
            # Step 3: Perform sentiment analysis on the article content
            content = article.get('content', '') or article.get('description', '')

            # Get both the sentiment label and confidence score from the analyze_sentiment function
            sentiment_label, confidence_score = analyze_sentiment(content)

            # Step 4: Store the article and its sentiment (including confidence) in the database
            store_news_article(conn, article, sentiment_label, confidence_score, source)

        # Close the database connection
        conn.close()

# Run the script to fetch headlines for each region
if __name__ == '__main__':
    # Fetch top headlines for APAC countries
    for country_code in APAC_COUNTRIES:
        print(f"Fetching top headlines for {country_code} (APAC)...")
        main(country_code, fetch_top_headlines)
        print(f"Finished processing headlines for {country_code} (APAC).")

    # Fetch top headlines for EMEA countries
    for country_code in EMEA_COUNTRIES:
        print(f"Fetching top headlines for {country_code} (EMEA)...")
        main(country_code, fetch_top_headlines)
        print(f"Finished processing headlines for {country_code} (EMEA).")

    # Fetch top headlines for Americas countries
    for country_code in AMERICAS_COUNTRIES:
        print(f"Fetching top headlines for {country_code} (Americas)...")
        main(country_code, fetch_top_headlines)
        print(f"Finished processing headlines for {country_code} (Americas).")