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
NEWS_API_URL = 'https://newsapi.org/v2/everything'
DB_SERVER = os.getenv('DB_SERVER')  # Get DB server from environment variables
DB_NAME = os.getenv('DB_NAME')  # Get DB name from environment variables

# Initialize the Hugging Face sentiment analysis pipeline (using a multilingual BERT-based model)
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Fetch news articles from NewsAPI
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

# Analyze sentiment using the BERT-based model from Hugging Face
def analyze_sentiment(text):
    result = sentiment_pipeline(text[:512])  # Limit input to 512 tokens (BERT max length)
    sentiment_label = result[0]['label']  # Extract the predicted label (e.g., '5 stars')
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
def store_news_article(conn, article, sentiment_label, confidence_score, stock_symbol):
    with conn.cursor() as cursor:
        insert_query = """
            INSERT INTO NewsArticles (
                stock_symbol, source_name, author, title, description, url, published_at, content, sentiment_label, confidence_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(insert_query, (
            stock_symbol,
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
def main(stock_symbol):
    # Step 1: Fetch news articles
    articles = fetch_news(stock_symbol)

    if articles:
        # Step 2: Connect to the database
        conn = connect_db()

        for article in articles:
            # Step 3: Perform sentiment analysis on the article content
            content = article.get('content', '') or article.get('description', '')
            
            # Get both the sentiment label and confidence score from the analyze_sentiment function
            sentiment_label, confidence_score = analyze_sentiment(content)

            # Step 4: Store the article and its sentiment (including confidence) in the database
            store_news_article(conn, article, sentiment_label, confidence_score, stock_symbol)

        # Close the database connection
        conn.close()

# Run the script for a specific stock symbol, e.g., Apple (AAPL)
if __name__ == '__main__':
    # Define stock symbols for each sector
    banks = ['JPM', 'BAC', 'BCS']  # JPMorgan, Bank of America, Barclays    
    tech = ['AAPL', 'MSFT', 'GOOGL']  # Apple, Microsoft, Alphabet (Google)
    traditional_industry = ['XOM', 'CVX', 'MMM']  # ExxonMobil, Chevron, 3M    
    oils = ['XOM', 'CVX', 'BP']  # Example: ExxonMobil, Chevron, BP
    chinese_stocks = ['BABA', 'TCEHY', 'JD']  # Example: Alibaba, Tencent, JD.com

    # Loop through each sector and stock symbol
    for stock_symbol in banks + tech + traditional_industry + oils + chinese_stocks:
        print(f"Fetching news for {stock_symbol}...")
        main(stock_symbol)
        print(f"Finished processing news for {stock_symbol}.")