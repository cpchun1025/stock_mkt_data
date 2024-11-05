import os
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants for DB connection
DB_SERVER = os.getenv('DB_SERVER')
DB_NAME = os.getenv('DB_NAME')

# SQLAlchemy setup
Base = declarative_base()

# Engine and session setup
DATABASE_URL = f"mssql+pyodbc://{DB_SERVER}/{DB_NAME}?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# Define NewsApiSource model
class NewsApiSource(Base):
    __tablename__ = 'NewsApiSource'

    id = Column(Integer, primary_key=True, autoincrement=True)
    api_source_id = Column(String(255), nullable=False, unique=True, index=True)  # API returned source ID (e.g., 'the-irish-times')
    api_source_name = Column(String(1024), nullable=False)  # API returned source name (e.g., 'The Irish Times')
    description = Column(Text)  # Optional description

    # Relationship to NewsSources
    news_sources = relationship('NewsSources', back_populates='api_source')

# Define NewsSources model
class NewsSources(Base):
    __tablename__ = 'NewsSources'

    id = Column(Integer, primary_key=True, autoincrement=True)
    api_source_id = Column(Integer, ForeignKey('NewsApiSource.id'), nullable=False, index=True)  # Link to NewsApiSource
    source_code = Column(String(10))  # Stock symbol or country code (optional)
    source_type = Column(String(50))  # Type of source: 'stock' or 'country'

    # Relationship to NewsApiSource
    api_source = relationship('NewsApiSource', back_populates='news_sources')

    # Relationship to NewsArticles
    articles = relationship('NewsArticles', back_populates='source')

# Define NewsArticles model
class NewsArticles(Base):
    __tablename__ = 'NewsArticles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(Integer, ForeignKey('NewsSources.id'), nullable=False, index=True)
    author = Column(String(1024))
    title = Column(Text)
    description = Column(Text)
    url = Column(String(2048), index=True)
    published_at = Column(DateTime, index=True)
    content = Column(Text)

    # Relationship to NewsSources
    source = relationship('NewsSources', back_populates='articles')

    # Relationship to SentimentAnalysis
    sentiments = relationship('SentimentAnalysis', back_populates='article')

# Define SentimentAnalysis model
class SentimentAnalysis(Base):
    __tablename__ = 'SentimentAnalysis'

    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(Integer, ForeignKey('NewsArticles.id'), nullable=False, index=True)
    model_name = Column(String(100), nullable=False)  # e.g., 'BERT', 'TextBlob'
    sentiment_label = Column(String(50))  # e.g., 'positive', 'neutral', 'negative'
    confidence_score = Column(Float, nullable=True)  # Confidence score for models like BERT
    polarity = Column(Float, nullable=True)  # Polarity score for TextBlob (-1 to 1)
    subjectivity = Column(Float, nullable=True)  # Subjectivity score for TextBlob (0 to 1)

    # Relationship to NewsArticles
    article = relationship('NewsArticles', back_populates='sentiments')

# Create all tables in the database
def create_tables():
    Base.metadata.create_all(engine)