from transformers import pipeline
from textblob import TextBlob

# Initialize the Hugging Face sentiment analysis pipeline
bert_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Perform sentiment analysis using BERT and TextBlob
def analyze_sentiment(text):
    if not text or text.strip() == "":
        return "neutral", 0.0, "neutral", 0.0, 0.0

    # BERT analysis
    bert_result = bert_pipeline(text[:512])  # Limit to 512 tokens
    bert_sentiment_label = bert_result[0]['label']
    bert_confidence_score = bert_result[0]['score']

    # TextBlob analysis
    textblob = TextBlob(text)
    textblob_polarity = textblob.sentiment.polarity
    textblob_subjectivity = textblob.sentiment.subjectivity
    textblob_sentiment_label = "positive" if textblob_polarity > 0 else "negative" if textblob_polarity < 0 else "neutral"

    return bert_sentiment_label, bert_confidence_score, textblob_sentiment_label, textblob_polarity, textblob_subjectivity