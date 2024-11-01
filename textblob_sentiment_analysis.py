from textblob import TextBlob

def analyze_sentiment(text):
    analysis = TextBlob(text)
    # Get the sentiment polarity (-1 to 1 scale)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0:
        return 'Positive'
    elif polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'