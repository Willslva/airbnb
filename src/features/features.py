from textblob import TextBlob

def calculate_sentiment(comment):
    if comment is None:
        return 0
    analysis = TextBlob(comment)
    return analysis.sentiment.polarity