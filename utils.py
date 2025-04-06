from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import emoji
import re
from textblob import TextBlob


def extract_emojis(text):
    return [char for char in text if emoji.is_emoji(char)]

def fetch_emoji_stats(messages):
    all_emojis = []
    for msg in messages:
        all_emojis += extract_emojis(msg)
    emoji_counts = Counter(all_emojis).most_common()
    return pd.DataFrame(emoji_counts, columns=['Emoji', 'Count'])


def most_common_words(messages, stopwords):
    words = []
    for msg in messages:
        msg = re.sub(r'[^\w\s]', '', msg)
        for word in msg.lower().split():
            if word not in stopwords:
                words.append(word)
    return pd.DataFrame(Counter(words).most_common(20), columns=['Word', 'Count'])

def generate_wordcloud(messages):
    text = " ".join(messages)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wc

def sentiment_analysis(messages):
    sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
    for msg in messages:
        score = TextBlob(msg).sentiment.polarity
        if score > 0.1:
            sentiments['positive'] += 1
        elif score < -0.1:
            sentiments['negative'] += 1
        else:
            sentiments['neutral'] += 1
    return sentiments

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(message):
    return analyzer.polarity_scores(message)['compound']

def add_sentiment_column(df):
    df['sentiment'] = df['message'].apply(analyze_sentiment)
    return df
