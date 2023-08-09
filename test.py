from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

tableau_data = [
    "This app is amazing and easy to use.",
    "Terrible experience, lots of bugs and crashes.",
    "I love the new features in the latest update.",
    "The customer support is very helpful and responsive.",
    "Disappointed with the performance and lack of features.",
    "Great app with 5 stars!"
]


def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return sentiment


def preprocess_text(text):
    text = re.sub(r'[0-9]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


preprocessed_data = [preprocess_text(text) for text in tableau_data]

sentiments = [analyze_sentiment(text) for text in preprocessed_data]

sentiment_labels = ['Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral' for sentiment in sentiments]

positive_text = ' '.join([text for text, label in zip(tableau_data, sentiment_labels) if label == 'Positive'])
negative_text = ' '.join([text for text, label in zip(tableau_data, sentiment_labels) if label == 'Negative'])

wordcloud_positive = WordCloud(width=800, height=800, background_color='white').generate(positive_text)
wordcloud_negative = WordCloud(width=800, height=800, background_color='white').generate(negative_text)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title('Positive Sentiment Word Cloud')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title('Negative Sentiment Word Cloud')
plt.axis('off')

plt.show()
