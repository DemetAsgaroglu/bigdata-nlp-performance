import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import time

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

df = pd.read_csv('data/cleaned_augmented_twitter_data.csv')
print(f"[INFO] Veri yüklendi: {len(df)} kayıt")

sid = SentimentIntensityAnalyzer()

def get_avg_sentiment(text):
    if pd.isna(text):
        return np.nan
    sentences = sent_tokenize(text)
    if not sentences:
        return 0.0
    scores = [sid.polarity_scores(sent)['compound'] for sent in sentences]
    return np.mean(scores)

start = time.time()

df['vader_sentiment'] = df['cleaned_text'].apply(get_avg_sentiment)

def categorize(score):
    if pd.isna(score):
        return 'Nötr'
    elif score > 0.05:
        return 'Pozitif'
    elif score < -0.05:
        return 'Negatif'
    else:
        return 'Nötr'

df['vader_category'] = df['vader_sentiment'].apply(categorize)

end = time.time()
print(f"[TIME] Toplam duygu analizi süresi: {end - start:.2f} saniye")
print(df['vader_category'].value_counts())
