import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import time

# VADER leksikonunu indir
nltk.download('vader_lexicon', quiet=True)

# -------------------------------
# 1. Veriyi Yükle
# -------------------------------
start_loading = time.time()
df = pd.read_csv('data/cleaned_augmented_twitter_data.csv')
end_loading = time.time()
loading_time = end_loading - start_loading
print(f"[INFO] Veri yüklendi. Toplam kayıt: {len(df)}")
print(f"[TIME] Veri yükleme süresi: {loading_time:.2f} saniye")

# -------------------------------
# 2. Duygu Skorlarını Hesapla
# -------------------------------
sid = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    if pd.isna(text):
        return np.nan
    scores = sid.polarity_scores(text)
    return scores['compound']

start_processing = time.time()
df['vader_sentiment'] = df['cleaned_text'].apply(get_vader_sentiment)

def categorize_sentiment(score):
    if pd.isna(score):
        return 'Nötr'
    elif score > 0.05:
        return 'Pozitif'
    elif score < -0.05:
        return 'Negatif'
    else:
        return 'Nötr'

df['vader_category'] = df['vader_sentiment'].apply(categorize_sentiment)
end_processing = time.time()
processing_time = end_processing - start_processing
print(f"[TIME] İşleme süresi: {processing_time:.2f} saniye")

# -------------------------------
# 3. Analiz ve Süre
# -------------------------------
start_analysis = time.time()
sentiment_counts = df['vader_category'].value_counts()
end_analysis = time.time()
analysis_time = end_analysis - start_analysis
print(f"[TIME] Analiz süresi: {analysis_time:.2f} saniye")

# -------------------------------
# 4. Toplam Süre
# -------------------------------
total_time = loading_time + processing_time + analysis_time
print(f"[TOTAL TIME] Toplam süre: {total_time:.2f} saniye")

# -------------------------------
# 5. Sonuçları Yazdır
# -------------------------------
print("\n[SONUÇ] Duygu Kategorisi Dağılımı:")
print(sentiment_counts)
