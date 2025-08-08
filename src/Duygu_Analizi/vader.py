import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Süre ölçümü için tekil sistemde duygu analizi
def run_sentiment_analysis():
    # Veri setini yükleme
    start_loading = time.time()
    df = pd.read_csv('data/cleaned_twitter_data.csv')
    end_loading = time.time()
    loading_time = end_loading - start_loading
    print(f"Veri yükleme süresi: {loading_time:.2f} saniye")

    # VADER ile duygu analizi
    nltk.download('vader_lexicon', quiet=True)
    sid = SentimentIntensityAnalyzer()

    def get_vader_sentiment(text):
        if pd.isna(text):
            return np.nan
        scores = sid.polarity_scores(text)
        return scores['compound']

    # Duygu skorlarını hesaplama - süre ölçümü
    start_processing = time.time()
    df['vader_sentiment'] = df['cleaned_text'].apply(get_vader_sentiment)

    # Duygu kategorilerini oluşturma
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

    # Sonuçları hesaplama
    start_analysis = time.time()
    sentiment_counts = df['vader_category'].value_counts()
    end_analysis = time.time()
    analysis_time = end_analysis - start_analysis

    total_time = loading_time + processing_time + analysis_time

    print(f"İşleme süresi: {processing_time:.2f} saniye")
    print(f"Analiz süresi: {analysis_time:.2f} saniye")
    print(f"Toplam süre: {total_time:.2f} saniye")

    # Sonuçları görselleştirme
    plt.figure(figsize=(10, 6))
    sns.countplot(x='vader_category', data=df)
    plt.title('Duygu Analizi Sonuçları')
    plt.xlabel('Duygu Kategorisi')
    plt.ylabel('Tweet Sayısı')
    plt.savefig('gorsel/DUYGU/duygu_analizi_sonuclari.png')
    plt.close()

    # Süre sonuçlarını görselleştirme
    time_data = {
        'Aşama': ['Veri Yükleme', 'İşleme', 'Analiz', 'Toplam'],
        'Süre (saniye)': [loading_time, processing_time, analysis_time, total_time]
    }
    time_df = pd.DataFrame(time_data)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Aşama', y='Süre (saniye)', data=time_df)
    plt.title('İşlem Süreleri')
    plt.grid(True, alpha=0.3)
    plt.savefig('gorsel/DUYGU/islem_sureleri.png')
    plt.close()

    # Duygu skorlarını CSV'ye kaydet
    df[['cleaned_text', 'vader_sentiment', 'vader_category']].to_csv('data/sentiment_scores.csv', index=False)
    print("Duygu skorları 'data/sentiment_scores.csv' dosyasına kaydedildi.")

    return {
        'loading_time': loading_time,
        'processing_time': processing_time,
        'analysis_time': analysis_time,
        'total_time': total_time,
        'sentiment_counts': sentiment_counts,
        'df': df
    }


# Fonksiyonu çalıştırma
results = run_sentiment_analysis()

print("\nDuygu analizi sonuçları:")
print(results['sentiment_counts'])

