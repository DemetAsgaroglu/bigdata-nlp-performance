import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Duygu analizi sonuçlarını içeren CSV'yi ve zaman bilgisini içeren orijinal veri setini yükle
df_sentiment = pd.read_csv('data/sentiment_scores.csv')  # Duygu skorlarını içeren dosya
df_twitter = pd.read_csv('data/cleaned_twitter_data.csv')  # Zaman bilgisini içeren dosya

# 'created_at' sütununu datetime formatına çevir
df_twitter['created_at'] = pd.to_datetime(df_twitter['created_at'], errors='coerce')

# 'created_at' ve duygu skorlarını birleştir
df_merged = pd.merge(df_sentiment, df_twitter[['created_at', 'cleaned_text']], left_index=True, right_index=True)

# Zaman sütunundaki eksik verileri temizle
df_merged = df_merged.dropna(subset=['created_at', 'vader_sentiment'])

# Zaman dilimini günlük olarak ayarlayalım
df_merged['date'] = df_merged['created_at'].dt.date

# Günlük duygu kategorilerini hesapla
df_merged['vader_category'] = df_merged['vader_sentiment'].apply(lambda score: 'Pozitif' if score > 0.05
                                                                else ('Negatif' if score < -0.05 else 'Nötr'))

# Günlük pozitif, nötr, negatif kategorilerinin sayısını hesapla
daily_sentiment_counts = df_merged.groupby(['date', 'vader_category']).size().unstack(fill_value=0).reset_index()

# Günlük duygu değişimi görselleştirelim
daily_sentiment_counts.set_index('date', inplace=True)

# Grafik çizimi
plt.figure(figsize=(12, 6))
daily_sentiment_counts.plot(kind='line', marker='o', figsize=(12, 6))
plt.title('Zamanla Duygu Kategorilerinin Değişimi')
plt.xlabel('Tarih')
plt.ylabel('Tweet Sayısı')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gorsel/DUYGU/zamanla_duygu_kategorileri.png')
plt.close()

print("Zamanla duygu kategorilerinin değişimi görselleştirildi.")
