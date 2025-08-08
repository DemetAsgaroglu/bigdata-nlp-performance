import pandas as pd
import matplotlib.pyplot as plt

# Temizlenmiş veriyi yükleyelim
twitter_df = pd.read_csv("data/cleaned_twitter_data.csv")

# 'created_at' sütununun tarih formatınını kontrol edelim
twitter_df.info()

# Eğer tarih formatı düzgün değilse, aşağıdaki gibi dönüştürebiliriz:
twitter_df['created_at'] = pd.to_datetime(twitter_df['created_at'], errors='coerce')

# Boş satır kontrolleri
twitter_df['created_at'].isnull().sum()

# Zaman dilimlerine göre tweet sayısını inceleyelim
tweets_per_day = twitter_df.groupby(twitter_df['created_at'].dt.date).size()

# Veriyi görselleştirelim (örneğin günlük tweet sayısı)
plt.figure(figsize=(10, 6))
tweets_per_day.plot(color='skyblue', title='Tweet Sayısı (Günlük)', legend=False)
plt.xlabel('Tarih')
plt.ylabel('Tweet Sayısı')
plt.grid(True)
plt.xticks(rotation=45)

# Görseli 'gorsel' klasörüne kaydedelim
plt.tight_layout()  # Görselin kesilmesini önler
plt.savefig('gorsel/tweet_sayisi_gunluk.png')

