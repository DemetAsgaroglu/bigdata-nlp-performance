import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import time
import os

df = pd.read_csv("data/cleaned_twitter_data.csv")

# cleaned_text sütununda boş olmayanları al
documents = df['cleaned_text'].dropna().tolist()


# 2. Belge-Terim Matrisi Oluştur

# CountVectorizer ile metni sayısal vektöre dönüştür
vectorizer = CountVectorizer(
    max_df=0.95,  # Çok sık geçen (genel) kelimeleri çıkar
    min_df=2,     # Çok nadir geçen kelimeleri çıkar
    stop_words='english'  # İngilizce stop-word'leri çıkar
)

# Vektör dönüşümü başlat
start_time = time.time()
dtm = vectorizer.fit_transform(documents)
end_time = time.time()

print(f"[INFO] Belge-terim matrisi oluşturuldu. Boyut: {dtm.shape}")
print(f"[TIME] Dönüşüm süresi: {end_time - start_time:.2f} saniye")


# 3. LDA Modelini Eğit

n_topics = 6  # Konu sayısı (veriye göre ayarlanabilir)
lda_model = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    learning_method='batch'  # alternatif: 'online'
)

# Modeli eğit
start_time = time.time()
lda_model.fit(dtm)
end_time = time.time()

print(f"[INFO] LDA modeli {n_topics} konu ile eğitildi.")
print(f"[TIME] Eğitim süresi: {end_time - start_time:.2f} saniye")


# 4. Konu Başlıklarını Görüntüle

feature_names = vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(lda_model.components_):
    print(f"\nKonu #{topic_idx + 1}:")
    top_words_idx = topic.argsort()[:-11:-1]  # En yüksek ağırlıklı 10 kelime
    top_words = [feature_names[i] for i in top_words_idx]
    print("Anahtar Kelimeler:", ", ".join(top_words))


# 5. Modeli ve Vectorizer'ı Kaydet

os.makedirs('models', exist_ok=True)

joblib.dump(lda_model, 'models/lda_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("[INFO] Model ve vectorizer 'models/' klasörüne kaydedildi.")

