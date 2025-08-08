import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import time

# -------------------------------
# 1. Veriyi Yükle
# -------------------------------
start_loading = time.time()
df = pd.read_csv("data/cleaned_augmented_twitter_data.csv")
documents = df['cleaned_text'].dropna().tolist()
end_loading = time.time()
loading_time = end_loading - start_loading
print(f"[INFO] Veri yüklendi. Toplam belge sayısı: {len(documents)}")
print(f"[TIME] Veri yükleme süresi: {loading_time:.2f} saniye")

# -------------------------------
# 2. Belge-Terim Matrisi Oluştur
# -------------------------------
vectorizer = CountVectorizer(
    max_df=0.95,
    min_df=2,
    stop_words='english'
)

start_vectorizing = time.time()
dtm = vectorizer.fit_transform(documents)
end_vectorizing = time.time()
vectorizing_time = end_vectorizing - start_vectorizing
print(f"[INFO] Belge-terim matrisi oluşturuldu. Boyut: {dtm.shape}")
print(f"[TIME] Vektörleme süresi: {vectorizing_time:.2f} saniye")

# -------------------------------
# 3. LDA Modelini Eğit
# -------------------------------
n_topics = 6  # Konu sayısı
lda_model = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    learning_method='batch'
)

start_training = time.time()
lda_model.fit(dtm)
end_training = time.time()
training_time = end_training - start_training
print(f"[INFO] LDA modeli eğitildi ({n_topics} konu).")
print(f"[TIME] Eğitim süresi: {training_time:.2f} saniye")

# -------------------------------
# 4. Toplam Süre
# -------------------------------
total_time = loading_time + vectorizing_time + training_time
print(f"[TOTAL TIME] Toplam süre: {total_time:.2f} saniye")

# -------------------------------
# 5. Konu Anahtar Kelimeleri
# -------------------------------
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda_model.components_):
    print(f"\nKonu #{topic_idx + 1}:")
    top_words_idx = topic.argsort()[:-11:-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print("Anahtar Kelimeler:", ", ".join(top_words))
