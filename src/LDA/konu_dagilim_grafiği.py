import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


# 1. Gerekli Dosyaları Yükle

# Veriyi yükle
df = pd.read_csv("data/cleaned_twitter_data.csv")
documents = df['cleaned_text'].dropna().tolist()

# Eğitilmiş LDA modeli ve vectorizer'ı yükle
lda_model = joblib.load("models/lda_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Belgeleri dönüştür
dtm = vectorizer.transform(documents)

# Belgelerin konu olasılıkları
doc_topic_dist = lda_model.transform(dtm)

# 2. Konu Dağılımı Görselleştir

# Her konunun belgelerdeki ortalama katkısı
topic_weights = np.mean(doc_topic_dist, axis=0)

# Konu başlıkları
topic_labels = [
    "AI Platformları ve Hizmetleri",
    "Yapay Zeka ve Kripto Yatırımları",
    "Yaratıcı İçerik Üretimi ve Sanat",
    "Görsel Üretimde Prompt ve Stil Tasarımı",
    "AI Ürün Lansmanları ve Topluluk Katılımı",
    "Kullanıcı Yorumları ve Kişisel Tepkiler"
]

# Barplot çizimi
plt.figure(figsize=(10, 6))
sns.barplot(x=topic_weights, y=topic_labels)
plt.title("Her Konunun Belgelerdeki Ortalama Ağırlığı")
plt.xlabel("Ortalama Ağırlık")
plt.ylabel("Konu")
plt.tight_layout()

# Görsel klasörü varsa oluştur
os.makedirs("gorsel/LDA", exist_ok=True)

# Görseli kaydet
plt.savefig("gorsel/LDA/konu_agirliklari.png", dpi=300)
plt.close()
