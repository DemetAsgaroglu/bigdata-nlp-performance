import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os



# Veriyi oku ve tarih sütununu dönüştür
df = pd.read_csv("data/cleaned_twitter_data.csv")
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

# NaN değerlerini kontrol et
print(df['cleaned_text'].isna().sum())  # NaN sayısını yazdır

# NaN değerleri 'empty' (boş string) ile dolduralım
df['cleaned_text'] = df['cleaned_text'].fillna('')

# Boş stringleri kontrol et
print(df['cleaned_text'].isna().sum())  # NaN değerlerinin olmadığını tekrar kontrol et

# 2023-01-01 ile 2023-05-01 arası verileri al
df_clean = df[(df['created_at'] >= '2023-01-01') & (df['created_at'] <= '2023-05-01')]

# Model ve vectorizer'ı yükle
lda_model = joblib.load("models/lda_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Belgeleri vektörleştir
documents = df_clean['cleaned_text'].tolist()
dtm = vectorizer.transform(documents)

# Belgelerin konu dağılımı
doc_topic_dist = lda_model.transform(dtm)

# Dominant topic seç (en yüksek olasılık)
dominant_topics = np.argmax(doc_topic_dist, axis=1)

# Konu başlıkları
topic_labels = [
    "AI Platformları ve Hizmetleri",
    "Yapay Zeka ve Kripto Yatırımları",
    "Yaratıcı İçerik Üretimi ve Sanat",
    "Görsel Üretimde Prompt ve Stil Tasarımı",
    "AI Ürün Lansmanları ve Topluluk Katılımı",
    "Kullanıcı Yorumları ve Kişisel Tepkiler"
]

# DataFrame'e dominant topic ve ay bilgisi ekle
document_topics_df = pd.DataFrame({
    'created_at': df_clean['created_at'].values,
    'dominant_topic': dominant_topics
})
document_topics_df['month'] = document_topics_df['created_at'].dt.month

# Sadece Ocak–Mayıs aylarını al (1–5)
document_topics_df = document_topics_df[document_topics_df['month'].isin([1, 2, 3, 4, 5])]

# Ay + Konu'ya göre gruplama
topic_time_df = document_topics_df.groupby(['month', 'dominant_topic']).size().reset_index(name='count')

# Pivot tablo (ay x konu)
pivot_df = topic_time_df.pivot(index='month', columns='dominant_topic', values='count').fillna(0)
pivot_df.columns = [topic_labels[i] if i < len(topic_labels) else f"Konu {i}" for i in pivot_df.columns]

# Ay isimleri
month_names = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs']
pivot_df.index = [month_names[i - 1] for i in pivot_df.index]  # 1 tabanlı aylar


# 2. Grafiği Çiz ve Kaydet

plt.figure(figsize=(14, 8))
pivot_df.plot(kind='line', marker='o', linewidth=2)
plt.title('Konuların Aya Göre Zamansal Dağılımı', fontsize=16)
plt.xlabel('Ay', fontsize=12)
plt.ylabel('Belge Sayısı', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Konular', fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()

# Kaydet
plt.savefig("gorsel/LDA/aylara_gore_konularin_zamansal_dagilimi.png", dpi=300)
plt.close()
