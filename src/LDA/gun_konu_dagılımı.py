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

# NaN değerlerini 'empty' (boş string) ile doldur
df['cleaned_text'] = df['cleaned_text'].fillna('')

# NaN değerlerinin olmadığını kontrol et
print(df['cleaned_text'].isna().sum())  # NaN değerlerinin olmadığını kontrol et

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

# DataFrame'e dominant topic ve gün bilgisi ekle
document_topics_df = pd.DataFrame({
    'created_at': df_clean['created_at'].values,
    'dominant_topic': dominant_topics
})
document_topics_df['day'] = document_topics_df['created_at'].dt.date  # Gün bazında gruplama

# Sadece Ocak–Mayıs aylarını al (1–5)
document_topics_df = document_topics_df[document_topics_df['created_at'].dt.month.isin([1, 2, 3, 4, 5])]

# Gün + Konu'ya göre gruplama
topic_time_df = document_topics_df.groupby(['day', 'dominant_topic']).size().reset_index(name='count')

# Pivot tablo (gün x konu)
pivot_df = topic_time_df.pivot(index='day', columns='dominant_topic', values='count').fillna(0)
pivot_df.columns = [topic_labels[i] if i < len(topic_labels) else f"Konu {i}" for i in pivot_df.columns]


# 2. Konsola Popüler Konuları Yazdırma


# Günlere göre popüler konuları yazdır
for day, day_data in pivot_df.iterrows():
    print(f"\nGün: {day}")
    popular_topic = day_data.idxmax()  # En popüler konu
    print(f"Popüler Konu: {popular_topic}, Belge Sayısı: {day_data.max()}")


# 3. Grafiği Çiz ve Kaydet

# Grafiği çiz ve kaydet
plt.figure(figsize=(18, 10))  # Grafik boyutunu büyüttük
pivot_df.plot(kind='line', marker='o', linewidth=3, markersize=6)  # Çizgi kalınlığı arttırıldı

plt.title('Konuların Günlere Göre Zamansal Dağılımı', fontsize=20)
plt.xlabel('Gün', fontsize=14)
plt.ylabel('Belge Sayısı', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

# X eksenindeki etiketleri döndürdük
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Daha iyi yerleşim için tight_layout kullanıldı
plt.tight_layout()

# Legende başlık ve yazı büyüklüğü eklendi
plt.legend(title='Konular', fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')


#kaydet
plt.savefig("gorsel/LDA/konularin_gunlere_gore_dagilimi.png", dpi=300)
plt.close()


# -------------------------------
# 4. Sadece Mart Ayı İçin Grafik
# -------------------------------

# Mart ayı verisini filtrele
march_df = document_topics_df[
    (document_topics_df['created_at'].dt.month == 3) &
    (document_topics_df['created_at'].dt.year == 2023)
]

# Gün ve dominant topic'e göre gruplama
march_grouped = march_df.groupby(['day', 'dominant_topic']).size().reset_index(name='count')

# Pivot tablo oluştur (gün x konu)
march_pivot = march_grouped.pivot(index='day', columns='dominant_topic', values='count').fillna(0)
march_pivot.columns = [topic_labels[i] if i < len(topic_labels) else f"Konu {i}" for i in march_pivot.columns]

# Konsola yazdır (isteğe bağlı)
print("\nMart Ayı Günlük En Popüler Konular:")
for day, row in march_pivot.iterrows():
    popular_topic = row.idxmax()
    print(f"{day}: {popular_topic} ({int(row.max())} tweet)")

# Grafik çizimi
plt.figure(figsize=(16, 9))
march_pivot.plot(
    kind='bar',
    stacked=True,
    colormap='tab20',
    figsize=(16, 9)
)

plt.title('Mart 2023 Günlerine Göre Konu Dağılımı (Yığılmış Grafik)', fontsize=20)
plt.xlabel('Tarih', fontsize=14)
plt.ylabel('Tweet Sayısı', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Konular', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, title_fontsize=13)

plt.tight_layout()
plt.savefig("gorsel/LDA/mart_konu_dagilimi_stackbar.png", dpi=300)
plt.close()
