import pandas as pd
import matplotlib.pyplot as plt
import ast  # String'i sözlüğe çevirmek için
from collections import Counter
import os

# CSV dosyasını yükle
df = pd.read_csv("data/cleaned_twitter_data.csv")

# entities sütunu sözlük olarak yorumlanıyor
df['entities'] = df['entities'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {})

# Hashtag'leri topla
hashtags = []
for row in df['entities']:
    if 'hashtags' in row:
        hashtags += ['#' + tag['tag'].lower() for tag in row['hashtags']]

# En sık geçen ilk 10 hashtag
top_hashtags = Counter(hashtags).most_common(10)

print("En Sık Kullanılan 10 Hashtag:")
for tag, count in top_hashtags:
    print(f"{tag}: {count} kez")

labels, counts = zip(*top_hashtags)

# Pie chart çiz
plt.figure(figsize=(8, 8))
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('En Sık Kullanılan Hashtagler')

# Kaydet
plt.savefig("gorsel/en_sik_hashtagler.png", dpi=300, bbox_inches='tight')
plt.close()

print("Grafik başarıyla kaydedildi: gorsel/en_sik_hashtagler.png")
