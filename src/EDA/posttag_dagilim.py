import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize
from collections import Counter
import matplotlib.pyplot as plt

# Gerekli nltk paketlerini indir
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Temizlenmiş veri dosyasını oku (CSV formatında)
twitter_df = pd.read_csv("data/cleaned_twitter_data.csv")

# POS tag'leri topla (İlk 5000 tweet üzerinden örnekleme yapılabilir)
tags = []
for text in twitter_df['cleaned_text'].dropna().head(5000):
    tokens = word_tokenize(text)
    tags += [tag for word, tag in pos_tag(tokens)]

# En sık geçen 10 POS etiketi
tag_counts = Counter(tags)
common_tags = tag_counts.most_common(10)

# Konsola detaylı yazdırma
print("POS Tag Frekansları (İlk 5000 Tweet):")
for tag, count in common_tags:
    print(f"{tag}: {count} adet")

print(f"\nToplam farklı POS etiketi sayısı: {len(tag_counts)}")
print(f"Toplam etiketlenen kelime sayısı: {sum(tag_counts.values())}")


# Görselleştirme
labels, counts = zip(*common_tags)
plt.figure(figsize=(10,6))
plt.bar(labels, counts, color='skyblue')
plt.title("POS Tag Dağılımı (İlk 5000 Tweet)")
plt.xlabel("POS Tags")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Görseli kaydet
plt.savefig("gorsel/pos_tag_dagilimi.png", dpi=300)

