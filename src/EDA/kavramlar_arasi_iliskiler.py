import pandas as pd
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# 1. Veri setini yükle
df = pd.read_csv("data/cleaned_twitter_data.csv")
texts = df['cleaned_text'].dropna().tolist()

# 2. Kelime çiftlerini oluştur (her tweet içindeki benzersiz kelime kombinasyonları)
word_pairs = []
for text in texts:
    words = list(set(text.split()))
    if len(words) > 1:
        word_pairs.extend(itertools.combinations(words, 2))

# 3. En sık geçen ilk 50 kelime çiftini seç
pair_counts = Counter(word_pairs).most_common(50)

# 4. Ağ grafiğini oluştur
G = nx.Graph()
for (w1, w2), weight in pair_counts:
    G.add_edge(w1, w2, weight=weight)

# 5. Düğüm boyutlarını ayarla (kaç bağlantısı olduğuna göre)
node_sizes = [G.degree(n) * 200 for n in G.nodes]

# 6. Kenar kalınlıkları ve renkler
edges = G.edges(data=True)
weights = [edata["weight"] for _, _, edata in edges]

# 7. Görselleştirme
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=0.5, seed=42)  # daha düzgün dağılmış bir yerleşim

# Arka planı beyaz yapalım
plt.gca().set_facecolor('white')

nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=node_sizes, alpha=0.8)
nx.draw_networkx_edges(G, pos, width=[w * 0.3 for w in weights], edge_color=weights, edge_cmap=plt.cm.Blues, alpha=0.6)
nx.draw_networkx_labels(G, pos, font_size=9, font_family='sans-serif')

plt.title("Kavramlar Arası İlişki Haritası (Top 50 Çift)", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig("kavram_ag_gelismis.png", dpi=300)
plt.show()

# 8. Konsola en güçlü bağlantıları yazdır
print("\nEn güçlü 10 kavram bağlantısı:")
for (w1, w2), weight in pair_counts[:10]:
    print(f"{w1} --- {w2} : {weight} kez birlikte geçti")
