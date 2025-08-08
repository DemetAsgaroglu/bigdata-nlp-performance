import pandas as pd
import matplotlib.pyplot as plt

# Veriyi oluştur
data = {
    'Yöntem': ['VADER', 'VADER', 'LDA', 'LDA'],
    'Veri Miktarı': [597894, 1193680, 597894, 1193680],
    'Tekil Süre (sn)': [142.18, 164.61, 454.73, 3436.71],
    'Dağıtık Süre (sn)': [78.27, 9.91, 597.88, 2715.03]
}

df = pd.DataFrame(data)

# Yüzdelik değişimi hesapla
df['Yüzdelik Değişim (%)'] = ((df['Tekil Süre (sn)'] - df['Dağıtık Süre (sn)']) / df['Tekil Süre (sn)']) * 100

# Bar grafiği oluştur
plt.figure(figsize=(5, 5))  # Genişlik: 12 inç, Yükseklik: 6 inç
bars = plt.bar(
    df['Yöntem'] + " (" + df['Veri Miktarı'].astype(str) + ")",
    df['Yüzdelik Değişim (%)'],
    color=['skyblue', 'orange', 'skyblue', 'orange'],
    width=0.3  # Çubuk genişliğini daralt
)

# Değerleri çubukların üzerine ekle
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

# Grafik detayları
plt.title('Tekil ve Dağıtık Sistemler Arasındaki Yüzdelik Süre Değişimi', fontsize=14)
plt.xlabel('Yöntem ve Veri Miktarı', fontsize=12)
plt.ylabel('Yüzdelik Değişim (%)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Grafiği kaydet
plt.tight_layout()
plt.savefig('gorsel/yuzdelik_degisim_postere_uygun3.png', dpi=300)
plt.close()