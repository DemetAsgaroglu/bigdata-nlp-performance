import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# Temizlenmiş veriyi yükleyelim
twitter_df = pd.read_csv("data/cleaned_twitter_data.csv")

# Boş satır kontrol edelim
twitter_df["cleaned_text"].isnull().sum()

# Boş verileri temizleyelim (isteğe bağlı)
twitter_df['cleaned_text'].fillna('', inplace=True)

# CountVectorizer ile kelime sıklığını sayma
vectorizer = CountVectorizer(stop_words='english', max_features=20)  # Sadece İngilizce stopword'leri çıkarıyoruz
X = vectorizer.fit_transform(twitter_df['cleaned_text'])

# Kelime sıklıklarını veriye dönüştür
word_freq = np.asarray(X.sum(axis=0)).flatten()

# Kelimeler ve sıklıkları
words = vectorizer.get_feature_names_out()

# Sıklıkları bir DataFrame'e dönüştür
word_freq_df = pd.DataFrame({
    'word': words,
    'frequency': word_freq
})

# Sıklığı en yüksek 10 kelimeyi göster
top_words = word_freq_df.sort_values(by='frequency', ascending=False).head(10)
print(top_words)

# Kelime sıklığı grafiği
plt.figure(figsize=(10,6))
plt.barh(top_words['word'], top_words['frequency'], color='skyblue')
plt.xlabel('Frequency')
plt.title('Popüler 10 Kelime')
plt.gca().invert_yaxis()  # Y eksenini tersten sıralamak için

# Grafiği kaydedelim
plt.savefig("gorsel/populer_10_kelime.png")

