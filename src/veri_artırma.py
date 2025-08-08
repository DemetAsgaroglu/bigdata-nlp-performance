import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# NLTK kaynaklarını indir
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

# Veriyi yükle
df = pd.read_csv('data/cleaned_twitter_data.csv')


# Synonym Replacement fonksiyonu
def synonym_replacement(text, n=1):
    words = word_tokenize(text)
    new_words = words.copy()

    # Boş satırları ve NaN değerleri atla
    if not isinstance(text, str) or text.strip() == "":
        return text

    # Her kelime için eşanlamlı kelimelerle değişim yap
    for i, word in enumerate(words):
        synonyms = set()

        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())

        # En az 1 eşanlamlı varsa, o kelimeyi değiştir
        if len(synonyms) > 0:
            synonym = list(synonyms)[0]  # İlk eşanlamlıyı al
            new_words[i] = synonym

    return " ".join(new_words)


# Veri artırma işlemi
augmented_data = []
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    text = row['cleaned_text']

    # NaN, boş ve geçersiz metinleri atla
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        continue

    augmented_data.append(text)  # Orijinal veriyi ekle

    # Synonym replacement ile yeni veri üret
    augmented_text = synonym_replacement(text)
    augmented_data.append(augmented_text)  # Yeni veriyi ekle

# Yeni verileri DataFrame'e kaydet
augmented_df = pd.DataFrame({'cleaned_text': augmented_data})
augmented_df.to_csv('data/cleaned_augmented_twitter_data.csv', index=False)

print("Veri artırma işlemi tamamlandı ve 'data/cleaned_augmented_twitter_data.csv' dosyasına kaydedildi.")

df_yeni=pd.read_csv("data/cleaned_augmented_twitter_data.csv")
print(len(df_yeni))

import pandas as pd

# Artırılmış veriyi oku
augmented_df = pd.read_csv('data/cleaned_augmented_twitter_data.csv')

# İlk 5 satırı görüntüle
print(augmented_df.head())

# Veri uzunluğunu yazdır
print("Toplam veri sayısı:", len(augmented_df))
