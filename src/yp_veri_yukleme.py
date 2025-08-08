import pandas as pd
import numpy as np

# Dosya yolları ve platform isimleri - Burada sözlük kullanmalısın
file_paths = {
    "chatgpt": "data/tez_veriseti/chatgpt.csv",
    "bard": "data/tez_veriseti/bard.csv",
    "midjourney": "data/tez_veriseti/midjourney.csv",
    "runway": "data/tez_veriseti/runway.csv",
    "fireflies": "data/tez_veriseti/fireflies.csv"
}

# CSV dosyalarını yükle ve her birini ayrı DataFrame'lere al
chatgpt_df = pd.read_csv(file_paths["chatgpt"])
bard_df = pd.read_csv(file_paths["bard"])
midjourney_df = pd.read_csv(file_paths["midjourney"])
runway_df = pd.read_csv(file_paths["runway"])
fireflies_df = pd.read_csv(file_paths["fireflies"])

# Veri çerçevelerinin ilk 5 kaydını göster
print("ChatGPT Data:")
print(chatgpt_df.head())

print("\nBard Data:")
print(bard_df.head())

print("\nMidJourney Data:")
print(midjourney_df.head())

print("\nRunway Data:")
print(runway_df.head())

print("\nFireflies Data:")
print(fireflies_df.head())


# Tüm verileri tutmak için liste
dfs = []

for platform, path in file_paths.items():
    df = pd.read_csv(path)
    df['platform'] = platform  # Her veri setine platform bilgisini ekle
    dfs.append(df)

# Hepsini tek bir DataFrame’e birleştir
combined_df = pd.concat(dfs, ignore_index=True)

# Veri çerçevelerinin ilk 5 kaydını göster
print("ChatGPT Data:")
print(chatgpt_df.head())

print("\nBard Data:")
print(bard_df.head())

print("\nMidJourney Data:")
print(midjourney_df.head())

print("\nRunway Data:")
print(runway_df.head())

print("\nFireflies Data:")
print(fireflies_df.head())

# Gereksiz sütunları kaldırma
chatgpt_df_clean = chatgpt_df[['author_id', 'created_at', 'entities', 'text']].dropna()
bard_df_clean = bard_df[['author_id', 'created_at', 'entities', 'text']].dropna()
midjourney_df_clean = midjourney_df[['author_id', 'created_at', 'entities', 'text']].dropna()
runway_df_clean = runway_df[['author_id', 'created_at', 'entities', 'text']].dropna()
fireflies_df_clean = fireflies_df[['author_id', 'created_at', 'entities', 'text']].dropna()

# Birleştirilmiş veri
twitter_df= pd.concat([chatgpt_df_clean, bard_df_clean, midjourney_df_clean, runway_df_clean, fireflies_df_clean], ignore_index=True)

#veri seti uzunluğu:
len(twitter_df)

twitter_df.shape

# Temiz veriyi görüntüleme
print(twitter_df.head())
# ilk 5 satırın text gözükmesi
twitter_df["text"].head(5)
#boş satır var mı
twitter_df.isnull().sum()
twitter_df.info()

#created_at string formatında bunu tarih formatına dönüştürelim.
twitter_df['created_at'] = pd.to_datetime(twitter_df['created_at'])
twitter_df.head(5)
twitter_df.info()

""" Veri ön temizleme:
Tüm metinleri küçük harfe çevir
URL’leri kaldır Mention (@...) ve hashtag (#...) işaretlerini kaldır
Noktalama işaretlerini, rakamları ve özel karakterleri temizle
Gereksiz boşlukları temizle
Temizlenmiş metni cleaned_text adında yeni bir sütuna ekle"""

import re
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords, words
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from collections import defaultdict

# Gerekli veri kümelerini indir
nltk.download('stopwords')
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# Yardımcılar
english_words = set(words.words())
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# POS etiket haritalama
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


def clean_text(text):
    # Küçük harfe çevir
    text = text.lower()

    # URL'leri kaldır
    text = re.sub(r'http\S+', '', text)

    # Mention ve hashtag işaretlerini kaldır
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)

    # Sayıları ve özel karakterleri kaldır
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize et (kelimelere ayır)
    tokens = word_tokenize(text)

    # Stopwords'leri çıkar
    tokens = [word for word in tokens if word not in stop_words]

    # Kelimeleri lemmatize et
    lemmatized_tokens = []
    for word, tag in pos_tag(tokens):
        tag = tag[0].upper()  # POS tag'ini büyük harfe çevir (J -> Adjective, V -> Verb)
        wordnet_pos = tag_map[tag]  # POS etiketine göre doğru lemmatizer'ı seç
        lemmatized_tokens.append(lemmatizer.lemmatize(word, wordnet_pos))

    return ' '.join(lemmatized_tokens)


# Temizleme işlemi twitter_df için
twitter_df['cleaned_text'] = twitter_df['text'].apply(clean_text)

# Temizlenmiş veriyi görmek için
print("Cleaned Data:")
twitter_df[['cleaned_text']].head(5)

#boş satır kontrol etme
twitter_df[['cleaned_text']].isnull().sum()

#Örnek Bir Satır için Orijinal Metin, Temizlenmiş Metin ve POS Tagleri
# Örnek olarak ilk satırı alalım
sample_row = twitter_df.iloc[0]

original_text = sample_row['text']
cleaned_text = sample_row['cleaned_text']

# Temizlenmiş metni token'lara ayır
tokens = word_tokenize(cleaned_text)

# POS tag'leri al
pos_tags = pos_tag(tokens)

print("\n--- Örnek Satır ---")
print("Orijinal Metin:")
print(original_text)

print("\nTemizlenmiş Metin:")
print(cleaned_text)

print("\nPOS Tag'leri (Temizlenmiş Metin Üzerinden):")
for word, tag in pos_tags:
    print(f"{word} --> {tag}")


###verisetini kaydet
twitter_df.to_csv("cleaned_twitter_data.csv", index=False)


df=pd.read_csv('data/cleaned_twitter_data.csv')
print(len(df))

print(df.isnull().sum())