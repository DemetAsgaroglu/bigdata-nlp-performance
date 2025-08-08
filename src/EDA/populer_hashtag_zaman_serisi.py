import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import ast

# Veriyi oku
df = pd.read_csv("data/cleaned_twitter_data.csv")

# entities sütununu sözlük formatına çevir
df['entities'] = df['entities'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {})

# created_at sütununu tarih formatına çevir
df['created_at'] = pd.to_datetime(df['created_at']).dt.date

# Günlük hashtagleri topla
daily_hashtags = defaultdict(list)

for _, row in df.iterrows():
    date = row['created_at']
    hashtags = row['entities'].get('hashtags', [])
    for tag in hashtags:
        daily_hashtags[date].append('#' + tag['tag'].lower())

# Günlük frekans tablosu (DataFrame)
hashtag_df = pd.DataFrame()

for date, tags in daily_hashtags.items():
    count = Counter(tags)
    for tag, freq in count.items():
        hashtag_df = pd.concat([hashtag_df, pd.DataFrame({'date': [date], 'hashtag': [tag], 'count': [freq]})])

# En sık geçen 5 hashtag’i al
top_hashtags = hashtag_df.groupby("hashtag")["count"].sum().sort_values(ascending=False).head(5).index

# Sadece bu hashtag’leri filtrele
filtered_df = hashtag_df[hashtag_df["hashtag"].isin(top_hashtags)]

# Pivot tablo (satır: tarih, sütun: hashtag, değer: count)
pivot_df = filtered_df.pivot_table(index="date", columns="hashtag", values="count", fill_value=0)

# Görselleştirme
plt.figure(figsize=(12, 6))
sns.lineplot(data=pivot_df)
plt.title("Günlere Göre En Popüler Hashtag'ler")
plt.xlabel("Tarih")
plt.ylabel("Frekans")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("gorsel/gunluk_hashtag_trendleri.png", dpi=300)
