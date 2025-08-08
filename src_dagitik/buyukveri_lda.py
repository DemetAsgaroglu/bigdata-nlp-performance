from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.clustering import LDA
from pyspark.sql.functions import split
import time

# Toplam süreyi başlat
total_start_time = time.time()

# Spark oturumu başlat
spark = SparkSession.builder \
    .appName("1200k_LDA_Model") \
    .master("spark://192.168.56.102:7077") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

print(f"[INFO] Veri seti okunuyor...")
df = spark.read.csv("file:///home/seed/Desktop/dagitik/cleaned_augmented_twitter_data.csv", header=True, inferSchema=True)
print(f"[INFO] Veri seti okuma islemi tamamlandı.")

df = df.select("cleaned_text").dropna()
print(f"[INFO] Veri seti dropna islemi tamamlandı.")

df = df.repartition(8)
print(f"[INFO] Veri seti repartition islemi tamamlandı.")

df = df.withColumn("tokens", split(df["cleaned_text"], " "))
print(f"[INFO] Tokenize islemi tamamlandı.")

cv = CountVectorizer(inputCol="tokens", outputCol="features", vocabSize=10000, minDF=2)
cv_model = cv.fit(df)
vectorized_df = cv_model.transform(df)
print(f"[INFO] Belge-Terim matrisi olusturuldu.")

# Konu sayısı 6 olarak ayarlandı
lda = LDA(k=6, seed=42, featuresCol="features")

print(f"[INFO] LDA modeli eğitimi başlıyor...")
start_time = time.time()
lda_model = lda.fit(vectorized_df)
end_time = time.time()
print(f"[INFO] LDA modeli eğitildi.")
print(f"[TIME] Eğitim süresi: {end_time - start_time:.2f} saniye")

# Konuları yazdır
print(f"\n[INFO] Konular yazdırılıyor:\n")
topics = lda_model.describeTopics(10)
vocab = cv_model.vocabulary

topics_list = topics.collect()
for i, topic in enumerate(topics_list):
    words = [vocab[idx] for idx in topic['termIndices']]
    print(f"Topic {i + 1}: {', '.join(words)}")

# Toplam süreyi hesapla
total_end_time = time.time()
print(f"\n[TIME] 1200k için Toplam süre: {total_end_time - total_start_time:.2f} saniye")

# Spark oturumunu kapat
spark.stop()
