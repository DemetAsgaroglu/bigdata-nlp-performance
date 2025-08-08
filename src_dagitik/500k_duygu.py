from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, trim, length
from pyspark.sql.types import StringType, FloatType
import time
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# === Spark başlat ===
def create_spark():
    return SparkSession.builder \
        .appName("VADER_Sentiment_Analysis_Spark") \
        .master("spark://192.168.56.102:7077") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

# === NLTK VADER ayarı ===
nltk.data.path.append("/home/seed/nltk_data")
try:
    nltk.data.find('sentiment/vader_lexicon')
except:
    nltk.download('vader_lexicon', download_dir="/home/seed/nltk_data")

analyzer = SentimentIntensityAnalyzer()

def get_vader_score(text):
    if not text:
        return None
    try:
        return float(analyzer.polarity_scores(text)['compound'])
    except:
        return None

def get_sentiment_label(score):
    if score is None:
        return "Nötr"
    elif score > 0.05:
        return "Pozitif"
    elif score < -0.05:
        return "Negatif"
    else:
        return "Nötr"

# === Spark başlat ===
spark = create_spark()

# === Zaman başlat ===
total_start = time.time()

# === Veri setini oku (multiline + tırnak escape destekli) ===
load_start = time.time()
df = spark.read.csv(
    "file:///home/seed/Desktop/dagitik/cleaned_twitter_data.csv",
    header=True,
    inferSchema=True,
    multiLine=True,
    quote='"',
    escape='"'
)
load_end = time.time()

print(f"✅ Dosyadan okunan toplam satır sayısı: {df.count()} satır ({load_end - load_start:.2f} saniye)")

# === Temiz metin olmayanları çıkar (boşlukları da hesaba kat) ===
df = df.filter((col("cleaned_text").isNotNull()) & (length(trim(col("cleaned_text"))) > 0))
print(f"✅ Geçerli temizlenmiş metin sayısı: {df.count()}")

# === UDF'leri kaydet ===
vader_udf = udf(get_vader_score, FloatType())
label_udf = udf(get_sentiment_label, StringType())

# === VADER hesaplama ===
process_start = time.time()
df = df.withColumn("vader_sentiment", vader_udf(col("cleaned_text")))
df = df.withColumn("vader_category", label_udf(col("vader_sentiment")))
process_end = time.time()
print(f"✅ Duygu hesaplama süresi: {process_end - process_start:.2f} saniye")

# === Sonuçları grupla ===
analysis_start = time.time()
result = df.groupBy("vader_category").count()
result_list = result.collect()
analysis_end = time.time()

# === Sonuç yaz ===
print("\n🎯 Duygu analizi sonuçları:")
for row in result_list:
    print(f"{row['vader_category']}: {row['count']} tweet")

# === Toplam süre ===
total_end = time.time()
print(f"\n⏱️ Toplam süre: {total_end - total_start:.2f} saniye")

# === Kapat ===
spark.stop()
