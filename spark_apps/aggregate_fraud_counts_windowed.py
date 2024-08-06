from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as pyspark_sum, col
from pyspark.sql.types import TimestampType, LongType, StringType, StructField, StructType


def quiet_logs(sc):
  logger = sc._jvm.org.apache.log4j
  logger.LogManager.getLogger("org"). setLevel(logger.Level.ERROR)
  logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)


conf = SparkConf() \
    .setAppName("Aggregate fraud counts for 1 minute windows") \
    .setMaster("spark://spark-master:7077")

ctx = SparkContext().getOrCreate(conf)

spark = SparkSession(ctx)

quiet_logs(spark)


schema = StructType([
    StructField('window_start', TimestampType(), True),
    StructField('window_end', TimestampType(), True),
    StructField('label', StringType(), False),
    StructField('count', LongType(), False)
])

fraud_counts = spark.read.parquet('hdfs://namenode:9000/real_time_processing/fraud_counts_windowed_sink', schema=schema)

fraud_counts_aggregated = fraud_counts.groupBy('window_start', 'window_end', 'label')\
                                        .agg(pyspark_sum('count').alias('count')).orderBy(col('window_start'), col('label'))

fraud_counts_aggregated.write.parquet('hdfs://namenode:9000/real_time_processing/fraud_counts_windowed_aggregated', mode='overwrite')