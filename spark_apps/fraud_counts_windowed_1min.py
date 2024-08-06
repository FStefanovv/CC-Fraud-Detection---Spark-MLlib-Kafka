from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, TimestampType

def quiet_logs(sc):
  logger = sc._jvm.org.apache.log4j
  logger.LogManager.getLogger("org"). setLevel(logger.Level.ERROR)
  logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)

spark = SparkSession.builder.appName('Number of frauds windowed per minute').getOrCreate()

quiet_logs(spark)

schema = StructType([
    StructField('id', IntegerType(), False),
    StructField('prediction', DoubleType(), False),
    StructField('timestamp', TimestampType(), False)
])

classified_transactions_stream = spark.readStream\
                                    .format('kafka')\
                                    .option("kafka.bootstrap.servers", "kafka1:19092,kafka2:19093") \
                                    .option("subscribe", "classified_transactions") \
                                    .load()


transactions_values = classified_transactions_stream.selectExpr('CAST(value AS STRING)')
transactions = transactions_values.withColumn('transaction', from_json(col('value'), schema)).select('transaction.*')

transactions = transactions.withColumn('label', when(col('prediction')==1.0, 'fraud').otherwise('not_fraud'))

counts_windowed = transactions.withWatermark('timestamp', '5 minutes')\
                                .groupBy('label', window('timestamp', '1 minute'))\
                                .count()

counts_windowed = counts_windowed.select(
    col("window.start").alias("window_start"),
    col("window.end").alias("window_end"),
    col("label"),
    col("count")
)

console_query = counts_windowed.writeStream\
    .outputMode("update")\
    .format("console")\
    .start()

hdfs_query = counts_windowed.coalesce(1).writeStream \
    .format("parquet")\
    .option("path", "hdfs://namenode:9000/real_time_processing/fraud_counts_windowed_sink")\
    .option("checkpointLocation", "hdfs://namenode:9000/real_time_processing/checkpoint_fraud_counts_windowed_sink") \
    .outputMode("append")\
    .start()

console_query.awaitTermination()
hdfs_query.awaitTermination()