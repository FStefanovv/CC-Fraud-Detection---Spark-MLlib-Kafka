from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, TimestampType

from pyspark.ml.feature import VectorAssembler, StandardScalerModel
from pyspark.ml.classification import RandomForestClassificationModel

def quiet_logs(sc):
  logger = sc._jvm.org.apache.log4j
  logger.LogManager.getLogger("org"). setLevel(logger.Level.ERROR)
  logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)

spark = SparkSession.builder.appName('Classify incoming transactions').getOrCreate()

quiet_logs(spark)

schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("V1", DoubleType(), True),
    StructField("V2", DoubleType(), True),
    StructField("V3", DoubleType(), True),
    StructField("V4", DoubleType(), True),
    StructField("V5", DoubleType(), True),
    StructField("V6", DoubleType(), True),
    StructField("V7", DoubleType(), True),
    StructField("V8", DoubleType(), True),
    StructField("V9", DoubleType(), True),
    StructField("V10", DoubleType(), True),
    StructField("V11", DoubleType(), True),
    StructField("V12", DoubleType(), True),
    StructField("V13", DoubleType(), True),
    StructField("V14", DoubleType(), True),
    StructField("V15", DoubleType(), True),
    StructField("V16", DoubleType(), True),
    StructField("V17", DoubleType(), True),
    StructField("V18", DoubleType(), True),
    StructField("V19", DoubleType(), True),
    StructField("V20", DoubleType(), True),
    StructField("V21", DoubleType(), True),
    StructField("V22", DoubleType(), True),
    StructField("V23", DoubleType(), True),
    StructField("V24", DoubleType(), True),
    StructField("V25", DoubleType(), True),
    StructField("V26", DoubleType(), True),
    StructField("V27", DoubleType(), True),
    StructField("V28", DoubleType(), True),
    StructField("Amount", DoubleType(), True),
    StructField("Class", DoubleType(), True),
    StructField("timestamp", TimestampType(), True)
])

amount_scaler_model = StandardScalerModel.load('hdfs://namenode:9000/models/amount_scaler')
rf_classifier = RandomForestClassificationModel.load('hdfs://namenode:9000/models/rf_model')

transactions_stream = spark \
                .readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", "kafka1:19092,kafka2:19093") \
                .option("subscribe", "transactions") \
                .load()

transactions_stream_values = transactions_stream.selectExpr('CAST(value AS STRING)')
transactions = transactions_stream_values.withColumn('transaction', from_json(col('value'), schema)).select('transaction.*')

transactions = transactions.withColumnRenamed('Class', 'Fraud')

assembler = VectorAssembler(inputCols=["Amount"], outputCol="Amount_Vector")
transactions = assembler.transform(transactions)

transactions = amount_scaler_model.transform(transactions)

features = [col for col in transactions.columns if col not in ['Fraud', 'id', 'timestamp', 'Amount', 'Amount_Vector']]
assembler = VectorAssembler(inputCols=features, outputCol="features")
transactions = assembler.transform(transactions)

predictions = rf_classifier.transform(transactions)

result = predictions.select('id', 'timestamp', 'prediction')

console_query = result.writeStream \
    .format("console") \
    .outputMode("append") \
    .start()

hdfs_query = result.writeStream \
    .format("csv")\
    .option("path", "hdfs://namenode:9000/real_time_processing/classification_results")\
    .option("checkpointLocation", "hdfs://namenode:9000/real_time_processing/spark_checkpoints/classification_hdfs")\
    .outputMode("append")\
    .start()

kafka_query = result.selectExpr("CAST(id AS STRING) AS key", "to_json(struct(*)) AS value") \
    .writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka1:19092,kafka2:19093") \
    .option("topic", "classified_transactions") \
    .option("checkpointLocation", "hdfs://namenode:9000/real_time_processing/spark_checkpoints/classification_kafka") \
    .outputMode("append") \
    .start()

    
console_query.awaitTermination()
hdfs_query.awaitTermination()
kafka_query.awaitTermination()