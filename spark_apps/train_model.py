from pyspark import SparkConf, SparkContext

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier



def quiet_logs(sc):
  logger = sc._jvm.org.apache.log4j
  logger.LogManager.getLogger("org"). setLevel(logger.Level.ERROR)
  logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)


conf = SparkConf() \
    .setAppName("Train model") \
    .setMaster("spark://spark-master:7077")

ctx = SparkContext().getOrCreate(conf)

spark = SparkSession(ctx)

quiet_logs(spark)

train_data = spark.read.csv('hdfs://namenode:9000/fraud_data/train_data.csv', header=True)

for column in train_data.columns:
        train_data = train_data.withColumn(column, col(column).cast(DoubleType()))

train_data = train_data.withColumnRenamed('Class', 'Fraud')

assembler = VectorAssembler(inputCols=["Amount"], outputCol="Amount_Vector")
train_data = assembler.transform(train_data)

amount_scaler = StandardScaler(inputCol='Amount_Vector', outputCol='Amount_Scaled', withMean=True, withStd=True)
amount_scaler_model = amount_scaler.fit(train_data)

amount_scaler_model.write().overwrite().save('hdfs://namenode:9000/models/amount_scaler')

train_data = amount_scaler_model.transform(train_data)

train_data = train_data.drop('Amount')
train_data = train_data.drop('Amount_Vector')
train_data = train_data.drop('id')

features = [col for col in train_data.columns if col != 'Fraud']

features_vector_assembler = VectorAssembler(inputCols=features, outputCol='features')

train_data = features_vector_assembler.transform(train_data)

rf_classifier = RandomForestClassifier(labelCol="Fraud", featuresCol="features", numTrees=200)

rf_model = rf_classifier.fit(train_data)

rf_model.write().overwrite().save('hdfs://namenode:9000/models/rf_model')