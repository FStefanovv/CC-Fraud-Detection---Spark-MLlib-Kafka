from pyspark import SparkConf, SparkContext

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType

from pyspark.ml.feature import VectorAssembler, StandardScalerModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.mllib.evaluation import MulticlassMetrics

def quiet_logs(sc):
  logger = sc._jvm.org.apache.log4j
  logger.LogManager.getLogger("org"). setLevel(logger.Level.ERROR)
  logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)


conf = SparkConf() \
    .setAppName("Evaluate model") \
    .setMaster("spark://spark-master:7077")

ctx = SparkContext().getOrCreate(conf)

spark = SparkSession(ctx)

quiet_logs(spark)

test_data = spark.read.csv('hdfs://namenode:9000/fraud_data/test_data.csv', header=True)

for column in test_data.columns:
        test_data = test_data.withColumn(column, col(column).cast(DoubleType()))

test_data = test_data.withColumnRenamed('Class', 'Fraud')
assembler = VectorAssembler(inputCols=["Amount"], outputCol="Amount_Vector")
test_data = assembler.transform(test_data)

amount_scaler_model = StandardScalerModel.load('hdfs://namenode:9000/models/amount_scaler')

test_data = amount_scaler_model.transform(test_data)

test_data = test_data.drop('Amount')
test_data = test_data.drop('Amount_Vector')
test_data = test_data.drop('id')

features = [col for col in test_data.columns if col != 'Fraud']

features_vector_assembler = VectorAssembler(inputCols=features, outputCol='features')

test_data = features_vector_assembler.transform(test_data)

rf_classifier = RandomForestClassificationModel.load('hdfs://namenode:9000/models/rf_model')

preds = rf_classifier.transform(test_data)

predictionAndLabels = preds.select(col("prediction"), col("Fraud")).rdd.map(lambda row: (float(row[0]), float(row[1])))
metrics = MulticlassMetrics(predictionAndLabels)

accuracy = metrics.accuracy
precision = metrics.weightedPrecision
recall = metrics.weightedRecall
f1_score = metrics.weightedFMeasure()

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")