# Databricks notebook source
# ALWAYS IMPORT THESE PACKAGES AT THE BEGINNING
from __future__ import division, absolute_import
from pyspark.sql import Row
from pyspark.sql.functions import col,when
from pyspark.ml import regression
from pyspark.ml import feature
from pyspark.ml import Pipeline
from pyspark.sql import functions as fn
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.ml.classification import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark_pipes import pipe
from pyspark.ml.feature import QuantileDiscretizer
import numpy as np

# COMMAND ----------

# import the dataset from spark table
us_visa = spark.sql("SELECT * FROM us_perm_visas_csv")

# COMMAND ----------

df = spark.sql("SELECT * FROM book1_csv")

# COMMAND ----------

# convert the dataset to a dataframe, and select wanted columns 
us_visa_test = us_visa.select("case_status","class_of_admission","country_of_citzenship","employer_state","naics_2007_us_title","pw_job_title_9089","pw_level_9089","pw_soc_title","wage_offer_from_9089","wage_offer_unit_of_pay_9089").toDF("case_status","class_of_admission","country_of_citzenship","employer_state","naics_2007_us_title","pw_job_title_9089","pw_level_9089","pw_soc_title","wage_offer_from_9089","wage_offer_unit_of_pay_9089")

# COMMAND ----------

# Drop na rows
new_df = us_visa_test.dropna(subset=["case_status","class_of_admission","country_of_citzenship","employer_state","naics_2007_us_title","pw_job_title_9089","pw_level_9089","pw_soc_title","wage_offer_from_9089","wage_offer_unit_of_pay_9089"])

# COMMAND ----------

# clean some unwanted data : we want only certified and denied applications, and for only H-1B visa-type
from pyspark.sql.functions import col, asc
whereDF = new_df.where(((col("case_status") == "Denied") | (col("case_status") == "Certified")) & ((col("class_of_admission")=="H-1B")|(col("class_of_admission")=="H-1B1")|(col("class_of_admission")=="H1B")))

# COMMAND ----------

# MAGIC %md After cleaning the dataset, now we have 10381 observations

# COMMAND ----------

whereDF.count()

# COMMAND ----------

# MAGIC %md Next, we only want to analyze wages in a yearly unit of pay. We arrive at 10115 observations.

# COMMAND ----------

whereDF_test = whereDF.select("*").where(fn.col("wage_offer_unit_of_pay_9089")=="yr")

# COMMAND ----------

whereDF_test.count()

# COMMAND ----------

#whereDF_test.groupby('pw_level_9089').agg(fn.count("*")).show()

# COMMAND ----------

# MAGIC %md Convert levels into numerical values

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
whereDF_test= whereDF_test.withColumn('pw_level_9089',
    F.when(whereDF_test['pw_level_9089']=='Level I','1').otherwise(whereDF_test['pw_level_9089']))
whereDF_test= whereDF_test.withColumn('pw_level_9089',
    F.when(whereDF_test['pw_level_9089']=='Level II','2').otherwise(whereDF_test['pw_level_9089']))
whereDF_test= whereDF_test.withColumn('pw_level_9089',
    F.when(whereDF_test['pw_level_9089']=='Level III','3').otherwise(whereDF_test['pw_level_9089']))
whereDF_test= whereDF_test.withColumn('pw_level_9089',
    F.when(whereDF_test['pw_level_9089']=='Level IV','4').otherwise(whereDF_test['pw_level_9089']))
whereDF_test=whereDF_test.withColumn('pw_level_9089', whereDF_test['pw_level_9089'].cast(IntegerType()))

# COMMAND ----------

# MAGIC %md Convert case status to numerical values (dummy variables) 1 for certified, 0 for denied application

# COMMAND ----------

whereDF_test= whereDF_test.withColumn('case_status',
    F.when(whereDF_test['case_status']=='Certified','1').otherwise(whereDF_test['case_status']))
whereDF_test= whereDF_test.withColumn('case_status',
    F.when(whereDF_test['case_status']=='Denied','0').otherwise(whereDF_test['case_status']))
whereDF_test=whereDF_test.withColumn('case_status', whereDF_test['case_status'].cast(IntegerType()))

# COMMAND ----------

# MAGIC %md Cocantenate all important text columns into one column called 'blurb'

# COMMAND ----------

from pyspark.sql.functions import *
whereDF_test = whereDF_test.withColumn('blurb', concat_ws(' ','country_of_citzenship','employer_state','pw_soc_title','naics_2007_us_title'))

# COMMAND ----------

# MAGIC %md Convert wage into integer

# COMMAND ----------

whereDF_test=whereDF_test.withColumn('wage', whereDF_test['wage_offer_from_9089'].cast(IntegerType()))

# COMMAND ----------

# MAGIC %md Create a new dataframe with only wanted columns

# COMMAND ----------

new_DF = whereDF_test.select("case_status","wage","blurb","pw_level_9089")

# COMMAND ----------

# MAGIC %md Create a tokenizer to convert blurb into a vector of vocabulery, and count each word's frequency. This also creates a new column 'words' that split all the words

# COMMAND ----------

from pyspark.ml.feature import RegexTokenizer
tokenizer = RegexTokenizer().setGaps(False)\
  .setPattern("\\p{L}+")\
  .setInputCol("blurb")\
  .setOutputCol("words")

# COMMAND ----------

# import stop words
import requests
stop_words = requests.get('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words').text.split()

# COMMAND ----------

stop_words[0:10]

# COMMAND ----------

# delete stop words from words
from pyspark.ml.feature import StopWordsRemover
SW_Remover = StopWordsRemover()\
  .setStopWords(stop_words)\
  .setCaseSensitive(False)\
  .setInputCol("words")\
  .setOutputCol("filtered")

# COMMAND ----------

# ignore words that its frequency is less than 5 times 
from pyspark.ml.feature import CountVectorizer
Count_Vectorizer = CountVectorizer(minTF=1., minDF=5., vocabSize=2**17)\
  .setInputCol("filtered")\
  .setOutputCol("tf")

# COMMAND ----------

# create tf-idf
from pyspark.ml.feature import IDF
idf = IDF().\
    setInputCol('tf').\
    setOutputCol('tfidf')

# COMMAND ----------

discretizer = feature.QuantileDiscretizer(numBuckets=10, inputCol="wage", outputCol="quant_wage")

# COMMAND ----------

#display(QuantileDiscretizer(numBuckets=10, inputCol="wage", outputCol="quant_wage"))

plt.figure()
feature.QuantileDiscretizer(numBuckets=10, inputCol="wage_offer_from_9089", outputCol="quant_wage").fit(whereDF_test).transform(whereDF_test).toPandas().iloc[:, -1].hist()
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
plt.xlabel('Wage quantiles')
plt.xlabel('Wages')
plt.title('Offered Wage by Quantiles')
display()

# COMMAND ----------

# now we want to create some models for predictions
# first, we want to divide the dataset into 3 sets 60% for training, 30% for validation and 10% for testing
training_df, validation_df, testing_df = new_DF.randomSplit([0.6, 0.3, 0.1], seed=0)

# COMMAND ----------

lambda_par = 0.02
alpha_par = 0.3
logr_1 = LogisticRegression(labelCol='case_status', featuresCol='tfidf')\
    .setRegParam(lambda_par)\
    .setMaxIter(100)\
    .setElasticNetParam(alpha_par)

# COMMAND ----------

VA_2 = feature.VectorAssembler(inputCols=['quant_wage'], outputCol='features')

# COMMAND ----------

VA_3 = feature.VectorAssembler(inputCols=['quant_wage', 'pw_level_9089'], outputCol='features_2')

# COMMAND ----------

VA_4 = feature.VectorAssembler(inputCols=['tfidf','quant_wage', 'pw_level_9089'], outputCol='features_3')

# COMMAND ----------

logr_2 = LogisticRegression(featuresCol='features', labelCol='case_status', maxIter=100, regParam=0.02, elasticNetParam=0.3)


# COMMAND ----------

logr_3 = LogisticRegression(featuresCol='features_2', labelCol='case_status', maxIter=100, regParam=0.02, elasticNetParam=0.3)


# COMMAND ----------

logr_4 = LogisticRegression(featuresCol='features_3', labelCol='case_status', maxIter=100, regParam=0.02, elasticNetParam=0.3)

# COMMAND ----------

logr_estimator_1 = Pipeline(stages=[tokenizer, SW_Remover, Count_Vectorizer, idf, logr_1])

# COMMAND ----------

logr_estimator_2 = Pipeline(stages=[discretizer, VA_2, logr_2])

# COMMAND ----------

logr_estimator_3 = Pipeline(stages=[discretizer, VA_3, logr_3])

# COMMAND ----------

logr_estimator_4 = Pipeline(stages=[tokenizer, SW_Remover, Count_Vectorizer, idf, discretizer, VA_4, logr_4])

# COMMAND ----------

logr_estimator_1.getStages()

# COMMAND ----------

logr_estimator_2.getStages()

# COMMAND ----------

logr_estimator_3.getStages()

# COMMAND ----------

logr_estimator_4.getStages()

# COMMAND ----------

grid_1 = ParamGridBuilder().addGrid(logr_1.regParam, [0., 0.01, 0.02, 0.03]).
    addGrid(logr_1.elasticNetParam, [0., 0.1, 0.2, 0.3, 0.4]).\
    build()

# COMMAND ----------

grid_2 = ParamGridBuilder().addGrid(logr_2.regParam, [0., 0.01, 0.02, 0.03]).\
    addGrid(logr_2.elasticNetParam, [0., 0.1, 0.2, 0.3, 0.4]).\
    build()

# COMMAND ----------

grid_3 = ParamGridBuilder().addGrid(logr_3.regParam, [0., 0.01, 0.02, 0.03]).\
    addGrid(logr_3.elasticNetParam, [0., 0.1, 0.2, 0.3, 0.4]).\
    build()

# COMMAND ----------

grid_4 = ParamGridBuilder(). addGrid(logr_4.regParam, [0., 0.01, 0.02, 0.03]).\
    addGrid(logr_4.elasticNetParam, [0., 0.1, 0.2, 0.3, 0.4]).\
    build()

# COMMAND ----------

models_1 = []
for k in range(len(grid_1)):
    print("Fitting model {}".format(k+1))
    model = logr_estimator_1.fit(training_df, grid_1[k])
    models_1.append(model)

# COMMAND ----------

models_2 = []
for k in range(len(grid_2)):
    print("Fitting model {}".format(k+1))
    model = logr_estimator_2.fit(training_df, grid_2[k])
    models_2.append(model)

# COMMAND ----------

models_3 = []
for k in range(len(grid_3)):
    print("Fitting model {}".format(k+1))
    model = logr_estimator_3.fit(training_df, grid_3[k])
    models_3.append(model)

# COMMAND ----------

models_4 = []
for k in range(len(grid_4)):
    print("Fitting model {}".format(k+1))
    model = logr_estimator_4.fit(training_df, grid_4[k])
    models_4.append(model)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(labelCol="case_status")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC $$
# MAGIC precision = \frac{TP}{TP+FP}, \quad recall = \frac{TP}{TP+FN} = TPR
# MAGIC $$

# COMMAND ----------

# estimate the accuracy of each of them:
auc_1 = [evaluator.evaluate(m.transform(validation_df), {evaluator.metricName: "areaUnderPR"})\
       for m in models_1]

# COMMAND ----------

# estimate the accuracy of each of them:
auc_2 = [evaluator.evaluate(m.transform(validation_df), {evaluator.metricName: "areaUnderPR"})\
       for m in models_2]

# COMMAND ----------

# estimate the accuracy of each of them:
auc_3 = [evaluator.evaluate(m.transform(validation_df), {evaluator.metricName: "areaUnderPR"})\
       for m in models_3]

# COMMAND ----------

# estimate the accuracy of each of them:
auc_4 = [evaluator.evaluate(m.transform(validation_df), {evaluator.metricName: "areaUnderPR"})\
       for m in models_4]

# COMMAND ----------

auc_4

# COMMAND ----------

# MAGIC %md Find the highest AUPR of the four best models

# COMMAND ----------

print(np.max(auc_1), np.max(auc_2), np.max(auc_3), np.max(auc_4))

# COMMAND ----------

print(np.argmax((np.max(auc_1), np.max(auc_2), np.max(auc_3), np.max(auc_4))))

# COMMAND ----------



# COMMAND ----------

BestModel_1 = np.argmax(auc_1)

best_model_1 = models_1[BestModel_1]

# COMMAND ----------

BestModel_2 = np.argmax(auc_2)

best_model_2 = models_2[BestModel_2]

# COMMAND ----------

BestModel_3 = np.argmax(auc_3)

best_model_3 = models_3[BestModel_3]

# COMMAND ----------

BestModel_4 = np.argmax(auc_4)

best_model_4 = models_4[BestModel_4]

# COMMAND ----------

d = {'wage': [90000], 'blurb': ["CHILE NY Professor, Higher Education"], 'pw_level_9089': [3]}
df = pd.DataFrame(data=d)

# COMMAND ----------

df = testing_df.show

# COMMAND ----------

best_model_4.transform(df)

# COMMAND ----------

testing_df.show(5)

# COMMAND ----------

display(best_model_4.transform(testing_df))

# COMMAND ----------

best_model_4.stages

# COMMAND ----------

# MAGIC %md Logistic Regression model coefficients

# COMMAND ----------

len(best_model_4.stages[-1].coefficients)

# COMMAND ----------

best_model_4.stages[-1].coefficients

# COMMAND ----------

best_model_4.stages

# COMMAND ----------

best_model_4.stages[5].getInputCols()

# COMMAND ----------

best_model_4.stages[3]

# COMMAND ----------

weights = best_model_4.stages[-1].coefficients.toArray()

# COMMAND ----------


#best_model_4.stages[0].vocabulary

# COMMAND ----------

# to weight words
import pandas as pd
vocabulary = IDF_vector_pipeline.stages[0].stages[-1].vocabulary
weights = logReg1_pipeline.stages[-1].coefficients.toArray()

# COMMAND ----------

coefficient_DF = pd.DataFrame({'word': vocabulary, 'weight': weights})
coefficient_DF = pd.DataFrame({'word': best_model_4.stages[0].vocabulary, 'weight': weights})

# COMMAND ----------

coefficient_DF.sort_values('weight', ascending=True).head(15)

# COMMAND ----------

coefficient_DF.sort_values('weight', ascending=False).head(100)

# COMMAND ----------

coefficient_DF.toPandas().weight.hist()

# COMMAND ----------

# MAGIC %md PLOTS

# COMMAND ----------

bestmodel = best_model_4.transform(df)

# COMMAND ----------

display(bestmodel)

# COMMAND ----------

best_model_4_df[9]

# COMMAND ----------


