import h5py
import numpy as np
import pandas as pd
import pyspark
from pyspark import SparkContext
from pyspark.rdd import RDD
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel, LinearRegressionSummary, LinearRegressionTrainingSummary
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, RidgeRegressionWithSGD
from pyspark.mllib import linalg as mllib_linalg
from pyspark.ml import linalg as ml_linalg

'''
March 7th:
1. Read the data, check the specific features and load through pyspark -- [Done with 2 features]
2. Start with basic models: linear regression + ridge -- [Done with 2 features]

TODO:
1. Illustrate the features with the dependent variable, check for id/timestamp
2. Join rows by timestamp, so that we may actually start constructing a dataset consisting of time series --> break into (train, test)
3. Normalize features + calculate correlation between different combination of features and dependent variable
'''

def init_spark():
    spark = SparkSession \
        .builder \
        .master("local[*]") \
        .appName("PySpark") \
        .config("config", "value") \
        .getOrCreate()
    return spark


# convert deprecated ml.vectors to mllib.vectors
def as_old(v):
    if isinstance(v, ml_linalg.SparseVector):
        return mllib_linalg.SparseVector(v.size, v.indices, v.values)
    if isinstance(v, ml_linalg.DenseVector):
        return mllib_linalg.DenseVector(v.values)
    raise ValueError("Unsupported type {0}".format(type(v)))


def load_data(filename):
    spark = init_spark()

    # probably not necessary with our dataset..
    spark.conf.set("spark.executor.memory", '32g')
    spark.conf.set('spark.executor.cores', '50')
    spark.conf.set('spark.cores.max', '50')
    spark.conf.set("spark.driver.memory",'32g')
    spark.sparkContext.setLogLevel("ERROR")

    # load df, choose a few features to start with without any null values
    # cast each column from string to float
    # need a specific set of methods to preprocess/fill/generate new features
    # need to fill null values with avg/median, test difference in between
    ind_vars = ['technical_20', 'fundamental_53', 'technical_30', 'technical_27', 'derived_0',\
        'fundamental_42', 'fundamental_48', 'technical_21', 'technical_24', 'fundamental_11',\
        'fundamental_44', 'technical_19', 'technical_13', 'technical_17', 'fundamental_9']
    total_features = list(['id', 'timestamp', 'y'] + ind_vars)
    df = spark.read.csv(filename, header=True).select(total_features)
    df = df.select(*(col(c).cast("float").alias(c) for c in df.columns)).na.fill(0)

    # separate training/testing based on 80/20
    timestamps = df.select("timestamp").distinct().sort(df.timestamp).rdd.flatMap(lambda x:x).collect()
    train_len = int(len(timestamps)*0.8)
    train_df, test_df = df.filter(df.timestamp.isin(list(timestamps)[:train_len])), df.filter(df.timestamp.isin(list(timestamps)[train_len:]))

    # set up ML pipeline
    assembler = VectorAssembler(inputCols=ind_vars, outputCol='features')
    lr = LinearRegression(featuresCol='features', labelCol='y', maxIter=10, regParam=0.00001)
    pipeline = Pipeline(stages=[assembler, lr])
    train_transformed = assembler.transform(train_df).select(['features','y'])
    test_transformed = assembler.transform(test_df).select(['features','y'])

    # train/testing using linear model
    lr_model = lr.fit(train_transformed)
    trainingSummary = lr_model.summary
    print("RMSE: ", trainingSummary.rootMeanSquaredError)
    print("R^2: ", trainingSummary.r2)
    lr_predictions = lr_model.transform(test_transformed)
    print(lr_predictions.show())
    lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="y", metricName="rmse")
    rmse = lr_evaluator.evaluate(lr_predictions)
    print("RMSE: ", rmse)
    r2 = lr_evaluator.evaluate(lr_predictions, {lr_evaluator.metricName: "r2"})
    print("r2: ", r2)

    # grid search not working...
    grid_lr = ParamGridBuilder().addGrid(lr.regParam, [0.01, 1.0]).build()
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid_lr, evaluator=lr_evaluator)
    cv_model = cv.fit(train_df)
    print(cv_model.bestModel.extractParamMap())
    '''
    train_transformed = assembler.transform(train_df).select(col("y"), col("features"))
    test_transformed = assembler.transform(test_df).select(col("y"), col("features"))
    train_labeledRDD = train_transformed.rdd.map(lambda row:LabeledPoint(row.label, as_old(row.features)))
    test_labeledRDD = test_transformed.rdd.map(lambda row:LabeledPoint(row.label, as_old(row.features)))
  
    # train the linear model with stochatic gradient descent
    linear_model = LinearRegressionWithSGD.train(train_labeledRDD, iterations=100, step=0.0000001)
    linear_preds = train_labeledRDD.map(lambda p: (p.y, linear_model.predict(p.features)))
    linear_MSE = linear_preds.map(lambda x:(x[0] - x[1])**2).reduce(lambda x,y:x + y) / linear_preds.count()
    print("Linear Regression Mean Squared Error = " + str(linear_MSE))
    
    # train the ridge model with stochastic gradient descent
    ridge_model = RidgeRegressionWithSGD.train(labeledRDD, iterations=100, step=0.0000001)
    ridge_preds = labeledRDD.map(lambda p: (p.y, ridge_model.predict(p.features)))
    ridge_MSE = ridge_preds.map(lambda x:(x[0] - x[1])**2).reduce(lambda x,y:x + y) / ridge_preds.count()
    print("Ridge Regression Mean Squared Error = " + str(ridge_MSE))
    # split: train = df.limit(timestamp_used_for_training), test = df.subtract(train)
    '''

if __name__ == "__main__":
    '''
    arr = []
    with pd.HDFStore('data/train.h5', 'r') as f:
        h5 = f.get('train')
        df = init_spark().createDataFrame(h5).limit(3)
        print(df.select(["fundamental_1","y"]).show())
        items = list(h5.items())
        for i in items:
            arr.append(np.array(h5.get(i[0])))
        rdd = sc.parallelize(arr)
        print(rdd.collect())
    '''
    load_data('data/train.csv')
