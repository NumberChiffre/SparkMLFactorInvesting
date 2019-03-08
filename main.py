import h5py
import numpy as np
import pandas as pd
import pyspark
from pyspark import SparkContext
from pyspark.rdd import RDD
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
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

    # load df, choose a few features to start with without any null values
    # cast each column from string to float
    df = spark.read.csv(filename, header=True).select(['id', 'timestamp', 'fundamental_2', 'fundamental_7', 'y'])
    df = df.select(*(col(c).cast("float").alias(c) for c in df.columns)).na.fill(0)
    num_id, num_timestamp = df.select("id").distinct().count(), df.select("timestamp").distinct().count()
    market_df = df.select(["timestamp", 'fundamental_2', 'fundamental_7', "y"]).groupBy("timestamp").sort(df.timestamp)
    print(market_df.show())

    # set up features as one vector and output variable combined as labeledpoint
    assembler = VectorAssembler(inputCols=['fundamental_2', 'fundamental_7'], outputCol="features")
    transformed = assembler.transform(df).select(col("y").alias("label"), col("features"))
    labeledRDD = transformed.rdd.map(lambda row:LabeledPoint(row.label, as_old(row.features)))
  
    # train the linear model with stochatic gradient descent
    linear_model = LinearRegressionWithSGD.train(labeledRDD, iterations=100, step=0.0000001)
    linear_preds = labeledRDD.map(lambda p: (p.label, linear_model.predict(p.features)))
    linear_MSE = linear_preds.map(lambda x:(x[0] - x[1])**2).reduce(lambda x,y:x + y) / linear_preds.count()
    print("Linear Regression Mean Squared Error = " + str(linear_MSE))

    # train the ridge model with stochastic gradient descent
    ridge_model = RidgeRegressionWithSGD.train(labeledRDD, iterations=100, step=0.0000001)
    ridge_preds = labeledRDD.map(lambda p: (p.label, ridge_model.predict(p.features)))
    ridge_MSE = ridge_preds.map(lambda x:(x[0] - x[1])**2).reduce(lambda x,y:x + y) / ridge_preds.count()
    print("Ridge Regression Mean Squared Error = " + str(ridge_MSE))
    # split: train = df.limit(timestamp_used_for_training), test = df.subtract(train)


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
