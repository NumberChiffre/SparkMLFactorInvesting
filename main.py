import h5py
import numpy as np
import pandas as pd
import pyspark
from pyspark import SparkContext
from pyspark.rdd import RDD
from pyspark.sql import SparkSession
'''
1. Read the data, check the specific features and load through pyspark
'''
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("PySpark") \
        .config("config", "value") \
        .getOrCreate()
    return spark

def load_data(filename):
    #sc = SparkContext("local", "first app")
    #print(sc.textFile(filename).take(20))
    spark = init_spark()
    spark.conf.set("spark.executor.memory", '32g')
    spark.conf.set('spark.executor.cores', '50')
    spark.conf.set('spark.cores.max', '50')
    spark.conf.set("spark.driver.memory",'32g')
    rdd = spark.read.csv(filename, header=True).limit(20).rdd
    print(rdd.collect())
    '''
    arr = []
    with h5py.File(filename, 'r') as f:
        h5 = f.get('/train')
        items = list(h5.items())
        for i in items16
            arr.append(np.array(h5.get(i[0])))
        rdd = sc.parallelize(arr)
        print(rdd.collect())
    '''
if __name__ == "__main__":
    load_data('data/train.csv')