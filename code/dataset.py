import pandas as pd 
import numpy as np 
import pyspark
from pyspark import SparkContext
from pyspark.rdd import RDD
from pyspark.sql import SparkSession
import environment, sys

def init_spark():
    spark = SparkSession \
        .builder \
        .master("local[*]") \
        .appName("PySpark") \
        .config("config", "value") \
        .getOrCreate()
    return spark

class Dataset(object):
    def __init__(self, features=[], select_features=False, filepath="data/train.h5"):
        self.ind = ['id', 'timestamp']
        self.features = features
        self.select_features = select_features
        self.filepath = filepath
        with pd.HDFStore(self.filepath, "r") as hfdata:
            self.fullset = hfdata.get("train")

    def load_data(self):
        with pd.HDFStore(self.filepath, "r") as hfdata:
            df = hfdata.get("train")
        spark = init_spark()
        if self.select_features:
            spark_df = spark.createDataFrame(df).select(self.features+self.ind)
        else:
            spark_df = spark.createDataFrame(df)
        spark_df = spark_df.select(*(col(c).cast("float").alias(c) for c in spark_df.columns if c not in ['id', 'timestamp']))
        self.fullset = spark_df

    # fill NaN with median of each column by default
    # best option to deal with this dataset due to a large amount of outliers
    def preprocess(self, fill_method='median', scale_method='normalize'):
        #self.load_data()
        df = self.fullset
        features = [c for c in df.columns if c not in self.ind and c not in ['y']]

        if fill_method == 'median':
            df_prep = df.fillna(df[features].dropna().median())
        elif fill_method == 'mean':
            df_prep = df.fillna(df[features].dropna().mean())
        elif fill_method == 'remove':
            df_prep = df.fillna(0)
        elif fill_method == 'none':
            df_prep = df
        else:
            raise Exception('Dataset preprocess fill method does not exist!')
            sys.exit(1)

        df_norm = df_prep[features+['y']]
        #df_norm = spark.createDataFrame(df_norm).rdd
        # scale, by default using sigmoid on features
        if scale_method != 'none':
            if scale_method == 'sigmoid':
                df_norm = df_norm.apply(lambda x:1/(1+np.exp(-x)))
                #df_norm = df_norm.map(lambda x:1/(1+np.exp(-x)))
            elif scale_method == 'normalize':
                df_norm = df_norm.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))
                #df_norm = df_norm.map(lambda x:(x-x.mean())/(x.max()-x.min()))
            else:
                raise Exception('Dataset preprocess fill method does not exist!')
                sys.exit(1)

        df_norm[self.ind] = df_prep[self.ind]
        df_norm = df_norm[self.ind + features + ['y']]
        self.features = features
        self.df_norm = df_norm
        return self.df_norm
        """
        #df_rdd = spark.createDataFrame(df_prep).rdd
        #df_rdd = df_rdd.map(lambda x:)
        # standardize each feature by (x[i] - x.min())/(x.max() - x.min())
        scaled_df = pd.DataFrame(df_prep.timestamp)
        for col, old_values in df_prep.iteritems():
            if col not in self.ind:
                scaled_df[str(col)] = self.scale_feature(old_values)
        return scaled_df
        """
    
    def signedLog(x):
        return np.sign(x) * np.log(np.abs(x))

    def signedExp(x):
        return np.sign(x) * np.exp(np.abs(x))


    def scale_feature(self, values):
        new_values = []
        for value in values:
            new_value = (value - values.min())/(values.max()-values.min())
            new_values.append(new_value)
        return new_values

def create():
    return Dataset()
    
if __name__ == '__main__':
    features = ['technical_20', 'technical_30']
    data_obj = Dataset(features)
    print(data_obj.fullset.describe())
    df_norm = data_obj.preprocess()
    print(df_norm.describe())
    clusters, singles = data_obj.cluster_corr()
    print(clusters)
    print(singles)
    