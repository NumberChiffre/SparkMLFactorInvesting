import pandas as pd 
import numpy as np 
import pyspark
from pyspark import SparkContext
from pyspark.rdd import RDD
from pyspark.sql import SparkSession
import environment, dataset

class FeatureSelection(object):
    def __init__(self, data_obj, features=[], select_features=False):
        self.data_obj = data_obj
        self.ind = data_obj.ind
        self.df_norm = data_obj.df_norm
        if select_features:
            self.features = features
        else:
            self.features = [c for c in self.df_norm if c not in self.ind + ['y']]
   
    # get features that share high correlations with each other
    # these features should be combined together as we can remove highly correlated variables
    def cluster_corr(self, corr_threshold=0.80):
        clusters, singles = [], []
        features = self.features
        for col in features:
            cluster = []
            for feature in features:
                coeff = np.corrcoef(self.df_norm[col].values, self.df_norm[feature].values)[0,1]  
                if coeff >= 0 and coeff >= corr_threshold or coeff <= 0 and coeff <= -corr_threshold:
                    cluster.append(feature)
            """
            for member in cluster:
                while member in features:
                    features.remove(member)
            """
            if len(cluster) > 1:
                clusters.append(cluster)
            elif len(cluster) == 1:
                singles.append(col)
        return clusters, singles

    # eventually we'll have to create extra features: lagged + fusion of multiple ones
    def generate_features(self):
        """
        Once highly correlated features are found, we aim to combine them to form new features
        We intend to create the following type of new features:
        - combination of highly correlated features through summation/sub, weighted sums
        - produce lagged features, will be based on sign of autocorrelation
        """
        pass

if __name__ == '__main__':
    features = ['technical_20', 'technical_30']
    data_obj = Dataset()
    print(data_obj.fullset.describe())
    df_norm = data_obj.preprocess()
    print(df_norm.describe())
    features_obj = FeatureSelection(data_obj)
    clusters, singles = features_obj.cluster_corr()
    print(clusters)
    print(singles)
    