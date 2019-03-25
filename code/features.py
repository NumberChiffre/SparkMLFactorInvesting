import pandas as pd 
import numpy as np 
import pyspark
from pyspark import SparkContext
from pyspark.rdd import RDD
from pyspark.sql import SparkSession
import environment
from dataset import Dataset

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
        features_df = self.df_norm[features]
        for col in features:
            cluster = []
            for feature in features:
                coeff = np.corrcoef(features_df[col].values, features_df[feature].values)[0,1]  
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
    def filter_features(self, num_features_threshold=14):
        """
        Once highly correlated features are found, we aim to combine them to form new features
        We intend to create the following type of new features:
        - combination of highly correlated features through summation/sub, weighted sums
        - produce lagged features, will be based on sign of autocorrelation
        """

        # keep high correlated features with output y and de-mean in addition to existing features
        num_features_keep = int(num_features_threshold/2)
        y = self.df_norm['y']
        features_df = self.df_norm
        features_corr_to_y = features_df[self.features].corrwith(y).sort_values(ascending=False)
        df_top_corr, df_bot_corr = features_corr_to_y.head(num_features_keep), features_corr_to_y.tail(num_features_keep)
        features_corr_df = features_df[list(df_top_corr.index) + list(df_bot_corr.index) + ['timestamp']]
        features_demean = features_corr_df.groupby('timestamp').apply(lambda x:x-x.mean())
        features_demean.columns = [col_name + '_demean' if col_name not in self.ind else col_name for col_name in features_demean.columns]
        df = pd.concat([features_demean.drop(['timestamp'], axis=1), self.df_norm], axis=1)
        return df
    
    def generate_features(self):
        pass

if __name__ == '__main__':
    features = ['technical_20', 'technical_30']
    data_obj = Dataset()
    #print(data_obj.fullset.describe())
    df_norm = data_obj.preprocess(fill_method='remove', scale_method='none')
    #print(df_norm.describe())
    features_obj = FeatureSelection(data_obj)
    print(features_obj.filter_features())
    #clusters, singles = features_obj.cluster_corr()
    #print(list(set(clusters)))
    