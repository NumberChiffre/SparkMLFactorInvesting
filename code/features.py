import pandas as pd 
import numpy as np 
import pyspark
import re
from pyspark import SparkContext
from pyspark.rdd import RDD
from pyspark.sql import SparkSession
import environment
from dataset import Dataset

# use the Dataset object to perform feature selection
class FeatureSelection(object):
    def __init__(self, data_obj, features=[], select_features=False, split_ratio=0.6):
        self.data_obj = data_obj
        self.ind = data_obj.ind
        self.df_norm = data_obj.df_norm
        if select_features:
            self.features = features
        else:
            self.features = [c for c in self.df_norm if c not in self.ind + ['y']]
        
        # split train/test data in order to avoid biases after feature selection
        # performing feature selection on all data + CV on test data can yield biases
        unique_timestamp = self.df_norm["timestamp"].unique()
        n = len(unique_timestamp)
        unique_idx = int(n*split_ratio) 
        timesplit = unique_timestamp[unique_idx]
        self.train = self.df_norm[self.df_norm.timestamp < timesplit]
        self.test = self.df_norm[self.df_norm.timestamp >= timesplit]

    # get features that share high correlations with each other
    # these features should be combined together as we can remove highly correlated variables
    def cluster_corr(self, features=[], corr_threshold=0.80):
        clusters, singles = [], []
        if features is None:
            features = self.features
        else:
            features = features
        features_df = self.train[features] # for the sake of feature selection..

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
            """
            elif len(cluster) == 1:
                singles.append(col)
            """
        return clusters

    #TODO: 
    # add extra features: lagged + fusion of existing features
    def filter_features(self, num_features_threshold=20):
        """
        Once highly correlated features are found, we aim to combine them to form new features
        We intend to create the following type of new features:
        - combination of highly correlated features through summation/sub, weighted sums
        - produce lagged features, will be based on sign of autocorrelation
        """

        #TODO: need check correlation between de-mean features, check for multicollinearity
        # keep high correlated features with output y and de-mean in addition to existing features
        num_features_keep = int(num_features_threshold/2)
        y = self.train['y']
        core_features = self.features
        features_df = self.train
        features_corr_to_y = features_df[core_features].corrwith(y).sort_values(ascending=False)
        df_top_corr, df_bot_corr = features_corr_to_y.head(num_features_keep), features_corr_to_y.tail(num_features_keep)
        selected_features = list(df_top_corr.index) + list(df_bot_corr.index)

        # check clusters containing highly correlated features, we will combine them
        features_clusters = list(set([tuple(x) for x in self.cluster_corr(features=selected_features)]))
        features_clusters = [list(x) for x in features_clusters]

        # for each cluster, we will create the sum of those features..to deal with multicollinearity
        for cluster in features_clusters:
            features_num = str(''.join(['_'+re.findall('\d+', s)[0] for s in cluster]))
            features_df['Combo_'+cluster[0][:4]+features_num] = features_df[cluster].sum(axis=1)

        # Combined features that share high corr with each other
        # add de-mean features by grouping timestamp
        features_df[[c+'_demean' for c in selected_features]] = features_df[['id']+selected_features].groupby('id')[selected_features].apply(lambda x:x-x.mean())
        
        # add lag features and diff features, from 1-10
        # add rolling mean/vol, from 1-10
        for lag in range(1,10):
            features_df[[c+'_lag'+str(lag) for c in selected_features]] = features_df[['id']+selected_features].groupby('id')[selected_features].shift(lag).fillna(0)
            features_df[[c+'_lagdiff'+str(lag) for c in selected_features]] = features_df[['id']+selected_features].groupby('id')[selected_features].diff(lag).fillna(0)
            #features_df[[c+'_rollmean'+str(2*lag) for c in selected_features]] = features_df[['id']+selected_features].groupby('id')[selected_features].rolling(2*lag).mean().fillna(0)
            #features_df[[c+'_rollvol'+str(2*lag) for c in selected_features]] = features_df[['id']+selected_features].groupby('id')[selected_features].rolling(2*lag).std().fillna(0)

        self.features = [c for c in features_df if c not in self.ind + ['y'] + core_features]
        self.train = features_df
        return self.train
    
    #TODO:
    # add model feature selection methods: Forward Selection + Backward Selection
    # CV will use the following percentiles to (train, validate) through Ensemble Trees:
    # (0-20th, 20-40th), (40-60th, 60-80th), (60-80th, 80-100th), (0-20th, 60-80th), (40-60th, 80-100th)
    def CV_features(self):
        pass

if __name__ == '__main__':
    features = ['technical_20', 'technical_30']
    data_obj = Dataset()
    #print(data_obj.fullset.describe())
    df_norm = data_obj.preprocess(fill_method='median', scale_method='none')
    #print(df_norm.describe())
    features_obj = FeatureSelection(data_obj)
    features = features_obj.filter_features()
    print(features)    