import pandas as pd 
import numpy as np 
import re
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, ExtraTreesRegressor
import xgboost as xgb
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
        unique_timestamps = len(unique_timestamp)
        unique_idx = int(unique_timestamps*split_ratio) 
        timesplit = unique_timestamp[unique_idx]
        self.train = self.df_norm[self.df_norm.timestamp < timesplit]
        self.test = self.df_norm[self.df_norm.timestamp >= timesplit]
        self.outpath = "output/"

    # get features that share high correlations with each other
    # these features should be combined together as we can remove highly correlated variables
    def cluster_corr(self, features, threshold=0.6):
        clusters = []
        data = self.train[features]
        for col in features:
            cluster_pos, cluster_neg = [], []
            for feature in features:
                coeff = np.corrcoef(data[col].values, data[feature].values)[0,1]
                coeff = np.round(coeff, decimals=2)
                if coeff >= threshold:
                    cluster_pos.append(feature)
                elif coeff <= -threshold:
                    cluster_neg.append(feature)
            if len(cluster_pos) > 1:
                clusters.append(cluster_pos)
            if len(cluster_neg) > 1:
                clusters.append(cluster_neg)
        return clusters


    def generate_sim_features(self, features, threshold=0.6):
        correlation_matrix = self.train[features].corr()  
        cluster = set()
        for i in range(len(features)):  
            correlated_features_pos, correlated_features_neg = set(), set()  
            for j in range(len(features)):
                if correlation_matrix.iloc[i, j] >= threshold:
                    correlated_features_pos.add(correlation_matrix.columns[i])
                elif correlation_matrix.iloc[i, j] <= -threshold:
                    correlated_features_neg.add(correlation_matrix.columns[i])
            if len(correlated_features_pos) > 1:
                cluster.add(correlated_features_pos)
            if len(correlated_features_neg) > 1:
                cluster.add(correlated_features_neg)
        return cluster

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
        return selected_features

    def generate_features(self, selected_features):
        # check clusters containing highly correlated features, we will combine them
        features_df = self.train
        features_clusters = list(set([tuple(x) for x in self.cluster_corr(features=selected_features)]))
        features_clusters = [list(x) for x in features_clusters]

        # for each cluster, we will create the sum of those features..to deal with multicollinearity
        for cluster in features_clusters:
            features_num = str(''.join(['_'+re.findall('\d+', s)[0] for s in cluster]))
            features_df['Combo_'+cluster[0][:4]+features_num] = features_df[cluster].sum(axis=1)

        # Combined features that share high corr with each other
        # add de-mean features by clustering timestamp
        features_df[[c+'_demean' for c in selected_features]] = features_df[['id']+selected_features].groupby('id')[selected_features].apply(lambda x:x-x.mean())
    
        # add lag features and diff features, from 1-10
        # add rolling mean/vol, from 1-10
        for lag in range(1,5):
            features_df[[c+'_lag'+str(lag) for c in selected_features]] = features_df[['id']+selected_features].groupby('id')[selected_features].shift(lag).fillna(0)
            features_df[[c+'_lagdiff'+str(lag) for c in selected_features]] = features_df[['id']+selected_features].groupby('id')[selected_features].diff(lag).fillna(0)
            #features_df[[c+'_rollmean'+str(2*lag) for c in selected_features]] = features_df[['id']+selected_features].groupby('id')[selected_features].rolling(2*lag).mean().fillna(0)
            #features_df[[c+'_rollvol'+str(2*lag) for c in selected_features]] = features_df[['id']+selected_features].groupby('id')[selected_features].rolling(2*lag).std().fillna(0)
            features_df['y_rolling_vol'+'_lag'+str(2*lag)] = features_df[['timestamp','y']].groupby('timestamp').mean().rolling(2*lag).std().fillna(0)  
        self.features = [c for c in features_df if c not in self.ind + ['y'] + self.features]
        self.train = features_df.fillna(0)
        return self.train
    
    # run several Trees models in parallel to get feature importance ranks
    def get_feature_importance(self, split_ratio=0.6):
    
        etr_features = ['technical_30_demean',
        'technical_43_demean',        
        'technical_20_demean',
        'technical_30_lag1',
        'technical_20_lagdiff6',
        'technical_43_lagdiff3',
        'technical_11_lagdiff3',
        'technical_2_lagdiff3',
        'technical_21_lagdiff1',
        'fundamental_60_demean'
        ]
    
        # feature extraction
        unique_timestamp = self.train["timestamp"].unique()
        unique_timestamps = len(unique_timestamp)
        unique_idx = int(unique_timestamps*split_ratio) 
        timesplit = unique_timestamp[unique_idx]

        x_train = self.train.loc[self.train.timestamp < timesplit, self.features].astype('float32')
        y_train = self.train.loc[self.train.timestamp < timesplit, 'y'].astype('float32')
        x_cv = self.train.loc[self.train.timestamp >= timesplit, self.features].astype('float32')
        y_cv = self.train.loc[self.train.timestamp >= timesplit, 'y'].astype('float32')
        
        """
        etr = ExtraTreesRegressor(n_estimators=len(self.features), min_samples_leaf=50,
                max_depth=10, n_jobs=-1, random_state=10000, verbose=0)
        model = etr.fit(x_train, y_train)
        features_rank = pd.DataFrame(model.feature_importances_, index=self.features, columns=['importance']).sort_values('importance', ascending=False)
        top_features = [features_rank.head(10).index]
        print(top_features)
        #print(model.score(x_cv, y_cv))
        #print(model.score(x_cv[features_rank.head(10).index], y_cv))
        """
        train_xgb = xgb.DMatrix(data=x_train[etr_features], label=y_train)
        params = {
            'objective' : 'reg:linear'
            ,'tree_method': 'hist'
            ,      'eta': 0.01
            ,'max_depth': 4
            ,'subsample': 0.90
            ,'colsample_bytree': 0.40
            ,'min_child_weight': 100
            ,     'seed': 10000
            ,   'silent': 1
            ,'base_score' : 0.0
            ,'reg_lambda': 0.5
            ,'reg_alpha': 0.0

        }
        model = xgb.train(params, train_xgb, num_boost_round=15)
        test_xgb = xgb.DMatrix(x_cv[etr_features])
        pred = xgb.predict(test_xgb)
        print(score(y_cv, pred))
        """
        models = [ExtraTreesClassifier(), RandomForestClassifier()]
        for model in models:
            model.fit(x_train, y_train)
            features_rank = pd.DataFrame(model.feature_importances_, index=self.features, columns=['importance']).sort_values('importance', ascending=False)
            print(features_rank)
            model.score(x_cv, y_cv)
            model.score(x_cv[features_rank.iloc[:,0:10]], y_cv)
        """
        print('done')

    def Wrapper_features_selection(self):
        features = self.features
        pmax = 1
        while (len(features) > 0):
            p = []
            X, y = self.train[features], self.train['y']
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            p = pd.Series(model.pvalues.values[1:],index = features)      
            pmax = max(p)
            feature_with_p_max = p.idxmax()
            if(pmax > 0.05):
                features.remove(feature_with_p_max)
            else:
                break
        filtered_features = features
        return filtered_features

    def LassoCV_features_selection(self):
        reg = LassoCV()
        X, y = self.train[features], self.train['y']
        reg.fit(X, y)
        coef = pd.Series(reg.coef_, index = X.columns)
        imp_coef = coef.sort_values()
        matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
        imp_coef.plot(kind = "barh")
        plt.title("Feature importance using Lasso Model")
        plt.savefig(self.outpath+'LassoFeatureImportance.png')
        return imp_coef

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
    filtered_features = features_obj.filter_features()
    new_df = features_obj.generate_features(filtered_features)
    backward_selection_features = features_obj.Wrapper_features_selection()
    print(backward_selection_features)
    features_obj.get_feature_importance()
    lasso_features = features_obj.LassoCV_features_selection()
    print(lasso_features)