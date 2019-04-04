import pandas as pd 
import numpy as np 
import re
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LassoCV, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, ExtraTreesRegressor

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import xgboost as xgb
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
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
    def cluster_corr(self, features, threshold=0.7):
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


    def generate_sim_features(self, features, threshold=0.8):
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


    def filter_features(self, num_features_threshold=10):
        """
        Once highly correlated features are found, we aim to combine them to form new features
        We intend to create the following type of new features:
        - combination of highly correlated features through summation/sub, weighted sums
        - produce lagged features, will be based on sign of autocorrelation
        """
        core_features = self.features
        features_df = self.train[core_features]
        """
        null_counts = features_df.isnull().sum()/len(features_df)
        null_counts = null_counts.sort_values()
        null_counts = pd.DataFrame({'Features':null_counts.index, '% Missing NaN':null_counts.values})
        core_features = list(null_counts[null_counts['% Missing NaN'] < 0.3]['Features'])
        std_df = self.train[core_features+['timestamp','id','y']].groupby('timestamp')
        features_df = std_df.mean()/std_df.std()
        """
        y = self.train['y']
        #y = features_df['y']
        features_corr_to_y = features_df[core_features].corrwith(y).abs().sort_values(ascending=False)
        print(features_corr_to_y)
        selected_features = list(features_corr_to_y.head(num_features_threshold).index)
        self.features = selected_features
        return selected_features

    # generate single feature linear models to assess their reward score
    def generate_single_reward(self):
        train = self.train
        top_features = self.features
        track_score = {}
        for feature in top_features:
            model = Ridge()
            model.fit(np.array(train[feature].values).reshape(-1,1), train.y.values)
            rewards = {}    
            env = environment.make(df)
            observation = env.reset()

            while True:
                test_x = np.array(observation.features[feature].values).reshape(-1,1)
                observation.target.y = model.predict(test_x)
                target = observation.target
                timestamp = observation.features["timestamp"][0]
                if timestamp % 100 == 0:
                    print("Timestamp #{}".format(timestamp))

                observation, reward, done, info = env.step(target)
                if done:
                    track_score[feature] = info['public_score']
                    print(feature, track_score[feature])
                    break
        rewards_df = pd.DataFrame.from_dict(track_score, orient='index', columns=['features','reward']).sort_values(by='reward', ascending=False)
        rewards_df.to_csv(self.outpath+'_Ridge_Reward.csv')

    # rank corr with y and single feature reward score
    # take top features with the least delta in rank
    def select_top_features(self, num_features_threshold=14, abs_rank_delta=25):
        y = self.train['y']
        train = self.train[self.features]
        corr_with_y = train.corrwith(y).abs().sort_values(ascending=False).to_frame(name='corr')
        corr_with_y.index.name = 'features'
        corr_with_y.reset_index(level=0, inplace=True)
        corr_with_y['rank_corr'] = corr_with_y['corr'].rank()
        reward_df = pd.read_csv(self.outpath+'_Ridge_Reward.csv')
        reward_df['rank_reward'] = reward_df['reward'].rank()
        rank_df = pd.merge(corr_with_y, reward_df, on='features')
        top_rank_df = rank_df[(abs(rank_df['rank_corr']-rank_df['rank_reward']) < abs_rank_delta) & (rank_df['rank_reward'] > 0)]
        top_rank_df.to_csv(self.outpath+'_Top_Ridge_Corr_Reward_Rank.csv')
        print(top_rank_df)
        return list(top_rank_df.head(num_features_threshold)['features'])


    # feature engineering on filtered features based on correlation against the output variable
    def generate_features(self, features=[]):
        if not features:
            selected_features = [c for c in self.train.columns if c not in self.ind + ['y']]
        else:
            selected_features = features
        print("selected features: ", selected_features)
        features_df = self.train
        features_clusters = list(set([tuple(x) for x in self.cluster_corr(features=selected_features)]))
        features_clusters = [list(x) for x in features_clusters]
        print(features_clusters)

        # for each cluster, we will create the sum of those features..to deal with multicollinearity
        for cluster in features_clusters:
            features_num = str(''.join(['_'+re.findall('\d+', s)[0] for s in cluster]))
            features_df['Combo_'+cluster[0][:4]+features_num] = features_df[cluster].sum(axis=1)

        # Combined features that share high corr with each other
        # add de-mean features by clustering timestamp
        features_df[[c+'_demean' for c in selected_features]] = features_df[['timestamp']+selected_features].groupby('timestamp')[selected_features].apply(lambda x:x-x.mean())

        # adding lag features
        for lag in range(1,2):
            features_df[[c+'_lag'+str(lag) for c in selected_features]] = features_df[['timestamp']+selected_features].groupby('timestamp')[selected_features].shift(lag).fillna(0)
            features_df[[c+'_lagdiff'+str(lag) for c in selected_features]] = features_df[['timestamp']+selected_features].groupby('timestamp')[selected_features].diff(lag).fillna(0)
            #features_df[[c+'_rollmean'+str(2*lag) for c in selected_features]] = features_df[['timestamp']+selected_features].groupby('timestamp')[selected_features].rolling(2*lag).mean().fillna(0)
            #features_df[[c+'_rollvol'+str(2*lag) for c in selected_features]] = features_df[['timestamp']+selected_features].groupby('timestamp')[selected_features].rolling(2*lag).std().fillna(0)
            features_df['y_rolling_vol'+'_lag'+str(2*lag)] = features_df[['timestamp','y']].groupby('timestamp').mean().rolling(2*lag).std().fillna(0)  
        features_df['y_cumsum'] = features_df[['timestamp', 'y']].groupby('timestamp').mean().cumsum().fillna(0)
        self.features = [c for c in features_df if c not in self.ind + ['y'] + selected_features]
        self.train = features_df.fillna(0)
        return self.train
    
    # run several Trees models in parallel to get feature importance ranks
    def get_feature_importance(self, split_ratio=0.6):

        """
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
        """
        etr_features = self.features

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
        etr = ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=17, verbose=0)
        clf = GridSearchCV(etr, {'max_depth': [4, 8],
                            'n_estimators': [50, 100]}, n_jobs=-1)
        clf.fit(x_train, y_train)
        print("best params for ETR: ", clf.best_params_)
        """
        params = {'n_estimators':100, 'max_depth':4, 'random_state':17, 'verbose':0}
        etr = ExtraTreesRegressor(n_jobs=-1, **params)
        num_features = 20
        title = 'Top ' + str(num_features) + ' Features [60 percent of Training + 40 percent for CV]'
        features_rank = self.generate_feature_importance(etr, x_train, y_train, num_features=num_features, title=title)    
        #features_rank = pd.DataFrame(list(model.get_booster().get_fscore().items()), index=self.features, columns=['importance']).sort_values('importance', ascending=False)
        top_features = [features_rank.head(10).index]
        print(features_rank)
        
        """
        model = etr.fit(x_train, y_train)
        features_rank = pd.DataFrame(model.feature_importances_, index=self.features, columns=['importance']).sort_values(by='importance', ascending=False)
        top_features = list(features_rank.head(14))
        print(top_features)
        
        plt.xticks(np.arange(len(features_rank.head(10))), features_rank.head(10).index, label="Features", rotation='vertical')
        plt.bar(np.arange(len(features_rank.head(10))), features_rank['importance'])
        plt.tight_layout()
        
        features_rank[top_features].sort_values(by='importance', inplace=True).plot(kind='barh', color='#0b3fe8', legend=None, figsize=(20,8))
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title('Top 10 ETR Features [60 percent of Training + 40 percent for CV]')
        #plt.tight_layout()
        plt.savefig('output/features_importance_etr.png')
        #print(model.score(x_cv, y_cv))
        #print(model.score(x_cv[features_rank.head(10).index], y_cv))
        """
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

        """
        base_score=0.5, booster='gbtree', colsample_bylevel=1, \
            colsample_bytree=1, gamma=0, importance_type='gain', \
            learning_rate=0.1, max_delta_step=0, max_depth=4, \
            min_child_weight=1, missing=None, n_estimators=100, \
            objective='reg:linear', random_state=0, reg_alpha=0, \
            reg_lambda=1, scale_pos_weight=1, seed=None, silent=True, \
            subsample=1
        
        xgb_ = xgb.XGBRegressor(n_jobs=-1)
        clf = xgb_.fit(x_train, y_train)
        clf = GridSearchCV(xgb_, {'max_depth': [4, 8],
                                    'min_child_weight':[50, 100],
                                    'reg_lambda:': [0.3, 0.6, 0.9],
                                    'reg_alpha': [0.3, 0.6, 0.9]}, n_jobs=-1)
     
        clf.fit(x_train, y_train)
        print("best params for XGB: ", clf.best_params_)
        """
        #params = {'n_estimators': 14, 'max_depth': 4, 'min_child_weight': 100, 'reg_alpha': 0.9, 'reg_lambda:': 0.3}
        xgb_ = xgb.XGBClassifier(n_jobs=-1, **params)
        features_rank = self.generate_feature_importance(xgb_, x_train, y_train)    
        #features_rank = pd.DataFrame(list(model.get_booster().get_fscore().items()), index=self.features, columns=['importance']).sort_values('importance', ascending=False)
        top_features = [features_rank.head(10).index]
        print(features_rank)
        #print(clf.best_score_)
        #print(clf.best_params_)
        #print(clf.score(x_cv, y_cv))
        #print(clf.score(x_cv[top_features], y_cv))
        xgb_.save_model('feature_select_xgb.model')
        fig, ax = plt.subplots(figsize=(12,18))
        xgb.plot_importance(model, ax=ax)
        plt.savefig('feature_importance_xgb.png')
        """
        train_xgb = xgb.DMatrix(data=x_train.loc[:,top_features], label=y_train)        
        model = xgb.train(params.items(), train_xgb, num_boost_round=15)
        print(model.get_score(importance_type="cover"))
        test_xgb = xgb.DMatrix(x_cv[top_features])
        pred = model.predict(test_xgb)
        xgb.plot_importance(model, color='#0b3fe8', importance_type="cover", ylabel="Features", xlabel="Feature Importance", title="XGB Feature Importance")
        plt.tight_layout()  
        plt.savefig('output/feature_importance_xgb.png')
        print(mean_squared_error(y_cv, pred))
        model.save_model('feature_select.model')
    
        models = [ExtraTreesClassifier(), RandomForestClassifier()]
        for model in models:
            model.fit(x_train, y_train)
            features_rank = pd.DataFrame(model.feature_importances_, index=self.features, columns=['importance']).sort_values('importance', ascending=False)
            print(features_rank)
            model.score(x_cv, y_cv)
            model.score(x_cv[features_rank.iloc[:,0:10]], y_cv)
        """
        print('done')


    def generate_feature_importance(self, clf, X_train, y_train, num_features=14, figsize=(14,8), title="Feature Importances"):     
        from xgboost.core import XGBoostError
        from lightgbm.sklearn import LightGBMError
    
        try: 
            if not hasattr(clf, 'feature_importances_'):
                clf.fit(X_train.values, y_train.values.ravel())

                if not hasattr(clf, 'feature_importances_'):
                    raise AttributeError("{} does not have feature_importances_ attribute".
                                        format(clf.__class__.__name__))
                    
        except (XGBoostError, LightGBMError, ValueError):
            clf.fit(X_train.values, y_train.values.ravel())
        clf_name = clf.__class__.__name__ 
        title = clf_name + " " + title
        feat_imp = pd.DataFrame({'importance':clf.feature_importances_})    
        feat_imp['Features'] = X_train.columns
        feat_imp.sort_values(by='importance', ascending=False, inplace=True)
        feat_imp = feat_imp.iloc[:num_features]
        feat_imp.sort_values(by='importance', inplace=True)
        feat_imp = feat_imp.set_index('Features', drop=True)
        feat_imp.plot.barh(title=title, figsize=figsize, color='#0b3fe8', legend=None)
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.subplots_adjust(left=0.15, right=0.9)
        plt.savefig(self.outpath+clf_name+'_feature_importance.png')
        return feat_imp


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
        features = self.features
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
    from multiprocessing import set_start_method
    set_start_method("forkserver")
    features = ['technical_20', 'technical_30']
    data_obj = Dataset()
    #print(data_obj.fullset.describe())
    df_norm = data_obj.preprocess(fill_method='median', scale_method='normalize')
    #print(df_norm.describe())
    features_obj = FeatureSelection(data_obj)
    filtered_features = features_obj.select_top_features()
    new_df = features_obj.generate_features(filtered_features)
    #backward_selection_features = features_obj.Wrapper_features_selection()
    #print(backward_selection_features)
    features_obj.get_feature_importance()
    lasso_features = features_obj.LassoCV_features_selection()
    print(lasso_features)