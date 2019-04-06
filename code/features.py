import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd 
import numpy as np 
import re
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LassoCV, Ridge
from sklearn.ensemble import ExtraTreesRegressor
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import xgboost as xgb
import lightgbm as lgb
import environment
from dataset import Dataset
from train import Model_Training, LinearModelGenerator


def get_reward(y_true, y_fit):
    R2 = 1 - np.sum((y_true - y_fit)**2) / np.sum((y_true - np.mean(y_true))**2)
    R = np.sign(R2) * math.sqrt(abs(R2))
    return(R)

# use the Dataset object to perform feature selection
class FeatureGenerator(object):
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
        self.min_y_cut, self.max_y_cut = -0.086, 0.086
        y_is_above_cut = (self.train.y >= self.max_y_cut)
        y_is_below_cut = (self.train.y <= self.min_y_cut)
        y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
        self.train = self.train.loc[y_is_within_cut, :]
        self.train = self.train.reset_index(drop=True)  
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
            y_actual_list = []
            y_pred_list = []
            overall_reward_list = []
            ts_list = [] 
            env = environment.make(self.df_norm)
            observation = env.reset()

            while True:
                test_x = np.array(observation.features[feature].values).reshape(-1,1)
                observation.target.y = model.predict(test_x).clip(self.min_y_cut, self.max_y_cut)
                target = observation.target
                timestamp = observation.features["timestamp"][0]
                actual_y = list(train[train["timestamp"] == timestamp]["y"].values)
                observation, reward, done, info = env.step(target)

                pred_y = list(target.y.values)
                y_actual_list.extend(actual_y)
                y_pred_list.extend(pred_y)
                overall_reward = get_reward(np.array(y_actual_list), np.array(y_pred_list))
                overall_reward_list.append(overall_reward)
                ts_list.append(timestamp)
                if done:
                    track_score[feature] = info['public_score']
                    print(feature, track_score[feature])
                    break

        rewards_df = pd.DataFrame.from_dict(track_score, orient='index', columns=['reward']).sort_values(by='reward', ascending=False)
        rewards_df.index.name = 'features'
        rewards_df.to_csv(self.outpath+'_Ridge_Reward.csv')

        fig = plt.figure(figsize=(12, 6))
        plt.plot(ts_list, r1_overall_reward_list, c='blue')
        plt.plot(ts_list, [0]*len(ts_list), c='red')
        plt.title("Cumulative R value change for Univariate Ridge (technical_20)")
        plt.ylim([-0.04,0.04])
        plt.xlim([850, 1850])
        plt.savefig(self.outpath+'_model.png')
        return overall_reward_list, ts_list

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

        # remove features to be combined from total set of features
        for feature_list in features_clusters:
            for feature in feature_list:
                if feature in selected_features:
                    selected_features.remove(feature)
        print("Clustered features based on multicollinearity: ", features_clusters)

        # for each cluster, we will create the sum of those features..to deal with multicollinearity
        for cluster in features_clusters:
            features_num = str(''.join(['_'+re.findall('\d+', s)[0] for s in cluster]))
            features_df['Combo_'+cluster[0][:4]+features_num] = features_df[cluster].sum(axis=1)

        # add value from technical 20 - 30, which shares a high corr with ewm of output y
        # check its market premium return
        selected_features.append('20_30')
        features_df['20_30'] = features_df['technical_20'] - features_df['technical_30']
        mkt_20_30 = features_df[['timestamp', '20_30']].groupby('timestamp')['20_30'].mean()
        features_df['mkt_20_30'] = mkt_20_30[features_df['timestamp']].values
        features_df['excess_20_30'] = features_df['20_30'] - features_df['mkt_20_30']   
        vol_20_30 = features_df[['timestamp', '20_30']].groupby('timestamp')['20_30'].std() * np.sqrt(255.)
        vol_20_30 = vol_20_30.rolling(window=2, min_periods=1).mean()
        features_df['vol_20_30'] = vol_20_30[features_df['timestamp']].values

        # de-mean each selected feature 
        features_df[[c+'_demean' for c in selected_features]] = features_df[['timestamp']+selected_features].groupby('timestamp')[selected_features].apply(lambda x:x-x.mean())

        # adding lag features
        for lag in [1,5,10,20]:
            features_df[[c+'_lag'+str(lag) for c in selected_features]] = features_df[['id']+selected_features].groupby('id')[selected_features].shift(lag)
            features_df[[c+'_lagdiff'+str(lag) for c in selected_features]] = features_df[['id']+selected_features].groupby('id')[selected_features].diff(lag)
            features_df[[c+'_rolling'+str(lag) for c in selected_features]] = features_df[['id']+selected_features].groupby('id')[selected_features].rolling(lag, 1).mean().reset_index(0, drop=True)
            features_df['y_rolling_vol'+'_lag'+str(lag)] = features_df[['timestamp','y']].groupby('timestamp')['y'].mean().rolling(lag).std()
            """
            vol = features_df[['timestamp']+selected_features].groupby('timestamp').std()
            vol = vol.rolling(window=lag, min_periods=1).mean()
            features_df[[c+'_vol_lag'+str(lag) for c in selected_features]] = vol[features_df['timestamp']].values
            """
        # addding lag y values
        #features_df['y_cumsum'] = features_df[['timestamp', 'y']].groupby('timestamp').mean().cumsum().fillna(0)
        for lag in [12,26]:
            features_df['y_ewm_'+str(lag)+'_mean'] = features_df.groupby('id')['y'].apply(lambda x:x.ewm(span=lag).mean())
        features_df['y_macd_mean'] = features_df['y_ewm_12_mean'] - features_df['y_ewm_26_mean']
        self.features = [c for c in features_df if c not in self.ind + ['y'] + selected_features]
        self.train = features_df
        return self.train


    # run several Trees models in parallel to get feature importance ranks
    def get_feature_importance(self, split_ratio=0.6):
        bds = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},
        {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},
        {'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 300)},
        {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 10)}]


        params = {
                   '0':{'max_depth': range(3,9), 
                    'min_samples_leaf': range(50, 101), 
                    'n_estimators': range(100,201)
                   },
                   '1':{'max_depth': range(3,9),
                    'min_child_weight': range(50, 101),
                    'reg_lambda:': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    'reg_alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    'colsample_bytree': [x/10 for x in range(3, 10)],
                    'subsample': [x/10 for x in range(5, 10)],
                    'n_estimators': range(100, 201),
                    'learning_rate': [0.01, 0.05, 0.1]
                   },
                   '2':{'num_leaves': range(50, 101),
                    'feature_fraction': [x/10 for x in range(4, 9)],
                    'bagging_fraction': [x/10 for x in range(5, 8)],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'n_estimators': range(100, 201),
                    'max_depth': range(3,9),
                    'reg_lambda:': [x/10 for x in range(0, 10)],
                    'reg_alpha': [x/10 for x in range(0, 10)]
                   }
        }
        """
        params = {
                   '0':{'max_depth': range(4, 9), 
                    'min_samples_leaf':range(50, 100), 
                    'n_estimators': stats.randint(100, 200)
                   },
                   '1':{'max_depth': stats.randint(3, 6),
                    'min_child_weight': stats.randint(50, 100),
                    'reg_lambda:': stats.uniform(0.01, 0.3),
                    'reg_alpha': stats.uniform(0.01, 0.3),
                    'colsample_bytree': stats.uniform(0.4, 0.9),
                    'subsample': stats.uniform(0.3, 0.9),
                    'n_estimators': stats.randint(100, 200),
                    'learning_rate': stats.uniform(0.01, 0.2)
                   },
                   '2':{'num_leaves': stats.randint(50, 100),
                    'feature_fraction': stats.uniform(0.4, 0.8),
                    'bagging_fraction': stats.uniform(0.5, 0.7),
                    'learning_rate': stats.uniform(0.01, 0.2),
                    'n_estimators': stats.randint(100, 200),
                    'max_depth': stats.randint(3, 6),
                    'reg_lambda:': stats.uniform(0.01, 0.3),
                    'reg_alpha': stats.uniform(0.01, 0.3)
                   }
        }
        """
        training = Model_Training(self.train, self.features)
        models = [ExtraTreesRegressor(n_jobs=-1), xgb.XGBRegressor(n_jobs=-1), lgb.LGBMRegressor(n_jobs=-1)]
        num_features = 20
        title = 'Top ' + str(num_features) + ' Features [60 percent of Training + 40 percent for CV]'
        i = 0
        for clf in models:
            if i != 0:
                print(params[str(i)])
                training.fit(clf, params=params[str(i)])
                features_rank = training.generate_feature_importance(num_features=num_features, title=title)
                features_rank.to_csv(self.outpath+clf.__class__.__name__+'_feature_importance.csv')
            i += 1
        print('Finished generating feature importance')


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
    print("Creating new set of features based on unprocessed dataset")
    features = data_obj.features
    df_norm = data_obj.preprocess(fill_method='median', scale_method='none')
    """
    df_raw = df_norm.ix[df_norm[abs(df_norm.y) < abs(df_norm.y).max()*0.95].index, :]
    lm_generator = LinearModelGenerator(train=df_raw, features=features, num_selected_features=30)
    top_features_lm_etr = lm_generator.generate_top_models()
    """
    features_obj = FeatureGenerator(data_obj)
    #features_obj.generate_single_reward()
    filtered_features = features_obj.select_top_features()
    new_df = features_obj.generate_features(filtered_features)
    features_obj.get_feature_importance()
