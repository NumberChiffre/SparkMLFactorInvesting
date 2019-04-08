import warnings
warnings.filterwarnings(action='ignore',category = DeprecationWarning)
warnings.simplefilter(action='ignore',category = DeprecationWarning)
import pandas as pd
pd.reset_option('all') 
import numpy as np 
import statsmodels.api as sm
from skopt import BayesSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import pickle, random, os
from joblib import load, dump
import environment

# general training of ensemble models using the entire set of selected models
class Model_Training(object):
    def __init__(self, train, features, num_splits=3):
        unique_timestamp = train["timestamp"].unique()
        unique_timestamps = len(unique_timestamp)
        unique_idx = int(unique_timestamps/num_splits)
        timesplit = unique_timestamp[unique_idx]
        self.train = train
        self.features = features
        self.cv_indexes = [(train.index[(train.timestamp >= timesplit*(split-1)) & (train.timestamp < timesplit*split)], train.index[(train.timestamp >= timesplit*split) & (train.timestamp < timesplit*(1+split))]) for split in range(1, num_splits)]
        self.cv_indexes.append((train.index[(train.timestamp >= 0) & (train.timestamp < timesplit)], train.index[train.timestamp >= timesplit*2]))
        self.X_train = train[features].astype('float32')
        self.y_train = train['y'].astype('float32')
        self.outpath = "output/"
    
    def fit(self, clf, params={}, features_dict={}, multiple_models=False):
        best_params = self.load_best_params(clf, params)
        print('Using the following params: \n', pd.DataFrame.from_dict(best_params, orient='index', columns=['values']))
        clf.set_params(**best_params)
        self.best_clf = clf.fit(self.X_train, self.y_train)
        dump(self.best_clf, 'model/'+clf.__class__.__name__+'.joblib')

    def load_best_params(self, clf, params={}):
        params_file = 'data/'+clf.__class__.__name__+'_randomizedsearchcv_best_params.pickle'
        if os.path.isfile(params_file):
            with open(params_file,'rb') as f:
                best_params = pickle.load(f)
        else:
            best_params = self.generate_best_params(clf, params)
        return best_params

    def generate_best_params(self, clf, params, n_jobs=-1):
        search_clf = RandomizedSearchCV(clf, param_distributions=params, scoring='neg_mean_squared_error', cv=self.cv_indexes, n_iter=25, n_jobs=n_jobs)
        search_clf.fit(self.X_train, self.y_train)
        dump(search_clf, 'model/'+clf.__class__.__name__+'_randomizedsearchcv.joblib')
        best_params = search_clf.best_params_
        with open('data/'+clf.__class__.__name__+'_randomizedsearchcv_best_params.pickle','wb') as f:
            pickle.dump(best_params, f)
        print(best_params)
        print(search_clf.best_score_)

    # for each set of train-validation data, we generate the cumulative reward score for each classifier
    def cross_validate_multiple(self, clfs, params_list=[], features_dict={}):
        i = 0
        for train_cv_indexes in self.cv_indexes:

            # produce results to generate cumulative reward charts for each set of indexes
            y_actual_dict, y_pred_dict = {}, {}
            cum_reward_dict, timestamp_dict = {}, {}

            for clf in clfs:
                # get params for the given classifier
                # train the classifier using the iterated train/validation indexes
                best_params = self.load_best_params(clf, params_list[i])
                print('Using the following params: \n', pd.DataFrame.from_dict(best_params, orient='index', columns=['values']))
                clf.set_params(**best_params)
                X_train, y_train = self.X_train.ix[train_cv_indexes[0], :], self.y_train.ix[train_cv_indexes[0]]
                clf.fit(X_train, y_train)
                dump(clf, 'model/'+clf.__class__.__name__+'train_sub'+str(i)+'.joblib')
                
                # set up environment for testingg
                env = environment.make(self.train, train_cv_indexes, use_cv=True, features=self.features)
                observation = env.reset()

                y_actual_list, y_pred_list = [], []
                cum_reward_list, timestamp_dict = [], []
                
                while True: 
                    observation.target.y = clf.predict(observation.features)
                    target = observation.target
                    timestamp = observation.features["timestamp"][0]
                    actual_y = list(test[test["timestamp"] == timestamp]["y"].values)
                    observation, reward, done, info = env.step(target)
                    
                    pred_y = list(target.y.values)
                    y_actual_dict.extend(actual_y)
                    y_pred_dict.extend(pred_y)
                    cum_reward = environment.get_reward(np.array(y_actual_list), np.array(y_pred_list))
                    cum_reward_dict.append(overall_reward)
                    timestamp_dict.append(timestamp)

                    # save results to produce charts for each classifier
                    if done:
                        print(info)
                        y_actual_dict[clf.__class__.__name__] = y_actual_list
                        y_pred_dict[clf.__class__.__name__] = y_pred_list
                        cum_reward_dict[clf.__class__.__name__] = cum_reward_list
                        timestamp_dict[clf.__class__.__name__] = timestamp_list
                        break

            # for the given set of train-validation indexes, print their actual reward scores
            fig = plt.figure(figsize=(12, 6))
            colors = iter(plt.cm.rainbow(np.linspace(0,1,len(cls))))
            for clf in cls:
                dict_key = clf.__class__.__name__
                plt.plot(timestamp_dict[dict_key], cum_reward_dict[dict_key], c=next(colors), label=dict_key)
            plt.plot(timestamp_dict[dict_key], [0]*len(timestamp_dict[dict_key]), c='red')
            start_train, end_train = train_cv_indexes[0][0][0], train_cv_indexes[0][0][-1]
            start_valid, end_valid = train_cv_indexes[0][1][0], train_cv_indexes[0][1][-1]
            plt.title("Cross-Validation Set #"+str(i+1)+": "+"Train Timestamps["+str(start_train)+"-"+str(end_train)+"], Test Timestamps["+str(start_valid)+"-"+str(end_valid)+"]")
            plt.ylim([-0.04, 0.04])
            plt.xlim(timestamp_dict[dict_key][0], timestamp_dict[dict_key][-1])
            plt.savefig(self.outpath+'cum_rewards_cv'+str(i+1)+'.png')       
            i += 1
            
    def generate_feature_importance(self, num_features=14, figsize=(13,8), title="Feature Importances"):     
        X_train, y_train = self.X_train, self.y_train
        """
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
        """
        clf = self.best_clf
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


# assuming that the Dataset object has not used preprocess()
# training linear models by taking a small subset of features with the lowest residuals
# aiming to reduce the potentials of overfitted models by combining them into Tree Models
class LinearModelGenerator(object):
    def __init__(self, train, features, num_top_models=10, num_selected_features=20, num_max_model_feature=2, random_seed=124124):
        self.train = train
        self.features = features
        self.num_top_models = num_top_models
        self.num_selected_features = num_selected_features
        self.num_max_model_feature = num_max_model_feature
        self.selected_features = []
        self.clfs = []
        self.random_seed = random_seed
        self.min_y_cut, self.max_y_cut = -0.086, 0.086

    def fit_recurrent(self):
        model = Ridge(fit_intercept=False)
        quantile_thresh, best_mse = 0.99, 1e15, 
        non_na_idx = train.dropna().index
        limitlen = len(train)*self.limit_size_train

        while True:
            model.fit(self.train.ix[non_na_idx], self.train.ix[non_na_idx])
            pred = model.predict(train.ix[non_na_idx])
            mse = mean_squared_error(y.ix[non_na_idx], pred)
            if mse < best_mse:
                best_mse = mse
                self.best_model = model
                residual = y.ix[non_na_idx] - pred
                non_na_idx = residual[abs(residual) <= abs(residual).quantile(quantitle_thresh)].index
                if len(non_na_idx) < limitlen:
                    break
            else:
                self.best_model = model
                break

    def predict_recurrent(self, test):
        return self.bestmodel.predict(test)

    def fit(self):
        prev_mse = np.zeros(self.num_selected_features) + 1e15
        random.seed(self.random_seed)
        random.shuffle(self.features)

        # pair up (i,j) features
        for feature in self.features[:self.num_selected_features]:
            self.selected_features.append([feature])
        remaining_features = self.features[self.num_selected_features:]
        
        # pairing up selected features with the remaining features
        # by default, combining two features
        for feature1 in remaining_features:
            mse = []
            feature_idx = 0
            for feature2 in self.selected_features:
                if len(feature2) < self.num_max_model_feature:
                    clf = Ridge(fit_intercept=False) 
                    clf.fit(self.train[[feature1] + feature2], self.train['y'])
                    pred = clf.predict(self.train[[feature1] + feature2])
                    mse.append(mean_squared_error(self.train['y'], pred))
                else:
                    mse.append(prev_mse[feature_idx])
                feature_idx += 1
            mse_delta = prev_mse - mse

            # checking for lowest residuals
            if mse_delta.max() > 0:
                top_feature_idx = mse_delta.argmax()
                prev_mse[top_feature_idx] = mse[top_feature_idx]
                self.selected_features[top_feature_idx].append(feature1)

        # generate linear models based on a set of features combined based on low residuals
        for feature in self.selected_features:
            clf = Ridge(fit_intercept=False) 
            clf.fit(self.train[feature], self.train['y'])
            self.clfs.append(clf)
            print(feature)
        print('Finished generating linear models through LinearModelGenerator')

    # add these linear models as trees for ExtraTreesRegressor model
    def transform(self, df):
        idx = 0
        for feature in self.selected_features:
            df[str(feature)+'_'+self.clfs[idx].__class__.__name__] = self.clfs[idx].predict(df[feature])
            idx += 1
        self.features = [c for c in df.columns if c not in ['timestamp', 'id', 'y']]
        return df
    
    def fit_transform(self):
        self.fit()
        return self.transform()
    
    # using LGBM + ETR to generate top models with checks on residual training
    def generate_top_models(self):
        residuals, top_residuals = [], []
        lm_generator_train = self.fit_transform(df=self.train)[self.features]
        total_features = lm_generator_train.columns
        clf = ExtraTreesRegressor(n_estimators=140, max_depth=4, n_jobs=-1)
        clf.fit(lm_generator_train, self.train['y'])

        for x in clf.estimators_:
            residuals.append(abs(x.predict(lm_generator_train) - self.train['y']))
        min_residuals = np.argmin(np.array(residuals).T, axis=1)
        top_models = pd.Series(min_residuals).value_counts().head(self.num_top_models).index
        
        for model in top_models:
            top_residuals.append(residuals[model])
        target_residuals = np.argmin(np.array(top_residuals).T, axis=1)
        clf_selection = ExtraTreesRegressor(n_estimators=140, max_depth=4, n_jobs=-1)
        clf_selection.fit(lm_generator_train, target_residuals)
        feature_importance = pd.DataFrame(clf_selection.feature_importances_, index=total_features).sort_values(by=[0], ascending=False).head(self.num_selected_features)
        dump(clf_selection, 'model/'+clf_selection.__class__.__name__+'_lm_trees.joblib')
        self.clf_selection = clf_selection
        self.total_features = total_features
        print(feature_importance)
        return feature_importance
    
    def load_top_models(self):
        if os.path.isfile('model/ExtraTreesRegressor_lm_trees.joblib'):
            clf = load('model/ExtraTreesRegressor_lm_trees.joblib')
            feature_importance = pd.DataFrame(clf.feature_importances_, index=self.total_features).sort_values(by=[0], ascending=False).head(self.num_selected_features)
        else:
            feature_importance = self.generate_top_models()
            clf = self.clf_selection

        env = environment.make()
        o = env.reset()
        y_actual_list = []
        y_pred_list = []
        r1_overall_reward_list = []
        ts_list = []
        idx = 0
        while True:
            idx += 1
            test = o.features
            features = [x for x in test.columns if x not in ['timestamp', 'id', 'y']]
            timestamp = o.features.timestamp[0]
            pred = o.target
            test = self.transform(test[features])
            selected_pred = clf.predict_proba(test.loc[:, features+self.selected_features])
            pred['y'] = selected_pred.clip(self.min_y_cut, self.max_y_cut)
            
            o, reward, done, info = env.step(pred)
            if reward > 0:
                countplus += 1
            
            if indice % 100 == 0:
                print(indice, countplus, reward, np.array(list(rewards.values())).mean(), info)
            
            actual_y = list(test[test["timestamp"] == timestamp]["y"].values)
            y_actual_list.extend(actual_y)
            y_pred_list.extend(pred['y'])
            overall_reward = get_reward(np.array(y_actual_list), np.array(y_pred_list))
            r1_overall_reward_list.append(overall_reward)
            ts_list.append(timestamp)

            if done:
                fig = plt.figure(figsize=(12, 6))
                plt.plot(ts_list, r1_overall_reward_list, c='blue')
                plt.plot(ts_list, [0]*len(ts_list), c='red')
                plt.title("Cumulative R value change for LinearML model")
                plt.ylim([-0.04,0.04])
                plt.savefig('output/linearML_rewards.png')
                print(info)
                break