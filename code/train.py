import pandas as pd 
import numpy as np 
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import pickle, random
from joblib import load, dump
import environment

# general training of ensemble models using the entire set of selected models
class Model_Training(object):
    def __init__(self, train, features, num_splits=3):
        unique_timestamp = train["timestamp"].unique()
        unique_timestamps = len(unique_timestamp)
        unique_idx = int(unique_timestamps/num_splits)
        timesplit = unique_timestamp[unique_idx]
        self.cv_indexes = [(train.index[(train.timestamp >= timesplit*(split-1)) & (train.timestamp < timesplit*split)], train.index[train.timestamp >= timesplit*split]) for split in range(1, num_splits)]
        self.cv_indexes.append((train.index[(train.timestamp >= 0) & (train.timestamp < timesplit)], train.index[train.timestamp >= timesplit*2]))
        self.X_train = train[features].astype('float32')
        self.y_train = train['y'].astype('float32')
        self.outpath = "output/"

    def fit(self, clf, params={}, early_stopping_rounds=10):
        if len(params) == 0:
            with open('data/'+clf.__class__.__name__+'_gridsearchcv_best_params.pickle','rb') as f:
                best_params = pickle.load(f)
        else:
            best_params = self.generate_best_params(clf, params)
        clf.set_params(**best_params, n_jobs=-1)
        self.best_clf = clf.fit(self.X_train, self.y_train, early_stopping_rounds=early_stopping_rounds)
        dump(self.best_clf, 'model/'+clf.__class__.__name__+'.joblib')

    def transform(self):
        pass

    def generate_best_params(self, clf, params, n_jobs=-1):
        gridsearch_clf = GridSearchCV(clf, param_grid=params, cv=self.cv_indexes, n_jobs=n_jobs)
        gridsearch_clf.fit(self.X_train, self.y_train)
        best_params = gridsearch_clf.best_params_
        with open('data/'+clf.__class__.__name__+'_gridsearchcv_best_params.pickle','wb') as f:
            pickle.dump(best_params, f)
        print(best_params)
        print(gridsearch_clf.best_score_)
        return best_params

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
# aiming to reduce the potentials of overfitted models
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
    
    def fit_recurrent(self):
        model = Ridge(fit_intercept=False, n_jobs=-1)
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
    def transform(self):
        idx = 0
        for feature in self.selected_features:
            self.train[str(feature)+'_'+self.clfs[idx].__class__.__name__] = self.clfs[idx].predict(self.train[feature])
            idx += 1
        return self.train
    
    def fit_transform(self):
        self.fit()
        return self.transform()
    
    def generate_top_models(self):
        residuals, top_residuals = [], []
        lm_generator_train = self.fit_transform()
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
        feature_importance = pd.DataFrame(clf_selection.feature_importances_, index=lm_generator_train.columns).sort_values(by=[0]).tail(self.num_selected_features)
        dump(clf_selection, 'model/'+clf_selection.__class__.__name__+'_lm_trees.joblib')
        print(feature_importance)
        return feature_importance