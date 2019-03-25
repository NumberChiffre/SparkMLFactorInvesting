import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import environment, dataset
from features import FeatureSelection

# captures analysis of the dataset
# can be passed through Dataset object to preprocess
class DataAnalysis(object):
    def __init__(self, env, cols, outpath="output/"):
        self.env = env
        self.df = env.fullset
        self.outpath = outpath
        self.cols = cols
        self.ind = ['id', 'timestamp', 'y']

    def analyze_nan(self, df_norm=None):
        # check % of NaN per feature, return dataframe containing % of NaN per feature
        # useful to look at which features to choose from
        if df_norm is None:
            df = self.df.drop(columns=self.ind)
        else:
            df = df_norm.drop(columns=self.ind) 
        null_counts = df.isnull().sum()/len(df)
        null_counts = null_counts.sort_values()
        null_counts = pd.DataFrame({'Features':null_counts.index, '% Missing NaN':null_counts.values})
        null_counts.set_index('Features',inplace=True)
        plt.figure(figsize=(20,10))
        plt.xticks(np.arange(len(null_counts)),null_counts.index,rotation='vertical')
        plt.ylabel('Percentage of rows with NaN')
        plt.bar(np.arange(len(null_counts)),null_counts['% Missing NaN'])
        plt.tight_layout()
        plt.savefig(self.outpath+'missing_nan.png')
        return null_counts
    
    def analyze_corr(self, df_norm=None):
        if df_norm is None:
            df, y = self.df.drop(columns=self.ind), self.df['y']
        else:
            df, y = df_norm.drop(columns=self.ind), df_norm['y']
        return df.corrwith(y).sort_values(ascending=False)

    def analyze_outliers(self):
        pass

    def security_regression(self):
        train_data, test_data = self.env.train[self.cols+self.ind], self.env.test[self.cols+self.ind]
        train_data = train_data.fillna(train_data[self.cols].dropna().median())
        test_data = test_data.fillna(test_data[self.cols].dropna().median())
        unique_id = np.unique(train_data['id'] )
        unique_id_len = len(unique_id)

        # fit linear regression to train on y for each security id
        security_id, mae, mse, rmse = [], [], [], []
        for i in range(unique_id_len):  
            train_id = train_data[train_data['id']==unique_id[i]]
            test_id = test_data[test_data['id']==unique_id[i]]

            # in case test data does not contain the security id
            if len(test_id) <= 1:
                continue
            model = LinearRegression().fit(train_id[self.cols],train_id['y'])
            y_pred = model.predict(test_id[self.cols])
            security_id.append(unique_id[i])
            mae.append(metrics.mean_absolute_error(test_id['y'], y_pred))
            mse.append(metrics.mean_squared_error(test_id['y'], y_pred))  
            rmse.append(np.sqrt(metrics.mean_squared_error(test_id['y'], y_pred)))
        output = pd.DataFrame(list(zip(security_id, mae, mse, rmse)), columns=['id','mae','mse','rmse'])
        output = output.sort_values('mse',ascending=True)
        """
        lr_model = list(train_data.groupby('id').apply(lambda df: LinearRegression().fit(df[cols], df['y'])))
        for model in lr_model:
            y_pred = model.predict(test_data[cols])
            print('Mean Absolute Error:', metrics.mean_absolute_error(test_data['y'], y_pred))  
            print('Mean Squared Error:', metrics.mean_squared_error(test_data['y'], y_pred))  
            print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_data['y'], y_pred)))
        """
        return output


if __name__ == "__main__":
    env = environment.make()
    cols = ['technical_20', 'technical_30', 'technical_29', 'technical_40']
    data_analysis = DataAnalysis(env, cols)

    # unprocessed data
    print(data_analysis.analyze_corr().head(6))
    print(data_analysis.analyze_corr().tail(6))
    top_nan_features = data_analysis.analyze_nan()
    print(data_analysis.df.describe())

    # preprocessed data
    data_obj = dataset.create()
    df = data_obj.preprocess()
    print(data_analysis.analyze_corr(df).head(6))
    print(data_analysis.analyze_corr(df).tail(6))
    features_obj = FeatureSelection(data_obj)
    clusters, singles = features_obj.cluster_corr()
    print(clusters)
    print(singles)
    