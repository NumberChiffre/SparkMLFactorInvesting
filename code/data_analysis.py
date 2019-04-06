import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import metrics
import environment, dataset
from dataset import Dataset
from features import FeatureGenerator

# captures analysis of the dataset
# can be passed through Dataset object to preprocess
class DataAnalysis(object):
    def __init__(self, cols, outpath="output/"):
        self.outpath = outpath
        self.cols = cols
        self.ind = ['id', 'timestamp', 'y']
        self.data_obj = Dataset()
        self.df = self.data_obj.fullset

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
    
    def analyze_corr_with_score(self):
        df = self.data_obj.preprocess(fill_method='median', scale_method='none')
        features_obj = FeatureGenerator(self.data_obj)
        train = features_obj.train
        corr_with_y = train[features_obj.features].corrwith(train.y).abs().sort_values(ascending=False).to_frame(name='corr')
        corr_with_y.index.name = 'features'
        corr_with_y.reset_index(level=0, inplace=True)
        corr_with_y['rank_corr'] = corr_with_y['corr'].rank()
        reward_df = pd.read_csv(self.outpath+'_Ridge_Reward.csv')
        reward_df['rank_reward'] = reward_df['reward'].rank()
        rank_df = pd.merge(corr_with_y, reward_df, on='features')
        top_rank_df = rank_df[(abs(rank_df['rank_corr']-rank_df['rank_reward']) < 30) & (rank_df['rank_reward'] > 0)]
        top_rank_df.to_csv(self.outpath+'_Top_Ridge_Corr_Reward_Rank.csv')

    # overfitting data using Ridge
    def analyze_selected_features(self):
        df = self.data_obj.preprocess(fill_method='median', scale_method='none')
        features_obj = FeatureGenerator(self.data_obj)
        train = features_obj.train
        filtered_features = features_obj.filter_features()
        top_features = filtered_features[:10]
        top_features = features_obj.features
        track_score = {}
        for feature in top_features:
            model = Ridge(fit_intercept=False, normalize=True)
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
                #rewards[timestamp] = reward
                if done:
                    """
                    rewards_df = pd.DataFrame.from_dict(rewards, orient='index')
                    rewards_df.plot(kind='line', color='#0b3fe8', legend=None, figsize=(16,8))
                    plt.xlabel('Timestamps')
                    plt.ylabel('Rewards')
                    plt.title('Ridge Regression Using '+feature+' [60 percent of Training + 40 percent for Testing]')
                    plt.savefig(self.outpath+feature+'_lm_rewards.png')
                    """
                    track_score[feature] = info['public_score']
                    print(feature, track_score[feature])
                    break
        rewards_df = pd.DataFrame.from_dict(track_score, orient='index', columns=['features','reward']).sort_values(by='reward', ascending=False)
        rewards_df.to_csv(self.outpath+'_Ridge_Reward.csv')
        print(rewards_df)

    def analyze_corr(self, df_norm=None):
        if df_norm is None:
            df, y = self.df.drop(columns=self.ind), self.df['y']
            df_features = self.df.drop(columns=self.ind + ['y'])
        else:
            df, y = df_norm.drop(columns=self.ind), df_norm['y']
            df_features = df_norm.drop(columns=self.ind + ['y'])

        corr = df_features.corr()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0,len(df_features.columns),1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(df_features.columns)
        ax.set_yticklabels(df_features.columns)
        plt.savefig(self.outpath+'features_corr.png')

        corr = corr.abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        high_corr = [column for column in upper.columns if any(upper[column] >= 0.90)]
        return df.corrwith(y).sort_values(ascending=False)

    def analyze_outliers(self):
        df = self.data_obj.preprocess(fill_method='none', scale_method='none')
        features_obj = FeatureGenerator(self.data_obj)
        train = features_obj.train
        y = train['y'].values
        plt.hist(y, bins=70, color='#0b3fe8')
        plt.xlabel('Returns')
        plt.ylabel('Count')
        plt.title('Empirical Return Distribution of The Output Value Y [VIX-Related Product]')
        plt.savefig(self.outpath+'y_distribution.png')
        plt.clf()

        """
        time_targets = train.groupby('timestamp')['y'].mean()
        plt.figure(figsize=(12, 5))
        plt.plot(time_targets)
        plt.xlabel('Timestamps')
        plt.ylabel('Mean of target')
        plt.title('Change in target over time - Red lines = new timeperiod')
        for i in timediff[timediff > 5].index:
            plt.axvline(x=i, linewidth=0.25, color='red')
        """

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
    cols = ['technical_20', 'technical_30', 'technical_29', 'technical_40']
    data_analysis = DataAnalysis(cols)
    data_analysis.analyze_outliers()
    data_analysis.analyze_corr_with_score()
    data_analysis.analyze_selected_features()
    # unprocessed data
    #print(data_analysis.analyze_corr().head(6))
    #print(data_analysis.analyze_corr().tail(6))
    #top_nan_features = data_analysis.analyze_nan()
    #print(data_analysis.df.describe())

    # preprocessed data
    data_obj = dataset.create()
    df = data_obj.preprocess()
    data_analysis.analyze_corr(df)
    #print(data_analysis.analyze_corr(df).head(6))
    #print(data_analysis.analyze_corr(df).tail(6))
    features_obj = FeatureGenerator(data_obj)
    clusters = features_obj.cluster_corr()
    print(clusters)
