"""
    1. loading all models and test them all together
    2. Require transforming testing dataset into usable features through FeatureSelection class
    3. Load each model with their params + subset of features
    4. For each model + params + distinct features, compute their scores
    5. Output on graph + all prediction related metrics such as R2, MSE, etf..
"""
from train import *
from features import *
from dataset import Dataset
import environment

class Model_Testing(object):
    def __init__(self, feature_obj):
        self.train = feature_obj.train
        self.test = feature_obj.test
        self.feature_obj = feature_obj
        self.models = [xgb.XGBRegressor(n_jobs=-1), Ridge(), ExtraTreesRegressor(n_jobs=-1)]
        self.num_top_features = 14
        self.modelpath = 'model/'   
        self.outpath = 'output/'

    # create new testing features based on the training set
    # another way is to transform all features during training, while ensuring no testing set was touched
    def generate_test_models(self):
        filtered_features = self.feature_obj.select_top_features(num_features_threshold=self.num_top_features)
        self.full_trans_data = self.feature_obj.generate_features(features=filtered_features, use_full_set=True)
        print("Features for testing:\n", self.feature_obj.features)
        features_dict = self.feature_obj.features_dict
        self.y_actual_dict, self.y_pred_dict = {}, {}
        self.cum_reward_dict, self.timestamp_dict = {}, {}

        # generate reward score for each model
        # load each trained model
        for clf in self.models:
            self.clf_name = clf.__class__.__name__
            
            if 'Regressor' in self.clf_name:
                model_file = self.modelpath+self.clf_name+'.joblib'
                trained_clf = load(model_file)
                features_file = self.outpath+self.clf_name+'_feature_importance.csv'
                features_df = pd.read_csv(features_file)
                top_features = list(features_df.tail(self.num_top_features)['Features'].values)
                top_features = self.feature_obj.features
                self.generate_single_model(trained_clf, top_features=top_features)
            else:
                for features_name in features_dict:
                    model_file = self.modelpath+self.clf_name+'_'+features_name+'.joblib'
                    trained_clf = load(model_file)
                    self.generate_single_model(trained_clf, top_features=features_dict[features_name], features_name=features_name)

        # generate reward chart
        self.generate_reward_chart()

    def generate_single_model(self, clf, top_features=[], features_name=""):  
        print("Testing: ", self.clf_name, " and params:\n", pd.DataFrame.from_dict(clf.get_params(), orient='index', columns=['values']))

        # set up environment for testing
        env = environment.make(self.full_trans_data, features=top_features)
        observation = env.reset()
        self.start_train, self.end_train = env.train["timestamp"].values[0], env.train["timestamp"].values[-1]
        self.start_test, self.end_test = env.test["timestamp"].values[0], env.test["timestamp"].values[-1]

        y_actual_list, y_pred_list = [], []
        cum_reward_list, timestamp_list = [], []
        
        while True: 
            observation.target.y = clf.predict(observation.features[top_features])
            target = observation.target
            timestamp = observation.features["timestamp"][0]
            y_test = env.test
            actual_y = list(y_test[y_test["timestamp"] == timestamp]["y"].values)
            observation, reward, done, info = env.step(target)
            
            pred_y = list(target.y.values)
            y_actual_list.extend(actual_y)
            y_pred_list.extend(pred_y)
            cum_reward = environment.get_reward(np.array(y_actual_list), np.array(y_pred_list))
            cum_reward_list.append(cum_reward)
            timestamp_list.append(timestamp)

            # save results to produce charts for each classifier
            if done:
                print(info)
                if 'Regressor' in self.clf_name:
                    key = self.clf_name
                else:
                    key = self.clf_name+'_'+features_name
                self.y_actual_dict[key] = y_actual_list
                self.y_pred_dict[key] = y_pred_list
                self.cum_reward_dict[key] = cum_reward_list
                self.timestamp_dict[key] = timestamp_list
                del env, observation
                gc.collect()
                break
    
    def generate_reward_chart(self):
        fig = plt.figure(figsize=(12, 6))
        colors = iter(plt.cm.rainbow(np.linspace(0,1,len(self.cum_reward_dict))))
        for model in self.cum_reward_dict:
            plt.plot(self.timestamp_dict[model], self.cum_reward_dict[model], c=next(colors), label=str(model))
        plt.plot(self.timestamp_dict[model], [0]*len(self.timestamp_dict[model]), c='red')
        plt.title("Out-Of-Sample Reward Performance: "+"Train Timestamps["+str(int(self.start_train))+" - "+str(int(self.end_train))+"], Test Timestamps["+str(int(self.start_test))+" - "+str(int(self.end_test))+"]")
        plt.ylim([-0.06, 0.06])
        plt.xlim(self.timestamp_dict[model][0], self.timestamp_dict[model][-1])
        plt.xlabel('Timestamps')
        plt.legend(framealpha=1, frameon=True)
        plt.savefig(self.outpath+'test_cum_rewards.png')      
        plt.clf() 

if __name__ == '__main__':
    from multiprocessing import set_start_method
    set_start_method("forkserver")  
    data_obj = Dataset()
    df_norm = data_obj.preprocess(fill_method='median', scale_method='none')
    feature_obj = FeatureGenerator(data_obj)
    model_testing = Model_Testing(feature_obj)
    model_testing.generate_test_models()