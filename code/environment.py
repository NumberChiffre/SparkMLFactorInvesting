import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from dataset import Dataset

# turn dataset into usable features mimicking Kagglegym
class Environment(object):
    def __init__(self, df=pd.DataFrame(), filepath="data/train.h5", split_ratio=0.6):
        if df.empty:
            with pd.HDFStore(filepath, "r") as hfdata:
                self.fullset = hfdata.get("train")
        else:
            self.fullset = df
        self.timestamp = 0
        self.unique_timestamp = self.fullset["timestamp"].unique()
        self.n = len(self.unique_timestamp)
        self.unique_idx = int(self.n*split_ratio)
        timesplit = self.unique_timestamp[self.unique_idx]
        self.split_ratio = split_ratio
        self.train = self.fullset[self.fullset.timestamp < timesplit]
        self.test = self.fullset[self.fullset.timestamp >= timesplit]

        # Needed to compute final score
        self.full = self.test.loc[:, ['timestamp', 'y']]
        self.full['y_hat'] = 0.0
        self.temp_test_y = None

    def reset(self):
        timesplit = self.unique_timestamp[self.unique_idx]
        self.unique_idx = int(self.n*self.split_ratio)
        self.unique_idx += 1
        subset = self.test[self.test.timestamp == timesplit]

        # reset index to conform to how kagglegym works
        target = subset.loc[:, ['id', 'y']].reset_index(drop=True)
        self.temp_test_y = target['y']
        target.loc[:, 'y'] = 0.0  # set the prediction column to zero
        features = subset.iloc[:, :110].reset_index(drop=True)
        observation = Observation(self.train, target, features)
        return observation

    def step(self, target):
        timesplit = self.unique_timestamp[self.unique_idx-1]
        # Since full and target have a different index we need
        # to do a _values trick here to get the assignment working
        y_hat = target.loc[:, ['y']]
        self.full.loc[self.full.timestamp == timesplit, ['y_hat']] = y_hat._values

        if self.unique_idx == self.n:
            done = True
            observation = None
            reward = r_score(self.temp_test_y, target.loc[:, 'y'])
            score = r_score(self.full['y'], self.full['y_hat'])
            info = {'public_score': score}
        else:
            reward = r_score(self.temp_test_y, target.loc[:, 'y'])
            done = False
            info = {}
            timesplit = self.unique_timestamp[self.unique_idx]
            self.unique_idx += 1
            subset = self.test[self.test.timestamp == timesplit]

            # reset index to conform to how kagglegym works
            target = subset.loc[:, ['id', 'y']].reset_index(drop=True)
            self.temp_test_y = target['y']

            # set the prediction column to zero
            target.loc[:, 'y'] = 0

            # column bound change on the subset
            # reset index to conform to how kagglegym works
            features = subset.iloc[:, 0:110].reset_index(drop=True)
            observation = Observation(self.train, target, features)
        return observation, reward, done, info

    def __str__(self):
        return "Environment()"

def make(dataset=[]):
    if dataset is None:
        return Environment()
    return Environment(dataset)

class Observation(object):
    def __init__(self, train, target, features):
        self.train = train
        self.target = target
        self.features = features

def r_score(y_true, y_pred, sample_weight=None, multioutput=None):
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)
    r = (np.sign(r2)*np.sqrt(np.abs(r2)))
    if r <= -1:
        return -1
    else:
        return r
