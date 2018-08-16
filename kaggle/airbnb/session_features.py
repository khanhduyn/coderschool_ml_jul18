import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction import DictVectorizer
def make_sessions_features(data, df_sessions):
    # Drop row with nan values from the "user_id" column as they're useless
    df_sessions = df_sessions.dropna(subset=["user_id"])

    # print df_sessions

    # Frequency of devices - by user
    device_freq = df_sessions.groupby('user_id').device_type.value_counts()
    
    # Frequency of actions taken - by user
    action_freq = df_sessions.groupby('user_id').action.value_counts()

    # Total list of users
    users = data.id.values
    def feature_dict(df):
        f_dict = dict(list(df.groupby(level='user_id')))
        res = {}
        for k, v in f_dict.items():
            v.index = v.index.droplevel('user_id')
            res[k] = v.to_dict()
        return res

    # Make a dictionary with the frequencies { 'user_id' : {"IPhone": 2, "Windows": 1}}
    action_dict = feature_dict(action_freq)
    device_dict = feature_dict(device_freq)

    # Transform to a list of dictionaries
    action_rows = [action_dict.get(k, {}) for k in users]
    device_rows = [device_dict.get(k, {}) for k in users]

    device_transf = DictVectorizer()
    tf = device_transf.fit_transform(device_rows)

    action_transf = DictVectorizer()
    tf2 = action_transf.fit_transform(action_rows)

    # Concatenate the two datasets
    # Those are row vectors with the frequencies of both device and actions [0, 0, 0, 2, 0, 1, ...]
    features = sp.hstack([tf, tf2])

    # We create a dataframe with the new features and we write it to disk
    df_sess_features = pd.DataFrame(features.todense())
    
    df_sess_features['id'] = users

    #left joining data and sessions on user_id
    final = pd.merge(data, df_sess_features, how='left', left_on='id', right_on='id')
    final.iloc[:, final.columns != 'age_bucket'].fillna(-1, inplace=True)

    # Using inplace because I have 8GB of RAM
    # final.ix[:, final.columns != 'age_bucket'] = final.ix[:, final.columns != 'age_bucket'].fillna(-1)

    final.drop(['id'], axis=1, inplace=True)
    return final
