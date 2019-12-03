# Written mostly by Justin Wang for a hackathon we did together (https://github.com/timot3/PYGHACK).
# I modified some lines to make it more relevant to the current problem.
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

from torch import as_tensor

'''
def extract_data_from_file(file, num_rows):
    list_data = []
    with open(file, 'r') as fin:
        line = fin.readline()
        counter = 1
        while line:
            try:
                temp = line.split(',')[2:]
                #                temp[-1] = temp[-1][:-1]
                list_data.append(list(map(float, temp)))  # convert string to float
            except:
                counter = counter
            #            if counter % 1000 == 0:
            #                print("%.2f%% Finished" % (counter / num_rows * 100))

            line = fin.readline()
            counter += 1
    raw_data = as_tensor(list_data).float()
    print(raw_data.shape)
    labels = raw_data[:, -1]
    features = raw_data[:, :-1]

    return labels, features
'''


def convert_to_float(arr):
    lb = LabelEncoder()
    for x in arr:
        arr[x] = lb.fit_transform(list(arr[x]))
    return arr


def convert_labels_to_float(arr):
    lb = LabelEncoder()
    arr = lb.fit_transform(list(arr))
    return arr


def extract_data(FILE, num_rows):
    list_data = []
    file = pd.read_csv(FILE)
    lb = LabelEncoder()
    # print(file.iloc[:, :-1])
    data_features = convert_to_float(file.iloc[:, :-1])  # all but last column

    data_labels = convert_labels_to_float(file.iloc[:, -1])

    # data_labels = lb.fit_transform(data_labels)

    '''labels = raw_data[:, -1]
    features = raw_data[:, :-1]'''
    return data_labels, data_features

    '''
    line = fin.readline()
            counter = 1
            while line:
                try:
                    temp = line.split(',')[2:]
                    #                temp[-1] = temp[-1][:-1]
                    list_data.append(list(map(float, temp)))  # convert string to float
                except:
                    counter = counter
                #            if counter % 1000 == 0:
                #                print("%.2f%% Finished" % (counter / num_rows * 100))
    
                line = fin.readline()
                counter += 1
        raw_data = as_tensor(list_data).float()
        print(raw_data.shape)
        labels = raw_data[:, -1]
        features = raw_data[:, :-1]
        
        return labels, features
    '''


def normalize(x):
    return (x - x.mean()) / x.std()


DATA_FILE = '../data/baylor-2014.csv'
NUM_ROWS = 1393
TRAIN_VALIDATION_SPLIT = 0.95


# np.random.seed(3)  # to produce a similar output


def RMSE(x, y):
    acc = 0
    for i in range(len(x)):
        acc = acc + (x[i] - y[i]) ** 2
    return math.sqrt(acc / len(x))


print("Loading in Data")
labels, features = extract_data(DATA_FILE, NUM_ROWS)
# perm = np.random.permutation(len(labels))
# features = features[perm].numpy()
# labels = labels[perm].numpy()
#
# xtrain = features[:int(len(features) * TRAIN_VALIDATION_SPLIT), :]
# ytrain = labels[:int(len(features) * TRAIN_VALIDATION_SPLIT)]
#
# xval = features[int(len(features) * TRAIN_VALIDATION_SPLIT):, :]
# yval = labels[int(len(features) * TRAIN_VALIDATION_SPLIT):]

clf = RandomForestRegressor(n_estimators=50, )

# print("Initializing Training")
clf.fit(features, labels)

print(*(clf.feature_importances_ * 100))

# y_pred = clf.predict(xval)

# acc = RMSE(y_pred, yval)

# print("Model Accuracy: ", acc)