# Written mostly by Justin Wang for a hackathon we did together (https://github.com/timot3/PYGHACK).
# I modified some lines to make it more relevant to the current problem.
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

import numpy as np
import math
import pandas as pd

from ML import bayes_classifier as bc


def get_table(FILE):
    f = pd.read_csv(FILE)
    return f


def convert_to_float(arr):
    lb = LabelEncoder()
    for x in arr:
        arr[x] = lb.fit_transform(list(arr[x]))
    return arr


def convert_labels_to_float(arr):
    lb = LabelEncoder()
    arr = lb.fit_transform(list(arr))
    return arr


def extract_data(file, num_rows):
    lb = LabelEncoder()
    data_features = convert_to_float(file.iloc[:, :-1])  # all but last column

    data_labels = convert_labels_to_float(file.iloc[:, -1])

    return data_labels, data_features


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
file = get_table(DATA_FILE)
labels, features = extract_data(file, NUM_ROWS)
# perm = np.random.permutation(len(labels))
# features = features[perm].numpy()
# labels = labels[perm].numpy()
#
# xtrain = features[:int(len(features) * TRAIN_VALIDATION_SPLIT), :]
# ytrain = labels[:int(len(features) * TRAIN_VALIDATION_SPLIT)]
#
# xval = features[int(len(features) * TRAIN_VALIDATION_SPLIT):, :]

clf = RandomForestRegressor(n_estimators=50, )

clf.fit(features, labels)

print(*(clf.feature_importances_ * 100))

# y_pred = clf.predict(xval)

# acc = RMSE(y_pred, yval)

# print("Model Accuracy: ", acc)

eval_data = labels[int(len(features) * TRAIN_VALIDATION_SPLIT):]

categorical_classifier = bc.Classifier(features, labels)
categorical_classifier.train()
