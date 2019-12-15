# Written mostly by Justin Wang for a hackathon we did together (https://github.com/timot3/PYGHACK).
# I modified some lines to make it more relevant to the current problem.
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sb

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


print("Loading in Data")
file = get_table(DATA_FILE)
labels, features = extract_data(file, NUM_ROWS)

clf = RandomForestClassifier(n_estimators=25, )

clf.fit(features[:int(len(features) * TRAIN_VALIDATION_SPLIT)], labels[:int(len(labels) * TRAIN_VALIDATION_SPLIT)])


acc = 100 * clf.score(features[int(len(features) * TRAIN_VALIDATION_SPLIT):],
                      labels[int(len(labels) * TRAIN_VALIDATION_SPLIT):])

print("Forest accuracy: \n" + str(acc))
print(*(clf.feature_importances_ * 100))

# categorical_classifier = bc.Classifier(features, labels)
# categorical_classifier.train()  # outputs training accuracy, also trains classifier


# plt.bar(reltrad.keys(), reltrad.values(), label="Count")
'''
plt.ylabel('Count')
plt.xlabel('Religion')
plt.title("Frequency of Religion in Baylor 2014 Survey")
plt.xticks(list(reltrad.keys()))
plt.legend(bbox_to_anchor=(1, 1), loc="upper right", borderaxespad=0.)
pylab.xticks(rotation=-60)

plt.show()
'''
rows = list(file.columns)

rows = [w.replace('_', ' ') for w in rows]

print(rows)
importance = clf.feature_importances_
importance = pd.DataFrame(importance, index=file.columns[:-1],
                          columns=["Importance"])

importance["Std"] = np.std([tree.feature_importances_
                            for tree in clf.estimators_], axis=0)

y = importance.ix[:, 0]
yerr = importance.ix[:, 1]

plt.bar(rows[:-1], y, yerr=yerr, align="center")
pylab.xticks(rotation=-60)
plt.title("Feature Importance When Predicting Social Status")
plt.ylabel("Importance Percentage")
plt.show()


