import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import pandas as pd
import collections
from collections import Counter
import seaborn as sns
from math import isnan

import operator

# voted_for_in_2012_election,political_spectrum,social_class,gender,citizenship,living_location,sexuality,income,education,reltrad
data = pd.read_csv('../data/baylor-2014.csv')

voted_for_in_2012_election = dict(Counter(data['voted_for_in_2012_election']))
political_spectrum = dict(Counter(data['political_spectrum']))
social_class = dict(Counter(data['social_class']))
gender = dict(Counter(data['gender']))
# citizenship = dict(Counter(data['citizenship']))
living_location = dict(Counter(data['living_location'].fillna('no-response')))
# sexuality = dict(Counter(data['sexuality']))
# income = dict(Counter(data['income']))
education = dict(Counter(data['education']))
reltrad = dict(Counter(data['reltrad']))

print(data['living_location'].fillna('no-response').value_counts(normalize=True))
features = list(data.columns.values)
print(features)

sorted_living_location = sorted(living_location.items(), key=lambda kv: kv[1], reverse=True)
living_location = dict(collections.OrderedDict(sorted_living_location))

print(living_location)


plt.bar(living_location.keys(), living_location.values(), label="Count")


plt.ylabel('Count')
plt.xlabel('Religion')
plt.title("Frequency of Living Location in Baylor 2014 Survey")
plt.xticks(list(living_location.keys()))
plt.legend(bbox_to_anchor=(1, 1), loc="upper right", borderaxespad=0.)
pylab.xticks(rotation=-60)

plt.show()


plt.title("Distribution of Religion in the United States")
plt.pie(reltrad.values(), labels=reltrad.keys(), autopct='%1.1f%%')
plt.show()


freq = data.groupby(['reltrad', 'social_class'])
print(freq.size())
heatmap = pd.DataFrame({'count': freq.size()}).unstack().transpose()
print(heatmap)
heatmap = sns.heatmap(heatmap / heatmap.sum(), cmap='plasma', annot=True, fmt=".2f")
plt.title("Heatmap of Social Class Given Religion")
plt.show()

