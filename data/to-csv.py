import pandas as pd
data = pd.io.stata.read_stata('Baylor Religion Survey, Wave IV (2014).DTA')
data.to_csv('baylor-2014.csv')
