import pandas as pd

data = pd.io.stata.read_stata('GSS2018.dta', convert_categoricals=False)
data.to_csv('GSS2018.csv')
