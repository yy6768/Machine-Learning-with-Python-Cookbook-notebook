import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

# Group rows, apply function to groups
print(dataframe.groupby('Sex').apply(lambda x: x.count()))