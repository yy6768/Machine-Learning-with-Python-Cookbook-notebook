import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

## Select missing values, show two rows
print(dataframe[dataframe['Age'].isnull()].head(2))
