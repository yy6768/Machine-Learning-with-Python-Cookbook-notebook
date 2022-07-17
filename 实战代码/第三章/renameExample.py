import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

# 替换一列
print(dataframe.rename(columns={'Pclass': 'Passenger Class'}).head(2))


# 同时替换两列
print(dataframe.rename(columns={'Pclass': 'Passenger Class','Lname': 'Last Name'}).head(2))



