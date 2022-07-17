import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

# 大写函数
def uppercase(x):
    return x.upper()
# apply作用后的前两行
print(dataframe['Name'].apply(uppercase)[0:2])