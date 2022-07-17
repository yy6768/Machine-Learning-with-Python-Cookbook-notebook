import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

# 替换female 为male
print(dataframe['Sex'].replace("female", "Woman").head(2))

# 替换 "female" and "male 为 "Woman" and "Man"
print(dataframe['Sex'].replace(["female", "male"], ["Woman", "Man"]).head(5))