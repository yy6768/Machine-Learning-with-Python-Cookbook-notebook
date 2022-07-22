import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

# 去除重复的行
print(dataframe.drop_duplicates().head(2))

#检查行数
print("Number Of Rows In The Original DataFrame:", len(dataframe))
print("Number Of Rows After Deduping:", len(dataframe.drop_duplicates()))

dataframe.drop_duplicates(subset=['Sex'])

# keep
# Drop duplicates
dataframe.drop_duplicates(subset=['Sex'], keep='last')