
import pandas as pd
# Create DataFrame
dataframe = pd.DataFrame()
# 用字典的方式添加新的一行
dataframe['Name'] = ['Jacky Jackson', 'Steven Stevenson']
dataframe['Age'] = [38, 25]
dataframe['Driver'] = [True, False]
# 展示dataframe
print(dataframe)

# 创建新的一行
new_person = pd.Series(['Molly Mooney', 40, True], index=['Name', 'Age', 'Driver'])
# 拼接一行
dataframe = dataframe.append(new_person,ignore_index=True)
print(dataframe)