# Load library
import pandas as pd
# Create URL
url = 'titanic.csv'
# Load data
dataframe = pd.read_csv(url)
# Select first row
print(dataframe.iloc[0])
print()
# Select three rows
print(dataframe.iloc[1:4])


# Set index
dataframe = dataframe.set_index(dataframe['Lname'])
print(dataframe.loc['Braund'])