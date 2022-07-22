import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

# Print first two names uppercased
for name in dataframe['Name'][0:2]:
    print(name.upper())
# Show first two names uppercased
print([name.upper() for name in dataframe['Name'][0:2]])
