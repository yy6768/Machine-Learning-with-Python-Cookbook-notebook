import pandas as pd

url = 'https://www.gairuo.com/file/data/dataset/GDP-China.csv'
df = pd.read_csv(url)
# show first two rows
print(df.head(2))  # also try tail(2) for last two rows

# show dimensions
print("Dimensions: {}".format(df.shape))

# show statistics
print(df.describe())
