# load library
import pandas as pd

# create url
url = 'https://raw.githubusercontent.com/domoritz/maps/master/data/iris.json'

# load data
df = pd.read_json(url, orient="columns")

# view first two rows
print(df.head(2))