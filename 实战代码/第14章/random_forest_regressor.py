# Load libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
# Load data with only two features
boston = datasets.load_boston()
features = boston.data[:,0:2]
target = boston.target
# Create random forest classifier object
randomforest = RandomForestRegressor(random_state=0, n_jobs=-1)
# Train model
model = randomforest.fit(features, target)
