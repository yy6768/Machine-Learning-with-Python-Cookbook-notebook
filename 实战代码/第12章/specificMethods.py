# Load libraries
from sklearn import linear_model, datasets
# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create cross-validated logistic regression
logit = linear_model.LogisticRegressionCV(Cs=100)
# Train model
print(logit.fit(features, target))