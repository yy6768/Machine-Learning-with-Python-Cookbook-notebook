import numpy as np
from sklearn.preprocessing import Normalizer

# create feature matrix
features = np.array([
    [0.5, 0.5],
    [1.1, 3.4],
    [1.5, 20.2],
    [1.63, 34.4],
    [10.9, 3.3]
])

# create normalizer
normalizer = Normalizer(norm="l2")

# normalize matrix
print(normalizer.transform(features))

# transform feature matrix
features_l1_norm = Normalizer(norm="l1").transform(features)
print("Sum of the first observation's values: {}".format(features_l1_norm[0,0] + features_l1_norm[0,1]))
print(features_l1_norm)