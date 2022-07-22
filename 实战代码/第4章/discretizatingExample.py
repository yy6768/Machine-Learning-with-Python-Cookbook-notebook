# Load libraries
import numpy as np
from sklearn.preprocessing import Binarizer

# Create feature
age = np.array([[6],
                [12],
                [20],
                [36],
                [65]])
# Create binarizer
binarizer = Binarizer(threshold=18)

# Transform feature
print(binarizer.fit_transform(age))

# bin feature
print(np.digitize(age, bins=[20,30,64]))

# Bin feature
print(np.digitize(age, bins=[20,30,64], right=True))