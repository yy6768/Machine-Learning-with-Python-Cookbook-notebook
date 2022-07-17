# Load libraries
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Create feature matrix
features = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])
# Create PolynomialFeatures object
polynomial_interaction = PolynomialFeatures(degree=2, include_bias=False)
# 创建多项式特征
print(polynomial_interaction.fit_transform(features))

# 只出现交叉项
interaction = PolynomialFeatures(degree=2,
                                 interaction_only=True, include_bias=False)
print(interaction.fit_transform(features))