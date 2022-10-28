## Chapter 13 Linear Regression

### 前言

- 本笔记是针对人工智能典型算法的课程中Machine Learning with Python Cookbook的学习笔记
- 学习的实战代码都放在代码压缩包中
- 实战代码的运行环境是**python3.9   numpy 1.23.1** **anaconda 4.12.0 **
- 上一章：[(97条消息) Machine Learning with Python Cookbook 学习笔记 第11章_五舍橘橘的博客-CSDN博客](https://blog.csdn.net/weixin_51083297/article/details/125941709?csdn_share_tail={"type"%3A"blog"%2C"rType"%3A"article"%2C"rId"%3A"125941709"%2C"source"%3A"weixin_51083297"}&ctrtid=NOBZ2)
- 代码仓库
  - Github:[yy6768/Machine-Learning-with-Python-Cookbook-notebook: 人工智能典型算法笔记 (github.com)](https://github.com/yy6768/Machine-Learning-with-Python-Cookbook-notebook)
  - Gitee: [机器学习笔记代码仓库: 主要存放一些笔记和代码 第一期主要围绕Machine Learning with Python Cookbook的学习笔记和代码 (gitee.com)](https://gitee.com/yyorange/Machine-Learning)



## 13.0 Introduction

- 线性回归是我们工具包中最简单的监督学习算法之一。 如果你曾经在大学里上过统计学入门课程，那么你所涵盖的最后一个主题可能是线性回归。
- 事实上，当目标向量是定量值（例如房价、年龄）时，线性回归及其扩展仍然是一种常见且有用的预测方法。 在本章中，我们将介绍用于创建性能良好的预测模型的各种线性回归方法（和一些扩展）。



## 13.1 Fitting a Line

- 问题：特征和目标向量之间的线性关系的模型

- `scikit-learn`库

- ```python
  # 加载库
  from sklearn.linear_model import LinearRegression
  from sklearn.datasets import fetch_california_housing
  # 加载加州房价
  housing = fetch_california_housing()
  features = housing.data[:,0:2]
  target = housing.target
  # 创建线性回归模型
  regression = LinearRegression()
  # 在线性模型中获取均值和方差
  model = regression.fit(features, target)
  
  # 查看差值
  print(model.intercept_)
  # 查看系数集合
  print(model.coef_)
  # 进行预测
  print(model.predict(features)[0])
  
  ```

  

### Disscussion

线性回归假设特征和目标向量之间的关系是近似线性的。 也就是说，特征对目标向量的影响（也称为系数、权重或参数）是恒定的。 在我们的解决方案中，为了解释起见，我们只使用两个特征训练了我们的模型。 这意味着我们的线性模型将是：
$$
\hat y =\hat \beta_0 +\hat \beta_1 x_1 + \hat \beta_2 x_2 + \epsilon
$$

- 其中 ŷ 是我们的目标，xi 是单个特征的数据，是通过拟合模型确定的系数，$\epsilon$是误差。 拟合模型后，我们可以查看每个参数的值。偏差或截距，可以使用intercept_ 查看

- $\hat \beta_i$可以通过_coef查看

  ```python
  # 查看差值
  print(model.intercept_)
  # 查看系数集合
  print(model.coef_)
  # 进行预测
  print(model.predict(features)[0])
  
  ```

  

- 优点：可解释性

### 关于scikit-learn的LinearRegression()的原理

1、模型公式：$\hat y = X\omega$

2、损失函数：$\Sigma^m_{i=1}(y_i-\hat y_i)^2$(SSE)

3、模型目标：$min_{arg\ \omega} || y- X\omega||^2$

4、矩阵求导：$\frac{\partial RSS}{\partial \omega}\\ =\frac{\partial(y-X\omega)^T(y-X\omega)}{\partial \omega} \\ =\frac{\part (y^T-\omega ^TX^T)(y-X\omega)}{\part \omega} \\=\frac{\part(y^Ty-y^TX\omega -\omega^TX^Ty+\omega^TX^TX\omega)}{\part \omega}\\=0	- X^Ty-X^Ty +2X^TX\omega\\= 2(X^TX\omega -X^Ty) = 0$

ps.在求导处使用了如下规则：

- $\frac{\part a}{\part A} =0$
- $\frac{\part C^TBA}{\part A} =B^TC$
- $\frac{\part A^TB^TC}{\part A} = B^TC$
- $\frac{\part A^TB^TBA}{\part A} = (B+B^T)A$ 

5、 解方程：

$X^TX\omega -X^Ty = 0 => \omega = (X^TX)^{-1}X^Ty$

前提是$X^TX$可逆

## 13.2 Handling Interactive Effects

- 问题：某个特征对其他特征有依赖
- 创建一个interaction term（交互项）

```python
# 加载库
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures

# 加载加州房价
housing = fetch_california_housing()
features = housing.data[:, 0:2]
target = housing.target

# 创建交互项
interaction = PolynomialFeatures(
    degree=3, include_bias=False, interaction_only=True)
features_interaction = interaction.fit_transform(features)
# 创建线性回归模型
regression = LinearRegression()
# Fit the linear regression
model = regression.fit(features_interaction, target)

```



### Discussion

- 某些特征存在数据相关性，比如咖啡的甜度的因素中，搅拌和糖的量应该是相互依赖的
- 解决方案：创建一个包含交互特征的相应值的乘积的新特征来解释交互效应

$\hat y = \hat \beta_0 + \hat \beta_1x_1 + \hat\beta_2x_2+\hat\beta_3x_1x_2+\epsilon$

在这里$\hat\beta_3x_1x_2$就是新添加的交互项

- 我们一般用两个特征相乘来表达交叉项

- PolynomialFeatures()可以创建交叉项

  三个重要参数：

  - interaction_only:只返回交叉项
  - include_bias:是否在交叉项里计算bias
  - degree：交叉项所拥有的最大特征数



## 13.3  Fitting a Nonlinear Relationship

- 拟合一个非线性关系
- 创建一个多项式回归
- noLinearRegression

```python
# 加载库
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import fetch_california_housing
# 加载加州房价
housing = fetch_california_housing()
features = housing.data[:,0:2]
target = housing.target
# 多项式 x^2 and x^3
polynomial = PolynomialFeatures(degree=3, include_bias=False)
features_polynomial = polynomial.fit_transform(features)
# 创建线性关系
regression = LinearRegression()
# 拟合线性关系
model = regression.fit(features_polynomial, target)

```



### Discussion

- $\hat y = \hat \beta_0 + \hat \beta_1x_1 + \hat\beta_2x_2^2+\hat\beta_3x_3^3+……+\hat\beta_dx_i^d+\epsilon$
- d是多项式的阶，
- 高阶的x实际上可以认为是新的特征
- 两个重要函数：
  - degree多项式的阶
  - include_bias:是否引入偏移



### 13.4 Reducing Variance with Regularization

- 减少线性模型的方差

- Use a learning algorithm that includes a shrinkage penalty (also called regularization) like ridge regression and lasso regression（使用包含收缩惩罚（正则化）的算法，如`岭回归`和`套索回归`）

- shrinkage_penalty.py

  ```python
  # Load libraries
  from sklearn.linear_model import Ridge
  from sklearn.datasets import fetch_california_housing
  from sklearn.preprocessing import StandardScaler
  # 加利福尼亚房价数据集
  housing = fetch_california_housing()
  features = housing.data
  target = housing.target
  # 标准化
  scaler = StandardScaler()
  features_standardized = scaler.fit_transform(features)
  # 岭回归
  regression = Ridge(alpha=0.5)
  # 适用模型
  model = regression.fit(features_standardized, target)
  
  
  
  ```

#### Discussion

- 传统线性模型的Loss函数：残差平方和：$RSS=\Sigma_{i=1}^{n}(y_i-\hat y_i)^2$

- 收缩惩罚（正则化）学习模型的损失函数与RSS相似，但是希望线性模型的系数可以尽可能的少。收缩惩罚的名称表示希望将模型缩小

- 有两种常用的正则化线性模型：岭回归和lasso回归。它们唯一的不同就是正则项的不同：

  - Ridge regression的损失函数：$RSS + \alpha\Sigma_{j=1}^p \hat\beta_j^2$

    $\hat\beta_j$是线性模型每一个变量前面的系数

  - Lasso regression的损失函数：$\frac{1}{2n}RSS+\alpha\Sigma_{j=1}^p|\hat\beta_j|$

    n是observation样本的数量	

    $\hat\beta_j$是线性模型每一个变量前面的系数

- 使用什么模型呢？

  - 岭回归往往比lasso回归作出更好的预测
  - lasso回归比岭回归更具有解释性
  - 可以使用两者的结合

- 对于超参数$\alpha$,决定了正则项的权重，在scikit包中，作为alpha参数输入模型中

- 我们可以使用scikit-learn中的`RigerCV`使用交叉检验法把$\alpha$作为参数进行训练

​		

```python

# Load library
from sklearn.linear_model import RidgeCV
# 创建一系列的alpha参数，使用交叉检验法进行比较
regr_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])
# 拟合模型
model_cv = regr_cv.fit(features_standardized, target)
# 查看结果
print(model_cv.coef_)
print(model_cv.alpha_)

```



- 最后：因为系数的值由实际特征的比例决定，所以最好在使用正则化之前进行一定的标准化



### 13.5 Reducing Features with Lasso Regression

- You want to simplify your linear regression model by reducing the number of features.
- 使用lasso模型：

lasso_regression.py

```python
# 加载python库
from sklearn.linear_model import Lasso
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
# 加利福尼亚房价数据集
housing = fetch_california_housing()
features = housing.data
target = housing.target
# 标准化数据
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
# Create lasso regression with alpha value
regression = Lasso(alpha=0.5)
# Fit the linear regression
model = regression.fit(features_standardized, target)

```

#### Discussion

- lasso回归模型是可以直接将系数降为0的

  ```python
  # 0.5的系数
  print(model.coef_)
  
  # 设置一个新模型为alpha=10
  regression_10 = Lasso(alpha=10)
  model = regression_10.fit(features_standardized, target)
  print(model.coef_)
  ```

  ![image-20220923135657343](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220923135657343.png)

- 这种效果的实际好处是，它意味着我们可以在特征矩阵中包含100个特征，然后通过调整拉索的α超参数，生成一个只使用10个（例如）最重要特征的模型。这使我们能够减少差异，同时提高模型的可解释性（因为较少的特征更容易解释）





