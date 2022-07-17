## Chapter 5. Handling Categorical Data

- 本笔记是针对人工智能典型算法的课程中Machine Learning with Python Cookbook的学习笔记
- 学习的实战代码都放在代码压缩包中
- 实战代码的运行环境是python3.9 numpy 1.23.1

上一章：[(89条消息) Machine Learning with Python Cookbook 学习笔记 第4章_五舍橘橘的博客-CSDN博客](https://blog.csdn.net/weixin_51083297/article/details/125833408?csdn_share_tail={"type"%3A"blog"%2C"rType"%3A"article"%2C"rId"%3A"125833408"%2C"source"%3A"weixin_51083297"}&ctrtid=nhame)



代码笔记仓库（给颗星再走qaq）：[yy6768/Machine-Learning-with-Python-Cookbook-notebook: 人工智能典型算法笔记 (github.com)](https://github.com/yy6768/Machine-Learning-with-Python-Cookbook-notebook)

### 5.0 Introduction

- 不是根据数量而是根据某些质量来衡量对象通常很有用
- 分类信息通常在数据中表示为向量或字符串列（例如，“Maine”、“Texas”、“Delaware”）。问题是大多数机器学习算法要求输入是数值。
- 我们的目标是进行转换，以正确传达类别中的信息（序数、类别之间的相对间隔等）。在本章中，我们将介绍进行这种转换的技术，以及克服处理分类数据时经常遇到的其他挑战。



### 5.1 Encoding Nominal Categorical Feature

对类别特征编码

encodingCategory.py

```python
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

feature = np.array([
    ["Texas"],
    ["California"],
    ["Texas"],
    ["Delaware"],
    ["Texas"]
])

# 创建 one-hot encoder
one_hot = LabelBinarizer()

# one-hot 对feature 编码
print(one_hot.fit_transform(feature))

# 输出各个类别
print(one_hot.classes_)

# 对编码结果解码
print(one_hot.inverse_transform(one_hot.transform(feature)))

# 方法2 使用pandas
# Import library
import pandas as pd

# 从feature中创建dummy编码
print(pd.get_dummies(feature[:, 0]))


multiclass_feature = [("Texas", "Florida"),
                      ("California", "Alabama"),
                      ("Texas", "Florida"),
                      ("Delware", "Florida"),
                      ("Texas", "Alabama")]

# Create 多类别编码
one_hot_multiclass = MultiLabelBinarizer()
# One-hot encode multiclass feature
print(one_hot_multiclass.fit_transform(multiclass_feature))

# 查看所有类别
print(one_hot_multiclass.classes_)
```

结果：

```
[[0 0 1]
 [1 0 0]
 [0 0 1]
 [0 1 0]
 [0 0 1]]
 
['California' 'Delaware' 'Texas']

['Texas' 'California' 'Texas' 'Delaware' 'Texas']

   California  Delaware  Texas
0           0         0      1
1           1         0      0
2           0         0      1
3           0         1      0
4           0         0      1

[[0 0 0 1 1]
 [1 1 0 0 0]
 [0 0 0 1 1]
 [0 0 1 1 0]
 [1 0 0 0 1]]
 
 ['Alabama' 'California' 'Delware' 'Florida' 'Texas']

```



#### Discussion

- encode的策略并不是为每个类分配一个数值
- encode的正确策略是为每一个类创建一个2元特征；通常这个特征被称为独热编码（one-hot)
  - 查阅资料：[(88条消息) 详细详解One Hot编码-附代码_chary8088的博客-CSDN博客_onehot解码代码](https://blog.csdn.net/chary8088/article/details/79032223
  - One-Hot编码，又称为一位有效编码，主要是采用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候只有一位有效。
  - One-Hot不适合原本有序的序列，因为会破坏其原本的顺序



### 5.2 Encoding Ordinal Categorical Features

编码有序序列

有效的方式是使用pandas DataFrame中的replace函数

encodeOrdinal.py

```python
# Load library
import pandas as pd

# Create features
dataframe = pd.DataFrame({"Score": ["Low", "Low", "Medium", "Medium", "High"]})
# 创建mapper
scale_mapper = {"Low": 1,
                "Medium": 2,
                "High": 3}
# 替换
print(dataframe["Score"].replace(scale_mapper))
```

结果

```
0    1
1    1
2    2
3    2
4    3
Name: Score, dtype: int64
```



#### Discussion

- 顺序序列是我们较为常见的序列，当需要对这些序列进行编码的时候，最好用数值去替换表示顺序的属性
- 当需要表示不同顺序序列之间的距离（程度）时，可以在mapper中凸显出来，例如

​	

```python
scale_mapper = {"Low":1,
				"Medium":2,
				"Barely More Than Medium": 2.1,	# 表示和medium类很贴近
				"High":3}
dataframe["Score"].replace(scale_mapper)
```





### 5.3 Encoding Dictionaries of Features

对字典类型的特征进行encode，通常使用`DictVectorizer`

dictVectorizerExample.py

```python
# Import library
from sklearn.feature_extraction import DictVectorizer
# Create dictionary
data_dict = [{"Red": 2, "Blue": 4},
             {"Red": 4, "Blue": 3},
             {"Red": 1, "Yellow": 2},
             {"Red": 2, "Yellow": 2}]
# 创建 dictionary vectorizer
dictvectorizer = DictVectorizer(sparse=False)
# 默认是生成一个产生稀疏矩阵的dicvectorizer2
dictvectorizer2 = DictVectorizer()
# 把字典转换为特征
features = dictvectorizer.fit_transform(data_dict)
features2 = dictvectorizer2.fit_transform(data_dict)
# 查看 feature matrix
print(features)
print(features2)

# 查看特征的名字 dictvectorizer.get_feature_names即将被遗弃
feature_names = dictvectorizer.get_feature_names_out()
# 查看
print(feature_names)

# 还可以使用pandas
import pandas as pd
# 通过列来表明不同的特征
print(pd.DataFrame(features, columns=feature_names))
```



结果：

```
[[4. 2. 0.]
 [3. 4. 0.]
 [0. 1. 2.]
 [0. 2. 2.]]
  (0, 0)	4.0
  (0, 1)	2.0
  (1, 0)	3.0
  (1, 1)	4.0
  (2, 1)	1.0
  (2, 2)	2.0
  (3, 1)	2.0
  (3, 2)	2.0
['Blue' 'Red' 'Yellow']
   Blue  Red  Yellow
0   4.0  2.0     0.0
1   3.0  4.0     0.0
2   0.0  1.0     2.0
3   0.0  2.0     2.0
```



#### Discussion

- 字典是一个比较常用的数据结构，特别时在自然语言处理中，但是我们最后都需要转换成矩阵的格式
- 字典类型的矩阵非常庞大，如何处理这样的矩阵也是我们的目标（可以使用稀疏矩阵）

- dictvectorizer.get_feature_names即将被遗弃,需要替换成dictvectorizer.get_feature_names_out()



### 5.4 Imputing Missing Class Values

填充丢失的类别值

最常用的方法仍然是KNN

imputing.py

```python
# load libraries
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[0, 2.10, 1.45],
            [1, 1.18, 1.33],
            [0, 1.22, 1.27],
            [1, -0.21, -1.19]])

X_with_nan = np.array([[np.nan, 0.87, 1.31],
                      [np.nan, -0.67, -0.22]])

# 训练 KNN learner
clf = KNeighborsClassifier(3, weights='distance')
trained_model = clf.fit(X[:,1:], X[:, 0])

# 预测丢失值
imputed_values = trained_model.predict(X_with_nan[:, 1:])

# 连接丢失的列和他们原来其他列
X_with_imputed = np.hstack((imputed_values.reshape(-1, 1), X_with_nan[:, 1:]))

# 连接原来的矩阵和有丢失列的矩阵
print(np.vstack((X_with_imputed, X)))

# 方法2 用最常用的值去替换缺失值
# impute已经被替代
from sklearn.impute import SimpleImputer
# Join the two feature matrices
X_complete = np.vstack((X_with_nan, X))
imputer = SimpleImputer(strategy='most_frequent')
print(imputer.fit_transform(X_complete))
```



结果

```
[[ 0.    0.87  1.31]
 [ 1.   -0.67 -0.22]
 [ 0.    2.1   1.45]
 [ 1.    1.18  1.33]
 [ 0.    1.22  1.27]
 [ 1.   -0.21 -1.19]]
 
 [[ 0.    0.87  1.31]
 [ 0.   -0.67 -0.22]
 [ 0.    2.1   1.45]
 [ 1.    1.18  1.33]
 [ 0.    1.22  1.27]
 [ 1.   -0.21 -1.19]]
```



#### Discussion

- KNN是最常用的方法，测量值比较精准，但是在大数据面前表现比较差
- 用最常见的特征值没有KNN准确，但是它在大数据情况时可拓展性比KNN好，而且没有KNN那么复杂



### 5.5 Handling Imbalanced Classes

我们如果遇到极度不平衡的类的目标向量，如何解决

- 作者认为首先应该去搜集更多的数据

- 如果搜集的数据已经足够多的话，应该更改评估模型的指标

- 再然后如果还不能解决就要使用内置的类权重参数、下采样或者上采样

- 评估模型指标在之后的章节会介绍，这一节关注类权重参数、下采样和上采样的方法

  handleImbalance.py

```python
# Load libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
# 加载莺尾花数据集
iris = load_iris()
# 创建特征矩阵
features = iris.data
# 创建目标向量
target = iris.target
# 移除前40个元素
features = features[40:,:]
target = target[40:]
# 将类0和非类0的observation分为两类
target = np.where((target == 0), 0, 1)
# 打印
print(target)

# 第一种处理，提供类权重

# 创建权重
weights = {0: .9, 1: 0.1}
# 根据权重创建随机森岭
print(RandomForestClassifier(class_weight=weights))

# Train a random forest with balanced class weights
print(RandomForestClassifier(class_weight="balanced"))

# 处理方式2 进行下采样
# Indicies of each class' observations
i_class0 = np.where(target == 0)[0]
i_class1 = np.where(target == 1)[0]
# 读取类的大小
n_class0 = len(i_class0)
n_class1 = len(i_class1)
# For every observation of class 0, randomly sample
# 随机取样，使得class1 Observation数量和class0一样多
i_class1_downsampled = np.random.choice(i_class1, size=n_class0, replace=False)
# 连接
# 下采样
print(np.hstack((target[i_class0], target[i_class1_downsampled])))
print(np.vstack((features[i_class0,:], features[i_class1_downsampled,:]))[0:5])

# 处理方式3 上采样
# 对于class0来说,随机创建样本直至数量和class1一样
i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace=True)
# 目标向量
print(np.concatenate((target[i_class0_upsampled], target[i_class1])))
# 特征矩阵
print(np.vstack((features[i_class0_upsampled,:], features[i_class1,:]))[0:5])
```



```python
[0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]


RandomForestClassifier(class_weight={0: 0.9, 1: 0.1})
RandomForestClassifier(class_weight='balanced')


[0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1]
[[5.  3.5 1.3 0.3]
 [4.5 2.3 1.3 0.3]
 [4.4 3.2 1.3 0.2]
 [5.  3.5 1.6 0.6]
 [5.1 3.8 1.9 0.4]]


[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
[[5.1 3.8 1.9 0.4]
 [5.1 3.8 1.6 0.2]
 [5.3 3.7 1.5 0.2]
 [5.3 3.7 1.5 0.2]
 [4.5 2.3 1.3 0.3]]


```



#### Discussion

- 在现实世界中，不平衡的课程无处不在——大多数访问者不会点击购买按钮，幸好许多类型的癌症很少见。出于这个原因，处理不平衡类是机器学习中的常见活动。
- 策略优先级从高到低分别是
  - 收集更多的观察结果——尤其是少数群体的观察结果。
  - 使用更适合不平衡类的模型评估指标。我们在后面的章节中讨论的一些更好的指标是混淆矩阵、精度、召回、F1 分数和 ROC 曲线。
  - 使用包含在某些模型实现中的类权重参数。这使我们可以让算法针对不平衡的类进行调整。幸运的是，许多 scikit-learn 分类器都有一个 class_weight 参数，使其成为一个不错的选择。
  - 下采样和上采样。通常我们应该尝试两者来查看哪个产生更好的结果。

