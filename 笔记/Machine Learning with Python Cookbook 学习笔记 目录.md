# Machine Learning with Python Cookbook 学习笔记 目录

## 前言

- 本笔记是针对人工智能典型算法的课程中Machine Learning with Python Cookbook的学习笔记
- 学习的实战代码都放在代码压缩包中
- 实战代码的运行环境是**python3.9   numpy 1.23.1 anaconda 4.12.0 **
- 代码仓库
  - Github:[yy6768/Machine-Learning-with-Python-Cookbook-notebook: 人工智能典型算法笔记 (github.com)](https://github.com/yy6768/Machine-Learning-with-Python-Cookbook-notebook)
  - Gitee:[yyorange/机器学习笔记代码仓库 (gitee.com)](https://gitee.com/yyorange/Machine-Learning)

- 预计11月前能够把所有笔记完整整理完毕
- 最新更新：7/22 第12章学习笔记[(97条消息) Machine Learning with Python Cookbook 学习笔记 第12章_五舍橘橘的博客-CSDN博客](https://blog.csdn.net/weixin_51083297/article/details/125941794?spm=1001.2014.3001.5502)

## 目录

### 第一部分——基础知识

#### 第1章 **Vectors, Matrices, and Arrays**:[(97条消息) Machine Learning with Python Cookbook 学习笔记 第1章_五舍橘橘的博客-CSDN博客](https://blog.csdn.net/weixin_51083297/article/details/125817027?spm=1001.2014.3001.5502)

主要讲述了：

- numpy的基本知识
- 对于数据结构的一些基本操作
- 线性代数操作



#### 第2章 Loading Data：[(97条消息) Machine Learning with Python Cookbook 学习笔记 第2章_五舍橘橘的博客-CSDN博客](https://blog.csdn.net/weixin_51083297/article/details/125829627?spm=1001.2014.3001.5502)

主要分为两部分：

- 从多个数据源获得数据（pandas库的基本知识）
- 通过工具生成数据（scikit-learn库的数据集部分）



#### 第3章 Data Wrangling（数据整理）：[(97条消息) Machine Learning with Python Cookbook 学习笔记 第3章_五舍橘橘的博客-CSDN博客](https://blog.csdn.net/weixin_51083297/article/details/125829788?spm=1001.2014.3001.5502)

主要讲述了：

- pandas的DataFrame



### 第二部分——数据预处理

#### 第4章 Handling Numerical Data：[(97条消息) Machine Learning with Python Cookbook 学习笔记 第4章_五舍橘橘的博客-CSDN博客](https://blog.csdn.net/weixin_51083297/article/details/125833408?spm=1001.2014.3001.5502)

主要讲述了如何处理数字类数据，包括：

- 数据标准化归一化等预处理方法
- 填充缺失值、扫描异常值、离散化等操作

#### 第5章 Handling Categorical Data：[(97条消息) Machine Learning with Python Cookbook 学习笔记 第5章_五舍橘橘的博客-CSDN博客](https://blog.csdn.net/weixin_51083297/article/details/125833474?spm=1001.2014.3001.5502)

主要讲述了如何预处理分类数据，包含：

- 编码（encode)
- 填充(impute)
- 处理不平衡的分类

#### 第6章 Handling Text：[(97条消息) Machine Learning with Python Cookbook 学习笔记 第6章_五舍橘橘的博客-CSDN博客](https://blog.csdn.net/weixin_51083297/article/details/125833559?spm=1001.2014.3001.5502)

主要讲述了如何预处理文本数据,包含了：

- 不同类型文本的清理
- ntlk自然语言处理库
- Porter算法
- 单词加权

#### 第7章 Handling Dates and Times：[(97条消息) Machine Learning with Python Cookbook 学习笔记 第7章_五舍橘橘的博客-CSDN博客](https://blog.csdn.net/weixin_51083297/article/details/125861659?spm=1001.2014.3001.5502)

主要讲述了如何处理时间和日期数据，包含：

- 不同格式日期的相互转换
- 日期编码
- 填充丢失日期、滞后等日期操作

#### 第8章 Handling Images[(97条消息) Machine Learning with Python Cookbook 学习笔记 第8章_五舍橘橘的博客-CSDN博客](https://blog.csdn.net/weixin_51083297/article/details/125875427?spm=1001.2014.3001.5502)

主要讲述了如何处理图像数据，包含：

- `opencv`库的基本操作
- 平滑图片、锐化图片、增强图片（直方图均衡化算法）
- 去除背景（Grabcut算法）
- 边缘检测（Canny算法）



### 第三部分——数据降维

#### 第9章 Dimensionality Reduction Using Feature Extraction： [(97条消息) Machine Learning with Python Cookbook 学习笔记 第9章_五舍橘橘的博客-CSDN博客](https://blog.csdn.net/weixin_51083297/article/details/125895300?spm=1001.2014.3001.5502)

主要讲述了使用特征提取(Feature Extraction)进行特征降维。主要包括：

- PCA（主成分分析）以及它的变种（Kernal PCA 非线性情况）
- LDA（线性判断降维算法）
- NMF（非负矩阵分解）
- TSVD（截断奇异值分解）（针对稀疏矩阵）



#### 第10章 Dimensionality Reduction Using Feature Selection：[(97条消息) Machine Learning with Python Cookbook 学习笔记 第10章_五舍橘橘的博客-CSDN博客](https://blog.csdn.net/weixin_51083297/article/details/125909496?csdn_share_tail={"type"%3A"blog"%2C"rType"%3A"article"%2C"rId"%3A"125909496"%2C"source"%3A"weixin_51083297"}&ctrtid=f37vM)

主要讲述了使用**特征选择**（Feature Selection）进行特征降维。主要包括：

- 设置阈值方差（移除方差较小的特征）
- 去除高度线性相关特征
- 移除无关变量（卡方统计量）
- **Recursively Eliminating Features**（RFE）





### 第四部分 模型评估与选择

#### 第11章 Model Evaluation： [(97条消息) Machine Learning with Python Cookbook 学习笔记 第11章_五舍橘橘的博客-CSDN博客](https://blog.csdn.net/weixin_51083297/article/details/125941709?csdn_share_tail={"type"%3A"blog"%2C"rType"%3A"article"%2C"rId"%3A"125941709"%2C"source"%3A"weixin_51083297"}&ctrtid=qCjFK)

主要讲述了不同的模型评估算法，来查看模型训练好坏的结果：

- **交叉验证模型（Model Evaluation）**
- 生成启发式模型（scikit-learn中 DummyXXXX模型）
- ROC曲线（评估分类器）
- 多分类器分类指标（macro，micro，weighted）
- 混淆矩阵
- MSE（评估线性模型）
- 轮廓系数（评估诸如聚类等无监督模型）
- learning curve（观测值数量对模型的影响）



#### 第12章  Model Selection[(97条消息) Machine Learning with Python Cookbook 学习笔记 第12章_五舍橘橘的博客-CSDN博客](https://blog.csdn.net/weixin_51083297/article/details/125941794?spm=1001.2014.3001.5502)

主要讲述了如何选择不同超参数，不同算法构成的模型：

- `GridSearchCV`：穷举法交叉检验
- `RandomizedSearchCV`:随机分布化交叉检验
- 多算法模型选择
- 预处理和模型选择
- 加速模型选择
- 嵌套交叉检验



____

施工中……

____



### 第五部分 学习算法

#### 第13章 Linear Regression

#### 第14章 Trees and Forests（决策树与随机森林）

#### 第15章 K-Nearest Neighbors（KNN算法）

#### 第16章 Logistic Regression（Sigmoid函数和逻辑回归）

#### 第17章 Support Vector Machines（支持向量机）

#### 第18章 Naive Bayes（朴素贝叶斯）

#### 第19章 Clustering（聚类）

#### 第20章  Neural Networks（神经网络）



### 第六部分 模型保存和导入

#### 第21章 Saving and Loading Trained Models 