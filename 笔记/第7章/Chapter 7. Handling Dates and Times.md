## Chapter 7. Handling Dates and Times

### 前言

- 本笔记是针对人工智能典型算法的课程中Machine Learning with Python Cookbook的学习笔记
- 学习的实战代码都放在代码压缩包中
- 实战代码的运行环境是**python3.9   numpy 1.23.1** **anaconda 4.12.0 **
- 上一章：[(92条消息) Machine Learning with Python Cookbook 学习笔记 第6章_五舍橘橘的博客-CSDN博客](https://blog.csdn.net/weixin_51083297/article/details/125833559?spm=1001.2014.3001.5501)
- 代码仓库
  - Github:[yy6768/Machine-Learning-with-Python-Cookbook-notebook: 人工智能典型算法笔记 (github.com)](https://github.com/yy6768/Machine-Learning-with-Python-Cookbook-notebook)
  - Gitee:[yyorange/机器学习笔记代码仓库 (gitee.com)](https://gitee.com/yyorange/Machine-Learning)

### 7.0 Introduction

- 日期和时间是机器学习中常见要处理的类型
-  `pandas` 库中的时间序列工具，它集中了许多其他库的功能。



### 7.1 Converting Strings to Dates

- 转换字符串为时间
- `pandas’ to_datetime`

strToDate.py

```python
import numpy as np
import pandas as pd

date_strings = np.array([
    '03-04-2005 11:35 PM',
    '23-05-2010 12:01 AM',
    '04-09-2009 09:09 PM'
])

# 转换 datetimes
print([pd.to_datetime(date, format='%d-%m-%Y %I:%M %p') for date in date_strings])


# coerce设置时 不会raise error 而会输出NaT
date_strings = np.append(date_strings,["13-13-3333 25:61 PP"])
print([pd.to_datetime(date, format='%d-%m-%Y %I:%M %p', errors='coerce') for date in date_strings])
```

![image-20220718114919034](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718114919034.png)

#### Discussion

- `pandas’ to_datetime`可以将字符串转换为时间

- 可以使用 format 参数来指定字符串的确切格式。

- 创建的日期格式

  | Code | Description                       | Example |
  | ---- | --------------------------------- | ------- |
  | %Y   | Full year                         | 2001    |
  | %m   | Month w/ zero padding             | 04      |
  | %d   | Day of the month w/ zero padding  | 09      |
  | %I   | Hour (12hr clock) w/ zero padding | 02      |
  | %p   | AM or PM                          | AM      |
  | %M   | Minute w/ zero padding            | 05      |
  | %S   | Second w/ zero padding            | 09      |



### 7.2 Handling Time Zones

- 处理时区
- 如果未指定，pandas 对象没有时区。 但是，我们可以在创建过程中使用 `tz `添加时区：

timeZone.py

```python
import pandas as pd
print(pd.Timestamp('2017-05-01 06:00:00', tz='Europe/London'))

date = pd.Timestamp('2017-05-01 06:00:00')
# 设置 time zone
date_in_london = date.tz_localize('Europe/London')
# 打印
print(date_in_london)

# 更改时区 time zone
print(date_in_london.tz_convert('Africa/Abidjan'))

# 通过pd一次性创建3个时间段
dates = pd.Series(pd.date_range('2/2/2002', periods=3, freq='M'))
# 设置时区
print(dates.dt.tz_localize('Africa/Abidjan'))
```

![image-20220718203729330](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718203729330.png)

#### Discussion

- 建议使用 pytz 库的字符串
- 导入 all_timezones 查看所有用于表示时区的字符串：

```python
# Load library
from pytz import all_timezones
# 展示前两个时区
print(all_timezones[0:2])
```

![image-20220718204818920](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718204818920.png)



### 7.3 Selecting Dates and Times

- 从一组日期中选择出特定的日期
- pandas dataFrame的切片访问

selectTime.py

```python
# Load library
import pandas as pd
# 创建空的DataFrame
dataframe = pd.DataFrame()
# 填充数据
dataframe['date'] = pd.date_range('1/1/2001', periods=100000, freq='H')
# Select 
print(dataframe[(dataframe['date'] > '2002-1-1 01:00:00') &
(dataframe['date'] <= '2002-1-1 04:00:00')])

```

![image-20220718205738464](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718205738464.png)

#### Discussion

- bool条件值或者切片
- 作者认为在大数据时应该使用切片，而数据量较小的时候使用bool值访问更加合适



### 7.4 Breaking Up Date Data into Multiple Features

- 将日期拆解成多个特征
- `pandas Series.dt`

seriesDt.py

```python
# Load library
import pandas as pd
# 空的dataFrame
dataframe = pd.DataFrame()
# 创建 150个数据
dataframe['date'] = pd.date_range('1/1/2001', periods=150, freq='W')
# 使用dt拆解成年月日小时分钟
dataframe['year'] = dataframe['date'].dt.year
dataframe['month'] = dataframe['date'].dt.month
dataframe['day'] = dataframe['date'].dt.day
dataframe['hour'] = dataframe['date'].dt.hour
dataframe['minute'] = dataframe['date'].dt.minute
# Show
print(dataframe.head(3))
```

![image-20220718210303226](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718210303226.png)

#### Discussion

- 分解时间作者认为有时很有用比如只考察某样事物一年间每月的变化



### 7.5 Calculating the Difference Between Dates

- 想要计算时间差
- `pandas`

timestamp.py

```python
# Load library
import pandas as pd
# Create data frame
dataframe = pd.DataFrame()
# 创建两个时间
dataframe['Arrived'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-04-2017')]
dataframe['Left'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-06-2017')]
# 计算持续时间
print(dataframe['Left'] - dataframe['Arrived'])

# 计算持续时间
print(pd.Series(delta.days for delta in (dataframe['Left'] - dataframe['Arrived'])))
```

![image-20220718211219675](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718211219675.png)

#### Discussion

- 计算时间差值很有用
- pandas中的TimeDelta使得计算时间差值很简单

```python
# 计算时间差
timedelta = pd.Timedelta('2 days 2 hours 15 minutes 30 seconds')
print(timedelta)
```

![image-20220718211626892](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718211626892.png)



### 7.6 Encoding Days of the Week

- 对星期编码
- `Series.dt`

weekdays.py

```python
# Load library
import pandas as pd
# Create dates
dates = pd.Series(pd.date_range("2/2/2002", periods=3, freq="M"))
# Show days of the week
print(dates.dt.weekday_name)
# Show days of the week
print(dates.dt.weekday)
```

报错

![image-20220718212047131](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718212047131.png)

根据查阅资料weekday_name已经被废弃，改为day_name

```python
# Load library
import pandas as pd
# Create dates
dates = pd.Series(pd.date_range("2/2/2002", periods=3, freq="M"))
# 星期几
print(dates.dt.day_name())
# 数字编号
print(dates.dt.weekday)
```

![image-20220718213120073](C:\Users\12587\AppData\Roaming\Typora\typora-user-images\image-20220718213120073.png)

#### Discussion

- 对于分析星期类的问题，pandas的转换会很有帮助





### 7.7 Creating a Lagged Feature

- 创建滞后特征
- 使用pandas库的`shift`

```python
# Load library
import pandas as pd
# Create data frame
dataframe = pd.DataFrame()
# Create data
dataframe["dates"] = pd.date_range("1/1/2001", periods=5, freq="D")
dataframe["stock_price"] = [1.1,2.2,3.3,4.4,5.5]
# 创建一个新的特征前移一天
dataframe["previous_days_stock_price"] = dataframe["stock_price"].shift(1)
# 展现dataframe
print(dataframe)
```

![image-20220718213839590](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718213839590.png)

#### Discussion

- 什么是**Lagged Feature**?
  - 数据通常基于有规律的间隔时间段（例如，每天、每小时、每三个小时），我们有兴趣使用过去的值进行预测,而这些过去的特征就可以称作**滞后特征（Lagged Feature）**

- 在我们的示例中因为往前移了一天，第一行没有数据，所以是NaN；



###  7.8 Using Rolling Time Windows 

- 计算这些流动时间的统计数据

statistic.py

```python
# Load library
import pandas as pd
# Create datetimes
time_index = pd.date_range("01/01/2010", periods=5, freq="M")
# 设置index
dataframe = pd.DataFrame(index=time_index)
# 新建一个特征
dataframe["Stock_Price"] = [1,2,3,4,5]
# 窗口大小为2，求出窗口时间中的平均值
print(dataframe.rolling(window=2).mean())
```

![image-20220718214653455](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718214653455.png)

#### Discussion

- 滚动时间窗口包含连续的一些时间，然后一次次的移动
- 每次可以求出窗口内的统计数据

### 7.9 Handling Missing Data in Time Series

- 处理丢失数据
- 类似于之前处理丢失数据的方式
- 我们对于时间数据还可以使用插值法

handlingMissing.py

```python
# Load libraries
import pandas as pd
import numpy as np

# Create date
time_index = pd.date_range("01/01/2010", periods=5, freq="M")
# set index
dataframe = pd.DataFrame(index=time_index)
# 创建一组有缺失的数据
dataframe["Sales"] = [1.0, 2.0, np.nan, np.nan, 5.0]
# 插入缺失值
print(dataframe.interpolate())

# 前值填充
print(dataframe.ffill())

# 后值填充
print(dataframe.bfill())

```

![image-20220718215920807](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718215920807.png)

#### Discussion

- 插值是一种填补由缺失值引起的空白的技术，实际上是在与空白接壤的已知值之间绘制一条直线或曲线，并使用该直线或曲线来预测合理的值。

  - 我们的解决方案中，两个缺失值的差距以 2.0 和 5.0 为界。通过拟合从 2.0 到 5.0 的线，我们可以对介于 3.0 和 4.0 之间的两个缺失值做出合理的猜测。

  -  如果我们认为两个已知点之间的线是非线性的，我们可以使用 interpolate 的方法来指定插值方法：

    ```python
    # 非线性 结果见下图
    print(dataframe.interpolate(method="quadratic"))
    ```

    

- 在某些情况下，我们的缺失值差距很大，并且不想在整个差距中插入值。在这些情况下，我们可以使用 limit 来限制插值的数量，使用 limit_direction 来设置是否从间隙之前的最后一个已知值向前插值，

```python
# 限制
print(dataframe.interpolate(limit=1, limit_direction="forward"))
```

![image-20220718221132787](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718221132787.png)

