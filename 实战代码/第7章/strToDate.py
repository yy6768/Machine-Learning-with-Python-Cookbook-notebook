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