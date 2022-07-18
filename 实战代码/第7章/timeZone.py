
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

# Load library
from pytz import all_timezones
# 展示前两个时区
print(all_timezones[0:2])