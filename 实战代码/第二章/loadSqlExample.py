import pandas as pd
from sqlalchemy import create_engine

# 初始化数据库连接
# 按实际情况依次填写MySQL的用户名、密码、IP地址、端口、数据库名
engine = create_engine('mysql+pymysql://root:444555@localhost:3306/lab5')

sql_query = 'select * from student;'
# 使用pandas的read_sql_query函数执行SQL语句，并存入DataFrame
df_read = pd.read_sql_query(sql_query, engine)

print(df_read.head(2))