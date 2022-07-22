# Load library
import pandas as pd

# Create DataFrame
employee_data = {'employee_id': ['1', '2', '3', '4'],
                 'name': ['Amy Jones', 'Allen Keys', 'Alice Bees',
                          'Tim Horton']}
dataframe_employees = pd.DataFrame(employee_data, columns=['employee_id',
                                                           'name'])
# Create DataFrame
sales_data = {'employee_id': ['3', '4', '5', '6'],
              'total_sales': [23456, 2512, 2345, 1455]}
dataframe_sales = pd.DataFrame(sales_data, columns=['employee_id', 'total_sales'])
# 自然连接
print(pd.merge(dataframe_employees, dataframe_sales, on='employee_id'))

# 外连接
print(pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='outer'))

# 左连接
print(pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='left'))

# 指定属性链接
print(pd.merge(dataframe_employees,
         dataframe_sales,
         left_on='employee_id',
         right_on='employee_id'))
