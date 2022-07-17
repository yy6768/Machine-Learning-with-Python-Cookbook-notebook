
import pandas as pd

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# 因为原书的excel无法访问，所以替换了一个url
url = "https://www.sample-videos.com/xls/Sample-Spreadsheet-10-rows.xls"

# 加载url
df = pd.read_excel(url, sheet_name=0, header=None)

# 打印前两行
print(df.head(2))