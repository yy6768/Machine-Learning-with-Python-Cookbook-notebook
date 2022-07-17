# Load library
from bs4 import BeautifulSoup

# 创建一些html文本
html = """
<div class='full_name'><span style='font-weight:bold'>
Masego</span> Azra</div>"
"""
# 转换 html
soup = BeautifulSoup(html, "lxml")
# 查找div，寻找class为fullname的标签，获得它的文本属性
print(soup.find("div", {"class": "full_name"}).text)