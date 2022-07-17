# 正则表达式模块
import re

# 新建一段文本
text_data = [" Interrobang. By Aishwarya Henriette ",
             "Parking And Going. By Karl Gautier",
             " Today Is The night. By Jarek Prakash "]
# 除去始末空格
strip_whitespace = [string.strip() for string in text_data]
# Show text
print(strip_whitespace)

# 除去.
remove_periods = [string.replace(".", "") for string in strip_whitespace]
# Show text
print(remove_periods)


# Create function
def capitalizer(string: str) -> str:
    return string.upper()


# 全部变成大写
print([capitalizer(string) for string in remove_periods])

# 新建一个正则表达式匹配函数
def replace_letters_with_upper_x(string: str) -> str:
    return re.sub(r"[a-zA-Z]", "X", string)


# 运用function
print([replace_letters_with_upper_x(string) for string in remove_periods])