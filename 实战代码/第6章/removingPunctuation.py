# Load libraries
import unicodedata
import sys

# 文本
text_data = ['Hi!!!! I. Love. This. Song....',
             '10000% Agree!!!! #LoveIT',
             'Right?!?!']
# 创建一个字典，标点映射到空置上
punctuation = dict.fromkeys(i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith('P'))

# 移除标点
print([string.translate(punctuation) for string in text_data])