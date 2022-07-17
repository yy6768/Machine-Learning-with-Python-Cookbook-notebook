# Load library

from nltk.tokenize import word_tokenize
# Create text
string = "The science of today is the technology of tomorrow"
# 分词
print(word_tokenize(string))

# Load library
from nltk.tokenize import sent_tokenize
# Create text
string = "The science of today is the technology of tomorrow. Tomorrow is today."
# 分句子
print(sent_tokenize(string))