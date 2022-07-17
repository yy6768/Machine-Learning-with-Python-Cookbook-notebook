# Load library

from nltk.corpus import stopwords

# 需要下载
import nltk
nltk.download('stopwords')
# Create word tokens
tokenized_words = ['i',
                   'am',
                   'going',
                   'to',
                   'go',
                   'to',
                   'the',
                   'store',
                   'and',
                   'park']
# Load stop words
stop_words = stopwords.words('english')
# Remove stop words
print([word for word in tokenized_words if word not in stop_words])