# Load library
from nltk.stem.porter import PorterStemmer
# Create word tokens
tokenized_words = ['i', 'am', 'humbled', 'by', 'this', 'traditional', 'meeting']
# Create stemmer
porter = PorterStemmer()
# Apply stemmer
print([porter.stem(word) for word in tokenized_words])