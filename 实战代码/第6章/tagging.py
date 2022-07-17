# Load libraries
from nltk import pos_tag
from nltk import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer

# 下载
import nltk
# nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
# Create text


text_data = "Chris loved outdoor running"
# 使用训练好的模型处理
text_tagged = pos_tag(word_tokenize(text_data))
# Show parts of speech
print(text_tagged)

# Filter words
print([word for word, tag in text_tagged if tag in ['NN', 'NNS', 'NNP', 'NNPS']])

# 用独热编码将词性统计转化为特征矩阵
# Create text
tweets = ["I am eating a burrito for breakfast",
          "Political science is an amazing field",
          "San Francisco is an awesome city"]
# Create list
tagged_tweets = []
# Tag each word and each tweet
for tweet in tweets:
    tweet_tag = pos_tag(word_tokenize(tweet))
    tagged_tweets.append([tag for word, tag in tweet_tag])
# Use one-hot encoding to convert the tags into features
one_hot_multi = MultiLabelBinarizer()
print(one_hot_multi.fit_transform(tagged_tweets))

print(one_hot_multi.classes_)

# Load library
from nltk.corpus import brown
from nltk.tag import UnigramTagger
from nltk.tag import BigramTagger
from nltk.tag import TrigramTagger

# Get some text from the Brown Corpus, broken into sentences
sentences = brown.tagged_sents(categories='news')
# Split into 4000 sentences for training and 623 for testing
train = sentences[:4000]
test = sentences[4000:]
# Create backoff tagger
unigram = UnigramTagger(train)
bigram = BigramTagger(train, backoff=unigram)
trigram = TrigramTagger(train, backoff=bigram)
# Show accuracy 0.8179229731754832
print(trigram.evaluate(test))
