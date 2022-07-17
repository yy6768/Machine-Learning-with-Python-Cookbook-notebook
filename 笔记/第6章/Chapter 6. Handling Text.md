## Chapter 6. Handling Text

- 本笔记是针对人工智能典型算法的课程中Machine Learning with Python Cookbook的学习笔记
- 学习的实战代码都放在代码压缩包中
- 实战代码的运行环境是python3.9 numpy 1.23.1

上一章：[[(89条消息) Machine Learning with Python Cookbook 学习笔记 第5章_五舍橘橘的博客-CSDN博客](https://blog.csdn.net/weixin_51083297/article/details/125833474?csdn_share_tail={"type"%3A"blog"%2C"rType"%3A"article"%2C"rId"%3A"125833474"%2C"source"%3A"weixin_51083297"}&ctrtid=Af1hc)]

代码笔记仓库（给颗星再走qaq）：[yy6768/Machine-Learning-with-Python-Cookbook-notebook: 人工智能典型算法笔记 (github.com)](https://github.com/yy6768/Machine-Learning-with-Python-Cookbook-notebook)

### 6.0 Introduction

- 在本章中，我们将介绍将文本转换为信息丰富的特征的策略。
- 因为处理文本信息的方法过多，本章将只能重点介绍。
- 本章介绍的这些常规技术是非常有价值的预处理工具

### 6.1 Cleaning Text

对非结构性文本进行一些基本的清理

常用的函数：

`strip`

`replace`

`split`

clean.py

```python
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
```

![image-20220717115227664](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717115227664.png)



#### Discussion

- 大多数文本数据都需要在我们使用它来构建功能之前进行清理。 
- 大多数基本的文本清理都可以使用 Python 的标准字符串操作来完成。
- 可以自定义函数完成处理



### 6.2 Parsing and Cleaning HTML

- 处理HTML文本
- 使用Beautiful Soup库



cleanHtml.py

- 需要安装两个库

- Beautiful Soup 4

  官网：[Beautiful Soup Documentation — Beautiful Soup 4.4.0 documentation (beautiful-soup-4.readthedocs.io)](https://beautiful-soup-4.readthedocs.io/en/latest/)

  [Beautiful Soup 中文文档](https://beautifulsoup.cn/)

  教程[BeautifulSoup详细使用教程！你学会了吗？ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/59822990)

  ```
  conda install bs4
  ```

- lxml

  [Python lxml库的安装和使用 (biancheng.net)](http://c.biancheng.net/python_spider/lxml.html)

  ```
  conda install lxml
  ```

  

  代码：

  ```
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
  ```

  ![image-20220717141910538](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717141910538.png)

#### Discussion

- Beautiful Soup 是一个强大的 Python 库，专为抓取 HTML 而设计。

- 支持原生和第三方解析器

  | 解析器           | 使用方法                                                     | 优势                                                  | 劣势                                            |
  | ---------------- | ------------------------------------------------------------ | ----------------------------------------------------- | ----------------------------------------------- |
  | Python标准库     | BeautifulSoup(markup, “html.parser”)                         | Python的内置标准库执行速度适中文档容错能力强          | Python 2.7.3 or 3.2.2)前 的版本中文档容错能力差 |
  | lxml HTML 解析器 | BeautifulSoup(markup, “lxml”)                                | 速度快文档容错能力强                                  | 需要安装C语言库                                 |
  | lxml XML 解析器  | BeautifulSoup(markup, [“lxml”, “xml”])BeautifulSoup(markup, “xml”) | 速度快唯一支持XML的解析器                             | 需要安装C语言库                                 |
  | html5lib         | BeautifulSoup(markup, “html5lib”)                            | 最好的容错性以浏览器的方式解析文档生成HTML5格式的文档 | 速度慢不依赖其他库                              |

- Beautiful Soup四大标签

  - Tag 
  -  NavigableString 
  -  BeautifulSoup 
  -  Comment

- 支持遍历文档树，查找文档树以及CSS选择器三个主要操作





### 6.3 Removing Punctuation

移除标点

removingPunctuation.py

```python
# Load libraries
import unicodedata
import sys

# 文本
text_data = ['Hi!!!! I. Love. This. Song....',
             '10000% Agree!!!! #LoveIT',
             'Right?!?!']
# 创建一个字典，用于处理标点
punctuation = dict.fromkeys(i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith('P'))
# For each string, 移除标点
print([string.translate(punctuation) for string in text_data])
```



![image-20220717143110998](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717143110998.png)

#### Discussion

- translate 是一种 Python 方法，因其超快的速度而广受欢迎。
- 在该例中，我们创建一个字典，用unicodedata.category方法查找出所有unicode中属于标点符号的字符，并把它映射到None上，最后再用translate方法把标点符号全部转换为空值



### 6.4 Tokenizing Text

- 把文本分解为单个单词

- NLTK——强大的python自然语言工具包，具有强大的文本集操作

- nltk包需要安装

  ```
  conda install nltk
  ```

- nltk需要下载数据包

  ```python
  # 主动下载
  import nltk
  nltk.download('punkt')
  
  # 离线下载
  # 下载到指定的文件夹里
  # http://www.nltk.org/nltk_data/
  ```

  

NTLKExample.py

```python
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
```

主动下载+结果

![image-20220717145912597](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717145912597.png)



#### Discussion

- 官网：[NLTK :: Natural Language Toolkit](https://www.nltk.org/)
- 首先nltk的运行是需要它自身的开源**语料库**、**词库**、**标记库**进行的
- 提供非常广泛的功能，例如词性分析，词性还原、还可以进行朴素贝叶斯分类



### 6.5 Removing Stop Words

移除去stop words(信息量极少的次，例如‘I' 'am'等)

仍然需要NLTK库



stopWorkds.py

```
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
```

![image-20220717151128933](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717151128933.png)

#### Discussion

- 虽然“stop words”可以指代我们想要在处理之前删除的任何一组词，但该术语通常指的是极其常见的词，它们本身包含的信息价值很少。 NLTK 有一个常见停用词列表，我们可以使用这些停用词在我们的标记词中查找和删除停用词：

```
# Show stop words
stop_words[:5]
```

['i', 'me', 'my', 'myself', 'we']

- 需要注意ntlk中的stop words都是小写的



### 6.6 Stemming Words

- 把单词转换成原型
- 需要使用nltk中的`PorterStemmer`

stemmingWords.py

```python
# Load library
from nltk.stem.porter import PorterStemmer
# Create word tokens
tokenized_words = ['i', 'am', 'humbled', 'by', 'this', 'traditional', 'meeting']
# 创建 stemmer
porter = PorterStemmer()
# 运用 stemmer
print([porter.stem(word) for word in tokenized_words])
```

![image-20220717151651206](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717151651206.png)

#### Discussion

- 虽然将句子中的词转换为词干可读性较差，但是更容易将句子观察比较

- PorterStemmer使用了现在非常常用的Porter算法来删除一些常见的前缀和后缀生成stemming

#### Porter算法：[(89条消息) Porter Algorithm ---------词干提取算法_noobzc1的博客-CSDN博客_词干提取算法](https://blog.csdn.net/noobzc1/article/details/8902881#)（网上有python实现，但是最经典的版本是java实现的）

- 定义一个类	

  ```java
  class Stemmer
  {  private char[] b;
     private int i,     /* b中的元素位置（偏移量） */
                 i_end, /* 要抽取词干单词的结束位置 */
                 j, k;
     private static final int INC = 50;
                       /* 随着b的大小增加数组要增长的长度（防止溢出） */
     public Stemmer()
     {  b = new char[INC];
        i = 0;
        i_end = 0;
     }
  }		
  ```

  

- 字符串处理add

  ```java
  /**
  * 增加一个字符到要存放待处理的单词的数组。添加完字符时，
  * 可以调用stem(void)方法来进行抽取词干的工作。
  */
  public void add(char ch)
  {  if (i == b.length)
     {  char[] new_b = new char[i+INC];
        for (int c = 0; c < i; c++) new_b[c] = b[c];
        b = new_b;
     }
     b[i++] = ch;
  }
   
  /** 增加wLen长度的字符数组到存放待处理的单词的数组b。
  */
  public void add(char[] w, int wLen)
  {  if (i+wLen >= b.length)
     {  char[] new_b = new char[i+wLen+INC];
        for (int c = 0; c < i; c++) new_b[c] = b[c];
        b = new_b;
     }
     for (int c = 0; c < wLen; c++) b[i++] = w[c];
  }
  ```

  

- 一系列辅助函数

  总结下来就是判断发音类，判断类型类和操作类，核心函数应该是m()返回辅音序列的个数

  - **cons(i)**：参数i：int型；返回值bool型。当i为辅音时，返回真；否则为假。

  - **m（）**

    ：返回值：int型。表示单词b介于0和j之间辅音序列的个度。现假设c代表辅音序列，而v代表元音序列。<..>表示任意存在。于是有如下定义；

    - <c><v>      结果为 0
    - <c>vc<v>    结果为 1
    - <c>vcvc<v>   结果为 2
    - <c>vcvcvc<v> 结果为 3
    - ....

  - **vowelinstem()**：返回值：bool型。从名字就可以看得出来，表示单词b介于0到i之间是否存在元音。

  - **doublec(j)**：参数j：int型；返回值bool型。这个函数用来表示在j和j-1位置上的两个字符是否是相同的辅音。

  - **cvc(i)**：参数i：int型；返回值bool型。对于i，i-1，i-2位置上的字符，它们是“辅音-元音-辅音”的形式，并且对于第二个辅音，它不能为w、x、y中的一个。这个函数用来处理以e结尾的短单词。比如说cav(e)，lov(e)，hop(e)，crim(e)。但是像snow，box，tray就辅符合条件。

  - **ends(s)**：参数：String；返回值：bool型。顾名思义，判断b是否以s结尾。

  - **setto(s)**：参数：String；void类型。把b在(j+1)...k位置上的字符设为s，同时，调整k的大小。

  - **r(s)**：参数：String；void类型。在m()>0的情况下，调用setto(s)。

  ```java
  // cons(i) 为真 <=> b[i] 是一个辅音
  private final boolean cons(int i)
  {  switch (b[i])
     {  case 'a': case 'e': case 'i': case 'o': case 'u': return false; //aeiou
        case 'y': return (i==0) ? true : !cons(i-1);
                  //y开头，为辅；否则看i-1位，如果i-1位为辅，y为元，反之亦然。
        default: return true;
     }
  }
   
  // m() 用来计算在0和j之间辅音序列的个数。 见上面的说明。 */
  private final int m()
  {  int n = 0; //辅音序列的个数，初始化
     int i = 0; //偏移量
     while(true)
     {  if (i > j) return n; //如果超出最大偏移量，直接返回n
        if (! cons(i)) break; //如果是元音，中断
        i++; //辅音移一位，直到元音的位置
     }
     i++; //移完辅音，从元音的第一个字符开始
     while(true)//循环计算vc的个数
     {  while(true) //循环判断v
        {  if (i > j) return n;
           if (cons(i)) break; //出现辅音则终止循环
              i++;
        }
        i++;
        n++;
        while(true) //循环判断c
        {  if (i > j) return n;
           if (! cons(i)) break;
           i++;
        }
        i++;
      }
  }
   
  // vowelinstem() 为真 <=> 0,...j 包含一个元音
  private final boolean vowelinstem()
  {  int i; for (i = 0; i <= j; i++) if (! cons(i)) return true;
     return false;
  }
   
  // doublec(j) 为真 <=> j,(j-1) 包含两个一样的辅音
  private final boolean doublec(int j)
  {  if (j < 1) return false;
     if (b[j] != b[j-1]) return false;
     return cons(j);
  }
   
  /* cvc(i) is 为真 <=> i-2,i-1,i 有形式： 辅音 - 元音 - 辅音
     并且第二个c不是 w,x 或者 y. 这个用来处理以e结尾的短单词。 e.g.
   
     cav(e), lov(e), hop(e), crim(e), 但不是
     snow, box, tray.
   
  */
  private final boolean cvc(int i)
  {  if (i < 2 || !cons(i) || cons(i-1) || !cons(i-2)) return false;
     {  int ch = b[i];
           if (ch == 'w' || ch == 'x' || ch == 'y') return false;
     }
        return true;
  }
   
  private final boolean ends(String s)
  {  int l = s.length();
     int o = k-l+1;
     if (o < 0) return false;
     for (int i = 0; i < l; i++) if (b[o+i] != s.charAt(i)) return false;
     j = k-l;
     return true;
  }
   
  // setto(s) 设置 (j+1),...k 到s字符串上的字符, 并且调整k值
  private final void setto(String s)
  {  int l = s.length();
     int o = j+1;
     for (int i = 0; i < l; i++) b[o+i] = s.charAt(i);
     k = j+l;
  }
   
  private final void r(String s) { if (m() > 0) setto(s); }
  ```

  

- 然后正式进入六步操作的工作

  - 第一步是否为ed或着ing结尾

    可以明显看到麻烦在于分类讨论，有许多诸如sses或者ies这样的结尾很难判断

  ```
  /* step1() 处理复数，ed或者ing结束的单词。比如：
   
        caresses  ->  caress
        ponies    ->  poni
        ties      ->  ti
        caress    ->  caress
        cats      ->  cat
   
        feed      ->  feed
        agreed    ->  agree
        disabled  ->  disable
   
        matting   ->  mat
        mating    ->  mate
        meeting   ->  meet
        milling   ->  mill
        messing   ->  mess
   
        meetings  ->  meet
  */
   
  private final void step1()
  {  if (b[k] == 's')
     {  if (ends("sses")) k -= 2; //以“sses结尾”
        else if (ends("ies")) setto("i"); //以ies结尾，置为i
        else if (b[k-1] != 's') k--; //两个s结尾不处理
     }
     if (ends("eed")) { if (m() > 0) k--; } //以“eed”结尾，当m>0时，左移一位
     else if ((ends("ed") || ends("ing")) && vowelinstem())
     {  k = j;
        if (ends("at")) setto("ate"); else
        if (ends("bl")) setto("ble"); else
        if (ends("iz")) setto("ize"); else
        if (doublec(k))//如果有两个相同辅音
        {  k--;
           {  int ch = b[k];
              if (ch == 'l' || ch == 's' || ch == 'z') k++;
           }
        }
        else if (m() == 1 && cvc(k)) setto("e");
    }
  }
  ```

  

  - 第二步 如果含有元音，并且以y结尾将y改成i

    ```java
    private final void step2() { if (ends("y") && vowelinstem()) b[k] = 'i'; }
    ```

    

  - 第三步 将双后缀的单词映射为单后缀。 和第一步一样需要分类讨论，有很多英语上的特殊情况例如

    所以只能一个一个进行判断，但是实际上就是枚举所有类别的双后缀然后转换成原来的模式

    ```java
    /* step3() 将双后缀的单词映射为单后缀。 所以 -ization ( = -ize 加上
       -ation) 被映射到 -ize 等等。 注意在去除后缀之前必须确保
       m() > 0. */
    private final void step3() { if (k == 0) return;  switch (b[k-1])
    {
        case 'a': if (ends("ational")) { r("ate"); break; }
                  if (ends("tional")) { r("tion"); break; }
                  break;
        case 'c': if (ends("enci")) { r("ence"); break; }
                  if (ends("anci")) { r("ance"); break; }
                  break;
        case 'e': if (ends("izer")) { r("ize"); break; }
                  break;
        case 'l': if (ends("bli")) { r("ble"); break; }
                  if (ends("alli")) { r("al"); break; }
                  if (ends("entli")) { r("ent"); break; }
                  if (ends("eli")) { r("e"); break; }
                  if (ends("ousli")) { r("ous"); break; }
                  break;
        case 'o': if (ends("ization")) { r("ize"); break; }
                  if (ends("ation")) { r("ate"); break; }
                  if (ends("ator")) { r("ate"); break; }
                  break;
        case 's': if (ends("alism")) { r("al"); break; }
                  if (ends("iveness")) { r("ive"); break; }
                  if (ends("fulness")) { r("ful"); break; }
                  if (ends("ousness")) { r("ous"); break; }
                  break;
        case 't': if (ends("aliti")) { r("al"); break; }
                  if (ends("iviti")) { r("ive"); break; }
                  if (ends("biliti")) { r("ble"); break; }
                  break;
        case 'g': if (ends("logi")) { r("log"); break; }
    } }
    ```

  -  第四步，处理 -ic-，-full，-ness等等后缀。和步骤3有着类似的处理。 也是分类讨论然后替换

    ```java
    private final void step4() { switch (b[k])
    {
        case 'e': if (ends("icate")) { r("ic"); break; }
                  if (ends("ative")) { r(""); break; }
                  if (ends("alize")) { r("al"); break; }
                  break;
        case 'i': if (ends("iciti")) { r("ic"); break; }
                  break;
        case 'l': if (ends("ical")) { r("ic"); break; }
                  if (ends("ful")) { r(""); break; }
                  break;
        case 's': if (ends("ness")) { r(""); break; }
                  break;
    } }
    ```

    

  - 第五步就是根据m（）的统计情况，处理<c>vcvc<v>的情况

    ```java
    private final void step5()
    {   if (k == 0) return;  switch (b[k-1])
        {  case 'a': if (ends("al")) break; return;
           case 'c': if (ends("ance")) break;
                     if (ends("ence")) break; return;
           case 'e': if (ends("er")) break; return;
           case 'i': if (ends("ic")) break; return;
           case 'l': if (ends("able")) break;
                     if (ends("ible")) break; return;
           case 'n': if (ends("ant")) break;
                     if (ends("ement")) break;
                     if (ends("ment")) break;
                     /* element etc. not stripped before the m */
                     if (ends("ent")) break; return;
           case 'o': if (ends("ion") && j >= 0 && (b[j] == 's' || b[j] == 't')) break;
                                     /* j >= 0 fixes Bug 2 */
                     if (ends("ou")) break; return;
                     /* takes care of -ous */
           case 's': if (ends("ism")) break; return;
           case 't': if (ends("ate")) break;
                     if (ends("iti")) break; return;
           case 'u': if (ends("ous")) break; return;
           case 'v': if (ends("ive")) break; return;
           case 'z': if (ends("ize")) break; return;
           default: return;
        }
        if (m() > 1) k = j;//调用对k赋值
    }
    ```

     

  - 第6步很好理解 除去末尾冗余的e

    ```java
    private final void step6()
    {  j = k;
       if (b[k] == 'e')
       {  int a = m();
          if (a > 1 || a == 1 && !cvc(k-1)) k--;
       }
       if (b[k] == 'l' && doublec(k) && m() > 1) k--;
    }
    ```

    

### 6.7 Tagging Parts of Speech

标记词性

tagging.py

```python
# Load libraries
from nltk import pos_tag
from nltk import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
# 下载
# import nltk
# nltk.download('averaged_perceptron_tagger')
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
# 标记每个词的词性
for tweet in tweets:
    tweet_tag = pos_tag(word_tokenize(tweet))
	tagged_tweets.append([tag for word, tag in tweet_tag])

# Use one-hot 编码
one_hot_multi = MultiLabelBinarizer()
print(one_hot_multi.fit_transform(tagged_tweets))

print(one_hot_multi.classes_)
```

![image-20220717155353522](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717155353522.png)

![image-20220717155647789](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717155647789.png)

#### Discussion

- 如果是非专业术语语句，使用 NLTK 的预训练词性标注器是最简单的方法

- Brown Corpus是NTLK使用的语料库

- NLTK uses the Penn Treebank parts for speech tags, some examples:

  

  | tag  | Parts of Speech                    |
  | ---- | ---------------------------------- |
  | NNP  | Proper noun, singular              |
  | NN   | Noun, singular or mass             |
  | RB   | Adverb                             |
  | VBD  | Verb, past tense                   |
  | VBG  | Verb, gerund or present participle |
  | JJ   | Adjective                          |
  | PRP  | Personal  pronoun                  |

- 这里我们使用一个退避 n-gram 标注器，其中 n 是我们在预测词的词性标签时考虑的先前词的数量。首先我们使用 TrigramTagger 考虑前面两个词；如果两个单词不存在，我们“退后”并使用 BigramTagger 考虑前一个单词的标签，最后如果失败，我们只使用 UnigramTagger 查看单词本身。

  ```python
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
  # backoff tagger
  unigram = UnigramTagger(train)
  bigram = BigramTagger(train, backoff=unigram)
  trigram = TrigramTagger(train, backoff=bigram)
  # 准确率 :0.8179229731754832
  print(trigram.evaluate(test))
  ```

  

### 6.8 Encoding Text as a Bag of Words

统计特定的词出现的次数

使用`scikit-learn’s CountVectorizer`

countExample.py

```python
# Load library
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# Create text
text_data = np.array(['I love Brazil. Brazil!',
'Sweden is best',
'Germany beats both'])
# 创建一个特征矩阵包含计数信息
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)
# 打印
print(bag_of_words)

print(bag_of_words.toarray())

# 展示特征的name get_feature_names即将废弃
print(count.get_feature_names_out())
```

![image-20220717160733410](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717160733410.png)

#### Discussion

- Bag-of-words models是文本转换成feature最常用的模型
- 大多数bag-of-words的矩阵都是稀疏矩阵，所以CountVectorizer 的一个很好的特性是默认情况下输出是一个稀疏矩阵
- CountVectorizer 带有许多有用的参数，可以轻松创建词袋特征矩阵。首先，虽然默认情况下每个特征都是一个单词，但不一定是这样。相反，我们可以将每个特征设置为两个单词（称为 2-gram）甚至三个单词（3-gram）的组合。

- ngram_range 设置我们的 n-gram 的最小和最大大小。 例如，(2,3) 将返回所有 2-gram 和 3-gram。 其次，我们可以使用内置列表或自定义列表的 stop_words 轻松删除低信息填充词。 最后，我们可以使用词汇将我们想要考虑的单词或短语限制在某个单词列表中。 例如，我们可以为仅出现的国家名称创建一个词袋特征矩阵：

  ```python
  # Create feature matrix with arguments
  count_2gram = CountVectorizer(ngram_range=(1, 2),
                                stop_words="english",
                                vocabulary=['brazil'])
  bag = count_2gram.fit_transform(text_data)
  # View feature matrix
  print(bag.toarray())
  # View the 1-grams and 2-grams
  print(count_2gram.vocabulary_)
  ```

  ![image-20220717161421563](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717161421563.png)

- N-Gram是一种基于统计语言模型的算法。它的基本思想是将文本里面的内容按照字节进行大小为N的滑动窗口操作，形成了长度是N的字节片段序列。

  [自然语言处理中N-Gram模型介绍 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/32829048)





### 6.9 Weighting Word Importance

- 给单词加权
- 使用词频-逆文档频率 (tf-idf) 比较文档（推文、电影评论、演讲稿等）中单词的频率与所有其他文档中单词的频率。 scikit-learn 使用 TfidfVectorizer 使这变得简单：

weightWords.py

```python
# Load libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# Create text
text_data = np.array(['I love Brazil. Brazil!',
'Sweden is best',
'Germany beats both'])
# 创建 the tf-idf 特征矩阵
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)
# 展示这个特征矩阵
print(feature_matrix)

# 转换为一般数组
print(feature_matrix.toarray())
# 特征名称
print(tfidf.vocabulary_)
```



![image-20220717162048935](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717162048935.png)

#### 

#### Discussion

- 一个词在文档中出现的越多，它对该文档的重要性就越大。 例如，如果经济一词频繁出现，则证明该文件可能是关于经济的。 我们称之为术语频率 (tf)。
- 如果一个词出现在许多文档中，那么它对任何单个文档的重要性都可能降低。 例如，如果某些文本数据中的每个文档都包含后面的单词，那么它可能是一个不重要的单词。 我们称此文档频率 (df)。
- 通过结合这两个统计数据，我们可以为每个单词分配一个分数，表示该单词在文档中的重要性。 具体来说，我们将 tf 乘以文档频率 (idf) 的倒数：

​	$$tf-idf(t, d) = tf(t,d) * idf(t)$$

- tf 和 idf 的计算方式有很多变化。 在 scikit-learn 中，tf 只是单词在文档中出现的次数，idf 的计算公式为：

  $$idf(t) = log(\frac{1 + n_d}{1 + df(d, t}) +1$$

​	其中 nd 是文档数，df(d,t) 是术语，t 的文档频率（即，该术语出现的文档数）。默认情况下，scikit-learn 	然后使用欧几里得范数（L2 范数）对 tf-idf 向量进行归一化。结果值越高，单词对文档越重要



