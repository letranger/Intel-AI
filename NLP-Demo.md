
# Table of Contents

1.  [特徵擷取](#org349c1c2)
    1.  [例: 出生地](#orga66cb4a)
        1.  [solution 1](#org8b76fe3)
    2.  [One Hot Encoding](#org1a99e2a)
    3.  [從文本中提取特徵](#orga9f543e)
        1.  [詞袋模型(bag-of-words model)](#orge1195e9)
        2.  [字嵌入(word embedding)](#org9f98130)
2.  [DEMO](#org2a1dfb2)
3.  [jeiba DEMO](#org2751d6a)
4.  [jeiba DEMO](#org3453e56)



<a id="org349c1c2"></a>

# 特徵擷取

如何處理類別變數(categorical variable)/名義變數(nominal variable)
為何要處理? 因為要計算。


<a id="orga66cb4a"></a>

## 例: 出生地

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">user</th>
<th scope="col" class="org-left">city</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">James</td>
<td class="org-left">Taipei</td>
</tr>


<tr>
<td class="org-left">Ruby</td>
<td class="org-left">Tainan</td>
</tr>


<tr>
<td class="org-left">Vanessa</td>
<td class="org-left">Kaohsiung</td>
</tr>
</tbody>
</table>


<a id="org8b76fe3"></a>

### solution 1

以數值代替city，例

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">user</th>
<th scope="col" class="org-right">city</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">James</td>
<td class="org-right">1</td>
</tr>


<tr>
<td class="org-left">Ruby</td>
<td class="org-right">2</td>
</tr>


<tr>
<td class="org-left">Vanessa</td>
<td class="org-right">3</td>
</tr>
</tbody>
</table>

問題: Kaohsiung的城市值為Taipei的三倍


<a id="org1a99e2a"></a>

## One Hot Encoding

    1  from sklearn.feature_extraction import DictVectorizer
    2  onehot_encoder = DictVectorizer()
    3  X = [
    4      {'city': 'Taipei'},
    5      {'city': 'Tainan'},
    6      {'city': 'Kaohsiung'}
    7  ]
    8  print(onehot_encoder.fit_transform(X).toarray())

    [[0. 0. 1.]
     [0. 1. 0.]
     [1. 0. 0.]]


<a id="orga9f543e"></a>

## 從文本中提取特徵

兩種常用的文本表示形式: 詞袋模型(bag-of-words)及字嵌入(word wmbedding)


<a id="orge1195e9"></a>

### 詞袋模型(bag-of-words model)

1.  語料庫(corpus)

        1  corpus = [
        2      'UNC played Duke in basketball',
        3      'Duke lost the basetball game'
        4  ]
    
    這個語料庫包含8個「字詞」，這組存了語料庫的\*詞彙(vocabulary)\* ，詞袋模型使用一個特徵向量(feature vector)來表示每個文件(document)，所以每個文件由「包含8個元素的向量」來表示。組成一個特徵量的元素數量稱為\*向量維度(vector&rsquo;s dimension)\*，一個字典(dictionary)會把「詞彙」映射到feature vector的index。
    
         1  from sklearn.feature_extraction.text import CountVectorizer
         2  
         3  corpus = [
         4      'UNC played Duke in basketball',
         5      'Duke lost the basetball game'
         6  ]
         7  vectorizer = CountVectorizer()
         8  print(vectorizer.fit_transform(corpus).todense())
         9  print(vectorizer.vocabulary_)
        10  # 加入第三份文件
        11  corpus.append('I ate a sandwich')
        12  print(vectorizer.fit_transform(corpus).todense())
        13  print(vectorizer.vocabulary_)
    
        [[0 1 1 0 1 0 1 0 1]
         [1 0 1 1 0 1 0 1 0]]
        {'unc': 8, 'played': 6, 'duke': 2, 'in': 4, 'basketball': 1, 'lost': 5, 'the': 7, 'basetball': 0, 'game': 3}
        [[0 0 1 1 0 1 0 1 0 0 1]
         [0 1 0 1 1 0 1 0 0 1 0]
         [1 0 0 0 0 0 0 0 1 0 0]]
        {'unc': 10, 'played': 7, 'duke': 3, 'in': 5, 'basketball': 2, 'lost': 6, 'the': 9, 'basetball': 1, 'game': 4, 'ate': 0, 'sandwich': 8}

2.  如何比較兩份文件的相關程度(距離)

    1.  歐幾里德距離
    
    2.  歐幾里德範數(Euclidean norm) / \(L^2\) norm
    
        \[ d = \left\| {x_0 - x_1} \right\| \]
        一個vector的Euclidean norm為該數的magnitude(量級)，:
        \[ \left\|x\right\| = \sqrt{x_1^2 + x_2^2 + ... + x_n^2 } \]
        可以使用scikit-learn函式庫的euclidean\_distance function來計算兩個或多個vector的distance:
        
             1  from sklearn.feature_extraction.text import CountVectorizer
             2  
             3  corpus = [
             4      'He is a senior high school student',
             5      'She teach English in a senior high schyool',
             6      'Queen will miss Remembrance Sunday service after spraining her back'
             7  ]
             8  vectorizer = CountVectorizer()
             9  
            10  print(vectorizer.fit_transform(corpus).todense())
            11  print(vectorizer.vocabulary_)
            12  
            13  from sklearn.metrics.pairwise import euclidean_distances
            14  X = vectorizer.fit_transform(corpus).todense()
            15  print('Distance between 1st and 2st documents:', euclidean_distances(X[0], X[1]))
            16  print('Distance between 1st and 3st documents:', euclidean_distances(X[0], X[2]))
            17  print('Distance between 2st and 3st documents:', euclidean_distances(X[1], X[2]))
        
            [[0 0 0 1 0 1 0 1 0 0 0 1 0 1 0 0 0 1 0 0 0]
             [0 0 1 0 0 1 1 0 0 0 0 0 1 1 0 1 0 0 0 1 0]
             [1 1 0 0 1 0 0 0 1 1 1 0 0 0 1 0 1 0 1 0 1]]
            {'he': 3, 'is': 7, 'senior': 13, 'high': 5, 'school': 11, 'student': 17, 'she': 15, 'teach': 19, 'english': 2, 'in': 6, 'schyool': 12, 'queen': 9, 'will': 20, 'miss': 8, 'remembrance': 10, 'sunday': 18, 'service': 14, 'after': 0, 'spraining': 16, 'her': 4, 'back': 1}
            Distance between 1st and 2st documents: [[3.]]
            Distance between 1st and 3st documents: [[4.]]
            Distance between 2st and 3st documents: [[4.12310563]]
    
    3.  Curse of dimensionality
    
        可以發現，如果文件量大而且主題相去甚遠，則
        
        -   代表document的vector會包含成千上萬的元素
        -   每個vector裡會包含很多0，即稀疏向量(sparse vectors)
        
        使用這種「高維度資料」會為所有的機器學習任務帶來一些問題，即「維數災難/維度詛咒」(curse of dimensionality):
        
        1.  「高維度資料」比「低維度資料」需要更多的記憶體和計算能力，
        2.  隨著feature vector空間維度的增加，model就需要更多的「訓練資料」，以確保有足夠多的訓練實例(instance)，否則將導致「過度擬合」。
    
    4.  如何解決curse of dimensionality
    
        -   將文本均轉為小寫
        -   過濾停用字

3.  停用字過濾

    即刪除語料庫大部份文件中「經常出現的字詞」，即停用字(stop words)，如
    
    -   determiner: the, a, an等限定詞
    -   auxiliary verbs: 如do, be, will等助動詞
    -   perpositions: 如on, around, beneath等介系詞
    
    停用字通常是「虛詞/功能詞」(functional words)，CountVectorizer可以透過stop\_words關鍵字參數來過濾，它本身有一個英語停用字基本列表。
    
         1  from sklearn.feature_extraction.text import CountVectorizer
         2  
         3  corpus = [
         4      'He is a senior high school student',
         5      'She teach English in a senior high schyool',
         6      'Queen will miss Remembrance Sunday service after spraining her back'
         7  ]
         8  
         9  vectorizer = CountVectorizer(stop_words='english')
        10  X = vectorizer.fit_transform(corpus).todense()
        11  print(X)
        12  print(vectorizer.vocabulary_)
        13  from sklearn.metrics.pairwise import euclidean_distances
        14  print('Distance between 1st and 2st documents:', euclidean_distances(X[0], X[1]))
        15  print('Distance between 1st and 3st documents:', euclidean_distances(X[0], X[2]))
        16  print('Distance between 2st and 3st documents:', euclidean_distances(X[1], X[2]))
    
        [[0 1 0 0 0 1 0 1 0 0 1 0 0]
         [1 1 0 0 0 0 1 1 0 0 0 0 1]
         [0 0 1 1 1 0 0 0 1 1 0 1 0]]
        {'senior': 7, 'high': 1, 'school': 5, 'student': 10, 'teach': 12, 'english': 0, 'schyool': 6, 'queen': 3, 'miss': 2, 'remembrance': 4, 'sunday': 11, 'service': 8, 'spraining': 9}
        Distance between 1st and 2st documents: [[2.23606798]]
        Distance between 1st and 3st documents: [[3.16227766]]
        Distance between 2st and 3st documents: [[3.31662479]]

4.  詞幹提取與詞性還原(Stemming and Lemmatization)

    1.  Stemming
    
        詞干提取是去除單詞的前後綴得到詞根的過程。
        大家常見的前後詞綴有「名詞的複數」、「進行式」、「過去分詞」…
        如: plays, played, playing -> play
    
    2.  Lemmatization
    
        詞形還原是基於詞典，將單詞的複雜形態轉變成最基礎的形態。
        詞形還原不是簡單地將前後綴去掉，而是會根據詞典將單詞進行轉換。比如「drove」會轉換為「drive」。
        如: is, are , been -> be
    
    3.  Demo
    
             1  # Download wordnet
             2  import nltk
             3  nltk.download('wordnet')
             4  # Lemmatization
             5  from nltk.stem.wordnet import WordNetLemmatizer
             6  lemmatizer = WordNetLemmatizer()
             7  print(lemmatizer.lemmatize('gathering', 'v'))
             8  print(lemmatizer.lemmatize('gathering', 'n'))
             9  # Stemming
            10  from nltk.stem import PorterStemmer
            11  stemmer = PorterStemmer()
            12  print(stemmer.stem('gathering'))
            13  print(stemmer.stem('gathered'))
        
            gather
            gathering
            gather
            gather
    
    4.  還原corpus詞性
    
             1  from nltk import word_tokenize
             2  from nltk.stem import PorterStemmer
             3  from nltk.stem.wordnet import WordNetLemmatizer
             4  from nltk import pos_tag
             5  
             6  import nltk
             7  nltk.download('punkt')
             8  
             9  wordnet_tag = ['v', 'n']
            10  corpus = [
            11      'He ate the sandwiches',
            12      'Every sandwich was eaten by him'
            13  ]
            14  # Stemming
            15  stemmer = PorterStemmer()
            16  for document in corpus:
            17      print('\nStemmed: ', end='')
            18      for token in word_tokenize(document):
            19          print(stemmer.stem(token), end=', ')
            20  
            21  # lemmatization
            22  nltk.download('averaged_perceptron_tagger')
            23  def lemmatize(token, tag):
            24      if tag[0].lower() in wordnet_tag:
            25          return lemmatizer.lemmatize(token, tag[0].lower())
            26      return token
            27  
            28  lemmatizer = WordNetLemmatizer()
            29  tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]
            30  for document in tagged_corpus:
            31      print('\nLemmization: ', end='')
            32      for token, tag in document:
            33          print(lemmatize(token, tag), end=', ')
            34  print('\n',tagged_corpus)
        
            
            Stemmed: he, ate, the, sandwich, 
            Stemmed: everi, sandwich, wa, eaten, by, him, 
            Lemmization: He, eat, the, sandwich, 
            Lemmization: Every, sandwich, be, eat, by, him, 
             [[('He', 'PRP'), ('ate', 'VBD'), ('the', 'DT'), ('sandwiches', 'NNS')], [('Every', 'DT'), ('sandwich', 'NN'), ('was', 'VBD'), ('eaten', 'VBN'), ('by', 'IN'), ('him', 'PRP')]]

5.  TF-IDF權重擴充詞袋

    在「詞袋模型」中，無論字詞是否出現在document中，corpus字典中的字詞都會進行編碼，但「語法]、「字詞順序」、「詞類」都不會編碼。
    依照常理，一個字詞在document中出現的頻率應該可以代表該document與字詞的相關程度。為減輕幫feature vector建立「詞類」特徵編碼的編碼負擔，可以使用整數來表示字詞出現在文字中的次數。
    
         1  import numpy as np
         2  from sklearn.feature_extraction.text import CountVectorizer
         3  
         4  corpus = ['The dog ate a sandwich, the wizard transfigured a sandwich, and I ate a sandwich']
         5  vectorizer = CountVectorizer(stop_words='english')
         6  frequencies = np.array(vectorizer.fit_transform(corpus).todense())[0]
         7  print(frequencies)
         8  print('Token indices:', vectorizer.vocabulary_)
         9  
        10  for token, index in vectorizer.vocabulary_.items():
        11      print('Token {0:s} : {1:d} times'.format(token, frequencies[index]))
    
        [2 1 3 1 1]
        Token indices: {'dog': 1, 'ate': 0, 'sandwich': 2, 'wizard': 4, 'transfigured': 3}
        Token dog : 1 times
        Token ate : 2 times
        Token sandwich : 3 times
        Token wizard : 1 times
        Token transfigured : 1 times

6.  問題1: document長度不同(只考慮raw term frequency的問題)

    如果我們對feature vector中對「原始詞頻(raw term frequency)」進行編碼，可以為「文件的意義」提供額外的資訊，但前提是要假設所有文件都有相似的長度。許多字詞也許在兩個文件中出現相同次數，但如果這兩份文件長度差異太大，則兩份文件在意義上仍可能會有很大的差異。
    
    1.  解決之道
    
        scikit-learn函式庫中的TfidfTransformer類別可以透過將「詞頻向量矩陣」(a matrix of term frequency vectors)轉換為一個「常態化詞頻權重矩陣」(a matrix of normalized term frequency weights)來緩和上述問題。
        
        -   TfidTransformer類別對raw term frequency做平滑(smooth)處理，並對其進行\(L^2\) 常態化(normalization)。平滑化、常態化後的詞頻可以由以下公式取得:
            \[ df(t,d)=\frac{f(t,d)}{\left\|x\right\|} \]
            \(f(t,d)\) 表示「字詞」在document中出現的frequency； \(\left\|x\right\|\) 為詞頻向量的\(L^2\) 範數。
        -   除了對raw term frequency進行常態化，還可以透過計算詞頻的「對數」(logarithm)將頻數縮放到一個「有限的範圍」內，來改善feature vector.詞頻的對放縮放值(logarithmically scaled term frequencies) 公式如下：
            \[ fd(t,d)=1+\log f(t,d) \]
            當sublinear\_tf關鍵字參數設置為True時，TfidTransformer就會計算「詞頻的對數縮放值」。
        
        上述「常態化」及「對數縮放」後的詞頻就可以代表一個document中「字詞」出現的頻數，也能緩和不同文件大小的影響。但仍存在一個
        問題: feature vector的權重資訊對於比較文件意義並無幫助。

7.  問題2: feature vector的權重資訊對於比較文件意義並無幫助

    一份關於「杜克大學籃球隊」文章的語料庫，其大部份document中可能都包含了coach, trip, flop等字詞，這些字詞可能因出現太過頻繁被當成停用字，同時對計算document相似性也沒有幫助。
    
    1.  解決: IDF
    
        反向文件頻率(Inverse Document Frequency, IDF)是一種衡量一個字詞在語料庫中是否「稀有」或是「常見」的方式。反向文件頻率公式如下：
        \[ idf(t,D)=\log \frac{N}{1+|d\in D:t\in d|} \]
        其中，\(N\) 是corpus中的文件總數；\(1+\lvert d\in D: t\in d \rvert\) 是corpus中包含該字卜呞的文件總數。一個字詞的tf-idf value是其「詞頻」和「反向文件頻率」的乘積。當use\_idf關鍵字參數被設為預設值True，TfidTransformer將回傳tf-idf權重。由於tf-idf feature vector經常用於表示文本，scikit-learn函式庫提供了一個TfidVectorizer轉換器類別，它封裝了CountVectorizer類別和TfidTransformer類別，讓我們使用TfidVectorizer類別，為corpus建立tf-idf feature vector。
        
            1  from sklearn.feature_extraction.text import TfidfVectorizer
            2  
            3  corpus = [
            4      'The dog ate a sandwich and I ate a sandwich',
            5      'The wizard transfigured a sandwich'
            6  ]
            7  
            8  vectorizer = TfidfVectorizer(stop_words='english')
            9  print(vectorizer.fit_transform(corpus).todense())
        
            [[0.75458397 0.37729199 0.53689271 0.         0.        ]
             [0.         0.         0.44943642 0.6316672  0.6316672 ]]


<a id="org9f98130"></a>

### 字嵌入(word embedding)


<a id="org2a1dfb2"></a>

# DEMO

     1  # coding: utf-8
     2  import sys
     3  import json
     4  from gensim.models import doc2vec
     5  from collections import namedtuple
     6  
     7  
     8  # Load data
     9  raw_doc = ["I love machine learning. Its awesome.",
    10          "I love coding in python",
    11          "I love building chatbots",
    12          "they chat amagingly well"]
    13  
    14  
    15  # Preprocess
    16  docs = []
    17  analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    18  for index, text in enumerate(raw_doc):
    19      words = text.split()
    20      docs.append(analyzedDocument(words, [index]))
    21  
    22  
    23  # Train
    24  model = doc2vec.Doc2Vec(docs, vector_size=20, window=300, min_count=1, workers=4, dm=1)
    25  
    26  
    27  # Save
    28  model.save('doc2vec.model')
    29  
    30  
    31  # Load
    32  model = doc2vec.Doc2Vec.load('doc2vec.model')
    33  print(model.docvecs[1].shape)
    34  print(model.docvecs[1])

    (20,)
    [-0.01887621  0.01302744 -0.02849153  0.0131293   0.02904045 -0.04056058
     -0.04169167 -0.04981408  0.02464924 -0.04563693  0.02922578  0.0340294
     -0.03257207 -0.02262998 -0.00628912  0.00822178 -0.00741135 -0.04270934
     -0.01805406  0.00866313]


<a id="org2751d6a"></a>

# jeiba DEMO

-   <https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html>
-   [Word Embedding 編碼矩陣](https://medium.com/ml-note/word-embedding-3ca60663999d)
-   [DOC2VEC gensim tutorial](https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5)
-   [Doc2Vec模型的介紹與gensim中Doc2Vec的使用](https://www.twblogs.net/a/5db38150bd9eee310ee69608)
-   [如何訓練 Doc2Vec 模型](https://clay-atlas.com/blog/2020/07/14/python-cn-nlp-gensim-doc2vec-model/)

     1  import jieba.posseg as pseg
     2  
     3  text = '我是王小明，在台南讀書的學生，我喜歡喝咖啡、不喜歡讀書'
     4  words = pseg.cut(text)
     5  wordList = []
     6  for w, f in words:
     7      wordList.append(w)
     8      print(w, f)
     9  
    10  print(wordList)
    11  word_index = {
    12      word: idx
    13      for idx, word in enumerate(wordList)
    14  }
    15  print(word_index)
    16  print(wordList)
    17  print([word_index[w] for w in wordList])

    我 r
    是 v
    王小明 nr
    ， x
    在 p
    台南 ns
    讀書 n
    的 uj
    學生 n
    ， x
    我 r
    喜歡 v
    喝咖啡 nr
    、 x
    不 d
    喜歡 v
    讀書 n
    ['我', '是', '王小明', '，', '在', '台南', '讀書', '的', '學生', '，', '我', '喜歡', '喝咖啡', '、', '不', '喜歡', '讀書']
    {'我': 10, '是': 1, '王小明': 2, '，': 9, '在': 4, '台南': 5, '讀書': 16, '的': 7, '學生': 8, '喜歡': 15, '喝咖啡': 12, '、': 13, '不': 14}
    ['我', '是', '王小明', '，', '在', '台南', '讀書', '的', '學生', '，', '我', '喜歡', '喝咖啡', '、', '不', '喜歡', '讀書']
    [10, 1, 2, 9, 4, 5, 16, 7, 8, 9, 10, 15, 12, 13, 14, 15, 16]

     1  # coding: utf-8
     2  import sys
     3  import json
     4  from gensim.models import doc2vec
     5  from collections import namedtuple
     6  
     7  
     8  # Load data
     9  raw_doc = ["I love machine learning. Its awesome.",
    10          "I love coding in python",
    11          "I love building chatbots",
    12          "they chat amagingly well"]
    13  
    14  
    15  # Preprocess
    16  docs = []
    17  analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    18  for index, text in enumerate(raw_doc):
    19      words = text.split()
    20      docs.append(analyzedDocument(words, [index]))
    21  
    22  
    23  # Train
    24  model = doc2vec.Doc2Vec(docs, vector_size=20, window=300, min_count=1, workers=4, dm=1)
    25  
    26  
    27  # Save
    28  model.save('doc2vec.model')
    29  
    30  
    31  # Load
    32  model = doc2vec.Doc2Vec.load('doc2vec.model')
    33  print(model.docvecs[1].shape)
    34  print(model.docvecs[1])

    (20,)
    [-0.01887621  0.01302744 -0.02849153  0.0131293   0.02904045 -0.04056058
     -0.04169167 -0.04981408  0.02464924 -0.04563693  0.02922578  0.0340294
     -0.03257207 -0.02262998 -0.00628912  0.00822178 -0.00741135 -0.04270934
     -0.01805406  0.00866313]


<a id="org3453e56"></a>

# jeiba DEMO

-   <https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html>
-   [Word Embedding 編碼矩陣](https://medium.com/ml-note/word-embedding-3ca60663999d)
-   [DOC2VEC gensim tutorial](https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5)
-   [Doc2Vec模型的介紹與gensim中Doc2Vec的使用](https://www.twblogs.net/a/5db38150bd9eee310ee69608)
-   [如何訓練 Doc2Vec 模型](https://clay-atlas.com/blog/2020/07/14/python-cn-nlp-gensim-doc2vec-model/)

     1  import jieba.posseg as pseg
     2  
     3  text = '我是王小明，在台南讀書的學生，我喜歡喝咖啡、不喜歡讀書'
     4  words = pseg.cut(text)
     5  wordList = []
     6  for w, f in words:
     7      wordList.append(w)
     8      print(w, f)
     9  
    10  print(wordList)
    11  word_index = {
    12      word: idx
    13      for idx, word in enumerate(wordList)
    14  }
    15  print(word_index)
    16  print(wordList)
    17  print([word_index[w] for w in wordList])

    我 r
    是 v
    王小明 nr
    ， x
    在 p
    台南 ns
    讀書 n
    的 uj
    學生 n
    ， x
    我 r
    喜歡 v
    喝咖啡 nr
    、 x
    不 d
    喜歡 v
    讀書 n
    ['我', '是', '王小明', '，', '在', '台南', '讀書', '的', '學生', '，', '我', '喜歡', '喝咖啡', '、', '不', '喜歡', '讀書']
    {'我': 10, '是': 1, '王小明': 2, '，': 9, '在': 4, '台南': 5, '讀書': 16, '的': 7, '學生': 8, '喜歡': 15, '喝咖啡': 12, '、': 13, '不': 14}
    ['我', '是', '王小明', '，', '在', '台南', '讀書', '的', '學生', '，', '我', '喜歡', '喝咖啡', '、', '不', '喜歡', '讀書']
    [10, 1, 2, 9, 4, 5, 16, 7, 8, 9, 10, 15, 12, 13, 14, 15, 16]

