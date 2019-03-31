
## Distance Measures.
### Author: Kevin Okiah

#### 03/17/2019

### 1.	Evaluate text similarity of Amazon book search results by doing the following:

> a.	Do a book search on Amazon. Manually copy the full book title (including subtitle) of each of the top 24 books listed in the first two pages of search results. 

> b.	In Python, run one of the text-similarity measures covered in this course, e.g., cosine similarity. Compare each of the book titles, pairwise, to every other one. 

> c.	Which two titles are the most similar to each other? Which are the most dissimilar? Where do they rank, among the first 24 results?



```python
import numpy as np
import pandas as pd
import selenium
from lxml import html
import urllib3
from bs4 import BeautifulSoup
import lxml
import urllib 
import nltk
import string
from urllib3 import request
from string import punctuation
from TextCleaningToolkit import *
import TextCleaningToolkit
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

http = urllib3.PoolManager()

def get_url_Bs(url):
    tree = BeautifulSoup(url)
    return tree

def get_url_Sel(url):
    tree = html.document_fromstring(url)
    return tree
```
### *** Commented thid section out to prevenet scrapping multiple times. The data is pickled for reproducability

#book Data science
'''
Pull books from Amazon. Target title is  Deep Learning with Pythonby Francois Chollet 
'''

url_pg1 = "https://www.amazon.com/s?k=deep+learning+with+python&crid=3V00KB95YFL3I&sprefix=Deep+learning%2Caps%2C192&ref=nb_sb_ss_i_3_13"
url_pg2 = "https://www.amazon.com/s?k=deep+learning+with+python&page=2&crid=3V00KB95YFL3I&qid=1552836600&sprefix=Deep+learning%2Caps%2C192&ref=sr_pg_2"

Title_div = "//span[3]/div[1]/div/div/div/div/div/div/div/div/div/div/div/h5/a/span"
myurl = http.request('GET', url_pg1).data
tree = get_url_Sel(myurl)
TitlesPg1 = tree.xpath(Title_div)

myurl2 = http.request('GET', url_pg2).data
tree2 = get_url_Sel(myurl2)
TitlesPg2 = tree2.xpath(Title_div)

bookTitles = []
for i in TitlesPg1:
    bookTitles =bookTitles+[i.text]
for i in TitlesPg2:
    bookTitles =bookTitles+[i.text]
    
pickle.dump(bookTitles, open( "AmazonBooks.p", "wb" ) , protocol=2) # save list as pickle

```python
with open('AmazonBooks.p', 'rb') as f:
     bookTitles = pickle.load(f)

len(bookTitles)
```




    30




```python
bookTitles
```




    [u'Deep Learning with Python',
     u'Deep Learning (Adaptive Computation and Machine Learning series)',
     u'Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems',
     u'Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow, 2nd Edition',
     u'Machine Learning with Python Cookbook: Practical Solutions from Preprocessing to Deep Learning',
     u'Deep Learning Cookbook: Practical Recipes to Get Started Quickly',
     u'Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn and Tensorflow: Step-by-Step Tutorial For Beginners.',
     u'Python Machine Learning: A Deep Dive Into Python Machine Learning and Deep Learning, Using Tensor Flow And Keras: From Beginner To Advance',
     u'Deep Learning with R',
     u"Deep Learning: A Practitioner's Approach",
     u'Python Deep Learning Projects: 9 projects demystifying neural network and deep learning models for building intelligent systems',
     u'Deep Learning With Python Illustrated Guide For Beginners And Intermediates "Learn By Doing Approach": The Future Is Here! Keras with Tensorflow Back End',
     u'Deep Learning with Python: With Natural Language Processing',
     u'Deep Learning with Keras: Implementing deep learning models and neural networks with the power of Python',
     u'Hands-On Transfer Learning with Python: Implement advanced deep learning and neural network models using TensorFlow and Keras',
     u'Natural Language Processing Recipes: Unlocking Text Data with Machine Learning and Deep Learning using Python',
     u'Hands-On Reinforcement Learning with Python: Master reinforcement and deep reinforcement learning using OpenAI Gym and TensorFlow',
     u'Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow (Step-by-Step Tutorial for Beginners)',
     u'Machine Learning for Beginners: Absolute Beginner\u2019s Guide to Understanding Machine Learning, Artificial Intelligence, Python Programming, Neural Networks Concepts and Big Data',
     u'Python Deep Learning: Next generation techniques to revolutionize computer vision, AI, speech and data analysis',
     u'Deep Learning from Scratch: From Basics to Building Real Neural Networks in Python with Keras',
     u'Deep Learning with Python: A Hands-on Introduction',
     u"Deep Learning: A Practitioner's Approach",
     u'Deep Learning with Keras: Implementing deep learning models and neural networks with the power of Python',
     u'Advanced Deep Learning with Keras: Apply deep learning techniques, autoencoders, GANs, variational autoencoders, deep reinforcement learning, policy gradients, and more',
     u'Make Your Own Neural Network',
     u'Deep Reinforcement Learning Hands-On: Apply modern RL methods, with deep Q-networks, value iteration, policy gradients, TRPO, AlphaGo Zero and more',
     u'Python Deep Learning Projects: 9 projects demystifying neural network and deep learning models for building intelligent systems',
     u'Deep Learning with Applications Using Python: Chatbots and Face, Object, and Speech Recognition With TensorFlow and Keras',
     u'Applied Deep Learning with Keras: Solve complex real-life problems with the simplicity of Keras']




```python
#leveraging Sarkar's codes
from normalization import normalize_corpus 
from utils import build_feature_matrix 
import numpy as np
```


```python
# normalize and extract features from the 32 books Titles from Amazon
norm_book_corpus = normalize_corpus(bookTitles, lemmatize=True) 
tfidf_vectorizer, tfidf_features = build_feature_matrix(norm_book_corpus,                                                         
                                                        feature_type='tfidf', 
                                                        ngram_range=(1, 1), 
                                                        min_df=0.0, 
                                                        max_df=1.0)
query_docs_tfidf = tfidf_vectorizer.transform(norm_book_corpus)
```


```python
norm_book_corpus
```




    [u'deep learning python',
     u'deep learning adaptive computation machine learning series',
     u'hands machine learning scikit learn tensorflow concept tool technique build intelligent system',
     u'python machine learning machine learning deep learning python scikit learn tensorflow 2nd edition',
     u'machine learning python cookbook practical solution preprocessing deep learning',
     u'deep learning cookbook practical recipe start quickly',
     u'python machine learning machine learning deep learning python scikit learn tensorflow step step tutorial beginner',
     u'python machine learning deep dive python machine learning deep learning use tensor flow kera beginner advance',
     u'deep learning r',
     u'deep learning practitioner approach',
     u'python deep learning project 9 project demystify neural network deep learning model build intelligent system',
     u'deep learning python illustrated guide beginner intermediate learn approach future kera tensorflow end',
     u'deep learning python natural language processing',
     u'deep learning kera implementing deep learning model neural network power python',
     u'hands transfer learning python implement advance deep learning neural network model use tensorflow kera',
     u'natural language processing recipe unlocking text data machine learning deep learning use python',
     u'hands reinforcement learning python master reinforcement deep reinforcement learn use openai gym tensorflow',
     u'python machine learning machine learning deep learning python scikit learn tensorflow step step tutorial beginner',
     u'machine learning beginner absolute beginner \u2019 guide understand machine learning artificial intelligence python programming neural network concept big data',
     u'python deep learning next generation technique revolutionize computer vision ai speech data analysis',
     u'deep learning scratch basic building real neural network python kera',
     u'deep learning python hands introduction',
     u'deep learning practitioner approach',
     u'deep learning kera implementing deep learning model neural network power python',
     u'advanced deep learning kera apply deep learning technique autoencoders gans variational autoencoders deep reinforcement learn policy gradient',
     u'neural network',
     u'deep reinforcement learning hands apply modern rl method deep q networks value iteration policy gradient trpo alphago',
     u'python deep learning project 9 project demystify neural network deep learning model build intelligent system',
     u'deep learning application use python chatbots face object speech recognition tensorflow kera',
     u'applied deep learning kera solve complex real life problem simplicity kera']




```python
tfidf_vectorizer
```




    TfidfVectorizer(analyzer=u'word', binary=False, decode_error=u'strict',
            dtype=<type 'numpy.float64'>, encoding=u'utf-8', input=u'content',
            lowercase=True, max_df=1.0, max_features=None, min_df=0.0,
            ngram_range=(1, 1), norm=u'l2', preprocessor=None, smooth_idf=True,
            stop_words=None, strip_accents=None, sublinear_tf=False,
            token_pattern=u'(?u)\\b\\w\\w+\\b', tokenizer=None, use_idf=True,
            vocabulary=None)




```python
def compute_cosine_similarity(doc_features, corpus_features, 
                              top_n=3):    
    # get document vectors    
    doc_features = doc_features.toarray()[0]    
    corpus_features = corpus_features.toarray()    
    # compute similarities    
    similarity = np.dot(doc_features,                        
                        corpus_features.T)    
    # get docs with highest similarity scores    
    top_docs = similarity.argsort()[::-1][:top_n]    
    top_docs_with_score = [(index, round(similarity[index], 3))                           
                           for index in top_docs]    
    # get docs with lowest similarity scores  
    bottom_docs = similarity.argsort()[::1][:top_n]    
    bottom_docs_with_score = [(index, round(similarity[index], 3))                           
                           for index in bottom_docs]  
    return top_docs_with_score, bottom_docs_with_score
```


```python
print 'Document Similarity Analysis using Cosine Similarity'     
print '='*100     
for index, doc in enumerate(norm_book_corpus):
    try:
        doc_tfidf = query_docs_tfidf[index] 
        top_similar_docs, bottom_similar_docs = compute_cosine_similarity(doc_tfidf, 
                                                     tfidf_features, 
                                                     top_n=1)
        print 'Document',index+1,':',norm_book_corpus[index]
        print '='*100
        print 'Most similar doc:' 
        print '-'*18
        n = len(top_similar_docs)
        for doc_index, sim_score in top_similar_docs:  
                print 'Doc num: {} Similarity Score: {}\nDoc: {}'. format(doc_index+2,sim_score, norm_book_corpus[doc_index+1]) 
                #print '='*90 

        print '-'*18
        print 'Most dissimilar doc:' 
        print '-'*18
        for doc_index, sim_score in bottom_similar_docs:  
                print 'Doc num: {} Similarity Score: {}\nDoc: {}'. format(doc_index,sim_score, norm_book_corpus[doc_index]) 
        print '='*100 
    except:
        print('Query Failed...')
```

    Document Similarity Analysis using Cosine Similarity
    ====================================================================================================
    Document 1 : deep learning python
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 2 Similarity Score: 1.0
    Doc: deep learning adaptive computation machine learning series
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 25 Similarity Score: 0.0
    Doc: neural network
    ====================================================================================================
    Document 2 : deep learning adaptive computation machine learning series
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 3 Similarity Score: 1.0
    Doc: hands machine learning scikit learn tensorflow concept tool technique build intelligent system
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 25 Similarity Score: 0.0
    Doc: neural network
    ====================================================================================================
    Document 3 : hands machine learning scikit learn tensorflow concept tool technique build intelligent system
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 4 Similarity Score: 1.0
    Doc: python machine learning machine learning deep learning python scikit learn tensorflow 2nd edition
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 25 Similarity Score: 0.0
    Doc: neural network
    ====================================================================================================
    Document 4 : python machine learning machine learning deep learning python scikit learn tensorflow 2nd edition
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 5 Similarity Score: 1.0
    Doc: machine learning python cookbook practical solution preprocessing deep learning
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 25 Similarity Score: 0.0
    Doc: neural network
    ====================================================================================================
    Document 5 : machine learning python cookbook practical solution preprocessing deep learning
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 6 Similarity Score: 1.0
    Doc: deep learning cookbook practical recipe start quickly
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 25 Similarity Score: 0.0
    Doc: neural network
    ====================================================================================================
    Document 6 : deep learning cookbook practical recipe start quickly
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 7 Similarity Score: 1.0
    Doc: python machine learning machine learning deep learning python scikit learn tensorflow step step tutorial beginner
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 25 Similarity Score: 0.0
    Doc: neural network
    ====================================================================================================
    Document 7 : python machine learning machine learning deep learning python scikit learn tensorflow step step tutorial beginner
    ====================================================================================================
    Most similar doc:
    ------------------
    Query Failed...
    Document 8 : python machine learning deep dive python machine learning deep learning use tensor flow kera beginner advance
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 9 Similarity Score: 1.0
    Doc: deep learning r
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 25 Similarity Score: 0.0
    Doc: neural network
    ====================================================================================================
    Document 9 : deep learning r
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 10 Similarity Score: 1.0
    Doc: deep learning practitioner approach
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 25 Similarity Score: 0.0
    Doc: neural network
    ====================================================================================================
    Document 10 : deep learning practitioner approach
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 11 Similarity Score: 1.0
    Doc: python deep learning project 9 project demystify neural network deep learning model build intelligent system
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 25 Similarity Score: 0.0
    Doc: neural network
    ====================================================================================================
    Document 11 : python deep learning project 9 project demystify neural network deep learning model build intelligent system
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 29 Similarity Score: 1.0
    Doc: deep learning application use python chatbots face object speech recognition tensorflow kera
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 29 Similarity Score: 0.04
    Doc: applied deep learning kera solve complex real life problem simplicity kera
    ====================================================================================================
    Document 12 : deep learning python illustrated guide beginner intermediate learn approach future kera tensorflow end
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 13 Similarity Score: 1.0
    Doc: deep learning python natural language processing
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 25 Similarity Score: 0.0
    Doc: neural network
    ====================================================================================================
    Document 13 : deep learning python natural language processing
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 14 Similarity Score: 1.0
    Doc: deep learning kera implementing deep learning model neural network power python
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 25 Similarity Score: 0.0
    Doc: neural network
    ====================================================================================================
    Document 14 : deep learning kera implementing deep learning model neural network power python
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 25 Similarity Score: 1.0
    Doc: advanced deep learning kera apply deep learning technique autoencoders gans variational autoencoders deep reinforcement learn policy gradient
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 2 Similarity Score: 0.03
    Doc: hands machine learning scikit learn tensorflow concept tool technique build intelligent system
    ====================================================================================================
    Document 15 : hands transfer learning python implement advance deep learning neural network model use tensorflow kera
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 16 Similarity Score: 1.0
    Doc: natural language processing recipe unlocking text data machine learning deep learning use python
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 5 Similarity Score: 0.045
    Doc: deep learning cookbook practical recipe start quickly
    ====================================================================================================
    Document 16 : natural language processing recipe unlocking text data machine learning deep learning use python
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 17 Similarity Score: 1.0
    Doc: hands reinforcement learning python master reinforcement deep reinforcement learn use openai gym tensorflow
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 25 Similarity Score: 0.0
    Doc: neural network
    ====================================================================================================
    Document 17 : hands reinforcement learning python master reinforcement deep reinforcement learn use openai gym tensorflow
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 18 Similarity Score: 1.0
    Doc: python machine learning machine learning deep learning python scikit learn tensorflow step step tutorial beginner
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 25 Similarity Score: 0.0
    Doc: neural network
    ====================================================================================================
    Document 18 : python machine learning machine learning deep learning python scikit learn tensorflow step step tutorial beginner
    ====================================================================================================
    Most similar doc:
    ------------------
    Query Failed...
    Document 19 : machine learning beginner absolute beginner ’ guide understand machine learning artificial intelligence python programming neural network concept big data
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 20 Similarity Score: 1.0
    Doc: python deep learning next generation technique revolutionize computer vision ai speech data analysis
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 26 Similarity Score: 0.012
    Doc: deep reinforcement learning hands apply modern rl method deep q networks value iteration policy gradient trpo alphago
    ====================================================================================================
    Document 20 : python deep learning next generation technique revolutionize computer vision ai speech data analysis
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 21 Similarity Score: 1.0
    Doc: deep learning scratch basic building real neural network python kera
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 25 Similarity Score: 0.0
    Doc: neural network
    ====================================================================================================
    Document 21 : deep learning scratch basic building real neural network python kera
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 22 Similarity Score: 1.0
    Doc: deep learning python hands introduction
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 2 Similarity Score: 0.013
    Doc: hands machine learning scikit learn tensorflow concept tool technique build intelligent system
    ====================================================================================================
    Document 22 : deep learning python hands introduction
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 23 Similarity Score: 1.0
    Doc: deep learning practitioner approach
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 25 Similarity Score: 0.0
    Doc: neural network
    ====================================================================================================
    Document 23 : deep learning practitioner approach
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 11 Similarity Score: 1.0
    Doc: python deep learning project 9 project demystify neural network deep learning model build intelligent system
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 25 Similarity Score: 0.0
    Doc: neural network
    ====================================================================================================
    Document 24 : deep learning kera implementing deep learning model neural network power python
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 25 Similarity Score: 1.0
    Doc: advanced deep learning kera apply deep learning technique autoencoders gans variational autoencoders deep reinforcement learn policy gradient
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 2 Similarity Score: 0.03
    Doc: hands machine learning scikit learn tensorflow concept tool technique build intelligent system
    ====================================================================================================
    Document 25 : advanced deep learning kera apply deep learning technique autoencoders gans variational autoencoders deep reinforcement learn policy gradient
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 26 Similarity Score: 1.0
    Doc: neural network
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 25 Similarity Score: 0.0
    Doc: neural network
    ====================================================================================================
    Document 26 : neural network
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 27 Similarity Score: 1.0
    Doc: deep reinforcement learning hands apply modern rl method deep q networks value iteration policy gradient trpo alphago
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 0 Similarity Score: 0.0
    Doc: deep learning python
    ====================================================================================================
    Document 27 : deep reinforcement learning hands apply modern rl method deep q networks value iteration policy gradient trpo alphago
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 28 Similarity Score: 1.0
    Doc: python deep learning project 9 project demystify neural network deep learning model build intelligent system
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 25 Similarity Score: 0.0
    Doc: neural network
    ====================================================================================================
    Document 28 : python deep learning project 9 project demystify neural network deep learning model build intelligent system
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 29 Similarity Score: 1.0
    Doc: deep learning application use python chatbots face object speech recognition tensorflow kera
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 29 Similarity Score: 0.04
    Doc: applied deep learning kera solve complex real life problem simplicity kera
    ====================================================================================================
    Document 29 : deep learning application use python chatbots face object speech recognition tensorflow kera
    ====================================================================================================
    Most similar doc:
    ------------------
    Doc num: 30 Similarity Score: 1.0
    Doc: applied deep learning kera solve complex real life problem simplicity kera
    ------------------
    Most dissimilar doc:
    ------------------
    Doc num: 25 Similarity Score: 0.0
    Doc: neural network
    ====================================================================================================
    Document 30 : applied deep learning kera solve complex real life problem simplicity kera
    ====================================================================================================
    Most similar doc:
    ------------------
    Query Failed...


### 2.	Now evaluate using a major search engine.

>a.	Enter one of the book titles from question 1a into Google, Bing, or Yahoo!. Copy the capsule of the first organic result and the 20th organic result. Take web results only (i.e., not video results), and skip sponsored results. 

>b.	Run the same text similarity calculation that you used for question 1b on each of these capsules in comparison to the original query (book title). 

>c.	Which one has the highest similarity measure? 



```python
#Google Search Results
book_title = ['Deep Learning with Python by Francois Chollet']
Capsule1 = ["Deep Learning with Python: Francois Chollet: 9781617294433 ...\
            https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438\
            Deep Learning with Python [Francois Chollet] on Amazon.com. *FREE* shipping \
            on qualifying offers. Summary Deep Learning with Python introduces the field ..."]
Capsule20 = ["Deep Learning with Python : Francois Chollet : 9781617294433\
            https://www.bookdepository.com/Deep-Learning-with-Python-Francois-Chollet/9781...\
            Dec 22, 2017 - Deep Learning with Python by Francois Chollet, 9781617294433,\
            available at Book Depository with free delivery worldwide."
           ]
Capsule51 = ["Deep Learning with Python by Francois Chollet (9781617294433)\
              https://www.allbookstores.com/Deep-Learning-Python-Francois-Chollet/9781617294...\
             Deep Learning with Python by Francois Chollet. Click here for the lowest price! \
             Paperback, 9781617294433, 1617294438."]

merged = book_title+Capsule1+Capsule20+Capsule51
```


```python
merged
```




    ['Deep Learning with Python by Francois Chollet',
     'Deep Learning with Python: Francois Chollet: 9781617294433 ...            https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438            Deep Learning with Python [Francois Chollet] on Amazon.com. *FREE* shipping             on qualifying offers. Summary Deep Learning with Python introduces the field ...',
     'Deep Learning with Python : Francois Chollet : 9781617294433            https://www.bookdepository.com/Deep-Learning-with-Python-Francois-Chollet/9781...            Dec 22, 2017 - Deep Learning with Python by Francois Chollet, 9781617294433,            available at Book Depository with free delivery worldwide.',
     'Deep Learning with Python by Francois Chollet (9781617294433)              https://www.allbookstores.com/Deep-Learning-Python-Francois-Chollet/9781617294...             Deep Learning with Python by Francois Chollet. Click here for the lowest price!              Paperback, 9781617294433, 1617294438.']




```python
# normalize and extract features from the 32 books Titles from Amazon
norm_book_corpus = normalize_corpus(merged, lemmatize=True) 
title_book_corpus = normalize_corpus(book_title, lemmatize=True) 
tfidf_vectorizer, tfidf_features = build_feature_matrix(norm_book_corpus,                                                         
                                                        feature_type='tfidf', 
                                                        ngram_range=(1, 1), 
                                                        min_df=0.0, 
                                                        max_df=1.0)
query_docs_tfidf = tfidf_vectorizer.transform(norm_book_corpus)
```


```python
print 'Document Similarity Analysis using Cosine Similarity'     
print '='*100     
index =0
doc_tfidf = query_docs_tfidf[index] 
top_similar_docs, bottom_similar_docs = compute_cosine_similarity(doc_tfidf, 
                                             tfidf_features, 
                                             top_n=1)
print 'Search Title',index+1,':',norm_book_corpus[index]
print '='*100
print 'Most similar doc:' 
print '-'*18
n = len(top_similar_docs)
for doc_index, sim_score in top_similar_docs:  
        print 'Search Result: {} Similarity Score: {}\nDoc: {}'. format(doc_index+1,sim_score, norm_book_corpus[doc_index+1]) 
        #print '='*90 

print '-'*18
print 'Most dissimilar doc:' 
print '-'*18
for doc_index, sim_score in bottom_similar_docs:  
        print 'Search Result: {} Similarity Score: {}\nDoc: {}'. format(doc_index+18,sim_score, norm_book_corpus[doc_index]) 
print '='*100 
```

    Document Similarity Analysis using Cosine Similarity
    ====================================================================================================
    Search Title 1 : deep learning python francois chollet
    ====================================================================================================
    Most similar doc:
    ------------------
    Search Result: 1 Similarity Score: 1.0
    Doc: deep learning python francois chollet 9781617294433 https www amazon com deep learning python francois chollet dp 1617294438 deep learning python francois chollet amazon com free ship qualify offer summary deep learning python introduce field
    ------------------
    Most dissimilar doc:
    ------------------
    Search Result: 20 Similarity Score: 0.69
    Doc: deep learning python francois chollet 9781617294433 https www bookdepository com deep learning python francois chollet 9781 dec 22 2017 deep learning python francois chollet 9781617294433 available book depository free delivery worldwide
    ====================================================================================================


 Search Result 1 has the highest Cosine Similarirt Measure
 
 **Summary:**
 
Search results that are top of the list in both Amazon and Google  have a high cosine similarity distance to the actual search.
We can conclude both amazon and google use some distance measure to rank its results.
 
