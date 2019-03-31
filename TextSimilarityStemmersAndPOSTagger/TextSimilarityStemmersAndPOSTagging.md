
## Text Similarity, Stemmers and POS Tagging
### Author: Kevin Okiah
**02/23/2019**

This document explores text similarity using Levenshtein distance measures, application of different NLTK Stemmers and POS tagging using StanfordPos Tagger,NLTK POS Tagger and Unigram Tagger

1.	Compare your given name with your nickname (if you don’t have a nickname, invent one for this assignment) by answering the following questions:

> a. What is the edit distance between your nickname and your given name?

> b. What is the percentage string match between your nickname and your given name?

Show your work for both calculations.




```python
# -*- coding: utf-8 -*-
import numpy as np
import Levenshtein
import future
import TextCleaningToolkit
from TextCleaningToolkit import *
import pandas as pd
import nltk
from nltk.corpus import brown
from nltk.corpus import state_union
from nltk import UnigramTagger
from nltk import BrillTagger
from nltk import BigramTagger
from nltk import DefaultTagger
from nltk import NgramTagger
from nltk import pos_tag##??
from nltk import PerceptronTagger
from nltk import StanfordPOSTagger
from nltk.tag.stanford import CoreNLPNERTagger
from nltk.tokenize import word_tokenize
import os
wd =os.getcwd()
```


```python
Name = "Kevin"
NickName = "Kev"
'''
distance is the number of letters that need to be swapped for the Name and nickname to match in my case
it is two letters 'i' and 'n'
'''
Distance = 2 

# "% similarity"  -  Used levenstein disctance measure to calculate percentage similarity
Levenshtein.ratio(Name, NickName)*100
```




    75.0



2.	Find a friend (or family member or classmate) who you know has read a certain book. Without your friend knowing, copy the first two sentences of that book. Now rewrite the words from those sentences, excluding stop words. Now tell your friend to guess which book the words are from by reading them just that list of words. Did you friend correctly guess the book on the first try? What did he or she guess? Explain why you think you friend either was or was not able to guess the book from hearing the list of words. 




```python
#Book: art-of-the-start-guy-kawasaki

sentences = "There are many ways to describe the ebb and flow, yin and yang, bubble blowing \
and bubble bursting phases of business cycles. Here's another one: microscopes and telescopes. \
In the microscope phase, there's a cry for level headed thinking, a return to fundamentals, and going \
'back to basics.'"
```


```python
Tokens =[]
Tokens = Tokens + Tokenizer_Tool(sentences.lower(),'word_tokenize') #plint sentences into tokens
words = [word for word in Tokens if word.isalpha()]#Remove punctuations
CleanTokens = remove_stopwords(words)#Remove stopwords
```


```python
CleanTokens
```




    ['many',
     'ways',
     'describe',
     'ebb',
     'flow',
     'yin',
     'yang',
     'bubble',
     'blowing',
     'bubble',
     'bursting',
     'phases',
     'business',
     'cycles',
     'another',
     'one',
     'microscopes',
     'telescopes',
     'microscope',
     'phase',
     'cry',
     'level',
     'headed',
     'thinking',
     'return',
     'fundamentals',
     'going',
     'basics']



Yes

My friend was still able to guess what the book was.

My friend could  identify the book by me just reading the tokens from the two sentences as he could spot familiar words from the book. He could associate the words in context and descern they are releated to the book art-of-the-start-guy-kawasaki which she had read.

3.	Run one of the stemmers available in Python. Run the same two sentences from question 2 above through the stemmer and show the results. How many of the outputted stems are valid morphological roots of the corresponding words? Express this answer as a percentage.



```python
# for this question I am using stemming function from my LexicalDiversityToolkit displayed below

import inspect
code, line_no = inspect.getsourcelines(stemming)
print(''.join(code))
```

    def stemming(tokens, Type = 'ps', rgxRule ='ing$|s$|ed$', MIN=4):
        '''
        Code adopted from text Text-analytics-with-python-a-practical-dipanjan-sarkar
        this function stems the tokens to get the root
        Stemmers: 
           - LancasterStemmer 
           - RegexpStemmer #user defined rules
           - SnowballStemmer # can stem other languages
           - PorterStemmer
        '''
        stemmers ={'ps':PorterStemmer(), 'ls':LancasterStemmer(),
                   'sn':SnowballStemmer("english"), 'rg': RegexpStemmer(rgxRule, MIN)}
        stemmer = stemmers[Type]
        stemmed_list =[]
        for i in tokens:
            stemmed_list = stemmed_list+[stemmer.stem(i)]
        return stemmed_list
    



```python
#Compare stemmers
Stemmers_Compare = pd.DataFrame()
LancasterStemmer=stemming(CleanTokens, 'ls') #lancasterStemmer
RegexpStemmer=stemming(CleanTokens, 'rg') #lancasterStemmer
PorterStemmer=stemming(CleanTokens, 'ps') #lancasterStemmer
Lema =lemming(CleanTokens)
```


```python
Stemmers_Compare =pd.DataFrame({"CleanTokens":CleanTokens,"LancasterStemmer":LancasterStemmer,"RegrexStemmer": RegexpStemmer,"PortersStemmer": PorterStemmer,"Lematization":Lema })
```


```python
Stemmers_Compare
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CleanTokens</th>
      <th>LancasterStemmer</th>
      <th>Lematization</th>
      <th>PortersStemmer</th>
      <th>RegrexStemmer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>many</td>
      <td>many</td>
      <td>many</td>
      <td>mani</td>
      <td>many</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ways</td>
      <td>way</td>
      <td>way</td>
      <td>way</td>
      <td>way</td>
    </tr>
    <tr>
      <th>2</th>
      <td>describe</td>
      <td>describ</td>
      <td>describe</td>
      <td>describ</td>
      <td>describe</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ebb</td>
      <td>eb</td>
      <td>ebb</td>
      <td>ebb</td>
      <td>ebb</td>
    </tr>
    <tr>
      <th>4</th>
      <td>flow</td>
      <td>flow</td>
      <td>flow</td>
      <td>flow</td>
      <td>flow</td>
    </tr>
    <tr>
      <th>5</th>
      <td>yin</td>
      <td>yin</td>
      <td>yin</td>
      <td>yin</td>
      <td>yin</td>
    </tr>
    <tr>
      <th>6</th>
      <td>yang</td>
      <td>yang</td>
      <td>yang</td>
      <td>yang</td>
      <td>yang</td>
    </tr>
    <tr>
      <th>7</th>
      <td>bubble</td>
      <td>bubbl</td>
      <td>bubble</td>
      <td>bubbl</td>
      <td>bubble</td>
    </tr>
    <tr>
      <th>8</th>
      <td>blowing</td>
      <td>blow</td>
      <td>blowing</td>
      <td>blow</td>
      <td>blow</td>
    </tr>
    <tr>
      <th>9</th>
      <td>bubble</td>
      <td>bubbl</td>
      <td>bubble</td>
      <td>bubbl</td>
      <td>bubble</td>
    </tr>
    <tr>
      <th>10</th>
      <td>bursting</td>
      <td>burst</td>
      <td>bursting</td>
      <td>burst</td>
      <td>burst</td>
    </tr>
    <tr>
      <th>11</th>
      <td>phases</td>
      <td>phas</td>
      <td>phase</td>
      <td>phase</td>
      <td>phase</td>
    </tr>
    <tr>
      <th>12</th>
      <td>business</td>
      <td>busy</td>
      <td>business</td>
      <td>busi</td>
      <td>busines</td>
    </tr>
    <tr>
      <th>13</th>
      <td>cycles</td>
      <td>cyc</td>
      <td>cycle</td>
      <td>cycl</td>
      <td>cycle</td>
    </tr>
    <tr>
      <th>14</th>
      <td>another</td>
      <td>anoth</td>
      <td>another</td>
      <td>anoth</td>
      <td>another</td>
    </tr>
    <tr>
      <th>15</th>
      <td>one</td>
      <td>on</td>
      <td>one</td>
      <td>one</td>
      <td>one</td>
    </tr>
    <tr>
      <th>16</th>
      <td>microscopes</td>
      <td>microscop</td>
      <td>microscope</td>
      <td>microscop</td>
      <td>microscope</td>
    </tr>
    <tr>
      <th>17</th>
      <td>telescopes</td>
      <td>telescop</td>
      <td>telescope</td>
      <td>telescop</td>
      <td>telescope</td>
    </tr>
    <tr>
      <th>18</th>
      <td>microscope</td>
      <td>microscop</td>
      <td>microscope</td>
      <td>microscop</td>
      <td>microscope</td>
    </tr>
    <tr>
      <th>19</th>
      <td>phase</td>
      <td>phas</td>
      <td>phase</td>
      <td>phase</td>
      <td>phase</td>
    </tr>
    <tr>
      <th>20</th>
      <td>cry</td>
      <td>cry</td>
      <td>cry</td>
      <td>cri</td>
      <td>cry</td>
    </tr>
    <tr>
      <th>21</th>
      <td>level</td>
      <td>level</td>
      <td>level</td>
      <td>level</td>
      <td>level</td>
    </tr>
    <tr>
      <th>22</th>
      <td>headed</td>
      <td>head</td>
      <td>head</td>
      <td>head</td>
      <td>head</td>
    </tr>
    <tr>
      <th>23</th>
      <td>thinking</td>
      <td>think</td>
      <td>think</td>
      <td>think</td>
      <td>think</td>
    </tr>
    <tr>
      <th>24</th>
      <td>return</td>
      <td>return</td>
      <td>return</td>
      <td>return</td>
      <td>return</td>
    </tr>
    <tr>
      <th>25</th>
      <td>fundamentals</td>
      <td>funda</td>
      <td>fundamental</td>
      <td>fundament</td>
      <td>fundamental</td>
    </tr>
    <tr>
      <th>26</th>
      <td>going</td>
      <td>going</td>
      <td>go</td>
      <td>go</td>
      <td>go</td>
    </tr>
    <tr>
      <th>27</th>
      <td>basics</td>
      <td>bas</td>
      <td>basic</td>
      <td>basic</td>
      <td>basic</td>
    </tr>
  </tbody>
</table>
</div>




```python
Stemmers_Compare.shape
```




    (28, 5)




```python
# lancaster Stemmer
n = 28
invalid_l = 14
print("Percentage correct from lancaster Stemmer")
print(100*(n-invalid_l)/n, "%")
```

    Percentage correct from lancaster Stemmer
    50.0 %



```python
#Regex Stemmer
invalid_r = 1
print("Percentage correct from Regex Stemmer")
print(100*(n-invalid_r)/n, "%")
```

    Percentage correct from Regex Stemmer
    96.42857142857143 %



```python
#porters Stemmer
invalid_p = 11
print("Percentage correct from porters Stemmer")
print(100*(n-invalid_p)/n, "%")
```

    Percentage correct from porters Stemmer
    60.714285714285715 %


Regex is the beststremmer  of the three that I have compared above.

I had some issues integrating pattern3 with python 3 to use it a tagger. I installed python 2 in my stem but they were conflicting with python 3 for my Capstone project. I plan to create different environments to be able to switch between python3 and python2 as needed.

**Homework 4**

1.	Run one of the part-of-speech (POS) taggers available in Python. 

> a. Find the longest sentence you can, longer than 10 words, that the POS tagger tags correctly. Show the input and output.





```python
Long_sentense = "In response, I made a conscious and fundamentally bad decision to distance myself from who I was"
tokens_L = nltk.word_tokenize(Long_sentense)
Short_sentense ="The complex houses married and single soldiers and their families."
tokens_S = nltk.word_tokenize(Short_sentense)
```


```python
#Pos tagging Long sentence 
print('---------Long Sentence ---------')
print(Long_sentense)
print('---------Pos Tagging------------')
nltk.pos_tag(tokens_L)
```

    ---------Long Sentence ---------
    In response, I made a conscious and fundamentally bad decision to distance myself from who I was
    ---------Pos Tagging------------





    [('In', 'IN'),
     ('response', 'NN'),
     (',', ','),
     ('I', 'PRP'),
     ('made', 'VBD'),
     ('a', 'DT'),
     ('conscious', 'JJ'),
     ('and', 'CC'),
     ('fundamentally', 'RB'),
     ('bad', 'JJ'),
     ('decision', 'NN'),
     ('to', 'TO'),
     ('distance', 'VB'),
     ('myself', 'PRP'),
     ('from', 'IN'),
     ('who', 'WP'),
     ('I', 'PRP'),
     ('was', 'VBD')]



> b. Find the shortest sentence you can, shorter than 10 words, that the POS tagger fails to tag 100 percent correctly. Show the input and output. Explain your conjecture as to why the tagger might have been less than perfect with this sentence.


```python
#Pos tagging Long sentence 
print('---------Short Sentence----------')
print(Short_sentense)
print('---------Pos Tagging---------------')
nltk.pos_tag(tokens_S)
```

    ---------Short Sentence----------
    The complex houses married and single soldiers and their families.
    ---------Pos Tagging---------------





    [('The', 'DT'),
     ('complex', 'JJ'),
     ('houses', 'NNS'),
     ('married', 'VBD'),
     ('and', 'CC'),
     ('single', 'JJ'),
     ('soldiers', 'NNS'),
     ('and', 'CC'),
     ('their', 'PRP$'),
     ('families', 'NNS'),
     ('.', '.')]



Here, "complex" is a noun (a housing complex) instead of an adjective, "houses" is a verb instead of a noun, and "married" is an adjective instead of the past tense of a verb.

The corpus used to train the tagger is not similar to my thus  the tagger fails to tag my text because the context, style is all very different.

reference: http://mentalfloss.com/article/49238/7-sentences-sound-crazy-are-still-grammatical 

2.	Run a different POS tagger in Python. Process the same two sentences from question 1.

> a. Does it produce the same or different output?






```python
def Compare_taggers(tokens):
    #tokens = nltk.word_tokenize(Long_sentense) #parse sentense
    brown_tagged_sents = brown.tagged_sents(categories='news') #training set
    summary = {}

    unigram_tagger = UnigramTagger(brown_tagged_sents)# train unigram tagger
    bigram_tagger = BigramTagger(brown_tagged_sents)# train Bigramtagger
    Stanfordpos_tagger = StanfordPOSTagger(wd+'/stanford-postagger/models/wsj-0-18-bidirectional-distsim.tagger',
                           wd+'/stanford-postagger/stanford-postagger.jar', encoding='utf-8')

    #print("-----------", 'Pos_tagger' ,"----------------- ")
    #print(nltk.pos_tag(tokens),'\n')
    ps = nltk.pos_tag(tokens)
    
    #print("-----------", 'unigram_tagger' ,"-----------------")
    #print(unigram_tagger.tag(tokens), '\n')
    un =unigram_tagger.tag(tokens)

    #print("-----------", 'StanfordPos _tagger' ,"----------------- ")
    #print(Stanfordpos_tagger.tag(tokens),'\n')
    st = Stanfordpos_tagger.tag(tokens)
    
    summary.update({'pos_tagger':dict((y, x) for y, x in ps)})
    summary.update({'unigram_tagger':dict((y, x) for y, x in un)})
    summary.update({'StanfordPos_tagger':dict((y, x) for y, x in st)})
    return(pd.DataFrame.from_dict(summary))
```


```python
#Pos tagging Long sentence 
print('----------------------------------')
print(Long_sentense)
print('----------------------------------')
Compare_taggers(tokens_L)
```

    ----------------------------------
    In response, I made a conscious and fundamentally bad decision to distance myself from who I was
    ----------------------------------


    /home/kevimwe/anaconda3/lib/python3.6/site-packages/nltk/tag/stanford.py:149: DeprecationWarning: 
    The StanfordTokenizer will be deprecated in version 3.2.5.
    Please use [91mnltk.tag.corenlp.CoreNLPPOSTagger[0m or [91mnltk.tag.corenlp.CoreNLPNERTagger[0m instead.
      super(StanfordPOSTagger, self).__init__(*args, **kwargs)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StanfordPos_tagger</th>
      <th>pos_tagger</th>
      <th>unigram_tagger</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>,</th>
      <td>,</td>
      <td>,</td>
      <td>,</td>
    </tr>
    <tr>
      <th>I</th>
      <td>PRP</td>
      <td>PRP</td>
      <td>PPSS</td>
    </tr>
    <tr>
      <th>In</th>
      <td>IN</td>
      <td>IN</td>
      <td>IN</td>
    </tr>
    <tr>
      <th>a</th>
      <td>DT</td>
      <td>DT</td>
      <td>AT</td>
    </tr>
    <tr>
      <th>and</th>
      <td>CC</td>
      <td>CC</td>
      <td>CC</td>
    </tr>
    <tr>
      <th>bad</th>
      <td>JJ</td>
      <td>JJ</td>
      <td>JJ</td>
    </tr>
    <tr>
      <th>conscious</th>
      <td>JJ</td>
      <td>JJ</td>
      <td>JJ</td>
    </tr>
    <tr>
      <th>decision</th>
      <td>NN</td>
      <td>NN</td>
      <td>NN</td>
    </tr>
    <tr>
      <th>distance</th>
      <td>VB</td>
      <td>VB</td>
      <td>NN</td>
    </tr>
    <tr>
      <th>from</th>
      <td>IN</td>
      <td>IN</td>
      <td>IN</td>
    </tr>
    <tr>
      <th>fundamentally</th>
      <td>RB</td>
      <td>RB</td>
      <td>QL</td>
    </tr>
    <tr>
      <th>made</th>
      <td>VBD</td>
      <td>VBD</td>
      <td>VBN</td>
    </tr>
    <tr>
      <th>myself</th>
      <td>PRP</td>
      <td>PRP</td>
      <td>PPL</td>
    </tr>
    <tr>
      <th>response</th>
      <td>NN</td>
      <td>NN</td>
      <td>NN</td>
    </tr>
    <tr>
      <th>to</th>
      <td>TO</td>
      <td>TO</td>
      <td>TO</td>
    </tr>
    <tr>
      <th>was</th>
      <td>VBD</td>
      <td>VBD</td>
      <td>BEDZ</td>
    </tr>
    <tr>
      <th>who</th>
      <td>WP</td>
      <td>WP</td>
      <td>WPS</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Pos tagging Short sentence 
print('----------------------------------')
print(Short_sentense)
print('----------------------------------')
Compare_taggers(tokens_S)
```

    ----------------------------------
    The complex houses married and single soldiers and their families.
    ----------------------------------


    /home/kevimwe/anaconda3/lib/python3.6/site-packages/nltk/tag/stanford.py:149: DeprecationWarning: 
    The StanfordTokenizer will be deprecated in version 3.2.5.
    Please use [91mnltk.tag.corenlp.CoreNLPPOSTagger[0m or [91mnltk.tag.corenlp.CoreNLPNERTagger[0m instead.
      super(StanfordPOSTagger, self).__init__(*args, **kwargs)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StanfordPos_tagger</th>
      <th>pos_tagger</th>
      <th>unigram_tagger</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>.</th>
      <td>.</td>
      <td>.</td>
      <td>.</td>
    </tr>
    <tr>
      <th>The</th>
      <td>DT</td>
      <td>DT</td>
      <td>AT</td>
    </tr>
    <tr>
      <th>and</th>
      <td>CC</td>
      <td>CC</td>
      <td>CC</td>
    </tr>
    <tr>
      <th>complex</th>
      <td>JJ</td>
      <td>JJ</td>
      <td>JJ</td>
    </tr>
    <tr>
      <th>families</th>
      <td>NNS</td>
      <td>NNS</td>
      <td>NNS</td>
    </tr>
    <tr>
      <th>houses</th>
      <td>NNS</td>
      <td>NNS</td>
      <td>NNS</td>
    </tr>
    <tr>
      <th>married</th>
      <td>JJ</td>
      <td>VBD</td>
      <td>VBN</td>
    </tr>
    <tr>
      <th>single</th>
      <td>JJ</td>
      <td>JJ</td>
      <td>AP</td>
    </tr>
    <tr>
      <th>soldiers</th>
      <td>NNS</td>
      <td>NNS</td>
      <td>NNS</td>
    </tr>
    <tr>
      <th>their</th>
      <td>PRP$</td>
      <td>PRP$</td>
      <td>PP$</td>
    </tr>
  </tbody>
</table>
</div>



Different taggers produce different results.

> b. Explain any differences as best you can.

The results could depend on the tag training set used. NLTK.Pos_tagger and StanfordPosTagger are trained and tested on the Wall Street Journal corpus where as Unigram_Tagger that I have is is trained on Brown Corpus.

It is interesting that for the the long sentence, the three taggers are able to tag the words the same but for . Unigram_Tagger struggles with the short sentense where are Pos_Tagger and StanfordPosTagger tag the sentence the same which cements our argument the Corpus used is the primary driver of how the Taggers perform.

3.	In a news article from this week’s news, find a random sentence of at least 10 words.

> a. Looking at the Penn tag set, manually POS tag the sentence yourself.


Ref: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

Article: https://www.cnn.com/2019/02/26/media/att-time-warner-merger-ruling/index.html




```python
sentence = "Leon, the judge who ruled against the Justice Department at trial, was appointed by President George W. Bush."
```


```python
sentence
```




    'Leon, the judge who ruled against the Justice Department at trial, was appointed by President George W. Bush.'


Leon= NNP,
the= DT,
judge= NN,
who = WP,
ruled =VBD,
against =IN,
the =NN,
Justice = NNP,
Department =NNP,
at = IN,
trial=NN,
was=VBD,
appointed=VBN,
by= IN,
President= NNP,
George  = NNP
W. =NNP, 
Bush=NNP,
> b. Now run the same sentences through both taggers that you implemented for questions 1 and 2. Did either of the taggers produce the same results as you had created manually?



```python
sentence_token = nltk.word_tokenize(sentence)
```


```python
Compare_taggers(sentence_token)
```

    /home/kevimwe/anaconda3/lib/python3.6/site-packages/nltk/tag/stanford.py:149: DeprecationWarning: 
    The StanfordTokenizer will be deprecated in version 3.2.5.
    Please use [91mnltk.tag.corenlp.CoreNLPPOSTagger[0m or [91mnltk.tag.corenlp.CoreNLPNERTagger[0m instead.
      super(StanfordPOSTagger, self).__init__(*args, **kwargs)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StanfordPos_tagger</th>
      <th>pos_tagger</th>
      <th>unigram_tagger</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>,</th>
      <td>,</td>
      <td>,</td>
      <td>,</td>
    </tr>
    <tr>
      <th>.</th>
      <td>.</td>
      <td>.</td>
      <td>.</td>
    </tr>
    <tr>
      <th>Bush</th>
      <td>NNP</td>
      <td>NNP</td>
      <td>NP</td>
    </tr>
    <tr>
      <th>Department</th>
      <td>NNP</td>
      <td>NNP</td>
      <td>NN-TL</td>
    </tr>
    <tr>
      <th>George</th>
      <td>NNP</td>
      <td>NNP</td>
      <td>NP</td>
    </tr>
    <tr>
      <th>Justice</th>
      <td>NNP</td>
      <td>NNP</td>
      <td>NN-TL</td>
    </tr>
    <tr>
      <th>Leon</th>
      <td>NNP</td>
      <td>NNP</td>
      <td>NP</td>
    </tr>
    <tr>
      <th>President</th>
      <td>NNP</td>
      <td>NNP</td>
      <td>NN-TL</td>
    </tr>
    <tr>
      <th>W.</th>
      <td>NNP</td>
      <td>NNP</td>
      <td>NP</td>
    </tr>
    <tr>
      <th>against</th>
      <td>IN</td>
      <td>IN</td>
      <td>IN</td>
    </tr>
    <tr>
      <th>appointed</th>
      <td>VBN</td>
      <td>VBN</td>
      <td>VBN</td>
    </tr>
    <tr>
      <th>at</th>
      <td>IN</td>
      <td>IN</td>
      <td>IN</td>
    </tr>
    <tr>
      <th>by</th>
      <td>IN</td>
      <td>IN</td>
      <td>IN</td>
    </tr>
    <tr>
      <th>judge</th>
      <td>NN</td>
      <td>NN</td>
      <td>NN</td>
    </tr>
    <tr>
      <th>ruled</th>
      <td>VBD</td>
      <td>VBD</td>
      <td>VBD</td>
    </tr>
    <tr>
      <th>the</th>
      <td>DT</td>
      <td>DT</td>
      <td>AT</td>
    </tr>
    <tr>
      <th>trial</th>
      <td>NN</td>
      <td>NN</td>
      <td>NN</td>
    </tr>
    <tr>
      <th>was</th>
      <td>VBD</td>
      <td>VBD</td>
      <td>BEDZ</td>
    </tr>
    <tr>
      <th>who</th>
      <td>WP</td>
      <td>WP</td>
      <td>WPS</td>
    </tr>
  </tbody>
</table>
</div>



> c. Explain any differences between the two taggers and your manual tagging as much as you can.

Similar to explanation in part two, pos_tagger and StanfordPosTagger tag the sentence the same as there were trained on the same Corpus where as unigram_tagger	 has some varition in tags nfor some words compared to the previous two.

My tagging and that of the pos_tagger and StanfordPosTagger match.


```python
#Session info
```


```python
%load_ext version_information
%version_information pandas, numpy, nltk, re, contractions
```




<table><tr><th>Software</th><th>Version</th></tr><tr><td>Python</td><td>3.6.8 64bit [GCC 7.3.0]</td></tr><tr><td>IPython</td><td>7.2.0</td></tr><tr><td>OS</td><td>Linux 4.15.0 46 generic x86_64 with debian buster sid</td></tr><tr><td>pandas</td><td>0.20.3</td></tr><tr><td>numpy</td><td>1.15.0</td></tr><tr><td>nltk</td><td>3.2.5</td></tr><tr><td>re</td><td>2.2.1</td></tr><tr><td>contractions</td><td>0.0.17</td></tr><tr><td colspan='2'>Sun Mar 31 18:27:14 2019 CDT</td></tr></table>


