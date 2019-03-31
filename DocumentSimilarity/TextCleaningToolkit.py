#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Author: Kevin Okiah
Changes:
    - 1/26/2019: Default code with basic functions for for lexical analysis
    -
'''

from __future__ import division
import future

#Library Imports
import nltk
try:
    import request
except:
    from urllib import request
import urllib

from nltk import word_tokenize
import re
import string
from string import punctuation
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer 
from nltk.stem import LancasterStemmer 
from nltk.stem import RegexpStemmer #user defined rules
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import contractions
import pandas as pd
import numpy as np




def lexical_diversity(text):
    '''
    This functon calculated lexical_diversity score for a given text
    '''
    return len(set(text))/len(text) 
    
def percentage(count, total):
    '''
    Functions returns a percentage for a given count
    '''
    return 100 * count / total

def FreqDistPlot(data, show =10):
    fdist1 = FreqDist(data) 
    fdist1.plot(show, cumulative=True)
    
def remove_stopwords(tokens): 
    '''
     Code adopted from text Text-analytics-with-python-a-practical-dipanjan-sarkar
     The Function removes stop words from a given toxens list.
    '''
    stopword_list = nltk.corpus.stopwords.words('english') 
    filtered_tokens = [token for token in tokens if token not in stopword_list] 
    return filtered_tokens 

def dropVowels(text):
    vowels = ('a', 'e', 'i', 'o', 'u', 'A', 'I', 'E', 'O', 'U')

    for char in text:

        if char in vowels:

            text = text.replace(char, '')

    return text

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

def lemming(tokens, pos = 'n'):
    '''
    Function to remove word affixes to get to a base form of the word
    
    pos:
    n =noun
    v= verb
    ad =adjectives
    '''
    lemma = WordNetLemmatizer() 
    
    lemmad_list =[]
    for i in tokens:
        List =list(nltk.pos_tag([i])[0])[1]
        if(List in ['NNPS','NNS', 'NN']): #nouns
            pos = 'n'
        if(List in ['JJS','JJR']): #adjectives
            pos ='a'
        if(List in ['RB','RBS','RBR','MB', 'VBN', 'VBD', 'VBG']): #verbs and adverbs
            pos ='v'
        lemmad_list= lemmad_list+ [lemma.lemmatize(i, pos)]
    return lemmad_list

def Tokenizer_Tool(sentence ="The brown fox wasn’t that quick and he couldn’t win the  race", Type ='RegexpTokenizer', TOKEN_PATTERN = r'\s+' ):
    '''
    This function utlilizes different nltk methods for tokenization.
    
    '''
    
    Tokenizers = {'word_tokenize':nltk.word_tokenize(sentence),
                  'TreebankWordTokenizer':nltk.TreebankWordTokenizer().tokenize(sentence),
                  'RegexpTokenizer':nltk.RegexpTokenizer(pattern=TOKEN_PATTERN,gaps =True).tokenize(sentence),
                 'WordPunctTokenizer':nltk.WordPunctTokenizer().tokenize(sentence),
                 'WhitespaceTokenizer':nltk.WhitespaceTokenizer().tokenize(sentence) }

    '''  
    SAMPLE REGEX Breakdowns

     .  for matching any single character   
     ^  for matching the start of the string  
     $  for matching the end of the string 
     *  for matching zero or more cases of the previous mentioned regex before the  *  symbol in the pattern 
     ?  for matching zero or one case of the previous mentioned regex before the  ?  symbol in the pattern  
     [...]  for matching any one of the set of characters inside the square brackets  
     [^...]  for matching a character not present in the square brackets after the  ^  symbol  
     |  denotes the OR  operator   for matching either the preceding or the next regex  
     +  for matching one or more cases of the previous mentioned regex before the  +  symbol in the pattern  
     \d  for matching decimal digits which is also depicted as  [0-9]   
     \D  for matching non-digits, also depicted as  [^0-9]   
     \s  for matching white space characters  
     \S  for matching non whitespace characters  
     \w  for matching alphanumeric characters also depicted as [a-zA-Z0-9_]   
     \W  for matching non alphanumeric characters also depicted as [^a-zA-Z0-9_]     
    '''
    #print(Tokenizers['word_tokenize'])
    return  Tokenizers[Type]


def remove_characters_after_tokenization(tokens): 
    '''
    Code adopted from text Text-analytics-with-python-a-practical-dipanjan-sarkar
    Function removed special Characters from a tokens list
    '''
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation))) 
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    return filtered_tokens 

 
def expand_contractions(sentence ="you're happy now"): 
    '''    
    CONTRACTION_MAP = { "isn't": "is not", 
                           ...... ........
                        "aren't": "are not"}
    This function expands the contractions
    '''
    expanded_contraction = contractions.fix(sentence)
    return expanded_contraction 
    
def case_conversion(token, Type ="small"):
    '''
    This function converts tokens to small or cap text. Small is default
    '''
    case = {'caps':token.upper(), 'small':token.lower()}
    
    return case[Type]
              
def Total_Vocab_size(Tokens):
    '''
    Function returns the actual size of text
    '''
    return(len(Tokens))                     
       
                    
def Vocab_size(Tokens):
    '''
    Function returns  unique words vocab size
    '''
    return(len(set(Tokens)))    

def MinMaxScaler(x):
    '''
    function to scale values between 0 and 1
    '''
    normalized =[]
    for i in x:
        normalized = normalized+ [(i-np.min(x))/(max(x)-min(x))]
    return normalized

def Normalized_Vocab_Score(Texts, BooksMapper):
    '''
    This function normalizes the vocabulary size score for texts passed in a list
    Normalizations:
        1. v_raw_score = Vocab_Size_Text(i)/ Max(Vocab_Size_Texts)
        2. v_sqrt_score = sqrt(v_raw_score)
        3. v_minmax_score = MinMaxScaler(v_raw_score)
        4. v_final_score = avg(v_sqrt_score, v_minmax_score)
    '''
    Summary = pd.DataFrame()
    VocabSize = []
    v_raw_score =[]
    v_sqrt_score =[]
    category =[]
    
    #Vocab Size
    f = lambda x:Vocab_size(x)
    VocabSize = [f(x) for x in list(Texts.values())]
    books = list(Texts.keys())
    
    #lexical diversity score
    f_lx = lambda x:lexical_diversity(x)
    Lexical_diversity = [f_lx(x) for x in list(Texts.values())]
    
    #v_raw Score
    f_raw = lambda x:x/max(VocabSize)
    v_raw_score = [f_raw(x) for x in VocabSize]
    
    for i in books:
        category = category + list(BooksMapper[BooksMapper.BookTitle==i].Category)
    
    
    Summary = pd.DataFrame({"Title":books,
                            "Category":category,
                            "VocabSize":VocabSize,
                            "Lexical_diversity":Lexical_diversity, 
                            'V_Raw_Score': v_raw_score,
                            'V_Sqrt_Score':np.sqrt(v_raw_score), 
                            'V_minmax_Score':MinMaxScaler(v_raw_score),
                           })
    Summary['V_Normalized_FinalScore'] = (Summary.V_Sqrt_Score+Summary.V_minmax_Score)/2
        
    return(Summary)
 
def Long_Word_Vocab_Size(text,minChar=14):
    '''
    Function to calculate longword vocab size after consulting section 3.2 of Chapter 1 Bird-Klein
    long word is defined as >14 Characters
    '''
    V = set(text)
    long_words = [w for w in V if len(w)>minChar]
    return len(long_words)

def Normalized_Long_Word_Vocab_Score(Texts,BooksMapper, minChar =14):
    '''
    This function normalizes the Long Word vocabulary size score for texts passed in a list
    Normalizations:
        1. v_raw_score = Vocab_Size_Text(i)/ Max(Vocab_Size_Texts)
        2. v_sqrt_score = sqrt(v_raw_score)
        3. v_minmax_score = MinMaxScaler(v_raw_score)
        4. v_final_score = avg(v_sqrt_score, v_minmax_score)
    '''
    Summary = pd.DataFrame()
    VocabSize = []
    v_raw_score =[]
    v_sqrt_score =[]
    category = []
    
    #Vocab Size
    f = lambda x:Long_Word_Vocab_Size(x, minChar)
    VocabSize = [f(x) for x in list(Texts.values())]
    books = list(Texts.keys())
    
    for i in books:
        category = category + list(BooksMapper[BooksMapper.BookTitle==i].Category)
        
    
    #lexical diversity score
    f_lx = lambda x:lexical_diversity(x)
    Lexical_diversity = [f_lx(x) for x in list(Texts.values())]
    
    #v_raw Score
    f_raw = lambda x:x/max(VocabSize)
    v_raw_score = [f_raw(x) for x in VocabSize]
    
    Summary = pd.DataFrame({"Title":books,
                            "Category":category,
                            "LongWordVocabSize":VocabSize,
                            'Lexical_diversity':Lexical_diversity, 
                            'V_Raw_Score': v_raw_score,
                            'V_Sqrt_Score':np.sqrt(v_raw_score), 
                            'V_minmax_Score':MinMaxScaler(v_raw_score),
                            #'v_final_score':np.mean(np.sqrt(v_raw_score),MinMaxScaler(v_raw_score))
                           })
    Summary['V_Normalized_LW_FinalScore'] = (Summary.V_Sqrt_Score+Summary.V_minmax_Score)/2
        
    return(Summary)
 
    
def Text_Difficulty_Score(data,BooksMapper,  LW =14):
    '''
    This function returns the summary for the test difficulty score
    Text difficulty score is calculated as the avg of Lexical_diversity_score, 
    Normalized vocab size score and Normalised long word vocabulary score
    '''
    Summary = pd.DataFrame()
    Normalized_Vocab = Normalized_Vocab_Score(data, BooksMapper)
    Normalized_LW_Vocab =Normalized_Long_Word_Vocab_Score(data,BooksMapper, LW)
    
    Summary['Title'] = Normalized_Vocab.Title
    Summary['Category'] = Normalized_Vocab.Category
    Summary['VocabSize']  = Normalized_Vocab.VocabSize
    Summary['LongWordVocabSize']  = Normalized_LW_Vocab.LongWordVocabSize
    Summary['Lexical_diversity']  = Normalized_Vocab.Lexical_diversity
    Summary['V_Normalized_FinalScore']  = Normalized_Vocab.V_Normalized_FinalScore
    Summary['V_Normalized_LW_FinalScore'] = Normalized_LW_Vocab.V_Normalized_LW_FinalScore
    Summary['Text_Difficulty_Score'] = 100*((Summary.Lexical_diversity+Summary.V_Normalized_FinalScore+ 
                                             Summary.V_Normalized_LW_FinalScore)/3)
    
    return(Summary)
 
 
 

 