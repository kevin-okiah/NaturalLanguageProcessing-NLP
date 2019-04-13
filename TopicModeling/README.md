


## Topic Modeling - GutenBerg Books

#### Author: Kevin Okiah

**04/14/2019**

In this notebook, I explore Topic modeling using "Latent Dirichlet allocation" (LDA). LDA is a generative probabilistic model for collections of
discrete data such as text corpora. LDA is a three-level hierarchical Bayesian model,in which each item of a collection is modeled as a finite mixture over an underlying set of topics. Each topic is, in turn, modeled as an infinite mixture over an underlying set of topic probabilities. In the context of text modeling, the topic probabilities provide an explicit representation of a document. [Link](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)

#### Problem Statement
![](https://github.com/kevimwe/NaturalLanguageProcessing-NLP/blob/master/TopicModeling/topic_modeling.JPG)


#### NLP Pipeline Approach
I follow an NLP pipleine approach to find books similar to `The Flag of My Country. Shikéyah Bidah Na'at'a'í; Navajo New World Readers 2` from [Children's Instructional Books (Bookshelf)](https://www.gutenberg.org/wiki/Children%27s_Instructional_Books_(Bookshelf)) to recommend to my son as shown in the flow chart below.
![](https://github.com/kevimwe/NaturalLanguageProcessing-NLP/blob/master/TopicModeling/nlp_pipeline.JPG)

Data Collection and preprocessing were performed on separated notebooks which can be found in the `Notebooks` Directory
* **Data Acqusitions**  ->  /Notebooks/Scrape_GuternsbergBooks.ipynb
* **Pre-processing and EDA** -> /Notebooks/TextCleaning_and_EDA.ipynb

### Modeling


```python
import os
from unipath import Path
import time
#visualization 
import seaborn as sns
import matplotlib.pyplot as plt
import pyLDAvis.sklearn
from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig
%matplotlib inline

# Data Manipulation and Statistics
import pandas as pd
import numpy as np

#Directory Navigation and Saving instances
import os
from unipath import Path
wd = os.getcwd()
p = Path(wd)
path = str(p.parent)

#Text Cleaning and Analytics
import spacy
nlp = spacy.load('en_core_web_sm')

wd = os.getcwd()
p = Path(wd)
path = str(p.parent)


# Topic Modeling
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings('ignore')
```


```python
# Read data
data1 = pd.read_csv("Data/GuternsbergBooksClean.csv", encoding='utf-8')
#data1 = pd.read_csv("Data/MovieReviewsWithSentimentsClean.csv", encoding='utf-8')
data1.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BookTitle</th>
      <th>Category</th>
      <th>url</th>
      <th>Body</th>
      <th>Corpus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A Primary Reader: Old-time Stories, Fairy Tale...</td>
      <td>Misc.</td>
      <td>http://www.gutenberg.org/cache/epub/7841/pg784...</td>
      <td>['CONTENTS.', 'THE UGLY DUCKLING', 'THE LITTLE...</td>
      <td>['ugly', 'duckling', 'little', 'pine', 'tree',...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Bird-Woman of the Lewis and Clark Expedition</td>
      <td>Misc.</td>
      <td>http://www.gutenberg.org/cache/epub/5742/pg574...</td>
      <td>['CONTENTS', 'THE BIRD-WOMAN', 'WHO THE WHITE ...</td>
      <td>['bird', 'woman', 'white', 'men', 'sacajawea',...</td>
    </tr>
  </tbody>
</table>
</div>



#### LDA -Latent Dirichlet Allocation


```python
def VisualizeGridSearchResults(Data):
    '''
    display Log Likelihood Score comparison of different models
    
    '''
    plt.figure(figsize=(12, 7))
    ax = sns.pointplot(x="param_n_components", y="mean_test_score", hue="param_learning_decay",
                       data=Data, palette="Set2")
    plt.title("Choosing Optimal LDA Model")
    plt.xlabel("Num Topics")
    plt.ylabel("Log Likelyhood Scores")
    plt.legend(title='Learning decay', loc='best')
```


```python
def OptimalTopicsGridSearch(Corpus=data1['Corpus'], NoOfTopics =[3,4,5,7,8,9, 10, 15, 25, 35, 40, 50],lda_=True, LearningDecay = [.3,.4, .5,.6,.7, .8],show =True):
        '''
        Function to run LDA Grid Search - LatentDirichletAllocation for topic modeling.
        *****************
        Inputs:
         - Corpus = Body of clean documents to identify topics
         - NoOfTopics = list of Number of topics for grid search
         - LearningDecay  = Learning Decay rate
         - show -  Flag to visualise grid searcg results
         - lda_  = Flag True to run LDA else false for NMF 
         '''
        #Define Search Param
        search_params = {'n_components': NoOfTopics, 'learning_decay': LearningDecay}
        
        if lda_ ==True:
            # Counter Vectorizer Object
            cv = CountVectorizer(max_df=0.90, min_df=5, stop_words='english') #max_df=0.95 (drop most frequent words) min_df=2 (drop unique words)

            # Init the Model
            model_ = LatentDirichletAllocation()
            
            #vectorize the data
            data_vectorized = cv.fit_transform(Corpus)
            GridSearchType =True
            
        else:
            #tfidf object
            cv = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
            data_vectorized = cv.fit_transform(Corpus)

            #NMF object
            model_ = NMF()
            GridSearchType =False
        
        if GridSearchType ==False:
            print("--------------------------------------------------")
            # RandomisedGridSearch 
            print("Performing Randomized Grid Search")
            print("--------------------------------------------------")
            start = time. time()
            model = RandomizedSearchCV(model_, param_distributions=search_params)

        else:
            #GridSearch
            print("Performing GridSearch")
            print("--------------------------------------------------")
            start = time. time()
            model = GridSearchCV(model_, param_grid=search_params)
        
        # Do the Grid Search
        model.fit(data_vectorized)

        print(model)
        # Best Model
        best_lda_model = model.best_estimator_
        print("--------------------------------------------------")
        # Model Parameters
        print("Best Model's Params: ", model.best_params_)

        # Log Likelihood Score
        print("Best Log Likelihood Score: ", model.best_score_)

        # Perplexity
        print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))
        end = time. time()
        print("--------------------------------------------------")
        print('Runtime :',end - start)
        GridResults = pd.DataFrame.from_dict(model.cv_results_)
        # Visualize gridSearch Results
        if show==True:
            VisualizeGridSearchResults(GridResults)
        return(GridResults)

GridResults = OptimalTopicsGridSearch(show=False, lda_=True)
# Saving Data for future analysis
GridResults.to_csv('Data/LDAGridResults.csv', header=True, index=False, encoding='utf-8')
```

    Performing GridSearch
    --------------------------------------------------
    GridSearchCV(cv='warn', error_score='raise-deprecating',
           estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
                 evaluate_every=-1, learning_decay=0.7,
                 learning_method='batch', learning_offset=10.0,
                 max_doc_update_iter=100, max_iter=10, mean_change_tol=0.001,
                 n_components=10, n_jobs=None, n_topics=None, perp_tol=0.1,
                 random_state=None, topic_word_prior=None,
                 total_samples=1000000.0, verbose=0),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'n_components': [3, 4, 5, 7, 8, 9, 10, 15, 25, 35, 40, 50], 'learning_decay': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)
    --------------------------------------------------
    Best Model's Params:  {'learning_decay': 0.5, 'n_components': 4}
    Best Log Likelihood Score:  -4875853.551934138
    Model Perplexity:  3178.461567301383
    --------------------------------------------------
    Runtime : 802.7736694812775



```python
GridResults = pd.read_csv('Data/LDAGridResults.csv')
```


```python
VisualizeGridSearchResults(GridResults)
```


![png](output_8_0.png)



```python
def LatentDirichletAllocation_Run(Corpus=data1['Corpus'], NoOfTopics =4, TopWords = 15, show =False, LearningDecay = 0.5):
    '''
    Function to run LDA - LatentDirichletAllocation for topic modeling.
    *****************
    Inputs:
     - Corpus = Body of clean documents to identify topics
     - NoOfTopics = Number of topics
     - TopWords = Number of top words to display per topic
    ******************
    The function:
    1. Takes a Corpus and vectorizes it using Sklearn CountVectorizer
    2. Fits LDA on the data to identify topics
    3. Returns a dataframe with the corpus and topic assigned  
    
    '''
    Topic =pd.DataFrame()
    # Counter Vectorizer Object
    cv = CountVectorizer(max_df=0.90, min_df=5) #max_df=0.95 (drop most frequent words) min_df=2 (drop unique words)
   
    #vectorize the data
    dtm = cv.fit_transform(Corpus)
    print(dtm.shape)
    
    # Latent Dirichlet Allocation Model
    LDA = LatentDirichletAllocation(n_components=NoOfTopics,learning_decay=LearningDecay,random_state=100, max_iter=10)
    
    #lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
    #data_lda = lda.fit_transform(data_vectorized)
    
    TFIDF = TfidfVectorizer(max_df=0.90, min_df=4, stop_words='english')
    
    # This can take awhile, we're dealing with a large amount of documents!
    X_Topics =LDA.fit(dtm)
    #X_Topics =TFIDF.fit(dtm)
    
    # number of features /words
    print("-------------------------------------------------------")
    print("                   Topics Summary                      ")
    print("-------------------------------------------------------")
    print("Number of Features/Words :", len(cv.get_feature_names()))
    print("Number of Topics:", len(LDA.components_))
    print("-------------------------------------------------------")
    Words_topics = pd.DataFrame() #Sortd
    temp_topic =pd.DataFrame() #sorted
    WordCloud = pd.DataFrame()
    temp_wordcloud = pd.DataFrame()
    n= 0
    if( show ==True):
        for index,topic in enumerate(LDA.components_):
            print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
            print([cv.get_feature_names()[i] for i in topic.argsort()[-TopWords:]])
            # print([topic[i] for i in topic.argsort()[-TopWords:]])
            #Top_topics = Top_topics + [[cv.get_feature_names()[i] for i in topic.argsort()[-TopWords:]]]
            Sortedwords = [[cv.get_feature_names()[i] for i in topic.argsort()[-TopWords:]]]
            temp_topic["Topic"] = "Topic_"+str(n)
            temp_topic["Top5Words"] =list(Sortedwords)[0][0:15]
            temp_wordcloud["Topic"] = "Topic_"+str(n)
            temp_wordcloud["Words"] =list(Sortedwords)
            Words_topics = pd.concat([Words_topics, temp_topic], axis=0)
            WordCloud = pd.concat([WordCloud, temp_wordcloud], axis=0)
            WordCloud['Topic'] = WordCloud['Topic'].replace('nan', np.nan).fillna('Topic_0')
            Words_topics['Topic'] = Words_topics['Topic'].replace('nan', np.nan).fillna('Topic_0')
            print('\n')
            n =n+1
    topic_results = LDA.transform(dtm)
    
    # column names
    topicnames = ["Topic" + str(i) for i in range(NoOfTopics)]

    # index names
    docnames = ["Book" + str(i) for i in range(len(Corpus))]
    
    Topic=pd.DataFrame(topic_results)
    Topic = Topic.round(3) 
    Topic.columns=topicnames
    Topic.index=docnames
    Topic['Dominant_Topic'] = topic_results.argmax(axis=1) # Map topic to book
    Topic['Dominant_Topic'] = topic_results.argmax(axis=1) # Map topic to book
    
    return(Topic,LDA,cv, dtm, topic_results, Words_topics, WordCloud)

Temp_LDA, lda, vectorizer_cv, data_vectorized_cv ,LdaTR, WordsTopics, WordCloud= LatentDirichletAllocation_Run(NoOfTopics =4, show =True)
#data['Topic_by_LDA'] = list(Temp_LDA.Dominant_Topic)
data = data1.copy()
data.index=Temp_LDA.index
#data.index=Temp_LDA.BookTitle	
data = data.join(Temp_LDA)
data.head(2)
```

    (104, 14453)
    -------------------------------------------------------
                       Topics Summary                      
    -------------------------------------------------------
    Number of Features/Words : 14453
    Number of Topics: 4
    -------------------------------------------------------
    THE TOP 15 WORDS FOR TOPIC #0
    ['make', 'bird', 'plant', 'come', 'insect', 'call', 'end', 'body', 'use', 'food', 'leave', 'form', 'small', 'animal', 'water']
    
    
    THE TOP 15 WORDS FOR TOPIC #1
    ['qui', 'ce', 'des', 'se', 'el', 'il', 'du', 'un', 'les', 'que', 'et', 'en', 'le', 'la', 'de']
    
    
    THE TOP 15 WORDS FOR TOPIC #2
    ['army', 'english', 'life', 'general', 'call', 'city', 'power', 'take', 'england', 'war', 'country', 'king', 'people', 'year', 'state']
    
    
    THE TOP 15 WORDS FOR TOPIC #3
    ['word', 'father', 'life', 'let', 'leave', 'take', 'eye', 'shall', 'mother', 'hear', 'boy', 'think', 'tell', 'go', 'come']
    
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BookTitle</th>
      <th>Category</th>
      <th>url</th>
      <th>Body</th>
      <th>Corpus</th>
      <th>Topic0</th>
      <th>Topic1</th>
      <th>Topic2</th>
      <th>Topic3</th>
      <th>Dominant_Topic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Book0</th>
      <td>A Primary Reader: Old-time Stories, Fairy Tale...</td>
      <td>Misc.</td>
      <td>http://www.gutenberg.org/cache/epub/7841/pg784...</td>
      <td>['CONTENTS.', 'THE UGLY DUCKLING', 'THE LITTLE...</td>
      <td>['ugly', 'duckling', 'little', 'pine', 'tree',...</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Book1</th>
      <td>The Bird-Woman of the Lewis and Clark Expedition</td>
      <td>Misc.</td>
      <td>http://www.gutenberg.org/cache/epub/5742/pg574...</td>
      <td>['CONTENTS', 'THE BIRD-WOMAN', 'WHO THE WHITE ...</td>
      <td>['bird', 'woman', 'white', 'men', 'sacajawea',...</td>
      <td>0.2</td>
      <td>0.014</td>
      <td>0.0</td>
      <td>0.786</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# top words per topic
# Top 5 Keywords for each Topic
df_top5words_stacked = pd.DataFrame(WordsTopics, columns=['Topic', 'Top5Words'])
df_top5words = df_top5words_stacked.groupby('Topic').agg(', \n'.join)
df_top5words.reset_index(level=0,inplace=True)
```


```python
df_top5words
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Topic</th>
      <th>Top5Words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Topic_0</td>
      <td>make, \nbird, \nplant, \ncome, \ninsect, \ncal...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Topic_1</td>
      <td>qui, \nce, \ndes, \nse, \nel, \nil, \ndu, \nun...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Topic_2</td>
      <td>army, \nenglish, \nlife, \ngeneral, \ncall, \n...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Topic_3</td>
      <td>word, \nfather, \nlife, \nlet, \nleave, \ntake...</td>
    </tr>
  </tbody>
</table>
</div>



### Determining the topics based on top 15 words


```python
Topic0 = list(WordsTopics[WordsTopics.Topic=='Topic_0']['Top5Words'])
Topic1 = list(WordsTopics[WordsTopics.Topic=='Topic_1']['Top5Words'])
Topic2 = list(WordsTopics[WordsTopics.Topic=='Topic_2']['Top5Words'])
Topic3 = list(WordsTopics[WordsTopics.Topic=='Topic_3']['Top5Words'])
#Topic4 = list(WordsTopics[WordsTopics.Topic=='Topic_4']['Top5Words'])

Summary = pd.DataFrame([Topic0, Topic1, Topic2, Topic3])
Summary.index = ['Topic_0','Topic_1', 'Topic_2', 'Topic_3'  ]
names = list(Summary.columns)
names_=[]
for i in names:
    names_=names_+['Word_'+str(names[i])]
Summary.columns = names_
Summary.T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Topic_0</th>
      <th>Topic_1</th>
      <th>Topic_2</th>
      <th>Topic_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Word_0</th>
      <td>make</td>
      <td>qui</td>
      <td>army</td>
      <td>word</td>
    </tr>
    <tr>
      <th>Word_1</th>
      <td>bird</td>
      <td>ce</td>
      <td>english</td>
      <td>father</td>
    </tr>
    <tr>
      <th>Word_2</th>
      <td>plant</td>
      <td>des</td>
      <td>life</td>
      <td>life</td>
    </tr>
    <tr>
      <th>Word_3</th>
      <td>come</td>
      <td>se</td>
      <td>general</td>
      <td>let</td>
    </tr>
    <tr>
      <th>Word_4</th>
      <td>insect</td>
      <td>el</td>
      <td>call</td>
      <td>leave</td>
    </tr>
    <tr>
      <th>Word_5</th>
      <td>call</td>
      <td>il</td>
      <td>city</td>
      <td>take</td>
    </tr>
    <tr>
      <th>Word_6</th>
      <td>end</td>
      <td>du</td>
      <td>power</td>
      <td>eye</td>
    </tr>
    <tr>
      <th>Word_7</th>
      <td>body</td>
      <td>un</td>
      <td>take</td>
      <td>shall</td>
    </tr>
    <tr>
      <th>Word_8</th>
      <td>use</td>
      <td>les</td>
      <td>england</td>
      <td>mother</td>
    </tr>
    <tr>
      <th>Word_9</th>
      <td>food</td>
      <td>que</td>
      <td>war</td>
      <td>hear</td>
    </tr>
    <tr>
      <th>Word_10</th>
      <td>leave</td>
      <td>et</td>
      <td>country</td>
      <td>boy</td>
    </tr>
    <tr>
      <th>Word_11</th>
      <td>form</td>
      <td>en</td>
      <td>king</td>
      <td>think</td>
    </tr>
    <tr>
      <th>Word_12</th>
      <td>small</td>
      <td>le</td>
      <td>people</td>
      <td>tell</td>
    </tr>
    <tr>
      <th>Word_13</th>
      <td>animal</td>
      <td>la</td>
      <td>year</td>
      <td>go</td>
    </tr>
    <tr>
      <th>Word_14</th>
      <td>water</td>
      <td>de</td>
      <td>state</td>
      <td>come</td>
    </tr>
  </tbody>
</table>
</div>



Based on the frequent words by topic above, we can deduce the topics as

>**Topic_0** - Science or Nature 

>**Topic_1** - Non English

>**Topic_2** - Geography, History, Civic

>**Topic_3** - General

### Visualizing  dominant Topic by probability.


```python
from pandas.tools.plotting import table
cm = sns.light_palette("green", as_cmap=True)

s = Temp_LDA.style.background_gradient(cmap=cm, subset=['Topic0','Topic1','Topic2','Topic3', 'Topic4', 'Topic5', 'Topic6'])
print("LDA Topics Book Assignment Heatmap");
s.to_excel('a.xlsx', engine='openpyxl')
s
```

    LDA Topics Book Assignment Heatmap





<style  type="text/css" >
    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row0_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row0_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row0_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row0_col3 {
            background-color:  #008000;
            background-color:  #008000;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row1_col0 {
            background-color:  #b8e6b8;
            background-color:  #b8e6b8;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row1_col1 {
            background-color:  #e3fee3;
            background-color:  #e3fee3;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row1_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row1_col3 {
            background-color:  #319b31;
            background-color:  #319b31;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row2_col0 {
            background-color:  #caf0ca;
            background-color:  #caf0ca;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row2_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row2_col2 {
            background-color:  #81c781;
            background-color:  #81c781;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row2_col3 {
            background-color:  #81c781;
            background-color:  #81c781;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row3_col0 {
            background-color:  #e4fee4;
            background-color:  #e4fee4;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row3_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row3_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row3_col3 {
            background-color:  #028102;
            background-color:  #028102;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row4_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row4_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row4_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row4_col3 {
            background-color:  #008000;
            background-color:  #008000;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row5_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row5_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row5_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row5_col3 {
            background-color:  #008000;
            background-color:  #008000;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row6_col0 {
            background-color:  #6cbc6c;
            background-color:  #6cbc6c;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row6_col1 {
            background-color:  #e4fee4;
            background-color:  #e4fee4;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row6_col2 {
            background-color:  #a8dda8;
            background-color:  #a8dda8;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row6_col3 {
            background-color:  #b9e7b9;
            background-color:  #b9e7b9;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row7_col0 {
            background-color:  #80c780;
            background-color:  #80c780;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row7_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row7_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row7_col3 {
            background-color:  #66b866;
            background-color:  #66b866;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row8_col0 {
            background-color:  #dcfadc;
            background-color:  #dcfadc;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row8_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row8_col2 {
            background-color:  #dbf9db;
            background-color:  #dbf9db;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row8_col3 {
            background-color:  #158b15;
            background-color:  #158b15;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row9_col0 {
            background-color:  #e3fee3;
            background-color:  #e3fee3;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row9_col1 {
            background-color:  #e3fee3;
            background-color:  #e3fee3;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row9_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row9_col3 {
            background-color:  #068306;
            background-color:  #068306;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row10_col0 {
            background-color:  #7dc57d;
            background-color:  #7dc57d;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row10_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row10_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row10_col3 {
            background-color:  #6abb6a;
            background-color:  #6abb6a;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row11_col0 {
            background-color:  #e1fde1;
            background-color:  #e1fde1;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row11_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row11_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row11_col3 {
            background-color:  #058205;
            background-color:  #058205;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row12_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row12_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row12_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row12_col3 {
            background-color:  #018001;
            background-color:  #018001;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row13_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row13_col1 {
            background-color:  #e4fee4;
            background-color:  #e4fee4;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row13_col2 {
            background-color:  #c7eec7;
            background-color:  #c7eec7;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row13_col3 {
            background-color:  #209220;
            background-color:  #209220;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row14_col0 {
            background-color:  #defbde;
            background-color:  #defbde;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row14_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row14_col2 {
            background-color:  #cef2ce;
            background-color:  #cef2ce;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row14_col3 {
            background-color:  #1f911f;
            background-color:  #1f911f;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row15_col0 {
            background-color:  #d4f6d4;
            background-color:  #d4f6d4;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row15_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row15_col2 {
            background-color:  #cbf1cb;
            background-color:  #cbf1cb;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row15_col3 {
            background-color:  #2e992e;
            background-color:  #2e992e;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row16_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row16_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row16_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row16_col3 {
            background-color:  #008000;
            background-color:  #008000;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row17_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row17_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row17_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row17_col3 {
            background-color:  #028102;
            background-color:  #028102;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row18_col0 {
            background-color:  #e3fee3;
            background-color:  #e3fee3;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row18_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row18_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row18_col3 {
            background-color:  #038103;
            background-color:  #038103;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row19_col0 {
            background-color:  #ddfbdd;
            background-color:  #ddfbdd;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row19_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row19_col2 {
            background-color:  #d1f4d1;
            background-color:  #d1f4d1;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row19_col3 {
            background-color:  #1f911f;
            background-color:  #1f911f;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row20_col0 {
            background-color:  #e2fde2;
            background-color:  #e2fde2;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row20_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row20_col2 {
            background-color:  #a8dda8;
            background-color:  #a8dda8;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row20_col3 {
            background-color:  #42a442;
            background-color:  #42a442;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row21_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row21_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row21_col2 {
            background-color:  #80c780;
            background-color:  #80c780;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row21_col3 {
            background-color:  #66b866;
            background-color:  #66b866;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row22_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row22_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row22_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row22_col3 {
            background-color:  #008000;
            background-color:  #008000;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row23_col0 {
            background-color:  #a4dba4;
            background-color:  #a4dba4;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row23_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row23_col2 {
            background-color:  #d8f8d8;
            background-color:  #d8f8d8;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row23_col3 {
            background-color:  #51ad51;
            background-color:  #51ad51;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row24_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row24_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row24_col2 {
            background-color:  #e0fce0;
            background-color:  #e0fce0;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row24_col3 {
            background-color:  #068306;
            background-color:  #068306;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row25_col0 {
            background-color:  #dbf9db;
            background-color:  #dbf9db;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row25_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row25_col2 {
            background-color:  #b8e6b8;
            background-color:  #b8e6b8;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row25_col3 {
            background-color:  #3aa03a;
            background-color:  #3aa03a;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row26_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row26_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row26_col2 {
            background-color:  #93d193;
            background-color:  #93d193;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row26_col3 {
            background-color:  #54ae54;
            background-color:  #54ae54;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row27_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row27_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row27_col2 {
            background-color:  #9ed79e;
            background-color:  #9ed79e;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row27_col3 {
            background-color:  #48a848;
            background-color:  #48a848;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row28_col0 {
            background-color:  #d4f6d4;
            background-color:  #d4f6d4;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row28_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row28_col2 {
            background-color:  #b9e7b9;
            background-color:  #b9e7b9;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row28_col3 {
            background-color:  #3ea23e;
            background-color:  #3ea23e;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row29_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row29_col1 {
            background-color:  #e2fde2;
            background-color:  #e2fde2;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row29_col2 {
            background-color:  #b3e3b3;
            background-color:  #b3e3b3;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row29_col3 {
            background-color:  #369e36;
            background-color:  #369e36;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row30_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row30_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row30_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row30_col3 {
            background-color:  #018001;
            background-color:  #018001;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row31_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row31_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row31_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row31_col3 {
            background-color:  #018001;
            background-color:  #018001;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row32_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row32_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row32_col2 {
            background-color:  #e0fce0;
            background-color:  #e0fce0;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row32_col3 {
            background-color:  #058305;
            background-color:  #058305;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row33_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row33_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row33_col2 {
            background-color:  #bce8bc;
            background-color:  #bce8bc;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row33_col3 {
            background-color:  #299729;
            background-color:  #299729;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row34_col0 {
            background-color:  #62b662;
            background-color:  #62b662;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row34_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row34_col2 {
            background-color:  #bbe8bb;
            background-color:  #bbe8bb;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row34_col3 {
            background-color:  #afe1af;
            background-color:  #afe1af;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row35_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row35_col1 {
            background-color:  #c6eec6;
            background-color:  #c6eec6;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row35_col2 {
            background-color:  #379e37;
            background-color:  #379e37;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row35_col3 {
            background-color:  #cef2ce;
            background-color:  #cef2ce;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row36_col0 {
            background-color:  #dbf9db;
            background-color:  #dbf9db;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row36_col1 {
            background-color:  #3aa03a;
            background-color:  #3aa03a;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row36_col2 {
            background-color:  #cff3cf;
            background-color:  #cff3cf;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row36_col3 {
            background-color:  #cef2ce;
            background-color:  #cef2ce;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row37_col0 {
            background-color:  #e3fee3;
            background-color:  #e3fee3;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row37_col1 {
            background-color:  #319b31;
            background-color:  #319b31;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row37_col2 {
            background-color:  #dbf9db;
            background-color:  #dbf9db;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row37_col3 {
            background-color:  #c2ecc2;
            background-color:  #c2ecc2;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row38_col0 {
            background-color:  #cff3cf;
            background-color:  #cff3cf;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row38_col1 {
            background-color:  #72bf72;
            background-color:  #72bf72;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row38_col2 {
            background-color:  #b9e7b9;
            background-color:  #b9e7b9;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row38_col3 {
            background-color:  #b8e6b8;
            background-color:  #b8e6b8;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row39_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row39_col1 {
            background-color:  #008000;
            background-color:  #008000;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row39_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row39_col3 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row40_col0 {
            background-color:  #6cbc6c;
            background-color:  #6cbc6c;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row40_col1 {
            background-color:  #e4fee4;
            background-color:  #e4fee4;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row40_col2 {
            background-color:  #a8dda8;
            background-color:  #a8dda8;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row40_col3 {
            background-color:  #b9e7b9;
            background-color:  #b9e7b9;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row41_col0 {
            background-color:  #cbf1cb;
            background-color:  #cbf1cb;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row41_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row41_col2 {
            background-color:  #269526;
            background-color:  #269526;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row41_col3 {
            background-color:  #daf9da;
            background-color:  #daf9da;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row42_col0 {
            background-color:  #2d992d;
            background-color:  #2d992d;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row42_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row42_col2 {
            background-color:  #d4f6d4;
            background-color:  #d4f6d4;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row42_col3 {
            background-color:  #caf0ca;
            background-color:  #caf0ca;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row43_col0 {
            background-color:  #daf9da;
            background-color:  #daf9da;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row43_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row43_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row43_col3 {
            background-color:  #0c860c;
            background-color:  #0c860c;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row44_col0 {
            background-color:  #2e992e;
            background-color:  #2e992e;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row44_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row44_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row44_col3 {
            background-color:  #b8e6b8;
            background-color:  #b8e6b8;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row45_col0 {
            background-color:  #94d294;
            background-color:  #94d294;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row45_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row45_col2 {
            background-color:  #acdfac;
            background-color:  #acdfac;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row45_col3 {
            background-color:  #8bcd8b;
            background-color:  #8bcd8b;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row46_col0 {
            background-color:  #1b8f1b;
            background-color:  #1b8f1b;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row46_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row46_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row46_col3 {
            background-color:  #caf0ca;
            background-color:  #caf0ca;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row47_col0 {
            background-color:  #67b967;
            background-color:  #67b967;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row47_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row47_col2 {
            background-color:  #cef2ce;
            background-color:  #cef2ce;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row47_col3 {
            background-color:  #96d396;
            background-color:  #96d396;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row48_col0 {
            background-color:  #2b982b;
            background-color:  #2b982b;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row48_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row48_col2 {
            background-color:  #d2f4d2;
            background-color:  #d2f4d2;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row48_col3 {
            background-color:  #cff3cf;
            background-color:  #cff3cf;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row49_col0 {
            background-color:  #b7e5b7;
            background-color:  #b7e5b7;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row49_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row49_col2 {
            background-color:  #e3fee3;
            background-color:  #e3fee3;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row49_col3 {
            background-color:  #319b31;
            background-color:  #319b31;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row50_col0 {
            background-color:  #b8e6b8;
            background-color:  #b8e6b8;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row50_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row50_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row50_col3 {
            background-color:  #2f9a2f;
            background-color:  #2f9a2f;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row51_col0 {
            background-color:  #a3daa3;
            background-color:  #a3daa3;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row51_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row51_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row51_col3 {
            background-color:  #44a544;
            background-color:  #44a544;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row52_col0 {
            background-color:  #b9e7b9;
            background-color:  #b9e7b9;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row52_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row52_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row52_col3 {
            background-color:  #2c982c;
            background-color:  #2c982c;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row53_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row53_col1 {
            background-color:  #058305;
            background-color:  #058305;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row53_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row53_col3 {
            background-color:  #e0fce0;
            background-color:  #e0fce0;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row54_col0 {
            background-color:  #56af56;
            background-color:  #56af56;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row54_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row54_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row54_col3 {
            background-color:  #90d090;
            background-color:  #90d090;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row55_col0 {
            background-color:  #98d498;
            background-color:  #98d498;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row55_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row55_col2 {
            background-color:  #a0d9a0;
            background-color:  #a0d9a0;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row55_col3 {
            background-color:  #94d294;
            background-color:  #94d294;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row56_col0 {
            background-color:  #2b982b;
            background-color:  #2b982b;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row56_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row56_col2 {
            background-color:  #bae7ba;
            background-color:  #bae7ba;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row56_col3 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row57_col0 {
            background-color:  #339c33;
            background-color:  #339c33;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row57_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row57_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row57_col3 {
            background-color:  #b3e3b3;
            background-color:  #b3e3b3;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row58_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row58_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row58_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row58_col3 {
            background-color:  #018001;
            background-color:  #018001;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row59_col0 {
            background-color:  #51ad51;
            background-color:  #51ad51;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row59_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row59_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row59_col3 {
            background-color:  #94d294;
            background-color:  #94d294;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row60_col0 {
            background-color:  #008000;
            background-color:  #008000;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row60_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row60_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row60_col3 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row61_col0 {
            background-color:  #abdfab;
            background-color:  #abdfab;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row61_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row61_col2 {
            background-color:  #ade0ad;
            background-color:  #ade0ad;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row61_col3 {
            background-color:  #74c074;
            background-color:  #74c074;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row62_col0 {
            background-color:  #95d395;
            background-color:  #95d395;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row62_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row62_col2 {
            background-color:  #e1fde1;
            background-color:  #e1fde1;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row62_col3 {
            background-color:  #56af56;
            background-color:  #56af56;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row63_col0 {
            background-color:  #3ca13c;
            background-color:  #3ca13c;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row63_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row63_col2 {
            background-color:  #dcfadc;
            background-color:  #dcfadc;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row63_col3 {
            background-color:  #b3e3b3;
            background-color:  #b3e3b3;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row64_col0 {
            background-color:  #2b982b;
            background-color:  #2b982b;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row64_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row64_col2 {
            background-color:  #bae7ba;
            background-color:  #bae7ba;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row64_col3 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row65_col0 {
            background-color:  #cbf1cb;
            background-color:  #cbf1cb;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row65_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row65_col2 {
            background-color:  #8dce8d;
            background-color:  #8dce8d;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row65_col3 {
            background-color:  #72bf72;
            background-color:  #72bf72;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row66_col0 {
            background-color:  #4eab4e;
            background-color:  #4eab4e;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row66_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row66_col2 {
            background-color:  #9ad59a;
            background-color:  #9ad59a;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row66_col3 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row67_col0 {
            background-color:  #42a442;
            background-color:  #42a442;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row67_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row67_col2 {
            background-color:  #ddfbdd;
            background-color:  #ddfbdd;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row67_col3 {
            background-color:  #ade0ad;
            background-color:  #ade0ad;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row68_col0 {
            background-color:  #178c17;
            background-color:  #178c17;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row68_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row68_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row68_col3 {
            background-color:  #cff3cf;
            background-color:  #cff3cf;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row69_col0 {
            background-color:  #0b860b;
            background-color:  #0b860b;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row69_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row69_col2 {
            background-color:  #dbf9db;
            background-color:  #dbf9db;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row69_col3 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row70_col0 {
            background-color:  #71bf71;
            background-color:  #71bf71;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row70_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row70_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row70_col3 {
            background-color:  #74c074;
            background-color:  #74c074;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row71_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row71_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row71_col2 {
            background-color:  #008000;
            background-color:  #008000;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row71_col3 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row72_col0 {
            background-color:  #e3fee3;
            background-color:  #e3fee3;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row72_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row72_col2 {
            background-color:  #038103;
            background-color:  #038103;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row72_col3 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row73_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row73_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row73_col2 {
            background-color:  #e0fce0;
            background-color:  #e0fce0;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row73_col3 {
            background-color:  #068306;
            background-color:  #068306;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row74_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row74_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row74_col2 {
            background-color:  #3ba13b;
            background-color:  #3ba13b;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row74_col3 {
            background-color:  #aadeaa;
            background-color:  #aadeaa;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row75_col0 {
            background-color:  #a2daa2;
            background-color:  #a2daa2;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row75_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row75_col2 {
            background-color:  #8ecf8e;
            background-color:  #8ecf8e;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row75_col3 {
            background-color:  #9bd69b;
            background-color:  #9bd69b;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row76_col0 {
            background-color:  #87cb87;
            background-color:  #87cb87;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row76_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row76_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row76_col3 {
            background-color:  #5eb45e;
            background-color:  #5eb45e;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row77_col0 {
            background-color:  #56b056;
            background-color:  #56b056;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row77_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row77_col2 {
            background-color:  #caf0ca;
            background-color:  #caf0ca;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row77_col3 {
            background-color:  #abdfab;
            background-color:  #abdfab;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row78_col0 {
            background-color:  #3fa33f;
            background-color:  #3fa33f;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row78_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row78_col2 {
            background-color:  #caf0ca;
            background-color:  #caf0ca;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row78_col3 {
            background-color:  #c2ecc2;
            background-color:  #c2ecc2;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row79_col0 {
            background-color:  #76c176;
            background-color:  #76c176;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row79_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row79_col2 {
            background-color:  #ccf1cc;
            background-color:  #ccf1cc;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row79_col3 {
            background-color:  #8acc8a;
            background-color:  #8acc8a;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row80_col0 {
            background-color:  #4fac4f;
            background-color:  #4fac4f;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row80_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row80_col2 {
            background-color:  #96d396;
            background-color:  #96d396;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row80_col3 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row81_col0 {
            background-color:  #b1e2b1;
            background-color:  #b1e2b1;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row81_col1 {
            background-color:  #d2f4d2;
            background-color:  #d2f4d2;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row81_col2 {
            background-color:  #6bbb6b;
            background-color:  #6bbb6b;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row81_col3 {
            background-color:  #c3ecc3;
            background-color:  #c3ecc3;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row82_col0 {
            background-color:  #caf0ca;
            background-color:  #caf0ca;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row82_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row82_col2 {
            background-color:  #b8e6b8;
            background-color:  #b8e6b8;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row82_col3 {
            background-color:  #4aa94a;
            background-color:  #4aa94a;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row83_col0 {
            background-color:  #48a848;
            background-color:  #48a848;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row83_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row83_col2 {
            background-color:  #9ed79e;
            background-color:  #9ed79e;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row83_col3 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row84_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row84_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row84_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row84_col3 {
            background-color:  #008000;
            background-color:  #008000;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row85_col0 {
            background-color:  #56af56;
            background-color:  #56af56;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row85_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row85_col2 {
            background-color:  #90d090;
            background-color:  #90d090;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row85_col3 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row86_col0 {
            background-color:  #70be70;
            background-color:  #70be70;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row86_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row86_col2 {
            background-color:  #d3f5d3;
            background-color:  #d3f5d3;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row86_col3 {
            background-color:  #89cc89;
            background-color:  #89cc89;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row87_col0 {
            background-color:  #59b159;
            background-color:  #59b159;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row87_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row87_col2 {
            background-color:  #c2ecc2;
            background-color:  #c2ecc2;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row87_col3 {
            background-color:  #b0e2b0;
            background-color:  #b0e2b0;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row88_col0 {
            background-color:  #88cb88;
            background-color:  #88cb88;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row88_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row88_col2 {
            background-color:  #e1fde1;
            background-color:  #e1fde1;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row88_col3 {
            background-color:  #62b662;
            background-color:  #62b662;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row89_col0 {
            background-color:  #7bc47b;
            background-color:  #7bc47b;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row89_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row89_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row89_col3 {
            background-color:  #6abb6a;
            background-color:  #6abb6a;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row90_col0 {
            background-color:  #a6dca6;
            background-color:  #a6dca6;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row90_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row90_col2 {
            background-color:  #b4e4b4;
            background-color:  #b4e4b4;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row90_col3 {
            background-color:  #71bf71;
            background-color:  #71bf71;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row91_col0 {
            background-color:  #8dce8d;
            background-color:  #8dce8d;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row91_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row91_col2 {
            background-color:  #d5f6d5;
            background-color:  #d5f6d5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row91_col3 {
            background-color:  #68ba68;
            background-color:  #68ba68;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row92_col0 {
            background-color:  #8ecf8e;
            background-color:  #8ecf8e;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row92_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row92_col2 {
            background-color:  #dcfadc;
            background-color:  #dcfadc;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row92_col3 {
            background-color:  #61b661;
            background-color:  #61b661;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row93_col0 {
            background-color:  #a5dba5;
            background-color:  #a5dba5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row93_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row93_col2 {
            background-color:  #93d193;
            background-color:  #93d193;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row93_col3 {
            background-color:  #94d294;
            background-color:  #94d294;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row94_col0 {
            background-color:  #e1fde1;
            background-color:  #e1fde1;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row94_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row94_col2 {
            background-color:  #148b14;
            background-color:  #148b14;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row94_col3 {
            background-color:  #d7f7d7;
            background-color:  #d7f7d7;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row95_col0 {
            background-color:  #008000;
            background-color:  #008000;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row95_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row95_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row95_col3 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row96_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row96_col1 {
            background-color:  #008000;
            background-color:  #008000;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row96_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row96_col3 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row97_col0 {
            background-color:  #008000;
            background-color:  #008000;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row97_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row97_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row97_col3 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row98_col0 {
            background-color:  #defbde;
            background-color:  #defbde;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row98_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row98_col2 {
            background-color:  #80c780;
            background-color:  #80c780;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row98_col3 {
            background-color:  #6ebd6e;
            background-color:  #6ebd6e;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row99_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row99_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row99_col2 {
            background-color:  #018001;
            background-color:  #018001;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row99_col3 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row100_col0 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row100_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row100_col2 {
            background-color:  #4eab4e;
            background-color:  #4eab4e;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row100_col3 {
            background-color:  #97d497;
            background-color:  #97d497;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row101_col0 {
            background-color:  #caf0ca;
            background-color:  #caf0ca;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row101_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row101_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row101_col3 {
            background-color:  #1d901d;
            background-color:  #1d901d;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row102_col0 {
            background-color:  #008000;
            background-color:  #008000;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row102_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row102_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row102_col3 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row103_col0 {
            background-color:  #008000;
            background-color:  #008000;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row103_col1 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row103_col2 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }    #T_41d740c6_5e32_11e9_bb92_4cedfb93a259row103_col3 {
            background-color:  #e5ffe5;
            background-color:  #e5ffe5;
        }</style>  
<table id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >Topic0</th> 
        <th class="col_heading level0 col1" >Topic1</th> 
        <th class="col_heading level0 col2" >Topic2</th> 
        <th class="col_heading level0 col3" >Topic3</th> 
        <th class="col_heading level0 col4" >Dominant_Topic</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row0" class="row_heading level0 row0" >Book0</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row0_col0" class="data row0 col0" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row0_col1" class="data row0 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row0_col2" class="data row0 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row0_col3" class="data row0 col3" >1</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row0_col4" class="data row0 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row1" class="row_heading level0 row1" >Book1</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row1_col0" class="data row1 col0" >0.2</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row1_col1" class="data row1 col1" >0.014</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row1_col2" class="data row1 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row1_col3" class="data row1 col3" >0.786</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row1_col4" class="data row1 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row2" class="row_heading level0 row2" >Book2</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row2_col0" class="data row2 col0" >0.119</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row2_col1" class="data row2 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row2_col2" class="data row2 col2" >0.44</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row2_col3" class="data row2 col3" >0.441</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row2_col4" class="data row2 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row3" class="row_heading level0 row3" >Book3</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row3_col0" class="data row3 col0" >0.008</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row3_col1" class="data row3 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row3_col2" class="data row3 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row3_col3" class="data row3 col3" >0.992</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row3_col4" class="data row3 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row4" class="row_heading level0 row4" >Book4</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row4_col0" class="data row4 col0" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row4_col1" class="data row4 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row4_col2" class="data row4 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row4_col3" class="data row4 col3" >1</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row4_col4" class="data row4 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row5" class="row_heading level0 row5" >Book5</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row5_col0" class="data row5 col0" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row5_col1" class="data row5 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row5_col2" class="data row5 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row5_col3" class="data row5 col3" >1</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row5_col4" class="data row5 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row6" class="row_heading level0 row6" >Book6</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row6_col0" class="data row6 col0" >0.531</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row6_col1" class="data row6 col1" >0.01</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row6_col2" class="data row6 col2" >0.266</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row6_col3" class="data row6 col3" >0.194</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row6_col4" class="data row6 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row7" class="row_heading level0 row7" >Book7</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row7_col0" class="data row7 col0" >0.444</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row7_col1" class="data row7 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row7_col2" class="data row7 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row7_col3" class="data row7 col3" >0.556</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row7_col4" class="data row7 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row8" class="row_heading level0 row8" >Book8</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row8_col0" class="data row8 col0" >0.04</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row8_col1" class="data row8 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row8_col2" class="data row8 col2" >0.05</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row8_col3" class="data row8 col3" >0.91</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row8_col4" class="data row8 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row9" class="row_heading level0 row9" >Book9</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row9_col0" class="data row9 col0" >0.015</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row9_col1" class="data row9 col1" >0.013</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row9_col2" class="data row9 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row9_col3" class="data row9 col3" >0.972</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row9_col4" class="data row9 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row10" class="row_heading level0 row10" >Book10</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row10_col0" class="data row10 col0" >0.456</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row10_col1" class="data row10 col1" >0.003</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row10_col2" class="data row10 col2" >0.003</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row10_col3" class="data row10 col3" >0.538</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row10_col4" class="data row10 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row11" class="row_heading level0 row11" >Book11</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row11_col0" class="data row11 col0" >0.022</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row11_col1" class="data row11 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row11_col2" class="data row11 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row11_col3" class="data row11 col3" >0.978</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row11_col4" class="data row11 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row12" class="row_heading level0 row12" >Book12</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row12_col0" class="data row12 col0" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row12_col1" class="data row12 col1" >0.004</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row12_col2" class="data row12 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row12_col3" class="data row12 col3" >0.996</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row12_col4" class="data row12 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row13" class="row_heading level0 row13" >Book13</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row13_col0" class="data row13 col0" >0.001</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row13_col1" class="data row13 col1" >0.008</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row13_col2" class="data row13 col2" >0.134</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row13_col3" class="data row13 col3" >0.857</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row13_col4" class="data row13 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row14" class="row_heading level0 row14" >Book14</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row14_col0" class="data row14 col0" >0.035</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row14_col1" class="data row14 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row14_col2" class="data row14 col2" >0.103</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row14_col3" class="data row14 col3" >0.862</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row14_col4" class="data row14 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row15" class="row_heading level0 row15" >Book15</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row15_col0" class="data row15 col0" >0.078</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row15_col1" class="data row15 col1" >0.007</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row15_col2" class="data row15 col2" >0.115</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row15_col3" class="data row15 col3" >0.8</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row15_col4" class="data row15 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row16" class="row_heading level0 row16" >Book16</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row16_col0" class="data row16 col0" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row16_col1" class="data row16 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row16_col2" class="data row16 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row16_col3" class="data row16 col3" >1</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row16_col4" class="data row16 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row17" class="row_heading level0 row17" >Book17</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row17_col0" class="data row17 col0" >0.004</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row17_col1" class="data row17 col1" >0.003</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row17_col2" class="data row17 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row17_col3" class="data row17 col3" >0.992</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row17_col4" class="data row17 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row18" class="row_heading level0 row18" >Book18</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row18_col0" class="data row18 col0" >0.015</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row18_col1" class="data row18 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row18_col2" class="data row18 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row18_col3" class="data row18 col3" >0.985</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row18_col4" class="data row18 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row19" class="row_heading level0 row19" >Book19</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row19_col0" class="data row19 col0" >0.039</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row19_col1" class="data row19 col1" >0.003</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row19_col2" class="data row19 col2" >0.091</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row19_col3" class="data row19 col3" >0.867</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row19_col4" class="data row19 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row20" class="row_heading level0 row20" >Book20</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row20_col0" class="data row20 col0" >0.017</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row20_col1" class="data row20 col1" >0.003</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row20_col2" class="data row20 col2" >0.267</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row20_col3" class="data row20 col3" >0.713</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row20_col4" class="data row20 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row21" class="row_heading level0 row21" >Book21</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row21_col0" class="data row21 col0" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row21_col1" class="data row21 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row21_col2" class="data row21 col2" >0.443</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row21_col3" class="data row21 col3" >0.557</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row21_col4" class="data row21 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row22" class="row_heading level0 row22" >Book22</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row22_col0" class="data row22 col0" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row22_col1" class="data row22 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row22_col2" class="data row22 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row22_col3" class="data row22 col3" >1</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row22_col4" class="data row22 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row23" class="row_heading level0 row23" >Book23</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row23_col0" class="data row23 col0" >0.286</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row23_col1" class="data row23 col1" >0.004</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row23_col2" class="data row23 col2" >0.061</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row23_col3" class="data row23 col3" >0.648</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row23_col4" class="data row23 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row24" class="row_heading level0 row24" >Book24</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row24_col0" class="data row24 col0" >0.004</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row24_col1" class="data row24 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row24_col2" class="data row24 col2" >0.026</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row24_col3" class="data row24 col3" >0.97</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row24_col4" class="data row24 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row25" class="row_heading level0 row25" >Book25</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row25_col0" class="data row25 col0" >0.05</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row25_col1" class="data row25 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row25_col2" class="data row25 col2" >0.203</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row25_col3" class="data row25 col3" >0.747</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row25_col4" class="data row25 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row26" class="row_heading level0 row26" >Book26</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row26_col0" class="data row26 col0" >0.002</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row26_col1" class="data row26 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row26_col2" class="data row26 col2" >0.363</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row26_col3" class="data row26 col3" >0.636</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row26_col4" class="data row26 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row27" class="row_heading level0 row27" >Book27</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row27_col0" class="data row27 col0" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row27_col1" class="data row27 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row27_col2" class="data row27 col2" >0.314</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row27_col3" class="data row27 col3" >0.686</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row27_col4" class="data row27 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row28" class="row_heading level0 row28" >Book28</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row28_col0" class="data row28 col0" >0.076</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row28_col1" class="data row28 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row28_col2" class="data row28 col2" >0.194</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row28_col3" class="data row28 col3" >0.73</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row28_col4" class="data row28 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row29" class="row_heading level0 row29" >Book29</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row29_col0" class="data row29 col0" >0.001</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row29_col1" class="data row29 col1" >0.016</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row29_col2" class="data row29 col2" >0.22</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row29_col3" class="data row29 col3" >0.763</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row29_col4" class="data row29 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row30" class="row_heading level0 row30" >Book30</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row30_col0" class="data row30 col0" >0.005</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row30_col1" class="data row30 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row30_col2" class="data row30 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row30_col3" class="data row30 col3" >0.994</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row30_col4" class="data row30 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row31" class="row_heading level0 row31" >Book31</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row31_col0" class="data row31 col0" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row31_col1" class="data row31 col1" >0.001</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row31_col2" class="data row31 col2" >0.004</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row31_col3" class="data row31 col3" >0.995</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row31_col4" class="data row31 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row32" class="row_heading level0 row32" >Book32</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row32_col0" class="data row32 col0" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row32_col1" class="data row32 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row32_col2" class="data row32 col2" >0.025</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row32_col3" class="data row32 col3" >0.975</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row32_col4" class="data row32 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row33" class="row_heading level0 row33" >Book33</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row33_col0" class="data row33 col0" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row33_col1" class="data row33 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row33_col2" class="data row33 col2" >0.18</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row33_col3" class="data row33 col3" >0.819</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row33_col4" class="data row33 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row34" class="row_heading level0 row34" >Book34</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row34_col0" class="data row34 col0" >0.573</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row34_col1" class="data row34 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row34_col2" class="data row34 col2" >0.185</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row34_col3" class="data row34 col3" >0.242</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row34_col4" class="data row34 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row35" class="row_heading level0 row35" >Book35</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row35_col0" class="data row35 col0" >0.001</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row35_col1" class="data row35 col1" >0.137</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row35_col2" class="data row35 col2" >0.76</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row35_col3" class="data row35 col3" >0.102</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row35_col4" class="data row35 col4" >2</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row36" class="row_heading level0 row36" >Book36</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row36_col0" class="data row36 col0" >0.048</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row36_col1" class="data row36 col1" >0.749</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row36_col2" class="data row36 col2" >0.098</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row36_col3" class="data row36 col3" >0.105</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row36_col4" class="data row36 col4" >1</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row37" class="row_heading level0 row37" >Book37</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row37_col0" class="data row37 col0" >0.014</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row37_col1" class="data row37 col1" >0.784</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row37_col2" class="data row37 col2" >0.047</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row37_col3" class="data row37 col3" >0.155</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row37_col4" class="data row37 col4" >1</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row38" class="row_heading level0 row38" >Book38</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row38_col0" class="data row38 col0" >0.101</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row38_col1" class="data row38 col1" >0.502</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row38_col2" class="data row38 col2" >0.194</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row38_col3" class="data row38 col3" >0.202</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row38_col4" class="data row38 col4" >1</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row39" class="row_heading level0 row39" >Book39</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row39_col0" class="data row39 col0" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row39_col1" class="data row39 col1" >1</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row39_col2" class="data row39 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row39_col3" class="data row39 col3" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row39_col4" class="data row39 col4" >1</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row40" class="row_heading level0 row40" >Book40</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row40_col0" class="data row40 col0" >0.531</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row40_col1" class="data row40 col1" >0.01</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row40_col2" class="data row40 col2" >0.266</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row40_col3" class="data row40 col3" >0.194</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row40_col4" class="data row40 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row41" class="row_heading level0 row41" >Book41</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row41_col0" class="data row41 col0" >0.114</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row41_col1" class="data row41 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row41_col2" class="data row41 col2" >0.834</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row41_col3" class="data row41 col3" >0.052</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row41_col4" class="data row41 col4" >2</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row42" class="row_heading level0 row42" >Book42</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row42_col0" class="data row42 col0" >0.803</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row42_col1" class="data row42 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row42_col2" class="data row42 col2" >0.075</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row42_col3" class="data row42 col3" >0.122</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row42_col4" class="data row42 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row43" class="row_heading level0 row43" >Book43</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row43_col0" class="data row43 col0" >0.054</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row43_col1" class="data row43 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row43_col2" class="data row43 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row43_col3" class="data row43 col3" >0.946</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row43_col4" class="data row43 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row44" class="row_heading level0 row44" >Book44</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row44_col0" class="data row44 col0" >0.799</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row44_col1" class="data row44 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row44_col2" class="data row44 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row44_col3" class="data row44 col3" >0.201</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row44_col4" class="data row44 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row45" class="row_heading level0 row45" >Book45</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row45_col0" class="data row45 col0" >0.354</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row45_col1" class="data row45 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row45_col2" class="data row45 col2" >0.25</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row45_col3" class="data row45 col3" >0.397</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row45_col4" class="data row45 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row46" class="row_heading level0 row46" >Book46</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row46_col0" class="data row46 col0" >0.879</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row46_col1" class="data row46 col1" >0.001</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row46_col2" class="data row46 col2" >0.001</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row46_col3" class="data row46 col3" >0.119</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row46_col4" class="data row46 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row47" class="row_heading level0 row47" >Book47</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row47_col0" class="data row47 col0" >0.548</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row47_col1" class="data row47 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row47_col2" class="data row47 col2" >0.104</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row47_col3" class="data row47 col3" >0.347</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row47_col4" class="data row47 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row48" class="row_heading level0 row48" >Book48</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row48_col0" class="data row48 col0" >0.811</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row48_col1" class="data row48 col1" >0.001</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row48_col2" class="data row48 col2" >0.086</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row48_col3" class="data row48 col3" >0.101</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row48_col4" class="data row48 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row49" class="row_heading level0 row49" >Book49</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row49_col0" class="data row49 col0" >0.206</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row49_col1" class="data row49 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row49_col2" class="data row49 col2" >0.012</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row49_col3" class="data row49 col3" >0.782</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row49_col4" class="data row49 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row50" class="row_heading level0 row50" >Book50</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row50_col0" class="data row50 col0" >0.198</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row50_col1" class="data row50 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row50_col2" class="data row50 col2" >0.006</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row50_col3" class="data row50 col3" >0.796</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row50_col4" class="data row50 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row51" class="row_heading level0 row51" >Book51</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row51_col0" class="data row51 col0" >0.291</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row51_col1" class="data row51 col1" >0.002</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row51_col2" class="data row51 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row51_col3" class="data row51 col3" >0.707</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row51_col4" class="data row51 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row52" class="row_heading level0 row52" >Book52</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row52_col0" class="data row52 col0" >0.192</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row52_col1" class="data row52 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row52_col2" class="data row52 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row52_col3" class="data row52 col3" >0.808</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row52_col4" class="data row52 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row53" class="row_heading level0 row53" >Book53</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row53_col0" class="data row53 col0" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row53_col1" class="data row53 col1" >0.973</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row53_col2" class="data row53 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row53_col3" class="data row53 col3" >0.027</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row53_col4" class="data row53 col4" >1</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row54" class="row_heading level0 row54" >Book54</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row54_col0" class="data row54 col0" >0.626</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row54_col1" class="data row54 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row54_col2" class="data row54 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row54_col3" class="data row54 col3" >0.374</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row54_col4" class="data row54 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row55" class="row_heading level0 row55" >Book55</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row55_col0" class="data row55 col0" >0.338</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row55_col1" class="data row55 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row55_col2" class="data row55 col2" >0.303</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row55_col3" class="data row55 col3" >0.359</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row55_col4" class="data row55 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row56" class="row_heading level0 row56" >Book56</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row56_col0" class="data row56 col0" >0.81</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row56_col1" class="data row56 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row56_col2" class="data row56 col2" >0.19</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row56_col3" class="data row56 col3" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row56_col4" class="data row56 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row57" class="row_heading level0 row57" >Book57</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row57_col0" class="data row57 col0" >0.776</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row57_col1" class="data row57 col1" >0.002</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row57_col2" class="data row57 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row57_col3" class="data row57 col3" >0.222</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row57_col4" class="data row57 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row58" class="row_heading level0 row58" >Book58</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row58_col0" class="data row58 col0" >0.004</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row58_col1" class="data row58 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row58_col2" class="data row58 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row58_col3" class="data row58 col3" >0.996</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row58_col4" class="data row58 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row59" class="row_heading level0 row59" >Book59</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row59_col0" class="data row59 col0" >0.645</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row59_col1" class="data row59 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row59_col2" class="data row59 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row59_col3" class="data row59 col3" >0.354</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row59_col4" class="data row59 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row60" class="row_heading level0 row60" >Book60</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row60_col0" class="data row60 col0" >1</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row60_col1" class="data row60 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row60_col2" class="data row60 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row60_col3" class="data row60 col3" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row60_col4" class="data row60 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row61" class="row_heading level0 row61" >Book61</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row61_col0" class="data row61 col0" >0.256</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row61_col1" class="data row61 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row61_col2" class="data row61 col2" >0.249</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row61_col3" class="data row61 col3" >0.495</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row61_col4" class="data row61 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row62" class="row_heading level0 row62" >Book62</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row62_col0" class="data row62 col0" >0.349</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row62_col1" class="data row62 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row62_col2" class="data row62 col2" >0.023</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row62_col3" class="data row62 col3" >0.628</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row62_col4" class="data row62 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row63" class="row_heading level0 row63" >Book63</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row63_col0" class="data row63 col0" >0.737</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row63_col1" class="data row63 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row63_col2" class="data row63 col2" >0.042</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row63_col3" class="data row63 col3" >0.221</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row63_col4" class="data row63 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row64" class="row_heading level0 row64" >Book64</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row64_col0" class="data row64 col0" >0.812</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row64_col1" class="data row64 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row64_col2" class="data row64 col2" >0.188</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row64_col3" class="data row64 col3" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row64_col4" class="data row64 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row65" class="row_heading level0 row65" >Book65</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row65_col0" class="data row65 col0" >0.115</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row65_col1" class="data row65 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row65_col2" class="data row65 col2" >0.385</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row65_col3" class="data row65 col3" >0.5</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row65_col4" class="data row65 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row66" class="row_heading level0 row66" >Book66</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row66_col0" class="data row66 col0" >0.657</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row66_col1" class="data row66 col1" >0.007</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row66_col2" class="data row66 col2" >0.332</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row66_col3" class="data row66 col3" >0.004</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row66_col4" class="data row66 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row67" class="row_heading level0 row67" >Book67</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row67_col0" class="data row67 col0" >0.714</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row67_col1" class="data row67 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row67_col2" class="data row67 col2" >0.038</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row67_col3" class="data row67 col3" >0.248</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row67_col4" class="data row67 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row68" class="row_heading level0 row68" >Book68</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row68_col0" class="data row68 col0" >0.899</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row68_col1" class="data row68 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row68_col2" class="data row68 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row68_col3" class="data row68 col3" >0.101</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row68_col4" class="data row68 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row69" class="row_heading level0 row69" >Book69</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row69_col0" class="data row69 col0" >0.95</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row69_col1" class="data row69 col1" >0.001</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row69_col2" class="data row69 col2" >0.049</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row69_col3" class="data row69 col3" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row69_col4" class="data row69 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row70" class="row_heading level0 row70" >Book70</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row70_col0" class="data row70 col0" >0.506</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row70_col1" class="data row70 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row70_col2" class="data row70 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row70_col3" class="data row70 col3" >0.494</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row70_col4" class="data row70 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row71" class="row_heading level0 row71" >Book71</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row71_col0" class="data row71 col0" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row71_col1" class="data row71 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row71_col2" class="data row71 col2" >1</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row71_col3" class="data row71 col3" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row71_col4" class="data row71 col4" >2</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row72" class="row_heading level0 row72" >Book72</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row72_col0" class="data row72 col0" >0.014</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row72_col1" class="data row72 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row72_col2" class="data row72 col2" >0.986</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row72_col3" class="data row72 col3" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row72_col4" class="data row72 col4" >2</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row73" class="row_heading level0 row73" >Book73</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row73_col0" class="data row73 col0" >0.004</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row73_col1" class="data row73 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row73_col2" class="data row73 col2" >0.026</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row73_col3" class="data row73 col3" >0.97</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row73_col4" class="data row73 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row74" class="row_heading level0 row74" >Book74</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row74_col0" class="data row74 col0" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row74_col1" class="data row74 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row74_col2" class="data row74 col2" >0.741</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row74_col3" class="data row74 col3" >0.259</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row74_col4" class="data row74 col4" >2</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row75" class="row_heading level0 row75" >Book75</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row75_col0" class="data row75 col0" >0.294</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row75_col1" class="data row75 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row75_col2" class="data row75 col2" >0.379</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row75_col3" class="data row75 col3" >0.326</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row75_col4" class="data row75 col4" >2</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row76" class="row_heading level0 row76" >Book76</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row76_col0" class="data row76 col0" >0.412</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row76_col1" class="data row76 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row76_col2" class="data row76 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row76_col3" class="data row76 col3" >0.588</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row76_col4" class="data row76 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row77" class="row_heading level0 row77" >Book77</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row77_col0" class="data row77 col0" >0.624</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row77_col1" class="data row77 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row77_col2" class="data row77 col2" >0.12</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row77_col3" class="data row77 col3" >0.256</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row77_col4" class="data row77 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row78" class="row_heading level0 row78" >Book78</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row78_col0" class="data row78 col0" >0.723</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row78_col1" class="data row78 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row78_col2" class="data row78 col2" >0.122</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row78_col3" class="data row78 col3" >0.155</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row78_col4" class="data row78 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row79" class="row_heading level0 row79" >Book79</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row79_col0" class="data row79 col0" >0.485</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row79_col1" class="data row79 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row79_col2" class="data row79 col2" >0.112</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row79_col3" class="data row79 col3" >0.402</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row79_col4" class="data row79 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row80" class="row_heading level0 row80" >Book80</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row80_col0" class="data row80 col0" >0.655</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row80_col1" class="data row80 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row80_col2" class="data row80 col2" >0.345</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row80_col3" class="data row80 col3" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row80_col4" class="data row80 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row81" class="row_heading level0 row81" >Book81</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row81_col0" class="data row81 col0" >0.23</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row81_col1" class="data row81 col1" >0.088</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row81_col2" class="data row81 col2" >0.534</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row81_col3" class="data row81 col3" >0.149</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row81_col4" class="data row81 col4" >2</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row82" class="row_heading level0 row82" >Book82</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row82_col0" class="data row82 col0" >0.124</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row82_col1" class="data row82 col1" >0.001</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row82_col2" class="data row82 col2" >0.196</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row82_col3" class="data row82 col3" >0.679</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row82_col4" class="data row82 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row83" class="row_heading level0 row83" >Book83</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row83_col0" class="data row83 col0" >0.687</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row83_col1" class="data row83 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row83_col2" class="data row83 col2" >0.313</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row83_col3" class="data row83 col3" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row83_col4" class="data row83 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row84" class="row_heading level0 row84" >Book84</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row84_col0" class="data row84 col0" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row84_col1" class="data row84 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row84_col2" class="data row84 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row84_col3" class="data row84 col3" >1</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row84_col4" class="data row84 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row85" class="row_heading level0 row85" >Book85</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row85_col0" class="data row85 col0" >0.627</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row85_col1" class="data row85 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row85_col2" class="data row85 col2" >0.373</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row85_col3" class="data row85 col3" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row85_col4" class="data row85 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row86" class="row_heading level0 row86" >Book86</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row86_col0" class="data row86 col0" >0.514</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row86_col1" class="data row86 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row86_col2" class="data row86 col2" >0.081</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row86_col3" class="data row86 col3" >0.405</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row86_col4" class="data row86 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row87" class="row_heading level0 row87" >Book87</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row87_col0" class="data row87 col0" >0.612</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row87_col1" class="data row87 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row87_col2" class="data row87 col2" >0.155</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row87_col3" class="data row87 col3" >0.233</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row87_col4" class="data row87 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row88" class="row_heading level0 row88" >Book88</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row88_col0" class="data row88 col0" >0.407</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row88_col1" class="data row88 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row88_col2" class="data row88 col2" >0.02</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row88_col3" class="data row88 col3" >0.573</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row88_col4" class="data row88 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row89" class="row_heading level0 row89" >Book89</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row89_col0" class="data row89 col0" >0.462</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row89_col1" class="data row89 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row89_col2" class="data row89 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row89_col3" class="data row89 col3" >0.538</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row89_col4" class="data row89 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row90" class="row_heading level0 row90" >Book90</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row90_col0" class="data row90 col0" >0.278</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row90_col1" class="data row90 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row90_col2" class="data row90 col2" >0.218</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row90_col3" class="data row90 col3" >0.505</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row90_col4" class="data row90 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row91" class="row_heading level0 row91" >Book91</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row91_col0" class="data row91 col0" >0.383</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row91_col1" class="data row91 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row91_col2" class="data row91 col2" >0.072</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row91_col3" class="data row91 col3" >0.545</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row91_col4" class="data row91 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row92" class="row_heading level0 row92" >Book92</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row92_col0" class="data row92 col0" >0.381</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row92_col1" class="data row92 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row92_col2" class="data row92 col2" >0.043</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row92_col3" class="data row92 col3" >0.576</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row92_col4" class="data row92 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row93" class="row_heading level0 row93" >Book93</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row93_col0" class="data row93 col0" >0.285</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row93_col1" class="data row93 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row93_col2" class="data row93 col2" >0.361</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row93_col3" class="data row93 col3" >0.354</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row93_col4" class="data row93 col4" >2</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row94" class="row_heading level0 row94" >Book94</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row94_col0" class="data row94 col0" >0.023</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row94_col1" class="data row94 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row94_col2" class="data row94 col2" >0.911</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row94_col3" class="data row94 col3" >0.065</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row94_col4" class="data row94 col4" >2</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row95" class="row_heading level0 row95" >Book95</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row95_col0" class="data row95 col0" >1</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row95_col1" class="data row95 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row95_col2" class="data row95 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row95_col3" class="data row95 col3" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row95_col4" class="data row95 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row96" class="row_heading level0 row96" >Book96</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row96_col0" class="data row96 col0" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row96_col1" class="data row96 col1" >1</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row96_col2" class="data row96 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row96_col3" class="data row96 col3" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row96_col4" class="data row96 col4" >1</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row97" class="row_heading level0 row97" >Book97</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row97_col0" class="data row97 col0" >1</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row97_col1" class="data row97 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row97_col2" class="data row97 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row97_col3" class="data row97 col3" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row97_col4" class="data row97 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row98" class="row_heading level0 row98" >Book98</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row98_col0" class="data row98 col0" >0.035</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row98_col1" class="data row98 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row98_col2" class="data row98 col2" >0.444</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row98_col3" class="data row98 col3" >0.52</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row98_col4" class="data row98 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row99" class="row_heading level0 row99" >Book99</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row99_col0" class="data row99 col0" >0.002</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row99_col1" class="data row99 col1" >0.002</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row99_col2" class="data row99 col2" >0.994</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row99_col3" class="data row99 col3" >0.002</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row99_col4" class="data row99 col4" >2</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row100" class="row_heading level0 row100" >Book100</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row100_col0" class="data row100 col0" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row100_col1" class="data row100 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row100_col2" class="data row100 col2" >0.657</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row100_col3" class="data row100 col3" >0.343</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row100_col4" class="data row100 col4" >2</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row101" class="row_heading level0 row101" >Book101</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row101_col0" class="data row101 col0" >0.122</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row101_col1" class="data row101 col1" >0.002</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row101_col2" class="data row101 col2" >0.002</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row101_col3" class="data row101 col3" >0.874</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row101_col4" class="data row101 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row102" class="row_heading level0 row102" >Book102</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row102_col0" class="data row102 col0" >1</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row102_col1" class="data row102 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row102_col2" class="data row102 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row102_col3" class="data row102 col3" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row102_col4" class="data row102 col4" >0</td> 
    </tr>    <tr> 
        <th id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259level0_row103" class="row_heading level0 row103" >Book103</th> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row103_col0" class="data row103 col0" >1</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row103_col1" class="data row103 col1" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row103_col2" class="data row103 col2" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row103_col3" class="data row103 col3" >0</td> 
        <td id="T_41d740c6_5e32_11e9_bb92_4cedfb93a259row103_col4" class="data row103 col4" >0</td> 
    </tr></tbody> 
</table> 




```python
sns.set(font_scale=1.5)
fig, ax = plt.subplots(figsize=(15,7))
sns.set_style("darkgrid")
g=sns.catplot(hue='Dominant_Topic', x='Dominant_Topic',
                 data=data, kind="count",ax=ax);
#ax.set_xticklabels(rotation=90)
ax.set_xticklabels(df_top5words.Topic)#+'\n'+df_top5words.Top5Words);
ax.set_title(r'Distributions Topics in the books')

Topic_Distribution = data['Dominant_Topic'].value_counts().reset_index(name="Num Documents")
Topic_Distribution.columns = ['Topic Num', 'Num Documents']
Topic_Distribution

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Topic Num</th>
      <th>Num Documents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>32</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




![png](output_17_1.png)



![png](output_17_2.png)


### Visualizing LDA  Results with pyLDAvis


```python
pyLDAvis.enable_notebook()
dash2 = pyLDAvis.sklearn.prepare(lda, data_vectorized_cv, vectorizer_cv, mds='tsne')
dash2
```





<link rel="stylesheet" type="text/css" href="https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css">


<div id="ldavis_el104961400367656731765778801472"></div>
<script type="text/javascript">

var ldavis_el104961400367656731765778801472_data = {"mdsDat": {"Freq": [45.8327528491568, 26.144344979388727, 22.55505547941353, 5.467846692040931], "cluster": [1, 1, 1, 1], "topics": [1, 2, 3, 4], "x": [-22.189210891723633, 34.612003326416016, 63.015480041503906, -50.59236526489258], "y": [-265.2434387207031, -151.63626098632812, -236.8407745361328, -180.03948974609375]}, "tinfo": {"Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4"], "Freq": [10103.0, 7896.0, 3926.0, 3551.0, 2775.0, 2575.0, 2209.0, 2012.0, 1957.0, 1820.0, 1559.0, 1554.0, 1436.0, 2200.0, 1263.0, 1246.0, 1154.0, 1108.0, 1262.0, 1073.0, 1007.0, 1662.0, 946.0, 946.0, 931.0, 886.0, 882.0, 2204.0, 835.0, 4947.0, 282.80561110188864, 207.54637074228026, 173.33327484597496, 167.28265691891923, 315.47771008006083, 158.21720132774863, 151.19235743400023, 154.1773319028708, 143.14129352078456, 142.13328970958412, 172.14900780081746, 144.12045930178704, 134.0761483445405, 129.0539873968138, 132.03230564818372, 124.01918895847615, 123.01101070531995, 112.9470367546919, 110.94467202813206, 109.9271953123469, 112.92474606541865, 108.92088308683148, 108.92071690098564, 125.84992179902004, 100.87863252080292, 125.82199349032304, 99.84999656379995, 95.84626514805174, 94.82600619061884, 93.82076220036254, 1749.6835785055657, 596.8473912187553, 1323.3805639565824, 1684.1120725583319, 383.389628916906, 243.31056912981998, 368.21763769323957, 220.45074509491752, 407.58323542653125, 357.57114401032607, 260.0961577417034, 2193.6085825787477, 1054.2226875462472, 1554.2730676519966, 486.20067359591195, 402.7937158650149, 235.81634847283826, 1741.361044809301, 2152.5829804278483, 962.2805462430124, 996.449793874468, 650.5326105729075, 506.9444490494864, 770.1645363610927, 330.8975446793145, 3354.912796652965, 918.5297833374686, 1158.050507635392, 515.3533738282467, 2075.7209012480607, 9476.900857503973, 4342.1878329840465, 3855.610550597436, 5893.542727937281, 1806.412373742114, 3317.6534447844865, 3159.909865833521, 871.2771908740942, 1682.041204206501, 2414.53890100071, 2968.745562966199, 4293.885250537907, 1112.529458316773, 2446.6965140555953, 1833.4073046823084, 1388.0226310428832, 1539.728603086431, 1890.9898832122767, 1695.673911373791, 1778.3565243346536, 2273.6005258576192, 2301.359754321626, 1766.0830261004419, 1559.2208209557612, 2183.4265296191656, 2422.118477448634, 2616.6107435278573, 2640.1483639164326, 2027.7119925164654, 1947.1983461419666, 1966.3313038814672, 1851.585956062421, 1871.1782731797502, 367.4167388902545, 234.60801521807974, 156.48636198005178, 147.69766854070252, 146.70631385812405, 126.21409656087593, 120.34593533152638, 89.10119716939458, 87.14978889221308, 267.1837206401444, 81.29304225047967, 81.2868989289125, 80.30753872366614, 144.15756723167436, 78.35667765359182, 79.31902464475678, 69.57567563986929, 69.57490349446385, 279.2414748206756, 66.64262331921802, 64.69091326127831, 64.68351817954773, 64.67807298541416, 62.738267290573354, 66.59487566126508, 152.42149962475276, 59.80929414589263, 65.59356038253365, 61.72201828183699, 57.85087282478506, 1073.4592177099717, 225.28588384007597, 484.3656396683293, 273.8143013029775, 161.84405395278887, 280.7528997959864, 1241.1367597147218, 530.783290317862, 370.32192329677997, 196.91958987481073, 221.14371737229445, 302.05525252030355, 241.46631711160154, 211.3826223494778, 911.290233616374, 433.6795943261704, 158.47218123019914, 631.5009927449571, 242.4207919307446, 761.9335085321255, 420.3031659232831, 698.3061490283075, 471.6027683049903, 2055.8771547049187, 408.1634185594085, 1035.659524459025, 1176.2971987862175, 747.2913693644532, 1605.8442191981248, 631.180286429104, 530.628358357512, 374.32404530493415, 2967.5486196255492, 964.5100881663452, 535.9333239351481, 1845.5181231544643, 1774.6093211678326, 1312.8190835284147, 830.2421935819699, 1409.1092319540548, 884.521384140824, 805.7576939359186, 1005.2871096632402, 895.6620889490792, 665.6487903110034, 1309.5689906819425, 1106.0379124524557, 1731.2534110176691, 1031.0019419476575, 864.1463220006083, 1056.4167430177022, 1244.1207296327398, 1161.1632637793412, 973.8554031315269, 937.4642153554588, 1017.4983282733109, 976.5400864886673, 1054.5352719925056, 1212.932062251675, 919.1293476885779, 931.9123794743944, 477.31831974879583, 293.5951919530474, 400.81711688843916, 221.49782622527084, 158.28260344520922, 201.44313465814253, 161.1328097776778, 139.50256044114073, 131.61350025292077, 115.80671420921209, 112.83217013671751, 152.06675003503585, 105.918624271581, 94.0817287114654, 87.16465492327944, 223.2564307091119, 84.18396566276304, 83.20186619196652, 81.23798356113751, 83.17034802663044, 86.10550403993442, 78.27455432101824, 78.2674861215293, 77.28585141095223, 123.2654109849991, 135.968385446165, 289.50300900079293, 76.29012491341747, 77.26806767456523, 75.30847792151488, 722.4720659984056, 444.0190553897084, 356.0956175965586, 276.5471120036907, 205.12811940882136, 128.89902832869825, 233.70364895382846, 151.76991783379958, 339.1847482349892, 287.98294974795414, 203.5497197081319, 429.2712206299668, 381.39742060774745, 471.5996894021445, 483.2553391781522, 271.8586064623882, 438.9697886771512, 248.9310869387832, 397.39620049922286, 252.94444910453515, 745.4136339943765, 733.563451475871, 954.390074390921, 713.9054300056114, 1505.5776809019978, 743.4918831898941, 1374.1278958838345, 475.9471805325805, 1907.3238483806751, 644.014221241475, 357.0165257687256, 1074.5268149141905, 962.9012134084477, 790.7814395975286, 1282.0337549997, 916.5535661343129, 1195.1586390165655, 925.9784313213163, 742.3956861556989, 874.2185678820668, 924.9179639991705, 1752.7233040684737, 1552.0969269300404, 1583.518730478234, 1900.5498055434462, 846.8373873768494, 569.9257591840848, 685.7413209921583, 681.8676794747676, 933.5843980281699, 1314.9359821381393, 1093.5648130210902, 992.9484213703136, 817.2024218396682, 728.9133173113233, 919.6034508429667, 790.7283183381231, 733.4894753756107, 1820.1181991165265, 1262.5784040689473, 885.515426587663, 2572.260347574014, 721.5691091400115, 544.1836641084611, 2770.160621877501, 2204.318461378177, 371.392018852955, 263.5118059301148, 529.7134039361856, 257.5236084686634, 588.9398424983027, 941.8039284100886, 1148.815064428242, 1239.9676733691692, 1068.1693805234934, 267.7370054578894, 471.99232913737154, 620.1417787693144, 306.8368008544232, 130.8608876645837, 155.64563425092052, 1537.6440100785412, 747.3550370602346, 60.963284700280305, 1094.4061961595232, 158.04459338982312, 3873.5917793786984, 330.11689020635026, 7784.045511171716, 3495.8361339445664, 1537.5778933254905, 1921.934278415573, 1411.070142836286, 933.0781340086348, 822.4195756102796, 9644.966910930065, 1959.0369290120068, 629.2711381642297, 838.1479633159734, 939.026799497755, 851.5993117879369, 915.1878131300763, 745.5955976175105], "Term": ["de", "la", "le", "en", "et", "que", "les", "un", "du", "il", "se", "el", "des", "son", "ce", "qui", "est", "dan", "pour", "au", "par", "sing", "sur", "plus", "al", "nous", "di", "point", "si", "water", "sigh", "aladdin", "tommy", "aunt", "tis", "ly", "maggie", "ter", "twas", "kate", "morn", "doth", "ing", "kitty", "hiawatha", "quoth", "winkle", "hark", "script", "tion", "glitter", "yon", "ty", "be", "magician", "sam", "aye", "willie", "magpie", "yonder", "thy", "tom", "dear", "oh", "fairy", "echo", "roar", "alice", "whisper", "bless", "ere", "cry", "thee", "thou", "robin", "hath", "softly", "voice", "heart", "song", "smile", "shout", "dance", "ride", "angel", "hear", "sky", "laugh", "tale", "love", "come", "tell", "boy", "go", "poor", "mother", "shall", "wave", "sit", "father", "eye", "think", "happy", "let", "face", "wind", "answer", "ask", "story", "lie", "fall", "word", "friend", "feel", "bird", "life", "leave", "take", "turn", "bear", "foot", "young", "water", "larva", "larv\u00e6", "pupa", "magnetic", "wireless", "antenn\u00e6", "oxygen", "zinc", "flavour", "magnet", "camper", "disc", "oval", "recipe", "carbon", "fjord", "lice", "louse", "donald", "chrysalis", "cockroach", "appendage", "skunk", "pollen", "cereal", "starch", "insects", "cicada", "membrane", "yap", "eclipse", "mineral", "vegetable", "moth", "temperature", "cooking", "insect", "wire", "electric", "motor", "porcelain", "caterpillar", "electricity", "buffaloes", "specie", "product", "acid", "method", "beetle", "inch", "reindeer", "sugar", "stem", "animal", "substance", "egg", "plant", "pupil", "food", "size", "material", "colour", "water", "camp", "current", "small", "form", "body", "fact", "use", "case", "fig", "cover", "different", "fruit", "end", "make", "leave", "ground", "number", "fly", "call", "bird", "point", "sun", "foot", "country", "take", "come", "mean", "year", "congress", "political", "parliament", "treaty", "commons", "consul", "administration", "charter", "frederic", "doctrine", "colonial", "legislature", "colonist", "nominate", "clergy", "napoleon", "magistrate", "turks", "confederate", "elector", "jealousy", "tarquin", "insurrection", "abolish", "clause", "institution", "military", "poland", "democratic", "cincinnati", "rome", "romans", "territory", "religion", "cum", "pitt", "slavery", "reform", "duke", "constitution", "cromwell", "national", "empire", "spain", "reign", "religious", "president", "senate", "roman", "revolution", "nation", "government", "army", "france", "war", "french", "england", "colony", "state", "lincoln", "minister", "general", "english", "history", "power", "law", "city", "united", "battle", "states", "thousand", "people", "country", "king", "year", "force", "public", "party", "act", "send", "take", "call", "life", "land", "death", "come", "form", "have", "il", "ce", "nous", "que", "terre", "jour", "et", "les", "op", "dit", "pas", "om", "los", "sur", "est", "qui", "au", "pret", "ind", "sa", "por", "casa", "fut", "el", "ne", "fil", "dan", "vous", "le", "las", "la", "en", "se", "du", "des", "plus", "si", "de", "un", "te", "di", "par", "al", "pour", "son"], "Total": [10103.0, 7896.0, 3926.0, 3551.0, 2775.0, 2575.0, 2209.0, 2012.0, 1957.0, 1820.0, 1559.0, 1554.0, 1436.0, 2200.0, 1263.0, 1246.0, 1154.0, 1108.0, 1262.0, 1073.0, 1007.0, 1662.0, 946.0, 946.0, 931.0, 886.0, 882.0, 2204.0, 835.0, 4947.0, 283.7482302362572, 208.28191082366712, 174.06765962856085, 168.0294281781922, 316.94839051124313, 158.97268338140304, 151.92897936787676, 154.94650745387244, 143.87854257641476, 142.87219656821435, 173.0580448295268, 144.8842944310982, 134.82161776606262, 129.79030533022723, 132.80761128520803, 124.7587641526882, 123.7524228863777, 113.68934425405817, 111.67698014719264, 110.67041719461751, 113.68866247499766, 109.66414070654463, 109.66411911227175, 126.76451002643985, 101.61389588180302, 126.767448332588, 100.60696627828318, 96.58236933405739, 95.57551568255843, 94.56938334995172, 1788.4675136008989, 604.6070582879296, 1354.7937584372353, 1731.263664347725, 388.2975100281421, 245.47174910547724, 373.2280425676212, 222.34825441232655, 414.43275045391346, 363.16096887811864, 263.55812876662463, 2296.9272853154753, 1088.172124910033, 1628.8002201343165, 497.7301456300714, 413.3823698841669, 239.3907236626176, 1847.474276084663, 2306.1593384145153, 1011.9526197106381, 1051.1029247248216, 679.6371000258135, 525.7809089348798, 809.2630806649112, 338.9392753722701, 3748.1533405143136, 975.3927021928846, 1244.403797218311, 537.7914272724906, 2311.7170525403035, 11609.772805252946, 5141.68103542172, 4538.362558231684, 7131.30561691681, 2039.8038922892881, 3912.383223966746, 3725.874869309288, 944.8323256802689, 1920.758891977114, 2847.5947129819206, 3581.8294902076873, 5351.938140214727, 1233.3452263836887, 2952.1487490050135, 2179.990691881155, 1601.6681650062158, 1818.234761129201, 2334.648991908968, 2077.0991975312136, 2245.804129821962, 3100.004978166423, 3201.17474790466, 2238.208247849854, 1893.4834337340305, 3349.695591288708, 4074.9824202294476, 4868.310982904491, 5009.861493283735, 3070.671639333097, 2913.6693831722755, 3126.08707717956, 3093.0676414099244, 4947.866674938379, 368.1585815316285, 235.3507025114459, 157.22839247808847, 148.4396315471312, 147.46350835818507, 126.95600513649252, 121.09703405241184, 89.84799992230955, 87.89492763134335, 269.54819172522485, 82.03571206384073, 82.03577082159221, 81.05942411494306, 145.52852627325925, 79.10625822746775, 80.08303561055614, 70.31733150041303, 70.31732531964481, 282.27075567497064, 67.38782451814056, 65.4347421571352, 65.43478970407165, 65.4351480912432, 63.4816868007408, 67.38712338807552, 154.3125205660792, 60.55205958271216, 66.4135386375241, 62.505747044438536, 58.599192598336685, 1101.8702109658352, 229.5312324431521, 498.4011316207246, 280.44459651248656, 165.06894555760414, 288.23157142734067, 1297.4674167266828, 554.2408886687726, 384.98463639195944, 202.20039151283265, 229.57902459229754, 316.81735712569053, 252.09577189530015, 219.93091929939885, 1003.0123385758162, 465.2038047996591, 164.202726483115, 695.350147187937, 256.24409806856056, 862.2581948304114, 461.13851426257315, 801.254932555301, 528.7677028482495, 2754.530166389272, 457.8924357499342, 1289.6369138696364, 1497.2885074471735, 900.8168000951325, 2136.7776675740224, 759.2595745980377, 623.6723465916006, 421.00371739759066, 4947.866674938379, 1298.9355415048224, 649.1728890444252, 3044.211196629302, 2999.1390609026253, 2049.5472728978743, 1145.896723774979, 2393.832807393489, 1317.0452773496615, 1170.4397160147757, 1615.718473416042, 1385.041497479589, 928.4280678131637, 2708.844447178401, 2177.4684546512817, 4868.310982904491, 2169.837928644268, 1574.6482714249219, 2574.672785425091, 3708.0983800022195, 3349.695591288708, 2204.4851867933967, 2152.702665197404, 3126.08707717956, 3169.269277116889, 5009.861493283735, 11609.772805252946, 2420.9728786288256, 4550.76690227792, 478.0634678324575, 294.3433873841359, 402.01575579024984, 222.23811974595748, 159.02263874979616, 202.48888117715708, 161.9845473935595, 140.25572389214608, 132.35354648648732, 116.54967840034266, 113.58665325710913, 153.1015574706822, 106.67212779429511, 94.81931573600335, 87.9050838932099, 225.2237488276245, 84.94172679076485, 83.95424746996999, 81.97869407215956, 83.95307244826571, 86.91760451977227, 79.01548959759042, 79.01540723819797, 78.02767384359835, 124.4597017530286, 137.28607990071058, 292.3449332633153, 77.03987426574834, 78.02760477447981, 76.05222502338263, 732.9896550412258, 449.45880911425564, 363.41012574022693, 281.5363338040587, 208.39241815025156, 130.39197516075376, 238.1090823724714, 154.1133747134797, 348.8334406480302, 295.19104719024716, 207.48086201785884, 445.4863974079744, 396.33667665654133, 493.6680554682685, 512.1661010258856, 282.4105522857229, 465.62135696883297, 257.94955582935927, 421.19904805453507, 263.61984874693314, 831.0525297650798, 818.8725700677242, 1083.8539797093872, 796.658675164818, 1807.0646716424014, 843.8014228644371, 1675.6041509556185, 524.0977348306583, 2523.583228949826, 757.6665721456128, 389.61898223553953, 1455.928035695861, 1281.8415871091124, 1019.1034852681214, 1864.9417148611526, 1277.0693087117538, 1831.4761966427388, 1325.2749297201863, 1029.5665993511573, 1297.1280343058338, 1470.240966859443, 3787.2536070817346, 3169.269277116889, 3317.1195432190743, 4550.76690227792, 1345.5552804143763, 721.6991918355575, 1038.8868709116123, 1041.8043560284848, 2049.30108473097, 5009.861493283735, 3708.0983800022195, 4074.9824202294476, 2380.8419462470506, 1510.6826389470586, 11609.772805252946, 2999.1390609026253, 1718.542817768759, 1820.8807779813505, 1263.4392050707554, 886.3418432159481, 2575.161162421673, 722.3857863129921, 544.9274348314221, 2775.7903134323155, 2209.66141441076, 372.29954367112697, 264.2765340424914, 531.4375396288186, 258.4985946865733, 591.2400857713062, 946.2895427443449, 1154.617737323608, 1246.2687475492871, 1073.6355367708811, 269.1127543172257, 475.5541161307922, 625.1279789650363, 309.6916249882786, 132.17041847611577, 157.25881877404248, 1554.2106159571708, 756.4578836243635, 61.74417331275264, 1108.5612008134701, 160.1736680184221, 3926.6659189334687, 334.77357832052587, 7896.613189360137, 3551.4368479494156, 1559.430729264364, 1957.7131247630803, 1436.8288322309852, 946.2958192760583, 835.7465579078245, 10103.395022319803, 2012.0853355033883, 645.7804716182329, 882.2989991658267, 1007.4945280426355, 931.0853201634204, 1262.8881539266981, 2200.0260300446844], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.7768, 0.7766, 0.7759, 0.7757, 0.7755, 0.7754, 0.7753, 0.7752, 0.775, 0.775, 0.7749, 0.7749, 0.7746, 0.7745, 0.7743, 0.7742, 0.7742, 0.7736, 0.7736, 0.7734, 0.7734, 0.7734, 0.7734, 0.7729, 0.7729, 0.7727, 0.7726, 0.7725, 0.7723, 0.7722, 0.7582, 0.7673, 0.7567, 0.7526, 0.7675, 0.7713, 0.7667, 0.7716, 0.7635, 0.7647, 0.7669, 0.7341, 0.7485, 0.7333, 0.7567, 0.7542, 0.7651, 0.721, 0.7113, 0.7298, 0.7268, 0.7364, 0.7437, 0.7307, 0.7562, 0.6693, 0.7201, 0.7083, 0.7376, 0.6725, 0.5772, 0.6112, 0.6171, 0.5895, 0.6587, 0.6153, 0.6154, 0.6991, 0.6475, 0.6152, 0.5924, 0.5599, 0.6771, 0.5924, 0.607, 0.637, 0.6139, 0.5694, 0.5773, 0.5468, 0.4701, 0.4502, 0.5433, 0.5859, 0.3522, 0.2599, 0.1593, 0.1396, 0.3652, 0.3771, 0.3166, 0.2671, -0.1922, 1.3395, 1.3384, 1.3368, 1.3365, 1.3364, 1.3357, 1.3353, 1.3332, 1.333, 1.3327, 1.3324, 1.3324, 1.3322, 1.3321, 1.332, 1.332, 1.3309, 1.3309, 1.3307, 1.3304, 1.3301, 1.33, 1.3299, 1.3298, 1.3297, 1.3292, 1.3292, 1.3291, 1.3289, 1.3287, 1.3154, 1.3229, 1.313, 1.3176, 1.3218, 1.3152, 1.2972, 1.2983, 1.3027, 1.3151, 1.3041, 1.2938, 1.2985, 1.3019, 1.2456, 1.2714, 1.306, 1.2452, 1.2861, 1.2178, 1.2488, 1.204, 1.2271, 1.049, 1.2266, 1.1222, 1.1003, 1.1547, 1.0559, 1.1568, 1.18, 1.224, 0.8303, 1.0439, 1.1498, 0.8411, 0.8168, 0.8961, 1.0193, 0.8116, 0.9434, 0.9682, 0.867, 0.9056, 1.0088, 0.6147, 0.6642, 0.3076, 0.5974, 0.7415, 0.4507, 0.2494, 0.2821, 0.5246, 0.5102, 0.2191, 0.1643, -0.2168, -0.9173, 0.373, -0.2443, 1.4877, 1.4867, 1.4862, 1.4859, 1.4845, 1.484, 1.4839, 1.4838, 1.4836, 1.4828, 1.4825, 1.4824, 1.4821, 1.4814, 1.4808, 1.4804, 1.4802, 1.4802, 1.4801, 1.4798, 1.4798, 1.4798, 1.4797, 1.4797, 1.4796, 1.4796, 1.4794, 1.4794, 1.4794, 1.4794, 1.4748, 1.477, 1.4689, 1.4713, 1.4734, 1.4777, 1.4705, 1.4739, 1.4612, 1.4645, 1.4701, 1.4521, 1.4508, 1.4435, 1.4311, 1.4511, 1.4303, 1.4536, 1.431, 1.4479, 1.3805, 1.3792, 1.362, 1.3795, 1.3067, 1.3627, 1.2909, 1.3928, 1.2092, 1.3267, 1.4018, 1.1854, 1.2031, 1.2356, 1.1144, 1.1575, 1.0624, 1.1307, 1.1622, 1.0946, 1.0257, 0.7187, 0.7753, 0.7498, 0.6161, 1.0262, 1.2531, 1.0738, 1.0653, 0.703, 0.1516, 0.2681, 0.0773, 0.4199, 0.7604, -1.0464, 0.1561, 0.6378, 2.9059, 2.9056, 2.9054, 2.9052, 2.9052, 2.9049, 2.9043, 2.9039, 2.9038, 2.9034, 2.903, 2.9025, 2.9024, 2.9015, 2.9012, 2.9012, 2.9012, 2.9012, 2.8988, 2.8983, 2.897, 2.8963, 2.896, 2.8956, 2.8942, 2.8936, 2.8934, 2.8929, 2.8927, 2.8923, 2.8919, 2.8905, 2.8922, 2.8878, 2.8882, 2.8922, 2.8902, 2.8599, 2.8796, 2.8804, 2.8549, 2.8359, 2.8171, 2.5843, 1.8242], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -7.9539, -8.2633, -8.4434, -8.4789, -7.8445, -8.5347, -8.5801, -8.5605, -8.6348, -8.6419, -8.4503, -8.628, -8.7002, -8.7384, -8.7156, -8.7782, -8.7863, -8.8717, -8.8896, -8.8988, -8.8719, -8.908, -8.908, -8.7635, -8.9847, -8.7638, -8.995, -9.0359, -9.0466, -9.0572, -6.1314, -7.207, -6.4107, -6.1696, -7.6496, -8.1043, -7.6899, -8.2029, -7.5884, -7.7193, -8.0376, -5.9053, -6.6381, -6.2499, -7.412, -7.6002, -8.1356, -6.1362, -5.9242, -6.7293, -6.6944, -7.1208, -7.3702, -6.952, -7.7968, -5.4804, -6.7758, -6.5441, -7.3538, -5.9606, -4.442, -5.2225, -5.3413, -4.917, -6.0995, -5.4916, -5.5403, -6.8287, -6.1709, -5.8094, -5.6027, -5.2337, -6.5842, -5.7961, -6.0847, -6.363, -6.2593, -6.0538, -6.1628, -6.1152, -5.8695, -5.8574, -6.1221, -6.2467, -5.91, -5.8062, -5.729, -5.72, -5.984, -6.0245, -6.0147, -6.0748, -6.0643, -7.1308, -7.5793, -7.9843, -8.0421, -8.0488, -8.1993, -8.2469, -8.5475, -8.5696, -7.4493, -8.6392, -8.6393, -8.6514, -8.0663, -8.676, -8.6638, -8.7948, -8.7948, -7.4052, -8.8379, -8.8676, -8.8677, -8.8678, -8.8983, -8.8386, -8.0106, -8.9461, -8.8538, -8.9146, -8.9794, -6.0586, -7.6199, -6.8544, -7.4248, -7.9506, -7.3998, -5.9135, -6.7629, -7.1229, -7.7545, -7.6384, -7.3266, -7.5505, -7.6836, -6.2224, -6.9649, -7.9717, -6.5892, -7.5466, -6.4014, -6.9963, -6.4886, -6.8811, -5.4088, -7.0256, -6.0945, -5.9671, -6.4208, -5.6558, -6.5897, -6.7632, -7.1121, -5.0418, -6.1656, -6.7532, -5.5167, -5.5559, -5.8573, -6.3155, -5.7865, -6.2522, -6.3455, -6.1242, -6.2397, -6.5365, -5.8598, -6.0287, -5.5807, -6.099, -6.2755, -6.0746, -5.9111, -5.9801, -6.156, -6.1941, -6.1122, -6.1532, -6.0764, -5.9365, -6.2138, -6.2, -6.7214, -7.2074, -6.8961, -7.4892, -7.8252, -7.5841, -7.8074, -7.9515, -8.0097, -8.1377, -8.1637, -7.8653, -8.2269, -8.3454, -8.4218, -7.4813, -8.4566, -8.4683, -8.4922, -8.4687, -8.434, -8.5294, -8.5294, -8.5421, -8.0752, -7.9772, -7.2214, -8.555, -8.5423, -8.568, -6.3069, -6.7937, -7.0144, -7.2672, -7.5659, -8.0306, -7.4355, -7.8672, -7.063, -7.2267, -7.5737, -6.8275, -6.9457, -6.7334, -6.709, -7.2843, -6.8051, -7.3724, -6.9046, -7.3564, -6.2756, -6.2917, -6.0285, -6.3188, -5.5726, -6.2782, -5.664, -6.7243, -5.3361, -6.4219, -7.0118, -5.9099, -6.0196, -6.2166, -5.7334, -6.069, -5.8035, -6.0587, -6.2797, -6.1162, -6.0599, -5.4207, -5.5422, -5.5222, -5.3397, -6.1481, -6.5441, -6.3591, -6.3647, -6.0505, -5.708, -5.8924, -5.9889, -6.1837, -6.298, -6.0656, -6.2166, -6.2918, -3.9658, -4.3316, -4.6863, -3.62, -4.8911, -5.1732, -3.5458, -3.7743, -5.5552, -5.8984, -5.2002, -5.9214, -5.0942, -4.6247, -4.426, -4.3497, -4.4988, -5.8825, -5.3155, -5.0426, -5.7462, -6.5984, -6.4249, -4.1345, -4.856, -7.3622, -4.4745, -6.4096, -3.2106, -5.6731, -2.5127, -3.3132, -4.1345, -3.9114, -4.2204, -4.634, -4.7603, -2.2983, -3.8923, -5.0279, -4.7413, -4.6277, -4.7254, -4.6534, -4.8583]}, "token.table": {"Topic": [3, 1, 2, 1, 2, 3, 3, 1, 4, 1, 1, 3, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 2, 2, 1, 2, 3, 1, 2, 3, 1, 4, 1, 1, 1, 3, 1, 1, 2, 3, 4, 1, 2, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 2, 1, 4, 1, 2, 3, 1, 2, 4, 2, 3, 2, 2, 3, 1, 2, 3, 1, 3, 3, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 3, 3, 2, 3, 4, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 3, 4, 1, 2, 3, 1, 2, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 3, 1, 3, 4, 1, 2, 3, 4, 1, 2, 3, 2, 4, 3, 1, 2, 3, 1, 1, 2, 3, 4, 1, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 3, 1, 2, 3, 1, 2, 3, 1, 3, 1, 4, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 4, 2, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 1, 4, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 4, 1, 2, 3, 3, 4, 1, 1, 2, 3, 4, 2, 2, 3, 3, 3, 4, 1, 1, 2, 3, 1, 1, 2, 3, 4, 1, 2, 3, 2, 2, 1, 2, 4, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 3, 3, 4, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 4, 2, 1, 2, 3, 1, 1, 1, 3, 2, 3, 2, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 2, 3, 2, 3, 1, 2, 3, 1, 1, 2, 1, 2, 3, 2, 3, 1, 3, 1, 2, 3, 4, 1, 2, 3, 1, 3, 4, 3, 4, 1, 2, 3, 1, 2, 3, 4, 4, 4, 2, 2, 1, 2, 3, 4, 1, 3, 1, 2, 3, 4, 2, 4, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 4, 1, 2, 3, 4, 3, 3, 2, 1, 2, 3, 1, 4, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 3, 4, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 1, 4, 1, 3, 4, 1, 1, 2, 1, 3, 1, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 4, 1, 3, 1, 2, 3, 1, 3, 4, 1, 3, 4, 1, 1, 1, 4, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 4, 1, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 1, 4, 1, 2, 3, 1, 2, 3, 3, 1, 3, 4, 1, 2, 3, 4, 2, 3, 1, 4, 2, 3, 1, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 3, 1, 1, 3, 1, 2, 3, 1, 3, 3, 1, 2, 3, 1, 1, 1, 4, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 1, 1, 1, 2, 3, 2], "Freq": [0.9868293671594227, 0.030450164300494433, 0.9622251918956242, 0.18717525883973973, 0.15741919204983237, 0.6546334693779615, 0.9939219671913061, 0.08484721892740639, 0.9150611459006359, 0.998646493963147, 0.989438844849342, 0.004497449294769736, 0.9765761127460071, 0.005900762010549892, 0.01475190502637473, 0.19204727051271706, 0.7464067829378946, 0.057723092649379984, 0.003993421504045156, 0.8469753372461072, 0.0918473255325324, 0.06104822235994668, 0.9924697919136263, 0.9933553740137627, 0.11625182207088842, 0.0027679005254973438, 0.8801923671081553, 0.8099718658151646, 0.1357805824766828, 0.054397898973308254, 0.004657073866088901, 0.9947509777965893, 0.9938735244810777, 0.9939669557612513, 0.2787580717759008, 0.7206916002011093, 0.9939690531184131, 0.6682295566013025, 0.19185430002061019, 0.13934319464824282, 0.00034320983903508086, 0.05073287579299378, 0.944411995531115, 0.6517010099894325, 0.3465986590003349, 0.0014926729500445086, 0.9857887567211258, 0.01376799939554645, 0.16345073101258129, 0.640629283043341, 0.19614087721509754, 0.8496456487386592, 0.13837589922403476, 0.011898564582958402, 0.03182817596679383, 0.9593921612847854, 0.009093764561941095, 0.3694616106704213, 0.3354819296890541, 0.29502992852075977, 0.13472569993524217, 0.7429160025000496, 0.1224079216554486, 0.9873748634858592, 0.9860155409665979, 0.00756598951209879, 0.9911446260849415, 0.10933564887744518, 0.6719586753926319, 0.21943057309431707, 0.04418949809762411, 0.9532306018201773, 0.9996523734034906, 0.9942552320293282, 0.9981767311518586, 0.9942448873974828, 0.9937732780693838, 0.9861644413025507, 0.22713917918374563, 0.12012168129909624, 0.6524791325110001, 0.00803472920081673, 0.9882716917004578, 0.9897038504131407, 0.9933560958169406, 0.9948351919852659, 0.9936991245211567, 0.009540205323756178, 0.0801377247195519, 0.9082275468215881, 0.07363355409692021, 0.8883532010402632, 0.03562913907915494, 0.816295043750731, 0.10448094207762337, 0.07924358343892292, 0.9935692253767392, 0.9880616045030177, 0.9977754672673917, 0.006775273230800335, 0.9756393452352482, 0.01693818307700084, 0.004938542769294588, 0.992647096628212, 0.013877730257624976, 0.9749105505981546, 0.010408297693218731, 0.20193929390001641, 0.3082729533442438, 0.4897027877075398, 0.3137922901432651, 0.622014302946709, 0.06374872955573237, 0.014459164911999335, 0.9832232140159548, 0.9551891407388029, 0.019591390762646366, 0.02525112587185532, 0.0047986390717871365, 0.983721009716363, 0.009597278143574273, 0.12015289195889722, 0.8256660267944732, 0.05391475921232568, 0.007216561425864032, 0.005412421069398024, 0.9868647749869064, 0.9642799717225833, 0.024725127480066236, 0.009509664415410091, 0.019894302811674994, 0.0003959065236154228, 0.025041087618675493, 0.9546296050676882, 0.9765323996813289, 0.005904957821202291, 0.016976753735956588, 0.5136750631759889, 0.0033097620049999284, 0.4825633003289896, 0.9868302406891785, 0.006959771251578278, 0.011135634002525246, 0.9820237235976951, 0.047602910169578655, 0.002266805246170412, 0.001133402623085206, 0.9497913981454027, 0.14079000546543094, 0.6469120251129545, 0.2122680082401882, 0.987374156283059, 0.9989536186271956, 0.9952837415951116, 0.003542697852665548, 0.9884127008936878, 0.003542697852665548, 0.9938965473478649, 0.004086400555223406, 0.005618800763432183, 0.008683601179849737, 0.9817577333924232, 0.02580027873268297, 0.9718104989310584, 0.989930616804237, 0.008147577093038988, 0.0018150958071975797, 0.9737989005615014, 0.024503793397167326, 0.1946284239389975, 0.8033268812780933, 0.0023262361426971815, 0.004503894084965443, 0.000643413440709349, 0.005790720966384141, 0.9895698718109789, 0.9886475572546434, 0.010390025008498083, 0.9610773132860727, 0.02597506252124521, 0.007933492834741535, 0.9559858865863549, 0.03173397133896614, 0.03532350353770606, 0.9613039177047149, 0.015486689572350628, 0.9843903044534146, 0.33003002476986537, 0.48360104300729717, 0.18642635627380538, 0.13607032416932643, 0.04356637572088083, 0.8200027430204145, 0.21765560019722707, 0.030424976371655397, 0.7512628781001064, 0.986499643235154, 0.0075884587941165696, 0.0037942293970582848, 0.0025982625270888084, 0.0008660875090296028, 0.0017321750180592057, 0.9951345478750137, 0.0018012887990150122, 0.9979139946543167, 0.8289060124489195, 0.12367981256816143, 0.04746177908936218, 0.8408292782288304, 0.08669761788611509, 0.05366995392949981, 0.018348702198119595, 0.13002917008885637, 0.7243235649245019, 0.14573739197878532, 0.9863570847318641, 0.00772603460625481, 0.005150689737503207, 0.7335472091225529, 0.08258051254853717, 0.1838706724713523, 0.8480841704720966, 0.03476618338581265, 0.1172919722309235, 0.8233502190856696, 0.09664726753860009, 0.07974719889797056, 0.004271898784351299, 0.6886300840374294, 0.0017087595137405195, 0.305867952959553, 0.987947473051697, 0.9864760919425815, 0.9898182107265969, 0.5806563103719481, 0.4101492065235968, 0.008933174005722279, 0.22463731593795863, 0.7515990195757533, 0.023399720410204022, 0.6289012274647763, 0.3253268302806091, 0.045424198524922806, 0.18951283809125283, 0.142692019268708, 0.6294798975031025, 0.03864575521860842, 0.14370790791884308, 0.5918365117307343, 0.2637423553684568, 0.0010002870620801144, 0.03389155336118579, 0.040167766946590557, 0.8962432999958019, 0.03012582520994292, 0.9973287720966101, 0.05688542197174222, 0.06162587380272074, 0.8805389276042598, 0.7890239890307422, 0.03797680581405045, 0.1729061629416179, 0.2283429458346177, 0.7173415185181857, 0.04308357468577692, 0.011847983038588654, 0.0063589438595291195, 0.9919952420865426, 0.05494777079539099, 0.2067409876176586, 0.7383606700630665, 0.9939425580352034, 0.8264966216029661, 0.08736128185589546, 0.08623946764265765, 0.0012211912287125406, 0.10258006321185341, 0.8963543618750047, 0.4631682333195881, 0.4751506950771098, 0.06175576444261175, 0.9024237303479458, 0.07783708725373117, 0.02027007480565916, 0.9939365975010137, 0.9748843428250795, 0.0241906784820119, 0.2560308625718541, 0.3171291365946829, 0.4265241415117478, 0.8951074556463141, 0.04535566839340489, 0.05949596501017229, 0.9335868359730025, 0.015176748378567155, 0.051167323104883554, 0.9939189382491516, 0.0775191147336872, 0.14620693791543535, 0.776172401953754, 0.99951629014266, 0.1136550520337758, 0.8837260168340526, 0.002319490857832159, 0.006308430309485339, 0.99252636869236, 0.993905890022116, 0.04239027453865224, 0.9564787400448624, 0.0007707322643391316, 0.0007707322643391316, 0.9908828933893146, 0.007284059685608549, 0.9906321172427627, 0.9871492500805957, 0.9894428231790082, 0.998298058104362, 0.9938952673146737, 0.5218382929667237, 0.0009043991212594865, 0.4775227360250089, 0.9939109063021584, 0.0002532731377414803, 0.00012663656887074016, 0.013930022575781416, 0.9857390520898414, 0.5069635142738511, 0.149526935444483, 0.3431559164554568, 0.9968530367353967, 0.9985098726806272, 0.00597418712083998, 0.00896128068125997, 0.9857408749385967, 0.930566109319616, 0.05303744664515946, 0.0160719535288362, 0.1840150713801561, 0.09866339997404115, 0.7180502998110773, 0.00229202080997115, 0.0002546689788856834, 0.010950766092084384, 0.9865876242031374, 0.5375580995523559, 0.3555647956916806, 0.10681322574215708, 0.006531612195986265, 0.9928050537899124, 0.002262790112273073, 0.9974378814899705, 0.828887772279338, 0.14260799024503526, 0.028453850785232687, 0.9954871509819572, 0.7916986064768492, 0.1349182664580907, 0.07302506831395009, 0.594358392315132, 0.16196388890503183, 0.24368203285257062, 0.14782227976988693, 0.0013198417836597048, 0.8499781086768499, 0.003382720570089368, 0.996211207891319, 0.9954872384835127, 0.8980337787095188, 0.012977366744357208, 0.08911125164458616, 0.9938814432724306, 0.9938854366576941, 0.9939585439916889, 0.9889132605805792, 0.9905464336120553, 0.007419823472749478, 0.9970383142119859, 0.9939784192797878, 0.37750259859982316, 0.5079292871671587, 0.1143529769481216, 0.04970558686691256, 0.8514086008493732, 0.09780776770586021, 0.41388319912427357, 0.3795994610730613, 0.20652854247718241, 0.991908791297559, 0.025886238857924654, 0.9088946087893545, 0.06615372152580744, 0.006841233667623028, 0.9919788818053392, 0.9802587543537268, 0.017426822299621808, 0.07186508172508117, 0.01283305030805021, 0.916279791994785, 0.9938861852358898, 0.021394600126420533, 0.977020072439871, 0.8480764306713022, 0.13214451918537168, 0.019936697285220486, 0.9742810017630327, 0.02472794420718357, 0.004440029105302533, 0.9901264904824649, 0.09506017630718427, 0.004813173483908065, 0.896453561377877, 0.0024065867419540326, 0.01571316215428556, 0.020202637055510005, 0.9629923663126436, 0.009253654633700679, 0.0026439013239144796, 0.9874971444820582, 0.9913591895317565, 0.9996143212479874, 0.14415917771565864, 0.5486939627591589, 0.3067351666813354, 0.9726999039366242, 0.01906122139543266, 0.0005776127695585654, 0.0075089660042613506, 0.998071189952975, 0.9965094137416539, 0.9869302782926167, 0.9909408677017058, 0.05459086721478699, 0.0039702448883481446, 0.009925612220870362, 0.932014987539727, 0.0024874646965870315, 0.9974733433313997, 0.20598970493506888, 0.08951889046243647, 0.6603221382498002, 0.045240729588543166, 0.0018816886753962616, 0.9972949979600186, 0.3825991471208937, 0.1544654941792428, 0.4628683953781413, 0.007669183619368829, 0.989324686898579, 0.16296124546899232, 0.7854197732439958, 0.0514262946766902, 0.0010567520004104286, 0.012681024004925143, 0.9859496163829299, 0.21864515256784658, 0.4418265116205033, 0.13880791843518892, 0.2009539472770872, 0.9865021292459369, 0.998833378296052, 0.9924121927910841, 0.8853792302421346, 0.025492646717935216, 0.08873402030665911, 0.006458037087944168, 0.9913086929994298, 0.9626314964638744, 0.03484638901226694, 0.14490594391197517, 0.09977130564431078, 0.03088159460419143, 0.7245297195598759, 0.19249931359207548, 0.12011099232486046, 0.6874209471449603, 0.05154402743945973, 0.0064430034299324665, 0.9428261685801176, 0.0037159145523857867, 0.9958651000393909, 0.0021495954884346914, 0.932924441980656, 0.06448786465304074, 0.11084950753031794, 0.09976455677728614, 0.7898027411535153, 0.992187209582648, 0.09990932672491828, 0.8292474118168218, 0.0699365287074428, 0.0007766504206358901, 0.9987724409377546, 0.003209580604396733, 0.0016047903021983664, 0.9949699873629871, 0.9939181494956172, 0.006871505027971614, 0.9894967240279123, 0.012977458988996287, 0.9862868831637178, 0.054669764250142826, 0.9430534333149637, 0.08674183301294136, 0.9107892466358843, 0.010655818236547453, 0.007103878824364969, 0.9838872171745482, 0.0035409441747356212, 0.03186849757262059, 0.9631368155280889, 0.003793341073342201, 0.03793341073342201, 0.9597152915555768, 0.9514829211871971, 0.00741415263262751, 0.0407778394794513, 0.9859923639937266, 0.002679327076069909, 0.010717308304279636, 0.9764327201535637, 0.016072966586889937, 0.006027362470083727, 0.05460601135314545, 0.9425472394434237, 0.006674693963418077, 0.0044497959756120515, 0.9878547065858754, 0.010914205875866084, 0.9850070802969141, 0.002728551468966521, 0.004799017322767746, 0.003199344881845164, 0.9917969133720007, 0.9939460141962113, 0.9939380510979043, 0.013466452600883663, 0.986257338102813, 0.03489054273058624, 0.9653050155462193, 0.42990266611587563, 0.11418527113633926, 0.4557651420570123, 0.8481229538945328, 0.06656154828033042, 0.08534908206913337, 0.957864130688678, 0.008828240835840351, 0.03384158987072134, 0.015554954880752002, 0.9835517624598573, 0.9973630487998667, 0.7097660391223362, 0.018646395943044427, 0.013232926153128302, 0.2586435566293259, 0.8756955425408184, 0.08069727056707898, 0.043212086819790684, 0.14224384336170756, 0.8310728255670137, 0.025024379850670776, 0.9933499334236023, 0.9421846174714019, 0.05741277320826823, 0.0010252280930047898, 0.01679902320459513, 0.9827428574688151, 0.30714031964512756, 0.6063968235988294, 0.08672197260568307, 0.9475760903821597, 0.03710388305713276, 0.01427072425274337, 0.9858360273500143, 0.012531813906991707, 0.45408553642418015, 0.003636320612005447, 0.20317941419580435, 0.3390868970695079, 0.950637392761608, 0.03853935376060573, 0.010870074137606744, 0.008102610561272398, 0.02835913696445339, 0.9561080462301429, 0.010128263201590496, 0.002990990125066377, 0.9082640013118232, 0.0877357103352804, 0.9850140445014056, 0.0064803555559303, 0.04002245649810826, 0.20407490194579955, 0.7556715301177471, 0.0038546695991161593, 0.32225037848611093, 0.6737962459255046, 0.08888591293838626, 0.8926415086578364, 0.015129517095895531, 0.003782379273973883, 0.8165233523828914, 0.11506431675678717, 0.0683645731358317, 0.04149446139873852, 0.891038960562385, 0.06551757062958713, 0.002183919020986238, 0.08861099896580009, 0.8711334827905417, 0.03993735164655779, 0.5551161427537035, 0.4352667997993474, 0.009290646740647758, 0.004227036038461923, 0.9954669870577828, 0.5269606761662388, 0.21058466414976587, 0.26248230649947124, 0.9576203224583151, 0.02417294017855941, 0.016735112431310362, 0.9871482211555975, 0.015485138432479135, 0.009291083059487481, 0.9740152074029376, 0.8444708977642503, 0.12641779906650455, 0.028784360402834882, 0.00019448892164077624, 0.9814080986145685, 0.018174224048417936, 0.9938913921363848, 0.9994659552827567, 0.019261983924475853, 0.9796094681590577, 0.9685967650449985, 0.030326084674084392, 0.8023261643729908, 0.15769987953674994, 0.039798666281194, 0.95407649188054, 0.045432213899073336, 0.26526263979235487, 0.10542489530208976, 0.6291485687382776, 0.9784913545768308, 0.021247240842239754, 0.9939422186017555, 0.9938526568691504, 0.0031550877995846043, 0.9874181781643924, 0.003307933595190594, 0.008269833987976484, 0.9938664101600544, 0.9944288596961998, 0.9886337201663166, 0.6604418310387791, 0.22633484840826007, 0.11333025503032303, 0.9938938596354758, 0.9939440619443463, 0.026340831109302992, 0.9736167574174446, 0.009054724971318696, 0.2920148803250279, 0.6987229436200927, 0.22223774290204718, 0.5885958265958354, 0.18923627356132963, 0.024076991881976324, 0.9711053392397118, 0.004012831980329388, 0.9423676543359981, 0.00487151573177713, 0.0525041139980424, 0.012486446896939224, 0.9864293048581987, 0.16435493685461916, 0.002213534503092514, 0.8333957404143315, 0.37814276796843993, 0.5998544817372152, 0.020614945127087586, 0.0014147511361726774, 0.921856689622563, 0.06456171993912324, 0.013759055068993477, 0.9844781802430722, 0.00482587343256408, 0.00965174686512816, 0.9939702314400354, 0.8665964837945153, 0.12674285750020647, 0.006243490517251551, 0.9939199341004537, 0.02886831398966303, 0.9580671705319419, 0.012629887370477577, 0.9968567928205043, 0.7187986227575135, 0.055604587071202694, 0.22554220149105814, 0.9897747294498782, 0.3775187868093271, 0.20480064569632878, 0.417731789129529, 0.9939438662240392, 0.9939791999293817, 0.5987583249733899, 0.25347004685698576, 0.14774975945617666, 0.9905618386269832], "Term": ["abolish", "acid", "acid", "act", "act", "act", "administration", "al", "al", "aladdin", "alice", "alice", "angel", "angel", "angel", "animal", "animal", "animal", "animal", "answer", "answer", "answer", "antenn\u00e6", "appendage", "army", "army", "army", "ask", "ask", "ask", "au", "au", "aunt", "aye", "battle", "battle", "be", "bear", "bear", "bear", "bear", "beetle", "beetle", "bird", "bird", "bird", "bless", "bless", "body", "body", "body", "boy", "boy", "boy", "buffaloes", "buffaloes", "buffaloes", "call", "call", "call", "camp", "camp", "camp", "camper", "carbon", "casa", "casa", "case", "case", "case", "caterpillar", "caterpillar", "ce", "cereal", "charter", "chrysalis", "cicada", "cincinnati", "city", "city", "city", "clause", "clause", "clergy", "cockroach", "colonial", "colonist", "colony", "colony", "colony", "colour", "colour", "colour", "come", "come", "come", "commons", "confederate", "congress", "constitution", "constitution", "constitution", "consul", "consul", "cooking", "cooking", "cooking", "country", "country", "country", "cover", "cover", "cover", "cromwell", "cromwell", "cry", "cry", "cry", "cum", "cum", "cum", "current", "current", "current", "dan", "dan", "dan", "dance", "dance", "dance", "de", "de", "de", "de", "dear", "dear", "dear", "death", "death", "death", "democratic", "des", "des", "des", "di", "di", "di", "di", "different", "different", "different", "disc", "dit", "doctrine", "donald", "donald", "donald", "doth", "du", "du", "du", "du", "duke", "duke", "echo", "echo", "eclipse", "eclipse", "eclipse", "egg", "egg", "egg", "el", "el", "el", "el", "elector", "electric", "electric", "electric", "electricity", "electricity", "electricity", "empire", "empire", "en", "en", "end", "end", "end", "england", "england", "england", "english", "english", "english", "ere", "ere", "ere", "est", "est", "est", "est", "et", "et", "eye", "eye", "eye", "face", "face", "face", "face", "fact", "fact", "fact", "fairy", "fairy", "fairy", "fall", "fall", "fall", "father", "father", "father", "feel", "feel", "feel", "fig", "fig", "fig", "fig", "fil", "fjord", "flavour", "fly", "fly", "fly", "food", "food", "food", "foot", "foot", "foot", "force", "force", "force", "force", "form", "form", "form", "form", "france", "france", "france", "france", "frederic", "french", "french", "french", "friend", "friend", "friend", "fruit", "fruit", "fruit", "fruit", "fut", "fut", "general", "general", "general", "glitter", "go", "go", "go", "government", "government", "government", "ground", "ground", "ground", "happy", "happy", "happy", "hark", "hath", "hath", "have", "have", "have", "hear", "hear", "hear", "heart", "heart", "heart", "hiawatha", "history", "history", "history", "il", "inch", "inch", "inch", "ind", "ind", "ing", "insect", "insect", "insect", "insect", "insects", "institution", "institution", "insurrection", "jealousy", "jour", "kate", "king", "king", "king", "kitty", "la", "la", "la", "la", "land", "land", "land", "larva", "larv\u00e6", "las", "las", "las", "laugh", "laugh", "laugh", "law", "law", "law", "le", "le", "le", "le", "leave", "leave", "leave", "legislature", "legislature", "les", "les", "let", "let", "let", "lice", "lie", "lie", "lie", "life", "life", "life", "lincoln", "lincoln", "lincoln", "los", "los", "louse", "love", "love", "love", "ly", "maggie", "magician", "magistrate", "magnet", "magnet", "magnetic", "magpie", "make", "make", "make", "material", "material", "material", "mean", "mean", "mean", "membrane", "method", "method", "method", "military", "military", "mineral", "mineral", "minister", "minister", "minister", "morn", "moth", "moth", "mother", "mother", "mother", "motor", "motor", "napoleon", "napoleon", "nation", "nation", "nation", "nation", "national", "national", "national", "ne", "ne", "ne", "nominate", "nous", "number", "number", "number", "oh", "oh", "oh", "oh", "om", "op", "oval", "oxygen", "par", "par", "par", "par", "parliament", "parliament", "party", "party", "party", "party", "pas", "pas", "people", "people", "people", "pitt", "pitt", "plant", "plant", "plant", "plus", "plus", "plus", "point", "point", "point", "point", "poland", "political", "pollen", "poor", "poor", "poor", "por", "por", "porcelain", "porcelain", "pour", "pour", "pour", "pour", "power", "power", "power", "president", "president", "president", "pret", "pret", "product", "product", "product", "public", "public", "public", "pupa", "pupil", "pupil", "pupil", "que", "que", "qui", "qui", "qui", "quoth", "recipe", "recipe", "reform", "reform", "reign", "reign", "reindeer", "reindeer", "religion", "religion", "religion", "religious", "religious", "religious", "revolution", "revolution", "revolution", "ride", "ride", "ride", "roar", "roar", "roar", "robin", "robin", "robin", "roman", "roman", "romans", "romans", "romans", "rome", "rome", "rome", "sa", "sa", "sa", "sam", "script", "se", "se", "senate", "senate", "send", "send", "send", "shall", "shall", "shall", "shout", "shout", "shout", "si", "si", "sigh", "sing", "sing", "sing", "sing", "sit", "sit", "sit", "size", "size", "size", "skunk", "sky", "sky", "sky", "slavery", "slavery", "small", "small", "small", "smile", "smile", "smile", "softly", "softly", "son", "son", "son", "son", "song", "song", "song", "spain", "spain", "spain", "spain", "specie", "specie", "specie", "starch", "starch", "state", "state", "state", "states", "states", "states", "stem", "stem", "stem", "stem", "story", "story", "story", "substance", "substance", "substance", "substance", "sugar", "sugar", "sugar", "sun", "sun", "sun", "sur", "sur", "take", "take", "take", "tale", "tale", "tale", "tarquin", "te", "te", "te", "tell", "tell", "tell", "tell", "temperature", "temperature", "ter", "terre", "territory", "territory", "thee", "thee", "think", "think", "think", "thou", "thou", "thousand", "thousand", "thousand", "thy", "thy", "tion", "tis", "tis", "tom", "tom", "tom", "tommy", "treaty", "turks", "turn", "turn", "turn", "twas", "ty", "un", "un", "united", "united", "united", "use", "use", "use", "vegetable", "vegetable", "vegetable", "voice", "voice", "voice", "vous", "vous", "war", "war", "war", "water", "water", "water", "water", "wave", "wave", "wave", "whisper", "whisper", "whisper", "willie", "wind", "wind", "wind", "winkle", "wire", "wire", "wire", "wireless", "word", "word", "word", "yap", "year", "year", "year", "yon", "yonder", "young", "young", "young", "zinc"]}, "R": 30, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [4, 1, 3, 2]};

function LDAvis_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

if(typeof(LDAvis) !== "undefined"){
   // already loaded: just create the visualization
   !function(LDAvis){
       new LDAvis("#" + "ldavis_el104961400367656731765778801472", ldavis_el104961400367656731765778801472_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
        new LDAvis("#" + "ldavis_el104961400367656731765778801472", ldavis_el104961400367656731765778801472_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js", function(){
         LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el104961400367656731765778801472", ldavis_el104961400367656731765778801472_data);
            })
         });
}
</script>



## Clustering


```python
lda_output = LdaTR
```


```python
# Construct the k-means clusters
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA as sklearnPCA

from sklearn.decomposition import FactorAnalysis


from scipy.spatial.distance import cdist
from yellowbrick.cluster import KElbowVisualizer
import hdbscan
#import plotly_express as px
%matplotlib inline
#.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}


sklearn_pca = sklearnPCA(n_components=2)
#Lda_PCA = FactorAnalysis(n_components = 2).fit_transform(Temp_LDA)

Lda_PCA = sklearn_pca.fit_transform(Temp_LDA)
```


```python
# diplay clustering results
#https://github.com/scikit-learn-contrib/hdbscan/blob/master/notebooks/Comparing%20Clustering%20Algorithms.ipynb
def plot_clusters(data,n_clusters, algorithm, args, kwds, n, show_labels =False):
    '''
    Function to cluster data and plot them
    
    n = real labels from the dataset to display
    show_label = flag to display original data labels if any
    algorithm = takes any algorithm supported by sklearn
    
    '''
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('bright', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    fig, ax = plt.subplots(figsize=(20,15))
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    #px.scatter(data.T, data.T[0], data.T[1])

    z = data.T[0]
    y = data.T[1]
    # label the points
    if show_labels:
        for i, txt in enumerate(n):
            ax.annotate(txt, (z[i], y[i]))
        
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title(str(n_clusters) +' Clusters found by {}'.format(str(algorithm.__name__)), fontsize=14)
    #plt.text(-0.3, 0.4, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
    return(labels)


#https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
from sklearn.metrics import silhouette_samples, silhouette_score
#plot_clusters(data, hdbscan.HDBSCAN, (), {'min_cluster_size':15})

AlgosDict ={"Kmean":cluster.KMeans,
            #"MiniBatch":cluster.MiniBatchKMeans,
            #"Spectral":cluster.SpectralClustering,
            #"Agglomerative":cluster.AgglomerativeClustering,
            #"MeanShift":cluster.MeanShift,
            #"AffinityProgation":cluster.AffinityPropagation,
            #"hdbscan":hdbscan.HDBSCAN            
           }

def run_Clustering_Algo(n_clusters = [4], Algos_Dict = AlgosDict, data =Lda_PCA, showlabels =True, flag = True, ClusterLabels = data.Dominant_Topic):
    """
    Function to run multiple clustering algos and plot results
    Setup for clustering Algos that require number of clusters to be specified
    Stores clustering labels
    flag =True (run algos that don't required cluster size)
    """
    temp = pd.DataFrame()
    temp["PCA1"] = data.T[0]
    temp["PCA2"] = data.T[1]
    Silhoute_scores = pd.DataFrame()
    for i in n_clusters:
        for key, value in Algos_Dict.items():
            if (key!="AffinityProgation") and (key!="MeanShift") and (key!="hdbscan"):
                labels = plot_clusters(data,i, value, (), {'n_clusters':i}, 
                                      n = ClusterLabels, show_labels=showlabels)
                temp[key+"_"+str(i)] = labels
                print("# of Clusters for " + key +" " + str(len(np.unique(labels)))) 
                silhouette_avg = silhouette_score(data, labels)
                print("#Avg silhouette "+ str(silhouette_avg))
                Silhoute_scores[key+"_"+str(i)] = silhouette_avg
                print("---------------------------------------")
                # Compute the silhouette scores for each sample
                sample_silhouette_values = silhouette_samples(data, labels)
            else:
                if flag:
                    labels = plot_clusters(data,i, value, (), {}, 
                                           n = ClusterLabels, show_labels=showlabels)
                    temp[key] = labels
                    print("# of Clusters for " + key +" " + str(len(np.unique(labels)))) 
                    
        
    return(temp, sample_silhouette_values)
    
#ClusterLabels, Silhoute_scores = run_Clustering_Algo()
```


```python
ClusterLabels, Silhoute_scores = run_Clustering_Algo(ClusterLabels = data.index)
```

    # of Clusters for Kmean 4
    #Avg silhouette 0.8271313582982562
    ---------------------------------------



![png](output_24_1.png)


### Conclusion

Books in the science category similar the `The Flag of My Country. Shikéyah Bidah Na'at'a'í; Navajo New World Readers 2`


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BookTitle</th>
      <th>Category</th>
      <th>url</th>
      <th>Body</th>
      <th>Corpus</th>
      <th>Topic0</th>
      <th>Topic1</th>
      <th>Topic2</th>
      <th>Topic3</th>
      <th>Dominant_Topic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Book0</th>
      <td>A Primary Reader: Old-time Stories, Fairy Tale...</td>
      <td>Misc.</td>
      <td>http://www.gutenberg.org/cache/epub/7841/pg784...</td>
      <td>['CONTENTS.', 'THE UGLY DUCKLING', 'THE LITTLE...</td>
      <td>['ugly', 'duckling', 'little', 'pine', 'tree',...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>1.000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Book1</th>
      <td>The Bird-Woman of the Lewis and Clark Expedition</td>
      <td>Misc.</td>
      <td>http://www.gutenberg.org/cache/epub/5742/pg574...</td>
      <td>['CONTENTS', 'THE BIRD-WOMAN', 'WHO THE WHITE ...</td>
      <td>['bird', 'woman', 'white', 'men', 'sacajawea',...</td>
      <td>0.200</td>
      <td>0.014</td>
      <td>0.00</td>
      <td>0.786</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Book2</th>
      <td>Dr. Scudder's Tales for Little Readers, About ...</td>
      <td>Misc.</td>
      <td>http://www.gutenberg.org/cache/epub/13539/pg13...</td>
      <td>['CONTENTS.', 'CHAPTER I.', 'General Remarks',...</td>
      <td>['chapter', 'general', 'remarks', 'chapter', '...</td>
      <td>0.119</td>
      <td>0.000</td>
      <td>0.44</td>
      <td>0.441</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Book3</th>
      <td>The Louisa Alcott Reader: a Supplementary Read...</td>
      <td>Misc.</td>
      <td>http://www.gutenberg.org/cache/epub/7425/pg742...</td>
      <td>['CONTENTS.', 'I. A CHRISTMAS DREAM', 'II. THE...</td>
      <td>['christmas', 'dream', 'ii', 'candy', 'country...</td>
      <td>0.008</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.992</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Book4</th>
      <td>Boy Blue and his friends, School ed.</td>
      <td>Misc.</td>
      <td>http://www.gutenberg.org/cache/epub/16046/pg16...</td>
      <td>['~CONTENTS~', 'LITTLE BOY BLUE', 'SNOWBALL', ...</td>
      <td>['little', 'boy', 'blue', 'snowball', 'fire', ...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>1.000</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# filter to topic 0
cols = ['BookTitle','Category', 'Topic0','Topic1','Topic2','Topic3','Dominant_Topic']

Final = data[cols]
Final = Final[Final.Dominant_Topic.isin([data.loc['Book6'].Dominant_Topic])] #books similar to Book 6
```


```python
Final
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BookTitle</th>
      <th>Category</th>
      <th>Topic0</th>
      <th>Topic1</th>
      <th>Topic2</th>
      <th>Topic3</th>
      <th>Dominant_Topic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Book6</th>
      <td>The Flag of My Country. Shikéyah Bidah Na'at'a...</td>
      <td>Misc.</td>
      <td>0.531</td>
      <td>0.010</td>
      <td>0.266</td>
      <td>0.194</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book34</th>
      <td>Gems of Poetry for Boys and Girls</td>
      <td>Poetry Readers</td>
      <td>0.573</td>
      <td>0.000</td>
      <td>0.185</td>
      <td>0.242</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book40</th>
      <td>The Flag of My Country. Shikéyah Bidah Na'at'a...</td>
      <td>Non-English Readers</td>
      <td>0.531</td>
      <td>0.010</td>
      <td>0.266</td>
      <td>0.194</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book42</th>
      <td>A Book of Natural HistoryYoung Folks' Library ...</td>
      <td>Science and Nature</td>
      <td>0.803</td>
      <td>0.000</td>
      <td>0.075</td>
      <td>0.122</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book44</th>
      <td>Wildflowers of the Farm</td>
      <td>Science and Nature</td>
      <td>0.799</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.201</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book46</th>
      <td>Book about Animals</td>
      <td>Science and Nature</td>
      <td>0.879</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.119</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book47</th>
      <td>Bird Day; How to prepare for it</td>
      <td>Science and Nature</td>
      <td>0.548</td>
      <td>0.000</td>
      <td>0.104</td>
      <td>0.347</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book48</th>
      <td>Child's Book of Water Birds</td>
      <td>Science and Nature</td>
      <td>0.811</td>
      <td>0.001</td>
      <td>0.086</td>
      <td>0.101</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book54</th>
      <td>The Burgess Animal Book for Children</td>
      <td>Science and Nature</td>
      <td>0.626</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.374</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book56</th>
      <td>The History of Insects</td>
      <td>Science and Nature</td>
      <td>0.810</td>
      <td>0.000</td>
      <td>0.190</td>
      <td>0.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book57</th>
      <td>The Insect Folk</td>
      <td>Science and Nature</td>
      <td>0.776</td>
      <td>0.002</td>
      <td>0.000</td>
      <td>0.222</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book59</th>
      <td>Little Busybodies;The Life of Crickets, Ants, ...</td>
      <td>Science and Nature</td>
      <td>0.645</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.354</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book60</th>
      <td>Outlines of Lessons in Botany, Part I; from Se...</td>
      <td>Science and Nature</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book63</th>
      <td>Camping For Boys</td>
      <td>Science and Nature</td>
      <td>0.737</td>
      <td>0.000</td>
      <td>0.042</td>
      <td>0.221</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book64</th>
      <td>Quadrupeds, What They Are and Where Found;A Bo...</td>
      <td>Science and Nature</td>
      <td>0.812</td>
      <td>0.000</td>
      <td>0.188</td>
      <td>0.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book66</th>
      <td>The Story of Eclipses</td>
      <td>Science and Nature</td>
      <td>0.657</td>
      <td>0.007</td>
      <td>0.332</td>
      <td>0.004</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book67</th>
      <td>Country Walks of a Naturalist with His Children</td>
      <td>Science and Nature</td>
      <td>0.714</td>
      <td>0.000</td>
      <td>0.038</td>
      <td>0.248</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book68</th>
      <td>On the Trail: An Outdoor Book for Girls</td>
      <td>Science and Nature</td>
      <td>0.899</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.101</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book69</th>
      <td>Our Common Insects;A Popular Account of the In...</td>
      <td>Science and Nature</td>
      <td>0.950</td>
      <td>0.001</td>
      <td>0.049</td>
      <td>0.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book70</th>
      <td>The Wonders of the Jungle;Book One</td>
      <td>Science and Nature</td>
      <td>0.506</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.494</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book77</th>
      <td>Little Journey to Puerto Rico : for Intermedia...</td>
      <td>Geography</td>
      <td>0.624</td>
      <td>0.000</td>
      <td>0.120</td>
      <td>0.256</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book78</th>
      <td>Where We Live;A Home Geography</td>
      <td>Geography</td>
      <td>0.723</td>
      <td>0.000</td>
      <td>0.122</td>
      <td>0.155</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book79</th>
      <td>Peeps at Many Lands: Norway</td>
      <td>Geography</td>
      <td>0.485</td>
      <td>0.000</td>
      <td>0.112</td>
      <td>0.402</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book80</th>
      <td>Commercial GeographyA Book for High Schools, C...</td>
      <td>Geography</td>
      <td>0.655</td>
      <td>0.000</td>
      <td>0.345</td>
      <td>0.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book83</th>
      <td>A Catechism of Familiar Things; Their History,...</td>
      <td>Uncategorized</td>
      <td>0.687</td>
      <td>0.000</td>
      <td>0.313</td>
      <td>0.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book85</th>
      <td>The Story of the Mind</td>
      <td>Uncategorized</td>
      <td>0.627</td>
      <td>0.000</td>
      <td>0.373</td>
      <td>0.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book86</th>
      <td>The Story of Glass</td>
      <td>Uncategorized</td>
      <td>0.514</td>
      <td>0.000</td>
      <td>0.081</td>
      <td>0.405</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book87</th>
      <td>The Story of Porcelain</td>
      <td>Uncategorized</td>
      <td>0.612</td>
      <td>0.000</td>
      <td>0.155</td>
      <td>0.233</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book95</th>
      <td>Electricity for Boys</td>
      <td>Uncategorized</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book97</th>
      <td>The Boy Mechanic: Volume 1700 Things for Boys ...</td>
      <td>Uncategorized</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book102</th>
      <td>Ontario Teachers' Manuals: Household Management</td>
      <td>Uncategorized</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Book103</th>
      <td>Ontario Teachers' Manuals: Household Science i...</td>
      <td>Uncategorized</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
%load_ext version_information
%version_information pandas, numpy, requests, bs4, selenium, lxml, urllib3, pyvirtualdisplay, unipath
```

    The version_information extension is already loaded. To reload it, use:
      %reload_ext version_information





<table><tr><th>Software</th><th>Version</th></tr><tr><td>Python</td><td>3.6.8 64bit [GCC 7.3.0]</td></tr><tr><td>IPython</td><td>7.2.0</td></tr><tr><td>OS</td><td>Linux 4.15.0 46 generic x86_64 with debian buster sid</td></tr><tr><td>pandas</td><td>0.22.0</td></tr><tr><td>numpy</td><td>1.16.2</td></tr><tr><td>requests</td><td>2.21.0</td></tr><tr><td>bs4</td><td>4.7.1</td></tr><tr><td>selenium</td><td>3.141.0</td></tr><tr><td>lxml</td><td>4.3.2</td></tr><tr><td>urllib3</td><td>1.24.1</td></tr><tr><td>pyvirtualdisplay</td><td>0.2.1</td></tr><tr><td>unipath</td><td>1.1</td></tr><tr><td colspan='2'>Sat Apr 13 16:24:38 2019 CDT</td></tr></table>




Refernces
* https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/
