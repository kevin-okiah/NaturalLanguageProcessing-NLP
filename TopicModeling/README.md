


## Topic Modeling - GutenBerg Books

#### Author: Kevin Okiah

**04/13/2019**

In this notebook, I explore Topic modeling using "Latent Dirichlet allocation" (LDA). LDA is a generative probabilistic model for collections of
discrete data such as text corpora. LDA is a three-level hierarchical Bayesian model,in which each item of a collection is modeled as a finite mixture over an underlying set of topics. Each topic is, in turn, modeled as an infinite mixture over an underlying set of topic probabilities. In the context of text modeling, the topic probabilities provide an explicit representation of a document. [Link](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)

#### Problem Statement
![](https://github.com/kevimwe/NaturalLanguageProcessing-NLP/blob/master/TopicModeling/topic_modeling.JPG)


#### NLP Pipeline Approach
I follow an NLP pipleine approach to find books similar to `The Flag of My Country. Shikéyah Bidah Na'at'a'í; Navajo New World Readers 2` from [Children's Instructional Books (Bookshelf)](https://www.gutenberg.org/wiki/Children%27s_Instructional_Books_(Bookshelf)) to recommend to my son as shown in the flow chart below.
![](https://github.com/kevimwe/NaturalLanguageProcessing-NLP/blob/master/TopicModeling/nlp_pipeline.JPG)

Below are links to notebooks for the seperate steps:
* [**Data Acqusitions**](https://github.com/kevimwe/NaturalLanguageProcessing-NLP/blob/master/TopicModeling/Notebooks/Scrape_GuternsbergBooks.ipynb)
* [**Pre-processing and EDA**](https://github.com/kevimwe/NaturalLanguageProcessing-NLP/blob/master/TopicModeling/Notebooks/TextCleaning_and_EDA.ipynb)
* [**Modeling and Results**](https://github.com/kevimwe/NaturalLanguageProcessing-NLP/blob/master/TopicModeling/TopicModeling_IMDB_ActionMoviesReviews-LDA.ipynb)

Here is the link to the summary report [**Powerpoint Summary**](https://github.com/kevimwe/NaturalLanguageProcessing-NLP/blob/master/TopicModeling/Results/TopicModelingTermPresentation.pdf) 



   
### References
* https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/
