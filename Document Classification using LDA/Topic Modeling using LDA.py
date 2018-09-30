# coding: utf-8

# ## Topic Modeling using LDA
# 
# #### Using the already available 20newsgroup dataset which already has data grouped into pre-defined 20 news categories

# In[1]:


import warnings

warnings.filterwarnings('ignore')


# In[2]:


from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train', shuffle = True)
newsgroups_test = fetch_20newsgroups(subset='test', shuffle = True)


# #### Check unique categories in the dataset

# In[3]:


set(newsgroups_train.target_names)


# #### check first 5 rows

# In[4]:


newsgroups_train.data[:5]


# ### Data Pre-processing

# In[5]:


#!pip install gensim


# In[6]:


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(400)


# ### Functions to perform pre-processing

# In[7]:


def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result


# #### Pre-process data

# In[8]:


processed_docs = []

for doc in newsgroups_train.data:
    processed_docs.append(preprocess(doc))


# #### Creating Bag of Words from the processed data

# In[9]:


dictionary = gensim.corpora.Dictionary(processed_docs)


# #### Filter extreme cases. Words with frequency less than 10 and words appearing in more than 20% of the documents

# In[10]:


dictionary.filter_extremes(no_below=10,no_above=0.2,keep_n= 100000)


# In[11]:


bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


# ### Running LDA on Bag of Words

# In[12]:


## Creating 8 topics from the dictionary created and bow corpus
lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 8, 
                                   id2word = dictionary,                                    
                                   passes = 10,
                                   workers = 2)


# #### Check words occuring for each topic

# In[13]:


for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")


# ### Classification of the topics
# #### Using the words in each topic and their corresponding weights, what categories were you able to infer?
# 
# * 0: Graphics Cards
# * 1: Politics
# * 2: Gun Violence 
# * 3: Sports
# * 4: Religion 
# * 5: Technology
# * 6: Driving
# * 7: Encryption

# ### Testing Model

# In[14]:


test_doc = newsgroups_test.data[10]
test_doc


# In[15]:


## Pre-processing test document
bow_vector = dictionary.doc2bow(preprocess(test_doc))

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))


# In[16]:


newsgroups_test.target[10]
<<<<<<< HEAD


# ## Visualizing LDA model

# In[17]:


#!pip install pyLDAvis


# In[18]:


import pyLDAvis.gensim


# In[19]:


lda_display = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary, sort_topics=False)


# Saliency: tells how much the term tells about the topic.
# 
# Relevance: a weighted average of the probability of the word given the topic and the word given the topic normalized by the probability of the topic.
# 
# The size of the bubble measures the importance of the topics, relative to the data.

# In[20]:


pyLDAvis.display(lda_display)

=======
>>>>>>> 6fb2f6388e9ae9c5a15cdec137c9b1daa4cdfc13
