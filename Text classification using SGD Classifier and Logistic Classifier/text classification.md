

```python
from io import StringIO
import requests
import json
import pandas as pd

df_data_1 = pd.read_csv(get_object_storage_file_with_credentials_2659b32ff9a04774afc2ee0815088bcf('project1', 'twitter_train.csv'))
df_data_1.head()
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
      <th>id</th>
      <th>label</th>
      <th>tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>@user when a father is dysfunctional and is s...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>@user @user thanks for #lyft credit i can't us...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>bihday your majesty</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>#model   i love u take with u all the time in ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>factsguide: society now    #motivation</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df_data_1.shape)

print(df_data_1.groupby(by=["label"])['label'].count())
```

    (31962, 3)
    label
    0    29720
    1     2242
    Name: label, dtype: int64



```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cross_validation import cross_val_predict
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from functools import lru_cache
from nltk.tag.perceptron import PerceptronTagger
```


```python
# Initiate lemmatizer
wnl = WordNetLemmatizer()

# Load tagger pickle
tagger = PerceptronTagger()

# Lookup if tag is noun, verb, adverb or an adjective
tags = {'N': wn.NOUN, 'V': wn.VERB, 'R': wn.ADV, 'J': wn.ADJ}

# Memoization of POS tagging and Lemmatizer
lemmatize_mem = lru_cache(maxsize=10000)(wnl.lemmatize)
tagger_mem = lru_cache(maxsize=10000)(tagger.tag)

whitelist = ["n't", "not"]
modified_stopwords = []
for idx, stop_word in enumerate(ENGLISH_STOP_WORDS):
    if stop_word not in whitelist:
        modified_stopwords.append(stop_word)

# POS tag sentences and lemmatize each word
def tokenizer(text):
    for token in wordpunct_tokenize(text):
        if token not in modified_stopwords:
            tag = tagger_mem(frozenset({token}))
            yield lemmatize_mem(token, tags.get(tag[0][1], wn.NOUN))
```


```python
# Pipeline definition
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        tokenizer=tokenizer,
        ngram_range=(2, 3),
        stop_words=modified_stopwords,
        sublinear_tf=True,
        min_df=0.0001,
        max_df=0.3
    )),
    ('classifier', MultinomialNB()),
])
```


```python
# Cross validate using k-fold
y_pred = cross_val_predict(
    pipeline, df_data_1.get('tweet'),
    y=df_data_1.get('label'),
    cv=10, n_jobs=-1, verbose=20
)

cm = confusion_matrix(df_data_1.get('label'), y_pred)

cm
```

    [Parallel(n_jobs=-1)]: Done   1 tasks      | elapsed:   27.9s
    [Parallel(n_jobs=-1)]: Done   2 out of  10 | elapsed:   28.3s remaining:  1.9min
    [Parallel(n_jobs=-1)]: Done   3 out of  10 | elapsed:   28.5s remaining:  1.1min
    [Parallel(n_jobs=-1)]: Done   4 out of  10 | elapsed:   28.7s remaining:   43.0s
    [Parallel(n_jobs=-1)]: Done   5 out of  10 | elapsed:   28.7s remaining:   28.7s
    [Parallel(n_jobs=-1)]: Done   6 out of  10 | elapsed:   28.7s remaining:   19.2s
    [Parallel(n_jobs=-1)]: Done   7 out of  10 | elapsed:   28.8s remaining:   12.3s
    [Parallel(n_jobs=-1)]: Done   8 out of  10 | elapsed:   28.8s remaining:    7.2s
    [Parallel(n_jobs=-1)]: Done  10 out of  10 | elapsed:   29.4s remaining:    0.0s
    [Parallel(n_jobs=-1)]: Done  10 out of  10 | elapsed:   29.4s finished





    array([[29681,    39],
           [ 1501,   741]])




```python

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-24-8b468a3c5682> in <module>()
    ----> 1 metrics(df_data_1.get('label')==y_pred)
    

    NameError: name 'metrics' is not defined



```python
pipeline_SGD = Pipeline([
    ('vectorizer', TfidfVectorizer(
        tokenizer=tokenizer,
        ngram_range=(1, 2),
        stop_words=modified_stopwords,
        sublinear_tf=True,
        min_df=0.0009
    )),
    ('classifier', SGDClassifier(
        alpha=1e-4, n_jobs=-1
    )),
])

# Cross validate using k-fold
y_pred_SGD = cross_val_predict(
    pipeline_SGD, df_data_1.get('tweet'),
    y=df_data_1.get('label'),
    cv=10, n_jobs=-1, verbose=20
)
```

    /usr/local/src/conda3_runtime/home/envs/DSX-Python35-Spark/lib/python3.5/site-packages/sklearn/linear_model/stochastic_gradient.py:128: FutureWarning: max_iter and tol parameters have been added in <class 'sklearn.linear_model.stochastic_gradient.SGDClassifier'> in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      "and default tol will be 1e-3." % type(self), FutureWarning)
    /usr/local/src/conda3_runtime/home/envs/DSX-Python35-Spark/lib/python3.5/site-packages/sklearn/linear_model/stochastic_gradient.py:128: FutureWarning: max_iter and tol parameters have been added in <class 'sklearn.linear_model.stochastic_gradient.SGDClassifier'> in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      "and default tol will be 1e-3." % type(self), FutureWarning)
    /usr/local/src/conda3_runtime/home/envs/DSX-Python35-Spark/lib/python3.5/site-packages/sklearn/linear_model/stochastic_gradient.py:128: FutureWarning: max_iter and tol parameters have been added in <class 'sklearn.linear_model.stochastic_gradient.SGDClassifier'> in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      "and default tol will be 1e-3." % type(self), FutureWarning)
    /usr/local/src/conda3_runtime/home/envs/DSX-Python35-Spark/lib/python3.5/site-packages/sklearn/linear_model/stochastic_gradient.py:128: FutureWarning: max_iter and tol parameters have been added in <class 'sklearn.linear_model.stochastic_gradient.SGDClassifier'> in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      "and default tol will be 1e-3." % type(self), FutureWarning)
    /usr/local/src/conda3_runtime/home/envs/DSX-Python35-Spark/lib/python3.5/site-packages/sklearn/linear_model/stochastic_gradient.py:128: FutureWarning: max_iter and tol parameters have been added in <class 'sklearn.linear_model.stochastic_gradient.SGDClassifier'> in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      "and default tol will be 1e-3." % type(self), FutureWarning)
    [Parallel(n_jobs=-1)]: Done   1 tasks      | elapsed:   15.5s
    [Parallel(n_jobs=-1)]: Done   2 out of   5 | elapsed:   15.7s remaining:   23.5s
    [Parallel(n_jobs=-1)]: Done   3 out of   5 | elapsed:   15.7s remaining:   10.5s
    [Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:   15.9s remaining:    0.0s
    [Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:   15.9s finished



```python
cm = confusion_matrix(df_data_1.get('label'), y_pred_SGD)

cm
```




    array([[29618,   102],
           [ 1488,   754]])




```python
from sklearn.linear_model import LogisticRegression
```


```python
# Pipeline definition
pipeline_logistic = Pipeline([
    ('vectorizer', TfidfVectorizer(
        tokenizer=tokenizer,
        ngram_range=(1, 2),
        stop_words=modified_stopwords,
        sublinear_tf=True,
        min_df=0.0001,
        max_df=0.3
    )),
    ('classifier', LogisticRegression()),
])

# Cross validate using k-fold
y_pred_logistic = cross_val_predict(
    pipeline_logistic, df_data_1.get('tweet'),
    y=df_data_1.get('label'),
    cv=10, n_jobs=-1, verbose=20
)

cm = confusion_matrix(df_data_1.get('label'), y_pred_logistic)

cm
```

    [Parallel(n_jobs=-1)]: Done   1 tasks      | elapsed:   35.9s
    [Parallel(n_jobs=-1)]: Done   2 out of  10 | elapsed:   36.3s remaining:  2.4min
    [Parallel(n_jobs=-1)]: Done   3 out of  10 | elapsed:   36.3s remaining:  1.4min
    [Parallel(n_jobs=-1)]: Done   4 out of  10 | elapsed:   36.5s remaining:   54.7s
    [Parallel(n_jobs=-1)]: Done   5 out of  10 | elapsed:   36.5s remaining:   36.5s
    [Parallel(n_jobs=-1)]: Done   6 out of  10 | elapsed:   36.6s remaining:   24.4s
    [Parallel(n_jobs=-1)]: Done   7 out of  10 | elapsed:   36.6s remaining:   15.7s
    [Parallel(n_jobs=-1)]: Done   8 out of  10 | elapsed:   36.6s remaining:    9.2s
    [Parallel(n_jobs=-1)]: Done  10 out of  10 | elapsed:   36.6s remaining:    0.0s
    [Parallel(n_jobs=-1)]: Done  10 out of  10 | elapsed:   36.6s finished





    array([[29657,    63],
           [ 1441,   801]])




```python
logistic_model = pipeline_logistic.fit(df_data_1.get('tweet'), df_data_1.get('label'))
```


```python
df_data_2 = pd.read_csv(get_object_storage_file_with_credentials_2659b32ff9a04774afc2ee0815088bcf('project1', 'test_tweets.csv'))
print(df_data_2.head())

print(df_data_2.shape)
```

          id                                              tweet
    0  31963  #studiolife #aislife #requires #passion #dedic...
    1  31964   @user #white #supremacists want everyone to s...
    2  31965  safe ways to heal your #acne!!    #altwaystohe...
    3  31966  is the hp and the cursed child book up for res...
    4  31967    3rd #bihday to my amazing, hilarious #nephew...
    (17197, 2)



```python
predictions = logistic_model.predict(df_data_2.get('tweet'))
```


```python
## Function to put file to the IBM datascience folder
def put_file(credentials, local_file_name): 
    """This functions returns a StringIO object containing the file content from Bluemix Object Storage V3.""" 
    f = open(local_file_name,'r') 
    my_data = f.read() 
    url1 = ''.join(['https://identity.open.softlayer.com', '/v3/auth/tokens']) 
    data = {'auth': {'identity': {'methods': ['password'], 'password': {'user': {'name': credentials['username'],'domain': {'id': credentials['domain_id']}, 'password': credentials['password']}}}}} 
    headers1 = {'Content-Type': 'application/json'} 
    resp1 = requests.post(url=url1, data=json.dumps(data), headers=headers1) 
    resp1_body = resp1.json() 
    for e1 in resp1_body['token']['catalog']: 
        if(e1['type']=='object-store'): 
            for e2 in e1['endpoints']: 
                if(e2['interface']=='public'and e2['region']== credentials['region']):    
                    url2 = ''.join([e2['url'],'/', credentials['container'], '/', local_file_name]) 
    
    s_subject_token = resp1.headers['x-subject-token'] 
    headers2 = {'X-Auth-Token': s_subject_token, 'accept': 'application/json'} 
    resp2 = requests.put(url=url2, headers=headers2, data = my_data ) 
    print (resp2)
```


```python
df = pd.DataFrame(predictions, columns = ['label'])

df.to_csv('Dataset.csv',index=False)

put_file(credentials_2661 ,'Dataset.csv')
```

    <Response [201]>

