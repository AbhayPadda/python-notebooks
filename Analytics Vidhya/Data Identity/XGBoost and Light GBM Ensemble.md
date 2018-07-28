

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
```


```python
train = pd.read_csv("train_HK6lq50.csv")
test = pd.read_csv("test_2nAIblo.csv")
```


```python
# Imputing missing value in age variable with median and in trainee_engagement_rating variable with mode of that variable.
train['age'].fillna(train['age'].median(), inplace=True)
train['trainee_engagement_rating'].fillna(train['trainee_engagement_rating'].mode()[0], inplace=True)
test['age'].fillna(train['age'].median(), inplace=True)
test['trainee_engagement_rating'].fillna(train['trainee_engagement_rating'].mode()[0], inplace=True)
```


```python
dummy_fields = ['difficulty_level','education',"city_tier","gender","is_handicapped","test_type", "program_id"] 
```


```python
for item in dummy_fields:
    dummies = pd.get_dummies(train.loc[:, item], prefix=item) 
    train = pd.concat([train, dummies], axis = 1)
    train = train.drop(item, axis =1)
    dummies = pd.get_dummies(test.loc[:, item], prefix=item) 
    test = pd.concat([test, dummies], axis = 1)
    test = test.drop(item, axis =1)
```


```python
train.drop('program_type', inplace=True, axis=1)
test.drop('program_type', inplace=True, axis=1)
```


```python
from sklearn.model_selection import train_test_split
```


```python
y = train.is_pass.values
train.drop(['id', 'is_pass'], inplace=True, axis=1)
#x, x_test, y, y_test = train_test_split(train, y, test_size=0.015, random_state=42, stratify=y)
x = train
```


```python
from imblearn.over_sampling import SMOTE
ros = SMOTE(random_state=0)
X_resampled, y_resampled = ros.fit_sample(x, y)
```


```python
import xgboost as xgb
data = np.array(X_resampled)  # 5 entities, each contains 10 features
label = np.array(y_resampled)  # binary target
dtrain = xgb.DMatrix(data, label=label)
```


```python
#data = np.array(x_test)  # 5 entities, each contains 10 features
#label = np.array(y_test)  # binary target
#dtest = xgb.DMatrix(data, label=label)
```


```python
parameters = {
    'objective': 'binary:logistic',
    'silent': 0,
    'eval_metric': 'auc'
}

#evallist = [(dtest, 'eval'), (dtrain, 'train')]

```


```python
num_round = 2500
bst = xgb.train(parameters, dtrain, num_round)
```


```python
test_ids = test['id']
test = test.drop('id',axis=1)
```


```python
dtest = xgb.DMatrix(test.values)
pred = bst.predict(dtest)
d = pd.DataFrame(pred)
d.columns = ['is_pass']
d.to_csv('xgb_submissions.csv')
```


```python
import lightgbm
gbm_train_data = lightgbm.Dataset(X_resampled, label=y_resampled)
#gbm_test_data = lightgbm.Dataset(x_test, label=y_test)
parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'verbose': 0
}

gbm_model = lightgbm.train(parameters,
                       gbm_train_data,
                       num_boost_round=2000)
```


```python
gbm_pred = gbm_model.predict(test)
output = pd.DataFrame({'is_pass': gbm_pred})
output.to_csv("lgbm_submissions.csv", index=False)
```


```python
xg_pred = bst.predict(dtest)
gbm_pred = gbm_model.predict(test)

pred = (0.65 * xg_pred) + (0.35 * gbm_pred)
```


```python
output = pd.DataFrame({'id': test_ids, 'is_pass': pred})
output.to_csv("final_output.csv", index=False)
```
