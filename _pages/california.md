---
title: "California housing prices"
permalink: /california/
header:
  image: "/images/newfront.jpg"
classes: wide
---

## Table of Contents:
* [1-Preprocessing the data](#preprocessing)
* [2-Linear Regression](#Regression)
    * [2.1-Training the model](#Training)
    * [2.2-Evaluating the model](#Evaluation)
* [3-XGBoost](#Xgboost)
    * [3.1-Training the model](#Training2)
    * [3.2-Evaluating the model](#Evaluation2)
* [4-XGBoost vs Linear Regression](#Comparison)

## Preprocessing the data <a class="anchor" id="preprocessing"></a>


```python
#importing the libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
#loading the dataset and obtaining info about columns
df=pd.read_csv("housing.csv")
list(df)
```




    ['longitude',
     'latitude',
     'housing_median_age',
     'total_rooms',
     'total_bedrooms',
     'population',
     'households',
     'median_income',
     'median_house_value',
     'ocean_proximity']




```python
#description of the numerical columns
df.describe()
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20433.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.569704</td>
      <td>35.631861</td>
      <td>28.639486</td>
      <td>2635.763081</td>
      <td>537.870553</td>
      <td>1425.476744</td>
      <td>499.539680</td>
      <td>3.870671</td>
      <td>206855.816909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.003532</td>
      <td>2.135952</td>
      <td>12.585558</td>
      <td>2181.615252</td>
      <td>421.385070</td>
      <td>1132.462122</td>
      <td>382.329753</td>
      <td>1.899822</td>
      <td>115395.615874</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.930000</td>
      <td>18.000000</td>
      <td>1447.750000</td>
      <td>296.000000</td>
      <td>787.000000</td>
      <td>280.000000</td>
      <td>2.563400</td>
      <td>119600.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.490000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2127.000000</td>
      <td>435.000000</td>
      <td>1166.000000</td>
      <td>409.000000</td>
      <td>3.534800</td>
      <td>179700.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.710000</td>
      <td>37.000000</td>
      <td>3148.000000</td>
      <td>647.000000</td>
      <td>1725.000000</td>
      <td>605.000000</td>
      <td>4.743250</td>
      <td>264725.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6445.000000</td>
      <td>35682.000000</td>
      <td>6082.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#count the values of the columns
df.count()
```




    longitude             20640
    latitude              20640
    housing_median_age    20640
    total_rooms           20640
    total_bedrooms        20433
    population            20640
    households            20640
    median_income         20640
    median_house_value    20640
    ocean_proximity       20640
    dtype: int64




```python
#We have missing values in the column total_bedrooms. We can drop the null rows or replace the null value for the mean.
#I choose to replace it with the mean
df['total_bedrooms'].fillna(df['total_bedrooms'].mean(), inplace=True)

```


```python
#I want information about the column "ocean_proximity"
df['ocean_proximity'].value_counts()

```




    <1H OCEAN     9136
    INLAND        6551
    NEAR OCEAN    2658
    NEAR BAY      2290
    ISLAND           5
    Name: ocean_proximity, dtype: int64




```python
#Transform the variable into a numerical one.
def map_age(age):
    if age == '<1H OCEAN':
        return 0
    elif age == 'INLAND':
        return 1
    elif age == 'NEAR OCEAN':
        return 2
    elif age == 'NEAR BAY':
        return 3
    elif age == 'ISLAND':
        return 4
df['ocean_proximity'] = df['ocean_proximity'].apply(map_age)
```


```python
#Obtaining info of the correlations with a heatmap
plt.figure(figsize=(15,8))
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df.corr(), linewidths=.5,annot=True,mask=mask,cmap='coolwarm')

```




    <matplotlib.axes._subplots.AxesSubplot at 0x2d174ac0eb8>




![png](/images/california/output_9_1.png)



```python
#There is a high correlation between households and populati
df.drop('households', axis=1, inplace=True)

```


```python
# let's create 2 more columns with the total bedrooms and rooms per population in the same block.
df['average_rooms']=df['total_rooms']/df['population']
df['average_bedrooms']=df['total_bedrooms']/df['population']

```


```python
#dropping the 2 columns we are not going to use
df.drop('total_rooms',axis=1,inplace=True)
df.drop('total_bedrooms',axis=1,inplace=True)
```


```python
#Obtaining info of the new correlations with a heatmap
plt.figure(figsize=(15,8))
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df.corr(), linewidths=.5,annot=True,mask=mask,cmap='coolwarm')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2d1768818d0>




![png](/images/california/output_13_1.png)



```python
#histogram to get the distributions of the different variables
df.hist(bins=70, figsize=(20,20))
plt.show()
```


![png](/images/california/output_14_0.png)



```python
#Finding Outliers
plt.figure(figsize=(15,5))
sns.boxplot(x=df['housing_median_age'])
plt.figure()
plt.figure(figsize=(15,5))
sns.boxplot(x=df['median_house_value'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2d177c1d8d0>




![png](/images/california/output_15_1.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/california/output_15_3.png)



```python
#removing outliers
df=df.loc[df['median_house_value']<500001,:]
```

# Linear Regression <a class="anchor" id="Regression"></a>

## Training the model <a class="anchor" id="Training"></a>


```python
#Choosing the dependant variable and the regressors. In this case we want to predict the housing price
X=df[['longitude',
 'latitude',
 'housing_median_age',
 'population',
 'median_income',
 'ocean_proximity',
 'average_rooms',
 'average_bedrooms']]
Y=df['median_house_value']
```


```python
#splitting the dataset into the train set and the test set

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state=0)
```


```python
#Training the model
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,Y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
             normalize=False)




```python
#Obtaining the predictions
y_pred = regressor.predict(X_test)

```

## Evaluating the model <a class="anchor" id="Evaluation"></a>


```python
#R2 score
from sklearn.metrics import r2_score
r2=r2_score(Y_test,y_pred)
print('the R squared of the linear regression is:', r2)
```

    the R squared of the linear regression is: 0.5526714001645363



```python
#Graphically
grp = pd.DataFrame({'prediction':y_pred,'Actual':Y_test})
grp = grp.reset_index()
grp = grp.drop(['index'],axis=1)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(20,10))
plt.plot(grp[:120],linewidth=2)
plt.legend(['Actual','Predicted'],prop={'size': 20})
```




    <matplotlib.legend.Legend at 0x2d1765c9dd8>




![png](/images/california/output_25_1.png)


# XGBoost <a class="anchor" id="Xgboost"></a>

## Training the model <a class="anchor" id="Training2"></a>


```python
import xgboost as xgb
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 1,eta=0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 2000)
```


```python
xg_reg.fit(X_train,Y_train)

y_pred2 = xg_reg.predict(X_test)
```

## Evaluating the model <a class="anchor" id="Evaluation2"></a>


```python
#Graphically
grp = pd.DataFrame({'prediction':y_pred2,'Actual':Y_test})
grp = grp.reset_index()
grp = grp.drop(['index'],axis=1)
plt.figure(figsize=(20,10))
plt.plot(grp[:120],linewidth=2)
plt.legend(['Actual','Predicted'],prop={'size': 20})
```




    <matplotlib.legend.Legend at 0x2d1783f10b8>




![png](/images/california/output_31_1.png)



```python
r2xgb=r2_score(Y_test,y_pred2)
print('the R squared of the xgboost method is:', r2xgb)
```

    the R squared of the xgboost method is: 0.8227763364288538



```python
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
```


![png](/images/california/output_33_0.png)



```python
#Doing cross validation to see the accuracy of the XGboost model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(regressor, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

    Accuracy: 43.75% (10.12%)


# Linear regression vs XGBoost <a class="anchor" id="Comparison"></a>


```python
#comparing the scores of both techniques
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

mae1 = mean_absolute_error(Y_test, y_pred)
rms1 = sqrt(mean_squared_error(Y_test, y_pred))
mae2 =mean_absolute_error(Y_test,y_pred2)
rms2 = sqrt(mean_squared_error(Y_test, y_pred2))

print('Stats for the linear regression: \n','mean squared error: ',rms1, '\n R2:',r2,' \n mean absolute error:',mae1 )
print('Stats xgboost: \n','mean squared error: ',rms2, '\n R2:',r2xgb,' \n mean absolute error:',mae2 )
```

    Stats for the linear regression:
     mean squared error:  65524.097680759056
     R2: 0.5526714001645363  
     mean absolute error: 47427.66363813204
    Stats xgboost:
     mean squared error:  41242.84075742074
     R2: 0.8227763364288538  
     mean absolute error: 27488.045577549237
