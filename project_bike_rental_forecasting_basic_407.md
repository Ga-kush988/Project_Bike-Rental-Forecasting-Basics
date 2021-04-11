

```python
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics  import mean_squared_error
from sklearn  import linear_model 
import matplotlib.pyplot as plt
import os
```


```python
np.random.seed(42)
```


```python
filePath = '/cxldata/datasets/project/bikes.csv'
bikesData = pd.read_csv(filePath)
print(bikesData.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 17379 entries, 0 to 17378
    Data columns (total 17 columns):
    instant       17379 non-null int64
    dteday        17379 non-null object
    season        17379 non-null int64
    yr            17379 non-null int64
    mnth          17379 non-null int64
    hr            17379 non-null int64
    holiday       17379 non-null int64
    weekday       17379 non-null int64
    workingday    17379 non-null int64
    weathersit    17379 non-null int64
    temp          17379 non-null float64
    atemp         17379 non-null float64
    hum           17379 non-null float64
    windspeed     17379 non-null float64
    casual        17379 non-null int64
    registered    17379 non-null int64
    cnt           17379 non-null int64
    dtypes: float64(4), int64(12), object(1)
    memory usage: 2.3+ MB
    None



```python
filePath = '/cxldata/datasets/project/bikes.csv'
bikesData = pd.read_csv(filePath)
```


```python
columnsToDrop = ['instant','casual','registered','atemp','dteday']

bikesData = bikesData.drop(columnsToDrop,axis=1)
```


```python
train_set = bikesData[:12165]
test_set = bikesData[12165:]
```


```python
import numpy as np
import pandas as pd

```


```python
np.random.seed(42)
from sklearn.model_selection import train_test_split

bikesData['dayCount'] = pd.Series(range(bikesData.shape[0]))/24

train_set, test_set = train_test_split(bikesData, test_size=0.3, random_state=42)

print(len(train_set), "train +", len(test_set), "test")

train_set.sort_values('dayCount', axis= 0, inplace=True)
test_set.sort_values('dayCount', axis= 0, inplace=True)
```

    12165 train + 5214 test


    /usr/local/anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      # Remove the CWD from sys.path while we load stuff.
    /usr/local/anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      # This is added back by InteractiveShellApp.init_path()



```python
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
```


```python
columnsToScale = ['temp','hum','windspeed']

scaler = StandardScaler()

train_set[columnsToScale] = scaler.fit_transform(train_set[columnsToScale])
test_set[columnsToScale] = scaler.transform(test_set[columnsToScale])
train_set[columnsToScale].describe()
```

    /usr/local/anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """
    /usr/local/anaconda/lib/python3.6/site-packages/pandas/core/indexing.py:494: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.obj[item] = s
    /usr/local/anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    /usr/local/anaconda/lib/python3.6/site-packages/pandas/core/indexing.py:494: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.obj[item] = s





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
      <th>temp</th>
      <th>hum</th>
      <th>windspeed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>1.216500e+04</td>
      <td>1.216500e+04</td>
      <td>1.216500e+04</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>1.658955e-15</td>
      <td>4.775031e-17</td>
      <td>-1.367550e-15</td>
    </tr>
    <tr>
      <td>std</td>
      <td>1.000041e+00</td>
      <td>1.000041e+00</td>
      <td>1.000041e+00</td>
    </tr>
    <tr>
      <td>min</td>
      <td>-2.476000e+00</td>
      <td>-3.245965e+00</td>
      <td>-1.552670e+00</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>-8.186290e-01</td>
      <td>-7.628859e-01</td>
      <td>-6.962541e-01</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>1.005628e-02</td>
      <td>1.307622e-02</td>
      <td>-2.069907e-01</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>8.387416e-01</td>
      <td>8.407692e-01</td>
      <td>5.264946e-01</td>
    </tr>
    <tr>
      <td>max</td>
      <td>2.599698e+00</td>
      <td>1.927116e+00</td>
      <td>5.419128e+00</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost
from xgboost import XGBRegressor

trainingCols = train_set.drop(['cnt'], axis=1)
trainingLabels = train_set['cnt']
```


```python
dec_reg = DecisionTreeRegressor(random_state = 42)

dt_mae_scores = -cross_val_score(dec_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_absolute_error")

display_scores(dt_mae_scores)

dt_mse_scores = np.sqrt(-cross_val_score(dec_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_squared_error"))

display_scores(dt_mse_scores)

 
```

    Scores: [42.94494659 50.37222679 36.95891537 44.26211997 46.99589154 71.98026316
     58.19901316 48.87417763 50.84868421 96.46217105]
    Mean: 54.7898409457034
    Standard deviation: 16.563759407187572
    Scores: [ 65.39786583  77.67402864  60.57274567  73.73250527  75.48574011
     113.22922285  96.5884429   82.11639785  86.86752618 149.13680359]
    Mean: 88.0801278896052
    Standard deviation: 24.927341207369675



```python
lin_reg = LinearRegression()
lr_mae_scores = -cross_val_score(lin_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_absolute_error")
display_scores(lr_mae_scores)
lr_mse_scores = np.sqrt(-cross_val_score(lin_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_squared_error"))
display_scores(lr_mse_scores)
```

    Scores: [ 66.96340699  80.48809095 113.84704981  93.17230086  76.11197672
      96.5220689  133.13798218 158.02254734 158.90195479 127.15674717]
    Mean: 110.43241256942311
    Standard deviation: 31.42696570529551
    Scores: [ 84.63836676 111.12038541 131.88324414 119.16350622 105.17621319
     127.72562924 174.97188817 187.31691741 205.60028279 164.30585678]
    Mean: 141.1902290118181
    Standard deviation: 37.55565075919517



```python
#Train a Random Forest Regressor

forest_reg = RandomForestRegressor(n_estimators=150, random_state=42)

rf_mae_scores = -cross_val_score(forest_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_absolute_error")

display_scores(rf_mae_scores)

rf_mse_scores = np.sqrt(-cross_val_score(forest_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_squared_error"))

display_scores(rf_mse_scores)
```

    Scores: [33.39666393 33.54451931 28.50225692 31.78826623 36.55658724 57.81963268
     40.96405702 40.84652961 37.57766447 84.69771382]
    Mean: 42.56938912059061
    Standard deviation: 15.980256848600963
    Scores: [ 45.64176074  50.97205843  43.37588352  52.2640926   60.46557726
      94.24478873  66.26045287  65.45672124  61.69916554 131.9727285 ]
    Mean: 67.23532294382946
    Standard deviation: 25.544513111074128



```python
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators':[120,150],'max_features':[10,12],'max_depth':[15,28]},
]
```


```python
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error')
```


```python
grid_search.fit(trainingCols, trainingLabels)

print(grid_search.best_estimator_)
print(grid_search.best_params_)
```

    RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                          max_depth=28, max_features=10, max_leaf_nodes=None,
                          max_samples=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=1,
                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                          n_estimators=150, n_jobs=None, oob_score=False,
                          random_state=42, verbose=0, warm_start=False)
    {'max_depth': 28, 'max_features': 10, 'n_estimators': 150}



```python
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
```

    [0.00424888 0.00145493 0.00570279 0.58348648 0.00215107 0.01790669
     0.06993018 0.01688336 0.09373438 0.03176755 0.00907719 0.16365649]



```python
final_model = grid_search.best_estimator_
test_set.sort_values('dayCount', axis= 0, inplace=True)
test_x_cols = (test_set.drop(['cnt'], axis=1)).columns.values
test_y_cols = 'cnt'

X_test = test_set.loc[:,test_x_cols]
y_test = test_set.loc[:,test_y_cols]
```

    /usr/local/anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      



```python
test_set.loc[:,'predictedCounts_test'] = final_model.predict(X_test)

mse = mean_squared_error(y_test, test_set.loc[:,'predictedCounts_test'])
final_mse = np.sqrt(mse)
print(final_mse)
test_set.describe()
```

    39.47930005837265


    /usr/local/anaconda/lib/python3.6/site-packages/pandas/core/indexing.py:376: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.obj[key] = _infer_fill_value(value)
    /usr/local/anaconda/lib/python3.6/site-packages/pandas/core/indexing.py:494: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.obj[item] = s





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
      <th>season</th>
      <th>yr</th>
      <th>mnth</th>
      <th>hr</th>
      <th>holiday</th>
      <th>weekday</th>
      <th>workingday</th>
      <th>weathersit</th>
      <th>temp</th>
      <th>hum</th>
      <th>windspeed</th>
      <th>cnt</th>
      <th>dayCount</th>
      <th>predictedCounts_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>5214.000000</td>
      <td>5214.000000</td>
      <td>5214.000000</td>
      <td>5214.000000</td>
      <td>5214.000000</td>
      <td>5214.000000</td>
      <td>5214.000000</td>
      <td>5214.000000</td>
      <td>5214.000000</td>
      <td>5214.000000</td>
      <td>5214.000000</td>
      <td>5214.000000</td>
      <td>5214.000000</td>
      <td>5214.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>2.481204</td>
      <td>0.504411</td>
      <td>6.480437</td>
      <td>11.471423</td>
      <td>0.030687</td>
      <td>3.001534</td>
      <td>0.685846</td>
      <td>1.422133</td>
      <td>-0.018492</td>
      <td>-0.004197</td>
      <td>0.017498</td>
      <td>185.322785</td>
      <td>360.850898</td>
      <td>186.725053</td>
    </tr>
    <tr>
      <td>std</td>
      <td>1.110062</td>
      <td>0.500028</td>
      <td>3.457991</td>
      <td>6.887845</td>
      <td>0.172484</td>
      <td>1.995486</td>
      <td>0.464223</td>
      <td>0.637995</td>
      <td>0.990859</td>
      <td>0.993451</td>
      <td>1.008611</td>
      <td>177.755171</td>
      <td>207.769276</td>
      <td>171.469555</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>-2.476000</td>
      <td>-3.245965</td>
      <td>-1.552670</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.526667</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>-0.818629</td>
      <td>-0.762886</td>
      <td>-0.696254</td>
      <td>41.000000</td>
      <td>180.781250</td>
      <td>50.300000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.010056</td>
      <td>0.013076</td>
      <td>0.037231</td>
      <td>136.000000</td>
      <td>362.375000</td>
      <td>143.943333</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>9.000000</td>
      <td>17.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>0.735156</td>
      <td>0.789038</td>
      <td>0.526495</td>
      <td>277.000000</td>
      <td>537.104167</td>
      <td>274.495000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>12.000000</td>
      <td>23.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>2.392526</td>
      <td>1.927116</td>
      <td>5.419128</td>
      <td>977.000000</td>
      <td>724.041667</td>
      <td>907.466667</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
