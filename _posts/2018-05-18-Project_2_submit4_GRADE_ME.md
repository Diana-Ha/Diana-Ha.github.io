
Note to self: Same as submission 2 but using Ridge


```python
# import everything for ease of use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import patsy

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn.linear_model import LinearRegression

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

%config InlineBackend.figure_format = 'retina'
%matplotlib inline

plt.style.use('fivethirtyeight')
sns.set_style('darkgrid')
```

    /Users/dianaha/anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
# load in data set
train = pd.read_csv('./train.csv')

test = pd.read_csv('./test.csv')
```

### Review and Clean Data_1

__Update NaN data from 'object' columns__


```python
# change column headers to lower case and add _ in spaces
train = train.rename(columns=lambda x: x.lower().replace(" ", "_"),)

test = test.rename(columns=lambda x: x.lower().replace(" ", "_"),)
```


```python
print(train.shape)

print(test.shape)
```

    (2051, 81)
    (879, 80)



```python
# drop 4 columns with too many null values
train.drop(['alley', 'pool_qc', 'fence', 'misc_feature'], axis=1, inplace=True)

test.drop(['alley', 'pool_qc', 'fence', 'misc_feature'], axis=1, inplace=True)
```


```python
# there are 1000 missing entries in 'fireplace_qu' that happen to coincide with having no fireplace
# change to None below
train['fireplace_qu'][(train['fireplaces']==0)].isnull().sum()
```




    1000




```python
test['fireplace_qu'][(test['fireplaces']==0)].isnull().sum()
```




    422




```python
# replace all NaN values with the string 'None'
train['fireplace_qu'].fillna(value='None', inplace=True)

test['fireplace_qu'].fillna(value='None', inplace=True)
```


```python
# replace 114 NaNs (113 for type) to 'None' as there is no garage
train['garage_cond'].fillna(value='None', inplace=True)
train['garage_finish'].fillna(value='None', inplace=True)
train['garage_qual'].fillna(value='None', inplace=True)
train['garage_type'].fillna(value='None', inplace=True)

test['garage_cond'].fillna(value='None', inplace=True)
test['garage_finish'].fillna(value='None', inplace=True)
test['garage_qual'].fillna(value='None', inplace=True)
test['garage_type'].fillna(value='None', inplace=True)
```


```python
# replace 55 NaNs to 'None' as there is no basement
train['bsmt_qual'].fillna(value='None', inplace=True)
train['bsmt_cond'].fillna(value='None', inplace=True)
train['bsmtfin_type_1'].fillna(value='None', inplace=True)

test['bsmt_qual'].fillna(value='None', inplace=True)
test['bsmt_cond'].fillna(value='None', inplace=True)
test['bsmtfin_type_1'].fillna(value='None', inplace=True)
```


```python
train[['bsmt_exposure', 'bsmtfin_type_2', 'bsmt_cond', 'bsmt_qual', 
      'bsmtfin_type_1', 'bsmt_full_bath', 'bsmt_half_bath', 'bsmt_unf_sf', 'bsmtfin_sf_1',
      'bsmtfin_sf_2', 'total_bsmt_sf']].iloc[1147]
```




    bsmt_exposure       No
    bsmtfin_type_2     NaN
    bsmt_cond           TA
    bsmt_qual           Gd
    bsmtfin_type_1     GLQ
    bsmt_full_bath       1
    bsmt_half_bath       0
    bsmt_unf_sf       1603
    bsmtfin_sf_1      1124
    bsmtfin_sf_2       479
    total_bsmt_sf     3206
    Name: 1147, dtype: object




```python
# replace 56 NaNs to 'None' as there is no basement for 55 and no type_2 for 1
train['bsmtfin_type_2'].fillna(value='None', inplace=True)

test['bsmtfin_type_2'].fillna(value='None', inplace=True)
```


```python
# replace 58 NaNs to 'None'; 55 NaNs are due to no basement, 3 NaNs are due to unfinished basements
train['bsmt_exposure'].fillna(value='None', inplace=True)

test['bsmt_exposure'].fillna(value='None', inplace=True)
```


```python
# temporarily replace 22 NaNs with "Missing" because these may be 'CBlock' but I am unsure
train['mas_vnr_type'].fillna(value='Missing', inplace=True)

test['mas_vnr_type'].fillna(value='Missing', inplace=True)
```


```python
train.shape
```




    (2051, 77)




```python
test.shape
```




    (879, 76)



__Split data into independent and target variables__


```python
target = 'saleprice'
y = train[target]
```


```python
# put predictor variables in their own group
X = train.drop(columns=target)
```

__Split data into test and train sets__


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```

__Find columns of interest__


```python
# adding back together to do correlation
X_train['y']= y_train
```

    /Users/dianaha/anaconda3/envs/dsi/lib/python3.6/site-packages/ipykernel/__main__.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      from ipykernel import kernelapp as app



```python
# find relevant numeric columns we care about
num_list= list(X_train.corr()[abs(X_train.corr()['y']) > 0.5].index)
```


```python
num_list
```




    ['overall_qual',
     'year_built',
     'year_remod/add',
     'total_bsmt_sf',
     '1st_flr_sf',
     'gr_liv_area',
     'full_bath',
     'garage_yr_blt',
     'garage_cars',
     'garage_area',
     'y']



Numeric columns to consider for continuous(not labeled) and categorical(labeled) variables:  

'overall_qual' - categorical  
'year_built' - discrete  
'year_remod/add' - discrete   
'total_bsmt_sf'  
'1st_flr_sf'  
'gr_liv_area'  
'full_bath' - discrete    
'garage_yr_blt' - discrete  
'garage_cars' - discrete  
'garage_area'  


```python
# put numeric columns we care about in a df
X_train_num_list = X_train[num_list].drop(['y'], axis=1)
```


```python
# create dummies for datatype 'objects'
X_train_str_dum = pd.get_dummies(X_train.select_dtypes(include=['object']))

test_dum = pd.get_dummies(test.select_dtypes(include=['object']))
```


```python
# add target to X to do correlation
X_train_str_dum['y'] = y_train
```


```python
# find list of moderately to strongly correlated columns with 'object' datatype
obj_list = list(X_train_str_dum.corr()[abs(X_train_str_dum.corr()['y']) >= 0.5].index)
```


```python
# object columns we care about in a df
X_train_obj_list = X_train_str_dum[obj_list].drop(['y'], axis=1)
```


```python
print(X_train_obj_list.columns)
print(X_train_num_list.columns)
```

    Index(['exter_qual_TA', 'foundation_PConc', 'bsmt_qual_Ex', 'kitchen_qual_Ex',
           'kitchen_qual_TA'],
          dtype='object')
    Index(['overall_qual', 'year_built', 'year_remod/add', 'total_bsmt_sf',
           '1st_flr_sf', 'gr_liv_area', 'full_bath', 'garage_yr_blt',
           'garage_cars', 'garage_area'],
          dtype='object')


__Fill in null values for non-'object' datatypes__  


```python
# add mean of column in null values of that column
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
```


```python
X_train_num_list[['total_bsmt_sf', '1st_flr_sf', 'gr_liv_area',
                  'garage_area']] = imp.fit_transform(X_train_num_list[
    ['total_bsmt_sf', '1st_flr_sf', 'gr_liv_area', 'garage_area']])



test[['total_bsmt_sf', '1st_flr_sf', 'gr_liv_area',
                  'garage_area']] = imp.fit_transform(test[['total_bsmt_sf', 
                    '1st_flr_sf', 'gr_liv_area', 'garage_area']])
```


```python
# add mode of column in null values of that column
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
```


```python
X_train_num_list[['overall_qual', 'year_built', 'year_remod/add', 'full_bath', 
                  'garage_yr_blt', 'garage_cars']] = imp.fit_transform(X_train_num_list[
    ['overall_qual', 'year_built', 'year_remod/add', 'full_bath', 'garage_yr_blt', 
     'garage_cars']])


test[['overall_qual', 'year_built', 'year_remod/add', 'full_bath', 
                  'garage_yr_blt', 'garage_cars']] = imp.fit_transform(test[
    ['overall_qual', 'year_built', 'year_remod/add', 'full_bath', 'garage_yr_blt', 
     'garage_cars']])
```


```python
# combine dfs so there is only one relevant X_train
X_train_all = pd.concat([X_train_num_list, X_train_obj_list], axis=1)

# combine dfs and drop columns so there is only one relevant test df
test_all = pd.concat([test, test_dum], axis=1)

test_all = test_all[['overall_qual', 'year_built', 'year_remod/add',
       'total_bsmt_sf', '1st_flr_sf', 'gr_liv_area', 'full_bath',
       'garage_yr_blt', 'garage_cars', 'garage_area', 'exter_qual_TA',
       'foundation_PConc', 'bsmt_qual_Ex', 'kitchen_qual_Ex',
       'kitchen_qual_TA']]
```


```python
X_train_all['y']= y_train
```


```python
X_train_all.corr()['y']
```




    overall_qual        0.800871
    year_built          0.569217
    year_remod/add      0.535336
    total_bsmt_sf       0.612144
    1st_flr_sf          0.605444
    gr_liv_area         0.682244
    full_bath           0.532711
    garage_yr_blt       0.458494
    garage_cars         0.644487
    garage_area         0.643931
    exter_qual_TA      -0.602455
    foundation_PConc    0.521373
    bsmt_qual_Ex        0.597145
    kitchen_qual_Ex     0.530172
    kitchen_qual_TA    -0.539865
    y                   1.000000
    Name: y, dtype: float64




```python
sns.pairplot(X_train_all)
```




    <seaborn.axisgrid.PairGrid at 0x1a31197908>




![png](/images/Project_2_submit4_GRADE_ME_files/Project_2_submit4_GRADE_ME_43_1.png)



```python
# categorical
sns.swarmplot(x="overall_qual", y="y", data=X_train_all)
```


![png](/images/Project_2_submit4_GRADE_ME_files/Project_2_submit4_GRADE_ME_44_0.png)



```python
sns.lmplot(x="total_bsmt_sf", y="y", hue='bsmt_qual_Ex', data=X_train_all)
```




    <seaborn.axisgrid.FacetGrid at 0x1a612962b0>




![png](/images/Project_2_submit4_GRADE_ME_files/Project_2_submit4_GRADE_ME_45_1.png)



```python
sns.lmplot(x="year_built", y="y", data=X_train_all)
```




    <seaborn.axisgrid.FacetGrid at 0x1a614930f0>




![png](/images/Project_2_submit4_GRADE_ME_files/Project_2_submit4_GRADE_ME_46_1.png)



```python
sns.lmplot(x='year_remod/add', y="y", data=X_train_all)
```




    <seaborn.axisgrid.FacetGrid at 0x1a61505f28>




![png](/images/Project_2_submit4_GRADE_ME_files/Project_2_submit4_GRADE_ME_47_1.png)



```python
sns.lmplot(x='1st_flr_sf', y="y", data=X_train_all)
```




    <seaborn.axisgrid.FacetGrid at 0x1a61deb908>




![png](/images/Project_2_submit4_GRADE_ME_files/Project_2_submit4_GRADE_ME_48_1.png)



```python
sns.lmplot(x='gr_liv_area', y="y", data=X_train_all)
```




    <seaborn.axisgrid.FacetGrid at 0x1a61de5e80>




![png](/images/Project_2_submit4_GRADE_ME_files/Project_2_submit4_GRADE_ME_49_1.png)



```python
sns.boxplot(x='full_bath', y="y", data=X_train_all, color='darkgreen')
```

    /Users/dianaha/anaconda3/envs/dsi/lib/python3.6/site-packages/seaborn/categorical.py:454: FutureWarning: remove_na is deprecated and is a private function. Do not use.
      box_data = remove_na(group_data)





    <matplotlib.axes._subplots.AxesSubplot at 0x1a63e75198>




![png](/images/Project_2_submit4_GRADE_ME_files/Project_2_submit4_GRADE_ME_50_2.png)



```python
sns.lmplot(x='garage_yr_blt', y="y", data=X_train_all)
```




    <seaborn.axisgrid.FacetGrid at 0x1a645c8278>




![png](/images/Project_2_submit4_GRADE_ME_files/Project_2_submit4_GRADE_ME_51_1.png)



```python
sns.lmplot(x='garage_area', y="y", data=X_train_all)
```




    <seaborn.axisgrid.FacetGrid at 0x1a64c03908>




![png](/images/Project_2_submit4_GRADE_ME_files/Project_2_submit4_GRADE_ME_52_1.png)



```python
sns.swarmplot(x='garage_cars', y="y", data=X_train_all)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a652779e8>




![png](/images/Project_2_submit4_GRADE_ME_files/Project_2_submit4_GRADE_ME_53_1.png)



```python
X_train_all = X_train_all.drop(['y'], axis=1)
```


```python
print(X_train_all.columns)

print(test_all.columns)
```

    Index(['overall_qual', 'year_built', 'year_remod/add', 'total_bsmt_sf',
           '1st_flr_sf', 'gr_liv_area', 'full_bath', 'garage_yr_blt',
           'garage_cars', 'garage_area', 'exter_qual_TA', 'foundation_PConc',
           'bsmt_qual_Ex', 'kitchen_qual_Ex', 'kitchen_qual_TA'],
          dtype='object')
    Index(['overall_qual', 'year_built', 'year_remod/add', 'total_bsmt_sf',
           '1st_flr_sf', 'gr_liv_area', 'full_bath', 'garage_yr_blt',
           'garage_cars', 'garage_area', 'exter_qual_TA', 'foundation_PConc',
           'bsmt_qual_Ex', 'kitchen_qual_Ex', 'kitchen_qual_TA'],
          dtype='object')



```python
print(X_train_all.shape)

print(test_all.shape)
```

    (1538, 15)
    (879, 15)


### Prep X_test data


```python
X_test_a = X_test[['overall_qual', 'year_built', 'year_remod/add',
       'total_bsmt_sf', '1st_flr_sf', 'gr_liv_area', 'full_bath',
        'garage_yr_blt', 'garage_cars', 'garage_area']]
```


```python
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X_test_a[['total_bsmt_sf', '1st_flr_sf', 'gr_liv_area',
                  'garage_area']] = imp.fit_transform(
    X_test_a[['total_bsmt_sf', '1st_flr_sf', 'gr_liv_area', 'garage_area']])
```

    /Users/dianaha/anaconda3/envs/dsi/lib/python3.6/site-packages/ipykernel/__main__.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    /Users/dianaha/anaconda3/envs/dsi/lib/python3.6/site-packages/pandas/core/indexing.py:537: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self.obj[item] = s



```python
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
X_test_a[['overall_qual', 'year_built', 'year_remod/add', 'full_bath', 
                  'garage_yr_blt', 'garage_cars']] = imp.fit_transform(X_test_a[
    ['overall_qual', 'year_built', 'year_remod/add', 'full_bath', 'garage_yr_blt', 
     'garage_cars']])
```

    /Users/dianaha/anaconda3/envs/dsi/lib/python3.6/site-packages/ipykernel/__main__.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    /Users/dianaha/anaconda3/envs/dsi/lib/python3.6/site-packages/pandas/core/indexing.py:537: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self.obj[item] = s



```python
X_test_b = X_test[['exter_qual', 'foundation', 'bsmt_qual',
                       'kitchen_qual']]
```


```python
X_test_dum = pd.get_dummies(X_test_b)
```


```python
X_test_dum = X_test_dum[['exter_qual_TA', 'foundation_PConc', 'bsmt_qual_Ex', 'kitchen_qual_Ex', 
            'kitchen_qual_TA']]
```


```python
# combine dfs so there is only one relevant X for test data from train/test split
X_test_all = pd.concat([X_test_a, X_test_dum], axis=1)
```

### Standardize the data_1


```python
ss = StandardScaler()
X_train_all_new = ss.fit_transform(X_train_all)
```


```python
ss = StandardScaler()
test_all_new = ss.fit_transform(test_all)
```

### Train the data_1
Ridge


```python
# start with linear regression
lr_model = LinearRegression()
lr_cv_mean_mse = -cross_val_score(lr_model, X_train_all_new, y_train, cv=5, scoring='neg_mean_squared_error').mean()
lr_cv_mean_r2 = cross_val_score(lr_model, X_train_all_new, y_train, cv=5, scoring='r2').mean()
lr_cv_mean_mse, lr_cv_mean_r2
```




    (1180943568.4552472, 0.805004217712747)




```python
# use ridge to develop model
alpha = 10.0
ridge_model = Ridge(alpha=alpha)
ridge_cv_mean_mse = -cross_val_score(ridge_model, X_train_all_new, y_train, cv=5, scoring='neg_mean_squared_error').mean()
ridge_cv_mean_r2 = cross_val_score(ridge_model, X_train_all_new, y_train, cv=5, scoring='r2').mean()
ridge_cv_mean_mse, ridge_cv_mean_r2
```




    (1178238019.627359, 0.805450017532136)




```python
# fit model
r_alphas = np.logspace(0, 5, 200)
ridge_model = RidgeCV(alphas=r_alphas, store_cv_values=True)
ridge_model = ridge_model.fit(X_train_all_new, y_train)
```


```python
# find optimal value of alpha
ridge_optimal_alpha = ridge_model.alpha_
ridge_optimal_alpha
```




    144.81182276745346



### Evaluate the model_1


```python
def plot_cv(alphas, cv_means, optimal_alpha, lr_mse, log=False):
    # alphas = list of alphas
    # cv_means = list of CV mean MSE
    # optimal_alpha
    # lr_mse
    fig = plt.figure(figsize=(12,8))
    ax = plt.gca()

    if log:
        ax.semilogx(alphas, cv_means, lw=2)
    else:
        ax.plot(alphas, cv_means, lw=2)
    ax.axvline(optimal_alpha)
    ax.axhline(lr_mse)
    ax.set_xlabel('alpha')
    ax.set_ylabel('Mean Squared Error')
```


```python
# average the CV scores for each value of alpha
ridge_cv_means = [np.mean(cv_alpha) for cv_alpha in ridge_model.cv_values_.T]
```


```python
plot_cv(ridge_model.alphas, ridge_cv_means, ridge_optimal_alpha, lr_cv_mean_mse, log=True)
```


![png](/images/Project_2_submit4_GRADE_ME_files/Project_2_submit4_GRADE_ME_76_0.png)



```python
# get average of R2's
ridge_opt = Ridge(alpha=ridge_optimal_alpha)
cross_val_score(ridge_opt, X_train_all_new, y_train, cv=5).mean()
```




    0.8067325464349666



### Get predictions for test.csv data


```python
final_predictions = ridge_model.fit(X_train_all_new, y_train).predict(test_all)
```


```python
final_predictions
```




    array([6.61318853e+07, 7.89337347e+07, 5.58799052e+07, 4.96432454e+07,
           6.20688541e+07, 4.63746633e+07, 5.13386006e+07, 5.73252046e+07,
           6.19033182e+07, 5.82636368e+07, 5.71317467e+07, 5.08626907e+07,
           6.10803971e+07, 8.62589678e+07, 6.32992349e+07, 5.21860950e+07,
           5.86318997e+07, 5.19132637e+07, 7.01079335e+07, 6.22788767e+07,
           5.14578632e+07, 4.79214839e+07, 6.87908187e+07, 5.03564852e+07,
           5.77930228e+07, 4.61371621e+07, 6.31041559e+07, 6.09197728e+07,
           5.15312554e+07, 3.56647063e+07, 4.75876387e+07, 5.22630110e+07,
           8.44299144e+07, 5.56965589e+07, 6.57445303e+07, 5.47875467e+07,
           6.20943218e+07, 4.49277657e+07, 4.34776811e+07, 6.18725999e+07,
           5.10552638e+07, 6.58512377e+07, 5.94123572e+07, 5.82341748e+07,
           6.53876745e+07, 5.09581225e+07, 6.47313018e+07, 4.79724837e+07,
           4.80199164e+07, 5.08774955e+07, 5.21949747e+07, 7.09976686e+07,
           7.06242997e+07, 5.39215692e+07, 4.82084717e+07, 5.13758837e+07,
           6.46017737e+07, 5.88079507e+07, 5.63164569e+07, 6.80576623e+07,
           8.19785256e+07, 5.47695194e+07, 5.47778467e+07, 6.07249229e+07,
           5.92940057e+07, 8.28856885e+07, 4.41777410e+07, 6.04386521e+07,
           4.04486982e+07, 4.63679838e+07, 4.25233945e+07, 8.44291382e+07,
           6.71236236e+07, 5.45151863e+07, 5.13442241e+07, 7.11986056e+07,
           1.41725401e+08, 6.26705525e+07, 5.51682688e+07, 4.15264508e+07,
           7.51274242e+07, 6.44064712e+07, 4.95388365e+07, 5.27992378e+07,
           5.53453030e+07, 5.95859055e+07, 8.55054780e+07, 4.71455582e+07,
           5.94628086e+07, 7.11295053e+07, 5.95434235e+07, 4.96471196e+07,
           5.26060086e+07, 5.20098765e+07, 5.74004460e+07, 5.12334209e+07,
           5.92886312e+07, 6.43202420e+07, 6.76835960e+07, 6.18534778e+07,
           6.10016273e+07, 7.16272665e+07, 6.92522980e+07, 4.72658071e+07,
           9.51157902e+07, 5.68219955e+07, 4.34314307e+07, 5.62057676e+07,
           7.03509044e+07, 7.74873107e+07, 7.20499602e+07, 6.33930542e+07,
           6.77078498e+07, 5.37384152e+07, 5.26299621e+07, 6.52413336e+07,
           1.08876207e+08, 6.24925493e+07, 5.17888974e+07, 7.35921828e+07,
           6.08238400e+07, 8.18720063e+07, 7.07890436e+07, 7.17424929e+07,
           5.65853595e+07, 7.04966468e+07, 8.18519576e+07, 4.58235857e+07,
           4.97584728e+07, 4.89685382e+07, 6.88080384e+07, 5.08354836e+07,
           6.16189655e+07, 5.53981945e+07, 5.34117424e+07, 5.77926827e+07,
           7.39526680e+07, 6.22642157e+07, 5.47877438e+07, 6.04386521e+07,
           4.44370299e+07, 6.15645957e+07, 5.17043343e+07, 5.47851556e+07,
           6.50033728e+07, 4.49150435e+07, 6.66911627e+07, 6.49211511e+07,
           5.34107698e+07, 4.12157132e+07, 7.23153701e+07, 6.27171479e+07,
           6.07660054e+07, 5.15840944e+07, 4.66135533e+07, 4.32862488e+07,
           5.22224108e+07, 4.49197150e+07, 4.80786035e+07, 7.27558313e+07,
           7.61644554e+07, 6.51383269e+07, 5.04602124e+07, 5.12306224e+07,
           1.00284013e+08, 6.44703147e+07, 6.86039178e+07, 5.47548875e+07,
           5.50016649e+07, 7.43171635e+07, 6.37418455e+07, 1.29703999e+08,
           5.06126667e+07, 6.80211115e+07, 5.02035816e+07, 6.54237498e+07,
           6.35191230e+07, 4.01406958e+07, 7.84477824e+07, 4.63550274e+07,
           5.67154233e+07, 5.86118613e+07, 6.77156954e+07, 7.56690948e+07,
           5.67386843e+07, 5.92089314e+07, 4.99729022e+07, 5.70703672e+07,
           7.01883714e+07, 4.96095570e+07, 4.40353416e+07, 6.09515763e+07,
           4.79804528e+07, 4.47342038e+07, 4.56756203e+07, 5.68335937e+07,
           4.80585195e+07, 6.81098557e+07, 5.73730719e+07, 5.80567085e+07,
           5.11695026e+07, 8.65411846e+07, 7.23446406e+07, 4.78906764e+07,
           6.18142065e+07, 5.33898289e+07, 4.57332169e+07, 5.71529026e+07,
           5.04177150e+07, 6.17011590e+07, 6.73965056e+07, 6.27929140e+07,
           6.57283318e+07, 5.38082786e+07, 8.10330072e+07, 6.70223076e+07,
           6.23908316e+07, 7.48418318e+07, 5.28755922e+07, 4.46081084e+07,
           5.09236787e+07, 5.58975886e+07, 5.28584492e+07, 6.34844524e+07,
           5.71293714e+07, 4.58944599e+07, 4.56788070e+07, 7.69359235e+07,
           6.86793474e+07, 5.75192278e+07, 7.98112987e+07, 6.34221246e+07,
           3.76549246e+07, 6.32890903e+07, 4.32186501e+07, 9.35069549e+07,
           5.60484311e+07, 4.45400153e+07, 9.25100562e+07, 8.56731789e+07,
           6.10927118e+07, 6.92719406e+07, 5.89548647e+07, 7.86000034e+07,
           7.61979076e+07, 8.15526361e+07, 7.16428434e+07, 4.72643081e+07,
           4.70814513e+07, 6.50335962e+07, 7.67083487e+07, 6.87886725e+07,
           6.07551404e+07, 7.04809871e+07, 7.01656487e+07, 6.30353064e+07,
           4.91216766e+07, 4.71029251e+07, 9.21594857e+07, 6.73665078e+07,
           7.82234055e+07, 5.25667076e+07, 5.66273972e+07, 4.99237190e+07,
           5.32803965e+07, 6.75736928e+07, 6.37273418e+07, 5.85999766e+07,
           6.30598110e+07, 6.03024625e+07, 5.05077821e+07, 7.61093465e+07,
           5.94873200e+07, 6.21559559e+07, 1.00654595e+08, 4.72133401e+07,
           5.50914977e+07, 4.63581659e+07, 6.32725957e+07, 4.59387899e+07,
           5.21749183e+07, 6.57612272e+07, 5.56914107e+07, 4.77268300e+07,
           4.47332106e+07, 5.19444087e+07, 6.57050917e+07, 7.43514766e+07,
           4.37627767e+07, 4.35719172e+07, 5.52191813e+07, 4.63359125e+07,
           5.88392649e+07, 4.64139350e+07, 4.25780960e+07, 6.23136291e+07,
           4.63679838e+07, 5.99208265e+07, 7.15162347e+07, 6.35616972e+07,
           5.94549381e+07, 6.72973509e+07, 7.73112484e+07, 5.01735471e+07,
           6.85854451e+07, 3.98899748e+07, 4.83166446e+07, 7.09141855e+07,
           6.03457738e+07, 5.39613092e+07, 5.04386831e+07, 7.56144019e+07,
           6.12219393e+07, 6.84440159e+07, 7.67636386e+07, 7.13829543e+07,
           7.74484808e+07, 7.86642132e+07, 7.06858043e+07, 7.35485809e+07,
           5.17569780e+07, 3.92872898e+07, 5.52423263e+07, 6.20146658e+07,
           6.86337816e+07, 4.82354560e+07, 4.93367573e+07, 5.91708334e+07,
           5.49251313e+07, 6.64766052e+07, 6.08039439e+07, 7.54314933e+07,
           6.14528265e+07, 6.56016748e+07, 5.75730732e+07, 4.23372645e+07,
           6.15999604e+07, 4.24878844e+07, 5.71941889e+07, 5.16236974e+07,
           4.44480909e+07, 6.83715945e+07, 5.35897999e+07, 5.35619213e+07,
           6.11415283e+07, 7.81727947e+07, 6.63569550e+07, 6.25747550e+07,
           4.62980701e+07, 5.69747299e+07, 5.27030523e+07, 6.33560922e+07,
           5.42677187e+07, 4.99139011e+07, 7.01014930e+07, 5.85843896e+07,
           8.98197342e+07, 5.79253782e+07, 6.42651466e+07, 7.13034616e+07,
           5.61732692e+07, 5.82288160e+07, 6.66558458e+07, 4.95764185e+07,
           7.37812332e+07, 5.78256534e+07, 6.90022537e+07, 6.95449764e+07,
           6.88120311e+07, 6.89895800e+07, 5.40597131e+07, 8.29075512e+07,
           4.41892943e+07, 6.42053170e+07, 9.13404131e+07, 4.64140470e+07,
           7.42532435e+07, 5.58263380e+07, 6.93176658e+07, 6.07966955e+07,
           5.11410703e+07, 5.98591982e+07, 6.45934266e+07, 7.59517567e+07,
           8.09664566e+07, 6.63024416e+07, 6.63915145e+07, 7.95681755e+07,
           5.59870591e+07, 5.51281922e+07, 5.45628357e+07, 4.63172924e+07,
           7.34180309e+07, 5.82805104e+07, 5.92905588e+07, 5.88589648e+07,
           4.15072324e+07, 6.02876996e+07, 5.75245261e+07, 4.71859622e+07,
           5.11158410e+07, 6.24676523e+07, 7.35543421e+07, 5.24891725e+07,
           7.37243576e+07, 6.23969613e+07, 5.04504853e+07, 6.10413745e+07,
           6.57636554e+07, 6.90290387e+07, 5.08771484e+07, 5.40857768e+07,
           5.98955611e+07, 8.35764930e+07, 6.22075883e+07, 7.31001332e+07,
           8.33351736e+07, 5.13708974e+07, 8.16426226e+07, 4.72264554e+07,
           7.01717791e+07, 8.56165963e+07, 5.12176643e+07, 9.11514384e+07,
           5.03196040e+07, 6.83259881e+07, 8.24099484e+07, 5.73286827e+07,
           6.24017757e+07, 4.74433310e+07, 7.10837544e+07, 4.84497671e+07,
           5.13839793e+07, 4.78374988e+07, 5.96772691e+07, 5.24907687e+07,
           7.90921114e+07, 6.15695914e+07, 5.63312455e+07, 4.74064479e+07,
           6.23685920e+07, 6.41249076e+07, 5.59526064e+07, 4.82406268e+07,
           5.70397105e+07, 5.35974049e+07, 6.40200893e+07, 4.83554380e+07,
           4.21812541e+07, 8.17004188e+07, 4.73922408e+07, 6.40964278e+07,
           7.30177721e+07, 6.13717493e+07, 6.39436233e+07, 5.88814615e+07,
           6.55530048e+07, 6.30788250e+07, 4.98935131e+07, 6.64156432e+07,
           4.91449130e+07, 5.00226384e+07, 5.14344916e+07, 5.29763675e+07,
           1.27382965e+08, 5.58392532e+07, 6.61714557e+07, 6.67127394e+07,
           7.50521687e+07, 4.25894036e+07, 4.44272120e+07, 4.73518383e+07,
           7.05662042e+07, 5.65362313e+07, 4.64983921e+07, 7.33725094e+07,
           6.77270695e+07, 5.04413353e+07, 5.91683946e+07, 4.93365254e+07,
           5.96780592e+07, 4.06054666e+07, 7.34812318e+07, 3.76658678e+07,
           5.71115298e+07, 9.41271693e+07, 5.40444293e+07, 4.63679838e+07,
           5.66943961e+07, 5.86020031e+07, 5.40185961e+07, 7.63662011e+07,
           6.62983274e+07, 7.36590812e+07, 7.12630340e+07, 7.78396007e+07,
           6.22874886e+07, 5.71270140e+07, 5.41175896e+07, 6.78606862e+07,
           5.44483614e+07, 5.63630059e+07, 7.70782906e+07, 5.27231656e+07,
           5.68641104e+07, 7.53029419e+07, 4.62420735e+07, 8.33431391e+07,
           4.74888368e+07, 4.93362078e+07, 7.17747373e+07, 6.14991117e+07,
           5.44115953e+07, 5.30848110e+07, 4.90438832e+07, 3.24778542e+07,
           6.51960335e+07, 5.91667504e+07, 5.99457137e+07, 7.10210664e+07,
           7.92115409e+07, 4.91304497e+07, 6.03987251e+07, 6.32254924e+07,
           4.95760722e+07, 6.46748533e+07, 9.35435987e+07, 5.18747914e+07,
           5.83357744e+07, 5.52811752e+07, 3.54793281e+07, 4.53819239e+07,
           6.03550319e+07, 5.35810831e+07, 5.81315996e+07, 6.93652297e+07,
           5.49752517e+07, 4.49197150e+07, 7.17630760e+07, 5.28526796e+07,
           6.08023972e+07, 5.17559806e+07, 4.57445906e+07, 3.90700939e+07,
           4.89317460e+07, 6.87367417e+07, 8.05649737e+07, 7.21394901e+07,
           4.50422028e+07, 4.38721669e+07, 6.98938842e+07, 5.29790367e+07,
           7.43891463e+07, 3.92407470e+07, 7.18655704e+07, 6.36607415e+07,
           4.24569074e+07, 5.08122843e+07, 5.81305838e+07, 4.88371038e+07,
           6.30181261e+07, 4.82356548e+07, 5.21343137e+07, 5.68676963e+07,
           5.51025928e+07, 5.25270602e+07, 4.80939070e+07, 5.18685093e+07,
           7.22654967e+07, 4.59486747e+07, 4.90202209e+07, 6.18253440e+07,
           4.74432765e+07, 4.61796171e+07, 4.99014922e+07, 6.57645767e+07,
           4.66840586e+07, 5.84715558e+07, 4.63679838e+07, 8.20298759e+07,
           4.94890159e+07, 6.38906936e+07, 5.89197741e+07, 8.16614148e+07,
           4.94689995e+07, 5.90628705e+07, 5.78065857e+07, 5.53887871e+07,
           5.38850690e+07, 7.53350984e+07, 6.56947441e+07, 5.07395982e+07,
           4.24203698e+07, 5.45787617e+07, 5.76220509e+07, 7.65512755e+07,
           7.14496545e+07, 5.46953639e+07, 4.72657468e+07, 4.55021506e+07,
           7.56121600e+07, 5.67469351e+07, 6.64796635e+07, 4.73032922e+07,
           5.90978299e+07, 6.54575903e+07, 4.27778396e+07, 4.98384131e+07,
           4.74836400e+07, 5.70420737e+07, 5.21759194e+07, 4.46615045e+07,
           9.14715684e+07, 6.33370048e+07, 4.99979398e+07, 5.43719897e+07,
           3.93281256e+07, 6.03951741e+07, 7.43717058e+07, 4.62868020e+07,
           5.36805727e+07, 5.84728614e+07, 5.75249643e+07, 6.73190551e+07,
           4.66281252e+07, 5.48063884e+07, 6.99020357e+07, 5.70207984e+07,
           5.23266395e+07, 9.34243853e+07, 5.23891403e+07, 7.22925957e+07,
           7.82791580e+07, 4.60719359e+07, 4.49290523e+07, 5.29588546e+07,
           5.72119162e+07, 6.27918389e+07, 6.34260350e+07, 5.66985952e+07,
           5.89087850e+07, 7.10489741e+07, 4.68946373e+07, 6.46911401e+07,
           6.86582522e+07, 5.24368916e+07, 3.89861579e+07, 7.57165649e+07,
           5.50815624e+07, 5.05034398e+07, 6.16058965e+07, 5.55962635e+07,
           4.51505256e+07, 5.03926693e+07, 5.21175175e+07, 3.75823335e+07,
           5.80196098e+07, 6.56113715e+07, 6.59384094e+07, 7.36023569e+07,
           6.34387189e+07, 5.20926251e+07, 6.67306154e+07, 7.20219538e+07,
           7.50618478e+07, 9.32572793e+07, 7.17962404e+07, 8.21411819e+07,
           6.62684691e+07, 5.76135479e+07, 8.17498080e+07, 6.22408273e+07,
           9.18528867e+07, 6.43347413e+07, 6.32089324e+07, 4.73078867e+07,
           5.32209336e+07, 6.20075233e+07, 4.63844812e+07, 5.79554932e+07,
           4.37343669e+07, 8.92356971e+07, 6.58416039e+07, 6.17693522e+07,
           6.92410278e+07, 6.43746459e+07, 6.00320697e+07, 6.94156076e+07,
           8.04885368e+07, 5.32594458e+07, 6.61969126e+07, 4.39723121e+07,
           4.72778008e+07, 6.59999765e+07, 4.98024243e+07, 6.36657052e+07,
           5.58388647e+07, 6.25463478e+07, 8.25390805e+07, 4.44095107e+07,
           7.02409059e+07, 4.63679838e+07, 5.46270817e+07, 6.15536530e+07,
           4.87530592e+07, 8.82335616e+07, 8.27087932e+07, 5.26318962e+07,
           5.70045832e+07, 5.15889307e+07, 7.23561900e+07, 6.74170024e+07,
           5.32604639e+07, 5.48081959e+07, 5.24962690e+07, 5.05041933e+07,
           7.51761420e+07, 8.10412701e+07, 6.38780911e+07, 6.55596297e+07,
           7.59004396e+07, 6.35535064e+07, 7.00868603e+07, 4.95428372e+07,
           5.09284689e+07, 7.56408834e+07, 6.33272481e+07, 6.27284298e+07,
           5.00842329e+07, 7.07602770e+07, 4.79011491e+07, 6.03889580e+07,
           6.43360299e+07, 7.32552431e+07, 5.08269280e+07, 7.48117968e+07,
           5.50963756e+07, 6.01876479e+07, 5.48416344e+07, 5.16059259e+07,
           5.56144927e+07, 6.54242278e+07, 6.54636325e+07, 6.35602893e+07,
           5.42407387e+07, 5.54660704e+07, 7.38281520e+07, 6.21639120e+07,
           7.84550641e+07, 5.12310228e+07, 8.16783219e+07, 7.34914486e+07,
           6.35015062e+07, 5.70731983e+07, 6.24906667e+07, 5.26817833e+07,
           4.99356895e+07, 9.78488378e+07, 4.46207983e+07, 5.53961148e+07,
           6.07109425e+07, 4.78528987e+07, 7.16209484e+07, 5.97108040e+07,
           7.25473169e+07, 6.98872291e+07, 7.85204911e+07, 5.81329956e+07,
           5.68693173e+07, 8.00483811e+07, 5.60574975e+07, 7.01510644e+07,
           5.62697144e+07, 7.08100434e+07, 7.81576333e+07, 5.47664979e+07,
           6.85397551e+07, 6.06491424e+07, 4.86478477e+07, 5.89047897e+07,
           7.93863596e+07, 5.72119162e+07, 8.81277453e+07, 4.78523423e+07,
           4.37434359e+07, 5.10221415e+07, 6.08460508e+07, 7.16428434e+07,
           4.59793966e+07, 7.75795702e+07, 6.23009102e+07, 5.52958968e+07,
           7.18523740e+07, 5.89909566e+07, 7.89336756e+07, 6.79640479e+07,
           5.85229449e+07, 5.15259366e+07, 6.62884161e+07, 6.92610421e+07,
           6.66002798e+07, 5.76992960e+07, 4.90548963e+07, 5.37204249e+07,
           6.15914475e+07, 7.58563974e+07, 4.92815304e+07, 4.85440375e+07,
           5.57324722e+07, 5.16588707e+07, 4.73593981e+07, 5.38169789e+07,
           8.23407732e+07, 5.38355056e+07, 6.92042309e+07, 6.13093465e+07,
           6.19882578e+07, 7.37731024e+07, 4.84090330e+07, 4.92016827e+07,
           5.44525871e+07, 6.19863058e+07, 4.71011011e+07, 9.25814153e+07,
           6.76489306e+07, 5.45686439e+07, 6.51101751e+07, 9.30805438e+07,
           5.30417887e+07, 7.20651595e+07, 5.73181951e+07, 5.47975091e+07,
           6.06156705e+07, 7.49624363e+07, 6.12307855e+07, 5.24193618e+07,
           4.96508684e+07, 7.60641398e+07, 5.28743623e+07, 4.89196088e+07,
           5.10035842e+07, 7.14599693e+07, 5.95444862e+07, 6.42417916e+07,
           4.83143458e+07, 5.26215538e+07, 5.93512826e+07, 8.44526740e+07,
           6.61116894e+07, 4.59297265e+07, 5.82611326e+07, 5.64742542e+07,
           6.04959699e+07, 4.57187372e+07, 4.97114767e+07, 6.36227247e+07,
           5.87256556e+07, 6.44434351e+07, 3.92127155e+07, 6.12433975e+07,
           4.36215334e+07, 7.90236628e+07, 4.34953654e+07, 5.43784728e+07,
           6.03373467e+07, 5.75452961e+07, 6.24937098e+07, 6.54717106e+07,
           7.27454607e+07, 7.92631935e+07, 6.70554855e+07, 6.69779443e+07,
           8.28460448e+07, 6.25286370e+07, 6.72419848e+07, 6.94629499e+07,
           5.45112086e+07, 4.69922932e+07, 4.79004578e+07])



### Save Data


```python
submission = pd.DataFrame(data=final_predictions, columns=['SalePrice'], index=test['id'])
```


```python
submission = submission.reset_index()
submission = submission.rename(columns={'id': 'Id'})
```


```python
submission.head(1)
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
      <th>Id</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2658</td>
      <td>6.613189e+07</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission.to_csv('./DianaHa_submit4.csv', index=False, index_label='Id')
```

### Analyze Results


```python
X_test_all.columns
```




    Index(['overall_qual', 'year_built', 'year_remod/add', 'total_bsmt_sf',
           '1st_flr_sf', 'gr_liv_area', 'full_bath', 'garage_yr_blt',
           'garage_cars', 'garage_area', 'exter_qual_TA', 'foundation_PConc',
           'bsmt_qual_Ex', 'kitchen_qual_Ex', 'kitchen_qual_TA'],
          dtype='object')




```python
# predictions based on test data from train/test split
y_hat = ridge_model.fit(X_train_all_new, y_train).predict(X_test_all)
```


```python
sns.residplot(y_test, y_hat, lowess=True)

# LOWESS (Locally Weighted Scatterplot Smoothing), sometimes called 
# LOESS (locally weighted smoothing), is a popular tool used in regression analysis 
# that creates a smooth line through a timeplot or scatter plot to help you to see 
# relationship between variables and foresee trends
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a32cb6be0>




![png](/images/Project_2_submit4_GRADE_ME_files/Project_2_submit4_GRADE_ME_89_1.png)



```python
# average error
metrics.mean_absolute_error(y_test, y_hat)
```




    60682703.338840164




```python
# there seems to be some large errors impacting this model, points to possible outliers
metrics.mean_squared_error(y_test, y_hat)
```




    3837444132501607.0




```python
np.sqrt(metrics.mean_squared_error(y_test, y_hat))
```




    61947107.53942921




```python
# plot y and predicted y values with linear reg fit - goal is straight line
sns.regplot(y_test, y_hat)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a3337d198>




![png](/images/Project_2_submit4_GRADE_ME_files/Project_2_submit4_GRADE_ME_93_1.png)

