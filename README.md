# Capstone project scenario overview: Working for Salifort Motors     
## About the company
Salifort Motors is a fictional French-based alternative energy vehicle manufacturer. Its global workforce of over 100,000 employees research, design, construct, validate, and distribute electric, solar, algae, and hydrogen-based vehicles. Salifort’s end-to-end vertical integration model has made it a global leader at the intersection of alternative energy and automobiles.        

## Your business case
As a data specialist working for Salifort Motors, you have received the results of a recent employee survey. The senior leadership team has tasked you with analyzing the data to come up with ideas for how to increase employee retention. To help with this, they would like you to design a model that predicts whether an employee will leave the company based on their  department, number of projects, average monthly hours, and any other data points you deem helpful. 

## The value of your deliverable
For this deliverable, you are asked to choose a method to approach this data challenge based on your prior course work. Select either a regression model or a tree-based machine learning model to predict whether an employee will leave the company. Both approaches are shown in the project exemplar, but only one is needed to complete your project.

<h2>Data Source</h2>
https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction/code?datasetId=1409252&sortBy=voteCount

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
```


```python
df0 = pd.read_csv("HR_comma_sep.csv")
df0.head()
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
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>Department</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>




```python
df0.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14999 entries, 0 to 14998
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   satisfaction_level     14999 non-null  float64
     1   last_evaluation        14999 non-null  float64
     2   number_project         14999 non-null  int64  
     3   average_montly_hours   14999 non-null  int64  
     4   time_spend_company     14999 non-null  int64  
     5   Work_accident          14999 non-null  int64  
     6   left                   14999 non-null  int64  
     7   promotion_last_5years  14999 non-null  int64  
     8   Department             14999 non-null  object 
     9   salary                 14999 non-null  object 
    dtypes: float64(2), int64(6), object(2)
    memory usage: 1.1+ MB
    


```python
df0.duplicated().sum()
```




    3008




```python
df0.drop_duplicates(inplace=True)
df0.duplicated().any()
```




    False




```python
df0['Department'] = df0['Department'].astype('category')
df0['salary'] = df0['salary'].astype('category')
df0.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 11991 entries, 0 to 11999
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype   
    ---  ------                 --------------  -----   
     0   satisfaction_level     11991 non-null  float64 
     1   last_evaluation        11991 non-null  float64 
     2   number_project         11991 non-null  int64   
     3   average_montly_hours   11991 non-null  int64   
     4   time_spend_company     11991 non-null  int64   
     5   Work_accident          11991 non-null  int64   
     6   left                   11991 non-null  int64   
     7   promotion_last_5years  11991 non-null  int64   
     8   Department             11991 non-null  category
     9   salary                 11991 non-null  category
    dtypes: category(2), float64(2), int64(6)
    memory usage: 867.0 KB
    


```python
df0.describe()
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
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11991.000000</td>
      <td>11991.000000</td>
      <td>11991.000000</td>
      <td>11991.000000</td>
      <td>11991.000000</td>
      <td>11991.000000</td>
      <td>11991.000000</td>
      <td>11991.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.629658</td>
      <td>0.716683</td>
      <td>3.802852</td>
      <td>200.473522</td>
      <td>3.364857</td>
      <td>0.154282</td>
      <td>0.166041</td>
      <td>0.016929</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.241070</td>
      <td>0.168343</td>
      <td>1.163238</td>
      <td>48.727813</td>
      <td>1.330240</td>
      <td>0.361234</td>
      <td>0.372133</td>
      <td>0.129012</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.090000</td>
      <td>0.360000</td>
      <td>2.000000</td>
      <td>96.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.480000</td>
      <td>0.570000</td>
      <td>3.000000</td>
      <td>157.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.660000</td>
      <td>0.720000</td>
      <td>4.000000</td>
      <td>200.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.820000</td>
      <td>0.860000</td>
      <td>5.000000</td>
      <td>243.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>310.000000</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
for_plotting = df0.drop(['salary', 'Department','promotion_last_5years', 'left','Work_accident'], axis=1)

num_plots = len(for_plotting)

plt.figure(figsize=(12,8))

for i,col in enumerate(for_plotting):
    plt.subplot(2, 3, i+1)
    plt.title(f'Boxplot for {col}', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    sns.boxplot(x=df0[col])
    
plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.show()
```


    
![png](output_7_0.png)
    



```python
labels = ['Did Not Leave', 'Left']
df0.groupby('left').size().plot(kind='pie', labels=labels, autopct='%1.1f%%', ylabel='', colors=['#51BAFF', '#00FFA2'])

plt.show()
```


    
![png](output_8_0.png)
    



```python
from sklearn.utils import resample

left_true = df0[df0['left'] == 1]
left_false  = df0[df0['left'] == 0]
print(left_true.shape)
print(left_false.shape)

left_downsample = resample(left_false,
             replace=True,
             n_samples=len(left_true),
             random_state=42)

print(left_downsample.shape)
```

    (1991, 10)
    (10000, 10)
    (1991, 10)
    


```python
df1 = pd.concat([left_downsample, left_true])

labels = ['Did Not Leave', 'Left']
df1.groupby('left').size().plot(kind='pie', labels=labels, autopct='%1.1f%%', ylabel='', colors=['#51BAFF', '#00FFA2'])

plt.show()
```


    
![png](output_10_0.png)
    



```python
plt.figure(figsize=(24, 8))

plt.subplot(1, 2, 1)
sns.histplot(data=df1, x='salary', shrink=.8, stat='percent')
plt.title('Distribution of Salary Levels (Downsampled Data)')
plt.xlabel('Salary')
plt.ylabel('Percent')

plt.subplot(1, 2, 2)
sns.histplot(data=df1, x='salary', hue='left', multiple='dodge', shrink=.8, stat='percent', palette='viridis')
plt.title('Distribution of Salary Levels for Employees Who Left or \n Did Not Left (Downsampled Data)')
plt.legend(labels=['left', 'stayed'])
plt.xlabel('Salary')
plt.ylabel('Percent')

plt.show()
```


    
![png](output_11_0.png)
    



```python
plt.figure(figsize=(14, 6))
sns.histplot(data=df1, x='Department', shrink=.8, stat='percent')
plt.title('Distribution of Different Departments (Downsampled Data)')
plt.xlabel('Departments')
plt.ylabel('Percent')

plt.figure(figsize=(14, 6))
sns.histplot(data=df1, x='Department', hue='left', multiple='dodge', shrink=.8, stat='percent', palette='viridis')
plt.title('Distribution of Different Departments for Employees Who Left or Did Not Left (Downsampled Data)')
plt.legend(labels=['left', 'stayed'])
plt.xlabel('Departments')
plt.ylabel('Percent')

plt.show()
```


    
![png](output_12_0.png)
    



    
![png](output_12_1.png)
    



```python
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(data=df0, x="satisfaction_level", hue="left", element="poly", stat='percent', palette='viridis')
plt.title('Distribution of Satisfaction Level for Employees \n Who Left or Did Not Left (Non Downsampled Data)')
plt.legend(labels=['left', 'stayed'], loc="lower left")
plt.xlabel('Satisfaction Level')
plt.ylabel('Percent')

plt.subplot(1, 2, 2)
sns.histplot(data=df1, x="satisfaction_level", hue="left", element="poly", stat='percent', palette='viridis')
plt.title('Distribution of Satisfaction Level for Employees \n Who Left or Did Not Left (Downsampled Data)')
plt.legend(labels=['left', 'stayed'], loc="lower left")
plt.xlabel('Satisfaction Level')
plt.ylabel('Percent')
```




    Text(0, 0.5, 'Percent')




    
![png](output_13_1.png)
    



```python
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(data=df1, x="last_evaluation", hue="left", element="poly", stat='percent', palette='viridis')
plt.title('Distribution of Last Evaluation for Employees \n Who Left or Did Not Left (Downsampled Data)')
plt.legend(labels=['left', 'stayed'], loc="lower left")
plt.xlabel('Last Evaluation')
plt.ylabel('Percent')

plt.subplot(1, 2, 2)
sns.histplot(data=df1, x="average_montly_hours", hue="left", element="poly", stat='percent', palette='viridis')
plt.title('Distribution of Average Monthly Hours for Employees \n Who Left or Did Not Left (Downsampled Data)')
plt.legend(labels=['left', 'stayed'], loc="lower left")
plt.xlabel('Average Monthly Hours')
plt.ylabel('Percent')

plt.show()
```


    
![png](output_14_0.png)
    



```python
plt.figure(figsize=(14, 6))
comparison_summary = df1.groupby(['left', 'Work_accident']).size().unstack()

plt.subplot(1, 2, 1)
sns.heatmap(comparison_summary, annot=True, fmt="d", cmap="viridis")
plt.title('Comparison of Employees Who Left vs Had an \n Work Accident (Downsampled Data)')
plt.xlabel('Work Accident')
plt.ylabel('Left Company')
plt.xticks([0.5, 1.5], ['No Accident', 'Accident'])
plt.yticks([0.5, 1.5], ['Stayed', 'Left'], rotation=0)

comparison_summary = df1.groupby(['left', 'promotion_last_5years']).size().unstack()

plt.subplot(1, 2, 2)
sns.heatmap(comparison_summary, annot=True, fmt="d", cmap="viridis")
plt.title('Comparison of Employees Who Left vs Promotions in \n the Last 5 Years (Downsampled Data)')
plt.xlabel('Promotion in Last 5 Years')
plt.ylabel('Left Company')
plt.xticks([0.5, 1.5], ['No Promotion', 'Promotion'])
plt.yticks([0.5, 1.5], ['Stayed', 'Left'], rotation=0)

plt.show()
```


    
![png](output_15_0.png)
    



```python
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(data=df0, x="number_project", hue="left", multiple='dodge', shrink=.8, stat='percent', palette='viridis')
plt.title('Distribution of Number of Projects for Employees \n Who Left or Did Not Left (Non Downsampled Data)')
plt.legend(labels=['left', 'stayed'])
plt.xlabel('Number of Projects')
plt.ylabel('Percent')

z = np.abs(stats.zscore(df0['time_spend_company']))
outliers = df0[z > 1.5]
df0_outliers = df0.drop(outliers.index)

z = np.abs(stats.zscore(df1['time_spend_company']))
outliers = df1[z > 1.5]
df1_outliers = df1.drop(outliers.index)

plt.subplot(1, 2, 2)
sns.histplot(data=df1_outliers, x="time_spend_company", hue="left", multiple='dodge', shrink=.8, stat='percent', palette='viridis')
plt.title('Distribution of Time Spend at the Company for \n Employees Who Left or Did Not Left (Downsampled Data)')
plt.legend(labels=['left', 'stayed'])
plt.xlabel('Time Spend at the Company')
plt.ylabel('Percent')

plt.show()
```


    
![png](output_16_0.png)
    



```python
label_encoder = LabelEncoder()
df0_outliers['salary'] = label_encoder.fit_transform(df0_outliers['salary'])

features = df0_outliers[['satisfaction_level', 'number_project', 'average_montly_hours', 
               'time_spend_company', 'promotion_last_5years', 'salary']]

vif_data = pd.DataFrame()
vif_data['Feature'] = features.columns
vif_data['VIF'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]

vif_data
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
      <th>Feature</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>satisfaction_level</td>
      <td>6.126429</td>
    </tr>
    <tr>
      <th>1</th>
      <td>number_project</td>
      <td>13.104632</td>
    </tr>
    <tr>
      <th>2</th>
      <td>average_montly_hours</td>
      <td>16.867282</td>
    </tr>
    <tr>
      <th>3</th>
      <td>time_spend_company</td>
      <td>11.321497</td>
    </tr>
    <tr>
      <th>4</th>
      <td>promotion_last_5years</td>
      <td>1.016087</td>
    </tr>
    <tr>
      <th>5</th>
      <td>salary</td>
      <td>5.176839</td>
    </tr>
  </tbody>
</table>
</div>




```python
label_encoder = LabelEncoder()
df1_outliers['salary'] = label_encoder.fit_transform(df1_outliers['salary'])

features = df1_outliers[['satisfaction_level', 'number_project', 'average_montly_hours', 
               'time_spend_company', 'promotion_last_5years', 'salary']]

vif_data = pd.DataFrame()
vif_data['Feature'] = features.columns
vif_data['VIF'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]

vif_data
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
      <th>Feature</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>satisfaction_level</td>
      <td>4.353971</td>
    </tr>
    <tr>
      <th>1</th>
      <td>number_project</td>
      <td>14.656249</td>
    </tr>
    <tr>
      <th>2</th>
      <td>average_montly_hours</td>
      <td>22.978714</td>
    </tr>
    <tr>
      <th>3</th>
      <td>time_spend_company</td>
      <td>13.979366</td>
    </tr>
    <tr>
      <th>4</th>
      <td>promotion_last_5years</td>
      <td>1.016832</td>
    </tr>
    <tr>
      <th>5</th>
      <td>salary</td>
      <td>5.461340</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_logreg = df0_outliers.copy()

y = df_logreg['left']
X = df_logreg.drop(columns=['last_evaluation', 'number_project', 'Work_accident', 'Department', 'left'])
```


```python
X_train, X_test0, y_train, y_test0 = train_test_split(X, y, test_size=0.25, stratify=y)
logr = LogisticRegression(max_iter=500).fit(X_train, y_train)
y_pred = logr.predict(X_test0)
```


```python
log_cm = confusion_matrix(y_test0, y_pred, labels=logr.classes_)
ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=logr.classes_).plot(values_format='')

plt.show()
```


    
![png](output_21_0.png)
    



```python
target_names = ['Predicted would not leave', 'Predicted would leave']
classification_rep = classification_report(y_test0, y_pred, target_names=target_names)

print(classification_rep)
```

                               precision    recall  f1-score   support
    
    Predicted would not leave       0.94      0.78      0.85       467
        Predicted would leave       0.81      0.95      0.87       470
    
                     accuracy                           0.86       937
                    macro avg       0.88      0.86      0.86       937
                 weighted avg       0.87      0.86      0.86       937
    
    


```python
df_logreg = df1_outliers.copy()

y = df_logreg['left']
X = df_logreg.drop(columns=['last_evaluation', 'number_project', 'Work_accident', 'Department', 'left'])
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
logr = LogisticRegression(max_iter=500).fit(X_train, y_train)
y_pred = logr.predict(X_test)
```


```python
log_cm = confusion_matrix(y_test0, y_pred, labels=logr.classes_)
ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=logr.classes_).plot(values_format='')

plt.show()
```


    
![png](output_25_0.png)
    



```python
target_names = ['Predicted would not leave', 'Predicted would leave']
classification_rep = classification_report(y_test0, y_pred, target_names=target_names)

print(classification_rep)
```

                               precision    recall  f1-score   support
    
    Predicted would not leave       0.48      0.42      0.45       467
        Predicted would leave       0.49      0.56      0.52       470
    
                     accuracy                           0.49       937
                    macro avg       0.49      0.49      0.48       937
                 weighted avg       0.49      0.49      0.48       937
    
    


```python

```
