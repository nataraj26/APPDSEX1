
# Implementing Data Preprocessing and Data Analysis

## AIM:
To implement Data analysis and data preprocessing using a data set

## ALGORITHM:
Step 1: Import the data set necessary

Step 2: Perform Data Cleaning process by analyzing sum of Null values in each column a dataset.

Step 3: Perform Categorical data analysis.

Step 4: Use Sklearn tool from python to perform data preprocessing such as encoding and scaling.

Step 5: Implement Quantile transfomer to make the column value more normalized.

Step 6: Analyzing the dataset using visualizing tools form matplot library or seaborn.

## CODING AND OUTPUT:
### Perform Data Cleaning process.
```python
import pandas as pd
import numpy as np
df = pd.read_csv('/content/Toyota (1).csv')
df.head()
df.duplicated()
df.info()
df = df.drop_duplicates()
df = df.dropna()
df.isnull().sum()
df['MetColor'] = df['MetColor'].astype('object')
df['Automatic'] = df['Automatic'].astype('object')  
print(df.isnull().sum())

```
### output
![image](https://github.com/user-attachments/assets/5745f3ba-a0b2-4341-a8ab-4b8693ce5f82)

### Perform Outlier Detection and Removal using IQR method for any Numerical columns
```python
Q1 = df['Price'].quantile(0.25) 
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
outlier_step = 1.5 * IQR
df_no_outliers = df[~((df['Price'] < (Q1 - outlier_step)) | (df['Price'] > (Q3 + outlier_step)))]
```

### output
![image](https://github.com/user-attachments/assets/07d6eb3c-66ec-40ff-a4d2-00e624de26bf)

### Identify the categorical data and perform categorical analysis.
### Perform Bivariate and multivariate analysis [Comapare the suitable columns in the dataset].
### Implement any two data Visualization 
```python
import pandas as pd
import numpy as np
!pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt 

df = pd.read_csv('/content/Toyota (1).csv')


# Identify categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns


print(f"Categorical columns: {categorical_cols}")

for col in categorical_cols:
    print(f"\nAnalysis of {col}:")
    print(df[col].value_counts()) 
    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x='FuelType', y='Price', data=df)
plt.title('Price vs. FuelType')
plt.xlabel('Fuel Type')
plt.ylabel('Price')
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(x='Age', y='Price', hue='Automatic', data=df)
plt.title('Price vs. Age vs. Automatic')
plt.xlabel('Age')
plt.ylabel('Price')
plt.show()

numerical_cols = df.select_dtypes(exclude=['object']).columns
corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Variables')
plt.show()
```
### output

![image](https://github.com/user-attachments/assets/41bfd384-1db8-4a9c-a554-bddc0555d078)
![image](https://github.com/user-attachments/assets/fe46c37b-0e5a-4e28-8852-40e3c7c086e9)
![image](https://github.com/user-attachments/assets/91c1f58b-0b6d-401a-8565-c60a54d20170)

## Perform any two operations[Scaling/Encoding/Transformation].
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
numerical_features = ['Age', 'KM']
df.replace('??', pd.NA, inplace=True)
df[numerical_features] = df[numerical_features].apply(pd.to_numeric, errors='coerce')
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
print(df.head())
from sklearn.preprocessing import OneHotEncoder
categorical_features = ['FuelType', 'Color']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(df[categorical_features])
encoded_df = pd.DataFrame(encoded_features)
encoded_df.columns = encoder.get_feature_names_out(categorical_features)
df = pd.concat([df, encoded_df], axis=1)
df = df.drop(categorical_features, axis=1)
print(df.head())
```
## Output
![image](https://github.com/user-attachments/assets/9dbfc1e5-dc96-4c3f-92c8-9193acfcd5d4)


## RESULT:
Thus Data analysis and Data preprocessing implemeted using a dataset.
