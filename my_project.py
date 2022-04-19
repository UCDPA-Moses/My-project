# importing necessary libraries


import numpy as np # linear algebra
import pandas as pd # data processing and preparation
import matplotlib.pyplot as plt # Visualization
import seaborn as sns # Visualization on top of matplotlib
import matplotlib
import seaborn as sns # Advanced visualization


#import data
df = pd.read_csv("https://raw.githubusercontent.com/UCDPA-Moses/UCDPA_Moses/main/Car_sales.csv")
print(df.head())

#Data Exploratory
print("Information about the dataset")
print(df.info())

print("Description of the dataset")
print(df.describe())

print("Shape, i.e Rows by Columns")
print(df.shape)

print(df.duplicated) #checking for duplicates

# Replacing space in column names
df.columns = df.columns.str.replace(' ', '_') #using the replace function

total_missing=df.isna().sum().sum() #checking for missing values in the dataset
print("Total missing values are:", total_missing)

#Replacing missing data with the mean of the column
print("Replacing missing data with the mean of the column")
df.fillna(df.mean(),inplace=True)
total_missing=df.isna().sum().sum() #checking for missing values in the dataset
print("Total missing values are:", total_missing)

#checking null values
print('null values in each column is:')
print(df.isnull().sum())

# Sorting all the data in the dataset with respexct to Engine size
df_sorted = df.sort_values(by = 'Engine_size', ascending = False) #sorting the data by Engine size
print("sorting the data by Engine size")
print(df_sorted.head()) 

# Correlation with Price_in_thousands
df_corr = df.corr()['Price_in_thousands'][:-1]
print(df_corr)

# Counting all the cars in the dataset by its category
categories_avail = df.groupby('Manufacturer').size()
print("The different categories / manufacturers of cars in this dataset are:")
print(categories_avail)

print("Categories of cars based on the vehicle type")
categories_avail2= df.groupby('Vehicle_type').size()
print(categories_avail2)

#Formatting the default setting the charts and graphs
sns.set_style('whitegrid')
matplotlib.rcParams['font.size'] =8
matplotlib.rcParams['figure.figsize'] = (10,6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


#plotting a bar chat for the groups of cars 
plt.pie(categories_avail, labels=categories_avail.index)
plt.title('Plot of Different Categories / Manufacturers of Cars')
plt.show()

# Creating histogram for continuous numerical variable
plt.hist(df['Horsepower'])
plt.title('Distribution of different Horsepower')
plt.xlabel('Horsepower value')
plt.ylabel('Number of cars')
plt.show()

df.hist(bins=15,xlabelsize=7)
plt.show()

plt.pie(categories_avail2, labels=categories_avail2.index, autopct='%1.1f%%')
plt.title('Plot of Different Vehicle type')
plt.show()

#Machine Learning development to predict the selling price of car

#Removing columns that does not contribute to the selling price
df = df.drop(['Model','Latest_Launch'], axis = 1) 

#Data encoding using dummy values
df=pd.get_dummies(df,drop_first=True)
print('Dataset with categorical values encoded')
print(df)

#Splitting dataset into features and target variables
target= df['Price_in_thousands']
features = df.drop(['Price_in_thousands'],axis = 1) 

#Feature importance
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(features,target)

#plotting the feature importance
feat_importances = pd.Series(model.feature_importances_, index=features.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Feature importance for the highest 10 variables')
plt.show()

#Performing data scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
sc.fit(features)
input_scaled = sc.transform(features)

#Splitting data into training and test set
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(input_scaled,target,test_size = 0.2, random_state=100)
