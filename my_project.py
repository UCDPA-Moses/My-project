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
plt.title('Plot of Different Categories / Manufacturers of Cars',size = 25)
plt.show()

# Creating histogram for continuous numerical variable
plt.hist(df['Horsepower'])
plt.title('Distribution of different Horsepower',size = 25)
plt.xlabel('Horsepower value',size = 18)
plt.ylabel('Number of cars',size = 18)
plt.show()

plt.pie(categories_avail2, labels=categories_avail2.index, autopct='%1.1f%%')
plt.title('Plot of Different Vehicle type', size = 25)
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
plt.title('Feature importance for the highest 10 variables',size = 25)
plt.show()

#Performing data scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
sc.fit(features)
input_scaled = sc.transform(features)

#Splitting data into training and test set
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(input_scaled,target,test_size = 0.2, random_state=100)

#Building linear regression model for the selling price

#fit the model for training set
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

#predict the values for the training dataset
y_pred = model.predict(x_train)

#Plotting the line of best fit
plt.scatter(y_train,y_pred)
plt.title('Plotting actual target vs predicted target',size = 25)
plt.xlabel("Target (y_train)", size = 18)
plt.ylabel("Predicted (y_hat)", size = 18)
plt.show()

# Performance Metrics of the training set
print("Linear regression accuracy is {:.2f}%".format(model.score(x_train, y_train) *100))

#Testing the model

#predict the values for the training dataset
y_pred_test = model.predict(x_test)

plt.scatter(y_test,y_pred_test)
plt.title('Linear Regression: Predicting target values from the feature values',size = 25)
plt.xlabel("Targets (y_test)", size = 18)
plt.ylabel("Predictions (y_pred_test)", size = 18)
plt.show()

from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
print("R_square value is: ", r2_score(y_test,y_pred_test))
print("MSE is: ", mean_squared_error(y_test,y_pred_test))
print("RMSE is: ", np.sqrt(mean_squared_error(y_test,y_pred_test)))

#Building a Random Forest model

from sklearn.ensemble import RandomForestRegressor as RFC

regressor=RFC() # creating a random forest regressor model
regressor.fit(x_train,y_train)

# Performance Metrics of the training set
print("Random forest regressor accuracy is {:.2f}%".format(regressor.score(x_train, y_train) *100))

predictions=regressor.predict(x_test)

plt.scatter(y_test,predictions)
plt.title('Random Forest: Predicting target values from the feature values',size = 25)
plt.xlabel("Targets (y_test)", size = 18)
plt.ylabel("Predictions (y_pred_test)", size = 18)
plt.show()

#  Performance Metrics
print("R_square value is: ", r2_score(y_test,predictions))
print('MSE:', mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))