# importing necessary libraries

<<<<<<< HEAD
import numpy as np # linear algebra
import pandas as pd # data processing and preparation
import matplotlib.pyplot as plt # Visualization
=======

import numpy as np # linear algebra
import pandas as pd # data processing and preparation
import matplotlib.pyplot as plt # Visualization
import seaborn as sns # Visualization on top of matplotlib
>>>>>>> 07f8199deb4acf626c4604de082f1a06b762f143
import matplotlib
import seaborn as sns # Advanced visualization


#import data
df = pd.read_csv("https://raw.githubusercontent.com/UCDPA-Moses/UCDPA_Moses/main/Car_sales.csv")
print(df.head())

#Data Exploratory
print(df.info())

print(df.describe())

print(df.shape)

print(df.duplicated) #checking for duplicates

# Dropping unnecessary columns
drop_cols = ['Vehicle_type']
df = df.drop(drop_cols, axis = 1)

print(df.describe())

# Sorting all the data in the dataset with respexct to Engine size
df_sorted = df.sort_values(by = 'Engine_size', ascending = False) #sorting the data by Engine size
print(df_sorted.head()) 

# Replacing space in column names
df.columns = df.columns.str.replace(' ', '_') #using the replace function

# Counting all the cars in the dataset by its category
categories_avail = df.groupby('Manufacturer').size()
print(categories_avail)


#Formatting the default setting the charts and graphs
sns.set_style('darkgrid')
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
