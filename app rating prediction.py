#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing library

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#1. Load the data file using pandas

df = pd.read_csv('googleplaystore.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


# Checking for null Values in the data
df.isnull()


# In[7]:


#2. Checking for null values count by each column
df.isnull().sum()


# In[8]:


#3. Droping the records with null in any of the column, Since the question demands of removing all the null items we will not go by removing through each column
df.dropna(inplace= True)


# In[9]:


df.isnull().sum()


# In[10]:


# Checking the revised Rows and columns
df.reset_index(drop= True, inplace = True)
df.shape


# In[11]:


#4. checking for the Data Types of each column
df.info()


# In[12]:


df['Size'].unique()


# In[13]:


#I) Start the cleaning with Size Column and converting in to numeric

df['Size'] = df['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace('Varies with device', 'nan') if 'Varies with device' in str(x) else x)

# Scaling all the values to Millions format (means that 19.0 => 19x10^6 => 19M)
df['Size'] = df['Size'].apply(lambda x: float(str(x).replace('k', ''))/1000 if 'k' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x : float(x))
df = df[pd.notnull(df['Size'])]
df['Size'].dtype


# In[14]:


df.shape


# In[15]:


df['Reviews'].unique()


# In[16]:


#II) Converting the Reviews column

df['Reviews'] = df['Reviews'].apply(lambda x : int(x))
df['Reviews'].dtype


# In[17]:


df['Rating'].dtype


# In[18]:


df['Price'].unique()


# In[19]:


#III) Moving on with the Cleaning and conversion of Price column

df['Price'] = df['Price'].apply((lambda x:str(x).replace('$','') if '$' in str(x) else str(x)))
df['Price'] = df['Price'].apply (lambda x: float(x))
df['Price'].dtype


# In[20]:


df.shape


# In[21]:


df['Installs'].unique()


# In[22]:


#IV) Cleaning and conversion of the Installs column

df['Installs'] = df['Installs'].apply (lambda x: str(x).replace('+','') if '+' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: str(x).replace(',', '') if ',' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: int (x))
df['Installs'].dtype


# In[23]:


df.shape


# In[24]:


#1) Avg. rating should be between 1 and 5, as only these values are allowed on the play store
df['Rating'].unique()


# In[25]:


df[df['Rating']>5].shape[0]


# In[26]:


#2) Reviews should not be more than installs as only those who installed can review the app
df[df['Reviews']>df['Installs']].shape[0]


# In[27]:


df.shape


# In[28]:


df.drop(df[df['Reviews']>df['Installs']].index, inplace = True)


# In[29]:


#3) For free apps price should be equal to 0
df[(df['Type']=='free')& (df['Price'] ==0)].shape[0]


# In[30]:


# Box Plot for Price
sns.set(rc={'figure.figsize':(10,6)})


# In[31]:


sns.boxplot(x= 'Price',data= df);


# In[32]:


# Boxplot for Reviews
sns.boxplot(x ='Reviews', data =df);


# In[33]:


# Histogram for Rating
df['Rating'].plot(kind= 'hist'); #we can use either to get the results
plt.hist(df['Rating'])


# In[34]:


# Histogram for Size
df['Size'].plot(kind= 'hist') #we can use either to get the results
plt.hist(df['Size'])


# In[35]:


#I) price of $200 and above for an application is expected to be very high
df[df['Price']>200].index.shape[0] #we can use either to get the results
df.loc[df['Price']>200].shape[0]


# In[36]:


#Dropping the Junk apps
df.drop(df[df['Price']>200].index, inplace= True)


# In[37]:


df.shape


# In[38]:


#II) Very few apps have very high no. of Reviews
df.loc[df['Reviews']>2000000].shape[0]


# In[39]:


#Dropping the Star apps as these will skew the analysis,
#checking the shape after dropping
df.drop(df[df['Reviews']>2000000].index, inplace= True)
df.shape


# In[42]:


#dropping the value more than the cutoff(threshold -95th percentile)
df.drop(df[df['Installs']>10000000].index, inplace= True)


# In[43]:


df.shape


# In[44]:


#1) Scatter plot/jointplot for Rating Vs. Price
sns.scatterplot(x = 'Rating', y = 'Price',data=df)


# In[45]:


sns.jointplot(x= 'Rating',y= 'Price',data= df)


# In[46]:


#2) Scatterplot/jointplot for Rating Vs. Size
sns.scatterplot(x= 'Rating',y= 'Size', data= df)


# In[47]:


sns.jointplot(x= 'Rating', y= 'Size', data= df)


# In[48]:


#3) Scatterplot for Ratings Vs. Reviews
sns.scatterplot(x= 'Rating',y= 'Reviews', data= df)


# In[49]:


#4) Boxplot for Ratings Vs. Content Rating
sns.set(rc={'figure.figsize':(14,8)})
sns.boxplot(x= 'Rating', y= 'Content Rating', data = df)


# In[50]:


#5) Boxplot for Ratings Vs. Category
sns.set(rc={'figure.figsize':(18,12)})
sns.boxplot(x= 'Rating', y = 'Category', data= df)


# In[51]:


#creating a copy of the data(df) to make all edits
inp1= df.copy()


# In[52]:


inp1.head()


# In[55]:


#1) apply log transformation to Reviews
reviews_skew = np.log1p(inp1['Reviews'])
inp1['Reviews']= reviews_skew


# In[56]:


reviews_skew.skew()


# In[57]:


#1) apply log transformation to Installs
Installs_skew = np.log1p(inp1['Installs'])
inp1['Installs']


# In[58]:


Installs_skew.skew()


# In[59]:


inp1.head()


# In[60]:


#2) Dropping the columns- App, Last Updated, Current Ver, Type, & Andriod Ver as these won't be useful for our model
inp1.drop(['App','Last Updated','Current Ver','Android Ver','Type'], axis= 1, inplace = True)


# In[61]:


inp1.head()


# In[62]:


inp1.shape


# In[63]:


#3) create a copy of dataframe
inp2 = inp1


# In[64]:


inp2.head()


# In[65]:


#get unique values in column category
inp2['Category'].unique()


# In[66]:


inp2.Category = pd.Categorical(inp2.Category)

x = inp2[['Category']]
del inp2['Category']

dummies = pd.get_dummies(x, prefix = 'Category')
inp2 = pd.concat([inp2,dummies], axis=1)
inp2.head()


# In[67]:


#get unique values in Column Genres
inp2["Genres"].unique()


# In[68]:


#Create an empty list
lists = []
#Get the total genres count and gernes count of perticular gerner count less than 20 append those into the list
for i in inp2.Genres.value_counts().index:
    if inp2.Genres.value_counts()[i]<20:
        lists.append(i)
#changing the gerners which are in the list to other
inp2.Genres = ['Other' if i in lists else i for i in inp2.Genres]


# In[69]:


inp2["Genres"].unique()


# In[70]:


#Storing the genres column into x varible and delete the genres col from dataframe inp2
#And concat the encoded cols to the dataframe inp2
inp2.Genres = pd.Categorical(inp2['Genres'])
x = inp2[["Genres"]]
del inp2['Genres']
dummies = pd.get_dummies(x, prefix = 'Genres')
inp2 = pd.concat([inp2,dummies], axis=1)


# In[71]:


inp2.head()


# In[72]:


#getting the unique values in Column "Content Rating"
inp2["Content Rating"].unique()


# In[73]:


#Applying one hot encoding 
#Storing the Content Rating column into x varible and delete the Content Rating col from dataframe inp2
#And concat the encoded cols to the dataframe inp2
inp2['Content Rating'] = pd.Categorical(inp2['Content Rating'])

x = inp2[['Content Rating']]
del inp2['Content Rating']

dummies = pd.get_dummies(x, prefix = 'Content Rating')
inp2 = pd.concat([inp2,dummies], axis=1)
inp2.head()


# In[74]:


inp2.shape


# In[75]:


#importing the neccessary libraries from sklearn to split the data and and for model building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn import metrics


# In[76]:


#Creating the variable X and Y which contains the X features as independent features and Y is the target feature 
df2 = inp2
X = df2.drop('Rating',axis=1)
y = df2['Rating']

#Dividing the X and y into test and train data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=5)


# In[77]:


#Create a linear reggression obj by calling the linear reggressor algorithm
lin_reggressor = LinearRegression()
lin_reggressor.fit(X_train,y_train)


# In[78]:


R2_Score_train_data = round(lin_reggressor.score(X_train,y_train),3)
print("The R2 value of the Training Set is : {}".format(R2_Score_train_data))


# In[79]:


# test the output by changing values, like 3750
y_pred = lin_reggressor.predict(X_test)
R2_Score_test_data =metrics.r2_score(y_test,y_pred)
R2_Score_test_data


# In[80]:


R2_Score_test_data = round(lin_reggressor.score(X_test,y_test),3)
print("The R2 value of the Training Set is : {}".format(R2_Score_test_data))


# In[ ]:




