#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.getcwd()


# # Data Exploration

# In[2]:


crash = pd.read_csv('C:/Users/iz/Desktop/New Folder/CrashTest_TrainData.csv',index_col =0)
crash.info()


# In[3]:


data=crash.copy()
data.head()


# # Q) How many distinct car types are there in the Train_Data?

# In[4]:


cart = pd.crosstab(index = data['CarType'], columns='Count')
cart
#Binary Classification


# # Q) How many missing values are there in Train_Data?

# In[5]:


data.isnull().sum()# Number of NUll values


# In[6]:


test = pd.read_csv('C:/Users/iz/Desktop/New Folder/CrashTest_TestData.csv',index_col =0)
testc = test.copy()
testc.info()


# In[7]:


testc.isnull().sum()


# # Q) What is the proportion of car types in the Test_Data?

# In[8]:


testc['CarType'].value_counts()


# # Q) What is the difference between third quartile values of the variable ManBI from Train_Data and Test_Data?

# In[9]:


B = data['ManBI'].median()
B1 = testc['ManBI'].median()
CrQ3 = np.quantile(data['ManBI'], .75)
CtQ3 = np.quantile(testc['ManBI'], .75)
diff = CrQ3 - CtQ3
print(diff)


# Train Set: Has 3 Null Values
# <br>Test Set : Has NO Null values

# In[10]:


# we can Either Remove the coloumns with null values or Impute them with mean ,median or mode.
data.info()
data['IntI'].fillna(data['IntI'].mean(), inplace=True)
print("\n----------------------------After Imputation------------------------------\n")
data.info()


# In[11]:


data.shape


# # Buliding Model
# Follow the steps given below to build the classifier models: 
# <br>• Drop the missing values 
# <br>• Ensure the datatypes of the columns are appropriate
# <br>• Map the categorical variables into integers

# In[12]:


train=crash.copy()
train.info()


# In[13]:


train=train.dropna()
train.info()
# Dropped Columns with null values


# In[14]:


from sklearn import preprocessing as p
label=train['CarType']
label[0:10]
le=p.LabelEncoder()
le.fit(label)
label=le.transform(label)
label#Catagorical Mapping


# In[15]:


#remove Target fom dataframe
train=train.drop(columns=['CarType'])
train.info()


# In[16]:


#remove Target fom dataframe and storing in list
actual_labels=testc['CarType']
testc=testc.drop(columns=['CarType'])
# map Lables
le.fit(actual_labels)
actual_labels=le.transform(actual_labels)
actual_labels#Catagorical Mapping "SUV"=1 "Hatchback"=0


# In[17]:


from sklearn.neighbors import KNeighborsClassifier
# Initiate KNN classifier
model_1 = KNeighborsClassifier(n_neighbors = 3) 
model_2 = KNeighborsClassifier(n_neighbors = 2)

# Fit the classifier to the data
model_1.fit(train,label)
model_2.fit(train,label)


# In[18]:


#test results


# # Q) Identify the list of indices of misclassified samples from the ‘model_1’.

# In[19]:


pred=model_1.predict(testc)
print("Predicted Lables    :",pred)
print("Actual Lables       :",actual_labels)
print("Indices of Misclassified labels:\n")
for i in range(0,len(actual_labels)):
    if pred[i]!=actual_labels[i]:
        print(i,end=" ")


# # Q) Rebuild the model (model_2) with 2 neighbors keeping the other modelling steps constant. Compare results of the two models (model_1 &amp; model_2).

# In[20]:


#Model_1 Accuracy (3NN) 
print("model_1:",model_1.score(testc,actual_labels))

#Model_2 Accuracy (2NN)
print("model_2:",model_2.score(testc,actual_labels))


# # Q) Build a logistic regression model (model_3) keeping the modelling steps constant. The accuracy of the model_3

# In[21]:


from sklearn.linear_model import LogisticRegression

# instantiate the model

model_3 = LogisticRegression()

# fit the model with data
model_3.fit(train,label)

#get predictions
pred=model_3.predict(testc)
print("predicions of model_3:",pred)
print("Actual               :",actual_labels)

# Accuracy
print("Accuracy:",model_3.score(testc,actual_labels))


# In[ ]:




