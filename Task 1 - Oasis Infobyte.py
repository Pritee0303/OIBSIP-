#!/usr/bin/env python
# coding: utf-8

#  # Task 1: Iris Flower Classification
# # Name of intern: Pritee Jamadade
# # Batch:March P-2 OIB-SIP

# # Step 1 : Import libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt


# # Step 2: Load Dataset

# In[2]:


data=sns.load_dataset('iris')
data


# In[3]:


data.head()


# In[4]:


data.isnull().sum()


# In[5]:


data.info()


# # Step 3 : Data Analysis

# In[6]:


data.describe()


# from this we can obaerve the count of each column along with their mean value, standard deviation, minimum and maximum values

# In[7]:


data.corr()


# # Step 4: Data Visualization

# In[8]:


sns.heatmap(data.corr(),annot=True,cmap='coolwarm')


# from the above graph we can observe that
# Petal width and petal length have high correlations, 
# Petal length and sepal width have good correlations,
# Petal Width and Sepal length have good correlations

# In[9]:


sns.scatterplot(x='sepal_length', y='sepal_width',
                hue='species', data=data )
 
plt.legend(bbox_to_anchor=(1, 1), loc=2)

plt.show()


# In the above plot we will see the relation between sepal length and sepal width
# 
# from this we can interpret that  the species setosa has smaller sepal lengths but larger sepal widths,
# versicolor species lies in the middle of the other two species in terms of sepal length and width and
# species virginica has larger sepal lengths but smaller sepal widths.

# In[10]:


sns.scatterplot(x='petal_length', y='petal_width',
                hue='species', data=data, )
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()


# In the above plot we will see the relation between petal length and petal width
# 
# from the above graph we will observe that 
# species setosa has smaller petal lengths and widths,
# versicolor species lies in the middle of the other two species in terms of petal length and width and 
# species ivrginica has the largest of petal lengths and widths

# In[11]:


fig, axes = plt.subplots(2, 2, figsize=(10,10))
 
axes[0,0].set_title("Sepal Length")
axes[0,0].hist(data['sepal_length'], bins=7)
 
axes[0,1].set_title("Sepal Width")
axes[0,1].hist(data['sepal_width'], bins=5);
 
axes[1,0].set_title("Petal Length")
axes[1,0].hist(data['petal_length'], bins=6);
 
axes[1,1].set_title("Petal Width")
axes[1,1].hist(data['petal_width'], bins=6);


# From the above histograms we can see that ,
# 
# The highest frequency of the sepal length is between 30 and 35 which is between 5.5 and 6 and 
# The highest frequency of the sepal Width is around 70 which is between 3.0 and 3.5 and 
# The highest frequency of the petal length is around 50 which is between 1 and 2 and
# The highest frequency of the petal width is between 40 and 50 which is between 0.0 and 0.5

# In[25]:


sns.boxplot(x = "species", y = "petal_length", data = data)


# In[12]:


sns.pairplot(data,hue="species")


# There is a strong correlation between columns for petal length and petal width 
# Setosa has short petal length and petal width
# 
# 
# 
# 

# # Model Accuracy

# In[14]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[15]:


data['species']=le.fit_transform(data['species'])
data


# In[53]:


from sklearn.model_selection import train_test_split
x=data.drop(columns=['species'])
y=data['species']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)


# In[54]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[55]:


model.fit(x_train,y_train) 


# In[56]:


print("Accuracy:",model.score(x_test,y_test)*100)


# In[57]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()


# In[58]:


model.fit(x_train,y_train)


# In[59]:


print("Accuracy:",model.score(x_test,y_test)*100)


# In[60]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()


# In[61]:


model.fit(x_train,y_train)


# In[62]:


print("Accuracy:",model.score(x_test,y_test)*100)


# # THANK YOU 

# In[ ]:




