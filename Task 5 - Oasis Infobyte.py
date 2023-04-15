#!/usr/bin/env python
# coding: utf-8

# # Task 4: Sales Prediction Using Python
# # Name of intern: Pritee Jamadade
# # Batch:March P-2 OIB-SIP

#  # Step 1 : Import Libraries

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# # Step 2 : Import Dataset 

# In[2]:


data=pd.read_csv(r"C:\Users\DNYANESH\Downloads\Advertising.csv")
data


# # Step 3 : Data Analysis

# In[3]:


data.head()


# In[4]:


data.tail()


# In[8]:


data.rename(columns={"Unnamed: 0": "Index"}, inplace=True)
data


# In[9]:


df=data.drop(columns='Index')
df


# In[10]:


data.isnull().sum()


# In[11]:


df.describe()


# In[12]:


df.corr()


# # Step 4: Data Visualization

# In[13]:


sns.scatterplot(x='Sales', y='TV',data=data)
plt.title('SALES OF TV')


# In[14]:


sns.scatterplot(x='Sales', y='Radio',data=data)
plt.title('SALES OF RADIO')


# In[15]:


sns.scatterplot(x='Sales', y='Newspaper',data=data)
plt.title('SALES OF NEWSPAPER')


# In[16]:


sns.pairplot(df)


# In[23]:


sns.lineplot(x='Sales',y='TV',data=df,color='purple')
plt.show()


# In[24]:


sns.lineplot(x='Sales',y='Radio',data=df,color='red')
plt.show()


# In[25]:


sns.lineplot(x='Sales',y='Newspaper',data=df,color='blue')
plt.show()


# In[18]:


sns.heatmap(df.corr(),annot=True,cmap='summer')


# # Step 5 : Sales Prediction

# In[27]:


x = np.array(df.drop(["Sales"], 1))
y = np.array(df["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

data = pd.DataFrame(data={"Predicted Sales": ypred.flatten()})
print(data)


# ## So this is how we can predict future sales of a product using Python

# In[ ]:




