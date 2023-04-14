#!/usr/bin/env python
# coding: utf-8

# # Task 2: Unemployment Analysis With Python
# # Name of intern: Pritee Jamadade
# # Batch:March P-2 OIB-SIP

# # Step 1: Import Libraries 

# In[38]:


import pandas as pd
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


# # Step 2 : Loading Dataset

# In[2]:


data=pd.read_csv(r"C:\Users\DNYANESH\OneDrive\Documents\unemployment data.csv")
data


# # Step 3 : Data Analysis  

# In[3]:


data.head()


# In[4]:


data.tail()


# In[7]:


data.info()


# In[12]:


data.columns=["States","Date", "Frequency","Estimated Unemployment Rate","Estimated Employed","Estimated Labour Participation Rate","Region","longitude","latitude"]


# In[13]:


print(data)


# In[8]:


data.isnull().sum()


# In[9]:


data.describe()


# In[10]:


data.corr()


# # Data Interpretation  

# In[44]:


print(data['States'].value_counts().idxmax())


# Andhra Pradesh is the state with highest unemployment rate

# In[45]:


print(data['States'].value_counts().idxmin())


# Sikkim is the state with lowest Unemployment rate

# # Step 4 : Data Visualization

# In[11]:


sns.heatmap(data.corr(),annot=True)
plt.style.use("seaborn-whitegrid")
plt.figure(figsize=(8,8))
plt.show()


# In[18]:


data.columns=["States","Date", "Frequency","Estimated Unemployment Rate","Estimated Employed","Estimated Labour Participation Rate","Region","longitude","latitude"]
plt.title("Unemployment in India")
sns.histplot(x="Estimated Employed",hue="Region",data=data)
plt.figure(figsize=(8,8))
plt.show()


# In[19]:


plt.title("Unemployment in India")
sns.histplot(x="Estimated Unemployment Rate",hue="Region",data=data)
plt.figure(figsize=(8,8))
plt.show()


# In[32]:


color=sns.color_palette()
unemployment=data[["States","Region","Estimated Unemployment Rate"]]
figure = px.sunburst(unemployment, path=["Region", "States"], 
                     values="Estimated Unemployment Rate", 
                     width=600, height=600, color_continuous_scale="RdY1Gn", 
                     title="Unemployment Rate in India")
figure.show()


# In[51]:


plot_ump=data[['Estimated Unemployment Rate','States']]
data_unemp=plot_ump.groupby('States').mean().reset_index()
data_unemp=data_unemp.sort_values('Estimated Unemployment Rate')
fig=px.bar(data_unemp,x='States',y='Estimated Unemployment Rate',color='States',title='Average Unemployment Rate In Each State',template='plotly')
fig.show()


# In[50]:


plot=px.pie(data,values='Estimated Unemployment Rate',names=data['States'],title='Unemployment Rate State wise',template='plotly')
plot.show()


# In[ ]:




