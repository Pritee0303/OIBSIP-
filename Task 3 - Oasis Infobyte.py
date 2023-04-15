#!/usr/bin/env python
# coding: utf-8

# # Task 3: Car Price Prediction 
# # Name of intern: Pritee Jamadade
# # Batch:March P-2 OIB-SIP

# # Step 1 : Import Libraries

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


# # Step 2 : Import Dataset

# In[2]:


data=pd.read_csv(r"C:\Users\DNYANESH\OneDrive\Documents\CarPrice.csv")
data


# # Step 3 : Data Analysis

# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data=data.drop(['car_ID'],axis=1)


# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# In[8]:


data['CarName'] = data['CarName'].apply(lambda x : x.split()[0])


# In[9]:


data['CarName'].unique()


# In[10]:


data['CarName'] = data['CarName'].replace({'maxda': 'mazda', 'nissan': 'Nissan', 'porcshce': 'porsche', 'toyouta': 'toyota', 'vokswagen': 'volkswagen', 'vw': 'volkswagen'})


# In[11]:


data.corr()


# # Step 4: Data Visualization

# In[12]:


plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap="flare")


# In[13]:


data.hist(bins=25,figsize=(20,10))


# In[14]:


n=pd.DataFrame(data['CarName'].value_counts()).reset_index().rename(columns={'index':'car_name','CarName': 'count'})
n


# In[15]:


plt.figure(figsize=(11,6))
plot = sns.barplot(x='car_name',y='count',data=n)
plot=plt.setp(plot.get_xticklabels(), rotation=90)


# In[16]:


sns.distplot(data['price'],kde=True)


# In[17]:


plt.figure(figsize=(4,4))
plt.title('Car Price Spread')
sns.boxplot(y=data.price)
plt.show()


# In[18]:


sns.pairplot(data[['horsepower','price','carbody']], hue="carbody")


# In[19]:


plt.figure(figsize = (25,8))
sns.boxplot(palette ="YlGnBu", data=data)


# In[20]:


sns.pairplot(data = data , x_vars = ['carwidth', 'carheight', 'curbweight', 'enginesize'] , y_vars = ['price'])
sns.pairplot(data = data , x_vars = ['wheelbase','carlength','peakrpm', 'citympg', 'highwaympg'] , y_vars = ['price'])
sns.pairplot(data = data , x_vars = ['boreratio', 'stroke', 'compressionratio', 'horsepower'] , y_vars = ['price'])


# In[21]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['CarName']=le.fit_transform(data['CarName'])
data['fueltype']=le.fit_transform(data['fueltype'])
data['aspiration']=le.fit_transform(data['aspiration'])
data['doornumber']=le.fit_transform(data['doornumber'])
data['drivewheel']=le.fit_transform(data['drivewheel'])
data['enginelocation']=le.fit_transform(data['enginelocation'])
data['enginetype']=le.fit_transform(data['enginetype'])
data['cylindernumber']=le.fit_transform(data['cylindernumber'])
data['carbody']=le.fit_transform(data['carbody'])
data['fuelsystem']=le.fit_transform(data['fuelsystem'])


# In[22]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data=data
VIF=pd.Series([variance_inflation_factor(vif_data.values,i) 
    for i in range(vif_data.shape[1])],index=vif_data.columns)
VIF


# In[23]:


x = data.drop(['price'] , axis = 1).values
y= data['price' ].values


# In[24]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x, y, test_size= 0.3 , random_state=42)


# In[25]:


x_train.shape


# In[26]:


y_train.shape


# In[27]:


from sklearn.linear_model import LinearRegression
reg =LinearRegression()


# In[28]:


reg.fit(x_train , y_train)
reg.score(x_train , y_train)


# In[29]:


reg.score(x_test , y_test)


# In[30]:


reg.coef_


# In[31]:


pd.DataFrame(reg.coef_ , data.columns[:-1] ,  columns=['Coeficient'])


# In[32]:


y_pred =reg.predict(x_test)
datal = pd.DataFrame({"Y_test": y_test , "Y_pred" : y_pred})
datal.head(10)


# In[33]:


plt.figure(figsize=(5,5))
plt.plot(datal[:50])
plt.legend(["Actual" , "Predicted"])


# In[34]:


from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(random_state=0)
dt_regressor.fit(x_train,y_train)
y_train_pred = dt_regressor.predict(x_train)
y_test_pred = dt_regressor.predict(x_test)
dt_regressor.score(x_test,y_test)


# In[35]:


print("Accuracy:",dt_regressor .score(x_test,y_test)*100)


# In[36]:


from sklearn.metrics import mean_absolute_error
dt_regressor.score(x_test,y_test_pred)


# In[ ]:




