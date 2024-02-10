#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()


# In[2]:


dir(iris)


# In[4]:


iris.feature_names


# In[8]:


df=pd.DataFrame (iris.data,columns=iris.feature_names)
df.head()


# In[9]:


df['target']=iris.target
df.head()


# In[10]:


iris.target_names


# In[12]:


df[df.target==1].head()


# In[13]:


df[df.target==2].head()


# In[ ]:


#0 to 50 is 'setosa', 50to 100'versicolor',100 to 150'virginica'


# In[16]:


df['flower_name']=df.target.apply(lambda x:iris.target_names[x])


# In[17]:


df.head()


# In[18]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


#separting 3 speices different df
df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]


# In[20]:


df0.head()


# In[21]:


df1.head()


# In[22]:


df2.head()


# In[26]:


plt.xlabel("sepal length (cm)")
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='blue')


# In[27]:


plt.xlabel("petal length (cm)")
plt.ylabel('petal width (cm)')
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='green')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='blue')


# In[28]:


from sklearn.model_selection import train_test_split


# In[30]:


#remove target col
X=df.drop(['target','flower_name'],axis='columns')
X.head()


# In[31]:


y=df.target


# In[32]:


y


# In[33]:


#30% as test 70% as training
X_train, X_test, Y_train, Y_test=train_test_split(X,y,test_size=0.3)


# In[34]:


len(X_train)


# In[35]:


len(X_test)


# In[36]:


from sklearn.svm import SVC
model = SVC()


# In[38]:


#to train model
model.fit(X_train,Y_train)


# In[39]:


#check accuracy
model.score(X_test,Y_test)


# In[ ]:




