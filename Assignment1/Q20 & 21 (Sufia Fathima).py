#!/usr/bin/env python
# coding: utf-8

# In[19]:


import math
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('Cars.csv')


# In[5]:


df.head()


# In[7]:


# Probability greater than 38
P_38 = 1 - stats.norm.cdf(38, loc = df['MPG'].mean(), scale = df['MPG'].std())
P_38


# In[8]:


#Probabiltiy less than 40
P_40 = stats.norm.cdf(40, loc = df['MPG'].mean(), scale = df['MPG'].std())
P_40


# In[9]:


#Probability between 20 and 50
P_20 = 1 - stats.norm.cdf(20, loc = df['MPG'].mean(), scale = df['MPG'].std())
P_50 = stats.norm.cdf(50, loc = df['MPG'].mean(), scale = df['MPG'].std())
print(P_50 - P_20)


# ## Q21

# In[10]:


pd.DataFrame(df['MPG']).plot(kind='density')


# ## Q21 b

# In[11]:


df1 = pd.read_csv('wc-at.csv')


# In[13]:


df1.head()


# In[14]:


pd.DataFrame(df1).plot(kind='density')


# In[22]:


#scipy.stats.zscore(0.95)


# In[23]:


(1+0.90)/2


# In[25]:


np.round((1+0.94)/2,3)


# In[27]:


np.round((1+0.60)/2,3)


# In[ ]:




