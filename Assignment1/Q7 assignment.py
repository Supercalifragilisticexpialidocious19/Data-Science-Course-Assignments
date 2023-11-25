#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import scipy.stats as stats


# In[2]:


df = pd.read_csv('Q7.csv')


# In[3]:


df


# In[5]:


np.mean(df)


# In[11]:


np.median(df['Points'])


# In[12]:


np.median(df['Score'])


# In[13]:


np.median(df['Weigh'])


# In[15]:


np.var(df)


# In[16]:


np.std(df)


# In[19]:


df.describe()


# In[20]:


df.mode()


# In[28]:


df['Points'].mode()


# In[27]:


df['Score'].mode()


# In[26]:


df['Weigh'].mode()


# In[30]:


range_points = df['Points'].max() - df['Points'].min()
range_points


# In[31]:


range_score = df['Score'].max() - df['Score'].min()
range_score


# In[32]:


range_weigh = df['Weigh'].max() - df['Weigh'].min()
range_weigh


# In[ ]:




