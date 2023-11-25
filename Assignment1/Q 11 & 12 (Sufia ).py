#!/usr/bin/env python
# coding: utf-8

# In[10]:


import math
import numpy as np


# ##Suppose we want to estimate the average weight of an adult male in    Mexico. We draw a random sample of 2,000 men from a population of 3,000,000 men and weigh them. We find that the average person in our sample weighs 200 pounds, and the standard deviation of the sample is 30 pounds. Calculate 94%,98%, and 96% confidence interval?

# In[1]:


#mean = 200
#std dev = 30
#sample = 2000
#population = 3000000


# In[4]:


#for 94% confidence interval
(200 +(1.555*(30/math.sqrt(2000))))


# In[5]:


(200 -(1.555*(30/math.sqrt(2000))))


# In[6]:


# for 98% confidence interval
(200 +(2.326*(30/math.sqrt(2000))))


# In[7]:


(200 -(2.326*(30/math.sqrt(2000))))


# In[8]:


#for 96% confidence interval
(200 +(2.04*(30/math.sqrt(2000))))


# In[9]:


(200 -(2.04*(30/math.sqrt(2000))))


# In[11]:


data = [34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56]


# In[13]:


np.mean(data)


# In[14]:


np.median(data)


# In[15]:


np.var(data)


# In[16]:


np.std(data)


# In[ ]:




