#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
from scipy.stats import skew,kurtosis


# In[12]:


df = pd.read_csv('Q9_a.csv',index_col=0)


# In[14]:


df.head()


# In[16]:


print(skew(df)) #speed has a longer tail on the left side, hence its mean,mode,median are all negative.


# In[17]:


print(kurtosis(df)) #speed has negative kurtosis - more data is present at the peak, for dist - more data is present at tails.


# In[ ]:




