#!/usr/bin/env python
# coding: utf-8

# **Aim: Use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.**

# ***Exploratory Data Analysis***

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


test_data = pd.read_csv("test.csv")
train_data = pd.read_csv("train.csv")


# In[3]:


test_data.describe


# In[4]:


train_data.describe


# **Droping rows with missing values**

# In[5]:


test_data.dropna(axis = 0, inplace = True)
train_data.dropna(axis = 0, inplace = True)


# In[6]:


test_data.describe


# In[7]:


train_data.describe


# **Finding relation between Sex of passenger and their survival rate**

# In[8]:


sns.catplot(x ="Sex", hue ="Survived",
kind ="count", data = train_data)


# We can say roughly 40% of the males and 90% of the females survived

# **Finding the rate of survival of a person based on their socio-economic status (label- pclass)**

# In[10]:


group = train_data.groupby(['Pclass', 'Survived'])
pclass_survived = group.size().unstack()
 

sns.heatmap(pclass_survived, annot = True, fmt ="d")


# A person with a higher class ticket has more chances of survival

# **Relation between age and survival**

# In[33]:


train_data["Age_Binned"] = pd.cut(x = train_data["Age"],  
                                  bins=np.linspace(min(train_data["Age"]), 
                max(train_data["Age"]),4), labels = ['Children', 'Adolescents', 'Adults'] , include_lowest = True )


# In[30]:


Age_Binned


# In[67]:


sns.swarmplot(x = 'Age_Binned', y ='Survived', data = train_data )


# Chances of survivng is better in case of children and adolescents

# **Fare and Survival**

# In[73]:


sns.relplot(x ='Survived', y = 'Fare', data = train_data )


# Survivals are more for people paying more than 100bucks as fare

# **Number of family members and Survival**

# In[74]:


train_data["No.Fm.Mem"] = train_data["SibSp"] + train_data["Parch"] 


# In[79]:


sns.swarmplot(x ='Survived', y = 'No.Fm.Mem', data = train_data )


# Rate of survival is more for people with less family members on board

# In[ ]:




