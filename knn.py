#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
irisData = load_iris()


# In[25]:


X= irisData.data
y = irisData.target
print(X)
print(y)


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state=42)


# In[31]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[32]:


knn.fit(X_train,y_train)


# In[33]:


knn.predict(X_test)
knn.score(X_test, y_test)

