#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
directory_train_neg = 'C:/Users/support/Desktop/PyShiz/.ipynb_checkpoints/aclImdb/train/neg'
reviews_train=[]
for filename in os.listdir(directory_train_neg):
    filename = "C:/Users/support/Desktop/PyShiz/.ipynb_checkpoints/aclImdb/train/neg/%s" % filename
    for line in open(filename,'r',encoding='latin1'):
        reviews_train.append(line.strip())

directory_train_pos = 'C:/Users/support/Desktop/PyShiz/.ipynb_checkpoints/aclImdb/train/pos'
for filename in os.listdir(directory_train_pos):
    filename = "C:/Users/support/Desktop/PyShiz/.ipynb_checkpoints/aclImdb/train/pos/%s" % filename
    for line in open(filename,'r',encoding='latin1'):
        reviews_train.append(line.strip())


reviews_test=[]
directory_test_pos = 'C:/Users/support/Desktop/PyShiz/.ipynb_checkpoints/aclImdb/test/pos'
for filename in os.listdir(directory_test_pos):
    filename = "C:/Users/support/Desktop/PyShiz/.ipynb_checkpoints/aclImdb/test/pos/%s" % filename
    for line in open(filename,'r',encoding='latin1'):
        reviews_test.append(line.strip())

directory_test_neg = 'C:/Users/support/Desktop/PyShiz/.ipynb_checkpoints/aclImdb/test/neg'
for filename in os.listdir(directory_test_neg):
    filename = "C:/Users/support/Desktop/PyShiz/.ipynb_checkpoints/aclImdb/test/neg/%s" % filename
    for line in open(filename,'r',encoding='latin1'):
        reviews_test.append(line.strip())


# In[1]:


pip install re


# In[9]:


pip install regex


# In[15]:


import re

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE=re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews=[REPLACE_NO_SPACE.sub("",line.lower()) for line in reviews]
    reviews=[REPLACE_WITH_SPACE.sub("",line) for line in reviews]
    
    return reviews

reviews_train_clean=preprocess_reviews(reviews_train)
reviews_test_clean=preprocess_reviews(reviews_test)


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer

print(reviews_train_clean[-1])
print(reviews_test_clean[-1])
cv=CountVectorizer(binary=True)
cv.fit(reviews_train_clean)
X=cv.transform(reviews_train_clean)
X_test=cv.transform(reviews_test_clean)


# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target=[1 if i < 12500 else 0 for i in range(25000)]

X_train, X_val,y_train,y_val =train_test_split(X,target,train_size=0.75)

for c in [0.01,0.05,0.25,0.5,1]:
    
    lr=LogisticRegression(C=c)
    lr.fit(X_train,y_train)
    print("Accuracy for c=%s: %s" %(c,accuracy_score(y_val,lr.predict(X_val))))


# In[29]:


final_model=LogisticRegression(C=0.05)
final_model.fit(X,target)
print("Final Accuracy :%s" % accuracy_score(target,final_model.predict(X_test)))


# In[31]:


feature_to_coef={
    word: coef for word,coef in zip(
        cv.get_feature_names(),final_model.coef_[0]
    )
}

for best_positive in sorted(
    feature_to_coef.items(),
    key=lambda x: x[1],
    reverse=True)[:5]:
    print(best_positive)


# In[33]:


for best_negative in sorted(
    feature_to_coef.items(),
    key=lambda x: x[1])[:5]:
    print(best_negative)


# In[ ]:




