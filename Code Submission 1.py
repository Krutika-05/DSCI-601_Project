#!/usr/bin/env python
# coding: utf-8

# In[30]:

# Import the data from the file data stored in local location
# Import classification report to evaluate initial models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
df=pd.read_csv('Users/krutikaparvatikar/Desktop/DSCI601_Project/data/train/en/truth.txt',sep=':::',header=None,engine='python')
df.columns=['id','label']


# In[22]:

# This gives a count of the labels present in the data.
# There are 2 label values in the data - 0 and 1
# The counts for each of the labels is 200.
df.label.value_counts()


# In[33]:
# Preprocessing the data is very crucial.
# In this we download packages required to preprocess the data

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
stop = stopwords.words('english')


# In[1]:
# This is the pre-processing step.
# All the preprocessing is done here.
# Removal of unnecessary tags and special characters in done in this method.

def preprocessing(text):
    text=text.str.replace('\d+', '')
    text=text.str.replace('RT','')
    text=text.str.replace('#USER#','')
    text=text.str.replace('#URL#','')
    text= text.str.lower()
    text = text.str.replace('[^\w\s]','')
    #text = text.apply(lambda x : [lemmatizer.lemmatize(y) for y in w_tokenizer.tokenize(x)])
    #text = text.apply(lambda x: [item for item in x if item not in stop])
    #text = text.apply(lambda x : " ".join(x))
    return text


# In[2]:

# Since the data is xml file we need to standarize the file type by labeling as train followed by the file number.
# This adds uniformity to data sets. 
df.id='train/en/'+df.id.astype(str)+'.xml'
df.head()


# In[5]:
# Converting XML files to text files.
# XML files have text placed inside them and to extract the text which are the tweets or posts, we need to convert the files to text.
# After converting the files, we obtain two columns - text and id
# Id is the label - 0 or 1 where 0 is not hateful tweet and 1 is hateful tweet.

import xml.etree.ElementTree as ET
def reader(df,ground=True):
    data=[]
    for x in df.iterrows():
        tree = ET.parse(x[1].id)
        root = tree.getroot()
        text=[x.text for x in root[0]]
        if ground:
        label=[x[1].label]*len(text)
        data.append(pd.DataFrame(zip(text,label),columns=['text','label']))
        else:
        data.append(pd.DataFrame(text,columns=['text']))
    return data


# In[2]:

# Sending this newly obtained data to the dataframe df.
data=reader(df)


# In[2]:


data[0].head()


# In[33]:
# Train test split to train the models.

from sklearn.model_selection import train_test_split
trainx,valx=train_test_split(data,test_size=0.1,)
trainx,testx=train_test_split(trainx,test_size=0.1)




