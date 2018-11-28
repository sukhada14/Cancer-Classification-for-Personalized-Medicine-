
# coding: utf-8

# In[1]:


import pandas as pd         
import seaborn as sb        #for visualization of data
from sklearn.model_selection import train_test_split #to split train test data
get_ipython().magic('matplotlib inline')


# In[2]:


data = pd.read_csv("/media/kirti/study/ML/Project/training_variants.csv")


# In[3]:


print(data.head())


# In[4]:


data.shape


# In[5]:


input = data[['ID', 'Gene','Variation']]
input.head()


# In[14]:


labels = data[['Class']]


# In[15]:


labels.head(10)


# In[16]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
input = input.apply(le.fit_transform)
input.head()


# In[17]:


input.head()


# In[18]:


"""
UPTILL now we are done with processing of gene and variations
so now we need to process text data
"""


# In[19]:


text = pd.read_csv("/media/kirti/study/ML/Project/training_text",sep='\|\|', header=None, skiprows=1, engine="python",names=["ID","Text"])


# In[23]:


#Convert a collection of raw documents to a matrix of TF-IDF features
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
vect = TfidfVectorizer(analyzer="word", tokenizer=nltk.word_tokenize,
    preprocessor=None, stop_words='english', max_features=1000, max_df= 0.8)    


# In[24]:


bag_of_words = vect.fit(text["Text"])
bag_of_words = vect.transform(text["Text"])
print(bag_of_words.shape)


# In[26]:


import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
svd = TruncatedSVD(n_components=200, n_iter=25, random_state=12)
truncated_bag_of_words = svd.fit_transform(bag_of_words)
print(truncated_bag_of_words.shape)


# In[16]:


# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.feature_extraction.text import CountVectorizer
# svd = PCA()
# PCA_bag_of_words = svd.fit_transform(bag_of_words.toArray())
# print(PCA_bag_of_words.shape)


# In[28]:


#for merging we need one common attribute i.e. ID
df = pd.DataFrame(truncated_bag_of_words)
newfeature = {'ID': range(0,3321)}
df['ID'] = pd.DataFrame(data =newfeature)
df.head()


# In[29]:


final = pd.merge(input, df, how="left",on="ID")   #Gene ,Mutations ,Text(truncatedSVD)


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(final, labels, test_size= 0.25, random_state=5)


# In[31]:


X_train=X_train.drop('ID', axis=1)
X_test=X_test.drop('ID', axis=1)


# In[32]:


#representation of testing data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train['Gene'], X_train['Variation'], y_train['Class'], c='g', marker='o')

ax.set_xlabel('Gene',fontsize=14)
ax.set_ylabel('Variation',fontsize=14)
ax.set_zlabel('Class',fontsize=14)
plt.title("Training data", fontsize=18)
plt.show()


# In[35]:


"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
"""

fig = plt.figure()
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111, projection='3d')
# x,y,z,color=red,marker=
ax.scatter(X_test['Gene'], X_test['Variation'], y_test['Class'], c='r', marker='o')

ax.set_xlabel('Gene',fontsize=14)
ax.set_ylabel('Variation',fontsize=14)
ax.set_zlabel('Class',fontsize=14)
plt.title("Testing data", fontsize=18)
plt.show()


# In[36]:


"""
%matplotlib inline
import matplotlib.pyplot as   plt
import seaborn as sns
"""

plt.figure(figsize=(12,8))
sb.countplot(x="Class", data=y_train,palette="Blues_d") #y_train
plt.ylabel('Frequency', fontsize=14)
plt.xlabel('Class', fontsize=14)
plt.title("Distribution of genetic mutation classes in training data", fontsize=18)
plt.show()


# In[37]:


"""
%matplotlib inline
import matplotlib.pyplot as   plt
import seaborn as sb
"""
plt.figure(figsize=(12,8))
sb.countplot(x="Class", data=y_test,palette="Blues_d") #y_train
plt.ylabel('Frequency', fontsize=14)
plt.xlabel('Class', fontsize=14)
plt.title("Distribution of genetic mutation classes in testing data", fontsize=18)
plt.show()


# In[21]:


X_train.head()


# In[22]:


X_test.head()


# In[36]:


y_train.head()


# In[74]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[78]:


from sklearn.svm import LinearSVC
lr = LinearSVC()


# In[40]:


from sklearn.ensemble import RandomForestClassifier
lr = RandomForestClassifier()


# In[46]:


from sklearn.ensemble import AdaBoostClassifier
lr = AdaBoostClassifier()


# In[70]:


from sklearn.neighbors import KNeighborsClassifier
lr=KNeighborsClassifier(n_neighbors=1)


# In[41]:


lr.fit(X_train,y_train)


# In[42]:


y_pred = lr.predict(X_test)


# In[48]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[ ]:


#Confusion matrix
#or curves


# In[ ]:


"""
RandomForestClassifier  :0.60409145607701564

K-NearestNeighbors      :0.45487364620938631

LogisticRegression      :0.56317689530685922

LinearSVC               :0.26353790613718414
"""


# y_test.head()
