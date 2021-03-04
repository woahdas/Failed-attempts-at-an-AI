#An AI that classifies Tweets to see if they're about actual disasters or not.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from keras.layers import Embedding
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense

##eliminating spaces

#df= pd.read_csv("/kaggle/input/nlp-getting-started/train.csv", delimiter=',', header=None, skiprows=1, names=['id','keyword', 'location', 'text','target' ])

#from sklearn.model_selection import train_test_split
#train = train_test_split(df)
#test = train_test_split(pd.read_csv("/kaggle/input/nlp-getting-started/test.csv"))

features = train[['keyword', 'location', 'text']]
target='target'

#train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
#test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')


#different form of one-hot encoding
#trainht = pd.get_dummies(train)
#testht = pd.get_dummies(test)

#another way of one-hot encoding
embedding_layer = Embedding(1000, 64)
max_features = 10000 #number of words to consider as features
maxlen = 20 #cuts off the text after this many words from the max_features most common words

train1 = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
num_words=max_features #loads data as lists of integers

#pickup = train1[train1.isin(np.arange(peak1lower,peak1upper))]

train = pd.DataFrame(index='id', columns = train1.set_index(list(train1.columns)).index, dtype=bool)


model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))

model.add(Flatten()) #flattens the 3d tensor of embeddings into a 2d tensor of shape (samples, maxlen * 8)

model.add(Dense(1, activation='sigmoid')) #adds the classifier on top
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

model.fit(train[features],train[target])
