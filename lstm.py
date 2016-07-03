import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras import backend as K
import theano
import theano.tensor as T
import ast

"""
def loss(y_true, y_pred):
        global output_feature;
        y_true_positive=y_true[:,:output_feature]
        y_true_negative=y_true[:,output_feature:]
        y_pred = K.l2_normalize(y_pred, axis=1)
        y_true_positive = K.l2_normalize(y_true_positive, axis=1)
        y_true_negative = K.l2_normalize(y_true_negative, axis=1)
        return K.maximum(K.sum(y_pred*y_true_negative, axis=1) - K.sum(y_pred*y_true_positive, axis=1)+0.6, 0.)
"""
"""
def cosine_proximity(y_true, y_pred):
    assert K.ndim(y_true) == 2
    assert K.ndim(y_pred) == 2
    y_true = K.l2_normalize(y_true, axis=1)
    y_pred = K.l2_normalize(y_pred, axis=1)
    return -K.mean(y_true * y_pred, axis=1)

def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)

def loss(y_true, y_pred):
        global output_feature;
        y_true_positive=y_true[:,:output_feature]
        y_true_negative=y_true[:,output_feature:]
        return 0.1*cosine_proximity(y_true_negative, y_pred)-cosine_proximity(y_true_positive, y_pred)
"""
def loss(y_true, y_pred):
        global output_feature;
        y_true_positive=y_true[:,:output_feature]
        y_true_negative=y_true[:,output_feature:]
        y_pred = K.l2_normalize(y_pred, axis=1)
        y_true_positive = K.l2_normalize(y_true_positive, axis=1)
        y_true_negative = K.l2_normalize(y_true_negative, axis=1)
        #return -K.exp(K.sum(y_pred*y_true, axis=1))/K.exp(K.sum(y_pred*y_true_negative, axis=1))
        #(K.exp(K.sum(y_pred*y_true, axis=1))+ 
        return K.sum(y_pred*y_true_negative, axis=1)-K.sum(y_pred*y_true_positive, axis=1)

def cos(v1, v2): 
    return np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2))) 

output_feature=7
input_feature=7
length=3
batch_size = 1
dictionary_size=5000
features=[]
labels=[]
labels_negative=[]
embedding_size=7
f=open('newdataset.txt')
for line in f:
    record=line.split('|||')
    if len(record) != length:
        print((record))
        continue
    features+=[ast.literal_eval(record[0])]
    labels+=[ast.literal_eval(record[1])]
    labels_negative+=[ast.literal_eval(record[2])]
features=np.array(features)
labels=np.array(labels)
labels_negative=np.array(labels_negative)
features = sequence.pad_sequences(features, maxlen=input_feature)
labels=sequence.pad_sequences(labels, maxlen=input_feature)
labels_negative=sequence.pad_sequences(labels_negative, maxlen=input_feature)
X_train=features
X_test=features


print('Build model...')
model = Sequential()
model.add(Embedding(dictionary_size, embedding_size, input_length=input_feature))
model.add(LSTM(embedding_size))
model.add(Dense(300))
model.add(Activation('relu'))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dense(output_feature))
model.compile(loss=loss, optimizer='adam')
print("Train...")

y_test=y_train=np.concatenate((labels, labels_negative), axis=1)
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=160, validation_data=(X_test, y_test))
result=np.array(model.predict(X_train,batch_size=batch_size))
for i,j in zip(result, labels):
    print(cos(i,j))
for i,j in zip(result, labels_negative):
    print(cos(i,j))
