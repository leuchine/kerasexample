import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.constraints import maxnorm
from keras import backend as K
from keras.callbacks import ModelCheckpoint
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

def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)

def loss(y_true, y_pred): 
        global output_feature;
        y_true_positive=y_true[:,:output_feature]
        y_true_negative=y_true[:,output_feature:]
        return 0.1*cosine_proximity(y_true_negative, y_pred)-cosine_proximity(y_true_positive, y_pred)
"""
"""
def cosine_proximity(y_true, y_pred):
    assert K.ndim(y_true) == 2
    assert K.ndim(y_pred) == 2
    y_true = K.l2_normalize(y_true, axis=1)
    y_pred = K.l2_normalize(y_pred, axis=1)
    return -K.mean(y_true * y_pred, axis=1)
"""
"""
def loss(y_true, y_pred):
        global output_feature;
        y_pred_positive=y_pred[:,:output_feature]
        y_pred_label=y_pred[:,output_feature:2*output_feature]
        y_pred_negative=y_pred[:,2*output_feature:]
        l=K.mean(y_true-y_true, axis=1)
        return -cosine_proximity(y_pred_positive, y_pred_negative)+cosine_proximity(y_pred_positive, y_pred_label)+l
        #return K.sum(y_pred_positive*y_true, axis=1)
"""
def euclidean(v1, v2): 
    return np.sqrt(np.sum((v1-v2)*(v1-v2)))

def cosine_proximity(y_true, y_pred):
    assert K.ndim(y_true) == 2
    assert K.ndim(y_pred) == 2
    y_true = K.l2_normalize(y_true, axis=1)
    y_pred = K.l2_normalize(y_pred, axis=1)
    return K.sum(y_true * y_pred, axis=1)

def mean_squared_error(y_true, y_pred):
    return K.square(K.sum((y_pred - y_true)*(y_pred - y_true), axis=-1))

def loss(y_true, y_pred):
        global output_feature;
        y_pred_positive=y_pred[:,:output_feature]
        y_pred_label=y_pred[:,output_feature:2*output_feature]
        y_pred_negative1=y_pred[:,2*output_feature:3*output_feature]
        y_pred_negative2=y_pred[:,3*output_feature:4*output_feature]
        y_pred_negative3=y_pred[:,4*output_feature:5*output_feature]
        y_pred_negative4=y_pred[:,5*output_feature:6*output_feature]
        y_pred_negative5=y_pred[:,6*output_feature:7*output_feature]
        y_pred_negative6=y_pred[:,7*output_feature:8*output_feature]
        l=K.min(y_true-y_true, axis=1)
        return K.maximum(cosine_proximity(y_pred_positive, y_pred_negative1)-cosine_proximity(y_pred_positive, y_pred_label)+0.3, 0.)\
        +K.maximum(cosine_proximity(y_pred_positive, y_pred_negative2)-cosine_proximity(y_pred_positive, y_pred_label)+0.3, 0.)\
        +K.maximum(cosine_proximity(y_pred_positive, y_pred_negative3)-cosine_proximity(y_pred_positive, y_pred_label)+0.3, 0.)\
        +K.maximum(cosine_proximity(y_pred_positive, y_pred_negative4)-cosine_proximity(y_pred_positive, y_pred_label)+0.3, 0.)\
        +K.maximum(cosine_proximity(y_pred_positive, y_pred_negative5)-cosine_proximity(y_pred_positive, y_pred_label)+0.3, 0.)\
        +K.maximum(cosine_proximity(y_pred_positive, y_pred_negative6)-cosine_proximity(y_pred_positive, y_pred_label)+0.3, 0.)\
        +K.maximum(0.98-cosine_proximity(y_pred_positive, y_pred_label), 0.)+l
"""
def loss(y_true, y_pred):
        global output_feature;
        y_pred_positive=y_pred[:,:output_feature]
        y_pred_label=y_pred[:,output_feature:2*output_feature]
        y_pred_negative=y_pred[:,2*output_feature:]
        l=K.min(y_true-y_true, axis=1)
        return cosine_proximity(y_pred_positive, y_pred_negative)-cosine_proximity(y_pred_positive, y_pred_label)+l
        #return K.maximum(cosine_proximity(y_pred_positive, y_pred_negative)-cosine_proximity(y_pred_positive, y_pred_label)+0.8, 0.)+l
        #return -cosine_proximity(y_pred_positive, y_pred_label)+l
"""
output_feature=300
input_feature=300
length=8
batch_size = 128
dictionary_size=6001
features=[]
labels=[]
labels_negative1=[]
labels_negative2=[]
labels_negative3=[]
labels_negative4=[]
labels_negative5=[]
labels_negative6=[]
answers=[]
embedding_size=300
f=open('vectorpair.txt')
#f=open('vectorpair2.txt')
for line in f:
    record=line.split('|||')
    if len(record) != length:
        print((record))
        continue
    features+=[ast.literal_eval(record[0])]
    labels+=[ast.literal_eval(record[1])]
    labels_negative1+=[ast.literal_eval(record[2])]
    labels_negative2+=[ast.literal_eval(record[3])]
    labels_negative3+=[ast.literal_eval(record[4])]
    labels_negative4+=[ast.literal_eval(record[5])]
    labels_negative5+=[ast.literal_eval(record[6])]
    labels_negative6+=[ast.literal_eval(record[7])]

f=open('vectorid.txt')
#f=open('vectorid2.txt')
for line in f:
    answers+=[ast.literal_eval(line)]
features=np.array(features)
labels=np.array(labels)
labels_negative1=np.array(labels_negative1)
labels_negative2=np.array(labels_negative2)
labels_negative3=np.array(labels_negative3)
labels_negative4=np.array(labels_negative4)
labels_negative5=np.array(labels_negative5)
labels_negative6=np.array(labels_negative6)
answers=np.array(answers)
features = sequence.pad_sequences(features, maxlen=input_feature)
labels=sequence.pad_sequences(labels, maxlen=input_feature)
labels_negative1=sequence.pad_sequences(labels_negative1, maxlen=input_feature)
labels_negative2=sequence.pad_sequences(labels_negative2, maxlen=input_feature)
labels_negative3=sequence.pad_sequences(labels_negative3, maxlen=input_feature)
labels_negative4=sequence.pad_sequences(labels_negative4, maxlen=input_feature)
labels_negative5=sequence.pad_sequences(labels_negative5, maxlen=input_feature)
labels_negative6=sequence.pad_sequences(labels_negative6, maxlen=input_feature)
answers=sequence.pad_sequences(answers, maxlen=input_feature)
print('Build model...')
model = Graph()
model.add_input(name='input1', input_shape=(input_feature,), dtype='float' )
model.inputs["input1"].input = T.imatrix()
model.add_input(name='input2', input_shape=(input_feature,), dtype='float')
model.inputs["input2"].input = T.imatrix()
model.add_input(name='input3', input_shape=(input_feature,), dtype='float')
model.inputs["input3"].input = T.imatrix()
model.add_input(name='input4', input_shape=(input_feature,), dtype='float')
model.inputs["input4"].input = T.imatrix()
model.add_input(name='input5', input_shape=(input_feature,), dtype='float')
model.inputs["input5"].input = T.imatrix()
model.add_input(name='input6', input_shape=(input_feature,), dtype='float')
model.inputs["input6"].input = T.imatrix()
model.add_input(name='input7', input_shape=(input_feature,), dtype='float')
model.inputs["input7"].input = T.imatrix()
model.add_input(name='input8', input_shape=(input_feature,), dtype='float')
model.inputs["input8"].input = T.imatrix()

shared_dense1=Embedding(dictionary_size, embedding_size, input_length=input_feature,init='he_uniform')
shared_dense2=LSTM(embedding_size,init='he_uniform')
shared_dense3=BatchNormalization()
shared_dense4=Dense(50,W_constraint = maxnorm(1.2))
shared_dense5=Dense(50,W_constraint = maxnorm(1.2))
shared_dense6=Dense(50,W_constraint = maxnorm(1.2))
shared_dense7=Dense(output_feature,W_constraint = maxnorm(1.2))
model.add_shared_node(shared_dense1, name="shared_dense1", inputs=['input1','input2','input3','input4','input5','input6','input7','input8'], merge_mode=None)
model.add_shared_node(shared_dense2, name="shared_dense2", inputs=["shared_dense1"], merge_mode=None)
model.add_shared_node(shared_dense3, name="shared_dense3", inputs=["shared_dense2"], merge_mode=None)
model.add_shared_node(shared_dense4, name="shared_dense4", inputs=["shared_dense3"], merge_mode=None)
model.add_shared_node(shared_dense5, name="shared_dense5", inputs=["shared_dense4"], merge_mode=None)
model.add_shared_node(shared_dense6, name="shared_dense6", inputs=["shared_dense5"], merge_mode=None)
model.add_shared_node(shared_dense7, name="shared_dense7", inputs=["shared_dense6"], merge_mode='concat')
model.add_node(Activation('softplus'), name='out', input='shared_dense7')
model.add_output(name='output', input='out')
adam=Adam(clipnorm=0.09)
model.compile(adam, {'output': loss})
print("Train...")
checkpointer = ModelCheckpoint(filepath="./weight/weights.{epoch:02d}.hdf5", verbose=1, save_best_only=False)
model.fit({'input1': features, 'input2': labels, 'input3': labels_negative1,'input4': labels_negative2,'input5': labels_negative3,'input6': labels_negative4,'input7': labels_negative5,'input8': labels_negative6, 'output': features},
          batch_size=batch_size, callbacks=[checkpointer],
          nb_epoch=30)
result=np.array(model.predict({'input1': features, 'input2': features, 'input3': features, 'input4': features, 'input5': features, 'input6': features, 'input7': features, 'input8': features}, batch_size=batch_size)['output'])
a=result[:, :output_feature]
result=np.array(model.predict({'input1': answers, 'input2': answers, 'input3': answers, 'input4': answers, 'input5': answers, 'input6': answers, 'input7': answers, 'input8': answers}, batch_size=batch_size)['output'])
b=result[:, :output_feature]
with open('result1.txt', 'w') as ff:
    for i in a:
        ff.write(str(i.tolist()))
        ff.write('\n')
with open('result2.txt', 'w') as ff:
    for i in b:
        ff.write(str(i.tolist()))
        ff.write('\n')
json_string = model.to_json()
open('my_model_architecture.json', 'w').write(json_string)
model.save_weights('my_model_weights.h5', overwrite=True)
