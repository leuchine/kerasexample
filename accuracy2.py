import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras import backend as K
import theano
import theano.tensor as T
import ast
def cos(v1, v2): 
    return np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2))) 
def euclidean(v1, v2): 
    v1=np.array(v1)
    v2=np.array(v2)
    return np.sqrt(np.sum((v1-v2)*(v1-v2)))
f=file("result.txt")

left=[]
right=[]
for l in f:
    line=ast.literal_eval(l)
    a=line[:200]
    b=line[200:400]
    c=line[400:]
    left+=[a]
    right+=[b]
for i in range(len(left)):
    sum=0
    dis=cos(left[i],right[i])
    for j in right:
        if dis< cos(left[i],j):
            sum+=1
    print(sum)
