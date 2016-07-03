import numpy as np
import math
import ast

def computeQuality(W, b, features, labels):
    quality=0
    for i in range(len(features)):
        x=features[i]
        y=labels[i]
        x=np.array(x)
        y=np.array(y)
        o=np.dot(W, x)+b
        quality+=np.sum(y*o)/np.sqrt(np.sum(y*y))/np.sqrt(np.sum(o*o))
    print(quality)

features=[]
labels=[]
f=open('word2vecdataset.txt')
for line in f:
    record=line.split('|')
    features+=[ast.literal_eval('['+record[0]+']')]
    labels+=[ast.literal_eval("["+record[1]+"]")]
print("Finish reading")
M=len(features)
dimension=300
a=32.0/M
W=np.random.rand(dimension, dimension)
b=np.random.rand(1,dimension)
computeQuality(W,b,features, labels)
for m in range(20):
    for n in range(len(features)):
        x=features[n]
        y=labels[n]
        x=np.array(x)
        y=np.array(y)
        o=np.dot(W, x)+b
        db=(-np.sum(o*o)*y+np.sum(y*o)*o)/np.sqrt(np.sum(y*y))/np.power(np.sum(o*o), 1.5)
        dW=np.ones((dimension, dimension))
        for i in range(dimension):
            for j in range(dimension):
                dW[i][j]=db[0][i]*x[j]
        b=b-a*db
        W=W-a*dW
    print("Finish phase: "+str(m))
    computeQuality(W, b, features, labels)

w=open('word2vecprediction.txt','w')
for i in features:
    o=np.dot(W, i)+b
    w.write(str(o))
    w.write('\n')
w.close()
"""
gradient check
"""
"""
epsilon=0.001
W[0][0]=W[0][0]+epsilon
o=np.dot(W, x)+b
loss2=np.sum((y-o)*(y-o))
W[0][0]=W[0][0]-2*epsilon
o=np.dot(W, x)+b
loss1=np.sum((y-o)*(y-o))
print((loss2-loss1)/(2*epsilon))
"""
