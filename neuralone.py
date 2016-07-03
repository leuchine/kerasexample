import numpy as np
import math

def computeDerivative(o, y):
    return (-np.sum(o*o)*y+np.sum(y*o)*o)/np.sqrt(np.sum(y*y))/np.power(np.sum(o*o), 1.5)

dimension=2
x=np.ones(dimension)
#y=np.ones(dimension)
y=np.array([1,1])
W=np.ones((dimension, dimension))
b=np.zeros((1,dimension))
o=np.dot(W, x)+b
y1=np.array([2,3])
y2=np.array([4,29])
print(o)
print(np.sum(y*o)/np.sqrt(np.sum(y*y))/np.sqrt(np.sum(o*o)))

db=computeDerivative(o,y)-computeDerivative(o,y1)-computeDerivative(o,y2)

print('db')
print(db)

dW=np.ones((dimension, dimension))
for i in range(dimension):
    for j in range(dimension):
        dW[i][j]=db[0][i]*x[j]
epsilon=0.001
W[0][0]=W[0][0]+epsilon
o=np.dot(W, x)+b
loss2=np.sum(y*o)/np.sqrt(np.sum(y*y))/np.sqrt(np.sum(o*o))-np.sum(y1*o)/np.sqrt(np.sum(y1*y1))/np.sqrt(np.sum(o*o))-np.sum(y2*o)/np.sqrt(np.sum(y2*y2))/np.sqrt(np.sum(o*o))
W[0][0]=W[0][0]-2*epsilon
o=np.dot(W, x)+b
loss1=np.sum(y*o)/np.sqrt(np.sum(y*y))/np.sqrt(np.sum(o*o))-np.sum(y1*o)/np.sqrt(np.sum(y1*y1))/np.sqrt(np.sum(o*o))-np.sum(y2*o)/np.sqrt(np.sum(y2*y2))/np.sqrt(np.sum(o*o))
print("loss")
print((loss2-loss1)/(2*epsilon))


