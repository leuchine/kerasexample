import numpy as np
import math
import ast

def computequality(x, y):
        #return np.sum(x*y)/np.sqrt(np.sum(y*y))/np.sqrt(np.sum(x*x))
        return np.sqrt(np.sum((y-x)*(y-x)))

answerlist=[]
with open('answervector.txt') as f:
        for l in f:
                answerlist+=[(l.split('|')[0], ast.literal_eval('['+l.split('|')[1]+']'))]
dataset=open('word2vecdataset.txt')
prediction=open('output.txt')
datasets=dataset.read().split('\n')
predictions=prediction.read().split('\n')

evaluate=[]
for i in range(len(predictions)):
        evaluate+=[(datasets[i].split("|")[2], ast.literal_eval(predictions[i])[0], ast.literal_eval(datasets[i].split("|")[1]))]
for t in evaluate:
	result=[]
	for a in answerlist:
		x=np.array(t[1])
		y=np.array(a[1])
		quality=computequality(x,y)
		result.append((quality,a[0]))
	result.sort()
	#result.reverse()
	for i in range(len(result)):
		if t[0]==result[i][1]:
			x=np.array(t[2])
			y=np.array(t[1])
			quality=computequality(x,y)
			print(quality)
			print(i)
			break
