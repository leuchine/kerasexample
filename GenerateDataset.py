import numpy as np
import ast

vector=open('word2vecvectors.txt')
vectors=vector.read().split('\n')
answervector=open('answervector.txt','w')
dictionary={}
print(len(vectors))
for i in range(len(vectors)/2):
		questionid=vectors[2*i].split('_')[0]
		answervector.write(questionid+"|"+vectors[2*i+1])
		try:
			l=dictionary.get(questionid, [])
			l+=[vectors[2*i+1]]
			dictionary[questionid]=l
		except AttributeError:
			print(questionid)
			print(dictionary.get(questionid, []))
			exit(0)
		answervector.write('\n')
answervector.close()

with open('word2vecdataset.txt','w') as f:
	for questionid, answers in dictionary.items():
		if len(answers)<=1:
			continue
		for i in range(len(answers)):
			for j in range(i+1, len(answers)):
				f.write(answers[i]+"|"+answers[j]+"|"+questionid+"\n")
