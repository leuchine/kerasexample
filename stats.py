import numpy as np
import math
import ast

features=[]
labels=[]
f=open('dataset2.txt')
for line in f:
    record=line.split('|')
    features+=[ast.literal_eval(record[0])]
    labels+=[ast.literal_eval(record[1])]
sum=0
s=0
for i in features:
    for j in i:
        sum+=abs(j)
        s+=j
print("Finish reading")
print(s/sum)
