import numpy as np
import ast

def cos(v1, v2): 
    return np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2))) 

def euclidean(v1, v2): 
    v1=np.array(v1)
    v2=np.array(v2)
    return np.sqrt(np.sum((v1-v2)*(v1-v2)))

r1=[]
id1=[]
str1=[]
with open("result1.txt") as f:
#with open("r1.txt") as f:
    for l in f:
        r1+=[ast.literal_eval(l)]
with open("pairid.txt") as f:
#with open("id1.txt") as f:
    for l in f:
        id1+=[l.strip()]
with open("cleanpair.txt") as f:
#with open("id1.txt") as f:
    for l in f:
        str1+=[l.strip()]
r2=[]
id2=[]
str2=[]
with open("result2.txt") as f:
#with open("r2.txt") as f:
    for l in f:
        r2+=[ast.literal_eval(l)]
with open("idid.txt") as f:
#with open("id2.txt") as f:
    for l in f:
        id2+=[l.strip()]
with open("cleanid.txt") as f:
#with open("id1.txt") as f:
    for l in f:
        str2+=[l.strip()]
f=open("compare.txt",'w')
r1=zip(r1, id1,str1)
r2=zip(r2, id2,str2)
for i in r1:
    questionid=i[1]
    cosvalue=[]
    for j in r2:
        if euclidean(i[0], j[0])<0.0001:
            continue
        cosvalue+=[(cos(i[0],j[0]),j[1],j[2])]
    cosvalue.sort()
    cosvalue.reverse()
    count=0
    flag=False
    for h in cosvalue:
        if h[1]==questionid:
            flag=True
            break
        count+=1    
    if cosvalue!=[]:
        f.write(i[2]+"|||||"+cosvalue[0][2]+"\n")

    if cosvalue!=[] and flag==True:
        print(count)
        see=[]
        for i in range(count+2):
        #for i in range(len(cosvalue)):
            see+=[cosvalue[i][0]]

        print(see)
    if cosvalue!=[] and flag==False:
        print(len(r2))
