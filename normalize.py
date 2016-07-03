import numpy as np
import math
import ast

f=open('word2vecvectors.txt')
w=open('newword2vecvectors.txt', 'w')
for line in f:
    if 'answer' in line:
        w.write(line)
    else:
        data=np.array(ast.literal_eval('['+line+']'))
        data=data/np.sqrt(np.sum(data*data))
        w.write(str(data.tolist()).strip('[').strip(']'))
        w.write('\n')
w.close()
