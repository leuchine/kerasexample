from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy
import ast
from pymongo import MongoClient

client=MongoClient()
db=client.qadb
documents=db.quora.find()
ids=[]
docs=[]
for document in documents:
    if document['question']==None:
        continue
    objectid=str(document['_id'])
    answer=document.get('answer', {}).items()
    answer.sort()
    description=document.get('description',"")
    if description==None:
        description=""
    question=document['question']+" "+description
    
    ids+=[objectid+"_question"]
    docs+=[LabeledSentence(question.split(),[objectid+"_question"])]
    for key, value in answer:
        ids+=[objectid+"_"+key]
        docs+=[LabeledSentence(value.split(),[objectid+"_"+key])]
print(len(docs))

f=open('vector.txt','w')
f2=open('id.txt','w')
model = Doc2Vec(min_count=1, window=10, size=30, sample=1e-4, negative=5, workers=8)
model.build_vocab(docs)
for epoch in range(100):
    print('Epoch %d' % epoch)
    model.train(docs)
for i in ids:
    f.write(str(model.docvecs[i].tolist()))
    f.write('\n')
    f2.write(i)
    f2.write('\n')
f.close()
f2.close()

"""
class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for uid, line in enumerate(open(self.filename)):
            yield LabeledSentence(line.split(), ['SENT_%s' % uid])

    def toArray(self):
        array=[]
        for uid, line in enumerate(open(self.filename)):
            array.append(LabeledSentence(line.split(), ['SENT_%s' % uid]))
        return array
"""
