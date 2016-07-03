import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

"""
#pos
def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']

def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']

def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']

def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return wn.NOUN

def clean(text):
    global lemmatizer
    global stops
    #remove special character
    text=re.sub("[^a-zA-Z0-9]", " ", text)
    text=text.lower() 
    text=nltk.word_tokenize(text)
    pos=nltk.pos_tag(text)
    meaningful_words=[]
    for key, value in pos:
        #remove stopwords
        if key not in stops:
            #lemmatization
            meaningful_words+=[lemmatizer.lemmatize(key, pos=penn_to_wn(value))]                            
    return( " ".join( meaningful_words )) 
"""
linelength=8
dictionary_size=6000
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = dictionary_size) 
sentences=[]
split1=[]
split2=[]
s1=file("cleanpair.txt")
for l in s1:
    sentences+=[l.strip()]
    split1+=[l.strip()]
s1=file("cleanid.txt")
for l in s1:
    sentences+=[l.strip()]
    split2+=[l.strip()]

train_data_features = vectorizer.fit(sentences)
dictionary={}
vocab = vectorizer.get_feature_names()
vocab=['||||']+vocab
#word to integer
for key, value in zip(vocab, range(len(vocab))):
    dictionary[key]=value
vocab=set(vocab)

newdataset=file('vectorpair.txt','w')
for i in range(len(split1)):
    feature=[]
    for word in split1[i].split():
        if word in vocab:
            feature+=[dictionary[word]]
    newdataset.write(str(feature))
    if (i+1)%8==0:
        newdataset.write('\n')
    else:
        newdataset.write('|||')
newdataset.close()

newdataset=file('vectorid.txt','w')
for i in range(len(split2)):
    feature=[]
    for word in split2[i].split():
        if word in vocab:
            feature+=[dictionary[word]]
    newdataset.write(str(feature))
    newdataset.write('\n')
newdataset.close()
