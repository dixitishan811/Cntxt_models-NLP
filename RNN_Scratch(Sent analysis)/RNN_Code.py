import numpy as np
import pandas as pd

### Pre-processors ###
import nltk
from sklearn import preprocessing
from nltk.corpus import stopwords
from nltk.stem.porter import *
ps = PorterStemmer()
from nltk.tokenize.treebank import TreebankWordDetokenizer

import warnings
warnings.filterwarnings("ignore")




### Data pre-processing ###

raw_data = pd.read_csv('train.csv')
raw_test = pd.read_csv('test.csv')

combi = raw_data.append(raw_test, ignore_index=True)

combi['tweet'] = combi['tweet'].str.replace('http\S+|www.\S+', '', case=False)
combi['tweet'] = combi['tweet'].str.replace('@\S+', '', case=False)
combi['tweet'] = combi['tweet'].str.replace("-", ' ', case=False)
combi['tweet'] = combi['tweet'].str.replace("#", ' ', case=False)
combi['tweet'] = combi['tweet'].str.replace('$\S+', '', case=False)

combi['tweet'] = combi['tweet'].apply(lambda x: str(x).lower())

combi['tweet'] = combi['tweet'].apply(
    lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

combi['tweet'] = combi['tweet'].str.replace("[^a-zA-Z#]", " ")

combi['tokenized'] = combi['tweet'].apply(lambda x: str(x).split())

combi['tokenized'] = combi['tokenized'].apply(
    lambda x: [ps.stem(w) for w in x])

combi['tokenized'] = combi['tokenized'].apply(
    lambda x: [word for word in x if word not in stopwords.words('english')])




train = combi[0:6336]
val=combi[6336:7920]


### Helper Functions ###


vocab = list(set([w for text in combi.tweet for w in text.split(' ')]))
vocab_size = len(vocab)
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}


def createInputs(text):

    inputs = []
    for w in text.split(' '):
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)
    return inputs
    
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
### Dims ###    

n_r = 64
n_x = vocab_size
n_y = 2


def intilization(n_r, n_x, n_y):
    params = {
        'Wx': np.random.rand(n_r, n_x) * 0.01,
        'Wr': np.random.rand(n_r, n_r) * 0.01,
        'Wy': np.random.rand(n_y, n_r) * 0.01,
        'br': np.zeros((n_r, 1)),
        'by': np.zeros((n_y, 1))
    }

    return params
    
    
def forward_prop(inputs, params):
    Wr = params['Wr']
    Wx = params['Wx']
    Wy = params['Wy']
    br = params['br']
    by = params['by']
    curr = np.zeros((Wr.shape[0], 1))
    prev_inputs = inputs
    prev = {0: curr}
    for i, inp in enumerate(inputs):
        curr = np.tanh(np.dot(Wx, inp) + np.dot(Wr, curr) + br)
        prev[i + 1] = curr
    y = softmax(np.dot(Wy,(curr)) + by)

    return y, params, prev, prev_inputs
    
    
    
 ### BPTT ###
 
def back_prop(d_y, params, prev, prev_inputs, learning_rate=0.001):

    Wr = params['Wr']
    Wx = params['Wx']
    Wy = params['Wy']
    br = params['br']
    by = params['by']

    n = len(prev_inputs)
    d_Wy = np.dot(d_y, prev[n].T)
    d_by = d_y

    d_Wr = np.zeros(Wr.shape)
    d_Wx = np.zeros(Wx.shape)
    d_br = np.zeros(br.shape)

    #Deriavtive for last hidden unit
    d_r = np.dot(Wy.T, d_y)

    for t in reversed(range(n)):

        dv = ((1 - prev[t + 1]**2) * d_r)

        d_br += dv

        d_Wr += np.dot(dv, prev[t].T)

        d_Wx += np.dot(dv, prev_inputs[t].T)

        d_r = np.dot(Wr, dv)

    # Clip to prevent exploding gradients.
    for d in [d_Wx, d_Wr, d_Wy, d_r, d_y]:
        np.clip(d, -10, 10, out=d)

    Wr -= learning_rate * d_Wr
    Wx -= learning_rate * d_Wx
    Wy -= learning_rate * d_Wy
    br -= learning_rate * d_br
    by -= learning_rate * d_by

    params.update({'Wx': Wx, 'Wr': Wr, 'Wy': Wy, 'br': br, 'by': by})
    return params
    
    
    
###    Train   ###

param = intilization(n_r, n_x, n_y)

ls = []
for epochs in range(10):
    loss = 0
    val_correct = 0
    for idx, x in enumerate(train['tweet']):
        inp = createInputs(x)
        label = int(combi.label[idx])

        y, params, prev, prev_inputs = forward_prop(inp, param)
        loss -= np.log(y[label])
        val_correct += int(np.argmax(y) == label)
        ls.append(np.argmax(y))
        d_y = y
        d_y[label] -= 1
        param = back_prop(d_y, params, prev, prev_inputs)
    print('--- Epoch %d' % (epochs + 1))
    print('Train:\tLoss %.3f | Accuracy: %.3f' %
          (loss / len(train), val_correct / len(train)))
    

### Validation Accuracy ###

loss = 0
val_correct = 0    
for idx, x in enumerate(val['tweet']):
        inp = createInputs(x)
        label = int(combi.label[idx])
    
        y, params, prev, prev_inputs = forward_prop(inp, param)
        loss -= np.log(y[label])
        val_correct += int(np.argmax(y) == label)
        ls.append(np.argmax(y))
print('Validation:\tLoss %.3f | Accuracy: %.3f' %(loss/len(val),val_correct / len(val)))
    
