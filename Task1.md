

```python
import os

import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cosine, euclidean
np.set_printoptions(precision=4)

os.listdir('data')
```




    ['dist_sim_data.txt',
     'EN-wform.w.2.ppmi.svd.500.rcv_vocab.txt',
     'EN_syn_verb.txt',
     'GoogleNews-vectors-rcv_vocab.txt',
     'SAT-package-V3.txt']



## Caculate the count matrix from file


```python
word_dict = {}
word_count = defaultdict(lambda: defaultdict(int))

def get_word_in_dict(word):
    if word not in word_dict:
        word_dict[word] = len(word_dict)
    return word_dict[word]

def parse_sentence(words):
    prev = ''
    for word in words:
        if prev != '':
            word_count[prev][word] += 1
            word_count[word][prev] += 1
        if word not in word_dict:
            word_dict[word] = len(word_dict)
        prev = word

with open(os.path.join('data', 'dist_sim_data.txt')) as f:
    for sentence in f:
        prev_word = ''
        parse_sentence(sentence.strip('\n').split(' '))

num_words = len(word_dict)
count_matrix = np.zeros((num_words, num_words))

for word1 in word_dict:
    for word2 in word_count[word1]:
        idx1, idx2 = word_dict[word1], word_dict[word2]
        count_matrix[idx1][idx2] = word_count[word1][word2]
```


```python
count_matrix
```




    array([[0., 8., 4., 9., 5., 3., 4.],
           [8., 0., 2., 0., 0., 0., 2.],
           [4., 2., 0., 0., 2., 0., 0.],
           [9., 0., 0., 0., 0., 3., 1.],
           [5., 0., 2., 0., 0., 0., 1.],
           [3., 0., 0., 3., 0., 0., 0.],
           [4., 2., 0., 1., 1., 0., 0.]])




```python
word_dict
```




    {'the': 0, 'men': 1, 'feed': 2, 'dogs': 3, 'women': 4, 'bite': 5, 'like': 6}



## Calculate the co-occurence probability matrix


```python
cooc_prop_matrix = count_matrix * 10 + 1
for row in cooc_prop_matrix:
    row /= np.sum(row)
cooc_prop_matrix
```




    array([[0.003 , 0.2404, 0.1217, 0.27  , 0.1513, 0.092 , 0.1217],
           [0.6378, 0.0079, 0.1654, 0.0079, 0.0079, 0.0079, 0.1654],
           [0.4713, 0.2414, 0.0115, 0.0115, 0.2414, 0.0115, 0.0115],
           [0.6642, 0.0073, 0.0073, 0.0073, 0.0073, 0.2263, 0.0803],
           [0.5862, 0.0115, 0.2414, 0.0115, 0.0115, 0.0115, 0.1264],
           [0.4627, 0.0149, 0.0149, 0.4627, 0.0149, 0.0149, 0.0149],
           [0.4713, 0.2414, 0.0115, 0.1264, 0.1264, 0.0115, 0.0115]])




```python
word_prop = np.sum(count_matrix, axis=1)
word_prop /= np.sum(word_prop)
word_prop
```




    array([0.375 , 0.1364, 0.0909, 0.1477, 0.0909, 0.0682, 0.0909])




```python
ppmi = np.empty((num_words, num_words))
for i in range(num_words):
    for j in range(num_words):
        ppmi[i][j] = max(np.log(cooc_prop_matrix[i][j] / word_prop[i] / word_prop[j]), 0)
```


```python
ppmi[word_dict['dogs']]
```




    array([2.4841, 0.    , 0.    , 0.    , 0.    , 3.112 , 1.7882])




```python
count_matrix[word_dict['dogs']]
```




    array([9., 0., 0., 0., 0., 3., 1.])



Here we can see that the ppmi vector gives a higher weight to "bite" and "like" over "the". 

This is right because "the" can (in both real English and this small corpus) appear together with **almost any noun** while the later two verbs are more informative on the meaning of the word "dog".


```python
def calc_dist(word1, word2, matrix):
    if word1 not in word_dict or word2 not in word_dict:
        return
    dist = euclidean(matrix[word_dict[word1]], matrix[word_dict[word2]])
    print('The distance between %s and %s is %.3f'%(word1, word2, dist))
```

### Distance calculated by using ppmi


```python
calc_dist('women', 'men', ppmi)
calc_dist('women', 'dogs', ppmi)
calc_dist('dogs', 'men', ppmi)
calc_dist('feed', 'like', ppmi)
calc_dist('feed', 'bite', ppmi)
calc_dist('like', 'bite', ppmi)
```

    The distance between women and men is 1.107
    The distance between women and dogs is 4.328
    The distance between dogs and men is 4.128
    The distance between feed and like is 2.334
    The distance between feed and bite is 5.299
    The distance between like and bite is 3.624


We can see the distances calculated agree with our intuition.


```python
from scipy.linalg import svd

U, E, Vt = svd(ppmi, full_matrices=False)
U = np.matrix(U)
E = np.matrix(np.diag(E))
Vt = np.matrix(Vt)
V = Vt.T
reduced_ppmi = ppmi * V[:, 0:3]
```


```python
reduced_ppmi.shape
# ppmi was a 7x7 matrix
```




    (7, 3)



### Distance calculated by using SVD-reduced ppmi


```python
calc_dist('women', 'men', reduced_ppmi)
calc_dist('women', 'dogs', reduced_ppmi)
calc_dist('dogs', 'men', reduced_ppmi)
calc_dist('feed', 'like', reduced_ppmi)
calc_dist('feed', 'bite', reduced_ppmi)
calc_dist('like', 'bite', reduced_ppmi)
```

    The distance between women and men is 0.851
    The distance between women and dogs is 2.704
    The distance between dogs and men is 2.185
    The distance between feed and like is 2.052
    The distance between feed and bite is 5.101
    The distance between like and bite is 3.558


We can see the reduced ppmi matrix still keep the information needed.
