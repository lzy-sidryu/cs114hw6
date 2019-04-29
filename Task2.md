

```python
import pandas as pd
from collections import defaultdict
import numpy as np
import csv
import os

os.listdir('data')
```




    ['dist_sim_data.txt',
     'EN-wform.w.2.ppmi.svd.500.rcv_vocab.txt',
     'EN_syn_verb.txt',
     'GoogleNews-vectors-rcv_vocab.txt',
     'readme.md',
     'SAT-package-V3.txt']




```python
google_vec_df = pd.read_csv(os.path.join('data','GoogleNews-vectors-rcv_vocab.txt'), sep=' ', header=None)
google_vec_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>291</th>
      <th>292</th>
      <th>293</th>
      <th>294</th>
      <th>295</th>
      <th>296</th>
      <th>297</th>
      <th>298</th>
      <th>299</th>
      <th>300</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lundazi</td>
      <td>0.106934</td>
      <td>0.144531</td>
      <td>-0.081543</td>
      <td>0.111816</td>
      <td>-0.016846</td>
      <td>-0.075684</td>
      <td>-0.196289</td>
      <td>0.040283</td>
      <td>-0.359375</td>
      <td>...</td>
      <td>-0.273438</td>
      <td>-0.062012</td>
      <td>-0.152344</td>
      <td>-0.072754</td>
      <td>-0.129883</td>
      <td>0.052490</td>
      <td>-0.347656</td>
      <td>-0.055908</td>
      <td>0.056152</td>
      <td>0.196289</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Eket</td>
      <td>-0.250000</td>
      <td>-0.017944</td>
      <td>-0.082520</td>
      <td>-0.031128</td>
      <td>-0.143555</td>
      <td>-0.292969</td>
      <td>0.012756</td>
      <td>0.154297</td>
      <td>-0.229492</td>
      <td>...</td>
      <td>-0.051025</td>
      <td>0.165039</td>
      <td>-0.384766</td>
      <td>-0.433594</td>
      <td>-0.310547</td>
      <td>0.171875</td>
      <td>-0.460938</td>
      <td>-0.099121</td>
      <td>-0.120605</td>
      <td>-0.318359</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Asir</td>
      <td>-0.073242</td>
      <td>0.103027</td>
      <td>-0.175781</td>
      <td>0.102539</td>
      <td>0.283203</td>
      <td>0.080566</td>
      <td>0.023560</td>
      <td>-0.188477</td>
      <td>-0.333984</td>
      <td>...</td>
      <td>-0.180664</td>
      <td>-0.115234</td>
      <td>0.220703</td>
      <td>-0.049805</td>
      <td>-0.249023</td>
      <td>0.542969</td>
      <td>-0.128906</td>
      <td>-0.101074</td>
      <td>0.167969</td>
      <td>0.437500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Simha</td>
      <td>0.088379</td>
      <td>0.116211</td>
      <td>-0.137695</td>
      <td>0.121582</td>
      <td>0.129883</td>
      <td>-0.554688</td>
      <td>0.302734</td>
      <td>-0.124512</td>
      <td>0.002457</td>
      <td>...</td>
      <td>-0.132812</td>
      <td>0.140625</td>
      <td>-0.267578</td>
      <td>-0.122559</td>
      <td>-0.155273</td>
      <td>-0.123535</td>
      <td>-0.318359</td>
      <td>0.179688</td>
      <td>0.146484</td>
      <td>0.367188</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HRCP</td>
      <td>-0.316406</td>
      <td>0.023438</td>
      <td>0.158203</td>
      <td>0.034180</td>
      <td>-0.119629</td>
      <td>-0.134766</td>
      <td>0.142578</td>
      <td>0.029053</td>
      <td>-0.215820</td>
      <td>...</td>
      <td>0.074219</td>
      <td>0.011902</td>
      <td>0.008606</td>
      <td>-0.018677</td>
      <td>-0.013428</td>
      <td>0.289062</td>
      <td>-0.194336</td>
      <td>0.093262</td>
      <td>0.006927</td>
      <td>-0.063477</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 301 columns</p>
</div>




```python
google_words = list(google_vec_df[0])
google_word_dict = {k:v for v, k in enumerate(google_words)}
len(google_words)
```




    140922




```python
google_vecs = google_vec_df.drop(0, axis=1).values
google_vecs.shape
```




    (140922, 300)




```python
ppmi_df = pd.read_csv(os.path.join('data','EN-wform.w.2.ppmi.svd.500.rcv_vocab.txt'), header=None, sep=' ', quoting=csv.QUOTE_NONE)
ppmi_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>491</th>
      <th>492</th>
      <th>493</th>
      <th>494</th>
      <th>495</th>
      <th>496</th>
      <th>497</th>
      <th>498</th>
      <th>499</th>
      <th>500</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>neo-classic</td>
      <td>0.183098</td>
      <td>0.129868</td>
      <td>0.176383</td>
      <td>-0.053295</td>
      <td>-0.055520</td>
      <td>0.043304</td>
      <td>-0.133636</td>
      <td>0.092237</td>
      <td>-0.094436</td>
      <td>...</td>
      <td>0.014659</td>
      <td>0.001169</td>
      <td>0.036292</td>
      <td>-0.008851</td>
      <td>0.027492</td>
      <td>-0.009265</td>
      <td>-0.052609</td>
      <td>-0.044142</td>
      <td>-0.059494</td>
      <td>0.012797</td>
    </tr>
    <tr>
      <th>1</th>
      <td>auberge</td>
      <td>0.187866</td>
      <td>0.017577</td>
      <td>0.115123</td>
      <td>0.056515</td>
      <td>-0.142405</td>
      <td>0.116539</td>
      <td>-0.062118</td>
      <td>0.135252</td>
      <td>-0.078248</td>
      <td>...</td>
      <td>-0.009394</td>
      <td>0.013320</td>
      <td>0.001646</td>
      <td>0.027234</td>
      <td>0.011830</td>
      <td>0.013212</td>
      <td>0.007839</td>
      <td>0.024726</td>
      <td>-0.017699</td>
      <td>-0.014483</td>
    </tr>
    <tr>
      <th>2</th>
      <td>deeps</td>
      <td>0.284431</td>
      <td>0.153209</td>
      <td>0.247658</td>
      <td>0.010080</td>
      <td>-0.218465</td>
      <td>0.113874</td>
      <td>-0.183892</td>
      <td>0.058796</td>
      <td>-0.098555</td>
      <td>...</td>
      <td>0.016969</td>
      <td>0.034882</td>
      <td>-0.005492</td>
      <td>-0.063476</td>
      <td>-0.023179</td>
      <td>-0.047171</td>
      <td>-0.061298</td>
      <td>-0.054737</td>
      <td>-0.041229</td>
      <td>0.038749</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1997-2007</td>
      <td>0.160917</td>
      <td>0.017448</td>
      <td>0.272126</td>
      <td>0.139675</td>
      <td>0.033443</td>
      <td>0.131233</td>
      <td>-0.163524</td>
      <td>0.094085</td>
      <td>-0.220436</td>
      <td>...</td>
      <td>-0.000998</td>
      <td>0.061768</td>
      <td>-0.032469</td>
      <td>-0.010853</td>
      <td>-0.019737</td>
      <td>-0.016588</td>
      <td>-0.034730</td>
      <td>-0.011669</td>
      <td>0.017729</td>
      <td>0.037919</td>
    </tr>
    <tr>
      <th>4</th>
      <td>refectory</td>
      <td>0.278820</td>
      <td>0.083047</td>
      <td>0.156682</td>
      <td>0.047539</td>
      <td>-0.244677</td>
      <td>0.088986</td>
      <td>-0.088450</td>
      <td>0.180136</td>
      <td>0.123281</td>
      <td>...</td>
      <td>-0.006584</td>
      <td>-0.037627</td>
      <td>-0.040274</td>
      <td>-0.009409</td>
      <td>-0.020078</td>
      <td>-0.003393</td>
      <td>0.025949</td>
      <td>-0.011941</td>
      <td>-0.027034</td>
      <td>-0.007744</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 501 columns</p>
</div>




```python
ppmi_words = ppmi_df[0].values.astype(str)
ppmi_word_dict = {k: v for v, k in enumerate(ppmi_words)}
len(ppmi_words)
```




    65362




```python
ppmi_vecs = ppmi_df.drop(0, axis=1).values
ppmi_vecs.shape
```




    (65362, 500)




```python
def google_get_dist(word1, word2, dist_func):
    if word1 not in google_word_dict or word2 not in google_word_dict:
        return np.inf
    idx1 = google_word_dict[word1]
    idx2 = google_word_dict[word2]
    return dist_func(google_vecs[idx1], google_vecs[idx2])

def ppmi_get_dist(word1, word2, dist_func):
    word1 = word1.replace('_', '-')
    word2 = word2.replace('_', '-')
    if word1 not in ppmi_word_dict or word2 not in ppmi_word_dict:
        return np.inf
    idx1 = ppmi_word_dict[word1]
    idx2 = ppmi_word_dict[word2]
    return dist_func(ppmi_vecs[idx1], ppmi_vecs[idx2])
```

## load the original synonym data


```python
syn_dict = defaultdict(set)
syn_verb = pd.read_csv(os.path.join('data','EN_syn_verb.txt'), sep='\t')
syn_verb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Input.word</th>
      <th>Answer.suggestion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>to_interpret</td>
      <td>to_clarify</td>
    </tr>
    <tr>
      <th>1</th>
      <td>to_interpret</td>
      <td>to_explain</td>
    </tr>
    <tr>
      <th>2</th>
      <td>to_interpret</td>
      <td>to_explain</td>
    </tr>
    <tr>
      <th>3</th>
      <td>to_interpret</td>
      <td>to_understand</td>
    </tr>
    <tr>
      <th>4</th>
      <td>to_interpret</td>
      <td>to_clarify</td>
    </tr>
  </tbody>
</table>
</div>



## remove the leading 'to_'


```python
syn_verb = syn_verb[syn_verb['Answer.suggestion'] != '0'].applymap(lambda word: word.split('_', 1)[1])
syn_verb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Input.word</th>
      <th>Answer.suggestion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>interpret</td>
      <td>clarify</td>
    </tr>
    <tr>
      <th>1</th>
      <td>interpret</td>
      <td>explain</td>
    </tr>
    <tr>
      <th>2</th>
      <td>interpret</td>
      <td>explain</td>
    </tr>
    <tr>
      <th>3</th>
      <td>interpret</td>
      <td>understand</td>
    </tr>
    <tr>
      <th>4</th>
      <td>interpret</td>
      <td>clarify</td>
    </tr>
  </tbody>
</table>
</div>



## remove synonym pairs which have out-of-voc words


```python
for index, row in syn_verb.iterrows():
    word1 = row['Input.word']
    word2 = row['Answer.suggestion']
    if word1 not in google_word_dict or word1.replace('_', '-') not in ppmi_word_dict:
        continue
    if word2 not in google_word_dict or word2.replace('_', '-') not in ppmi_word_dict:
        continue
    syn_dict[word1].add(word2)
    syn_dict[word2].add(word1)
question_word_set = set(syn_dict)
```

## generate the dataset


```python
import random
random.seed(123)
question_set = []
columns = ['given_word', 'choice1', 'choice2', 'choice3', 'choice4', 'choice5', 'correct_answer']

for i in range(1000):
    given_word = random.sample(question_word_set, 1)[0]
    answer = random.sample(syn_dict[given_word], 1)[0]
    choices = random.sample(question_word_set.difference(syn_dict[given_word]).difference({given_word}), 4)
    choices.append(answer)
    random.shuffle(choices)
    question = [given_word, *choices, answer]
    question_set.append(question)
len(syn_dict)
```




    455



## save the dataset to file


```python
dataset = pd.DataFrame(question_set, columns = columns)
dataset.to_csv('synonym_dataset.csv')
```

## check the dataset generated


```python
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>given_word</th>
      <th>choice1</th>
      <th>choice2</th>
      <th>choice3</th>
      <th>choice4</th>
      <th>choice5</th>
      <th>correct_answer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tabulate</td>
      <td>bound</td>
      <td>split</td>
      <td>achieve</td>
      <td>record</td>
      <td>pontificate</td>
      <td>record</td>
    </tr>
    <tr>
      <th>1</th>
      <td>restrain</td>
      <td>deteriorate</td>
      <td>constrict</td>
      <td>see</td>
      <td>expand</td>
      <td>reach</td>
      <td>constrict</td>
    </tr>
    <tr>
      <th>2</th>
      <td>enclose</td>
      <td>consist</td>
      <td>brawl</td>
      <td>contrast</td>
      <td>accomplish</td>
      <td>touch</td>
      <td>consist</td>
    </tr>
    <tr>
      <th>3</th>
      <td>visit</td>
      <td>tour</td>
      <td>leave</td>
      <td>collect</td>
      <td>obstruct</td>
      <td>associate</td>
      <td>tour</td>
    </tr>
    <tr>
      <th>4</th>
      <td>alter</td>
      <td>edit</td>
      <td>swell</td>
      <td>identify</td>
      <td>land</td>
      <td>explain</td>
      <td>edit</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(dataset)
```




    1000



## read the dataset back


```python
synonym_dataset = pd.read_csv('synonym_dataset.csv')
```


```python
synonym_dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>given_word</th>
      <th>choice1</th>
      <th>choice2</th>
      <th>choice3</th>
      <th>choice4</th>
      <th>choice5</th>
      <th>correct_answer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>tabulate</td>
      <td>bound</td>
      <td>split</td>
      <td>achieve</td>
      <td>record</td>
      <td>pontificate</td>
      <td>record</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>restrain</td>
      <td>deteriorate</td>
      <td>constrict</td>
      <td>see</td>
      <td>expand</td>
      <td>reach</td>
      <td>constrict</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>enclose</td>
      <td>consist</td>
      <td>brawl</td>
      <td>contrast</td>
      <td>accomplish</td>
      <td>touch</td>
      <td>consist</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>visit</td>
      <td>tour</td>
      <td>leave</td>
      <td>collect</td>
      <td>obstruct</td>
      <td>associate</td>
      <td>tour</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>alter</td>
      <td>edit</td>
      <td>swell</td>
      <td>identify</td>
      <td>land</td>
      <td>explain</td>
      <td>edit</td>
    </tr>
  </tbody>
</table>
</div>



## Test the accuracy of both approaches


```python
from scipy.spatial.distance import cosine, euclidean
google_cosine_count, google_euclidean_count, ppmi_cosine_count, ppmi_euclidean_count = 0, 0, 0, 0
# google_out_of_v_count, ppmi_out_of_v_count = 0, 0
for index, row in synonym_dataset.iterrows():
    word = row['given_word']
    choices = [row['choice' + num] for num in list('12345')]
    correct_answer = row['correct_answer']
    
    google_cosine_dists = [google_get_dist(word, candidate, cosine) for candidate in choices]
#     if any([dist == np.inf for dist in google_cosine_dists]):
#         google_out_of_v_count += 1
#     else:
    google_cosine_answer = choices[np.argmin(google_cosine_dists)]
    if google_cosine_answer == correct_answer:
        google_cosine_count += 1
    google_euclidean_dists = [google_get_dist(word, candidate, euclidean) for candidate in choices]
    google_euclidean_answer = choices[np.argmin(google_euclidean_dists)]
    if google_euclidean_answer == correct_answer:
        google_euclidean_count += 1
    
    ppmi_cosine_dists = [ppmi_get_dist(word, candidate, cosine) for candidate in choices]
#     if any([dist == np.inf for dist in ppmi_cosine_dists]):
#         ppmi_out_of_v_count += 1
#     else:
    ppmi_cosine_answer = choices[np.argmin(ppmi_cosine_dists)]
    if ppmi_cosine_answer == correct_answer:
        ppmi_cosine_count += 1
    ppmi_euclidean_dists = [ppmi_get_dist(word, candidate, euclidean) for candidate in choices]
    ppmi_euclidean_answer = choices[np.argmin(ppmi_euclidean_dists)]
    if ppmi_euclidean_answer == correct_answer:
        ppmi_euclidean_count += 1

# print("google's accuracy on the dataset after removing %d questions that contains out-of-vocabulary words:\
#        %.3f using cosine, %.3f using euclidean"\
#        %(google_out_of_v_count, google_cosine_count / (1000 - google_out_of_v_count),\
#         google_euclidean_count / (1000 - google_out_of_v_count)))
# print("Classic Approach's accuracy on the dataset after removing %d questions that contains out-of-vocabulary words:\
#        %.3f using cosine, %.3f using euclidean"\
#        %(ppmi_out_of_v_count, ppmi_cosine_count / (1000 - ppmi_out_of_v_count),\
#         ppmi_euclidean_count / (1000 - ppmi_out_of_v_count)))

print("Google's accuracy on the dataset:           %.3f using cosine, %.3f using euclidean"\
       %(google_cosine_count / 1000, google_euclidean_count / 1000))
print("Classic Approach's accuracy on the dataset: %.3f using cosine, %.3f using euclidean"\
       %(ppmi_cosine_count / 1000, ppmi_euclidean_count / 1000))

```

    Google's accuracy on the dataset:           0.712 using cosine, 0.545 using euclidean
    Classic Approach's accuracy on the dataset: 0.542 using cosine, 0.542 using euclidean
    

### Result

* might be slightly different from the data in this table

|  Accuracy	| cosine 	| euclidean 	|
|----------	|--------	|-----------	|
| google 	| 0.712 	| 0.545 	|
| COMPOSES 	| 0.542 	| 0.542 	|

## The SAT Questions


```python
sat_questions = []
with open(os.path.join('data','SAT-package-V3.txt'), 'r') as f:
    content = f.read()
    entries = content.split('\n\n')[1:]
    for entry in entries:
        question = [i for i in entry.split('\n')[1:] if i]
        answer = question[-1]
        question = [x.split()[:-1] for x in question[:-1]]
        question.append(answer)
        sat_questions.append(question)

sat_questions[:5]
```




    [[['lull', 'trust'],
      ['balk', 'fortitude'],
      ['betray', 'loyalty'],
      ['cajole', 'compliance'],
      ['hinder', 'destination'],
      ['soothe', 'passion'],
      'c'],
     [['ostrich', 'bird'],
      ['lion', 'cat'],
      ['goose', 'flock'],
      ['ewe', 'sheep'],
      ['cub', 'bear'],
      ['primate', 'monkey'],
      'a'],
     [['word', 'language'],
      ['paint', 'portrait'],
      ['poetry', 'rhythm'],
      ['note', 'music'],
      ['tale', 'story'],
      ['week', 'year'],
      'c'],
     [['coop', 'poultry'],
      ['aquarium', 'fish'],
      ['forest', 'wildlife'],
      ['crib', 'nursery'],
      ['fence', 'yard'],
      ['barn', 'tool'],
      'a'],
     [['legend', 'map'],
      ['subtitle', 'translation'],
      ['bar', 'graph'],
      ['figure', 'blueprint'],
      ['key', 'chart'],
      ['footnote', 'information'],
      'd']]



Here, I choose to make prediction based on the **cosine similarity of the difference** between each pair of word vectors.


```python
def similarity(sample1, sample2, word1, word2):
    if sample1 not in google_word_dict or sample2 not in google_word_dict:
        return np.inf
    if word1 not in google_word_dict or word2 not in google_word_dict:
        return np.inf
    
    vec1 = google_vecs[google_word_dict[sample1]] - google_vecs[google_word_dict[sample2]]
    vec2 = google_vecs[google_word_dict[word1]] - google_vecs[google_word_dict[word2]]
    return cosine(vec1, vec2)
```


```python
passed_question = 0
correct_count = 0
for question in sat_questions:
    correct_answer = ord(question[-1]) - ord('a')
    sample1, sample2 = question[0]
    candidates = question[1:-1]
    dists = [similarity(sample1, sample2, word1, word2) for word1, word2 in candidates]
    if any([x == np.inf for x in dists]):
        passed_question += 1
        continue
    answer = np.argmin(dists)
    if answer == correct_answer:
        correct_count += 1

print("With %d out of %d questions including out-of-voc words are passed, %d out of %d are answered correctly.\nFinal accuracy is %.3f"\
      %(passed_question, len(sat_questions), correct_count, len(sat_questions) - passed_question,\
        correct_count / (len(sat_questions) - passed_question)))
```

    With 115 out of 374 questions including out-of-voc words are passed, 114 out of 259 are answered correctly.
    Final accuracy is 0.440

### We can see that this accuracy is significantly higher than random guess (0.2).
