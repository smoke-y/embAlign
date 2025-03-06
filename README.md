# Aligning embeddings

## PREPARING DATA

### DATA
Download en-hi full(not train or test) <a href="https://github.com/facebookresearch/MUSE/#Download">here</a>. Place it in the root dir.

### EMBEDDING
Download english and hindi embeddings from <a href="https://fasttext.cc/docs/en/pretrained-vectors.html">here</a>. These embeddings are trained from wiki articles. Place them in the root dir.

Run ```python3 srcipts/data.py```. This will reduce the number of words in our embedding by picking the 100,000 most frequent words. It will also generate test and train dataset for supervised training.

## SUPERVISED TRAINING
Run ```python3 scripts/sup.py``` to find W using SVD algorithm. In the first pass, we use the dataset which already has the correct pairs. For the consecutive passes, we find the nearest neighbour by calculating cosine similarity and use them as reference points.

2 arguments can be passed to the script:
1) ```--iter``` number of iterations to approximate W
2) ```--dict``` how many pairs to use from the dataset in the first pass.

## UNSUPERVISED TRAINING
Run ```python3 scripts/unsup.py``` to run an adversial network followed by procrustes algorithm.

## EVALUATION
Run ```python3 scripts/eval.py w_[sup/unsup].bin``` to evaluate W on testing data. The script report precision@1, precision@5, and analyzes cosine similarity.

### SUPERVISED
Using 26K pair
```
total pairs processed:  2000
precision@1: 6.05%
precision@5: 15.1%
[cosine similarity]
greater than 0: 2000
less than 0: 0
strong corelation: 6
moderate corelation: 937
weak corelation: 2931
[strong corelation words]
hypochlorite हाइपोक्लोराइट
irritability चिड़चिड़ापन
muzaffarnagar मुज़फ्फरनगर
virudhunagar विरुधुनगर
constitutionalism संविधानवाद
odorless गंधहीन
```

Using 10K pair
```
total pairs processed:  2000
precision@1: 4.6%
precision@5: 11.55%
[cosine similarity]
greater than 0: 1999
less than 0: 1
strong corelation: 1
moderate corelation: 550
weak corelation: 2549
[strong corelation words]
constitutionalism संविधानवाद
```

Using 5K pair
```
total pairs processed:  2000
precision@1: 4.3999999999999995%
precision@5: 10.2%
[cosine similarity]
greater than 0: 1999
less than 0: 1
strong corelation: 1
moderate corelation: 327
weak corelation: 2326
[strong corelation words]
constitutionalism संविधानवाद
```


### UNSUPERVISED
```
total pairs processed:  1966
precision@1: 0.0%
precision@5: 0.0%
[cosine similarity]
greater than 0: 1956
less than 0: 10
strong corelation: 0
moderate corelation: 3
weak corelation: 1969
```

paper: https://arxiv.org/abs/1710.04087<br>
repo:  https://github.com/facebookresearch/MUSE<br>
This is my submission to SarvamAI