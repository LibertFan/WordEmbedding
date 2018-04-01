# WordEmbedding
## Introduction
This is an implementation of [Distributed Representations ofWords and Phrases
and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-andphrases) 
with pytorch.
* In order to train word2vec model, IMDB dataset published by stanford [
Learning word vectors for sentiment analysis](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf) is utilized. 
* The final consequence is saved with the format of `txt`.

## Requirement
* python 3.6
* pytorch > 0.1
* numpy
* nltk with punkt for word tokenize

## Run
Step 1: Download nltk and punkt

Open the command line:

```bash
pip install nltk
python
import nltk
nltk.download('punkt')
```
Step 2: Download IMDB dataset

You can download thew dataset in [IMDB-stanford](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)

Then you can create an directory for dataset with the following commands:
```bash
cd # where you save the 
cd WordEmbedding
mkdir Data
cd Data
mkdir raw
cd raw
mv ~/Downloads/aclImdb_v1.tar.gz .
tar -zxvf aclImdb_v1.tar.gz
```

Step 3: PreProcess Ddataset

Open `DataProcess` directory in `Code`.

```bash
python FormalizeData.py
python BuildVocab.py
```
After the upper processing, you can run the following commands:
```bash
python Streamer.py
```
and modify the parameters in Streamer to check whether the preprocess is appropriate. 

Step 4: Train

Open `Starter` directory in `Code`.

```bash
python StartSkipGram.py
```
to run the model.

## Hyperparameters

First, we explain some important hyperparameters:

* `neg_sampling_num` time of negative sampling. 6 
* `neg_sampling_p_factor` \lambda for U^{\lambda} which is the possibility of negative sampling. 0.75
* `sub_sampling_p_factor` t for 1 - \sqrt{\frac{t}{U}} which determines whether add the piece of data into training sequence. 1e-4
* `window_size` left and right windows_size words of target words would be extracted as positive samples. 5  
* `emb_dim` The dimension of word embedding. 20
* `epochs` 500000
* `learnimg_rate` 1.0
* `learning_rate_factor` learning_rate multiply learning_rate_factor to update learning rate every 500 epochs. 0.99
* `num_units` None
* `min_counts` None
* `display_every` display training situation such as epoch, loss, etc every certain times. 5000
* `save_every` save model every certain times. 50000


## Reference
* [Distributed Representations ofWords and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-andphrases)
* [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)
* [Learning word vectors for sentiment analysis](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)
