# Intro

This repo is to implement a multi-modal natural language model with tensorflow.

Dependencies:

- [X] [tensorflow](https://www.tensorflow.org)

- [X] [lasagne](https://https://github.com/Lasagne/Lasagne)

- [X] [Theano](https://github.com/Theano/Theano)

Datasets:

- [X] [IAPR TC-12](http://www.imageclef.org/photodata)

Sample annotations:

```
<DOC>
<DOCNO>annotations/01/1000.eng</DOCNO>
<TITLE>Godchild Cristian Patricio Umaginga Tuaquiza</TITLE>
<DESCRIPTION>a dark-skinned boy wearing a knitted woolly hat and a light and dark grey striped jumper with a grey zip, leaning on a grey wall;</DESCRIPTION>
<NOTES></NOTES>
<LOCATION>Quilotoa, Ecuador</LOCATION>
<DATE>April 2002</DATE>
<IMAGE>images/01/1000.jpg</IMAGE>
<THUMBNAIL>thumbnails/01/1000.jpg</THUMBNAIL>
</DOC>
```


References:

1. [Dong, Jianfeng, Xirong Li, and Cees GM Snoek. "Word2VisualVec: Image and video to sentence matching by visual feature prediction." CoRR, abs/1604.06838 (2016).](https://arxiv.org/pdf/1604.06838.pdf)

2. [Karpathy, Andrej, and Li Fei-Fei. "Deep visual-semantic alignments for generating image descriptions." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)

3. [Kiros, Ryan, Ruslan Salakhutdinov, and Rich Zemel. "Multimodal neural language models." Proceedings of the 31st International Conference on Machine Learning (ICML-14). 2014.](http://proceedings.mlr.press/v32/kiros14.pdf)

4. [word2vec tutorial](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

# Overview

1. Firstly, a word embedding with word2vec net is trained against iaprtc12 datasets.

2. Secondly, the filtered (meaning, if the description is too long, we only keep the first sentence) word vectors for each description of image are used as target output of a CNN network



# Network Design

Coming soon ...

# Training

Coming soon ...

# Testing and Results

Coming soon ...
