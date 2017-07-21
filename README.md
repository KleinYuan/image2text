# Intro

This repo is to implement a multi-modal natural language model with tensorflow.

|**Dependencies**             |  **DataSets**|
| --- | --- |
|[tensorflow](https://www.tensorflow.org) <br/>[lasagne](https://https://github.com/Lasagne/Lasagne) <br/>[Theano](https://github.com/Theano/Theano) |[IAPR TC-12](http://www.imageclef.org/photodata)|


# Project Overview

1. Firstly, a word embedding with word2vec net is trained against iaprtc12 datasets.

2. Secondly, the filtered (meaning, if the description is too long, we only keep the first sentence) word vectors for each description of image are used as target output of a CNN network

# Setup

For various systems, you need to use different tools to install tensorflow, lasagne, theano, nolearn, ... dependencies, first.

Then, simply run below scripts to download the datasets

Run:

```bash setup.sh```


# Network Design

**Word2Vec**             |  **StoryNet**
:-------------------------:|:-------------------------:
![word2vec](https://www.tensorflow.org/images/softmax-nplm.png)|![storynet](https://user-images.githubusercontent.com/8921629/28401184-23dfdb4e-6ccd-11e7-8883-cf7749444d32.png)

# Training

Run:

```python train.py```


**Optimizer**             |  **Loss**
:-------------------------:|:-------------------------:
MomentumOptimizer  | MSE Loss


![learning_curve](https://user-images.githubusercontent.com/8921629/28445982-bd35c1e6-6d7c-11e7-8100-cfdeee644167.png)

# Pre-trained Model

Coming soon ...

# Testing and Results

Coming soon ...

# Data Sets

The image collection of the IAPR TC-12 Benchmark consists of 20,000 still natural images taken from locations around the world and comprising an assorted cross-section of still natural images. This includes pictures of different sports and actions, photographs of people, animals, cities, landscapes and many other aspects of contemporary life.

Each image is associated with a text caption in up to three different languages (English, German and Spanish) . These annotations are stored in a database which is managed by a benchmark administration system that allows the specification of parameters according to which different subsets of the image collection can be generated.

The IAPR TC-12 Benchmark is now available free of charge and without copyright restrictions.

More [details](http://www.imageclef.org/photodata).

Sample annotations:

`

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

`


# References:

1. [Dong, Jianfeng, Xirong Li, and Cees GM Snoek. "Word2VisualVec: Image and video to sentence matching by visual feature prediction." CoRR, abs/1604.06838 (2016).](https://arxiv.org/pdf/1604.06838.pdf)

2. [Karpathy, Andrej, and Li Fei-Fei. "Deep visual-semantic alignments for generating image descriptions." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)

3. [Kiros, Ryan, Ruslan Salakhutdinov, and Rich Zemel. "Multimodal neural language models." Proceedings of the 31st International Conference on Machine Learning (ICML-14). 2014.](http://proceedings.mlr.press/v32/kiros14.pdf)

4. [word2vec tutorial](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
