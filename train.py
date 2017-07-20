from word2vecNet import Word2Vec
from storyNet import StoryNet
import pickle

data_fp = './iaprtc12'
embeddings_fp = 'embeddings.pickle'
dict_vocabulary = 'dict_vocabulary.pickle'
reversed_dict_vocabulary = 'reversed_dict_vocabulary.pickle'


def load_pickle(fp):
    with open(fp, 'r') as f:
        content = pickle.load(f)
    return content


# Train word2Vec and obtain the word embeddings
word2Vec = Word2Vec(training_data_path=data_fp, model_path='./word2vec-models')
word2Vec.train()

# Word embeddings will be stored locally, therefore, here we can either load from local pickle or from word2Vec attributes
embeddings = load_pickle(embeddings_fp)
dict_vocabulary = load_pickle(dict_vocabulary)
reversed_dict_vocabulary = load_pickle(reversed_dict_vocabulary)

# Train our story net with the CNN
storyNet = StoryNet(training_data_fp_list='./image_fps.list',
                    data_root_dir=data_fp,
                    embeddings=embeddings,
                    reversed_dict_vocabulary=reversed_dict_vocabulary,
                    dict_vocabulary=dict_vocabulary)
storyNet.train()
