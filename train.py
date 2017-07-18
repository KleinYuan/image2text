from word2vecNet import Word2Vec
from vanillaCNNNet import VanillaCNN
import pickle

embeddings_fp = 'embeddings.pickle'
dict_vocabulary = 'dict_vocabulary.pickle'
reversed_dict_vocabulary = 'reversed_dict_vocabulary.pickle'
#
# storyNet = Word2Vec(training_data_path='./iaprtc12', model_path='./models')
# storyNet.train()

with open(embeddings_fp, 'r') as f:
    embeddings = pickle.load(f)

with open(dict_vocabulary, 'r') as f:
    dict_vocabulary = pickle.load(f)

with open(reversed_dict_vocabulary, 'r') as f:
    reversed_dict_vocabulary = pickle.load(f)

vanillaCNNNet = VanillaCNN(training_data_fp_list='./image_fps.list',
                           data_root_dir='./iaprtc12',
                           embeddings=embeddings,
                           reversed_dict_vocabulary=reversed_dict_vocabulary,
                           dict_vocabulary=dict_vocabulary)
vanillaCNNNet.train()
