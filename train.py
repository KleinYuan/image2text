from services.word2vecNet import Word2Vec
from services.storyNet import StoryNet
import pickle

# Run Setup Script to download this data sets
training_data_fp = './iaprtc12'
generated_data_fp = './data'

# Those are names of the pickle/list files that the word2vec trained objects are going to be saved with
# Some of the lists may not necessary to be outputed but in terms of better understanding the process, I output it
embeddings_fp = '%s/embeddings.pickle' % generated_data_fp
dict_vocabulary_fp = '%s/dict_vocabulary.pickle' % generated_data_fp
reversed_dict_vocabulary_fp = '%s/reversed_dict_vocabulary.pickle' % generated_data_fp
image_fps = '%s/image_fps.list' % generated_data_fp
annotation_fps_fp = '%s/annotation_fps_fp.list' % generated_data_fp
word2vec_model_path = './word2vec-models'


def load_pickle(fp):
    with open(fp, 'r') as f:
        content = pickle.load(f)
    return content


# Train word2Vec and obtain the word embeddings
word2Vec = Word2Vec(training_data_path=training_data_fp,
                    embeddings_fp=embeddings_fp,
                    dict_vocabulary_fp=dict_vocabulary_fp,
                    reversed_dict_vocabulary_fp=reversed_dict_vocabulary_fp,
                    image_fps_fp=image_fps,
                    annotation_fps_fp=annotation_fps_fp,
                    model_path = word2vec_model_path)
word2Vec.train()

# Word embeddings will be stored locally, therefore, here we can either load from local pickle or from word2Vec attributes
embeddings = load_pickle(embeddings_fp)
dict_vocabulary = load_pickle(dict_vocabulary_fp)
reversed_dict_vocabulary = load_pickle(reversed_dict_vocabulary_fp)

# Train our story net with the CNN
storyNet = StoryNet(training_data_fp_list=image_fps,
                    data_root_dir=training_data_fp,
                    embeddings=embeddings,
                    reversed_dict_vocabulary=reversed_dict_vocabulary,
                    dict_vocabulary=dict_vocabulary)
storyNet.train()
