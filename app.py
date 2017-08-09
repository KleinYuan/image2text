import cv2
from services.prediction import PredictionApp
from services.dataService import *
from config import config

generated_data_fp = './data'

embeddings_fp = '%s/embeddings.pickle' % generated_data_fp
dict_vocabulary_fp = '%s/dict_vocabulary.pickle' % generated_data_fp
reversed_dict_vocabulary_fp = '%s/reversed_dict_vocabulary.pickle' % generated_data_fp

frozen_graph_filename='./vcnn-models/storyNet.pb'
test_image_fp = './test-images/test.jpg'

embeddings = load_pickle(embeddings_fp)
dict_vocabulary = load_pickle(dict_vocabulary_fp)
reversed_dict_vocabulary = load_pickle(reversed_dict_vocabulary_fp)

predictionApp = PredictionApp(net_name='storyNet',
                              embeddings=embeddings,
                              reversed_dict_vocabulary=reversed_dict_vocabulary,
                              dropout=True,
                              image_size=config.image_size,
                              frozen_graph_filename=frozen_graph_filename)

input_image = cv2.imread(test_image_fp)
prediction = predictionApp.predict(input_image)
