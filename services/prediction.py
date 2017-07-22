import numpy as np
import cv2
from copy import deepcopy
from services.dataService import *


class PredictionApp:
    def __init__(self, net_name, embeddings, reversed_dict_vocabulary, dropout=True, image_size=96, num_word2describe=80, frozen_graph_filename=None):
        self.net_name = net_name
        self.image_size = image_size
        self.num_channels = 3
        self.num_word2describe = num_word2describe
        self.dropout = dropout
        self.frozen_graph_filename = frozen_graph_filename

        self.embeddings = embeddings
        self.reversed_dict_vocabulary = reversed_dict_vocabulary

        self.input_image = None
        self.input_image_bounding_box = None
        self.input_image_params = None

        self.graph = None
        self.session = None

        # Load Graph
        self._init_model()

    def _pre_processing(self, img):
        self.input_image_origin = img
        self.input_image = deepcopy(img)
        self.input_image = cv2.resize(self.input_image, (self.image_size, self.image_size))
        self.input_image = cv2.normalize(self.input_image, self.input_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.input_image = [self.input_image]

        self.input_image = np.array(self.input_image)
        self.input_image = self.input_image.astype(np.float32)
        self.input_image = self.input_image.reshape(-1, self.image_size, self.image_size, self.num_channels)

    def _init_model(self):
        print 'Loading %s' % self.net_name
        self.graph = self._load_frozen_graph()

    def predict(self, img):
        self._pre_processing(img)
        tf_x = self.graph.get_tensor_by_name('prefix/Placeholder:0')
        val_is_training = self.graph.get_tensor_by_name('prefix/Placeholder_2:0')
        y = self.graph.get_tensor_by_name('prefix/out/add:0')

        config = tf.ConfigProto(device_count={'GPU': 0})
        config.gpu_options.allow_growth = True

        with tf.Session(graph=self.graph, config=config) as self.session:
            p = self.session.run(y, feed_dict={
                tf_x: self.input_image,
                val_is_training: False
            })

        predicted_vectors = p[0]
        print 'Predicted Vectors are: '
        print predicted_vectors

        predicted_words = list(chunks(predicted_vectors, self.num_word2describe))
        predicted_words = np.array(predicted_words)
        predicted_words = predicted_words.astype(np.float32)

        _, num_words = predicted_words.shape
        voc_size, embedding_size = self.embeddings.shape

        predicted_sentence = search_closest_word(embedding_input=self.embeddings,
                                                 words_input=predicted_words,
                                                 voc_size=voc_size,
                                                 embedding_size=embedding_size,
                                                 num_words=num_words)

        predicted_sentence = predicted_sentence[0].tolist()

        description = []
        for word in predicted_sentence:
            description.append(self.reversed_dict_vocabulary[word])

        print description
        return description

    def _load_frozen_graph(self):
        with tf.gfile.GFile(self.frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="prefix",
                op_dict=None,
                producer_op_list=None
            )
        return graph
