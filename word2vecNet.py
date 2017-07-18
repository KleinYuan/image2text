import tensorflow as tf
import random
import xml.etree.ElementTree as ET
import os
import os.path
import collections
import numpy as np
import math
import re
# import cv2
from nolearn.lasagne import BatchIterator
import pickle


class Word2Vec:

    def __init__(self, training_data_path, model_path):
        self.training_data_path = training_data_path
        self.vocabulary_annotation_eng_path = '%s/annotations_complete_eng' % self.training_data_path
        self.vocabulary_size = 4000
        self.bad_list_fp = './bad.list'
        self.annotation_fps_fp = './annotation_fps_fp.list'
        self.image_fps_fp = './image_fps.list'
        self.model_path = model_path
        self.data_set_name = 'IAPR TC-12 Benchmark'
        self.image_size = 96
        self.embedding_size = 128  # Dimension of the embedding vector.
        self.skip_window = 1  # How many words to consider left and right. ---- Window Size |--| Word|--|
        self.num_skips = 2  # How many times to reuse an input to generate a label.
        self.batch_size = 128
        self.data_index = 0
        self.learning_rate = 1.0

        self.valid_size = 16  # Random set of words to evaluate similarity on.
        self.valid_window = 100  # Only pick dev samples in the head of the distribution.
        self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
        self.num_sampled = 64  # Number of negative examples to sample.
        self.num_steps = 50001

        self.annotation_fps = []
        self.image_fps = []
        self.bad_list = []

        # Placeholder properties
        self.vocabulary = []
        self.images = None
        self.features = None
        self.vocabulary_data = None
        self.reversed_dict_vocabulary_unk_tokenized = None
        self.dict_vocabulary_unk_tokenized = None
        self.nn_init = None
        self.train_inputs = None
        self.train_labels = None
        self.valid_dataset = None
        self.session = None
        self.loss = None
        self.optimizer = None
        self.similarity = None
        self.normalized_embeddings = None
        self.final_embeddings = None

    # This method read file from path and grab description and image part
    # then abstract the description and return image file path and abstracted description
    def _read_xml_file(self, fp):
        print 'Reading %s' % fp
        tree = ET.parse(fp)
        root = tree.getroot()
        image_fp = None
        description = None

        for branch in root:
            if branch.tag == 'DESCRIPTION':
                description = branch.text
            if branch.tag == 'IMAGE':
                image_fp = branch.text

        abstracted_description = self._abstract_description(description=description)

        print ' Abstracted description as'
        print abstracted_description
        return image_fp, abstracted_description

    # This methods abstract a sentence e.g: ['a yellow building with white columns; two ..']
    # into ['a', 'a', 'yellow', 'building', 'with', 'white', 'columns', ...]
    @staticmethod
    def _abstract_description(description):
        description = description.split(';')
        abstracted_description = []
        for paragraph in description:
            for word in paragraph.split(' '):
                if (word != ' ') and (word != ''):
                    word = re.sub(r'[^\w]', ' ', word)
                    word = word.split(' ')
                    abstracted_description.extend(word)
        return abstracted_description

    def _construct_vocabulary(self):
        print 'Open training data and scan all descriptions to construct a vocabulary object'
        for annotation_fp in self.annotation_fps:
            print 'Constructing from %s' % annotation_fp
            try:
                image_fp, abstracted_description = self._read_xml_file(annotation_fp)
                self.image_fps.append("%s|%s" % (image_fp, abstracted_description))
                self.vocabulary.extend(abstracted_description)
            except:
                self.bad_list.append(annotation_fp)
        print 'vocabulary size is %s before removing repeated' % len(self.vocabulary)

    def _prepare_images(self):
        print 'Open training data and pre-process images'

    def _get_data_from_path_into_memory(self):
        print 'Getting data from path into memory'
        for dir_path, _, file_names in os.walk(self.vocabulary_annotation_eng_path):
            for file_name in file_names:
                    if file_name.endswith('eng'):
                        annotation_fp = '%s/%s' % (dir_path, file_name)
                        print 'Found %s' % annotation_fp
                        self.annotation_fps.append(annotation_fp)

    def _log_bad_list(self):
        f = open(self.bad_list_fp, 'w')
        print 'There are %s out of %s bad list' % (len(self.bad_list), len(self.annotation_fps))
        for bad_file in self.bad_list:
            f.write('%s\n' % bad_file)
        f.close()

    def _log_annotation_fps(self):
        f = open(self.annotation_fps_fp, 'w')
        for annotation_fp in self.annotation_fps:
            if annotation_fp not in self.bad_list:
                f.write('%s\n' % annotation_fp)
        f.close()

    def _log_images_fps(self):
        f = open(self.image_fps_fp, 'w')
        for image_fp in self.image_fps:
            f.write('%s\n' % image_fp)
        f.close()

    def _log_info(self):
        self._log_bad_list()
        self._log_annotation_fps()
        self._log_images_fps()

    # This method is to choose most common vocabulary_size words from master vocabulary
    # Then mark all others as UNK (a token to represent not common words)
    # Then use the count order of each Non-UNK words to construct a dictionary for each words and map each word to a number
    # It will eventually output a self.vocabulary_data like [0, 3547, 557, 2, 436 ... ] to contains quantified words
    # Then with self.reversed_dict_vocabulary_unk_tokenized, you can easily find the words with like, self.reversed_dict_vocabulary_unk_tokenized[self.vocabulary_data[2]]
    def _unk_tokenize(self):
        count = [['UNK', -1]]
        count.extend(collections.Counter(self.vocabulary).most_common(self.vocabulary_size - 1))
        dict_vocabulary_unk_tokenized = dict()
        for word, _ in count:
            # Embed each word in the dict_vocabulary_unk_tokenized with key to be the word and value to be the order in Count, from 0 to 3999
            dict_vocabulary_unk_tokenized[word] = len(dict_vocabulary_unk_tokenized)

        # Counting UNK in master vocabulary from selected dict_vocabulary_unk_tokenized
        self.vocabulary_data = list()
        unk_count = 0
        for word in self.vocabulary:
            if word in dict_vocabulary_unk_tokenized:
                index = dict_vocabulary_unk_tokenized[word]
            else:
                index = 0  # dict_vocabulary_unk_tokenized['UNK']
                unk_count += 1
            self.vocabulary_data.append(index)

        # Update UNK Count
        count[0][1] = unk_count

        # Reverse dict_vocabulary_unk_tokenized with number as key and word as value
        self.reversed_dict_vocabulary_unk_tokenized = dict(zip(dict_vocabulary_unk_tokenized.values(), dict_vocabulary_unk_tokenized.keys()))
        self.dict_vocabulary_unk_tokenized = dict_vocabulary_unk_tokenized
        '''
        # count[:5]: [['UNK', 2015], ('a', 54710), ('in', 25148), ('the', 23733), ('and', 23123)]
        # self.vocabulary_data[:5]: [0, 3547, 557, 2, 436]
        # dict_vocabulary_unk_tokenized['a'], dict_vocabulary_unk_tokenized['the'] : 1 3
        # self.reversed_dict_vocabulary_unk_tokenized[0], reversed_dict_vocabulary_unk_tokenized[2]: UNK in
        '''
        print 'len of vocabulary_data is ', len(self.vocabulary_data)

    def _load_data(self):
        print 'Preparing data for training with %s' % self.data_set_name
        self._get_data_from_path_into_memory()
        self._construct_vocabulary()
        self._log_info()
        self._unk_tokenize()
        batch, labels = self._create_batch(batch_size=8)
        for i in range(8):
            print(batch[i], self.reversed_dict_vocabulary_unk_tokenized[batch[i]],
                  '->', labels[i, 0], self.reversed_dict_vocabulary_unk_tokenized[labels[i, 0]])

    def _create_batch(self, batch_size):

        assert batch_size % self.num_skips == 0
        assert self.num_skips <= 2 * self.skip_window

        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.vocabulary_data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.vocabulary_data)
        for i in range(batch_size // self.num_skips):
            target = self.skip_window  # target label at the center of the buffer
            targets_to_avoid = [self.skip_window]
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * self.num_skips + j] = buffer[self.skip_window]
                labels[i * self.num_skips + j, 0] = buffer[target]
            buffer.append(self.vocabulary_data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.vocabulary_data)
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self.data_index = (self.data_index + len(self.vocabulary_data) - span) % len(self.vocabulary_data)
        return batch, labels

    def _define_graph(self):
        print 'Defining Graph ...'
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            with tf.device('/cpu:0'):
                # Look up embeddings for inputs.
                embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)

                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size], stddev=1.0 / math.sqrt(self.embedding_size)))
                nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=self.train_labels,
                               inputs=embed,
                               num_sampled=self.num_sampled,
                               num_classes=self.vocabulary_size))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            # Compute the cosine similarity between minibatch examples and all embeddings.

            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            self.normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, self.valid_dataset)
            self.similarity = tf.matmul(valid_embeddings, self.normalized_embeddings, transpose_b=True)
            self.nn_init = tf.global_variables_initializer()

    def train(self):
        self._load_data()
        self._define_graph()
        print 'Training ...'
        with tf.Session(graph=self.graph) as self.session:
            self.nn_init.run()
            print 'Initialized!'
            saver = tf.train.Saver()
            avg_loss = 0
            for step in xrange(self.num_steps):
                batch_input, batch_labels = self._create_batch(self.batch_size)
                feed_dict = {
                    self.train_inputs: batch_input,
                    self.train_labels: batch_labels
                }

                _, loss_val = self.session.run(
                    [self.optimizer, self.loss],
                    feed_dict = feed_dict
                )

                avg_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        avg_loss /= 2000
                    print 'Average Loss at step %s: %s' % (step, avg_loss)
                    avg_loss = 0

                if step % 10000 == 0:
                    sim = self.similarity.eval()
                    for i in xrange(self.valid_size):
                        valid_word = self.reversed_dict_vocabulary_unk_tokenized[self.valid_examples[i]]
                        top_k = 8
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = 'Nearest to %s' % valid_word

                        for k in xrange(top_k):
                            close_word = self.reversed_dict_vocabulary_unk_tokenized[nearest[k]]
                            log_str = '%s %s, ' % (log_str, close_word)
                        print log_str

            self.final_embeddings = self.normalized_embeddings.eval()

            # print self.final_embeddings[2], len(self.final_embeddings)
            # print self.reversed_dict_vocabulary_unk_tokenized[2], len(self.reversed_dict_vocabulary_unk_tokenized)
            with open('embeddings.pickle', 'wb') as f:
                pickle.dump(self.final_embeddings, f)
            with open('dict_vocabulary.pickle', 'wb') as f:
                pickle.dump(self.dict_vocabulary_unk_tokenized, f)
            with open('reversed_dict_vocabulary.pickle', 'wb') as f:
                pickle.dump(self.reversed_dict_vocabulary_unk_tokenized, f)

            saver.save(self.session, '%s/model' % self.model_path, global_step=self.num_steps)

    def predict(self):
        print 'Predicting with model in %s' % self.model_path
