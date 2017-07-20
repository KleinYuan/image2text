import tensorflow as tf
import os
import os.path
import numpy as np
import ast
import cv2

from sklearn.utils import shuffle
from nolearn.lasagne import BatchIterator
from itertools import chain
#from sklearn.model_selection import train_test_split


class StoryNet:

    def __init__(self, training_data_fp_list, data_root_dir, embeddings, reversed_dict_vocabulary, dict_vocabulary, img_size=96, embedding_size=128):
        self.training_data_fp_list = training_data_fp_list
        self.embeddings = embeddings
        self.reversed_dict_vocabulary = reversed_dict_vocabulary
        self.dict_vocabulary = dict_vocabulary
        self.data_root_dir = data_root_dir
        self.session = None
        self.x_batch = None
        self.y_batch = None
        self.predictions = None
        self.is_training = None
        self.img_size = img_size
        self.num_channels = 3
        self.num_word2describe = 80
        self.embedding_size = embedding_size
        self.num_output = self.num_word2describe * self.embedding_size
        self.feed_X = None
        self.feed_Y = None

        self.graph = None
        self.learning_rate = 0.0001
        self.num_epochs = 1001
        self.cross_entropy = None
        self.optimizer = None
        self.batch_size = 128
        self.saver = None

        self.current_epoch = None

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_valid = None
        self.y_valid = None
        self.model_parent_dir = './vcnn-models'
        self.model_path = self.model_parent_dir + '/model.ckpt'


    @staticmethod
    def fully_connected(input, size):
        weights = tf.get_variable('weights', shape=[input.get_shape()[1], size],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=[size], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input, weights) + biases

    @staticmethod
    def conv_relu(input, kernel_size, depth):
        weights = tf.get_variable('weights', shape=[kernel_size, kernel_size, input.get_shape()[3], depth],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=[depth], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.relu(conv + biases)

    @staticmethod
    def pool(input, size):
        return tf.nn.max_pool(input, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')

    def fully_connected_relu(self, input, size):
        return tf.nn.relu(self.fully_connected(input, size))

    def _define_net(self, input, training):

        with tf.variable_scope('conv1'):
            conv1 = self.conv_relu(input, kernel_size=3, depth=32)
            pool1 = self.pool(conv1, size=2)
            pool1 = tf.cond(training, lambda: tf.nn.dropout(pool1, keep_prob=0.8), lambda: pool1)

        with tf.variable_scope('conv2'):
            conv2 = self.conv_relu(pool1, kernel_size=2, depth=64)
            pool2 = self.pool(conv2, size=2)
            pool2 = tf.cond(training, lambda: tf.nn.dropout(pool2, keep_prob=0.7), lambda: pool2)

        with tf.variable_scope('conv3'):
            conv3 = self.conv_relu(pool2, kernel_size=2, depth=128)
            pool3 = self.pool(conv3, size=2)
            pool3 = tf.cond(training, lambda: tf.nn.dropout(pool3, keep_prob=0.6), lambda: pool3)

        with tf.variable_scope('conv4'):
            conv4 = self.conv_relu(pool3, kernel_size=2, depth=256)
            pool4 = self.pool(conv4, size=2)
            pool4 = tf.cond(training, lambda: tf.nn.dropout(pool4, keep_prob=0.5), lambda: pool4)

        shape = pool4.get_shape().as_list()
        flattened = tf.reshape(pool4, [-1, shape[1] * shape[2] * shape[3]])

        with tf.variable_scope('fc5'):
            fc5 = self.fully_connected_relu(flattened, size=1000)
            fc5 = tf.cond(training, lambda: tf.nn.dropout(fc5, keep_prob=0.5), lambda: fc5)
        with tf.variable_scope('fc6'):
            fc6 = self.fully_connected_relu(fc5, size=1000)
        with tf.variable_scope('out'):
            prediction = self.fully_connected(fc6, size=self.num_output)
        return prediction

    def _define_graph(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.x_batch = tf.placeholder(tf.float32, shape=(None, self.img_size, self.img_size, self.num_channels))
            self.y_batch = tf.placeholder(tf.float32, shape=(None, self.num_output))
            self.is_training = tf.placeholder(tf.bool)
            self.current_epoch = tf.Variable(0)

            self.predictions = self._define_net(input=self.x_batch,
                                                training=self.is_training)

            self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_batch * tf.log(self.predictions), reduction_indices=[1]))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9,
                                                    beta2=0.999, epsilon=0.001, use_locking=False,
                                                    name='Adam').minimize(self.cross_entropy)

            tf.summary.scalar("cross_entropy", self.cross_entropy)

    def _get_batch_predictions(self, batches):
        predicts = []
        batch_iterator = BatchIterator(batch_size=128)
        for x_batch, _ in batch_iterator(batches):
            [p_batch] = self.session.run([self.predictions], feed_dict={
                self.x_batch: x_batch,
                self.is_training: False
            })
            predicts.extend(p_batch)
        return predicts

    def _load_data(self, test=False):
        with open(self.training_data_fp_list, 'r') as f:
            training_data = f.read()
        training_images = []
        training_labels = []
        training_data = training_data.split('\n')
        bad_list = []
        space_embedding = self.embeddings[self.dict_vocabulary['']]
        for data in training_data:
            try:
                data_split = data.split('|')
                image_fp = data_split[0]
                description = ast.literal_eval(data_split[1])
                print 'image_fp: ', image_fp
                print 'description: ', description
            except:
                continue

            # Construct training images -- X
            img = cv2.imread('%s/%s' % (self.data_root_dir, image_fp))
            if img is not None:

                img = cv2.resize(img, (self.img_size, self.img_size))
                img = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                training_images.append(img)
            else:
                bad_list.append(data)
                continue

            # Construct labels -- Y
            feature = []
            num_sentences = description.count('')
            for word in description:
                if word in self.dict_vocabulary:
                    word_embedding = self.embeddings[self.dict_vocabulary[word]]
                else:
                    # if the word is not in the voc dict, meaning it is an UNK
                    word_embedding = self.embeddings[0]
                feature.append(word_embedding)

            # If feature is too long,  we only get the first sentence
            if (len(description) > self.num_word2describe) and (num_sentences >= 1):
                print description
                feature = feature[:description.index('')]
                description = description[:description.index('')]

            # Pad feature
            while len(feature) < self.num_word2describe:
                feature.append(space_embedding)
                description.append('')

            if len(feature) == self.num_word2describe:
                feature = list(chain.from_iterable(feature))
                training_labels.append(feature)
            if len(feature) > self.num_output:
                raise

        print 'Bad List: '
        print bad_list

        training_images = np.array(training_images)
        training_images = training_images.astype(np.float32)
        training_images = training_images.reshape(-1, self.img_size, self.img_size, self.num_channels)

        training_labels = np.array(training_labels)

        if not test:
            training_labels = training_labels.astype(np.float32)
            training_images, training_labels = shuffle(training_images, training_labels, random_state=42)

        self.feed_X = training_images
        self.feed_Y = training_labels
        print 'Sample X', self.feed_X[0].shape
        print 'Sample Y', self.feed_Y[0].shape

    def train(self):

        self._load_data()
        self._define_graph()

        # self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.feed_X, self.feed_Y, test_size =0.2)
        # self.x_test, self.x_valid, self.y_test, self.y_valid = train_test_split(self.x_test, self.y_test, test_size=0.4)
        self.x_train = self.feed_X
        self.y_train = self.feed_Y

        if not os.path.exists(self.model_parent_dir):
            os.makedirs(self.model_parent_dir)

        with tf.Session(graph=self.graph) as self.session:
            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            writer = tf.summary.FileWriter('./log/train', self.session.graph)

            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                batch_iterator = BatchIterator(batch_size=self.batch_size, shuffle=True)

                print '%s / %s iterations' % (self.current_epoch, self.num_epochs)

                for x_batch, y_batch in batch_iterator(self.x_train, self.y_train):
                    _, summary = self.session.run([self.optimizer, summary_op], feed_dict={
                        self.x_batch: x_batch,
                        self.y_batch: y_batch,
                        self.is_training: True
                    })

                writer.add_summary(summary, epoch)

                if (epoch % 300 == 0) and (epoch > 0):
                    save_path = self.saver.save(self.session, '%s_%s_' % (self.model_path, epoch))
                    print 'Snapshot Model file: ', save_path

            save_path = self.saver.save(self.session, self.model_path)
            print 'Model file: ', save_path
