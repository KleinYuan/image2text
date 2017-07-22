import pickle
import tensorflow as tf


def load_pickle(fp):
    with open(fp, 'r') as f:
        content = pickle.load(f)
    return content


def chunks(seq, n):
    return (seq[i:i+n] for i in xrange(0, len(seq), n))


def search_closest_word(embedding_input, words_input, voc_size, embedding_size, num_words):
    embeddings = tf.placeholder(tf.float32, [voc_size, embedding_size])
    words = tf.placeholder(tf.float32, [embedding_size, num_words])

    normalized_embeddings = tf.nn.l2_normalize(x=embeddings, dim=1)
    normalized_words = tf.nn.l2_normalize(x=words, dim=1)

    cosine_similarity = tf.matmul(a=tf.transpose(normalized_words,[1, 0]), b=tf.transpose(normalized_embeddings, [1, 0]))

    closest_words = tf.argmax(cosine_similarity, 1)

    sess = tf.Session()
    predicted_closest_words = sess.run([closest_words], feed_dict={
        embeddings: embedding_input,
        words: words_input
    })

    return predicted_closest_words

