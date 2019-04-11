from gensim.models import Word2Vec
import numpy as np
import data_loader as dl
import tensorflow as tf
import datetime
import W2VCNNmodel as nn_model

vector_dim = 200
filter_sizes = [3, 4, 5]
num_classes = 3
num_filters = 10
num_epochs = 5
batch_size = 100
num_dense_sizes = [20, 30, 20]


def preparation():

    def convert_text_to_indexes(text, wv):
        index_data = np.zeros(max_sen_lenth, dtype=int)
        string_data = text.split()
        i = 0
        for word in string_data:
            if word in wv:
                index_data[i] = wv.vocab[word].index
            i += 1
        return index_data.tolist()

    def convert_labels_to_classes(label):
        if label == '1':
            return [1, 0, 0]
        if label == '0':
            return [0, 1, 0]
        if label == '-1':
            return [0, 0, 1]

    print('Loading embedding model...')
    model = Word2Vec.load('models/w2v/shares/model_shares.w2v')
    print('Embedding model load complete.')

    print('Loading train and test data...')
    raw_texts, max_sen_lenth = dl.load_marked_texts()
    raw_labels = dl.load_marked_labels()
    print('Data load complete.')

    print('Tokenize data...')
    text_indexes = [convert_text_to_indexes(text, model.wv) for text in raw_texts]
    classes_labels = [convert_labels_to_classes(label) for label in raw_labels]
    print('Tokenization complete.')

    print('Splitting data into parts...')
    x_train, y_train, x_test, y_test = dl.get_sets(text_indexes, classes_labels, 0.2)
    print('Splitting complete.')

    print('Building embedding matrix from model...')
    embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print('Embedding matrix building complete.')

    print('Words in vocabulary: {}'.format(len(model.wv.vocab)))

    return list(x_train), list(y_train), list(x_test), list(y_test), embedding_matrix, max_sen_lenth


def train_nn(x_train, y_train, x_test, y_test, embedding_matrix, max_sen_lenth):

    def train_step(x_batch, y_batch):
        feed_dict = {CNN.data: x_batch, CNN.labels: y_batch, CNN.dropout_keep_prob: 0.5, CNN.embedding_placeholder: embedding_matrix}
        _, step, summaries, loss, accuracy = sess.run([train_op, global_step, merged, CNN.loss, CNN.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print('{}: step {}, loss {:g}, acc {:g}'.format(time_str, step, loss, accuracy))
        summary_writer.add_summary(summaries, step)

    def test_step(x_batch, y_batch, writer=None):
        feed_dict = {CNN.data: x_batch, CNN.labels: y_batch, CNN.dropout_keep_prob: 1.0, CNN.embedding_placeholder: embedding_matrix}
        step, summaries, loss, accuracy = sess.run([global_step, merged, CNN.loss, CNN.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print('{}: step {}, loss {:g}, acc {:g}'.format(time_str, step, loss, accuracy))
        if writer:
            writer.add_summary(summaries, step)

    tf.reset_default_graph()
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess = tf.Session(config=session_conf)

    print('Setting up NN-model...')
    CNN = nn_model.CNN(vector_dim, len(embedding_matrix), num_classes, max_sen_lenth, num_filters, filter_sizes, num_dense_sizes)
    print('Model setting up complete.')

    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-4)
    grads_and_vars = optimizer.compute_gradients(CNN.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    tf.summary.scalar('Loss', CNN.loss)
    tf.summary.scalar('Accuracy', CNN.accuracy)
    merged = tf.summary.merge_all()
    logdir_train = 'tensorboard/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-train/'
    logdir_test = 'tensorboard/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-test/'

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(logdir_train, sess.graph)
    summary_writer_test = tf.summary.FileWriter(logdir_test, sess.graph)

    print('Getting batches from data...')
    batches = dl.get_batches(zip(x_train, y_train), batch_size, num_epochs)
    print('Completed. Batches size: {}'.format(batch_size))
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % 50 == 0:
            print('\nEvaluation:')
            test_step(x_test, y_test, writer=summary_writer_test)
            print('')
        if current_step % 1000 == 0:
            path = saver.save(sess, 'models/NN/w2vcnn.ckpt', global_step=current_step)
            print('Saved model checkpoint to {}\n'.format(path))


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, embedding_matrix, max_sen_lenth = preparation()
    train_nn(x_train, y_train, x_test, y_test, embedding_matrix, max_sen_lenth)


