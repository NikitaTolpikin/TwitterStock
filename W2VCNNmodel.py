import tensorflow as tf


class CNN:
    def __init__(self, embedding_dim, vocab_size, num_classes, max_sentence_len, num_filters, filter_sizes, num_danse_sizes, l2_reg_lambda=0.0):
        self.data = tf.placeholder(tf.int32, [None, max_sentence_len])
        self.labels = tf.placeholder(tf.float32, [None, num_classes])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        l2_loss = tf.constant(0.0)

        with tf.name_scope('embedding'):
            W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=False, name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
            self.embedding = W.assign(self.embedding_placeholder)
            embedded_data = tf.nn.embedding_lookup(self.embedding, self.data)
            embedded_data_expanded = tf.expand_dims(embedded_data, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size, embedding_dim, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
            conv = tf.nn.conv2d(embedded_data_expanded, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

            pooled = tf.nn.max_pool(h, ksize=[1, max_sentence_len - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                    padding='VALID', name='pool')
            pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        dense = []
        i=0
        for i in range(0, len(num_danse_sizes)):
            if i==0:
                with tf.name_scope('dense'):
                    W = tf.Variable(tf.truncated_normal([num_filters_total, num_danse_sizes[i]], stddev=0.1), name='W')
                    b = tf.Variable(tf.constant(0.1, shape=[num_danse_sizes[i]]), name='b')
                    dense.append(tf.nn.sigmoid(tf.nn.xw_plus_b(self.h_drop, W, b)))
            else:
                with tf.name_scope('dense'):
                    W = tf.Variable(tf.truncated_normal([num_danse_sizes[i-1], num_danse_sizes[i]], stddev=0.1), name='W')
                    b = tf.Variable(tf.constant(0.1, shape=[num_danse_sizes[i]]), name='b')
                    dense.append(tf.nn.sigmoid(tf.nn.xw_plus_b(dense[i-1], W, b)))
            i+=1

        with tf.name_scope('output'):
            W = tf.Variable(tf.truncated_normal([num_danse_sizes[-1], num_classes], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(dense[-1], W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.labels)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
