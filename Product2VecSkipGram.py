import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import reduce

import tensorflow as tf
import numpy as np
import math
from sklearn.utils import shuffle


class Product2VecSkipGram:
    def __init__(self, data, cv_data, batch_size, num_skips, skip_window, vocabulary_size, embedding_size=32,
                 num_negative_sampled=64, len_ratio = 0.5):
        self.data = data
        self.cv_data = cv_data
        self.data_index = 0
        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window
        self.embedding_size = embedding_size
        self.num_negative_sampled = num_negative_sampled
        self.vocabulary_size = vocabulary_size
        self.len_ratio = len_ratio
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        self.build_graph()

    def predict(self, products):
        result = []
        for i in range(0, len(products), self.batch_size):
            batch = products[i:i+self.batch_size]
            batch = self.sess.run(self.gathered, feed_dict={self.train_inputs: batch})
            result.append(batch)
        return np.concatenate(result, axis=0)

    def train(self, num_steps, cv_every_n_steps, cv_steps, lrs):
        with ThreadPoolExecutor(max_workers=2) as executor:
            average_loss = 0
            learning_rate = 1.0
            current = executor.submit(self.generate_batch)
            for step in range(num_steps):
                if step in lrs:
                    learning_rate = lrs[step]
                batch_inputs, batch_labels = current.result()
                current = executor.submit(self.generate_batch)
                feed_dict = {self.train_inputs: batch_inputs,
                             self.train_labels: batch_labels,
                             self.learning_rate: learning_rate}

                _, loss_val = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0
                if step % cv_every_n_steps == 0:
                    self.data = shuffle(self.data, random_state=0)
                    self.save_model(step)
                    cv_loss = 0
                    for batch_inputs, batch_labels in self.generate_test(cv_steps):
                        feed_dict = {self.train_inputs: batch_inputs,
                                     self.train_labels: batch_labels,
                                     self.learning_rate: learning_rate}
                        loss_val = self.sess.run(self.loss, feed_dict=feed_dict)
                        cv_loss += loss_val
                    print('CV',cv_loss / cv_steps)

    def save_model(self, step):
        self.saver.save(self.sess, 'models/prod2vec_skip_gram', global_step=step)

    def load_model(self, path):
        self.saver.restore(self.sess, path)

    def build_graph(self):
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.learning_rate = tf.placeholder(tf.float32)

        # variables
        embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))

        softmax_weights = tf.Variable(tf.truncated_normal([self.embedding_size, self.vocabulary_size],
                                                          stddev=1.0 / math.sqrt(self.embedding_size)))
        softmax_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

        self.gathered = tf.gather(embeddings, self.train_inputs)

        prediction = tf.matmul(self.gathered, softmax_weights) + softmax_biases
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.train_labels, logits=prediction))

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()


    def inc(self):
        self.data_index = (self.data_index + 1) % len(self.data)

    def inc_cv(self, data_index):
        return (data_index + 1) % len(self.cv_data)

    def generate_batch(self):
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        counter = 0
        while counter < self.batch_size:
            current = self.data.iloc[self.data_index]
            if len(current) == 1:
                warnings.warn("lenght is one", RuntimeWarning)
                self.inc()
                continue

            span = min(2 * self.skip_window + 1, len(current))

            x = target = np.random.randint(0, len(current))

            targets_to_avoid = [x]

            for j in range(self.num_skips):  # target varies!!! X constant!
                while target in targets_to_avoid and len(targets_to_avoid) != span:
                    target = np.random.randint(0, span)
                if len(targets_to_avoid) == span or counter == self.batch_size:
                    break
                targets_to_avoid.append(target)
                batch[counter] = current[x]
                labels[counter] = current[target]
                counter += 1
            self.inc()

        return batch, labels

    def generate_test(self, num_steps):
        data_index = 0
        for _ in range(num_steps):
            batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
            labels = np.ndarray(shape=(self.batch_size), dtype=np.int32)

            counter = 0
            while counter < self.batch_size:
                current = self.cv_data.iloc[data_index]
                if len(current) == 1:
                    warnings.warn("lenght is one", RuntimeWarning)
                    data_index = self.inc_cv(data_index)
                    continue

                span = min(2 * self.skip_window + 1, len(current))

                x = target = np.random.randint(0, len(current))

                targets_to_avoid = [x]

                for j in range(self.num_skips):  # target varies!!! X constant!
                    while target in targets_to_avoid and len(targets_to_avoid) != span:
                        target = np.random.randint(0, span)
                    if len(targets_to_avoid) == span or counter == self.batch_size:
                        break
                    targets_to_avoid.append(target)
                    batch[counter] = current[x]
                    labels[counter] = current[target]
                    counter += 1
                data_index = self.inc_cv(data_index)

            yield batch, labels
