from datetime import datetime

import tensorflow as tf

from tensorflow.python.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dot, Dense, Reshape
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pydot
import graphviz


class P2VTensorFlow:
    def __init__(self, seed, num_epoch, data_streamer, num_n_samples, size_vector, model_name):
        self.model_name = model_name
        self.data_streamer = data_streamer
        self.num_epoch = num_epoch
        self.seed = seed
        self.n_products = data_streamer.get_num_products()
        self.num_n_samples = num_n_samples
        self.size_embedding = size_vector
        self.epoch_done = 0
        self.w_context = None
        self.w_center = None
        self.b_context = None
        self.b_center = None
        self.center_emb = None
        self.center_bias = None
        self.context_emb = None
        self.context_bias = None
        self.model = None

    def initialize_network(self):
        # Inputs needed for model
        center_input = tf.keras.Input(shape=[1, ], name='center_input')
        context_input = tf.keras.Input(shape=[1, ], name='positive_context')

        # Defining embedding layers needed in network
        self.w_context = Embedding(input_dim=self.n_products,
                                   output_dim=self.size_embedding,
                                   input_length=1,
                                   embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                                       stddev=.08, seed=self.seed),
                                   name='context_embedding')
        self.w_center = Embedding(input_dim=self.n_products,
                                  output_dim=self.size_embedding,
                                  input_length=1,
                                  embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                                      stddev=.08, seed=self.seed),
                                  name='center_embedding')
        self.b_context = Embedding(input_dim=self.n_products,
                                   output_dim=1,
                                   input_length=1,
                                   embeddings_initializer=tf.initializers.Constant(-1.5)
                                   , name='context_bias')
        self.b_center = Embedding(input_dim=self.n_products,
                                  output_dim=1,
                                  input_length=1,
                                  embeddings_initializer=tf.initializers.Constant(-1.5),
                                  name='center_bias')

        self.center_emb = self.w_center(center_input)
        self.center_emb = Reshape((self.size_embedding,), name='Reshape_center_emb')(self.center_emb)

        self.center_bias = self.b_center(center_input)
        self.center_bias = Reshape((1,), name='Reshape_center_bias')(self.center_bias)

        #self.context_emb = self.w_context(context_input)
        self.context_emb = self.w_center(context_input)
        self.context_emb = Reshape((self.size_embedding,), name='Reshape_context_emb')(self.context_emb)

        self.context_bias = self.b_context(context_input)
        self.context_bias = Reshape((1,), name='Reshape_context_bias')(self.context_bias)

        # Calculate similarity of center and context vector using dot product
        similarity = Dot(axes=1)([self.center_emb, self.context_emb])
        # Calculate score by adding bias to similarity
        score = similarity + self.center_bias + self.context_bias

        # Create output layer to get probabilities
        output = Dense(1, activation='sigmoid', name='Calculate_probabilities')(score)

        # Create model
        self.model = Model(inputs=[center_input, context_input],
                           outputs=[output],
                           name='product2vec')

        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                           optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, epsilon=1e-08),
                           metrics=['accuracy'])

    def train_network(self):
        csv_logger = tf.keras.callbacks.CSVLogger(self.model_name+'.log', append=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=r".\logs", histogram_freq=1,
                                                              write_graph=True, write_images=True, embeddings_freq=100)
        for i in (range(self.num_epoch)):
            num_samples = self.data_streamer.get_num_samples()
            data_stream = self.data_streamer.create_batches()
            with tqdm(total=num_samples // self.data_streamer.batch_size) as pbar:
                for x, train_data in enumerate(data_stream):
                    batch_size = train_data[0].shape[0]
                    if x == num_samples // self.data_streamer.batch_size - 1:
                        self.model.fit(x=[train_data[0][:-10], train_data[1][:-10]],
                                       y=train_data[2][:-10],
                                       epochs=1,
                                       shuffle=True,
                                       batch_size=batch_size,
                                       verbose=0,
                                       callbacks=[csv_logger],
                                       )
                        output_val = self.model.predict([train_data[0][:10], train_data[1][:10]])
                        print(output_val)
                        print(train_data[2][:10])
                    else:
                        self.model.fit(x=[train_data[0], train_data[1]],
                                       y=train_data[2],
                                       epochs=1,
                                       shuffle=True,
                                       batch_size=batch_size,
                                       verbose=0,
                                       )

                    pbar.update(1)

        self.model.save("simulated_test.h5")

    def train_one_epoch_and_log(self):
        num_samples = self.data_streamer.get_num_samples()
        data_stream = self.data_streamer.create_batches()
        loss_history = []
        accuracy_history = []
        with tqdm(total=num_samples // self.data_streamer.batch_size) as pbar:
            for x, train_data in enumerate(data_stream):
                batch_size = train_data[0].shape[0]
                if x == num_samples // self.data_streamer.batch_size - 1:
                    history = self.model.fit(x=[train_data[0][:-10], train_data[1][:-10]],
                                             y=train_data[2][:-10],
                                             epochs=1,
                                             shuffle=True,
                                             batch_size=batch_size,
                                             verbose=0,
                                             )
                    output_val = self.model.predict([train_data[0][:10], train_data[1][:10]])
                    print(output_val)
                    print(train_data[2][:10])
                    loss_history.append(history.history['loss'])
                    accuracy_history.append(history.history['accuracy'])
                else:
                    history = self.model.fit(x=[train_data[0], train_data[1]],
                                             y=train_data[2],
                                             epochs=1,
                                             shuffle=True,
                                             batch_size=batch_size,
                                             verbose=0,
                                             )
                    loss_history.append(history.history['loss'])
                    accuracy_history.append(history.history['accuracy'])

                pbar.update(1)

        steps = np.arange(len(loss_history))
        loss_history = np.array(loss_history)
        accuracy_history = np.array(accuracy_history)

        self.plot_loss(loss_history, steps)
        self.plot_acc(accuracy_history, steps)

    def plot_acc(self, acc, steps):
        plt.plot(steps, acc)
        plt.show()

    def plot_loss(self, loss, steps):
        plt.plot(steps, loss)
        plt.show()


class centerContextLabel:

    def __init__(self, loc_center, loc_pos, loc_neg, n_neg_samples, batch_size):
        self.batch_size = batch_size
        self.center_products = np.loadtxt(loc_center, delimiter=",",
                                          dtype=np.int32)
        self.positive_products = np.loadtxt(loc_pos, delimiter=",",
                                            dtype=np.int32)
        self.negative_products = np.loadtxt(loc_neg, delimiter=",",
                                            dtype=np.int32)
        self.n_neg_samples = n_neg_samples
        self.batch_cache_center = []
        self.batch_cache_context = []
        self.batch_cache_labels = []

    def create_batches(self):
        for i, center_product in enumerate(self.center_products):
            indexes = np.array(range(self.n_neg_samples + 1))
            np.random.shuffle(indexes)
            for index in indexes:
                if index == 0:
                    self.batch_cache_center.append(np.array(center_product))
                    self.batch_cache_context.append(np.array(self.positive_products[i]))
                    self.batch_cache_labels.append(1)
                else:
                    self.batch_cache_center.append(np.array(center_product))
                    self.batch_cache_context.append(np.array(self.negative_products[i][index - 1]))
                    self.batch_cache_labels.append(0)
            if len(self.batch_cache_center) == self.batch_size or i == self.center_products.shape[0] - 1:
                yield [np.array(self.batch_cache_center), np.array(self.batch_cache_context),
                       np.array(self.batch_cache_labels)]
                self.clear_cache()

    def clear_cache(self):
        self.batch_cache_center = []
        self.batch_cache_context = []
        self.batch_cache_labels = []

    def get_num_products(self):
        return max(np.max(self.center_products), np.max(self.positive_products), np.max(self.negative_products)) + 1

    def get_num_samples(self):
        return self.positive_products.shape[0] + self.negative_products.size

    def get_batch_size(self):
        return self.batch_size


# Batch size must be multiple of 21
if __name__ == '__main__':
    # Batch size must be multiple of 21
    print("{} INFO: Initializing data streamer".format(
        datetime.now().strftime("%H:%M:%S")))
    location_center = r'large_data\center_products_simulated'
    location_pos = r'large_data\positive_context_products_simulated'
    location_neg = r'large_data\negative_context_products_simulated'
    train_streamer = centerContextLabel(loc_center=location_center, loc_pos=location_pos, loc_neg=location_neg,
                                        n_neg_samples=20, batch_size=2016)
    test = P2VTensorFlow(1000, 5, data_streamer=train_streamer, num_n_samples=20, size_vector=30,
                         model_name='simulated_test')
    test.initialize_network()
    test.train_network()
