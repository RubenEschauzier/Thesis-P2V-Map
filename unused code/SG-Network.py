from random import random, randint
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, dot, Dense, Reshape, Add
import numpy as np


class P2VTensorFlow:
    def __init__(self, seed, num_epoch, train_center, train_pos_context, train_neg_context, num_n_samples, size_vector):
        self.num_epoch = num_epoch
        self.seed = seed
        self.n_products = max(max(x) for x in train_neg_context) + 1
        self.train_center = train_center
        self.train_pos_context = train_pos_context
        self.train_neg_context = train_neg_context
        self.num_n_samples = num_n_samples
        self.size_embedding = size_vector
        self.epoch_done = 0
        self.w_context = None
        self.w_center = None
        self.b_context = None
        self.b_center = None
        self.center_emb = None
        self.positive_emb = None
        self.negative_emb = None
        self.center_bias = None
        self.pos_bias = None
        self.neg_bias = None
        self.model = None

    def initialize_network(self):
        # Inputs needed for model
        center_vector = tf.keras.Input(shape=[1, ], name='center_input')
        positive_vector = tf.keras.Input(shape=[1, ], name='positive_context')
        negative_vector = tf.keras.Input(shape=[self.num_n_samples, ], name='negative_context')

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

        self.center_emb = self.w_center(center_vector)
        self.center_emb = Reshape((self.size_embedding,), name='Reshape_center_emb')(self.center_emb)
        self.center_bias = self.b_center(center_vector)
        self.center_bias = Reshape((1,), name='Reshape_center_bias')(self.center_bias)

        self.positive_emb = self.w_context(positive_vector)
        self.positive_emb = Reshape((self.size_embedding,), name='Reshape_positive_emb')(self.positive_emb)
        self.pos_bias = self.b_context(positive_vector)
        self.pos_bias = Reshape((1,), name='Reshape_positive_bias')(self.pos_bias)

        self.negative_emb = self.w_context(negative_vector)
        self.neg_bias = self.b_context(negative_vector)
        self.neg_bias = Reshape((self.num_n_samples,), name='Reshape_negative_bias')(self.neg_bias)

        logit_pos_context = tf.einsum('ij,ij->i', self.center_emb, self.positive_emb)

        logit_neg_context = tf.einsum('ik,ijk->ij', self.center_emb, self.negative_emb)

        test = Add(name='pos_logit')([logit_pos_context, self.pos_bias, self.center_bias])
        logit_pos_context = logit_pos_context + self.pos_bias + self.center_bias

        test2 = Add(name='neg_logit')([logit_neg_context, self.neg_bias, tf.reshape(self.center_bias, shape=[-1, 1])])
        logit_neg_context = logit_neg_context + self.neg_bias + tf.reshape(self.center_bias, shape=[-1, 1])
        output_pos = Dense(1, activation='sigmoid', name='pos_prediction')(test)
        output_neg = Dense(self.num_n_samples, activation='sigmoid', name='neg_prediction')(test2)
        # Calculate score by adding bias to similarity
        # output = (Dense(1 + self.num_n_samples, activation='softmax', name='Calculate_probabilities')(score))

        # Create model
        self.model = Model([center_vector, positive_vector, negative_vector],
                           [output_pos, output_neg],
                           name='product2vec')

        self.model.compile(loss=[tf.keras.losses.BinaryCrossentropy(from_logits=False),
                                 tf.keras.losses.BinaryCrossentropy(from_logits=False)],
                           loss_weights=[1., 1],
                           optimizer='adam',
                           metrics={'pos_prediction': 'accuracy', 'neg_prediction': 'accuracy'}
                           )

    def train_network(self):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=r".\logs", histogram_freq=1,
                                                              write_graph=True, write_images=True, embeddings_freq=1)
        labels = np.zeros((self.train_center.shape[0], self.num_n_samples))
        labels_pos = np.ones((self.train_center.shape[0]))
        """for i in range(self.train_center.shape[0]):
            index = randint(0, self.num_n_samples)
            labels[i][0] = 1"""

        """test_model = Model(inputs=self.model.input, outputs=self.model.get_layer(
            'Calculate_probabilities').output)
        output_test = test_model.predict( 
            [self.train_center[:10], self.train_pos_context[:10], self.train_neg_context[:10]])
        print(output_test)
        print(self.train_center[:10])
        print(self.train_pos_context[:10])
        print(self.train_neg_context[:10])
        print(labels[:10])"""

        self.model.fit(x=[self.train_center, self.train_pos_context, self.train_neg_context],
                       y={"pos_prediction": labels_pos, "neg_prediction": labels},
                       epochs=self.num_epoch,
                       shuffle=True,
                       batch_size=64,
                       validation_split=0.2,
                       callbacks=[tensorboard_callback]
                       )
        prediction = self.model.predict(
            x=[self.train_center[:-20], self.train_pos_context[:-20], self.train_neg_context[:-20]])
        print(prediction)


# Load small dataset from disk
center = np.loadtxt(r'large_data\small_center_products', delimiter=",", dtype=np.int32)
pos_context = np.loadtxt(r'large_data\small_positive_context_products', delimiter=",", dtype=np.int32)
neg_context = np.loadtxt(r'large_data\small_negative_context_products', delimiter=",", dtype=np.int32)

test = P2VTensorFlow(1000, 2, center, pos_context, neg_context, 20, 30)
test.initialize_network()
test.train_network()
