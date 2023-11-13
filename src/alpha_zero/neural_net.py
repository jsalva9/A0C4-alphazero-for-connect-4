import tensorflow as tf
import numpy as np
from src.boards.bitboard import ConnectGameBitboard as Game
from src.utils import NNConf


class NeuralNetwork:
    def __init__(self, game: Game):
        self.row = game.h
        self.column = game.w
        self.action_size = game.w
        self.pi = None
        self.v = None

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.states = tf.keras.Input(tf.float32, shape=[None, self.row, self.column])
            self.training = tf.keras.Input(tf.bool)

            # Input Layer
            input_layer = tf.reshape(self.states, [-1, self.row, self.column, 1])

            # Redo in Tf2
            conv1 = tf.keras.layers.Conv2D(
                filters=256,
                kernel_size=[3, 3],
                padding="same",
                strides=1)(input_layer)

            # Redo in Tf2
            batch_norm1 = tf.keras.layers.BatchNormalization(
                training=self.training)(conv1)

            relu1 = tf.nn.relu(batch_norm1)

            resnet_in_out = relu1

            # Residual Tower
            for i in range(NNConf['resnet_blocks']):
                conv2 = tf.keras.layers.Conv2D(
                    filters=256,
                    kernel_size=[3, 3],
                    padding="same",
                    strides=1)(resnet_in_out)

                batch_norm2 = tf.keras.layers.BatchNormalization(
                    training=self.training)(conv2)

                relu2 = tf.nn.relu(batch_norm2)

                conv3 = tf.keras.layers.Conv2D(
                    filters=256,
                    kernel_size=[3, 3],
                    padding="same",
                    strides=1)(relu2)

                batch_norm3 = tf.keras.layers.BatchNormalization(
                    training=self.training)(conv3)

                resnet_skip = tf.add(batch_norm3, resnet_in_out)

                resnet_in_out = tf.nn.relu(resnet_skip)

            # Policy head
            conv4 = tf.keras.layers.Conv2D(
                filters=2,
                kernel_size=[1, 1],
                padding="same",
                strides=1)(resnet_in_out)

            batch_norm4 = tf.keras.layers.BatchNormalization(training=self.training)(conv4)

            relu4 = tf.nn.relu(batch_norm4)

            relu4_flat = tf.reshape(relu4, [-1, self.row * self.column * 2])

            logits = tf.keras.layers.Dense(units=self.action_size)(relu4_flat)

            self.pi = tf.nn.softmax(logits)

            # Value Head
            conv5 = tf.keras.layers.Conv2D(
                filters=1,
                kernel_size=[1, 1],
                padding="same",
                strides=1)(resnet_in_out)

            batch_norm5 = tf.keras.layers.BatchNormalization(training=self.training)(conv5)

            relu5 = tf.nn.relu(batch_norm5)

            relu5_flat = tf.reshape(relu5, [-1, self.action_size])

            dense1 = tf.keras.layers.Dense(units=256)(relu5_flat)

            relu6 = tf.nn.relu(dense1)

            dense2 = tf.keras.layers.Dense(units=1)(relu6)

            self.v = tf.nn.tanh(dense2)

            # Loss Function
            self.train_pis = tf.keras.Input(tf.float32, shape=[None, self.action_size])
            self.train_vs = tf.keras.Input(tf.float32, shape=[None])

            # self.loss_pi = tf.losses.softmax_cross_entropy(self.train_pis, self.pi)
            # translate to tf2
            self.loss_pi = tf.nn.softmax_cross_entropy_with_logits(self.train_pis, self.pi)

            self.loss_v = tf.losses.mean_squared_error(self.train_vs, tf.reshape(self.v, shape=[-1, ]))
            self.total_loss = self.loss_pi + self.loss_v

            # Stochastic gradient descent with momentum
            # global_step = tf.Variable(0, trainable=False)

            # learning_rate = tf.train.exponential_decay(CFG.learning_rate,
            #                                            global_step,
            #                                            20000,
            #                                            0.96,
            #                                            staircase=True)

            optimizer = tf.keras.optimizers.SGD(
                learning_rate=NNConf['learning_rate'],
                momentum=NNConf['momentum'],
                nesterov=False)

            self.train_op = optimizer.minimize(self.total_loss)

            # Create a saver for writing training checkpoints.
            self.saver = tf.train.Saver()

            # Create a session for running Ops on the Graph.
            self.sess = tf.Session()

            # Initialize the session.
            self.sess.run(tf.global_variables_initializer())


class NNWrapper:
    def __init__(self, game):
        """Initializes NeuralNetworkWrapper with game state and TF session."""
        self.game = game
        self.net = NeuralNetwork(self.game)
        self.sess = self.net.sess

    def predict(self, state):
        """Predicts move probabilities and state values given a game state.

        Args:
            state: A list containing the game state in matrix form.

        Returns:
            A probability vector and a value scalar
        """
        state = state[np.newaxis, :, :]

        pi, v = self.sess.run([self.net.pi, self.net.v], feed_dict={self.net.states: state, self.net.training: False})

        return pi[0], v[0][0]

    def train(self, training_data):
        """Trains the network using states, pis and vs from self play games.

        Args:
            training_data: A list containing states, pis and vs
        """
        print("\nTraining the network.\n")

        for epoch in range(NNConf['epochs']):
            print("Epoch", epoch + 1)

            examples_num = len(training_data)

            # Divide epoch into batches.
            for i in range(0, examples_num, NNConf['batch_size']):
                states, pis, vs = map(list,
                                      zip(*training_data[i:i + NNConf['batch_size']]))

                feed_dict = {self.net.states: states,
                             self.net.train_pis: pis,
                             self.net.train_vs: vs,
                             self.net.training: True}

                self.sess.run(self.net.train_op,
                              feed_dict=feed_dict)

                pi_loss, v_loss = self.sess.run(
                    [self.net.loss_pi, self.net.loss_v],
                    feed_dict=feed_dict)

                # Record pi and v loss to a file.
                if NNConf['record_loss']:

                    file_path = f'{NNConf["model_directory"]}{NNConf["loss_file"]}'

                    with open(file_path, 'a') as loss_file:
                        loss_file.write('%f|%f\n' % (pi_loss, v_loss))

        print("\n")

    def save_model(self, filename="current_model"):
        """Saves the network model at the given file path.

        Args:
            filename: A string representing the model name.
        """
        # Create directory if it doesn't exist.
        file_path = f'{NNConf["model_directory"]}{filename}'

        print("Saving model:", filename, "at", NNConf["model_directory"])
        self.net.saver.save(self.sess, file_path)

    def load_model(self, filename="current_model"):
        """Loads the network model at the given file path.

        Args:
            filename: A string representing the model name.
        """
        file_path = f'{NNConf["model_directory"]}{filename}'

        print("Loading model:", filename, "from", NNConf["model_directory"])
        self.net.saver.restore(self.sess, file_path)
