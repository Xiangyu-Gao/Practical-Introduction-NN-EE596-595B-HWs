import tensorflow as tf
from AlexNetHelper import AlexNetHelper
from DogsAndCatsHelper import DogsAndCatsHelper
import  numpy as np

class AlexNet:

    def __init__(self,
                 batch_size = 32,
                 epochs = 100):
        """
        Args:
            batch_size: number of examples per batch_size
            epochs: number of iterations over all training examples
        """
        self.batch_size = batch_size
        self.epochs = epochs

    @staticmethod
    def inference(images, num_classes):
        """Builds AlexNet Graph

        Args:
            images: train/test images
            num_classes: number of classes
        Returns:
            final_node: last node in the graph
        """

        # Layer 1
        with tf.name_scope("conv1") as scope:
            kernel = AlexNetHelper.instintiate_weights('W1', [11, 11, 3, 96])
            conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
            biases = AlexNetHelper.instintiate_bias('b1', [96])
            conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)

        with tf.name_scope('batch_norm1') as scope:
            mean, var = tf.nn.moments(conv1, [0, 1, 2])
            batch_norm1 = tf.nn.batch_normalization(conv1, mean, var, 0, 1, 0, name=scope)

        with tf.name_scope('pool1') as scope:
            pool1 = tf.nn.max_pool(batch_norm1, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name=scope)

        # Layer 2
        with tf.name_scope("conv2") as scope:
            kernel = AlexNetHelper.instintiate_weights('W2', [5, 5, 96, 256])
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = AlexNetHelper.instintiate_bias('b2', [256])
            conv2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)

        with tf.name_scope('batch_norm2') as scope:
            mean, var = tf.nn.moments(conv2, [0, 1, 2])
            batch_norm2 = tf.nn.batch_normalization(conv2, mean, var, 0, 1, 0, name=scope)

        with tf.name_scope('pool2') as scope:
            pool2 = tf.nn.max_pool(batch_norm2, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name=scope)

        # Layer 3
        with tf.name_scope("conv3") as scope:
            kernel = AlexNetHelper.instintiate_weights('W3', [3, 3, 256, 384])
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = AlexNetHelper.instintiate_bias('b3', [384])
            conv3 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)

        with tf.name_scope('batch_norm3') as scope:
            mean, var = tf.nn.moments(conv3, [0, 1, 2])
            batch_norm3 = tf.nn.batch_normalization(conv3, mean, var, 0, 1, 0, name=scope)

        # Layer 4
        with tf.name_scope("conv4") as scope:
            kernel = AlexNetHelper.instintiate_weights('W4', [3, 3, 384, 384])
            conv = tf.nn.conv2d(batch_norm3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = AlexNetHelper.instintiate_bias('b4', [384])
            conv4 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)

        with tf.name_scope('batch_norm4') as scope:
            mean, var = tf.nn.moments(conv4, [0, 1, 2])
            batch_norm4 = tf.nn.batch_normalization(conv4, mean, var, 0, 1, 0, name=scope)

        # Layer 5
        with tf.name_scope("conv5") as scope:
            kernel = AlexNetHelper.instintiate_weights('W5', [3, 3, 384, 256])
            conv = tf.nn.conv2d(batch_norm4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = AlexNetHelper.instintiate_bias('b5', [256])
            conv5 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)

        with tf.name_scope('batch_norm5') as scope:
            mean, var = tf.nn.moments(conv5, [0, 1, 2])
            batch_norm5 = tf.nn.batch_normalization(conv5, mean, var, 0, 1, 0, name=scope)

        with tf.name_scope('pool5') as scope:
            pool5 = tf.nn.max_pool(batch_norm5, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name=scope)

        # Fully Connected 6
        with tf.name_scope('FC6') as scope:
            pool5_flat = tf.contrib.layers.flatten(pool5)
            fc6 = tf.layers.dense(pool5_flat, units=4096, activation=tf.nn.relu, name=scope)
            # fc6_dropout = tf.layers.dropout(fc6, 0.5)

        with tf.name_scope('FC7') as scope:
            fc7 = tf.layers.dense(fc6, units=4096, activation=tf.nn.relu, name=scope)
            # fc7_dropout = tf.layers.dropout(fc7, 0.5)

        with tf.name_scope('classes') as scope:
            predictions = tf.layers.dense(fc7, units=num_classes, activation=tf.nn.softmax, name=scope)

        return predictions

    def run(self):
        """
        Runtime for AlexNet
        :param
        train_data: training data
        num_classes: num_classes
        :return: None
        """

        train_images, train_labels, valid_images= DogsAndCatsHelper.get_data()
        num_classes = train_labels.shape[1]

        images = tf.placeholder(tf.float32, shape=[None, *train_images[0].shape])
        labels = tf.placeholder(tf.float32, shape=[None, 2])

        predictions = self.inference(images, num_classes)
        init = tf.global_variables_initializer()

        loss = tf.nn.l2_loss(tf.square(predictions - labels))
        learner = tf.train.AdamOptimizer(0.001)
        optimizer = learner.minimize(loss)

        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            sess.run(init)
            for _ in range(self.epochs):
                l_total = 0
                acc = 0
                for i in range(0, len(train_images), self.batch_size):
                    l, acc, _ = sess.run(fetches=[loss, accuracy, optimizer],
                                         feed_dict={images: train_images[i: self.batch_size + i],
                                                    labels: train_labels[i: self.batch_size + i]})
                    l_total += np.sum(l)

                    print(l_total, acc)
                # val_acc, _ = sess.run([accuracy, predictions],
                #                        feed_dict={images: valid_images, labels: valid_labels})
                # print("Validation accuracy: ", val_acc)
