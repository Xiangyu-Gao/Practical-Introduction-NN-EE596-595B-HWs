import tensorflow as tf
import pickle
import numpy as np
import timeit
import scipy.io as spio
import os
import cv2
import matplotlib.pyplot as plt
import time

from tensorflow.contrib.layers import flatten
from networks import VGG_like, VGG16, ResNet50, AlexNet
from utils import fetch_data, mini_batch
from tqdm import tqdm

tf.reset_default_graph()

# learning rate
lr = 0.00001
# number of traning steps
num_steps = 1
# batch_size
batch_size = 10
# num_input = 784
num_classes = 3

# fetch the data
# directory = "Desktop/ee596prepro/2019_04_09_bms1000/data"


car5 = fetch_data("/mnt/disk1/temp/ee596prepro/2019_05_09_cm1s004/data", [0, 1, 0])
bike5 = fetch_data("/mnt/disk1/temp/ee596prepro/2019_04_30_bm1s006/data", [1, 0, 0])
ped5 = fetch_data("/mnt/disk1/temp/ee596prepro/2019_04_30_pm1s005/data", [0, 0, 1])

test_set = car5[150:]

np.random.shuffle(test_set)

# np.random.shuffle(full_list)
# print(full_list[0])

# print(len(full_list))
# print(len(full_list[0]))
# print(len(full_list[0][0]))
# print(np.asarray(full_list).shape)
# print(full_list[0])

test_set_data = []
test_set_labels = []

# print(np.asarray(training_set).shape)
# print(training_set[0][0][0])
# print(training_set[0][:][1])
# print(training_set[0][:][0].shape)
# print(training_set[0][:][0])
# split into training, valid, and testing


for i in range(len(test_set)):
    test_set_data.append(test_set[i][0])
    test_set_labels.append(test_set[i][1])

# print(train_set_data.shape)

test_set_data, test_set_labels = mini_batch(test_set_data, test_set_labels, batch_size)

print('the batch number of testing set is ', len(test_set_data))

test_set = None

bike3 = None
car3 = None
ped3 = None

# tf graph input
X = tf.placeholder(tf.float32, [None, 256, 256, 4], name='X')
Y = tf.placeholder(tf.int32, [None, num_classes], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# TAKE ABS VALUE OF IMAGES


# predicted labels
logits = VGG16(X, keep_prob)

# define loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y), name='loss')
l2_loss = tf.losses.get_regularization_loss()
loss += l2_loss
# define optimizer
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)
# optimizer = tf.train.AdamOptimizer()
# train_op = optimizer.minimize(loss)

# compare the predicted labels with true labels
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))

# compute the accuracy by taking average
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


# Initialize the variables
def get_session():
    """Create a session that dynamically allocates memory."""
    #See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type ='BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.90
    session = tf.Session(config=config)
    # session = tf.Session()
    return session


sess = get_session()

# Initialize the variables
init = tf.global_variables_initializer()
# Save model
saver = tf.train.Saver()
acc_list = []
steps = []

# with tf.Session() as sess:
#sess.run(init)
# restore weights
weights_path = '/home/admin-cmmb/Documents/Practical-Introduction-NN-hw-master/model'
saver.restore(sess, weights_path)

for i in range(num_steps):
    start_time = time.time()
    # # epoch training
    # acc_t = 0
    # for j in range(len(train_set_data)):
    #     # fetch batch
    #     batch_x = train_set_data[j]
    #     # print("running")
    #     batch_y = train_set_labels[j]
    #     # run optimization
    #     sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
    #     loss1 = sess.run(loss, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1})
    #     acc1 = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1})
    #     # print('[%d, %d] loss: %.7f accuracy: %.7f' % (i, j, loss1, acc1))
    #
    #     acc_t += acc1
    #
    # print('[%d, %d] loss: %.7f accuracy: %.7f' % (i, j, loss1, acc1))
    # acc_t = acc_t / len(train_set_data)
    # print("step " + str(i) + ", Accuracy training= {:.3f}".format(acc_t))

    # epoch testing
    acc = 0
    los = 0
    for j in range(len(test_set_data)):
        # fetch batch
        batch_x = test_set_data[j]
        batch_y = test_set_labels[j]
        # run optimization
        acc2 = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
        loss2 = sess.run(loss, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
        print('[%d, %d] loss: %.7f accuracy: %.7f' % (i, j, loss2, acc2))

        acc += acc2
        los += loss2

    acc = acc / len(test_set_data)
    los = los / len(test_set_data)
    print("step " + str(i) + ", Accuracy testing = {:.3f}, Loss testing = {:.3f}".format(acc, los))
    acc_list.append(acc)
    steps.append(i)

    print("--- %s seconds ---" % (time.time() - start_time))

saver.save(sess, '/home/admin-cmmb/Documents/Practical-Introduction-NN-hw-master/model')
print("Training finished!")

# acc = 0
# for k in range(len(test_set_data)):
# #fetch batch
#     batch_x = test_set_data[k]
#     batch_y = test_set_labels[k]
#     #run optimization
#     acc += sess.run(accuracy, feed_dict={X:batch_x, Y:batch_y})


# acc = acc/len(test_set_data)
# print("Test Accuracy= {:.3f}".format(acc))

# #print the first images
# batch_x = test_set_data[0]
# batch_y = test_set_labels[0]
# #run optimization
# guesses = sess.run(just_soft, feed_dict={X:batch_x, Y:batch_y})


# for images in range(10):
#     cur_img = batch_x[images]
#     b,g,r = cv2.split(cur_img)
#     frame_rgb = cv2.merge((r,g,b))
#     plt.imshow(frame_rgb)
#     print("Guess:", guesses[images])
#     plt.show()

plt.figure()
# plot epoch vs accuracy
plt.plot(steps, acc_list, '--', lw=4)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.title('Epoch vs accuracy')
plt.show()