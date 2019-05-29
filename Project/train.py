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
from networks import VGG_like, VGG16, ResNet50
from utils import fetch_data, mini_batch
from tqdm import tqdm

tf.reset_default_graph()

#learning rate
lr = 0.00001
#number of traning steps
num_steps = 40
#batch_size
batch_size = 10
#num_input = 784
num_classes = 3

#fetch the data
#directory = "Desktop/ee596prepro/2019_04_09_bms1000/data"

bike1 = fetch_data("D:/tmp/ee596prepro/2019_04_09_bms1000/data", [1, 0, 0])
bike2 = fetch_data("D:/tmp/ee596prepro/2019_04_09_bms1001/data", [1, 0, 0])
bike3 = fetch_data("D:/tmp/ee596prepro/2019_04_09_bms1002/data", [1, 0, 0])
bike4 = fetch_data("D:/tmp/ee596prepro/2019_04_30_bm1s005/data", [1, 0, 0])
bike5 = fetch_data("D:/tmp/ee596prepro/2019_04_30_bm1s006/data", [1, 0, 0])

car1 = fetch_data("D:/tmp/ee596prepro/2019_04_09_cms1000/data", [0, 1, 0])
car2 = fetch_data("D:/tmp/ee596prepro/2019_04_09_cms1001/data", [0, 1, 0])
car3 = fetch_data("D:/tmp/ee596prepro/2019_04_09_cms1002/data", [0, 1, 0])
car4 = fetch_data("D:/tmp/ee596prepro/2019_04_30_cm1s000/data", [0, 1, 0])
car5 = fetch_data("D:/tmp/ee596prepro/2019_05_09_cm1s004/data", [0, 1, 0])
car6 = fetch_data("D:/tmp/ee596prepro/2019_05_09_cs1m001/data", [0, 1, 0])

ped1 = fetch_data("D:/tmp/ee596prepro/2019_04_09_pms1000/data", [0, 0, 1])
ped2 = fetch_data("D:/tmp/ee596prepro/2019_04_09_pms1001/data", [0, 0, 1])
ped3 = fetch_data("D:/tmp/ee596prepro/2019_04_09_pms2000/data", [0, 0, 1])
ped4 = fetch_data("D:/tmp/ee596prepro/2019_04_30_pm1s004/data", [0, 0, 1])
ped5 = fetch_data("D:/tmp/ee596prepro/2019_04_30_pm1s005/data", [0, 0, 1])

#full_list = bike1 + bike2 + bike3 + car1 + car2 + car3 + ped1 + ped2 + ped3
#full_list = 

training_set = bike1 + car1  + ped1 + bike2 + car2 + ped2  + bike3[0:200] + car3[0:200]  + ped3[0:200] +\
            bike4 + bike5 + car4 + car5 + car6 + ped4 + ped5 
valid_set =  bike3[-20:] + car3[-20:]  + ped3[-20:] + bike3[-60:-40] + car3[-60:-40]  + ped3[-60:-40] + \
bike3[-100:-80] + car3[-100:-80]  + ped3[-100:-80]
#test_set = full_list[975:999]
np.random.shuffle(training_set)
np.random.shuffle(valid_set)

#np.random.shuffle(full_list)
#print(full_list[0])

#print(len(full_list))
#print(len(full_list[0]))
#print(len(full_list[0][0]))
#print(np.asarray(full_list).shape)
#print(full_list[0])


train_set_data = []
train_set_labels = []
valid_set_data = []
valid_set_labels = []
#test_set_data = []
#test_set_labels = []

#print(np.asarray(training_set).shape)
#print(training_set[0][0][0])
#print(training_set[0][:][1])
#print(training_set[0][:][0].shape)
#print(training_set[0][:][0])
#split into training, valid, and testing
for i in range(len(training_set)):
    train_set_data.append(training_set[i][0])
    train_set_labels.append(training_set[i][1])
    
for i in range(len(valid_set)):
    valid_set_data.append(valid_set[i][0])
    valid_set_labels.append(valid_set[i][1])

# for i in range(len(test_set)):
#     test_set_data.append(test_set[i][0])
#     test_set_labels.append(test_set[i][1])

print(np.asarray(train_set_data).shape)
#print(train_set_data.shape)


train_set_data, train_set_labels = mini_batch(train_set_data,train_set_labels,batch_size)
valid_set_data, valid_set_labels = mini_batch(valid_set_data,valid_set_labels,batch_size)
#test_set_data, test_set_labels = mini_batch(test_set_data,test_set_labels,4)
print('the batch number of training set is ', len(train_set_data))
print('the batch number of validation set is ', len(valid_set_data))

training_set = None
valid_set = None
#test_set = None
full_list = None
bike1 = None
car1 = None
ped1 = None
bike2 = None
car2 = None
ped2 = None
bike3 = None
car3 = None
ped3 = None
bike4 = None
bike5 = None
car4 = None 
car5 = None
car6 = None
ped4 = None
ped5 = None

#tf graph input
X = tf.placeholder(tf.float32, [None,256,256,4], name='X')
Y = tf.placeholder(tf.int32, [None,num_classes], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

#TAKE ABS VALUE OF IMAGES


#predicted labels
logits = VGG16(X, keep_prob)

#define loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y),name='loss')
l2_loss = tf.losses.get_regularization_loss()
loss += l2_loss
#define optimizer
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)
#optimizer = tf.train.AdamOptimizer()
#train_op = optimizer.minimize(loss)

#compare the predicted labels with true labels
correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(Y,1))

#compute the accuracy by taking average
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32),name='accuracy')

#Initialize the variables
def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.gpu_options.allocator_type ='BFC'
    #config.gpu_options.per_process_gpu_memory_fraction = 0.90
    session = tf.Session(config=config)
    return session
sess = get_session()

#Initialize the variables
init = tf.global_variables_initializer()
#Save model
saver = tf.train.Saver()
acc_list = []
steps = []

#with tf.Session() as sess:
sess.run(init)

for i in range(num_steps):
    start_time = time.time()
    #epoch training
    acc_t = 0
    for j in range(len(train_set_data)):
        #fetch batch
        batch_x = train_set_data[j]
        #print("running")
        batch_y = train_set_labels[j]
        #run optimization
        sess.run(train_op, feed_dict={X:batch_x, Y:batch_y, keep_prob:0.5})
        loss1 = sess.run(loss, feed_dict={X:batch_x, Y:batch_y, keep_prob:1})
        acc1 = sess.run(accuracy, feed_dict={X:batch_x, Y:batch_y, keep_prob:1})
        #print('[%d, %d] loss: %.7f accuracy: %.7f' % (i, j, loss1, acc1))

        acc_t += acc1

    print('[%d, %d] loss: %.7f accuracy: %.7f' % (i, j, loss1, acc1))
    acc_t = acc_t/len(train_set_data)
    print("step "+str(i)+", Accuracy training= {:.3f}".format(acc_t))

    #epoch validation
    acc = 0
    for j in range(len(valid_set_data)):
        #fetch batch
        batch_x = valid_set_data[j]
        batch_y = valid_set_labels[j]
        #run optimization
        acc += sess.run(accuracy, feed_dict={X:batch_x, Y:batch_y, keep_prob:1})

    acc = acc/len(valid_set_data)
    print("step "+str(i)+", Accuracy validation= {:.3f}".format(acc))
    acc_list.append(acc)
    steps.append(i)

    print("--- %s seconds ---" % (time.time() - start_time))

saver.save(sess, 'C:/Users/Xiangyu Gao/Documents/GitHub/Practical-Introduction-NN-hw/Project/model/VGG_model_30.ckpt')
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
plt.plot(steps,acc_list,'--',lw=4)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.title('Epoch vs accuracy')