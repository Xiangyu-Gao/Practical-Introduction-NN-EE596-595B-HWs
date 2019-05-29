import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten

num_classes = 3

def VGG_like(x, keep_prob):    
    #first conv/pool pair
    #filters, then kernel size
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    conv1 = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, kernel_regularizer= regularizer, \
        kernel_initializer=tf.contrib.layers.xavier_initializer())
    #pool size, then stride
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
    
    #second conv/pool pair
    conv2 = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu, kernel_regularizer = regularizer, 
        kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
    #drop2 = tf.nn.dropout(pool2, 0.95)

    #third conv/pool pair
    conv3 = tf.layers.conv2d(pool2, 128, 3, activation=tf.nn.relu, kernel_regularizer = regularizer, \
        kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2)
    drop3 = tf.nn.dropout(pool3, keep_prob = keep_prob)
    
   
    #flatten to connect to fully connected
    full_in = flatten(drop3)
    
    #fully connected layer
    full1 = tf.contrib.layers.fully_connected(inputs=full_in, num_outputs=1024, activation_fn=tf.nn.relu, \
        weights_regularizer = regularizer)
    drop4 = tf.nn.dropout(full1, keep_prob = keep_prob)
    full2 = tf.contrib.layers.fully_connected(inputs=drop4, num_outputs=256, activation_fn=tf.nn.relu, \
        weights_regularizer = regularizer)
    drop5 = tf.nn.dropout(full2, keep_prob = keep_prob)
    logits = tf.contrib.layers.fully_connected(inputs=drop5, num_outputs=num_classes, activation_fn=None, \
        weights_regularizer = regularizer)
    return logits

# def VGG_like(x):    

#     regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
#     #first conv/pool pair
#     #filters, then kernel size
#     conv1 = tf.layers.conv2d(x, 30, 5, activation=tf.nn.relu)
#     #pool size, then stride
#     pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
    
#     #second conv/pool pair
#     conv2 = tf.layers.conv2d(pool1, 15, 3, activation=tf.nn.relu)
#     pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
#     drop2 = tf.nn.dropout(pool2, 0.95)
    
#     #flatten to connect to fully connected
#     full_in = flatten(drop2)
    
#     #fully connected layer
#     full1 = tf.contrib.layers.fully_connected(inputs=full_in, num_outputs=128, activation_fn=tf.nn.relu, weights_regularizer = regularizer)
#     drop3 = tf.nn.dropout(full1, 0.95)
#     full2 = tf.contrib.layers.fully_connected(inputs=drop3, num_outputs=50, activation_fn=tf.nn.relu, weights_regularizer = regularizer)
#     drop4 = tf.nn.dropout(full2, 0.95)
#     logits = tf.contrib.layers.fully_connected(inputs=drop4, num_outputs=num_classes, activation_fn=None, weights_regularizer = regularizer)
#     return logits

def VGG16(x, keep_prob):    
    #first conv/pool pair
    #filters, then kernel size
    conv1 = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu, padding = "same")
    conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu, padding = "same")
    #pool size, then stride
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2) #usually 2,2 trying 5,5
    
    #filters, then kernel size
    conv3 = tf.layers.conv2d(pool2, 128, 3, activation=tf.nn.relu, padding = "same")
    conv4 = tf.layers.conv2d(conv3, 128, 3, activation=tf.nn.relu, padding = "same")
    #pool size, then stride
    pool4 = tf.layers.max_pooling2d(conv4, 2, 2)
    
    #filters, then kernel size
    conv5 = tf.layers.conv2d(pool4, 256, 3, activation=tf.nn.relu, padding = "same")
    conv6 = tf.layers.conv2d(conv5, 256, 3, activation=tf.nn.relu, padding = "same")
    conv7 = tf.layers.conv2d(conv6, 256, 3, activation=tf.nn.relu, padding = "same")
    #pool size, then stride
    pool7 = tf.layers.max_pooling2d(conv7, 2, 2)
    
    #filters, then kernel size
    conv8 = tf.layers.conv2d(pool7, 512, 3, activation=tf.nn.relu, padding = "same")
    conv9 = tf.layers.conv2d(conv8, 512, 3, activation=tf.nn.relu, padding = "same")
    conv10 = tf.layers.conv2d(conv9, 512, 3, activation=tf.nn.relu, padding = "same")
    #pool size, then stride
    pool10 = tf.layers.max_pooling2d(conv10, 2, 2)
    
    
    #filters, then kernel size
    conv11 = tf.layers.conv2d(pool10, 512, 3, activation=tf.nn.relu, padding = "same")
    conv12 = tf.layers.conv2d(conv11, 512, 3, activation=tf.nn.relu, padding = "same")
    conv13 = tf.layers.conv2d(conv12, 512, 3, activation=tf.nn.relu, padding = "same")
    #pool size, then stride
    pool13 = tf.layers.max_pooling2d(conv13, 2, 2)
    
    
    
    #flatten to connect to fully connected
    full_in = flatten(pool13)
    
    #fully connected layer
    full1 = tf.contrib.layers.fully_connected(inputs=full_in, num_outputs=4096, activation_fn=tf.nn.relu)
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=4096, activation_fn=tf.nn.relu)
    logits = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=num_classes, activation_fn=None)
    return logits


def ResNet50(x):    
    #first conv/pool pair
    #filters, then kernel size
    #first layer

    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    conv1 = tf.layers.conv2d(x, 64, 7, activation=tf.nn.relu, padding = "same", kernel_regularizer=regularizer)
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2) #did 2 not 3
    
    #repeat x3
    #second layer with shortcuts
    shortcut2 = tf.layers.conv2d(pool1, 256, 1, activation=tf.nn.relu, padding = "same", kernel_regularizer=regularizer)
    conv2 = pool1
    for i in range(3):
        #filters, then kernel size
        conv2 = tf.layers.conv2d(conv2, 64, 1, activation=tf.nn.relu, padding = "same", kernel_regularizer=regularizer)
        conv2 = tf.layers.conv2d(conv2, 64, 3, activation=tf.nn.relu, padding = "same", kernel_regularizer=regularizer)
        conv2 = tf.layers.conv2d(conv2, 256, 1, activation=None, padding = "same", kernel_regularizer=regularizer)
        
        conv2 = shortcut2 + conv2
        conv2 = tf.nn.relu(conv2)
        shortcut2 = conv2
        
        
    
    #lose weight at the start of layers, first shortcut needs to be reshaped
    #filter, kernel, then stride
    shortcut3 = tf.layers.conv2d(conv2, 512, 1, 2, activation=tf.nn.relu, kernel_regularizer=regularizer) #changed filters
    conv3 = tf.layers.conv2d(conv2, 128, 1, 2, activation=tf.nn.relu, kernel_regularizer=regularizer) #need stride 2
    conv3 = tf.layers.conv2d(conv3, 128, 3, activation=tf.nn.relu, padding = "same", kernel_regularizer=regularizer)
    conv3 = tf.layers.conv2d(conv3, 512, 1, activation=None, padding = "same", kernel_regularizer=regularizer)
    conv3 = conv3 + shortcut3
    conv3 = tf.nn.relu(conv3)
    
    shortcut3 = conv3
    
    #third layer with shortcuts
    for i in range(3):
        #filters, then kernel size
        conv3 = tf.layers.conv2d(conv3, 128, 1, activation=tf.nn.relu, padding = "same", kernel_regularizer=regularizer)
        conv3 = tf.layers.conv2d(conv3, 128, 3, activation=tf.nn.relu, padding = "same", kernel_regularizer=regularizer)
        conv3 = tf.layers.conv2d(conv3, 512, 1, activation=None, padding = "same", kernel_regularizer=regularizer)
        
        conv3 = shortcut3 + conv3
        conv3 = tf.nn.relu(conv3)
        shortcut3 = conv3
    
    
    #lose weight at the start of layers, first shortcut needs to be reshaped
    #filter, kernel, then stride
    shortcut4 = tf.layers.conv2d(conv3, 1024, 1, 2, activation=tf.nn.relu, kernel_regularizer=regularizer)
    conv4 = tf.layers.conv2d(conv3, 256, 1, 2, activation=tf.nn.relu, kernel_regularizer=regularizer) #need stride 2
    conv4 = tf.layers.conv2d(conv4, 256, 3, activation=tf.nn.relu, padding = "same", kernel_regularizer=regularizer)
    conv4 = tf.layers.conv2d(conv4, 1024, 1, activation=None, padding = "same", kernel_regularizer=regularizer)
    conv4 = conv4 + shortcut4
    conv4 = tf.nn.relu(conv4)
    
    shortcut4 = conv4
    
    #fourth layer with shortcuts
    for i in range(5):
        #filters, then kernel size
        conv4 = tf.layers.conv2d(conv4, 256, 1, activation=tf.nn.relu, padding = "same", kernel_regularizer=regularizer)
        conv4 = tf.layers.conv2d(conv4, 256, 3, activation=tf.nn.relu, padding = "same", kernel_regularizer=regularizer)
        conv4 = tf.layers.conv2d(conv4, 1024, 1, activation=None, padding = "same", kernel_regularizer=regularizer)
        
        conv4 = shortcut4 + conv4
        conv4 = tf.nn.relu(conv4)
        shortcut4 = conv4
    
    #lose weight at the start of layers, first shortcut needs to be reshaped
    #filter, kernel, then stride
    shortcut5 = tf.layers.conv2d(conv4, 2048, 1, 2, activation=tf.nn.relu, kernel_regularizer=regularizer)
    conv5 = tf.layers.conv2d(conv4, 512, 1, 2, activation=tf.nn.relu, kernel_regularizer=regularizer) #need stride 2
    conv5 = tf.layers.conv2d(conv5, 512, 3, activation=tf.nn.relu, padding = "same", kernel_regularizer=regularizer)
    conv5 = tf.layers.conv2d(conv5, 2048, 1, activation=None, padding = "same", kernel_regularizer=regularizer)
    conv5 = conv5 + shortcut5
    conv5 = tf.nn.relu(conv5)
    
    shortcut5 = conv5
    
    #fifth layer with shortcuts
    for i in range(2):
        #filters, then kernel size
        conv5 = tf.layers.conv2d(conv5, 512, 1, activation=tf.nn.relu, padding = "same", kernel_regularizer=regularizer)
        conv5 = tf.layers.conv2d(conv5, 512, 3, activation=tf.nn.relu, padding = "same", kernel_regularizer=regularizer)
        conv5 = tf.layers.conv2d(conv5, 2048, 1, activation=None, padding = "same", kernel_regularizer=regularizer)
        
        conv5 = shortcut5 + conv5
        conv5 = tf.nn.relu(conv5)
        shortcut5 = conv5
    
    #do an average pool, 7 by 7
    pool5 = tf.layers.average_pooling2d(conv5, 7, 7)
    
    
    #flatten to connect to fully connected
    full_in = flatten(pool5)
    
    #fully connected layer
    full1 = tf.contrib.layers.fully_connected(inputs=full_in, num_outputs=1000, activation_fn=tf.nn.relu)
    logits = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=num_classes, activation_fn=None)
    return logits