import pickle
import numpy as np
import timeit
import scipy.io as spio
import os
import cv2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


def fetch_data(directory, label): 
    item_list = []
    step = 0
    for file in tqdm(os.listdir(directory)):
        full_img_str = directory + "/" + file
        #print(full_img_str)

        mat = spio.loadmat(full_img_str, squeeze_me=True)
        data = np.abs(mat["data_store"])
        smaller_data = data[70:170]
        #print(data.shape)
        #print(data)
        ###append the img and label to the list###
        sub_list = [smaller_data, label]
        #print(sub_list)
        item_list.append(sub_list)
        
    return item_list

def mini_batch(features,labels,mini_batch_size):
    """
    Args:
        features: features for one batch
        labels: labels for one batch
        mini_batch_size: the mini-batch size you want to use.
    Hint: Use "yield" to generate mini-batch features and labels
    """
    #split the data into batches
    amount_of_data = len(features)
    number_of_bunches = amount_of_data/mini_batch_size
    
    bunches_features = []
    bunches_labels = []
    
    #loop over breaking the data into batches
    for i in range(int(number_of_bunches)):
        current_range = i * mini_batch_size
        f_b = features[current_range:current_range+mini_batch_size]
        l_b = labels[current_range:current_range+mini_batch_size]
        
        bunches_features.append(f_b)
        bunches_labels.append(l_b)
    
    #return the mini-batched data
    return bunches_features, bunches_labels