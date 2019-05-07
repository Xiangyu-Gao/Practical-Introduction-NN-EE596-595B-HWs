from glob import glob
from scipy import misc
import numpy as np
import os, cv2, random
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker


class DogsAndCatsHelper:

    @staticmethod
    def get_data():
        
        TRAIN_DIR = 'C:/Users/Xiangyu Gao/Downloads/train/'
        TEST_DIR = 'C:/Users/Xiangyu Gao/Downloads/test1/'

        ROWS = 227
        COLS = 227
        CHANNELS = 3
        #print(os.listdir(TRAIN_DIR)[:5])
        train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
        train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
        train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

        test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]


        # slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset
        train_images = train_dogs[:] + train_cats[:]
        random.shuffle(train_images)
        test_images =  test_images[:]

        def read_image(file_path):
            img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
            return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


        def prep_data(images):
            count = len(images)
            data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

            for i, image_file in enumerate(images):
                image = read_image(image_file)
                data[i] = image.T
                #if i%250 == 0: print('Processed {} of {}'.format(i, count))
            
            return data

        train = prep_data(train_images)
        test = prep_data(test_images)
        train = np.swapaxes(train, 1,3)
        test = np.swapaxes(test, 1,3)

        print("Train shape: {}".format(train.shape))
        print("Test shape: {}".format(test.shape))
        labels = []
        
        for i in train_images:
            if 'dog' in i.split('/')[-1]:
                labels.append([1,0])
            else:
                labels.append([0,1])

        #sns.countplot(labels).set_title('Cats and Dogs')
        labels = np.array(labels)

        return train, labels, test
