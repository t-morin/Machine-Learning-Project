# -*- coding: utf-8 -*-

import os 
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

path = r'C:\Users\thoma\OneDrive\Bureau\MachineLearningProject\Images_subset_manual'
i = 1

#appending the pics to the training data list

training_data = []
for doss in os.listdir(path):
    for img in os.listdir(os.path.join(path,doss)):
        pic = cv2.imread(os.path.join(path,doss,img))
        pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
        pic = cv2.resize(pic,(256,256))
        training_data.append([pic])
        i = i +1
        if i > 600:
            i = 1
            break


np.save(os.path.join(r'C:\Users\thoma\OneDrive\Bureau\MachineLearningProject','features'),np.array(training_data))

#loading the saved file once again
saved = np.load(os.path.join(r'C:\Users\thoma\OneDrive\Bureau\MachineLearningProject','features.npy'))

plt.imshow(saved[1000].reshape(256,256,3))
plt.imshow(np.array(training_data[1000]).reshape(256,256,3))


