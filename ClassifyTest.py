# -*- coding: utf-8 -*-


import cv2
import os

#Convert image into numpy array
pic = cv2.imread(r'C:\Users\thoma\OneDrive\Bureau\MachineLearningProject\7.jpg')
pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
pic = cv2.resize(pic,(256,256))

import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms, models
from sklearn import preprocessing

#Extract the Features of the image
model = models.resnet18(pretrained=True)
layer = model._modules.get('avgpool')
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

def get_vector(pic):
    x = pic.reshape(256,256,3)    #Prendre une image du data set
    x = Image.fromarray(x)       #Convertir en image PIL
    input_tensor = Variable(normalize(transform(x)).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(o.data.size(1)))
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
     # 6. Run the model on our transformed image
    model(input_tensor)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding

Img_feat = get_vector(pic).numpy()

#Scale les features
min_max_scaler = preprocessing.MinMaxScaler()
Img_feat = min_max_scaler.fit_transform(Img_feat.reshape(-1,1))

#import sklearn.external.joblib as extjoblib
import joblib
#from sklearn.externals import joblib

Img_feat = Img_feat.reshape(1,-1)
rbf = joblib.load('ClassifierTrained.pkl')
output = rbf.predict(Img_feat)
print('Right Class 8')
print(output)
