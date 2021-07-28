# -*- coding: utf-8 -*-


# import the opencv library
#pip install opencv-python
import cv2
import time
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms, models
from sklearn import preprocessing
import joblib
#import cv2.cv as cv

model = models.resnet18(pretrained=True)
layer = model._modules.get('avgpool')
model.eval()
rbf = joblib.load('ClassifierTrained.pkl')
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


# define a video capture object
vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
frame_rate = 0.3
prev = 0

while (True):

    # Capture the video frame
    # by frame
    time_elapsed = time.time() - prev
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if time_elapsed > 1./frame_rate:
        prev = time.time()
        frame = cv2.resize(frame,(256,256))
    ############################## PROCESSING   ###############################
        Img_feat = get_vector(frame).numpy()
        #Scale les features
        min_max_scaler = preprocessing.MinMaxScaler()
        Img_feat = min_max_scaler.fit_transform(Img_feat.reshape(-1,1))
        Img_feat = Img_feat.reshape(1,-1)
        output = rbf.predict(Img_feat)
        print('Right Class 8')
        print(output)
    ###################################################################################################
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()




