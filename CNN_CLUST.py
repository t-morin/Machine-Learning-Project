# -*- coding: utf-8 -*-


import os
import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms, models

model = models.resnet18(pretrained=True)
layer = model._modules.get('avgpool')
model.eval()

img_bank = np.load(os.path.join(r'C:\Users\thoma\OneDrive\Bureau\MachineLearningProject','features.npy'))

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

def get_vector(img_num):
    x = img_bank[img_num].reshape(256,256,3)    #Prendre une image du data set
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

feature_img = []
for i in range(6000):
    y = get_vector(i)
    feature_img.append([y])

#pour enregistrer la liste de features 
Lizte = []
for i in range(6000):
    a = torch.stack(feature_img[i])
    b = a.numpy()
    Lizte.append(b)


np.save(os.path.join(r'C:\Users\thoma\OneDrive\Bureau\MachineLearningProject','Features_Img'),np.array(Lizte))
