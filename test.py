from __future__ import print_function
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.models as models
from img_loader_test import img_loader
import numpy as np
import math
import sys
import scipy.misc
import cv2
import matplotlib.pyplot as plt

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            output = model(data[0])
    return output

def main():
    model = models.resnet18()
    for params in model.parameters():
        params.requires_grad = False
    #only the final classification layer is learned
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    model.double()

    model.load_state_dict(torch.load('Resnetmodel.pt'))
   
    img_path = 'data/test/'
    img_name = sys.argv[1]
    data_test = [img_path, img_name] 
    test_loader = torch.utils.data.DataLoader(img_loader(data_test))
    prediction = test(model, test_loader)

    print(prediction)
    prediction=prediction[0]
    img = cv2.imread(img_name, 0)
    #plt.imshow(img)
    #plt.scatter([prediction[0]*326], [predicition[1]*490])
    #plt.show()


if __name__ == '__main__':
    main()
