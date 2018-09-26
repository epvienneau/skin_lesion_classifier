from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from img_loader import img_loader
import torch.utils.model_zoo as model_zoo
import math
import numpy as np
import matplotlib.pyplot as plt
import csv

training_loss = []
validation_loss = []

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target= data.to(device), target.to(device)
        optimizer.zero_grad() 
        output = model(data)
        criterion = nn.MSELoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        training_loss.append(loss.item())
  
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            criterion = nn.MSELoss()
            test_loss += criterion(output, target).item() 
            #pred = output.max(1, keepdim=True)[1]
            correct += output.eq(target.view_as(output)).sum().item()

    test_loss /= len(test_loader.dataset)
    validation_loss.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    parser = argparse.ArgumentParser(description='PyTorch Object Detection Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # parse txt file to create dictionary for dataloader
    probs_train = []
    probs_test = []
    img_file_train = []
    img_file_test = []
    img_path_train = ['data/train/']*9015
    img_path_test = ['data/test/']*1000
    with open('data/labels/Train_labels.csv', 'r') as f:
        next(f)
        for count, line in enumerate(f):
            file_info = line.split()[0] #get single line
            file_info = file_info.split(',', 1) #separate file name from probs
            img_file_train.append(file_info[0]) #pull out img file str
            probs = file_info[1] #probs, as a single str with commas in it
            probs = probs.split(',') #probs, as a list of strings
            probs = list(map(int, probs)) #probs as a list of ints
            probs_train.append(probs)
    with open('data/labels/Test_labels.csv', 'r') as f:
        next(f)
        for count, line in enumerate(f):
            file_info = line.split()[0] #get single line
            file_info = file_info.split(',', 1) #separate file name from probs
            img_file_test.append(file_info[0]) #pull out img file str
            probs = file_info[1] #probs, as a single str with commas in it
            probs = probs.split(',') #probs, as a list of strings
            probs = list(map(int, probs)) #probs as a list of ints
            probs_test.append(probs)
    data_train = [img_path_train, img_file_train, probs_train]
    data_test = [img_path_test, img_file_test, probs_test] 
    train_loader = torch.utils.data.DataLoader(img_loader(data_train), batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(img_loader(data_test), batch_size=args.batch_size, shuffle=True, **kwargs) 

    model = models.resnet18(pretrained=True, **kwargs).to(device)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    model.double()
    #need to use adam optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    #epoch_axis = range(args.epochs)
    #plt.plot(epoch_axis, training_loss, 'r', epoch_axis, validation_loss, 'b')
    #plt.show()
    #torch.save(model.state_dict(), './Resnetmodel.pt')
    #with open('loss.csv', 'w', newline='') as csvfile:
    #    losswriter = csv.writer(csvfile, dialect='excel', delimiter=' ', 
    #            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #losswriter.writerow(str(args.epochs))
        #losswriter.writerow(str(args.batch-size))
    #    losswriter.writerow('training')
    #    print(len(training_loss))
    #    print(len(validation_loss))
    #    for item in training_loss:
    #        losswriter.writerow(str(round(item, 4)))
    #    losswriter.writerow('validation')
    #    for item in validation_loss:
    #        losswriter.writerow(str(round(item, 4)))


if __name__ == '__main__':
    main()

