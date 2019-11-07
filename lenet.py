from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import struct
import numpy as np

# LeNet-5
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Applies a 2D convolution over an input signal composed of several input planes
        # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(1, 6, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        # Applies a linear transformation to the incoming data: y = xA^T + b
        # (in_features, out_features, bias=True)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc  = nn.Linear(84, 10)
    def forward(self, x):
        # 1st layer: Convolution 1
        # Applies the rectified linear unit function element-wise, return type: tensor
        # (input, inplace=False)
        x = F.relu(self.conv1(x))
        # 2nd layer: Subsampling 1
        # (kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        x = F.max_pool2d(x, 2, 2)
        # 3rd layer: Convolution 2
        x = F.relu(self.conv2(x))
        # 4th layer: Subsampling 2
        x = F.max_pool2d(x, 2, 2)
        # Returns a new tensor with the same data as the `self` tensor but of a different `shape`
        # the size -1 is inferred from other dimensions
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        # 5th layer: Full connection 1
        x = F.relu(self.fc1(x))
        # 6th layer: Full connection 2
        x = F.relu(self.fc2(x))
        # 7th layer: Gaussian connection
        x = self.fc(x)
        # Applies a softmax followed by a logarithm
        # (input, dim=None, _stacklevel=3, dtype=None)
        return F.log_softmax(x, dim=1)

# Hyperperamaters for training
class Hyper():
    def __init__(self):
        self.log_interval = 10
        self.lr = 0.001
        self.momentum = 0.5
        self.batch_size = 32
        self.test_batch_size = 32
        self.epoch = 50
        self.save_model = 1

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(data.shape)
        # print(target.shape)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # loss_list.append(loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    # print(loss_list)

def test(args, model, device, test_loader, train_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    if train_set:
        print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), accuracy))
    else: 
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss, accuracy

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def main():
    train_imgs = read_idx('MNIST/train-images-idx3-ubyte')
    train_labels = read_idx('MNIST/train-labels-idx1-ubyte')
    test_imgs = read_idx('MNIST/t10k-images-idx3-ubyte')
    test_labels = read_idx('MNIST/t10k-labels-idx1-ubyte')

    args = Hyper()
    device = torch.device('cpu')
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True)
    # print(train_loader)
    # train_loaderr = [train_imgs, train_labels]
    # test_loaderr = [test_imgs, test_labels]
    # train_loader = torch.utils.data.DataLoader(train_loaderr, batch_size=args.batch_size)
    # test_loader = torch.utils.data.DataLoader(test_loaderr, batch_size=args.test_batch_size)
    # print('OK')

    # train_imgs = np.reshape(train_imgs, (60000, 1, 28, 28))
    # train_labels = np.reshape(train_labels, (60000, 1, 1, 1))
    # tensor_x = torch.stack([torch.Tensor(i) for i in train_imgs]) # transform to torch tensors
    # tensor_y = torch.stack([torch.Tensor(i) for i in train_labels])
    # # create your datset
    # my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y) 
    # # create your dataloader
    # train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.test_batch_size, shuffle=True) 
    
    # Construct the model
    model = Net().to(device)
    # Implements stochastic gradient descent (optionally with momentum)
    # (params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Train
    training_loss = []
    training_accuracy = []
    test_loss = []
    test_accuracy = []
    for epoch in range(1, args.epoch + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        loss, accuracy = test(args, model, device, test_loader, train_set=True)
        training_loss.append(loss)
        training_accuracy.append(accuracy)
        loss, accuracy = test(args, model, device, test_loader)
        test_loss.append(loss)
        test_accuracy.append(accuracy)
    print(training_loss)
    print(training_accuracy)
    print(test_loss)
    print(test_accuracy)

    if args.save_model:
        torch.save(model.state_dict(), 'mnist_cnn.pt')

if __name__ == '__main__':
    main()
