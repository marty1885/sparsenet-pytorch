from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

import sparsenet as S

print("CUDA Available: ",torch.cuda.is_available())

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    curr_idx = 0
    loss_func = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        curr_idx += len(data)
        if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader)-1:
            print('\033[K', end='') #]
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, curr_idx, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()/len(data)), end='\r')
    print('')

def test(model, device, test_loader, noise=0):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if noise != 0:
                data = (torch.randn(data.shape).to(device)*noise + data)
                data = torch.clamp(data, 0, 1)
            output = model(data)
            loss_func = nn.CrossEntropyLoss()
            test_loss += loss_func(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), noise={}\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), noise))
    return correct / len(test_loader.dataset)

def train_model(model, loader, save_path, args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using CUDA: {}".format(use_cuda))

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        loader('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        loader('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    print("traning for {} epochs.".format(args.epochs))
    model = model.to(device)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
    #optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    best_accuracy = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        accuracy = test(model.eval(), device, test_loader)
        if accuracy > best_accuracy and args.save_model:
            accuracy = best_accuracy
            torch.save(model.state_dict(),save_path)
        
    # Test the model against white noise
    noise = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for n in noise:
        test(model, device, test_loader, n)
        
    return model

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Sparse CNN MNIST trainer')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                        help='learning rate (default: 0.02)')
    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model-type', default='kwinner'
        , help='The model type to attack')
    args = parser.parse_args()
    model = S.Net()
    save_path = "sparse_cnn.pt"
    if args.model_type == "kwinner":
        model = S.Net()
        save_path = "sparse_cnn.pt"
    elif args.model_type == "lenet":
        model = S.Lenet()
        save_path = "lenet.pt"
    elif args.model_type == "dropout":
        model = S.Lenet()
        save_path = "lenet_drop.pt"
    elif args.model_type == "brelu":
        model = S.BreluLenet()
        save_path = "lenet_brelu.pt"
    else:
        print("Model type must be drouput, lenet or kwinner.")
        exit(0)
    train_model(model, datasets.MNIST, save_path, args)

if __name__ == '__main__':
    main()
