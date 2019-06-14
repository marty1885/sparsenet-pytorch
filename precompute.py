from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np

import sparsenet as S

print("IFGSM: Iterative Fast Gradient Signed Method")
print("Attacking SparseNet with precomuted IFGSM on LeNet.")

epsilons = [0, 0.003, .1, .15, .2, .25, .3, .4, .5, .6, .7, .8]
lenet_pretrained_model = "lenet.pt"
use_cuda=True

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
lenet = S.Lenet().to(device)
print()

lenet.load_state_dict(torch.load(lenet_pretrained_model, map_location='cpu'))
lenet.train(False)
lenet.eval()

pretrained_model = "spaese_cnn.pt"
model = S.Net().to(device)
#pretrained_model = "lenet_drop.pt"
#model = S.LenetDropout().to(device)
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
model.train(False)
model.eval()

parser = argparse.ArgumentParser(description='Precomputed adversial attack')
parser.add_argument('--iter', type=int, default=4, metavar='ITER',
                        help='number of IFGSM iterations (default: 4)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
args = parser.parse_args()

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def ifgsm_attack(model, image, target, epsolon, num_iter):
    perturbed_data = None
    data = image
    for i in range(num_iter):
        output = model(data)
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsolon/num_iter, data_grad)
        data = Variable(perturbed_data.data, requires_grad=True)
    return perturbed_data


def generate_adversial_example(model, device, test_loader, epsilon, num_iter):
    adv_image = []
    adv_pred = []
    real_pred = []
    for data, target in test_loader:

        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1].cpu().numpy().flatten() # get the index of the max log-probability
        target_pred = target.cpu().numpy().flatten()
        init_correct = init_pred == target_pred

        perturbed_data = ifgsm_attack(model, data, target, epsilon, num_iter)

        # Re-classify the perturbed image
        output = model(perturbed_data)
        final_pred = np.argmax(output.detach().cpu().numpy(), axis=1).flatten() # get the index of the max log-probabilityi

        adv_image.append(perturbed_data.detach().cpu().numpy())
        adv_pred.append(final_pred)
        real_pred.append(target.cpu().numpy())

    return np.concatenate(adv_image), np.concatenate(adv_pred), np.concatenate(real_pred)


accuracies = []
examples = []

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=args.test_batch_size, shuffle=True)

print("Test results:")
for eps in epsilons:
    correct = 0
    adv_image, adv_pred, real_pred = generate_adversial_example(lenet, device, test_loader, eps, args.iter)
    for i in range(len(adv_image)//args.test_batch_size):
         image = adv_image[i*args.test_batch_size:(i+1)*args.test_batch_size]
         real_y = real_pred[i*args.test_batch_size:(i+1)*args.test_batch_size]

         pred = model.forward(torch.FloatTensor(image).cuda())
         pred_class = np.argmax(pred.detach().cpu(), axis=1).flatten()

         for j in range(len(real_y)):
             real = real_y[j]
             p = pred_class[j]
             if real == p:
                 correct += 1
    print("epsolon = {}\t: victim accuracy = {}, defender accuracy =  {}".format(eps, (real_pred==adv_pred).mean(), correct/len(real_pred)))

