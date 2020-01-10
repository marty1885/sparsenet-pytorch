import foolbox
from foolbox.criteria import TargetClassProbability, Misclassification
from foolbox.attacks import *

#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

import sparsenet as S

parser = argparse.ArgumentParser(description="FoolBox attacker")
parser.add_argument('--no-cuda', action='store_true', default=False
        , help='disable CUDA')
parser.add_argument('--model-type', default='kwinner'
        , help='The model type to attack')
args = parser.parse_args()
use_cuda=not args.no_cuda and torch.cuda.is_available()

# Define what device we are using
print("CUDA Available: ",use_cuda)
device = torch.device("cuda:0" if (use_cuda) else "cpu")
model_type = args.model_type
if model_type == "dropout":
    model = S.LenetDropout().to(device)
    pretrained_model = "lenet_drop.pt"
elif model_type == "lenet":
    model = S.Lenet().to(device)
    pretrained_model = "lenet.pt"
elif model_type == "kwinner":
    model = S.Net().to(device)
    pretrained_model = "spaese_cnn.pt"
else:
    print("Model type must be drouput, lenet or kwinner.")
    exit(0)
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
model.train(False)
model.eval()
model = model.to(device)

fbmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=128, shuffle=True)

dists = []
success_attacks = 0
good_predictions = 0
for i, (data, real_label) in enumerate(test_loader):
    batch_size = data.shape[0]
    label = model.forward(data.to(device)).detach().argmax(dim=-1).cpu()
    correct_preds = (label == real_label)
    num_good = (correct_preds).sum().item()
    good_predictions += num_good

    data = data[correct_preds]
    label = label[correct_preds]

    if data.shape[0] == 0:
        continue

    criterion = Misclassification()
    attack = DeepFoolAttack(fbmodel, criterion)
    #attack = (fbmodel, criterion)
    #attack = EADAttack(fbmodel, criterion, distance=foolbox.distances.Linfinity)
    image = data.numpy()
    adversarial = attack(image, labels=label.numpy())
    adversarial = torch.Tensor(adversarial).to(device)
    adv_pred = model.forward(adversarial).argmax(-1).cpu()
    num_adv = (adv_pred != label).float() * torch.isnan(adversarial.view(num_good, -1)).any(-1).cpu().float() # avoid nans
    num_adv = num_adv.sum().item()
    success_attacks += num_adv
    curr_samples = (i+1)*batch_size
    print('Progress: {}/10000 - clean accuracy {:.4f}%, success attacks {:.4f}%'
            .format(curr_samples, good_predictions/curr_samples*100, success_attacks/good_predictions*100)
            , end='\r')


#print("average MSE: ", np.mean(np.array(dists)))
print()
print("Attack success rate: {:.4f}".format(success_attacks/good_predictions))
print("Clssifcation rate: {:.4f} (if not attacked)".format(good_predictions/10000))
