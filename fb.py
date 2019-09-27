import foolbox
from foolbox.criteria import TargetClassProbability, Misclassification
from foolbox.attacks import LBFGSAttack, GradientAttack, DeepFoolAttack, FGSM, MomentumIterativeAttack

#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import random

import numpy as np
import matplotlib.pyplot as plt

import sparsenet as S
pretrained_model = "spaese_cnn.pt"
use_cuda=True

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
model = S.Net().to(device)

model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
model.eval()
#model = model.cpu()

fbmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)

dists = []
success_attacks = 0
for i, (data, real_label) in enumerate(test_loader):
    image = np.asarray(data.numpy())
    label = np.argmax(model.forward(data.cuda()).cpu().detach().numpy())
    if label != np.argmax(real_label.detach().numpy()):
        continue

    adv_target = random.randint(0, 9)
    while(adv_target == label):
        adv_target = random.randint(0, 9)
    criterion = Misclassification()
    #attack = DeepFoolAttack(fbmodel, criterion)
    attack = MomentumIterativeAttack(fbmodel, criterion, distance=foolbox.distances.Linfinity)
    print(i, end='\r')
    image = image.reshape(1,28,28)
    #print("running attack")
    adversarial = attack(image, label=label)
    success_attacks += 1 if adversarial is not None else 0
    if adversarial is None:
        continue
    #"""
    fig, axis = plt.subplots(1, 3) 

    axis[0].set_title("Orig: " + str(label))
    axis[0].imshow(image[0], cmap="gray")

    axis[1].set_title("Adv: " + str(adv_target))
    axis[1].imshow(adversarial[0], cmap="gray")

    axis[2].set_title("Distance: {:.4f}".format(foolbox.distances.MSE(image, adversarial, (0,1)).value))
    axis[2].imshow(adversarial[0] - image[0])

    
    plt.show()
    break
    #"""

    dist = foolbox.distances.Linf(image, adversarial, (0,1)).value
    dists.append(dist)

    
#print("average MSE: ", np.mean(np.array(dists)))
#`print("Attack success rate: {:.4f}".format(success_attacks/10000))
