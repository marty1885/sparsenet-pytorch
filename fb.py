import foolbox
from foolbox.criteria import Misclassification
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
import sys

import numpy as np
import eagerpy as ep

import sparsenet as S

def foolbox_attack(fbmodel, device, attack, test_loader, use_cuda):
    success_attacks = 0
    curr_samples = 0
    robustness = 0
    print("\nAttack: {}".format(str(attack)))
    for i, (data, real_label) in enumerate(test_loader):
        batch_size = data.shape[0]
        if use_cuda:
            data = data.to(device)
            real_label = real_label.to(device)
        _, _, success = attack(fbmodel, data, real_label, epsilons=[0.3])
        success = success[0].cpu().numpy()
        success_attacks += success.sum()
        curr_samples += batch_size
        robustness = 1 - success_attacks/curr_samples
        print('\r\033[K', end='')
        print('Progress: {}/10000 - adversial robustness {:.4f}%, success attacks {:.4f}%'
                .format(curr_samples, robustness*100, (1-robustness)*100)
                , end='')
        sys.stdout.flush()
    print()
    return robustness

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FoolBox attacker")
    parser.add_argument('--no-cuda', action='store_true', default=False
            , help='disable CUDA')
    parser.add_argument('--model-type', default='kwinner'
            , help='The model type to attack')
    args = parser.parse_args()
    use_cuda=(not args.no_cuda) and torch.cuda.is_available()

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
        pretrained_model = "sparse_cnn.pt"
    else:
        print("Model type must be drouput, lenet or kwinner.")
        exit(0)
    model.load_state_dict(torch.load(pretrained_model, map_location=device))
    model.train(False)
    model.eval()
    model = model.to(device)

    fbmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1))

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=128, shuffle=True)
    if not use_cuda:
        model = model.to('cpu')
    #attack = LinfDeepFoolAttack() # Gradient Based
    #attack = LinfPGD()
    #attack = DDNAttack()
    #attack = InversionAttack()
    attack = SaltAndPepperNoiseAttack()
    robustness = foolbox_attack(fbmodel, device, attack, test_loader, use_cuda)
    print("Attack success rate: {:.4f}".format((1-robustness)*100))
