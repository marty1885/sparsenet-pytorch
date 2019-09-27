from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

import sparsenet as S

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "spaese_cnn.pt"
use_cuda=True

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
model = S.Net().to(device)

model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
model.eval()

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

def test( model, device, test_loader, epsilon ):

    correct = 0
    adv_examples = []
    data_len = 0

    for data, target in test_loader:

        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        data_len += data.shape[0]

        output = model(data)
        # get the index of the max log-probability
        init_pred = output.max(1, keepdim=True)[1].cpu().numpy().flatten()

        target_pred = target.cpu().numpy().flatten()
        init_correct = init_pred == target_pred

        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)
        pred_tensor = output.detach().cpu().numpy()
        final_pred = np.argmax(pred_tensor, axis=1).flatten() # get the index of the max log-probability

        adv = perturbed_data.squeeze().detach().cpu().numpy()
        for i, ex in enumerate(adv):
            if init_correct[i] == 0:
                continue
            
            adv_ex = ex.squeeze()

            # Check for success
            if int(final_pred[i]) == int(target_pred[i]):
                correct += 1
                # Special case for saving 0 epsilon examples
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_examples.append( (init_pred[i], final_pred[i], adv_ex) )
            else:
                #if pred_tensor[i, final_pred[i]] < 0.5:
                #    correct += 1
                #    continue
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_examples.append( (init_pred[i], final_pred[i], adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/data_len
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, data_len, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


accuracies = []
examples = []

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])), 
        batch_size=1000, shuffle=True)

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)


cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
