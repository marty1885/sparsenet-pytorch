from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

import sparsenet as S

print("IFGSM: Iterative Fast Gradient Signed Method")
print("Attacking LeNet with IFGSM.")

epsilons = [0, 0.003, .1, .15, .2, .25, .3]

parser = argparse.ArgumentParser(description="Attach MNIST models")
parser.add_argument('--no-cuda', action='store_true', default=False
        , help='disable CUDA')
parser.add_argument('--model-type', default='kwinner'
        , help='The model type to attack')
parser.add_argument('--num-iteration', type=int, default=100, metavar='N'
        , help='The amount of iteratoins of attack')
args = parser.parse_args()

# Define what device we are using
use_cuda = torch.cuda.is_available() and not args.no_cuda
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

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
num_iter = args.num_iteration
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
model.train(False)
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

def test( model, device, test_loader, epsilon, num_iter ):

    correct = 0
    adv_examples = []
    data_len = 0

    for data, target in test_loader:

        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        data_len += data.shape[0]

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1].cpu().numpy().flatten() # get the index of the max log-probability
        target_pred = target.cpu().numpy().flatten()
        init_correct = init_pred == target_pred

        for i in range(num_iter):

            output = model(data)
            loss = F.nll_loss(output, target)
            model.zero_grad()
            loss.backward()

            # Collect datagrad
            data_grad = data.grad.data

            # Call FGSM Attack
            perturbed_data = fgsm_attack(data, epsilon/num_iter, data_grad)
            data = Variable(perturbed_data.data, requires_grad=True)

        # Re-classify the perturbed image
        output = model(perturbed_data)
        final_pred = np.argmax(output.detach().cpu().numpy(), axis=1).flatten() # get the index of the max log-probability

        adv = perturbed_data.squeeze().detach().cpu().numpy()
        for i, ex in enumerate(adv):
            adv_ex = ex.squeeze()
            if init_correct[i] == 0:
                continue
            # Check for success
            # If the initial prediction is wrong, dont bother attacking, just move on
            if int(final_pred[i]) == int(target_pred[i]):
                correct += 1
                # Special case for saving 0 epsilon examples
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_examples.append( (init_pred[i], final_pred[i], adv_ex) )
            else:
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
    acc, ex = test(model, device, test_loader, eps, num_iter)
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
