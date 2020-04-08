from ifgsm import ifgsm_attack
import pandas as pd
import torch
from torchvision import datasets, transforms

import argparse

import sparsenet as S

from fb import foolbox_attack
import foolbox
import foolbox.attacks as fa

if __name__ == '__main__':
    ifgsm_epsilons = [0, 0.003, .1, .15, .2, .25, .3]
    
    parser = argparse.ArgumentParser(description="Attach MNIST models")
    parser.add_argument('--no-cuda', action='store_true', default=False
            , help='disable CUDA')
    parser.add_argument('--model-type', default='kwinner'
            , help='The model type to attack')
    args = parser.parse_args()
    
    use_cuda=(not args.no_cuda) and torch.cuda.is_available()
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
    elif model_type == "brelu":
        model = S.BreluLenet().to(device)
        pretrained_model = "lenet_brelu.pt"
    else:
        print("Model type must be drouput, lenet or kwinner.")
        exit(0)
    print("Model type: ", model_type)
    
    model.load_state_dict(torch.load(pretrained_model, map_location=device))
    model.train(False)
    model.eval()
    model = model.to(device)
    if not use_cuda:
        model = model.to('cpu')

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=128, shuffle=True)
        
    attack_result = {}

    # perform IFGSM (Our implementation is fster than FoolBox's)
    ifgsm_iterations = 4
    print("IFSGM @ {} iterations".format(ifgsm_iterations))
    for epsilon in ifgsm_epsilons:
        accuracy, _ = ifgsm_attack(model, device, test_loader, epsilon, ifgsm_iterations)
        attack_result["IFGSM_{}".format(epsilon)] = accuracy
        
    # Use FoolBox for attack
    fbmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1))
    fb_attacks = {"Linf PGD":fa.LinfPGD(), "DDN":fa.DDNAttack(), "Linf DeepFool":fa.LinfDeepFoolAttack()
                     , "Basic Iterative":fa.LinfBasicIterativeAttack(), "Salt and Pepper":fa.SaltAndPepperNoiseAttack()}
    for name, attack in fb_attacks.items():
        robustness = foolbox_attack(fbmodel, device, attack, test_loader, use_cuda)
        attack_result[name] = robustness
    
    result_df = pd.DataFrame(attack_result.items(), columns=["attack", "accuracy"])
    result_df.to_csv("{}.csv".format(model_type), index=False)
        
