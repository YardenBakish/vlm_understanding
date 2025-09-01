import argparse
import consts
import random
import numpy as np
import models
import defenses
import attacks
import utils
import time
import re
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


import glob
import os

def find_files_with_prefix(directory, prefix):
    pattern = os.path.join(directory, prefix + "*")
    return [f for f in glob.glob(pattern) if os.path.isfile(f)]


torch.manual_seed(consts.SEED)
random.seed(consts.SEED)
np.random.seed(consts.SEED)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, help='1 for adversarial training, 0 for evaluating')
    return parser.parse_args()

def run_standard_training(reg_layer=None, reg_strength=1e-4, dropout = False):
    # load training set
    transforms_tr = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.RandomResizedCrop((32,32))
    ])
    data_tr = utils.TMLDataset('train', transform=transforms_tr)

    # init model
    model = models.SimpleCNN(reg_layer=reg_layer, dropout=dropout)
    model.to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # execute training
    model = utils.standard_train(model, data_tr, criterion, optimizer,
                                 scheduler, device,
                                 reg_strength=(reg_strength if reg_layer else 0.0))

    # save model
    model.to('cpu')
    tag = "standard" if reg_layer is None else f"{reg_layer}_sparse"
    tag = tag if dropout == False else f"{tag}_dropout"
    tag = tag if reg_strength==1e-4 else f"{tag}_{reg_strength}"
    torch.save(model.state_dict(), f"trained-models/simple-cnn-{tag}")
    print(f"done with {tag}")
    return tag


def adjust_paths(mpaths,dropout=False, comp_single=False):
    new_d = mpaths
    output_file = "eval_results.txt"
    if dropout:
        new_d = {}
        for k in mpaths:
            new_d[k] = f"{mpaths[k]}_dropout"
            output_file = "eval_results_dropout.txt"
    if comp_single:
        new_d = {}
        variant = comp_single
        
        files = find_files_with_prefix("trained-models",f"simple-cnn-{variant}")
        files = [f for f in files if "dropout" not in f]

        for f in files:
            suff = f.split("trained-models/simple-cnn-")[-1]
            
            new_d[suff] = f"trained-models/simple-cnn-{suff}"
        
        output_file = f"eval_results_{variant}.txt"
        
    return output_file, new_d 

def run_evaluation(dropout=False, comp_single=False):
    # Load all trained models (skip any that aren't present)
    trained_models = {}
    mpaths = {
        'standard':      'trained-models/simple-cnn-standard',
        'conv1_sparse':  'trained-models/simple-cnn-conv1_sparse',
        'conv2_sparse':  'trained-models/simple-cnn-conv2_sparse',
        'conv3_sparse':  'trained-models/simple-cnn-conv3_sparse',
        'conv4_sparse':  'trained-models/simple-cnn-conv4_sparse',
        'fc1_sparse':    'trained-models/simple-cnn-fc1_sparse',
        'fc2_sparse':    'trained-models/simple-cnn-fc2_sparse',
        'fc3_sparse':    'trained-models/simple-cnn-fc3_sparse',
    }

    output_file, mpaths = adjust_paths(mpaths,dropout=dropout, comp_single=comp_single)


    for mtype, path in mpaths.items():
        try:
            model = models.SimpleCNN()               # reg_layer is irrelevant at eval
            model.load_state_dict(torch.load(path))
            model.eval()
            model.to(device)
            trained_models[mtype] = model
        except Exception as e:
            print(f"[warn] Skipping {mtype}: couldn't load '{path}' ({e})")

    # Test data
    data_test = utils.TMLDataset('test', transform=transforms.ToTensor())
    loader_test = DataLoader(data_test,
                             batch_size=consts.BATCH_SIZE,
                             shuffle=True,
                             num_workers=2)

    # Evaluate: accuracy + PGD success rate
    results = []
    print('Model accuracy and robustness:')
    for mtype, model in trained_models.items():
        acc = utils.compute_accuracy(model, loader_test, device)
        attack = attacks.PGDAttack(model, eps=consts.PGD_Linf_EPS)
        x_adv, y = utils.run_whitebox_attack(
            attack, loader_test, False, device, n_classes=4
        )
        sr = utils.compute_attack_success(
            model, x_adv, y, consts.BATCH_SIZE, False, device
        )
       
        bb_attack = attacks.NESBBoxPGDAttack(model)
        tmp_line = ""
        for momentum in [0, 0.9]:
            for targeted in [False, True]:
                bb_attack.momentum = momentum
                x_adv, y, n_queries = utils.run_blackbox_attack(bb_attack, loader_test, targeted, device)
                srNes = utils.compute_attack_success(model, x_adv, y, consts.BATCH_SIZE, targeted, device)
                median_queries = torch.median(n_queries)
                if targeted:
                    print(f'Targeted black-box attack (momentum={momentum:0.2f}):')
                else:
                    print(f'Untargeted black-box attack (momentum={momentum:0.2f}):')
                print(f'\t- success rate: {srNes:0.4f}\n\t- median(# queries): {median_queries}')
                tmp_line = tmp_line + f" | nes_sr={srNes:.4f},queries={median_queries}"

        line = f"{mtype:12s} | acc={acc:.4f} | pgd_sr={sr:.4f}"
        print('\t' + line)
        results.append(line)
        results.append(tmp_line)

    # Save to file
    with open(output_file, "w") as f:
        f.write("\n".join(results))
    print(f"Saved evaluation to {output_file}")


def visualize(file_path):
    # Storage
    layers = []
    accs = []
    pgd_srs = []
    nes_values = [[] for _ in range(4)]  # 4 NES configs
    
    with open(file_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]  # clean + skip empty lines

    # Process in chunks of 2 lines (first line = acc/pgd, second line = NES values)
    for i in range(0, len(lines), 2):
       
        if i == 0:
            continue

        line1 = lines[i]
        line2 = lines[i+1] if i+1 < len(lines) else ""

        # Match layer + acc + pgd_sr
        #match = re.match(r"(\w+)\s*\|\s*acc=([\d.]+)\s*\|\s*pgd_sr=([\d.]+)", line1)
        
        pattern = r"([\w.\-]+)\s*\|\s*acc=([\d.]+)\s*\|\s*pgd_sr=([\d.]+)"
        #pattern = r"([\w\-]+)\s*\|\s*acc=([\d.]+)\s*\|\s*pgd_sr=([\d.]+)"
        match = re.match(pattern, line1)
        
        if not match:
            continue

        layer, acc, pgd = match.groups()
        
        ####
        print(layer)
        tmp = layer.split("_")[-1]
        layer = tmp
        
        ####
        
        
        layer = layer.replace("_sparse", "")  # strip suffix
        layers.append(layer)
        accs.append(float(acc))
        pgd_srs.append(float(pgd))

        # Extract NES values from second line
        nes_matches = re.findall(r"nes_sr=([\d.]+),queries=\d+", line2)
        for j, val in enumerate(nes_matches):
            nes_values[j].append(float(val))

    # -------- Plot 1: Accuracy vs PGD_SR --------
    plt.figure(figsize=(8, 5))
    #plt.plot(layers, accs, marker="o", label="Accuracy")
    plt.plot(layers, pgd_srs, marker="o", label="PGD Success Rate")
    plt.xlabel("Regularization Strength")
    plt.title("PGD-Success Rate - Conv4 Layer - L1 Regularization")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("layer_acc_pgd.png", dpi=300)
    plt.close()

    # -------- Plot 2: NES curves --------
    nes_labels = [
        "momentum=0",
        #"NES (momentum=0, targeted=True)",
        "momentum=0.9",
        #"NES (momentum=0.9, targeted=True)"
    ]

    plt.figure(figsize=(8, 5))
    nes_values = [nes_values[0], nes_values[2]]
    #nes_labels=nes_labels[0,2]

    for j, nes_curve in enumerate(nes_values):
        plt.plot(layers, nes_curve, marker="o", label=nes_labels[j])
    plt.xlabel("Regularization Strength")
    plt.title("NES-Success Rate - Conv4 Layer - L1 Regularization")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("layer_nes.png", dpi=300)
    plt.close()




if __name__=='__main__':

    args = parse_arguments()

    if args.train == 1:
        #variants = ["fc1", "conv3", "conv1", "conv2",  "conv4", "fc2", "fc3", "conv3",  None] #,  
        #for reg_layer in variants:
        #    #tag = run_standard_training(reg_layer=reg_layer)  # uses default reg_strength
        #    tag = run_standard_training(reg_layer=reg_layer, dropout=True)  # uses default reg_strength
        
        variants = ["fc3"] #,  f
        for reg_layer in variants:
            for reg_strength in [1e-5, 2e-4, 1e-2, 0.1, 1.0,10.0]:
                tag = run_standard_training(reg_layer=reg_layer,reg_strength=reg_strength)  # uses default reg_strength


        #run_evaluation()
        #run_evaluation(dropout=True)
        run_evaluation(comp_single='fc3_sparse')
        #run_evaluation(comp_single='conv4_sparse')
        print(f"Finished training variant: {tag}")
    elif args.train == 0:
       run_evaluation(comp_single='fc1_sparse')
    else:
        #plot_sparsification_results("eval_results.txt")
        #visualize("eval_results_dropout.txt")
        visualize("eval_results_conv4_sparse.txt")

