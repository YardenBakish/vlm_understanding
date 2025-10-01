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


L1_REG_STRENGH = [1e-5, 2e-4, 1e-2, 0.1, 1.0,10.0]
DROPOUT_REG_STRENGH = [0.1, 0.2, 0.3, 0.4,0.5,0.6]



def find_files_with_prefix(directory, prefix):
    pattern = os.path.join(directory, prefix + "*")
    return [f for f in glob.glob(pattern) if os.path.isfile(f)]


torch.manual_seed(consts.SEED)
random.seed(consts.SEED)
np.random.seed(consts.SEED)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices = ["train", "eval", "vis"])
    parser.add_argument('--l1', action='store_true')
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--per_layer', action='store_true')
    parser.add_argument('--layers', nargs="+", default=["None", "conv1", "conv2","conv3", "conv4", "fc1", "fc2", "fc3"])
    parser.add_argument('--vis_text')

    args = parser.parse_args()
    return args


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
    model = models.SimpleCNN(reg_layer=reg_layer, dropout=dropout, reg_strength=reg_strength)
    model.to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    print(dropout)
    exit(1)

    # execute training
    model = utils.standard_train(model, data_tr, criterion, optimizer,
                                 scheduler, device,
                                 reg_strength=(reg_strength if (reg_layer and dropout == False) else 0.0))

    # save model
    model.to('cpu')
    tag = "standard" if reg_layer=="None" else f"{reg_layer}_sparse"
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



        


if __name__=='__main__':

    args = parse_arguments()

    if args.mode == "train":
        #variants = ["conv3", "conv1", "conv2",  "conv4", "fc1", "fc2", "fc3", "conv3",  None] #,  

        for reg_layer in args.layers:
            if args.per_layer:
                print(args.l1)
                exit(1)
                if args.l1 is None and args.dropout is None:
                    print("When choosing 'per_layer', you must specify --l1 or --dropout")
                    exit(1)
                else:
                    reg_values = L1_REG_STRENGH if args.l1 else DROPOUT_REG_STRENGH
                    for reg_strength in reg_values:
                        tag = run_standard_training(reg_layer=reg_layer,reg_strength=reg_strength)
            else:
                tag = run_standard_training(reg_layer=reg_layer,dropout = args.dropout)
                #tag = run_standard_training(reg_layer=reg_layer)  # uses default reg_strength
                #tag = run_standard_training(reg_layer=reg_layer, dropout=True)  # uses default reg_strength
            print(f"Finished training variant: {tag}")


        #for reg_layer in variants:
        #    #tag = run_standard_training(reg_layer=reg_layer)  # uses default reg_strength
        #    tag = run_standard_training(reg_layer=reg_layer, dropout=True)  # uses default reg_strength
        
        #variants = ["fc3"] #,  f
        #for reg_layer in variants:
        #    for reg_strength in [1e-5, 2e-4, 1e-2, 0.1, 1.0,10.0]:
        #        tag = run_standard_training(reg_layer=reg_layer,reg_strength=reg_strength)  # uses default reg_strength


        #run_evaluation()
        #run_evaluation(dropout=True)
        run_evaluation(comp_single='fc3_sparse')
        #run_evaluation(comp_single='conv4_sparse')
        print(f"Finished training variant: {tag}")
    elif args.mode == "eval":
       run_evaluation(comp_single='fc1_sparse')
    else:
        #plot_sparsification_results("eval_results.txt")
        #utils.visualize("eval_results_dropout.txt")
        utils.visualize(args.vis_text)

        #utils.visualize_queries()
        #utils.visualize_queries_dual("eval_results.txt","eval_results_dropout.txt")
        #utils.visualize("eval_results_conv4_sparse.txt")

