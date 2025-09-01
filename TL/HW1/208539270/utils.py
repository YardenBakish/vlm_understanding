import gzip
import struct
from os import path
import numpy as np
import models
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def load_pretrained_cnn(cnn_id, n_classes=4, models_dir='trained-models/'):
    """
    Loads one of the pre-trained CNNs that will be used throughout the HW
    """
    if not isinstance(cnn_id, int) or cnn_id < 0 or cnn_id > 2:
        raise ValueError(f'Unknown cnn_id {id}')
    model = eval(f'models.SimpleCNN{cnn_id}(n_classes=n_classes)')
    fpath = path.join(models_dir, f'simple-cnn-{cnn_id}')
    model.load_state_dict(torch.load(fpath))
    return model


class TMLDataset(Dataset):
    """
    Used to load the dataset used throughout the HW
    """

    def __init__(self, fpath='dataset.npz', transform=None):
        with gzip.open(fpath, 'rb') as fin:
            self.data = np.load(fin, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def compute_accuracy(model, data_loader, device):
    """
    Evaluates and returns the (benign) accuracy of the model 
    (a number in [0, 1]) on the labeled data returned by 
    data_loader.
    """
    total = 0
    acc   = 0
    for data in data_loader:
        x, y = data[0].to(device), data[1].to(device)
        total+=data[1].shape[0]
        pred = torch.argmax(model(x),dim=-1) 
        acc+= (pred==y).sum()
   
    return acc/total



def run_whitebox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the white-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=c_x+randint(1, n_classes)%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    """
    X = []
    Y = []
    for data in data_loader:
        x, y = data[0].to(device), data[1].to(device)
        if targeted == True:
            y=(y+torch.randint_like(y,1, n_classes))%n_classes
        x_tags =attack.execute(x,y,targeted)
        X.append(x_tags)
        Y.append(y)
    
    X = torch.cat(X)
    Y = torch.cat(Y)

    return X,Y


def run_blackbox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the black-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=(c_x+randint(1, n_classes))%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    3- The number of queries made to create each adversarial example.
    """
    X , Y, Q = [] , [], []
    for data in data_loader:
        x, y = data[0].to(device), data[1].to(device)
        if targeted == True:
            y=(y+torch.randint_like(y,1, n_classes))%n_classes
        x_tags, queries = attack.execute(x,y,targeted)
        X.append(x_tags)
        Y.append(y)
        Q.append(queries)
    
    X = torch.cat(X)
    Y = torch.cat(Y)
    Q = torch.cat(Q)

    return X,Y,Q



def compute_attack_success(model, x_adv, y, batch_size, targeted, device):
    """
    Returns the success rate (a float in [0, 1]) of targeted/untargeted
    attacks. y contains the true labels in case of untargeted attacks,
    and the target labels in case of targeted attacks.
    """
    preds       = torch.argmax(model(x_adv), dim=1)
    num_correct = torch.sum(preds==y) if targeted else torch.sum(preds!=y)
    return num_correct / len(x_adv)
    
   


def binary(num):
    """
    Given a float32, this function returns a string containing its
    binary representation (in big-endian, where the string only
    contains '0' and '1' characters).
    """
    packed = struct.pack('>f', num)  
    binary = ''.join(f'{byte:08b}' for byte in packed)  
    return binary


def float32(binary):
    """
    This function inverts the "binary" function above. I.e., it converts 
    binary representations of float32 numbers into float32 and returns the
    result.
    """
    byteArr = int(binary, 2).to_bytes(4, byteorder='big')
    num = struct.unpack('>f', byteArr)[0]
    return torch.tensor(num, dtype=torch.float32)


def random_bit_flip(w):
    """
    This functoin receives a weight in float32 format, picks a
    random bit to flip in it, flips the bit, and returns:
    1- The weight with the bit flipped
    2- The index of the flipped bit in {0, 1, ..., 31}
    """
    w_bin = binary(w)

    rand_idx       = np.random.randint(0, 31)
    flipped_bit    = str(1 - int(w_bin[rand_idx]))
    flipped_weight = w_bin[:rand_idx]+flipped_bit+w_bin[rand_idx+1:]
    flipped_weight = float32(flipped_weight)

    return flipped_weight, rand_idx
