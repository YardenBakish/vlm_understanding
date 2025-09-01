import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torch_models
"""
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, (3,3))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, (3,3))
        self.conv3 = nn.Conv2d(64, 128, (3,3))
        self.conv4 = nn.Conv2d(128, 128, (3,3))
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = (x-0.5)*2.
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""



class SimpleCNN(nn.Module):
    def __init__(self, n_classes=4, reg_layer=None,dropout=False):
        super().__init__()
        self.reg_layer = reg_layer
        self.conv1 = nn.Conv2d(3, 64, (3,3))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, (3,3))
        self.conv3 = nn.Conv2d(64, 128, (3,3))
        self.conv4 = nn.Conv2d(128, 128, (3,3))
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_classes)
        self._activation = None
        self.dropout = dropout
        self.p_conv = 0.2
        self.p_fc = 0.5

    def forward(self, x):
        x = (x - 0.5) * 2.

        # conv1
        x = F.relu(self.conv1(x))
        if self.reg_layer == "conv1":
            x = F.dropout(x, p=self.p_conv, training=self.training and self.dropout)
            self._activation = x

        # conv2
        x = self.pool(F.relu(self.conv2(x)))
        if self.reg_layer == "conv2":
            x = F.dropout(x, p=self.p_conv, training=self.training and self.dropout)
            self._activation = x

        # conv3
        x = F.relu(self.conv3(x))
        if self.reg_layer == "conv3":
            x = F.dropout(x, p=self.p_conv, training=self.training and self.dropout)
            self._activation = x

        # conv4
        x = self.pool(F.relu(self.conv4(x)))
        if self.reg_layer == "conv4":
            x = F.dropout(x, p=self.p_conv, training=self.training and self.dropout)
            self._activation = x

        # flatten
        x = torch.flatten(x, 1)

        # fc1
        x = F.relu(self.fc1(x))
        if self.reg_layer == "fc1":
            x = F.dropout(x, p=self.p_fc, training=self.training and self.dropout)
            self._activation = x

        # fc2
        x = F.relu(self.fc2(x))
        if self.reg_layer == "fc2":
            x = F.dropout(x, p=self.p_fc, training=self.training and self.dropout)
            self._activation = x

        # fc3
        x = self.fc3(x)
        if self.reg_layer == "fc3":
            x = F.dropout(x, p=self.p_fc, training=self.training and self.dropout)
            self._activation = x

        return x
