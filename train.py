import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
import utils

argumentparser = argparse.ArgumentParser(description = 'train.py')
argumentparser.add_argument('data_dir', nargs = '*', action = "store", default = "flowers/")
argumentparser.add_argument('--gpu', dest = "gpu", action = "store", default = "gpu")
argumentparser.add_argument('--save_dir', dest = "save_dir", action = "store", default = "checkpoint.pth")
argumentparser.add_argument('--learning_rate', dest = "learning_rate", action = "store", default = 0.001)
argumentparser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
argumentparser.add_argument('--epochs', dest = "epochs", action = "store", type = int, default = 3)
argumentparser.add_argument('--arch', dest = "arch", action = "store", default = "densenet121", type = str)
argumentparser.add_argument('--hidden_units', type = int, dest = "hidden_units", action = "store", default = 256)


parser = argumentparser.parse_args()
data_dir = parser.data_dir
save_dir = parser.save_dir
lr = parser.learning_rate
arch = parser.arch
dropout = parser.dropout
hidden_units = parser.hidden_units
power = parser.gpu
epochs = parser.epochs
learning_rate = parser.learning_rate

def train_the_model():
    model, optimizer = utils.model_setup(arch, dropout, hidden_units, lr, power)
    model = utils.train(model, epochs, optimizer, power)
    utils.validate_testset(model, power)
    utils.save_checkpoint(model, arch, dropout, epochs, save_dir, hidden_units, optimizer)


if __name__ == "__main__":
    train_the_model()