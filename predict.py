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
import json


argumentparser = argparse.ArgumentParser(description = 'predict.py')
argumentparser.add_argument('input', default = 'flowers/test/100/image_07896.jpg', action="store", type = str)
argumentparser.add_argument('checkpoint', default = 'checkpoint.pth', action="store",type = str)
argumentparser.add_argument('--top_k', default = 5, dest = "top_k", action = "store", type = int)
argumentparser.add_argument('--category_names', dest = "category_names", action = "store", default='cat_to_name.json')
argumentparser.add_argument('--gpu', default = "gpu", action = "store", dest = "gpu")

parser = argumentparser.parse_args()
test_image = parser.input
top_k = parser.top_k
power = parser.gpu
checkpoint = parser.checkpoint


def predict_image():
    if parser.checkpoint == 'checkpoint':
        checkpoint = 'checkpoint.pth'
        
    category_names = parser.category_names

    model = utils.load_checkpoint(checkpoint)

    with open(category_names, 'r') as json_file:
        cat_to_name = json.load(json_file, strict = False)

    processed_img = utils.process_image(test_image)
    top_probs, class_names = utils.predict(processed_img, model, top_k, cat_to_name, power)

    print("Categories with class names:")
    for i in range(len(top_probs)):
        print(class_names[i], ": ", top_probs[i])


if __name__ == "__main__":
    predict_image()
    