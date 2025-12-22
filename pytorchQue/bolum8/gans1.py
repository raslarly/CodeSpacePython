"""
image generation: MNIST veri seti
"""
# kutuphanelerin import edilmesi
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# %% veriseti hazirlama
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# min batch boyutu
batch_size = 128

# goruntu boyutu
image_size = 28*28

transform = transforms.Compose([
    # goruntuleri tensore cevir
    transforms.ToTensor(),
    # Normalizasyon -> -1 ile 1 arasina sikistir
    transforms.Normalize((0.5,),(0.5,))])

# MNIST verisetini yukleme
dataset = datasets.MNIST(root='./data', train=True, transform=transform,
                         download=True)

# verisetini batchler halinde yukle
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %% discriminator olustur

# Ayirt edici: Generator un uretmis oldugu goruntulerin gercek mi sahte mi 
# oldugunu ayirt edecek

class discriminator():

# %% generator olustur


# %% gan training


# %% model testing and performance evaluation



















