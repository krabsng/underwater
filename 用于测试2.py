from .models.spu_gan_model import VGG19_Discriminator
import torch.nn as nn
import torch
if __name__ == "__main__":
    x = torch.randn(3,256,256)
    net = VGG19_Discriminator()
    x = net(x)