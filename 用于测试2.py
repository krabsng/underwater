import torch
from models.spu_model import DualPathDownsampling,DualPathUpsampling,SwinTransformerBlock
if __name__ == '__main__':
    net = SwinTransformerBlock(dim=int(64 * 2 ** 4), num_heads=4, window_size=4)
    Feature_in = torch.randn(2, 1024, 30, 40)
    Feature_out = net(Feature_in)
    print(Feature_out.shape)
