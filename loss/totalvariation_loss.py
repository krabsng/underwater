import torch
import torch.nn as nn

class TotalVariationLoss(nn.Module):
    """
        该损失函数有助于减少生成图像中的噪声和伪影，使得图像更加平滑和自然。
    """
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        # Compute total variation loss as sum of differences between neighboring pixels
        tv_loss = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])) + \
                  torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))

        return tv_loss / (batch_size * channels * height * width)



