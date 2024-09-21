import torch
import torch.nn as nn


class PSNR(nn.Module):
    """峰值信噪比评价指标（有参考图像）
    """

    def __init__(self, max_val=0):
        super().__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        """
        parameters:
            a() -- 输入的图像
            b() -- 参考图像
        """
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return 0

        return 10 * torch.log10((1.0 / mse))

def getPSNR():
    """给外部调用用来获取PSNR的实例
        Parameters:
            无参数
        returns:
            返回一个PSNR的实例
    """
    # 创建一个psnr对象
    return PSNR()


