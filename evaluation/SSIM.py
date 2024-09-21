from IQA_pytorch import SSIM


def getSSIM():
    """给外部调用用来获取SSIM的实例
        Parameters:
            无参数
        returns:
            返回一个SSIM的实例
    """
    return SSIM()
