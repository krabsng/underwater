from models.global_net import Global_pred
import torch
import torch.nn as nn
from models.krabs_model import KrabsNet
from models.prompt_ir_model import PromptGenBlock



if __name__ == '__main__':
    a = '3;/home/ljp/a-wxk/MicroExpress/MicroExpressionRecognition-master/datasets/train/others/img-30 (60).jpg'.split(';')[1].split('')
    print(a)



