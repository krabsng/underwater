import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


def setup(rank, world_size):
    """
    初始化进程组
    """
    dist.init_process_group(
        backend='nccl',  # 推荐使用 NCCL 后端
        init_method='env://',  # 使用环境变量进行初始化
        world_size=world_size,
        rank=rank
    )
    torch.manual_seed(0)


def cleanup():
    """
    销毁进程组
    """
    dist.destroy_process_group()


def train(rank, world_size):
    """
    每个进程的训练过程
    """
    setup(rank, world_size)

    # 设置当前进程的 GPU
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # 数据预处理和加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # 定义模型并移动到对应 GPU
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    ).to(device)

    # 包装模型
    ddp_model = DDP(model, device_ids=[rank])

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # 训练循环
    for epoch in range(5):
        sampler.set_epoch(epoch)  # 设置种子以确保每个 epoch 的数据打乱方式不同
        ddp_model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 99:
                print(f'Rank {rank}, Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

    cleanup()


def main():
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise ValueError("需要至少一个 GPU 来运行")
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
