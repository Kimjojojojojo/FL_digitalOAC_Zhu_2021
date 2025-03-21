import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from dataclasses import dataclass, field

def MNIST_dataloader(batch, batch_set, subset_size, params):
    transform = transforms.Compose([transforms.ToTensor()])
    full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    # ✅ torch로 인덱스 생성 및 셔플
    indices = torch.randperm(len(full_train_dataset))

    # ✅ K명 사용자에게 데이터 분배
    train_datasets = []
    for k in range(params.K):
        subset_indices = indices[k * subset_size : (k + 1) * subset_size]
        train_datasets.append(Subset(full_train_dataset, subset_indices.tolist()))  # torch tensor → list

    # ✅ 사용자별 고정 DataLoader 생성
    train_loaders = [
        DataLoader(train_datasets[k], batch_size=batch_set[k], shuffle=False)
        for k in range(params.K)
    ]

    return train_loaders, test_loader
