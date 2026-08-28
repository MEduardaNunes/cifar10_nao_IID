"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor


class Net(nn.Module):
    """Small CIFAR-10 CNN using GroupNorm for federated stability."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


fds = None  # Cache FederatedDataset

normalize = Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
train_transforms = Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor(), normalize])
eval_transforms = Compose([ToTensor(), normalize])


def apply_transforms(batch, transforms):
    """Apply a split-specific transform to a FederatedDataset batch."""
    batch["img"] = [transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int, alpha: float = 0.5):
    """Load Dirichlet-partitioned CIFAR10 data."""
    global fds
    if fds is None:
        # Particionamento heterogêneo usando Dirichlet
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",        # Nome da coluna de classe no CIFAR-10
            alpha=alpha,                 # Concentração: menor = mais heterogêneo
            min_partition_size=10,       # Garante um mínimo de amostras por nó
            self_balancing=True,
            seed=42
        )
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
        
    partition = fds.load_partition(partition_id)
    
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    # Construct dataloaders
    train_dataset = partition_train_test["train"].with_transform(
        lambda batch: apply_transforms(batch, train_transforms)
    )
    test_dataset = partition_train_test["test"].with_transform(
        lambda batch: apply_transforms(batch, eval_transforms)
    )
    
    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return trainloader, testloader


def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(
        lambda batch: apply_transforms(batch, eval_transforms)
    )
    return DataLoader(dataset, batch_size=128)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    net.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    avg_trainloss = running_loss / (epochs * len(trainloader))
    train_accuracy = correct / total

    return avg_trainloss, train_accuracy


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


import matplotlib.pyplot as plt
import numpy as np


def plot_client_distribution(num_partitions=4):
    """Plot the class distribution of each federated client."""

    global fds

    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=0.5,
            min_partition_size=10,
            self_balancing=True,
            seed=42,
        )

        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )

    class_names = [
        "avião",
        "automóvel",
        "pássaro",
        "gato",
        "cervo",
        "cachorro",
        "sapo",
        "cavalo",
        "navio",
        "caminhão",
    ]

    distributions = []

    for partition_id in range(num_partitions):
        partition = fds.load_partition(partition_id)

        labels = partition["label"]

        counts = np.bincount(
            labels,
            minlength=10
        )

        distributions.append(counts)

    distributions = np.array(distributions)

    # Converter para proporção
    distributions = distributions / distributions.sum(axis=1, keepdims=True)

    # Gráfico
    x = np.arange(num_partitions)
    bottom = np.zeros(num_partitions)

    plt.figure(figsize=(12, 6))

    for class_id in range(10):
        plt.bar(
            x,
            distributions[:, class_id],
            bottom=bottom,
            label=class_names[class_id],
        )

        bottom += distributions[:, class_id]

    plt.xlabel("Cliente")
    plt.ylabel("Proporção das amostras")
    plt.title("Distribuição não-IID do CIFAR-10 entre os clientes")
    plt.xticks(x, [f"Cliente {i}" for i in range(num_partitions)])
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.show()
