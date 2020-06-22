import numpy as np
from tqdm import tqdm

import jittor as jt
from jittor import nn

jt.flags.use_cuda = 1

from pointnet import PointNet
from modelnet40_loader import ModelNet40
from utils import LRScheduler


def freeze_random_seed():
    np.random.seed(0)


def train(net, optimizer, epoch, dataloader):
    net.train()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [TRAIN]')
    for pts, normals, labels in pbar:
        output = net(pts, normals)
        loss = nn.cross_entropy_loss(output, labels)
        optimizer.step(loss)

        pred = np.argmax(output.data, axis=1)
        acc = np.mean(pred == labels.data) * 100

        pbar.set_description(f'Epoch {epoch} [TRAIN] loss = {loss.data[0]:.2f}, acc = {acc:.2f}')

def evaluate(net, epoch, dataloader):
    total_acc = 0
    total_num = 0

    net.eval()
    for pts, normals, labels in tqdm(dataloader, desc=f'Epoch {epoch} [Val]'):
        pts = jt.float32(pts.numpy())
        normals = jt.float32(normals.numpy())
        labels = jt.int32(labels.numpy())

        output = net(pts, normals)
        pred = np.argmax(output.data, axis=1)
        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0]

    acc = total_acc / total_num
    return acc


if __name__ == '__main__':
    freeze_random_seed()

    net = PointNet(n_classes=40)
    optimizer = nn.Adam(net.parameters(), lr=1e-3)

    lr_scheduler = LRScheduler(optimizer)

    batch_size = 32
    train_dataloader = ModelNet40(n_points=4096, batch_size=batch_size, train=True, shuffle=True)
    val_dataloader = ModelNet40(n_points=4096, batch_size=batch_size, train=False, shuffle=False)

    step = 0
    best_acc = 0
    for epoch in range(1000):
        lr_scheduler.step(len(train_dataloader) * batch_size)

        train(net, optimizer, epoch, train_dataloader)
        acc = evaluate(net, epoch, val_dataloader)

        best_acc = max(best_acc, acc)
        print(f'val acc={acc:.4f}, best={best_acc:.4f}')
