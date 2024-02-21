import random

import torch
import torchvision
from torchvision import datasets
from torchvision import transforms

from PIL import Image

from torch.utils.tensorboard import SummaryWriter
import argparse
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
import matplotlib

import matplotlib.pyplot as plt
from matplotlib import offsetbox
import seaborn as sns
import os
class MNIST(datasets.MNIST):

    def __init__(self, *args, **kwargs):
        super(MNIST, self).__init__(*args, **kwargs)
        if kwargs["train"] is True:
            self.data, self.labels = self.train_data, self.train_labels
        else:
            self.data, self.labels = self.test_data, self.test_labels

    def __getitem__(self, idx):
        x1, t1 = self.data[idx], self.labels[idx]

        is_diff = random.randint(0, 1)
        while True:
            idx2 = random.randint(0, len(self)-1)
            x2, t2 = self.data[idx2], self.labels[idx2]
            if is_diff and t1 != t2:
                break
            if not is_diff and t1 == t2:
                break

        x1, x2 = Image.fromarray(x1.numpy()), Image.fromarray(x2.numpy())
        if self.transform is not None:
            x1, x2 = self.transform(x1), self.transform(x2)
        return x1, x2, int(is_diff)
def get_loaders(batch_size):
    train_loader = torch.utils.data.DataLoader(
        MNIST("./data", train=True, download=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])),
        batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        MNIST("./data", train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor()])),
        batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader
class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1,out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.c3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool_Max =nn.MaxPool2d(stride=2,kernel_size=2)
        self.pool_Avg =nn.AvgPool2d(2)
        self.fc4 = nn.Linear(in_features=128*2*2, out_features=64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(in_features=64, out_features=32)
        self.bn5=nn.BatchNorm1d(32)
        self.out=nn.Linear(in_features=32,out_features=2)

    def forward(self, x):
        #print(x.shape)
        h=self.c1(x)
        #print(h.shape)
        h=self.bn1(h)
        #print(h.shape)
        h = self.pool_Max(h)
        #print(h.shape)
        h=self.c2(h)
        h=self.bn2(h)
        h = self.pool_Max(h)
        #print(h.shape)
        h=self.c3(h)
        h=self.bn3(h)
        h = self.pool_Avg(h)
        #print(h.shape)
        h=h.flatten(start_dim=1)
        ##print(h.shape)
        h=self.fc4(h)
        h=self.bn4(h)
        h=self.fc5(h)
        h=self.bn5(h)
        h=self.out(h)

        return h
def contractive_loss(o1, o2, y):
    g, margin = F.pairwise_distance(o1, o2), 5.0
    loss = (1 - y) * (g ** 2) + y * (torch.clamp(margin - g, min=0)**2)
    return torch.mean(loss)
def grade(o1, o2):
    g, margin = F.pairwise_distance(o1, o2), 5.0
    loss=(g ** 2)
    return torch.mean(loss)
def train_loop(args):
    comment=f'batch size {args.batch_size} learning rate {args.lr}'
    tb=SummaryWriter(comment=comment)
    train_loader,test_loader=get_loaders(args.batch_size)
    net=Siamese().cuda()
    optimizer=optim.Adam(net.parameters(),lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5, 10], 0.1)
    cudnn.benckmark = True
    #images1,images2,labels=next(iter(train_loader))
    images1,images2,labels=next(iter(train_loader))
    images=torch.cat([images1,images2],dim=0)
    grid=torchvision.utils.make_grid(images)
    tb.add_image("images",grid)
    tb.add_graph(net,images.cuda())
    print("\t".join(["Epoch", "TrainLoss", "TestLoss"]))
    for epoch in range(args.epochs):
        scheduler.step()
        train_loss=0
        train_cnt=0
        net.train()
        for x1,x2,y in tqdm(train_loader,total=len(train_loader),leave=False):
            x1=Variable(x1.cuda())
            x2=Variable(x2.cuda())
            y=Variable(y.float().cuda().reshape(y.size(0),1))
            o1,o2=net(x1),net(x2)
            loss=contractive_loss(o1,o2,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss=loss.item()*y.size(0)
            train_cnt+=y.size(0)
        net.eval()

        test_loss, test_cnt = 0, 0
        with torch.no_grad():
            for x1, x2, y in tqdm(test_loader, total=len(test_loader), leave=False):
                x1, x2 = Variable(x1.cuda()), Variable(x2.cuda())
                y = Variable(y.float().cuda()).view(y.size(0), 1)

                o1, o2 = net(x1), net(x2)
                loss = contractive_loss(o1, o2, y)
                test_loss = loss.item() * y.size(0)
                test_cnt += y.size(0)
        tb.add_scalar("test loss",test_loss,epoch)
        tb.add_scalar("train loss",train_loss,epoch)
        for name, weight in net.named_parameters():
            tb.add_histogram(f'{name}.weight',weight,epoch)
            tb.add_histogram(f'{name}.weight.grad',weight.grad,epoch)
        if (epoch + 1) % 5 == 0:
            torch.save(net, "./checkpoint/{}_MINTS.tar".format(epoch+1))
        print("{}\t{:.6f}\t{:.6f}".format(epoch, train_loss / train_cnt, test_loss / test_cnt))
    tb.close()
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args([])
train_loop(args)   