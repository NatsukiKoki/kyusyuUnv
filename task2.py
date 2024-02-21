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
import os

if not os.path.exists('./checkpoint/'):
        os.makedirs('./checkpoint/')
		
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
class CIFAR10(datasets.CIFAR10):

    def __init__(self, *args, **kwargs):
        super(CIFAR10, self).__init__(*args, **kwargs)
        

    def __getitem__(self, idex):
        x1, t1 = self.data[idex], self.targets[idex]

        is_diff = random.randint(0, 1)
        while is_diff:
            idex2 = random.randint(0, len(self)-1)
            x2, t2 = self.data[idex2], self.targets[idex2]
            if t1 != t2:
                break
        while not is_diff:
            idex2 = random.randint(0, len(self)-1)
            x2, t2 = self.data[idex2], self.targets[idex2]
            if t1 == t2:
                break
        x1, x2 = Image.fromarray(x1), Image.fromarray(x2)
        if self.transform is not None:
            x1, x2 = self.transform(x1), self.transform(x2)
        return x1, x2, int(is_diff)
def get_loaders(batch_size):
    train_loader = torch.utils.data.DataLoader(
         CIFAR10(root="./data", train=True, download=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])),
        batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
         CIFAR10(root="./data", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor()])),
        batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader
	
class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3,out_channels=96, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.c2 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(192)
        self.c3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm2d(384)
        self.pool_Max =nn.MaxPool2d(stride=2,kernel_size=2)
        self.pool_Avg =nn.AvgPool2d(2)
        self.fc4 = nn.Linear(in_features=384*3*3, out_features=126)
        self.bn4 = nn.BatchNorm1d(126)
        self.fc5 = nn.Linear(in_features=126, out_features=42)
        self.bn5=nn.BatchNorm1d(42)
        self.out=nn.Linear(in_features=42,out_features=2)

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
        #print(h.shape)
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
    comment=f'Siamese batch size {args.batch_size} learning rate {args.lr}'
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
        
        train_loss=0
        train_cnt=0
        for x1,x2,y in tqdm(train_loader,total=len(train_loader),leave=False):
            x1=Variable(x1.cuda())
            x2=Variable(x2.cuda())
            y=Variable(y.float().cuda().reshape(y.size(0),1))
            o1,o2=net(x1),net(x2)
            loss=contractive_loss(o1,o2,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()*y.size(0)
            train_cnt+=y.size(0)
        #net.eval()
        scheduler.step()
        test_loss, test_cnt = 0, 0
        with torch.no_grad():
            for x1, x2, y in tqdm(test_loader, total=len(test_loader), leave=False):
                x1, x2 = Variable(x1.cuda()), Variable(x2.cuda())
                y = Variable(y.float().cuda()).view(y.size(0), 1)

                o1, o2 = net(x1), net(x2)
                loss = contractive_loss(o1, o2, y)
                test_loss += loss.item() * y.size(0)
                test_cnt += y.size(0)
        tb.add_scalar("test loss",test_loss,epoch)
        tb.add_scalar("train loss",train_loss,epoch)
        for name, weight in net.named_parameters():
            tb.add_histogram(f'{name}.weight',weight,epoch)
            tb.add_histogram(f'{name}.weight.grad',weight.grad,epoch)
        if (epoch + 1) % 2 == 0:
            torch.save(net, "./checkpoint/{}.tar".format(epoch+1))
        print("{}\t{:.6f}\t{:.6f}".format(epoch, train_loss / train_cnt, test_loss / test_cnt))
    tb.close()
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.01)
args = parser.parse_args([])
if not os.path.exists('./checkpoint/100.tar'):
	train_loop(args)
def distribution(args):
    number_of_items = 15
    #sns.set(style="whitegrid", font_scale=1.5)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10("./data", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor()
        ])), batch_size=100, shuffle=True)
    net = torch.load(args.ck_path)

    net.eval()

    inputs, embs, targets = [], [], []
    for x1, t in tqdm(test_loader, total=len(test_loader)):
        x1 = Variable(x1.cuda())
        o1 = net(x1)
        #print(x1.shape)
        inputs.append(x1.cpu().data.numpy())
        embs.append(o1.cpu().data.numpy())
        targets.append(t.numpy())
    #print(inputs.count())
    inputs = np.array(inputs).reshape(-1,3,32, 32)
    inputs=inputs.transpose(0,2,3,1)
    embs = np.array(embs).reshape((-1, 2))
    targets = np.array(targets).reshape((-1,))

    n_plots = args.n_plots

    plt.figure(figsize=(32, 32))
    ax = plt.subplot(111)
    ax.set_title("CIFAR10 2D embeddigs")
    for x, e, t in zip(inputs[:n_plots], embs[:n_plots], targets[:n_plots]):
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(x, zoom=0.5, cmap=plt.cm.brg),
            xy=e, frameon=False)
        ax.add_artist(imagebox)
    
    ax.set_xlim(embs[:, 0].min(), embs[:, 0].max())
    ax.set_ylim(embs[:, 1].min(), embs[:, 1].max())
    plt.tight_layout()
    plt.savefig("./vis.png")
    plt.show()
    plt.close()
def visualize(args):
    number_of_items = 30
    #sns.set(style="whitegrid", font_scale=1.5)

    _, test_loader = get_loaders(100)
    net = torch.load(args.ck_path)

    net.eval()
    test_loss, test_cnt = 0, 0
    input1, input2 = [], []
    err= []
    for x1, x2, y in tqdm(test_loader, total=len(test_loader), leave=False):
        x1, x2 = Variable(x1.cuda()), Variable(x2.cuda())
        y = Variable(y.float().cuda()).view(y.size(0), 1)
        #print(x1.shape)
        o1, o2 = net(x1), net(x2)

        input1.append(x1.cpu().data.numpy())
        input2.append(x2.cpu().data.numpy())
        
        loss = grade(o1, o2)
        test_loss += loss.item() * y.size(0)
        test_cnt += y.size(0)

        err.append(loss.item())

    #print("\t Average loss: {:.6f}".format(test_loss / test_cnt))
    

    input1 = np.array(input1).reshape(-1,3, 32, 32)
    input2 = np.array(input2).reshape(-1,3, 32, 32)
    input1=input1.transpose(0,2,3,1)
    input2=input2.transpose(0,2,3,1)

    plt.figure(figsize=(120, 10))
    for item in range(number_of_items*2):
        display = plt.subplot(2, number_of_items,item+1)
        if item<30:
            plt.imshow(input1[item])
            #print(item)
        else:
            plt.imshow(input2[item-30])
            display.set_title("Error: {:.6f}".format(err[item-30]))
            #print(item)
        #plt.imshow(input2[item])
        
        display.get_xaxis().set_visible(False)
        display.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig('./1.png')
    plt.show()
    plt.close()
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument("--ck_path", type=str, default=r".\checkpoint\100.tar")
parser.add_argument("--n_plots", type=int, default=500)
args_for_show = parser.parse_args([])
distribution(args_for_show)
visualize(args_for_show)
