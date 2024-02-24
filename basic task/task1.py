
from numpy import true_divide
import argparse
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets
from torch.utils.data.dataset import Subset
from torchvision.utils import save_image
import math
from IPython.display import Image
import matplotlib.pyplot as plt
from torch.optim import Adam
import random
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter


batch_size = 250
epochs = 20
DEVICE="cuda"

def linear_beta_schedule(timesteps, start=0.0001,end=0.01):
    return torch.linspace(start,end,timesteps).to(DEVICE)
def get_index(vals, t,x_shape):
    batch_size=t.shape[0]
    out = vals.gather(-1,t.cuda())
    return out.reshape(batch_size,*((1,)*(len(x_shape)-1))).to(t.device)
def forward_diffusion_sample(x0,t,device="cuda:0"):
    noise=torch.rand_like(x0)
    sqrt_alphas_cumprod_t = get_index(sqrt_alphas_cumprod,t,x0.shape)
    sqrt_one_minus_alphas_cumprod_t=get_index(sqrt_one_minus_alphas_cumprod,t,x0.shape)
    aaa=sqrt_alphas_cumprod_t.to(device) * x0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
    aaa=F.sigmoid(aaa)
    aaa=aaa*2-1
    return aaa, noise.to(device)
# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)
print(betas.device)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
print(alphas_cumprod_prev.device)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
print(sqrt_alphas_cumprod.device)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def tensor2var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def flatten(x):
    return tensor2var(x.view(x.size(0), -1))

def saveImage(x, path='real_image.png'):
    save_image(x, path)

# load data
dataset = datasets.MNIST(root='../data', train=True, transform=transforms.ToTensor(), download=True)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
)

# fixed input
fixed_x, _ = next(iter(data_loader))
saveImage(fixed_x)
fixed_x = flatten(fixed_x)

Image('real_image.png')

class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, z_dim*2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, image_size),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = tensor2var(torch.randn(*mu.size()))
        z = mu + std * esp
        return z
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

#vae = VAE()
train=False

if not os.path.exists("./checkpoint/VAE_20.tar"):
    vae=VAE()
    train=True
else:
    vae=torch.load("./checkpoint/VAE_20.tar")
    train=False
if torch.cuda.is_available():
    vae.cuda()



def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
    return BCE + KLD
def loss_fn_(recon_x,recon_noi_x,x,mu,logvar,mu_no,logvar_no):
    BCE = F.binary_cross_entropy(recon_x, x)
    BCE_noi=F.binary_cross_entropy(recon_noi_x,x)
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
    KLD_no = -0.5 * torch.sum(1 + logvar_no - mu_no**2 -  logvar_no.exp())
    return 5*BCE+5*BCE_noi+KLD*5+KLD_no*5

if not os.path.exists('reconstructed'):
    os.makedirs('reconstructed')
def train_loop():
    for epoch in range(epochs):
        comment=f'VAE batch size {batch_size} learning rate {1e-3}'
        tb=SummaryWriter(comment=comment)
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
        images,_=next(iter(data_loader))
        images=flatten(images)
        grid=torchvision.utils.make_grid(images)
        tb.add_image("images",grid)
        #tb.add_graph(vae,images.cuda())
        vae.train()
        aa=0
        input1, input2,input3,input4 = np.empty((1,1,28,28)),np.empty((1,1,28,28)),np.empty((1,1,28,28)),np.empty((1,1,28,28))
        for images, idx in tqdm(data_loader,total=len(data_loader),leave=False):
            
            aa+=1
            if aa%10==0:
                input1=np.vstack((images.cpu().data.numpy(),input1))
            images=Variable(images.cuda())
            images=images*2-1
            y=torch.randint(1, T//10, (batch_size,), device=DEVICE).long()
            x2,noise=forward_diffusion_sample(images, y, DEVICE)
            x2=(x2+1)/2
            if aa%10==0:
                input3=np.vstack((x2.cpu().data.numpy(),input3))
                
            images=(images+1)/2
            x2=flatten(x2)
            images = flatten(images)
            recon_images, mu, logvar = vae(images)
            rec_noi,no_mu,no_logvar=vae(x2)
            
            loss=loss_fn(recon_images,images,mu,logvar)    
            #loss = loss_fn_(recon_images,rec_noi, images, mu, logvar,no_mu,no_logvar)
            if aa%10==0:
                rec_noi=rec_noi.view(rec_noi.size(0), 1, 28, 28).data.cpu()
                input4=np.vstack((rec_noi.cpu().data.numpy(),input4))
            if aa%10==0:
                recon_images=recon_images.view(recon_images.size(0), 1, 28, 28).data.cpu()
                input2=np.vstack((recon_images.data.numpy(),input2))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        vae.eval()
        for name, weight in vae.named_parameters():
                tb.add_histogram(f'{name}.weight',weight,epoch)
                tb.add_histogram(f'{name}.weight.grad',weight.grad,epoch)
        input1 = np.array(input1).reshape(-1, 28, 28)
        input2 = np.array(input2).reshape(-1, 28, 28)
        input3 = np.array(input3).reshape(-1, 28, 28)
        input4 = np.array(input4).reshape(-1, 28, 28)
        number_of_items=5
        plt.figure(figsize=(120, 40))
        for item in range(number_of_items*4):
            display = plt.subplot(4, number_of_items,item+1)
            if item<number_of_items:
                plt.imshow(input1[item])
            else:
                if item<number_of_items*2:
                    plt.imshow(input2[item-number_of_items])
                else:
                    if item<number_of_items*3:
                        plt.imshow(input3[item-number_of_items*2])
                    else:
                        plt.imshow(input4[item-number_of_items*3])
                        
                
            display.get_xaxis().set_visible(False)
            display.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig('./ans{}.png'.format(epoch+1))
        #plt.show()
        plt.close()
        if (epoch + 1) % 5 == 0:
            torch.save(vae, "./checkpoint/VAE_{}.tar".format(epoch+1))
        #print("{}\t{:.6f}".format(epoch, train_loss / train_cnt))
        print("Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, loss.data/batch_size))
        recon_x, _, _ = vae(fixed_x)
        print(recon_x.shape)
        saveImage(recon_x.view(recon_x.size(0), 1, 28, 28).data.cpu(), f'reconstructed/recon_image_{epoch}.png')

if train:
    train_loop()

sample = tensor2var(torch.randn(128, 20))
recon_x = vae.decoder(sample)
# recon_x, _, _ = vae(fixed_x)

save_image(recon_x.view(recon_x.size(0), 1, 28, 28).data.cpu(), 'sample_image.png')
Image('sample_image.png')

def visualize():
    number_of_items = 30
    #sns.set(style="whitegrid", font_scale=1.5)

    test_loader = data_loader
    net = torch.load('./checkpoint/VAE_20.tar')
    #net=vae
    net.eval()
    data_transforms = [
            transforms.ToTensor()
        ]
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t * 255.)
    ])
    data_transform = transforms.Compose(data_transforms)
    #test_loss, test_cnt = 0, 0
    input1, input2,input3 = [], [],[]
    err= []
    for x1,_tp in tqdm(test_loader, total=len(test_loader), leave=False):
        x1=Variable(x1.cuda())
        #print(x1)
        x1=x1*2-1
        y=torch.randint(20, T//2, (batch_size,), device=DEVICE).long()
        x2,noise=forward_diffusion_sample(x1, y, DEVICE)
        x2=(x2+1)/2
        x2=flatten(x2)
        o2,_,__ = net(x2)
        #print(o2)
        #o2=F.normalize(o2)
        #o2=data_transform(o2)
       # print(o2)
        input1.append(x1.cpu().data.numpy())
        input2.append(x2.cpu().data.numpy())
        input3.append(o2.cpu().data.numpy())
    

    input1 = np.array(input1).reshape(-1, 28, 28)
    input2 = np.array(input2).reshape(-1, 28, 28)
    input3 = np.array(input3).reshape(-1, 28, 28)
    plt.figure(figsize=(20, 10))
    for item in range(number_of_items*3):
        display = plt.subplot(3, number_of_items,item+1)
        if item<number_of_items:
            plt.imshow(input1[item])
        else:
            if item<number_of_items*2:
                plt.imshow(input2[item-number_of_items])
                #plt.imshow(input3[item-number_of_items])
            else:    
                plt.imshow(input3[item-number_of_items*2])
        display.get_xaxis().set_visible(False)
        display.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig('./ans.png')
    plt.show()
    plt.close()
visualize()

