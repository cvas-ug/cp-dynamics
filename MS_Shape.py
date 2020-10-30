#pylint:skip-file
import torch
import os
import time
import seaborn as sns
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import time
from statistics import mean
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader
from utils.Picture_Dataset import PictureDataset
from utils.BatchRandomSampler import BatchRandomSampler
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn.functional as F
import pandas as pd

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
softmax=nn.Softmax(dim=1)

fig_dir='./MS_Shape/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

class AlexNet (nn.Module):
    def __init__(self):
        super (AlexNet,self).__init__()
        self.models=models.alexnet(pretrained=True)
        self.models=freeze(self.models)
        self.models.features[0]=nn.Conv2d(1,64, kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        num_feature=self.models.classifier[1].in_features
        classiifer=nn.Sequential(
            nn.Linear(in_features=num_feature,out_features=9216,bias=True),
            nn.Linear(in_features=9216,out_features=512,bias=True),
            nn.Linear(in_features=512,out_features=5,bias=True)
        )
        self.models.classifier=classiifer
    
    def forward(self,x):
        x=self.models(x)
        return x
    
    def loss_fn(self,pred_label,target_index_index):
        loss=criterion(pred_label,target_index_index)
        return loss

def freeze(model,use_feature=True):
    if use_feature:
        for param in model.parameters():
            param.requires_grad=False
    return model


class Features(nn.Module):
    def __init__(self,model):
        super(Features,self).__init__()
        self.model=nn.Sequential(
            model.models.features,
        )
        self.model=freeze(self.model)
    
    def forward(self,x):
        x=self.model(x)
        return x

class Classifier(nn.Module):
    def __init__(self,model):
        super(Classifier,self).__init__()
        self.avgpool=model.models.avgpool
        self.fc=model.models.classifier
        self.avgpool=freeze(self.avgpool)
        self.fc=freeze(self.fc)
    
    def forward(self,x):
        x=x.reshape(-1,256,15,15)
        x=self.avgpool(x)
        x=x.reshape(x.size()[0],-1)
        x=self.fc(x)
        return x

class LSTM(nn.Module):
    def __init__(self,num_features=225):
        super(LSTM,self).__init__()
        self.in_features=num_features
        self.hidden_features=num_features
        self.n_layer=1
        self.lstm=nn.LSTM(self.in_features,self.hidden_features,self.n_layer)
    
    def init_hidden_layer(self,batch_size):
        return torch.zeros(self.n_layer,batch_size,self.hidden_features)
    
    def forward(self,x):
        time_step=x.shape[0]
        batch_size=x.shape[1]
        x=x.view(time_step,batch_size,-1)
        hidden_0=Variable(self.init_hidden_layer(batch_size)).to(device)
        c_0=Variable(self.init_hidden_layer(batch_size)).to(device)
        out,(hidden_0,c_0)=self.lstm(x,(hidden_0,c_0))
        out=out[-1,:,:]
        out=F.relu(out)
        out=out.view(1,-1)
        return out

def move_average(dataloader,features,classifier,model):
    item_item=int(len(dataloader)/10)
    start_time=time.time()
    data=[]
    for t, input_ in enumerate(dataloader):
        pant=[]
        shirt=[]
        sweater=[]
        towel=[]
        t_shirt=[]
        inputs=Variable(input_['Image'][:,0:1,:,:]).to(device)
        Labels=input_['Label']
        for i in range (len(inputs)-3):
            input_frames=inputs[i:i+3]
            target_lable=Labels[i+3:i+4]
            feature=features(input_frames)
            pred_feature=model(feature)
            pred_label=classifier(pred_feature)
            possibility=softmax(pred_label)
            pant.append(possibility[0][0].item())
            shirt.append(possibility[0][1].item())
            sweater.append(possibility[0][2].item())
            towel.append(possibility[0][3].item())
            t_shirt.append(possibility[0][4].item())
        if (t+2)%(item_item+1)==0:
            print ('['+str(target_lable.item())+']','[Batch: %d/%d][Duration: %f][pant: %f][shirt: %f][sweater: %f][towel: %f][t-shirt: %f]'%(t+1,len(dataloader),time.time()-start_time,mean(pant),mean(shirt),mean(sweater),mean(towel),mean(t_shirt)))
            start_time=time.time()
        move_pant=[]
        move_shirt=[]
        move_sweater=[]
        move_towel=[]
        move_t_shirt=[]
        for i in range(len(pant)):
            pant_mean=mean(pant[:i+1])
            move_pant.append(pant_mean)
            shirt_mean=mean(shirt[:i+1])
            move_shirt.append(shirt_mean)
            sweater_mean=mean(sweater[:i+1])
            move_sweater.append(sweater_mean)
            towel_mean=mean(towel[:i+1])
            move_towel.append(towel_mean)
            t_shirt_mean=mean(t_shirt[:i+1])
            move_t_shirt.append(t_shirt_mean)
        data.append(move_pant)
        data.append(move_shirt)
        data.append(move_sweater)
        data.append(move_towel)
        data.append(move_t_shirt)
    return data

def plots(data):
    x=list(range(len(data[0])))
    data=pd.DataFrame({'x':x,'pant':data[0],'shirt':data[1],'sweater':data[2],'towel':data[3],'t-shirt':data[4]})
    fig=plt.figure()
    fig.add_subplot(111)
    subplot=plt.subplot()
    subplot.plot('x','pant',data=data,color='purple',label='pant')
    subplot.plot('x','shirt',data=data,color='red',label='shirt')
    subplot.plot('x','sweater',data=data,color='blue',label='sweater')
    subplot.plot('x','towel',data=data,color='green',label='towel')
    subplot.plot('x','t-shirt',data=data,color='orange',label='t-shirt')
    subplot.set_xticks(np.arange(0,197,20))
    subplot.set_yticks(np.arange(0,1,0.1))
    subplot.set_xlabel('Frame')
    subplot.set_ylabel('Probability')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.title('An Unseen Shirt Video Sequence Move Average')
    plt.savefig(fig_dir+'MS_Shape.png')
    plt.show()

transform=T.Compose([
    T.Resize((256,256)),
    T.ToTensor()
])
gartment_dataset=PictureDataset(file_path='./Database/Real/depth/',csv_path='./csv_clothes/real/depth/shirt_sequence.csv',idx_column=4,transforms=transform)
date_len=len(gartment_dataset)
indices=list(range(date_len))
sampler=BatchRandomSampler(indices,200)
dataloader=DataLoader(dataset=gartment_dataset,batch_size=200,sampler=sampler,num_workers=4)
alexnet_Path='./alexnet_model/alexnet_shape_dict.pth'
LSTM_Path='./lstm_model/lstm_shape_dict.pth'
alexnet=AlexNet()
alexnet.load_state_dict(torch.load(alexnet_Path))
features=Features(alexnet)
classifier=Classifier(alexnet)
features=features.to(device)
classifier=classifier.to(device)
model=LSTM()
model.load_state_dict(torch.load(LSTM_Path))
model=freeze(model)
model=model.to(device)
data=move_average(dataloader,features,classifier,model)
plots(data)
print('finished!')