#pylint:skip-file
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader
import time
from statistics import mean
from torch.autograd import Variable
import pandas as pd
from matplotlib import pyplot as plt
import os
from utils.Picture_Dataset import PictureDataset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models
import random
import seaborn as sns

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic=True

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
fig_dir='./alexnet_figure/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
model_dir='./alexnet_model/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

criterion=nn.CrossEntropyLoss()

def freeze(model,feature_extract=True,):
    if feature_extract:
        for params in model.parameters():
            params.requires_grad=False
    return model

class AlexNet (nn.Module):
    def __init__(self):
        super (AlexNet,self).__init__()
        self.models=models.alexnet(pretrained=True)
        self.models=freeze(self.models)
        self.models.features[0]=nn.Conv2d(1,64, kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        num_feature=self.models.classifier[1].in_features
        print ('num_feature:',num_feature)
        classiifer=nn.Sequential(
            nn.Linear(in_features=num_feature,out_features=9216,bias=True),
            nn.Linear(in_features=9216,out_features=512,bias=True),
            nn.Linear(in_features=512,out_features=3,bias=True)
        )
        self.models.classifier=classiifer
    
    def forward(self,x):
        x=self.models(x)
        return x
    
    def loss_fn(self,pred_label,target_index):
        loss=criterion(pred_label,target_index)
        return loss


def main(train_loader,vali_loader,test_loader):
    alexnet=AlexNet()
    alexnet=alexnet.to(device)
    params_to_update=[]
    print(alexnet)
    print('-----------------------')
    for name,params in alexnet.named_parameters():
        if params.requires_grad==True:
            params_to_update.append(params)
            print('\t',name)
    print('-----------------------')
    for fng in alexnet.models.features[0].weight:
        print(fng)
    print('------weight_check-----')
    optimiser=optim.SGD(params_to_update,lr=0.001,momentum=0.9)
    epoch=36
    train=[]
    acc_train=[]
    vali=[]
    acc_vali=[]
    epochs=[]
    for n in range(epoch):
        alexnet.train()
        start_time=time.time()
        item_time=int(len(train_loader)/10)
        train_loss=[]
        acc=0
        total=0
        for t, input_ in enumerate(train_loader):
            inputs=Variable(input_['Image'].to(device))
            labels=Variable(input_['Label'].to(device))
            optimiser.zero_grad()
            pred_label=alexnet(inputs)
            loss=alexnet.loss_fn(pred_label,labels)
            loss.backward()
            train_loss.append(loss.item())
            optimiser.step()
            _,pred_index=torch.max(pred_label,1)
            for i in range(len(pred_label)):
                if pred_index[i]==labels[i]:
                    acc+=1
                total+=1
            if (t+2)%(item_time+1)==0:
                print ('[Train][Epoch %d/%d][Batch: %d/%d][Duration: %f][CorssEntropy: %f]'%(n+1,epoch,t+1,len(train_loader),time.time()-start_time,mean(train_loss)))
                start_time=time.time()
        avg_loss=mean(train_loss)
        acc=100*(acc/total)
        train.append(avg_loss)
        acc_train.append(acc)
        print ('avg_loss(tria):',avg_loss)
        print ('acc(train):',acc)

        alexnet.eval()
        vali_loss=[]
        acc=0
        item_time=int(len(vali_loader)/10)
        start_time=time.time()
        total=0
        with torch.no_grad():
            for t,input_ in enumerate(vali_loader):
                inputs=Variable(input_['Image'].to(device))
                labels=Variable(input_['Label'].to(device))
                pred_label=alexnet(inputs)
                loss=alexnet.loss_fn(pred_label,labels)
                vali_loss.append(loss.item())
                _,pred_index=torch.max(pred_label,1)
                for i in range(len(pred_label)):
                    if pred_index[i]==labels[i]:
                        acc+=1
                    total+=1
                if (t+2)%(item_time+1)==0:
                    print ('[Vali][Epoch: %d/%d][Batch: %d/%d][Duration: %f][CrossEntropy: %f]'%(n+1,epoch,t+1,len(vali_loader),time.time()-start_time,mean(vali_loss)))
                    start_time=time.time()
            avg_loss=mean(vali_loss)
            acc=100*(acc/total)
            vali.append(avg_loss)
            acc_vali.append(acc)
            print ('avg_loss(vali):',avg_loss)
            print ('acc (vali):',acc)
            epochs.append(n+1)
    
    alexnet.eval()
    test_loss=[]
    acc=0
    item_time=int(len(test_loader)/10)
    start_time=time.time()
    total=0
    confusion_matrix=[]
    with torch.no_grad():
        for t,input_ in enumerate(test_loader):
            inputs=Variable(input_['Image'].to(device))
            labels=Variable(input_['Label'].to(device))
            pred_label=alexnet(inputs)
            loss=alexnet.loss_fn(pred_label,labels)
            test_loss.append(loss.item())
            _,pred_index=torch.max(pred_label,1)
            for i in range(len(pred_label)):
                if pred_index[i]==labels[i]:
                    acc+=1
                total+=1
                sample={'target_index':labels[i],'pred_index':pred_index[i]}
                confusion_matrix.append(sample)
            if (t+2)%(item_time+1)==0:
                print ('[Test][Batch: %d/%d][Duration: %f][CrossEntropy: %f]'%(t+1,len(test_loader),time.time()-start_time,mean(test_loss)))
                start_time=time.time()
        avg_loss=mean(test_loss)
        acc=100*(acc/total)
        print ('avg_loss(test):',avg_loss)
        print ('acc (test):',acc)
    torch.save(alexnet.state_dict(),model_dir+'./alexnet_weight_dict.pth')
    return train,acc_train,vali,acc_vali,epochs,confusion_matrix

transforms=T.Compose([T.Resize((256,256)),T.ToTensor()])
root='./data'

cloth_train=PictureDataset(file_path='./Database/Real/depth/',csv_path='./csv_clothes/real/depth/LOOD_75_full.csv',idx_column=6,transforms=transforms)
cloth_test=PictureDataset(file_path='./Database/Real/depth/',csv_path='./csv_clothes/real/depth/LOOD_25_full.csv',idx_column=6,transforms=transforms)
len_train=len(cloth_train)
indices=list(range(len_train))
np.random.shuffle(indices)
n_train=int(len_train*0.8)
train_indices=indices[:n_train]
vali_indices=indices[n_train:]
len_test=len(cloth_test)
test_indices=list(range(len_test))

train_sampler=SubsetRandomSampler(train_indices)
vali_sampler=SubsetRandomSampler(vali_indices)
test_sampler=SubsetRandomSampler(test_indices)

train_loader=DataLoader(dataset=cloth_train,batch_size=100,sampler=train_sampler,num_workers=4)
vali_loader=DataLoader(dataset=cloth_train,batch_size=100,sampler=vali_sampler,num_workers=4)
test_loader=DataLoader(dataset=cloth_test,batch_size=100,sampler=test_sampler,num_workers=4)


train,acc_train,vali,acc_vali,epochs,confusion_matrix=main(train_loader,vali_loader,test_loader)

zero_to_zero=0
zero_to_one=0
zero_to_two=0

one_to_zero=0
one_to_one=0
one_to_two=0

two_to_zero=0
two_to_one=0
two_to_two=0

for x in range (len(confusion_matrix)):
    if confusion_matrix[x]["target_index"]==0:
        if confusion_matrix[x]["pred_index"]==0:
            zero_to_zero+=1
        if confusion_matrix[x]["pred_index"]==1:
            zero_to_one+=1
        if confusion_matrix[x]["pred_index"]==2:
            zero_to_two+=1
        if confusion_matrix[x]["pred_index"]==3:
            zero_to_three+=1
        if confusion_matrix[x]["pred_index"]==4:
            zero_to_four+=1
    if confusion_matrix[x]["target_index"]==1:
        if confusion_matrix[x]["pred_index"]==0:
            one_to_zero+=1
        if confusion_matrix[x]["pred_index"]==1:
            one_to_one+=1
        if confusion_matrix[x]["pred_index"]==2:
            one_to_two+=1
        if confusion_matrix[x]["pred_index"]==3:
            one_to_three+=1
        if confusion_matrix[x]["pred_index"]==4:
            one_to_four+=1
    if confusion_matrix[x]["target_index"]==2:
        if confusion_matrix[x]["pred_index"]==0:
            two_to_zero+=1
        if confusion_matrix[x]["pred_index"]==1:
            two_to_one+=1
        if confusion_matrix[x]["pred_index"]==2:
            two_to_two+=1
        if confusion_matrix[x]["pred_index"]==3:
            two_to_three+=1
        if confusion_matrix[x]["pred_index"]==4:
            two_to_four+=1
    if confusion_matrix[x]["target_index"]==3:
        if confusion_matrix[x]["pred_index"]==0:
            three_to_zero+=1
        if confusion_matrix[x]["pred_index"]==1:
            three_to_one+=1
        if confusion_matrix[x]["pred_index"]==2:
            three_to_two+=1
        if confusion_matrix[x]["pred_index"]==3:
            three_to_three+=1
        if confusion_matrix[x]["pred_index"]==4:
            three_to_four+=1
    if confusion_matrix[x]["target_index"]==4:
        if confusion_matrix[x]["pred_index"]==0:
            four_to_zero+=1
        if confusion_matrix[x]["pred_index"]==1:
            four_to_one+=1
        if confusion_matrix[x]["pred_index"]==2:
            four_to_two+=1
        if confusion_matrix[x]["pred_index"]==3:
            four_to_three+=1
        if confusion_matrix[x]["pred_index"]==4:
            four_to_four+=1

zero=zero_to_zero+zero_to_one+zero_to_two+1
one=one_to_zero+one_to_one+one_to_two+1
two=two_to_zero+two_to_one+two_to_two+1
    
print (zero)
print (one)
print (two)

z_z=zero_to_zero/zero
z_o=zero_to_one/zero
z_tw=zero_to_two/zero

o_z=one_to_zero/one
o_o=one_to_one/one
o_tw=one_to_two/one

tw_z=two_to_zero/two
tw_o=two_to_one/two
tw_tw=two_to_two/two

z=[z_z*100,z_o*100,z_tw*100]
o=[o_z*100,o_o*100,o_tw*100]
tw=[tw_z*100,tw_o*100,tw_tw*100]

total=[z,o,tw]
total=np.array(total,dtype=np.float32).reshape(3,3)

ax=sns.heatmap(total,annot=True,cmap="YlGnBu",vmin=0,vmax=100,fmt=".2f",xticklabels=False,yticklabels=False,cbar_kws={"label":"Classification Accuracy(%)[ResCla] Color Bar"})
plt.title("Prediction Result")
plt.savefig(os.path.join(fig_dir,"confusion_matrix_%f.png"%time.time()))

df=pd.DataFrame({'x':epochs,'train':train,'acc_train':acc_train,'vali':vali,'acc_vali':acc_vali})
plt.figure()
subplot=plt.subplot()
subplot.plot('x','train',color='red',data=df,label='train')
subplot.plot('x','vali',color='blue',data=df,label='vali')
plt.legend(loc='upper right')
subplot.set_xlabel('Epoch')
subplot.set_ylabel('CrossEntropy')
subplot2=subplot.twinx()
subplot2.plot('x','acc_train',color='green',data=df,label='acc_train',linestyle='--')
subplot2.plot('x','acc_vali',color='yellow',data=df,label='acc_vali',linestyle='--')
plt.grid(True)
subplot2.set_ylabel('Accuracy (%)')
plt.legend(loc='upper left')
plt.title('MNIST Classification')
plt.savefig(fig_dir+'MNIST_graph_%f.png'%time.time())

plt.show()    