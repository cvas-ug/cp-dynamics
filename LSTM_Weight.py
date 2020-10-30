#pylint:skip-file
import torch
import torch.nn as nn
import os
import numpy as np
import random
import seaborn as sns
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import time
from statistics import mean
from torchvision.transforms import transforms as T
from utils.Picture_Dataset import PictureDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from utils.BatchRandomSampler import BatchRandomSampler
from torchvision import models
import pandas as pd


torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic=True

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
criterion=nn.CrossEntropyLoss()
mse=nn.MSELoss(reduction='sum')

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
            nn.Linear(in_features=512,out_features=3,bias=True)
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

fig_dir='./lstm_figure/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
model_dir='./lstm_model/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

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
    
def train(features,classifier,train_loader,vali_loader,test_loader):
    print (features)
    Epoch=36
    model=LSTM()
    model=model.to(device)
    param_update=[]
    print ('---------------------------')
    for name,param in model.named_parameters():
        if param.requires_grad==True:
            print('\t',name)
            param_update.append(param)
    print ('----------------------------')
    for fng in model.lstm.weight_ih_l0:
        print (fng)
    print ('--------weight check--------')
    optimiser=optim.Adam(param_update,lr=0.0001)
    scheduler=optim.lr_scheduler.StepLR(optimiser,step_size=15,gamma=0.1,last_epoch=-1)
    train_mse=[]
    train_criterion=[]
    vali_mse=[]
    vali_criterion=[]
    test_mse=[]
    test_criterion=[]
    train_acc=[]
    vali_acc=[]
    test_acc=[]
    epochs=[]
    for n in range(Epoch):
        total=0
        criterion_loss=[]
        mse_loss=[]
        model.train()
        acc=0
        item_time=int(len(train_loader)/10)
        start_time=time.time()
        for t,input_ in enumerate(train_loader):
            inputs=Variable(input_['Image']).to(device)
            labels=Variable(input_['Label'].to(device))
            for i in range (len(inputs)-3):
                optimiser.zero_grad()
                time_step=inputs[i:i+3]
                label=labels[i+3:i+4]
                groud_truth=inputs[i+3:i+4]
                feature=features(time_step)
                pred_ft=model(feature)
                pred_label=classifier(pred_ft)
                loss1=criterion(pred_label,label)
                _,pred_index=torch.max(pred_label,1)
                if pred_index==label:
                    acc+=1
                total+=1
                groud_ft=features(groud_truth)
                loss2=mse(pred_ft,groud_ft.view(1,-1))
                loss=loss2+1000*loss1
                loss.backward()
                optimiser.step()
                mse_loss.append(loss2.item())
                criterion_loss.append(loss1.item())
            if (t+2)%(item_time+1)==0:
                print('[Train][Epoch: %d/%d][Batch: %d/%d][Duration: %fs][CrossEntropy: %f][MSE: %f]'%(n+1,Epoch,t+1,len(train_loader),time.time()-start_time,mean(criterion_loss),mean(mse_loss)))
                start_time=time.time()
        avg_criterion=mean(criterion_loss)
        avg_mse=mean(mse_loss)
        acc=acc/total*100
        train_criterion.append(avg_criterion)
        train_mse.append(avg_mse)
        train_acc.append(acc)
        print ('avg_criterion (train):',avg_criterion)
        print ('avg_mse (train):',avg_mse)
        print ('acc (train):',acc)
        scheduler.step()

        start_time=time.time()
        criterion_loss=[]
        mse_loss=[]
        acc=0
        total=0
        model.eval()
        item_time=int(len(vali_loader)/10)
        with torch.no_grad():
            for t,input_ in enumerate(vali_loader):
                inputs=Variable(input_['Image']).to(device)
                labels=Variable(input_['Label']).to(device)
                for i in range(len(inputs)-3):
                    time_step=inputs[i:i+3]
                    label=labels[i+3:i+4]
                    groud_truth=inputs[i+3:i+4]
                    feature=features(time_step)
                    pred_ft=model(feature)
                    pred_label=classifier(pred_ft)
                    loss1=criterion(pred_label,label)
                    _,pred_index=torch.max(pred_label,1)
                    if pred_index==label:
                        acc+=1
                    total+=1
                    groud_ft=features(groud_truth)
                    loss2=mse(groud_ft.view(1,-1),pred_ft)
                    mse_loss.append(loss2.item())
                    criterion_loss.append(loss1.item())
                if (t+2)%(item_time+1)==0:
                    print('[Vali][Epoch: %d/%d][Batch:]%d/%d][Duration: %fs][CrossEntropy: %f][MSE: %f]'%(n+1,Epoch,t+1,len(vali_loader),time.time()-start_time,mean(criterion_loss),mean(mse_loss)))
                    start_time=time.time()
            avg_criterion=mean(criterion_loss)
            avg_mse=mean(mse_loss)
            acc=acc/total*100
            vali_criterion.append(avg_criterion)
            vali_mse.append(avg_mse)
            vali_acc.append(acc)
            print ('avg_criterion (vali):',avg_criterion)
            print ('avg_mse (vali):',avg_mse)
            print ('acc (vali):',acc)
            epochs.append(n+1)
            
    start_time=time.time()
    criterion_loss=[]
    mse_loss=[]
    total=0
    acc=0
    model.eval()
    item_time=int(len(test_loader)/10)
    confusion_matrix=[]
    with torch.no_grad():
        for t, input_ in enumerate(test_loader):
            inputs=Variable(input_['Image']).to(device)
            labels=Variable(input_['Label']).to(device)
            for i in range (len(inputs)-3):
                time_step=inputs[i:i+3]
                label=labels[i+3:i+4]
                groud_truth=inputs[i+3:i+4]
                feature=features(time_step)
                pred_ft=model(feature)
                pred_label=classifier(pred_ft)
                loss1=criterion(pred_label,label)
                _,pred_index=torch.max(pred_label,1)
                if pred_index==label:
                    acc+=1
                total+=1
                sample={'target_index':label,'pred_index':pred_index}
                confusion_matrix.append(sample)
                groud_ft=features(groud_truth)
                loss2=mse(groud_ft.view(1,-1),pred_ft)
                criterion_loss.append(loss1.item())
                mse_loss.append(loss2.item())
            if (t+2)%(item_time+1)==0:
                print ('[Test][Batch: %d/%d][Duration: %fs][CrossEntropy: %f][MSE: %f]'%(t+1,len(test_loader),time.time()-start_time,mean(criterion_loss),mean(mse_loss)))
                start_time=time.time()
        avg_criterion=mean(criterion_loss)
        avg_mse=mean(mse_loss)
        acc=acc/total*100
        test_mse.append(avg_mse)
        test_criterion.append(avg_criterion)
        test_acc.append(acc)
        print ('avg_criterion (test):',avg_criterion)
        print ('avg_mse (test):',avg_mse)
        print ('acc (test):',acc)
        torch.save(model.state_dict(),model_dir+'lstm_weight_dict.pth')
        return train_mse,train_criterion,train_acc,vali_mse,vali_criterion,vali_acc,confusion_matrix,epochs

batch_size=200
transforms=T.Compose([T.Resize((256,256)),T.ToTensor()])
train_dataset=PictureDataset(file_path='./Database/Real/depth/',csv_path='./csv_clothes/real/depth/LOOD_75_full.csv',idx_column=6,transforms=transforms)
test_dataset=PictureDataset(file_path='./Database/Real/depth/',csv_path='./csv_clothes/real/depth/LOOD_25_full.csv',idx_column=6,transforms=transforms)
len_data=len(train_dataset)
split_indices=list(range(len_data))
indices=list(range(len_data))
rand_num=torch.randperm(int((len_data)/batch_size))
for i in range (len(rand_num)):
    t=rand_num[i]
    rand_indices=split_indices[t*batch_size:(t+1)*batch_size]
    indices[i*batch_size:(i+1)*batch_size]=rand_indices
n_train=int(len_data*0.8)
train_indices=indices[:n_train]
vali_indices=indices[n_train:]
len_test=len(test_dataset)
test_indices=list(range(len_test))

train_sampler=BatchRandomSampler(train_indices,batch_size)
vali_sampler=BatchRandomSampler(vali_indices,batch_size)
test_sampler=BatchRandomSampler(test_indices,batch_size)

train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,sampler=train_sampler,num_workers=4)
vali_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,sampler=vali_sampler,num_workers=4)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,sampler=test_sampler,num_workers=4)

alexnet=AlexNet()
alexnet.load_state_dict(torch.load('./alexnet_model/alexnet_weight_dict.pth'))
features=Features(alexnet)
classifier=Classifier(alexnet)
features=features.to(device)
classifier=classifier.to(device)

train_mse,train_criterion,train_acc,vali_mse,vali_criterion,vali_acc,confusion_matrix,epochs=train(features,classifier,train_loader,vali_loader,test_loader)

zero_to_zero=0
zero_to_one=0
zero_to_two=0

one_to_zero=0
one_to_one=0
one_to_two=0

two_to_zero=0
two_to_one=0
two_to_two=0

three_to_zero=0
three_to_one=0
three_to_two=0

four_to_zero=0
four_to_one=0
four_to_two=0

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
plt.savefig(os.path.join(fig_dir,"cp_confusion_matrix_10_loss1_loss2%f.png"%time.time()))

df=pd.DataFrame({'x':epochs,'train_mse':train_mse,'train_criterion':train_criterion,'train_acc':train_acc,'vali_mse':vali_mse,'vali_criterion':vali_criterion,'vali_acc':vali_acc})

plt.figure()
subplot=plt.subplot()
subplot.plot('x','train_criterion',color='red',data=df,label='train_criterion')
subplot.plot('x','vali_criterion',color='blue',data=df,label='vali_criterion')
plt.legend(loc='upper right')
subplot.set_xlabel('Epoch')
subplot.set_ylabel('CrossEntropy')
subplot2=subplot.twinx()
subplot2.plot('x','train_acc',color='green',data=df,label='train_acc',linestyle='--')
subplot2.plot('x','vali_acc',color='yellow',data=df,label='vali_acc',linestyle='--')
plt.grid(True)
subplot2.set_ylabel('Accuracy (%)')
plt.legend(loc='upper left')
plt.title('CP Classification')
plt.savefig(fig_dir+'CP_classification_graph_%f.png'%time.time())

plt.figure()
subplot=plt.subplot()
subplot.plot('x','train_mse',color='red',data=df,label='train_mse')
subplot.plot('x','vali_mse',color='blue',data=df,label='vali_mse')
plt.grid(True)
plt.legend(loc='upper right')
subplot.set_xlabel('Epoch')
subplot.set_ylabel('MSE Loss')
plt.title('CP MSE')
plt.savefig(fig_dir+'CP_MSE_%f.png'%time.time())






