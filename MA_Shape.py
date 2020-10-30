#pylint: skip-file
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

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic=True

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
softmax=nn.Softmax(dim=1)

fig_dir='./MA_Shape/'
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
    pairs=[]
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
            input_enc=features(input_frames)
            pred_enc=model(input_enc)
            pred_label=classifier(pred_enc)
            possibility=softmax(pred_label)
            pant.append(possibility[0][0].item())
            shirt.append(possibility[0][1].item())
            sweater.append(possibility[0][2].item())
            towel.append(possibility[0][3].item())
            t_shirt.append(possibility[0][4].item())
        if (t+2)%(item_item+1)==0:
            print ('['+str(target_lable.item())+']','[Batch: %d/%d][Duration: %f][pant: %f][shirt: %f][sweater: %f][towel: %f][t_shirt: %f]'%(t+1,len(dataloader),time.time()-start_time,mean(pant),mean(shirt),mean(sweater),mean(towel),mean(t_shirt)))
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
            tshirt_mean=mean(t_shirt[:i+1])
            move_t_shirt.append(tshirt_mean)
        movavg_pant=mean(move_pant)
        movavg_shirt=mean(move_shirt)
        movavg_sweater=mean(move_sweater)
        movavg_towel=mean(move_towel)
        movavg_t_shirt=mean(move_t_shirt)
        possibilities=[movavg_pant,movavg_shirt,movavg_sweater,movavg_towel,movavg_t_shirt]
        possibilities=torch.FloatTensor(possibilities)
        possibilities=possibilities
        _,max_index=torch.max(possibilities,0)
        pair={'Target':Labels[0],'Prediction':max_index}
        pairs.append(pair)
    return pairs

def statistics_method(pairs):
    test_critia=pairs
    zero_to_zero=0
    zero_to_one=0
    zero_to_two=0
    zero_to_three=0
    zero_to_four=0

    one_to_zero=0
    one_to_one=0
    one_to_two=0
    one_to_three=0
    one_to_four=0

    two_to_zero=0
    two_to_one=0
    two_to_two=0
    two_to_three=0
    two_to_four=0

    three_to_zero=0
    three_to_one=0
    three_to_two=0
    three_to_three=0
    three_to_four=0

    four_to_zero=0
    four_to_one=0
    four_to_two=0
    four_to_three=0
    four_to_four=0
    for x in range (len(test_critia)):
        if test_critia[x]["Target"]==0:
            if test_critia[x]["Prediction"]==0:
                zero_to_zero+=1
            if test_critia[x]["Prediction"]==1:
                zero_to_one+=1
            if test_critia[x]["Prediction"]==2:
                zero_to_two+=1
            if test_critia[x]["Prediction"]==3:
                zero_to_three+=1
            if test_critia[x]["Prediction"]==4:
                zero_to_four+=1
        if test_critia[x]["Target"]==1:
            if test_critia[x]["Prediction"]==0:
                one_to_zero+=1
            if test_critia[x]["Prediction"]==1:
                one_to_one+=1
            if test_critia[x]["Prediction"]==2:
                one_to_two+=1
            if test_critia[x]["Prediction"]==3:
                one_to_three+=1
            if test_critia[x]["Prediction"]==4:
                one_to_four+=1
        if test_critia[x]["Target"]==2:
            if test_critia[x]["Prediction"]==0:
                two_to_zero+=1
            if test_critia[x]["Prediction"]==1:
                two_to_one+=1
            if test_critia[x]["Prediction"]==2:
                two_to_two+=1
            if test_critia[x]["Prediction"]==3:
                two_to_three+=1
            if test_critia[x]["Prediction"]==4:
                two_to_four+=1
        if test_critia[x]["Target"]==3:
            if test_critia[x]["Prediction"]==0:
                three_to_zero+=1
            if test_critia[x]["Prediction"]==1:
                three_to_one+=1
            if test_critia[x]["Prediction"]==2:
                three_to_two+=1
            if test_critia[x]["Prediction"]==3:
                three_to_three+=1
            if test_critia[x]["Prediction"]==4:
                three_to_four+=1
        if test_critia[x]["Target"]==4:
            if test_critia[x]["Prediction"]==0:
                four_to_zero+=1
            if test_critia[x]["Prediction"]==1:
                four_to_one+=1
            if test_critia[x]["Prediction"]==2:
                four_to_two+=1
            if test_critia[x]["Prediction"]==3:
                four_to_three+=1
            if test_critia[x]["Prediction"]==4:
                four_to_four+=1
    zero=zero_to_zero+zero_to_one+zero_to_two+zero_to_three+zero_to_four
    one=one_to_zero+one_to_one+one_to_two+one_to_three+one_to_four
    two=two_to_zero+two_to_one+two_to_two+two_to_three+two_to_four
    three=three_to_zero+three_to_one+three_to_two+three_to_three+three_to_four
    four=four_to_zero+four_to_one+four_to_two+four_to_three+four_to_four

    z_z=zero_to_zero/zero
    z_o=zero_to_one/zero
    z_tw=zero_to_two/zero
    z_th=zero_to_three/zero
    z_f=zero_to_four/zero

    o_z=one_to_zero/one
    o_o=one_to_one/one
    o_tw=one_to_two/one
    o_th=one_to_three/one
    o_f=one_to_four/one

    tw_z=two_to_zero/two
    tw_o=two_to_one/two
    tw_tw=two_to_two/two
    tw_th=two_to_three/two
    tw_f=two_to_four/two

    th_z=three_to_zero/three
    th_o=three_to_one/three
    th_tw=three_to_two/three
    th_th=three_to_three/three
    th_f=three_to_four/three

    f_z=four_to_zero/four
    f_o=four_to_one/four
    f_tw=four_to_two/four
    f_th=four_to_three/four
    f_f=four_to_four/four

    z=[z_z*100,z_o*100,z_tw*100,z_th*100,z_f*100]
    o=[o_z*100,o_o*100,o_tw*100,z_th*100,z_f*100]
    tw=[tw_z*100,tw_o*100,tw_tw*100,tw_th*100,tw_f*100]
    th=[th_z*100,th_o*100,th_tw*100,th_th*100,th_f*100]
    f=[f_z*100,f_o*100,f_tw*100,f_th*100,f_f*100]

    plt.figure()
    total=[z,o,tw,th,f]
    total=np.array(total,dtype=np.float32).reshape(5,5)
    ax=sns.heatmap(total,annot=True,cmap="YlGnBu",vmin=0,vmax=100,fmt=".2f",xticklabels=False,yticklabels=False,cbar_kws={"label":"Classification Accuracy(%)[ResCla] Color Bar"})
    plt.title('Move Average Result (%)')
    plt.savefig(fig_dir+'MA_Shape.png')

transform=T.Compose([
    T.Resize((256,256)),
    T.ToTensor()
])
gartment_dataset=PictureDataset(file_path='./Database/Real/depth/',csv_path='./csv_clothes/real/depth/LOOD_25_full.csv',idx_column=4,transforms=transform)
date_len=len(gartment_dataset)
indices=list(range(date_len))
sampler=BatchRandomSampler(indices,200)
dataloader=DataLoader(dataset=gartment_dataset,batch_size=200,sampler=sampler,num_workers=4)
alexnet_Path='./alexnet_model/alexnet_shape_dict.pth'
LSTM_Path='./lstm_model/lstm_shape_dict.pth'
alexnet=AlexNet()
print (alexnet)
alexnet.load_state_dict(torch.load(alexnet_Path))
features=Features(alexnet)
classifier=Classifier(alexnet)
features=features.to(device)
classifier=classifier.to(device)
model=LSTM()
model.load_state_dict(torch.load(LSTM_Path))
model=freeze(model)
model=model.to(device)
pairs=move_average(dataloader,features,classifier,model)
statistics_method(pairs)
print('finished!')







