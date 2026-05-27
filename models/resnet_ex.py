import torch
import torch.nn as nn
from torchvision.models import resnet,alexnet
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    def __init__(self, dim, hdim, r=8):
        super(Attention, self).__init__()
        self.dim = dim
        self.hdim = hdim
        self.r = r
        self.layer2 = nn.Sequential(
            nn.Linear(self.dim, self.r),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(self.r, self.hdim),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.layer2(input)
        x = self.layer3(x)
        # print(x.shape)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        # print('att.shape',x.shape)
        return x


class Local(nn.Module):
    def __init__(self, dim, hdim, r=16,num_att=6):
        super(Local, self).__init__()
        self.dim = dim
        self.hdim = hdim
        self.r = r
        self.num_att=num_att
        self.layer = Attention(dim, hdim, self.r)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, p):
        # print('p.shape',p.shape)
        a = self.avgpool(p).squeeze()
        if p.shape[0]==1:
            a=a.view(1,-1)
        a = self.layer(a)
        a = a.view(a.shape[0], a.shape[1], 1, 1)
        a = a * p
        a = a.sum(1).view(a.shape[0], -1)
        a = torch.softmax(a, dim=1)
        # print(a.shape)
        a = a.view(a.shape[0], 1, p.shape[2], p.shape[3])
        result=self.avgpool(a * p).squeeze()
        if p.shape[0]==1:
            result=result.view(1,-1)
        return result

class Fusion(nn.Module):
    def __init__(self, dim=512, hdim=512, r=8,num_att=16):
        super(Fusion, self).__init__()
        self.dim = dim
        self.hdim = hdim
        self.r = r
        self.num_att=num_att
        self.layer2 = nn.Sequential(
            nn.Linear(self.dim, self.r),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(self.r, self.hdim),
            nn.ReLU()
        )

    def forward(self,x1,x2):
        N=x1.shape[0]
        x1_=x1.view(-1,self.num_att,self.dim).view(-1,self.dim)
        x2_=x2.view(-1,self.num_att,self.dim).view(-1,self.dim)
        x1_=self.layer2(x1_)
        x1_=self.layer3(x1_)
        x2_=self.layer2(x2_)
        x2_=self.layer3(x2_)
        x1_=x1_.view(-1,self.num_att,self.dim)
        x2_=x2_.view(-1,self.num_att,self.dim)
        Fx1_=F.normalize(x1_,dim=2)
        Fx2_=F.normalize(x2_,dim=2)
        S=Fx1_.matmul(Fx2_.permute(0,2,1))
        return x1+(S.matmul(x2.view(-1,self.num_att,self.dim)).view(-1,(self.num_att)*self.dim))


class Model(nn.Module):
    def __init__(self, num_classes=1000,num_att=6,mode='single_cls',dim=512):
        super(Model, self).__init__()
        self.num_classes=num_classes
        self.num_att=num_att
        ResNet=resnet.resnet18(pretrained=True)
        self.conv1=ResNet.conv1
        self.bn1=ResNet.bn1
        self.relu=ResNet.relu
        self.maxpool=ResNet.maxpool
        self.layer1=ResNet.layer1
        self.layer2=ResNet.layer2
        self.layer3=ResNet.layer3
        self.layer4=ResNet.layer4
        self.avgpool=ResNet.avgpool
        self.dim=dim

        self.attibute1 = Local( self.dim,  self.dim)
        self.attibute2 = Local( self.dim,  self.dim)
        self.attibute3 = Local( self.dim,  self.dim)
        self.attibute4 = Local( self.dim,  self.dim)
        self.attibute5 = Local( self.dim,  self.dim)
        self.attibute6 = Local( self.dim,  self.dim)
        self.attibute7 = Local( self.dim,  self.dim)
        self.attibute8 = Local( self.dim,  self.dim)
        self.attibute9 = Local( self.dim,  self.dim)
        self.attibute10 = Local( self.dim,  self.dim)
        self.attibute11 = Local( self.dim,  self.dim)
        self.attibute12 = Local( self.dim,  self.dim)
        self.attibute13 = Local( self.dim,  self.dim)
        self.attibute14 = Local( self.dim,  self.dim)
        self.attibute15 = Local( self.dim,  self.dim)
        self.attibute16 = Local( self.dim,  self.dim)

        self.ex_non_attribute=Fusion(self.dim,self.dim,num_att=self.num_att)
        self.ex_dis_attribute=Fusion(self.dim,self.dim,num_att=self.num_att)

        self.cls = nn.Linear(self.dim*(num_att+1),num_classes)

        self.softmax = nn.Softmax(dim=1)
        self.feature_transfer=nn.Sequential(
            nn.Linear(self.dim*(num_att+1),512),
            nn.ReLU()
        )

    def compositional_exchange(self,att,att_bank):
        ex_index = ((torch.sign(torch.rand(att.shape[0], self.num_att, 1) - 0.5) + 1) / 2).to(att.device)
        ex_index = ex_index.repeat(1, 1, self.dim)
        ex_index = ex_index.view(att.shape[0], -1)
        att = ex_index * att + (1 - ex_index) * att_bank
        return att

    def cal_other_pro_att(self,attribute, label):
        device = label.device
        C = attribute.shape[0]
        N = label.shape[0]
        vs_label = 1 - label
        random_matrix = torch.rand(N, C).to(device)
        random_matrix = vs_label * random_matrix
        idx = torch.max(random_matrix, dim=1)[1]
        result = attribute[idx]
        return idx,result

    def random_select_att(self,pro,label_oh):
        selected_pro = (label_oh.matmul(pro)) / (torch.sum(label_oh, dim=1, keepdim=True) + 1e-5)
        pro_ver_idx,selected_pro_vers=self.cal_other_pro_att(pro,label_oh)
        return selected_pro,selected_pro_vers,pro_ver_idx

    def forward(self, x,pro_att=None,label_oh=None,N=100):
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_4 = self.layer4(x)
        # print(x_4.shape)

        att1 = self.attibute1(x_4)
        att2 = self.attibute2(x_4)
        att3 = self.attibute3(x_4)
        att4 = self.attibute4(x_4)
        att5 = self.attibute5(x_4)
        att6 = self.attibute6(x_4)

        # print('att6.shape',att6.shape)

        att = torch.cat((att1, att2, att3, att4, att5, att6), dim=1)
        # att = torch.cat((att1, att2, att3, att4, att5, att6,att7,att8,att9,att10,att11,att12,att13,att14,att15,att16), dim=1)
        x = self.avgpool(x_4)
        x = torch.flatten(x, 1)

        x_cat = torch.cat((x, att), dim=1)
        x_ft_tf = self.feature_transfer(x_cat)
        y = self.cls(x_cat)

        if pro_att is not None:
            MixY=[]
            MixFt=[]
            for i in range(N):
                pro_att=pro_att[:,(self.dim):].contiguous()
                selected_pro, selected_pro_vers, pro_ver_idx=self.random_select_att(pro_att,label_oh)
                att_mix=self.compositional_exchange(selected_pro,selected_pro_vers)
                att_mix= self.ex_dis_attribute(att, att_mix)
                att_mix= torch.cat((x, att_mix), dim=1)
                y_mix = self.cls(att_mix)
                ft_mix = self.feature_transfer(att_mix)
                MixY.append(y_mix)
                MixFt.append(ft_mix)
            MixY=torch.stack(MixY,dim=0).view(-1,y.shape[1])
            MixFt=torch.stack(MixFt,dim=0).view(-1,x_ft_tf.shape[1])
            return x_cat,x_ft_tf,y,MixFt,MixY
        return x_cat,x_ft_tf,y


# model_test=Model()
# img=torch.rand(2,3,256,256)
# x=model_test(img)


