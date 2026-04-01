import torch
import torch.nn as nn
from torchvision.models import resnet,alexnet
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from models.attention import NonLocalAttention


class Model(nn.Module):
    def __init__(self, num_classes=1000,mode='single_cls',dim=512,K=100):
        super(Model, self).__init__()
        self.num_classes=num_classes
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
        self.CrossAtt=NonLocalAttention(256)
        self.dim=dim
        self.cls = nn.Linear(self.dim,num_classes)
        self.K=K

    def select_posi_pro(self,pro,label_oh):
        C, D, H, W = pro.shape
        max_id=torch.max(label_oh,dim=1)[1]
        result=pro[max_id]
        return result

    def select_nega_pro(self,pro,label_oh):
        rand_matrix=torch.rand_like(label_oh)
        rand_matrix=rand_matrix*(1-label_oh)
        max_idx=torch.max(rand_matrix,dim=1)[1]
        result=pro[max_idx]
        return result

    def compositional_exchange(self,posi,nega):
        K=self.K
        B,D,H,W=posi.shape
        random_matrix=torch.rand(B,D).to(posi.device)
        random_matrix_sort=torch.sort(random_matrix,dim=1)[0]
        random_matrix_sort=(random_matrix_sort[:,K]).view(-1,1)
        random_matrix[random_matrix<random_matrix_sort]=1
        random_matrix[random_matrix<1]=0
        random_matrix=random_matrix.view(B,D,1,1)
        return posi*(1-random_matrix)+random_matrix*nega

    def patch_set(self):
        patch_index_set = []
        for i in range(4):
            for j in range(4):
                patch_index = torch.zeros(1, 16, 16)
                h1=(i*4)
                h2=(i+1)*4
                w1=(j*4)
                w2=(j+1)*4
                patch_index[0,h1:h2,w1:w2] = 1
                # a=patch_index.numpy()
                patch_index_set.append(patch_index)
        patch_index_set=torch.stack(patch_index_set)
        return patch_index_set.view(16,1,-1)   #16,1,16*16

    def cal_few_last_layers(self,x):
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.cls(x)
        return x,y

    def forward(self, x,pro_att_ori=None,label_oh=None,N=3):
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)
        score,assembly,x=self.CrossAtt(x_3,x_3) #score.shape=B,HW,HW,  assembly.shape=B,HW,D
        x = self.layer4(x+x_3)
        x_4=x.clone()
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.cls(x)
        if label_oh is not None:
            mask = (self.patch_set()).to(x.device)
            masked_x3_set = []
            # mask_x3_set=[]
            for i in range(x.shape[0]):
                cur_score = (score[i]).view(1, 16 * 16, 16 * 16).repeat(16, 1, 1)
                cur_masked = cur_score * (1 - mask)
                # cur_mask=cur_score*mask
                cur_masked = torch.softmax(cur_masked, dim=2)
                # cur_mask=torch.softmax(cur_mask,dim=2)
                cur_assembly = (assembly[i]).view(1, 16 * 16, 256).repeat(16, 1, 1)
                cur_masked = (cur_masked.matmul(cur_assembly)).permute(0, 2, 1).view(16, 256, 16, 16)
                # cur_mask=(cur_mask.matmul(cur_assembly)).permute(0,2,1).view(16,256,16,16)
                masked_x3_set.append(cur_masked)
                # mask_x3_set.append(cur_mask)
            masked_x3_set = torch.stack(masked_x3_set)  # shape=B,16,256,16,16
            masked_x3_set = masked_x3_set.view(-1, 256, 16, 16)
            # mask_x3_set=torch.stack(mask_x3_set)
            # mask_x3_set=mask_x3_set.view(-1,256,16,16)
            masked_x, masked_y = self.cal_few_last_layers(masked_x3_set+x_3.view(x.shape[0],1,256,16,16).repeat(1,16,1,1,1).view(-1,256,16,16))
            # mask_x,mask_y=self.cal_few_last_layers(mask_x3_set)
            label_repeat = label_oh.view(label_oh.shape[0], 1, -1).repeat(1, 16, 1).view(-1, label_oh.shape[1])
            masked_x_view=masked_x.view(label_oh.shape[0],16,512)
            x_view=x.view(label_oh.shape[0],1,512).repeat(1,16,1)
            x_sim=((masked_x_view-x_view)**2).sum(2)
            # print('x_sim',x_sim)

            sort_sim = torch.sort(-x_sim, dim=1)[1]
            min_patch_index = sort_sim[:, :N]
            max_patch_index=sort_sim[:,(-N):]

            pair_index=torch.rand(x.shape[0],x.shape[0]).to(x.device)
            pair_index=torch.sort(pair_index,dim=1)[1]
            pair_index=pair_index[:,0]

            sim_pair_index = torch.rand(x.shape[0], x.shape[0]).to(x.device)
            S=label_oh.matmul(label_oh.T)
            sim_pair_index=S*sim_pair_index
            sim_pair_index = torch.sort(-sim_pair_index, dim=1)[1]
            sim_pair_index = sim_pair_index[:, 0]


            x_3_clone=x_3.clone()
            pair_x3=x_3_clone[pair_index]
            sim_pair_x3=x_3_clone[sim_pair_index]
            #
            ex_x_3=torch.zeros_like(x_3)
            ex_min_patch_index=min_patch_index[pair_index]

            sim_ex_x_3=torch.zeros_like(x_3)
            ex_max_patch_index=max_patch_index[sim_pair_index]

            for i in range(x.shape[0]):

                #change dis similar
                x3_mask=min_patch_index[i]
                rand_idx=torch.rand(1,N)
                rand_idx=torch.max(rand_idx,dim=1)[1]
                x3_mask=(mask[x3_mask[rand_idx]]).view(-1)   #16*16,1
                x3_mask=torch.where(x3_mask==1)[0]

                ex_x3_mask=ex_min_patch_index[i]
                rand_idx = torch.rand(1, N)
                rand_idx = torch.max(rand_idx, dim=1)[1]
                ex_x3_mask = (mask[ex_x3_mask[rand_idx]]).view(-1)
                ex_x3_mask=torch.where(ex_x3_mask==1)[0]


                cur_x3=(x_3_clone[i] ).view(256,16*16).permute(1,0)  #16*16,256
                cur_ex_x3=pair_x3[i].view(256,16*16).permute(1,0)

                cur_x3[x3_mask]=cur_ex_x3[ex_x3_mask]
                cur_x3=cur_ex_x3.permute(1,0).view(256,16,16)
                ex_x_3[i]=cur_x3

                # change similar
                x3_mask = max_patch_index[i]
                rand_idx = torch.rand(1, N)
                rand_idx = torch.max(rand_idx, dim=1)[1]
                x3_mask = (mask[x3_mask[rand_idx]]).view(-1)  # 16*16,1
                x3_mask = torch.where(x3_mask == 1)[0]

                ex_x3_mask = ex_max_patch_index[i]
                rand_idx = torch.rand(1, N)
                rand_idx = torch.max(rand_idx, dim=1)[1]
                ex_x3_mask = (mask[ex_x3_mask[rand_idx]]).view(-1)
                ex_x3_mask = torch.where(ex_x3_mask == 1)[0]

                cur_x3 = (x_3_clone[i]).view(256, 16 * 16).permute(1, 0)  # 16*16,256
                cur_ex_x3 = sim_pair_x3[i].view(256, 16 * 16).permute(1, 0)

                cur_x3[x3_mask] = cur_ex_x3[ex_x3_mask]
                cur_x3 = cur_ex_x3.permute(1, 0).view(256, 16, 16)
                sim_ex_x_3[i] = cur_x3
            # ex_x_3=ex_x_3
            _,_,ex_x_3=self.CrossAtt(x_3,ex_x_3)
            ex_x,ex_y=self.cal_few_last_layers(ex_x_3+x_3)

            # _, _, sim_ex_x_3 = self.CrossAtt(x_3, sim_ex_x_3)

            _, _, sim_ex_x_3 = self.CrossAtt(sim_ex_x_3,x_3)
            sim_ex_x, sim_ex_y = self.cal_few_last_layers(sim_ex_x_3 + x_3)
            return x_sim,x,y,sim_ex_x, sim_ex_y,ex_x,ex_y
        return x_4,x,y

class Model_NIH(nn.Module):
    def __init__(self, num_classes=1000,mode='single_cls',dim=512,K=100):
        super(Model_NIH, self).__init__()
        self.num_classes=num_classes
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
        self.CrossAtt=NonLocalAttention(256)
        self.dim=dim
        # self.cls=[]
        self.cls = nn.ModuleList([nn.Linear(self.dim,2) for i in range(self.num_classes)])
        self.K=K

    def select_posi_pro(self,pro,label_oh):
        C, D, H, W = pro.shape
        max_id=torch.max(label_oh,dim=1)[1]
        result=pro[max_id]
        return result

    def select_nega_pro(self,pro,label_oh):
        rand_matrix=torch.rand_like(label_oh)
        rand_matrix=rand_matrix*(1-label_oh)
        max_idx=torch.max(rand_matrix,dim=1)[1]
        result=pro[max_idx]
        return result

    def compositional_exchange(self,posi,nega):
        K=self.K
        B,D,H,W=posi.shape
        random_matrix=torch.rand(B,D).to(posi.device)
        random_matrix_sort=torch.sort(random_matrix,dim=1)[0]
        random_matrix_sort=(random_matrix_sort[:,K]).view(-1,1)
        random_matrix[random_matrix<random_matrix_sort]=1
        random_matrix[random_matrix<1]=0
        random_matrix=random_matrix.view(B,D,1,1)
        return posi*(1-random_matrix)+random_matrix*nega

    def patch_set(self):
        patch_index_set = []
        for i in range(4):
            for j in range(4):
                patch_index = torch.zeros(1, 16, 16)
                h1=(i*4)
                h2=(i+1)*4
                w1=(j*4)
                w2=(j+1)*4
                patch_index[0,h1:h2,w1:w2] = 1
                # a=patch_index.numpy()
                patch_index_set.append(patch_index)
        patch_index_set=torch.stack(patch_index_set)
        return patch_index_set.view(16,1,-1)   #16,1,16*16

    def cal_few_last_layers(self,x):
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y=[]
        for i in range(self.num_classes):
            cur_y=self.cls[i](x)
            y.append(cur_y)
        return x,y

    def forward(self, x,pro_att_ori=None,label_oh=None,N=3):
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)
        score,assembly,x=self.CrossAtt(x_3,x_3) #score.shape=B,HW,HW,  assembly.shape=B,HW,D
        x = self.layer4(x+x_3)
        x_4=x.clone()
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y=[]
        for i in range(self.num_classes):
            cur_y=self.cls[i](x)
            y.append(cur_y)
        if label_oh is not None:
            mask = (self.patch_set()).to(x.device)
            masked_x3_set = []
            # mask_x3_set=[]
            for i in range(x.shape[0]):
                cur_score = (score[i]).view(1, 16 * 16, 16 * 16).repeat(16, 1, 1)
                cur_masked = cur_score * (1 - mask)
                # cur_mask=cur_score*mask
                cur_masked = torch.softmax(cur_masked, dim=2)
                # cur_mask=torch.softmax(cur_mask,dim=2)
                cur_assembly = (assembly[i]).view(1, 16 * 16, 256).repeat(16, 1, 1)
                cur_masked = (cur_masked.matmul(cur_assembly)).permute(0, 2, 1).view(16, 256, 16, 16)
                # cur_mask=(cur_mask.matmul(cur_assembly)).permute(0,2,1).view(16,256,16,16)
                masked_x3_set.append(cur_masked)
                # mask_x3_set.append(cur_mask)
            masked_x3_set = torch.stack(masked_x3_set)  # shape=B,16,256,16,16
            masked_x3_set = masked_x3_set.view(-1, 256, 16, 16)
            # mask_x3_set=torch.stack(mask_x3_set)
            # mask_x3_set=mask_x3_set.view(-1,256,16,16)
            masked_x, masked_y = self.cal_few_last_layers(masked_x3_set+x_3.view(x.shape[0],1,256,16,16).repeat(1,16,1,1,1).view(-1,256,16,16))
            # mask_x,mask_y=self.cal_few_last_layers(mask_x3_set)
            label_repeat = label_oh.view(label_oh.shape[0], 1, -1).repeat(1, 16, 1).view(-1, label_oh.shape[1])
            masked_x_view=masked_x.view(label_oh.shape[0],16,512)
            x_view=x.view(label_oh.shape[0],1,512).repeat(1,16,1)
            x_sim=((masked_x_view-x_view)**2).sum(2)
            # print('x_sim',x_sim)

            sort_sim = torch.sort(-x_sim, dim=1)[1]
            min_patch_index = sort_sim[:, :N]
            max_patch_index=sort_sim[:,(-N):]

            pair_index=torch.rand(x.shape[0],x.shape[0]).to(x.device)
            pair_index=torch.sort(pair_index,dim=1)[1]
            pair_index=pair_index[:,0]

            sim_pair_index = torch.rand(x.shape[0], x.shape[0]).to(x.device)
            S=label_oh.matmul(label_oh.T)
            S[S>0]=1
            sim_pair_index=S*sim_pair_index
            sim_pair_index = torch.sort(-sim_pair_index, dim=1)[1]
            sim_pair_index = sim_pair_index[:, 0]


            x_3_clone=x_3.clone()
            pair_x3=x_3_clone[pair_index]
            sim_pair_x3=x_3_clone[sim_pair_index]
            #
            ex_x_3=torch.zeros_like(x_3)
            ex_min_patch_index=min_patch_index[pair_index]

            sim_ex_x_3=torch.zeros_like(x_3)
            ex_max_patch_index=max_patch_index[sim_pair_index]

            for i in range(x.shape[0]):

                #change dis similar
                x3_mask=min_patch_index[i]
                rand_idx=torch.rand(1,N)
                rand_idx=torch.max(rand_idx,dim=1)[1]
                x3_mask=(mask[x3_mask[rand_idx]]).view(-1)   #16*16,1
                x3_mask=torch.where(x3_mask==1)[0]

                ex_x3_mask=ex_min_patch_index[i]
                rand_idx = torch.rand(1, N)
                rand_idx = torch.max(rand_idx, dim=1)[1]
                ex_x3_mask = (mask[ex_x3_mask[rand_idx]]).view(-1)
                ex_x3_mask=torch.where(ex_x3_mask==1)[0]


                cur_x3=(x_3_clone[i] ).view(256,16*16).permute(1,0)  #16*16,256
                cur_ex_x3=pair_x3[i].view(256,16*16).permute(1,0)

                cur_x3[x3_mask]=cur_ex_x3[ex_x3_mask]
                cur_x3=cur_ex_x3.permute(1,0).view(256,16,16)
                ex_x_3[i]=cur_x3

                # change similar
                x3_mask = max_patch_index[i]
                rand_idx = torch.rand(1, N)
                rand_idx = torch.max(rand_idx, dim=1)[1]
                x3_mask = (mask[x3_mask[rand_idx]]).view(-1)  # 16*16,1
                x3_mask = torch.where(x3_mask == 1)[0]

                ex_x3_mask = ex_max_patch_index[i]
                rand_idx = torch.rand(1, N)
                rand_idx = torch.max(rand_idx, dim=1)[1]
                ex_x3_mask = (mask[ex_x3_mask[rand_idx]]).view(-1)
                ex_x3_mask = torch.where(ex_x3_mask == 1)[0]

                cur_x3 = (x_3_clone[i]).view(256, 16 * 16).permute(1, 0)  # 16*16,256
                cur_ex_x3 = sim_pair_x3[i].view(256, 16 * 16).permute(1, 0)

                cur_x3[x3_mask] = cur_ex_x3[ex_x3_mask]
                cur_x3 = cur_ex_x3.permute(1, 0).view(256, 16, 16)
                sim_ex_x_3[i] = cur_x3
            # ex_x_3=ex_x_3
            _,_,ex_x_3=self.CrossAtt(x_3,ex_x_3)
            ex_x,ex_y=self.cal_few_last_layers(ex_x_3+x_3)

            # _, _, sim_ex_x_3 = self.CrossAtt(x_3, sim_ex_x_3)

            _, _, sim_ex_x_3 = self.CrossAtt(sim_ex_x_3,x_3)
            sim_ex_x, sim_ex_y = self.cal_few_last_layers(sim_ex_x_3 + x_3)
            return x_sim,x,y,sim_ex_x, sim_ex_y,ex_x,ex_y
        return x_4,x,y

# model_test=Model()
# img=torch.rand(100,3,256,256)
# label=torch.rand(100,5)
# label_long=torch.max(label,dim=1)[1]
# label_long=label_long.view(-1)
# label_oh = torch.tensor(np.eye(5, dtype=np.uint8)[label_long.cpu().numpy()]).float().to(
#                     label.device)
# _,_,_,_,_,_=model_test(img,label_oh=label_oh)
# print(x.shape)


