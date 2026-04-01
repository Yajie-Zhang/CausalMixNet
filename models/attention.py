import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from models import common



class NonLocalAttention(nn.Module):
    def __init__(self, channel=256, reduction=2, average=True,
                 conv=common.default_conv):
        super(NonLocalAttention, self).__init__()
        self.conv_match1 = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match2 = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = common.BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())

    def forward(self, input1,input2):
        x_embed_1 = self.conv_match1(input1)
        x_embed_2 = self.conv_match2(input2)
        x_assembly = self.conv_assembly(input2)

        N, C, H, W = x_embed_1.shape
        x_embed_1 = x_embed_1.permute(0, 2, 3, 1).view((N, H * W, C))
        x_embed_2 = x_embed_2.view(N, C, H * W)
        score = torch.matmul(x_embed_1, x_embed_2)
        score_f = F.softmax(score, dim=2)
        x_assembly =x_assembly.view(N, -1, H * W).permute(0, 2, 1)
        x_final = torch.matmul(score_f, x_assembly)
        return score_f,x_assembly,x_final.permute(0, 2, 1).view(N, -1, H, W)


# Module=NonLocalAttention(256)
# a=torch.rand(3,256,3,3)
# b=torch.rand(3,256,3,3)
# c=Module(a,b)
# print(c.shape)