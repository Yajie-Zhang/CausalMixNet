import torch
import torch.nn as nn
from torchvision.models import resnet,alexnet
from models.resnet_cbam import *

class Model(nn.Module):
    def __init__(self, num_classes=1000,mode='single_cls'):
        super(Model, self).__init__()
        self.num_classes=num_classes
        # ResNet=resnet50_cbam(pretrained=True)
        ResNet = resnet.resnet18(pretrained=True)
        self.conv1=ResNet.conv1
        self.bn1=ResNet.bn1
        self.relu=ResNet.relu
        self.maxpool=ResNet.maxpool
        self.layer1=ResNet.layer1
        self.layer2=ResNet.layer2
        self.layer3=ResNet.layer3
        self.layer4=ResNet.layer4
        self.avgpool=ResNet.avgpool
        self.head=nn.Linear(512,num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y=self.head(x)
        return y



if __name__ == '__main__':
    model=Model()
    a=torch.rand(2,3,224,224)
    b=model(a)
    print()


