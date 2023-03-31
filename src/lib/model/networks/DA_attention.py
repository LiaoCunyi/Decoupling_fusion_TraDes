from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb
import torchvision

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class DABasicBlock(nn.Module):
  def __init__(self, inplanes, planes, stride=1, downsample=None, fixed_block=False):
    super(DABasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.domain_attention = DomainAttention(planes, reduction=16, nclass_list=['Car','Pedestrian'], fixed_block=fixed_block)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    out = self.domain_attention(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)
    return out

class DomainAttention(nn.Module):
    def __init__(self, planes, reduction=16, nclass_list=None, fixed_block=False):
        super(DomainAttention, self).__init__()
        self.planes = planes

        self.n_datasets = len(nclass_list)

        self.fixed_block = fixed_block
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.SE_Layers = nn.ModuleList([SELayer(planes, reduction, with_sigmoid=False) for num_class in nclass_list])
        self.fc_1 = nn.Linear(planes, self.n_datasets)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, _, _ = x.size()

        if self.fixed_block:
            SELayers_Matrix = self.SE_Layers[0](x).view(b, c, 1, 1)
            SELayers_Matrix = self.sigmoid(SELayers_Matrix)
        else:
            weight = self.fc_1(self.avg_pool(x).view(b, c))
            weight = self.softmax(weight).view(b, self.n_datasets, 1)
            for i, SE_Layer in enumerate(self.SE_Layers):
                if i == 0:
                    SELayers_Matrix = SE_Layer(x).view(b, c, 1)
                else:
                    SELayers_Matrix = torch.cat((SELayers_Matrix, SE_Layer(x).view(b, c, 1)), 2)
            SELayers_Matrix = torch.matmul(SELayers_Matrix, weight).view(b, c, 1, 1)
            SELayers_Matrix = self.sigmoid(SELayers_Matrix)
        return x*SELayers_Matrix


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, with_sigmoid=True):
        super(SELayer, self).__init__()
        self.with_sigmoid = with_sigmoid
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if with_sigmoid:
            self.fc = nn.Sequential(
                    nn.Linear(channel, channel // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(channel // reduction, channel),
                    nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                    nn.Linear(channel, channel // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(channel // reduction, channel),
            )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y