
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import torchvision
import sys
import numpy as n
import copy
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
from Harnet import HarDNetFeatureExtractor

class conv1x1(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(conv1x1, self).__init__()
        self.bn = nn.BatchNorm2d(out_planes)
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,padding=0, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 

    def forward(self, feature):
        output = self.conv(feature)
        output = self.bn(output)
        output = self.avgpool(output)
        output = output.squeeze(-1).squeeze(-1)
        return output
    

class DGMnet(nn.Module):
    def __init__(self):
        super(DGMnet, self).__init__()
        # 加載預訓練的 HarDNet
        self.base_model = HarDNetFeatureExtractor(arch=68, pretrained=True)
        # 根據 HarDNet 的輸出設置特徵金字塔
        self.downsample=conv1x1(654,128)
        self.fc = nn.Sequential(
            nn.Linear(128,128//2),
            nn.LeakyReLU(),
            nn.Linear(128//2, 2),
        )
    
        self.avgpooling = nn.AdaptiveAvgPool2d(1)
        self.loss_L1 = nn.L1Loss()
    def forward(self, x_in):
        features = self.base_model(x_in['origin_face'])
        last_feature=features[3]
        last_feature=self.downsample(last_feature)
        # 計算 gaze (目光方向)
        gaze = self.fc(last_feature)
        #print(gaze.shape)
        return gaze

        

if __name__ == '__main__':
    m = DGMnet().cuda()
    feature = {
        "origin_face": torch.zeros(10, 3, 224, 224).cuda(),
        "left_eye": torch.zeros(10, 3, 112, 112).cuda(),
        "right_eye": torch.zeros(10, 3, 112, 112).cuda(),
        "gaze_origin": torch.zeros(10, 3).cuda()
    }
    output = m(feature)
