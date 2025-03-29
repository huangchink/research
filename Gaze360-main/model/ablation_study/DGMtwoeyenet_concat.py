
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
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=8):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
class EyeImageModel(nn.Module):
    def __init__(self):
        super(EyeImageModel, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.features1_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, dilation=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CBAM(64),
            nn.Conv2d(64, 64, 3, 1, 1, dilation=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            CBAM(64),
            nn.Conv2d(64, 128, 3, 1, 1, dilation=(3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            CBAM(128),
            nn.Conv2d(128, 48, 3, 1, 1, dilation=(2, 2)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

        self.features1_2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            CBAM(48),
            nn.Conv2d(48, 64, 1, 1, dilation=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CBAM(64),
            nn.Conv2d(64, 64, 3, 1, dilation=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CBAM(64),
            nn.Conv2d(64, 128, 3, 1, dilation=(3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.features1_3 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        x1 = self.features1_3(self.features1_2(self.features1_1(x)))
        return x1

class DGMnet(nn.Module):
    def __init__(self):
        super(DGMnet, self).__init__()
        self.lefteyeModel = EyeImageModel()
        self.righteyeModel = EyeImageModel()

        # 加載預訓練的 HarDNet
        self.base_model = HarDNetFeatureExtractor(arch=68, pretrained=True)
        # 根據 HarDNet 的輸出設置特徵金字塔
        # 微調輸出金字塔特徵

        self.downsample=conv1x1(654,128)


        self.fc = nn.Sequential(
            nn.Linear(384, 384//2),
            nn.LeakyReLU(),
            nn.Linear(384//2, 2),
        )
        self.avgpooling = nn.AdaptiveAvgPool2d(1)
    def forward(self, x_in):
        # Get face and eye gaze feature
        xEyeL = self.lefteyeModel(x_in['left_eye'])
        xEyeR = self.righteyeModel(x_in['right_eye'])
        # GAZE_features = torch.cat((xEyeL, xEyeR), 1)
        face_features = self.base_model(x_in['origin_face'])
        face_feature=self.downsample(face_features[3])
        xEyeL=self.avgpooling(xEyeL)
        xEyeR=self.avgpooling(xEyeR)
        xEyeL=xEyeL.flatten(1)
        xEyeR=xEyeR.flatten(1)
        # print(face_feature.shape)
        # print(xEyeL.shape)
        # print(xEyeR.shape)
        fusion_input = torch.cat([face_feature, xEyeL,xEyeR], dim=1)


        # multilevelfeature, feature_list = self.MFSEA([features[0],features[1],features[2],features[3],xEyeL,xEyeR])
        # multilevelfeature, feature_list = self.MFSEA([features[3],xEyeL,xEyeR])

        # gaze_list = []
        # for feature in feature_list:
        #     gaze_list.append(self.fc(feature))
        # clstoken_fusion_output = self.transformer(multilevelfeature, 1)
        # # 計算 gaze (目光方向)
        gaze = self.fc(fusion_input)
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