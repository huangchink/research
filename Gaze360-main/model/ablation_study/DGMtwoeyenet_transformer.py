
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

def ep0(x):
    return x.unsqueeze(0)
class conv1x1(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(conv1x1, self).__init__()
        
        #self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,padding=0, bias=False)

        self.bn = nn.BatchNorm2d(out_planes)


        self.conv = nn.Sequential(
            CBAM(in_planes),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,padding=0, bias=False),
            nn.BatchNorm2d(out_planes),
            #nn.ReLU(inplace=False),
            CBAM(out_planes),
        )



        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 

    def forward(self, feature):
        output = self.conv(feature)
        #output = self.bn(output)
        output = self.avgpool(output)
        output = output.squeeze(-1).squeeze(-1)
 
        return output
class MFSEA(nn.Module):

    def __init__(self, input_nums, hidden_num):

        super(MFSEA, self).__init__()

        # input_nums: [64, 128, 256, 512]
        length = len(input_nums)

        self.input_nums = input_nums
        self.hidden_num = hidden_num
        
        self.length = length

        layerList = []

        for i in range(length):
            layerList.append(self.__build_layer(input_nums[i], hidden_num))

        self.layerList = nn.ModuleList(layerList)


    def __build_layer(self, input_num, hidden_num):
        layer = conv1x1(input_num, hidden_num)
        
        return layer

    def forward(self, feature_list):

        out_feature_list = []

        out_feature_gather =[]

        for i, feature in enumerate(feature_list):
            result = self.layerList[i](feature)
            #print(result.shape)
            # Dim [B, C] -> [1, B, C]
            out_feature_list.append(result)

            out_feature_gather.append(ep0(result))

            
        # [L, B, C]
        feature = torch.cat(out_feature_gather, 0)
        return feature, out_feature_list
        

import torch
import torch.nn as nn
import torch.nn.functional as F
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return self.weight * (x / (norm + self.eps))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.5):
        super().__init__()
        # 多頭自注意力模組
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前饋網路
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 殘差連接用的 Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 激活函數 (ReLU)
        self.activation = nn.ReLU(inplace=True)
        
        # 使用自定義的 RMSNorm
        self.rms1 = RMSNorm(d_model)
        self.rms2 = RMSNorm(d_model)

    def pos_embed(self, src, pos):
        """
        將位置編碼 pos 與輸入 src 相加。
        假設 src 的 shape 為 (seq_len, batch_size, d_model)
        而 pos 的 shape 為 (seq_len, d_model)
        """
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos

    def forward(self, src, pos):
        """
        Args:
            src (Tensor): 輸入張量，shape = (seq_len, batch_size, d_model)
            pos (Tensor): 位置編碼張量，shape = (seq_len, d_model)
        """
        # 1. 先 RMSNorm，再做 Attention (加上位置編碼)
        src_norm = self.rms1(src)
        q = k = self.pos_embed(src_norm, pos)
        attn_out, _ = self.self_attn(q, k, value=src_norm)
        src = src + self.dropout1(attn_out)  # 殘差連接
        
        # 2. 再 RMSNorm，然後做 Feed-Forward
        src_norm = self.rms2(src)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(ff_out)  # 殘差連接
        
        return src



class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class PositionalEncoder():
    # encode low-dim, vec to high-dims.

    def __init__(self, number_freqs, include_identity=False):
        freq_bands = torch.pow(2, torch.linspace(0., number_freqs - 1, number_freqs))
        self.embed_fns = []
        self.output_dim = 0

        if include_identity:
            self.embed_fns.append(lambda x:x)
            self.output_dim += 1

        for freq in freq_bands:
            for transform_fns in [torch.sin, torch.cos]:
                self.embed_fns.append(lambda x, fns=transform_fns, freq=freq: fns(x*freq))
                self.output_dim += 1

    def encode(self, vecs):
        # inputs: [B, N]
        # outputs: [B, N*number_freqs*2]
        return torch.cat([fn(vecs) for fn in self.embed_fns], -1)

    def getDims(self):
        return self.output_dim
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
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
class Transformer(nn.Module):

    def __init__(self, input_dim, nhead=8, hidden_dim=512, layer_num = 6, pred_num=1, length=4, dropout=0.3):

        super(Transformer, self).__init__()

        self.pnum = pred_num
        # input feature + added token
        # self.length = length + 1
        self.length = length

        # The input feature should be [L, Batch, Input_dim]
        encoder_layer = TransformerEncoderLayer(
                  input_dim,
                  nhead = nhead,
                  dim_feedforward = hidden_dim,
                  dropout=dropout)

        encoder_norm = nn.LayerNorm(input_dim)

        self.encoder = TransformerEncoder(encoder_layer, num_layers = layer_num, norm = encoder_norm)

        self.cls_token = nn.Parameter(torch.randn(pred_num, 1, input_dim))

        self.token_pos_embedding = nn.Embedding(pred_num, input_dim)
        self.pos_embedding = nn.Embedding(length, input_dim)


    def forward(self, feature, num = 1):

        # feature: [Length, Batch, Dim]
        batch_size = feature.size(1)

        # cls_num, 1, Dim -> cls_num, Batch_size, Dim
        for i in range(num):

            cls = self.cls_token[i, :, :].repeat((1, batch_size, 1))
            #print('cls',cls.shape)
            #print('feature',feature.shape)
            feature_in = torch.cat([cls, feature], 0)

            # position
            position = torch.from_numpy(np.arange(self.length)).cuda()
            pos_feature = self.pos_embedding(position)

            token_position = torch.Tensor([i]).long().cuda()
            token_pos_feature = self.token_pos_embedding(token_position)

            pos_feature = torch.cat([token_pos_feature,pos_feature], 0)

            # feature [Length, batch, dim]
            feature_out = self.encoder(feature_in, pos_feature)

            # [batch, dim, length]
            # feature = feature.permute(1, 2, 0)
            #print(feature_out.shape)
            # get the first dimension, [pnum, batch, dim]
            feature_out = feature_out[0, :, :]

        return feature_out
      

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

        self.transformer = Transformer(input_dim=128, nhead=16, hidden_dim=256,
                                       layer_num=6, pred_num=1, length=len([124, 262,128]))
        self.fc = nn.Sequential(
            nn.Linear(128, 128//2),
            nn.LeakyReLU(),
            nn.Linear(128//2, 2),
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
        face_feature=face_feature.unsqueeze(0)
        xEyeL=xEyeL.unsqueeze(0)
        xEyeR=xEyeR.unsqueeze(0)
        # print(face_feature.shape)
        # print(xEyeL.shape)
        # print(xEyeR.shape)
        fusion_input = torch.cat([face_feature, xEyeL,xEyeR], dim=0)

        clstoken_fusion_output = self.transformer(fusion_input, 1)
        # 計算 gaze (目光方向)
        gaze = self.fc(clstoken_fusion_output)

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