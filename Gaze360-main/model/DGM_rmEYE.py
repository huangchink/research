
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
class MLEnhance(nn.Module):

    def __init__(self, input_nums, hidden_num):

        super(MLEnhance, self).__init__()

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
        
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return self.weight * (x / (norm + self.eps))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
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
            nn.Conv2d(64, 32, 3, 1, dilation=(3, 3)),
            nn.BatchNorm2d(32),
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

        feature_list = []

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
        # self.eyeModel = EyeImageModel()

        # 加載預訓練的 HarDNet
        self.base_model = HarDNetFeatureExtractor(arch=68, pretrained=True)
        # 根據 HarDNet 的輸出設置特徵金字塔
        self.mle = MLEnhance(input_nums=[124, 262,328,654], hidden_num=128)
        # 微調輸出金字塔特徵
        self.transformer = Transformer(input_dim=128, nhead=16, hidden_dim=256,
                                       layer_num=6, pred_num=1, length=len([124, 262,328,654]))
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
        )
        # self.mix = nn.Sequential(
        #     CBAM(64),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        #     CBAM(128),
        # )
        self.avgpooling = nn.AdaptiveAvgPool2d(1)
        self.loss_L1 = nn.L1Loss()
    def forward(self, x_in):
        # Get face and eye gaze feature
        # xFace = self.faceModel(x_in['origin_face'])
        # xEyeL = self.eyeModel(x_in['left_eye'])
        # xEyeR = self.eyeModel(x_in['right_eye'])
        # print(xEyeL.shape)
        # print(xEyeR.shape)
        features = self.base_model(x_in['origin_face'])
        x1,x2,x3,x4=features[0],features[1],features[2],features[3]
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)


        pyramidfeature, feature_list = self.mle([x1, x2 ,x3,x4])
        gaze_list = []
        for feature in feature_list:
            #print(feature.shape)
            gaze_list.append(self.fc(feature))
        #print(pyramidfeature.shape)


        # GAZE_features = torch.cat((xEyeL, xEyeR), 1)
        # GAZE_features = self.mix(GAZE_features)
        # # print('GAZE_features',GAZE_features.shape)

        # GAZE_features = self.avgpooling(GAZE_features)

        # GAZE_features = torch.squeeze(GAZE_features,dim=2)
        #print('fuck',GAZE_features.shape)

        #print('fuck',GAZE_features.shape)
        #print(pyramidfeature.shape)
        # GAZE_features = GAZE_features.permute(2, 0, 1)        # => [50, B, 128]
        #print('GAZE_features',GAZE_features.shape)

        #print(GAZE_features.shape)
        # gaze1 = self.fc(GAZE_features)
        # pyramidfeature = self.transformer(pyramidfeature, 1)

        # fusion_input = torch.cat([pyramidfeature, GAZE_features], dim=0)  # [5, 10, 128]
        pyramidfeature = self.transformer(pyramidfeature, 1)
        #print(pyramidfeature.shape)

        # 計算 gaze (目光方向)
        gaze = self.fc(pyramidfeature)
        #print(gaze.shape)
        return gaze,gaze_list[0],gaze_list[1],gaze_list[2],gaze_list[3]

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.data.size(0),-1)



class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.add_module('layer1',ConvLayer(in_channels, out_channels, kernel))
        self.add_module('layer2',DWConvLayer(out_channels, out_channels, stride=stride))
        
    def forward(self, x):
        return super().forward(x)

class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels,  stride=1,  bias=False):
        super().__init__()
        out_ch = out_channels
        
        groups = in_channels
        kernel = 3
        #print(kernel, 'x', kernel, 'x', out_channels, 'x', out_channels, 'DepthWise')
        
        self.add_module('dwconv', nn.Conv2d(groups, groups, kernel_size=3,
                                          stride=stride, padding=1, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(groups))
    def forward(self, x):
        return super().forward(x)  

class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = 1
        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
        self.add_module('conv', nn.Conv2d(in_channels, out_ch, kernel_size=kernel,          
                                          stride=stride, padding=kernel//2, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_ch))
        self.add_module('relu', nn.ReLU6(True))                                          
    def forward(self, x):
        return super().forward(x)


class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
          return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
          dv = 2 ** i
          if layer % dv == 0:
            k = layer - dv
            link.append(k)
            if i > 0:
                out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
          ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
          in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0 # if upsample else in_channels
        for i in range(n_layers):
          outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
          self.links.append(link)
          use_relu = residual_out
          if dwconv:
            layers_.append(CombConvLayer(inch, outch))
          else:
            layers_.append(ConvLayer(inch, outch))
          
          if (i % 2 == 0) or (i == n_layers - 1):
            self.out_channels += outch
        #print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)
        
    def forward(self, x):
        layers_ = [x]
        
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:            
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
            
        t = len(layers_)
        out_ = []
        for i in range(t):
          if (i == 0 and self.keepBase) or \
             (i == t-1) or (i%2 == 1):
              out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out
        
        

if __name__ == '__main__':
    m = DGMnet().cuda()
    feature = {
        "origin_face": torch.zeros(10, 3, 224, 224).cuda(),
        "left_eye": torch.zeros(10, 3, 112, 112).cuda(),
        "right_eye": torch.zeros(10, 3, 112, 112).cuda(),
        "gaze_origin": torch.zeros(10, 3).cuda()
    }
    output = m(feature)
