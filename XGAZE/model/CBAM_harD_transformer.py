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
from model.hardnet import HarDNet

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
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


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def pos_embed(self, src, pos):
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos
        

    def forward(self, src, pos):
                # src_mask: Optional[Tensor] = None,
                # src_key_padding_mask: Optional[Tensor] = None):
                # pos: Optional[Tensor] = None):

        q = k = self.pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
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
        # self.features1_1 = nn.Sequential(
        #     nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0),
        #     nn.GroupNorm(3, 24),
        #     nn.ReLU(),
        #     nn.Conv2d(24, 48, kernel_size=5, stride=1, padding=0),
        #     )
        # eyeconv = models.vgg16(pretrained=True).features
        self.features1_1 = nn.Sequential(
            nn.Conv2d(3,   64,  3, 1, 1,  dilation=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CBAM(64),
            nn.Conv2d(64,  64,  3, 1, 1,  dilation=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            CBAM(64),
            nn.Conv2d(64,  128, 3, 1, 1,  dilation=(3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            CBAM(128),
            nn.Conv2d(128, 48, 3, 1, 1,  dilation=(2, 2)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
    
        self.features1_2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # SELayer(48, 16),
            CBAM(48),
            # nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=1),

            nn.Conv2d(48, 64, 1, 1,  dilation=(2, 2)), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CBAM(64),
            nn.Conv2d(64, 64, 3, 1,  dilation=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CBAM(64),
            nn.Conv2d(64, 32, 3, 1,  dilation=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            )
        self.features1_3 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):

        x1 = self.features1_3((self.features1_2(self.features1_1(x))))

        return x1 #[128,64,5,5]

class DGMnet(nn.Module):
    def __init__(self):
        super(DGMnet, self).__init__()
        self.eyeModel = EyeImageModel()
        self.base_model = HarDNet(arch=68, depth_wise=True, pretrained=True)

        maps = 128
        nhead = 16
        dim_feature = 7*7
        dim_feedforward=maps*4
        dropout = 0.2
        num_layers=6
        encoder_layer = TransformerEncoderLayer(
                        maps, 
                        nhead, 
                        dim_feedforward, 
                        dropout)

        encoder_norm = nn.LayerNorm(maps) 
        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))

        self.pos_embedding = nn.Embedding(dim_feature+1, maps)
        self.fc = nn.Sequential(
            nn.Linear(128, 2)
        )
        self.fcto49 = nn.Sequential(
            nn.Linear(25,49),
        )
        self.mix = nn.Sequential(
                         CBAM(1024)  ,
                         nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0),
                         nn.ReLU(),
                         CBAM(64),
                         )
        # self.mix192 = nn.Sequential(
        #                  CBAM(192)  ,
        #                  nn.Conv2d(192, 32, kernel_size=1, stride=1, padding=0),
        #                  nn.ReLU(),
        #                  CBAM(32),
        #                  )

    def forward(self, x_in):
        batch_size= x_in['origin_face'].shape[0]
        # Get face and eye gaze feature

        features = self.base_model(x_in['origin_face'])
        features = self.mix(features) 
        #print(features.shape) 

        xEyeL = self.eyeModel(x_in['left_eye'])
        xEyeR = self.eyeModel(x_in['right_eye'])
        # print(xEyeL.shape) [10, 64, 5, 5]
        # print(xEyeR.shape) [10, 64, 5, 5] 
        # print(features.shape) [10, 64, 7, 7]
        features = features.flatten(2)# [10, 64, 49]
        xEyeL    = xEyeL.flatten(2)# [10, 32, 25]
        xEyeR    = xEyeR.flatten(2)# [10, 32, 25]
        xEyeL    = self.fcto49(xEyeL)
        xEyeR    = self.fcto49(xEyeR)

        fusion_input = torch.cat([features,xEyeL,xEyeR], dim=1)  # [10, 128,49]
        #print(fusion_input.shape)

        #print(fusion_input.shape)# [10, 192,5,5]

        # fusion_input=self.mix192(fusion_input)# [10, 32, 5,5]
        #print(fusion_input.shape)
        # fusion_input = fusion_input.flatten(2)# [10, 32, 25]
        #print(fusion_input.shape)
        fusion_input = fusion_input.permute(2, 0, 1)# [49, 10, 128]
        #print(fusion_input.shape)

        cls = self.cls_token.repeat( (1, batch_size, 1))# [1, 10, 128]

        #print(cls.shape)#([1, 10, 32])

        # print(fusion_input.shape)


        fusion_input = torch.cat([cls, fusion_input], 0)# [50, 10, 128]
        #print(fusion_input.shape)
        position = torch.from_numpy(np.arange(0, 50)).cuda()

        pos_feature = self.pos_embedding(position)

        # feature is [HW, batch, channel]
        feature = self.encoder(fusion_input, pos_feature)
  
        feature = feature.permute(1, 2, 0)# [10, 32 ,50]

        feature = feature[:,:,0]
        # 計算 gaze (目光方向)
        #print(feature.shape)

        gaze = self.fc(feature)

        return gaze

        

def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 ** 2)

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Model Size (MB): {model_size:.2f}")

if __name__ == '__main__':
    m = DGMnet().cuda()
    print_model_size(m)

    feature = {
        "origin_face": torch.zeros(10, 3, 224, 224).cuda(),
        "left_eye": torch.zeros(10, 3, 112, 112).cuda(),
        "right_eye": torch.zeros(10, 3, 112, 112).cuda(),
        "gaze_origin": torch.zeros(10, 3).cuda()
    }
    output = m(feature)
