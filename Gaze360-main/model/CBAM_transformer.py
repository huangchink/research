# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models
# import math
# import torchvision
# import sys
# import numpy as np
# import copy
# import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
# from model.hardnet import HarDNet

# # --------------------------------------------------
# # Helper: 建立多層 Encoder
# # --------------------------------------------------
# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# # --------------------------------------------------
# # Transformer Encoder 與 Layer
# # --------------------------------------------------
# class TransformerEncoder(nn.Module):
#     def __init__(self, encoder_layer, num_layers, norm=None):
#         super().__init__()
#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm

#     def forward(self, src, pos):
#         output = src
#         for layer in self.layers:
#             output = layer(output, pos)
#         if self.norm is not None:
#             output = self.norm(output)
#         return output

# class TransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)

#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#         self.activation = nn.ReLU(inplace=True)

#     def pos_embed(self, src, pos):
#         # pos: [total_tokens, d_model]
#         # src: [total_tokens, B, d_model]
#         # 需要將 pos unsqueeze(1) 再相加
#         return src + pos.unsqueeze(1)

#     def forward(self, src, pos):
#         # src: [seq_len, batch, d_model]
#         # pos: [seq_len, d_model]
#         q = k = self.pos_embed(src, pos)
#         src2 = self.self_attn(q, k, value=src)[0]
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)

#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src

# # --------------------------------------------------
# # CBAM 模組
# # --------------------------------------------------
# class ChannelAttentionModule(nn.Module):
#     def __init__(self, channel, ratio=16):
#         super(ChannelAttentionModule, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.shared_MLP = nn.Sequential(
#             nn.Conv2d(channel, channel // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channel // ratio, channel, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avgout = self.shared_MLP(self.avg_pool(x))
#         maxout = self.shared_MLP(self.max_pool(x))
#         return self.sigmoid(avgout + maxout)

# class SpatialAttentionModule(nn.Module):
#     def __init__(self):
#         super(SpatialAttentionModule, self).__init__()
#         self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avgout = torch.mean(x, dim=1, keepdim=True)
#         maxout, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avgout, maxout], dim=1)
#         out = self.conv2d(out)
#         return self.sigmoid(out)

# class CBAM(nn.Module):
#     def __init__(self, channel):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttentionModule(channel)
#         self.spatial_attention = SpatialAttentionModule()

#     def forward(self, x):
#         out = self.channel_attention(x) * x
#         out = self.spatial_attention(out) * out
#         return out

# # --------------------------------------------------
# # 眼睛模型
# # --------------------------------------------------
# class EyeImageModel(nn.Module):
#     def __init__(self):
#         super(EyeImageModel, self).__init__()
#         self.features1_1 = nn.Sequential(
#             nn.Conv2d(3,   64,  3, 1, 1,  dilation=(2, 2)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             CBAM(64),
#             nn.Conv2d(64,  64,  3, 1, 1,  dilation=(3, 3)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             CBAM(64),
#             nn.Conv2d(64,  128, 3, 1, 1,  dilation=(3, 3)),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             CBAM(128),
#             nn.Conv2d(128, 48, 3, 1, 1,  dilation=(2, 2)),
#             nn.BatchNorm2d(48),
#             nn.ReLU(),
#         )

#         self.features1_2 = nn.Sequential(
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             CBAM(48),
#             nn.Conv2d(48, 64, 1, 1, dilation=(2, 2)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             CBAM(64),
#             nn.Conv2d(64, 64, 3, 1, dilation=(3, 3)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             CBAM(64),
#             nn.Conv2d(64, 32, 3, 1, dilation=(3, 3)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#         )

#         self.features1_3 = nn.Sequential(
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )

#     def forward(self, x):
#         # x: [B, 3, 112, 112]
#         x = self.features1_1(x)    # -> [B, 48, H, W], 可能大約 H=14, W=14 (視 dilation/stride 而定)
#         x = self.features1_2(x)    # -> [B, 32, H', W']
#         x = self.features1_3(x)    # -> [B, 32, 5, 5] (原程式裡標註)
#         return x

# # --------------------------------------------------
# # 主網路
# # --------------------------------------------------
# class DGMnet(nn.Module):
#     def __init__(self):
#         super(DGMnet, self).__init__()
#         self.eyeModel = EyeImageModel()
#         self.base_model = HarDNet(arch=68, depth_wise=True, pretrained=True)

#         # mix face feature
#         self.mix = nn.Sequential(
#             CBAM(1024),
#             nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             CBAM(64),
#         )

#         # === 這裡是關鍵改動：將 token 維度做 concat ===
#         self.d_model = 128  # Transformer的embedding大小
#         self.face_proj = nn.Linear(64, self.d_model)  # [64 -> d_model]
#         self.eye_proj  = nn.Linear(32, self.d_model)  # [32 -> d_model]

#         # Transformer設定
#         maps = self.d_model
#         nhead = 16
#         dim_feedforward = maps * 4
#         dropout = 0.2
#         num_layers = 6

#         encoder_layer = TransformerEncoderLayer(
#             d_model=maps,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout
#         )
#         encoder_norm = nn.LayerNorm(maps)
#         self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

#         # 位置編碼: 假設最大 token 數(不含 batch) = 49 (臉) + 25(左眼) + 25(右眼) + 1(cls) = 100
#         self.max_token_num = 49 + 25 + 25 + 1
#         self.pos_embedding = nn.Embedding(self.max_token_num, maps)

#         # CLS token
#         self.cls_token = nn.Parameter(torch.randn(1, 1, maps))

#         # 最後做 gaze
#         self.fc = nn.Sequential(
#             nn.Linear(self.d_model, 2)
#         )

#     def forward(self, x_in):
#         """
#         x_in = {
#             'origin_face': [B, 3, 224, 224],
#             'left_eye':    [B, 3, 112, 112],
#             'right_eye':   [B, 3, 112, 112],
#             'gaze_origin': [B, 3]  # 用於Loss, 不在此forward中處理
#         }
#         """
#         batch_size = x_in['origin_face'].shape[0]

#         # 1) Face features
#         face_feat = self.base_model(x_in['origin_face'])  # e.g. [B, 1024, 7, 7]
#         face_feat = self.mix(face_feat)                   # -> [B, 64, 7, 7]
#         # 攤平空間 -> [B, 64, 49], 再 -> [B, 49, 64]
#         face_feat = face_feat.flatten(2).transpose(1, 2)  # => [B, 49, 64]
#         face_feat = self.face_proj(face_feat)             # => [B, 49, 128]

#         # 2) Eyes feature: 左眼
#         left_eye = self.eyeModel(x_in['left_eye'])        # -> [B, 32, 5, 5]
#         left_eye = left_eye.flatten(2).transpose(1, 2)    # -> [B, 25, 32]
#         left_eye = self.eye_proj(left_eye)                # -> [B, 25, 128]

#         # 3) Eyes feature: 右眼
#         right_eye = self.eyeModel(x_in['right_eye'])      # -> [B, 32, 5, 5]
#         right_eye = right_eye.flatten(2).transpose(1, 2)  # -> [B, 25, 32]
#         right_eye = self.eye_proj(right_eye)              # -> [B, 25, 128]

#         # 4) Token 拼接 [face(49) + left(25) + right(25)] => [B, 99, 128]
#         fusion_tokens = torch.cat([face_feat, left_eye, right_eye], dim=1)  # => [B, 49+25+25, 128] = [B, 99, 128]

#         # 5) 轉成 Transformer 的輸入: [seq_len, B, d_model]
#         fusion_tokens = fusion_tokens.permute(1, 0, 2)    # => [99, B, 128]

#         # 6) 加上CLS token => [1 + 99 = 100, B, 128]
#         cls = self.cls_token.repeat(1, batch_size, 1)     # => [1, B, 128]
#         fusion_tokens = torch.cat([cls, fusion_tokens], dim=0)  # => [100, B, 128]

#         # 7) Positional Embedding
#         #   先產生 [0..99] 的 index，與 [B, 128] 無關，只是 seq_len
#         device = fusion_tokens.device
#         seq_len = fusion_tokens.shape[0]  # 100
#         position_ids = torch.arange(seq_len, device=device)  # [0..99]

#         # 取得對應的pos編碼 -> [100, 128]
#         pos_feature = self.pos_embedding(position_ids)

#         # 8) Transformer Encoder
#         feature = self.encoder(fusion_tokens, pos_feature)  # => [100, B, 128]

#         # 9) 取出 CLS token (位於序列最前面)
#         feature = feature[0, :, :]  # => [B, 128]

#         # 10) 預測 gaze
#         gaze = self.fc(feature)     # => [B, 2]
#         return gaze

# # --------------------------------------------------
# # 測試
# # --------------------------------------------------
# def print_model_size(model):
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     model_size = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 ** 2)

#     print(f"Total Parameters: {total_params}")
#     print(f"Trainable Parameters: {trainable_params}")
#     print(f"Model Size (MB): {model_size:.2f}")

# if __name__ == '__main__':
#     m = DGMnet().cuda()
#     print_model_size(m)

#     feature = {
#         "origin_face": torch.zeros(10, 3, 224, 224).cuda(),
#         "left_eye":    torch.zeros(10, 3, 112, 112).cuda(),
#         "right_eye":   torch.zeros(10, 3, 112, 112).cuda(),
#         "gaze_origin": torch.zeros(10, 3).cuda()
#     }
#     output = m(feature)
#     print("Output shape:", output.shape)  # [10, 2]
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models
# import math
# import torchvision
# import sys
# import numpy as n
# import copy
# import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
# import numpy as np
# from model.hardnet import HarDNet

# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
# class TransformerEncoder(nn.Module):

#     def __init__(self, encoder_layer, num_layers, norm=None):
#         super().__init__()
#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm

#     def forward(self, src, pos):
#         output = src
#         for layer in self.layers:
#             output = layer(output, pos)

#         if self.norm is not None:
#             output = self.norm(output)

#         return output


# class TransformerEncoderLayer(nn.Module):

#     def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)

#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#         self.activation = nn.ReLU(inplace=True)

#     def pos_embed(self, src, pos):
#         batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
#         return src + batch_pos
        

#     def forward(self, src, pos):
#                 # src_mask: Optional[Tensor] = None,
#                 # src_key_padding_mask: Optional[Tensor] = None):
#                 # pos: Optional[Tensor] = None):

#         q = k = self.pos_embed(src, pos)
#         src2 = self.self_attn(q, k, value=src)[0]
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)

#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src

# class ChannelAttentionModule(nn.Module):
#     def __init__(self, channel, ratio=16):
#         super(ChannelAttentionModule, self).__init__()

#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
 
#         self.shared_MLP = nn.Sequential(
#             nn.Conv2d(channel, channel // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channel // ratio, channel, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
 
#     def forward(self, x):
#         avgout = self.shared_MLP(self.avg_pool(x))
#         maxout = self.shared_MLP(self.max_pool(x))
#         return self.sigmoid(avgout + maxout)
 
# class SpatialAttentionModule(nn.Module):
#     def __init__(self):
#         super(SpatialAttentionModule, self).__init__()
#         self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
#         self.sigmoid = nn.Sigmoid()
 
#     def forward(self, x):

#         avgout = torch.mean(x, dim=1, keepdim=True)
#         maxout, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avgout, maxout], dim=1)
#         out = self.sigmoid(self.conv2d(out))
#         return out
 
# class CBAM(nn.Module):
#     def __init__(self, channel):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttentionModule(channel)
#         self.spatial_attention = SpatialAttentionModule()
 
#     def forward(self, x):
#         out = self.channel_attention(x) * x
#         out = self.spatial_attention(out) * out
#         return out

# class EyeImageModel(nn.Module):
#     def __init__(self):
#         super(EyeImageModel, self).__init__()
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1)
#         # self.features1_1 = nn.Sequential(
#         #     nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0),
#         #     nn.GroupNorm(3, 24),
#         #     nn.ReLU(),
#         #     nn.Conv2d(24, 48, kernel_size=5, stride=1, padding=0),
#         #     )
#         # eyeconv = models.vgg16(pretrained=True).features
#         self.features1_1 = nn.Sequential(
#             nn.Conv2d(3,   64,  3, 1, 1,  dilation=(2, 2)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             CBAM(64),
#             nn.Conv2d(64,  64,  3, 1, 1,  dilation=(3, 3)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             CBAM(64),
#             nn.Conv2d(64,  128, 3, 1, 1,  dilation=(3, 3)),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             CBAM(128),
#             nn.Conv2d(128, 48, 3, 1, 1,  dilation=(2, 2)),
#             nn.BatchNorm2d(48),
#             nn.ReLU(),
#         )
    
#         self.features1_2 = nn.Sequential(
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             # SELayer(48, 16),
#             CBAM(48),
#             # nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=1),

#             nn.Conv2d(48, 64, 1, 1,  dilation=(2, 2)), 
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             CBAM(64),
#             nn.Conv2d(64, 64, 3, 1,  dilation=(3, 3)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             CBAM(64),
#             nn.Conv2d(64, 64, 3, 1,  dilation=(3, 3)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),

#             )
#         self.features1_3 = nn.Sequential(
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )

#     def forward(self, x):

#         x1 = self.features1_3((self.features1_2(self.features1_1(x))))

#         return x1 #[128,64,5,5]

# class DGMnet(nn.Module):
#     def __init__(self):
#         super(DGMnet, self).__init__()
#         self.eyeModel = EyeImageModel()
#         self.base_model = HarDNet(arch=68, depth_wise=True, pretrained=True)

#         maps = 192
#         nhead = 8
#         dim_feature = 7*7
#         dim_feedforward=192*4
#         dropout = 0.1
#         num_layers=3
#         encoder_layer = TransformerEncoderLayer(
#                         maps, 
#                         nhead, 
#                         dim_feedforward, 
#                         dropout)

#         encoder_norm = nn.LayerNorm(maps) 
#         self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

#         self.cls_token = nn.Parameter(torch.randn(1, 1, maps))

#         self.pos_embedding = nn.Embedding(dim_feature+1, maps)
#         self.fc = nn.Sequential(
#             nn.Linear(192, 2)
#         )
#         self.fcto49 = nn.Sequential(
#             nn.Linear(25,49),
#         )
#         self.mix = nn.Sequential(
#                          CBAM(1024)  ,
#                          nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0),
#                          nn.ReLU(),
#                          CBAM(64),
#                          )
#         self.mix192 = nn.Sequential(
#                          CBAM(192)  ,
#                          nn.Conv2d(192, 32, kernel_size=1, stride=1, padding=0),
#                          nn.ReLU(),
#                          CBAM(32),
#                          )

#     def forward(self, x_in):
#         batch_size= x_in['origin_face'].shape[0]
#         # Get face and eye gaze feature

#         features = self.base_model(x_in['origin_face'])
#         features = self.mix(features) 
#         #print(features.shape) 

#         xEyeL = self.eyeModel(x_in['left_eye'])
#         xEyeR = self.eyeModel(x_in['right_eye'])
#         # print(xEyeL.shape) [10, 64, 5, 5]
#         # print(xEyeR.shape) [10, 64, 5, 5] 
#         # print(features.shape) [10, 64, 7, 7]
#         features = features.flatten(2)# [10, 64, 49]
#         xEyeL    = xEyeL.flatten(2)# [10, 32, 25]
#         xEyeR    = xEyeR.flatten(2)# [10, 32, 25]
#         xEyeL    = self.fcto49(xEyeL)
#         xEyeR    = self.fcto49(xEyeR)

#         fusion_input = torch.cat([features,xEyeL,xEyeR], dim=1)  # [10, 192,49]
#         #print(fusion_input.shape)

#         #print(fusion_input.shape)# [10, 192,5,5]

#         # fusion_input=self.mix192(fusion_input)# [10, 32, 5,5]
#         #print(fusion_input.shape)
#         # fusion_input = fusion_input.flatten(2)# [10, 32, 25]
#         #print(fusion_input.shape)
#         fusion_input = fusion_input.permute(2, 0, 1)# [49, 10, 192]
#         #print(fusion_input.shape)

#         cls = self.cls_token.repeat( (1, batch_size, 1))# [1, 10, 32]

#         #print(cls.shape)#([1, 10, 32])

#         # print(fusion_input.shape)


#         fusion_input = torch.cat([cls, fusion_input], 0)# [50, 10, 192]
#         #print(fusion_input.shape)
#         position = torch.from_numpy(np.arange(0, 50)).cuda()

#         pos_feature = self.pos_embedding(position)

#         # feature is [HW, batch, channel]
#         feature = self.encoder(fusion_input, pos_feature)
  
#         feature = feature.permute(1, 2, 0)# [10, 32 ,50]

#         feature = feature[:,:,0]
#         # 計算 gaze (目光方向)
#         #print(feature.shape)

#         gaze = self.fc(feature)

#         return gaze

        

# def print_model_size(model):
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     model_size = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 ** 2)

#     print(f"Total Parameters: {total_params}")
#     print(f"Trainable Parameters: {trainable_params}")
#     print(f"Model Size (MB): {model_size:.2f}")

# if __name__ == '__main__':
#     m = DGMnet().cuda()
#     print_model_size(m)

#     feature = {
#         "origin_face": torch.zeros(10, 3, 224, 224).cuda(),
#         "left_eye": torch.zeros(10, 3, 112, 112).cuda(),
#         "right_eye": torch.zeros(10, 3, 112, 112).cuda(),
#         "gaze_origin": torch.zeros(10, 3).cuda()
#     }
#     output = m(feature)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models
# import math
# import torchvision
# import sys
# import numpy as n
# import copy
# import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
# import numpy as np
# from model.hardnet import HarDNet

# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
# class TransformerEncoder(nn.Module):

#     def __init__(self, encoder_layer, num_layers, norm=None):
#         super().__init__()
#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm

#     def forward(self, src, pos):
#         output = src
#         for layer in self.layers:
#             output = layer(output, pos)

#         if self.norm is not None:
#             output = self.norm(output)

#         return output


# class TransformerEncoderLayer(nn.Module):

#     def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)

#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#         self.activation = nn.ReLU(inplace=True)

#     def pos_embed(self, src, pos):
#         batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
#         return src + batch_pos
        

#     def forward(self, src, pos):
#                 # src_mask: Optional[Tensor] = None,
#                 # src_key_padding_mask: Optional[Tensor] = None):
#                 # pos: Optional[Tensor] = None):

#         q = k = self.pos_embed(src, pos)
#         src2 = self.self_attn(q, k, value=src)[0]
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)

#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src

# class ChannelAttentionModule(nn.Module):
#     def __init__(self, channel, ratio=16):
#         super(ChannelAttentionModule, self).__init__()

#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
 
#         self.shared_MLP = nn.Sequential(
#             nn.Conv2d(channel, channel // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channel // ratio, channel, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
 
#     def forward(self, x):
#         avgout = self.shared_MLP(self.avg_pool(x))
#         maxout = self.shared_MLP(self.max_pool(x))
#         return self.sigmoid(avgout + maxout)
 
# class SpatialAttentionModule(nn.Module):
#     def __init__(self):
#         super(SpatialAttentionModule, self).__init__()
#         self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
#         self.sigmoid = nn.Sigmoid()
 
#     def forward(self, x):

#         avgout = torch.mean(x, dim=1, keepdim=True)
#         maxout, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avgout, maxout], dim=1)
#         out = self.sigmoid(self.conv2d(out))
#         return out
 
# class CBAM(nn.Module):
#     def __init__(self, channel):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttentionModule(channel)
#         self.spatial_attention = SpatialAttentionModule()
 
#     def forward(self, x):
#         out = self.channel_attention(x) * x
#         out = self.spatial_attention(out) * out
#         return out

# class EyeImageModel(nn.Module):
#     def __init__(self):
#         super(EyeImageModel, self).__init__()
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1)
#         # self.features1_1 = nn.Sequential(
#         #     nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0),
#         #     nn.GroupNorm(3, 24),
#         #     nn.ReLU(),
#         #     nn.Conv2d(24, 48, kernel_size=5, stride=1, padding=0),
#         #     )
#         # eyeconv = models.vgg16(pretrained=True).features
#         self.features1_1 = nn.Sequential(
#             nn.Conv2d(3,   64,  3, 1, 1,  dilation=(2, 2)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             CBAM(64),
#             nn.Conv2d(64,  64,  3, 1, 1,  dilation=(3, 3)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             CBAM(64),
#             nn.Conv2d(64,  128, 3, 1, 1,  dilation=(3, 3)),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             CBAM(128),
#             nn.Conv2d(128, 48, 3, 1, 1,  dilation=(2, 2)),
#             nn.BatchNorm2d(48),
#             nn.ReLU(),
#         )
    
#         self.features1_2 = nn.Sequential(
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             # SELayer(48, 16),
#             CBAM(48),
#             # nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=1),

#             nn.Conv2d(48, 64, 1, 1,  dilation=(2, 2)), 
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             CBAM(64),
#             nn.Conv2d(64, 64, 3, 1,  dilation=(3, 3)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             CBAM(64),
#             nn.Conv2d(64, 64, 3, 1,  dilation=(3, 3)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),

#             )
#         self.features1_3 = nn.Sequential(
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )

#     def forward(self, x):

#         x1 = self.features1_3((self.features1_2(self.features1_1(x))))

#         return x1 #[128,64,5,5]

# class DGMnet(nn.Module):
#     def __init__(self):
#         super(DGMnet, self).__init__()
#         self.eyeModel = EyeImageModel()
#         self.base_model = HarDNet(arch=68, depth_wise=True, pretrained=True)

#         maps = 64
#         nhead = 8
#         dim_feature = 7*7
#         dim_feedforward=192*4
#         dropout = 0.1
#         num_layers=3
#         encoder_layer = TransformerEncoderLayer(
#                         maps, 
#                         nhead, 
#                         dim_feedforward, 
#                         dropout)

#         encoder_norm = nn.LayerNorm(maps) 
#         self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

#         self.cls_token = nn.Parameter(torch.randn(1, 1, maps))

#         self.pos_embedding = nn.Embedding(dim_feature+1, maps)
#         self.fc = nn.Sequential(
#             nn.Linear(64, 2)
#         )
#         self.fcto49 = nn.Sequential(
#             nn.Linear(25,49),
#         )
#         self.mix = nn.Sequential(
#                          CBAM(1024)  ,
#                          nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0),
#                          nn.ReLU(),
#                          CBAM(64),
#                          )
#         self.mix192 = nn.Sequential(
#                          CBAM(192)  ,
#                          nn.Conv2d(192, 32, kernel_size=1, stride=1, padding=0),
#                          nn.ReLU(),
#                          CBAM(32),
#                          )

#     def forward(self, x_in):
#         batch_size= x_in['origin_face'].shape[0]
#         # Get face and eye gaze feature

#         features = self.base_model(x_in['origin_face'])
#         features = self.mix(features) 
#         #print(features.shape) 

#         # xEyeL = self.eyeModel(x_in['left_eye'])
#         # xEyeR = self.eyeModel(x_in['right_eye'])
#         # print(xEyeL.shape) [10, 64, 5, 5]
#         # print(xEyeR.shape) [10, 64, 5, 5] 
#         # print(features.shape) [10, 64, 7, 7]
#         features = features.flatten(2)# [10, 64, 49]
#         # xEyeL    = xEyeL.flatten(2)# [10, 32, 25]
#         # xEyeR    = xEyeR.flatten(2)# [10, 32, 25]
#         # xEyeL    = self.fcto49(xEyeL)
#         # xEyeR    = self.fcto49(xEyeR)

#         # fusion_input = torch.cat([features,xEyeL,xEyeR], dim=1)  # [10, 192,49]
#         #print(fusion_input.shape)

#         #print(fusion_input.shape)# [10, 192,5,5]

#         # fusion_input=self.mix192(fusion_input)# [10, 32, 5,5]
#         #print(fusion_input.shape)
#         # fusion_input = fusion_input.flatten(2)# [10, 32, 25]
#         #print(fusion_input.shape)
#         fusion_input = features.permute(2, 0, 1)# [49, 10, 192]
#         #print(fusion_input.shape)

#         cls = self.cls_token.repeat( (1, batch_size, 1))# [1, 10, 32]

#         #print(cls.shape)#([1, 10, 32])

#         #print(fusion_input.shape)


#         fusion_input = torch.cat([cls, fusion_input], 0)# [50, 10, 192]
#         #print(fusion_input.shape)
#         position = torch.from_numpy(np.arange(0, 50)).cuda()

#         pos_feature = self.pos_embedding(position)

#         # feature is [HW, batch, channel]
#         feature = self.encoder(fusion_input, pos_feature)
  
#         feature = feature.permute(1, 2, 0)# [10, 32 ,50]

#         feature = feature[:,:,0]
#         # 計算 gaze (目光方向)
#         #print(feature.shape)

#         gaze = self.fc(feature)

#         return gaze

        

# def print_model_size(model):
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     model_size = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 ** 2)

#     print(f"Total Parameters: {total_params}")
#     print(f"Trainable Parameters: {trainable_params}")
#     print(f"Model Size (MB): {model_size:.2f}")

# if __name__ == '__main__':
#     m = DGMnet().cuda()
#     print_model_size(m)

#     feature = {
#         "origin_face": torch.zeros(10, 3, 224, 224).cuda(),
#         "left_eye": torch.zeros(10, 3, 112, 112).cuda(),
#         "right_eye": torch.zeros(10, 3, 112, 112).cuda(),
#         "gaze_origin": torch.zeros(10, 3).cuda()
#     }
#     output = m(feature)
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
from model.hardnet import HarDNetFeatureExtractor

        
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

        self.activation = nn.ReLU(inplace=False)

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
            nn.MaxPool2d(kernel_size=2, stride=2),
            # SELayer(48, 16),
            CBAM(48),
            # nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=1),

            nn.Conv2d(48, 64, 1, 1,  dilation=(2, 2)), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CBAM(64),
            nn.Conv2d(64, 64, 3, 1,padding=1,  dilation=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CBAM(64),
            nn.Conv2d(64, 16, 3,1, padding=1,  dilation=(3, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            )
        self.features1_3 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):

        x1 = self.features1_3((self.features1_2(self.features1_1(x))))

        return x1 #[128,64,5,5]
class ProjectionModule(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(ProjectionModule, self).__init__()
        self.linearPJ = nn.Sequential(
            CBAM(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            CBAM(out_channels)
        )
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
    
    def forward(self, x):
        out = self.linearPJ(x)
        # 殘差連接，確保尺寸一致
        res = self.residual(x)
        out = out + res
        return out
class DGMnet(nn.Module):
    def __init__(self):
        super(DGMnet, self).__init__()

        self.eyeModel   = EyeImageModel()
        self.base_model = HarDNetFeatureExtractor(arch=68, pretrained=True)

        self.d_model        = 32  
        nhead               = 2 
        num_layers          = 6
        dim_feedforward     = self.d_model * 4 
        dropout             = 0.2
        total_channel       = 782+self.d_model
        encoder_layer = TransformerEncoderLayer(
            d_model         = self.d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout
        )
        encoder_norm = nn.LayerNorm(self.d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self.cls_token = nn.Parameter(torch.randn(1, self.d_model,1))

        self.pos_embedding = nn.Embedding(50, self.d_model)
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, 2)
        )
        # 結合臉部特徵後進行簡單卷積處理
        self.linearPJ = ProjectionModule(total_channel,self.d_model)


    def forward(self, x_in):
        """
        x_in: {
          'origin_face': [B,3,224,224],
          'left_eye':    [B,3,112,112],
          'right_eye':   [B,3,112,112],
          ...
        }
        """
        B = x_in['origin_face'].shape[0]

        # 1. Face 網路
        x,features = self.base_model(x_in['origin_face'])
        face_feature=features[-1]
        # features = features.flatten(2)                   # => [B, 1056, 49]
        # 2. Eye 網路
        eyeL = self.eyeModel(x_in['left_eye'])  # e.g. [B, 64, 7,7]
        eyeR = self.eyeModel(x_in['right_eye']) # e.g. [B, 64, 7,7]

        fuison_input_cnn = torch.cat([face_feature, eyeL, eyeR], dim=1)  # => [B, 1056, 7,7]
        
        fuison_input_transformer = self.linearPJ(fuison_input_cnn)                    # [B, 32,   7, 7]
        fuison_input_transformer = fuison_input_transformer.flatten(2)                  # => [B, 32, 49]

        # 加 CLS token：形狀 [B, 1, 49]
        cls = self.cls_token.repeat(B, 1, 1)                # => [B, 32, 1]
        fuison_input_transformer = torch.cat([cls, fuison_input_transformer], dim=2)# => [B, 32, 50]
        #print(fusion_input.shape)

        # Transformer 需 [seq_len, batch_size, d_model] => permute
        fuison_input_transformer = fuison_input_transformer.permute(2, 0, 1)        # => [50, B, 128]

        # Positional embedding: arange(129)
        pos_idx     = torch.arange(0, 50, device=fuison_input_transformer.device)  # [0..49]
        pos_feature = self.pos_embedding(pos_idx)           # => [129, 49]

        # 送入 Transformer
        output = self.encoder(fuison_input_transformer, pos_feature)    # => [129, B, 49]

        # 取 CLS token (sequence 維度 = 0) => [B, 49]
        cls_feature = output[0]                             # => [B, 49]

        # 最後全連接輸出 => [B,2]
        gaze = self.fc(cls_feature)
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
