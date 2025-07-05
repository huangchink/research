import torch
import torch.nn as nn
import math
from .twoeyenet_transformer_1054 import DGMnet  # 假設 DGMnet 定義在此模組中
import os

# -----------------------------------------------------------------------------
# DMSnet 定義 (多流 Transformer + Transformer Fusion with Class Token)
# -----------------------------------------------------------------------------
class DMSnet(nn.Module):
    def __init__(self, nhead=1, num_layers=6, channel_num_layers=6, seq_length=15, out_dim=2):
        """
        Args:
            nhead: 時間流 Transformer 多頭注意力頭數
            num_layers: 時間流 Transformer encoder 層數
            channel_num_layers: 通道流 Transformer encoder 層數
            seq_length: 輸入序列長度 (必須與輸入資料中 sequence 長度一致，例如 30)
            out_dim: 最終分類類別數 (例如 2)
        """
        super(DMSnet, self).__init__()
        # 使用外部 DGMnet 提取 gaze 預測 (輸出 shape 期望 (B, seq, 2))
        self.DriverGaze = DGMnet()
        checkpoint_path = '/home/remote/tchuang/research/Gaze360/savemodel/checkpoint/best_model_epoch_10.44.pt'
        if not os.path.exists(checkpoint_path):
            print("找不到模型檔案，請確認路徑！")
            exit()
        self.DriverGaze.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.DriverGaze.eval()
        for param in self.DriverGaze.parameters():
            param.requires_grad = False

        # 每個 frame token 由 gaze (2) + head pose (3) + EAR (2) 組成，共 7 維
        self.token_dim = 8 
        self.seq_length = seq_length

        # 時間流 Transformer
        self.pos_embedding = nn.Embedding(seq_length, self.token_dim)
        temporal_layer = nn.TransformerEncoderLayer(d_model=self.token_dim, nhead=nhead, dropout=0.1, dim_feedforward=10)
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=num_layers)

        # 通道流 Transformer
        # 將原始 token (B, seq, 7) 轉換成 (B, 7, seq)
        # 這裡我們設定 d_model = seq_length (例如 30)
        channel_layer = nn.TransformerEncoderLayer(d_model=self.seq_length, nhead=1, dropout=0.1, dim_feedforward=10)
        self.channel_transformer = nn.TransformerEncoder(channel_layer, num_layers=channel_num_layers)

        # ---- Fusion with Transformer using a learnable class token ----
        fusion_dim = self.token_dim*self.seq_length  # 設定共同的嵌入維度

        # 5) Cross-Attention Fusion
        self.cls_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))

        # 5.2) Cross-Attention module
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=nhead,
            dropout=0.1,
            batch_first=False   # 輸入輸出格式為 (L, B, D)
        )

        # 5.3) 殘差＋FFN（Pre-LN 風格）
        self.norm1    = nn.LayerNorm(fusion_dim)
        self.mlp_fuse = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        self.norm2    = nn.LayerNorm(fusion_dim)

        # 5.4) 最終分類頭
        self.fc_out = nn.Linear(fusion_dim, out_dim)



    def forward(self, x):
        """
        Args:
            x: 字典，包含：
                "origin_face", "left_eye", "right_eye": (B, seq, C, H, W)
                "head_pose": (B, seq, 3)
                "ear": (B, seq, 2)
        """
        B, seq, C, H, W = x["origin_face"].shape
        # 取得 gaze 預測
        origin_face = x["origin_face"].view(B * seq, C, H, W)
        left_eye = x["left_eye"].view(B * seq, 3, 112, 112)
        right_eye = x["right_eye"].view(B * seq, 3, 112, 112)
                # 組合成 token: (B, seq, 8)

        x_img = {"origin_face": origin_face, "left_eye": left_eye, "right_eye": right_eye}
        gaze_flat = self.DriverGaze(x_img)[0]  # (B*seq, 2)
        gaze = gaze_flat.view(B, seq, 2)
        
        headpose = x["head_pose"]  # (B, seq, 3)
        mar=x["mar"]
        ear = x["ear"]            # (B, seq, 2)
        ear_avg   = ear.mean(dim=-1, keepdim=True)                            # (B,S,1)
        ear_shift = torch.cat([ear_avg[:, :1], ear_avg[:, :-1]], dim=1)       # (B,S,1)
        ear_dyn   = ear_avg - ear_shift                                       # (B,S,1)


        # 組合成 token: (B, seq, 7)
        token = torch.cat((gaze, headpose, ear_avg,ear_dyn,mar), dim=-1)
        # token = torch.cat((gaze, ear), dim=-1)
        
        # --------------------------
        # 時間流處理
        # --------------------------
        token_t = token.transpose(0, 1)  # (seq, B, 7)
        # pos_input = torch.arange(0, token_t.size(0), device=token_t.device).unsqueeze(1).float()  # (seq, 1)
        # pos_embedding = self.pos_encoder.encode(pos_input).unsqueeze(1)  # (seq, 1, 7)
        pos_ids = torch.arange(seq, device=token_t.device)
        pos_emb = self.pos_embedding(pos_ids).unsqueeze(1)  # (S,1,D)
        token_t = token_t + pos_emb
        T = self.temporal_encoder(token_t)  # (seq, B, 7)
        T_out = T.transpose(0, 1)  # (B, seq, 7)
        
        # --------------------------
        # 通道流處理
        # --------------------------
        channel_tokens = token.transpose(1, 2)  # (B, 7, seq)
        channel_tokens = channel_tokens.transpose(0, 1)  # (7, B, seq)
        C_out = self.channel_transformer(channel_tokens)  # (7, B, seq)
        #print(C_out.shape)
        C_out = C_out.transpose(0, 1)  # (B, 7, seq)

        #print(C_proj.shape)

        # --- 5) Cross-Attention Fusion ---
        # 投影成兩個 fusion token：(B,1,D) 每個
        D = self.token_dim * self.seq_length  # =8*seq
        T_proj = T_out.reshape(B, 1, D)           # (B,1,D)
        C_proj = C_out.reshape(B, 1, D)           # (B,1,D)

        # 拼成 (B,2,D)，再轉成 (2,B,D)
        seq_tokens = torch.cat([T_proj, C_proj], dim=1).transpose(0,1)  # (2,B,D)
        learnable_query = self.cls_token.expand(B,1,D).transpose(0,1)       # (1,B,D)

        # Cross-Attention
        attn_out, _ = self.cross_attn(learnable_query, seq_tokens, seq_tokens)   # (1,B,D)
        attn_out = attn_out.squeeze(0)                                # (B,D)

        # 殘差＋FFN（Pre-LN）
        y  = self.norm1(attn_out + learnable_query.squeeze(0))  # (B,D)
        y2 = self.mlp_fuse(y)                       # (B,D)
        fused = self.norm2(y + y2)                  # (B,D)

        # 分類頭
        out = self.fc_out(fused)                    # (B,out_dim)
        
        return out, gaze

def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = sum( p.numel() for p in model.parameters()) / (1024 ** 2)

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Model Size (M): {model_size:.2f}")
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 注意：此處 seq_length 必須與 dummy 輸入中的 sequence 長度一致
    model = DMSnet(seq_length=10, out_dim=2).to(device)
    # 建立 dummy 輸入，假設 batch_size=10, seq_length=30
    B = 10
    seq = 10  # 必須與模型的 seq_length 參數一致
    dummy_input = {
        "origin_face": torch.zeros(B, seq, 3, 224, 224).to(device),
        "left_eye": torch.zeros(B, seq, 3, 112, 112).to(device),
        "right_eye": torch.zeros(B, seq, 3, 112, 112).to(device),
        "head_pose": torch.zeros(B, seq, 3).to(device),
        "ear": torch.zeros(B, seq, 2).to(device),
        "mar": torch.zeros(B, seq, 1).to(device)

    }
    # 為方便測試，暫時以 lambda 模擬 DriverGaze 輸出 gaze (B, seq, 2)
    # dummy_gaze = torch.zeros(B, seq, 2).to(device)
    # model.DriverGaze = lambda x: (dummy_gaze.view(B * seq, 2),)
    # output, gaze = model(dummy_input)
    # print("Output shape:", output.shape)  # 預期 (B, out_dim)
    print_model_size(model)