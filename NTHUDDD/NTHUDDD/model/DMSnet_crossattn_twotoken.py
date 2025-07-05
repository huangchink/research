import torch
import torch.nn as nn

# -------------------------------------------------------------------
# Positional Encoder（若未再使用可刪） -------------------------------
# -------------------------------------------------------------------
class PositionalEncoder():
    def __init__(self, number_freqs, include_identity=False):
        freq_bands = torch.pow(2, torch.linspace(0., number_freqs - 1, number_freqs))
        self.embed_fns, self.output_dim = [], 0
        if include_identity:
            self.embed_fns.append(lambda x: x)
            self.output_dim += 1
        for freq in freq_bands:
            for f in [torch.sin, torch.cos]:
                self.embed_fns.append(lambda x, fns=f, freq=freq: fns(x * freq))
                self.output_dim += 1
    def encode(self, vecs):
        return torch.cat([fn(vecs) for fn in self.embed_fns], -1)
    def getDims(self):
        return self.output_dim

# -------------------------------------------------------------------
#                         D M S n e t
# -------------------------------------------------------------------
class DMSnet(nn.Module):
    def __init__(self, nhead=5, num_layers=6, channel_num_layers=6,
                 seq_length=30, out_dim=2):
        super().__init__()

        # 一個 frame 的 token 組成：gaze(2)+EAR_avg(1)+EAR_dyn(1)+MAR(1)=5
        self.token_dim  = 5
        self.seq_length = seq_length

        # ── Temporal stream ──────────────────────────────────────────
        self.temporal_pos_embedding = nn.Embedding(seq_length, self.token_dim)
        self.temp_cls_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))

        temporal_layer  = nn.TransformerEncoderLayer(
            d_model=self.token_dim, nhead=nhead,
            dropout=0.4, dim_feedforward=10)
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=num_layers)

        # ── Channel (modality) stream ────────────────────────────────
        self.channel_pos_embedding = nn.Embedding(self.token_dim, seq_length)
        self.channel_cls_token = nn.Parameter(torch.zeros(1, 1, seq_length))

        channel_layer = nn.TransformerEncoderLayer(
            d_model=seq_length, nhead=nhead,
            dropout=0.4, dim_feedforward=10)
        self.modality_encoder = nn.TransformerEncoder(
            channel_layer, num_layers=channel_num_layers)

        # ── Cross-stream fusion ──────────────────────────────────────
        self.fusion_dim = self.token_dim * self.seq_length   # 5×30 = 150
        self.t_proj = nn.Linear(self.token_dim,  self.fusion_dim)
        self.c_proj = nn.Linear(self.seq_length, self.fusion_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.fusion_dim, num_heads=nhead,
            dropout=0.1, batch_first=False)

        self.fusion_cls_token = nn.Parameter(torch.zeros(1, 1, self.fusion_dim))

        # FFN (Pre-LN)
        self.norm1 = nn.LayerNorm(self.fusion_dim)
        self.mlp_fuse = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.ReLU(), nn.Linear(self.fusion_dim, self.fusion_dim))
        self.norm2 = nn.LayerNorm(self.fusion_dim)

        # ── Head ─────────────────────────────────────────────────────
        self.fc_out = nn.Linear(self.fusion_dim, out_dim)

    # ----------------------------------------------------------------
    def forward(self, x):
        """
        x 需包含：
          gaze   : (B,S,2)
          ear    : (B,S,2)
          mar    : (B,S,1)
        其餘影像張量可照舊傳入但這裡不再使用
        """
        B, S = x['gaze'].shape[:2]

        # --------- 建 token (B,S,5) ---------------------------------
        gaze = x['gaze']                       # (B,S,2)
        ear  = x['ear']                        # (B,S,2)
        mar  = x['mar']                        # (B,S,1)

        ear_avg   = ear.mean(-1, keepdim=True)                     # (B,S,1)
        ear_shift = torch.cat([ear_avg[:, :1, :], ear_avg[:, :-1, :]], 1)
        ear_dyn   = ear_avg - ear_shift                            # (B,S,1)
        token     = torch.cat([gaze, ear_avg, ear_dyn, mar], -1)   # (B,S,5)

        # ============================================================
        # 1) Temporal stream  (token dim = 5)
        # ============================================================
        # (S,B,5)
        t_input = token.transpose(0,1)
        # Positional emb.
        t_pos   = self.temporal_pos_embedding(
                     torch.arange(S, device=t_input.device)
                  ).unsqueeze(1)
        t_input = t_input + t_pos
        # 加 CLS
        temp_cls = self.temp_cls_token.expand(-1, B, -1)           # (1,B,5)
        t_in     = torch.cat([temp_cls, t_input], 0)               # (S+1,B,5)
        t_out    = self.temporal_encoder(t_in)                     # (S+1,B,5)
        t_cls    = t_out[0]                                        # (B,5)

        # ============================================================
        # 2) Channel (modality) stream  (d_model = S)
        # ============================================================
        # (B,5,S) → (5,B,S)
        c_input  = token.transpose(1,2).transpose(0,1)
        # Positional emb. 對「通道 index」做
        c_pos    = self.channel_pos_embedding(
                     torch.arange(self.token_dim, device=c_input.device)
                  ).unsqueeze(1)                                   # (5,1,S)
        c_input  = c_input + c_pos
        # 加 CLS
        ch_cls   = self.channel_cls_token.expand(-1, B, -1)        # (1,B,S)
        c_in     = torch.cat([ch_cls, c_input], 0)                 # (5+1,B,S)
        c_out    = self.modality_encoder(c_in)                     # (6,B,S)
        c_cls    = c_out[0]                                        # (B,S)

        # ============================================================
        # 3) Cross-stream fusion  (維度統一到 fusion_dim)
        # ============================================================
        t_vec = self.t_proj(t_cls)                                 # (B,D)
        c_vec = self.c_proj(c_cls)                                 # (B,D)

        seq_tokens = torch.stack([t_vec, c_vec], 0)                # (2,B,D)
        fusion_cls = self.fusion_cls_token.expand(-1, B, -1)       # (1,B,D)

        attn_out, _ = self.cross_attn(fusion_cls, seq_tokens, seq_tokens)  # (1,B,D)
        attn_out = attn_out.squeeze(0)                             # (B,D)

        # Pre-LN + FFN
        y  = self.norm1(attn_out + fusion_cls.squeeze(0))
        y2 = self.mlp_fuse(y)
        fused = self.norm2(y + y2)

        # Head
        out = self.fc_out(fused)                                   # (B,out_dim)
        return out , fused

# -------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = DMSnet(seq_length=30, out_dim=2).to(device)

    B, S = 2, 30
    dummy = {
        # 影像張量可不填，先給 shape 佔位
        'origin_face': torch.zeros(B,S,1,224,224, device=device),
        'left_eye'   : torch.zeros(B,S,1,112,112, device=device),
        'right_eye'  : torch.zeros(B,S,1,112,112, device=device),
        'head_pose'  : torch.zeros(B,S,3,         device=device),
        # 主要用得到的三項
        'gaze' : torch.randn(B,S,2, device=device),
        'ear'  : torch.rand (B,S,2, device=device),
        'mar'  : torch.rand (B,S,1, device=device),
    }
    out = model(dummy)
    print('Output shape =>', out.shape)   #  (B,2)
