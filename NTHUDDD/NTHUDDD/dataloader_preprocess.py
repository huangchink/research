# offline_dataset.py

import os
import glob
import torch
from torch.utils.data import Dataset

class OfflineSequenceDataset(Dataset):
    def __init__(self, seq_dir):
        # 讀取所有 .pt 檔案路徑
        all_files = glob.glob(os.path.join(seq_dir, "*.pt"))
        # 過濾掉檔名含有 "_yawning_" 的檔案
        self.files = [
            f for f in all_files
            if "sunglasses" not in os.path.basename(f)
        ]
        # （可選）排序，保證每次順序一致
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        bundle = torch.load(self.files[idx])
        data = {
            "origin_face": bundle["face"],
            "left_eye":    bundle["left"],
            "right_eye":   bundle["right"],
            "head_pose":   bundle["head"],
            "ear":         bundle["ear"],
            "mar":         bundle["mar"],
            "gaze":        bundle["gaze"]
        }
        return data, bundle["label"]
