#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import torch
from torch.utils.data import Dataset

class OfflineVBDDDDataset(Dataset):
    """
    Flat structure: all .pt 都放在同一層 preproc_root，
    透過 subjects 清單過濾檔名中包含該 subject 的檔案。
    """
    def __init__(self, preproc_root, subjects=None):
        """
        Args:
            preproc_root (str): 預處理後 .pt 全部放的資料夾
            subjects (list of str, optional): 只載入檔名中包含這些
                subject 字串的檔案；若為 None，則載入全部 .pt。
        """
        self.files = []
        if subjects:
            for subj in subjects:
                # 只抓檔名中含有 subj 的 .pt
                pattern = os.path.join(preproc_root, f"*{subj}*.pt")
                self.files += glob.glob(pattern)
            # 去重並排序
            self.files = sorted(set(self.files))
        else:
            # subjects None：全部 .pt
            pattern = os.path.join(preproc_root, "*.pt")
            self.files = sorted(glob.glob(pattern))

        if not self.files:
            raise ValueError(f"No .pt files found in {preproc_root} for subjects={subjects}")

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
            "mar":         bundle["mar"]

        }
        return data, bundle["label"]
