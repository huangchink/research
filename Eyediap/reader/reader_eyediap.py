import numpy as np
import cv2 
import os
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image

def gazeto2d(gaze):
  yaw = np.arctan2(-gaze[0], -gaze[2])
  pitch = np.arcsin(-gaze[1])
  return np.array([yaw, pitch])

class loader(Dataset): 
    def __init__(self, path, root, header=True):
        self.lines = []
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    lines = f.readlines()
                    if header:
                        lines.pop(0)
                    self.lines.extend(lines)
        else:
            with open(path) as f:
                self.lines = f.readlines()
                if header:
                    self.lines.pop(0)        
        self.root = root
        self.facetransform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.eyetransform = transforms.Compose([
            transforms.Resize([112,112]),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")

        face = line[0]
        lefteye = line[1]
        righteye = line[2]
        name = line[3]
        gaze2d = line[6]

        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        fimg = Image.open(os.path.join(self.root, face))
        rimg = Image.open(os.path.join(self.root, lefteye))
        limg = Image.open(os.path.join(self.root, righteye))

        fimg = self.facetransform(fimg)
        rimg = self.eyetransform(rimg)
        limg = self.eyetransform(limg)

        img = {"face": fimg,
               "lefteyeimg": limg,
               "righteyeimg": rimg,
               "name": name}

        return img, label



def txtload(labelpath, imagepath, batch_size, shuffle=True, num_workers=0, header=True):
  dataset = loader(labelpath, imagepath, header)
  print(f"[Read Data]: Total num: {len(dataset)}")
  print(f"[Read Data]: Label path: {labelpath}")
  load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  return load


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 設定標籤路徑與影像根目錄 (請自行確認資料夾結構是否正確)
    label_path = "/home/tchuang/research/Eyediap/Data_norm/Label_crossval"
    image_root = "/home/tchuang/research/Eyediap/Data_norm/Image"  # 確定你的影像放在這裡
    all_train_labels = sorted([os.path.join(label_path, f) 
                               for f in os.listdir(label_path) if f.endswith(".label")])
    for fold in range(4):
        leave_out_file = all_train_labels[fold]
        base_name = os.path.basename(leave_out_file)
        train_labels = [f for i, f in enumerate(all_train_labels) if i != fold]
        print(train_labels)
        #train_dataset = txtload(train_labels, image_root,16, shuffle=True, num_workers=4, header=True)
        val_dataset = txtload(leave_out_file, image_root, 16, num_workers=4, header=True)



