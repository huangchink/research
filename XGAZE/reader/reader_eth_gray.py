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
  def __init__(self, path, root, header=True, train=True):
    self.lines = []
    if isinstance(path, list):
      for i in path:
        with open(i) as f:
          line = f.readlines()
          if header: line.pop(0)
          self.lines.extend(line)
    else:
      with open(path) as f:
        self.lines = f.readlines()
        if header: self.lines.pop(0)

    self.root = '/home/tchuang/research/XGAZE/xgaze_train/face_eye'
    self.train = train
    self.facetransform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.Grayscale(num_output_channels=1),  # 轉成單通道灰階圖
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    self.eyetransform = transforms.Compose([
        transforms.Resize([112, 112]),
        transforms.Grayscale(num_output_channels=1),  # 轉成單通道灰階圖
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    face = line[0]
    name = line[0]
    gaze2d = line[1]
    head2d = line[1]

    if self.train:
        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)
        head2d = line[2]

    headpose = np.array(head2d.split(",")).astype("float")
    headpose = torch.from_numpy(headpose).type(torch.FloatTensor)




    file_path = face.split("/")[0] 
    file_name = face.split("/")[-1]  # 提取 '1.jpg'
    file_stem = file_name.split(".")[0]  # 提取 '1'
    face=file_stem+'_face.jpg'
    left_eye=file_stem+'_left.jpg'
    right_eye=file_stem+'_right.jpg'
    #print(os.path.join(self.root,face))

    fimg = Image.open(os.path.join(self.root,face))
    rimg = Image.open(os.path.join(self.root, right_eye))
    limg = Image.open(os.path.join(self.root, left_eye))
    fimg = self.facetransform(fimg)
    rimg = self.eyetransform(rimg)
    limg = self.eyetransform(limg)

    # img = {"face":torch.from_numpy(fimg).type(torch.FloatTensor),
    #         "head_pose": headpose,
    #         "name":name}
    

    img = {"left"     :limg,
           "right"    :rimg,
           "face"     :fimg,
           "head_pose":headpose,
           "name":name}
    if self.train:
        return img, label
    else:
        return img

def txtload(labelpath, imagepath, batch_size, train=True, num_workers=0, header=True):
  dataset = loader(labelpath, imagepath, header, train)
  print(f"[Read Data]: Total num: {len(dataset)}")
  print(f"[Read Data]: Label path: {labelpath}")
  load = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)
  return load


if __name__ == "__main__":
  path = './p00.label'
  d = loader(path)
  print(len(d))
  (data, label) = d.__getitem__(0)

