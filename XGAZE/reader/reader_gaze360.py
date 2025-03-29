import numpy as np
import cv2 
import os
from torch.utils.data import Dataset, DataLoader
import torch

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
          line = f.readlines()
          if header: line.pop(0)
          self.lines.extend(line)
    else:
      with open(path) as f:
        self.lines = f.readlines()
        if header: self.lines.pop(0)

    self.root = root

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    face = line[0]
    lefteye = line[1]
    righteye = line[2]
    name = line[3]
    gaze2d = line[5]

    label = np.array(gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)


    # rimg = cv2.imread(os.path.join(self.root, righteye))/255.0
    # rimg = rimg.transpose(2, 0, 1)

    # limg = cv2.imread(os.path.join(self.root, lefteye))/255.0
    # limg = limg.transpose(2, 0, 1)

    
    fimg = cv2.imread(os.path.join(self.root, face))
    lefteyeimg = cv2.imread(os.path.join(self.root, lefteye))
    righteyeimg = cv2.imread(os.path.join(self.root, righteye))

    fimg = cv2.resize(fimg, (224, 224))/255.0
    fimg = fimg.transpose(2, 0, 1)
    lefteyeimg = cv2.resize(lefteyeimg, (112, 112))/255.0
    righteyeimg = cv2.resize(righteyeimg, (112, 112))/255.0
    lefteyeimg = lefteyeimg.transpose(2, 0, 1)
    righteyeimg = righteyeimg.transpose(2, 0, 1)

    img = {"face":torch.from_numpy(fimg).type(torch.FloatTensor),
           "lefteyeimg":torch.from_numpy(lefteyeimg).type(torch.FloatTensor),
           "righteyeimg":torch.from_numpy(righteyeimg).type(torch.FloatTensor),
            "name":name}


    # img = {"left":torch.from_numpy(limg).type(torch.FloatTensor),
    #        "right":torch.from_numpy(rimg).type(torch.FloatTensor),
    #        "face":torch.from_numpy(fimg).type(torch.FloatTensor),
    #        "head_pose":headpose,
    #        "name":name}

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
    label_path = '/home/tchuang/research/Gaze360-main/FaceBased/Label/train.label'
    image_root = '/home/tchuang/research/Gaze360-main/FaceBased/Images'  # 確定你的影像放在這裡

    # 建立 dataset 實例，需同時提供 path 與 root
    d = loader(path=label_path, root=image_root)
    print("Dataset length:", len(d))

    # 取出第一筆資料 (包含影像與標籤)
    data, label = d[0]

    # 將 face 影像的維度從 (C, H, W) 轉為 (H, W, C) 以符合 matplotlib 的顯示格式
    face_img = data["face"].numpy().transpose(1, 2, 0)

    # 視覺化 face 影像
    plt.imshow(face_img)
    plt.title("Face Image")
    plt.axis('off')
    plt.show()
    lefteyeimg = data["lefteyeimg"].numpy().transpose(1, 2, 0)
    plt.imshow(lefteyeimg)
    plt.title("lefteyeimg Image")
    plt.axis('off')
    plt.show()
    righteyeimg = data["righteyeimg"].numpy().transpose(1, 2, 0)
    plt.imshow(righteyeimg)
    plt.title("righteyeimg Image")
    plt.axis('off')
    plt.show()
    # 若需要視覺化 left eye 與 right eye，只需同樣取得並轉置後 show 即可
    # left_eye_img = data["lefteyeimg"].numpy().transpose(1, 2, 0)
    # right_eye_img = data["righteyeimg"].numpy().transpose(1, 2, 0)
    # plt.imshow(left_eye_img)
    # plt.title("Left Eye Image")
    # plt.axis('off')
    # plt.show()
