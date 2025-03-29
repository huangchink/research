import numpy as np
import cv2 
import os
from torch.utils.data import Dataset, DataLoader
import torch
import mediapipe as mp

# 初始化 MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

def extract_eyes(face_image):
    """從輸入的 face_image 中提取左眼和右眼圖像"""
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            # 如果無法檢測到臉部，回傳空陣列
            return np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((0, 0, 3), dtype=np.uint8)

        # 提取臉部特徵點
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = face_image.shape

        try:
            # 提取左眼區域
            left_eye_img = face_image[
                int(h * face_landmarks.landmark[27].y):int(h * face_landmarks.landmark[23].y),
                int(w * face_landmarks.landmark[130].x):int(w * face_landmarks.landmark[243].x)
            ]

            # 提取右眼區域
            right_eye_img = face_image[
                int(h * face_landmarks.landmark[257].y):int(h * face_landmarks.landmark[253].y),
                int(w * face_landmarks.landmark[463].x):int(w * face_landmarks.landmark[359].x)
            ]
        except Exception as e:
            print(f"Error extracting eyes: {e}")
            # 返回空圖像以避免程序崩潰
            left_eye_img = None
            right_eye_img = None

        return left_eye_img, right_eye_img


# 修改 Dataset 類
class loader(Dataset): 
    def __init__(self, path, root, header=True, train=True):
        self.lines = []
        self.valid_samples = []  # 用於存放成功提取眼睛的樣本
        self.root = root
        self.train = train

        # 讀取所有資料
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

        # 過濾有效樣本
        self._filter_valid_samples()

    def _filter_valid_samples(self):
        """過濾掉無法成功提取眼睛的樣本"""
        with open('/home/tchuang/research/XGAZE/xgaze_train/Labels/Labels_faceeye', "w") as filter_file:
            for idx, line in enumerate(self.lines):

                try:
                    line = line.strip().split(" ")
                    face_path = line[0]

                    file_path = face_path.split("/")[0] 
                    file_name = face_path.split("/")[-1]  # 提取 '1.jpg'
                    file_id   = file_name.split(".")[0]  # 提取 '1'


                    face_img = cv2.imread(os.path.join(self.root, face_path))
                    # 嘗試讀取臉部圖像
                    fimg = cv2.imread(os.path.join(self.root, face_path)) 
                    fimg = fimg.astype(np.float32)

                    # 嘗試提取眼睛
                    left_eye, right_eye = extract_eyes((fimg).astype(np.uint8))
                    if left_eye.size == 0 or right_eye.size == 0:
                        print(f"Skipping sample {idx}: Unable to extract eyes")
                        continue

                    # 如果成功提取，記錄該樣本
                    # self.valid_samples.append(line)
                    debug_dir = '/home/tchuang/research/XGAZE/xgaze_train/face_eye/'

                    # 保存眼睛圖像到 debug 資料夾
                    left_eye_path  = os.path.join(debug_dir, file_id+"_left.jpg")
                    right_eye_path = os.path.join(debug_dir, file_id+"_right.jpg")
                    face_path      = os.path.join(debug_dir, file_id+"_face.jpg")
                    right_eye = cv2.resize(right_eye, (112, 112))
                    left_eye = cv2.resize(left_eye, (112, 112))

                    # 保存圖片
                    cv2.imwrite(left_eye_path, left_eye)
                    cv2.imwrite(right_eye_path, right_eye)
                    cv2.imwrite(face_path, face_img)
                    filter_file.write(" ".join(line) + "\n")  # 寫入 Label_filter.txt

                    #print(f"Saved left eye to {left_eye_path}")
                    #print(f"Saved right eye to {right_eye_path}")
                    #print(f"Saved face to {face_path}")


                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    continue

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        line = self.valid_samples[idx]
        face_path = line[0]
        gaze2d = line[1]
        head2d = line[2]

        # 讀取標籤
        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        headpose = np.array(head2d.split(",")).astype("float")
        headpose = torch.from_numpy(headpose).type(torch.FloatTensor)

        # 讀取臉部圖像
        fimg = cv2.imread(os.path.join(self.root, face_path)) 
        fimg = fimg.astype(np.float32)

        # 提取左眼和右眼
        left_eye, right_eye = extract_eyes((fimg).astype(np.uint8))
        rimg = cv2.resize(right_eye, (112, 112))
        limg = cv2.resize(left_eye, (112, 112))

        # 處理臉部

        img = {
            "left":  limg,
            "right": rimg,
            "face":  fimg,
            "head_pose": headpose,
            "name": face_path,
        }

        return img, label
def gazeto2d(gaze):
  yaw = np.arctan2(-gaze[0], -gaze[2])
  pitch = np.arcsin(-gaze[1])
  return np.array([yaw, pitch])

def txtload(labelpath, imagepath, batch_size, train=True, num_workers=0, header=True):
  dataset = loader(labelpath, imagepath, header, train)
  print(f"[Read Data]: Total num: {len(dataset)}")
  print(f"[Read Data]: Label path: {labelpath}")
  load = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)
  return load
if __name__ == "__main__":
    # 設定測試資料
    path = '/home/tchuang/research/XGAZE/xgaze_train/train/train.label'
    root = '/home/tchuang/research/XGAZE/xgaze_train/image_for_pretrain'
    debug_dir = '/home/tchuang/research/XGAZE/xgaze_train/face_eye/'
    os.makedirs(debug_dir, exist_ok=True)

    # 初始化資料集
    dataset = loader(path, root, header=True, train=True)
    print(f"Valid Dataset size: {len(dataset)}")

    # 測試並保存眼睛圖片
    for idx in range(len(dataset)):
        try:
            sample_img, sample_label = dataset[idx]

            # 打印樣本資訊
            print(f"Processing sample {idx + 1}/{len(dataset)}")
            print("Face shape:", sample_img["face"].shape)
            print("Left eye shape:", sample_img["left"].shape)
            print("Right eye shape:", sample_img["right"].shape)
            print("Gaze label:", sample_label)

            # 保存眼睛圖像到 debug 資料夾
            left_eye_path = os.path.join(debug_dir, f"{idx + 1}_left.jpg")
            right_eye_path = os.path.join(debug_dir, f"{idx + 1}_right.jpg")
            face_path = os.path.join(debug_dir, f"{idx + 1}_face.jpg")

            # 將圖像轉換為原始格式 (HWC)
            left_eye_img = (sample_img["left"]).astype(np.uint8)
            right_eye_img = (sample_img["right"]).astype(np.uint8)
            face_img = (sample_img["face"]).astype(np.uint8)

            # 保存圖片
            cv2.imwrite(left_eye_path, left_eye_img)
            cv2.imwrite(right_eye_path, right_eye_img)
            cv2.imwrite(face_path, face_img)

            print(f"Saved left eye to {left_eye_path}")
            print(f"Saved right eye to {right_eye_path}")
            print(f"Saved face to {face_path}")

        except Exception as e:
            print(f"Error processing sample {idx + 1}: {e}")
            continue

    print("Debug images saved to ./debug")






