# dataset_NTHUDDD_DGM.py

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import mediapipe as mp
from torchvision import transforms
from PIL import Image

# --- helper functions & constants ---
def detect_face(image_array, face_detection):
    results = face_detection.process(image_array)
    if results.detections:
        d = results.detections[0].location_data.relative_bounding_box
        h, w, _ = image_array.shape
        x, y = int(d.xmin * w), int(d.ymin * h)
        bw, bh = int(d.width * w), int(d.height * h)
        x, y = max(0, x), max(0, y)
        return image_array[y:y+bh, x:x+bw], (x, y, bw, bh)
    return None, None

def _eye_aspect_ratio(landmarks, indices, w, h):
    pts = np.array([[landmarks.landmark[i].x * w,
                     landmarks.landmark[i].y * h] for i in indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0

LEFT_EYE_INDICES  = [362, 380, 374, 263, 386, 385]
RIGHT_EYE_INDICES = [33, 159, 158, 133, 153, 145]

# MAR landmark indices (p1..p8)
MOUTH_INDICES = [61, 291, 39, 181, 0, 17, 269, 405]
def _mouth_aspect_ratio(landmarks, w, h):
    pts = [(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in MOUTH_INDICES]
    p1, p5 = np.array(pts[0]), np.array(pts[1])
    p2, p8 = np.array(pts[2]), np.array(pts[3])
    p3, p7 = np.array(pts[4]), np.array(pts[5])
    p4, p6 = np.array(pts[6]), np.array(pts[7])
    A = np.linalg.norm(p2 - p8)
    B = np.linalg.norm(p3 - p7)
    C = np.linalg.norm(p4 - p6)
    D = np.linalg.norm(p1 - p5)
    return (A + B + C) / (2.0 * D) if D > 0 else 0

class NTHUDDDDataset(Dataset):
    def __init__(self,
                 root_dir,                # e.g. ".../NTHU_modify/training" or ".../NTHU_modify/testing"
                 sequence_length=30,
                 stride=30,
                 frame_skip=3,
                 max_sequences_per_video=100):
        self.seq_len = sequence_length
        self.stride = stride
        self.skip = frame_skip
        self.max_seqs = max_sequences_per_video

        # transforms (grayscale input)
        self.face_tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])
        ])
        self.eye_tf = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])
        ])

        # 直接讀 NTHU_modify 下的 normal(0) / drowsiness(1)
        self.video_infos = []
        for label_name, lbl in [("normal", 0), ("drowsiness", 1)]:
            folder = os.path.join(root_dir, label_name)
            if not os.path.isdir(folder):
                continue
            for vid_path in glob.glob(os.path.join(folder, "*.avi")) + \
                             glob.glob(os.path.join(folder, "*.mp4")):
                self.video_infos.append((vid_path, lbl))

        # 預處理，生成所有序列
        self.data = self._prepare_data()

    def _prepare_data(self):
        # Mediapipe detectors
        face_det = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5)

        sequences = []
        for vid_path, vid_lbl in self.video_infos:
            cap = cv2.VideoCapture(vid_path)
            faces, lefts, rights = [], [], []
            heads, ears, mars = [], [], []
            idxs = []
            cnt = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cnt += 1
                if cnt % self.skip != 0:
                    continue

                img, _ = detect_face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), face_det)
                if img is None:
                    continue

                res = face_mesh.process(img)
                if not res.multi_face_landmarks:
                    continue
                lm = res.multi_face_landmarks[0]

                h, w, _ = img.shape
                # left eye crop
                l_y1, l_y2 = int(h * lm.landmark[27].y), int(h * lm.landmark[23].y)
                l_x1, l_x2 = int(w * lm.landmark[130].x), int(w * lm.landmark[243].x)
                # right eye crop
                r_y1, r_y2 = int(h * lm.landmark[257].y), int(h * lm.landmark[253].y)
                r_x1, r_x2 = int(w * lm.landmark[463].x), int(w * lm.landmark[359].x)

                left_crop  = img[l_y1:l_y2, l_x1:l_x2]
                right_crop = img[r_y1:r_y2, r_x1:r_x2]

                # head pose 5-point
                le_c = ((lm.landmark[133].x + lm.landmark[33].x)/2 * w,
                        (lm.landmark[145].y + lm.landmark[159].y)/2 * h)
                re_c = ((lm.landmark[362].x + lm.landmark[263].x)/2 * w,
                        (lm.landmark[374].y + lm.landmark[386].y)/2 * h)
                nose = (lm.landmark[1].x * w, lm.landmark[1].y * h)
                ml   = (lm.landmark[61].x * w, lm.landmark[61].y * h)
                mr   = (lm.landmark[291].x * w, lm.landmark[291].y * h)
                pts5 = [le_c[0], re_c[0], nose[0], ml[0], mr[0],
                        le_c[1], re_c[1], nose[1], ml[1], mr[1]]
                roll  = pts5[6] - pts5[5]
                yaw   = (pts5[2] - pts5[0]) - (pts5[1] - pts5[2])
                eye_y = (pts5[5] + pts5[6]) / 2
                mou_y = (pts5[8] + pts5[9]) / 2
                pitch = (eye_y - pts5[7]) / (pts5[7] - mou_y) if (pts5[7] - mou_y) != 0 else 0

                # EAR
                er = (_eye_aspect_ratio(lm, LEFT_EYE_INDICES,  w, h),
                      _eye_aspect_ratio(lm, RIGHT_EYE_INDICES, w, h))

                # MAR
                mar = _mouth_aspect_ratio(lm, w, h)

                # append
                faces.append(self.face_tf(Image.fromarray(img)))
                lefts.append(self.eye_tf(Image.fromarray(left_crop)))
                rights.append(self.eye_tf(Image.fromarray(right_crop)))
                heads.append(torch.tensor([roll, yaw, pitch], dtype=torch.float))
                ears.append(torch.tensor(er, dtype=torch.float))
                mars.append(torch.tensor([mar], dtype=torch.float))
                idxs.append(cnt-1)

            cap.release()

            # 如果不夠一整段就跳過
            if len(faces) < self.seq_len:
                continue

            seq_count = 0
            # 滑動窗切片
            for start in range(0, len(faces) - self.seq_len + 1, self.stride):
                seq_faces = torch.stack(faces[start:start+self.seq_len])
                seq_left  = torch.stack(lefts[start:start+self.seq_len])
                seq_right = torch.stack(rights[start:start+self.seq_len])
                seq_head  = torch.stack(heads[start:start+self.seq_len])
                seq_ear   = torch.stack(ears[start:start+self.seq_len])
                seq_mar   = torch.stack(mars[start:start+self.seq_len])

                sequences.append((seq_faces, seq_left, seq_right,
                                  seq_head, seq_ear, seq_mar, vid_lbl))
                seq_count += 1
                if seq_count >= self.max_seqs:
                    break

        face_mesh.close()
        face_det.close()
        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        f, l, r, h, e, m, lbl = self.data[idx]
        return {
            "origin_face": f,
            "left_eye":    l,
            "right_eye":   r,
            "head_pose":   h,
            "ear":         e,
            "mar":         m
        }, lbl

# --- usage example ---
if __name__ == '__main__':
    train_ds = NTHUDDDDataset(
        root_dir="/home/remote/tchuang/research/NTHUDDD/NTHU_modify/training",
        sequence_length=30, stride=30, frame_skip=1
    )
    test_ds  = NTHUDDDDataset(
        root_dir="/home/remote/tchuang/research/NTHUDDD/NTHU_modify/testing",
        sequence_length=30, stride=30, frame_skip=1
    )
    print("train videos:", len(train_ds.video_infos), "→ sequences:", len(train_ds))
    print("test  videos:", len(test_ds.video_infos),  "→ sequences:", len(test_ds))
