#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import cv2
import torch
import numpy as np
from PIL import Image
import mediapipe as mp
from torchvision import transforms

# --- helper functions ---

def detect_face(image_array, face_detection):
    results = face_detection.process(image_array)
    if results.detections:
        d = results.detections[0].location_data.relative_bounding_box
        h, w, _ = image_array.shape
        xmin = max(0, int(d.xmin * w))
        ymin = max(0, int(d.ymin * h))
        bw   = int(d.width * w)
        bh   = int(d.height * h)
        return image_array[ymin:ymin+bh, xmin:xmin+bw], (xmin, ymin, bw, bh)
    return None, None

def _eye_aspect_ratio(landmarks, indices, w, h):
    pts = np.array([[landmarks.landmark[i].x * w,
                     landmarks.landmark[i].y * h] for i in indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0

LEFT_EYE_INDICES  = [362, 380, 374, 263, 386, 385]
RIGHT_EYE_INDICES = [33, 159, 158, 133, 153, 145]
MOUTH_INDICES = [61, 291, 39, 181, 0, 17, 269, 405]

# transforms
face_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])
eye_tf = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])
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
    return (A + B + C) / (2.0 * D) if D > 0 else 0.0
def _process_frame(frame, face_det, face_mesh):
    # 1) detect face
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_crop, _ = detect_face(rgb, face_det)
    if face_crop is None:
        return None
    # 2) face mesh
    res = face_mesh.process(face_crop)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0]
    h, w, _ = face_crop.shape

    # 左右眼 crop（同原始方法）
    ly1, ly2 = int(h * lm.landmark[27].y), int(h * lm.landmark[23].y)
    lx1, lx2 = int(w * lm.landmark[130].x), int(w * lm.landmark[243].x)
    ry1, ry2 = int(h * lm.landmark[257].y), int(h * lm.landmark[253].y)
    rx1, rx2 = int(w * lm.landmark[463].x), int(w * lm.landmark[359].x)
    left_eye  = face_crop[ly1:ly2, lx1:lx2]
    right_eye = face_crop[ry1:ry2, rx1:rx2]

    # headpose 五點計算
    le_c = ((lm.landmark[133].x+lm.landmark[33].x)/2*w,
            (lm.landmark[145].y+lm.landmark[159].y)/2*h)
    re_c = ((lm.landmark[362].x+lm.landmark[263].x)/2*w,
            (lm.landmark[374].y+lm.landmark[386].y)/2*h)
    nose = (lm.landmark[1].x*w, lm.landmark[1].y*h)
    ml   = (lm.landmark[61].x*w, lm.landmark[61].y*h)
    mr   = (lm.landmark[291].x*w, lm.landmark[291].y*h)
    pts5 = [le_c[0], re_c[0], nose[0], ml[0], mr[0],
            le_c[1], re_c[1], nose[1], ml[1], mr[1]]
    roll = pts5[6] - pts5[5]
    yaw  = (pts5[2]-pts5[0]) - (pts5[1]-pts5[2])
    eye_y = (pts5[5]+pts5[6])/2
    mou_y = (pts5[8]+pts5[9])/2
    pitch = (eye_y-pts5[7])/(pts5[7]-mou_y) if (pts5[7]-mou_y)!=0 else 0

    # EAR
    er_l = _eye_aspect_ratio(lm, LEFT_EYE_INDICES,  w, h)
    er_r = _eye_aspect_ratio(lm, RIGHT_EYE_INDICES, w, h)

    # to tensor
    t_face  = face_tf(Image.fromarray(face_crop))
    t_leye  = eye_tf(Image.fromarray(left_eye))
    t_reye  = eye_tf(Image.fromarray(right_eye))
    t_head  = torch.tensor([roll, yaw, pitch], dtype=torch.float)
    t_ear   = torch.tensor([er_l, er_r], dtype=torch.float)
    mar_val = _mouth_aspect_ratio(lm, w, h)
    mar_val = torch.tensor([mar_val], dtype=torch.float)

    return t_face, t_leye, t_reye, t_head, t_ear ,mar_val

def process_video(vid_path, label, out_dir,
                  seq_len=30, skip=3, stride=15, max_seqs=24):
    os.makedirs(out_dir, exist_ok=True)
    face_det  = mp.solutions.face_detection.FaceDetection(1,0.5)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(vid_path)
    buf = []  # 暫存單幀結果
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        if frame_idx % skip != 0:
            continue
        out = _process_frame(frame, face_det, face_mesh)
        if out is not None:
            buf.append(out)
    cap.release()
    face_mesh.close(); face_det.close()

    # 切片並儲存
    seq_i = 0
    for start in range(0, len(buf)-seq_len+1, stride):
        if seq_i >= max_seqs:
            break
        seq = buf[start:start+seq_len]
        faces, leyes, reyes, heads, ears ,mars= zip(*seq)
        bundle = {
            "face":  torch.stack(faces),
            "left":  torch.stack(leyes),
            "right": torch.stack(reyes),
            "head":  torch.stack(heads),
            "ear":   torch.stack(ears),
            "mar":   torch.stack(mars),
            "label": label
        }
        base = os.path.splitext(os.path.basename(vid_path))[0]
        torch.save(bundle, os.path.join(
            out_dir, f"{base}_seq{seq_i}.pt"
        ))
        seq_i += 1

if __name__ == "__main__":
    ROOT      = "/home/remote/tchuang/research/VBDDD/VBDDD_dataset"
    OUT_DIR   = "/home/remote/tchuang/research/VBDDD/preprocessed_10_mar"
    SUBJECTS  = [f'subject{i}' for i in range(1,38)]
    for sub in SUBJECTS:
        for vid in glob.glob(os.path.join(ROOT, f"{sub}*.avi")):
            lbl = 0 if "driving" in vid.lower() else 1
            print(f"Processing {vid} → {lbl}")
            process_video(
                vid_path=vid,
                label=lbl,
                out_dir=OUT_DIR,
                seq_len=10,
                skip=3,
                stride=10,
                max_seqs=240000
            )
    print("VBDDD 離線預處理完成！")
