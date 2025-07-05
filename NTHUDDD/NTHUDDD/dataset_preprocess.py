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

# === 導入 DGMnet gaze 模型 ===
from model.twoeyenet_transformer_1054_gray import DGMnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gaze_model = DGMnet()
gaze_model.load_state_dict(torch.load(
    "/home/tchuang/research/NTHUDDD/loadmodel/eyetracking_gray.pt", map_location=device))
gaze_model.eval().to(device)
for p in gaze_model.parameters():
    p.requires_grad = False

# --- EAR / MAR 計算用 landmark index ---
LEFT_EYE_INDICES  = [362, 380, 374, 263, 386, 385]
RIGHT_EYE_INDICES = [33, 159, 158, 133, 153, 145]
MOUTH_INDICES = [61, 291, 39, 181, 0, 17, 269, 405]

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
    pts = np.array([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0

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

# --- transforms (灰階) ---
face_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])
eye_tf = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

def process_video(vid_path, label, out_dir,
                  seq_len=30, skip=3, stride=30):
    os.makedirs(out_dir, exist_ok=True)

    face_det  = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                refine_landmarks=True, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(vid_path)
    faces, lefts, rights = [], [], []
    heads, ears, mars, gazes = [], [], [], []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % skip != 0:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img, _ = detect_face(img_rgb, face_det)
        if img is None:
            continue

        res = face_mesh.process(img)
        if not res.multi_face_landmarks:
            continue
        lm = res.multi_face_landmarks[0]
        h, w, _ = img.shape

        # --- 眼睛裁切 ---
        l_y1, l_y2 = int(h * lm.landmark[27].y), int(h * lm.landmark[23].y)
        l_x1, l_x2 = int(w * lm.landmark[130].x), int(w * lm.landmark[243].x)
        r_y1, r_y2 = int(h * lm.landmark[257].y), int(h * lm.landmark[253].y)
        r_x1, r_x2 = int(w * lm.landmark[463].x), int(w * lm.landmark[359].x)
        left_crop  = img[l_y1:l_y2, l_x1:l_x2]
        right_crop = img[r_y1:r_y2, r_x1:r_x2]

        # --- Head pose ---
        le_c = ((lm.landmark[133].x + lm.landmark[33].x)/2 * w,
                (lm.landmark[145].y + lm.landmark[159].y)/2 * h)
        re_c = ((lm.landmark[362].x + lm.landmark[263].x)/2 * w,
                (lm.landmark[374].y + lm.landmark[386].y)/2 * h)
        nose = (lm.landmark[1].x * w, lm.landmark[1].y * h)
        ml   = (lm.landmark[61].x * w, lm.landmark[61].y * h)
        mr   = (lm.landmark[291].x * w, lm.landmark[291].y * h)
        pts5 = [le_c[0], re_c[0], nose[0], ml[0], mr[0],
                le_c[1], re_c[1], nose[1], ml[1], mr[1]]
        roll = pts5[6] - pts5[5]
        yaw  = (pts5[2] - pts5[0]) - (pts5[1] - pts5[2])
        eye_y = (pts5[5] + pts5[6]) / 2
        mou_y = (pts5[8] + pts5[9]) / 2
        pitch = (eye_y - pts5[7]) / (pts5[7] - mou_y) if abs(pts5[7] - mou_y) > 1e-3 else 0.0

        # --- EAR / MAR ---
        er_left  = _eye_aspect_ratio(lm, LEFT_EYE_INDICES,  w, h)
        er_right = _eye_aspect_ratio(lm, RIGHT_EYE_INDICES, w, h)
        mar_val  = _mouth_aspect_ratio(lm, w, h)

        # --- Gaze 預測 ---
        face_tensor  = face_tf(Image.fromarray(img)).unsqueeze(0).to(device)
        left_tensor  = eye_tf(Image.fromarray(left_crop)).unsqueeze(0).to(device)
        right_tensor = eye_tf(Image.fromarray(right_crop)).unsqueeze(0).to(device)
        x_img = {"origin_face": face_tensor, "left_eye": left_tensor, "right_eye": right_tensor}
        with torch.no_grad():
            gaze_out = gaze_model(x_img)[1]
        gaze = gaze_out.squeeze(0).cpu()

        # --- 收集資料 ---
        faces.append(face_tensor.squeeze(0).cpu())
        lefts.append(left_tensor.squeeze(0).cpu())
        rights.append(right_tensor.squeeze(0).cpu())
        heads.append(torch.tensor([roll, yaw, pitch], dtype=torch.float))
        ears.append(torch.tensor([er_left, er_right], dtype=torch.float))
        mars.append(torch.tensor([mar_val], dtype=torch.float))
        gazes.append(gaze)

    cap.release()
    face_mesh.close()
    face_det.close()

    seq_idx = 0
    for start in range(0, len(faces) - seq_len + 1, stride):
        bundle = {
            "face":  torch.stack(faces[start:start+seq_len]),
            "left":  torch.stack(lefts[start:start+seq_len]),
            "right": torch.stack(rights[start:start+seq_len]),
            "head":  torch.stack(heads[start:start+seq_len]),
            "ear":   torch.stack(ears[start:start+seq_len]),
            "mar":   torch.stack(mars[start:start+seq_len]),
            "gaze":  torch.stack(gazes[start:start+seq_len]),  # ✅ 新增 gaze
            "label": label
        }
        fn = os.path.splitext(os.path.basename(vid_path))[0]
        out_path = os.path.join(out_dir, f"{fn}_seq{seq_idx}.pt")
        torch.save(bundle, out_path)
        seq_idx += 1

if __name__ == "__main__":
    TRAIN_ROOT = "/home/tchuang/research/NTHUDDD/NTHU_segment/training"
    TEST_ROOT  = "/home/tchuang/research/NTHUDDD/NTHU_segment/testing"
    OUT_TRAIN_DIR = "/home/tchuang/research/NTHUDDD/preprocessed_nosunglasses_nonight_gaze_pt_120_stride3_aug/train"
    OUT_TEST_DIR  = "/home/tchuang/research/NTHUDDD/preprocessed_nosunglasses_nonight_gaze_pt_120_stride3_aug/test"

    for cls_name, lbl in [("normal", 0), ("drowsiness", 1)]:
        vids = glob.glob(os.path.join(TRAIN_ROOT, cls_name, "*.avi")) + \
               glob.glob(os.path.join(TRAIN_ROOT, cls_name, "*.mp4"))
        for vid in vids:
            # if "sunglasses" in vid:
            #     continue
            if cls_name =="normal":   
                print(f"[Train] {vid}")
                process_video(vid, lbl, OUT_TRAIN_DIR, seq_len=120, skip=3, stride=60) #data augmentation
            else:
                print(f"[Train] {vid}")
                process_video(vid, lbl, OUT_TRAIN_DIR, seq_len=120, skip=3, stride=120)

    for cls_name, lbl in [("normal", 0), ("drowsiness", 1)]:
        vids = glob.glob(os.path.join(TEST_ROOT, cls_name, "*.avi")) + \
               glob.glob(os.path.join(TEST_ROOT, cls_name, "*.mp4"))
        for vid in vids:
            # if "sunglasses" in vid:
            #     continue
            print(f"[Test] {vid}")
            process_video(vid, lbl, OUT_TEST_DIR, seq_len=120, skip=3, stride=120)

    # print("✅ Gaze-enhanced .pt 資料預處理完成")
