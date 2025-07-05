#!/usr/bin/env python3
import os
import glob
import random
import cv2
import numpy as np
import mediapipe as mp
from model.twoeyenet_transformer_1054_gray import DGMnet  # 假設 DGMnet 定義在此模組中
import torch
import math
from torchvision import transforms
from PIL import Image

# --- 偵測人臉並回傳裁切區域 + bounding box ---
def detect_face(img_rgb, face_detector):
    results = face_detector.process(img_rgb)
    if results.detections:
        bb = results.detections[0].location_data.relative_bounding_box
        h, w, _ = img_rgb.shape
        x, y = int(bb.xmin * w), int(bb.ymin * h)
        bw, bh = int(bb.width * w), int(bb.height * h)
        x, y = max(0, x), max(0, y)
        return img_rgb[y:y+bh, x:x+bw], (x, y, bw, bh)
    return None, None

# --- EAR helper ---
def eye_aspect_ratio(lm, indices, w, h):
    pts = np.array([[lm.landmark[i].x * w, lm.landmark[i].y * h] for i in indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0

LEFT_EYE_INDICES  = [362, 380, 374, 263, 386, 385]
RIGHT_EYE_INDICES = [33, 159, 158, 133, 153, 145]

# --- MAR helper ---
MOUTH_INDICES = [61, 291, 39, 181, 0, 17, 269, 405]
def mouth_aspect_ratio(lm, w, h):
    pts = [np.array([lm.landmark[i].x * w, lm.landmark[i].y * h]) for i in MOUTH_INDICES]
    p1, p5 = pts[0], pts[1]
    p2, p8 = pts[2], pts[3]
    p3, p7 = pts[4], pts[5]
    p4, p6 = pts[6], pts[7]
    A = np.linalg.norm(p2 - p8)
    B = np.linalg.norm(p3 - p7)
    C = np.linalg.norm(p4 - p6)
    D = np.linalg.norm(p1 - p5)
    return (A + B + C) / (2.0 * D) if D > 0 else 0

# --- gaze 2D→3D helper ---
def gazeto3d(gaze):
    gaze_3d = np.zeros(3)
    gaze_3d[0] = -math.cos(gaze[1]) * math.sin(gaze[0])
    gaze_3d[1] = -math.sin(gaze[1])
    gaze_3d[2] = -math.cos(gaze[1]) * math.cos(gaze[0])
    return gaze_3d

# --- 處理並輸出影片 ---
def process_video(video_path, out_path='output.avi'):
    # 載入模型
    DriverGaze = DGMnet()
    ckpt = '/home/tchuang/research/NTHUDDD/loadmodel/eyetracking_gray.pt'
    DriverGaze.load_state_dict(torch.load(ckpt, map_location='cpu'))
    DriverGaze.eval()

    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    # Transforms
    face_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485],[0.229])
    ])
    eye_tf = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485],[0.229])
    ])

    # Mediapipe
    face_det  = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
    )

    while True:
        ret, frame = cap.read()
        if not ret: break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop, bbox = detect_face(img_rgb, face_det)
        if crop is not None and bbox is not None:
            x, y, bw, bh = bbox
            res = face_mesh.process(crop)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0]
                hh, ww, _ = crop.shape

                # HeadPose
                le   = ((lm.landmark[133].x+lm.landmark[33].x)/2*ww,
                        (lm.landmark[145].y+lm.landmark[159].y)/2*hh)
                re   = ((lm.landmark[362].x+lm.landmark[263].x)/2*ww,
                        (lm.landmark[374].y+lm.landmark[386].y)/2*hh)
                nose = (lm.landmark[1].x*ww, lm.landmark[1].y*hh)
                ml   = (lm.landmark[61].x*ww, lm.landmark[61].y*hh)
                mr   = (lm.landmark[291].x*ww, lm.landmark[291].y*hh)
                roll = le[1] - re[1]
                yaw  = (nose[0]-le[0]) - (re[0]-nose[0])
                eye_y= (le[1]+re[1])/2
                mou_y= (ml[1]+mr[1])/2
                pitch = (eye_y - nose[1])/(nose[1]-mou_y) if (nose[1]!=mou_y) else 0

                # EAR / MAR
                er_l = eye_aspect_ratio(lm, LEFT_EYE_INDICES,  ww, hh)
                er_r = eye_aspect_ratio(lm, RIGHT_EYE_INDICES, ww, hh)
                mar  = mouth_aspect_ratio(lm, ww, hh)

                # Gaze 預測
                left_crop  = crop[int(hh*lm.landmark[27].y):int(hh*lm.landmark[23].y),
                                  int(ww*lm.landmark[130].x):int(ww*lm.landmark[243].x)]
                right_crop = crop[int(hh*lm.landmark[257].y):int(hh*lm.landmark[253].y),
                                  int(ww*lm.landmark[463].x):int(ww*lm.landmark[359].x)]
                face_pil = Image.fromarray(crop)
                left_eye_pil = Image.fromarray(left_crop)
                right_eye_pil = Image.fromarray(right_crop)
                feat = {
                    "origin_face": face_tf(face_pil).unsqueeze(0),
                    "left_eye":    eye_tf(left_eye_pil).unsqueeze(0),
                    "right_eye":   eye_tf(right_eye_pil).unsqueeze(0)
                }
                with torch.no_grad():
                    _, gaze_pred, *_ = DriverGaze(feat)
                gaze_pred = gaze_pred[0].cpu().numpy()
                vec3d = gazeto3d(gaze_pred)

                # 計算原始畫面上箭頭的座標
                center_crop = (bw//2, bh//2)
                center_abs  = (x + center_crop[0], y + center_crop[1])
                factor = 80
                end_pt = (int(center_abs[0] + vec3d[0]*factor),
                          int(center_abs[1] + vec3d[1]*factor))
                cv2.arrowedLine(frame, center_abs, end_pt, (0,0,255), 2, tipLength=0.2)

                # 疊加文字
                cv2.putText(frame,
                    f"L_EAR:{er_l:.2f} R_EAR:{er_r:.2f} MAR:{mar:.2f}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
                cv2.putText(frame,
                    f"roll:{roll:.2f} yaw:{yaw:.2f} pitch:{pitch:.2f}",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
                cv2.putText(frame,
                    f"gaze yaw:{gaze_pred[0]:.2f} pitch:{gaze_pred[1]:.2f}",
                    (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)
                cv2.putText(frame,
                    f"vec3d:[{vec3d[0]:.2f},{vec3d[1]:.2f},{vec3d[2]:.2f}]",
                    (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)

        writer.write(frame)
        cv2.imshow('EAR/MAR/HeadPose/Gaze', frame)
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    face_mesh.close()
    face_det.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    train_root = '/home/tchuang/research/NTHUDDD/Training_Evaluation_Dataset/TrainingDataset/035/night_noglasses/'
    vids = glob.glob(os.path.join(train_root, '**', 'yawning.avi'), recursive=True) \
         + glob.glob(os.path.join(train_root, '**', '*.mp4'), recursive=True)
    if not vids:
        raise RuntimeError(f"找不到影片，請確認 {train_root} 下 .avi/.mp4")
    choice = random.choice(vids)
    print(f"Processing: {choice}")
    process_video(choice, out_path='output.avi')
