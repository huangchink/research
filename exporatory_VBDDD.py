import os, glob, cv2
import torch
import numpy as np
from model.twoeyenet_transformer_1054 import DGMnet
from torchvision import transforms
from PIL import Image
import mediapipe as mp

# 定義 EAR 所需的 landmark indices
LEFT_EYE_EAR_INDICES = [362, 380, 374, 263, 386, 385]
RIGHT_EYE_EAR_INDICES = [33, 159, 158, 133, 153, 145]
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
    return (A + B + C) / (2.0 * D) if D > 0 else 0.0



def _eye_aspect_ratio(face_landmarks, eye_indices, w, h):
    """
    使用原始 EAR 算法：
    A = ||p2 - p6||, B = ||p3 - p5||, C = ||p1 - p4||
    EAR = (A + B) / (2*C)
    """
    pts = []
    for idx in eye_indices:
        x = face_landmarks.landmark[idx].x * w
        y = face_landmarks.landmark[idx].y * h
        pts.append([x, y])
    pts = np.array(pts)
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    ear = (A + B) / (2.0 * C) if C > 0 else 0
    return ear


def detect_face(image_array, face_detection):
    results = face_detection.process(image_array)
    if results.detections:
        d = results.detections[0]
        bbox = d.location_data.relative_bounding_box
        h, w, _ = image_array.shape
        xmin = max(0, int(bbox.xmin * w)); ymin = max(0, int(bbox.ymin * h))
        width = int(bbox.width * w); height = int(bbox.height * h)
        return image_array[ymin:ymin+height, xmin:xmin+width]
    return None


def _process_frame_for_eyes(frame, facetransform, eyetransform):
    face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                refine_landmarks=True, min_detection_confidence=0.5)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_crop = detect_face(rgb, face_detection)
    face_detection.close()
    if face_crop is None:
        face_mesh.close(); return None

    results = face_mesh.process(face_crop)
    face_mesh.close()
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0]
    h, w, _ = face_crop.shape

    # 裁切眼部 ROI
    def crop(idy1, idy2, idx1, idx2):
        y1 = int(lm.landmark[idy1].y * h); y2 = int(lm.landmark[idy2].y * h)
        x1 = int(lm.landmark[idx1].x * w);  x2 = int(lm.landmark[idx2].x * w)
        return face_crop[y1:y2, x1:x2]
    left_raw = crop(27, 23, 130, 243)
    right_raw= crop(257,253,463,359)

    # 計算 headpose (roll, yaw, pitch)
    le = ((lm.landmark[133].x+lm.landmark[33].x)/2, (lm.landmark[145].y+lm.landmark[159].y)/2)
    re = ((lm.landmark[362].x+lm.landmark[263].x)/2, (lm.landmark[374].y+lm.landmark[386].y)/2)
    nose = (lm.landmark[1].x, lm.landmark[1].y)
    ml = (lm.landmark[61].x, lm.landmark[61].y)
    mr = (lm.landmark[291].x, lm.landmark[291].y)
    lep = (int(le[0]*w), int(le[1]*h)); rep = (int(re[0]*w), int(re[1]*h))
    np_point = (int(nose[0]*w), int(nose[1]*h))
    mlp = (int(ml[0]*w), int(ml[1]*h)); mrp = (int(mr[0]*w), int(mr[1]*h))
    pt = [lep[1], rep[1], np_point[1], mlp[1], mrp[1], lep[0], rep[0], np_point[0], mlp[0], mrp[0]]
    roll = pt[1] - pt[0]
    yaw = (pt[2] - pt[5]) - (pt[5] - pt[2])
    eye_y = (pt[0] + pt[1]) / 2; mou_y = (pt[3] + pt[4]) / 2
    pitch = (eye_y - pt[2]) / ((pt[2] - mou_y) + 1e-6)
    headpose = [roll, yaw, pitch]

    # 使用原始 EAR 計算
    left_eye_ear = _eye_aspect_ratio(lm, LEFT_EYE_EAR_INDICES, w, h)
    right_eye_ear = _eye_aspect_ratio(lm, RIGHT_EYE_EAR_INDICES, w, h)


    mar = _mouth_aspect_ratio(lm, w, h)

    ear = (left_eye_ear, right_eye_ear)

    # transforms
    face_img = Image.fromarray(face_crop)
    left_img = Image.fromarray(left_raw)
    right_img = Image.fromarray(right_raw)
    return facetransform(face_img), eyetransform(left_img), eyetransform(right_img), headpose, ear ,mar


def analyze_subject_from_videos(root_dir, subjects, ckpt_path, batch_size=32, device='cpu'):
    facetransform = transforms.Compose([
        transforms.Resize([224,224]), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    eyetransform = transforms.Compose([
        transforms.Resize([112,112]), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    device = torch.device(device)
    model = DGMnet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    data = {'normal':{'hp':[], 'ear':[], 'gaze':[], 'mar':[]}, 'drowsy':{'hp':[], 'ear':[], 'gaze':[] , 'mar':[]}}

    videos = []
    for s in subjects:
        videos += glob.glob(os.path.join(root_dir, f"{s}*.avi"))
    if not videos:
        print("找不到影片"); return

    with torch.no_grad():
        for vid in videos:
            print(vid)
            label = 'normal' if 'driving' in os.path.basename(vid).lower() else 'drowsy'
            cap = cv2.VideoCapture(vid)
            buf, hpbuf, earbuf,marbuf = [], [], [],[]
            while True:
                ret, frame = cap.read()
                if not ret: break
                res = _process_frame_for_eyes(frame, facetransform, eyetransform)
                if res is None: continue
                f, l, r, hp, er,mar = res
                buf.append((f, l, r)); hpbuf.append(hp); earbuf.append(er) ;marbuf.append(mar)
                if len(buf) >= batch_size:
                    of = torch.stack([x[0] for x in buf]).to(device)
                    le = torch.stack([x[1] for x in buf]).to(device)
                    re = torch.stack([x[2] for x in buf]).to(device)
                    gz = model({'origin_face': of, 'left_eye': le, 'right_eye': re})[0].cpu().numpy()
                    data[label]['hp'].append(np.array(hpbuf)); data[label]['ear'].append(np.array(earbuf)); data[label]['mar'].append(np.array(marbuf)[:,np.newaxis]); data[label]['gaze'].append(gz)
                    buf.clear(); hpbuf.clear(); earbuf.clear() ;marbuf.clear()
            if buf:
                of = torch.stack([x[0] for x in buf]).to(device)
                le = torch.stack([x[1] for x in buf]).to(device)
                re = torch.stack([x[2] for x in buf]).to(device)
                gz = model({'origin_face': of, 'left_eye': le, 'right_eye': re})[0].cpu().numpy()
                data[label]['hp'].append(np.array(hpbuf)); data[label]['ear'].append(np.array(earbuf)); data[label]['mar'].append(np.array(marbuf)[:,np.newaxis]); data[label]['gaze'].append(gz)
            cap.release()

    def stats(arrs):
        x = np.vstack(arrs)
        return x.mean(axis=0), x.std(axis=0)

    for lbl in ['normal', 'drowsy']:
        mu_hp, sd_hp = stats(data[lbl]['hp'])
        mu_er, sd_er = stats(data[lbl]['ear'])
        mu_gz, sd_gz = stats(data[lbl]['gaze'])
        mu_gmar, sd_gmar = stats(data[lbl]['mar'])

        print(f"=== {lbl.upper()} ===")
        print(f"Headpose μ={mu_hp}, σ={sd_hp}")
        print(f"EAR      μ={mu_er}, σ={sd_er}")
        print(f"Gaze     μ={mu_gz}, σ={sd_gz}\n")
        print(f"Mar      μ={mu_gmar}, σ={sd_gmar}\n")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir', required=True)
    p.add_argument('--subjects', type=str, nargs='+', default=[f'subject{i}' for i in range(1,38)],
                        help='List of subject names to use')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    analyze_subject_from_videos(a.root_dir, a.subjects, a.checkpoint, a.batch_size, a.device)
