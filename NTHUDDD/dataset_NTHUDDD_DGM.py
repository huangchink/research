# nthuddd_dataset.py

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# --- visualization function ---
def visualize_sequence(face_seq, left_seq, right_seq, head_seq, ear_seq, label, frame_indices=None):
    T = face_seq.size(0)
    idxs = frame_indices or list(range(min(5, T)))
    fig, axes = plt.subplots(5, len(idxs), figsize=(3*len(idxs), 10))
    for i, ix in enumerate(idxs):
        img_f = face_seq[ix].permute(1,2,0).squeeze().cpu().numpy()
        axes[0, i].imshow(img_f, cmap='gray'); axes[0, i].axis('off'); axes[0, i].set_title(f'Face {ix}')
        img_l = left_seq[ix].permute(1,2,0).squeeze().cpu().numpy()
        axes[1, i].imshow(img_l, cmap='gray'); axes[1, i].axis('off'); axes[1, i].set_title(f'L-eye {ix}')
        img_r = right_seq[ix].permute(1,2,0).squeeze().cpu().numpy()
        axes[2, i].imshow(img_r, cmap='gray'); axes[2, i].axis('off'); axes[2, i].set_title(f'R-eye {ix}')
        hp = head_seq[ix].cpu().numpy()
        txt = f"roll:{hp[0]:.2f}\nyaw:{hp[1]:.2f}\npitch:{hp[2]:.2f}"
        axes[3, i].axis('off'); axes[3, i].text(0.5, 0.5, txt, ha='center', va='center'); axes[3, i].set_title('HeadPose')
        er = ear_seq[ix].cpu().numpy()
        txt2 = f"L_EAR:{er[0]:.2f}\nR_EAR:{er[1]:.2f}"
        axes[4, i].axis('off'); axes[4, i].text(0.5, 0.5, txt2, ha='center', va='center'); axes[4, i].set_title('EAR')
    plt.suptitle(f'Label = {label}', fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

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
    pts = np.array([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0

LEFT_EYE_INDICES = [362, 380, 374, 263, 386, 385]
RIGHT_EYE_INDICES = [33, 159, 158, 133, 153, 145]

class NTHUDDDDataset(Dataset):
    def __init__(self,
                 root_dir,
                 sequence_length=30,
                 stride=30,
                 max_sequences_per_video=100,
                 frame_skip=3,
                 eval=False):
        self.seq_len = sequence_length
        self.stride = stride
        self.max_seqs = max_sequences_per_video
        self.skip = frame_skip
        self.eval = eval

        # transforms for grayscale (single channel)
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

        # only nonsleepyCombination & sleepyCombination for training
        self.valid = None if eval else {"nonsleepyCombination", "sleepyCombination"}
        self.video_infos = []

        # gather video-label pairs
        for subj in os.listdir(root_dir):
            subj_dir = os.path.join(root_dir, subj)
            if not os.path.isdir(subj_dir):
                continue

            if self.eval:
                # in eval, videos and labels sit under subj_dir, skip any sunglasses files
                for ext in ('*.avi','*.mp4'):
                    for vid in glob.glob(os.path.join(subj_dir, ext)):
                        base = os.path.splitext(os.path.basename(vid))[0]
                        if 'sunglass' in base.lower():  # skip sunglasses
                            continue
                        pattern = os.path.join(subj_dir, f"{base}*_drowsiness.txt")
                        for lbl in glob.glob(pattern):
                            if 'sunglass' in os.path.basename(lbl).lower():
                                continue
                            print(vid)
                            print(lbl)
                            self.video_infos.append((vid, lbl))
            else:
                # in train, videos under scenario subfolders, skip any 'sunglass' folder
                for scen in os.listdir(subj_dir):
                    if 'sunglass' in scen.lower():
                        continue
                    scen_dir = os.path.join(subj_dir, scen)
                    if not os.path.isdir(scen_dir):
                        continue
                    for ext in ('*.avi','*.mp4'):
                        for vid in glob.glob(os.path.join(scen_dir, ext)):
                            base = os.path.splitext(os.path.basename(vid))[0]
                            if base not in self.valid:
                                continue
                            lbl = os.path.join(scen_dir, f"{subj}_{base}_drowsiness.txt")
                            if os.path.exists(lbl):
                                print(vid)
                                print(lbl)
                                self.video_infos.append((vid, lbl))

        # prepare sequences
        self.data = self._prepare_data()

    def _prepare_data(self):
        face_det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        sequences = []

        for vid, label_path in self.video_infos:
            with open(label_path, 'r') as f:
                raw = f.read()
            labels = [int(c) for c in raw if c in ('0','1')]

            cap = cv2.VideoCapture(vid)
            faces, lefts, rights, heads, ears, idxs = [], [], [], [], [], []
            cnt = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cnt += 1
                if cnt % self.skip != 0:
                    continue

                img = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) if frame.ndim < 3 else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                crop, _ = detect_face(img, face_det)
                if crop is None:
                    continue

                res = face_mesh.process(crop)
                if not res.multi_face_landmarks:
                    continue
                lm = res.multi_face_landmarks[0]

                h, w, _ = crop.shape
                # eye crops
                left_crop = crop[int(h*lm.landmark[27].y):int(h*lm.landmark[23].y), int(w*lm.landmark[130].x):int(w*lm.landmark[243].x)]
                right_crop = crop[int(h*lm.landmark[257].y):int(h*lm.landmark[253].y), int(w*lm.landmark[463].x):int(w*lm.landmark[359].x)]

                # headpose (5-point)
                le_c = ((lm.landmark[133].x+lm.landmark[33].x)/2*w, (lm.landmark[145].y+lm.landmark[159].y)/2*h)
                re_c = ((lm.landmark[362].x+lm.landmark[263].x)/2*w, (lm.landmark[374].y+lm.landmark[386].y)/2*h)
                nose = (lm.landmark[1].x*w, lm.landmark[1].y*h)
                ml = (lm.landmark[61].x*w, lm.landmark[61].y*h)
                mr = (lm.landmark[291].x*w, lm.landmark[291].y*h)
                pts5 = [le_c[0], re_c[0], nose[0], ml[0], mr[0], le_c[1], re_c[1], nose[1], ml[1], mr[1]]
                roll = pts5[6] - pts5[5]
                yaw  = (pts5[2] - pts5[0]) - (pts5[1] - pts5[2])
                eye_y = (pts5[5] + pts5[6]) / 2
                mou_y = (pts5[8] + pts5[9]) / 2
                pitch = (eye_y - pts5[7]) / (pts5[7] - mou_y) if (pts5[7] - mou_y) != 0 else 0
                hp = [roll, yaw, pitch]

                # EAR
                er = (_eye_aspect_ratio(lm, LEFT_EYE_INDICES, w, h), _eye_aspect_ratio(lm, RIGHT_EYE_INDICES, w, h))

                faces.append(self.face_tf(Image.fromarray(crop)))
                lefts.append(self.eye_tf(Image.fromarray(left_crop)))
                rights.append(self.eye_tf(Image.fromarray(right_crop)))
                heads.append(torch.tensor(hp, dtype=torch.float))
                ears.append(torch.tensor(er, dtype=torch.float))
                idxs.append(cnt-1)

            cap.release()
            if len(faces) < self.seq_len:
                continue

            proc_lbl = [labels[i] if i < len(labels) else 0 for i in idxs]
            seq_count = 0
            for start in range(0, len(faces)-self.seq_len+1, self.stride):
                segment = proc_lbl[start:start+self.seq_len]
                if len(set(segment)) != 1:
                    continue
                sequences.append((
                    torch.stack(faces[start:start+self.seq_len]),
                    torch.stack(lefts[start:start+self.seq_len]),
                    torch.stack(rights[start:start+self.seq_len]),
                    torch.stack(heads[start:start+self.seq_len]),
                    torch.stack(ears[start:start+self.seq_len]),
                    segment[0]
                ))
                seq_count += 1
                if seq_count >= self.max_seqs:
                    break

        face_mesh.close()
        face_det.close()
        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        f, l, r, h, e, lbl = self.data[idx]
        return {"origin_face": f, "left_eye": l, "right_eye": r, "head_pose": h, "ear": e}, lbl

if __name__ == '__main__':
    train_root = '/home/remote/tchuang/research/NTHUDDD/Training_Evaluation_Dataset/Training Dataset'
    eval_root  = '/home/remote/tchuang/research/NTHUDDD/Training_Evaluation_Dataset/Evaluation Dataset'
    train_ds = NTHUDDDDataset(train_root, sequence_length=30, stride=30, frame_skip=3, eval=False)
    eval_ds  = NTHUDDDDataset(eval_root,  sequence_length=30, stride=30, frame_skip=3, eval=True)
    print('Train videos:', len(train_ds.video_infos), 'sequences:', len(train_ds))
    print('Eval  videos:', len(eval_ds.video_infos),  'sequences:', len(eval_ds))
    if len(train_ds):
        sample, lbl = train_ds[0]
        visualize_sequence(
            sample['origin_face'],
            sample['left_eye'],
            sample['right_eye'],
            sample['head_pose'],
            sample['ear'],
            lbl
        )
    if len(eval_ds):
        sample, lbl = eval_ds[0]
        visualize_sequence(
            sample['origin_face'],
            sample['left_eye'],
            sample['right_eye'],
            sample['head_pose'],
            sample['ear'],
            lbl
        )
