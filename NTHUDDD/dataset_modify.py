# dataset_modify.py

import os
import glob
import cv2

def segment_videos(root_dirs, out_root="NTHU_modify", eval=False):
    """
    将 root_dirs 下所有视频，按 *_drowsiness.txt 标签切分成全 0 / 全 1 的片段，
    并分别存到 out_root/normal 和 out_root/drowsiness 目录。
    """
    out_normal = os.path.join(out_root, "normal")
    out_drowsy = os.path.join(out_root, "drowsiness")
    os.makedirs(out_normal, exist_ok=True)
    os.makedirs(out_drowsy, exist_ok=True)

    for root_dir in root_dirs:
        for subj in os.listdir(root_dir):
            subj_dir = os.path.join(root_dir, subj)
            if not os.path.isdir(subj_dir):
                continue

            if eval:
                # EvaluationDataset: 视频和 txt 直接在 subj_dir
                video_paths = []
                for ext in ("*.avi", "*.mp4"):
                    video_paths += glob.glob(os.path.join(subj_dir, ext))

                for vid_path in video_paths:
                    basename = os.path.splitext(os.path.basename(vid_path))[0]
                    print(f"Found eval video: {vid_path}")

                    # 取去掉 _mix 之前的前缀来匹配标签
                    prefix = basename.split("_mix")[0]
                    pattern = os.path.join(subj_dir, f"{prefix}*_drowsiness.txt")
                    lbl_files = glob.glob(pattern)
                    if not lbl_files:
                        print(f"  ↳ no label for {basename}, skip")
                        continue
                    label_file = lbl_files[0]
                    print(f"  ↳ using label: {label_file}")

                    _process_and_save(subj, basename, vid_path, label_file,
                                      out_normal, out_drowsy)

            else:
                # TrainingDataset: 视频在 subj_dir/<scenario>/ 下
                for scen in os.listdir(subj_dir):
                    scen_dir = os.path.join(subj_dir, scen)
                    if not os.path.isdir(scen_dir):
                        continue

                    video_paths = []
                    for ext in ("*.avi", "*.mp4"):
                        video_paths += glob.glob(os.path.join(scen_dir, ext))

                    for vid_path in video_paths:
                        basename = os.path.splitext(os.path.basename(vid_path))[0]
                        print(f"Found train video: {vid_path}")

                        # label 文件名为 "<subj>_<basename>_drowsiness.txt"
                        label_file = os.path.join(
                            scen_dir,
                            f"{subj}_{basename}_drowsiness.txt"
                        )
                        if not os.path.exists(label_file):
                            print(f"  ↳ no label for {basename}, skip")
                            continue
                        print(f"  ↳ using label: {label_file}")

                        _process_and_save(subj, basename, vid_path, label_file,
                                          out_normal, out_drowsy)


def _process_and_save(subj, basename, vid_path, label_file, out_normal, out_drowsy):
    # 1) 读取标签，只保留 '0'/'1'
    with open(label_file, 'r') as f:
        raw = f.read()
    labels = [int(c) for c in raw if c in ('0', '1')]
    if not labels:
        print("  ↳ empty labels, skip")
        return

    # 2) 读入所有帧
    cap = cv2.VideoCapture(vid_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # 3) 找到标签中的连续段
    segments = []
    curr = labels[0]
    start = 0
    for i, lbl in enumerate(labels):
        if lbl != curr:
            segments.append((start, i-1, curr))
            start = i
            curr = lbl
    segments.append((start, len(labels)-1, curr))

    # 4) 导出每一段
    for s, e, lbl in segments:
        dst_dir = out_drowsy if lbl == 1 else out_normal
        # 在文件名前加上 subject，避免跨 subject 覆写
        out_name = f"{subj}_{basename}_{s}_{e}.avi"
        out_path = os.path.join(dst_dir, out_name)
        if os.path.exists(out_path):
            continue

        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        end_frame = min(e, len(frames)-1)
        for fr in frames[s:end_frame+1]:
            writer.write(fr)
        writer.release()
        print(f"    ◦ saved: {out_path}")


if __name__ == "__main__":
    # 处理训练集
    train_roots = [
        "/home/remote/tchuang/research/NTHUDDD/Training_Evaluation_Dataset/TrainingDataset"
    ]
    segment_videos(train_roots, out_root="NTHU_modify/training", eval=False)

    # 处理评估集
    eval_roots = [
        "/home/remote/tchuang/research/NTHUDDD/Training_Evaluation_Dataset/EvaluationDataset"
    ]
    segment_videos(eval_roots, out_root="NTHU_modify/testing", eval=True)
