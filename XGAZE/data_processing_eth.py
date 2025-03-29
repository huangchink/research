import os
import h5py
import numpy as np
import cv2

path = "/home/tchuang/research/XGAZE/xgaze_train/test/"
imo_path = "/home/tchuang/research/XGAZE/xgaze_train/image_for_pretrain_test/"
annoo_path = "/home/tchuang/research/XGAZE/xgaze_train/test/test.label"
test = True

if not os.path.exists(imo_path):
    os.makedirs(imo_path)

if not os.path.exists(os.path.dirname(annoo_path)):
    os.makedirs(os.path.dirname(annoo_path))

if not test:
    with open(annoo_path, 'w') as outfile:
        outfile.write("face gaze head origin cam_index frame_index normmat\n")
else:
    with open(annoo_path, 'w') as outfile:
        outfile.write("face head origin cam_index frame_index normmat\n")


def process_person(h5files_path, imo_metapath, sub_id, annoo_metapath, begin_num, test):
    datas = h5py.File(h5files_path, 'r')
    if not test:
        keys = ["cam_index", "face_gaze", "face_head_pose",
            "face_mat_norm", "face_patch", "frame_index"]
    else:
        keys = ["cam_index", "face_head_pose",
            "face_mat_norm", "face_patch", "frame_index"]
    length = datas[keys[0]].shape[0]
    print(f"==> Length: {length}")

    imo_path = os.path.join(imo_metapath, sub_id)
    if not os.path.exists(imo_path):
        os.makedirs(imo_path)

    with open(annoo_metapath, 'a') as outfile:
        for i in range(length):
            img = datas["face_patch"][i,:]
            cv2.imwrite(os.path.join(imo_path, f"{begin_num}.jpg"), img)

            im_path = os.path.join(sub_id, f"{begin_num}.jpg")
            head = ",".join(list(datas["face_head_pose"][i, :].astype("str")))
            norm_mat = ",".join(list(datas["face_mat_norm"][i, :].astype("str").flatten()))
            cam_index = ",".join(list(datas["cam_index"][i, :].astype("str")))
            frame_index = ",".join(list(datas["frame_index"][i, :].astype("str")))
            if not test: 
                gaze = ",".join(list(datas["face_gaze"][i, :].astype("str")))
                outfile.write(f"{im_path} {gaze} {head} {os.path.join(sub_id, str(i)+'.jpg')} {cam_index} {frame_index} {norm_mat}\n")
            else: 
                outfile.write(f"{im_path} {head} {os.path.join(sub_id, str(i)+'.jpg')} {cam_index} {frame_index} {norm_mat}\n")
            begin_num += 1
    datas.close()
    return begin_num

filenames = os.listdir(path)
filenames.sort()
num = 1

for count, filename in enumerate(filenames):
    print(f"Processing.. {filename}, [{count}/{len(filenames)}]")
    sub_id = filename.split(".")[0]
    file_path = os.path.join(path, filename)
    num = process_person(file_path, 
        imo_path,
        sub_id,
        annoo_path,
        num,
        test)
