import numpy as np
import scipy.io as sio
import cv2 
import os
import sys
sys.path.append("../core/")
import data_processing_core as dpc


root = "/home/cyh/GazeDataset20200519/Original/EyeDiap/Data"
out_root = "/home/cyh/GazeDataset20200519/FaceBased/EyeDiap_temp"
scale =False

def ImageProcessing_Diap():
    folders = os.listdir(root)
    folders.sort(key=lambda x:int(x.split("_")[0]))
    
    count_dict = {}
    for i in range(20):
        count_dict[str(i)] = 0

    for folder in folders:
        if "FT" not in folder:
            video_path = os.path.join(root, folder, "rgb_vga.mov")
            head_path = os.path.join(root, folder, "head_pose.txt")
            anno_path = os.path.join(root, folder, "eye_tracking.txt")
            camparams_path = os.path.join(root, folder, "rgb_vga_calibration.txt")
            target_path = os.path.join(root, folder, "screen_coordinates.txt")
            
            number = int(folder.split("_")[0])
            count = count_dict[str(number)]
            person = "p" + str(number)

            im_outpath = os.path.join(out_root, "Image", person)
            label_outpath = os.path.join(out_root, "Label", f"{person}.label")

            if not os.path.exists(im_outpath):
                os.makedirs(im_outpath)
            if not os.path.exists(os.path.join(out_root, "Label")):
                os.makedirs(os.path.join(out_root, "Label"))
            if not os.path.exists(label_outpath):
                with open(label_outpath, 'w') as outfile:
                    outfile.write("Face Left Right metapath 3DGaze 3DHead 2DGaze 2DHead Rvec Svec GazeOrigin\n")

            print(f"Start Processing p{number}: {folder}")
            count_dict[str(number)] = ImageProcessing_PerVideos(video_path, head_path,anno_path, camparams_path, target_path, im_outpath, label_outpath, folder, count, person)


def ImageProcessing_PerVideos(video_path, head_path, anno_path, camparams_path, target_path, im_outpath, label_outpath, folder, count, person):
   
    # Read annotations
    with open(head_path) as infile:
        head_info = infile.readlines()
    with open(anno_path) as infile:
        anno_info = infile.readlines()
    with open(target_path) as infile:
        target_info = infile.readlines()
    length = len(target_info) - 1

    # Read camera parameters
    cam_info = CamParamsDecode(camparams_path)
    camera = cam_info["intrinsics"]
    cam_rot = cam_info["R"]
    cam_trans = cam_info["T"]*1000

    # Read video
    cap = cv2.VideoCapture(video_path)

    # create handle of label
    outfile = open(label_outpath, 'a')
    if not os.path.exists(os.path.join(im_outpath, "left")):
        os.makedirs(os.path.join(im_outpath, "left"))
    
    if not os.path.exists(os.path.join(im_outpath, "right")):
        os.makedirs(os.path.join(im_outpath, "right"))

    if not os.path.exists(os.path.join(im_outpath, "face")):
        os.makedirs(os.path.join(im_outpath, "face"))

    num = 1
    # Image Processing 
    for index in range(1, length+1):
        ret, frame = cap.read()

        if (index-1) % 15 != 0:
            continue

        progressbar = "".join(["\033[41m%s\033[0m" % '   '] * int(index/length * 20))
        progressbar = "\r" + progressbar + f" {index}|{length}"
        print(progressbar, end="", flush=True)

        # Calculate rotation and transition of head pose.
        head = head_info[index]
        head = list(map(eval, head.strip().split(";")))
        if len(head) != 13:
            print("[Error Head]")
            continue
        
        head_rot = head[1:10]
        head_rot = np.array(head_rot).reshape([3,3])
        head1 = cv2.Rodrigues(head_rot)[0].T[0]
        head2d = dpc.HeadTo2d(head1)
        print(head2d, end="------")

        head_rot = np.dot(cam_rot, head_rot) 
        head1 = cv2.Rodrigues(head_rot)[0].T[0]
        head2d = dpc.HeadTo2d(head1)
        print(head2d, end="------")
        exit()

            # rotate the head into camera coordinate system
        head_trans = np.array(head[10:13])*1000
        head_trans = np.dot(cam_rot, head_trans)
        head_trans = head_trans + cam_trans

        # Calculate the 3d coordinates of origin.
        anno = anno_info[index]
        anno = list(map(eval, anno.strip().split(";")))
        if len(anno) != 19:
            print("[Error anno]")
            continue
        anno = np.array(anno)

        left3d = anno[13:16]*1000
        left3d = np.dot(cam_rot, left3d) + cam_trans
        right3d = anno[16:19]*1000
        right3d = np.dot(cam_rot, right3d) + cam_trans

        face3d = (left3d + right3d)/2
        face3d = (face3d + head_trans)/2

        left2d = anno[1:3]
        right2d = anno[3:5]

        # Calculate the 3d coordinates of target
        target = target_info[index]
        target = list(map(eval,target.strip().split(";")))
        if len(target) != 6:
            print("[Error target]")
            continue
        target3d = np.array(target)[3:6]*1000

        # target3d = target3d.T - cam_trans
        # target3d = np.dot(np.linalg.inv(cam_rot), target3d)

        # Normalize the left eye image
        norm = dpc.norm(center = face3d,
                        gazetarget = target3d,
                        headrotvec = head_rot,
                        imsize = (224, 224),
                        camparams = camera)

        # Acquire essential info
        im_face = norm.GetImage(frame)
        gaze = norm.GetGaze(scale=scale)
        head = norm.GetHeadRot(vector=False)
        head = cv2.Rodrigues(head)[0].T[0]

        origin = norm.GetCoordinate(face3d)
        rvec, svec = norm.GetParams()

        gaze2d = dpc.GazeTo2d(gaze)
        head2d = dpc.HeadTo2d(head)

        # Crop Eye Image
        left2d = norm.GetNewPos(left2d)
        right2d = norm.GetNewPos(right2d)
        #   im_face = ipt.circle(im_face, left2d, 2)
        #im_face = ipt.circle(im_face, [left2d[0]+10, left2d[1]], 2)
        # cv2.imwrite("eye.jpg", im_face)
        
        
        im_left = norm.CropEyeWithCenter(left2d)
        im_left = dpc.EqualizeHist(im_left)
        im_right = norm.CropEyeWithCenter(right2d)
        im_right = dpc.EqualizeHist(im_right)

        # Save the acquired info
        cv2.imwrite(os.path.join(im_outpath, "face", str(count + num)+".jpg"), im_face)
        cv2.imwrite(os.path.join(im_outpath, "left", str(count + num)+".jpg"), im_left)
        cv2.imwrite(os.path.join(im_outpath, "right", str(count + num)+".jpg"), im_right)
        
        save_name_face = os.path.join(person, "face", str(count + num) + ".jpg")
        save_name_left = os.path.join(person, "left", str(count + num) + ".jpg")
        save_name_right = os.path.join(person, "right", str(count + num) + ".jpg")
        save_metapath = folder + f"_{index}"
        # save_flag = "left"
        save_gaze = ",".join(gaze.astype("str"))
        save_head = ",".join(head.astype("str"))
        save_gaze2d = ",".join(gaze2d.astype("str"))
        save_head2d = ",".join(head2d.astype("str"))
        save_origin = ",".join(origin.astype("str"))
        save_rvec = ",".join(rvec.astype("str"))
        save_svec = ",".join(svec.astype("str"))

        save_str = " ".join([save_name_face, save_name_left, save_name_right, save_metapath, save_gaze, save_head, save_gaze2d, save_head2d, save_rvec, save_svec, save_origin]) 
        outfile.write(save_str + "\n")
        num += 1

    count += (num-1)
    outfile.close()
    print("")
    return count

def CamParamsDecode(path):
    cal = {}
    fh = open(path, 'r')
    # Read the [resolution] section
    fh.readline().strip()
    cal['size'] = [int(val) for val in fh.readline().strip().split(';')]
    cal['size'] = cal['size'][0], cal['size'][1]
    # Read the [intrinsics] section
    fh.readline().strip()
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(';')])
    cal['intrinsics'] = np.array(vals).reshape(3, 3)
    # Read the [R] section
    fh.readline().strip()
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(';')])
    cal['R'] = np.array(vals).reshape(3, 3)
    # Read the [T] section
    fh.readline().strip()
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(';')])
    cal['T'] = np.array(vals).reshape(3)
    fh.close()
    return cal


if __name__ == "__main__":
    ImageProcessing_Diap()
