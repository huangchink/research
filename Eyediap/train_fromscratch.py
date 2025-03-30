import numpy as np
import random
import torch
import os
import time
import sys
import yaml
import importlib
import torch.optim as optim
import torch.nn as nn
from model.twoeyenet_transformer_1054 import DGMnet  # 使用 DGMnet
from model.warm.warmup_scheduler.scheduler import GradualWarmupScheduler
def load_pretrained_backbone(model, pretrained_path):
    print(f"Loading pretrained backbone from {pretrained_path}")
    pretrained_dict = torch.load(pretrained_path, map_location="cpu")
    print(f"Original {len(pretrained_dict)} layers from pretrained model.")

    model_dict = model.state_dict()
    # 篩選可用的權重
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"Loaded {len(pretrained_dict)} layers from pretrained model.")
# 設定隨機種子函式
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# gaze 與角度計算
def gazeto3d(gaze):
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt

def angular(gaze, label):
    total = np.sum(gaze * label)
    return np.arccos(min(total / (np.linalg.norm(gaze) * np.linalg.norm(label)), 0.9999999)) * 180 / np.pi

def evaluate_model(net, test_dataset, device):
    net.eval()
    accs = 0
    count = 0
    with torch.no_grad():
        for j, (data, label) in enumerate(test_dataset):
            data["origin_face"] = data["face"].to(device)
            data["left_eye"] = data["lefteyeimg"].to(device)
            data["right_eye"] = data["righteyeimg"].to(device)
            data["gaze_origin"] = torch.zeros(data["origin_face"].size(0), 3).to(device)

            label = label.to(device)
            gts = label.to(device)

            gaze, _, _, _, _ = net(data)
            for k, gaze_item in enumerate(gaze):
                gaze_item = gaze_item.cpu().detach().numpy()
                count += 1
                accs += angular(gazeto3d(gaze_item), gazeto3d(gts.cpu().numpy()[k]))
    
    avg_acc = accs / count
    print(f'Total Num: {count}, avg: {avg_acc}')
    return avg_acc

def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = sum(p.numel() for p in model.parameters()) / (1024 ** 2)

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Model Size (MB): {model_size:.2f}")

if __name__ == "__main__":
    config = yaml.load(open(f"{sys.argv[1]}"), Loader=yaml.FullLoader)
    readername = config["reader"]
    dataloader = importlib.import_module("reader." + readername)

    config = config["train"]
    imagepath = config["data"]["image"]
    labelpath = config["data"]["label"]
    modelname = config["save"]["model_name"]

    savepath = os.path.join(config["save"]["save_path"], f"checkpoint")
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Read data")

    # **獲取所有 label 檔案**
    all_labels = sorted([os.path.join(labelpath, f) for f in os.listdir(labelpath) if f.endswith(".label")])

    epoch_validation_errors=[]
    # **4-Fold Cross Validation**
    for fold in range(len(all_labels)):  # 總共 4 folds
        print(f"\n### Fold {fold+1}/{len(all_labels)}: Using {all_labels[fold]} as validation set ###")

        # **切分訓練與驗證集**
        train_labels = all_labels[:fold] + all_labels[fold+1:]
        val_label = all_labels[fold]

        train_dataset = dataloader.txtload(train_labels, imagepath, config["params"]["batch_size"], shuffle=True, num_workers=4, header=True)
        val_dataset = dataloader.txtload(val_label, imagepath, 16, num_workers=4, header=True)

        set_seed(3)  # 固定 seed

        print("Model building")
        net = DGMnet()

        # pretrained_path = "/home/tchuang/research/Eyediap/loadmodel/pretrain_model.pt"

        # 載入預訓練的 Backbone
        # load_pretrained_backbone(net, pretrained_path)
        print_model_size(net)
        net.to(device)
        print(f'device:{device}')

        print("Optimizer building")
        base_lr = config["params"]["lr"]
        decaysteps = config["params"]["decay_step"]
        decayratio = config["params"]["decay"]

        optimizer = optim.Adam(net.parameters(), lr=base_lr, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decaysteps, gamma=decayratio)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=scheduler)

        print("Training")
        for epoch in range(1, config["params"]["epoch"] + 1):
            net.train()
            epoch_loss = 0.0
            num_batches = 0
            current_lr = scheduler.get_last_lr()[0]

            for i, (data, label) in enumerate(train_dataset):
                # 數據準備
                data["origin_face"] = data["face"].to(device)
                data["left_eye"] = data["lefteyeimg"].to(device)
                data["right_eye"] = data["righteyeimg"].to(device)
                data["gaze_origin"] = torch.zeros(data["origin_face"].size(0), 3).to(device)
                label = label.to(device)
                
                # 前向傳播
                gaze, gaze1, gaze2, gaze3, gaze4 = net(data)

                # 計算 loss
                loss = nn.L1Loss()(gaze, label)
                loss1 = nn.L1Loss()(gaze1, label)
                loss2 = nn.L1Loss()(gaze2, label)
                loss3 = nn.L1Loss()(gaze3, label)
                loss4 = nn.L1Loss()(gaze4, label)

                batch_loss = 0.2 * loss + 0.2 * loss1 + 0.2 * loss2 + 0.2 * loss3 + 0.2 * loss4 
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                epoch_loss += batch_loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            scheduler.step()

            print(f"[Epoch {epoch}]: Training loss: {avg_loss:.4f}, lr: {current_lr}")
            
            if epoch == config["params"]["epoch"]:
                val_error = evaluate_model(net, val_dataset, device)
                print(f"Fold {fold}, Epoch {epoch}, validation error: {val_error:.2f}")
                epoch_validation_errors.append(val_error)

    avg_val_error = np.mean(epoch_validation_errors)


    print(f"\n===average validation error: {avg_val_error:.2f} degrees ===")
