import numpy as np
import random
import torch
import os
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
    print(f"Model Size (M): {model_size:.2f}")

if __name__ == "__main__":
    # 讀取配置檔
    config = yaml.load(open(f"{sys.argv[1]}"), Loader=yaml.FullLoader)
    readername = config["reader"]
    dataloader = importlib.import_module("reader." + readername)

    config = config["train"]
    imagepath = config["data"]["image"]
    # config["data"]["label"] 為標籤資料夾的父目錄
    label_base_path = config["data"]["label"]

    # 設定訓練與驗證標籤的資料夾
    train_label_dir = os.path.join(label_base_path, "train")
    val_label_dir   = os.path.join(label_base_path, "test")

    # 讀取訓練資料夾中所有 .label 檔案（預期有 3 個）
    all_train_labels = sorted([os.path.join(train_label_dir, f) 
                               for f in os.listdir(train_label_dir) if f.endswith(".label")])
    if len(all_train_labels) != 3:
        raise ValueError("預期在 {} 中有 3 個訓練 label 檔案".format(train_label_dir))

    # 用於記錄所有 epoch (從第5 epoch 到最後 epoch) 每個 fold 的驗證誤差
    epoch_validation_errors = {}

    # 建立儲存模型的資料夾
    savepath = os.path.join(config["save"]["save_path"], "checkpoint")
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # 3-fold Cross Validation
    for fold in range(3):
        # 取出本 fold 捨棄的訓練 label 檔案
        leave_out_file = all_train_labels[fold]
        base_name = os.path.basename(leave_out_file)
        # 驗證檔案直接使用相同檔名：例如捨棄 train1.label，驗證用 /test/train1.label
        val_file = os.path.join(val_label_dir, base_name)
        
        # 訓練標籤檔為除去該 fold 留出的檔案
        train_labels = [f for i, f in enumerate(all_train_labels) if i != fold]

        print(f"\n### Fold {fold+1}/3: 使用訓練標籤 {train_labels}，驗證標籤 {val_file} ###")

        # 建立 dataset
        train_dataset = dataloader.txtload(train_labels, imagepath, 
                                             config["params"]["batch_size"], shuffle=True, num_workers=4, header=True)
        val_dataset = dataloader.txtload(val_file, imagepath, 48, num_workers=4, header=True)

        set_seed(3)  # 固定 seed

        print("Model building")
        net = DGMnet()
        print_model_size(net)
        pretrained_path = "/home/tchuang/research/Eyediap/loadmodel/pretrain_model_5_09.pt"

        # 載入預訓練的 Backbone
        load_pretrained_backbone(net, pretrained_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        print(f"device: {device}")

        print("Optimizer building")
        base_lr = config["params"]["lr"]
        decaysteps = config["params"]["decay_step"]
        decayratio = config["params"]["decay"]

        optimizer = optim.Adam(net.parameters(), lr=base_lr, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decaysteps, gamma=decayratio)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=config["params"]["warmup"], after_scheduler=scheduler)

        print("Training")
        for epoch in range(1, config["params"]["epoch"] + 1):
            net.train()
            epoch_loss = 0.0
            leyeloss_epoch = 0.0
            reyeloss_epoch = 0.0
            clstokenloss_epoch = 0.0
            num_batches = 0
            for i, (data, label) in enumerate(train_dataset):
                # 數據準備
                data["origin_face"] = data["face"].to(device)
                data["left_eye"] = data["lefteyeimg"].to(device)
                data["right_eye"] = data["righteyeimg"].to(device)
                data["gaze_origin"] = torch.zeros(data["origin_face"].size(0), 3).to(device)
                label = label.to(device)
                
                # 前向傳播
                gaze, gaze1, gaze2, gaze3, gaze4 = net(data)
                
                # 分別計算 loss
                loss = nn.L1Loss()(gaze, label)
                lowface_loss = nn.L1Loss()(gaze1, label)
                face_loss = nn.L1Loss()(gaze2, label)
                left_loss = nn.L1Loss()(gaze3, label)
                right_loss = nn.L1Loss()(gaze4, label)
                
                # 組合各部分 loss
                batch_loss = 0.2 * loss  + 0.2 * lowface_loss + 0.2 * face_loss+ 0.2 * left_loss + 0.2 * right_loss
                
                # 反向傳播和更新
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                # 累計 loss
                epoch_loss += batch_loss.item()
                leyeloss_epoch += left_loss.item()
                reyeloss_epoch += right_loss.item()
                clstokenloss_epoch += loss.item()
                num_batches += 1

            clstokenloss_epoch_loss = clstokenloss_epoch / num_batches
            avg_loss = epoch_loss / num_batches
            lavg_loss = leyeloss_epoch / num_batches
            ravg_loss = reyeloss_epoch / num_batches

            current_lr = scheduler.get_last_lr()[0]
            log = f"[{epoch}/{config['params']['epoch']}]: loss: {avg_loss:.4f}, clstoken_loss: {clstokenloss_epoch_loss:.4f}, leyeloss: {lavg_loss:.4f}, reyeloss: {ravg_loss:.4f}, lr: {current_lr}"
            print(log)

            # 從第5個 epoch 開始，計算並記錄驗證誤差
            if epoch >= config["params"]["warmup"]:
                val_error = evaluate_model(net, val_dataset, device)
                print(f"Fold {fold+1}, Epoch {epoch}: Validation error {val_error:.2f} degrees")
                if epoch not in epoch_validation_errors:
                    epoch_validation_errors[epoch] = []
                epoch_validation_errors[epoch].append(val_error)

            scheduler.step()

        # 儲存每個 fold 的最後模型（依需求可調整儲存邏輯）
        torch.save(net.state_dict(), os.path.join(savepath, f"fold_{fold+1}_best_model.pt"))

    # 計算所有 fold 中各 epoch 的平均驗證誤差，並找出最佳 epoch
    best_epoch = None
    best_avg_error = float("inf")
    for epoch in sorted(epoch_validation_errors.keys()):
        avg_val_error = np.mean(epoch_validation_errors[epoch])
        print(f"Epoch {epoch} average validation error: {avg_val_error:.2f}")
        if avg_val_error < best_avg_error:
            best_avg_error = avg_val_error
            best_epoch = epoch

    print(f"\n=== 最終 3-Fold Cross Validation: Best Epoch: {best_epoch} with average error: {best_avg_error:.2f} degrees ===")
