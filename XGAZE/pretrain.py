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
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

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

def evaluate_model(net, val_loader, device):
    net.eval()
    accs = 0
    count = 0
    with torch.no_grad():
        for j, (data, label) in enumerate(val_loader):
            data["origin_face"] = data["face"].to(device)
            data["left_eye"] = data["left"].to(device)
            data["right_eye"] = data["right"].to(device)
            data["gaze_origin"] = torch.zeros(data["origin_face"].size(0), 3).to(device)

            label = label.to(device)
            gts = label.to(device)

            gaze, gaze1, gaze2, gaze3, gaze4 = net(data)
            for k, gaze_item in enumerate(gaze):
                gaze_np = gaze_item.cpu().detach().numpy()
                count += 1
                accs += angular(gazeto3d(gaze_np), gazeto3d(gts.cpu().numpy()[k]))
    print(f'[Validation] Total Samples: {count}, Average Error: {accs/count:.2f}°')
    return accs / count

def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 ** 2)

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Model Size (MB): {model_size:.2f}")

if __name__ == "__main__":
    # 讀取設定檔
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


    print("Reading data...")
    # 先建立完整資料集，再做切分（例如 80% 作為訓練集，20% 作為驗證集）
    full_dataset = dataloader.loader(labelpath, imagepath, header=True, train=True)
    total_samples = len(full_dataset)
    train_size = int(0.9 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"[Data Split] Train: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples")

    # 建立 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config["params"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["params"]["batch_size"], shuffle=False)

    # 若需要其他測試參數，這裡讀取 test 參數（但我們將驗證用 val_loader）
    config2 = yaml.load(open(f"{sys.argv[1]}"), Loader=yaml.FullLoader)["test"]

    for seed in range(3,42):  # 從 1 跑到 499
        print(f"Running with seed {seed}")
        set_seed(seed)

        print("Building model...")
        net = DGMnet()
        print_model_size(net)
        net.to(device)
        print(torch.cuda.get_device_name(0))
        length = len(train_loader)
        total_iter = length * config["params"]["epoch"]
        print("Building optimizer...")
        base_lr = config["params"]["lr"]
        decaysteps = config["params"]["decay_step"]
        decayratio = config["params"]["decay"]

        optimizer = optim.Adam(net.parameters(), lr=base_lr, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=decaysteps, 
            gamma=decayratio
        )
        scheduler = GradualWarmupScheduler(
            optimizer, 
            multiplier=1, 
            total_epoch=3, 
            after_scheduler=scheduler
        )

        print("Start training...")
        cur = 0
        timebegin = time.time()
        best_val_error = float('inf')  # 初始化最佳驗證誤差為無限大
        best_epoch = -1  # 用於記錄最佳 epoch

        for epoch in tqdm(range(1, config["params"]["epoch"] + 1)):
            net.train()
            epoch_loss = 0.0
            leyeloss_epoch=0.0
            reyeloss_epoch=0.0
		
            clstokenloss_epoch=0.0
            for i, (data, label) in enumerate(train_loader):
                data["origin_face"] = data["face"].to(device)
                data["left_eye"] = data["left"].to(device)
                data["right_eye"] = data["right"].to(device)
                data["gaze_origin"] = torch.zeros(data["origin_face"].size(0), 3).to(device)

                label = label.to(device)
                # forward
                gaze, gaze1, gaze2, gaze3, gaze4 = net(data)
                # loss calculation
                # 分別計算 loss
                loss = nn.L1Loss()(gaze, label)
                lowface_loss = nn.L1Loss()(gaze1, label)
                face_loss = nn.L1Loss()(gaze2, label)

                left_loss = nn.L1Loss()(gaze3, label)
                right_loss = nn.L1Loss()(gaze4, label)
                batch_loss = 0.2 * loss +0.2 * lowface_loss+0.2 * face_loss + 0.2 * left_loss + 0.2 * right_loss
                
                # 反向傳播和更新
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                # 累計 loss（注意取 .item() 得到純數值）
                epoch_loss += batch_loss.item()
                leyeloss_epoch+=left_loss.item()
                reyeloss_epoch+=right_loss.item()

                clstokenloss_epoch+=loss.item()
                cur += 1

            clstokenloss_epoch_loss=clstokenloss_epoch/ cur
            avg_loss = epoch_loss / cur
            lavg_loss = leyeloss_epoch / cur
            ravg_loss = reyeloss_epoch / cur
            total = length * config["params"]["epoch"]

            current_lr = scheduler.get_last_lr()[0]
            timeend = time.time()
            resttime = (timeend - timebegin) / cur * (total - cur) / 3600

            log = f"[{epoch}/{config['params']['epoch']}]: loss: {avg_loss:.4f}, clstoken_loss: {clstokenloss_epoch_loss:.4f}, leyeloss: {lavg_loss:.4f},reyeloss: {ravg_loss:.4f}, lr: {current_lr}, rest time: {resttime:.2f}h"
            print(log)

            # 評估當前 epoch 的驗證誤差
            print("Evaluating on validation data...")
            current_val_error = evaluate_model(net, val_loader, device)

            # 更新最佳模型
            if current_val_error < best_val_error:
                best_val_error = current_val_error
                best_epoch = epoch
                model_save_path = os.path.join(savepath, f"seed_{seed}_best_model_epoch{epoch}_{best_val_error:.2f}.pt")
                torch.save(net.state_dict(), model_save_path)
                print(f"New best model for seed {seed} saved at epoch {epoch} with validation error {best_val_error:.2f}°")

            scheduler.step()
