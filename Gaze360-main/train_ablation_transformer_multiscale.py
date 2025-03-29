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
from model.ablation_study.DGMtwoeyenet_transformer_multiscale import DGMnet  
from model.warm.warmup_scheduler.scheduler import GradualWarmupScheduler

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

            gaze = net(data)
            for k, gaze in enumerate(gaze):
                gaze = gaze.cpu().detach().numpy()
                count += 1
                accs += angular(gazeto3d(gaze), gazeto3d(gts.cpu().numpy()[k]))
    print(f'Total Num: {count}, avg: {accs/count}')

    return accs / count

def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = sum( p.numel() for p in model.parameters()) / (1024 ** 2)

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Model Size (M): {model_size:.2f}")

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
    dataset = dataloader.txtload(labelpath, imagepath, config["params"]["batch_size"], shuffle=True, num_workers=4, header=True)
    config2 = yaml.load(open(f"{sys.argv[1]}"), Loader=yaml.FullLoader)["test"]
    test_imagepath = config2["data"]["image"]
    test_labelpath = config2["data"]["label"]
    test_dataset = dataloader.txtload(test_labelpath, test_imagepath, 32, num_workers=4, header=True)
    seed=3
    print(f"Running with seed {seed}")
    set_seed(seed)

    print("Model building")
    net = DGMnet()
    print_model_size(net)
    print("CUDA 是否可用:", torch.cuda.is_available())
    net.to(device)
    print(f'device:{device}')
    length = len(dataset)
    timebegin = time.time()

    total = length * config["params"]["epoch"]
    print("Optimizer building")
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

    print("Training")
    cur = 0
    timebegin = time.time()
    best_epoch = -1  # 用於記錄最佳epoch

    for epoch in range(1, config["params"]["epoch"] + 1):
        net.train()
        epoch_loss = 0.0
        num_batches = 0
        for i, (data, label) in enumerate(dataset):
            # 數據準備
            data["origin_face"] = data["face"].to(device)
            data["left_eye"] = data["lefteyeimg"].to(device)
            data["right_eye"] = data["righteyeimg"].to(device)
            data["gaze_origin"] = torch.zeros(data["origin_face"].size(0), 3).to(device)
            label = label.to(device)
            
            # 前向傳播
            gaze = net(data)
            
            # 分別計算 loss
            loss = nn.L1Loss()(gaze, label)

            
            # 組合各部分 loss
            
            # 反向傳播和更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累計 loss（注意取 .item() 得到純數值）
            epoch_loss += loss.item()
            num_batches += 1
            cur += 1

        avg_loss = epoch_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        timeend = time.time()
        resttime = (timeend - timebegin) / cur * (total - cur) / 3600

        log = f"[{epoch}/{config['params']['epoch']}]: loss: {avg_loss:.4f} ,lr: {current_lr}, rest time: {resttime:.2f}h"
        print(log)

        # 評估當前 epoch 的 test error
        print("Evaluating on test data...")


        current_test_error = evaluate_model(net, test_dataset, device)


        print(f"test error {current_test_error:.2f} degrees")

        scheduler.step()
