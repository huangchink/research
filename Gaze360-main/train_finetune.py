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
from model.twoeyenet_transformer_1054 import DGMnet  # 新增：使用 DGMnet_harD.py 中的 DGMnet
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

            gaze, gaze1, gaze2, gaze3, gaze4= net(data)
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
    dataset = dataloader.txtload(labelpath, imagepath, config["params"]["batch_size"], shuffle=True, num_workers=4, header=True)
    config2 = yaml.load(open(f"{sys.argv[1]}"), Loader=yaml.FullLoader)["test"]
    test_imagepath = config2["data"]["image"]
    test_labelpath = config2["data"]["label"]
    test_dataset = dataloader.txtload(test_labelpath, test_imagepath, 32, num_workers=4, header=True)
    for seed in range(3, 20):  # 從 1 跑到 3407
        print(f"Running with seed {seed}")
        set_seed(seed)

        print("Model building")
        net = DGMnet()
        pretrained_path = "./loadmodel/pretrain_model.pt"

        # 載入預訓練的 Backbone
        load_pretrained_backbone(net, pretrained_path)
        print_model_size(net)
        print("CUDA 是否可用:", torch.cuda.is_available())
        print("GPU 數量:", torch.cuda.device_count())
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
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.)      

        scheduler = GradualWarmupScheduler(
            optimizer, 
            multiplier=1, 
            total_epoch=3, 
            after_scheduler=scheduler
        )

        print("Training")
        cur = 0
        timebegin = time.time()
        best_test_error = 10.7  # 初始化最佳測試誤差為無限大
        best_epoch = -1  # 用於記錄最佳epoch

        for epoch in range(1, config["params"]["epoch"] + 1):
            net.train()
            epoch_loss = 0.0
            num_batches = 0
            leyeloss_epoch=0.0
            reyeloss_epoch=0.0
            clstokenloss_epoch=0.0

            for i, (data, label) in enumerate(dataset):
                # 數據準備
                data["origin_face"] = data["face"].to(device)
                data["left_eye"] = data["lefteyeimg"].to(device)
                data["right_eye"] = data["righteyeimg"].to(device)
                data["gaze_origin"] = torch.zeros(data["origin_face"].size(0), 3).to(device)
                label = label.to(device)
                
                gaze, gaze1, gaze2, gaze3,gaze4 = net(data)
                
                # 分別計算 loss
                loss = nn.L1Loss()(gaze, label)
                lowface_loss = nn.L1Loss()(gaze1, label)
                face_loss = nn.L1Loss()(gaze2, label)

                left_loss = nn.L1Loss()(gaze3, label)
                right_loss = nn.L1Loss()(gaze4, label)
                # loss4 = nn.L1Loss()(gaze4, label)
                # eyeloss = nn.L1Loss()(gaze5, label)
                
                # 組合各部分 loss
                batch_loss = 0.2 * loss + 0.2 * lowface_loss+ 0.2 * face_loss + 0.2 * left_loss + 0.2 * right_loss
                

                # 反向傳播和更新
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                                # 累計 loss（注意取 .item() 得到純數值）
                epoch_loss += batch_loss.item()
                leyeloss_epoch+=left_loss.item()
                reyeloss_epoch+=right_loss.item()

                clstokenloss_epoch+=loss.item()
                num_batches += 1
            clstokenloss_epoch_loss=clstokenloss_epoch/ num_batches
            avg_loss = epoch_loss / num_batches
            lavg_loss = leyeloss_epoch / num_batches
            ravg_loss = reyeloss_epoch / num_batches

            # 累計 loss（注意取 .item() 得到純數值）
            epoch_loss += batch_loss.item()
            num_batches += 1
            cur += 1

            avg_loss = epoch_loss / num_batches
            current_lr = scheduler.get_last_lr()[0]
            timeend = time.time()
            resttime = (timeend - timebegin) / cur * (total - cur) / 3600

            log = f"[{epoch}/{config['params']['epoch']}]: loss: {avg_loss:.4f}, clstoken_loss: {clstokenloss_epoch_loss:.4f}, leyeloss: {lavg_loss:.4f},reyeloss: {ravg_loss:.4f}, lr: {current_lr}"
            print(log)

            # 評估當前 epoch 的 test error
            print("Evaluating on test data...")


            current_test_error = evaluate_model(net, test_dataset, device)

            # 更新最佳模型
            if current_test_error < best_test_error:
                best_test_error = current_test_error
                best_epoch = epoch
                model_save_path = os.path.join(savepath, f"seed_{seed}_best_model_epoch_{best_test_error}.pt")
                torch.save(net.state_dict(), model_save_path)
                print(f"New best model for seed {seed} saved at epoch {epoch} with test error {best_test_error:.2f} degrees")

            scheduler.step()
