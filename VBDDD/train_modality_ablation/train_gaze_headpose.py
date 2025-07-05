import os
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys




sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("sys.path:", sys.path)

# 載入模型與資料集模組
from model.modality_ablation.DMSnet_transformer_gaze_headpose import DMSnet

# from ..model.modality_ablation.DMSnet_transformer_gazeonly import DMSnet
from dataset_VBDDD_DGM import VBDDDDataset
from sklearn.metrics import precision_score, recall_score, f1_score
from dataloader_processed import OfflineVBDDDDataset
def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, labels) in enumerate(dataloader):
        for key in data:
            data[key] = data[key].to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        outputs, gaze = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    accuracy = correct / total * 100
    return epoch_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(dataloader):
            for key in data:
                data[key] = data[key].to(device)
            labels = labels.to(device).long()
            outputs, gaze = model(data)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)

            # 收集所有 batch 的預測和真實值
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            total += labels.size(0)

    epoch_loss = running_loss / total
    accuracy = (sum([p==l for p, l in zip(all_preds, all_labels)]) / total) * 100

    # 計算 precision, recall, f1 (binary classification)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0) * 100
    recall    = recall_score(all_labels, all_preds, average='binary', zero_division=0) * 100
    f1        = f1_score(all_labels, all_preds, average='binary', zero_division=0) * 100

    return epoch_loss, accuracy, precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description="Train DMSnet for driver monitoring (classification)")
    parser.add_argument('--data_root', type=str, default='/home/remote/tchuang/research/VBDDD/',
                        help='Root directory of VBDDD dataset')
    parser.add_argument('--subjects', type=str, nargs='+', default=[f'subject{i}' for i in range(1,38)],
                        help='List of subject names to use')
    parser.add_argument('--sequence_length', type=int, default=10, help='Sequence length')
    parser.add_argument('--stride', type=int, default=10, help='Stride for sequence sampling')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--step_size', type=int, default=100, help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR scheduler')
    parser.add_argument('--save_path', type=str, default='./checkpoint/bestDGMnet.pth', help='Path to save best model')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    #4
    seed=4
    print(f"Running with seed {seed}")
    set_seed(seed)
    # 將 subjects 隨機打散，前 10% 當驗證、後 90% 當訓練
    all_subjects = args.subjects.copy()
    random.shuffle(all_subjects)
    n_total = len(all_subjects)
    n_val = max(1, int(np.round(0.1 * n_total,0)))
    val_subjects = all_subjects[:n_val]
    train_subjects = all_subjects[n_val:]
    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Val   subjects ({len(val_subjects)}): {val_subjects}")
    root=args.data_root+f'preprocessed_{args.sequence_length}'

    # 針對不同 subject 集合分別建立 Dataset
    train_dataset = OfflineVBDDDDataset(preproc_root=root,subjects=train_subjects)
    val_dataset   = OfflineVBDDDDataset(preproc_root=root,subjects=val_subjects)
    # print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    # print(f"Val   subjects ({len(val_subjects)}): {val_subjects}")

    print(f"  Train samples: {len(train_dataset)}")
    print(f"    Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    # 建立模型、損失、優化器與 scheduler
    model = DMSnet(out_dim=2, seq_length=args.sequence_length).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_val_acc = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
            f"P: {val_prec:.4f}%, R: {val_rec:.4f}%, F1: {val_f1:.4f}% | "
            f"Time: {epoch_time:.2f}s LR: {current_lr:.1e}"
        )
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"Model saved to {args.save_path}")

if __name__ == '__main__':
    main()
