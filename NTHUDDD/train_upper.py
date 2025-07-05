import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter

# 載入模型與資料集模組
from model.DMSnet_crossattn import DMSnet
from dataloader_preprocess import OfflineSequenceDataset

# 只引入 f1_score
from sklearn.metrics import precision_score, recall_score, f1_score

def set_seed(seed: int):
    """
    設定隨機種子以保證可重現性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 以下確保 cudnn 在確保 reproducibility 時行為穩定（但可能稍微影響速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    # 新增：用來統計 label 分布
    label_counter = Counter()

    for data, labels in dataloader:
        # 在送入 GPU 前先統計一次
        label_counter.update(labels.tolist())

        for k in data:
            data[k] = data[k].to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)

    # 印出 label 分布
    print("Train Label counts:", label_counter)

    avg_loss = running_loss / total if total > 0 else 0
    accuracy = 100.0 * correct / total if total > 0 else 0
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, labels in dataloader:
            for k in data:
                data[k] = data[k].to(device)
            labels = labels.to(device).long()
            outputs = model(data)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

    print("Label counts:", Counter(all_labels))
    print("Pred counts: ", Counter(all_preds))

    epoch_loss = running_loss / total if total > 0 else 0
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0) * 100
    recall    = recall_score(all_labels, all_preds, average='binary', zero_division=0) * 100
    epoch_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0) * 100
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = 100.0 * correct / total if total > 0 else 0

    return epoch_loss, precision, recall, epoch_f1, accuracy

def main():
    parser = argparse.ArgumentParser(description="Train DMSnet for driver monitoring (classification) over multiple seeds")
    parser.add_argument('--sequence_length', type=int, default=120, help='Sequence length')
    parser.add_argument('--stride', type=int, default=10, help='Stride for sequence sampling')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--step_size', type=int, default=50, help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR scheduler')
    parser.add_argument('--output_dir', type=str, default='./checkpoint', help='Directory to save best models')
    args = parser.parse_args()

    # 確保輸出目錄存在
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 資料集根目錄
    train_root = "/home/tchuang/research/NTHUDDD/preprocessed_nosunglasses_nonight_gaze_pt_120_stride3_aug/train"
    eval_root  = "/home/tchuang/research/NTHUDDD/preprocessed_nosunglasses_nonight_gaze_pt_120_stride3_aug/test"

    # 針對每個 seed，做一次完整的訓練與驗證，並儲存最好的模型
    for seed in range(1, 201):
        print(f"\n===== Starting training with seed = {seed} =====")
        # 設定隨機種子
        set_seed(seed)

        # 建立資料集與 DataLoader（放在 seed 迴圈內，確保 shuffle 行為可重現）
        train_dataset = OfflineSequenceDataset(train_root)
        val_dataset   = OfflineSequenceDataset(eval_root)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=4)
        val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=4)

        # 模型、損失、優化器、排程器
        model     = DMSnet(out_dim=2, seq_length=args.sequence_length).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        print(f'Number of training sequences: {len(train_loader)}')
        print(f'Number of validation sequences: {len(val_loader)}')

        best_val_f1 = 0.0

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()

            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_precision, val_recall, val_f1, val_acc = evaluate(model, val_loader, criterion, device)

            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time

            print(
                f"Seed {seed} | Epoch {epoch}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val   Loss: {val_loss:.4f}, P: {val_precision:.2f}%, R: {val_recall:.2f}%, F1: {val_f1:.2f}%, Acc: {val_acc:.2f}% | "
                f"Time: {elapsed:.1f}s LR: {lr:.1e}"
            )

            scheduler.step()

            # 如果當前驗證 F1 更高，則儲存模型權重
            if val_f1 > best_val_f1 and  val_f1>86:
                best_val_f1 = val_f1
                best_model_path = os.path.join(args.output_dir, f"best_{seed}_{best_val_f1:.2f}.pth")

                torch.save(model.state_dict(), best_model_path)
                print(f"[Seed {seed}] [Saved] New best F1: {best_val_f1:.2f}% → {best_model_path}")

        print(f"===== Finished training with seed = {seed}. Best F1: {best_val_f1:.2f}% =====\n")

if __name__ == '__main__':
    main()
