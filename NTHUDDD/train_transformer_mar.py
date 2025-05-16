import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter

# 載入模型與資料集模組
from model.DMSnet_transformer import DMSnet
from dataset_NTHUDDD_DGM_MAR import NTHUDDDDataset

# 只引入 f1_score
from sklearn.metrics import precision_score, recall_score, f1_score

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    for data, labels in dataloader:
        for k in data:
            data[k] = data[k].to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        outputs, _ = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)

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
            outputs, _ = model(data)
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

    return epoch_loss , precision , recall , epoch_f1, accuracy

def main():
    parser = argparse.ArgumentParser(description="Train DMSnet for driver monitoring (classification)")
    parser.add_argument('--sequence_length', type=int, default=10, help='Sequence length')
    parser.add_argument('--stride', type=int, default=10, help='Stride for sequence sampling')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--step_size', type=int, default=150, help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR scheduler')
    parser.add_argument('--save_path', type=str, default='./checkpoint/bestDGMnet.pth', help='Path to save best model')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 建立資料集與 DataLoader
    train_root = '/home/remote/tchuang/research/NTHUDDD/Training_Evaluation_Dataset/TrainingDataset'
    eval_root  = '/home/remote/tchuang/research/NTHUDDD/Training_Evaluation_Dataset/EvaluationDataset'
    train_dataset = NTHUDDDDataset(train_root, sequence_length=args.sequence_length, max_sequences_per_video=100,stride=args.stride, frame_skip=1, eval=False)
    val_dataset   = NTHUDDDDataset(eval_root,  sequence_length=args.sequence_length, max_sequences_per_video=100,stride=args.stride, frame_skip=1, eval=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 模型、損失、優化器、排程器
    model     = DMSnet(out_dim=2, seq_length=args.sequence_length).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print('train sequences:', len(train_loader))
    print('validation sequences:', len(val_loader))
    best_val_f1 = 0.0
    for epoch in range(1, args.epochs+1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,val_precision,val_recall ,val_f1, val_acc = evaluate(model, val_loader, criterion, device)

        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val   Loss: {val_loss:.4f}, P: {val_precision:.2f}%, R: {val_recall:.2f}%,  F1: {val_f1:.2f}%, Acc: {val_acc:.2f}% | "
            f"Time: {elapsed:.1f}s LR: {lr:.1e}"
        )

        scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), args.save_path)
            print(f"[Saved] New best F1: {best_val_f1:.2f}% -> {args.save_path}")

if __name__ == '__main__':
    main()
