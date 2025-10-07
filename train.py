# train.py
import torch, torch.nn as nn, torch.optim as optim, numpy as np
from torch.utils.data import DataLoader
from dataset import PairFoodDataset
from model import SiameseMobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd, os, time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BS_TRAIN, BS_VAL = 16, 32
EPOCHS = 30
LR = 3e-4

def get_class_weights(csv_path):
    df = pd.read_csv(csv_path)
    counts = df['label'].value_counts().sort_index() # 1..7
    counts = counts.values
    weights = (1.0 / (counts + 1e-6))
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)

def run():
    train_set = PairFoodDataset("splits/train.csv", augment=True)
    val_set   = PairFoodDataset("splits/val.csv", augment=False)
    test_set  = PairFoodDataset("splits/test.csv", augment=False)

    train_loader = DataLoader(train_set, batch_size=BS_TRAIN, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=BS_VAL, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_set, batch_size=BS_VAL, shuffle=False, num_workers=4)

    model = SiameseMobileNetV2(num_classes=7, pretrained=True).to(DEVICE)

    class_w = get_class_weights("splits/train.csv").to(DEVICE)  # ukuran [7]
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, EPOCHS+1):
        model.train()
        loss_sum, n = 0.0, 0
        for xb, xa, y in train_loader:
            xb, xa, y = xb.to(DEVICE), xa.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb, xa)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.item()*y.size(0); n += y.size(0)

        # validation
        model.eval()
        correct, total, vloss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, xa, y in val_loader:
                xb, xa, y = xb.to(DEVICE), xa.to(DEVICE), y.to(DEVICE)
                logits = model(xb, xa)
                vloss = criterion(logits, y)
                vloss_sum += vloss.item()*y.size(0)
                pred = logits.argmax(1)
                correct += (pred==y).sum().item()
                total += y.size(0)
        val_acc = correct/total
        scheduler.step()

        print(f"[{epoch}/{EPOCHS}] train_loss={loss_sum/n:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(),"checkpoints/best.pth")

    # Test set evaluation
    model.load_state_dict(torch.load("checkpoints/best.pth", map_location=DEVICE))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, xa, y in test_loader:
            xb, xa = xb.to(DEVICE), xa.to(DEVICE)
            logits = model(xb, xa)
            pred = logits.argmax(1).cpu().numpy()
            y_true.extend(y.numpy()); y_pred.extend(pred)
    print(classification_report(y_true, y_pred, target_names=[f"lvl{i}" for i in range(1,8)]))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
if __name__ == "__main__":
    run()
