import torch, torch.nn as nn, torch.optim as optim, numpy as np, random, os, pandas as pd
from torch.utils.data import DataLoader
from dataset import PairFoodDataset
from model import SiameseMobileNetV2
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ================================================================
# Fungsi hitung bobot kelas
# ================================================================
def get_class_weights(csv_path):
    df = pd.read_csv(csv_path)
    counts = df['label'].value_counts().sort_index()
    counts = counts.values
    weights = (1.0 / (counts + 1e-6))
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)

# ================================================================
# Fungsi training & evaluasi
# ================================================================
def train_and_validate(lr, batch_size, weight_decay, optimizer_type):
    print(f"\n🔍 [GRID SEARCH] LR={lr} | BS={batch_size} | WD={weight_decay} | OPT={optimizer_type}")

    # Dataset
    train_set = PairFoodDataset("splits/train.csv", augment=True)
    val_set   = PairFoodDataset("splits/val.csv", augment=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=4)

    # Model
    model = SiameseMobileNetV2(num_classes=7, pretrained=True).to(DEVICE)
    class_w = get_class_weights("splits/train.csv").to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    # Optimizer
    if optimizer_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError("Unknown optimizer")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Training singkat per kombinasi
    EPOCHS = 10
    best_val_acc = 0.0
    best_metrics = {}

    for epoch in range(1, EPOCHS + 1):
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
            loss_sum += loss.item() * y.size(0)
            n += y.size(0)

        # Validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, xa, y in val_loader:
                xb, xa, y = xb.to(DEVICE), xa.to(DEVICE), y.to(DEVICE)
                logits = model(xb, xa)
                pred = logits.argmax(1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

        # Hitung metrik
        val_acc = np.mean(np.array(y_true) == np.array(y_pred))
        val_mae = mean_absolute_error(y_true, y_pred)
        val_mse = mean_squared_error(y_true, y_pred)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss={loss_sum/n:.4f} | Acc={val_acc:.4f} | MAE={val_mae:.4f} | MSE={val_mse:.4f}")

        # Simpan best model per kombinasi
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = {"acc": val_acc, "mae": val_mae, "mse": val_mse}

    return best_metrics


# ================================================================
# GRID SEARCH
# ================================================================
def run_grid_search():
    grid_params = {
        "lr": [1e-3, 3e-4, 1e-4],
        "batch_size": [8, 16],
        "weight_decay": [1e-4, 1e-5],
        "optimizer": ["adamw", "adam"]
    }

    results = []
    for lr in grid_params["lr"]:
        for bs in grid_params["batch_size"]:
            for wd in grid_params["weight_decay"]:
                for opt in grid_params["optimizer"]:
                    metrics = train_and_validate(lr, bs, wd, opt)
                    results.append({
                        "lr": lr,
                        "batch_size": bs,
                        "weight_decay": wd,
                        "optimizer": opt,
                        "val_acc": metrics["acc"],
                        "val_mae": metrics["mae"],
                        "val_mse": metrics["mse"]
                    })
                    torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv("gridsearch_results.csv", index=False)
    print("\n✅ Grid search selesai. Hasil disimpan di gridsearch_results.csv\n")
    print(df.sort_values("val_acc", ascending=False).head())

# ================================================================
# TESTING (setelah best model dipilih)
# ================================================================
def evaluate_test(best_model_path="checkpoints/best.pth"):
    test_set = PairFoodDataset("splits/test.csv", augment=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

    model = SiameseMobileNetV2(num_classes=7, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, xa, y in test_loader:
            xb, xa, y = xb.to(DEVICE), xa.to(DEVICE), y.to(DEVICE)
            logits = model(xb, xa)
            pred = logits.argmax(1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print("\n=== HASIL EVALUASI DATA UJI ===")
    print(f"Akurasi : {acc:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"MSE      : {mse:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    run_grid_search()
    # Setelah selesai dan best model diketahui, bisa jalankan manual:
    # evaluate_test("checkpoints/best.pth")
