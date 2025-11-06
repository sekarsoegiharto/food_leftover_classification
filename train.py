import torch, torch.nn as nn, torch.optim as optim, numpy as np, random, os, pandas as pd
from torch.utils.data import DataLoader
import shutil
from dataset import PairFoodDataset
from model import SiameseMobileNetV2
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, cohen_kappa_score

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
def train_and_validate(lr=1e-4, batch_size=16, weight_decay=1e-5, optimizer_type="adamw"):
    # Dataset
    train_set = PairFoodDataset("splits/train.csv", augment=True)
    val_set   = PairFoodDataset("splits/val.csv", augment=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

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
    EPOCHS = 1000
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
        # val_kappa = cohen_kappa_score(y_true, y_pred)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss={loss_sum/n:.4f} | Acc={val_acc:.4f} | MAE={val_mae:.4f} | MSE={val_mse:.4f}")

        # Simpan best model per kombinasi
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = {"acc": val_acc, "mae": val_mae, "mse": val_mse}

            # Simpan model terbaik sementara
            os.makedirs("checkpoints", exist_ok=True)
            save_path = f"checkpoints/best_temp.pth"
            torch.save(model.state_dict(), save_path)
            print(f"💾 Model disimpan: {save_path}")

    return best_metrics

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

    # Ubah indeks 0–6 → 1–7
    y_true = np.array(y_true) + 1
    y_pred = np.array(y_pred) + 1

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    print("\n=== HASIL EVALUASI DATA UJI ===")
    print(f"Akurasi : {acc:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"MSE      : {mse:.4f}")
    print(f"Kappa   : {kappa:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

def save_predictions(model_path="checkpoints/best.pth", out_csv="cnn_predictions.csv"):
    # test_set = PairFoodDataset("splits/test.csv", augment=False)
    # test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # model = SiameseMobileNetV2(num_classes=7, pretrained=False).to(DEVICE)
    # model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    # model.eval()

    # preds, probs = [], []
    # with torch.no_grad():
    #     for xb, xa, y in test_loader:
    #         xb, xa = xb.to(DEVICE), xa.to(DEVICE)
    #         logits = model(xb, xa)
    #         prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
    #         pred = prob.argmax() + 1  # kelas 1..7
    #         preds.append(pred)
    #         probs.append(prob)

    # # Simpan ke CSV
    # df_out = pd.DataFrame({
    #     "pred": preds,
    #     **{f"prob_{i+1}": [p[i] for p in probs] for i in range(7)}
    # })
    # df_out.to_csv(out_csv, index=False)
    # print(f"✅ Prediksi CNN tersimpan di {out_csv}")
    df_test = pd.read_csv("splits/test.csv")
    img_ids = df_test.iloc[:, 0].apply(os.path.basename).tolist()  # kolom pertama = path gambar before
    if "img_id" in df_test.columns:
        img_ids = df_test["img_id"].tolist()

    test_set = PairFoodDataset("splits/test.csv", augment=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = SiameseMobileNetV2(num_classes=7, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    preds, probs = [], []

    with torch.no_grad():
        for xb, xa, y in test_loader:
            xb, xa = xb.to(DEVICE), xa.to(DEVICE)
            logits = model(xb, xa)
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = prob.argmax() + 1  # kelas 1..7
            preds.append(pred)
            probs.append(prob)

    # Pastikan jumlah gambar sama
    if len(img_ids) != len(preds):
        print(f"⚠️ Jumlah img_id ({len(img_ids)}) ≠ prediksi ({len(preds)}), disesuaikan otomatis.")
        img_ids = img_ids[:len(preds)]

    # Simpan ke CSV
    df_out = pd.DataFrame({
        "img_id": img_ids,
        "pred": preds,
        **{f"prob_{i+1}": [p[i] for p in probs] for i in range(7)}
    })
    df_out.to_csv(out_csv, index=False)
    print(f"✅ Prediksi CNN tersimpan di {out_csv}")

if __name__ == "__main__":
    train_and_validate()

    # 1️⃣ Pastikan folder checkpoint ada
    os.makedirs("checkpoints", exist_ok=True)

    # 2️⃣ Cek apakah ada model sementara (best_temp.pth)
    temp_path = "checkpoints/best_temp.pth"
    best_path = "checkpoints/best.pth"

    if os.path.exists(temp_path):
        shutil.move(temp_path, best_path)
        print(f"💾 Model terbaik dipindahkan dari {temp_path} ke {best_path}")
    elif not os.path.exists(best_path):
        # train_and_validate()
        print(f"Lakukan training model terlebih dahulu")
    else:
        print("ℹ️ Menggunakan model terakhir di checkpoints/best.pth")

    # 3️⃣ Evaluasi model terbaik
    evaluate_test(best_path)

    # 4️⃣ Simpan hasil prediksi untuk perbandingan manual vs CNN
    save_predictions(best_path, "cnn_predictions.csv")
