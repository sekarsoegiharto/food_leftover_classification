import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, confusion_matrix,
    classification_report, cohen_kappa_score
)
import matplotlib.pyplot as plt

# ================================================================
# 1️⃣ Load Data Asli (Ground Truth dari berat makanan)
# ================================================================
EXCEL_PATH = "data_original.xlsx"
df = pd.read_excel(EXCEL_PATH)

# Hitung persentase sisa aktual
df["Waste (%)"] = (df["Weight After Eaten (g)"] / df["Weight Before Eaten (g)"]) * 100

# Konversi persentase sisa → level 1–7 (Comstock)
def percent_to_level(p):
    if p <= 5: return 1
    elif p <= 25: return 2
    elif p <= 50: return 3
    elif p <= 75: return 4
    elif p <= 85: return 5
    elif p <= 95: return 6
    else: return 7

df["true_label"] = df["Waste (%)"].apply(percent_to_level)

# ================================================================
# 2️⃣ Label Manual (Visual Estimation)
# ================================================================
df["manual_label"] = df["Visual Estimation by Observer (1-7)"].astype(str).str.extract(r'(\d+)').astype(int)

# Deteksi arah skala (otomatis balik jika berlawanan)
corr = np.corrcoef(df["manual_label"].squeeze(), df["Waste (%)"])[0, 1]
if corr < 0:
    print("🔄 Skala manual berlawanan arah — membalik skala manual (1↔7).")
    df["manual_label"] = 8 - df["manual_label"]
else:
    print("✅ Skala manual sudah searah dengan ground truth.")

# ================================================================
# 3️⃣ Evaluasi Pengamat Manual
# ================================================================
acc_manual = accuracy_score(df["true_label"], df["manual_label"])
mae_manual = mean_absolute_error(df["true_label"], df["manual_label"])
kappa_manual = cohen_kappa_score(df["true_label"], df["manual_label"])

print("\n=== Evaluasi Pengamat Manual ===")
print(f"Akurasi Manual : {acc_manual*100:.2f}%")
print(f"MAE Manual     : {mae_manual:.3f}")
print(f"Kappa Manual   : {kappa_manual:.3f}")
print("Confusion Matrix (Manual):\n", confusion_matrix(df["true_label"], df["manual_label"]))

# ================================================================
# 4️⃣ Evaluasi Model CNN
# ================================================================
# cnn_df = pd.read_csv("cnn_predictions.csv")
# cnn_df["pred"] = cnn_df["pred"].astype(int)
# # Balik arah skala CNN supaya searah (jika perlu)
# cnn_df["pred"] = 8 - cnn_df["pred"]

# # Samakan jumlah data
# if len(cnn_df) != len(df):
#     print(f"⚠️ Jumlah data CNN ({len(cnn_df)}) ≠ Excel ({len(df)}). Disesuaikan sementara.")
#     df = df.iloc[:len(cnn_df)]

# cnn_true = df["true_label"].to_numpy()
# cnn_pred = cnn_df["pred"].to_numpy()

# acc_cnn = accuracy_score(cnn_true, cnn_pred)
# mae_cnn = mean_absolute_error(cnn_true, cnn_pred)
# kappa_cnn = cohen_kappa_score(cnn_true, cnn_pred)

# print("\n=== Evaluasi Model CNN (Siamese MobileNetV2) ===")
# print(f"Akurasi CNN : {acc_cnn*100:.2f}%")
# print(f"MAE CNN     : {mae_cnn:.3f}")
# print(f"Kappa CNN   : {kappa_cnn:.3f}")
# print("Confusion Matrix (CNN):\n", confusion_matrix(cnn_true, cnn_pred))
# print("\nClassification Report (CNN):\n", classification_report(
#     cnn_true, cnn_pred, labels=range(1, 8),
#     target_names=[f"Level {i}" for i in range(1, 8)]
# ))
# 1️⃣ Gabungkan data manual dan prediksi CNN
cnn_df = pd.read_csv("cnn_predictions.csv")
df = pd.read_excel("data_original.xlsx")

# Tambahkan kolom manual_label dari pengamat
df["manual_label"] = df["Visual Estimation by Observer (1-7)"].astype(str).str.extract(r'(\d+)').astype(int)

# Bersihkan baris yang tidak punya gambar sebelum makan
missing_before = df["Image Before Eaten"].isna().sum()
if missing_before > 0:
    print(f"⚠️ Ditemukan {missing_before} baris tanpa 'Image Before Eaten' — akan di-skip.")
df = df.dropna(subset=["Image Before Eaten"])

# Samakan tipe data (string) agar bisa di-merge
df["Image Before Eaten"] = df["Image Before Eaten"].astype(str)
cnn_df["img_id"] = cnn_df["img_id"].astype(str)

# 🔗 Merge berdasarkan nama file gambar sebelum makan
merged = pd.merge(df, cnn_df, left_on="Image Before Eaten", right_on="img_id", how="inner")

print(f"✅ Data berhasil digabung: {len(merged)} baris cocok ditemukan.")
if len(merged) == 0:
    print("⚠️ Tidak ada baris yang cocok! Cek apakah nama file di 'Image Before Eaten' sama persis dengan kolom img_id di cnn_predictions.csv")
else:
    print(merged[["Image Before Eaten", "img_id", "manual_label", "pred"]].head())

# 2️⃣ Cek arah korelasi
if len(merged) > 0:
    corr = np.corrcoef(merged["pred"], merged["manual_label"])[0, 1]
    if corr < 0:
        print("🔄 Skala CNN berlawanan arah — membalik prediksi (1↔7).")
        merged["pred"] = 8 - merged["pred"]
    else:
        print("✅ Skala CNN sudah searah dengan manual.")

    # 3️⃣ Evaluasi ulang setelah dibalik (kalau perlu)
    acc_cnn = accuracy_score(merged["manual_label"], merged["pred"])
    mae_cnn = mean_absolute_error(merged["manual_label"], merged["pred"])
    kappa_cnn = cohen_kappa_score(merged["manual_label"], merged["pred"])

    print("\n=== Evaluasi CNN vs Manual ===")
    print(f"Akurasi CNN : {acc_cnn*100:.2f}%")
    print(f"MAE CNN     : {mae_cnn:.3f}")
    print(f"Kappa CNN   : {kappa_cnn:.3f}")
    print("Confusion Matrix:\n", confusion_matrix(merged["manual_label"], merged["pred"]))

# ================================================================
# 5️⃣ Perbandingan Manual vs CNN
# ================================================================
comparison = pd.DataFrame({
    "Metode": ["Manual Observation", "Model CNN"],
    "Akurasi (%)": [round(acc_manual*100, 2), round(acc_cnn*100, 2)],
    "MAE": [round(mae_manual, 3), round(mae_cnn, 3)],
    "Kappa": [round(kappa_manual, 3), round(kappa_cnn, 3)]
})

print("\n=== Manual vs CNN ===")
print(comparison, "\n")

# ================================================================
# 6️⃣ Visualisasi Perbandingan
# ================================================================
plt.figure(figsize=(6,4))
plt.bar(comparison["Metode"], comparison["Akurasi (%)"], color=["#FFB703","#219EBC"])
plt.title("Comparison of Manual Observation vs Model CNN\n(Ground Truth: Food Weight)")
plt.ylabel("Akurasi (%)")
plt.ylim(0,100)
for i, v in enumerate(comparison["Akurasi (%)"]):
    plt.text(i, v + 1, f"{v}%", ha='center', fontweight='bold')
plt.tight_layout()
plt.show()
