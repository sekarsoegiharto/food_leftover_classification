import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, confusion_matrix,
    classification_report, cohen_kappa_score
)
import matplotlib.pyplot as plt

# ================================================================
# 1️⃣ Load Data dari Excel
# ================================================================
EXCEL_PATH = "data_original.xlsx"
df = pd.read_excel(EXCEL_PATH)

# Hitung persentase sisa aktual (Comstock)
df["Waste (%)"] = (df["Weight After Eaten (g)"] / df["Weight Before Eaten (g)"]) * 100

# Fungsi konversi persen → level (1–7) sesuai Comstock (1=habis, 7=tidak dimakan)
def percent_to_level(p):
    if p <= 5: return 7
    elif p <= 25: return 6
    elif p <= 50: return 5
    elif p <= 75: return 4
    elif p <= 85: return 3
    elif p <= 95: return 2
    else: return 1

# Tambahkan label ground truth
df["true_label"] = df["Waste (%)"].apply(percent_to_level)

# ================================================================
# 2️⃣ Ambil label manual & sesuaikan arah skalanya
# ================================================================
df["manual_label"] = df["Visual Estimation by Observer (1-7)"].astype(str).str.extract(r'(\d+)').astype(int)

# Deteksi arah skala manual (cek korelasi dengan berat aktual)
corr = np.corrcoef(df["manual_label"], df["Waste (%)"])[0, 1]
# if corr < 0:
#     print("🔄 Skala manual terbalik terhadap Comstock. Membalik label manual...")
#     df["manual_label"] = 8 - df["manual_label"]
# else:
#     print("✅ Skala manual sudah searah dengan Comstock.")

print("Distribusi ground truth (berdasarkan berat makanan):\n", df["true_label"].value_counts().sort_index())
print()

# ================================================================
# 3️⃣ Evaluasi Pengamat Manual vs Ground Truth Berat
# ================================================================
acc_manual = accuracy_score(df["true_label"], df["manual_label"])
mae_manual = mean_absolute_error(df["true_label"], df["manual_label"])
kappa_manual = cohen_kappa_score(df["true_label"], df["manual_label"])

print("=== Evaluasi Pengamat Manual ===")
print("Akurasi Manual :", round(acc_manual*100, 2), "%")
print("MAE Manual     :", round(mae_manual, 3))
print("Kappa Manual   :", round(kappa_manual, 3))
print("Confusion Matrix (Manual):\n", confusion_matrix(df["true_label"], df["manual_label"]))
print()

# ================================================================
# 4️⃣ Evaluasi CNN vs Ground Truth Berat
# ================================================================
cnn_df = pd.read_csv("cnn_predictions.csv")
cnn_df["pred"] = cnn_df["pred"].astype(int)

# Samakan jumlah data CNN dengan subset Excel
if len(cnn_df) != len(df):
    print(f"⚠️ Jumlah data CNN ({len(cnn_df)}) ≠ Excel ({len(df)}). Disamakan sementara untuk evaluasi.")
    cnn_true = df.loc[:len(cnn_df)-1, "true_label"].to_numpy()
else:
    cnn_true = df.loc[cnn_df.index, "true_label"].to_numpy()

acc_cnn = accuracy_score(cnn_true, cnn_df["pred"])
mae_cnn = mean_absolute_error(cnn_true, cnn_df["pred"])
kappa_cnn = cohen_kappa_score(cnn_true, cnn_df["pred"])

print("=== Evaluasi Model CNN (Siamese MobileNetV2) ===")
print("Akurasi CNN :", round(acc_cnn*100, 2), "%")
print("MAE CNN     :", round(mae_cnn, 3))
print("Kappa CNN   :", round(kappa_cnn, 3))
print("Confusion Matrix (CNN):\n", confusion_matrix(cnn_true, cnn_df["pred"]))
print()
print(classification_report(
    cnn_true, cnn_df["pred"],
    labels=range(1, 8),
    target_names=[f"Level {i}" for i in range(1, 8)]
))

# ================================================================
# 5️⃣ Visualisasi Perbandingan
# ================================================================
comparison = pd.DataFrame({
    "Metode": ["Pengamat Manual", "Model CNN"],
    "Akurasi (%)": [round(acc_manual*100, 2), round(acc_cnn*100, 2)],
    "MAE": [round(mae_manual, 3), round(mae_cnn, 3)],
    "Kappa": [round(kappa_manual, 3), round(kappa_cnn, 3)]
})

print("\n=== Perbandingan Manual vs CNN (berdasarkan berat aktual) ===")
print(comparison, "\n")

plt.figure(figsize=(6,4))
plt.bar(comparison["Metode"], comparison["Akurasi (%)"], color=["#FFB703","#219EBC"])
plt.title("Perbandingan Akurasi Pengamat Manual vs Model CNN\n(Ground Truth: Berat Makanan)")
plt.ylabel("Akurasi (%)")
plt.ylim(0,100)
for i, v in enumerate(comparison["Akurasi (%)"]):
    plt.text(i, v + 1, f"{v}%", ha='center', fontweight='bold')
plt.tight_layout()
plt.show()
