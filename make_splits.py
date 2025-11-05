import pandas as pd, os
from sklearn.model_selection import train_test_split

EXCEL = "data_original.xlsx"
IMG_BEFORE_DIR = "images/before"
IMG_AFTER_DIR  = "images/after"
OUT_DIR = "splits"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load Excel & tampilkan kolom
df = pd.read_excel(EXCEL)
print("Kolom ditemukan:", df.columns.tolist())

rename_map = {
    'Image Before Eaten': 'before',
    'Image After Eaten': 'after',
    'Visual Estimation by Observer (1-7)': 'label',
}
df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

# 3. Pastikan label ada
if 'label' not in df.columns:
    raise ValueError("Kolom label tidak ditemukan. Cek kembali header Excel yang sesuai.")

# Jika label berupa string "Level X", ubah jadi angka
df['label'] = df['label'].astype(str).str.extract(r'(\d+)').astype(int)

# 4. Tambahkan path gambar
df['before_path'] = df['before'].astype(str).apply(lambda x: os.path.join(IMG_BEFORE_DIR, x))
df['after_path']  = df['after'].astype(str).apply(lambda x: os.path.join(IMG_AFTER_DIR,  x))

# 4. Tambahkan path gambar
df['before'] = df['before'].astype(str).str.strip()
df['after']  = df['after'].astype(str).str.strip()

df['before_path'] = df['before'].apply(lambda x: os.path.join(IMG_BEFORE_DIR, x))
df['after_path']  = df['after'].apply(lambda x: os.path.join(IMG_AFTER_DIR,  x))

# Debug: cek contoh data
print(df[['before','after','label']].head())

# Cek berapa banyak file yang ketemu
df['before_exists'] = df['before_path'].apply(os.path.isfile)
df['after_exists']  = df['after_path'].apply(os.path.isfile)

print("Jumlah baris awal:", len(df))
print("Before tidak ditemukan:", (~df['before_exists']).sum())
print("After tidak ditemukan:",  (~df['after_exists']).sum())

# Sekarang filter
df = df[df['label'].between(1,7)]
df = df[df['before_exists'] & df['after_exists']]
df = df.reset_index(drop=True)
print("Sisa data setelah filter:", len(df))

df['before'] = df['before'].astype(str).str.strip() + ".jpg"
df['after']  = df['after'].astype(str).str.strip() + ".jpg"

# 5. Split dataset 70/15/15 (stratified by label)
train_df, tmp = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(tmp, test_size=0.50, random_state=42, stratify=tmp['label'])

for name, part in [('train',train_df), ('val',val_df), ('test',test_df)]:
    part[['before_path','after_path','label']].to_csv(os.path.join(OUT_DIR, f"{name}.csv"), index=False)

print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

warns = []
def _per_class(df_part, name):
    counts = df_part['label'].value_counts().sort_index()
    print(f"\n{name} per class:")
    for c, v in counts.items():
        print(f"  class {int(c)}: {v}")
    print("  total:", counts.sum())

_per_class(train_df, "TRAIN")
_per_class(val_df,   "VAL")
_per_class(test_df,  "TEST")

if warns:
    print("\n".join(warns))
