# ✅ age_split_testset.py – 각 연령별 Fracture 10명 + Normal 10명 추출
import os
import shutil
import pandas as pd
from load_new_dxmodule_0620_ao import get_combined_dataset_ao

# 설정
OUTPUT_DIR = "test_set_images_0704"
CSV_PATH = "test_set_0704.csv"
YEARS = list(range(0, 20))  # 0~19세 대상
N_PER_CLASS = 10  # Fracture / Normal 각 10명씩

# 1. Kaggle 데이터만 필터링
df = get_combined_dataset_ao()
df = df[df['source'] == 'kaggle']
df = df[df['metal'] != 1]  # 👉 metal == 1 제외
df = df[df['age'].notnull()]
df['age'] = df['age'].astype(int)

# 2. test set 추출
os.makedirs(OUTPUT_DIR, exist_ok=True)
test_rows = []

for i in YEARS:
    bin_df = df[(df['age'] >= i) & (df['age'] < i + 1)]
    if len(bin_df) == 0:
        continue

    fracture_df = bin_df[bin_df["fracture_visible"] == 1]
    normal_df = bin_df[bin_df["fracture_visible"] == 0]

    if len(fracture_df) == 0 or len(normal_df) == 0:
        print(f"⚠️ Age {i}~{i+1}: 한쪽 클래스가 부족 → 건너뜀")
        continue

    sampled_frac = fracture_df.sample(min(N_PER_CLASS, len(fracture_df)), random_state=42)
    sampled_norm = normal_df.sample(min(N_PER_CLASS, len(normal_df)), random_state=42)
    sampled = pd.concat([sampled_frac, sampled_norm], ignore_index=True)

    target_dir = os.path.join(OUTPUT_DIR, f"age_{i}_{i+1}")
    os.makedirs(target_dir, exist_ok=True)

    for _, row in sampled.iterrows():
        src = row['image_path']
        if not os.path.exists(src):
            continue
        fname = os.path.basename(src)
        dst = os.path.join(target_dir, fname)
        shutil.copyfile(src, dst)

    test_rows.append(sampled)

# 3. test_set.csv 저장
if test_rows:
    final_df = pd.concat(test_rows, ignore_index=True)
    final_df[['filestem', 'image_path', 'age', 'fracture_visible', 'gender', 'ao_primary', 'ao_subtypes']].to_csv(CSV_PATH, index=False)
    print(f"✅ Test set saved: {CSV_PATH} with {len(final_df)} samples")
    print(f"🗂️ Total images copied to test set: {len(final_df)}")
else:
    print("❌ No test data was extracted due to class imbalance.")

# ✅ 백업
try:
    shutil.copy("test_set.csv", "age_test.csv")
    print("📁 Saved age_test.csv for fixed test set.")
except Exception as e:
    print(f"❌ Failed to copy test_set.csv to age_test.csv: {e}")
