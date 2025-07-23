import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# ✅ 경로 설정
csv_path = "/mnt/data/KimJG/Yolo_test/YOLOv9-Fracture-Detection/GRAZPEDWRI-DX/test_data_0704.csv"
pred_dir = Path("/mnt/data/KimJG/Yolo_test/YOLOv9-Fracture-Detection/runs/detect/exp_0712/labels")

# ✅ CSV 로드
df = pd.read_csv(csv_path)

# ✅ GT와 예측 리스트 생성
y_true = []
y_pred = []

for _, row in df.iterrows():
    filestem = row["filestem"]
    gt = int(row["fracture_visible"])  # 0 or 1
    y_true.append(gt)

    txt_path = pred_dir / f"{filestem}.txt"
    if txt_path.exists():
        found = False
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 0 and parts[0] == "3":  # class 3 = fracture
                    found = True
                    break
        y_pred.append(1 if found else 0)
    else:
        y_pred.append(0)

# ✅ Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fracture"])
disp.plot(cmap=plt.cm.Blues)
plt.title("YOLO vs fracture_visible (class 3)")
plt.savefig("yolo_confusion_matrix_from_fracture_visible.png")
plt.show()
