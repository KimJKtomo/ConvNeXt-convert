import sys
sys.path.append("/home/kimgk3793/Wrist_fracture/YOLOv9-Fracture-Detection/")

import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from timm import create_model
from torch.nn import Sigmoid
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from unified_dataset_0704 import UnifiedDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from models.common import DetectMultiBackend


# ✅ 설정
MODEL_BASE_PATH = "models_convnext/0722_best_ddp_convnextv2_agegroup{}.pt"
YOLO_WEIGHT = "/home/kimgk3793/Wrist_fracture/YOLOv9-Fracture-Detection/runs/train/yolov9-c7/weights/best.pt"
GT_LABEL_DIR = "/home/kimgk3793/Wrist_fracture/YOLOv9-Fracture-Detection/GRAZPEDWRI-DX/data/labels/test"
OUTPUT_DIR = "test_final_inference_results_gradcampp_yolo"
os.makedirs(OUTPUT_DIR, exist_ok=True)
img_dir ="original/"


CONF_THRES = 0.25
IOU_THRES = 0.3
CLASS_NAMES = {3: "fracture"}

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def AGE_GROUP_FN(age):
    age = float(age)
    if age < 5:
        return 0
    elif age < 10:
        return 1
    elif age < 15:
        return 2
    else:
        return 3

# ✅ 데이터 로딩
df_test = pd.read_csv("test_data_0704.csv")
df_test["age_group_label"] = df_test["age"].astype(float).apply(AGE_GROUP_FN)
df_test["label"] = df_test["fracture_visible"].apply(lambda x: 1.0 if x == 1 else 0.0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sigmoid = Sigmoid()

# ✅ YOLO 로딩
yolo_model = DetectMultiBackend(YOLO_WEIGHT, device=DEVICE)
yolo_model.eval()

all_results = []

# ✅ 연령 그룹별 실행
for AGE_GROUP in [0, 1, 2, 3]:
    print(f"\n🔍 Running inference for Age Group {AGE_GROUP}...")

    model = create_model("convnextv2_large.fcmae_ft_in22k_in1k_384", pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(MODEL_BASE_PATH.format(AGE_GROUP), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    cam = GradCAMPlusPlus(model=model, target_layers=[model.stages[-1].blocks[-1].conv_dw])

    df_group = df_test[df_test["age_group_label"] == AGE_GROUP].reset_index(drop=True)
    dataset = UnifiedDataset(df_group, transform=transform, task="fracture_only")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    group_dir = os.path.join(OUTPUT_DIR, f"agegroup{AGE_GROUP}")
    os.makedirs(group_dir, exist_ok=True)

    for i, (img_tensor, label) in enumerate(tqdm(loader)):
        img_tensor = img_tensor.to(DEVICE)

        with torch.no_grad():
            output = model(img_tensor).squeeze()
            prob = sigmoid(output).item()
            pred = int(prob > 0.5)

        # GradCAM++
        grayscale_cam = cam(input_tensor=img_tensor)[0]
        img_name = df_group.iloc[i]["filestem"]
        img_path = img_dir+img_name+".png"
        orig = cv2.imread(img_path)
        h, w = orig.shape[:2]
        rgb_img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB) / 255.0
        grayscale_cam = cv2.resize(grayscale_cam, (w, h))
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # YOLO Detection
        yolo_input = letterbox(orig, new_shape=640, stride=32, auto=True)[0]
        yolo_input = yolo_input[:, :, ::-1].transpose(2, 0, 1)
        yolo_input = np.ascontiguousarray(yolo_input)
        im = torch.from_numpy(yolo_input).to(DEVICE).float() / 255.0
        im = im.unsqueeze(0)

        with torch.no_grad():
            pred_yolo = yolo_model(im)[0][1]
            pred_yolo = non_max_suppression(pred_yolo, CONF_THRES, IOU_THRES, classes=[3])[0]

        overlay = cam_image.copy()

        # ✅ YOLO BBOX Overlay
        if pred_yolo is not None and len(pred_yolo) > 0:
            pred_yolo[:, :4] = scale_boxes(im.shape[2:], pred_yolo[:, :4], orig.shape).round()
            for *xyxy, conf, cls_id in pred_yolo:
                x1, y1, x2, y2 = map(int, xyxy)
                cls_name = CLASS_NAMES.get(int(cls_id), f"cls{int(cls_id)}")
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(overlay, f"YOLO: {cls_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # ✅ 정답 BBOX Overlay (filestem 기준)
        filestem = df_group.iloc[i]["filestem"]  # CSV에서 filestem 추출
        gt_txt_path = os.path.join(
            "home/kimgk3793/Wrist_fracture/YOLOv9-Fracture-Detection/GRAZPEDWRI-DX/data/labels",
            f"{filestem}.txt"
        )

        if os.path.exists(gt_txt_path):
            with open(gt_txt_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id, xc, yc, bw, bh = map(float, parts)
                    if int(cls_id) != 3:
                        continue
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(overlay, "GT: fracture", (x1, y2 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # ✅ 저장
        save_name = os.path.basename(img_path).replace(".png", f"_pred{pred}_label{int(label.item())}.png")
        cv2.imwrite(os.path.join(group_dir, save_name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        all_results.append({
            "filename": os.path.basename(img_path),
            "true_label": int(label.item()),
            "pred_label": pred,
            "probability": prob,
            "age_group": AGE_GROUP
        })

# ✅ CSV 저장
df_result = pd.DataFrame(all_results)
df_result.to_csv(os.path.join(OUTPUT_DIR, "final_inference_summary.csv"), index=False)

# ✅ Confusion Matrix
cm = confusion_matrix(df_result["true_label"], df_result["pred_label"])
print("\n📊 Final Confusion Matrix (All Age Groups):\n", cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fracture"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Final Test Set Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "final_confusion_matrix.png"))
plt.close()
