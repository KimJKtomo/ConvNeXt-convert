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

# ✅ 설정
MODEL_BASE_PATH = "0721_convnextv2_base_fracture_agegroup{}.pt"
OUTPUT_BASE_DIR = "0721_gradcampp_results_convnextv2"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# ✅ 이미지 변환
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ 나이 그룹 라벨 함수
def AGE_GROUP_FN(age):
    age = float(age)
    if age < 1.5:
        return 0
    elif age < 5:
        return 1
    elif age < 10:
        return 2
    elif age < 15:
        return 3
    else:
        return 4

# ✅ 각 연령대 모델에 대해 GradCAM++ 수행
for AGE_GROUP in [0, 1, 2, 3, 4]:
    print(f"\n🔍 GradCAM++ for Age Group {AGE_GROUP}...")
    MODEL_PATH = MODEL_BASE_PATH.format(AGE_GROUP)
    OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, f"agegroup{AGE_GROUP}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ 모델 로딩
    model = create_model(
        "convnextv2_base.fcmae_ft_in22k_in1k_384",
        pretrained=False,
        num_classes=1,
        drop_rate=0.0,
        drop_path_rate=0.0
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # ✅ CAM 대상 레이어
    target_layers = [model.stages[-1].blocks[-1]]
    sigmoid = Sigmoid()

    # ✅ 테스트셋 필터링
    df_test = pd.read_csv("test_set_0704.csv")
    df_test = df_test[df_test["age"].astype(float).apply(AGE_GROUP_FN) == AGE_GROUP]
    df_test["label"] = df_test["fracture_visible"].apply(lambda x: 1.0 if x == 1 else 0.0)

    val_dataset = UnifiedDataset(df_test, transform=transform, task="fracture_only")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    results = []
    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        for i, (img_tensor, label) in enumerate(tqdm(val_loader)):
            img_tensor = img_tensor.to(DEVICE)
            with torch.no_grad():
                output = model(img_tensor).squeeze()
                prob = sigmoid(output).item()
                pred = int(prob > 0.5)

            # ✅ 원본 이미지 로딩
            img_path = df_test.iloc[i]["image_path"]
            orig = cv2.imread(img_path)
            h, w = orig.shape[:2]
            rgb_img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB) / 255.0

            # ✅ CAM 계산
            grayscale_cam = cam(input_tensor=img_tensor)[0]
            grayscale_cam = cv2.resize(grayscale_cam, (w, h))
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            save_name = os.path.basename(img_path).replace(".png", f"_pred{pred}_label{int(label.item())}.png")
            save_path = os.path.join(OUTPUT_DIR, save_name)
            cv2.imwrite(save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

            results.append({
                "filename": os.path.basename(img_path),
                "true_label": int(label.item()),
                "pred_label": pred,
                "probability": prob,
                "age_group": AGE_GROUP
            })

    # ✅ 결과 저장
    df_group = pd.DataFrame(results)
    df_group.to_csv(os.path.join(OUTPUT_DIR, "gradcam_summary.csv"), index=False)

    # ✅ Confusion Matrix 출력
    cm = confusion_matrix(df_group["true_label"], df_group["pred_label"])
    print(f"\n📊 Confusion Matrix for Age Group {AGE_GROUP}:\n{cm}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fracture"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Fracture Confusion Matrix (Age Group {AGE_GROUP})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_agegroup{AGE_GROUP}.png"))
    plt.close()
