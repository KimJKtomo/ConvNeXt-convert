# run_final_inference_convnextv2.py
import sys
sys.path.append("/mnt/data/KimJG/Yolo_test/YOLOv9-Fracture-Detection")
import os, torch, cv2, numpy as np, pandas as pd
from PIL import Image
from pathlib import Path
from timm import create_model
from torchvision import transforms
from torch.nn import Sigmoid
from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =======================
# Config & Constants
# =======================
# df_test = pd.read_csv("test_set_0704.csv")
CSV_PATH = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250623_Single/test_set_0704.csv"
# IMG_DIR = "/mnt/data/KimJG/Yolo_test/YOLOv9-Fracture-Detection/GRAZPEDWRI-DX/data/images/test"
# CSV_PATH = "/mnt/data/KimJG/Yolo_test/YOLOv9-Fracture-Detection/GRAZPEDWRI-DX/test_data.csv"
YOLO_WEIGHT = "/mnt/data/KimJG/Yolo_test/YOLOv9-Fracture-Detection/runs/train/yolov9-c3/weights/best.pt"
MODEL_BASE_PATH = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250623_Single/0722_best_ddp_convnextv2_agegroup{}.pt"
# MODEL_BASE_PATH = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250623_Single/0718_convnextv2_base_fracture_agegroup{}.pt"

SAVE_DIR = "final_outputs_convnextv2_0723"
os.makedirs(f"{SAVE_DIR}/overlay", exist_ok=True)

DEVICE = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cpu")
CONF_THRES = 0.25
IOU_THRES = 0.3
ALPHA = 0.6
FINAL_THRESH = 0.4
IMG_SIZE = 384

CLASS_NAMES = {3: "fracture"}  # YOLO

# =======================
# Transform
# =======================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =======================
# Age Group 함수
# =======================
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

# =======================
# Load YOLO Model
# =======================
yolo_model = DetectMultiBackend(YOLO_WEIGHT, device=DEVICE)
yolo_model.eval()

# =======================
# Load ConvNeXt Models
# =======================
conv_models = {}
cam_modules = {}
sigmoid = Sigmoid()

for age_group in range(4):
    model = create_model(
        "convnextv2_large.fcmae_ft_in22k_in1k_384",
        pretrained=False,
        num_classes=1,
        drop_rate=0.0,
        drop_path_rate=0.0
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_BASE_PATH.format(age_group), map_location=DEVICE))
    model.eval()
    conv_models[age_group] = model

    cam = HiResCAM(model=model, target_layers=[model.stages[-1].blocks[-1]].conv_dw)
    cam_modules[age_group] = cam

# =======================
# Load test_set_0704.csv
# =======================
df = pd.read_csv(CSV_PATH)
df["label"] = df["fracture_visible"].apply(lambda x: 1.0 if x == 1 else 0.0)
df["age_group"] = df["age"].apply(AGE_GROUP_FN)
y_true = df["fracture_visible"].astype(int)

results = []

print(f"\\n🚀 Start Inference for {len(df)} images...")
for _, row in df.iterrows():
    img_path = row["image_path"]
    filename = os.path.basename(img_path)
    if not os.path.exists(img_path):
        continue

    # ========== Classification ==========
    age_group = row["age_group"]
    model = conv_models[age_group]
    cam = cam_modules[age_group]

    image_pil = Image.open(img_path).convert("RGB")
    img_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor).squeeze()
        fx_prob = sigmoid(output).item()
        fx_pred = int(fx_prob > FINAL_THRESH)

    # ========== Grad-CAM ==========
    img_cv2 = cv2.imread(img_path)
    h, w = img_cv2.shape[:2]
    rgb_img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB) / 255.0
    grayscale_cam = cam(input_tensor=img_tensor)[0]
    grayscale_cam = cv2.resize(grayscale_cam, (w, h))
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # ========== YOLO Detection ==========
    yolo_input = letterbox(img_cv2, new_shape=640, stride=32, auto=True)[0]
    yolo_input = yolo_input[:, :, ::-1].transpose(2, 0, 1)
    yolo_input = np.ascontiguousarray(yolo_input)
    im = torch.from_numpy(yolo_input).to(DEVICE).float() / 255.0
    im = im.unsqueeze(0)

    with torch.no_grad():
        pred = yolo_model(im)[0][1]
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=[3])[0]

    det_score = 0.0
    overlay = cam_image.copy()

    if pred is not None and len(pred) > 0:
        pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], img_cv2.shape).round()
        for *xyxy, conf, cls_id in pred:
            x1, y1, x2, y2 = map(int, xyxy)
            cls_name = CLASS_NAMES.get(int(cls_id), f"cls{int(cls_id)}")
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(overlay, f"YOLO: {cls_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            det_score = max(det_score, conf.item())

    # ========== Soft Voting ==========
    final_score = ALPHA * det_score + (1 - ALPHA) * fx_prob
    final_pred = int(final_score > FINAL_THRESH)

    # ========== Save Image ==========
    save_path = os.path.join(SAVE_DIR, "overlay", filename)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # ========== Save Results ==========
    results.append({
        "filename": filename,
        "age": row["age"],
        "age_group": age_group,
        "fx_prob": round(fx_prob, 4),
        "fx_pred": fx_pred,
        "det_score": round(det_score, 4),
        "det_pred": int(det_score > CONF_THRES),
        "final_score": round(final_score, 4),
        "final_pred": final_pred
    })

# =======================
# Save CSV
# =======================
df_out = pd.DataFrame(results)
df_out.to_csv(os.path.join(SAVE_DIR, "ensemble_results_convnextv2.csv"), index=False)
print(f"\\n✅ Saved inference results to ensemble_results_convnextv2.csv")
print(f"🖼️ Overlay images saved to: {SAVE_DIR}/overlay/")

   #------

#
# # =======================
# # Load CSV + Run Inference
# # =======================
# df = pd.read_csv(CSV_PATH)
# df["filename"] = df["filestem"].apply(lambda x: f"{x}.png")
# df["age_group"] = df["age"].apply(AGE_GROUP_FN)
#
# results = []
#
# print(f"\n🚀 Start Inference for {len(df)} images...")
# for _, row in df.iterrows():
#     filename = row["filename"]
#     img_path = os.path.join(IMG_DIR, filename)
#     if not os.path.exists(img_path):
#         continue
#
#     # ========== Classification ==========
#     age_group = row["age_group"]
#     model = conv_models[age_group]
#     cam = cam_modules[age_group]
#
#     image_pil = Image.open(img_path).convert("RGB")
#     img_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         output = model(img_tensor).squeeze()
#         fx_prob = sigmoid(output).item()
#         fx_pred = int(fx_prob > FINAL_THRESH)
#
#     # ========== Grad-CAM ==========
#     img_cv2 = cv2.imread(img_path)
#     h, w = img_cv2.shape[:2]
#     rgb_img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB) / 255.0
#     grayscale_cam = cam(input_tensor=img_tensor)[0]
#     grayscale_cam = cv2.resize(grayscale_cam, (w, h))
#     cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
#
#     # ========== YOLO Detection ==========
#     yolo_input = letterbox(img_cv2, new_shape=640, stride=32, auto=True)[0]
#     yolo_input = yolo_input[:, :, ::-1].transpose(2, 0, 1)
#     yolo_input = np.ascontiguousarray(yolo_input)
#     im = torch.from_numpy(yolo_input).to(DEVICE).float() / 255.0
#     im = im.unsqueeze(0)
#
#     with torch.no_grad():
#         pred = yolo_model(im)[0][1]
#         pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=[3])[0]
#
#     det_score = 0.0
#     overlay = cam_image.copy()
#
#     if pred is not None and len(pred) > 0:
#         pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], img_cv2.shape).round()
#         for *xyxy, conf, cls_id in pred:
#             x1, y1, x2, y2 = map(int, xyxy)
#             cls_name = CLASS_NAMES.get(int(cls_id), f"cls{int(cls_id)}")
#             cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
#             cv2.putText(overlay, f"YOLO: {cls_name} {conf:.2f}", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
#             det_score = max(det_score, conf.item())
#
#     # ========== Soft Voting ==========
#     final_score = ALPHA * det_score + (1 - ALPHA) * fx_prob
#     final_pred = int(final_score > FINAL_THRESH)
#
#     # ========== Save Image ==========
#     save_path = os.path.join(SAVE_DIR, "overlay", filename)
#     cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
#
#     # ========== Save Results ==========
#     results.append({
#         "filename": filename,
#         "age": row["age"],
#         "age_group": age_group,
#         "fx_prob": round(fx_prob, 4),
#         "fx_pred": fx_pred,
#         "det_score": round(det_score, 4),
#         "det_pred": int(det_score > CONF_THRES),
#         "final_score": round(final_score, 4),
#         "final_pred": final_pred
#     })
#
# # =======================
# # Save CSV
# # =======================
# df_out = pd.DataFrame(results)
# df_out.to_csv(os.path.join(SAVE_DIR, "ensemble_results_convnextv2.csv"), index=False)
# print(f"\n✅ Saved inference results to ensemble_results_convnextv2.csv")
# print(f"🖼️ Overlay images saved to: {SAVE_DIR}/overlay/")
