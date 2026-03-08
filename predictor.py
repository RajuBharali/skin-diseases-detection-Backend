# ==========================================================
# IMPORT LIBRARIES
# ==========================================================
import numpy as np
import json
import cv2
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess


# ==========================================================
# LOAD CONFIGURATION
# ==========================================================
with open("model_config.json") as f:
    CONFIG = json.load(f)

IMG_SIZE = CONFIG["image_size"]


# ==========================================================
# LOAD MODELS (LOAD ONCE INTO MEMORY)
# ==========================================================
stage1_model = load_model(CONFIG["stage1"]["model_path"])
stage2_model = load_model(CONFIG["stage2"]["model_path"])
stage3_model = load_model(CONFIG["stage3"]["model_path"])


# ==========================================================
# AUTO CROP SKIN / LESION AREA
# ==========================================================
def auto_crop_skin(pil_image):

    img = np.array(pil_image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return pil_image

    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    cropped = img[y:y+h, x:x+w]

    return Image.fromarray(cropped)


# ==========================================================
# IMAGE ENHANCEMENT (CLAHE CONTRAST)
# ==========================================================
def enhance_image(pil_image):

    img = np.array(pil_image)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8,8)
    )

    cl = clahe.apply(l)

    merged = cv2.merge((cl,a,b))

    enhanced = cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2RGB
    )

    return Image.fromarray(enhanced)


# ==========================================================
# IMAGE PREPROCESSING PIPELINE
# ==========================================================
def preprocess_image(pil_image):

    # Auto crop lesion
    pil_image = auto_crop_skin(pil_image)

    # Enhance contrast
    pil_image = enhance_image(pil_image)

    # Resize for model
    img = pil_image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img)

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ==========================================================
# MAIN PREDICTION FUNCTION
# ==========================================================
def predict_skin_from_array(pil_image):

    img_array = preprocess_image(pil_image)

    # ======================================================
    # STAGE 1 — Healthy vs Diseased
    # ======================================================
    img_s1 = mobilenet_preprocess(img_array.copy())

    s1 = float(
        stage1_model.predict(
            img_s1,
            verbose=0
        )[0][0]
    )

    healthy_prob = s1
    diseased_prob = 1 - s1

    stage1_report = {
        "healthy_probability": round(healthy_prob, 4),
        "diseased_probability": round(diseased_prob, 4)
    }

    if diseased_prob < CONFIG["stage1"]["healthy_threshold"]:

        return {
            "stage1": stage1_report,
            "final_decision": {
                "stage": 1,
                "result": "Healthy",
                "confidence_percent": round(healthy_prob * 100, 2),
                "type": "Normal",
                "medical_advice": "✅ Skin appears healthy."
            }
        }


    # ======================================================
    # STAGE 2 — CANCER CLASSIFICATION
    # ======================================================
    img_s2 = eff_preprocess(img_array.copy())

    s2 = stage2_model.predict(
        img_s2,
        verbose=0
    )[0]

    classes2 = CONFIG["stage2"]["classes"]

    mel_prob = float(s2[classes2.index("mel")])
    bcc_prob = float(s2[classes2.index("bcc")])
    nv_prob  = float(s2[classes2.index("nv")])

    max_cancer_prob = max(mel_prob, bcc_prob)

    cancer_class = "mel" if mel_prob >= bcc_prob else "bcc"

    stage2_report = {
        "mel": round(mel_prob, 4),
        "bcc": round(bcc_prob, 4),
        "nv": round(nv_prob, 4)
    }


    # ======================================================
    # STAGE 3 — GENERAL SKIN DISEASE MODEL
    # ======================================================
    s3 = stage3_model.predict(
        img_s2,
        verbose=0
    )[0]

    classes3 = CONFIG["stage3"]["classes"]

    idx3 = int(np.argmax(s3))

    general_class = classes3[idx3]

    general_prob = float(s3[idx3])

    stage3_report = {
        classes3[i]: round(float(s3[i]), 4)
        for i in range(len(classes3))
    }


    # ======================================================
    # FINAL SAFE DECISION ENGINE
    # ======================================================

    # 🔴 Strong Cancer Rule
    if max_cancer_prob >= 0.75 and max_cancer_prob > nv_prob:

        return {
            "stage1": stage1_report,
            "stage2": stage2_report,
            "final_decision": {
                "stage": 2,
                "result": cancer_class,
                "confidence_percent": round(max_cancer_prob * 100, 2),
                "type": "Cancer",
                "medical_advice": "⚠️ Possible skin cancer detected."
            }
        }

    # 🟢 Benign Mole Protection
    if nv_prob >= 0.60 and nv_prob > mel_prob and nv_prob > bcc_prob:

        return {
            "stage1": stage1_report,
            "stage2": stage2_report,
            "final_decision": {
                "stage": 2,
                "result": "nv",
                "confidence_percent": round(nv_prob * 100, 2),
                "type": "Benign Lesion",
                "medical_advice": "🟢 Likely benign mole."
            }
        }

    # 🟡 Use General Skin Disease Model
    return {
        "stage1": stage1_report,
        "stage2": stage2_report,
        "stage3": stage3_report,
        "final_decision": {
            "stage": 3,
            "result": general_class,
            "confidence_percent": round(general_prob * 100, 2),
            "type": "General Skin Condition",
            "medical_advice": "🩺 Non-cancer skin condition."
        }
    }