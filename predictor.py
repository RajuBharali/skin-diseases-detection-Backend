# ==========================================================
# IMPORT LIBRARIES
# ==========================================================
import numpy as np
import json

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
# LOAD MODELS
# ==========================================================
stage1_model = load_model(CONFIG["stage1"]["model_path"])   # Healthy vs Diseased
stage2_model = load_model(CONFIG["stage2"]["model_path"])   # Cancer Model
stage3_model = load_model(CONFIG["stage3"]["model_path"])   # General Model


# ==========================================================
# MAIN PREDICTION FUNCTION
# ==========================================================
def predict_skin_from_array(pil_image):

    # ======================================================
    # PREPROCESS IMAGE
    # ======================================================
    img = pil_image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # ======================================================
    # STAGE 1 : HEALTHY VS DISEASE
    # ======================================================
    img_s1 = mobilenet_preprocess(img_array.copy())

    s1 = float(stage1_model.predict(img_s1, verbose=0)[0][0])

    healthy_prob = s1
    diseased_prob = 1 - s1

    stage1_report = {
        "healthy_probability": round(healthy_prob, 4),
        "diseased_probability": round(diseased_prob, 4)
    }

    # ✅ Healthy exit
    if diseased_prob < CONFIG["stage1"]["healthy_threshold"]:
        return {
            "stage1": stage1_report,
            "final_decision": {
                "stage": 1,
                "result": "Healthy",
                "confidence_percent": round(healthy_prob * 100, 2),
                "type": "Healthy",
                "medical_advice": CONFIG["warnings"]["healthy"]
            }
        }

    # ======================================================
    # STAGE 2 : CANCER MODEL (bcc, mel, nv)
    # ======================================================
    img_s2 = eff_preprocess(img_array.copy())

    s2 = stage2_model.predict(img_s2, verbose=0)[0]
    classes2 = CONFIG["stage2"]["classes"]

    # ✅ Safety check
    assert len(classes2) == len(s2), "Stage2 class mismatch"

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
    # STAGE 3 : GENERAL DISEASE MODEL
    # ======================================================
    s3 = stage3_model.predict(img_s2, verbose=0)[0]
    classes3 = CONFIG["stage3"]["classes"]

    # ✅ Safety check
    assert len(classes3) == len(s3), "Stage3 class mismatch"

    idx3 = int(np.argmax(s3))
    general_class = classes3[idx3]
    general_prob = float(s3[idx3])

    stage3_report = {
        classes3[i]: round(float(s3[i]), 4)
        for i in range(len(classes3))
    }

    # ======================================================
    # FINAL DECISION ENGINE
    # ======================================================

    # 🔴 HIGH RISK CANCER
    if max_cancer_prob >= 0.80:
        return {
            "stage1": stage1_report,
            "stage2": stage2_report,
            "final_decision": {
                "stage": 2,
                "result": cancer_class,
                "confidence_percent": round(max_cancer_prob * 100, 2),
                "type": "Cancer (High Risk)",
                "medical_advice": CONFIG["warnings"]["cancer_high"]
            }
        }

    # 🟠 MODERATE RISK
    if max_cancer_prob >= 0.65:
        return {
            "stage1": stage1_report,
            "stage2": stage2_report,
            "final_decision": {
                "stage": 2,
                "result": cancer_class,
                "confidence_percent": round(max_cancer_prob * 100, 2),
                "type": "Cancer (Moderate Risk)",
                "medical_advice": CONFIG["warnings"]["cancer_moderate"]
            }
        }

    # 🟢 BENIGN MOLE
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

    # 🟡 GENERAL DISEASE (Fallback)
    return {
        "stage1": stage1_report,
        "stage2": stage2_report,
        "stage3": stage3_report,
        "final_decision": {
            "stage": 3,
            "result": general_class,
            "confidence_percent": round(general_prob * 100, 2),
            "type": "General Skin Condition",
            "medical_advice": CONFIG["warnings"]["general"]
        }
    }