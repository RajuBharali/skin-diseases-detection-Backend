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
stage1_model = load_model(CONFIG["stage1"]["model_path"])
stage2_model = load_model(CONFIG["stage2"]["model_path"])   # General Skin Model
stage3_model = load_model(CONFIG["stage3"]["model_path"])   # Cancer Model


# ==========================================================
# MAIN PREDICTION FUNCTION
# ==========================================================
def predict_skin_from_array(pil_image):

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
        "healthy_probability": round(healthy_prob,4),
        "diseased_probability": round(diseased_prob,4)
    }

    if diseased_prob < CONFIG["stage1"]["healthy_threshold"]:
        return {
            "stage1": stage1_report,
            "final_decision": {
                "stage":1,
                "result":"Healthy",
                "confidence_percent": round(healthy_prob*100,2),
                "type":"Healthy",
                "medical_advice":"✅ Skin appears healthy."
            }
        }

    # ======================================================
    # STAGE 2 : GENERAL SKIN DISEASE MODEL
    # ======================================================
    img_s2 = eff_preprocess(img_array.copy())

    s2 = stage2_model.predict(img_s2, verbose=0)[0]

    classes2 = CONFIG["stage2"]["classes"]

    idx2 = int(np.argmax(s2))

    general_class = classes2[idx2]
    general_prob = float(s2[idx2])

    stage2_report = {
        classes2[i]: round(float(s2[i]),4)
        for i in range(len(classes2))
    }


    # ======================================================
    # STAGE 3 : CANCER MODEL
    # ======================================================
    s3 = stage3_model.predict(img_s2, verbose=0)[0]

    classes3 = CONFIG["stage3"]["classes"]

    mel_prob = float(s3[classes3.index("mel")])
    bcc_prob = float(s3[classes3.index("bcc")])
    nv_prob  = float(s3[classes3.index("nv")])

    max_cancer_prob = max(mel_prob, bcc_prob)
    cancer_class = "mel" if mel_prob >= bcc_prob else "bcc"

    stage3_report = {
        "mel": round(mel_prob,4),
        "bcc": round(bcc_prob,4),
        "nv": round(nv_prob,4)
    }


    # ======================================================
    # FINAL DECISION ENGINE
    # ======================================================

    # 🔴 Strong cancer detection
    if max_cancer_prob >= 0.70 and max_cancer_prob > general_prob:

        return {
            "stage1": stage1_report,
            "stage2": stage2_report,
            "stage3": stage3_report,
            "final_decision": {
                "stage":3,
                "result": cancer_class,
                "confidence_percent": round(max_cancer_prob*100,2),
                "type":"Cancer",
                "medical_advice":"⚠️ Possible skin cancer detected. Please consult a dermatologist."
            }
        }

    # 🟢 Benign Mole Protection
    if nv_prob >= 0.60 and nv_prob > mel_prob and nv_prob > bcc_prob:

        return {
            "stage1": stage1_report,
            "stage3": stage3_report,
            "final_decision": {
                "stage":3,
                "result":"nv",
                "confidence_percent": round(nv_prob*100,2),
                "type":"Benign Lesion",
                "medical_advice":"🟢 Likely benign mole."
            }
        }

    # 🟡 Default to general disease
    return {
        "stage1": stage1_report,
        "stage2": stage2_report,
        "stage3": stage3_report,
        "final_decision": {
            "stage":2,
            "result": general_class,
            "confidence_percent": round(general_prob*100,2),
            "type":"General Skin Condition",
            "medical_advice":"🩺 Non-cancer skin condition."
        }
    }