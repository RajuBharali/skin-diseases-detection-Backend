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
# LOAD MODELS (LOAD ONCE)
# ==========================================================
stage1_model = load_model(CONFIG["stage1"]["model_path"])
stage2_model = load_model(CONFIG["stage2"]["model_path"])
stage3_model = load_model(CONFIG["stage3"]["model_path"])


# ==========================================================
# LESION REGION DETECTION
# ==========================================================
def detect_lesion_roi(pil_image):

    img = np.array(pil_image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (7,7), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        3
    )

    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return pil_image

    largest = max(contours, key=cv2.contourArea)

    x,y,w,h = cv2.boundingRect(largest)

    pad = int(max(w,h) * 0.2)

    x = max(x-pad, 0)
    y = max(y-pad, 0)

    w = min(w + pad*2, img.shape[1]-x)
    h = min(h + pad*2, img.shape[0]-y)

    crop = img[y:y+h, x:x+w]

    return Image.fromarray(crop)


# ==========================================================
# SHARPEN IMAGE (TEXTURE ENHANCEMENT)
# ==========================================================
def sharpen_image(pil_image):

    img = np.array(pil_image)

    kernel = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ])

    sharp = cv2.filter2D(img, -1, kernel)

    return Image.fromarray(sharp)


# ==========================================================
# COLOR NORMALIZATION
# ==========================================================
def normalize_color(pil_image):

    img = np.array(pil_image)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    y, cr, cb = cv2.split(img)

    y = cv2.equalizeHist(y)

    img = cv2.merge((y, cr, cb))

    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

    return Image.fromarray(img)


# ==========================================================
# CLAHE CONTRAST ENHANCEMENT
# ==========================================================
def enhance_image(pil_image):

    img = np.array(pil_image)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    l,a,b = cv2.split(lab)

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

    # Detect lesion region
    pil_image = detect_lesion_roi(pil_image)

    # Improve texture
    pil_image = sharpen_image(pil_image)

    # Normalize lighting
    pil_image = normalize_color(pil_image)

    # Improve contrast
    pil_image = enhance_image(pil_image)

    # Resize
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
                "confidence_percent":round(healthy_prob*100,2),
                "type":"healthy",
                "medical_advice":"✅ Skin appears healthy."
            }
        }


    # ======================================================
    # STAGE 2 : SKIN CANCER MODEL
    # ======================================================
    img_s2 = eff_preprocess(img_array.copy())

    s2 = stage2_model.predict(img_s2, verbose=0)[0]

    classes2 = CONFIG["stage2"]["classes"]

    mel_prob = float(s2[classes2.index("mel")])
    bcc_prob = float(s2[classes2.index("bcc")])
    nv_prob  = float(s2[classes2.index("nv")])

    max_cancer_prob = max(mel_prob, bcc_prob)

    cancer_class = "mel" if mel_prob >= bcc_prob else "bcc"

    stage2_report = {
        "mel":round(mel_prob,4),
        "bcc":round(bcc_prob,4),
        "nv":round(nv_prob,4)
    }


    # ======================================================
    # STAGE 3 : GENERAL SKIN DISEASE MODEL
    # ======================================================
    s3 = stage3_model.predict(img_s2, verbose=0)[0]

    classes3 = CONFIG["stage3"]["classes"]

    idx3 = int(np.argmax(s3))

    general_class = classes3[idx3]

    general_prob = float(s3[idx3])

    stage3_report = {
        classes3[i]: round(float(s3[i]),4)
        for i in range(len(classes3))
    }


    # ======================================================
    # FINAL DECISION ENGINE
    # ======================================================

    # Strong cancer detection
    if max_cancer_prob >= 0.75 and max_cancer_prob > nv_prob:

        return {
            "stage1":stage1_report,
            "stage2":stage2_report,
            "final_decision":{
                "stage":2,
                "result":cancer_class,
                "confidence_percent":round(max_cancer_prob*100,2),
                "type":"cancer",
                "medical_advice":"⚠️ Possible skin cancer detected. Please consult a dermatologist."
            }
        }


    # Benign mole protection
    if nv_prob >= 0.60 and nv_prob > mel_prob and nv_prob > bcc_prob:

        return {
            "stage1":stage1_report,
            "stage2":stage2_report,
            "final_decision":{
                "stage":2,
                "result":"nv",
                "confidence_percent":round(nv_prob*100,2),
                "type":"disease",
                "medical_advice":"🟢 Likely benign mole."
            }
        }


    # General disease model
    return {
        "stage1":stage1_report,
        "stage2":stage2_report,
        "stage3":stage3_report,
        "final_decision":{
            "stage":3,
            "result":general_class,
            "confidence_percent":round(general_prob*100,2),
            "type":"disease",
            "medical_advice":"🩺 Non-cancer skin condition."
        }
    }