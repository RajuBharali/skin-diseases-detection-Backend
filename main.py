from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import uuid

from predictor import predict_skin

app = FastAPI(
    title="AI Skin Disease Detection API",
    version="3.1"
)

# ===============================
# Enable CORS (for Next.js)
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Create uploads folder if missing
# ===============================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        # Generate unique filename
        unique_name = f"{uuid.uuid4().hex}_{file.filename}"
        file_location = os.path.join(UPLOAD_FOLDER, unique_name)

        # Save uploaded file
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run AI prediction
        result = predict_skin(file_location)

        # Optional: delete file after prediction
        os.remove(file_location)

        return result

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )