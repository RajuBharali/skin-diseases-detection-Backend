from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image

from predictor import predict_skin_from_array

app = FastAPI(
    title="AI Skin Disease Detection API",
    version="3.2"
)

# ===============================
# Enable CORS (Next.js Production)
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://onelifeai.koyeb.app"],  # Change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image into memory
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        # Run prediction directly from image
        result = predict_skin_from_array(image)

        return result

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
