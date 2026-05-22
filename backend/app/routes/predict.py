import io
import logging
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from PIL import Image

from app.models.schemas import PredictionResponse
from app.services.ml_service import MLService, get_ml_service
from app.routes.dataset import validate_image_file

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Inference & Prediction"])

@router.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(..., description="Single image to classify"),
    ml_service: MLService = Depends(get_ml_service)
):
    """
    Classifies a single image using the trained Teachable Machine model.
    1. Validates that the uploaded file is a valid image.
    2. Converts file bytes into a PIL Image.
    3. Runs feature extraction and prediction.
    4. Returns winning category name, confidence score, and scores for all classes.
    
    If the model weights are not found, returns a 400 Bad Request warning.
    """
    # 1. Image Validation
    validate_image_file(file)
    
    try:
        # 2. Read file bytes and load into PIL
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.load()  # Force load bytes
    except Exception as e:
        logger.error(f"Failed to read/decode prediction image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is corrupted or cannot be opened as a valid image."
        )

    # 3. Perform prediction
    try:
        predicted_class, confidence, all_confidences = ml_service.predict_image(pil_image)
    except ValueError as e:
        # Raised if model weights are missing or not trained yet
        logger.warning(f"Prediction requested but model is not trained: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred on the server during model prediction."
        )
        
    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        all_confidences=all_confidences
    )

from fastapi.responses import FileResponse
from app.core.config import settings
import os

@router.get("/download-model")
async def download_model():
    """
    Downloads the trained scikit-learn Logistic Regression model.
    """
    if not os.path.exists(settings.model_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model file not found. Have you trained the model yet?"
        )
    return FileResponse(
        path=settings.model_path,
        filename="teachable_machine_model.joblib",
        media_type="application/octet-stream"
    )
