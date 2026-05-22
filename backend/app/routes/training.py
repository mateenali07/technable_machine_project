import logging
from fastapi import APIRouter, Depends, HTTPException, status
from app.models.schemas import GenericResponse
from app.services.ml_service import MLService, get_ml_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Training Engine"])

@router.post("/train", response_model=GenericResponse)
def train_model(ml_service: MLService = Depends(get_ml_service)):
    """
    Triggers the transfer learning pipeline:
    1. Scans the 'data/' directory.
    2. Validates categories (requires at least 2 distinct directories with images).
    3. Preprocesses images and runs them through MobileNetV3 to extract features.
    4. Fits a Logistic Regression classifier on top of the features.
    5. Saves model weights and mapping metadata to disk.
    
    Catches dataset validation errors gracefully and returns descriptive messages.
    """
    logger.info("Training endpoint triggered. Starting feature extraction and fitting classifier...")
    
    # Run the ML classifier training pipeline
    success, message = ml_service.train_classifier()
    
    if not success:
        # A dataset validation error or training error occurred.
        # We return a 400 Bad Request to inform the user/client rather than crashing the server.
        logger.warning(f"Training failed: {message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
        
    logger.info("Training complete. Classifier is ready.")
    return GenericResponse(
        success=True,
        message=message
    )
