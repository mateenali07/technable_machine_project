import logging
import os
import shutil
import uuid
from typing import List
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, status
from app.core.config import settings
from app.models.schemas import UploadSamplesResponse, GenericResponse, TrainingStatusResponse, ClassSampleCount
from app.services.ml_service import MLService, get_ml_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Dataset Management"])

# Supported file extensions for validation
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def validate_image_file(file: UploadFile) -> None:
    """
    Validates that the uploaded file is indeed an image.
    Checks both the content-type header and the filename extension.
    """
    # 1. Check Content Type Header
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{file.filename}' is not a valid image. Received content type: {file.content_type}"
        )
    
    # 2. Check File Extension
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

@router.post("/upload-sample", response_model=UploadSamplesResponse)
async def upload_sample(
    class_name: str = Form(..., description="Category label name, e.g., 'dog'"),
    files: List[UploadFile] = File(..., description="List of image files to add to this category")
):
    """
    Uploads a batch of images to a specific class directory.
    - Validates image content types and extensions.
    - Creates the category subfolder dynamically if it doesn't exist.
    - Saves images with random UUIDs to avoid filename conflicts.
    """
    # Sanitize class_name (prevent path traversal attacks or invalid chars)
    class_name_clean = "".join(c for c in class_name if c.isalnum() or c in ("-", "_")).strip()
    if not class_name_clean:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid category name. It must contain letters, numbers, hyphens, or underscores."
        )

    # 1. Create class subdirectory path
    class_dir = settings.DATA_DIR / class_name_clean
    os.makedirs(class_dir, exist_ok=True)
    
    saved_count = 0
    for file in files:
        # Validate each image
        validate_image_file(file)
        
        # Generate random unique filename keeping the original extension
        file_ext = Path(file.filename or ".jpg").suffix.lower()
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        target_path = class_dir / unique_filename
        
        try:
            # Write bytes to disk
            with open(target_path, "wb") as buffer:
                # Read chunks to handle larger files efficiently
                while content := await file.read(1024 * 1024):  # 1MB chunks
                    buffer.write(content)
            saved_count += 1
        except Exception as e:
            logger.error(f"Error saving file {file.filename}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error saving file '{file.filename}' to disk."
            )
            
    # Calculate total files in folder currently
    total_samples = len([p for p in class_dir.iterdir() if p.is_file()])
    
    logger.info(f"Saved {saved_count} samples for class '{class_name_clean}'. Total samples in class: {total_samples}")
    
    return UploadSamplesResponse(
        class_name=class_name_clean,
        files_uploaded=saved_count,
        total_samples=total_samples
    )

@router.get("/dataset-status", response_model=TrainingStatusResponse)
def get_dataset_status():
    """
    Scans the dataset directory and returns:
    - If a trained model already exists on disk.
    - All categories currently defined and the number of image samples in each.
    """
    classes_info = []
    
    if settings.DATA_DIR.exists():
        for folder in sorted(settings.DATA_DIR.iterdir()):
            if folder.is_dir():
                # Count files with valid extensions
                sample_count = len([
                    p for p in folder.iterdir() 
                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
                ])
                classes_info.append(
                    ClassSampleCount(class_name=folder.name, sample_count=sample_count)
                )
                
    model_exists = settings.model_path.exists() and settings.class_mapping_path.exists()
    
    msg = "Ready to train." if len(classes_info) >= 2 else "Requires at least 2 classes with data before training."
    if model_exists:
        msg = "Model trained and ready for prediction."
        
    return TrainingStatusResponse(
        model_trained=model_exists,
        classes=classes_info,
        message=msg
    )

@router.post("/clear", response_model=GenericResponse)
def clear_all_data():
    """
    Completely deletes all uploaded images, labels, and saved model weights.
    Resets the workspace to start fresh.
    """
    # 1. Reset ML Service Singleton state
    try:
        from app.services.ml_service import get_ml_service
        ml_service = get_ml_service()
        ml_service.reset_model()
    except Exception as e:
        logger.warning(f"Could not reset ML service state: {e}")

    # 2. Delete raw data directories
    if settings.DATA_DIR.exists():
        try:
            shutil.rmtree(settings.DATA_DIR)
            os.makedirs(settings.DATA_DIR, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to delete data directory: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not delete image dataset on disk: {e}"
            )
            
    # 3. Delete saved models
    if settings.SAVED_MODELS_DIR.exists():
        try:
            shutil.rmtree(settings.SAVED_MODELS_DIR)
            os.makedirs(settings.SAVED_MODELS_DIR, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to delete models directory: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not delete model weights on disk: {e}"
            )

    logger.info("Workspace has been fully reset. All datasets and models cleared.")
    return GenericResponse(
        success=True,
        message="All datasets and model files have been successfully deleted from disk."
    )
