from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class HealthResponse(BaseModel):
    """
    Standard schema for service health status.
    """
    status: str = Field(..., description="Service status, e.g., 'healthy'")
    device: str = Field(..., description="Hardware device running PyTorch feature extraction (cpu or cuda)")

class ClassSampleCount(BaseModel):
    """
    Shows count of samples available for a specific class label.
    """
    class_name: str = Field(..., description="Custom name of the category/class")
    sample_count: int = Field(..., description="Number of uploaded image samples")

class TrainingStatusResponse(BaseModel):
    """
    Reports the status of the model training engine.
    """
    model_trained: bool = Field(..., description="Whether a valid trained model.pkl exists on disk")
    classes: List[ClassSampleCount] = Field(..., description="List of all class categories and their sample counts")
    message: str = Field(..., description="Status description or error details")

class UploadSamplesResponse(BaseModel):
    """
    Schema returned upon uploading image samples.
    """
    class_name: str = Field(..., description="Name of the class where files were uploaded")
    files_uploaded: int = Field(..., description="Number of successfully saved files")
    total_samples: int = Field(..., description="Total sample files currently inside this class folder")

class PredictionResponse(BaseModel):
    """
    Detailed classification result returned by the inference endpoint.
    """
    predicted_class: str = Field(..., description="Name of the class with the highest confidence")
    confidence: float = Field(..., description="Confidence score of the winning class (0.0 to 1.0)")
    all_confidences: Dict[str, float] = Field(..., description="Full mapping of all classes to their respective confidence scores")

class GenericResponse(BaseModel):
    """
    Standard message response for resets or general actions.
    """
    success: bool = Field(..., description="Action success indicator")
    message: str = Field(..., description="Detailed text message describing the action result")
