import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Avoid OpenMP duplicate runtime library conflict crashes in Anaconda environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Settings(BaseSettings):
    """
    Centralized configuration for the Teachable Machine Backend.
    Uses pydantic-settings to automatically load environment variables
    from the OS, falling back to clean default values.
    """
    
    # Project Root Directory (backend/app/)
    APP_DIR: Path = Path(__file__).resolve().parent.parent
    BACKEND_ROOT: Path = APP_DIR.parent

    # Data & Model Directories
    # Storing uploaded training samples and model checkpoints.
    # Note: These directories are ignored by git in our .gitignore.
    DATA_DIR: Path = BACKEND_ROOT / "data"
    SAVED_MODELS_DIR: Path = BACKEND_ROOT / "saved_models"
    
    # ML Hyperparameters & Architecture Settings
    IMAGE_SIZE: int = 224  # Standard size for MobileNetV3 input
    
    # Image Normalization values (standard ImageNet mean and standard deviation)
    # PyTorch models pre-trained on ImageNet expect input images normalized this way.
    NORM_MEAN: list[float] = [0.485, 0.456, 0.406]
    NORM_STD: list[float] = [0.229, 0.224, 0.225]
    
    # Saved file names
    MODEL_FILENAME: str = "model.joblib"
    CLASS_MAPPING_FILENAME: str = "class_mapping.joblib"

    @property
    def model_path(self) -> Path:
        return self.SAVED_MODELS_DIR / self.MODEL_FILENAME

    @property
    def class_mapping_path(self) -> Path:
        return self.SAVED_MODELS_DIR / self.CLASS_MAPPING_FILENAME

    class Config:
        env_prefix = "TM_"  # Envs like TM_IMAGE_SIZE will override defaults

# Centralized settings instance
settings = Settings()

# Ensure directories exist automatically when configuration is loaded
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.SAVED_MODELS_DIR, exist_ok=True)
