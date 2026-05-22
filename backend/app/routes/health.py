from fastapi import APIRouter, Depends
from app.models.schemas import HealthResponse
from app.services.ml_service import MLService, get_ml_service

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
def check_health(ml_service: MLService = Depends(get_ml_service)):
    """
    Check API Server and PyTorch device health.
    Returns:
    - status: "healthy"
    - device: "cuda" if GPU acceleration is active, otherwise "cpu".
    """
    return HealthResponse(
        status="healthy",
        device=str(ml_service.device)
    )
