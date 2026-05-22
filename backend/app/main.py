import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging
from app.routes import health, dataset, training, predict

# 1. Setup Logging Configuration
# Initializes the logger before starting the FastAPI app to capture all boot/lifespan sequences.
setup_logging()
logger = logging.getLogger("app.main")

# 2. FastAPI Lifespan Manager
# Lifespan events allow us to run setup code before the server starts accepting requests,
# and cleanup code when the server shuts down.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Sequence:
    logger.info("FastAPI Teachable Machine backend starting up...")
    logger.info(f"Storage directories initialized:")
    logger.info(f" - Dataset raw directory: {settings.DATA_DIR}")
    logger.info(f" - Saved models directory: {settings.SAVED_MODELS_DIR}")
    
    # Pre-warming PyTorch model:
    # Loading MobileNetV3 weights can take a few seconds. We trigger it now so the very first
    # web request receives a fast response, rather than waiting for weights to download/load on demand.
    try:
        from app.services.ml_service import get_ml_service
        logger.info("Pre-warming PyTorch MobileNetV3 model...")
        ml_service = get_ml_service()
        logger.info(f"Model successfully loaded. Running on device: {ml_service.device}")
    except Exception as e:
        logger.error(f"Error pre-warming machine learning model: {e}")
        logger.warning("Application will start, but model loading will be retried on-demand.")
        
    yield
    
    # Shutdown Sequence:
    logger.info("FastAPI Teachable Machine backend shutting down...")

# 3. Initialize FastAPI Application
app = FastAPI(
    title="Teachable Machine Backend API",
    description=(
        "Production-style REST API reproducing Google's Teachable Machine pipeline. "
        "Allows uploading image categories, training a Logistic Regression model on top of "
        "pre-trained MobileNetV3 PyTorch features on-the-fly, and running real-time predictions."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# 4. Configure Cross-Origin Resource Sharing (CORS)
# Cross-Origin Resource Sharing (CORS) is a browser security mechanism.
# Since our Streamlit frontend may run on a different port or host (e.g. localhost:8501)
# than our FastAPI backend (e.g. localhost:8000), we must enable CORS to allow the frontend
# browser requests to communicate with this API safely.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains; '*' is perfect for local dev
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all request headers
)

# 5. Include Routers
# Mounts the routing modules defined in app/routes/ directly to the root path.
app.include_router(health.router)
app.include_router(dataset.router)
app.include_router(training.router)
app.include_router(predict.router)

# Route index for documentation convenience
@app.get("/", tags=["Root"])
def root():
    """
    Root endpoint redirecting developers to the interactive API documentation.
    """
    return {
        "message": "Teachable Machine Backend API is running.",
        "documentation": "/docs",
        "health": "/health",
        "dataset_status": "/dataset-status"
    }
