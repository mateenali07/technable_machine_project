import logging
import sys

def setup_logging():
    """
    Configures centralized logging for the backend application.
    Provides human-readable output indicating timestamp, log level, module name, and the actual message.
    Useful for monitoring endpoint triggers and machine learning progress.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Optional: Suppress noisy logging from third-party libraries (like PyTorch/urllib3)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
