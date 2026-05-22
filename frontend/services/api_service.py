import logging
import requests
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class APIService:
    """
    APIService abstracts all network communication with our FastAPI backend.
    It encapsulates standard HTTP methods (GET, POST) and handles connection errors
    gracefully to ensure the Streamlit frontend remains stable even if the backend is down.
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        logger.info(f"APIService initialized pointing to backend: {self.base_url}")

    def check_health(self) -> Tuple[bool, str]:
        """
        Calls GET /health to check if the backend is active and what device is running.
        Returns:
            - is_healthy (bool)
            - device_name (str)
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=3.0)
            if response.status_code == 200:
                data = response.json()
                return True, data.get("device", "cpu")
        except requests.exceptions.RequestException:
            pass
        return False, "offline"

    def get_dataset_status(self) -> Optional[Dict[str, Any]]:
        """
        Calls GET /dataset-status to retrieve:
        - If a trained model is saved on disk.
        - Categories details (names and image sample counts).
        """
        try:
            response = requests.get(f"{self.base_url}/dataset-status", timeout=3.0)
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching dataset status: {e}")
        return None

    def upload_samples(self, class_name: str, files: List[Tuple[str, bytes]]) -> Tuple[bool, str]:
        """
        Calls POST /upload-sample to upload multiple images to a specific category.
        
        Args:
            class_name: The target category name.
            files: A list of tuples containing (filename, file_bytes).
            
        Returns:
            - success (bool)
            - message (str)
        """
        try:
            # Construct standard multipart/form-data payload
            # FastAPI receives 'class_name' as a form string, and 'files' as list of files
            data = {"class_name": class_name}
            
            # Prepare files list formatted for the requests library:
            # [("files", (filename, file_bytes, "image/jpeg")), ...]
            upload_files = []
            for filename, file_bytes in files:
                upload_files.append(("files", (filename, file_bytes, "image/jpeg")))
                
            response = requests.post(
                f"{self.base_url}/upload-sample",
                data=data,
                files=upload_files,
                timeout=15.0  # Allow slightly longer timeout for multiple file uploads
            )
            
            if response.status_code == 200:
                result = response.json()
                return True, f"Successfully uploaded {result['files_uploaded']} images (Total: {result['total_samples']})."
            else:
                detail = response.json().get("detail", "Unknown server error.")
                return False, f"Upload failed: {detail}"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during file upload: {e}")
            return False, "Network error: Could not reach the backend server."

    def train_model(self) -> Tuple[bool, str]:
        """
        Calls POST /train to trigger the transfer learning pipeline on the backend.
        
        Returns:
            - success (bool)
            - message (str)
        """
        try:
            # Long timeout is needed because training extracts features from all images sequentially
            response = requests.post(f"{self.base_url}/train", timeout=120.0)
            if response.status_code == 200:
                data = response.json()
                return True, data.get("message", "Model trained successfully!")
            else:
                detail = response.json().get("detail", "Training failed on server.")
                return False, detail
        except requests.exceptions.Timeout:
            return False, "Training request timed out. If you have a large dataset, wait a moment and refresh."
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during training: {e}")
            return False, "Network error: Could not connect to backend to trigger training."

    def predict_image(self, filename: str, image_bytes: bytes) -> Tuple[bool, Dict[str, Any]]:
        """
        Calls POST /predict to classify a single image.
        
        Returns:
            - success (bool)
            - prediction_details (Dict): contains predicted_class, confidence, all_confidences
        """
        try:
            files = {"file": (filename, image_bytes, "image/jpeg")}
            response = requests.post(f"{self.base_url}/predict", files=files, timeout=5.0)
            
            if response.status_code == 200:
                return True, response.json()
            else:
                detail = response.json().get("detail", "Prediction failed on server.")
                return False, {"message": detail}
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during prediction: {e}")
            return False, {"message": "Network error: Backend server is offline."}

    def clear_all(self) -> Tuple[bool, str]:
        """
        Calls POST /clear to delete all datasets and trained models, resetting the workspace.
        
        Returns:
            - success (bool)
            - message (str)
        """
        try:
            response = requests.post(f"{self.base_url}/clear", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                return True, data.get("message", "Workspace cleared successfully.")
            else:
                detail = response.json().get("detail", "Clear failed on server.")
                return False, detail
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during clearing: {e}")
            return False, "Network error: Could not connect to backend server."
