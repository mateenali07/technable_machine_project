import logging
import os
import joblib
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression

from app.core.config import settings

logger = logging.getLogger(__name__)

class MLService:
    """
    MLService handles the core Machine Learning operations of our Teachable Machine.
    It encapsulates:
    1. Feature Extraction: Loading a pre-trained MobileNetV3 backbone and stripping its final classification layer.
    2. Data Preprocessing: Resizing, converting to tensor, and normalizing images uniformly.
    3. Transfer Learning: Fitting a fast Logistic Regression classifier on top of the extracted features.
    4. Inference/Prediction: Running new images through the feature extractor and classifier.
    """

    def __init__(self):
        # Determine if a GPU is available to accelerate feature extraction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"MLService initialized using device: {self.device}")

        # 1. Initialize MobileNetV3 Large Feature Extractor
        # MobileNetV3 is designed for mobile/edge use cases. It is lightweight, fast, and has 
        # strong representational capabilities. We use the DEFAULT pre-trained weights (trained on ImageNet).
        logger.info("Loading pre-trained MobileNetV3 Large weights...")
        self.weights = models.MobileNet_V3_Large_Weights.DEFAULT
        self.feature_extractor = models.mobilenet_v3_large(weights=self.weights)
        
        # Transfer Learning Concept:
        # Instead of training a massive neural network from scratch (which requires millions of images
        # and hours of compute), we keep the feature extraction "backbone" intact. It has already learned to
        # recognize edges, textures, shapes, and complex objects from ImageNet. We only replace the final 
        # classifier head (which converts features to 1,000 ImageNet categories) with an Identity layer, 
        # allowing us to extract the raw 960-dimensional high-level feature vectors.
        self.feature_extractor.classifier = nn.Identity()
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()  # Set model to evaluation mode (turns off dropout, batchnorm updates)

        # 2. Define Image Preprocessing Transforms
        # Uniformity is critical: during both training and real-time prediction (inference), 
        # images must go through the exact same preprocessing pipeline:
        # - Resized to settings.IMAGE_SIZE (224x224 pixels).
        # - Converted to PyTorch Tensor.
        # - Normalized using the ImageNet channel mean and standard deviation.
        self.preprocess = transforms.Compose([
            transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=settings.NORM_MEAN, std=settings.NORM_STD),
        ])

        # Lazy-loaded model weights
        self.classifier: Optional[LogisticRegression] = None
        self.class_mapping: Optional[Dict[int, str]] = None
        self._load_saved_model_if_exists()

    def _load_saved_model_if_exists(self) -> None:
        """
        Attempts to load the saved classifier and class mappings from disk if they exist.
        Allows the API to handle predictions immediately upon restart without re-training.
        """
        if settings.model_path.exists() and settings.class_mapping_path.exists():
            try:
                self.classifier = joblib.load(settings.model_path)
                self.class_mapping = joblib.load(settings.class_mapping_path)
                logger.info(f"Successfully loaded saved model and class mapping from {settings.SAVED_MODELS_DIR}")
            except Exception as e:
                logger.error(f"Failed to load saved model: {e}. Starting fresh.")
                self.classifier = None
                self.class_mapping = None

    def preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Applies PyTorch preprocessing transformations to a PIL Image.
        Returns a 3D Tensor of shape [3, 224, 224].
        """
        # Ensure image is in RGB mode (removes Alpha channel from PNGs or converts Grayscale)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        return self.preprocess(pil_image)

    @torch.no_grad()
    def extract_features_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Extracts high-level feature vectors for a batch of PIL Images.
        Returns a PyTorch Tensor of shape [batch_size, 960].
        """
        if not images:
            return torch.empty(0, 960).to(self.device)
            
        # Preprocess each image and stack them into a single batch tensor: shape [batch_size, 3, 224, 224]
        batch_tensors = torch.stack([self.preprocess_image(img) for img in images]).to(self.device)
        
        # Extract features using our MobileNetV3 backbone
        features = self.feature_extractor(batch_tensors)
        return features

    def train_classifier(self) -> Tuple[bool, str]:
        """
        Scans the data directory, extracts features for all samples,
        trains a Logistic Regression classifier on top, and saves the weights to disk.
        """
        data_dir = settings.DATA_DIR
        
        # 1. Read class folders inside the data directory
        if not data_dir.exists():
            return False, f"Dataset directory '{data_dir}' does not exist."
            
        class_folders = [f for f in data_dir.iterdir() if f.is_dir()]
        
        # Validation Rule: Must have at least 2 categories to train a classifier.
        if len(class_folders) < 2:
            return False, f"Need at least 2 class categories to train a model. Currently found: {len(class_folders)}"

        # 2. Gather image samples and associate them with labels
        features_list = []
        labels_list = []
        
        # Map class names to integer labels: {0: "Class A", 1: "Class B"}
        temp_class_mapping: Dict[int, str] = {}
        # Inverse mapping for training loop: {"Class A": 0, "Class B": 1}
        class_to_idx: Dict[str, int] = {}
        
        for idx, folder in enumerate(sorted(class_folders)):
            class_name = folder.name
            temp_class_mapping[idx] = class_name
            class_to_idx[class_name] = idx
            
            # Find all images in this folder (supporting JPG, JPEG, PNG, WEBP)
            supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}
            image_paths = [p for p in folder.iterdir() if p.suffix.lower() in supported_extensions]
            
            # Validation Rule: Each folder must have at least 1 training sample
            if not image_paths:
                return False, f"Category '{class_name}' is empty. Upload at least one image to this category."
            
            # We process images in small sub-batches to prevent memory issues
            batch_size = 32
            for i in range(0, len(image_paths), batch_size):
                chunk_paths = image_paths[i:i + batch_size]
                pil_images = []
                for p in chunk_paths:
                    try:
                        # Open and load image data
                        img = Image.open(p)
                        img.load()  # Force load PIL image bytes
                        pil_images.append(img)
                    except Exception as e:
                        logger.warning(f"Skipping corrupted image {p.name}: {e}")
                
                if pil_images:
                    # Extract features for this sub-batch: returns shape [len(pil_images), 960]
                    batch_features = self.extract_features_batch(pil_images)
                    features_list.append(batch_features.cpu())
                    labels_list.extend([idx] * len(pil_images))

        if not features_list:
            return False, "No valid images found to train on."

        # 3. Concatenate all batches into single arrays
        # Combine PyTorch tensors and convert them to NumPy arrays for scikit-learn
        X = torch.cat(features_list, dim=0).numpy()
        y = labels_list
        
        # 4. Train the Classifier
        # Why Logistic Regression?
        # A 960-dimensional feature representation from MobileNetV3 is extremely rich.
        # Over such clean features, a simple linear model like Logistic Regression performs
        # remarkably well, trains in milliseconds, and provides probability estimates.
        logger.info(f"Training Logistic Regression classifier on {X.shape[0]} samples across {len(temp_class_mapping)} classes...")
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
        try:
            clf.fit(X, y)
        except Exception as e:
            return False, f"Failed to train classifier: {e}"

        # 5. Persist models to disk
        try:
            # Ensure folder exists
            os.makedirs(settings.SAVED_MODELS_DIR, exist_ok=True)
            joblib.dump(clf, settings.model_path)
            joblib.dump(temp_class_mapping, settings.class_mapping_path)
            
            # Update active instance variables
            self.classifier = clf
            self.class_mapping = temp_class_mapping
            
            logger.info("Successfully saved model and class mapping.")
            return True, "Model trained successfully!"
        except Exception as e:
            return False, f"Model trained but failed to save weights to disk: {e}"

    def predict_image(self, pil_image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        """
        Runs inference on a single image.
        Returns:
            - The predicted class name (str)
            - The confidence of the prediction (float)
            - A full dictionary of confidence scores for all classes {class_name: confidence}
        """
        if self.classifier is None or self.class_mapping is None:
            # Try reloading again in case a new model was saved on disk by another thread/process
            self._load_saved_model_if_exists()
            if self.classifier is None or self.class_mapping is None:
                raise ValueError("No trained model weights found. Please train the model first.")

        # 1. Feature extraction for single image (batch size = 1)
        features = self.extract_features_batch([pil_image])
        features_np = features.cpu().numpy()

        # 2. Get prediction probabilities
        probabilities = self.classifier.predict_proba(features_np)[0]
        
        # Find index of the highest probability
        predicted_idx = int(probabilities.argmax())
        predicted_class = self.class_mapping[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # 3. Structure confidence output for all categories
        all_confidences = {
            self.class_mapping[idx]: float(prob)
            for idx, prob in enumerate(probabilities)
        }

        return predicted_class, confidence, all_confidences

    def reset_model(self) -> None:
        """
        Resets active classifier and mapping values.
        """
        self.classifier = None
        self.class_mapping = None


# Singleton instance pattern for lifespan management
_ml_service_instance: Optional[MLService] = None

def get_ml_service() -> MLService:
    """
    Dependency Injection provider function.
    Ensures that only a single instance of MLService (and thus only one copy
    of the pre-trained MobileNetV3 PyTorch model) is initialized and kept in memory.
    """
    global _ml_service_instance
    if _ml_service_instance is None:
        _ml_service_instance = MLService()
    return _ml_service_instance

