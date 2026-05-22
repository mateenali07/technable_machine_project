import streamlit as st
import logging
from typing import Dict, List, Tuple, Any

# Configure Streamlit page layout at the absolute top of the script
st.set_page_config(
    page_title="Technable Machine - Custom AI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from services.api_service import APIService
from components.custom_styles import inject_custom_styles
from components.class_card import render_class_card
from components.prediction_panel import render_prediction_panel

# 1. Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("frontend.app")

# 2. Initialize API Service
# Points to our FastAPI backend running locally
api_service = APIService()

# 3. Inject CSS styles
inject_custom_styles()

# 4. State Management Checkpoints
# Streamlit is stateless by default. To preserve lists of classes, training status,
# and user inputs across reruns, we utilize Streamlit's st.session_state dictionary.
if "classes" not in st.session_state:
    # Default classes to mirror Google's Teachable Machine baseline setup (2 initial classes)
    st.session_state.classes = [
        {"name": "Class 1", "sample_count": 0},
        {"name": "Class 2", "sample_count": 0}
    ]

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

if "backend_healthy" not in st.session_state:
    st.session_state.backend_healthy = False

if "backend_device" not in st.session_state:
    st.session_state.backend_device = "offline"

# 5. Helper service calls
def refresh_dataset_status():
    """
    Fetches the current dataset state and training status from the FastAPI backend.
    Synchronizes the local session state with the backend storage.
    """
    status_data = api_service.get_dataset_status()
    if status_data:
        st.session_state.model_trained = status_data.get("model_trained", False)
        
        # Sync class sample counts from backend
        backend_classes = status_data.get("classes", [])
        backend_class_map = {item["class_name"]: item["sample_count"] for item in backend_classes}
        
        # Merge or update local classes
        for local_cls in st.session_state.classes:
            name = local_cls["name"]
            # Clean name to match backend directory naming logic
            clean_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip()
            local_cls["sample_count"] = backend_class_map.get(clean_name, 0)

def handle_upload(class_name: str, files: List[Tuple[str, bytes]]):
    """
    Callback triggered when images are uploaded or captured for a class.
    """
    # Clean the class name before uploading
    clean_name = "".join(c for c in class_name if c.isalnum() or c in ("-", "_")).strip()
    success, message = api_service.upload_samples(clean_name, files)
    if success:
        st.toast(message, icon="✅")
        refresh_dataset_status()
    else:
        st.error(message)

def handle_predict(filename: str, image_bytes: bytes) -> Tuple[bool, Dict[str, Any]]:
    """
    Inference callback called by the prediction panel.
    """
    return api_service.predict_image(filename, image_bytes)


# 6. Check Backend Health
st.session_state.backend_healthy, st.session_state.backend_device = api_service.check_health()

# Sync status at start of run
if st.session_state.backend_healthy:
    refresh_dataset_status()

# 7. Render Layout
# Main visual header
st.markdown(
    """
    <div class="main-title">🧠 Technable Machine</div>
    <div class="sub-title">Train a real-time transfer learning model in seconds using PyTorch and FastAPI</div>
    """,
    unsafe_allow_html=True
)

# Status indicators bar
status_col1, status_col2 = st.columns([1, 1])
with status_col1:
    if st.session_state.backend_healthy:
        st.markdown(
            f"""
            <span class="status-badge status-online">
                🟢 API Server: Online (PyTorch device: {st.session_state.backend_device})
            </span>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <span class="status-badge status-offline">
                🔴 API Server: Offline (FastAPI backend not detected)
            </span>
            """,
            unsafe_allow_html=True
        )
        
with status_col2:
    if st.session_state.model_trained:
        st.markdown(
            """
            <div class="text-right">
                <span class="status-badge status-trained">
                    🎓 Model Status: Trained & Ready
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="text-right">
                <span class="status-badge status-offline">
                    🎓 Model Status: Untrained
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<div class='mb-4'></div>", unsafe_allow_html=True)

# If the backend is offline, render a warning dashboard instead of the workflow
if not st.session_state.backend_healthy:
    st.error("### ⚠️ Connection Error: FastAPI Backend Offline")
    st.info(
        "To start the application, make sure your FastAPI backend server is running locally on port 8000.\n\n"
        "**How to start the backend:**\n"
        "1. Open Anaconda Prompt or your terminal.\n"
        "2. Navigate to the backend workspace directory:\n"
        "   `cd backend`\n"
        "3. Run the following command:\n"
        "   `uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`"
    )
    st.stop() # Stops Streamlit rendering immediately

# If backend is healthy, show the decoupled panels: Ingestion grid vs Training/Inference
layout_col1, layout_col2 = st.columns([5, 4])

# COLUMN 1: Category Setup & Data Ingestion
with layout_col1:
    st.markdown(
        """
        <div class="section-header">
            📁 Category Setup & Dataset Ingestion
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Render interactive controls to rename or add classes
    for idx, cls_info in enumerate(st.session_state.classes):
        col_rename, col_space = st.columns([3, 1])
        with col_rename:
            new_name = st.text_input(
                f"Rename Category {idx + 1}:",
                value=cls_info["name"],
                key=f"class_rename_key_{idx}"
            )
            # Ensure name matches alphanumeric format constraints
            clean_new_name = "".join(c for c in new_name if c.isalnum() or c in ("-", "_")).strip()
            if clean_new_name and clean_new_name != cls_info["name"]:
                cls_info["name"] = clean_new_name
                st.rerun()

        # Render ingestion card for files and webcam grabs
        render_class_card(
            class_index=idx,
            class_name=cls_info["name"],
            sample_count=cls_info["sample_count"],
            on_upload=handle_upload
        )
        
    # Button to add new categories dynamically
    if st.button("➕ Add Custom Category"):
        new_idx = len(st.session_state.classes) + 1
        st.session_state.classes.append({"name": f"Class {new_idx}", "sample_count": 0})
        st.toast(f"Added Class {new_idx}!", icon="📂")
        st.rerun()

# COLUMN 2: Training Dashboard & Prediction Panel
with layout_col2:
    st.markdown(
        """
        <div class="section-header">
            ⚙️ Training Controller & Engine
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 1. Training Card
    with st.container():
        # Calculate summary parameters
        total_samples = sum(c["sample_count"] for c in st.session_state.classes)
        active_classes_count = len(st.session_state.classes)
        
        st.markdown(
            f"""
            <div class="glass-card mb-4">
                <div style="font-weight:600; font-size:1.1rem; margin-bottom:12px; color:#ffffff;">
                    Training Summary
                </div>
                <p class="status-text">Active Categories: <b>{active_classes_count}</b></p>
                <p class="status-text">Total Image Samples Uploaded: <b>{total_samples}</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col_train, col_clear = st.columns([2, 1])
        
        # Validation rules: Require at least 2 categories and each category needs at least 1 image
        valid_for_training = (active_classes_count >= 2) and all(c["sample_count"] > 0 for c in st.session_state.classes)
        
        with col_train:
            train_btn = st.button(
                "🚀 Train Transfer Learning Model",
                disabled=not valid_for_training,
                key="btn_train_pipeline"
            )
            
            if not valid_for_training:
                st.warning("⚠️ Training requires at least 2 classes, and each class must have at least 1 sample uploaded.")
                
            if train_btn:
                # Trigger training endpoint
                with st.spinner("Extracting features using MobileNetV3 & fitting LogisticRegression classifier..."):
                    success, message = api_service.train_model()
                if success:
                    st.success(message)
                    st.toast("Model trained successfully!", icon="🎓")
                    refresh_dataset_status()
                    st.rerun()
                else:
                    st.error(f"Training failed: {message}")

        with col_clear:
            # Custom styled clear/reset button
            st.markdown('<div class="clear-button">', unsafe_allow_html=True)
            clear_btn = st.button(
                "🗑️ Reset Project",
                key="btn_clear_workspace"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if clear_btn:
                with st.spinner("Clearing project files..."):
                    success, message = api_service.clear_all()
                if success:
                    # Reset frontend state parameters
                    st.session_state.classes = [
                        {"name": "Class 1", "sample_count": 0},
                        {"name": "Class 2", "sample_count": 0}
                    ]
                    st.session_state.model_trained = False
                    st.success(message)
                    st.toast("Project reset complete.", icon="🗑️")
                    st.rerun()
                else:
                    st.error(message)

    st.markdown("---")

    # 2. Prediction Panel
    # Only render testing utilities and prediction components if a trained model weights file exists
    if st.session_state.model_trained:
        render_prediction_panel(on_predict=handle_predict)
    else:
        st.info("💡 **Preview Meter Unavailable**: Upload image samples and click the 'Train' button to activate the prediction dashboard.")
