import streamlit as st
import logging
from typing import Dict, List, Tuple, Any
import requests

st.set_page_config(
    page_title="Technable Machine - Premium AI Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from services.api_service import APIService
from components.custom_styles import inject_custom_styles
from components.class_card import render_class_card
from components.prediction_panel import render_prediction_panel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("frontend.app")

api_service = APIService()
inject_custom_styles()

# State Management for Step-by-Step UX
if "classes" not in st.session_state:
    st.session_state.classes = []

if "step" not in st.session_state:
    st.session_state.step = 1 # 1: Add Classes, 2: Upload Data, 3: Train, 4: Predict

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

if "backend_healthy" not in st.session_state:
    st.session_state.backend_healthy, st.session_state.backend_device = api_service.check_health()

def refresh_dataset_status():
    status_data = api_service.get_dataset_status()
    if status_data:
        st.session_state.model_trained = status_data.get("model_trained", False)
        backend_classes = status_data.get("classes", [])
        backend_class_map = {item["class_name"]: item["sample_count"] for item in backend_classes}
        for local_cls in st.session_state.classes:
            name = local_cls["name"]
            clean_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip()
            local_cls["sample_count"] = backend_class_map.get(clean_name, 0)

def handle_upload(class_name: str, files: List[Tuple[str, bytes]]):
    clean_name = "".join(c for c in class_name if c.isalnum() or c in ("-", "_")).strip()
    success, message = api_service.upload_samples(clean_name, files)
    if success:
        st.toast(f"Data uploaded for {class_name}", icon="✅")
        refresh_dataset_status()
    else:
        st.error(message)

def handle_predict(filename: str, image_bytes: bytes) -> Tuple[bool, Dict[str, Any]]:
    return api_service.predict_image(filename, image_bytes)

# --- HEADER ---
st.markdown(
    """
    <div class="main-title">🤖 Technable Machine <span style="font-weight:300;">Premium</span></div>
    <div class="sub-title">Train Explainable AI (XAI) models directly in your browser.</div>
    """, unsafe_allow_html=True
)

if not st.session_state.backend_healthy:
    st.error("### ⚠️ Connection Error: FastAPI Backend Offline on port 8000")
    st.stop()

st.markdown("<hr style='opacity: 0.2; margin-top:0;'>", unsafe_allow_html=True)

# --- STEP 1: ADD CLASSES ---
if st.session_state.step == 1:
    st.markdown('<div class="section-header">Step 1: Define 3 Classes</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.info(f"You have added {len(st.session_state.classes)}/3 classes.")
        
        # Manual Add
        with st.form("add_class_form", clear_on_submit=True):
            new_class_name = st.text_input("Enter Class Name:")
            submit_btn = st.form_submit_button("➕ Add Class", use_container_width=True)
            if submit_btn and new_class_name:
                clean_name = "".join(c for c in new_class_name if c.isalnum() or c in ("-", "_")).strip()
                if len(st.session_state.classes) < 3 and clean_name not in [c["name"] for c in st.session_state.classes]:
                    st.session_state.classes.append({"name": clean_name, "sample_count": 0})
                    st.rerun()

        # Display current classes
        for c in st.session_state.classes:
            st.markdown(f"🏷️ **{c['name']}**")

    with col2:
        st.markdown("<div style='margin-bottom: 20px;'>Or skip manual setup:</div>", unsafe_allow_html=True)
        if st.button("🚀 Start Demo Mode (Auto-fill Classes)", use_container_width=True):
            api_service.clear_all()
            st.session_state.classes = [
                {"name": "Cat", "sample_count": 0},
                {"name": "Dog", "sample_count": 0},
                {"name": "Car", "sample_count": 0}
            ]
            st.session_state.step = 2
            st.rerun()

    if len(st.session_state.classes) == 3:
        if st.button("Next: Collect Data ➡️", use_container_width=True):
            st.session_state.step = 2
            st.rerun()

# --- STEP 2: INGESTION ---
if st.session_state.step >= 2:
    st.markdown('<div class="section-header">Step 2: Dataset Collection</div>', unsafe_allow_html=True)
    
    cols = st.columns(3)
    for idx, cls_info in enumerate(st.session_state.classes):
        with cols[idx]:
            render_class_card(
                class_index=idx,
                class_name=cls_info["name"],
                sample_count=cls_info["sample_count"],
                on_upload=handle_upload
            )

    # Check if ready for training (all 3 classes must have data)
    ready_to_train = all(c["sample_count"] > 0 for c in st.session_state.classes) and len(st.session_state.classes) == 3
    
    st.markdown('<div class="mt-4"></div>', unsafe_allow_html=True)
    if ready_to_train and st.session_state.step == 2:
        if st.button("Next: Train Engine ➡️", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

# --- STEP 3: TRAINING ---
if st.session_state.step >= 3:
    st.markdown('<div class="section-header">Step 3: Neural Engine Training</div>', unsafe_allow_html=True)
    
    total_samples = sum(c["sample_count"] for c in st.session_state.classes)
    st.markdown(
        f"""
        <div class="glass-card mb-4 animate-fade-in text-center">
            <h3>Training Ready</h3>
            <p>Total Images: <b>{total_samples}</b> across 3 categories.</p>
        </div>
        """, unsafe_allow_html=True
    )
    
    if not st.session_state.model_trained:
        train_btn = st.button("🚀 TRAIN MODEL NOW", use_container_width=True)
        if train_btn:
            with st.spinner("Extracting MobileNetV3 features and fitting linear classifier..."):
                success, message = api_service.train_model()
            if success:
                refresh_dataset_status()
                st.session_state.step = 4
                st.rerun()
            else:
                st.error(f"Training failed: {message}")
    else:
        st.success("🎓 Model is Trained and Ready for Inference!")
        if st.session_state.step == 3:
            if st.button("Next: Test Predictions ➡️", use_container_width=True):
                st.session_state.step = 4
                st.rerun()

# --- STEP 4: PREDICTION ---
if st.session_state.step == 4 and st.session_state.model_trained:
    st.markdown('<div class="section-header">Step 4: Real-time Prediction (Explainable AI)</div>', unsafe_allow_html=True)
    render_prediction_panel(on_predict=handle_predict)

# --- FOOTER ---
st.markdown("---")
if st.button("🗑️ Restart Entire Project", type="secondary"):
    api_service.clear_all()
    st.session_state.classes = []
    st.session_state.step = 1
    st.session_state.model_trained = False
    st.rerun()
