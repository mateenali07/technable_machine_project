import streamlit as st
from PIL import Image
from typing import Dict, Any, Callable, Tuple

def render_confidence_bar(class_name: str, confidence: float, is_top: bool) -> str:
    """
    Constructs an HTML snippet for a custom-styled confidence progress bar.
    Applies gradient green for the winning category, and deep indigo for others.
    """
    pct = confidence * 100
    fill_class = "confidence-bar-fill-top" if is_top else ""
    return f"""
    <div class="confidence-item">
        <div class="confidence-row">
            <span class="confidence-label">{class_name}</span>
            <span class="confidence-value">{pct:.1f}%</span>
        </div>
        <div class="confidence-bar-bg">
            <div class="confidence-bar-fill {fill_class}" style="width: {pct:.1f}%;"></div>
        </div>
    </div>
    """

def render_prediction_panel(
    on_predict: Callable[[str, bytes], Tuple[bool, Dict[str, Any]]]
):
    """
    Renders the live testing interface.
    Features:
    1. Input source toggle (Webcam vs. Uploaded image file).
    2. Prediction visualizer card.
    3. Custom horizontal confidence meters.
    """
    st.markdown(
        """
        <div class="section-header">
            🔍 Real-Time Inference & Live Preview Meter
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        test_source = st.radio(
            "Select Inference Source:",
            ["Test Image File", "Live Webcam Testing"],
            key="test_source_selection"
        )
        
    # We will fetch and pass image bytes to the backend prediction API
    target_image_bytes = None
    filename = "test_image.jpg"

    with col2:
        if test_source == "Test Image File":
            uploaded_file = st.file_uploader(
                "Upload a test image:",
                type=["png", "jpg", "jpeg", "webp"],
                key="inference_file_uploader"
            )
            if uploaded_file is not None:
                target_image_bytes = uploaded_file.read()
                filename = uploaded_file.name
                
        elif test_source == "Live Webcam Testing":
            webcam_image = st.camera_input(
                "Snap a frame to classify instantly:",
                key="inference_webcam"
            )
            if webcam_image is not None:
                target_image_bytes = webcam_image.read()
                filename = "webcam_inference.jpg"

    st.markdown("<div class='mt-4'></div>", unsafe_allow_html=True)

    if target_image_bytes is not None:
        # Trigger prediction
        with st.spinner("Classifying image..."):
            success, prediction = on_predict(filename, target_image_bytes)
            
        if success:
            predicted_class = prediction.get("predicted_class", "Unknown")
            confidence = prediction.get("confidence", 0.0)
            all_confidences = prediction.get("all_confidences", {})
            
            # Show winning category in a highlighted neon card
            st.markdown(
                f"""
                <div class="predict-winning-card">
                    <div class="status-text">WINNING CATEGORY</div>
                    <div class="predict-winning-label">🎉 {predicted_class}</div>
                    <div class="predict-winning-pct">Confidence Score: {confidence * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Build HTML progress bars list for all categories
            bars_html = ""
            # Sort categories by confidence descending
            sorted_classes = sorted(all_confidences.items(), key=lambda item: item[1], reverse=True)
            
            for class_name, conf_val in sorted_classes:
                is_top = (class_name == predicted_class)
                bars_html += render_confidence_bar(class_name, conf_val, is_top)
                
            # Wrap progress bar list inside a card
            st.markdown(
                f"""
                <div class="glass-card">
                    <div style="font-weight:600; font-size:1.1rem; margin-bottom:16px; color:#ffffff;">
                        All Categories Probability Breakdown
                    </div>
                    {bars_html}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error(prediction.get("message", "Prediction failed. Check backend connection."))
    else:
        # Visual prompt when no image has been uploaded/snapped yet
        st.info("Upload an image file or snap a webcam photo to activate the prediction dashboard.")
