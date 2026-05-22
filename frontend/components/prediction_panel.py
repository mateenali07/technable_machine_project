import streamlit as st
from PIL import Image
from typing import Dict, Any, Callable, Tuple


# Semantic category mapping for human-readable explanations
SEMANTIC_MAP = {
    # Animals
    "cat": ("Animal", "🐱", "fur texture, pointed ears, and feline body shape"),
    "dog": ("Animal", "🐶", "fur patterns, snout shape, and canine body structure"),
    "bird": ("Animal", "🐦", "feathers, beak shape, and wing structure"),
    "fish": ("Animal", "🐟", "scales, fin shape, and aquatic body form"),
    "horse": ("Animal", "🐴", "mane, elongated face, and hooved limbs"),
    "elephant": ("Animal", "🐘", "large ears, trunk, and grey skin texture"),
    "rabbit": ("Animal", "🐰", "long ears, small body, and soft fur texture"),
    # People
    "person": ("Human", "🧑", "facial features, body posture, and clothing patterns"),
    "human": ("Human", "🧑", "facial features, body posture, and clothing patterns"),
    "face": ("Human", "🧑", "facial geometry, skin tone, and eye/nose/mouth patterns"),
    "man": ("Human", "🧑", "facial features, body posture, and clothing patterns"),
    "woman": ("Human", "🧑", "facial features, body posture, and clothing patterns"),
    # Vehicles
    "car": ("Vehicle", "🚗", "wheel shapes, windshield reflections, and metal body panels"),
    "truck": ("Vehicle", "🚛", "large body frame, wheel count, and cargo structure"),
    "bus": ("Vehicle", "🚌", "elongated body, window rows, and door placement"),
    "bike": ("Vehicle", "🏍️", "two-wheel structure, handlebar shape, and frame geometry"),
    "bicycle": ("Vehicle", "🚲", "two-wheel structure, pedal mechanism, and thin frame"),
    "airplane": ("Vehicle", "✈️", "wing span, fuselage shape, and engine placement"),
    "boat": ("Vehicle", "🚢", "hull shape, deck structure, and water-line patterns"),
    # Objects
    "phone": ("Object", "📱", "rectangular shape, screen reflection, and bezel edges"),
    "laptop": ("Object", "💻", "hinged screen, keyboard layout, and trackpad area"),
    "book": ("Object", "📚", "rectangular shape, page edges, and spine structure"),
    "chair": ("Object", "🪑", "seat shape, leg structure, and backrest form"),
    "bottle": ("Object", "🍼", "cylindrical shape, cap area, and label placement"),
    "cup": ("Object", "☕", "cylindrical shape, handle form, and rim edge"),
    # Nature
    "tree": ("Nature", "🌳", "trunk shape, leaf clusters, and branch patterns"),
    "flower": ("Nature", "🌸", "petal shapes, color gradients, and stem structure"),
    "mountain": ("Nature", "⛰️", "peak geometry, rock texture, and elevation lines"),
}


def get_semantic_info(class_name: str) -> tuple:
    """
    Returns (category_type, emoji, feature_explanation) for a class name.
    Falls back to a generic 'Class' category if not found in the semantic map.
    """
    key = class_name.lower().strip()
    if key in SEMANTIC_MAP:
        return SEMANTIC_MAP[key]
    # Partial match: check if any key is a substring of the class name
    for map_key, value in SEMANTIC_MAP.items():
        if map_key in key or key in map_key:
            return value
    return ("Class", "🏷️", "visual patterns, shapes, and textures learned during training")


def get_confidence_label(confidence: float) -> tuple:
    """Returns (label_text, css_color) based on confidence level."""
    if confidence >= 0.85:
        return ("Very High Confidence", "#10b981")
    elif confidence >= 0.65:
        return ("High Confidence", "#22d3ee")
    elif confidence >= 0.40:
        return ("Moderate Confidence", "#f59e0b")
    else:
        return ("Low Confidence", "#ef4444")


def render_confidence_bar(class_name: str, confidence: float, is_top: bool) -> str:
    """
    Constructs an HTML snippet for a custom-styled confidence progress bar.
    Applies gradient green for the winning category, and deep indigo for others.
    """
    pct = confidence * 100
    fill_class = "confidence-bar-fill-top" if is_top else ""
    semantic_type, emoji, _ = get_semantic_info(class_name)
    return f"""
    <div class="confidence-item">
        <div class="confidence-row">
            <span class="confidence-label">{emoji} {semantic_type}: {class_name}</span>
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
    Renders the premium prediction interface with:
    1. Input source toggle (Webcam vs. Uploaded image file).
    2. Top prediction hero card with semantic label.
    3. Confidence progress bars for all classes.
    4. AI Insight Panel with human-readable XAI explanation.
    """
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        test_source = st.radio(
            "Select Inference Source:",
            ["📂 Upload Image", "📷 Webcam Snapshot"],
            key="test_source_selection",
            horizontal=True
        )
        
    # We will fetch and pass image bytes to the backend prediction API
    target_image_bytes = None
    filename = "test_image.jpg"

    with col2:
        if test_source == "📂 Upload Image":
            uploaded_file = st.file_uploader(
                "Upload a test image:",
                type=["png", "jpg", "jpeg", "webp"],
                key="inference_file_uploader"
            )
            if uploaded_file is not None:
                target_image_bytes = uploaded_file.read()
                filename = uploaded_file.name
                
        elif test_source == "📷 Webcam Snapshot":
            webcam_image = st.camera_input(
                "Snap a frame to classify:",
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
            
            # Get semantic info for the predicted class
            semantic_type, emoji, feature_explanation = get_semantic_info(predicted_class)
            conf_label, conf_color = get_confidence_label(confidence)
            pct = confidence * 100
            
            # --- HERO PREDICTION CARD ---
            st.markdown(
                f"""
                <div class="predict-winning-card animate-fade-in">
                    <div style="font-size: 0.85rem; color: #94a3b8; letter-spacing: 2px; text-transform: uppercase;">
                        TOP PREDICTION
                    </div>
                    <div class="predict-winning-label">{emoji} {predicted_class}</div>
                    <div style="font-size: 1rem; color: #a5b4fc; margin-bottom: 8px;">
                        {semantic_type}: {predicted_class}
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {conf_color};">
                        {pct:.1f}%
                    </div>
                    <div style="font-size: 0.85rem; color: {conf_color}; margin-top: 4px;">
                        {conf_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # --- ALL CONFIDENCE BARS ---
            bars_html = ""
            sorted_classes = sorted(all_confidences.items(), key=lambda item: item[1], reverse=True)
            
            for class_name, conf_val in sorted_classes:
                is_top = (class_name == predicted_class)
                bars_html += render_confidence_bar(class_name, conf_val, is_top)
                
            st.markdown(
                f"""
                <div class="glass-card animate-fade-in">
                    <div style="font-weight:600; font-size:1.1rem; margin-bottom:16px; color:#ffffff;">
                        📊 Probability Breakdown
                    </div>
                    {bars_html}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # --- AI INSIGHT PANEL (XAI) ---
            # Build human-readable explanation
            runner_up = ""
            if len(sorted_classes) > 1:
                runner_up_name = sorted_classes[1][0]
                runner_up_conf = sorted_classes[1][1] * 100
                runner_up = f"The runner-up class was <b>{runner_up_name}</b> at {runner_up_conf:.1f}%."
            
            st.markdown(
                f"""
                <div class="glass-card animate-fade-in" style="border-left: 3px solid #6366f1;">
                    <div style="font-weight:700; font-size:1.1rem; margin-bottom:12px; color:#a5b4fc;">
                        🧠 AI Insight Panel — Explainable AI (XAI)
                    </div>
                    <div style="color: #e2e8f0; line-height: 1.8;">
                        <p>
                            <b>🏆 Prediction:</b> <span style="color: white;">{predicted_class}</span>
                        </p>
                        <p>
                            <b>📊 Confidence:</b> <span style="color: {conf_color};">{pct:.1f}%</span>
                        </p>
                        <p>
                            <b>🔍 Interpretation:</b> This image is classified as 
                            "<span style="color: #a5b4fc;">{semantic_type}: {predicted_class}</span>" 
                            with <span style="color: {conf_color};">{conf_label.lower()}</span>.
                        </p>
                        <p>
                            <b>💡 How the model decided:</b> The neural network (MobileNetV3) detected 
                            features like <i>{feature_explanation}</i> that closely match the 
                            "{predicted_class}" training data.
                        </p>
                        <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 8px;">
                            {runner_up}
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error(prediction.get("message", "Prediction failed. Check backend connection."))
    else:
        # Visual prompt when no image has been uploaded/snapped yet
        st.markdown(
            """
            <div class="glass-card text-center animate-fade-in" style="padding: 40px;">
                <div style="font-size: 3rem; margin-bottom: 16px;">🔮</div>
                <div style="font-size: 1.2rem; color: #a5b4fc; font-weight: 600;">
                    Awaiting Input
                </div>
                <p class="status-text" style="margin-top: 8px;">
                    Upload an image or snap a webcam photo to see the AI prediction.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
