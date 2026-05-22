import streamlit as st
from PIL import Image
import io
from typing import Callable

def render_class_card(
    class_index: int,
    class_name: str,
    sample_count: int,
    on_upload: Callable[[str, list], None]
):
    """
    Renders an ingestion card for a single class category.
    Allows user to:
    1. View category details and active sample counts.
    2. Upload image files from disk.
    3. Capture frames via Webcam.
    """
    # Create HTML structure for the card header using our CSS class rules
    st.markdown(
        f"""
        <div class="glass-card mb-4">
            <div class="class-header">
                <span class="class-title">Class {class_index + 1}: {class_name}</span>
                <span class="sample-counter">{sample_count} samples</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # We place the controls inside columns for clean alignments
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # User input option toggle
        input_type = st.radio(
            "Select Input Source:",
            ["File Upload", "Webcam Capture"],
            key=f"input_type_{class_name}_{class_index}",
            horizontal=True
        )

    with col2:
        st.write("") # Spacer to align with radio button heights
        st.write("") 

    if input_type == "File Upload":
        # Multi-file uploader
        uploaded_files = st.file_uploader(
            "Drag & drop image files:",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key=f"uploader_{class_name}_{class_index}"
        )
        
        if uploaded_files:
            # Trigger upload to backend
            files_to_send = []
            for file in uploaded_files:
                files_to_send.append((file.name, file.read()))
                
            if st.button(f"Add Files to {class_name}", key=f"btn_upload_{class_name}_{class_index}"):
                with st.spinner(f"Uploading files for {class_name}..."):
                    on_upload(class_name, files_to_send)

    elif input_type == "Webcam Capture":
        # Webcam camera frame grabber
        cam_image = st.camera_input(
            f"Capture training frame for {class_name}",
            key=f"camera_{class_name}_{class_index}"
        )
        
        if cam_image is not None:
            # Generate a generic filename with bytes
            image_bytes = cam_image.read()
            files_to_send = [("webcam_frame.jpg", image_bytes)]
            
            if st.button(f"Save Frame to {class_name}", key=f"btn_cam_{class_name}_{class_index}"):
                with st.spinner(f"Saving frame..."):
                    on_upload(class_name, files_to_send)
                    
    # Divider for visual separation between cards
    st.markdown("---")
