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
    Renders a compact, premium ingestion card for a single class category.
    Designed to sit cleanly in a 3-column grid layout.
    """
    # Emoji based on index for visual variety
    emojis = ["🔴", "🟢", "🔵"]
    emoji = emojis[class_index % 3]

    # Card header
    st.markdown(
        f"""
        <div class="glass-card animate-fade-in">
            <div class="class-header">
                <span class="class-title">{emoji} {class_name}</span>
                <span class="sample-counter">{sample_count} samples</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Input type selector
    input_type = st.radio(
        "Input:",
        ["📂 Upload", "📷 Webcam"],
        key=f"input_type_{class_name}_{class_index}",
        horizontal=True
    )

    if input_type == "📂 Upload":
        uploaded_files = st.file_uploader(
            "Drop images here:",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key=f"uploader_{class_name}_{class_index}"
        )

        if uploaded_files:
            files_to_send = []
            for file in uploaded_files:
                files_to_send.append((file.name, file.read()))

            if st.button(
                f"⬆️ Upload {len(files_to_send)} files",
                key=f"btn_upload_{class_name}_{class_index}",
                use_container_width=True
            ):
                with st.spinner(f"Uploading to {class_name}..."):
                    on_upload(class_name, files_to_send)

    elif input_type == "📷 Webcam":
        cam_image = st.camera_input(
            f"Capture frame",
            key=f"camera_{class_name}_{class_index}"
        )

        if cam_image is not None:
            image_bytes = cam_image.read()
            files_to_send = [("webcam_frame.jpg", image_bytes)]

            if st.button(
                f"💾 Save Frame",
                key=f"btn_cam_{class_name}_{class_index}",
                use_container_width=True
            ):
                with st.spinner("Saving..."):
                    on_upload(class_name, files_to_send)
