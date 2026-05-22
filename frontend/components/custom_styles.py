import streamlit as st
from pathlib import Path

def inject_custom_styles():
    """
    Reads the style.css file and injects it into the Streamlit markdown engine.
    Also hides default Streamlit headers, menus, and footers to deliver a 
    standalone SaaS dashboard experience.
    """
    css_path = Path(__file__).resolve().parent.parent / "styles" / "style.css"
    
    if css_path.exists():
        with open(css_path, "r") as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.warning("Custom CSS file not found. Falling back to default styles.")
        
    # Additional UI Polish: Hide the default Streamlit main menu and footer
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display:none;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def render_glass_card(content: str):
    """
    Renders custom HTML formatted inside a glassmorphic card.
    """
    st.markdown(
        f'<div class="glass-card">{content}</div>',
        unsafe_allow_html=True
    )
