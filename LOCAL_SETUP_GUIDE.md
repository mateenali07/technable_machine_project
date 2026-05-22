# 🚀 Local Setup Guide

This guide details the exact steps to get both the FastAPI backend and Streamlit frontend running on a local Windows machine.

## Prerequisites
- Windows 10/11
- Anaconda or Miniconda installed
- A modern web browser

## 1. Environment Configuration
Open your Anaconda Prompt and create a fresh Python 3.13 environment:

```powershell
conda create -n technable python=3.13 -y
conda activate technable
```

## 2. Setting Up the Backend
The backend requires PyTorch and FastAPI to serve the machine learning pipeline.

```powershell
# Navigate into the backend folder
cd backend

# Install all backend requirements
pip install -r requirements.txt

# Copy the example environment variables
copy .env.example .env

# Start the FastAPI Server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Leave this terminal window open! The backend must remain active.

## 3. Setting Up the Frontend
Open a **second, completely separate** Anaconda Prompt window.

```powershell
# Activate the same environment
conda activate technable

# Navigate to the frontend folder
cd frontend

# Install frontend requirements (Streamlit)
pip install -r requirements.txt

# Launch the Application
streamlit run app.py
```

Your browser will automatically open `http://localhost:8501`. You can now begin uploading images and training your model!
