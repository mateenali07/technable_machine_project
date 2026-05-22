# 🛠️ Troubleshooting Guide

If you encounter issues while running the Teachable Machine Clone, consult this guide for common errors and their solutions.

## 1. PyTorch OpenMP Crash (`libiomp5md.dll`)
**Symptom**: The backend server crashes immediately upon startup with an error mentioning `libiomp5md.dll` or duplicate OpenMP runtimes.
**Cause**: Anaconda environments on Windows often bundle an Intel OpenMP library that conflicts with PyTorch's bundled version.
**Solution**: 
This is already mitigated via the `.env` file! Ensure your `backend/.env` file contains:
```env
KMP_DUPLICATE_LIB_OK="TRUE"
```

## 2. "Connection Error: FastAPI Backend Offline"
**Symptom**: The Streamlit frontend shows a red warning indicating the backend is offline.
**Cause**: The frontend cannot reach `http://127.0.0.1:8000`.
**Solution**:
1. Check the terminal where you ran `uvicorn app.main:app ...`
2. Ensure there are no Python errors preventing startup.
3. Ensure you are running the backend on port `8000`. If you changed the port, you must update the `base_url` in `frontend/services/api_service.py`.

## 3. "Training requires at least 2 classes"
**Symptom**: The Train button is disabled or throws an error.
**Cause**: Transfer learning classifiers require distinct categories to learn boundaries.
**Solution**: 
1. Ensure you have at least 2 categories created.
2. Ensure you have uploaded at least 1 image to *each* category.

## 4. File Upload Timeouts
**Symptom**: Uploading hundreds of images via the webcam fails or hangs.
**Cause**: HTTP POST limits over localhost can sometimes bottle-neck if attempting to send 100+ high-res images in a single payload.
**Solution**: Upload in batches. Record 30 frames, let them upload, then record another 30 frames.
