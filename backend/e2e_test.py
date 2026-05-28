import requests
import io
import time
from PIL import Image

BASE_URL = "http://127.0.0.1:8000"

def create_dummy_image(color=(255, 0, 0)):
    # Create a 224x224 RGB image
    img = Image.new("RGB", (224, 224), color)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

def run_e2e_tests():
    print("=== Starting Teachable Machine Backend E2E Tests ===")
    
    # 1. Health check
    print("Testing /health...")
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    print("Health check OK:", r.json())
    
    # 2. Clear dataset and models
    print("Testing /clear...")
    r = requests.post(f"{BASE_URL}/clear")
    assert r.status_code == 200, f"Clear failed: {r.text}"
    print("Clear OK:", r.json())
    
    # 3. Upload samples for Class A ("Cat") and Class B ("Dog")
    print("Testing /upload-sample...")
    # Class A: Cat (Red images)
    for i in range(3):
        img_data = create_dummy_image(color=(255, 0, 0))
        files = {"files": ("cat_sample.jpg", img_data, "image/jpeg")}
        data = {"class_name": "Cat"}
        r = requests.post(f"{BASE_URL}/upload-sample", files=files, data=data)
        assert r.status_code == 200, f"Upload sample A failed: {r.text}"
        
    # Class B: Dog (Blue images)
    for i in range(3):
        img_data = create_dummy_image(color=(0, 0, 255))
        files = {"files": ("dog_sample.jpg", img_data, "image/jpeg")}
        data = {"class_name": "Dog"}
        r = requests.post(f"{BASE_URL}/upload-sample", files=files, data=data)
        assert r.status_code == 200, f"Upload sample B failed: {r.text}"
        
    print("Samples upload OK.")
    
    # 4. Check dataset status
    print("Testing /dataset-status...")
    r = requests.get(f"{BASE_URL}/dataset-status")
    assert r.status_code == 200, f"Dataset status failed: {r.text}"
    status = r.json()
    print("Dataset status OK:", status)
    classes_dict = {item["class_name"]: item["sample_count"] for item in status["classes"]}
    assert classes_dict["Cat"] == 3, f"Expected 3 Cat samples, got {classes_dict.get('Cat')}"
    assert classes_dict["Dog"] == 3, f"Expected 3 Dog samples, got {classes_dict.get('Dog')}"
    
    # 5. Train model
    print("Testing /train...")
    r = requests.post(f"{BASE_URL}/train")
    assert r.status_code == 200, f"Train failed: {r.text}"
    train_res = r.json()
    print("Train OK:", train_res)
    assert train_res["success"] is True
    
    # 6. Predict using Cat (Red) dummy image
    print("Testing /predict (Cat)...")
    img_data = create_dummy_image(color=(255, 0, 0))
    files = {"file": ("test_cat.jpg", img_data, "image/jpeg")}
    r = requests.post(f"{BASE_URL}/predict", files=files)
    assert r.status_code == 200, f"Predict failed: {r.text}"
    pred_res = r.json()
    print("Predict Cat OK:", pred_res)
    assert pred_res["predicted_class"] == "Cat"
    
    # Predict using Dog (Blue) dummy image
    print("Testing /predict (Dog)...")
    img_data = create_dummy_image(color=(0, 0, 255))
    files = {"file": ("test_dog.jpg", img_data, "image/jpeg")}
    r = requests.post(f"{BASE_URL}/predict", files=files)
    assert r.status_code == 200, f"Predict failed: {r.text}"
    pred_res = r.json()
    print("Predict Dog OK:", pred_res)
    assert pred_res["predicted_class"] == "Dog"
    
    # 7. Download model
    print("Testing /download-model...")
    r = requests.get(f"{BASE_URL}/download-model")
    assert r.status_code == 200, f"Download model failed: {r.text}"
    assert len(r.content) > 0, "Model zip file is empty"
    print("Download model OK. Received zip file size:", len(r.content), "bytes")
    
    # 8. Clear dataset again to clean up
    print("Cleaning up via /clear...")
    r = requests.post(f"{BASE_URL}/clear")
    assert r.status_code == 200, f"Final clear failed: {r.text}"
    print("Clean up OK:", r.json())
    
    print("\n=== ALL E2E TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_e2e_tests()
