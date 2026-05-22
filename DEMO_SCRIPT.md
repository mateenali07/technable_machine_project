# 🎤 Suggested Demo Script & Flow

If you are presenting this project at a hackathon, interview, or class presentation, follow this sequence to wow your audience.

## Preparation
1. Ensure both the FastAPI backend and Streamlit frontend are running.
2. Open the UI to `http://localhost:8501`.
3. Have 3 physical props ready (e.g., your hand, a coffee mug, and a pen).

## The Flow

### 1. Introduction (1 min)
* **What you say**: "Welcome to my Teachable Machine clone! I've built a fully decoupled AI application that lets anyone train an image classifier instantly directly in the browser. It leverages PyTorch MobileNetV3 for feature extraction and a FastAPI backend to handle the heavy lifting."
* **Action**: Show the gorgeous, glassmorphic UI landing page. Point out that the backend status shows "Online".

### 2. Data Ingestion (1.5 mins)
* **What you say**: "Let's build an AI that can recognize objects on my desk. I'll create three classes: 'Empty Desk', 'Coffee Mug', and 'Pen'."
* **Action**: Rename the classes in the UI to match. Add a third class using the `+ Add Category` button.
* **What you say**: "Instead of writing scripts, I can use my webcam to gather data in real-time."
* **Action**: 
   - Record ~30 frames of the empty desk.
   - Hold up the coffee mug and record ~30 frames.
   - Hold up the pen and record ~30 frames.

### 3. Transfer Learning (30 seconds)
* **What you say**: "Now for the magic. Training a neural network usually takes hours, but I've implemented Transfer Learning. The backend extracts structural features from a pre-trained PyTorch model and trains a Scikit-Learn classifier on top of them."
* **Action**: Click `Train Transfer Learning Model`.
* **What you say**: "And it's done! In just milliseconds, the API processed our data and saved the model."

### 4. Real-time Inference (1 min)
* **What you say**: "Let's test it out. Watch the confidence meters."
* **Action**: Start the prediction webcam preview.
   - Show the empty desk (the "Empty Desk" bar should hit 99% and turn Green).
   - Bring in the coffee mug. Watch the model instantly switch to "Coffee Mug".
   - Bring in the pen.
* **What you say**: "The decoupled architecture means inference requests are handled concurrently by FastAPI, keeping the Streamlit UI completely unblocked and buttery smooth."

### 5. Conclusion
* **What you say**: "You can even download the resulting `.joblib` model straight from the UI to use in your own scripts. This project demonstrates full-stack ML engineering, from API design to dynamic frontend state management."
